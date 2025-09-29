import os
from typing import BinaryIO, Iterable, Iterator
import regex as re
import multiprocessing as mp
from pathlib import Path
from time import time
from line_profiler import profile
import pickle
from tqdm import tqdm

class CountDict(dict):

    def __setitem__(self, key, value):
        if not isinstance(value, int):
            raise TypeError(f"Value for key '{key}' must be an int, got {type(value).__name__}")
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        """Updates the dictonary, adds to all values of excisting keys."""
        temp = dict(*args, **kwargs)
        self._update_counts(temp)

    def _update_counts(self, mapping: dict):
        for k, v in mapping.items():
            if not isinstance(v, int):
                raise TypeError(f"Values needs to be of type int, found {type(v).__name__} at key {k}")
            super().update({k: self.get(k, 0) + v})

    def sort(self):
        return self.__class__(sorted(self.items(), key=lambda item: item[1], reverse=True))
    
    def __sub__(self, other):
        res = {}
        for key in set(list(self.keys()) + list(other.keys())):
            res[key] = self.get(key, 0) - other.get(key, 0)
        return self.__class__(res)

class Tokenizer:

    def __init__(self, special_tokens: list[str] = None, num_processes: int = 4,
                 vocab: dict = None, merges: list = None):
        self.counts = CountDict()
        self.special_tokens = [re.escape(spt) for spt in special_tokens] if special_tokens else []
        self.num_processes = num_processes
        self.vocab = vocab if vocab else {}
        self._vocab_list = list(self.vocab.values())
        self.merges = merges if merges else []

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str,
                   special_tokens: list[str] | None = None):
        with open(vocab_filepath, 'rb') as file:
            vocab = pickle.load(file)
        with open(merges_filepath, 'rb') as file:
            merges = pickle.load(file)
        return cls(special_tokens=special_tokens,
                   vocab=vocab,
                   merges=merges)

    def _find_chunk_boundaries(
        self,
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
    ) -> list[int]:
        """
        Chunk the file into parts that can be counted independently.
        May return fewer chunks if the boundaries end up overlapping.
        """
        assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

        # Get total file size in bytes
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        chunk_size = file_size // desired_num_chunks

        # Initial guesses for chunk boundary locations, uniformly spaced
        # Chunks start on previous index, don't include last index
        chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            file.seek(initial_position)  # Start at boundary guess
            while True:
                mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

                # If EOF, this boundary should be at the end of the file
                if mini_chunk == b"":
                    chunk_boundaries[bi] = file_size
                    break

                # Find the special token in the mini chunk
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    break
                initial_position += mini_chunk_size

        # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
        return sorted(set(chunk_boundaries))
    
    def _pretokenize_chunk(self, chunk: str) -> list[tuple]:
        words = []
        if self.special_tokens:
            sub_chunks = re.split(rf"{'|'.join(self.special_tokens)}", chunk)
        else:
            sub_chunks = [chunk]

        for subch in sub_chunks:
            tk_iter = re.finditer(
                r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
                subch
            )
            words += [tuple(bytes(match.group(), encoding="utf-8")) for match in tk_iter]
        return words
    
    def _pretokenize_parallell(self, in_queue, out_list):
        while True:
            chunk = in_queue.get()
            if chunk is None:
                return
            chunk = chunk.decode("utf-8", errors="ignore")
            counts = self._pretokenize_chunk(chunk)
            out_list.append(counts)

    def pretokenize(self, file: Path, num_processes: int = 4):
        """Read file and store token counts.

        Parameters
        ----------
        file : Path
            Path to file
        num_processes : int, optional
            Number of processes, by default 4
        """
        counts = CountDict()
        if num_processes == 1:
            with open(file, "rb") as f:
                chunk = f.read()
                # Run pre-tokenization on your chunk and store the counts for each pre-token
                chunk = chunk.decode("utf-8", errors="ignore")
                for ptk in self._pretokenize_chunk(chunk):
                    counts.update({ptk: 1})
        else:
            manager = mp.Manager()
            out_list = manager.list()
            in_queue = manager.Queue()

            pool = []
            for _ in range(num_processes):
                p = mp.Process(target=self._pretokenize_parallell, args=(in_queue, out_list))
                p.start()
                pool.append(p)

            with open(file, "rb") as f:
                boundaries = self._find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
                
                for start, end in zip(boundaries[:-1], boundaries[1:]):
                    f.seek(start)
                    chunk = f.read(end - start)
                    in_queue.put(chunk)

            # Put None to end process
            for _ in range(num_processes):
                in_queue.put(None)

            for p in pool:
                p.join()

            for res in out_list:
                for ptk in res:
                    counts.update({ptk: 1})

        self.counts.update(counts)
        self.counts = self.counts.sort()

    def _adj_pairs(self, sequence):
        if len(sequence) <= 1:
            return []
        return [(x1, x2) for x1, x2 in zip(sequence[:-1], sequence[1:])]

    def _train_loop(self, byte_pairs: CountDict, vocab: dict, num_merges: int):

        def _contains(target: tuple[int], check: tuple[int]):
            return [
                ind for ind in range(len(target) - 1)
                if target[ind] == check[0] and target[ind + 1] == check[1]
            ]

        def _update_key(old_key: tuple[int], replace_val: int, replace_at: list[int]):
            ind = 0
            new_key = []
            while ind < len(old_key):
                if ind in replace_at:
                    new_key.append(replace_val)
                    ind += 1
                else:
                    new_key.append(old_key[ind])
                ind += 1
            return tuple(new_key)

        def _count_byte_pair_change(new_key, old_key):
            new_byte_pairs = CountDict()
            for pair in self._adj_pairs(new_key):
                new_byte_pairs.update({pair: 1})
            old_byte_pairs = CountDict()
            for pair in self._adj_pairs(old_key):
                old_byte_pairs.update({pair: 1})
            return new_byte_pairs - old_byte_pairs

        merges = []
        for iter in tqdm(range(num_merges)):
            # Find the lexicographically greatest, most occuring pair
            max_val = -1
            max_byte_pairs = {}
            for key, value in byte_pairs.items():
                if max_val == -1:
                    max_val = value
                if value >= max_val:
                    max_byte_pairs[(vocab[key[0]], vocab[key[1]])] = key
                else:
                    break
            max_pair = max_byte_pairs[max(max_byte_pairs)]

            # Next free position in vocabulary, this will be the encoding value
            # for the byte pair
            tk_ind = len(vocab)

            # Find byte pairs affected by the comming merge
            #affected_pairs = CountDict()
            edit_keys = []
            for key, count in self.counts.items():
                found = _contains(key, max_pair)
                if found:
                    new_key = _update_key(key, tk_ind, found)
                    edit_keys.append((key, new_key))
                    # Update counts in byte pairs
                    btp_change = _count_byte_pair_change(new_key, key)
                    for btp, change in btp_change.items():
                        byte_pairs[btp] = byte_pairs.get(btp, 0) + change * count
                        if byte_pairs[btp] == 0:
                            # Remove byte pair
                            del byte_pairs[btp]
                        elif byte_pairs[btp] < 0:
                            raise ValueError("Something has gone terribly wrong...")                            

            # Update pretoken counts
            for (old_key, new_key) in edit_keys:
                self.counts[new_key] = self.counts.pop(old_key)

            # Re-sort byte pairs
            byte_pairs = byte_pairs.sort()

            # Add to merge list
            merges.append((vocab[max_pair[0]], vocab[max_pair[1]]))
            # Update vocabulary
            vocab[len(vocab)] = vocab[max_pair[0]] + vocab[max_pair[1]]
        return vocab, merges

    def train(self, input_path: str, vocab_size: int, special_tokens: list[str]):

        # Pre-tokenize
        st_time = time()
        self.pretokenize(Path(input_path), self.num_processes)
        print(f"Pretokenization done, time: {time() - st_time}")

        # Init vocabulary
        vocab = {}
        for i in range(256):
            vocab[i] = bytes([i])

        # Init byte pairs
        byte_pairs = CountDict()
        for tk_bytes, count in self.counts.items():
            for b1, b2 in zip(tk_bytes[:-1], tk_bytes[1:]):
                byte_pairs.update({(b1, b2): count})
        byte_pairs = byte_pairs.sort()

        vocab, merges = self._train_loop(
            byte_pairs,
            vocab,
            num_merges=max(vocab_size - 256 - len(special_tokens), 0)
        )
        i = len(vocab)
        for spt in special_tokens:
            vocab[i] = bytes(spt, encoding='utf-8')
            i += 1

        self.vocab = vocab
        self._vocab_list = list(vocab.values())
        self.merges = merges
    
    def _tokenize(self, pre_token: list[bytes]) -> list[bytes]:
        if len(pre_token) <= 1:
            return pre_token
        possible_merges = [(pre_token[i], pre_token[i + 1]) for i in range(len(pre_token) - 1)]
        for merge in self.merges:
            if merge not in possible_merges:
                continue
            ind = possible_merges.index(merge)
            pre_token[ind] += pre_token[ind + 1]
            del pre_token[ind + 1]
            if len(pre_token) <= 1:
                return pre_token
            possible_merges = [(pre_token[i], pre_token[i + 1]) for i in range(len(pre_token) - 1)]
        return pre_token

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        pre_tks = self._pretokenize_chunk(text)
        out_list = []
        for b_tokens in pre_tks:
            # Convert from int to bytes
            tokens = self._tokenize([bytes([byte]) for byte in b_tokens])
            out_list += [self._vocab_list.index(tk) for tk in tokens] # ValueError if not preset!
        return out_list

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator
        that lazily yields token IDs. This is required for memory-efficient tokenization
        of large files that we cannot directly load into memory."""
        pass

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        bts =  b"".join([self.vocab[t_id] for t_id in ids])
        return bts.decode()

if __name__=='__main__':
    base_dir = Path.cwd()
    path = base_dir / "data" / "TinyStoriesV2-GPT4-train.txt"
    # tk = Tokenizer(num_processes=4, special_tokens=["<|endoftext|>"])

    # tk.train(path, 10000)
    # with open('tinystories_vocab.pkl', 'wb') as f:
    #     pickle.dump(tk.vocab, f)
    # with open('tinystories_merges.pkl', 'wb') as f:
    #     pickle.dump(tk.merges, f)
    # import pdb; pdb.set_trace()
    tk = Tokenizer.from_files('tinystories_vocab.pkl', 'tinystories_merges.pkl',
                              special_tokens=["<|endoftext|>"])
    enc = tk.encode("encode this sentence plz")
    print(tk.decode(enc))
    import pdb; pdb.set_trace()
