import os
from typing import BinaryIO, Iterable, Iterator
import regex as re
import multiprocessing as mp
from pathlib import Path
from time import time
from line_profiler import profile
import pickle
from tqdm import tqdm
import datasets
from collections import defaultdict


class KeyWrapper:
    def __init__(self, values):
        self.values = list(values)  # mutable

    def __hash__(self):
        # Still hashable! Cast to tuple for hashing
        return hash(tuple(self.values))

    def __eq__(self, other):
        return isinstance(other, KeyWrapper) and self.values == other.values

    def __repr__(self):
        return f"KeyWrapper({self.values})"
    
    def update(self, values):
        self.values = list(values)


class CountDict(dict):

    def __setitem__(self, key, value):
        if not isinstance(value, int):
            raise TypeError(f"Value for key '{key}' must be an int, got {type(value).__name__}")
        super().__setitem__(key, value)

    def update(self, *args, **kwargs):
        """Updates the dictonary, adds to all values of excisting keys."""
        temp = dict(*args, **kwargs) if kwargs else args[0]
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
        self.special_tokens = special_tokens if special_tokens else []
        self.special_tokens.sort(key=len, reverse=True)  # Longest first
        self.num_processes = num_processes
        self.vocab = vocab if vocab else {}
        self._vocab_list = list(self.vocab.values())
        self.merges = merges if merges else []
        self.windows_platform = True if os.name == 'nt' else False

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
    
    def _pretokenize_chunk(self, chunk: str) -> list[re.Scanner | str]:
        """Splits text chunks into suitable pre-tokens
        
        Returns
        -------
        list
            List with re.Scanner objects or strings for special characters
        """
        words = []
        if self.special_tokens:
            special_tks = [re.escape(spt) for spt in self.special_tokens]
            sub_chunks = re.split(rf"({'|'.join(special_tks)})", chunk)
        else:
            sub_chunks = [chunk]

        for subch in sub_chunks:
            if self.windows_platform:
                subch = re.sub(r"\r\n", "\n", subch)
            tk_iter = re.finditer(
                r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
                subch
            ) if subch not in self.special_tokens else subch
            words.append(tk_iter)
        return words
    
    def _pretokenize_parallell(self, in_queue, out_list):
        while True:
            chunk = in_queue.get()
            if chunk is None:
                return
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8", errors="ignore")
            counts = self._count_pretokens(self._pretokenize_chunk(chunk))
            out_list.append(counts)
    
    def _count_pretokens(self, ptks: list[re.Scanner]) -> CountDict:
        counts = CountDict()
        for ptk in ptks:
            if isinstance(ptk, str):
                continue  # no need to count special tokens
            for match in ptk:
                counts.update({tuple(bytes(match.group(), encoding="utf-8")): 1})
        return counts
    
    def pretokenize_from_ds(self, ds, num_processes=4):
        counts = CountDict()
        manager = mp.Manager()
        out_list = manager.list()
        in_queue = manager.Queue()

        pool = []
        for _ in range(num_processes):
            p = mp.Process(target=self._pretokenize_parallell, args=(in_queue, out_list))
            p.start()
            pool.append(p)

        for sample in ds:
            text = sample.get('text', '')
            if text:
                in_queue.put(text)

        # Put None to end process
        for _ in range(num_processes):
            in_queue.put(None)

        for p in pool:
            p.join()

        for res in out_list:
            counts.update(res)

        self.counts.update(counts)
        self.counts = self.counts.sort()

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
                counts = self._count_pretokens(self._pretokenize_chunk(chunk))
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
                counts.update(res)

        self.counts.update(counts)
        self.counts = self.counts.sort()

    def _adj_pairs(self, sequence):
        if len(sequence) <= 1:
            return []
        return [(x1, x2) for x1, x2 in zip(sequence[:-1], sequence[1:])]

    def _train_loop(self, byte_pairs: CountDict, pair_index: defaultdict[tuple, list[KeyWrapper]], num_merges: int):

        def _contains(target: tuple[int], check: tuple[int]):
            return [
                ind for ind in range(len(target) - 1)
                if target[ind] == check[0] and target[ind + 1] == check[1]
            ]

        def _update_key(old_key: tuple[int], replace_val: int, replace_at: list[int]) -> tuple:
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
            diff = defaultdict(int)
            # Subtract counts from old_key
            for i in range(len(old_key) - 1):
                pair = (old_key[i], old_key[i+1])
                diff[pair] -= 1
            # Add counts from new_key
            for i in range(len(new_key) - 1):
                pair = (new_key[i], new_key[i+1])
                diff[pair] += 1
            return diff

        merges = []
        for iter in tqdm(range(num_merges)):
            # Find the lexicographically greatest, most occuring pair
            max_val = -1
            max_byte_pairs = {}
            for key, value in byte_pairs.items():
                if value > max_val:
                    max_byte_pairs = {(self.vocab[key[0]], self.vocab[key[1]]): key}
                    max_val = value
                elif value == max_val:
                    max_byte_pairs[(self.vocab[key[0]], self.vocab[key[1]])] = key

            max_pair = max_byte_pairs[max(max_byte_pairs)]

            # Next free position in vocabulary, this will be the encoding value
            # for the byte pair
            tk_ind = len(self.vocab)

            # Find byte pairs affected by the comming merge
            edit_keys = []
            for key_obj in pair_index.pop(max_pair, KeyWrapper([])):
                key = tuple(key_obj.values)

                found = _contains(key, max_pair)
                if not found:
                    continue

                count = self.counts[key]
                # Get new key
                new_key = _update_key(key, tk_ind, found)
                edit_keys.append((key, new_key))

                # Update keys in fast lookup index
                key_obj.update(new_key)
                # Update counts in byte pairs and add new pairs to lookup
                btp_change = _count_byte_pair_change(new_key, key)
                for btp, change in btp_change.items():
                    byte_pairs[btp] = byte_pairs.get(btp, 0) + change * count
                    if byte_pairs[btp] == 0:
                        # Remove byte pair
                        del byte_pairs[btp]
                    elif byte_pairs[btp] < 0:
                        raise ValueError("Something has gone terribly wrong...")   
                    if change > 0:
                        pair_index[btp].append(key_obj)
            
            # Update pretoken counts
            for (old_key, new_key) in edit_keys:
                self.counts[new_key] = self.counts.pop(old_key)
            # Add to merge list
            merges.append((self.vocab[max_pair[0]], self.vocab[max_pair[1]]))
            # Update vocabulary
            self.vocab[len(self.vocab)] = self.vocab[max_pair[0]] + self.vocab[max_pair[1]]

        return merges

    def train(self, input_path: str, vocab_size: int):
        # Pre-tokenize
        if len(self.counts) == 0:
            st_time = time()
            self.pretokenize(Path(input_path), self.num_processes)
            print(f"Pretokenization done, time: {time() - st_time}")

        # Init vocabulary
        vocab = {i: bytes([i]) for i in range(256)}
        for spt in self.special_tokens:
            vocab[len(vocab)] = bytes(spt, encoding='utf-8')
        self.vocab = vocab

        # Init byte pairs and fast lookup:
        pair_index = defaultdict(list)
        byte_pairs = CountDict()
        for tk_bytes, count in self.counts.items():
            pi_key = KeyWrapper(tk_bytes)
            for b1, b2 in zip(tk_bytes[:-1], tk_bytes[1:]):
                byte_pairs.update({(b1, b2): count})
                pair_index[(b1, b2)].append(pi_key)

        merges = self._train_loop(
            byte_pairs, pair_index,
            num_merges=max(vocab_size - 256 - len(self.special_tokens), 0)
        )

        self._vocab_list = list(self.vocab.values())
        self.merges = merges
    
    def _tokenize(self, pre_token: list[bytes]) -> list[bytes]:
        if len(pre_token) <= 1:
            return pre_token
        possible_merges = [(pre_token[i], pre_token[i + 1]) for i in range(len(pre_token) - 1)]
        for merge in self.merges:
            if merge not in possible_merges:
                continue
            inds = [ind for ind, mrg in enumerate(possible_merges) if mrg == merge]
            for ind in sorted(inds, reverse=True):
                pre_token[ind] += pre_token[ind + 1]
                del pre_token[ind + 1]
                if len(pre_token) <= 1:
                    return pre_token
            possible_merges = [(pre_token[i], pre_token[i + 1]) for i in range(len(pre_token) - 1)]
        return pre_token

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        scanners = self._pretokenize_chunk(text)
        out_list = []
        for scr in scanners:
            # Convert from int to bytes
            if isinstance(scr, str):
                out_list.append(self._vocab_list.index(bytes(scr, encoding='utf-8')))
            else:
                for match in scr:
                    tokens = self._tokenize([bytes([c]) for c in match.group().encode("utf-8")])
                    out_list += [self._vocab_list.index(tk) for tk in tokens] # ValueError if not present!
        return out_list

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator
        that lazily yields token IDs. This is required for memory-efficient tokenization
        of large files that we cannot directly load into memory."""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        bts =  b"".join([self.vocab[t_id] for t_id in ids])
        return bts.decode(errors='replace')

if __name__=='__main__':
    base_dir = Path.cwd()
    path = base_dir / "data" / "TinyStoriesV2-GPT4-train.txt"
    tk = Tokenizer(num_processes=30, special_tokens=["<|endoftext|>"])
    ds = datasets.load_dataset("stanford-cs336/owt-sample", split="train")
    tk.pretokenize_from_ds(ds, 30)
    tk.train("", 32000)
    with open('owt_vocab.pkl', 'wb') as f:
        pickle.dump(tk.vocab, f)
    with open('owt_merges.pkl', 'wb') as f:
        pickle.dump(tk.merges, f)
    # tk = Tokenizer.from_files('tinystories_vocab.pkl', 'tinystories_merges.pkl',
    #                           special_tokens=["<|endoftext|>"])
    enc = tk.encode("encode this sentence <|endoftext|> plz")
    print(tk.decode(enc))
    import pdb; pdb.set_trace()
