import os
from typing import BinaryIO
import regex as re
import multiprocessing as mp
from pathlib import Path
from time import time


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

class Tokenizer:

    def __init__(self, special_tokens: list[str] = None, num_processes: int = 4):
        self.counts = CountDict()
        self.special_tokens = special_tokens if special_tokens else []
        self.num_processes = num_processes

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
    
    def _pretokenize_chunk(self, chunk):
        chunk = chunk.decode("utf-8", errors="ignore")
        counts = CountDict()
        if self.special_tokens:
            sub_chunks = re.split(rf"{'|'.join(self.special_tokens)}", chunk)
        else:
            sub_chunks = [chunk]

        for subch in sub_chunks:
            tk_iter = re.finditer(
                r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
                subch
            )
            for match in tk_iter:
                word = match.group()
                counts.update({tuple(bytes(word, encoding="utf-8")): 1})
        return counts
    
    def _pretokenize_parallell(self, in_queue, out_list):
        while True:
            chunk = in_queue.get()
            if chunk is None:
                return
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
                counts.update(self._pretokenize_chunk(chunk))
        else:
            manager = mp.Manager()
            out_list = manager.list()
            in_queue = manager.Queue()

            pool = []
            print("start processes")
            for _ in range(num_processes):
                p = mp.Process(target=self._pretokenize_parallell, args=(in_queue, out_list))
                p.start()
                pool.append(p)

            print("produce data")
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

            print("update counts")
            for res in out_list:
                counts.update(res)

        self.counts.update(counts)
        self.counts = self.counts.sort()

    def _train_loop(self, byte_pairs: CountDict, vocab: dict, num_merges: int):

        def _contains(target: tuple[int], check: tuple[int]):
            found = []
            for ind in range(len(target) - 1):
                if check[0] == target[ind] and check[1] == target[ind + 1]:
                    found.append(ind)
            return found
        
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
            assert len(old_key) - len(replace_at) == len(new_key)
            return tuple(new_key)

        merges = []
        for iter in range(num_merges):
            # Find the lexicographically greatest, most occuring pair
            max_val = -1
            max_byte_pairs = []
            for key, value in byte_pairs.items():
                if max_val == -1:
                    max_val = value
                if value >= max_val:
                    max_byte_pairs.append(key)
                else:
                    break
            max_pair = max(max_byte_pairs)

            # Next free position in vocabulary, this will be the encoding value
            # for the byte pair
            tk_ind = len(vocab)

            # Find byte pairs affected by the comming merge
            affected_pairs = CountDict()
            edit_keys = []
            for key, count in self.counts.items():
                found = _contains(key, max_pair)
                for ind in found:
                    if ind != 0:
                        if ind == len(key) - 1:
                            affected_pairs.update({(key[ind - 1], key[ind]): count})
                        else:
                            affected_pairs.update({(key[ind - 1], key[ind], key[ind + 1]): count})
                    if ind < len(key) - 2:
                        affected_pairs.update({(key[ind], key[ind + 1], key[ind + 2]): count})
                if found:
                    new_key = _update_key(key, tk_ind, found)
                    edit_keys.append((key, new_key))

            # Update pretoken counts
            for (old_key, new_key) in edit_keys:
                self.counts[new_key] = self.counts.pop(old_key)

            # Update counts in byte pairs
            for key, count in affected_pairs.items():
                if (key[0], key[1]) == max_pair:
                    edit_key = (key[1], key[2])
                    new_key = (tk_ind, key[2])
                elif (key[1], key[2]) == max_pair:
                    edit_key = (key[0], key[1])
                    new_key = (key[0], tk_ind)
                else:
                    raise IndexError("Uh Oh")
                if edit_key not in byte_pairs:
                    raise KeyError("Something has gone terribly wrong...")
                if byte_pairs[edit_key] == count:
                    # Replace it
                    del byte_pairs[edit_key]
                else:
                    byte_pairs[edit_key] = byte_pairs[edit_key] - count
                byte_pairs[new_key] = count
            # Remove merged pair
            del byte_pairs[max_pair]
            # Re-sort it
            byte_pairs = byte_pairs.sort()

            # Add to merge list
            merges.append(max_pair)
            # Update vocabulary
            vocab[len(vocab)] = vocab[max_pair[0]] + vocab[max_pair[1]]
        return vocab, merges

    def train(self, input_path: str, vocab_size: int, special_tokens: list[str]):
        self.special_tokens = [re.escape(spt) for spt in special_tokens]

        # Pre-tokenize
        self.pretokenize(Path(input_path), self.num_processes)

        # Init vocabulary
        # vocab = {i: spt for i, spt in enumerate(special_tokens)}
        # offset = len(special_tokens
        vocab = {}
        for i in range(256):
            vocab[i] = bytes([i])

        # Init byte pairs
        byte_pairs = CountDict()
        for tk_bytes, count in self.counts.items():
            for b1, b2 in zip(tk_bytes[:-1], tk_bytes[1:]):
                byte_pairs.update({(b1, b2): count})
        byte_pairs = byte_pairs.sort()

        vocab, merges = self._train_loop(byte_pairs, vocab, num_merges=max(vocab_size - 256, 0))
        i = len(vocab)
        for spt in special_tokens:
            vocab[i] = spt
            i += 1

        self.vocab = vocab
        return merges


if __name__=='__main__':
    base_dir = Path.cwd()
    path = base_dir / "data" / "TinyStoriesV2-GPT4-valid.txt"
    tk = Tokenizer(num_processes=8)
    import pdb; pdb.set_trace()
    merges = tk.train(path, 1000, special_tokens=["<|endoftext|>"])
    import pdb; pdb.set_trace()
