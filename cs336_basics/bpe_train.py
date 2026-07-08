import os
from typing import BinaryIO
from collections import defaultdict
import regex as re

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_tokens: list[bytes],
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    # assert isinstance(split_special_tokens, list[bytes]), "Must represent special token as a list of bytestring"

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
            flag = False
            for split_special_token in split_special_tokens:
                found_at = mini_chunk.find(split_special_token)
                if found_at != -1:
                    chunk_boundaries[bi] = initial_position + found_at
                    flag = True
                    break
            if flag == True:
                break
            initial_position += mini_chunk_size
    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

from multiprocessing import Pool

def pre_tok(start: int, end: int, input_path: str, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    """
    the each-process processing function of pre-tokenization.
    parallelized reading chunks and processing
    """
    # result: dict[tuple[bytes, ...], int]
    result = defaultdict(int)
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode('utf-8', errors="ignore")
        parts = re.split("|".join(map(re.escape, special_tokens)), chunk)

        # pretokenizes each part and count occurrences.
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+'"""
        for part in parts:
            for pretok in re.finditer(PAT, part):
                pretok = pretok.group()
                key = tuple(bytes([x]) for x in pretok.encode('utf-8'))
                result[key] += 1
    
    return result
            


def pre_tokenization(input_path: str | os.PathLike, special_tokens: list[str]) -> dict[tuple[bytes, ...], int]:
    """
    pre splits the corpus by chunks(which is parallel) and run pre-tokenization
    then return each token(the pre tokenized word)'s occurence count dict.
    """
    ## Usage
    with open(input_path, "rb") as f:
        num_processes = 4
        pool = Pool(num_processes)
        all_jobs = []
        boundaries = find_chunk_boundaries(f, num_processes, [x.encode('utf-8') for x in special_tokens])

        for start, end in zip(boundaries[:-1], boundaries[1:]):
            job = pool.apply_async(pre_tok, args=(start, end, input_path, special_tokens,))
            all_jobs.append(job)

        dicts = [job.get() for job in all_jobs]

        pool.close()
        pool.join()

        results = defaultdict(int)

        for d in dicts:
            for k,v in d.items():
                results[k] += v
        return results


def count_adjacent_pairs(pretoks: dict[tuple[bytes,...],int]) -> dict[tuple[bytes, bytes], int]:
    """
    count the adj pairs count
    """
    counts = defaultdict(int)
    for k,v in pretoks.items():
        for byte1, byte2 in zip(k, k[1:]):
            counts[(byte1, byte2)] += v
    return counts

def do_merge(pretoks: dict[tuple[bytes,...],int], pair: tuple[bytes, bytes]) -> dict[tuple[bytes,...],int]:
    """
    merge pretoks by pair, and return merged dict
    """
    new_toks = {}
    for key,value in pretoks.items():
        if pair[0] not in key and pair[1] not in key:
            new_toks[key] = value
            continue
        new_key = []

        i = 0
        while i < len(key):
            if i + 1 < len(key) and key[i] == pair[0] and key[i+1] == pair[1]:
                new_key.append(pair[0]+pair[1])
                i += 2
            else:
                new_key.append(key[i])
                i += 1
        new_toks[tuple(new_key)] = value
    return new_toks

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab: dict[int, bytes] = {}
    merge: list[tuple[bytes, bytes]] = []

    # 1. pre construct a vocab
    for i, x in enumerate(special_tokens):
        vocab[i] = x.encode('utf-8')
    for i in range(len(special_tokens), len(special_tokens) + 256):
        vocab[i] = bytes([i - len(special_tokens)])
    
    # 2. run pretokenization
    pretoks: dict[tuple[bytes,...], int] = pre_tokenization(input_path, special_tokens)
    
    # 3. run BPE merge
    
    num_merges = vocab_size - 256 - len(special_tokens)

    for i in range(num_merges):
        # 1. count
        counts = count_adjacent_pairs(pretoks)

        # 2. Find the most common pair
        pair = max(counts, key=lambda k: (counts[k], k))
        # 3. add new vocab
        new_vocab_idx = i + 256 + len(special_tokens)
        vocab[new_vocab_idx] = pair[0] + pair[1]

        # 4. do merge
        merge.append(pair)
        pretoks = do_merge(pretoks, pair)
    
    return vocab, merge
