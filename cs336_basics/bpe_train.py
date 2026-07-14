import os
from typing import BinaryIO
from collections import defaultdict
import regex as re

class Token:
    def __init__(self, symbols: tuple[bytes, ...], count: int):
        self.symbols = symbols
        self.count = count

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
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
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


def count_adjacent_pairs(tokens: list[Token]) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[int]]]:
    """
    count the adj pairs count cache, and the locations of pair
    """
    counts = defaultdict(int)
    pair_locations = defaultdict(set[int])
    for index, token in enumerate(tokens):
        for byte1, byte2 in zip(token.symbols, token.symbols[1:]):
            counts[(byte1, byte2)] += token.count
            pair_locations[(byte1, byte2)].add(index) # ensuring each Token_{i} is different(the dict has ensured that)
    return counts, pair_locations

def do_merge(tokens: list[Token], pair: tuple[bytes, bytes], pair_locations: dict[tuple[bytes,bytes], set[int]], counts: dict[tuple[bytes, bytes], int]):
    """
        1. locate where pair exists in tokens by pair_locations
        2. merge the pair in these tokens, update tokens token by token
        3. update counts by the merge process, and update pair_locations by the merge process
    """
    for idx in pair_locations[pair]:
        token_count = tokens[idx].count
        token_symbols = tokens[idx].symbols
        new_symbols = []
        i = 0
        counts_changes = defaultdict(int)
        while i < len(token_symbols):
            if i + 1 < len(token_symbols) and token_symbols[i] == pair[0] and token_symbols[i+1] == pair[1]:

                # count change is linear.
                # decrease old adjacent neighburing counts.
                counts_changes[pair] -= 1 * token_count
                if i > 0:
                    left_adj = (new_symbols[-1], token_symbols[i])
                    counts_changes[left_adj] -= 1 * token_count
                if i + 2 < len(token_symbols):
                    right_adj = (token_symbols[i+1], token_symbols[i+2])
                    counts_changes[right_adj] -= 1 * token_count
                
                new_symbols.append(pair[0]+pair[1]) # pair[0] is key[i], pair[1] is key[i+1]

                # increase new adjacent neighburing counts. notice that this is linear processing
                # so using new-old bounding, keeps the invariant of loop
                if len(new_symbols) > 1:
                    # the invariant is that, tail is the just merged pair
                    left_adj = (new_symbols[-2], new_symbols[-1])
                    counts_changes[left_adj] += 1 * token_count
                if i + 2 < len(token_symbols):
                    # the invariant is that, i + 2 is the linear process's right neighbor
                    right_adj = (new_symbols[-1], token_symbols[i+2])
                    counts_changes[right_adj] += 1 * token_count
                i += 2
            else:
                new_symbols.append(token_symbols[i])
                i += 1
        tokens[idx].symbols = new_symbols
        # the new adjacent counts, finally added to locations.
        # and update variation to counts
        for k, v in counts_changes.items():
            counts[k] += v
            if v > 0:
                pair_locations[k].add(idx)


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
    # add: dict->list
    tokens : list[Token] = []
    for symbols, count in pretoks.items():
        tokens.append(Token(symbols, count))

    # 3. run BPE merge
    num_merges = vocab_size - 256 - len(special_tokens)

    # 1. count once, and see index and cache.
    counts, pair_locations = count_adjacent_pairs(tokens)

    for i in range(num_merges):
        # 2. Find the most common pair
        pair = max(counts, key=lambda k: (counts[k], k))
        # 3. add new vocab
        new_vocab_idx = i + 256 + len(special_tokens)
        vocab[new_vocab_idx] = pair[0] + pair[1]

        # 4. do merge
        merge.append(pair)
        do_merge(tokens, pair, pair_locations, counts)
    
    return vocab, merge
