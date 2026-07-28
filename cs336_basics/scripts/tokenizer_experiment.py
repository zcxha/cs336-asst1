import pathlib
import numpy as np
from os.path import getsize
# 2.7 experiments
# a. What is tokenizer trained on tinystories' compression ratio? (bytes/token)\
DATADIR = (pathlib.Path(__file__).resolve().parent.parent.parent) / "data" 
print(DATADIR)
from cs336_basics.bpe_tokenizer import Tokenizer
def get_compression_ratio(s: str, token: list[int]) -> float:
    a = len(s.encode('utf-8'))
    b = len(token)
    return float(a) / b
# with open(DATADIR / "TinyStoriesV2-GPT4-valid.txt", "r") as f:
#     content = f.read()

# texts = content.strip().split("<|endoftext|>")

# compression_ratio = 0
# for s in texts[:10]:
#     token = tokenizer.encode(s)
#     compression_ratio += get_compression_ratio(s, token)
#     decoded = tokenizer.decode(token)
#     assert decoded == s
# print(compression_ratio / len(texts[:10]))

# total_bytes = sum(len(s.encode('utf-8')) for s in texts)
# print(f"{total_bytes} bytes")
# import time
# import cProfile
# with open(DATADIR / "TinyStoriesV2-GPT4-valid.txt", "r") as f:
#     content = f.read()
# start = time.perf_counter()
# cProfile.run('tokenizer.encode(content)')
# end = time.perf_counter()
# print(f"{end - start} s")

# throughput = total_bytes / (end - start)
# print(f"{throughput} bytes / s")

# 2.7 d tokenize all dataset.
import time
from tqdm import tqdm
# tinystories dataset
# with open(DATADIR / "TinyStoriesV2-GPT4-valid.txt", "r") as f:
#     content = f.read()
# start = time.perf_counter()
# token_ids = tokenizer.encode(content)
# end = time.perf_counter()
# print(f'[valid] took {end - start}s')
# data = np.array(token_ids, dtype=np.uint16)
# np.save(DATADIR / "TinyStoriesV2-GPT4-valid.npy", data)

def get_owt_by_tinystories_tokenizer_compression_ratio():
    tokenizer = Tokenizer.from_files(DATADIR / "vocab_tinystories.pkl", DATADIR / "merge_tinystories.pkl", special_tokens=["<|endoftext|>"])
    with open(DATADIR / "owt_valid.txt", "r") as f:
        content = f.read()
    tokens = tokenizer.encode(content)
    return get_compression_ratio(content, tokens)  

def get_tinystories_compression_ratio():
    tokenizer = Tokenizer.from_files(DATADIR / "vocab_tinystories.pkl", DATADIR / "merge_tinystories.pkl", special_tokens=["<|endoftext|>"])
    with open(DATADIR / "TinyStoriesV2-GPT4-valid.txt", "r") as f:
        content = f.read()
    tokens = tokenizer.encode(content)
    return get_compression_ratio(content, tokens)

def get_owt_compression_ratio():
    tokenizer = Tokenizer.from_files(DATADIR / "vocab_owt.pkl", DATADIR / "merge_owt.pkl", special_tokens=["<|endoftext|>"])
    with open(DATADIR / "owt_valid.txt", "r") as f:
        content = f.read()
    tokens = tokenizer.encode(content)
    return get_compression_ratio(content, tokens)

def encode_file(vocab_path: pathlib.Path, merge_path: pathlib.Path, corpus_path: pathlib.Path, output_path: pathlib.Path, cr):
    estimated_iterations = getsize(corpus_path) / cr
    tokenizer = Tokenizer.from_files(vocab_path, merge_path, special_tokens=["<|endoftext|>"])
    num_tokens = 0
    start = time.perf_counter()
    with open(corpus_path, "r") as f:
        for token_id in tqdm(tokenizer.encode_iterable(f), desc="encode_counting progress", total=estimated_iterations):
            num_tokens += 1
    end = time.perf_counter()

    print(f'[train] tokenizer counting took {end - start}s')

    memmap = np.memmap(output_path, dtype=np.uint16, mode="w+", shape=(num_tokens,))

    with open(corpus_path, "r") as f:
        i = 0
        for token_id in tqdm(tokenizer.encode_iterable(f), desc="encode progress", total=num_tokens):
            memmap[i] = token_id
            i += 1

    memmap.flush()

def get_tokenizer_compression_ratio(vocab: pathlib.Path, merge: pathlib.Path, corpus: pathlib.Path):
    tokenizer = Tokenizer.from_files(vocab, merge, special_tokens=["<|endoftext|>"])
    with open(corpus, "r") as f:
        content = f.read()
    tokens = tokenizer.encode(content)
    return get_compression_ratio(content, tokens)  

if __name__ == '__main__':
    # encode_tinystories_train()
    # owt_cr = 4.367378879789006
    # tiny_stories_cr = get_tinystories_compression_ratio()
    # hybrid_cr = get_owt_by_tinystories_tokenizer_compression_ratio()

    encode_file(DATADIR / "vocab_tinystories.pkl", DATADIR / "merge_tinystories.pkl", DATADIR / "TinyStoriesV2-GPT4-valid.txt", DATADIR / "TinyStoriesV2-GPT4-valid.map", 4)

    # encode_file(DATADIR / "vocab_owt.pkl", DATADIR / "merge_owt.pkl", DATADIR / "owt_train.txt", DATADIR / "owt_train.map", owt_cr)
    # encode_file(DATADIR / "vocab_owt.pkl", DATADIR / "merge_owt.pkl", DATADIR / "owt_valid.txt", DATADIR / "owt_valid.map", owt_cr)