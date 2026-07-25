import pathlib
import numpy as np
# 2.7 experiments
# a. What is tokenizer trained on tinystories' compression ratio? (bytes/token)\
DATADIR = (pathlib.Path(__file__).resolve().parent.parent.parent) / "data" 
print(DATADIR)
from cs336_basics.bpe_tokenizer import Tokenizer
def get_compression_ratio(s: str, token: list[int]) -> float:
    a = len(s.encode('utf-8'))
    b = len(token)
    return float(a) / b

tokenizer = Tokenizer.from_files(DATADIR / "vocab_tinystories.pkl", DATADIR / "merge_tinystories.pkl", special_tokens=["<|endoftext|>"])

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
# tinystories dataset
# with open(DATADIR / "TinyStoriesV2-GPT4-valid.txt", "r") as f:
#     content = f.read()
# start = time.perf_counter()
# token_ids = tokenizer.encode(content)
# end = time.perf_counter()
# print(f'[valid] took {end - start}s')
# data = np.array(token_ids, dtype=np.uint16)
# np.save(DATADIR / "TinyStoriesV2-GPT4-valid.npy", data)

# count total number of tokens
num_tokens = 0
start = time.perf_counter()
with open(DATADIR / "TinyStoriesV2-GPT4-train.txt", "r") as f:
    for token_id in tokenizer.encode_iterable(f):
        num_tokens += 1
end = time.perf_counter()

print(f'[train] tokenizer took {end - start}s')

memmap = np.memmap(DATADIR / "TinyStoriesV2-GPT4-train.map", dtype=np.uint16, mode="w+", shape=(num_tokens,))

with open(DATADIR / "TinyStoriesV2-GPT4-train.txt", "r") as f:
    i = 0
    for token_id in tokenizer.encode_iterable(f):
        memmap[i] = token_id

memmap.flush()
# token_ids = tokenizer.encode(content)
# data = np.array(token_ids, dtype=np.uint16)
# np.save(DATADIR / "TinyStoriesV2-GPT4-train.npy", data)

