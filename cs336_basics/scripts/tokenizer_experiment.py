import pathlib
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

with open(DATADIR / "TinyStoriesV2-GPT4-valid.txt", "r") as f:
    content = f.read()

texts = content.strip().split("<|endoftext|>")

# compression_ratio = 0
# for s in texts[:10]:
#     token = tokenizer.encode(s)
#     compression_ratio += get_compression_ratio(s, token)
#     decoded = tokenizer.decode(token)
#     assert decoded == s
# print(compression_ratio / len(texts[:10]))

total_bytes = sum(len(s.encode('utf-8')) for s in texts)
print(f"{total_bytes} bytes")
import time
import cProfile
with open(DATADIR / "TinyStoriesV2-GPT4-valid.txt", "r") as f:
    content = f.read()
start = time.perf_counter()
cProfile.run('tokenizer.encode(content)')
end = time.perf_counter()
print(f"{end - start} s")

throughput = total_bytes / (end - start)
print(f"{throughput} bytes / s")