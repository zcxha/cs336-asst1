from cs336_basics.bpe_train import train_bpe
import pickle
import cProfile
import pathlib
DATADIR = (pathlib.Path(__file__).resolve().parent.parent.parent) / "data" 

def train_bpe_tinystories():
    vocab, merge = train_bpe(
        DATADIR / "TinyStoriesV2-GPT4-train.txt",
        10000,
        ["<|endoftext|>"]
        )
    
    with open(DATADIR / "bpe" / "vocab_tinystories.pkl", "wb") as f:
        pickle.dump(vocab, f)
    
    with open(DATADIR / "bpe" / "merge_tinystories.pkl", "wb") as f:
        pickle.dump(merge, f)

# def train_bpe_openwebtext():
#     vocab, merge = train_bpe(
#         "../data/TinyStoriesV2-GPT4-train.txt",
#         10000,
#         ["<|endoftext|>"]
#         )
    
#     with open("../data/vocab_tinystories.pkl", "wb") as f:
#         pickle.dump(vocab, f)
    
#     with open("../data/merge_tinystories.pkl", "wb") as f:
#         pickle.dump(merge, f)

if __name__ == '__main__':
    # for profiling
    # cProfile.run('train_bpe_tinystories()')
    train_bpe_tinystories()
