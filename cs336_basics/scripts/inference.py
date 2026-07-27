import torch
import os
import typing
import argparse
import cs336_basics.scripts.model_config as config
from jaxtyping import Float
from torch import Tensor
from cs336_basics.prenorm_transformer_block import TransformerLM
from cs336_basics.bpe_tokenizer import Tokenizer
from cs336_basics.generate import decoding as generate
from pathlib import Path
def load_model(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module):
    obj = torch.load(src)
    model.load_state_dict(obj["model"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Inference")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("tokenizer_vocab_path", type=Path)
    parser.add_argument("tokenizer_merge_path", type=Path)
    parser.add_argument("--maximum_tokens_generate", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--top_p", type=float, default=0.8)
    args = parser.parse_args()
    tokenizer = Tokenizer.from_files(args.tokenizer_vocab_path, args.tokenizer_merge_path, special_tokens=config.special_tokens)
    model = TransformerLM(config.vocab_size, config.context_length, config.num_layers, config.d_model, config.num_heads, config.d_ff, config.theta)

    load_model(args.model_path, model)

    text = input("user input ")

    out_tokens = generate(model, tokenizer.encode(text), args.temperature, args.top_p, args.maximum_tokens_generate, tokenizer.encode(config.eos_token)[0])

    print(tokenizer.decode(out_tokens))