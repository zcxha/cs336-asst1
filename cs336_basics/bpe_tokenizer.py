from __future__ import annotations

from typing import Iterable, Iterator
from collections import defaultdict
import pickle
import regex as re

class Tokenizer:
    vocab = defaultdict(bytes)
    vocab_rev = defaultdict(int)
    merges = []
    special_tokens = []
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes,bytes]], special_tokens: list[str] | None = None):
        """
        Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens.
        """
        max_idx = max(vocab)

        if special_tokens is not None:
            self.special_tokens = special_tokens
            for x in special_tokens:
                if x.encode('utf-8') not in vocab.values():
                    max_idx += 1
                    vocab[max_idx] = x.encode('utf-8')
        self.vocab = vocab
        self.merges = merges
        self.vocab_rev = {v: k for k,v in vocab.items()}
        self.merge_sets = set(self.merges)
        self.merge_dict = {k:v for v,k in enumerate(self.merges)}
        

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> Tokenizer:
        """
        Class method that constructs and returns a Tokenizer from a serialized vocabulary 
        and list of merges (in the same format that your BPE training code output) and (optionally) a list of special tokens.
        """
        vocab = defaultdict(bytes)
        merges = []
        with open(vocab_filepath, 'rb') as f: 
            vocab = pickle.load(f)
        with open(merges_filepath, 'rb') as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens = special_tokens)

    @staticmethod
    def _split(text: str, special_tokens_list: list[str]) -> list[str]:
        if special_tokens_list == []:
            return [text]
        else:
            special_tokens_list = sorted(special_tokens_list, key=lambda x: -len(x))
            escaped_tokens = [re.escape(token) for token in special_tokens_list]
            pattern = "(" + "|".join(escaped_tokens) + ")"
            parts = re.split(pattern, text)
            result = [p for p in parts if p != ""]
            return result

    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        res = []
        
        # pretokenize by splitted parts and merge dynamicly, then append to result

        # split by special tokens
        splited_parts = Tokenizer._split(text, self.special_tokens)
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        # part-by-part merge and insert special tokens.
        for part in splited_parts:
            if part in self.special_tokens:
                res.append(self.vocab_rev[part.encode('utf-8')])
            else:
                for pretoken in re.finditer(PAT, part):
                    pretoken = tuple(bytes([b]) for b in pretoken.group().encode('utf-8'))
                    while True:
                        cur_merge_idx = defaultdict(list[int])
                        for (idx, a),b in zip(enumerate(pretoken), pretoken[1:]):
                            cur_merge_idx[(a,b)].append(idx)
                        if not any(x in self.merge_sets for x in cur_merge_idx.keys()):
                            break
                        cur_merge_pair = min(cur_merge_idx.keys(), key = lambda x : self.merge_dict[x] if x in self.merge_sets else 1e42)
                        cur_merge_indices = cur_merge_idx[cur_merge_pair]
                        new_pretoken = []
                        i = 0
                        while i < len(pretoken):
                            if i in cur_merge_indices:
                                new_pretoken.append(cur_merge_pair[0]+cur_merge_pair[1])
                                i += 2
                            else:
                                new_pretoken.append(pretoken[i])
                                i += 1
                        pretoken = tuple(new_pretoken)
                    for vocab in pretoken:
                        res.append(self.vocab_rev[vocab])
        return res
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs.
        This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
        """
        for line in iterable:
            splited_parts = Tokenizer._split(line, self.special_tokens)
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            
            # part-by-part merge and insert special tokens.
            for part in splited_parts:
                if part in self.special_tokens:
                    yield self.vocab_rev[part.encode('utf-8')]
                else:
                    for pretoken in re.finditer(PAT, part):
                        pretoken = tuple(bytes([b]) for b in pretoken.group().encode('utf-8'))
                        while True:
                            cur_merge_idx = defaultdict(list[int])
                            for (idx, a),b in zip(enumerate(pretoken), pretoken[1:]):
                                cur_merge_idx[(a,b)].append(idx)
                            if not any(x in self.merge_sets for x in cur_merge_idx.keys()):
                                break
                            cur_merge_pair = min(cur_merge_idx.keys(), key = lambda x : self.merge_dict[x] if x in self.merge_sets else 1e42)
                            cur_merge_indices = cur_merge_idx[cur_merge_pair]
                            new_pretoken = []
                            i = 0
                            while i < len(pretoken):
                                if i in cur_merge_indices:
                                    new_pretoken.append(cur_merge_pair[0]+cur_merge_pair[1])
                                    i += 2
                                else:
                                    new_pretoken.append(pretoken[i])
                                    i += 1
                            pretoken = tuple(new_pretoken)
                        for vocab in pretoken:
                            yield self.vocab_rev[vocab]


    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        res = bytes()
        for id in ids:
            res += self.vocab[id]
        return res.decode('utf-8', errors="replace")