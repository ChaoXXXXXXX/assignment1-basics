import os
import regex
import time
import psutil
from collections import defaultdict
from typing import List, Tuple
from collections import defaultdict
from typing import Iterable,Iterator,List,Set,Tuple

class Tokenizer:
    def __init__(self,vocab:dict[int,bytes], merges:list[tuple[bytes,bytes]], special_tokens:list[str] | None = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.id_to_token = vocab
        self.token_to_id = {v: k for k, v in vocab.items()}
        #merges的优先级
        self.merges_priority = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        Loads the tokenizer from the given vocabulary and merges files.

        Args:
           vocab_filepath (str): Path to the vocabulary file (JSON format).
           merges_filepath (str): Path to the merges file.
           special_tokens (list[str] | None): A list of special tokens for the tokenizer.

        Returns:
            Tokenizer: A tokenizer instance.
        """
        import json
        with open(vocab_filepath, 'r', encoding='utf-8') as f:
            # Need to convert keys back to int because JSON stores keys as strings
            vocab_raw = json.load(f)
            vocab = {int(k): v.encode('utf-8') if isinstance(v, str) else bytes(v) for k, v in vocab_raw.items()}

        merges = []
        with open(merges_filepath, 'r', encoding='utf-8') as f:
            for line in f:
                pair = tuple(line.strip().split())
                if len(pair) == 2:
                    merges.append((pair[0].encode('utf-8'), pair[1].encode('utf-8')))
        
        return cls(vocab, merges, special_tokens)




        
    def merge_tokens(self,seq:tuple[bytes,...],pair:tuple[bytes,bytes]):
        A,B = pair
        out = []
        i = 0
        while i < len(seq):
            if seq[i] == A and i+1 < len(seq) and seq[i+1] == B:
                out.append(A+B)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return tuple(out)

    def get_pbe_merges(self,piece: bytes) -> list[bytes]:
        """
        处理非特殊符号的字节段word,进行pbe编码
        """
        symbols = [bytes([b]) for b in piece]
        while True:
            #记录合并对
            pairs = set()
            for i in range(len(symbols)-1):
                pair = (symbols[i],symbols[i+1])
                if pair in self.merges_priority:
                    pairs.add(pair)
            if not pairs:
                break
            
            #选择优先级最高的合并对
            best_pair = min(pairs, key=lambda x: self.merges_priority[x])

            new_symbols = self.merge_tokens(tuple(symbols),best_pair)
            symbols = new_symbols   

        return symbols
            



    def encode(self,text:str) -> list[int]:
        #预分词处理
        pre_tokens = []
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        sorted_special_tokens = sorted(self.special_tokens, key=lambda x: len(x), reverse=True)
        special_tokens_pattern = '(' + '|'.join(map(regex.escape, sorted_special_tokens)) + ')'
        if self.special_tokens:
            chunks = regex.split(special_tokens_pattern, text)
        else:
            chunks = [text]
        
        final_tokens = []
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_tokens:
                final_tokens.append(self.token_to_id[chunk.encode('utf-8')])
            else:

                for word in regex.findall(PAT,chunk):
                    if not word:
                        continue
                    merged_pieces = self.get_pbe_merges(word.encode('utf-8'))
                    for piece in merged_pieces:
                        final_tokens.append(self.token_to_id[piece])

        return final_tokens
            

    def encode_iterable(self, iterable: Iterable[str]) ->Iterator[int]:
        for text in iterable:
            yield from self.encode(text)
    

    def decode(self,ids: list[int]) -> str:
        all_bytes = b''.join(self.vocab[id] for id in ids)
        return all_bytes.decode('utf-8',errors='replace')

                
        




        
