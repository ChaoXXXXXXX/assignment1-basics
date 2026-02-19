import os
from collections import Counter
import regex 
from collections import defaultdict
from typing import List, Tuple, Dict, Set


def merge_tokens(seq: tuple[bytes,...], pair: tuple[bytes,bytes]):
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
    

    

def Bpe_train(input_path: str, vocab_size: int, special_tokens: list[str], **kwargs) -> tuple[dict[int, bytes],list[tuple[bytes,bytes]]]:

    if not isinstance(input_path, str):
        raise ValueError("input_path must be a string")
    if not isinstance(vocab_size, int):
        raise ValueError("vocab_size must be an integer")
    if not isinstance(special_tokens, list):
        raise ValueError("special_tokens must be a list")
    if not isinstance(kwargs, dict):
        raise ValueError("kwargs must be a dictionary")

    # 初始化词表
    vocab: Dict[int,bytes] = {i:bytes([i]) for i in range(256)}
    vocab_current_size = 256

    # 初始化token频率表
    token_frequency_table = defaultdict(int)
    # 初始化existing_tokens
    existing_tokens: set[bytes] = set(vocab.values())

    #添加特殊符号到vocab
    for token in special_tokens:
        if len(vocab) >= vocab_size:
            break
        if token.encode('utf-8') not in existing_tokens: 
            vocab[vocab_current_size] = token.encode('utf-8')
            vocab_current_size += 1
            existing_tokens.add(token.encode('utf-8'))


    #加载文本
    try:
        with open(input_path, 'r',encoding="utf-8",errors = "ignore") as f:
            text = f.read()
    except FileNotFoundError:
        text = ""

    #分词
    #首先将special_tokens中的token进行转义
    chunks = regex.split('|'.join(map(regex.escape,special_tokens)),text)
    #然后进行小分割
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for chunk in chunks:
        for word in regex.findall(PAT,chunk):
            word_bytes = word.encode('utf-8')
            bytes_list = [bytes([x]) for x in word_bytes]  #将word_bytes转换为bytes_list
            token_frequency_table[tuple(bytes_list)] += 1 #统计token频率
    
    merges: List[Tuple[bytes,bytes]] = [] #用来记录合并的字节对
    
    #统计字节对频率
    pair_count = defaultdict(int)
    for token in token_frequency_table:
        for i in range(len(token)-1):
            pair_count[(token[i],token[i+1])] += token_frequency_table[token] 
    
    #开始BPE
    while len(vocab) < vocab_size:
        if not pair_count:
            break
        max_count = max(pair_count.values())
        #找到频率最高的字节对，可能有多个
        candidates = [k for k,v in pair_count.items() if v == max_count]
        #选择lexicographically biggest
        best_pair = max(candidates)
        #记录合并
        merges.append(best_pair)
        new_token_bytes = best_pair[0] + best_pair[1]
        vocab[vocab_current_size] = new_token_bytes
        vocab_current_size += 1
        #更新token_frequency_table
        new_token_frequency_table = defaultdict(int)
        for seq,freq in token_frequency_table.items():
            new_seq = merge_tokens(seq,best_pair)
            new_token_frequency_table[new_seq] += freq
        token_frequency_table = new_token_frequency_table
    
        #更新pair_count
        pair_count = defaultdict(int)
        for seq, freq in token_frequency_table.items():
            for i in range(len(seq) - 1):
                pair_count[(seq[i], seq[i+1])] += freq

        
        
    return vocab,merges





        
    

    
    


    


    
