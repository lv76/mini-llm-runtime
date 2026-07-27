import torch
import regex as re
from collections import Counter

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def merge_token_sequence(tokens, pair):
    merge = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
            merge.append(tokens[i] + tokens[i+1])
            i += 2
        else:
            merge.append(tokens[i])
            i += 1
    return tuple(merge)

def bpe(input_path, vocab_size, special_tokens, **kwargs):
    vocab = {i: bytes([i]) for i in range(256)}

    for special_token in special_tokens:
        vocab[len(vocab)] = special_token.encode()
    
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    chunks = re.split(f"({pattern})", text)

    word_freq = Counter()

    for chunk in chunks:
        if chunk in special_tokens:
            continue
        for match in re.finditer(PAT, chunk):
           word = match.group(0)
           tokens = tuple(bytes([b]) for b in word.encode("utf-8"))
           word_freq[tokens] += 1

    merges = []
    while len(vocab) < vocab_size:
        pair_counts = Counter()
        for tokens, freq in word_freq.items():
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                pair_counts[pair] += freq

        if(len(pair_counts) == 0):
            break
        best_pair = max(pair_counts, key = lambda pair: (pair_counts[pair], pair))
        merges.append(best_pair)
        vocab[len(vocab)]  = best_pair[0] + best_pair[1]

        new_word_freq = Counter()
        for tokens, freq in word_freq.items():
          new_tokens = merge_token_sequence(tokens, best_pair)
          new_word_freq[new_tokens] += freq
        word_freq = new_word_freq
    
    return vocab, merges
