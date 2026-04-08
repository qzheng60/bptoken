import regex as re
from helper import pair_stats, merge_pairs
from base import Tokenizer
from typing import Sequence

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):
    def __init__(self, pattern=None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.regex_ = re.compile(self.pattern)

    def _len_nested(self, l):
        return sum(len(x) for x in l)

    def train(self, text, target_vocab_size, verbose=False):
        assert target_vocab_size >= 256, "Target vocab size must be at least 256"
        n_merges = target_vocab_size - 256
        text_chunks = re.findall(self.regex_, text)
        tokens = [list(chunk.encode('utf-8')) for chunk in text_chunks] # encode text as bytes and convert to list of ints
        if verbose:
            print(f"Initial token count: {self._len_nested(tokens)}")
        for i in range(n_merges):
            counts = {}
            for chunk in tokens:
                counts = pair_stats(chunk, counts)
            if (not counts) or (max(counts.values()) < 2):
                raise ValueError(f"No more pairs to merge after {i} merges. Consider reducing target_vocab_size or providing more training data.")
            best_pair = max(counts, key=counts.get)
            new_idx = 256 + i
            self.merges[best_pair] = new_idx
            tokens = [merge_pairs(chunk, best_pair, new_idx) for chunk in tokens]
            if verbose:
                print(f"Merge {i+1}/{n_merges}: merged pair {best_pair} into index {new_idx}, count = {counts[best_pair]}, new token count: {self._len_nested(tokens)}")
        self._build_vocab()
        print(f"Training complete. Training compression ratio: {self.compression_ratio(text):.4f}")
    
    def decode(self, token_idxs):
        bytes_list = []
        for idx in token_idxs:
            if idx in self.vocab:
                bytes_list.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                bytes_list.append(self.inverse_special_tokens[idx])
            else:
                raise ValueError(f"Token index {idx} not in vocabulary")
        text_bytes = b''.join(bytes_list)
        text = text_bytes.decode('utf-8', errors='replace')
        return text
    
    def _encode_chunk(self, chunk):
        tokens = list(map(int, chunk.encode('utf-8')))
        while len(tokens) >= 2:
            pairs = set(zip(tokens, tokens[1:]))
            matched_pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if matched_pair not in self.merges:
                break
            matched_idx = self.merges[matched_pair]
            tokens = merge_pairs(tokens, matched_pair, matched_idx)
        return tokens

    def _encode_nonspecial(self, text):
        text_chunks = re.findall(self.regex_, text)
        token_idxs = []
        for chunk in text_chunks:
            token_idxs.extend(self._encode_chunk(chunk))
        return token_idxs

    def _encode_special(self, text, special):
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        token_idxs = []
        for chunk in special_chunks:
            if chunk in special:
                token_idxs.append(self.special_tokens[chunk])
            else:
                token_idxs.extend(self._encode_nonspecial(chunk))
        return token_idxs

    def encode(self, text, allowed_special = "none"):
        if allowed_special == "none":
            special = {}
        elif allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens), "Special tokens found in text but allowed_special is set to 'none_raise'"
        elif isinstance(allowed_special, Sequence):
            special = {token: self.special_tokens[token] for token in allowed_special if token in self.special_tokens}
        else:
            raise ValueError(f"Invalid value for allowed_special: {allowed_special}")
        if special:
            return self._encode_special(text, special)
        else:
            return self._encode_nonspecial(text)