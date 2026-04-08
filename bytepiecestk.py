import regex as re
from regextk import RegexTokenizer
from helper import bytepiece_stats, merge_bytepieces, merge_pairs

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class BytePieceTokenizer(RegexTokenizer):
    def __init__(self, pattern=None):
        super().__init__(pattern = pattern)

    def _len_nested(self, l):
        return sum(len(x) for sl in l for x in sl)

    def train(self, text, target_vocab_size, verbose=False):
        """
        Character-aware byte-piece training.

        Initial state:
            each regex chunk -> list of characters
            each character -> list of raw byte ids

        Learned tokens:
            - full multibyte character pieces can collapse directly
            - adjacent atomic pieces can merge into larger pieces

        Vocabulary:
            - always keeps raw bytes 0..255
            - only adds learned surface byte-pieces
            - does NOT add meaningless intermediate partial-byte merges
        """
        assert target_vocab_size >= 256, "Target vocab size must be at least 256"
        n_merges = target_vocab_size - 256
        text_chunks = re.findall(self.regex_, text)
        tokens = [[list(c.encode('utf-8')) for c in chunk] for chunk in text_chunks]
        if verbose:
            print(f"Initial token count: {self._len_nested(tokens)}")
        for i in range(n_merges):
            counts = {}
            for chunk in tokens:
                counts = bytepiece_stats(chunk, counts)
            if (not counts) or (max(counts.values()) < 2):
                raise ValueError(f"No more pairs to merge after {i} merges. Consider reducing target_vocab_size or providing more training data.")
            best_tuple = max(counts, key=counts.get)
            new_idx = 256 + i
            self.merges[best_tuple] = new_idx
            tokens = [merge_bytepieces(chunk, best_tuple, new_idx) for chunk in tokens]
            if verbose:
                print(f"Merge {i+1}/{n_merges}: merged pair {best_tuple} into index {new_idx}, count = {counts[best_tuple]}, new token count: {self._len_nested(tokens)}")
        self._build_vocab()
        print(f"Training complete. Training compression ratio: {self.compression_ratio(text):.4f}")
    
    def _encode_chunk(self, chunk):
        tokens = []
        for c in chunk:
            b = c.encode("utf-8")
            idx = self.inverse_vocab.get(b)
            if idx is not None:
                tokens.append([idx])
            else:
                tokens.append(list(b))
        tokens = [x for c in tokens for x in c]
        while len(tokens) >= 2:
            pairs = set(zip(tokens, tokens[1:]))
            matched_tuple = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if matched_tuple not in self.merges:
                break
            matched_idx = self.merges[matched_tuple]
            tokens = merge_pairs(tokens, matched_tuple, matched_idx)
        return tokens