from helper import pair_stats, merge_pairs
from base import Tokenizer

class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def train(self, text, target_vocab_size, verbose = False):
        assert target_vocab_size >= 256, "Target vocab size must be at least 256 (for all byte values)"
        n_merges = target_vocab_size - 256
        tokens = list(map(int, text.encode('utf-8'))) # encode text as bytes and convert to list of ints
        if verbose:
            print(f"Initial token count: {len(tokens)}")
        for i in range(n_merges):
            counts = pair_stats(tokens)
            if not counts:
                break
            best_pair = max(counts, key=counts.get)
            if counts[best_pair] < 2:
                raise ValueError(f"No more pairs to merge after {i} merges. Consider reducing target_vocab_size or providing more training data.")
            new_idx = 256 + i
            self.merges[best_pair] = new_idx
            tokens = merge_pairs(tokens, best_pair, new_idx)
            if verbose:
                print(f"Merge {i+1}/{n_merges}: merged pair {best_pair} into index {new_idx}, count = {counts[best_pair]}, new token count: {len(tokens)}")
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
    
    def encode(self, text):
        tokens = list(map(int, text.encode('utf-8')))
        while len(tokens) >= 2:
            pairs = set(zip(tokens, tokens[1:]))
            matched_pair = min(pairs, key=lambda p: self.merges.get(p, float("inf")))
            if matched_pair not in self.merges:
                break
            matched_idx = self.merges[matched_pair]
            tokens = merge_pairs(tokens, matched_pair, matched_idx)
        return tokens