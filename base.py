import json
class Tokenizer:
    """Base class for tokenizers."""
    
    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = {} # (int, int) -> int
        self.pattern = "" # regex str pattern for tokenization
        self.special_tokens = {} # str -> int, e.g. {"<|PAD|>": 256, "<|UNK|>": 257, '<|endoftext|>': 100257}}
        self.inverse_special_tokens = {} # int -> bytes
        self._build_vocab() # int -> bytes, built from merges and special tokens

    @classmethod
    def from_model(cls, model_name):
        # Implementation for loading tokenizer from a pre-trained model
        tok = cls()
        tok.load(model_name)
        return tok

    def train(self, text, target_vocab_size,verbose = False):
        """Train the tokenizer on the given text to create a vocabulary of the specified size."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def decode(self, token_idxs):
        """Decode a list of token IDs back into text."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def encode(self, text):
        """Encode the given text into a list of token IDs."""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def _build_vocab(self):
        """Build the vocabulary (e.g., all byte values)."""
        self.vocab = {i: bytes([i]) for i in range(256)}
        for parts, idx in self.merges.items():
            self.vocab[idx] = b"".join(self.vocab[p] for p in parts)
        self.inverse_vocab = {v: k for k,v in self.vocab.items()}
        if self.special_tokens:
            self.inverse_special_tokens = {v: k.encode('utf-8') for k, v in self.special_tokens.items()}
        
    def save(self, file_prefix):
        """Save the tokenizer's merges and special tokens to files."""
        model = {
            "merges": {"_".join([str(x) for x in pair]): idx for pair, idx in self.merges.items()},
            "special_tokens": self.special_tokens
        }
        with open(f"{file_prefix}_model.json", "w", encoding="utf-8") as f:
            json.dump(model, f, ensure_ascii=False, indent=4)

    def load(self, file_prefix):
        """Load the tokenizer's merges and special tokens from files."""
        with open(f"{file_prefix}_model.json", "r", encoding="utf-8") as f:
            model = json.load(f)
        self.merges = {
            tuple(map(int, pair.split("_"))): idx
            for pair, idx in model["merges"].items()
        }
        self.special_tokens = model["special_tokens"]
        self._build_vocab()
    
    def compression_ratio(self, text):
        """Calculate the compression ratio of the tokenizer on the given text."""
        original_size = len(list(map(int, text.encode('utf-8'))))
        encoded = self.encode(text)
        compressed_size = len(encoded)
        return original_size/compressed_size if compressed_size > 0 else 1.0
    
    def register_special_token(self, special_tokens):
        """Register a special token with a specific ID."""
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 256, "<|UNK|>": 257}
        for token_str, token_id in special_tokens.items():
            if token_id in self.vocab:
                raise ValueError(f"Token ID {token_id} already exists in the vocabulary.")
            else:
                self.special_tokens[token_str] = token_id
        self.inverse_special_tokens = {idx: token.encode('utf-8') for token, idx in self.special_tokens.items()}