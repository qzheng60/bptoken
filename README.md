# BytePieceTokenizer

A simple byte-piece tokenizer implementation, inspired by the `minbpe` repository.

## Overview

`BytePieceTokenizer` provides a lightweight tokenizer interface for training a vocabulary from text, encoding text into token IDs, and decoding token IDs back into text.

## Import

```python
from bptoken.bytepiecestk import BytePieceTokenizer
```

## Example

```python
text = "This is a simple example."

tokenizer = BytePieceTokenizer()
tokenizer.train(text, 400)

tokenizer.save("bytepiece_tokenizer")
loaded_tokenizer = BytePieceTokenizer.from_model("bytepiece_tokenizer")
encoded = loaded_tokenizer.encode(text)
decoded = loaded_tokenizer.decode(encoded)
```

## API

### `BytePieceTokenizer()`
Creates a new tokenizer instance.

### `tokenizer.train(text, vocab_size)`
Trains the tokenizer on the given text with the target vocabulary size.

### `tokenizer.encode(text)`
Encodes input text into a list of token IDs.

### `tokenizer.decode(tokens)`
Decodes a list of token IDs back into text.

## Notes

- Inspired by the design of the `minbpe` repository.
- Intended as a simple tokenizer implementation.

## TODO

- Stress test on more complex text inputs
- Evaluate performance on larger training corpora
- Compare performance with other tokenization
