# BytePieceTokenizer

A simple byte-piece tokenizer implementation, inspired by the `minbpe` repository.

## Overview

`BytePieceTokenizer` provides a lightweight tokenizer interface for training a vocabulary from text, encoding text into token IDs, and decoding token IDs back into text.

## Import

```python
from bptoken.bytepiecestk import BytePieceTokenizer
```

## Quick Start

```python
from bptoken.bytepiecestk import BytePieceTokenizer

tokenizer = BytePieceTokenizer()
tokenizer.train(text, 400)

encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)
```

## Example

```python
text = "This is a simple example."

from bptoken.bytepiecestk import BytePieceTokenizer

tokenizer = BytePieceTokenizer()
tokenizer.train(text, 400)

encoded = tokenizer.encode(text)
decoded = tokenizer.decode(encoded)

print(encoded)
print(decoded)
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
