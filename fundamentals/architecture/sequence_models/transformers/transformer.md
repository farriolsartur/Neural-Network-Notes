# Transformer Overview

The Transformer architecture was originally introduced to address machine translation tasks, but it has since proven effective for a wide range of sequence-to-sequence problems.

![Transformer](/docs/images/fundamentals/architecture/sequence_models/transformer/transformer.jpg)

## Main Components

- **Encoder:** Processes the input word embeddings and generates a context-rich representation. This representation encodes both semantic and positional information about each token.

- **Decoder:** Uses the encoder's output along with its own previously generated outputs to produce the next token in the sequence. The decoder generates queries from its previously generated output and uses the encoder's output as keys and values, given that it has processed information about the whole input sequence.

## Embeddings

An embedding layer converts input tokens (represented as integers) into continuous vectors. These embeddings capture semantic meaning. In some implementations, the same embedding matrix is also shared (tied) with the final linear layer that produces output logits, helping ensure that the embedding and output spaces remain consistent.

## Positional Encoding

To inject information about the order of tokens, a positional encoding is added to the token embeddings. The standard positional encoding uses sine and cosine functions at different frequencies:

$$ PE(pos, 2i) = \sin\left(\frac{pos}{10000^{\frac{2i}{d}}}\right), \quad PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{\frac{2i}{d}}}\right) $$

Here:
- $d$ is the embedding dimension.
- $i$ is an index that iterates over embedding dimensions, with $2i$ and $2i+1$ indexing even and odd positions respectively.
- $pos$ is the token's position in the sequence (starting from 0).

These sinusoidal patterns allow the model to learn both absolute and relative positioning of tokens. The linear changes in the positional encoding's phase help the model reason about distances between words.

## Add & Norm Layers

"Add & Norm" layers combine residual connections with layer normalization. The residual connections ensure stable gradients and help preserve positional and semantic information, while layer normalization speeds up training and stabilizes learning.

## Masked Multi-Head Attention

In the decoder, masked multi-head attention ensures that the model cannot "peek" at future tokens when generating the next token. This masking allows the model to be trained in parallel over entire sequences while still maintaining the causal property necessary for language generation.

## Key Points

1. **Encoder-Decoder Structure:** The encoder processes input embeddings to form contextual representations, while the decoder uses these plus its own history to generate outputs.

2. **Shared and Positional Embeddings:** Continuous embeddings and sinusoidal positional encodings allow the model to capture both semantic meaning and token order.

3. **Residual Connections and Normalization:** Add & Norm layers help maintain important information and stabilize training.

4. **Parallelized Training with Masking:** Masked multi-head attention in the decoder enables parallel training over sequences while preserving causality.