# Multi-Head Attention

Multi-head attention extends the idea of self-attention by applying it multiple times in parallel. Each application of self-attention is referred to as a "head."

When using multi-head attention, the input queries $Q$, keys $K$, and values $V$ are projected into different subspaces for each head using parameter matrices $W_i^Q$, $W_i^K$, and $W_i^V$. Each head learns to focus on different types of relationships within the same sequence, allowing the model to simultaneously consider multiple aspects of each token. For instance, considering the example *"He went to Norway last month"*, a single head might focus on the temporal aspect ("last month"), while another might focus on the location ("Norway").

By combining (concatenating) the outputs of multiple attention heads, the model can capture richer, more nuanced representations of the input. These concatenated outputs are then projected back to the original dimensionality using a matrix $W_o$. This process ensures that the overall output dimension remains manageable (e.g., typically 512), while still benefiting from the diverse perspectives each head provides.

### Formulas

$$
Attention(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

$$
head_i = Attention(W_i^Q Q,\; W_i^K K,\; W_i^V V)
$$

$$
Multi\text{-}Head(Q,K,V) = \text{concat}(head_1,\dots,head_h)W_o
$$

Because we split the total dimensionality $d_{model}$ across $h$ heads (each with dimension $d_k = d_{model}/h$), the concatenation of all heads returns to the original dimension $d_{model}$.

### Key Points

1. **Multiple Perspectives:** Multi-head attention applies multiple self-attentions in parallel. Each head can focus on different relationships within the same sequence.

2. **Dimensionality Management:** To keep the output dimension constant, queries, keys, and values are projected into smaller subspaces, ensuring that after concatenation the total dimensionality remains the same.

3. **Richer Representations:** By asking multiple "questions" about each token (one per head), the model creates more comprehensive and contextually informed representations.

4. **Preserving Information:** Concatenating the outputs of each head (rather than adding them) preserves the distinct informational content each head provides. A subsequent linear projection $W_o$ then synthesizes this multi-view information into a single output.