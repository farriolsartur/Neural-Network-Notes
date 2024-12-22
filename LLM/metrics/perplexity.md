# Perplexity

Perplexity is a simple NLP metric that aims to capture the degree of uncertainty a model has in predicting the next outputs. It is called "perplexity" because a high value indicates that the model is perplexed or confused about the expected output. It should be noted that perplexity should not be the only metric used to evaluate a model: a model that merely memorizes responses will achieve low perplexity but likely generalize poorly, whereas a more versatile model that produces novel outputs could be penalized by perplexity alone.

## How to compute it?

One of the simplest ways to calculate perplexity is via the *normalized probability* of a sentence. The probability of a certain sentence can be expressed as:

$$ P(\text{sentence}) \;=\; P(w_1)\,P(w_2 \mid w_1)\,\cdots\,P(w_n \mid w_{n-1}, \ldots, w_1) $$

**Where** $w_1, w_2, \ldots, w_n$ are the tokens (e.g., words) in the sentence, and each $P(w_i \mid w_{i-1}, \ldots, w_1)$ is the conditional probability of the $i$-th token given all its predecessors.

For example, having the sentence "*a dog.*," if the probability of *a* is 0.5, of *dog* (given the previous token) is 0.7, and of *.* (given the previous tokens) is 0.4, the overall probability is:

$$ P(\text{a dog.}) = 0.5 \times 0.7 \times 0.4 = 0.14 $$

To show why we normalize this probability, consider a longer sentence "*a dog barked loudly.*," with the probability of *barked* being 0.8, *loudly* being 0.7, and *.* being 0.6. Then:

$$ P(\text{a dog barked loudly.}) = 0.5 \times 0.7 \times 0.8 \times 0.7 \times 0.6 = 0.1176 $$

Even though the *average* probabilities of the words in the longer sentence are higher, the total probability is lower, simply because more factors are multiplied together. To avoid penalizing longer sequences, we take the geometric mean:

$$ P_{\text{norm}}(\text{sentence}) \;=\; \bigl(P(\text{sentence})\bigr)^{\tfrac{1}{n}} $$

Using our two examples:

$$ P_{\text{norm}}(\text{a dog.}) = 0.14^{\tfrac{1}{3}} \approx 0.519 $$

$$ P_{\text{norm}}(\text{a dog barked loudly.}) = 0.1176^{\tfrac{1}{5}} \approx 0.6518 $$

Given this normalized probability, *perplexity* is simply its reciprocal:

$$ \text{Perplexity}(\text{sentence}) = \frac{1}{P_{\text{norm}}(\text{sentence})} $$

## How it also can be derived from Cross Entropy

Perplexity can also be understood via cross entropy. Suppose you have a true distribution $p$ (in practice, approximated by the empirical distribution of your test data) and a model distribution $q$. The average cross entropy over $N$ tokens is:

$$ H(p, q) \;=\; -\frac{1}{N} \sum_{i=1}^{N} \log \bigl(q(w_i)\bigr) $$

where $w_i$ are the tokens in your test set. If the logarithm is base 2, perplexity is given by:

$$ \text{Perplexity} = 2^{H(p, q)} $$

and if the logarithm is the natural logarithm ($\ln$),

$$ \text{Perplexity} = e^{H(p, q)} $$

Hence, perplexity is essentially the exponential of the cross entropy.
