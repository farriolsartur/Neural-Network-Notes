# Loss Functions

## 1. Binary Cross Entropy (BCE)

The Binary Cross-Entropy (BCE) loss function, also known as log loss, is a fundamental loss function used for binary classification tasks. It measures the discrepancy between the predicted probabilities and the actual binary labels, providing a continuous and differentiable measure of error that can be minimized using optimization algorithms.

$$
BCE = -\frac{1}{N} \sum_{i=1}^{N} y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)
$$

Where:

- $N$ is the number of samples
- $y_i$ is the true label (0 or 1) for the $i$-th sample
- $\hat{y}_i$ is the predicted probability for the $i$-th sample

The BCE function comprises two complementary parts. The green term activates when y_i = 1, while the red term activates when y_i = 0. This mutual exclusivity ensures that for each sample, only one term contributes to the loss, simplifying the computation and interpretation of binary cross-entropy.

$$ BCE = -\frac{1}{N} \sum_{i=1}^{N} \color{#4CAF50}{y_i \log(\hat{y}_i)} \color{white} {+} \color{#F44336}{(1 - y_i) \log(1 - \hat{y}_i)} $$

The BCE loss function calculates the average loss over all samples in the dataset by comparing the true labels with the predicted probabilities.

- When $y_i = 1$: The loss focuses on $-\log(\hat{y}_i)$, penalizing the model heavily if it predicts a probability $\hat{y}_i$ that is far from 1. That is due to the second term being 0: $(1 - 1)log(1-\hat{y}_i) = 0log(1-\hat{y}_i) = 0$.
- When $y_i = 0$: The loss focuses on $-\log(1 - \hat{y}_i)$, penalizing the model if it predicts a probability $\hat{y}_i$ that is close to 1.  That is due to the first term of the formula being 0: $0log(\hat{y}_i) = 0$.

### Why are logarithms used?

![Log Function](/docs/images/fundamentals/functions/loss_functions/log_function.jpg)

1. Numerical stability: they convert multiplication into addition, which is computationally more stable and efficient, thus preventing underflow in computations.
2. Convexity: this property ensures that there is a single global minimum which eases the optimization process and makes it more reliable.
3. Scale-invariance: logarithms make the loss scale-invariant, meaning it's sensitive to relative rather than absolute differences in probabilities. $log(0.1) - log(0.01) = log(0.01) - log(0.001)$.

## 2. Cross Entropy (CE)

The Cross-Entropy (CE) loss function is a widely used metric in multi-class classification tasks. It measures the difference between two probability distributions: the true distribution $y_{i,c}$ and $\hat{y}_{i,c}$ the predicted distribution for each sample in the dataset.

$$ CE = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c}) $$
Where:

- $CE$ is the Cross Entropy Loss
- $N$ is the number of samples
- $C$ is the number of classes
- $y_{i,c}$ is the true probability of sample $i$ belonging to class $c$ (-usually 0 or 1)
- $\hat{y}_{i,c}$ is the predicted probability of sample $i$ belonging to class $c$