# Noise Injection

Noise injection is a technique used to enhance the robustness of machine learning models by introducing randomness into the training data. While noise can be added at both the input and output levels, this document focuses on input-level noise injection. By perturbing the input examples fed into the network, we aim to improve the model's ability to generalize to unseen data.

At the input level, noise injection involves adding random noise to each element of the input vector $x$ before it is processed by the network. The noise is typically drawn from a uniform distribution ranging from $-\epsilon$ to $\epsilon$, where $\epsilon$ is a small value that determines the magnitude of the noise. This process effectively creates new training samples that are slight variations of the original data.

For example, consider an input vector:
$x = [0.5, 0.7, 0.2]$

Suppose we choose $\epsilon = 0.1$. We generate a noise vector $\delta$ where each element $\delta_i$ is a random value between $-0.1$ and $0.1$:
$\delta = [0.03, -0.08, 0.05]$

Adding this noise to the original input yields the noisy input vector:
$x_{\text{noisy}} = x + \delta = [0.53, 0.62, 0.25]$

$$
\begin{bmatrix}
0.5 \\
0.7 \\
0.2 \\
\end{bmatrix}
+
\begin{bmatrix}
0.03 \\
-0.08 \\
0.05 \\
\end{bmatrix}
=
\begin{bmatrix}
0.53\\
0.62\\
0.25\\
\end{bmatrix}
$$

It's important to note that noise injection should only be applied during the training phase. Introducing noise during validation or real-world inference can skew results and hinder performance. The goal is to expose the model to slight variations during training to make it more resilient to variability in real data.

Moreover, the noise added should differ in each training epoch. For each original input $x$, the model is trained on multiple noisy versions across epochs. If the training consists of $n$ epochs, we effectively create $n$ different samples for each $x$, each slightly modified. This approach helps prevent the model from overfitting to the exact training data and enhances its robustness.

By consistently exposing the model to these small perturbations, noise injection at the input level can significantly improve the model's ability to generalize, leading to better performance on unseen data.
