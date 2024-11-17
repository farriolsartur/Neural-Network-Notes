# Label Smoothing

Label smoothing is a regularization technique designed to prevent overfitting in classification tasks, particularly in neural networks. The core issue it addresses stems from the nature of categorical labels (like 0 and 1) and how neural networks process them.

In classification problems with categorical labels, the network's final layer (using sigmoid or softmax activation) would theoretically need an infinite input to produce exactly 0 or 1. During training, this pushes the model to continuously increase its weights to approach these extreme values, even when the accuracy improvements become negligible. This behavior can lead to:
- Overconfident predictions
- Poor generalization
- Numerical instability

Label smoothing addresses this by introducing a small adjustment factor $\epsilon$ to the target labels. For a binary classification problem, instead of using hard labels [0, 1], we use:
- 0 → $\epsilon$
- 1 → $1 - \epsilon$

For example, in a binary classification problem with $\epsilon = 0.025$, a one-hot encoded label would be transformed as:

$$ \begin{bmatrix} 0\end{bmatrix} \longrightarrow \begin{bmatrix} 0.025\end{bmatrix} $$

$$ \begin{bmatrix} 1\end{bmatrix} \longrightarrow \begin{bmatrix} 0.975\end{bmatrix} $$


For multi-class problems with $K$ classes, the smoothing is applied as follows:
$$ y_{smooth} = (1 - \epsilon)y + \epsilon\frac{1}{K} $$

This is done to ensure that the one-hot encoded label still adds up to 1.

For example, in a 4-class classification problem with $\epsilon = 0.1$, a one-hot encoded label would be transformed as:

$$ \begin{bmatrix} 0 \\ 0 \\ 1 \\ 0 \end{bmatrix} \longrightarrow \begin{bmatrix} 0.025 \\ 0.025 \\ 0.925 \\ 0.025 \end{bmatrix} $$

This vector is the new correct label for a single example $x$

Label smoothing can be implemented in two ways:
1. **Dataset Modification**: Directly modifying the labels in the training dataset
2. **Loss Function Integration**: Applying smoothing during the loss calculation, keeping the original dataset intact

The second approach is generally preferred as it offers more flexibility and doesn't alter the original data. Common values for $\epsilon$ range from 0.1 to 0.2, though this hyperparameter should be tuned for each specific application.