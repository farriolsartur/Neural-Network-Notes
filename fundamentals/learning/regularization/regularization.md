# Regularization

## Introduction
Regularization is a technique used in machine learning to prevent overfitting by adding a penalty term to the loss function. This penalty discourages the model from assigning excessively large weights to the features, thereby reducing variance and improving the model's generalization to new data.

The most common forms of regularization involve Lp norms, with L1 (Lasso) and L2 (Ridge) regularizations being the preferred choices due to their effectiveness and computational efficiency. To illustrate how regularization operates, for the rest of the document, we will consider a simple model with just two weights and use the Mean Squared Error (MSE) as the loss function:

$$ \text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 $$

## Isocontours
To understand the impact of regularization, it is helpful to examine the geometric interpretation of the loss function. An isocontour of a function is a set of points where the function takes on the same value. For the MSE loss function with two weights, the isocontours are ellipses centered around the point corresponding to the optimal weights without regularization.

![MSE isocontour](/docs/images/fundamentals/learning/regularization/regularization/mse_isosurface.jpg)

The number of each elipse is the MSE loss.


## L1 Regularization
L1 regularization adds a penalty equal to the sum of the absolute values of the weights, multiplied by a regularization parameter $\lambda$. The modified loss function becomes:

$$ \text{Loss} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{i=1}^n |w_i| $$

Because the L1 penalty involves the absolute value of the weights, its derivative with respect to each weight is either $\lambda$ or $-\lambda$, depending on the sign of the weight. This means that all weights are penalized equally, regardless of their magnitude.

Geometrically, the isocontours of the L1 regularization term form a diamond shape in two dimensions (or a polyhedron in higher dimensions).

![MSE isosurface with L1 norm](/docs/images/fundamentals/learning/regularization/regularization/isosurface_l1.jpg)

Adding the L1 penalty shifts the optimal point away from the minimum of the unpenalized MSE loss because it increases the total loss by $\lambda$ times the sum of the absolute values of the weights. For example, if the unregularized optimal weights are $w_1=2$ and $w_2=6$, the additional regularization term becomes $\lambda(|2|+|6|)=8\lambda$, increasing the total loss.

![MSE isosurface with L1 norm distance](/docs/images/fundamentals/learning/regularization/regularization/isosurface_l1_closest_point.jpg)

The new optimal point is the red one, balancing the MSE loss and the regularization penalty.

Due to the diamond shape of the L1 isocontours, the point that minimizes the total loss often lies at one of the vertices of the diamond. At these vertices, one or more weights are exactly zero. This characteristic explains why L1 regularization tends to produce sparse models, effectively setting some weights to zero and acting as a form of feature selection.

### Why Do Weights Remain at Zero?
When a weight reaches zero under L1 regularization, it corresponds to a "kink" or vertex in the loss function's geometry. At this point, the derivative (technically, the subgradient) of the L1 term is not uniquely defined but ranges between $-\lambda$ and $\lambda$. Unless the gradient from the MSE loss component is large enough to overcome this interval, the weight remains at zero. This makes it difficult for weights that have been set to zero to become non-zero again during optimization, reinforcing the sparsity induced by L1 regularization.

## L2 Regularization
L2 regularization adds a penalty equal to the sum of the squares of the weights, multiplied by a regularization parameter $\lambda$. The modified loss function becomes:

$$ \text{Loss} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 + \lambda \sum_{i=1}^n w_i^2 $$

Because the L2 penalty involves the squares of the weights, the derivative with respect to each weight is proportional to the weight itself ($2\lambda w_i$). This means that weights with larger magnitudes experience a stronger push towards zero compared to smaller weights. Thus, L2 regularization penalizes large weights more heavily than small ones.

Geometrically, the isocontours of the L2 regularization term form circles in two dimensions (or hyperspheres in higher dimensions).

![MSE isosurface with L2 norm](/docs/images/fundamentals/learning/regularization/regularization/isosurface_l2.jpg)

As with L1 regularization, adding the L2 penalty shifts the optimal point away from the minimum of the unpenalized MSE loss. For instance, if the unregularized optimal weights are $w_1=2$ and $w_2=6$, the L2 regularization term adds $\lambda(w_1^2 + w_2^2) = \lambda(4 + 36) = 40\lambda$ to the loss, increasing the total loss.


![MSE isosurface with L2 norm distance](/docs/images/fundamentals/learning/regularization/regularization/isosurface_l2_closest_point.jpg)

just as before, the new optimal point is the red one, balancing the MSE loss and the regularization penalty.

Because the isocontours of L2 regularization are circular, the point that minimizes the total loss does not lie on a vertex (since circles have no vertices). This means that, unlike L1 regularization, L2 does not encourage weights to become exactly zero. Instead, it shrinks all weights towards zero proportionally, reducing their magnitude but not eliminating them entirely. This leads to smoother models where irrelevant features have smaller, but typically non-zero, weights. Additionally, the smoothness of the L2 penalty contributes to a more stable and well-behaved loss function during optimization.