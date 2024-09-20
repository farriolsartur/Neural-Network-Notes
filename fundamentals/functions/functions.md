# Deep Learning Functions

## Activation Functions

Before delving into the specific functions, it's crucial to understand how activation functions are applied in neural networks:

1. **Scalar Input/Output Functions (e.g., Sigmoid, ReLU)**:
   - These functions operate at the individual neuron level.
   - In a layer, each neuron independently applies the activation function to its scalar output.
   - The results from all neurons are then combined to form the layer's output vector or tensor.

2. **Vector Input/Output Functions (e.g., Softmax)**:
   - These functions are typically applied after the neuron-level computations.
   - First, the outputs from all neurons in a layer are collected into a vector.
   - Then, the vector-based activation function is applied to this collected output.

This distinction is important for understanding the flow of data through a neural network and how different types of activation functions are integrated into the network architecture.

### 1. Sigmoid

- **Usage**: Often used in the output layer for binary classification problems or as gates in LSTM units.
- **Input dimension**: Scalar ($\mathbb{R}$)
- **Output dimension**: Scalar in range (0, 1)
- **Formula**: $\sigma(x) = \frac{1}{1 + e^{-x}}$

![Sigmoid Function](/docs/images/fundamentals/functions/functions/sigmoid.jpg)

### 2. Softmax

- **Usage**: Commonly used in the output layer for multi-class classification problems or in networks that include pseudolabels (DINO).
- **Input dimension**: Vector ($\mathbb{R}^n$)
- **Output dimension**: Vector ($\mathbb{R}^n$) where the sum of all elements of the vector sum to 1 <sup><a href="#sum to 1 note">1</a></sup>
- **Formula**: $\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}$ for $i = 1, \ldots, n$

![Softmax Function](/docs/images/fundamentals/functions/functions/softmax.jpg)

### 3. Tanh (Hyperbolic Tangent)

- **Usage**: Commonly used as an activation function in hidden layers or in RNN/LSTM units.
- **Input dimension**: Scalar ($\mathbb{R}$)
- **Output dimension**: Scalar in range (-1, 1)
- **Formula**: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

![tanh Function](/docs/images/fundamentals/functions/functions/tanh.jpg)


### 4. ReLU (Rectified Linear Unit)

- **Usage**: Widely used activation function in hidden layers of neural networks.
- **Input dimension**: Scalar ($\mathbb{R}$)
- **Output dimension**: Scalar ($\mathbb{R}^+$)
- **Formula**: $\text{ReLU}(x) = \max(0, x)$

![ReLU Function](/docs/images/fundamentals/functions/functions/relu.jpg)

### 5. Leaky ReLU

- **Usage**: Variant of ReLU used to address the "dying ReLU"<sup><a href="#dying-relu-note">2</a></sup> problem in hidden layers.
- **Input dimension**: Scalar ($\mathbb{R}$)
- **Output dimension**: Scalar ($\mathbb{R}$)
- **Formula**: $\text{Leaky\_ReLU}(x) = \max(\alpha x, x)$, where $\alpha$ is a small constant (e.g., 0.01)

![Leaky ReLU Function](/docs/images/fundamentals/functions/functions/leaky_relu.jpg)

## Concept Clarification

1. <a id="sum to 1 note"></a>**Softmax normalizes to probabilities**:
The softmax function transforms a vector of real numbers into a probability distribution. When applied to a vector, it ensures all output elements are between 0 and 1 and sum to 1. This property is particularly useful in multiclass classification, where we want to interpret the outputs as probabilities for each class. It makes it ideal for multiclass classification output layers in which saturation of the function (one element of the vector obtains a much higher probability or value than the others) is not an is sue, but something positive. Despite that, when used in the middle of a network or for pseudolabels (DINO) the potential saturation of the function might cause a vanishing gradient problem for the values that obtained a close to zero probability, which is why centering or normalization techniques are often applied to the data before passing it through the softmax function.

2. <a id="dying-relu-note"></a>**Dying ReLU**: this problem occurs when a large number of neurons in a network output zero for any input, effectively making those neurons inactive or "dead." Once a neuron is dead, it remains inactive and no longer contributes to learning because its gradient is zero, meaning it cannot update its weights during backpropagation. 