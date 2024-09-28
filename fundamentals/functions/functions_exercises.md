# Exercises on Activation Functions

## Exercises

### 1. Identify the Function from the Plots

Given the following plots, identify the activation function corresponding to each plot.
- `plot1`

![plot 1](/docs/images/fundamentals/functions/activation_functions/plot_1.jpg)
****
- `plot2`

![plot2](/docs/images/fundamentals/functions/activation_functions/plot_2.jpg)
****
- `plot3`

![plot3](/docs/images/fundamentals/functions/activation_functions/plot_3.jpg)
****
- `plot4`

![plot4](/docs/images/fundamentals/functions/activation_functions/plot_4.jpg)



### 2. Write Down the Formulas

Given the names of the activation functions below, write down their mathematical formulas:
- Softmax
- Tanh
- Leaky ReLU 
- Sigmoid

### 3. Explain the Vector Input/Output Function

Which activation function has a vector as input/output instead of a scalar? WHat does it imply?


### 4. Compute the Gradient of the ReLU Function

Consider the function $f(x_1, x_2) = \text{ReLU}(mx_1 + nx_2 + b)$.  Compute the gradient of this function with respect to the weights $m$, $n$, and bias $b$, when $x_1$ is 1, $x_2$ is 2, $m$ is -3, $n$ is 1 and $b$ is -4 . Use the result to explain how the gradient can be 0 and how this relates to the dying ReLU problem.

## Solutions

### 1. Identify the Function from the Plots

- `plot1`: This plot corresponds to the **Softmax** function.
- `plot2`: This plot corresponds to the **ReLU** function.
- `plot3`: This plot corresponds to the **Sigmoid** function.
- `plot4`: This plot corresponds to the **Tanh** function.

### 2. Write Down the Formulas

- **Softmax**:
$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}$

- **Tanh**:
$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

- **Leaky ReLU**:
$\text{Leaky ReLU}(x) = \max(\alpha x, x)$

- **Sigmoid**:
$\sigma(x) = \frac{1}{1 + e^{-x}}$

### 3. Explain the Vector Input/Output Function

The **Softmax** function has a vector input/output instead of a scalar. This implies that the Softmax function is applied after all neuron outputs in a layer are collected into a vector. The function then normalizes this vector so that each element represents a probability, and the sum of all elements equals 1. This is different from scalar functions like Sigmoid and ReLU, which operate independently on each neuron's output.

### 4. Compute the Gradient of the ReLU Function

Consider the function $f(x_1, x_2) = \text{ReLU}(mx_1 + nx_2 + b)$.

Given:
- $x_1 = 1$
- $x_2 = 2$
- $m = -3$
- $n = 1$
- $b = -4$

Step 1: Calculate the input to the ReLU function
$z = mx_1 + nx_2 + b$
$z = (-3)(1) + (1)(2) + (-4) = -3 + 2 - 4 = -5$

Step 2: Apply the ReLU function
$\text{ReLU}(z) = \max(0, z) = \max(0, -5) = 0$

Step 3: Compute the gradient
The gradient of ReLU is defined as:
$\frac{\partial \text{ReLU}(z)}{\partial z} = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z \leq 0 \end{cases}$

In this case, $z = -5 < 0$, so the gradient is 0.

Therefore, the gradients with respect to $m$, $n$, and $b$ are:

$\frac{\partial f}{\partial m} = \frac{\partial \text{ReLU}(z)}{\partial z} \cdot \frac{\partial z}{\partial m} = 0 \cdot x_1 = 0$

$\frac{\partial f}{\partial n} = \frac{\partial \text{ReLU}(z)}{\partial z} \cdot \frac{\partial z}{\partial n} = 0 \cdot x_2 = 0$

$\frac{\partial f}{\partial b} = \frac{\partial \text{ReLU}(z)}{\partial z} \cdot \frac{\partial z}{\partial b} = 0 \cdot 1 = 0$

Explanation of zero gradient and the dying ReLU problem:

1. Zero Gradient: In this case, all gradients are zero because the input to the ReLU function is negative. When the input is negative, ReLU outputs zero, and its derivative is also zero.

2. Dying ReLU Problem: This example illustrates the "dying ReLU" problem. When weights and biases combine to consistently produce negative inputs to a ReLU activation, that neuron will always output zero, and its weights will not be updated during backpropagation due to the zero gradient. This neuron becomes "dead" or inactive, reducing the network's capacity to learn.

3. Implications: If many neurons in a network end up in this state, a significant portion of the network becomes unused, effectively reducing its capacity and ability to learn complex patterns. This is particularly problematic in deep networks where this effect can compound through layers.

4. Solutions: To address this, alternatives like Leaky ReLU or Parametric ReLU have been developed. These functions allow a small, non-zero gradient when the input is negative, helping to prevent neurons from completely "dying" and allowing the network to recover during training.
