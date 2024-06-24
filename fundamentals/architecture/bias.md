# Purpose of Bias in Neural Networks
The bias can be defined as the constant which is added to the product of features and weights and serves several important purposes, primarily related to enhancing the flexibility and performance of the network. Here are the key functions of bias in neural networks:

## 1. Allowing for Non-Zero Output with Zero Input
Without a bias term, if the input to a neuron is zero, the output will always be zero (assuming no activation function or a linear activation function). The bias term enables the neuron to produce a non-zero output even when the input is zero, allowing the network to fit the data more accurately.

### Example
Given a simple network that meets the following requirements:
* Analyzes images.
* Outputs a 1 if a dog is not present on an image and 0 otherwise.
* Uses ReLU as the activation function after each layer, including the last one, which is modified to cap higger values than 1 to 1.

When using this network, the bias term would be required when analyzing a black image. Given that all the features (pixels) would be 0, multiplying them by the weights would also result in 0. When this 0 is passed by the ReLU activation function, the output would be also 0. Repeating this for every layer would result in the final outcome being 0, indicating that there is a dog present in the completely black image.

$$
\\{ReLU}(w * 0) = 0
$$

## 2. Shifting Activation Functions And Handling Data Distributions
Bias helps in shifting the activation function to the left or right. This shifting allows the model to better fit the data by moving the decision boundary. For example, in a logistic regression model, the bias term allows the logistic function to be centered around a different point than zero, which allows the model to addapt the data distribution and, especially, 
 to its mean (assuming it might be different from 0).

### Example
The sigmoid function activation function is:

$$
\sigma(x) = \frac{1}{1 + e^{-x}}

$$
$$
x = w \cdot a + b
$$

Where:
- w is the weight.
- a is the input.
- b is the bias.

Let's see how modifying the bias might impact the function.

![Sigmoid Function with Different Biases](docs\fundamentals\architecture\bias\sigmoid_function_bias.png)

As it can be seen on the image, the change of the bias has a direct impact over the activation function. Despite it not making it any steeper nor gradual, it shifted it. Whenever the bias is increased, a lower output of w*x will be able to activate the neuron.
