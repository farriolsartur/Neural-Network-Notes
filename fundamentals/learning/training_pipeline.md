# Neural Network Training Pipeline

When a neural network is first initialized, its weights and biases are randomly set, thus usually producing a different output than the expected one for the available data. 

![Neural Netwrok mistake](/docs\images\fundamentals\learning\training_pipeline\neural_network_error.jpg)

In order to make the neural network fit the available data and produce the expected output (in this case 2), it is necessary to train it. This file will discuss the iterative process that is followed to train any neural network, independently from its architecture or purpose.

## 1. Forward Propagation

Firstly, the data is fed into the network, often using mini-batches ([more information about mini-batches here](/docs\images\fundamentals\learning\datasets\mini-batches.md)). Despite that, we will assume that only one example (instance) of the dataset is employed. The data passes through the network from left to right, being modified by the weights and biases of the neurons that compose the different layers of the network till the last layer is reached and it produces the output of the network. This process is known as forward propagation.

![Forward propagation](/docs\images\fundamentals\learning\training_pipeline\forward_prop.jpg)

## 2. Output and Loss Computation

Once forward propagation is complete, the network produces an output. This output is the network's prediction or response to the given input. In a classification task, this might be probabilities for each class, while in a regression task, it could be a continuous value.

To assess the quality of this output, we compare it to the expected result, often referred to as the ground truth or target. This comparison is quantified using a loss function, which measures the discrepancy between the predicted and actual outputs.

The choice of loss function depends on the specific task and the nature of the output. For regression problems, Mean Squared Error (MSE) is commonly used. It calculates the average squared difference between predicted and actual values. For classification tasks, Cross-Entropy loss is often preferred. This loss function penalizes confident incorrect predictions more heavily than less confident ones.

The loss value provides a single scalar that represents how well the network is performing. A lower loss indicates better performance, while a higher loss suggests that the network's predictions are far from the desired output.

## 3. Modifying Network Output

To improve the network's performance and reduce the loss, we need to modify its output. There are three primary ways to achieve this:

Firstly, we can modify the weights of the network. Weights determine the strength of connections between neurons. By adjusting these weights, we can change how much influence each input has on the subsequent layer's activations. This is the most common way of training neural networks.

Secondly, we can modify the biases. Biases act as thresholds for neuron activation. Adjusting biases can shift the activation function, allowing the network to better fit the data. This is particularly useful when the optimal decision boundary doesn't pass through the origin.

Lastly, we can modify the previous activations. While we typically don't directly alter these during training, changes to weights and biases in earlier layers will indirectly modify the activations of subsequent layers. This highlights the interconnected nature of neural networks, where changes in one part of the network can have far-reaching effects.

## 4. Derivatives and Gradient Descent

To systematically modify the network's parameters (weights and biases), we use an optimization algorithm called gradient descent. This algorithm relies on computing partial derivatives of the loss function with respect to each parameter.

These partial derivatives indicate how a small change in a parameter would affect the loss, assuming all other parameters remain constant. It's important to note that this assumption of constancy for other parameters is an approximation, as in reality, all parameters are being updated simultaneously. This can sometimes lead to minor discrepancies between expected and actual changes in loss.

The collection of these partial derivatives forms the gradient, which is a vector pointing in the direction of steepest increase in the loss function. To minimize the loss, we update the parameters in the opposite direction of this gradient.

The learning rate, a hyperparameter, determines the size of the steps we take in this opposite direction. A larger learning rate can lead to faster convergence but risks overshooting the minimum, while a smaller learning rate provides more precise updates but may slow down the learning process.

## 5. Backpropagation and the Chain Rule

Backpropagation is an efficient algorithm for computing the gradients needed for gradient descent. It leverages the chain rule of calculus to propagate the error gradient backwards through the network.

The process begins at the output layer, where we compute the gradient of the loss with respect to the network's output. From there, we move backwards through the network, layer by layer. For each layer, we compute three key gradients:

1. The gradient of the loss with respect to the layer's output. This tells us how changes in the layer's output affect the final loss.

2. The gradient of the layer's output with respect to its input. This gradient represents how sensitive the layer's output is to changes in its input.

3. The gradients of the layer's output with respect to its weights and biases. These gradients indicate how adjusting the layer's parameters would influence its output.

The chain rule allows us to combine these gradients to compute how each parameter in the layer contributes to the final loss. By multiplying these gradients, we can trace the path of influence from any parameter all the way to the loss function.

This process is repeated for each layer, moving backwards through the network, hence the name "backpropagation". The result is a set of gradients for all parameters in the network, which can then be used to update these parameters using gradient descent.

The power of backpropagation lies in its efficiency. Rather than separately computing the influence of each parameter on the loss, it reuses computations, significantly speeding up the training process for deep neural networks with millions of parameters.

By iterating through cycles of forward propagation, loss computation, backpropagation, and parameter updates, the neural network gradually refines its parameters. This iterative process allows the network to learn complex patterns in the data and improve its performance on the given task over time.