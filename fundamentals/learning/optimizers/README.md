# Optimizers

## Basic

Stochastic Gradient Descent (SGD) updates parameters using the formula:

$\theta_{t} = \theta_{t-1} - \alpha \frac{\partial J}{\partial \theta_{t-1}}$

where:

- $\theta_{t}$ is an array containing all the updated parameters.
- $\theta_{t-1}$ is an array containing all the parameter values from the previous optimization step.
- $\alpha$ is the learning rate, a hyperparameter usually ranging from 0.001 to 0.1 that indicates the magnitude of the step to be taken by the optimizer.
- $\frac{\partial J}{\partial \theta_{t-1}}$ is the gradient of the loss function $J$ with respect to the parameters $\theta$, evaluated at $\theta_{t-1}$.

## Improvements

This formula contains two main components that can be further optimized:

1. **Gradient ($\frac{\partial J}{\partial \theta_{t-1}}$)**: The gradient can incorporate information from previous iterations to avoid drastic changes when gradients vary significantly. By accumulating a "velocity," the optimizer can navigate the parameter space more effectively, especially in regions where the gradient landscape causes slow progress.

2. **Learning Rate ($\alpha$)**: Instead of using a single global learning rate, each parameter might benefit from its own adaptive learning rate at each optimization step. A learning rate suitable for one parameter might be insufficient or excessive for another.

These improvements usually do not result in an increase in the model's accuracy but lead to faster convergence and reduced hyperparameter tuning. In most cases, different optimization algorithms should achieve similar results if perfectly tuned.

## New Optimizers of this section

- **Improving the Gradient ($\frac{\partial J}{\partial \theta_{t-1}}$ issue)**:
    - **SGD with Momentum**

- **Improving the Learning Rate ($\alpha$ issue)**:
    - **Adagrad**
    - **RMSProp**

- **Combining Both Approaches and refining them**:
    - **Adam**
