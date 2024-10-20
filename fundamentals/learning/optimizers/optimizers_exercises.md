# Exercises on Optimizers

## Exercises

### 1. Optimizers that Improve the Gradient

Which optimizer enhances the gradient calculation by incorporating information from previous iterations to smooth out updates and accelerate convergence?

### 2. Optimizers that Adjust the Learning Rate

Which optimizers improve the learning rate by adapting it individually for each parameter during each optimization step?

### 3. Combining Gradient and Learning Rate Improvements

Which optimizer combines both gradient improvement and adaptive learning rates? How does it refine these approaches. Is it considered the standard choice for optimizers in modern deep learning?

### 4. Adam Optimizer Formulas

Write down all the formulas used in the Adam optimizer, explaining each component.

## Solutions

### 1. Optimizers that Improve the Gradient

**Answer:** The optimizer that improves the gradient by incorporating information from previous iterations is **Stochastic Gradient Descent with Momentum**. Momentum helps accelerate gradients in the right direction, leading to faster convergence by accumulating a velocity vector that also dampens oscillations.

### 2. Optimizers that Adjust the Learning Rate

**Answer:** The optimizers that improve the learning rate by adapting it for each parameter are **Adagrad** and **RMSProp**. These optimizers adjust the learning rate individually for each parameter based on the historical gradients, allowing parameters that receive infrequent updates to have larger learning rates.

### 3. Combining Gradient and Learning Rate Improvements

**Answer:** The **Adam** optimizer combines both gradient improvement (through momentum) and adaptive learning rates. It refines these approaches by maintaining exponentially decaying averages of past gradients (first moment) and past squared gradients (second moment). Adam is widely considered the go-to optimizer in modern deep learning because it is computationally efficient, has little memory requirement, and works well with sparse gradients and noisy problems.

### 4. Adam Optimizer Formulas

**Answer:**

The Adam optimizer updates parameters using the following formulas:

1. **Initialize** the first moment vector $v_0$ and the second moment vector $E_0$ to zeros.

2. **At each iteration $t$**, compute the gradients:
   $g_t = \frac{\partial J}{\partial \theta_{t-1}}$

3. **Update biased first moment estimate**:
   $v_t = \beta_1 v_{t-1} + (1 - \beta_1) g_t$

4. **Update biased second moment estimate**:
   $E_t = \beta_2 E_{t-1} + (1 - \beta_2) g_t^2$

5. **Compute bias-corrected first moment estimate**:
   $\hat{v}_t = \frac{v_t}{1 - \beta_1^t}$

6. **Compute bias-corrected second moment estimate**:
   $\hat{E}_t = \frac{E_t}{1 - \beta_2^t}$

7. **Calculate parameter-Specific Learning Rate**:
   $\alpha_t = \frac{\eta}{\sqrt{\hat{E}_t} + \epsilon}$

8. **Update parameters**:
   $\theta_t = \theta_{t-1} - \alpha_t \hat{v}_t$

**Where:**
- $\theta_t$ are the parameters at iteration $t$.
- $\eta$ is the initial learning rate.
- $\alpha_t$ is the parameter-specific learning rate at iteration $t$.
- $\beta_1$ is the exponential decay rate for the first moment estimates (typically $0.9$).
- $\beta_2$ is the exponential decay rate for the second moment estimates (typically $0.999$).
- $\epsilon$ is a small constant to prevent division by zero (typically $10^{-8}$).

**Explanation of Each Component:**

- **First Moment Estimate ($v_t$)**: Represents the exponentially decaying average of past gradients, acting like momentum.
- **Second Moment Estimate ($E_t$)**: Represents the exponentially decaying average of past squared gradients, helping to adapt the learning rate.
- **Bias Correction ($\hat{v}_t$ and $\hat{E}_t$)**: Adjusts the estimates to account for their initialization at zero, providing unbiased estimates.
- **Parameter-Specific Learning Rate ($\alpha_t$)**: Computes a new learning rate for each parameter at each iteration, taking into account the bias-corrected second moment estimate.
- **Parameter Update ($\theta_t$)**: Updates the parameters by moving in the direction determined by the bias-corrected first moment estimate, scaled by the parameter-specific learning rate.

Adam effectively combines the benefits of two other extensions of stochastic gradient descent: **AdaGrad** (adaptive learning rates) and **Momentum** (exponentially weighted average of gradients), making it highly effective for a wide range of optimization problems.