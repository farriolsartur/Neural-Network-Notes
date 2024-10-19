# Adaptive Moment Estimation (Adam)

Adam is an optimization algorithm that combines the advantages of two other extensions of stochastic gradient descent: **Adaptive Gradient Algorithm (Adagrad)** and **Root Mean Square Propagation (RMSProp)**. By computing adaptive learning rates for each parameter, Adam adapts the learning rate based on the first (momentum) and second (Adagrad) moments of the gradients, providing efficient and robust optimization for neural networks.

## Update Equations:

1. **Compute Gradient:**

   $g_t = \frac{\partial J}{\partial \theta_{i-1}}$

   - $g_t$ is the gradient of the cost function $J$ with respect to the parameters $\theta$ at iteration $t-1$.

2. **Update Biased First Moment Estimate (Mean of Gradients):**

   $v_t = \beta_1 v_{t-1} + (1 - \beta_1) g_t$

   - $v_t$ is the first moment estimate (the mean of the gradients).
   - $\beta_1$ is the exponential decay rate for the first moment estimates (typically $\beta_1 = 0.9$).

3. **Update Biased Second Moment Estimate (Uncentered Variance of Gradients):**

   $E_t = \beta_2 E_{t-1} + (1 - \beta_2) g_t^2$

   - $E_t$ is the second moment estimate (the uncentered variance of the gradients).
   - $\beta_2$ is the exponential decay rate for the second moment estimates (typically $\beta_2 = 0.999$).
   - $g_t^2$ denotes element-wise squaring of $g_t$.

4. **Compute Bias-Corrected First Moment Estimate:**

   $\hat{v}_t = \frac{v_t}{1 - \beta_1^t}$

5. **Compute Bias-Corrected Second Moment Estimate:**

   $\hat{E}_t = \frac{E_t}{1 - \beta_2^t}$

6. **Calculate Parameter-Specific Learning Rate:**

   $\alpha_t = \frac{\eta}{\sqrt{\hat{E}_t} + \epsilon}$

   - $\eta$ is the global learning rate (often set to $0.001$).
   - $\epsilon$ is a small constant (e.g., $1 \times 10^{-8}$) added for numerical stability.

7. **Parameter Update:**

   $\theta_t = \theta_{t-1} - \alpha_t \hat{v}_t$

## Purpose and Mechanics:

**Adaptive Learning Rates with Momentum and Bias Correction:** Adam adapts the learning rate for each parameter by utilizing estimates of both the **first moment (mean)** and the **second moment (uncentered variance)** of the gradients. The algorithm includes bias correction terms to account for the initialization of the moment estimates at zero, which can introduce bias in the early stages of training.

By combining **momentum** (first moment estimates) and **adaptive learning rates** (second moment estimates), Adam provides a robust optimization algorithm that can handle sparse gradients and noisy problems effectively.

## Key Concepts:

1. **First Moment Estimate (Momentum):**
   - $v_t$ represents the exponentially decaying average of past gradients.
   - The decay rate $\beta_1$ controls the influence of past gradients versus the current gradient.
   - Similar to the momentum term in SGD with momentum, it helps accelerate convergence in relevant directions while dampening oscillations.

2. **Second Moment Estimate (Adaptive Learning Rate):**
   - $E_t$ represents the exponentially decaying average of past squared gradients.
   - The decay rate $\beta_2$ adjusts how much past gradients influence the current second moment estimate.
   - By considering the second moment, Adam adapts the learning rate for each parameter individually, similar to RMSProp, taking into account their gradient history.

3. **Bias Correction:**
   - Since $v_0$ and $E_0$ are initialized at zero, the estimates $v_t$ and $E_t$ are biased towards zero at the initial time steps.
   - Bias correction terms $\hat{v}_t$ and $\hat{E}_t$ compensate for this bias, providing unbiased estimates of the first and second moments.

## Advantages of Adam:

- **Efficient Computation:** Adam is computationally efficient and has relatively low memory requirements (the ones seen during RMSProp), making it suitable for large-scale problems and datasets.
- **Bias Correction:** The inclusion of bias correction terms improves the estimates in early iterations, leading to better convergence properties.
- **Robust Default Parameters:** The default hyperparameters ($\alpha = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 1 \times 10^{-8}$) work well across a wide range of problems.

## Limitations:

- **Sensitivity to Hyperparameters:** While generally robust, certain problems may require tuning of the hyperparameters, especially the learning rate $\eta$, to achieve optimal performance.
- **Potential for Non-Convergence:** In some cases, Adam may not converge to the optimal solution and can get stuck in local minima.
- **Overfitting Risk:** Due to its adaptability, Adam may overfit to training data if proper regularization techniques are not employed.
- **Complexity:** The algorithm introduces additional computational overhead due to the calculation of first and second moment estimates and bias correction.

## Conclusion:

Adam combines the strengths of both momentum-based and adaptive learning rate optimization methods to provide a powerful and efficient algorithm for training deep neural networks. By incorporating bias correction and considering both the first and second moments of the gradients, Adam effectively balances the speed of convergence and stability, making it one of the most widely used optimization algorithms in machine learning today.

Through its adaptive nature and robust performance, Adam has become a go-to optimizer for practitioners and researchers, facilitating faster convergence and improved performance in a variety of complex optimization problems.