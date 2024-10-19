# Root Mean Square Propagation (RMSProp)

RMSProp is an optimization algorithm that improves upon Adagrad by incorporating an exponentially decaying average of past squared gradients. This modification allows RMSProp to adapt the learning rate for each parameter individually while giving more weight to recent gradients. By doing so, RMSProp prevents the learning rates from decreasing too rapidly, addressing one of the main limitations of Adagrad.

## Update Equations:

1. **Compute Exponentially Weighted Moving Average of Squared Gradients:**

   $E_{t,i} = \gamma E_{t-1,i} + (1-\gamma) \left(\frac{\partial J}{\partial \theta_{i-1}}\right)^2$

   - $E_{t,i}$ is the exponentially weighted moving average of the squared gradients for parameter $\theta_i$ at iteration $t$.
   - $\gamma$ is the decay rate (typically around 0.9).
   - $\frac{\partial J}{\partial \theta_{i-1}}$ is the gradient of the cost function $J$ with respect to $\theta_i$ at iteration $t-1$.

2. **Calculate New Learning Rate ($\alpha_{t,i}$):**

   $\alpha_{t,i} = \frac{\eta}{\sqrt{E_{t,i} + \epsilon}}$

   - $\eta$ is the initial global learning rate.
   - $\epsilon$ is a small constant (e.g., $1 \times 10^{-8}$) added for numerical stability to prevent division by zero.

3. **Parameter Update:**

   $\theta_{t,i} = \theta_{t-1,i} - \alpha_{t,i} \cdot \frac{\partial J}{\partial \theta_{i-1}}$

## Purpose and Mechanics:

**Adaptive Learning Rates with Emphasis on Recent Gradients:** RMSProp adjusts the learning rate for each parameter by considering the moving average of the squared gradients. This method places more emphasis on recent gradients through the decay rate $\gamma$, allowing the algorithm to adapt quickly to changes and preventing the learning rates from diminishing too rapidly.

## Key Concepts:

1. **Exponentially Weighted Moving Average (EWMA):**
   - The EWMA $E_{t,i}$ gives more weight to recent squared gradients and less to older ones.
   - The decay rate $\gamma$ controls the rate at which the influence of past gradients decreases.
   - This approach prevents the accumulation of gradients from growing indefinitely, unlike in Adagrad.

2. **Per-Parameter Adaptive Learning Rates:**
   - Each parameter $\theta_i$ has its own adaptive learning rate $\alpha_{t,i}$.
   - Parameters with consistently large gradients receive smaller learning rates, reducing their update magnitude.
   - Parameters with smaller gradients receive larger learning rates, allowing for more significant updates.

## Advantages of RMSProp:

- **Efficient Learning:** RMSProp adjusts learning rates dynamically, leading to faster convergence compared to algorithms with fixed learning rates.
- **Suitable for Non-Stationary Problems:** The emphasis on recent gradients makes RMSProp effective in settings where the cost function changes over time.
- **Reduced Need for Manual Tuning:** While hyperparameters like $\eta$ and $\gamma$ still need to be set, RMSProp reduces the sensitivity to the initial learning rate choice compared to standard gradient descent.

## Hyperparameters:

- **Initial Learning Rate ($\eta$):** Commonly set to $0.001$ by default but may require tuning based on the specific problem.
- **Decay Rate ($\gamma$):** Typically set around $0.9$. A higher value places more emphasis on past gradients, while a lower value emphasizes recent gradients more.

## Limitations:

- **Still sensible to Hyperparameters:** RMSProp's performance can be affected by the choice of $\gamma$ and $\eta$, necessitating careful tuning, despite improving upon SGD.
- **No Bias Correction:** Unlike algorithms like Adam, RMSProp does not include bias correction terms, which can lead to biased estimates in the early stages of training.
- **Memory Consumption:** RMSProp requires additional memory to store the moving average $E_{t,i}$ for each parameter, increasing the overall memory usage.

## Conclusion:

RMSProp enhances the Adagrad algorithm by incorporating an exponentially decaying average of past squared gradients, effectively adapting the learning rate for each parameter while mitigating the issue of indefinite learning rate decay. This makes RMSProp particularly well-suited for training deep neural networks and other complex models where efficient and adaptive optimization is crucial.

By balancing the consideration of past and recent gradients, RMSProp provides a more robust and responsive optimization strategy, contributing to its widespread adoption in the field of deep learning.