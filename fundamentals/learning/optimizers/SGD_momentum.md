# Stochastic Gradient Descent (SGD) with Momentum

SGD with momentum is an optimization algorithm that enhances the standard SGD by incorporating a momentum term. This momentum term helps accelerate convergence in the direction of consistent gradients and dampens oscillations in high-curvature areas of the parameter space.

## Update Equations:

1. **Velocity Update:**
   $v_t = \beta v_{t-1} + \frac{\partial J}{\partial \theta_{i-1}}$
   
   - $v_t$ is the velocity vector at iteration $t$.
   - $\beta$ is the momentum coefficient (typically close to 1, e.g., 0.9).
   - $\frac{\partial J}{\partial \theta_{i-1}}$ is the gradient of the cost function $J$ with respect to parameters $\theta$ at iteration $t-1$.

2. **Parameter Update:**
   $\theta_t = \theta_{t-1} - \alpha v_t$
   
   - $\theta_t$ are the updated parameters.
   - $\alpha$ is the learning rate.

## Purpose of Momentum:

- **Accelerate Convergence:** By accumulating past gradients, momentum allows the optimizer to build up speed in directions where the gradient consistently points, moving more swiftly towards minima.
- **Dampen Oscillations:** Momentum reduces fluctuations in the parameter updates, especially in directions with high curvature, leading to smoother convergence.

## Dampened Momentum Variant:

In some implementations, a dampening factor is introduced to prevent extreme accelerations and improve numerical stability. The modified velocity update is:

$v_t = \beta v_{t-1} + (1 - \beta) \frac{\partial J}{\partial \theta_{i-1}}$

This equation scales the gradient by $(1 - \beta)$, effectively computing an exponential moving average (EMA) of the gradients. While this variant better dampens oscillations, it may accumulate less velocity, potentially slowing down convergence.

## Key Concepts:

1. **Optimizer Components:**
   - **Gradient:** Provides the direction for updating parameters.
   - **Learning Rate ($\alpha$)**: Determines the step size in the direction of the gradient.

2. **Impact of Momentum:**
   - Momentum modifies the gradient term by adding a fraction of the previous velocity, allowing for accumulated movement in consistent directions and reduced oscillations.

3. **Trade-offs Between Variants:**
   - **Standard Momentum (without dampening):** Better at accumulating velocity, leading to faster convergence but may experience larger oscillations.
   - **Dampened Momentum (with EMA):** Provides greater numerical stability and better dampens oscillations but may not accelerate convergence as effectively.

## Hyperparameters:

- **Learning Rate ($\alpha$):** Typically set to a small value (e.g., 0.01 or 0.001), the learning rate controls how large the updates to the parameters will be. It requires tuning depending on the problem.
- **Momentum Coefficient ($\beta$):** A value between 0 and 1 (commonly set to 0.9) that determines how much of the previous velocity is retained. Higher values accumulate more momentum, but if too high, may lead to overshooting.

## Conclusion:

SGD with momentum improves optimization by combining the benefits of accelerated convergence and reduced oscillations. The choice between using standard momentum or its dampened variant depends on the specific optimization problem and the desired balance between speed and stability.
