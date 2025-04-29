# Motion Estimation
- Motivation
- Basic Questions
- Exhaustive Search
- Flow-Constraint Equation
- Gradient-based Search

## Matching Criterion

**Sum of Squared Difference (SSD)**
$$
D = \Sigma_x \Sigma_y \left[I(x,y) - T(x,y)\right]^2
$$

**Cross-Correlation**
$$
C = \Sigma_x \Sigma_y I(x,y)T(x,y)
$$

**Normalized Cross-Correlation**
$$
N = \frac{\Sigma_x \Sigma_y \left[I(x,y) - \bar{I}\right]\left[T(x,y) - \bar{T}\right]}{\sqrt{\left[\Sigma_x \Sigma_y \left(I(x,y) - \bar{I}\right)^2\right]\left[\Sigma_x \Sigma_y \left(T(x,y) - \bar{T}\right)^2\right]}}
$$

## Exhaustive Search
Search all locations nearby (search window). Easy to implement but computationally intensive and can't handle rotation.

## Gradient-based Search
- Let $I(x,y,\tau) \to I$ (current image)
- Let $I(x,y,0) \to$ reference image or template

Assuming a pure translational motion and using Constant Brightness Constraint:
- $I(x,y,0) = I(x+u,y+v,\tau) \forall (x,y) \in \R$
- Where $(u,v)$ is the displacement

$$
\begin{align*}
\left(u^*, v^*\right) &= \argmin_{(u,v)}D(u,v) \\
&= \argmin_{(u,v)}\Sigma_x \Sigma_y \left[I(x+u,y+v,\tau) - I(x,y,0)\right]^2
\end{align*}
$$