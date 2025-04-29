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

## Flow Constraint Equation
We perform Taylor expansion of $I(x+u,y+v,t)$ with respect to $(x,y,0)$

$$
I(x+u,y+v,\tau) = I(x,y,0) + \frac{\delta I(x,y,0)}{\delta x}u + \frac{\delta I(x,y,0)}{\delta y}v + \frac{\delta I(x,y,0)}{\delta t}\tau + O(t^2)
$$

Denote:
- $I_x = \frac{\delta I(x,y,0)}{\delta x}$
- $I_y = \frac{\delta I(x,y,0)}{\delta y}$
- $I_t = \frac{\delta I(x,y,0)}{\delta t}$

Since $I(x+u,y+v,\tau) = I(x,y,0)$:

$$
\rightarrow I_x u + I_y v + I_t \tau = 0
$$

### Solution
$$
D(u,v) = \Sigma_x \Sigma_y \left(I_x u + I_y v + I_t \tau\right)^2
$$

$$
\nabla D(u,v) = \begin{bmatrix}\Sigma_x \Sigma_y \left(I_x u + I_y v + I_t \tau\right)I_x \\ \Sigma_x \Sigma_y \left(I_x u + I_y v + I_t \tau\right)I_y \end{bmatrix} = 0
$$

$$
\Sigma_x \Sigma_y \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}\begin{bmatrix} u \\ v \end{bmatrix} = -\tau \Sigma_x \Sigma_y \begin{bmatrix}I_x I_t \\ I_y I_t\end{bmatrix}
$$

$$
\begin{bmatrix} u \\ v \end{bmatrix} = - \tau \left(\Sigma_x \Sigma_y \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}\right)^{-1}\left(\Sigma_x \Sigma_y \begin{bmatrix}I_x I_t \\ I_y I_t\end{bmatrix}\right)
$$
- $I_t$ is the image difference

This is a closed form solution. Important note is that we cannot determine $\tau$. We solve for $u/\tau$ and $v/\tau$ velocities. This only provides a direction to search.

## Handling Rotation
Assuming a pure rotation:
$$
R(\theta) = \begin{bmatrix}\cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}
$$

**Objective Function**
$$
D(\theta) = \Sigma_x \Sigma_y \left[I\left(R(\theta)\begin{bmatrix}x\\ y\end{bmatrix}, \tau\right) - I(x,y,0)\right]^2
$$

**Taylor Expansion**
$$
I\left(R(\theta)\begin{bmatrix}x\\ y\end{bmatrix}, \tau\right) = I(x,y,0) + \frac{\delta I}{\delta\theta}\theta + \frac{\delta I}{\delta t}\tau + o(t^2) \\
\text{ where } \frac{\delta I}{\delta\theta} = -\frac{delta I}{\delta x}y + \frac{\delta I}{\delta y}x = I_{\theta}
$$

**Derivative**
$$
D(\theta) = \Sigma_x \Sigma_y (I_{\theta} + I_t \tau)^2 \rightarrow \nabla D(\theta) = \Sigma_x \Sigma_y (I_{\theta}\theta + I_t \tau)I_{\theta}
$$

**Solution**
$$
\rightarrow \theta = -\tau \frac{\Sigma_x \Sigma_y I_{\theta}I_t}{\Sigma_x \Sigma_y I_{\theta}^2}
$$