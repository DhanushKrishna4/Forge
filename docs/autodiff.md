# Reverse-mode differentiation and updates

`grad(loss, parameters)` requires a scalar floating loss and constructs a fresh backward graph. It traverses reachable forward nodes in reverse topological order, accumulates contributions for each node, and returns one ordinary tensor per requested target. The original forward nodes never change.

```python
import numpy as np
from forge_ml import Graph, cross_entropy, grad, evaluate

g = Graph('float64')
x = g.input('x', (4, 2))
w = g.parameter('w', (2, 3))
y = g.input('y', (4,), 'int32')
loss = cross_entropy(x @ w, y)
dw, = grad(loss, [w])
values = evaluate([loss, dw], {
    'x': np.ones((4, 2), dtype='float64'),
    'w': np.ones((2, 3), dtype='float64'),
    'y': np.array([0, 1, 2, 0], dtype='int32'),
})
```

A repeated parameter or shared subexpression receives the sum of contributions from every downstream consumer. An unused target receives an explicit zero graph. There is no mutable `.grad` field and no accumulation across execution steps. Integer labels, comparison masks and branch choices are nondifferentiable. ReLU uses derivative zero at zero. Higher-order derivatives are rejected.

## Why broadcast gradients need reductions

If `a.shape == (3,)` and `b.shape == (2,3)`, then `a+b` uses each element of `a` twice. Its gradient must sum over the new leading axis. If `a.shape == (2,1)`, its single column was expanded to three columns, so its gradient sums over the expanded trailing axis instead. Forge left-pads the original shape with ones, finds **every** expanded axis, sums with retained dimensions, then reshapes back to the original shape. This also handles scalar broadcasts.

A reduction's backward operation reverses that process: reinsert removed axes with size one, broadcast to the original shape, and divide by the number of reduced elements for a mean. Reshape restores the original shape. Matrix transpose transposes the upstream gradient.

For `Y = A @ B`, use `dA = dY @ B.T` and `dB = A.T @ dY`. These are ordinary rank-2 matrix multiplications in the backward graph.

For softmax output `p` and upstream derivative `u`, the Jacobian-vector product is `p * (u - sum(u*p, axis, keepdims=True))`. No rank-3 Jacobian is allocated. Stable cross-entropy uses the closed-form `(p - one_hot(labels))/batch_size`, scaled by its upstream scalar derivative.

Nodes created while processing a forward operation retain its node ID in `provenance`. This connects derivative work to source operations. Compile the combined forward/backward/update graph only after differentiating; requested loss and gradient outputs stay roots of compilation.

## Optimizers are graph outputs

`optimizer_updates(parameters, gradients, kind=...)` returns a mapping of binding names to update tensors. Parameters, optimizer moments and the step counter are mutable named bindings, never constants.

- SGD: `p_next = p - lr*g`.
- Momentum: `v_next = momentum*v + g`, then `p_next = p - lr*v_next`. Velocity starts at zero; no dampening or Nesterov variant is implied.
- Adam: `m_next = beta1*m + (1-beta1)*g`; `v_next = beta2*v + (1-beta2)*g*g`; increment scalar `state/step`; divide moments by `1-beta**step`; then `p_next = p - lr*m_hat/(sqrt(v_hat)+eps)`.

Adam's epsilon is outside the square root. Beta values may be zero. Bias-correction powers are expressed as portable graph operations `exp(step*log(beta))`, with a constant correction of one for beta zero. Optimizer state uses the graph's arithmetic dtype; the bounded application training budget stays far below float32's exact-integer step limit.

`initial_bindings` validates parameter arrays and initializes all state to zero. `apply_updates` evaluates all requested outputs and every update from a single snapshot of old bindings. Only after successful evaluation does it replace parameter and state arrays. A failed operation leaves the old bindings intact. The GPU schedule follows the same out-of-place update boundary: it must not overwrite old parameter/state storage while a forward or backward consumer still needs it.

A checkpoint must include every parameter, every moment or velocity, and Adam's completed step counter. Tests compare uninterrupted and resumed updates, check explicit optimizer formulas, and compare five complete MLP updates independently against PyTorch using identical arrays.

## Numerical validation

Fast tests compare forward results with NumPy and use central finite differences in float64 (`epsilon=1e-6`, or `1e-4` for logits scaled by 1000). Typical finite-difference tolerances are relative `2e-6` and absolute `2e-8`, allowing subtraction cancellation without tolerating incorrect scaling. Ordinary finite-difference fixtures stay away from nondifferentiable points; ReLU at zero is checked separately.

Development-only PyTorch tests use relative `3e-5`, absolute `3e-7` for float32 and relative `3e-11`, absolute `3e-12` for float64. Error assertions include maximum absolute and relative error; the five-step training checks print both. Relative errors near zero can look large, so the absolute tolerance remains essential. These tolerances apply to small native CPU fixtures, not an assertion of universal CPU/GPU bitwise agreement.
