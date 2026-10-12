# Tensor contract

Forge builds an append-only graph of immutable nodes. Expressions do not run until `evaluate` or a compiled execution plan runs. A tensor is a graph and stable integer node ID. Data lives in named bindings; replacing a parameter array between completed steps does not change the graph. Mixing graphs or arithmetic dtypes is an error. Data-dependent Python branches, in-place tensor mutation, dynamic shapes and higher-order derivatives are unsupported.

## Shapes, dtypes and storage

- Ranks 0, 1 and 2 represent scalars `()`, vectors `(N,)` and matrices `(M,N)`.
- Every dimension is a positive static integer. Empty dimensions, inferred `-1` reshape dimensions and ranks above two are rejected.
- Arithmetic supports float32 and native NumPy float64. WebGPU uses float32; float64 is for native validation.
- Class-label inputs use int32 and are nondifferentiable. No implicit arithmetic casts or broadcasting labels into batches occurs.
- Python numeric literals are explicitly constructed as constants with the neighboring tensor's dtype. External NumPy bindings must already have their declared dtype and exact shape; float64 arrays are never silently downcast.
- Operations return materialized contiguous arrays. Transpose, reshape and broadcast create independent storage. There are no aliased views exposed by execution.
- Graph defaults bound a tensor to 1,048,576 elements, a model to 100,000 parameter elements and a graph to 20,000 nodes. The reference interpreter rejects reachable tensor storage above 128 MiB before execution. Configure these through `Graph(max_elements=..., max_parameters=..., max_nodes=..., max_bytes=...)`. NumPy temporary storage and Python/runtime overhead are additional and are not claimed as Forge-owned tensor storage.

## Public operations

| Expression | Semantics and restrictions |
|---|---|
| `a+b`, `a-b`, `a*b`, `a/b`, `-a` | NumPy broadcasting, bounded to rank 2. Division requires every denominator to be nonzero. |
| `a @ b` | Rank-2 `(M,K) @ (K,N) → (M,N)` only. |
| `x.relu()` | `max(x,0)`, with derivative zero at zero. |
| `x.tanh()`, `x.sigmoid()` | Standard functions. Sigmoid uses a sign-dependent stable exponential formula. |
| `x.exp()` | Exponential. Overflow is a numerical error, not clipping. |
| `x.log()` | Natural logarithm; requires strictly positive input. |
| `x.sqrt()` | Nonnegative input. The derivative at zero is singular; evaluating it is a numerical error. |
| `x.sum(axis=None, keepdims=False)` | Reduce all axes by default. An integer or tuple selects axes, negative axes normalize by rank, duplicate/invalid axes fail. Empty tuple reduces no axes. Scalars have no axes; use `None` or `()` for a scalar reduction. |
| `x.mean(axis=None, keepdims=False)` | Sum divided by the product of reduced dimensions. Same axes rules. |
| `x.reshape(shape)` | Explicit dimensions, unchanged element count, rank ≤2; no inferred dimension. |
| `x.transpose()` or `x.T` | Matrices only; exchange dimensions 0 and 1. |
| `x.broadcast_to(shape)` | NumPy trailing-axis broadcast, preserving or expanding dimensions of length one. |
| `x.softmax(axis=-1)` | Vectors or matrices, one axis. Subtract the maximum along the selected axis before exponentiating. |
| `cross_entropy(logits, labels)` | Floating `(N,C)` logits, int32 `(N,)` labels in `[0,C)`, scalar mean loss. |
| `mse(prediction, target)` | Mean of the squared broadcast difference across all output elements. |

`keepdims=True` retains reduced axes with size 1. `keepdims=False` removes them. Dtypes remain unchanged during reductions.

## Internal executable operations

The backward graph and optimizers use exactly the same operations, plus:

| Operation | Purpose |
|---|---|
| `greater(a,b)` | Broadcast comparison; returns floating masks 0 or 1 in the arithmetic dtype, with zero derivative. |
| `where(condition,a,b)` | Broadcast selection; select `a` when condition is nonzero. Gradients flow into selected values, never into the condition. |
| `one_hot(labels, classes)` | Int32 labels to floating `(N,C)` indicators; labels are not differentiated. |
| reduction to shape | Represented by ordinary `sum(..., keepdims=True)` and `reshape`, rather than a second hidden evaluator. |

## Stable cross-entropy

For each row, let `s = logits - max(logits)`. Compute

`loss = mean(log(sum(exp(s))) - s[row, label])`.

This avoids taking `log` of a probability that underflowed to zero. The gradient is `(softmax(logits) - one_hot(labels))/N`. Large finite logits, ties, one-class outputs and batch size one are covered by numerical tests.

Bindings and constants must be finite. The NumPy reference raises `GraphError` for domain violations or nonfinite results. Finite inputs do not guarantee finite arithmetic: subtraction or a reduction may still overflow at the extremes of a dtype. Forge does not clip or silently change these functions. GPU arithmetic follows portable float32 behavior; the browser validates input buffers and returned readbacks, but unobserved intermediate domain failures are not individually trapped by a shader.
