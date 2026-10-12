"""Reference NumPy interpreter for the same IR consumed by the GPU compiler."""
from __future__ import annotations
import numpy as np
import math
from .graph import GraphError


def _labels(labels, classes):
    if labels.dtype != np.dtype('int32') or labels.ndim != 1:
        raise GraphError('Class labels must be an int32 vector')
    if np.any(labels < 0) or np.any(labels >= classes):
        raise GraphError(f'Class labels must lie in [0, {classes})')


def apply_op(op, inputs, attrs, dtype, shape):
    """Execute one operation; also used by compiler folding and plan validation.

    IEEE overflow/nonfinite results are errors, not silently clipped values.
    Every returned array owns contiguous storage, including layout operations.
    """
    a = inputs[0] if inputs else None
    b = inputs[1] if len(inputs) > 1 else None
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
            if op == 'constant': out = np.array(attrs['value'], dtype=dtype).reshape(shape)
            elif op == 'add': out = a + b
            elif op == 'sub': out = a - b
            elif op == 'mul': out = a * b
            elif op == 'div':
                if np.any(b == 0): raise GraphError('Division requires a nonzero denominator')
                out = a / b
            elif op == 'neg': out = -a
            elif op == 'matmul': out = a @ b
            elif op == 'relu': out = np.maximum(a, 0)
            elif op == 'tanh': out = np.tanh(a)
            elif op == 'sigmoid':
                e = np.exp(-np.abs(a))
                out = np.where(a >= 0, 1 / (1 + e), e / (1 + e))
            elif op == 'exp': out = np.exp(a)
            elif op == 'log':
                if np.any(a <= 0): raise GraphError('Log requires strictly positive input')
                out = np.log(a)
            elif op == 'sqrt':
                if np.any(a < 0): raise GraphError('Square root requires nonnegative input')
                out = np.sqrt(a)
            elif op in ('sum', 'mean'):
                fn = np.sum if op == 'sum' else np.mean
                out = fn(a, axis=tuple(attrs['axis']), keepdims=attrs['keepdims'], dtype=dtype)
            elif op == 'reshape': out = a.reshape(shape)
            elif op == 'transpose': out = a.T
            elif op == 'broadcast_to': out = np.broadcast_to(a, shape)
            elif op == 'softmax':
                axis = attrs['axis']
                e = np.exp(a - np.max(a, axis=axis, keepdims=True))
                out = e / np.sum(e, axis=axis, keepdims=True)
            elif op == 'cross_entropy':
                _labels(b, a.shape[1])
                if b.shape != (a.shape[0],): raise GraphError('Label batch size does not match logits')
                shifted = a - np.max(a, axis=1, keepdims=True)
                # Working entirely with shifted logits avoids log(underflowed softmax).
                log_z = np.log(np.sum(np.exp(shifted), axis=1))
                out = np.mean(log_z - shifted[np.arange(a.shape[0]), b], dtype=dtype)
            elif op == 'one_hot':
                classes = attrs['classes']
                _labels(a, classes)
                out = np.zeros(shape, dtype=dtype)
                out[np.arange(a.shape[0]), a] = 1
            elif op == 'greater': out = (a > b).astype(dtype)
            elif op == 'where': out = np.where(a != 0, b, inputs[2])
            else: raise GraphError(f'Unsupported operation: {op}')
    except FloatingPointError as exc:
        raise GraphError(f'{op} produced a nonfinite value or violated its numerical domain') from exc
    out = np.array(out, dtype=dtype, copy=True, order='C')
    if out.shape != tuple(shape):
        raise GraphError(f'{op} produced shape {out.shape}, expected {shape}')
    if not np.all(np.isfinite(out)):
        raise GraphError(f'{op} produced nonfinite values')
    return out


def validate_binding(node, value):
    """Validate without dtype coercion; callers own any explicit conversion."""
    value = np.asarray(value)
    if value.shape != node.shape:
        raise GraphError(f'Binding {node.name!r}: expected shape {node.shape}, got {value.shape}')
    if value.dtype != np.dtype(node.dtype):
        raise GraphError(f'Binding {node.name!r}: expected dtype {node.dtype}, got {value.dtype}')
    if not np.all(np.isfinite(value)):
        raise GraphError(f'Binding {node.name!r} contains nonfinite values')
    return np.array(value, copy=True, order='C')


def evaluate(outputs, bindings, capture=False):
    """Execute reachable nodes; return output-id values, or all reachable values.

    All input/parameter/state arrays are copied before execution. Consequently a
    complete optimizer update observes one consistent version of every binding.
    """
    if not outputs:
        return {}
    graph = outputs[0].graph
    if any(t.graph is not graph for t in outputs):
        raise GraphError('All outputs must belong to one graph')
    graph.validate()
    needed = set()
    stack = [t.id for t in outputs]
    while stack:
        i = stack.pop()
        if i not in needed:
            needed.add(i)
            stack.extend(graph._nodes[i].inputs)
    # This simple interpreter retains intermediates for the execution. Bound its
    # graph-owned arrays independently of Python/NumPy runtime overhead.
    required_bytes = sum(math.prod(graph._nodes[i].shape) * np.dtype(graph._nodes[i].dtype).itemsize for i in needed)
    if required_bytes > graph.max_bytes:
        raise GraphError(f'Execution requires {required_bytes:,} tensor bytes, exceeding {graph.max_bytes:,} byte limit')
    values = {}
    # Take a consistent snapshot of leaves before any operation is executed.
    for i in sorted(needed):
        node = graph._nodes[i]
        if node.kind in ('input', 'parameter', 'state'):
            if node.name not in bindings:
                raise GraphError(f'Missing binding {node.name!r}')
            values[i] = validate_binding(node, bindings[node.name])
    for i in sorted(needed):
        if i in values:
            continue
        node = graph._nodes[i]
        values[i] = apply_op(node.op, [values[j] for j in node.inputs], node.attrs, node.dtype, node.shape)
    return values if capture else {t.id: values[t.id] for t in outputs}
