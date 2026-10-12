"""Reverse-mode differentiation builds ordinary operations in the same graph."""
import math
from .graph import GraphError
from .tensor import Tensor, one_hot, where


def reduce_to_shape(value, shape):
    """Sum every axis introduced or expanded by NumPy broadcasting."""
    if value.shape == shape:
        return value
    padded = (1,) * (value.ndim - len(shape)) + tuple(shape)
    axes = tuple(i for i, (source, target) in enumerate(zip(value.shape, padded)) if target == 1 and source != 1)
    if axes:
        value = value.sum(axis=axes, keepdims=True)
    return value.reshape(shape)


def grad(loss, parameters):
    """Differentiate scalar loss; return fresh executable gradient tensors.

    ReLU's derivative is zero at zero. Labels, masks and branch decisions are
    nondifferentiable. Higher-order differentiation is deliberately unsupported.
    """
    if loss.shape != () or loss.dtype == 'int32':
        raise GraphError('grad requires a floating point scalar loss')
    graph = loss.graph
    parameters = list(parameters)
    if any(p.graph is not graph or p.dtype != loss.dtype for p in parameters):
        raise GraphError('Differentiation targets must have the same graph and floating dtype as loss')
    needed, stack = set(), [loss.id]
    while stack:
        i = stack.pop()
        if i not in needed:
            needed.add(i)
            stack.extend(graph._nodes[i].inputs)
    if needed & getattr(graph, '_backward_ids', set()):
        raise GraphError('Higher-order differentiation is not supported')
    start = len(graph._nodes)
    grads = {loss.id: graph.constant(1, loss.dtype)}
    old_provenance = graph._provenance
    try:
        for i in sorted(needed, reverse=True):
            if i not in grads:
                continue
            node, upstream = graph._nodes[i], grads[i]
            graph._provenance = (i,)
            xs = [Tensor(graph, j) for j in node.inputs]
            y = Tensor(graph, i)
            op = node.op
            if op in ('input', 'parameter', 'state', 'constant', 'greater', 'one_hot'):
                continue
            if op == 'add': gs = (upstream, upstream)
            elif op == 'sub': gs = (upstream, -upstream)
            elif op == 'mul': gs = (upstream * xs[1], upstream * xs[0])
            elif op == 'div': gs = (upstream / xs[1], -(upstream * xs[0]) / (xs[1] * xs[1]))
            elif op == 'neg': gs = (-upstream,)
            elif op == 'matmul': gs = (upstream @ xs[1].T, xs[0].T @ upstream)
            elif op == 'relu': gs = (upstream * xs[0].greater(0),)
            elif op == 'tanh': gs = (upstream * (1 - y * y),)
            elif op == 'sigmoid': gs = (upstream * y * (1 - y),)
            elif op == 'exp': gs = (upstream * y,)
            elif op == 'log': gs = (upstream / xs[0],)
            elif op == 'sqrt': gs = (upstream * 0.5 / y,)
            elif op in ('sum', 'mean'):
                axes = node.attrs['axis']
                expanded = tuple(1 if axis in axes else n for axis, n in enumerate(xs[0].shape))
                gx = upstream if node.attrs['keepdims'] else upstream.reshape(expanded)
                gx = gx.broadcast_to(xs[0].shape)
                if op == 'mean':
                    gx = gx / math.prod(xs[0].shape[a] for a in axes)
                gs = (gx,)
            elif op == 'reshape': gs = (upstream.reshape(xs[0].shape),)
            elif op == 'transpose': gs = (upstream.T,)
            elif op == 'broadcast_to': gs = (reduce_to_shape(upstream, xs[0].shape),)
            elif op == 'softmax':
                # The Jacobian-vector product needs no rank-3 explicit Jacobian.
                axis = node.attrs['axis']
                gs = (y * (upstream - (upstream * y).sum(axis=axis, keepdims=True)),)
            elif op == 'cross_entropy':
                probabilities = xs[0].softmax(axis=1)
                gs = (upstream * (probabilities - one_hot(xs[1], xs[0].shape[1], node.dtype)) / xs[0].shape[0], None)
            elif op == 'where': gs = (None, where(xs[0], upstream, 0), where(xs[0], 0, upstream))
            else: raise GraphError(f'No differentiation rule for {op}')
            for x, contribution in zip(xs, gs):
                if contribution is None or x.dtype == 'int32':
                    continue
                contribution = reduce_to_shape(contribution, x.shape)
                grads[x.id] = contribution if x.id not in grads else grads[x.id] + contribution
        result = []
        for p in parameters:
            if p.id in grads:
                result.append(grads[p.id])
            else:
                zero = graph.constant(0, p.dtype)
                result.append(zero if p.shape == () else zero.broadcast_to(p.shape))
        return result
    finally:
        graph._provenance = old_provenance
        graph._backward_ids = getattr(graph, '_backward_ids', set()) | set(range(start, len(graph._nodes)))
