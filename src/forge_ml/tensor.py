"""Small graph-building tensor API. No eager execution or implicit mutation."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from .graph import GraphError, normalize_shape


def broadcast_shape(*shapes):
    try:
        return tuple(np.broadcast_shapes(*shapes))
    except ValueError as exc:
        raise GraphError(f'Incompatible broadcast shapes: {shapes}') from exc


def axes_for(axis, rank):
    if axis is None:
        return tuple(range(rank))
    axes = (axis,) if isinstance(axis, (int, np.integer)) else tuple(axis)
    if any(isinstance(a, (bool, np.bool_)) or not isinstance(a, (int, np.integer)) or not -rank <= a < rank for a in axes):
        raise GraphError(f'Invalid reduction axis {axis} for rank {rank}')
    normalized = tuple(sorted(int(a) % rank for a in axes))
    if len(set(normalized)) != len(normalized):
        raise GraphError('Reduction axes may not repeat')
    return normalized


@dataclass(frozen=True, slots=True)
class Tensor:
    graph: object
    id: int

    @property
    def node(self):
        return self.graph._nodes[self.id]

    @property
    def shape(self):
        return self.node.shape

    @property
    def dtype(self):
        return self.node.dtype

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return math.prod(self.shape)

    def __bool__(self):
        raise TypeError('Graph tensors cannot control Python branches; use explicit selection')

    def __eq__(self, other):
        raise TypeError('Tensor equality is not Python data-dependent control flow; compare node ids for graph identity')

    def __ne__(self, other):
        raise TypeError('Tensor inequality is not Python data-dependent control flow; compare node ids for graph identity')

    def __array__(self, *args, **kwargs):
        raise TypeError('Graph tensors have no eager array value; call evaluate')

    def _coerce(self, other):
        if not isinstance(other, Tensor):
            other = self.graph.constant(other, self.dtype)
        if other.graph is not self.graph:
            raise GraphError('Cannot combine tensors from different graphs')
        if other.dtype != self.dtype:
            raise GraphError('Mixed dtype arithmetic is not supported')
        if self.dtype == 'int32':
            raise GraphError('Integer labels are nondifferentiable and cannot be used for arithmetic')
        return other

    def _binary(self, op, other):
        other = self._coerce(other)
        return self.graph.add(op, (self, other), broadcast_shape(self.shape, other.shape), self.dtype)

    def _unary(self, op):
        if self.dtype == 'int32':
            raise GraphError('Integer labels do not support floating point operations')
        return self.graph.add(op, (self,), self.shape, self.dtype)

    def __add__(self, other): return self._binary('add', other)
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self._binary('sub', other)
    def __rsub__(self, other): return (-self) + other
    def __mul__(self, other): return self._binary('mul', other)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self._binary('div', other)
    def __rtruediv__(self, other): return self._coerce(other)._binary('div', self)
    def __neg__(self): return self._unary('neg')

    def __matmul__(self, other):
        other = self._coerce(other)
        if self.ndim != 2 or other.ndim != 2 or self.shape[1] != other.shape[0]:
            raise GraphError('Matmul requires matrices (M,K) @ (K,N)')
        return self.graph.add('matmul', (self, other), (self.shape[0], other.shape[1]), self.dtype)

    def relu(self): return self._unary('relu')
    def tanh(self): return self._unary('tanh')
    def sigmoid(self): return self._unary('sigmoid')
    def exp(self): return self._unary('exp')
    def log(self): return self._unary('log')
    def sqrt(self): return self._unary('sqrt')
    def greater(self, other): return self._binary('greater', other)

    def _reduce(self, op, axis=None, keepdims=False):
        if self.dtype == 'int32':
            raise GraphError('Integer labels do not support reductions')
        if not isinstance(keepdims, bool):
            raise GraphError('keepdims must be a bool')
        axes = axes_for(axis, self.ndim)
        shape = tuple(1 if i in axes else n for i, n in enumerate(self.shape)) if keepdims else tuple(n for i, n in enumerate(self.shape) if i not in axes)
        return self.graph.add(op, (self,), shape, self.dtype, {'axis': axes, 'keepdims': keepdims})

    def sum(self, axis=None, keepdims=False): return self._reduce('sum', axis, keepdims)
    def mean(self, axis=None, keepdims=False): return self._reduce('mean', axis, keepdims)

    def reshape(self, shape):
        shape = normalize_shape(shape, self.graph.max_elements)
        if math.prod(shape) != self.size:
            raise GraphError('Reshape must preserve the number of elements; inferred -1 dimensions are not supported')
        return self.graph.add('reshape', (self,), shape, self.dtype, {'shape': shape})

    def transpose(self):
        if self.ndim != 2:
            raise GraphError('Transpose is supported for matrices only')
        return self.graph.add('transpose', (self,), tuple(reversed(self.shape)), self.dtype, {'axes': (1, 0)})

    @property
    def T(self): return self.transpose()

    def broadcast_to(self, shape):
        shape = normalize_shape(shape, self.graph.max_elements)
        if broadcast_shape(self.shape, shape) != shape:
            raise GraphError(f'Cannot broadcast {self.shape} to {shape}')
        return self.graph.add('broadcast_to', (self,), shape, self.dtype, {'shape': shape})

    def softmax(self, axis=-1):
        axes = axes_for(axis, self.ndim)
        if len(axes) != 1:
            raise GraphError('Softmax requires exactly one valid axis on a vector or matrix')
        if self.dtype == 'int32':
            raise GraphError('Softmax requires floating point values')
        return self.graph.add('softmax', (self,), self.shape, self.dtype, {'axis': axes[0]})


def where(condition, x, y):
    """Select x when a floating comparison mask is nonzero, otherwise y."""
    if not isinstance(x, Tensor):
        if not isinstance(y, Tensor):
            raise GraphError('At least one selected branch must be a Tensor')
        x = y._coerce(x)
    y = x._coerce(y)
    condition = x._coerce(condition)
    return x.graph.add('where', (condition, x, y), broadcast_shape(condition.shape, x.shape, y.shape), x.dtype)


def one_hot(labels, classes, dtype=None):
    if labels.dtype != 'int32' or labels.ndim != 1 or not isinstance(classes, int) or isinstance(classes, bool) or classes <= 0:
        raise GraphError('One-hot labels must be an int32 vector and classes a positive integer')
    return labels.graph.add('one_hot', (labels,), (labels.shape[0], classes), dtype or labels.graph.dtype, {'classes': classes})


def cross_entropy(logits, labels):
    if not isinstance(labels, Tensor) or labels.graph is not logits.graph:
        raise GraphError('Labels and logits must belong to the same graph')
    if logits.dtype == 'int32' or logits.ndim != 2 or labels.dtype != 'int32' or labels.shape != (logits.shape[0],):
        raise GraphError('Cross-entropy requires floating logits (batch,classes) and int32 labels (batch,)')
    return logits.graph.add('cross_entropy', (logits, labels), (), logits.dtype)


def mse(prediction, target):
    delta = prediction - target
    return (delta * delta).mean()
