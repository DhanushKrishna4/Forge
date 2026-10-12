"""Immutable typed nodes in a bounded, append-only static computation graph."""
from __future__ import annotations
from dataclasses import dataclass
from types import MappingProxyType
from collections.abc import Mapping
import math
import numpy as np


class GraphError(ValueError):
    """Invalid graph structure, dtype, shape, or resource request."""


def normalize_dtype(dtype):
    name = np.dtype(dtype).name
    if name not in ('float32', 'float64', 'int32'):
        raise GraphError(f'Unsupported dtype {name}; use float32, float64, or int32 labels')
    return name


def normalize_shape(shape, max_elements=1_048_576):
    if not isinstance(shape, (tuple, list)):
        raise GraphError('Shape must be a tuple or list of positive static dimensions')
    if len(shape) > 2 or any(isinstance(d, (bool, np.bool_)) or not isinstance(d, (int, np.integer)) or d <= 0 for d in shape):
        raise GraphError('Only rank 0–2 tensors with positive integer dimensions are supported')
    shape = tuple(int(d) for d in shape)
    if math.prod(shape) > max_elements:
        raise GraphError(f'Tensor exceeds {max_elements:,} element limit')
    return shape


def freeze(value):
    if isinstance(value, Mapping):
        if any(not isinstance(k, str) for k in value):
            raise GraphError('Attribute names must be strings')
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, np.ndarray):
        return freeze(value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(freeze(v) for v in value)
    if isinstance(value, np.generic):
        return freeze(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise GraphError('Attributes must contain finite numbers')
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise GraphError('Attributes must contain immutable JSON-compatible values')


def thaw(value):
    if isinstance(value, Mapping):
        return {k: thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [thaw(v) for v in value]
    return value


@dataclass(frozen=True, slots=True)
class Node:
    id: int
    op: str
    inputs: tuple[int, ...]
    shape: tuple[int, ...]
    dtype: str
    attrs: Mapping
    name: str | None
    kind: str
    provenance: tuple[int, ...]

    def to_dict(self):
        return {field: thaw(getattr(self, field)) for field in self.__dataclass_fields__}


class Graph:
    """Graph construction mutates the node sequence, never an existing node.

    Parameters and state are named external bindings, not graph constants.
    """
    def __init__(self, dtype='float32', *, max_elements=1_048_576, max_nodes=20_000, max_parameters=100_000, max_bytes=128 * 1024 * 1024):
        self.dtype = normalize_dtype(dtype)
        if self.dtype == 'int32':
            raise GraphError('The graph arithmetic dtype must be floating point')
        limits = (max_elements, max_nodes, max_parameters, max_bytes)
        if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in limits):
            raise GraphError('Graph resource limits must be positive integers')
        self.max_elements, self.max_nodes, self.max_parameters, self.max_bytes = limits
        self._nodes = []
        self._names = set()
        self._parameter_elements = 0
        self._provenance = None

    @property
    def nodes(self):
        return tuple(self._nodes)

    def add(self, op, inputs=(), shape=(), dtype=None, attrs=None, *, name=None, kind='operation', provenance=None):
        from .tensor import Tensor
        if len(self._nodes) >= self.max_nodes:
            raise GraphError(f'Graph exceeds {self.max_nodes:,} node limit')
        shape = normalize_shape(shape, self.max_elements)
        dtype = normalize_dtype(dtype or self.dtype)
        ids = tuple(t.id if isinstance(t, Tensor) else t for t in inputs)
        if any(isinstance(t, Tensor) and t.graph is not self for t in inputs):
            raise GraphError('Cannot combine tensors from different graphs')
        if any(not isinstance(i, int) or i < 0 or i >= len(self._nodes) for i in ids):
            raise GraphError('Every input must refer to an earlier node in this graph')
        if name is not None:
            if not isinstance(name, str) or not name or len(name) > 200 or name in self._names:
                raise GraphError('Binding names must be unique, nonempty strings of at most 200 characters')
        if kind == 'parameter':
            if self._parameter_elements + math.prod(shape) > self.max_parameters:
                raise GraphError(f'Parameters exceed {self.max_parameters:,} element limit')
        i = len(self._nodes)
        origin = provenance if provenance is not None else self._provenance
        node = Node(i, str(op), ids, shape, dtype, freeze(attrs or {}), name, kind, tuple(origin) if origin is not None else (i,))
        validate_node(node, self._nodes)
        self._nodes.append(node)
        if name is not None:
            self._names.add(name)
        if kind == 'parameter':
            self._parameter_elements += math.prod(shape)
        return Tensor(self, i)

    def input(self, name, shape, dtype=None):
        return self.add('input', shape=shape, dtype=dtype, name=name, kind='input')

    def parameter(self, name, shape):
        return self.add('parameter', shape=shape, name=name, kind='parameter')

    def state(self, name, shape):
        if not name.startswith('state/'):
            raise GraphError('Optimizer state names must start with state/')
        return self.add('state', shape=shape, name=name, kind='state')

    def constant(self, value, dtype=None):
        dtype = normalize_dtype(dtype or self.dtype)
        # Inspect shape and size before a potentially allocating dtype conversion.
        if isinstance(value, (tuple, list)):
            if len(value) > self.max_elements:
                raise GraphError('Constant exceeds tensor element limit')
            if value and isinstance(value[0], (tuple, list)):
                shape = normalize_shape((len(value), len(value[0])), self.max_elements)
                if any(not isinstance(row, (tuple, list)) or len(row) != shape[1] or any(isinstance(x, (list, tuple)) for x in row) for row in value):
                    raise GraphError('Constants must be rectangular and have rank at most 2')
        raw = np.asarray(value)
        shape = normalize_shape(raw.shape, self.max_elements)
        if raw.dtype.kind not in 'fiu':
            raise GraphError('Constants must contain real numeric values')
        if not np.all(np.isfinite(raw)):
            raise GraphError('Constants must be finite')
        if dtype == 'int32' and (np.any(raw != np.floor(raw)) or np.any(raw < -2**31) or np.any(raw > 2**31-1)):
            raise GraphError('Integer constants must fit exactly into int32')
        with np.errstate(over='ignore'):
            a = np.asarray(raw, dtype=dtype)
        if not np.all(np.isfinite(a)):
            raise GraphError('Constant is outside the dtype finite range')
        return self.add('constant', shape=shape, dtype=dtype, attrs={'value': a.reshape(-1).tolist()}, kind='constant')

    def to_dict(self):
        return {'version': 1, 'dtype': self.dtype, 'nodes': [n.to_dict() for n in self._nodes]}

    def validate(self):
        """Check structural invariants without allocating tensor storage."""
        if len(self._nodes) > self.max_nodes:
            raise GraphError('Graph node limit exceeded')
        names = set()
        for i, node in enumerate(self._nodes):
            if node.id != i or any(j < 0 or j >= i for j in node.inputs):
                raise GraphError('Graph must be acyclic and topologically ordered')
            normalize_shape(node.shape, self.max_elements)
            normalize_dtype(node.dtype)
            validate_node(node, self._nodes)
            if node.name is not None:
                if node.name in names:
                    raise GraphError('Duplicate binding name')
                names.add(node.name)
        return self


def validate_node(node, nodes):
    """Re-derive each supported operation's static type and shape."""
    from .tensor import broadcast_shape, axes_for
    op, attrs = node.op, node.attrs
    attribute_keys = {'constant': {'value'}, 'sum': {'axis', 'keepdims'}, 'mean': {'axis', 'keepdims'}, 'reshape': {'shape'}, 'transpose': {'axes'}, 'broadcast_to': {'shape'}, 'softmax': {'axis'}, 'one_hot': {'classes'}}
    if set(attrs) != attribute_keys.get(op, set()):
        raise GraphError(f'Invalid attributes for {op}')
    inputs = [nodes[i] for i in node.inputs]
    if op in ('input', 'parameter', 'state', 'constant'):
        if inputs or node.kind != op:
            raise GraphError('Leaf kind must match operation, with no input edges')
        if op != 'constant' and not node.name:
            raise GraphError('Mutable leaves require names')
        if op in ('parameter', 'state') and node.dtype == 'int32':
            raise GraphError('Parameters and optimizer state must be floating point')
        if op == 'constant' and len(attrs.get('value', ())) != math.prod(node.shape):
            raise GraphError('Constant data size does not match shape')
        return
    unary = ('neg', 'relu', 'tanh', 'sigmoid', 'exp', 'log', 'sqrt', 'sum', 'mean', 'reshape', 'transpose', 'broadcast_to', 'softmax', 'one_hot')
    binary = ('add', 'sub', 'mul', 'div', 'matmul', 'greater', 'cross_entropy')
    expected_count = 1 if op in unary else 2 if op in binary else 3 if op == 'where' else None
    if expected_count is None:
        raise GraphError(f'Unsupported operation: {op}')
    if len(inputs) != expected_count or node.kind != 'operation':
        raise GraphError(f'{op} has invalid input count or node kind')
    if op not in ('cross_entropy', 'one_hot') and any(n.dtype != node.dtype for n in inputs):
        raise GraphError('Operation input/output dtypes must match')
    if node.dtype == 'int32' and op not in ('reshape', 'transpose', 'broadcast_to'):
        raise GraphError('Integer labels do not support arithmetic')
    a = inputs[0]
    if op in ('add', 'sub', 'mul', 'div', 'greater', 'where'):
        expected = broadcast_shape(*(n.shape for n in inputs))
    elif op == 'matmul':
        b = inputs[1]
        if len(a.shape) != 2 or len(b.shape) != 2 or a.shape[1] != b.shape[0]:
            raise GraphError('Matmul shape mismatch')
        expected = (a.shape[0], b.shape[1])
    elif op in ('sum', 'mean'):
        axes = axes_for(attrs.get('axis'), len(a.shape))
        if not isinstance(attrs.get('keepdims'), bool):
            raise GraphError('Reduction keepdims must be bool')
        expected = tuple(1 if i in axes else d for i, d in enumerate(a.shape)) if attrs['keepdims'] else tuple(d for i,d in enumerate(a.shape) if i not in axes)
    elif op == 'reshape':
        expected = tuple(attrs.get('shape', ()))
        if math.prod(expected) != math.prod(a.shape):
            raise GraphError('Reshape must preserve element count')
    elif op == 'transpose':
        if len(a.shape) != 2 or tuple(attrs.get('axes', ())) != (1, 0):
            raise GraphError('Only matrix transpose is supported')
        expected = a.shape[::-1]
    elif op == 'broadcast_to':
        expected = tuple(attrs.get('shape', ()))
        if broadcast_shape(a.shape, expected) != expected:
            raise GraphError('Invalid broadcast target shape')
    elif op == 'softmax':
        axes = axes_for(attrs.get('axis'), len(a.shape))
        if len(axes) != 1:
            raise GraphError('Softmax needs one valid axis')
        expected = a.shape
    elif op == 'cross_entropy':
        b = inputs[1]
        if len(a.shape) != 2 or a.dtype != node.dtype or b.dtype != 'int32' or b.shape != (a.shape[0],):
            raise GraphError('Cross-entropy requires (N,C) logits and (N,) int32 labels')
        expected = ()
    elif op == 'one_hot':
        classes = attrs.get('classes')
        if len(a.shape) != 1 or a.dtype != 'int32' or not isinstance(classes, int) or isinstance(classes, bool) or classes <= 0:
            raise GraphError('One-hot requires int32 labels and a positive class count')
        expected = (a.shape[0], classes)
    else:
        expected = a.shape
    if node.shape != expected:
        raise GraphError(f'{op} shape {node.shape} does not match inferred shape {expected}')
