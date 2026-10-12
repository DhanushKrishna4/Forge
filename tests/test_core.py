import dataclasses
import json
import numpy as np
import pytest
from forge_ml import Graph, GraphError, cross_entropy, evaluate, grad, mse, where


def value(output, bindings):
    return evaluate([output], bindings)[output.id]


def finite_difference(loss, parameter, bindings, eps=1e-6):
    original = bindings[parameter.node.name]
    result = np.zeros_like(original)
    for index in np.ndindex(original.shape):
        plus, minus = original.copy(), original.copy()
        plus[index] += eps
        minus[index] -= eps
        hi = value(loss, {**bindings, parameter.node.name: plus})
        lo = value(loss, {**bindings, parameter.node.name: minus})
        result[index] = (hi - lo) / (2 * eps)
    return result


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('operation,reference', [
    (lambda x: x + 2, lambda x: x + 2),
    (lambda x: 2 - x, lambda x: 2 - x),
    (lambda x: x * x, lambda x: x * x),
    (lambda x: x / 2, lambda x: x / 2),
    (lambda x: 2 / x, lambda x: 2 / x),
    (lambda x: -x, lambda x: -x),
    (lambda x: x.relu(), lambda x: np.maximum(x, 0)),
    (lambda x: x.tanh(), np.tanh),
    (lambda x: x.sigmoid(), lambda x: 1 / (1 + np.exp(-x))),
    (lambda x: x.exp(), np.exp),
    (lambda x: x.log(), np.log),
    (lambda x: x.sqrt(), np.sqrt),
])
@pytest.mark.parametrize('shape', [(), (3,), (2, 3)])
def test_forward_elementwise(dtype, operation, reference, shape):
    g = Graph(dtype)
    x = g.input('x', shape)
    data = np.linspace(0.3, 1.3, max(1, int(np.prod(shape)))).reshape(shape).astype(dtype)
    np.testing.assert_allclose(value(operation(x), {'x': data}), reference(data), rtol=2e-6, atol=2e-7)


@pytest.mark.parametrize('axis,keepdims', [(None, False), (None, True), (0, False), (1, False), (-1, True), ((0, 1), False), ((), False)])
@pytest.mark.parametrize('op', ['sum', 'mean'])
def test_reductions_and_gradients(axis, keepdims, op):
    g = Graph('float64'); x = g.parameter('x', (2, 3))
    a = np.array([[0.2, 0.4, 0.8], [1.1, 1.4, 2.1]])
    y = getattr(x, op)(axis, keepdims)
    np.testing.assert_allclose(value(y, {'x': a}), getattr(np, op)(a, axis=axis, keepdims=keepdims))
    loss = (y * y).sum()
    dx = grad(loss, [x])[0]
    np.testing.assert_allclose(value(dx, {'x': a}), finite_difference(loss, x, {'x': a}), rtol=2e-6, atol=2e-8)


@pytest.mark.parametrize('operation', [
    lambda x: (x + 1) * x - x / (x + 2),
    lambda x: 3 / x,
    lambda x: x.relu(),
    lambda x: x.tanh(),
    lambda x: x.sigmoid(),
    lambda x: x.exp(),
    lambda x: x.log(),
    lambda x: x.sqrt(),
    lambda x: x.softmax(axis=1),
    lambda x: x.softmax(axis=0),
    lambda x: x.T.reshape((2, 3)),
    lambda x: where(x.greater(0.7), x * x, -x),
])
def test_finite_difference_operations(operation):
    g = Graph('float64'); x = g.parameter('x', (2, 3))
    a = np.array([[0.2, 0.4, 0.8], [1.1, 1.4, 2.1]])
    y = operation(x)
    coefficients = g.constant(np.arange(1, 7).reshape(y.shape) / 7)
    loss = (y * coefficients).sum()
    dx = grad(loss, [x])[0]
    np.testing.assert_allclose(value(dx, {'x': a}), finite_difference(loss, x, {'x': a}), rtol=2e-6, atol=2e-8)


@pytest.mark.parametrize('left,right', [ ((), (2, 3)), ((3,), (2, 3)), ((2, 1), (1, 3)), ((1,), (2, 3)), ((1, 1), (2, 3)), ((2, 3), ()) ])
def test_broadcast_reduction_every_expanded_axis(left, right):
    g = Graph('float64'); x = g.parameter('x', left); y = g.parameter('y', right)
    bindings = {'x': np.full(left, 0.7), 'y': np.full(right, 1.2)}
    loss = ((x * y + x) / (y + 2)).sum()
    for parameter, derivative in zip([x, y], grad(loss, [x, y])):
        np.testing.assert_allclose(value(derivative, bindings), finite_difference(loss, parameter, bindings), rtol=2e-6, atol=2e-8)


def test_matmul_fanout_repeated_and_unused_parameter():
    rng = np.random.default_rng(3)
    g = Graph('float64'); x = g.parameter('x', (3, 5)); w = g.parameter('w', (5, 2)); unused = g.parameter('unused', ())
    shared = x @ w
    loss = (shared * shared + shared.tanh() + shared).sum()
    bindings = {'x': rng.normal(size=x.shape), 'w': rng.normal(size=w.shape)}
    derivatives = grad(loss, [x, w, unused])
    for parameter, derivative in zip([x, w], derivatives):
        np.testing.assert_allclose(value(derivative, bindings), finite_difference(loss, parameter, bindings), rtol=2e-6, atol=3e-8)
    assert value(derivatives[-1], {}) == 0
    # Evaluating again produces fresh values; no hidden .grad accumulation exists.
    np.testing.assert_array_equal(value(derivatives[0], bindings), value(derivatives[0], bindings))


@pytest.mark.parametrize('batch,classes', [(1, 1), (1, 2), (4, 3), (3, 11)])
@pytest.mark.parametrize('large', [False, True])
def test_stable_cross_entropy(batch, classes, large):
    rng = np.random.default_rng(4)
    g = Graph('float64'); x = g.parameter('x', (batch, classes)); labels = g.input('labels', (batch,), 'int32')
    a = rng.normal(size=x.shape) * (1000 if large else 1)
    y = rng.integers(classes, size=batch, dtype='int32')
    loss = cross_entropy(x, labels); dx = grad(loss, [x])[0]
    bindings = {'x': a, 'labels': y}
    shifted = a - a.max(1, keepdims=True)
    expected = np.mean(np.log(np.exp(shifted).sum(1)) - shifted[np.arange(batch), y])
    assert float(value(loss, bindings)) == pytest.approx(expected)
    np.testing.assert_allclose(value(dx, bindings), finite_difference(loss, x, bindings, eps=1e-4 if large else 1e-6), rtol=3e-6, atol=2e-8)


def test_tied_logits_and_relu_zero_convention():
    g = Graph('float64'); x = g.parameter('x', (2, 3)); labels = g.input('labels', (2,), 'int32')
    loss = cross_entropy(x, labels)
    result = value(grad(loss, [x])[0], {'x': np.full((2, 3), 10000.), 'labels': np.array([0, 2], dtype='int32')})
    np.testing.assert_allclose(result, (np.full((2, 3), 1/3) - np.eye(3)[[0, 2]]) / 2)
    dx = grad(x.relu().sum(), [x])[0]
    np.testing.assert_array_equal(value(dx, {'x': np.array([[-1., 0, 2], [0, -2, 3]])}), [[0, 0, 1], [0, 0, 1]])


def test_contiguous_layout_mse_and_stable_sigmoid():
    g = Graph(); x = g.input('x', (2, 3)); a = np.arange(6, dtype='float32').reshape((3, 2)).T
    result = value(x.T, {'x': a})
    assert result.flags.c_contiguous
    np.testing.assert_array_equal(result, a.T)
    assert value(mse(x, 0), {'x': a}) == np.mean(a*a)
    s = g.input('s', (3,))
    np.testing.assert_allclose(value(s.sigmoid(), {'s': np.array([-10000, 0, 10000], dtype='float32')}), [0, 0.5, 1])


def test_immutable_graph_and_serialization():
    g = Graph(); x = g.parameter('x', (2, 3)); y = x.mean(1)
    with pytest.raises(dataclasses.FrozenInstanceError): x.node.shape = (1,)
    with pytest.raises(TypeError): y.node.attrs['axis'] = (0,)
    with pytest.raises(TypeError): bool(x)
    with pytest.raises(TypeError): bool(x == 0)
    with pytest.raises(TypeError): bool(x != 0)
    with pytest.raises(TypeError): np.asarray(x)
    assert isinstance(g.nodes, tuple)
    serialized = json.loads(json.dumps(g.to_dict()))
    assert serialized['nodes'][y.id]['attrs']['axis'] == [1]


@pytest.mark.parametrize('shape', [(0,), (-1,), (1, 2, 3), (True,), (1.5,), (1_048_577,)])
def test_bad_shapes(shape):
    with pytest.raises(GraphError): Graph().input('x', shape)


def test_resource_limits_and_invalid_operations():
    with pytest.raises(GraphError): Graph(max_parameters=3).parameter('x', (2, 2))
    g = Graph(max_nodes=1); x = g.input('x', ())
    with pytest.raises(GraphError): x + 1
    g = Graph(); x = g.input('x', (2, 3)); y = g.input('y', (2,))
    for operation in [lambda: x + y, lambda: x @ x, lambda: x.reshape((5,)), lambda: x.sum((1, 1)), lambda: x.sum(2), lambda: y.T, lambda: x.broadcast_to((3, 3)), lambda: x + Graph().constant(1), lambda: x + g.constant(1, 'float64')]:
        with pytest.raises(GraphError): operation()
    with pytest.raises(GraphError): g.input('x', ())
    with pytest.raises(GraphError): g.constant(np.inf)
    with pytest.raises(GraphError): g.constant(2**40, 'int32')
    with pytest.raises(GraphError): g.constant(0.5, 'int32')
    with pytest.raises(GraphError): g.input('bad', (), 'float16')


def test_binding_and_domain_errors():
    g = Graph(); x = g.input('x', (2,))
    with pytest.raises(GraphError): value(x, {})
    with pytest.raises(GraphError): value(x, {'x': np.ones(2, dtype='float64')})
    with pytest.raises(GraphError): value(x, {'x': np.ones(3, dtype='float32')})
    with pytest.raises(GraphError): value(x, {'x': np.array([np.nan, 1], dtype='float32')})
    for expression, data in [(x.log(), [-1, 1]), (x.sqrt(), [-1, 0]), (1/x, [0, 1]), (x.exp(), [10000, 1])]:
        with pytest.raises(GraphError): value(expression, {'x': np.array(data, dtype='float32')})
    with pytest.raises(GraphError): grad(x, [x])
    derivative = grad((x*x).sum(), [x])[0]
    with pytest.raises(GraphError): grad(derivative.sum(), [x])


def test_invalid_labels():
    g = Graph(); x = g.input('x', (2, 3)); labels = g.input('labels', (2,), 'int32'); loss = cross_entropy(x, labels)
    for data in [np.array([-1, 0], dtype='int32'), np.array([0, 3], dtype='int32'), np.array([0, 1], dtype='int64')]:
        with pytest.raises(GraphError): value(loss, {'x': np.ones((2, 3), dtype='float32'), 'labels': data})
    with pytest.raises(GraphError): cross_entropy(x, g.input('wrong', (2,), 'float32'))
    with pytest.raises(GraphError): cross_entropy(x, g.input('wrong-shape', (2, 1), 'int32'))
    with pytest.raises(GraphError): labels + 1


def test_scalar_reductions_vector_softmax_and_bounded_execution():
    g = Graph('float64'); scalar = g.parameter('scalar', ())
    output = scalar.mean().sum((), True)
    np.testing.assert_array_equal(value(grad(output, [scalar])[0], {}), np.array(1.))
    x = g.parameter('x', (3,)); p = x.softmax(); loss = (p * g.constant([1., 2., 4.])).sum()
    bindings = {'x': np.array([.2, .5, -.8])}
    np.testing.assert_allclose(value(grad(loss, [x])[0], bindings), finite_difference(loss, x, bindings), atol=2e-8)
    tiny = Graph(max_bytes=8); a = tiny.input('a', (2,)); b = -a
    with pytest.raises(GraphError, match='tensor bytes'): value(b, {'a': np.ones(2, dtype='float32')})


def test_raw_graph_node_validation():
    g = Graph(); x = g.input('x', (2,))
    for create in [lambda: g.add('unknown', (x,), (2,)), lambda: g.add('add', (x,), (2,)), lambda: g.add('neg', (x,), (3,)), lambda: g.add('neg', (x,), (2,), attrs={'unexpected':1})]:
        with pytest.raises(GraphError): create()
    array = np.array([1., 2.])
    c = g.add('constant', shape=(2,), attrs={'value':array}, kind='constant')
    array[0] = 100
    assert c.node.attrs['value'] == (1., 2.)
