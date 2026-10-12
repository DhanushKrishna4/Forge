"""Independent development-only PyTorch reference; Forge never imports torch.

Run explicitly: python -m pytest tests/test_reference.py -v -s
The reference inputs are identical arrays, not independently seeded initializers.
"""
import numpy as np
import pytest
from forge_ml import Graph, cross_entropy, evaluate, grad, optimizer_updates, initial_bindings, apply_updates

torch = pytest.importorskip('torch', reason='Install the dev-reference extra to run independent reference checks')


def assert_reference(actual, expected, dtype):
    expected = expected.detach().numpy()
    difference = np.abs(actual - expected)
    absolute = float(np.max(difference))
    relative = float(np.max(difference / np.maximum(np.abs(expected), 1e-12)))
    tolerance = 3e-5 if dtype == 'float32' else 3e-11
    atol = 3e-7 if dtype == 'float32' else 3e-12
    assert np.allclose(actual, expected, rtol=tolerance, atol=atol), f'max_abs={absolute:.8g}; max_relative={relative:.8g}; dtype={dtype}'
    return absolute, relative


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('shape', [(), (3,), (3, 5)])
def test_torch_composite_all_pointwise_operations(dtype, shape):
    rng = np.random.default_rng(29)
    array = rng.uniform(.2, 1.7, size=shape).astype(dtype)
    g = Graph(dtype); x = g.parameter('x', shape)
    def composite(v):
        shared = v * v + 0.4
        return ((shared.log().exp().sqrt() / (v+2)) - (-v).tanh() + v.sigmoid() + (v-.3).relu()).mean()
    loss = composite(x); derivative = grad(loss, [x])[0]
    values = evaluate([loss, derivative], {'x':array})
    tx = torch.tensor(array, requires_grad=True); expected = composite(tx); expected.backward()
    assert_reference(values[loss.id], expected, dtype)
    assert_reference(values[derivative.id], tx.grad, dtype)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('axis,keepdims', [(0, False), (1, True), ((0, 1), False), ((0, 1), True)])
def test_torch_reduction_broadcast_layout_and_shared_parameter(dtype, axis, keepdims):
    rng = np.random.default_rng(67)
    x = rng.normal(size=(3, 5)).astype(dtype); b = rng.normal(size=(5,)).astype(dtype); c = np.array(.7, dtype=dtype)
    g = Graph(dtype); px = g.parameter('x', x.shape); pb = g.parameter('b', b.shape); pc = g.parameter('c', ())
    shared = px * pb + pc
    transformed = shared.T.reshape((3, 5))
    reduced = transformed.mean(axis, keepdims)
    loss = (reduced * reduced).sum() + (shared*px).mean()
    derivatives = grad(loss, [px, pb, pc]); values = evaluate([loss, *derivatives], {'x':x, 'b':b, 'c':c})
    tx, tb, tc = [torch.tensor(a, requires_grad=True) for a in (x, b, c)]
    ts = tx*tb+tc; tr = ts.T.reshape(3,5).mean(dim=axis, keepdim=keepdims); tloss = (tr*tr).sum() + (ts*tx).mean(); tloss.backward()
    assert_reference(values[loss.id], tloss, dtype)
    for derivative, tensor in zip(derivatives, [tx, tb, tc]): assert_reference(values[derivative.id], tensor.grad, dtype)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('batch,classes,magnitude', [(1, 2, 1), (7, 3, 1), (4, 11, 1000), (3, 5, 0)])
def test_torch_stable_cross_entropy(dtype, batch, classes, magnitude):
    rng = np.random.default_rng(17)
    array = (rng.normal(size=(batch, classes))*magnitude).astype(dtype)
    labels = rng.integers(classes, size=batch, dtype='int32')
    g = Graph(dtype); x = g.parameter('x', array.shape); y = g.input('y', (batch,), 'int32')
    loss = cross_entropy(x, y); derivative = grad(loss, [x])[0]; values = evaluate([loss, derivative], {'x':array, 'y':labels})
    tx = torch.tensor(array, requires_grad=True); expected = torch.nn.functional.cross_entropy(tx, torch.tensor(labels.astype('int64'))); expected.backward()
    assert_reference(values[loss.id], expected, dtype); assert_reference(values[derivative.id], tx.grad, dtype)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('axis', [0, 1])
def test_torch_softmax_jacobian_vector_product(dtype, axis):
    rng = np.random.default_rng(37); a = rng.normal(size=(3, 5)).astype(dtype); weights = rng.normal(size=(3, 5)).astype(dtype)
    g = Graph(dtype); x = g.parameter('x', a.shape); p = x.softmax(axis); loss = (p * g.constant(weights)).sum(); dx = grad(loss, [x])[0]
    values = evaluate([p, dx], {'x': a})
    tx = torch.tensor(a, requires_grad=True); expected = tx.softmax(axis); (expected * torch.tensor(weights)).sum().backward()
    assert_reference(values[p.id], expected, dtype); assert_reference(values[dx.id], tx.grad, dtype)


@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('kind', ['sgd', 'momentum', 'adam'])
def test_torch_complete_mlp_training_steps(dtype, kind):
    rng = np.random.default_rng(83)
    x = rng.normal(size=(7, 3)).astype(dtype); y = rng.integers(4, size=7, dtype='int32')
    initial = {'w1': rng.normal(0,.3,(3,5)).astype(dtype), 'b1': rng.normal(0,.1,(5,)).astype(dtype), 'w2': rng.normal(0,.3,(5,4)).astype(dtype), 'b2': rng.normal(0,.1,(4,)).astype(dtype)}
    g = Graph(dtype); px = g.input('x', x.shape); py = g.input('y', y.shape, 'int32')
    parameters = [g.parameter(name, a.shape) for name,a in initial.items()]
    w1,b1,w2,b2 = parameters; logits = (px@w1+b1).tanh()@w2+b2; loss = cross_entropy(logits,py)
    derivatives = grad(loss,parameters); updates = optimizer_updates(parameters, derivatives, kind, lr=.013)
    bindings = initial_bindings(g,initial); bindings.update(x=x,y=y)
    tp = [torch.tensor(a, requires_grad=True) for a in initial.values()]
    optimizer = torch.optim.Adam(tp,lr=.013) if kind == 'adam' else torch.optim.SGD(tp,lr=.013,momentum=.9 if kind == 'momentum' else 0)
    max_abs, max_rel = 0., 0.
    for _ in range(5):
        optimizer.zero_grad(); tw1,tb1,tw2,tb2 = tp
        expected_logits = (torch.tensor(x)@tw1+tb1).tanh()@tw2+tb2
        expected_loss = torch.nn.functional.cross_entropy(expected_logits,torch.tensor(y.astype('int64'))); expected_loss.backward()
        values = apply_updates(updates,bindings,[loss,logits,*derivatives])
        for actual,expected in [(values[loss.id],expected_loss),(values[logits.id],expected_logits),*[(values[d.id],p.grad) for d,p in zip(derivatives,tp)]]:
            a,r = assert_reference(actual,expected,dtype); max_abs,max_rel=max(max_abs,a),max(max_rel,r)
        optimizer.step()
        for name,p in zip(initial,tp):
            a,r=assert_reference(bindings[name],p,dtype); max_abs,max_rel=max(max_abs,a),max(max_rel,r)
            if kind == 'adam':
                state=optimizer.state[p]
                assert_reference(bindings[f'state/m/{name}'],state['exp_avg'],dtype)
                assert_reference(bindings[f'state/v/{name}'],state['exp_avg_sq'],dtype)
                assert bindings['state/step'] == state['step'].item()
            elif kind == 'momentum': assert_reference(bindings[f'state/momentum/{name}'],optimizer.state[p]['momentum_buffer'],dtype)
    print(f'{kind} {dtype} 5-step MLP reference: max_abs={max_abs:.8g}, max_relative={max_rel:.8g} (near-zero references use 1e-12 denominator floor)')
