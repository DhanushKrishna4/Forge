import numpy as np
import pytest
from forge_ml import Graph, GraphError, grad, optimizer_updates, initial_bindings, apply_updates


@pytest.mark.parametrize('kind', ['sgd', 'momentum', 'adam'])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
def test_optimizer_many_steps_against_explicit_formula(kind, dtype):
    g = Graph(dtype); p = g.parameter('p', (2, 3)); target = g.input('target', (2, 3)); loss = ((p-target)*(p-target)).mean()
    gradients = grad(loss, [p]); updates = optimizer_updates([p], gradients, kind, lr=.035)
    initial = np.arange(6, dtype=dtype).reshape(2, 3) / 4
    bindings = initial_bindings(g, {'p': initial}); bindings['target'] = np.full((2, 3), .7, dtype=dtype)
    expected = initial.copy(); m = np.zeros_like(initial); v = np.zeros_like(initial)
    for step in range(1, 9):
        derivative = 2 * (expected - bindings['target']) / expected.size
        if kind == 'sgd': delta = derivative
        elif kind == 'momentum':
            m = .9*m + derivative; delta = m
        else:
            m = .9*m + .1*derivative; v = .999*v + .001*derivative*derivative
            delta = (m/(1-.9**step)) / (np.sqrt(v/(1-.999**step)) + 1e-8)
        expected = expected - .035*delta
        apply_updates(updates, bindings, [loss])
        np.testing.assert_allclose(bindings['p'], expected, rtol=2e-5 if dtype == 'float32' else 1e-12, atol=2e-6 if dtype == 'float32' else 1e-13)
        if kind == 'adam':
            assert bindings['state/step'] == step
            np.testing.assert_allclose(bindings['state/m/p'], m, rtol=2e-5, atol=1e-7)
            np.testing.assert_allclose(bindings['state/v/p'], v, rtol=2e-5, atol=1e-8)
    assert np.linalg.norm(bindings['p']-bindings['target']) < np.linalg.norm(initial-bindings['target'])


def test_update_commit_is_atomic_and_resume_includes_step():
    g = Graph('float64'); p = g.parameter('p', (2,)); bad = g.input('bad', ()); loss = (p*p).sum() + bad.log()
    updates = optimizer_updates([p], grad(loss, [p]), 'adam'); bindings = initial_bindings(g, {'p': np.array([1., 2.])}); bindings['bad'] = np.array(1.)
    for _ in range(4): apply_updates(updates, bindings, [loss])
    checkpoint = {name: a.copy() for name, a in bindings.items()}
    apply_updates(updates, bindings, [loss]); expected = {name: a.copy() for name, a in bindings.items()}
    resumed = {name: a.copy() for name, a in checkpoint.items()}; apply_updates(updates, resumed, [loss])
    for name in expected: np.testing.assert_array_equal(expected[name], resumed[name])
    resumed['bad'] = np.array(-1.)
    before = {name: a.copy() for name, a in resumed.items()}
    with pytest.raises(GraphError): apply_updates(updates, resumed, [loss])
    for name in before: np.testing.assert_array_equal(before[name], resumed[name])


def test_invalid_optimizer_configuration():
    for settings in [{'kind':'rmsprop'}, {'lr':-1}, {'eps':0}, {'beta1':1}, {'beta2':-1}, {'momentum':float('nan')}]:
        g = Graph(); p = g.parameter('p', ()); d = grad(p*p, [p])
        with pytest.raises(GraphError): optimizer_updates([p], d, **settings)


def test_adam_zero_betas_and_zero_gradient_are_finite():
    g = Graph('float64'); p = g.parameter('p', (2,)); loss = (p*p).sum()
    updates = optimizer_updates([p], grad(loss, [p]), beta1=0, beta2=0)
    bindings = initial_bindings(g, {'p': np.array([0., 1.])})
    apply_updates(updates, bindings)
    assert bindings['p'][0] == 0
    assert np.isfinite(bindings['p']).all()
