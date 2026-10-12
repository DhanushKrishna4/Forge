"""Out-of-place optimizer updates are graph outputs, including all state."""
import math
import numpy as np
from .graph import GraphError
from .interpreter import evaluate, validate_binding


def optimizer_updates(parameters, gradients, kind='adam', lr=0.01, momentum=0.9, beta1=0.9, beta2=0.999, eps=1e-8):
    parameters, gradients = list(parameters), list(gradients)
    if not parameters or len(parameters) != len(gradients):
        raise GraphError('Optimizer requires matching nonempty parameter and gradient lists')
    graph = parameters[0].graph
    if len({p.id for p in parameters}) != len(parameters):
        raise GraphError('Optimizer parameters must not repeat')
    if kind not in ('sgd', 'momentum', 'adam'):
        raise GraphError('Optimizer must be sgd, momentum, or adam')
    if not math.isfinite(lr) or lr <= 0 or not math.isfinite(eps) or eps <= 0:
        raise GraphError('Learning rate and epsilon must be finite and positive')
    if not 0 <= momentum < 1 or not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
        raise GraphError('Momentum and Adam betas must be in [0,1)')
    for p, g in zip(parameters, gradients):
        if p.graph is not graph or g.graph is not graph or p.node.kind != 'parameter' or p.shape != g.shape or p.dtype != g.dtype:
            raise GraphError('Optimizer targets must be named parameters with matching gradients')
    updates = {}
    if kind == 'adam':
        step = graph.state('state/step', ())
        next_step = step + 1
        updates[step.node.name] = next_step
        # exp(t*log(beta)) expresses scalar bias correction using portable ops.
        correction1 = 1 - (next_step * math.log(beta1)).exp() if beta1 else graph.constant(1)
        correction2 = 1 - (next_step * math.log(beta2)).exp() if beta2 else graph.constant(1)
    for p, g in zip(parameters, gradients):
        name = p.node.name
        if kind == 'sgd':
            delta = g
        elif kind == 'momentum':
            velocity = graph.state(f'state/momentum/{name}', p.shape)
            delta = momentum * velocity + g
            updates[velocity.node.name] = delta
        else:
            m = graph.state(f'state/m/{name}', p.shape)
            v = graph.state(f'state/v/{name}', p.shape)
            next_m = beta1 * m + (1 - beta1) * g
            next_v = beta2 * v + (1 - beta2) * g * g
            updates[m.node.name], updates[v.node.name] = next_m, next_v
            # Epsilon is outside sqrt, matching conventional Adam.
            delta = (next_m / correction1) / ((next_v / correction2).sqrt() + eps)
        updates[name] = p - lr * delta
    return updates


def initial_bindings(graph, parameters):
    """Copy validated parameter values and initialize every optimizer state to 0."""
    result = {}
    for node in graph.nodes:
        if node.kind == 'parameter':
            if node.name not in parameters:
                raise GraphError(f'Missing initial parameter {node.name}')
            result[node.name] = validate_binding(node, parameters[node.name])
        elif node.kind == 'state':
            result[node.name] = np.zeros(node.shape, dtype=node.dtype)
    return result


def apply_updates(updates, bindings, outputs=()):
    """Execute and atomically commit update outputs; return requested read values."""
    requested = list(outputs)
    values = evaluate(requested + list(updates.values()), bindings)
    # Evaluation must finish successfully before replacing any mutable binding.
    committed = {name: values[t.id] for name, t in updates.items()}
    bindings.update(committed)
    return {t.id: values[t.id] for t in requested}
