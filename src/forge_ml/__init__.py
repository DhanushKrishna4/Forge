"""Forge: inspectable static tensor graphs, differentiation and compilation."""
from .graph import Graph, Node, GraphError
from .tensor import Tensor, cross_entropy, mse, one_hot, where
from .autodiff import grad
from .interpreter import evaluate
from .optim import optimizer_updates, initial_bindings, apply_updates

__version__ = '0.1.0'
__all__ = ['Graph', 'Node', 'GraphError', 'Tensor', 'cross_entropy', 'mse', 'one_hot', 'where', 'grad', 'evaluate', 'optimizer_updates', 'initial_bindings', 'apply_updates']
