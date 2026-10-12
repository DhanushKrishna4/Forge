"""Inspect an explicit scalar loss, its gradient graph, and a GPU plan."""
import numpy as np
from forge_ml import Graph,evaluate,grad
from forge_ml.compiler import compile_graph

g=Graph('float64')
x=g.parameter('x',(2,3));b=g.parameter('bias',(3,))
h=(x+b).tanh()
loss=(h*h).mean()
dx,db=grad(loss,[x,b])
bindings={'x':np.array([[.1,.2,.3],[.4,.5,.6]]),'bias':np.zeros(3)}
for ident,array in evaluate([loss,dx,db],bindings).items():print(ident,array)
plan=compile_graph(g,{'loss':loss,'dx':dx,'db':db})
print(plan['stats'])
print('float64 is native validation only; portable GPU plans require float32')
