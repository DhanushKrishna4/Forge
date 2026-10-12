#!/usr/bin/env python3
"""Generate GPU conformance plans from Forge and explicit NumPy ground truth.

The browser consumes the generated plans only; it never mirrors tensor semantics.
Run `python scripts/generate_gpu_fixtures.py` from the repository root.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from forge_ml import Graph, Tensor, cross_entropy, evaluate, grad, initial_bindings, optimizer_updates, where
from forge_ml.compiler import compile_graph

DISABLED=dict(dce=False,fold=False,cse=False,fusion=False,reuse=False,tiled=False)


def flatten(values):
    return {name:np.asarray(value).reshape(-1).tolist() for name,value in values.items()}


def fixture(name,g,outputs,bindings,updates=None,steps=1,options=None):
    updates=updates or {}
    current={k:v.copy() for k,v in bindings.items()};expected=[]
    for _ in range(steps):
        values=evaluate(list(outputs.values())+list(updates.values()),current)
        named={key:values[t.id] for key,t in {**outputs,**updates}.items()}
        current.update({key:values[t.id].copy() for key,t in updates.items()})
        persistent={n.name:current[n.name] for n in g.nodes if n.kind in ('parameter','state')}
        expected.append({'values':flatten(named),'bindings':flatten(persistent)})
    return dict(name=name,plan=compile_graph(g,outputs,options=options,updates=updates),bindings=flatten(bindings),expected=expected,rtol=3e-5,atol=3e-6,tolerance_notes='Float32 reductions, matmul accumulation and transcendental implementations may round differently; absolute tolerance protects near-zero values.')


def build_fixtures():
    result=[];rng=np.random.default_rng(83)
    for kind in ('sgd','momentum','adam'):
        for batch in (1,5):
            g=Graph();x=g.input('x',(batch,3));labels=g.input('labels',(batch,),dtype='int32')
            w=g.parameter('w',(3,7));b=g.parameter('b',(7,));v=g.parameter('v',(7,3))
            hidden=(x@w+b).tanh();logits=hidden@v;loss=cross_entropy(logits,labels)
            parameters=[w,b,v];grads=grad(loss,parameters);updates=optimizer_updates(parameters,grads,kind=kind,lr=.01)
            outputs={'loss':loss,'logits':logits,**{f'grad:{p.node.name}':d for p,d in zip(parameters,grads)}}
            bindings=initial_bindings(g,{p.node.name:rng.normal(0,.25,p.shape).astype('float32') for p in parameters})
            bindings.update(x=rng.normal(size=x.shape).astype('float32'),labels=(np.arange(batch)%3).astype('int32'))
            for optimized in (False,True):
                case=fixture(f'{kind}-batch{batch}-'+('optimized' if optimized else 'baseline'),g,outputs,bindings,updates,steps=4,options={} if optimized else DISABLED)
                if kind=='adam':
                    case.update(rtol=2e-4,atol=2e-5,tolerance_notes='Four float32 Adam steps: exp-based bias correction subtracts a number near 1 (beta2=0.999), amplifying a transcendental ULP. Tolerances cover accumulated update/state error; recorded absolute and relative errors must still be reported.')
                result.append(case)
    g=Graph();x=g.input('x',(3,5));a=g.parameter('a',(1,5));b=g.parameter('b',(3,1));positive=(x+a).exp()+.5
    terms=(positive.log()/positive.sqrt()).sigmoid()+(-x).tanh()+(x-b).relu()
    selected=where(x.greater(0),terms,terms*.5)
    outputs={'selected':selected,'all_sum':selected.sum(),'all_mean':selected.mean(),'row_sum':selected.sum(axis=1),
             'col_sum_keep':selected.sum(axis=0,keepdims=True),'row_mean_keep':selected.mean(axis=1,keepdims=True),
             'col_mean':selected.mean(axis=0),'reshape':selected.reshape((5,3)), 'transpose':selected.T,
             'softmax_rows':selected.softmax(axis=1),'softmax_cols':selected.softmax(axis=0),
             'broadcast':a.broadcast_to((3,5)), 'identity_reduction':selected.sum(axis=())}
    da,db=grad(selected.mean()+outputs['softmax_cols'].sum(),[a,b]);outputs.update(gradient_a=da,gradient_b=db)
    bindings={'x':rng.normal(size=x.shape).astype('float32'),'a':rng.normal(0,.2,a.shape).astype('float32'),'b':rng.normal(0,.2,b.shape).astype('float32')}
    for optimized in (False,True): result.append(fixture('all-ops-'+('optimized' if optimized else 'baseline'),g,outputs,bindings,options={} if optimized else DISABLED))
    for count in (1,7,65):
        g=Graph();x=g.input('x',(count,));out=(x*2+1).tanh();loss=out.sum();result.append(fixture(f'reduction-boundary-{count}',g,{'out':out,'sum':loss,'mean':out.mean()}, {'x':np.linspace(-1,1,count,dtype='float32')}))
    g=Graph();x=g.input('x',(3,5));labels=g.input('labels',(3,),dtype='int32');loss=cross_entropy(x,labels);dx=grad(loss,[x])[0]
    result.append(fixture('stable-large-tied-logits',g,{'loss':loss,'gradient':dx,'probabilities':x.softmax()}, {'x':np.array([[10000,10001,9999,-10000,5000],[9999,9999,9999,9999,9999],[-10000,-9999,-9998,-10000,-9990]],np.float32),'labels':np.array([1,4,0],np.int32)}))
    # A common logit translation must not change CE. Large offsets catch native
    # shader compilers reassociating (max - target) with the small loss sum.
    base_logits=np.array([[0,1,-1,2,-2],[3,3,3,3,3],[-10,-9,-8,-10,0]],np.float32)
    for offset in (0.0, 10000.0, -10000.0, 1000000.0):
        g=Graph();x=g.input('x',(3,5));labels=g.input('labels',(3,),dtype='int32');loss=cross_entropy(x,labels);dx=grad(loss,[x])[0]
        case=fixture(f'ce-translation-{offset:g}',g,{'loss':loss,'gradient':dx},{'x':base_logits+np.float32(offset),'labels':np.array([1,4,0],np.int32)})
        case.update(rtol=3e-6,atol=5e-7,tolerance_notes='Exactly representable integer logits share a common shift. Stable CE must agree with the unshifted float64 expression; offset cancellation must not consume accuracy.')
        result.append(case)
    g=Graph();x=g.input('x',(7,));activation=x.sigmoid();dx=grad(activation.sum(),[x])[0]
    result.append(fixture('stable-extreme-sigmoid',g,{'activation':activation,'gradient':dx},{'x':np.array([-1000,-100,-10,0,10,100,1000],np.float32)}))
    g=Graph();labels=g.input('labels',(1,3),dtype='int32');x=g.input('x',(3,2));flat=labels.T.reshape((3,));loss=cross_entropy(x,flat)
    result.append(fixture('integer-label-layouts',g,{'loss':loss,'flat_labels':flat},{'labels':np.array([[0,1,0]],np.int32),'x':np.arange(6,dtype='float32').reshape(3,2)}))
    # Direct-binding updates exercise simultaneous copies and precommit read semantics.
    g=Graph();a=g.parameter('a',(3,));b=g.parameter('b',(3,))
    result.append(fixture('binding-swap',g,{'old_a':a,'old_b':b},{'a':np.array([1,2,3],np.float32),'b':np.array([4,5,6],np.float32)}, {'a':b,'b':a},steps=3))
    result[-1].update(rtol=0.0,atol=0.0,tolerance_notes='Binding swap consists only of exact float32 copies; no numerical approximation is permitted.')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='web/public/validation/gpu-fixtures.json');args=parser.parse_args()
    fixtures=build_fixtures();path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({'version':1,'generator':'scripts/generate_gpu_fixtures.py','reference':'unoptimized Forge NumPy interpreter','fixtures':fixtures},separators=(',',':'),allow_nan=False)+'\n')
    print(f'{len(fixtures)} fixtures -> {path} ({path.stat().st_size:,} bytes)')


if __name__=='__main__':main()
