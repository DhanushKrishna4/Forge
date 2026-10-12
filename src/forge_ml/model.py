"""Model construction and bounded experiments shared by native Python and Pyodide."""
from __future__ import annotations
from collections import OrderedDict
from copy import deepcopy
import math
import time
import numpy as np
from .graph import Graph
from .tensor import Tensor, cross_entropy
from .autodiff import grad
from .interpreter import evaluate
from .optim import optimizer_updates, initial_bindings
from .datasets import dataset_2d, load_digits_asset

DEFAULT_CONFIG = dict(dataset='moons',seed=42,samples=240,noise=.12,hidden=[16,16],activation='tanh',
                      optimizer='adam',learningRate=.01,batchSize=32,backend='cpu',
                      compiler=dict(dce=True,fold=True,cse=True,fusion=True,reuse=True,tiled=True))
MAX_STEPS=10000
MAX_CACHE=8

def validate_config(value):
    if not isinstance(value,dict) or set(value)-set(DEFAULT_CONFIG)-{'customPoints'}:
        raise ValueError('Unknown experiment configuration fields')
    c=deepcopy(DEFAULT_CONFIG); c.update(value)
    for key,lo,hi in [('seed',0,2**31-1),('samples',6 if c['dataset']=='custom' else 20,2048),('batchSize',1,128)]:
        if type(c[key]) is not int or not lo<=c[key]<=hi: raise ValueError(f'{key} must be integer in [{lo},{hi}]')
    for key,lo,hi in [('noise',0,.8),('learningRate',1e-6,1.)]:
        if type(c[key]) not in (float,int) or not math.isfinite(c[key]) or not lo<=c[key]<=hi: raise ValueError(f'Invalid {key}')
    if not isinstance(c['hidden'],list) or not 1<=len(c['hidden'])<=3 or any(type(n)is not int or not 1<=n<=64 for n in c['hidden']):
        raise ValueError('Use 1–3 hidden layers with 1–64 units')
    for key, choices in dict(dataset=['xor','moons','spiral','digits','custom'],activation=['relu','tanh','sigmoid'],optimizer=['sgd','momentum','adam'],backend=['cpu','webgpu']).items():
        if c[key] not in choices: raise ValueError(f'Unsupported {key}')
    if not isinstance(c['compiler'],dict) or set(c['compiler'])-set(DEFAULT_CONFIG['compiler']) or any(type(v)is not bool for v in c['compiler'].values()):
        raise ValueError('Compiler options must be known boolean fields')
    c['compiler']={**DEFAULT_CONFIG['compiler'],**c['compiler']}
    dims=[64 if c['dataset']=='digits' else 2,*c['hidden'],10 if c['dataset']=='digits' else 2]
    if sum((a+1)*b for a,b in zip(dims,dims[1:]))>100000: raise ValueError('Model exceeds 100000 parameters')
    return c


def build_mlp(config, batch, training=True, dtype='float32'):
    dims=[64 if config['dataset']=='digits' else 2,*config['hidden'],10 if config['dataset']=='digits' else 2]
    g=Graph(dtype=dtype)
    x=g.input('x',(batch,dims[0])); y=g.input('labels',(batch,),dtype='int32')
    h=x; params=[]
    for i,(a,b) in enumerate(zip(dims,dims[1:])):
        w=g.parameter(f'layer{i}.weight',(a,b)); bias=g.parameter(f'layer{i}.bias',(b,))
        params.extend([w,bias]); h=h@w+bias
        if i<len(dims)-2: h=getattr(h,config['activation'])()
    logits=h; loss=cross_entropy(logits,y); probs=logits.softmax()
    forward_end=len(g.nodes)
    outputs={'loss':loss,'probabilities':probs}
    grads=[]; updates={}
    if training:
        grads=grad(loss,params)
        norm=sum((v*v).sum() for v in grads).sqrt()
        outputs.update({'gradient_norm':norm,**{'grad:'+p.node.name:v for p,v in zip(params,grads)}})
        updates=optimizer_updates(params,grads,kind=config['optimizer'],lr=config['learningRate'])
        outputs.update({'gradient_norm:'+p.node.name:(v*v).sum().sqrt() for p,v in zip(params,grads)})
        outputs.update({'parameter_norm:'+p.node.name:(updates[p.node.name]*updates[p.node.name]).sum().sqrt() for p in params})
    return dict(graph=g,params=params,grads=grads,outputs=outputs,updates=updates,forward_end=forward_end)


class Experiment:
    def __init__(self, config=None, digits=None):
        self.config=validate_config(config or {})
        c=self.config
        self.data=load_digits_asset(digits) if c['dataset']=='digits' else dataset_2d(c['dataset'],c['samples'],c['noise'],c['seed'],c.get('customPoints'))
        self.rng=np.random.default_rng(c['seed']+1)
        self.cache=OrderedDict(); self.plans=OrderedDict()
        self.batch=min(c['batchSize'],len(self.data['splits']['train']))
        self.training=self.program(self.batch,True)
        rng=np.random.default_rng(c['seed'])
        params={}
        for p in self.training['params']:
            if len(p.shape)==2:
                # Xavier for saturating activations, He for ReLU; explicit arrays are shared with references.
                scale=np.sqrt((2 if c['activation']=='relu' else 1)/p.shape[0])
                params[p.node.name]=(rng.normal(0,scale,p.shape)).astype(np.float32)
            else: params[p.node.name]=np.zeros(p.shape,np.float32)
        self.bindings=initial_bindings(self.training['graph'],params)
        self.initial={k:v.copy() for k,v in self.bindings.items()}
        # No backward execution has occurred yet. Unknown norms must not look like measured zeros.
        self.step=0; self.elapsed=0.; self.last_ms=0.; self.gradient_norm=None; self.history=[]
        self.last_batch=None; self.last_grads={}; self.parameter_summaries={}; self.backend_metadata={}

    def program(self,batch,training=False):
        key=(batch,training)
        if key not in self.cache:
            self.cache[key]=build_mlp(self.config,batch,training)
            while len(self.cache)>MAX_CACHE: self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return self.cache[key]

    def plan(self,program=None,options=None):
        from .compiler import compile_graph
        p=program or self.training
        opts=options or self.config['compiler']
        key=(tuple(p['graph'].nodes[0].shape),bool(p['grads']),tuple(sorted(opts.items())))
        if key not in self.plans:
            self.plans[key]=compile_graph(p['graph'],p['outputs'],opts,p['updates'])
            while len(self.plans)>MAX_CACHE:self.plans.popitem(last=False)
        self.plans.move_to_end(key)
        return self.plans[key]

    def next_batch(self):
        if self.step>=MAX_STEPS:raise ValueError(f'Training is bounded to {MAX_STEPS} steps; reset to start a new experiment')
        ids=self.rng.choice(self.data['splits']['train'],self.batch,replace=False)
        self.last_batch={'x':self.data['x'][ids].copy(),'labels':self.data['y'][ids].copy()}
        return self.last_batch

    def cpu_step(self,steps=1,compiled=False):
        if type(steps)is not int or not 1<=steps<=100:raise ValueError('Execute 1–100 bounded steps')
        p=self.training
        for _ in range(steps):
            rng_before=deepcopy(self.rng.bit_generator.state); batch_before=self.last_batch
            try:
                start=time.perf_counter(); batch=self.next_batch(); bindings={**self.bindings,**batch}
                if compiled:
                    from .compiler import execute_plan
                    named=execute_plan(self.plan(),bindings)
                else:
                    values=evaluate(list(p['outputs'].values())+list(p['updates'].values()),bindings)
                    named={k:values[v.id] for k,v in {**p['outputs'],**p['updates']}.items()}
                updates={k:np.array(named[k],copy=True).reshape(v.shape) for k,v in p['updates'].items()}
                if any(not np.isfinite(a).all() for a in updates.values()):raise ValueError('Nonfinite update; lower learning rate')
            except Exception:
                self.rng.bit_generator.state=rng_before; self.last_batch=batch_before
                raise
            # Commit together after every old parameter/state consumer completed.
            self.bindings.update(updates)
            self.gradient_norm=float(named['gradient_norm']); self.last_grads={p.node.name:np.array(named['grad:'+p.node.name],copy=True) for p in p['params']}
            self.parameter_summaries={p.node.name:dict(gradientNorm=float(named['gradient_norm:'+p.node.name]),valueNorm=float(named['parameter_norm:'+p.node.name])) for p in p['params']}
            self.step+=1; self.last_ms=(time.perf_counter()-start)*1000; self.elapsed+=self.last_ms
        return self.step

    def evaluate(self,x,y=None):
        p=self.program(len(x),False)
        b={**self.bindings,'x':np.asarray(x,dtype=np.float32),'labels':np.zeros(len(x),np.int32) if y is None else np.asarray(y,dtype=np.int32)}
        targets=[p['outputs']['probabilities']] if y is None else list(p['outputs'].values())
        v=evaluate(targets,b)
        return {k:v[t.id] for k,t in p['outputs'].items() if t.id in v}

    def evaluation_inputs(self):
        # All 2D display points share one evaluation; digits test is separate and never used for training selection.
        x=self.data['x']; y=self.data['y']
        if self.config['dataset']=='digits':return {'x':x,'labels':y},None
        n=40; domain=(-2.5,2.5,-2.5,2.5)
        xx,yy=np.meshgrid(np.linspace(domain[0],domain[1],n),np.linspace(domain[2],domain[3],n))
        grid=np.column_stack((xx.ravel(),yy.ravel())).astype(np.float32)
        return {'x':np.concatenate([x,grid]),'labels':np.concatenate([y,np.zeros(len(grid),np.int32)])},dict(width=n,height=n,xmin=domain[0],xmax=domain[1],ymin=domain[2],ymax=domain[3])

    def snapshot(self, probabilities=None, losses=None):
        inputs,boundary=self.evaluation_inputs()
        probs=self.evaluate(inputs['x'])['probabilities'] if probabilities is None else np.asarray(probabilities,dtype=np.float32).reshape(len(inputs['x']),-1)
        n=len(self.data['y']); actual=probs[:n]; y=self.data['y']; splits=self.data['splits']
        def score(split):
            ids=splits[split]; q=actual[ids]; labels=y[ids]
            # Display CE from graph evaluation, not naive log(probabilities), when available.
            if losses and split in losses:loss=float(losses[split])
            else:loss=float(self.evaluate(self.data['x'][ids],labels)['loss'])
            return loss,float((q.argmax(1)==labels).mean())
        train_loss,train_acc=score('train'); val_loss,val_acc=score('validation')
        if not self.history or self.history[-1]['step']!=self.step:
            self.history.append(dict(step=self.step,trainLoss=train_loss,validationLoss=val_loss));self.history=self.history[-400:]
        metrics=dict(step=self.step,trainLoss=train_loss,validationLoss=val_loss,trainAccuracy=train_acc,validationAccuracy=val_acc,gradientNorm=self.gradient_norm,elapsedMs=self.elapsed,stepMs=self.last_ms)
        p=self.training; graph=p['graph'].to_dict()
        if not isinstance(graph,dict):graph={'nodes':graph}
        graph.update(forwardIds=list(range(p['forward_end'])),backwardIds=list(range(p['forward_end'],len(p['graph'].nodes))),parameterIds=[t.id for t in p['params']],gradientIds={a.node.name:b.id for a,b in zip(p['params'],p['grads'])})
        parameters=[dict(name=t.node.name,shape=list(t.shape),**self.parameter_summaries.get(t.node.name,dict(gradientNorm=None,valueNorm=float(np.linalg.norm(self.bindings[t.node.name]))))) for t in p['params']]
        result=dict(config=self.config,backend=self.config['backend'],metrics=metrics,history=self.history,graph=graph,plan=self.plan(),parameters=parameters,parameterCount=sum(v.size for k,v in self.bindings.items() if not k.startswith('state/')),
                    notes=['Float32. Fresh gradients per step; optimizer updates commit together.','Inspection is separate from benchmark timing.'])
        if boundary:
            boundary['probabilities']=probs[n:,1].tolist();result['boundary']=boundary
            def points(ids):return [dict(x=float(self.data['x'][i,0]),y=float(self.data['x'][i,1]),label=int(y[i])) for i in ids]
            result.update(samples=points(splits['train']),validationSamples=points(splits['validation']))
        else:
            ids=splits['validation'];pred=actual[ids].argmax(1);conf=np.zeros((10,10),dtype=int)
            for label,guess in zip(y[ids],pred):conf[label,guess]+=1
            incorrect=[i for i in ids if actual[i].argmax()!=y[i]][:8];correct=[i for i in ids if actual[i].argmax()==y[i]][:12]
            examples=[dict(pixels=(self.data['x'][i]*16).tolist(),label=int(y[i]),prediction=int(actual[i].argmax()),probabilities=actual[i].tolist()) for i in incorrect+correct]
            result['digits']=dict(examples=examples,confusion=conf.tolist(),accuracy=float((pred==y[ids]).mean()),split='fixed validation split · 266 samples',checkpointStep=self.step)
            evaluation=self.backend_metadata.get('evaluation',{})
            if evaluation.get('step')==self.step and evaluation.get('datasetChecksum')==self.data['checksum']:
                result['digits']['testAccuracy']=evaluation['accuracy']
        return result

    def inspect(self,node_id):
        if type(node_id)is not int or not 0<=node_id<len(self.training['graph'].nodes):raise ValueError('Invalid node id')
        # A separate graph rebuild keeps instrumentation out of the training plan/cache.
        p=build_mlp(self.config,self.batch,True);t=Tensor(p['graph'],node_id)
        b={**self.bindings,**(self.last_batch or {'x':self.data['x'][self.data['splits']['train'][:self.batch]],'labels':self.data['y'][self.data['splits']['train'][:self.batch]]})}
        targets=[t];dt=None
        if node_id<p['forward_end'] and t.dtype!='int32':dt=grad(p['outputs']['loss'],[t])[0];targets.append(dt)
        values=evaluate(targets,b)
        return dict(nodeId=node_id,shape=list(t.shape),dtype=t.dtype,values=values[t.id].ravel()[:64].tolist(),gradient=values[dt.id].ravel()[:64].tolist() if dt is not None else None,gradientShape=list(dt.shape) if dt is not None else None,totalElements=int(np.prod(t.shape)),instrumentation='Separate NumPy instrumented execution of the current parameters, bounded to 64 displayed values. Training buffers and benchmark timing are unaffected.')
