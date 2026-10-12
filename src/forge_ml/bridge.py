"""JSON-only Pyodide boundary. No imported executable code or per-op JS calls."""
import json
import sys
import numpy as np
from .model import Experiment,build_mlp
from .serialization import export_experiment,import_experiment
from .compiler import compile_graph

experiment=None
digits=None

def flat(bindings):return {k:v.ravel().tolist() for k,v in bindings.items()}

def handle(command,payload=None):
    global experiment,digits
    payload=payload or {}
    if command=='assets':digits=payload['digits'];return {'python':sys.version.split()[0],'numpy':np.__version__}
    if command=='init':
        candidate=import_experiment(payload['checkpoint'],digits) if isinstance(payload.get('checkpoint'),dict) else Experiment(payload.get('config'),digits)
        if payload.get('config') and isinstance(payload.get('checkpoint'),dict):
            from .model import validate_config
            # Checkpoint carries architecture/training state. Only backend and compiler may be selected at restore.
            candidate.config=validate_config({**candidate.config,**{k:payload['config'][k] for k in ('backend','compiler') if k in payload['config']}})
        experiment=candidate
        return {'plan':experiment.plan(),'bindings':flat({**experiment.bindings,'x':np.zeros((experiment.batch,experiment.data['x'].shape[1]),np.float32),'labels':np.zeros(experiment.batch,np.int32)})}
    if command=='import':
        experiment=import_experiment(payload['experiment'],digits)
        if payload.get('backend') in ('cpu','webgpu'):experiment.config['backend']=payload['backend']
        return {'config':experiment.config,'plan':experiment.plan(),'bindings':flat({**experiment.bindings,'x':np.zeros((experiment.batch,experiment.data['x'].shape[1]),np.float32),'labels':np.zeros(experiment.batch,np.int32)})}
    if command=='bench-plans':
        from .benchmark import benchmark_plans
        return benchmark_plans()
    if command=='benchmark':
        from .benchmark import run_benchmarks
        # Small NumPy fixtures can finish within one coarse browser timer tick.
        # Retain raw batch totals while reporting amortized per-execution times.
        return run_benchmarks(payload.get('repetitions',7),iterations_per_sample=20)
    if command=='validate-case':
        from .compiler import execute_plan
        case=payload['case'];plan=case['plan'];spec=plan['bindings']
        bindings={k:np.asarray(v,dtype=spec[k]['dtype']).reshape(spec[k]['shape']) for k,v in case['bindings'].items()}
        steps=[]
        for _ in case['expected']:
            values=execute_plan(plan,bindings)
            for k in plan['updates']:bindings[k]=values[k].copy()
            steps.append({'values':flat(values),'bindings':flat({k:v for k,v in bindings.items() if spec[k]['kind'] in ('parameter','state')})})
        return steps
    if experiment is None:raise ValueError('Initialize an experiment first')
    e=experiment
    if command=='step':e.cpu_step(payload.get('steps',1),compiled=True);return e.snapshot()
    if command=='snapshot':return e.snapshot(payload.get('probabilities'),payload.get('losses'))
    if command=='batch':return flat(e.next_batch())
    if command=='gpu-complete':
        e.step+=1;e.last_ms=payload['ms'];e.elapsed+=payload['ms'];e.gradient_norm=payload['gradientNorm']
        e.parameter_summaries=payload.get('parameters',{})
        return {'step':e.step}
    if command=='sync':
        values=payload['bindings']
        restored={}
        for k,old in e.bindings.items():
            a=np.asarray(values[k],np.float32).reshape(old.shape)
            if not np.isfinite(a).all():raise ValueError('Nonfinite GPU state; lower the learning rate')
            restored[k]=a
        e.bindings.update(restored)
        return {'step':e.step}
    if command=='eval-plan':
        kind=payload.get('kind','display')
        if kind=='display':inputs,_=e.evaluation_inputs()
        elif kind=='drawing':
            pixels=payload['pixels']
            if not isinstance(pixels,list) or len(pixels)!=64 or any(type(x)not in (int,float) or not 0<=x<=16 for x in pixels):raise ValueError('Drawing must contain 64 pixels in [0,16]')
            if e.config['dataset']!='digits':raise ValueError('Select digits before drawing inference')
            inputs={'x':np.asarray(pixels,np.float32).reshape(1,64)/16,'labels':np.zeros(1,np.int32)}
        elif kind in ('train','validation'):
            ids=e.data['splits'][kind];inputs={'x':e.data['x'][ids],'labels':e.data['y'][ids]}
        else:raise ValueError('Unknown evaluation request')
        p=e.program(len(inputs['x']),False)
        outputs={'probabilities':p['outputs']['probabilities']} if kind in ('display','drawing') else {'loss':p['outputs']['loss']}
        plan=compile_graph(p['graph'],outputs,e.config['compiler'])
        bindings={**{k:v for k,v in e.bindings.items() if not k.startswith('state/')},**inputs}
        # Prune unused labels removed by DCE before sending to strict GPU bindings.
        bindings={k:v for k,v in bindings.items() if k in plan['bindings']}
        return {'plan':plan,'bindings':flat(bindings),'readNames':list(outputs)}
    if command=='export':return export_experiment(e)
    if command=='inspect':return e.inspect(payload['nodeId'])
    if command=='predict':
        if e.config['dataset']!='digits':raise ValueError('Select digits')
        pixels=payload['pixels']
        if not isinstance(pixels,list) or len(pixels)!=64 or any(type(v)not in (float,int) or not 0<=v<=16 for v in pixels):raise ValueError('Drawing must contain 64 pixels in [0,16]')
        return {'probabilities':e.evaluate(np.asarray(pixels,np.float32).reshape(1,64)/16)['probabilities'].ravel().tolist()}
    raise ValueError('Unknown worker command')

def call(command,payload_json='{}'):
    return json.dumps(handle(command,json.loads(payload_json)),separators=(',',':'),allow_nan=False)
