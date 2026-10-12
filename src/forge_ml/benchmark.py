"""Small reproducible fixtures; timings include completed work, never submission alone."""
import platform
import statistics
import time
from datetime import datetime,timezone
import numpy as np
from .graph import Graph
from .compiler import compile_graph, execute_plan
from .model import Experiment


def fixtures():
    rng=np.random.default_rng(7)
    g=Graph();x=g.input('x',(64,64));y=x
    for _ in range(4):y=(y*.7+.2).tanh()
    yield 'elementwise chain',g,{'result':y},{'x':rng.normal(size=x.shape).astype(np.float32)},{}
    g=Graph();x=g.input('x',(32,32));a=(x*.5).tanh();b=(x*.5).tanh();y=a+b
    yield 'common subexpressions',g,{'result':y},{'x':rng.normal(size=x.shape).astype(np.float32)},{}
    g=Graph();x=g.input('x',(32,32));y=x
    for _ in range(6):y=(y@x)*.01
    yield 'scratch reuse',g,{'result':y},{'x':rng.normal(size=x.shape).astype(np.float32)},{}
    for size in (15,32,65):
        g=Graph();x=g.input('a',(size,size+3));y=g.input('b',(size+3,size-2));z=x@y
        yield f'matmul {size}×{size+3}×{size-2}',g,{'result':z},{'a':rng.normal(size=x.shape).astype(np.float32),'b':rng.normal(size=y.shape).astype(np.float32)},{}
    exp=Experiment({'hidden':[16,16],'batchSize':32});p=exp.training
    yield 'MLP complete Adam step',p['graph'],p['outputs'],{**exp.bindings,**exp.next_batch()},p['updates']


def benchmark_plans():
    cases=[]
    for name,g,outputs,bindings,updates in fixtures():
        for optimized in (False,True):
            opts={k:optimized for k in ('dce','fold','cse','fusion','reuse','tiled')}
            start=time.perf_counter();plan=compile_graph(g,outputs,opts,updates);elapsed=(time.perf_counter()-start)*1000
            cases.append(dict(name=name,configuration='optimized' if optimized else 'unoptimized',plan=plan,bindings={k:v.ravel().tolist() for k,v in bindings.items()},compilationMs=elapsed,
                              shapes={k:list(v.shape) for k,v in bindings.items()},dtype='float32'))
    return cases


def _execute_and_commit(plan,bindings):
    values=execute_plan(plan,bindings)
    # Include the out-of-place binding commit in every full-step execution,
    # including warmups and each member of an amortized timing sample.
    for name in plan['updates']:bindings[name]=values[name]


def run_benchmarks(repetitions=15,iterations_per_sample=1):
    if type(repetitions)is not int or not 3<=repetitions<=50:raise ValueError('Use 3–50 repetitions')
    if type(iterations_per_sample)is not int or not 1<=iterations_per_sample<=100:raise ValueError('Use 1–100 iterations per sample')
    rows=[]
    for case in benchmark_plans():
        plan=case['plan'];spec=plan['bindings']
        bindings={k:np.asarray(v,dtype=spec[k]['dtype']).reshape(spec[k]['shape']) for k,v in case['bindings'].items()}
        for _ in range(3):_execute_and_commit(plan,bindings)
        total_times=[]
        for _ in range(repetitions):
            start=time.perf_counter()
            for _ in range(iterations_per_sample):_execute_and_commit(plan,bindings)
            total_times.append((time.perf_counter()-start)*1000)
        times=[elapsed/iterations_per_sample for elapsed in total_times]
        stats=plan['stats'];rows.append(dict(name=case['name'],backend='native NumPy',configuration=case['configuration'],medianMs=statistics.median(times),minMs=min(times),maxMs=max(times),stdMs=statistics.pstdev(times),compilationMs=case['compilationMs'],executionMs=statistics.median(times),uploadMs=0,readbackMs=0,kernels=stats.get('kernel_count',len(plan['kernels'])),scratchBytes=stats.get('scratch_bytes',0),repetitions=repetitions,iterationsPerSample=iterations_per_sample,totalSamplesMs=total_times,samplesMs=times,shapes=case['shapes'],dtype='float32',compiler=plan['options']))
    return dict(rows=rows,timingMethod=f'perf_counter wall clock around synchronous NumPy completion and update commit; 3 individual warmups; {iterations_per_sample} executions per sample, with total elapsed time divided by iterationsPerSample',environment={'platform':platform.platform(),'python':platform.python_version(),'numpy':np.__version__,'backend':'native NumPy'},timestamp=datetime.now(timezone.utc).isoformat(),notes=['Fusion, tiling and GPU scratch capacities describe generated GPU plans, not NumPy acceleration.','CPU execution excludes runtime startup and uses no GPU transfers.','CPU samples may amortize multiple executions to reduce browser timer quantization; sample variation describes batch averages, not individual execution latency. Raw totalSamplesMs and iterationsPerSample are retained.','CPU and WebGPU timing boundaries differ: GPU samples include queue completion and output readback, while synchronous NumPy needs neither.','Matmul BLAS behavior depends on installation; see launch environment.'])
