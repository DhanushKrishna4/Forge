"""Verify a built wheel in a disposable environment, independent of the checkout.

Requires a built dist/forge_ml-*.whl and access to the pinned NumPy wheel.
No reference framework, dataset download or long model training is performed.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
SMOKE=r'''
import json
import importlib.metadata
from pathlib import Path
import sys
import numpy as np
import forge_ml
from forge_ml import Graph, cross_entropy, grad, evaluate, optimizer_updates, initial_bindings, apply_updates
from forge_ml.compiler import compile_graph, execute_plan
assert Path(forge_ml.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), forge_ml.__file__
g=Graph();x=g.input('x',(3,2));w=g.parameter('w',(2,3));y=g.input('y',(3,),'int32')
loss=cross_entropy(x@w,y);dw,=grad(loss,[w]);updates=optimizer_updates([w],[dw],lr=.01)
bindings=initial_bindings(g,{'w':np.arange(6,dtype=np.float32).reshape(2,3)/10})
bindings.update(x=np.array([[1,0],[0,1],[1,1]],np.float32),y=np.array([0,1,2],np.int32))
plan=compile_graph(g,{'loss':loss,'dw':dw},updates=updates)
reference=evaluate([loss,dw,*updates.values()],bindings)
compiled=execute_plan(plan,bindings,simulate_memory=True)
for name,t in {'loss':loss,'dw':dw,**updates}.items():np.testing.assert_allclose(compiled[name],reference[t.id],rtol=3e-6,atol=3e-7)
before=float(reference[loss.id]);apply_updates(updates,bindings);after=float(evaluate([loss],bindings)[loss.id])
assert after<before and float(bindings['state/step'])==1
assert all(kernel['wgsl'].startswith('//') or '@compute' in kernel['wgsl'] for kernel in plan['kernels'])
print(json.dumps({'pass':True,'forgeVersion':importlib.metadata.version('forge-ml'),'numpyVersion':np.__version__,'importedFromCleanEnvironment':True,'sourceDirectoryOnPath':False,'graphNodes':len(g.nodes),'generatedKernels':len(plan['kernels']),'oneStepLossBefore':before,'oneStepLossAfter':after,'adamStep':float(bindings['state/step'])}))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheel',type=Path)
    parser.add_argument('--report',type=Path,help='Optional machine-readable verification result')
    args=parser.parse_args()
    matches=sorted((ROOT/'dist').glob('forge_ml-*.whl'))
    wheel=args.wheel or (matches[-1] if matches else None)
    if wheel is None or not wheel.is_file():parser.error('Build a wheel first with python -m build')
    wheel=wheel.resolve()
    with tempfile.TemporaryDirectory(prefix='forge-wheel-check-') as folder:
        temp=Path(folder);environment=temp/'venv'
        subprocess.run([sys.executable,'-m','venv',str(environment)],check=True)
        executable=environment/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
        env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
        env['PIP_CACHE_DIR']=str(temp/'pip-cache')
        subprocess.run([str(executable),'-m','pip','install','--only-binary=:all:',str(wheel)],cwd=temp,env=env,check=True)
        subprocess.run([str(executable),'-m','pip','check'],cwd=temp,env=env,check=True)
        run=subprocess.run([str(executable),'-I','-c',SMOKE],cwd=temp,env=env,check=True,text=True,capture_output=True)
        report=json.loads(run.stdout)
        cli=environment/('Scripts/forge.exe' if os.name=='nt' else 'bin/forge')
        help_result=subprocess.run([str(cli),'train','--help'],cwd=temp,env=env,check=True,text=True,capture_output=True)
        assert '--digits-asset' in help_result.stdout
        absent=subprocess.run([str(cli),'train','--dataset','digits','--steps','10','--digits-asset',str(temp/'missing-digits.json')],cwd=temp,env=env,text=True,capture_output=True)
        assert absent.returncode==2 and 'Digits asset not found' in absent.stderr
        report.update(cliHelp=True,cliMissingAssetError=True,wheel=wheel.name,wheelSha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),timestamp=datetime.now(timezone.utc).isoformat())
    # The temporary environment and its downloaded dependency are now removed.
    report['temporaryEnvironmentRemoved']=True
    output=json.dumps(report,indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(output+'\n')
    print(output)


if __name__=='__main__':main()
