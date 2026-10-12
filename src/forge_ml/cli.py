"""Reproducible local training, evaluation and benchmark commands."""
import argparse
import json
import platform
import resource
import time
from pathlib import Path
import numpy as np
from .model import Experiment,DEFAULT_CONFIG
from .serialization import export_experiment,import_experiment


def train(args):
    digits=str(args.digits_asset) if args.dataset=='digits' else None
    c={**DEFAULT_CONFIG,'dataset':args.dataset,'hidden':[64,32] if args.dataset=='digits' else [32,32],
       'activation':'relu' if args.dataset=='digits' else 'tanh','learningRate':.003 if args.dataset=='digits' else .01,'batchSize':64,'seed':args.seed}
    exp=Experiment(c,digits)
    start=time.perf_counter();exp.cpu_step(10)
    pilot=(time.perf_counter()-start)/10
    report={'platform':platform.platform(),'python':platform.python_version(),'numpy':np.__version__,'seed':args.seed,
            'pilot':{'steps':10,'ms_per_step':pilot*1000,'peakProcessRSSBytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'estimatedRemainingSeconds':max(0,args.steps-10)*pilot},
            'model':c,'split':{k:len(v) for k,v in exp.data['splits'].items()},'datasetChecksum':exp.data['checksum'],'validation':[]}
    print(json.dumps(report['pilot']),flush=True)
    best=None;best_loss=float('inf');training_start=time.perf_counter()
    # Fixed budget; choose the lowest validation loss without looking at held-out test accuracy.
    while exp.step<args.steps:
        exp.cpu_step(min(50,args.steps-exp.step))
        ids=exp.data['splits']['validation'];result=exp.evaluate(exp.data['x'][ids],exp.data['y'][ids]);loss=float(result['loss'])
        accuracy=float((result['probabilities'].argmax(1)==exp.data['y'][ids]).mean())
        report['validation'].append({'step':exp.step,'loss':loss,'accuracy':accuracy})
        print(f'step={exp.step} validation_loss={loss:.5f} validation_accuracy={accuracy:.4f}',flush=True)
        if loss<best_loss:best_loss=loss;best=export_experiment(exp)
    if best is not None:exp=import_experiment(best,digits)
    ids=exp.data['splits']['test'];result=exp.evaluate(exp.data['x'][ids],exp.data['y'][ids]);pred=result['probabilities'].argmax(1)
    report.update(selectedStep=exp.step,trainingSeconds=time.perf_counter()-training_start,
                  test={'loss':float(result['loss']),'accuracy':float((pred==exp.data['y'][ids]).mean()),'rows':len(ids)},
                  peakProcessRSSBytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  forgePersistentTensorBytes=sum(v.nbytes for v in exp.bindings.values()))
    conf=np.zeros((10 if args.dataset=='digits' else 2,)*2,dtype=int)
    for a,b in zip(exp.data['y'][ids],pred):conf[a,b]+=1
    report['test']['confusion']=conf.tolist()
    exp.backend_metadata['evaluation']={'step':exp.step,'datasetChecksum':exp.data['checksum'],'accuracy':report['test']['accuracy'],'split':'fixed held-out test','rows':len(ids)}
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(export_experiment(exp),separators=(',',':')))
    out.with_suffix('.metrics.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('model','validation')},indent=2))


def main():
    parser=argparse.ArgumentParser(prog='forge')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('train');p.add_argument('--dataset',choices=['digits','xor','moons','spiral'],default='moons');p.add_argument('--steps',type=int,default=800);p.add_argument('--seed',type=int,default=42);p.add_argument('--output',default='benchmarks/trained-experiment.json')
    p.add_argument('--digits-asset',type=Path,help='Prepared digits.json path; required for digits training outside a Forge source checkout')
    p=sub.add_parser('benchmark');p.add_argument('--output',default='benchmarks/native.json');p.add_argument('--repetitions',type=int,default=15)
    args=parser.parse_args()
    if args.command=='train':
        if not 10<=args.steps<=10000:parser.error('Use 10–10000 steps')
        if args.digits_asset is not None and args.dataset!='digits':parser.error('--digits-asset is only used with --dataset digits')
        if args.dataset=='digits':
            default_asset=Path(__file__).resolve().parents[2]/'web/public/assets/digits.json'
            args.digits_asset=(args.digits_asset or default_asset).expanduser().resolve()
            if not args.digits_asset.is_file():
                parser.error(f'Digits asset not found: {args.digits_asset}. Pass --digits-asset /path/to/prepared/digits.json; a source checkout can generate it with scripts/prepare_digits.py.')
        train(args)
    else:
        from .benchmark import run_benchmarks
        result=run_benchmarks(args.repetitions);Path(args.output).write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
