"""Versioned bounded JSON. Imports reconstruct trusted graphs; never execute code."""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
import numpy as np
from .model import Experiment, MAX_STEPS

MAX_IMPORT_BYTES=8*1024*1024

def canonical(obj):return json.dumps(obj,sort_keys=True,separators=(',',':'),allow_nan=False)

def checksum_payload(obj):
    """Hash JSON semantics across Python/JS, not their different number spellings.

    JavaScript rewrites 1.0 as 1 and exponent notation differently. Encode each
    finite number as its exact binary64 value, normalize signed zero, and type-tag
    containers/strings to avoid ambiguous encodings. Large RNG ints are strings.
    """
    def normalize(value):
        if value is None:return ['null']
        if type(value)is bool:return ['bool',value]
        if type(value)in (int,float):
            if not math.isfinite(value) or (type(value)is int and abs(value)>2**53-1):raise ValueError('JSON numbers must be finite and integers browser-safe')
            return ['number',(float(value) or 0.).hex()]
        if isinstance(value,str):return ['string',value]
        if isinstance(value,list):return ['list',[normalize(x) for x in value]]
        if isinstance(value,dict) and all(isinstance(k,str) for k in value):return ['object',[[k,normalize(value[k])] for k in sorted(value)]]
        raise ValueError('Unsupported JSON value')
    return hashlib.sha256(canonical(normalize(obj)).encode()).hexdigest()

def export_experiment(exp):
    rng=deepcopy(exp.rng.bit_generator.state)
    for key in ('state','inc'):rng['state'][key]=str(rng['state'][key])
    payload=dict(config=exp.config,dataset={'id':exp.data['id'],'checksum':exp.data['checksum'],'splits':exp.data['splits']},
                 bindings={k:dict(shape=list(v.shape),dtype=str(v.dtype),data=v.ravel().tolist()) for k,v in exp.bindings.items()},
                 rng=rng,step=exp.step,elapsedMs=exp.elapsed,history=exp.history,
                 metadata={**exp.backend_metadata,'backend':exp.config['backend'],'dtype':'float32','framework':'forge-ml 0.1.0'})
    # A checkpoint is a snapshot: later training, inspection or caller edits must
    # not mutate its content (or invalidate the checksum) through shared lists.
    payload=deepcopy(payload)
    return dict(format='forge-experiment',version=1,payload=payload,sha256=checksum_payload(payload))

def validate_metadata(metadata, step, dataset):
    """Validate optional reporting data before any model is constructed.

    A checksum protects consistency, not the truth of user-supplied evaluation
    claims. The interface identifies a saved evaluation as reported metadata.
    """
    if not isinstance(metadata,dict) or len(canonical(metadata))>16384:raise ValueError('Invalid backend metadata')
    if 'backend' in metadata and metadata['backend'] not in ('cpu','webgpu'):raise ValueError('Invalid backend metadata backend')
    if 'dtype' in metadata and metadata['dtype']!='float32':raise ValueError('Invalid backend metadata dtype')
    if 'framework' in metadata and (not isinstance(metadata['framework'],str) or not 1<=len(metadata['framework'])<=100):raise ValueError('Invalid backend metadata framework')
    if 'evaluation' not in metadata:return
    evaluation=metadata['evaluation']
    required={'step','accuracy','datasetChecksum'}
    if not isinstance(evaluation,dict) or not required<=set(evaluation) or set(evaluation)-required-{'split','rows'}:raise ValueError('Invalid evaluation metadata fields')
    if type(evaluation['step'])is not int or not 0<=evaluation['step']<=step:raise ValueError('Invalid evaluation step')
    accuracy=evaluation['accuracy']
    if type(accuracy)not in (float,int) or not math.isfinite(accuracy) or not 0<=accuracy<=1:raise ValueError('Invalid evaluation accuracy')
    checksum=evaluation['datasetChecksum']
    if not isinstance(checksum,str) or len(checksum)!=64 or any(c not in '0123456789abcdef' for c in checksum) or not isinstance(dataset,dict) or checksum!=dataset.get('checksum'):raise ValueError('Invalid evaluation dataset checksum')
    if 'split' in evaluation and evaluation['split']!='fixed held-out test':raise ValueError('Invalid evaluation split')
    if 'rows' in evaluation:
        rows=evaluation['rows'];splits=dataset.get('splits')
        if type(rows)is not int or not 1<=rows<=2048 or not isinstance(splits,dict) or not isinstance(splits.get('test'),list) or rows!=len(splits['test']):raise ValueError('Invalid evaluation row count')


def import_experiment(obj,digits=None):
    if isinstance(obj,str):
        if len(obj.encode())>MAX_IMPORT_BYTES:raise ValueError('Import exceeds 8 MiB')
        try:obj=json.loads(obj,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
        except (RecursionError,json.JSONDecodeError) as e:raise ValueError('Invalid JSON') from e
    try:serialized=canonical(obj)
    except (TypeError,ValueError,RecursionError) as e:raise ValueError('Invalid JSON experiment') from e
    if len(serialized.encode())>MAX_IMPORT_BYTES:raise ValueError('Import exceeds 8 MiB')
    if not isinstance(obj,dict) or set(obj)!={'format','version','payload','sha256'} or obj['format']!='forge-experiment' or obj['version']!=1:
        raise ValueError('Unsupported experiment format/version')
    p=obj['payload']
    if checksum_payload(p)!=obj['sha256']:raise ValueError('Experiment checksum mismatch')
    if not isinstance(p,dict) or set(p)!={'config','dataset','bindings','rng','step','elapsedMs','history','metadata'}:raise ValueError('Invalid experiment structure')
    if type(p['step'])is not int or not 0<=p['step']<=MAX_STEPS:raise ValueError('Invalid step counter')
    if type(p['elapsedMs'])not in (int,float) or not math.isfinite(p['elapsedMs']) or not 0<=p['elapsedMs']<=1e10:raise ValueError('Invalid elapsed time')
    validate_metadata(p['metadata'],p['step'],p['dataset'])
    if not isinstance(p['bindings'],dict) or len(p['bindings'])>32:raise ValueError('Invalid bindings')
    # Validate lengths and numeric content before NumPy allocates imported arrays.
    for name,b in p['bindings'].items():
        if not isinstance(name,str) or len(name)>100 or not isinstance(b,dict) or set(b)!={'shape','dtype','data'}:raise ValueError('Invalid tensor record')
        if b['dtype']!='float32' or not isinstance(b['shape'],list) or len(b['shape'])>2 or any(type(v)is not int or not 1<=v<=100000 for v in b['shape']):raise ValueError('Invalid tensor type or shape')
        count=math.prod(b['shape'])
        if count>100000 or not isinstance(b['data'],list) or len(b['data'])!=count:raise ValueError('Invalid tensor length')
        if any(type(v)not in (int,float) or not math.isfinite(v) or abs(v)>1e20 for v in b['data']):raise ValueError('Invalid tensor values')
    exp=Experiment(p['config'],digits)
    if p['dataset']!={'id':exp.data['id'],'checksum':exp.data['checksum'],'splits':exp.data['splits']}:raise ValueError('Dataset identity or split mismatch')
    if set(p['bindings'])!=set(exp.bindings):raise ValueError('Parameter or optimizer state set mismatch')
    restored={}
    for name,old in exp.bindings.items():
        b=p['bindings'][name]
        if b['shape']!=list(old.shape):raise ValueError('Model tensor shape mismatch')
        restored[name]=np.asarray(b['data'],np.float32).reshape(old.shape)
    if 'state/step' in restored and float(restored['state/step'])!=p['step']:raise ValueError('Adam counter differs from experiment step')
    rng=deepcopy(p['rng'])
    if not isinstance(rng,dict) or set(rng)!={'bit_generator','state','has_uint32','uinteger'} or rng.get('bit_generator')!='PCG64' or len(canonical(rng))>1024:raise ValueError('Invalid RNG state')
    if not isinstance(rng['state'],dict) or set(rng['state'])!={'state','inc'}:raise ValueError('Invalid RNG state')
    for key in ('state','inc'):
        v=rng['state'][key]
        if not isinstance(v,str) or not 1<=len(v)<=39 or not v.isascii() or not v.isdigit() or not 0<=int(v)<2**128:raise ValueError('Invalid RNG integer')
        rng['state'][key]=int(v)
    if rng['state']['inc']%2!=1 or type(rng['has_uint32'])is not int or rng['has_uint32']not in (0,1) or type(rng['uinteger'])is not int or not 0<=rng['uinteger']<2**32:raise ValueError('Invalid RNG state flags')
    try:exp.rng.bit_generator.state=rng
    except (ValueError,TypeError,KeyError,OverflowError) as e:raise ValueError('Invalid RNG state') from e
    history=p['history']
    if not isinstance(history,list) or len(history)>400:raise ValueError('Invalid history')
    previous=-1
    for h in history:
        if not isinstance(h,dict) or set(h)!={'step','trainLoss','validationLoss'} or any(type(v)not in (int,float) or not math.isfinite(v) for v in h.values()) or type(h['step'])is not int or not previous<h['step']<=p['step'] or h['trainLoss']<0 or h['validationLoss']<0:raise ValueError('Invalid metrics')
        previous=h['step']
    exp.backend_metadata=deepcopy(p['metadata'])
    exp.bindings=restored;exp.step=p['step'];exp.elapsed=p['elapsedMs'];exp.history=history
    # Last-step gradients and batches are transient measurements, absent from this format.
    # A nonzero restored step counter does not make a fresh measurement available.
    exp.gradient_norm=None;exp.parameter_summaries={};exp.last_grads={};exp.last_batch=None
    return exp
