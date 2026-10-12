"""Untrusted JSON validation, exact RNG/state continuation, safe formats only."""
from copy import deepcopy
import hashlib
import json
import numpy as np
import pytest
from forge_ml.model import Experiment
from forge_ml.serialization import export_experiment, import_experiment, canonical, MAX_IMPORT_BYTES, checksum_payload


def reseal(obj):
    obj['sha256']=checksum_payload(obj['payload'])
    return obj


@pytest.mark.parametrize('kind',['sgd','momentum','adam'])
def test_checkpoint_exact_multi_step_continuation(kind):
    exp=Experiment({'dataset':'xor','optimizer':kind,'samples':80,'batchSize':13,'hidden':[7,5]})
    exp.cpu_step(4)
    obj=export_experiment(exp); restored=import_experiment(json.dumps(obj))
    assert restored.step==exp.step
    assert canonical(restored.rng.bit_generator.state)==canonical(exp.rng.bit_generator.state)
    for _ in range(3):
        exp.cpu_step();restored.cpu_step()
        np.testing.assert_array_equal(exp.last_batch['x'],restored.last_batch['x'])
        np.testing.assert_array_equal(exp.last_batch['labels'],restored.last_batch['labels'])
        for name in exp.bindings:np.testing.assert_array_equal(exp.bindings[name],restored.bindings[name])
    assert restored.step==exp.step==7


def test_custom_points_round_trip_and_initial_parameters_preserved():
    points=[{'x':float(i%4)/4,'y':float(i//4)/4,'label':i%2} for i in range(8)]
    exp=Experiment({'dataset':'custom','customPoints':points,'hidden':[4],'samples':20})
    initial={k:a.copy() for k,a in exp.initial.items()};exp.cpu_step(2)
    restored=import_experiment(export_experiment(exp))
    assert restored.config['customPoints']==points and restored.data['checksum']==exp.data['checksum']
    for name in initial:np.testing.assert_array_equal(initial[name],restored.initial[name])


@pytest.mark.parametrize('mutation',[
    lambda p:p.__setitem__('step',-1),
    lambda p:p.__setitem__('step',10001),
    lambda p:p.__setitem__('step',True),
    lambda p:p.__setitem__('elapsedMs',-1),
    lambda p:p['config'].__setitem__('hidden',[10000000]),
    lambda p:p['config'].__setitem__('code','__import__("os")'),
    lambda p:p['bindings'].__setitem__('rogue',{'shape':[],'dtype':'float32','data':[0]}),
    lambda p:p['bindings']['layer0.weight'].__setitem__('shape',[100001]),
    lambda p:p['bindings']['layer0.weight'].__setitem__('shape',[1,1,1]),
    lambda p:p['bindings']['layer0.weight'].__setitem__('dtype','float64'),
    lambda p:p['bindings']['layer0.weight'].__setitem__('data',[]),
    lambda p:p['bindings']['layer0.weight']['data'].__setitem__(0,'1.0'),
    lambda p:p['bindings']['layer0.weight']['data'].__setitem__(0,1e21),
    lambda p:p['bindings']['state/step']['data'].__setitem__(0,6),
    lambda p:p['dataset'].__setitem__('checksum','bad'),
    lambda p:p['dataset']['splits']['train'].__setitem__(0,99999),
    lambda p:p.__setitem__('rng',{'bit_generator':'MT19937'}),
    lambda p:p.__setitem__('history',[{'step':1000,'trainLoss':1.,'validationLoss':1.}]),
    lambda p:p.__setitem__('history',[{}]*401),
    lambda p:p.__setitem__('history',[{'step':0.5,'trainLoss':1.,'validationLoss':1.}]),
    lambda p:p.__setitem__('history',[{'step':0,'trainLoss':-1.,'validationLoss':1.}]),
    lambda p:p.__setitem__('history',[{'step':0,'trainLoss':1.,'validationLoss':1.}]*2),
    lambda p:p.__setitem__('metadata',None),
    lambda p:p.__setitem__('metadata',{'large':'x'*17000}),
])
def test_invalid_payloads_fail_before_execution(mutation):
    exp=Experiment({'samples':40,'hidden':[4]});obj=export_experiment(exp)
    mutation(obj['payload']);reseal(obj)
    with pytest.raises(ValueError):import_experiment(obj)


def test_version_checksum_nonfinite_and_oversize_imports():
    obj=export_experiment(Experiment({'samples':40,'hidden':[4]}))
    changed=deepcopy(obj);changed['version']=2
    with pytest.raises(ValueError):import_experiment(changed)
    changed=deepcopy(obj);changed['payload']['config']['seed']=23
    with pytest.raises(ValueError,match='checksum'):import_experiment(changed)
    for source in ['{"x":NaN}', 'this is not json', ' '* (MAX_IMPORT_BYTES+1), b'not a pickle', ['not-an-object']]:
        with pytest.raises(ValueError):import_experiment(source)
    changed=deepcopy(obj);changed['payload']['bindings']['layer0.weight']['data'][0]=float('nan')
    with pytest.raises(ValueError):import_experiment(changed)


def test_backend_metadata_round_trips():
    exp=Experiment({'samples':40,'hidden':[4]})
    exp.backend_metadata={'adapter':'test adapter','timingMethod':'synchronized wall clock'}
    restored=import_experiment(export_experiment(exp))
    assert restored.backend_metadata['adapter']=='test adapter'
    assert restored.backend_metadata['timingMethod']=='synchronized wall clock'


def test_export_is_a_detached_snapshot_of_configuration_and_metrics():
    exp=Experiment({'samples':40,'hidden':[4]})
    exp.history=[{'step':0,'trainLoss':1.,'validationLoss':1.}]
    obj=export_experiment(exp)
    original=canonical(obj)
    exp.history.append({'step':1,'trainLoss':.9,'validationLoss':.9})
    exp.config['hidden'][0]=8
    exp.data['splits']['train'].reverse()
    assert canonical(obj)==original
    import_experiment(obj)


def test_browser_json_round_trip_preserves_rng_and_checksum():
    import shutil,subprocess
    if not shutil.which('node'):pytest.skip('Node is required for the browser JSON boundary check')
    exp=Experiment({'samples':40,'hidden':[4]});exp.cpu_step(3)
    exported=export_experiment(exp)
    exported['payload']['metadata']['numberSpelling']=[1.0,-0.0,1e-7,1e-20]
    reseal(exported)
    roundtrip=subprocess.run(['node','-e',"let s='';process.stdin.on('data',c=>s+=c);process.stdin.on('end',()=>process.stdout.write(JSON.stringify(JSON.parse(s))));"],input=json.dumps(exported),text=True,capture_output=True,check=True).stdout
    restored=import_experiment(roundtrip)
    exp.cpu_step();restored.cpu_step()
    for name in exp.bindings:np.testing.assert_array_equal(exp.bindings[name],restored.bindings[name])
