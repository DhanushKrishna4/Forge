"""Bounded acceptance checks: genuine learning, deterministic data and artifact metrics."""
import json
from pathlib import Path
import numpy as np
import pytest
from forge_ml.datasets import dataset_2d, load_digits_asset, stratified_split
from forge_ml.model import Experiment, validate_config, MAX_CACHE, MAX_STEPS
from forge_ml.serialization import import_experiment

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'web/public/assets'


@pytest.mark.parametrize('dataset',['xor','moons','spiral'])
def test_small_demonstrations_train_from_random_initialization(dataset):
    exp=Experiment({'dataset':dataset,'samples':100,'noise':.05,'hidden':[12,12],'learningRate':.02,'batchSize':32})
    ids=exp.data['splits']['train']; x,y=exp.data['x'][ids],exp.data['y'][ids]
    before=exp.evaluate(x,y); original={k:a.copy() for k,a in exp.bindings.items()}
    exp.cpu_step(30)
    after=exp.evaluate(x,y)
    assert float(after['loss']) < .85*float(before['loss'])
    assert not np.allclose(before['probabilities'],after['probabilities'])
    assert exp.step==30 and np.isfinite(exp.gradient_norm) and exp.gradient_norm>0
    assert any(not np.array_equal(original[k],exp.bindings[k]) for k in original if not k.startswith('state/'))


@pytest.mark.parametrize('kind',['sgd','momentum','adam'])
def test_compiled_native_training_matches_unoptimized_interpreter(kind):
    config={'dataset':'moons','samples':80,'hidden':[7,5],'batchSize':13,'optimizer':kind}
    expected,actual=Experiment(config),Experiment(config)
    expected.cpu_step(5); actual.cpu_step(5,compiled=True)
    for name in expected.bindings:
        np.testing.assert_allclose(actual.bindings[name],expected.bindings[name],rtol=3e-6,atol=3e-7)
    np.testing.assert_allclose(actual.gradient_norm,expected.gradient_norm,rtol=3e-6)


def test_seeded_data_partition_and_custom_points():
    for kind in ['xor','moons','spiral']:
        a,b=dataset_2d(kind,n=63,seed=77),dataset_2d(kind,n=63,seed=77)
        assert a['checksum']==b['checksum'] and a['splits']==b['splits']
        ids=sum(a['splits'].values(),[])
        assert sorted(ids)==list(range(63))
        assert all(len(indices)>0 for indices in a['splits'].values())
    points=[{'x':float(i)/4,'y':float(i%3)/3,'label':i%2} for i in range(8)]
    exp=Experiment({'dataset':'custom','customPoints':points,'samples':20,'batchSize':128})
    assert exp.batch==len(exp.data['splits']['train'])
    exp.cpu_step()
    for mutate in [lambda p:p[:-3],lambda p:[{**v,'label':0} for v in p],lambda p:[{**v,'x':float('inf')} for v in p],lambda p:[{**v,'label':.5} for v in p]]:
        with pytest.raises(ValueError): dataset_2d('custom',points=mutate(points))


def test_program_cache_bound_and_batch_size_one():
    exp=Experiment({'hidden':[4],'batchSize':1,'samples':40})
    for batch in range(1,MAX_CACHE+6): exp.program(batch,False)
    assert len(exp.cache)==MAX_CACHE
    exp.cpu_step(2)
    assert exp.step==2
    exp.step=MAX_STEPS
    with pytest.raises(ValueError):exp.cpu_step()
    for steps in [0,101,-1,1.5,True]:
        with pytest.raises(ValueError):exp.cpu_step(steps)


def test_forward_tensor_inspector_values_and_gradients_are_real():
    exp=Experiment({'hidden':[4],'batchSize':3,'samples':40})
    exp.cpu_step(2)
    parameter=exp.training['params'][0]
    inspected=exp.inspect(parameter.id)
    np.testing.assert_array_equal(np.array(inspected['values']),exp.bindings[parameter.node.name].ravel())
    assert len(inspected['gradient'])==parameter.size
    assert inspected['gradientShape']==list(parameter.shape)
    assert np.isfinite(inspected['gradient']).all()
    with pytest.raises(ValueError):exp.inspect(-1)


@pytest.mark.parametrize('field,value',[('samples',10000000),('hidden',[65]),('hidden',[2,2,2,2]),('batchSize',0),('seed',-1),('learningRate',float('nan')),('noise',float('inf')),('activation','gelu'),('optimizer','invalid'),('backend','cuda'),('compiler',{'fusion':'yes'})])
def test_configuration_rejects_unsupported_or_excessive_values(field,value):
    with pytest.raises(ValueError):validate_config({field:value})


def test_delivered_digits_checkpoint_reproduces_reported_held_out_metrics():
    dataset=ASSETS/'digits.json'; checkpoint=ASSETS/'digits-checkpoint.json'; report=json.loads((ASSETS/'digits-checkpoint.metrics.json').read_text())
    exp=import_experiment(checkpoint.read_text(),dataset)
    assert exp.data['splits']==stratified_split(exp.data['y'],2026)
    assert exp.data['checksum']==report['datasetChecksum']
    assert exp.step==report['selectedStep']
    ids=exp.data['splits']['test']; y=exp.data['y'][ids]
    values=exp.evaluate(exp.data['x'][ids],y); predicted=values['probabilities'].argmax(1)
    accuracy=float(np.mean(predicted==y))
    assert accuracy==report['test']['accuracy'] and accuracy>=.90
    np.testing.assert_allclose(values['loss'],report['test']['loss'],rtol=2e-6,atol=2e-7)
    confusion=np.zeros((10,10),dtype=int)
    for truth,guess in zip(y,predicted):confusion[truth,guess]+=1
    np.testing.assert_array_equal(confusion,report['test']['confusion'])
    # Resume is real training on the same fixture; test rows are never a batch source.
    before={k:a.copy() for k,a in exp.bindings.items()}; exp.cpu_step()
    assert exp.step==report['selectedStep']+1
    assert any(not np.array_equal(before[k],exp.bindings[k]) for k in before if not k.startswith('state/'))


def test_digits_asset_rejects_corrupt_pixels_checksum_and_splits():
    obj=json.loads((ASSETS/'digits.json').read_text())
    for mutate in [lambda x:x['pixels'].__setitem__(0,17), lambda x:x.__setitem__('checksum','0'*64),lambda x:x['splits']['train'].__setitem__(0,x['splits']['validation'][0])]:
        changed=json.loads(json.dumps(obj));mutate(changed)
        with pytest.raises(ValueError):load_digits_asset(changed)


@pytest.mark.parametrize('field,value',[('labels',None),('splits',None),('splits',{'train':None}),('provenance',None)])
def test_digits_asset_reports_malformed_metadata(field,value):
    obj=json.loads((ASSETS/'digits.json').read_text());obj[field]=value
    with pytest.raises(ValueError):load_digits_asset(obj)


def test_digits_fixed_split_cannot_be_changed_even_if_partition_remains_valid():
    obj=json.loads((ASSETS/'digits.json').read_text())
    train,validation=obj['splits']['train'],obj['splits']['validation']
    train[0],validation[0]=validation[0],train[0]
    with pytest.raises(ValueError,match='fixed'):load_digits_asset(obj)


def test_failed_training_step_preserves_rng_and_last_complete_state():
    from forge_ml.serialization import canonical
    exp=Experiment({'hidden':[4],'samples':40});exp.cpu_step()
    original=exp.bindings['layer0.weight'].copy()
    exp.bindings['layer0.weight'][0,0]=np.nan
    state=canonical(exp.rng.bit_generator.state);previous_batch=exp.last_batch
    with pytest.raises(ValueError):exp.cpu_step()
    assert state==canonical(exp.rng.bit_generator.state)
    assert exp.last_batch is previous_batch and exp.step==1
    exp.bindings['layer0.weight']=original
    exp.cpu_step();assert exp.step==2
