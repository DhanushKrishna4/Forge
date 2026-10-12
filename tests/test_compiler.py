"""Pass-by-pass output/gradient equivalence and physical lifetime invariants."""
import json
import numpy as np
import pytest

from forge_ml import Graph, cross_entropy, evaluate, grad, initial_bindings, optimizer_updates
from forge_ml.compiler import compile_graph, execute_plan, validate_plan_memory

DISABLED = dict(dce=False,fold=False,cse=False,fusion=False,reuse=False,tiled=False)


def fixture(dtype='float32', optimizer='adam', seed=4):
    rng = np.random.default_rng(seed)
    g=Graph(dtype);x=g.input('x',(5,3));y=g.input('y',(5,),dtype='int32')
    w=g.parameter('w',(3,7));b=g.parameter('b',(7,));v=g.parameter('v',(7,3))
    hidden=(x@w+b).tanh();logits=hidden@v
    # Both shared nodes and independently constructed common expressions.
    shared=hidden*hidden;loss=cross_entropy(logits,y)+shared.mean()*0.001+(hidden*hidden).mean()*0.001
    gradients=grad(loss,[w,b,v]); updates=optimizer_updates([w,b,v],gradients,kind=optimizer)
    unused=(x+1).exp(); assert unused.shape==(5,3)
    const=(g.constant([1,2,3])+g.constant([2,3,4])).sum()
    outputs={'loss':loss,'constant':const,**{f'gradient:{p.node.name}':d for p,d in zip([w,b,v],gradients)}}
    bindings=initial_bindings(g,{p.node.name:rng.normal(0,.2,p.shape).astype(dtype) for p in [w,b,v]})
    bindings.update(x=rng.normal(size=x.shape).astype(dtype),y=np.array([0,1,2,2,1],np.int32))
    return g,outputs,updates,bindings


@pytest.mark.parametrize('dtype',['float32','float64'])
@pytest.mark.parametrize('enabled',[None,'dce','fold','cse','fusion','reuse','tiled','all'])
def test_passes_preserve_outputs_and_gradients(dtype,enabled):
    g,outputs,updates,bindings=fixture(dtype)
    options={} if enabled=='all' else {**DISABLED,**({enabled:True} if enabled else {})}
    plan=compile_graph(g,outputs,options,updates)
    direct=evaluate(list(outputs.values())+list(updates.values()),bindings)
    for replay in (False,True):
        results=execute_plan(plan,bindings,simulate_memory=replay)
        for name,tensor in {**outputs,**updates}.items():
            np.testing.assert_allclose(results[name],direct[tensor.id],rtol=3e-6 if dtype=='float32' else 1e-12,atol=3e-7 if dtype=='float32' else 1e-13)
    validate_plan_memory(plan)
    json.dumps(plan,allow_nan=False)


@pytest.mark.parametrize('kind',['sgd','momentum','adam'])
def test_repeated_updates_use_consistent_old_state(kind):
    g,outputs,updates,bindings=fixture(optimizer=kind)
    reference={k:v.copy() for k,v in bindings.items()};actual={k:v.copy() for k,v in bindings.items()}
    plan=compile_graph(g,outputs,updates=updates)
    for _ in range(5):
        wanted=evaluate(list(outputs.values())+list(updates.values()),reference)
        result=execute_plan(plan,actual,simulate_memory=True)
        for name,tensor in {**outputs,**updates}.items(): np.testing.assert_allclose(result[name],wanted[tensor.id],rtol=3e-6,atol=3e-7)
        reference.update({name:wanted[t.id] for name,t in updates.items()})
        actual.update({name:result[name] for name in updates})


def test_parameters_never_fold_and_cache_excludes_binding_values():
    g=Graph();w=g.parameter('w',(3,));out=(w+2)*3
    p=compile_graph(g,{'out':out})
    assert next(n for n in p['nodes'] if n['id']==w.id)['op']=='parameter'
    a=execute_plan(p,{'w':np.array([1,2,3],np.float32)})['out']
    b=execute_plan(p,{'w':np.array([4,5,6],np.float32)})['out']
    assert not np.array_equal(a,b)
    assert p['cache_key']==compile_graph(g,{'out':out})['cache_key']


def test_fusion_and_reuse_produce_real_reductions():
    g=Graph();x=g.input('x',(64,64));a=(x+1).tanh();b=(a*2).sigmoid();c=(b+3).relu();d=(c*4).tanh()
    plain=compile_graph(g,{'out':d},DISABLED)
    fused=compile_graph(g,{'out':d},{**DISABLED,'fusion':True})
    reused=compile_graph(g,{'out':d},{**DISABLED,'reuse':True})
    assert fused['stats']['kernel_count'] < plain['stats']['kernel_count']
    assert reused['stats']['scratch_bytes'] < plain['stats']['scratch_bytes']
    for kernel in fused['kernels']: assert len(kernel['node_ids'])<=8 and len(kernel['inputs'])+1<=8


def test_instrumentation_preserves_every_source_value():
    g,outputs,updates,bindings=fixture()
    plan=compile_graph(g,outputs,{'instrument':True},updates)
    actual=execute_plan(plan,bindings,simulate_memory=True)
    expected=evaluate([g_tensor(g,n.id) for n in g.nodes],bindings)
    for node in g.nodes: np.testing.assert_allclose(actual[f'tensor:{node.id}'],expected[node.id],rtol=3e-6,atol=3e-7)
    assert all(len(k['node_ids'])==1 for k in plan['kernels'])


def g_tensor(g,ident):
    from forge_ml import Tensor
    return Tensor(g,ident)


@pytest.mark.parametrize('seed',range(10))
def test_random_bounded_graphs(seed):
    rng=np.random.default_rng(seed);g=Graph('float64');x=g.input('x',(3,5));b=g.parameter('b',(5,));values=[x+b]
    for _ in range(12):
        a=values[int(rng.integers(len(values)))];op=int(rng.integers(4))
        values.append((a*float(rng.uniform(.1,.5))+float(rng.uniform(-.5,.5))).tanh() if op<2 else a+values[int(rng.integers(len(values)))]*.1)
    loss=values[-1].mean()+values[-2].mean();db=grad(loss,[b])[0]
    bindings={'x':rng.normal(size=x.shape),'b':rng.normal(size=b.shape)}
    expected=evaluate([loss,db],bindings)
    for options in (DISABLED,{}, {'reuse':False}, {'max_fusion':2,'max_bindings':4}):
        p=compile_graph(g,{'loss':loss,'db':db},options)
        got=execute_plan(p,bindings,simulate_memory=True)
        np.testing.assert_allclose(got['loss'],expected[loss.id],rtol=1e-12,atol=1e-13)
        np.testing.assert_allclose(got['db'],expected[db.id],rtol=1e-12,atol=1e-13)


def test_tiled_matmul_edges_and_no_barrier_early_exit():
    g=Graph();a=g.input('a',(3,11));b=g.input('b',(11,5));c=a@b
    p=compile_graph(g,{'c':c});kernel=p['kernels'][0]
    assert kernel['dispatch']==[1,1,1]
    assert 'workgroupBarrier' in kernel['wgsl']
    assert 'return;' not in kernel['wgsl']
    assert 'row < 3u' in kernel['wgsl'] and 'col < 5u' in kernel['wgsl']


def test_invalid_options_and_labels():
    g,outputs,updates,bindings=fixture()
    with pytest.raises(ValueError): compile_graph(g,outputs,{'max_fusion':100})
    with pytest.raises(ValueError): compile_graph(g,outputs,{'max_allocation_bytes':1})
    p=compile_graph(g,outputs,updates=updates)
    bindings['y'][0]=3
    with pytest.raises(ValueError,match='labels'): execute_plan(p,bindings)


def test_integer_layouts_preserve_label_validation():
    g=Graph();labels=g.input('labels',(1,3),dtype='int32');x=g.input('x',(3,2))
    flat=labels.T.reshape((3,));loss=cross_entropy(x,flat)
    p=compile_graph(g,{'loss':loss},DISABLED)
    bindings={'labels':np.array([[0,1,0]],np.int32),'x':np.arange(6,dtype='float32').reshape(3,2)}
    np.testing.assert_allclose(execute_plan(p,bindings)['loss'],evaluate([loss],bindings)[loss.id])
    assert 'array<i32>' in p['kernels'][0]['wgsl']
    bindings['labels'][0,0]=2
    with pytest.raises(ValueError,match='labels'):execute_plan(p,bindings)


def test_invalid_recorded_shape_and_work_limits_are_rejected():
    g=Graph();x=g.input('x',(3,));bad=x.exp();records=g.to_dict();records['nodes'][bad.id]['shape']=[2]
    class InvalidGraph:
        def to_dict(self):return records
    with pytest.raises(ValueError,match='recorded shape'):compile_graph(InvalidGraph(),{'bad':bad.id})
    g=Graph();x=g.input('x',(4,));out=x.softmax()
    with pytest.raises(ValueError,match='work cap'):compile_graph(g,{'out':out},{'max_kernel_work':3})
    with pytest.raises(ValueError,match='boolean'):compile_graph(g,{'out':out},{'reuse':'no'})


def test_output_identity_and_names_are_unambiguous():
    g=Graph();a=g.parameter('a',(3,));other=Graph().input('x',(3,))
    with pytest.raises(ValueError,match='distinct'):compile_graph(g,{'a':a},updates={'a':a})
    with pytest.raises(ValueError,match='different graph'):compile_graph(g,{'out':other})
    with pytest.raises(ValueError,match='integer node'):compile_graph(g,{'out':0.5})
    with pytest.raises(ValueError,match='nonempty'):compile_graph(g,{'':a})


def test_numpy_retained_array_cap_even_when_gpu_fusion_fits():
    g=Graph();x=g.input('x',(64,));out=x
    for _ in range(8):out=(out+.1).tanh()
    plan=compile_graph(g,{'out':out},{'max_allocation_bytes':1024})
    assert plan['stats']['allocated_bytes']<=1024
    with pytest.raises(ValueError,match='NumPy retained'):execute_plan(plan,{'x':np.zeros(64,np.float32)})


def test_constant_folding_checks_capacity_before_large_materialization():
    g=Graph();x=g.constant(np.ones(64,np.float32));out=x
    for _ in range(8):out=out.exp()
    with pytest.raises(ValueError,match='Constant folding would exceed'):compile_graph(g,{'out':out},{'max_allocation_bytes':1024})


def test_cross_entropy_translation_and_shader_target_gap_regression():
    # Large offsets are exactly representable; the loss is independent of them.
    # The GPU regression fixture additionally executes the emitted source on the
    # actual driver, where arithmetic reassociation caused a ~1.5e-4 loss error.
    base=np.array([[0,1,-1,2,-2],[3,3,3,3,3],[-10,-9,-8,-10,0]],np.float64)
    labels=np.array([1,4,0],np.int32)
    shifted=base-base.max(axis=1,keepdims=True)
    expected=np.mean(np.log(np.exp(shifted).sum(axis=1))-shifted[np.arange(3),labels])
    for offset in (0,10000,-10000,1000000):
        g=Graph();x=g.input('x',(3,5));y=g.input('labels',(3,),dtype='int32');loss=cross_entropy(x,y)
        plan=compile_graph(g,{'loss':loss})
        actual=execute_plan(plan,{'x':(base+offset).astype('float32'),'labels':labels})['loss']
        np.testing.assert_allclose(actual,expected,rtol=3e-7,atol=3e-7)
        shader=next(k['wgsl'] for k in plan['kernels'] if k['op']=='cross_entropy')
        assert 'let target_gap = max(maximum -' in shader
        assert 'loss += log(total) + target_gap' in shader
