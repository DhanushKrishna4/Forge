"""Benchmark timings retain quantized totals and commit every measured step."""
import numpy as np
import pytest
from forge_ml import benchmark,bridge
from forge_ml.graph import Graph
from forge_ml.compiler import compile_graph


def test_amortized_samples_preserve_totals_and_commit_each_step(monkeypatch):
    graph=Graph();parameter=graph.parameter('p',());updated=parameter+1
    plan=compile_graph(graph,{'result':updated},updates={'p':updated})
    case={'name':'counter','plan':plan,'configuration':'optimized','bindings':{'p':[0]},'compilationMs':0,'shapes':{'p':[]}}
    monkeypatch.setattr(benchmark,'benchmark_plans',lambda:[case])
    # A deliberately coarse clock still supplies useful per-execution averages
    # when one sample covers multiple completed executions.
    ticks=iter([0,.01,.02,.04,.1,.16])
    monkeypatch.setattr(benchmark.time,'perf_counter',lambda:next(ticks))
    execute=benchmark.execute_plan;seen=[]
    def record(plan,bindings):
        seen.append(float(bindings['p']))
        return execute(plan,bindings)
    monkeypatch.setattr(benchmark,'execute_plan',record)
    report=benchmark.run_benchmarks(3,iterations_per_sample=20)
    row=report['rows'][0]
    assert seen==list(range(63))  # Three individual warmups, then 3 × 20 steps.
    assert row['iterationsPerSample']==20 and row['repetitions']==3
    np.testing.assert_allclose(row['totalSamplesMs'],[10,20,60])
    np.testing.assert_allclose(row['samplesMs'],[.5,1,3])
    assert row['medianMs']==pytest.approx(1)
    assert row['minMs']==pytest.approx(.5) and row['maxMs']==pytest.approx(3)
    assert row['stdMs']==pytest.approx(np.std([.5,1,3]))
    assert '3 individual warmups' in report['timingMethod']


@pytest.mark.parametrize('iterations',[0,101,1.5,True])
def test_iterations_are_bounded_integers(iterations):
    with pytest.raises(ValueError,match='iterations per sample'):
        benchmark.run_benchmarks(3,iterations_per_sample=iterations)


def test_browser_uses_explicit_batched_measurement(monkeypatch):
    requests=[]
    def measured(repetitions,iterations_per_sample=1):
        requests.append((repetitions,iterations_per_sample))
        return {'rows':[]}
    monkeypatch.setattr(benchmark,'run_benchmarks',measured)
    assert bridge.handle('benchmark',{'repetitions':7})=={'rows':[]}
    assert requests==[(7,20)]
