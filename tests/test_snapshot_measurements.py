"""Measurements absent from a checkpoint remain unknown until execution resumes."""
import json
import numpy as np
import pytest

from forge_ml.model import Experiment
from forge_ml.serialization import export_experiment, import_experiment, checksum_payload


CONFIG = {'dataset': 'xor', 'samples': 40, 'hidden': [4], 'batchSize': 8}


def assert_unknown_gradient_norms(snapshot):
    assert snapshot['metrics']['gradientNorm'] is None
    assert all(p['gradientNorm'] is None for p in snapshot['parameters'])
    # Parameter norms are available from the actual restored bindings independently.
    assert all(np.isfinite(p['valueNorm']) and p['valueNorm'] >= 0 for p in snapshot['parameters'])
    roundtrip = json.loads(json.dumps(snapshot, allow_nan=False))
    assert roundtrip['metrics']['gradientNorm'] is None
    assert all(p['gradientNorm'] is None for p in roundtrip['parameters'])


def assert_measured_gradient_norms(snapshot):
    assert np.isfinite(snapshot['metrics']['gradientNorm'])
    assert snapshot['metrics']['gradientNorm'] > 0
    assert all(np.isfinite(p['gradientNorm']) and p['gradientNorm'] >= 0 for p in snapshot['parameters'])


def test_initial_snapshot_has_unknown_gradients_until_completed_step():
    exp = Experiment(CONFIG)
    initial = exp.snapshot()
    assert initial['metrics']['step'] == 0
    assert_unknown_gradient_norms(initial)
    exp.cpu_step()
    trained = exp.snapshot()
    assert trained['metrics']['step'] == 1
    assert_measured_gradient_norms(trained)


def test_restored_nonzero_step_does_not_fabricate_gradient_measurements():
    exp = Experiment(CONFIG)
    exp.cpu_step(3)
    assert_measured_gradient_norms(exp.snapshot())
    restored = import_experiment(export_experiment(exp))
    snapshot = restored.snapshot()
    assert snapshot['metrics']['step'] == 3
    assert_unknown_gradient_norms(snapshot)
    for name in exp.bindings:
        np.testing.assert_array_equal(restored.bindings[name], exp.bindings[name])
    samples = exp.data['x'][:8]
    np.testing.assert_array_equal(restored.evaluate(samples)['probabilities'], exp.evaluate(samples)['probabilities'])
    # Resuming collects fresh measurements and retains exact optimizer/RNG continuation.
    exp.cpu_step()
    restored.cpu_step()
    assert_measured_gradient_norms(restored.snapshot())
    assert restored.step == 4
    for name in exp.bindings:
        np.testing.assert_array_equal(restored.bindings[name], exp.bindings[name])
    np.testing.assert_array_equal(restored.gradient_norm, exp.gradient_norm)


def test_export_backend_identity_follows_cpu_recovery_not_stale_metadata():
    exp = Experiment({**CONFIG, 'backend': 'webgpu'})
    exp.backend_metadata = {
        'backend': 'webgpu', 'dtype': 'incorrect-old-type', 'framework': 'incorrect-old-version',
        'adapter': 'previous GPU adapter',
        'evaluation': {'step': 0, 'accuracy': .9, 'datasetChecksum': exp.data['checksum']},
    }
    restored = import_experiment(export_experiment(exp))
    # This mirrors the validated backend override used for device-loss recovery.
    restored.config['backend'] = 'cpu'
    exported = export_experiment(restored)
    metadata = exported['payload']['metadata']
    assert metadata['backend'] == exported['payload']['config']['backend'] == 'cpu'
    assert metadata['dtype'] == 'float32'
    assert metadata['framework'] == 'forge-ml 0.1.0'
    assert metadata['adapter'] == 'previous GPU adapter'
    assert metadata['evaluation'] == exp.backend_metadata['evaluation']
    assert import_experiment(exported).backend_metadata['backend'] == 'cpu'


@pytest.mark.parametrize('mutation', [
    lambda m:m.__setitem__('evaluation', None),
    lambda m:m.__setitem__('evaluation', 42),
    lambda m:m.__setitem__('evaluation', 'reported score'),
    lambda m:m.__setitem__('evaluation', []),
    lambda m:m.__setitem__('evaluation', {}),
    lambda m:m['evaluation'].__setitem__('accuracy', '0.9'),
    lambda m:m['evaluation'].__setitem__('accuracy', True),
    lambda m:m['evaluation'].__setitem__('accuracy', 1.1),
    lambda m:m['evaluation'].__setitem__('accuracy', -.1),
    lambda m:m['evaluation'].__setitem__('step', True),
    lambda m:m['evaluation'].__setitem__('step', -1),
    lambda m:m['evaluation'].__setitem__('step', 1),
    lambda m:m['evaluation'].__setitem__('datasetChecksum', '0' * 64),
    lambda m:m['evaluation'].__setitem__('split', 'training'),
    lambda m:m['evaluation'].__setitem__('rows', True),
    lambda m:m['evaluation'].__setitem__('rows', 9999),
    lambda m:m['evaluation'].__setitem__('code', 'unrecognized'),
    lambda m:m.__setitem__('backend', None),
    lambda m:m.__setitem__('dtype', 'float64'),
    lambda m:m.__setitem__('framework', 42),
])
def test_invalid_optional_metadata_cannot_replace_active_experiment(mutation, monkeypatch):
    from forge_ml import bridge
    exp = Experiment(CONFIG)
    exp.backend_metadata = {'evaluation': {'step': 0, 'accuracy': .9, 'datasetChecksum': exp.data['checksum']}}
    monkeypatch.setattr(bridge, 'experiment', exp)
    exported = export_experiment(exp)
    mutation(exported['payload']['metadata'])
    exported['sha256'] = checksum_payload(exported['payload'])
    with pytest.raises(ValueError, match='metadata|evaluation'):
        bridge.handle('import', {'experiment': exported})
    assert bridge.experiment is exp
    assert exp.step == 0


def test_historical_evaluation_metadata_can_survive_resumed_training():
    exp = Experiment(CONFIG)
    exp.backend_metadata = {'evaluation': {
        'step': 0, 'accuracy': .5, 'datasetChecksum': exp.data['checksum'],
        'split': 'fixed held-out test', 'rows': len(exp.data['splits']['test']),
    }}
    exp.cpu_step(2)
    restored = import_experiment(export_experiment(exp))
    assert restored.step == 2
    assert restored.backend_metadata['evaluation']['step'] == 0
