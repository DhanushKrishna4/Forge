"""Installed-style CLI asset resolution and short bounded training."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from forge_ml import cli

ROOT=Path(__file__).resolve().parents[1]


def invoke(tmp_path,*args):
    env={**os.environ,'PYTHONPATH':str(ROOT/'src')}
    return subprocess.run([sys.executable,'-m','forge_ml.cli',*args],cwd=tmp_path,env=env,capture_output=True,text=True)


def test_explicit_missing_digits_asset_fails_clearly_without_training(tmp_path):
    output=tmp_path/'output.json'
    result=invoke(tmp_path,'train','--dataset','digits','--steps','10','--digits-asset',str(tmp_path/'missing.json'),'--output',str(output))
    assert result.returncode==2
    assert 'Digits asset not found' in result.stderr and '--digits-asset' in result.stderr
    assert not output.exists()


def test_explicit_digits_asset_trains_from_another_working_directory(tmp_path):
    output=tmp_path/'outputs'/'experiment.json'
    asset=ROOT/'web/public/assets/digits.json'
    result=invoke(tmp_path,'train','--dataset','digits','--steps','10','--digits-asset',str(asset),'--output',str(output))
    assert result.returncode==0,result.stderr
    checkpoint=json.loads(output.read_text());metrics=json.loads(output.with_suffix('.metrics.json').read_text())
    assert checkpoint['payload']['step']==10 and checkpoint['payload']['config']['dataset']=='digits'
    assert metrics['test']['rows']==279 and 0<=metrics['test']['accuracy']<=1
    assert metrics['datasetChecksum']==json.loads(asset.read_text())['checksum']


def test_installed_default_missing_asset_gives_actionable_error(monkeypatch,tmp_path,capsys):
    monkeypatch.setattr(cli,'__file__',str(tmp_path/'venv/lib/forge_ml/cli.py'))
    monkeypatch.setattr(sys,'argv',['forge','train','--dataset','digits','--steps','10'])
    with pytest.raises(SystemExit) as error:cli.main()
    assert error.value.code==2
    assert '--digits-asset /path/to/prepared/digits.json' in capsys.readouterr().err


def test_digits_asset_is_rejected_for_an_unrelated_dataset(tmp_path):
    result=invoke(tmp_path,'train','--dataset','moons','--digits-asset',str(ROOT/'web/public/assets/digits.json'))
    assert result.returncode==2 and 'only used with --dataset digits' in result.stderr
