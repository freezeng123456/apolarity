import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

import bscc_sweep as sweep
from torch_pinn import records
from train_fixed_wall import parse_args, sample


def test_selection_excludes_failed_candidate_and_breaks_ties_by_lower_lr():
    rows = [dict(eligible=False, score=0., lr=1e-5),
            dict(eligible=True, score=.1, lr=1e-3),
            dict(eligible=True, score=.1, lr=1e-4)]
    assert sweep.pick_candidate(rows)['lr'] == 1e-4
    with pytest.raises(RuntimeError):
        sweep.pick_candidate([rows[0]])


def test_selection_balances_terminal_and_spacetime_error_and_rejects_nonfinite():
    def result(a, b):
        return dict(errors=dict(spacetime=dict(relative_l2=a), terminal=dict(relative_l2=b)))
    assert sweep.validation_score(result(.1, .3)) == pytest.approx(np.sqrt(.05))
    assert sweep.validation_score(result(.01, .5)) > sweep.validation_score(result(.2, .2))
    for value in (float('nan'), float('inf'), -1.):
        with pytest.raises(ValueError):
            sweep.validation_score(result(.1, value))


def test_validation_test_and_training_streams_are_separate():
    for case in sweep.PLAN['cases']:
        streams = [sample(np.random.default_rng(seed), 400, case) for seed in
                   (sweep.PLAN['screen_seed'] + 1, sweep.PLAN['screen_eval_seed_base'] + 1,
                    sweep.PLAN['confirmation_eval_seed_base'] + 1)]
        for i in range(3):
            for j in range(i):
                assert not np.array_equal(streams[i], streams[j])
    assert sweep.PLAN['screen_seed'] != sweep.PLAN['confirmation_seed']
    cfgs = [sweep.screen_config(i) for i in range(12)]
    assert [c['case'] for c in cfgs[:3]] == sweep.PLAN['cases']
    assert len({(c['case'], c['lr']) for c in cfgs}) == 12
    assert all(c['expected_device'] == 'V100' and c['depth'] == 5 for c in cfgs)


def test_t4_remains_default_but_v100_requires_explicit_selection(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_name', lambda device: 'Tesla V100-SXM2-32GB')
    with pytest.raises(RuntimeError, match='requires T4'):
        records.configure('cuda')
    assert records.configure('cuda', 'V100').type == 'cuda'
    monkeypatch.setattr(sys, 'argv', ['train_fixed_wall.py', '--out', '/unused', '--case', 'kdv1d', '--method', 'nested_jvp'])
    args = parse_args()
    assert args.expected_device == 'T4' and args.eval_seed_base == 20261100


def test_verified_source_snapshot_rejects_modified_or_unlisted_code(tmp_path, monkeypatch):
    here = tmp_path / 'experiments/torch_benchmarks'; here.mkdir(parents=True)
    code = here / 'example.py'; code.write_text('value = 1\n')
    relative = str(code.relative_to(tmp_path))
    manifest = dict(source_commit='frozen-commit', files={relative: hashlib.sha256(code.read_bytes()).hexdigest()})
    (tmp_path / 'SOURCE_MANIFEST.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(records, 'HERE', here)
    assert records.source_identity() == ('frozen-commit', 'sha256_verified_export')
    code.write_text('value = 2\n')
    with pytest.raises(RuntimeError, match='mismatch'):
        records.source_identity()
    code.write_text('value = 1\n'); (here / 'unexpected.py').write_text('pass\n')
    with pytest.raises(RuntimeError, match='inventory'):
        records.source_identity()


def test_result_seal_detects_unsealed_extra_artifact(tmp_path):
    (tmp_path / 'summary.json').write_text('{"status":"COMPLETE"}\n')
    records.seal(tmp_path)
    assert sweep.verify_seal(tmp_path) == 1
    (tmp_path / 'unexpected.pt').write_bytes(b'unsealed')
    with pytest.raises(ValueError, match='inventory'):
        sweep.verify_seal(tmp_path)


def test_queued_campaign_requires_successful_preflight_and_bounded_arrays(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep, 'source_identity', lambda: ('frozen', 'git'))
    (tmp_path / 'submission_preflight.json').write_text(json.dumps(dict(job_id='123', source_commit='frozen', plan_sha256=sweep.sha(sweep.PLAN_PATH))))
    commands = []
    def submit(command, **kwargs):
        commands.append(command)
        return str(123 + len(commands)) + '\n'
    monkeypatch.setattr(sweep.subprocess, 'check_output', submit)
    sweep.submit(tmp_path, 'campaign')
    assert commands[0][commands[0].index('--array') + 1] == '0-11%3'
    assert commands[0][commands[0].index('--dependency') + 1] == 'afterok:123'
    assert commands[1][commands[1].index('--array') + 1] == '0-2%3'
    assert commands[1][commands[1].index('--dependency') + 1] == 'afterok:123,afterany:124'
    assert all('--kill-on-invalid-dep=yes' in c for c in commands)
    with pytest.raises(RuntimeError, match='already submitted'):
        sweep.submit(tmp_path, 'campaign')
    assert len(commands) == 2
