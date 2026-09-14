import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_fixed_wall import train
from plot_training_trajectories import log_interpolate,seed_stats


def test_dense_capture_preserves_paired_updates_and_saved_states(tmp_path):
    torch.set_num_threads(1)
    for mode,interval in [('off',0),('dense',1e-9)]:
        out=tmp_path/mode;out.mkdir()
        args=SimpleNamespace(out=out,case='kdv1d',method='shared_jet_linear',device='cpu',
            width=8,depth=3,seed=51,lr=1e-3,eval_seed_base=20261400,eval_points=32,
            constraint_side=4,batch=4,seconds=30,checkpoint_seconds=100,
            trajectory_seconds=interval,max_steps=3)
        train(args)
    load=lambda mode,name:json.loads((tmp_path/mode/name).read_text())
    assert load('off','summary.json')['final_hash']==load('dense','summary.json')['final_hash']
    assert load('off','batch_hashes.json')==load('dense','batch_hashes.json')
    points=load('dense','trajectory_checkpoints.json')
    assert [p['step'] for p in points]==[0,1,2,3]
    for p in points:
        state=torch.load(tmp_path/'dense'/p['file'],map_location='cpu',weights_only=False)
        assert state['step']==p['step']
    assert 'optimizer' not in torch.load(tmp_path/'dense'/points[1]['file'],weights_only=False)
    assert 'optimizer' in torch.load(tmp_path/'dense'/points[-1]['file'],weights_only=False)


def test_plot_does_not_extrapolate_or_invent_single_seed_band():
    with pytest.raises(ValueError,match='extrapolation'):
        log_interpolate([0,10],[1,.1],[-1,5,10])
    np.testing.assert_allclose(log_interpolate([0,10],[1,.01],[0,5,10]),[1,.1,.01])
    mean,std=seed_stats([[1,2,3]])
    assert std is None
    mean,std=seed_stats([[1,2],[3,4]])
    np.testing.assert_allclose(mean,[2,3]);np.testing.assert_allclose(std,[2**.5,2**.5])
