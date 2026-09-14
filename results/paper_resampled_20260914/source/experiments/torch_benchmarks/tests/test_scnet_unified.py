from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scnet_unified import config, PLAN
from torch_pinn.model import MLP


def test_formal_matrix_preserves_architecture_and_dense_pair_protocol():
    for case in PLAN['cases']:
        for seed in PLAN['seeds']:
            a=config(case,PLAN['methods'][0],seed)
            b=config(case,PLAN['methods'][1],seed)
            assert {k:v for k,v in a.items() if k!='method'}=={k:v for k,v in b.items() if k!='method'}
            assert a['lr']==1e-4 and a['depth']==5
            assert len(MLP(2 if case=='kdv1d' else 3,128,a['depth']).layers)==5
            assert a['trajectory_seconds']==2 and a['max_steps']==0
            assert a['seconds']==1200 and a['eval_points']==10000
