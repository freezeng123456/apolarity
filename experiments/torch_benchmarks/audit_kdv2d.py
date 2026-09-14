"""Read-only raw-result audit with independent CPU replay of saved full losses.

The report must live outside the sealed input directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch_pinn.model import MLP
from torch_pinn.problem_kdv2d import constraints, exact, loss, sample_numpy
from torch_pinn.records import digest, save


def read(path): return json.loads(path.read_text())


def check_manifest(root):
    root=root.resolve()
    entries=set()
    for line in (root/'SHA256SUMS').read_text().splitlines():
        expected,rel=line.split('  ',1)
        path=(root/rel).resolve()
        if not path.is_relative_to(root): raise AssertionError('manifest traversal')
        assert path not in entries
        entries.add(path)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected, rel
    actual={p.resolve() for p in root.rglob('*') if p.is_file() and p.name!='SHA256SUMS'}
    assert entries==actual, 'manifest file coverage'
    return len(entries)


def finite_tree(value):
    if torch.is_tensor(value): assert bool(torch.isfinite(value).all())
    elif isinstance(value,dict):
        for v in value.values(): finite_tree(v)
    elif isinstance(value,(list,tuple)):
        for v in value: finite_tree(v)


def audit(root):
    torch.set_num_threads(1)
    count=check_manifest(root)
    manifest=read(root/'manifest.json')
    assert (root/'status.txt').read_text().strip()=='COMPLETE'
    assert all(r['exit_code']==0 for r in read(root/'progress.json'))
    assert read(root/'acceptance.json')['gate']=='PASS'
    cells=[]
    for seed in manifest['seeds']:
        paired=[]
        for method in ('nested_jvp','shared_jet_linear'):
            directory=root/f'{method}_{seed}'
            check_manifest(directory)
            cfg=read(directory/'config.json')
            assert cfg['source_commit']==manifest['source_commit']
            assert all(cfg['source_sha256'][k]==v for k,v in manifest['source_sha256'].items())
            summary=read(directory/'summary.json')
            rows=read(directory/'validation.json')
            metrics=read(directory/'metrics.json')
            hashes=read(directory/'batch_hashes.json')
            assert len(metrics)==len(hashes)==cfg['steps']==manifest['steps']
            assert [r['step'] for r in metrics]==list(range(1,cfg['steps']+1))
            assert [r['step'] for r in rows]==[0]+[i for i in range(1,cfg['steps']+1) if i%cfg['eval_every']==0 or i==cfg['steps']]
            assert all(np.isfinite(list(r.values())).all() for r in metrics)
            rng=np.random.default_rng(seed+1)
            for i,(m,h) in enumerate(zip(metrics,hashes)):
                pts=torch.as_tensor(sample_numpy(rng,cfg['batch']))
                assert digest((pts,))==h
                assert m['lr_used']==cfg['lr']*(1-i/cfg['decay_steps'])
                assert m['next_lr']==cfg['lr']*(1-(i+1)/cfg['decay_steps'])
            points=np.load(directory/'evaluation_points.npz')
            fixed=torch.from_numpy(points['loss_points'])
            data=constraints(cfg['constraint_side'])
            assert digest((fixed,))==summary['evaluation_point_hash']
            assert digest(data.values())==summary['constraint_hash']
            for k,v in data.items(): assert np.array_equal(v.numpy(),points[k])
            model=MLP(3,cfg['width'],cfg['depth'],seed+2)
            assert digest(model.parameters())==summary['initial_parameter_hash']
            replay=[]
            for r in rows:
                step=r['step']
                name='initial.pt' if step==0 else ('final.pt' if step==cfg['steps'] else f'checkpoint_{step:06d}.pt')
                state=torch.load(directory/name,map_location='cpu',weights_only=False)
                finite_tree(state)
                assert state['step']==step
                assert state['optimizer']['param_groups'][0]['lr']==cfg['lr']*(1-step/cfg['decay_steps'])
                for item in state['optimizer']['state'].values(): assert int(item['step'])==step
                model.load_state_dict(state['model'])
                if step==0: assert digest(model.parameters())==summary['initial_parameter_hash']
                with torch.no_grad():
                    value,terms=loss(model,fixed,data,'nested_jvp',independent=True)
                np.testing.assert_allclose(float(value),r['fixed_loss'],rtol=2e-4,atol=2e-7)
                for k,v in terms.items(): np.testing.assert_allclose(float(v),r['fixed_parts'][k],rtol=2e-4,atol=2e-7)
                replay.append(dict(step=step,recorded=r['fixed_loss'],cpu=float(value),absolute_difference=abs(float(value)-r['fixed_loss'])))
            assert state['sampler_state']==rng.bit_generator.state
            assert digest(model.parameters())==summary['final_parameter_hash']!=summary['initial_parameter_hash']
            saved=np.load(directory/'predictions.npz')
            xyzt=torch.from_numpy(saved['points'])
            with torch.no_grad(): pred=torch.cat([model(p) for p in xyzt.split(4096)]).numpy()
            assert np.isfinite(saved['prediction']).all()
            np.testing.assert_allclose(pred,saved['prediction'],rtol=2e-4,atol=2e-6)
            np.testing.assert_allclose(exact(xyzt).numpy(),saved['target'],rtol=2e-6,atol=2e-7)
            error=float(np.linalg.norm(saved['prediction']-saved['target'])/np.linalg.norm(saved['target']))
            np.testing.assert_allclose(error,summary['test']['relative_l2'],rtol=2e-6,atol=1e-9)
            cells.append(dict(method=method,seed=seed,summary=summary,replay=replay))
            paired.append((summary,hashes))
        assert paired[0][1]==paired[1][1]
        for k in ('initial_parameter_hash','constraint_hash','evaluation_point_hash'):
            assert paired[0][0][k]==paired[1][0][k]
    return dict(audit='PASS',payload_files=count,cells=cells,
                replayed_losses=sum(len(c['replay']) for c in cells),
                max_cpu_gpu_loss_absolute_difference=max(r['absolute_difference'] for c in cells for r in c['replay']))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    if a.report.resolve().is_relative_to(a.root.resolve()): p.error('report must be outside sealed input')
    a.report.parent.mkdir(parents=True,exist_ok=True)
    report=audit(a.root)
    save(a.report,report)
    print(json.dumps({k:v for k,v in report.items() if k!='cells'},indent=2))
