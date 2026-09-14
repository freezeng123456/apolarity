"""Bounded V100 learning-rate screening, followed by independent-seed pairs."""
import argparse
import hashlib
import fcntl
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

from torch_pinn.model import MLP
from torch_pinn.records import configure, digest, save, seal, source_identity
from train_fixed_wall import PROBLEMS, full_loss, sample

HERE = Path(__file__).resolve().parent
PLAN_PATH = HERE / 'bscc_sweep_plan.json'
PLAN = json.loads(PLAN_PATH.read_text())
METHODS = PLAN['confirmation_methods']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validation_score(summary):
    values = [summary['errors'][name]['relative_l2'] for name in ('spacetime', 'terminal')]
    if not all(math.isfinite(x) and x >= 0 for x in values):
        raise ValueError('nonfinite or negative validation error')
    return math.hypot(*values) / math.sqrt(2)


def pick_candidate(rows):
    eligible = [r for r in rows if r.get('eligible')]
    if not eligible:
        raise RuntimeError('no complete finite validated candidate')
    return min(eligible, key=lambda r: (r['score'], r['lr']))


def cell_config(case, method, lr, seed, eval_base):
    return dict(case=case, method=method, lr=lr, seed=seed,
                eval_seed_base=eval_base, width=PLAN['width'],
                depth=PLAN['hidden_layers'] + 1, batch=PLAN['batch'],
                seconds=PLAN['seconds'], expected_device=PLAN['expected_device'])


def screen_config(index):
    if not 0 <= index < 12:
        raise ValueError('screen index must be 0..11')
    case = PLAN['cases'][index % 3]
    lr = PLAN['learning_rates'][case][index // 3]
    return cell_config(case, PLAN['screen_method'], lr, PLAN['screen_seed'],
                       PLAN['screen_eval_seed_base'])


def verify_runtime():
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('GPU worker must be launched through Slurm')
    commit, source_kind = source_identity()
    configure('cuda', PLAN['expected_device'])
    versions = dict(torch=torch.__version__, numpy=np.__version__,
                    python_major_minor=list(sys.version_info[:2]))
    if versions != PLAN['runtime']:
        raise RuntimeError(f'unexpected runtime: {versions}')
    return dict(source_commit=commit, source_kind=source_kind,
                plan_sha256=sha(PLAN_PATH), runtime=versions,
                python=sys.executable, hostname=platform.node(),
                gpu=torch.cuda.get_device_name(0),
                cuda_runtime=torch.version.cuda,
                visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                job_id=os.environ['SLURM_JOB_ID'],
                array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'))


def verify_seal(path):
    lines = (path / 'SHA256SUMS').read_text().splitlines()
    listed = set()
    for line in lines:
        expected, name = line.split('  ', 1)
        target = path / name
        if not target.resolve().is_relative_to(path.resolve()):
            raise ValueError('invalid artifact path')
        if sha(target) != expected:
            raise ValueError(f'checksum mismatch: {name}')
        listed.add(name)
    actual = {str(f.relative_to(path)) for f in path.rglob('*')
              if f.is_file() and f.name != 'SHA256SUMS'}
    if actual != listed:
        raise ValueError('unsealed artifact inventory')
    return len(listed)


def validate_cell(path, expected, commit, smoke=False):
    if (path / 'status.txt').read_text().strip() != 'COMPLETE':
        raise ValueError('cell is not COMPLETE')
    count = verify_seal(path)
    config, summary = read(path / 'config.json'), read(path / 'summary.json')
    for key, value in expected.items():
        if config[key] != value:
            raise ValueError(f'configuration mismatch: {key}')
    if config['source_commit'] != commit:
        raise ValueError('source commit mismatch')
    if not smoke and (summary['terminal'] != 'wall_time_budget' or
                      summary['wall_seconds'] < PLAN['seconds']):
        raise ValueError('incomplete training budget')
    if not math.isfinite(summary['final_fixed_loss']):
        raise ValueError('nonfinite final loss')
    validation_score(summary)
    curves = read(path / 'loss_curve.json')
    if not all(math.isfinite(row['fixed_loss']) for row in curves):
        raise ValueError('nonfinite loss assessment')
    state = torch.load(path / curves[-1]['file'], map_location='cpu', weights_only=False)
    model = MLP(2 if config['case'] == 'kdv1d' else 3, config['width'], config['depth'])
    model.load_state_dict(state['model'])
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('nonfinite final parameters')
    if digest(model.parameters()) != summary['final_hash'] or summary['final_hash'] == summary['initial_hash']:
        raise ValueError('incorrect or unchanged final parameters')
    rows = [json.loads(line) for line in (path / 'steps.jsonl').read_text().splitlines()]
    if len(rows) != summary['steps'] or state['step'] != summary['steps']:
        raise ValueError('step records mismatch')
    if not all(row['lr'] == expected['lr'] and math.isfinite(row['loss']) for row in rows):
        raise ValueError('invalid training records')
    with np.load(path / 'predictions.npz') as z, np.load(path / 'evaluation_points.npz') as pts:
        for label in ('spacetime', 'terminal', 'extra'):
            prediction, target = z[label + '_prediction'], z[label + '_target']
            if not np.isfinite(prediction).all():
                raise ValueError('nonfinite predictions')
            reference = PROBLEMS[config['case']].exact(torch.from_numpy(pts[label])).numpy()
            np.testing.assert_allclose(target, reference, rtol=1e-5, atol=2e-7)
            error = np.linalg.norm(prediction - target) / np.linalg.norm(target)
            if not np.isclose(error, summary['errors'][label]['relative_l2'], rtol=2e-5):
                raise ValueError('saved prediction error mismatch')
    return dict(summary=summary, hashed_files=count)


def launch_cell(path, config, smoke=False):
    cmd = [sys.executable, str(HERE / 'train_fixed_wall.py'), '--out', str(path)]
    for key, value in config.items():
        cmd += ['--' + key.replace('_', '-'), str(value)]
    cmd += ['--eval-points', str(32 if smoke else PLAN['eval_points']),
            '--checkpoint-seconds', str(1200 if smoke else PLAN['checkpoint_seconds'])]
    if smoke:
        cmd += ['--max-steps', '2']
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix('.log').open('w') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                       timeout=PLAN['child_timeout_seconds'], check=True)


def preflight(root):
    out = root / 'preflight'
    out.mkdir(exist_ok=False)
    environment = verify_runtime()
    save(out / 'environment.json', environment)
    for name, cmd in [('tests.log', [sys.executable, '-m', 'pytest', str(HERE / 'tests'), '-q']),
                      ('requirements.txt', [sys.executable, '-m', 'pip', 'freeze']),
                      ('gpu.txt', ['nvidia-smi'])]:
        with (out / name).open('w') as stream:
            subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=300)
    gates = []
    for case in PLAN['cases']:
        module = PROBLEMS[case]
        dim = 2 if case == 'kdv1d' else 3
        model = MLP(dim, PLAN['width'], PLAN['hidden_layers'] + 1, PLAN['screen_seed'] + 2, device='cuda')
        data = module.constraints(128 if dim == 2 else 16, device='cuda')
        points = torch.as_tensor(sample(np.random.default_rng(PLAN['screen_seed'] + 1), PLAN['batch'], case), device='cuda')
        values, gradients = [], []
        for method in METHODS:
            model.zero_grad(set_to_none=True)
            value, _ = full_loss(module, model, points, data, method)
            value.backward()
            values.append(value.detach())
            gradients.append(torch.cat([p.grad.flatten() for p in model.parameters()]).detach().clone())
        torch.testing.assert_close(values[0], values[1], rtol=1e-3, atol=1e-5)
        torch.testing.assert_close(gradients[0], gradients[1], rtol=1e-3, atol=1e-5)
        gate = dict(case=case, loss_absolute_difference=float(abs(values[0]-values[1])),
                    gradient_max_absolute_difference=float((gradients[0]-gradients[1]).abs().max()),
                    gradient_relative_l2_difference=float(torch.linalg.vector_norm(gradients[0]-gradients[1]) / torch.linalg.vector_norm(gradients[0])),
                    resumed=[])
        del model, gradients, values, data, points
        for method in METHODS:
            cfg = cell_config(case, method, 1e-4, PLAN['screen_seed'], PLAN['screen_eval_seed_base'])
            path = out / (case + '_' + method)
            launch_cell(path, cfg, smoke=True)
            validate_cell(path, cfg, environment['source_commit'], smoke=True)
            cp = read(path / 'checkpoints.json')[-1]
            state = torch.load(path / cp['file'], map_location='cuda', weights_only=False)
            resumed = MLP(dim, PLAN['width'], PLAN['hidden_layers'] + 1, device='cuda')
            resumed.load_state_dict(state['model'])
            opt = torch.optim.Adam(resumed.parameters(), lr=1e-4, foreach=False)
            opt.load_state_dict(state['optimizer'])
            rng = np.random.default_rng(); rng.bit_generator.state = state['sampler_state']
            p = torch.as_tensor(sample(rng, PLAN['batch'], case), device='cuda')
            data = module.constraints(128 if dim == 2 else 16, device='cuda')
            before = digest(resumed.parameters())
            opt.zero_grad(set_to_none=True)
            value, _ = full_loss(module, resumed, p, data, method)
            value.backward()
            if not bool(torch.isfinite(value)) or not all(bool(torch.isfinite(x.grad).all()) for x in resumed.parameters()):
                raise ValueError('nonfinite resumed update')
            opt.step()
            after = digest(resumed.parameters())
            if before == after or not all(bool(torch.isfinite(x).all()) for x in resumed.parameters()):
                raise ValueError('invalid resumed parameters')
            steps = {int(x['step'].item()) for x in opt.state.values()}
            if steps != {3}:
                raise ValueError('optimizer resume step mismatch')
            torch.save(dict(model=resumed.state_dict(), optimizer=opt.state_dict(), step=3,
                            sampler_state=rng.bit_generator.state), out / f'{case}_{method}_resumed.pt')
            gate['resumed'].append(dict(method=method, previous_step=2, step=3,
                                        loss=float(value.detach()), initial_hash=before, final_hash=after))
            del resumed, opt, state, p, data, value
        gates.append(gate)
    save(out / 'PASS.json', dict(status='PASS', **environment, gates=gates))
    seal(out)
    print(json.dumps(dict(status='PASS', job_id=environment['job_id']), indent=2), flush=True)


def require_preflight(root, environment=None):
    gate = read(root / 'preflight/PASS.json')
    commit, _ = source_identity()
    if gate['status'] != 'PASS' or gate['source_commit'] != commit or gate['plan_sha256'] != sha(PLAN_PATH):
        raise ValueError('preflight does not match current source/configuration')
    verify_seal(root / 'preflight')
    if environment and gate['runtime'] != environment['runtime']:
        raise ValueError('worker environment differs from preflight')
    return gate


def screen(root, index):
    environment = verify_runtime(); require_preflight(root, environment)
    cfg = screen_config(index)
    path = root / 'screening' / f'{index:02d}_{cfg["case"]}'
    save(root / 'state' / f'screen_{index:02d}.json', dict(stage='TRAINING', config=cfg, **environment))
    launch_cell(path, cfg)
    checked = validate_cell(path, cfg, environment['source_commit'])
    save(root / 'state' / f'screen_{index:02d}.json',
         dict(stage='COMPLETE', config=cfg, score=validation_score(checked['summary']),
              **environment, **checked))


def select(root, case, commit):
    rows = []
    for i in range(12):
        cfg = screen_config(i)
        if cfg['case'] != case:
            continue
        row = dict(index=i, lr=cfg['lr'], eligible=False)
        try:
            checked = validate_cell(root / 'screening' / f'{i:02d}_{case}', cfg, commit)
            row.update(eligible=True, score=validation_score(checked['summary']), **checked)
        except (ValueError, OSError, KeyError, AssertionError, RuntimeError) as exc:
            row['reason'] = str(exc)
        rows.append(row)
    return pick_candidate(rows), rows


def confirm(root, index):
    if not 0 <= index < 3:
        raise ValueError('confirmation index must be 0..2')
    environment = verify_runtime(); require_preflight(root, environment)
    case = PLAN['cases'][index]
    winner, rows = select(root, case, environment['source_commit'])
    save(root / 'selection' / f'{case}.json', dict(case=case, selected=winner, candidates=rows,
         selection=PLAN['selection'], test_data_used_for_selection=False, **environment))
    out = root / 'confirmation' / case
    out.mkdir(exist_ok=False)
    order = METHODS if index % 2 == 0 else list(reversed(METHODS))
    summaries = {}
    for method in order:
        cfg = cell_config(case, method, winner['lr'], PLAN['confirmation_seed'],
                          PLAN['confirmation_eval_seed_base'])
        save(root / 'state' / f'confirm_{index}.json', dict(stage='TRAINING', method=method, selected_lr=winner['lr'], **environment))
        launch_cell(out / method, cfg)
        summaries[method] = validate_cell(out / method, cfg, environment['source_commit'])['summary']
    left, right = [out / method for method in METHODS]
    a, b = [summaries[method] for method in METHODS]
    if a['initial_hash'] != b['initial_hash']:
        raise ValueError('paired initialization mismatch')
    hashes = [read(p / 'batch_hashes.json') for p in (left, right)]
    n = min(map(len, hashes))
    if hashes[0][:n] != hashes[1][:n]:
        raise ValueError('paired sampling prefix mismatch')
    with np.load(left / 'evaluation_points.npz') as x, np.load(right / 'evaluation_points.npz') as y:
        if x.files != y.files or not all(np.array_equal(x[k], y[k]) for k in x.files):
            raise ValueError('paired evaluation points mismatch')
    result = dict(status='COMPLETE', case=case, selected_lr=winner['lr'],
                  seed=PLAN['confirmation_seed'], evaluation_seed_base=PLAN['confirmation_eval_seed_base'],
                  matched_batch_prefix=n, update_count_ratio=b['steps']/a['steps'],
                  throughput_ratio=b['steps_per_second']/a['steps_per_second'],
                  methods=summaries, **environment)
    save(out / 'paired_summary.json', result)
    seal(out)
    save(root / 'state' / f'confirm_{index}.json', result)
    report(root)


def report(root):
    with (root / '.report.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        _report_locked(root)


def _report_locked(root):
    states = [read(p) for p in sorted((root / 'state').glob('*.json'))]
    selected = {p.stem: read(p) for p in sorted((root / 'selection').glob('*.json'))}
    complete = [s for s in states if s.get('status') == 'COMPLETE']
    result = dict(status='COMPLETE' if len(complete) == 3 else 'PARTIAL',
                  complete_pairs=len(complete), states=states, selected=selected,
                  source_commit=source_identity()[0], plan_sha256=sha(PLAN_PATH))
    # Each writer has a unique temporary file; final reports are atomic snapshots.
    temp = root / f'.campaign_summary.{os.getpid()}.tmp'
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temp.replace(root / 'campaign_summary.json')
    print(json.dumps(dict(status=result['status'], complete_pairs=len(complete)), indent=2), flush=True)


def submit(root, phase):
    source_identity()
    for name in ('logs', 'state', 'selection', 'screening', 'confirmation'):
        (root / name).mkdir(parents=True, exist_ok=True)
    script = HERE / 'bscc_job.sh'
    def sbatch(mode, options):
        path = root / f'submission_{mode}.json'
        if path.exists():
            raise RuntimeError(f'{mode} already submitted; inspect {path}')
        cmd = ['sbatch', '--parsable', '--partition', PLAN['partition'], '--gpus=1',
               '--job-name', 'apol-' + mode, '--output', str(root / 'logs' / (mode + '-%A_%a.log')),
               *options, str(script), str(root), mode]
        output = subprocess.check_output(cmd, text=True).strip()
        job = output.split(';', 1)[0]
        if not job.isdigit():
            raise RuntimeError(f'unrecognized submission response: {output}')
        save(path, dict(job_id=job, command=cmd, submitted_at=time.time(),
                        source_commit=source_identity()[0], plan_sha256=sha(PLAN_PATH)))
        print(mode, job, flush=True)
        return job
    if phase == 'preflight':
        sbatch('preflight', ['--time', PLAN['preflight_time_limit']])
    else:
        # A scheduler success dependency plus the worker's sealed PASS gate
        # permits all arrays to queue before a GPU becomes available.
        pre = read(root / 'submission_preflight.json')
        if pre['source_commit'] != source_identity()[0] or pre['plan_sha256'] != sha(PLAN_PATH):
            raise ValueError('preflight submission belongs to another source/configuration')
        pre_id = pre['job_id']
        job = sbatch('screen', ['--time', PLAN['screen_time_limit'], '--array', '0-11%3',
                     '--dependency', 'afterok:' + pre_id, '--kill-on-invalid-dep=yes'])
        sbatch('confirm', ['--time', PLAN['confirmation_time_limit'], '--array', '0-2%3',
                          '--dependency', 'afterok:' + pre_id + ',afterany:' + job,
                          '--kill-on-invalid-dep=yes'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('preflight', 'screen', 'confirm', 'report', 'submit-preflight', 'submit-campaign'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args(); root = args.root.resolve()
    if args.mode.startswith('submit-'):
        submit(root, args.mode.removeprefix('submit-')); return
    if args.mode == 'report':
        report(root); return
    try:
        if args.mode == 'preflight': preflight(root)
        elif args.mode == 'screen': screen(root, args.index)
        else: confirm(root, args.index)
    except BaseException as exc:
        (root / 'state').mkdir(parents=True, exist_ok=True)
        save(root / 'state' / f'{args.mode}_{args.index:02d}_failure.json',
             dict(stage='FAILED', mode=args.mode, index=args.index, error=repr(exc), time=time.time()))
        raise


if __name__ == '__main__':
    main()
