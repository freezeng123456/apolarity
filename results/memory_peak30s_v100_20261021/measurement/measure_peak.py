"""Peak GPU allocation during warmed, repeated paper workloads; no checkpoints."""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    root = Path(__file__).resolve().parent
    plan = json.loads((root / 'RUN_PLAN.json').read_text())
    task = plan['tasks'][args.task]
    bundle = 'kp' if task['case'] == 'kp2d' else 'partials_kdv'
    source = root / 'sources' / bundle
    for name, expected in plan['source_sha256'][bundle].items():
        assert sha(source / name) == expected, name
    sys.path.insert(0, str(source))
    import numpy as np
    import torch
    from torch_pinn.records import configure, digest, sync
    configure('cuda', 'V100')
    assert torch.__version__ == '2.5.1+cu121' and np.__version__ == '1.26.4'
    assert len(os.sched_getaffinity(0)) >= 10
    cpu_model = next(line.split(':', 1)[1].strip() for line in
                     Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
    if task['kind'] == 'partial':
        import stdepp_scaling_campaign as core
        model, points = core.make_model_and_points(task['seed'], 1000, 128, 4, 'cuda')
        alpha = core.alpha_from_exponents(task['exponents'])
        schedule = core.prepare('waring_batched' if task['method'] == 'wdd' else 'stdepp',
                                alpha, points.shape[-1], dtype=points.dtype, device=points.device)
        prepared = core.prepared_partials.prepare_evaluation(model, points, schedule.directions,
            schedule.weights, len(alpha), conjugate_reuse=task['method'] == 'wdd')
        evaluate = core.sparse_jets.prepare_curve_evaluation(model, points, (1,),
            prepared.directions.unsqueeze(0), len(alpha), prepared.weights)
        preparation = dict(schedule=prepared.metadata(), backend=evaluate.metadata())
        initial_hash = digest(model.parameters())

        def update():
            with torch.no_grad():
                value = evaluate()
                sync('cuda')
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError('nonfinite partial derivative')
                return value[:8].detach().cpu().tolist()

        warm_hash = hashlib.sha256(points.detach().cpu().numpy().tobytes()).hexdigest()
    else:
        from train_fixed_wall import PROBLEMS, full_loss, sample
        from torch_pinn.constraint_sampling import ConstraintSampler, as_tensors
        from torch_pinn.model import MLP
        from torch_pinn import stdepp
        case = task['case']
        module = PROBLEMS[case]
        dim = 2 if case == 'kdv1d' else 3
        side = 128 if dim == 2 else 16
        model = MLP(dim, 128, 5, task['seed'] + 2, device='cuda')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4 if case == 'kp2d' else 1e-5,
                                     betas=(.9, .999), eps=1e-8, foreach=False)
        rng = np.random.default_rng(task['seed'] + 1)
        sampler = ConstraintSampler(case, side, task['seed'])
        initial_hash = digest(model.parameters())
        warm_digest = hashlib.sha256()
        warming = True
        preparation = dict(batch=400, width=128, depth=5, constraint_side=side,
                           lr=optimizer.param_groups[0]['lr'], resample_every_update=True)

        def update():
            pn = sample(rng, 400, case)
            cn = sampler.sample_numpy()
            if warming:
                warm_digest.update(pn.tobytes())
                for name in sorted(cn):
                    warm_digest.update(name.encode())
                    warm_digest.update(cn[name].tobytes())
            points = torch.as_tensor(pn, device='cuda')
            data = as_tensors(case, cn, 'cuda')
            optimizer.zero_grad(set_to_none=True)
            if case == 'kp2d':
                value, parts = full_loss(module, model, points, data,
                    'stdepp' if task['method'] == 'stdepp' else 'shared_jet_linear')
            elif task['method'] == 'stdepp':
                value, parts = stdepp.pde_loss(case, model, points, data)
            else:
                value, parts = full_loss(module, model, points, data, 'shared_jet_linear')
            value.backward()
            if not bool(torch.isfinite(value)) or not all(
                    p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
                raise FloatingPointError('nonfinite objective or gradient')
            optimizer.step()
            sync('cuda')
            return float(value.detach())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'config.json', dict(task=task, mode=args.mode, seconds=args.seconds,
        warmup_calls=10, primary_metric=plan['primary_metric'], preparation=preparation,
        source_bundle=bundle, source_hashes=plan['source_sha256'][bundle],
        runner_sha256=sha(Path(__file__)), plan_sha256=sha(root / 'RUN_PLAN.json'),
        torch=torch.__version__, numpy=np.__version__, gpu=torch.cuda.get_device_name(),
        cpu_model=cpu_model,
        threads=torch.get_num_threads(), dtype='float32', tf32=False, compile=False,
        job_id=os.environ.get('SLURM_JOB_ID'), array_id=os.environ.get('SLURM_ARRAY_TASK_ID'),
        cpu_affinity=sorted(os.sched_getaffinity(0)), checkpoints_saved=False))
    (out / 'status.txt').write_text('RUNNING\n')
    try:
        first_warm_result = None
        for index in range(10):
            warm_value = update()
            if index == 0:
                first_warm_result = warm_value
        if task['kind'] == 'pde':
            warming = False
            warm_hash = warm_digest.hexdigest()
            optimizer.zero_grad(set_to_none=True)
        gc.collect()
        sync('cuda')
        torch.cuda.reset_peak_memory_stats()
        baseline_allocated = torch.cuda.memory_allocated()
        baseline_reserved = torch.cuda.memory_reserved()
        stop, ready = threading.Event(), threading.Event()
        samples, failures = [], []
        began = time.perf_counter()

        def telemetry():
            try:
                while not stop.is_set():
                    text = subprocess.check_output(['nvidia-smi',
                        '--query-compute-apps=pid,gpu_uuid,used_gpu_memory',
                        '--format=csv,noheader,nounits'], text=True, timeout=5)
                    rows = list(csv.reader(text.splitlines()))
                    own = [r for r in rows if int(r[0].strip()) == os.getpid()]
                    assert len(own) == 1, 'process GPU telemetry unavailable'
                    samples.append(dict(time_s=time.perf_counter()-began,
                        gpu_uuid=own[0][1].strip(), process_used_mib=float(own[0][2]),
                        allocated_bytes=torch.cuda.memory_allocated(),
                        allocated_peak_bytes=torch.cuda.max_memory_allocated(),
                        reserved_peak_bytes=torch.cuda.max_memory_reserved()))
                    ready.set()
                    stop.wait(.1)
            except Exception as exc:
                failures.append(repr(exc))
                ready.set()

        monitor = threading.Thread(target=telemetry, daemon=True)
        monitor.start()
        if not ready.wait(8) or failures:
            raise RuntimeError('telemetry startup failed: ' + repr(failures))
        calls = 0
        try:
            while time.perf_counter()-began < args.seconds and not failures:
                last_value = update()
                calls += 1
        finally:
            stop.set()
            monitor.join(timeout=8)
        sync('cuda')
        elapsed = time.perf_counter()-began
        assert not monitor.is_alive() and not failures and len(samples) >= 2 and calls > 0
        peak = torch.cuda.max_memory_allocated()
        assert peak >= baseline_allocated > 0
        with (out / 'memory_samples.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
            writer.writeheader()
            writer.writerows(samples)
        summary = dict(task=task, mode=args.mode, requested_seconds=args.seconds,
            elapsed_seconds=elapsed, complete_calls_or_updates=calls, warmup_calls=10,
            peak_allocated_bytes=peak, peak_allocated_mib=peak/2**20,
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            sampled_process_peak_mib=max(s['process_used_mib'] for s in samples),
            baseline_allocated_bytes=baseline_allocated, baseline_reserved_bytes=baseline_reserved,
            primary_metric=plan['primary_metric'], sample_count=len(samples),
            initial_hash=initial_hash, warmup_samples_sha256=warm_hash,
            first_warm_result=first_warm_result, last_result=last_value,
            final_parameter_hash=digest(model.parameters()), finite=True, checkpoints_saved=False,
            window_end_rule='First complete call/update ending at or after the 30-second target; actual elapsed time retained.')
        save(out / 'summary.json', summary)
        (out / 'status.txt').write_text('COMPLETE\n')
        print(json.dumps(dict(task=args.task, status='COMPLETE', peak_allocated_mib=peak/2**20)), flush=True)
    except Exception as exc:
        save(out / 'failure.json', dict(error=repr(exc)))
        (out / 'status.txt').write_text('FAILED\n')
        raise
    finally:
        files = sorted(p for p in out.iterdir() if p.is_file() and p.name != 'SHA256SUMS')
        (out / 'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.name+'\n' for p in files))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=int, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--mode', choices=('gate', 'formal'), required=True)
    parser.add_argument('--seconds', type=float, default=30)
    run(parser.parse_args())
