"""Full KP training: official case-6 logic ported to the shared PyTorch runner."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

import train_equal_updates as trainer
from equal_updates_campaign import SEEDS
from torch_pinn import problem_kp2d as problem
from torch_pinn.model import MLP
from torch_pinn.records import configure, run_record, save, source_identity

PROTOCOL = 'kp_official_case6_pytorch_full_training_v1'
METHODS = ('shared_jet_linear', 'stdepp')


def fingerprint():
    root = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for f in sorted(root.rglob('*.py')):
        h.update(str(f.relative_to(root)).encode()); h.update(f.read_bytes())
    return h.hexdigest()


def plan():
    return dict(protocol=PROTOCOL, source_fingerprint=fingerprint(),
        problem='KP-II line soliton', pde='u_tx+6*(u_x**2+u*u_xx)+u_xxxx+3*u_yy=0',
        problem_source=problem.SOURCE_URL, spatial_domain=[[-3, 3], [-3, 3]], time_domain=[-.5, .5],
        exact='18*sigmoid(3*x+6*y-63*t)*sigmoid(-3*x-6*y+63*t)',
        constraints='Exact initial data and four lateral spatial Dirichlet faces',
        source_training_settings_reproduced=False,
        protocol_basis='Existing apolarity 2D equal-update training, with the cited KP problem',
        methods=list(METHODS), seeds=list(SEEDS), updates=20000, batch=400,
        width=128, depth=5, activation='tanh', constraint_side=16, points_per_surface=256,
        loss='MSE(PDE)+10*(MSE(IC)+sum_four_faces_MSE(Dirichlet))',
        optimizer=dict(name='Adam', lr=1e-4, betas=[.9, .999], eps=1e-8, foreach=False),
        resample_every_update=True, eval_every=100, eval_points=10000, eval_seed_base=20261400,
        evaluation='Space-time relative L2 on shared uniform independent points',
        stdepp=dict(source='sail-sg/stde/stde/equations.py KdV2d_res case=6',
            commit='fae88663b1d2f1b0666d4d250c38a3c34b852ac0',
            construction='5th-order x/t curved expansion + 4th-order x + spatial order-2 Hessian diagonal',
            curve_raw_tangents={'2': [1, 0, 0], '3': [0, 0, 1]},
            upstream_correction='Use function value, not indices, from hess_diag return tuple'),
        wdd='Order-4 x expansion plus order-2 directions (1,0,1),(1,0,-1),(0,1,0)',
        execution=dict(backend='eager PyTorch', torch='2.5.1+cu121', numpy='1.26.4',
            python='3.11.7', gpu='V100', allocated_cpus=10, threads=1, dtype='float32', tf32=False),
        checkpoints_saved=False, weights_saved=False,
        timing='Shared synchronized training-update timing; warm-up and evaluation excluded',
        paired_order=[list(METHODS if i % 2 == 0 else METHODS[::-1]) for i in range(5)],
        anomaly_policy='Preserve every run. Investigate paired discrepancies and timing jitter; '
                       'at most one full paired confirmation per flagged seed, without selection by RE.')


def relative_error(a, b):
    return float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(b).clamp_min(1e-30))


def objective_gate(device, full_size):
    dtype = torch.float32 if full_size else torch.float64
    width, depth, batch, side = (128, 5, 400, 16) if full_size else (8, 3, 7, 3)
    points = torch.as_tensor(problem.sample_numpy(np.random.default_rng(92), batch),
                             device=device, dtype=dtype)
    data = problem.constraints(side, dtype=dtype, device=device)
    reference = None
    rows = []
    methods = METHODS if full_size else ('nested_jvp',)+METHODS
    for method in methods:
        model = MLP(3, width, depth, 91, device=device, dtype=dtype)
        d = problem.components(model, points, method)
        value, parts = problem.loss(model, points, data, method)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False)
        value.backward()
        g = torch.cat([p.grad.detach().flatten() for p in model.parameters()])
        optimizer.step()
        parameters = torch.cat([p.detach().flatten() for p in model.parameters()])
        rtol, atol = (1e-3, 2e-6) if full_size else (1e-8, 1e-10)
        current = dict(components={k: v.detach() for k, v in d.items()},
                       loss=value.detach(), gradient=g, parameters=parameters,
                       terms={k: v.detach() for k, v in parts.items()})
        if reference is None:
            reference = current
        for key in ('loss', 'gradient', 'parameters'):
            torch.testing.assert_close(current[key], reference[key], rtol=rtol, atol=atol)
        for group in ('components', 'terms'):
            for key in reference[group]:
                torch.testing.assert_close(current[group][key], reference[group][key], rtol=rtol, atol=atol)
        rows.append(dict(method=method, full_size=full_size, loss=float(value.detach()),
            gradient_relative_error=relative_error(g, reference['gradient']),
            parameter_after_Adam_relative_error=relative_error(parameters, reference['parameters']),
            components={k: relative_error(current['components'][k], reference['components'][k]) for k in d}))
    return rows


def preflight(output, device):
    output.mkdir(parents=True, exist_ok=False)
    configure(device, 'V100'); source_identity()
    if device == 'cuda':
        assert torch.__version__ == '2.5.1+cu121' and np.__version__ == '1.26.4'
    # Include the soliton peak instead of testing only exponentially small tails.
    points = torch.tensor([[0., 0., 0.], [.2, -.1, 0.], [1., 0., 1/21],
                           [-1., 1., 0.], [.3, -.2, .01]], dtype=torch.float64, device=device)
    residual = problem.residual(problem.independent_components(problem.exact, points))
    assert float(residual.abs().max()) < 1e-8, residual
    rows = objective_gate(device, False)+objective_gate(device, True)
    save(output/'PASS.json', dict(status='PASS', source_fingerprint=fingerprint(),
        exact_solution_max_abs_residual=float(residual.abs().max()), checks=rows,
        torch=torch.__version__, numpy=np.__version__,
        gpu=torch.cuda.get_device_name() if device == 'cuda' else 'CPU'))
    print('PREFLIGHT_PASS', flush=True)


def run_cell(output, lane, position, gate, device):
    passed = json.loads((gate/'PASS.json').read_text())
    assert passed['status'] == 'PASS' and passed['source_fingerprint'] == fingerprint()
    seed = SEEDS[lane]
    method = (METHODS if lane % 2 == 0 else METHODS[::-1])[position]
    args = trainer.parse_args(['--out', str(output/f'kp2d_seed{seed}_{method}'),
        '--case', 'kp2d', '--method', 'shared_jet_linear', '--updates', '20000',
        '--eval-every', '100', '--device', device, '--expected-device', 'V100',
        '--seed', str(seed), '--batch', '400', '--width', '128', '--depth', '5',
        '--constraint-side', '16', '--eval-points', '10000', '--eval-seed-base', '20261400',
        '--lr', '1e-4'])
    args.method = method
    args.campaign_protocol = PROTOCOL
    args.paired_lane, args.method_position = lane, position
    args.record_components = True
    run_record(args, trainer.train)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('plan', 'preflight', 'cell', 'lane'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--lane', type=int, choices=range(5))
    parser.add_argument('--position', type=int, choices=range(2))
    parser.add_argument('--device', default='cuda', choices=('cuda', 'cpu'))
    args = parser.parse_args()
    if args.stage == 'plan':
        save(args.out, plan())
    elif args.stage == 'preflight':
        preflight(args.out, args.device)
    elif args.stage == 'cell':
        run_cell(args.out, args.lane, args.position, args.gate, args.device)
    else:
        for position in range(2):
            subprocess.run([sys.executable, __file__, 'cell', '--out', str(args.out),
                '--gate', str(args.gate), '--lane', str(args.lane), '--position', str(position),
                '--device', args.device], check=True)


if __name__ == '__main__':
    main()
