"""Check the V100 campaign preserves the paper protocol and saved state."""
import pytest

from beijing_v100 import CASES, METHODS, PDE_SEEDS, pde_command, plan, resume_check
from torch_pinn.records import run_record
from train_fixed_wall import parse_args, train


def test_campaign_covers_paper_and_resolves_v100(tmp_path):
    campaign = plan(tmp_path)
    assert len(campaign['partial_cells']) == 12
    assert len(campaign['pde_cells']) == 30
    assert len({(c['case'], c['seed'], c['method']) for c in campaign['pde_cells']}) == 30
    for cell in campaign['pde_cells']:
        args = parse_args(cell['command'][2:])
        assert args.expected_device == 'V100' and args.device == 'cuda'
        assert args.seconds == 1200 and args.max_steps == 0
        assert args.width == 128 and args.depth == 5 and args.batch == 400
        assert args.eval_points == 10000 and args.eval_seed_base == 20261400
        assert args.constraint_side == (128 if args.case == 'kdv1d' else 16)
        assert args.lr == (1e-4 if args.case == 'ch2d' else 1e-5)
    assert {c['seed'] for c in campaign['pde_cells']} == set(PDE_SEEDS)


@pytest.mark.parametrize('case', CASES)
@pytest.mark.parametrize('method', METHODS)
def test_checkpoint_restore_makes_finite_adam_update(tmp_path, case, method):
    out = tmp_path/'smoke'
    cmd = pde_command(case, PDE_SEEDS[0], method, out, smoke=True)
    args = parse_args(cmd[2:] + ['--device', 'cpu'])
    run_record(args, train)
    restored = resume_check(out, device='cpu')
    assert restored['restored_step'] == 3 and restored['changed_parameters']
