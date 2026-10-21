"""Observe five new FP32 KP seed pairs using the unchanged formal-run source."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

SEEDS = (20260934, 20260935, 20260936, 20260937, 20260938)
METHODS = ('shared_jet_linear', 'stdepp')
PROTOCOL = 'kp_fp32_new_seeds_20261020_v1'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify_source(formal_root, campaign):
    source = Path(campaign.__file__).resolve().parent
    prior = formal_root / 'results/attempt1/kp2d_seed20260919_shared_jet_linear/config.json'
    expected = json.loads(prior.read_text())['source_sha256']
    actual = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(source.rglob('*')) if p.suffix in ('.py', '.md')}
    assert actual == expected, 'source must match the previous formal campaign verbatim'
    return actual


def analyze_pair(root, seed):
    dirs = [root / 'results/cells' / f'kp2d_seed{seed}_{m}' for m in METHODS]
    configs = [json.loads((d / 'config.json').read_text()) for d in dirs]
    summaries = [json.loads((d / 'summary.json').read_text()) for d in dirs]
    curves = [list(csv.DictReader((d / 'curve.csv').open())) for d in dirs]
    assert all((d / 'status.txt').read_text().strip() == 'COMPLETE' for d in dirs)
    assert all(c['dtype'] == 'float32' and c['lr'] == 1e-4 for c in configs)
    keys = ('initial_hash', 'interior_samples_sha256', 'constraint_samples_sha256',
            'spacetime_test_sha256', 'terminal_test_sha256')
    matching = {k: summaries[0][k] == summaries[1][k] for k in keys}
    assert all(matching.values()), matching
    differences = []
    for a, b in zip(*curves):
        assert a['step'] == b['step']
        differences.append(dict(step=int(a['step']), absolute_RE_difference=abs(
            float(a['spacetime_relative_l2']) - float(b['spacetime_relative_l2']))))
    assert len(curves[0]) == len(curves[1]) == 201
    assert differences[-1]['step'] == 20000
    report = dict(seed=seed, methods=METHODS, matched_inputs=matching,
        final_RE=[s['errors']['spacetime']['relative_l2'] for s in summaries],
        final_absolute_RE_difference=differences[-1]['absolute_RE_difference'],
        mean_RE_over_19000_through_20000=[statistics.mean(
            float(r['spacetime_relative_l2']) for r in c if int(r['step']) >= 19000)
            for c in curves],
        max_absolute_RE_difference=max(differences, key=lambda r: r['absolute_RE_difference']),
        first_difference_above_1e_4=next((r for r in differences if r['absolute_RE_difference'] > 1e-4), None),
        training_seconds=[s['wall_seconds'] for s in summaries],
        stdepp_over_wdd_time_ratio=summaries[1]['wall_seconds'] / summaries[0]['wall_seconds'],
        weights_saved=False)
    save(root / 'results/pairs' / f'seed{seed}.json', report)
    print('PAIR_COMPLETE', json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=('plan', 'preflight', 'lane'))
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--formal-root', type=Path, required=True)
    p.add_argument('--lane', type=int, choices=range(5))
    a = p.parse_args()
    import kp_paired_campaign as campaign
    hashes = verify_source(a.formal_root, campaign)
    campaign.SEEDS = SEEDS
    campaign.PROTOCOL = PROTOCOL
    if a.stage == 'plan':
        plan = campaign.plan()
        plan.update(seeds=list(SEEDS), protocol=PROTOCOL,
            cohort_role='additional unseen-seed observation; preserve earlier formal cohort',
            source_matches_previous_formal_campaign=True,
            source_sha256=hashes,
            launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            previous_formal_root=str(a.formal_root),
            observations=['final RE', 'mean RE over evaluations at steps 19000 through 20000',
                          'paired absolute RE difference versus updates', 'training time ratio'],
            anomaly_policy='Keep all five seeds and all curves; no replacement or selection by RE.',
            implementation='Previous frozen mlp_jet/curved_jet implementation; same code as formal cohort.')
        save(a.root / 'RUN_PLAN.json', plan)
        print('PLAN_SAVED', json.dumps(dict(seeds=list(SEEDS), updates=20000, lr=1e-4,
                                           dtype='float32', source_fingerprint=campaign.fingerprint())), flush=True)
    elif a.stage == 'preflight':
        campaign.preflight(a.root / 'results/gate', 'cuda')
    else:
        assert a.lane is not None
        for position in range(2):
            campaign.run_cell(a.root / 'results/cells', a.lane, position,
                              a.root / 'results/gate', 'cuda')
        analyze_pair(a.root, SEEDS[a.lane])


if __name__ == '__main__':
    main()
