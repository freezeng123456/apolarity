"""Re-run the reported KP cohort using the original frozen training source."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'source'
EXPECTED = json.loads((SOURCE / 'SOURCE_MANIFEST.json').read_text())
for name, expected in EXPECTED.items():
    assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == expected, name
sys.path.insert(0, str(SOURCE))
import kp_paired_campaign as campaign

campaign.SEEDS = tuple(range(20260936, 20260941))
campaign.PROTOCOL = 'kp_paper_fp32_seeds36_40_20261021'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('stage', choices=('plan', 'preflight', 'lane'))
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--lane', type=int, choices=range(5))
parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
a = parser.parse_args()
if a.stage == 'plan':
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / 'RUN_PLAN.json').write_text(json.dumps(campaign.plan(), indent=2)+'\n')
elif a.stage == 'preflight':
    campaign.preflight(a.out / 'gate', a.device)
else:
    if a.lane is None:
        parser.error('--lane is required for lane stage')
    for position in range(2):
        campaign.run_cell(a.out / 'cells', a.lane, position, a.out / 'gate', a.device)
