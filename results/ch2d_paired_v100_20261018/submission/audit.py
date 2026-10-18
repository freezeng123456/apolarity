"""Inspect all seeds and automatically confirm flagged complete groups once."""
from pathlib import Path
import hashlib,json,os,subprocess,sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source/experiments/torch_benchmarks'))
import ch2d_paired_campaign as paired
from torch_pinn.records import save
for name,sha in json.loads((ROOT/'SCHEDULER_SOURCE.json').read_text()).items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
attempt=int(os.environ.get('CH_ATTEMPT','1'))
references=json.loads((ROOT/'BASELINE_PAIRS.json').read_text())
target=ROOT/('AUDIT_ATTEMPT%d.json'%attempt)
assert not target.exists()
if attempt==1:
    audit=paired.audit(ROOT/'results/attempt1',references)
else:
    prior=json.loads((ROOT/'AUDIT_ATTEMPT1.json').read_text())
    flagged=prior['flagged_lanes']
    groups=[paired.inspect_group(ROOT/'results/attempt2' if i in flagged else ROOT/'results/attempt1',i,references)
            for i in range(5)]
    for group in groups:
        group['attempt']=2 if group['lane'] in flagged else 1
    audit=paired.audit_groups(groups)
audit['attempt']=attempt
audit['job_id']=os.environ['SLURM_JOB_ID']
audit['automatic_confirmation_repeats_remaining']=1 if attempt==1 else 0
save(target,audit)
print(json.dumps(audit,indent=2),flush=True)
if attempt==1 and audit['flagged_lanes']:
    intent=ROOT/'RETRY_INTENT.json'
    with intent.open('x') as f:json.dump(dict(lanes=audit['flagged_lanes'],attempt=2,paired_methods=3),f)
    common=['sbatch','--parsable','--partition=gpu','--nodes=1','--ntasks=1','--gres=gpu:1','--cpus-per-task=10',
        '--export=ALL,CH_ATTEMPT=2']
    command=common+['--array='+','.join(str(i) for i in audit['flagged_lanes'])+'%5',
        '--job-name=apo-ch-confirm','--output='+str(ROOT/'results/scheduler/repeat-%A_%a.log'),str(ROOT/'worker.sh'),'formal']
    p=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
    save(ROOT/'RETRY_SUBMISSION.json',dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
    assert p.returncode==0
    job=p.stdout.strip().split(';')[0];assert job.isdigit()
    command=common+['--dependency=afterany:'+job,'--job-name=apo-ch-audit2',
        '--output='+str(ROOT/'results/scheduler/audit2-%j.log'),str(ROOT/'worker.sh'),'audit']
    p=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
    save(ROOT/'FINAL_AUDIT_SUBMISSION.json',dict(command=command,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
    assert p.returncode==0
