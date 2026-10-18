"""Complete the predeclared matrix sequentially with fresh cell processes."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source'; OUT=ROOT/'results'
assert os.environ.get('SLURM_CPUS_PER_TASK')=='10'
assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
plan=json.loads((ROOT/'RUN_PLAN.json').read_text())
record=dict(status='RUNNING',job_id=os.environ['SLURM_JOB_ID'],
    driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    plan_sha256=hashlib.sha256((ROOT/'RUN_PLAN.json').read_bytes()).hexdigest(),
    cpu_affinity=sorted(os.sched_getaffinity(0)),commands=[],checkpoints=False,pde_rerun=False)
def save():
    (ROOT/'FORMAL_RUN.json').write_text(json.dumps(record,indent=2)+'\n')
save()
for cell in plan['cells']:
    command=[sys.executable,str(SOURCE/'experiments/torch_benchmarks/stdepp_fixed_batch_campaign.py'),
        'cell','--out',str(OUT),'--index',str(cell['index'])]
    began=time.time()
    p=subprocess.run(command,cwd=SOURCE)
    record['commands'].append(dict(index=cell['index'],command=command,returncode=p.returncode,
        start_utc=datetime.datetime.fromtimestamp(began,datetime.timezone.utc).isoformat(),
        elapsed_seconds=time.time()-began))
    if p.returncode: record['status']='FAILED'
    save()
    if p.returncode: raise SystemExit(p.returncode)
record.update(status='COMPLETE',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
save()
