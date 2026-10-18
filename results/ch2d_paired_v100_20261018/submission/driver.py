"""Run one seed's complete method group, retaining every attempt."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'source'
sys.path.insert(0,str(SOURCE/'experiments/torch_benchmarks'))
import ch2d_paired_campaign as paired
from torch_pinn.records import save
assert os.environ.get('SLURM_CPUS_PER_TASK')=='10'
assert all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
for name,sha in json.loads((ROOT/'SCHEDULER_SOURCE.json').read_text()).items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha,name
attempt=int(os.environ.get('CH_ATTEMPT','1'))
lane=int(os.environ['SLURM_ARRAY_TASK_ID'])
out=ROOT/'results'/('attempt'+str(attempt))
plan=paired.plan(out)
assert plan['source_fingerprint']==json.loads((ROOT/'RUN_PLAN.json').read_text())['source_fingerprint']
references=json.loads((ROOT/'BASELINE_PAIRS.json').read_text())
record_path=ROOT/'results'/('attempt%d-lane%d.json'%(attempt,lane))
assert not record_path.exists()
gpu=os.environ.get('SLURM_JOB_GPUS',os.environ['CUDA_VISIBLE_DEVICES'])
query=['nvidia-smi','-i',gpu,'--query-gpu=name,uuid,pci.bus_id,pstate,temperature.gpu,clocks.sm,clocks.mem,utilization.gpu,memory.used,power.draw,power.limit','--format=csv']
def telemetry():
    p=subprocess.run(query,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
    return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
record=dict(status='RUNNING',attempt=attempt,lane=lane,seed=plan['lanes'][lane]['seed'],
    job_id=os.environ['SLURM_JOB_ID'],array_job_id=os.environ['SLURM_ARRAY_JOB_ID'],
    node=os.uname().nodename,cpu_affinity=sorted(os.sched_getaffinity(0)),
    assigned_gpu=gpu,hardware_before=telemetry(),commands=[],checkpoints=False)
save(record_path,record)
failed=False
for index in plan['lanes'][lane]['indices']:
    cell=plan['cells'][index]
    command=[sys.executable,str(SOURCE/'experiments/torch_benchmarks/ch2d_paired_campaign.py'),
        'cell','--out',str(out),'--gate',str(ROOT/'results/preflight'),'--index',str(index)]
    log=ROOT/'results/scheduler'/('attempt%d-lane%d-%s.log'%(attempt,lane,cell['method']))
    start=time.time()
    print('START',attempt,lane,cell['name'],flush=True)
    with log.open('x') as stream:
        p=subprocess.run(command,cwd=SOURCE,stdout=stream,stderr=subprocess.STDOUT)
    record['commands'].append(dict(index=index,method=cell['method'],returncode=p.returncode,
        start_utc=datetime.datetime.fromtimestamp(start,datetime.timezone.utc).isoformat(),
        elapsed_seconds=time.time()-start,log=str(log.relative_to(ROOT)),hardware_after=telemetry()))
    failed=failed or bool(p.returncode)
    save(record_path,record)
    print('FINISH',attempt,lane,cell['name'],p.returncode,flush=True)
record['group_audit']=paired.inspect_group(out,lane,references)
record['status']='FAILED' if failed else 'COMPLETE'
record['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
save(record_path,record)
raise SystemExit(int(failed))
