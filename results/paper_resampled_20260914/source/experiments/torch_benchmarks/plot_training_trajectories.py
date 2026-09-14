"""RSSE-style plots from real checkpoints; interpolation is not new evidence.

Style and one-sided seed band follow RSSE commit 1681e967, scripts/plot/
plot_poly3_unscaled_five_seed_paper.py. Single-seed curves have no band.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch_pinn.model import MLP

CASES = ('kdv1d', 'kdv2d', 'ch2d')
METHODS = ('nested_jvp', 'shared_jet_linear')
TITLES = dict(kdv1d='1D KdV', kdv2d='2D KdV', ch2d='2D Cahn–Hilliard')
LABELS = dict(nested_jvp='JVP', shared_jet_linear='Shared jet')
COLORS = dict(nested_jvp='#c63d2f', shared_jet_linear='#1856a5')
read = lambda path: json.loads(path.read_text())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


def style():
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':10.5,'axes.titlesize':11.5,
        'axes.labelsize':10.5,'legend.fontsize':8.6,'xtick.labelsize':9,
        'ytick.labelsize':9,'axes.linewidth':.8,'savefig.facecolor':'white',
        'figure.facecolor':'white','path.simplify':False})


def axis_style(ax):
    ax.grid(which='major',color='.82',linewidth=.65,alpha=.75)
    ax.grid(which='minor',color='.90',linewidth=.45,alpha=.45)
    ax.tick_params(direction='in',top=True,right=True,width=.7)


def export(fig, out, name):
    for suffix in ('png','pdf','svg'):
        fig.savefig(out/f'{name}.{suffix}',dpi=300,bbox_inches='tight',pad_inches=.04)
    plt.close(fig)


def log_interpolate(x, y, grid):
    x,y,grid = map(lambda a:np.asarray(a,dtype=float),(x,y,grid))
    if len(x)<2 or np.any(np.diff(x)<=0) or not np.isfinite(x).all():
        raise ValueError('strictly increasing observed coordinates required')
    if not np.isfinite(y).all() or np.any(y<=0):
        raise ValueError('positive finite metric required')
    if grid[0]<x[0] or grid[-1]>x[-1]:
        raise ValueError('extrapolation would fabricate unobserved coverage')
    return 10**np.interp(grid,x,np.log10(y))


def seed_stats(curves):
    values=np.asarray(curves,dtype=float)
    if values.ndim!=2 or not len(values):raise ValueError('nonempty seed matrix required')
    mean=np.mean(values,axis=0)
    return mean, np.std(values,axis=0,ddof=1) if len(values)>1 else None


def recover_error_curve(directory, out):
    config=read(directory/'config.json')
    manifest=directory/'trajectory_checkpoints.json'
    if not manifest.exists():manifest=directory/'checkpoints.json'
    checkpoints=read(manifest)
    key=f"{config['case']}_{config['method']}_seed{config['seed']}"
    cache=out/'curves'/f'{key}.json'
    sources={str(directory/'config.json'):sha(directory/'config.json'),
             str(manifest):sha(manifest),str(directory/'evaluation_points.npz'):sha(directory/'evaluation_points.npz')}
    sources.update({str(directory/cp['file']):sha(directory/cp['file']) for cp in checkpoints})
    if cache.exists() and read(cache)['source_sha256']==sources:return read(cache)
    module=__import__('train_fixed_wall').PROBLEMS[config['case']]
    model=MLP(2 if config['case']=='kdv1d' else 3,config['width'],config['depth'])
    with np.load(directory/'evaluation_points.npz') as data:
        points={name:torch.from_numpy(data[name].copy()) for name in ('spacetime','terminal')}
    targets={name:module.exact(p) for name,p in points.items()}
    rows=[]
    with torch.no_grad():
        for cp in checkpoints:
            state=torch.load(directory/cp['file'],map_location='cpu',weights_only=False)
            assert state['step']==cp['step']
            model.load_state_dict(state['model'])
            row={**cp}
            for name,p in points.items():
                pred=torch.cat([model(chunk) for chunk in p.split(2048)])
                row[name]=float(torch.linalg.vector_norm(pred-targets[name])/torch.linalg.vector_norm(targets[name]))
            rows.append(row)
    summary=read(directory/'summary.json')
    for name in points:
        assert np.isclose(rows[-1][name],summary['errors'][name]['relative_l2'],rtol=2e-3,atol=1e-7)
    result=dict(config=config,rows=rows,source_sha256=sources,
                evaluation='CPU checkpoint forward replay on original fixed evaluation points',
                raw_observations=len(rows))
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(result,indent=2)+'\n')
    print('Recovered actual error observations:',key,len(rows),flush=True)
    return result


def plot_error_panels(curves,out,points):
    fig,axes=plt.subplots(1,3,figsize=(15.2,4.1))
    metadata=[]
    for ax,case in zip(axes,CASES):
        group=[c for c in curves if c['config']['case']==case]
        rates={c['config']['lr'] for c in group}
        if len(rates)!=1:raise ValueError('learning rates cannot be pooled')
        seeds={m:{c['config']['seed'] for c in group if c['config']['method']==m} for m in METHODS}
        if seeds[METHODS[0]]!=seeds[METHODS[1]]:raise ValueError('unequal paired seed sets')
        for seed in seeds[METHODS[0]]:
            paired=[c for c in group if c['config']['seed']==seed]
            if len(paired)!=2:raise ValueError('duplicate or missing seed/method')
            for name in ('batch','width','depth','seconds','eval_seed_base','eval_points'):
                if paired[0]['config'][name]!=paired[1]['config'][name]:raise ValueError(f'paired {name} mismatch')
        lo=max(c['rows'][0]['time'] for c in group)
        hi=min(1200.,min(c['rows'][-1]['time'] for c in group))
        grid=np.linspace(lo,hi,points)
        for method in METHODS:
            selected=[c for c in group if c['config']['method']==method]
            samples=[log_interpolate([r['time'] for r in c['rows']],[r['spacetime'] for r in c['rows']],grid) for c in selected]
            mean,std=seed_stats(samples)
            if std is not None:
                ax.fill_between(grid,mean,mean+std,color=COLORS[method],alpha=.16,linewidth=0,zorder=1)
            ax.plot(grid,mean,color=COLORS[method],lw=2,label=LABELS[method],zorder=3)
            metadata.append(dict(case=case,method=method,seeds=sorted(seeds[method]),
                raw_observations=[c['raw_observations'] for c in selected],plot_points=points,
                band='mean to mean + sample std' if std is not None else None))
        ax.set(yscale='log',xlim=(lo,hi),xlabel='Training wall time (s)',
               title=f"{TITLES[case]}  ($\\eta={next(iter(rates)):g}$; $n={len(seeds[METHODS[0]])}$)")
        ax.legend(loc='upper right',framealpha=.93,edgecolor='.75');axis_style(ax)
    axes[0].set_ylabel(r'Relative $L^2$ error (space–time)')
    fig.subplots_adjust(left=.065,right=.992,bottom=.17,top=.89,wspace=.26)
    export(fig,out,'rsse_style_error_trajectories')
    return metadata


def density_study(root,out):
    # Existing 2D KdV is already lr=1e-3 for both methods, so this is a valid
    # density/axis comparison without relabeling the other historical runs.
    case='kdv2d'; arrays={}; identities={}
    for m in METHODS:
        directory=root/'confirmation'/case/m
        config=read(directory/'config.json');assert config['lr']==1e-3
        rows=[json.loads(line) for line in (directory/'steps.jsonl').read_text().splitlines()]
        arrays[m]=rows;identities[m]=sha(directory/'steps.jsonl')
    fig,axes=plt.subplots(1,3,figsize=(15.2,4.1))
    n=min(len(a) for a in arrays.values())
    for m in METHODS:
        rows=arrays[m];indices=np.unique(np.linspace(0,len(rows)-1,121).astype(int))
        color=COLORS[m]
        axes[0].plot([rows[i]['time'] for i in indices],[rows[i]['loss'] for i in indices],color=color,lw=1.5)
        axes[1].plot([r['time'] for r in rows],[r['loss'] for r in rows],color=color,lw=.5,alpha=.65,rasterized=True)
        axes[2].plot([r['step'] for r in rows[:n]],[r['loss'] for r in rows[:n]],color=color,
            lw=.7 if m=='nested_jvp' else 1.1,alpha=.65,linestyle='--' if m=='nested_jvp' else '-',
            zorder=4 if m=='nested_jvp' else 3,rasterized=True)
    titles=['(a) Sparse display: 121 observed points/method',
            '(b) Every recorded update, wall-time axis',
            f'(c) Every recorded update, common steps 1–{n}']
    handles=[Line2D([0],[0],color=COLORS[m],lw=2,label=LABELS[m]) for m in METHODS]
    for j,ax in enumerate(axes):
        ax.set(yscale='log',title=titles[j],xlabel='Optimizer update' if j==2 else 'Training wall time (s)')
        ax.legend(handles=handles,loc='upper right',framealpha=.93,edgecolor='.75');axis_style(ax)
        ax.set_xlim(0,n if j==2 else 1200)
    axes[0].set_ylabel('Raw minibatch loss (no smoothing)')
    fig.subplots_adjust(left=.065,right=.992,bottom=.17,top=.89,wspace=.26)
    export(fig,out,'density_and_axis_study_kdv2d_lr1e3')
    a,b=[np.array([r['loss'] for r in arrays[m][:n]]) for m in METHODS]
    rel=np.abs(a-b)/np.maximum(np.abs(a),1e-30)
    return dict(case=case,lr=.001,common_steps=n,raw_points={m:len(a) for m,a in arrays.items()},
        first_50_max_relative_loss_difference=float(np.max(rel[:50])),
        common_step_median_relative_loss_difference=float(np.median(rel)),
        fraction_common_steps_within_10_percent=float(np.mean(rel<.1)),source_sha256=identities)


def main():
    # Plotting is an offline dependency; GPU training and numerical tests do
    # not require Matplotlib to be installed in the cluster environment.
    global plt, Line2D
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--run-manifest',type=Path,help='Optional JSON array of explicit run directories for matched multi-seed plots.')
    parser.add_argument('--points',type=int,default=601)
    a=parser.parse_args()
    if a.points<2:parser.error('at least two display points required')
    a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);style()
    runs=[Path(p) for p in read(a.run_manifest)] if a.run_manifest else [a.root/'confirmation'/case/m for case in CASES for m in METHODS]
    curves=[recover_error_curve(p,a.out) for p in runs]
    panels=plot_error_panels(curves,a.out,a.points)
    density=density_study(a.root,a.out)
    manifest=dict(reference='RSSE 1681e967d795e159811eee0769d1d51f5e9e3fb0: scripts/plot/plot_poly3_unscaled_five_seed_paper.py',
        plotter_sha256=sha(Path(__file__)),
        panels=panels,density_study=density,
        no_synthetic_variance=True,interpolation_is_new_measurement=False,
        historical_learning_rates_preserved=True)
    (a.out/'plot_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(density,indent=2),flush=True)


if __name__=='__main__':main()
