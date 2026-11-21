"""E18 saved-data plots for suffix comparison and uninterrupted replication."""

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(output):
    output=Path(output);rows=json.loads((output/'aggregate.json').read_text())
    modes=('independent','elastic','random');arms=('continue','independent','gd')
    colors=('#5d7182','#b16428','#17756e');cases=('quadratic','double_well','rosenbrock')
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharey='row')
    for col,case in enumerate(cases):
        for j,arm in enumerate(arms):
            g=[next(r for r in rows if (r['case'],r['prefix_method'],r['arm'])==(case,m,arm)) for m in modes]
            axes[0,col].bar(np.arange(3)+(j-1)*.24,[r['final_hits']/r['runs'] for r in g],.22,color=colors[j],label=arm)
            axes[1,col].plot(np.arange(3),[max(r['median_relative_final_gap'],1e-18) for r in g],'o-',color=colors[j])
        axes[0,col].set_title(case.replace('_',' '));axes[0,col].set_ylim(0,1.08)
        axes[1,col].set_yscale('log');axes[1,col].grid(axis='y',alpha=.2)
        for ax in axes[:,col]:ax.set_xticks(np.arange(3),modes)
        axes[1,col].set_xlabel('Exploration prefix rule')
    axes[0,0].set_ylabel('Final target fraction (failures as misses)')
    axes[1,0].set_ylabel('Median final relative gap (valid runs)')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3,bbox_to_anchor=(.5,.96),frameon=False)
    fig.suptitle('E18: fixed handoff near half of a shared 6000-query cap',y=1.)
    fig.tight_layout(rect=(0,0,1,.9));fig.savefig(output/'handoff_results.png',dpi=170);plt.close(fig)
    rows=json.loads((output/'replication_aggregate.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(12,3.6),sharey=True)
    for ax,case in zip(axes,cases):
        g=[next(r for r in rows if (r['case'],r['arm'])==(case,s)) for s in ('weak','constant','ramp')]
        ax.bar(np.arange(3),[r['final_hits']/r['runs'] for r in g],color=colors)
        ax.set_xticks(np.arange(3),('weak','constant','ramp'));ax.set_title(case.replace('_',' '));ax.set_ylim(0,1.08)
    axes[0].set_ylabel('Final target fraction')
    fig.suptitle('E17 protocol replicated: elastic population, 20 fresh seeds per arm')
    fig.tight_layout();fig.savefig(output/'cooling_replication.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');plot(p.parse_args().output)
