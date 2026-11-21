"""E19 stored timing curves; no new optimizer executions."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(output):
    output=Path(output);rows=json.loads((output/'aggregate.json').read_text());safety=json.loads((output/'config.json').read_text())['stage']=='safety'
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,case in zip(axes,('quadratic','double_well','rosenbrock')):
        zero=next(r for r in rows if r['case']==case and r['rule']=='independent')
        for mode,color in [('elastic','#17756e'),('random','#b16428')]:
            g=sorted([r for r in rows if r['case']==case and r['rule']==mode],key=lambda r:r['fraction'])
            if not g:continue
            g=[zero]+g;y=[r['harmed']/r['runs'] if safety else r['final_hits']/r['runs'] for r in g]
            ax.plot([r['fraction'] for r in g],y,'o-',color=color,label=mode)
        ax.set_title(case.replace('_',' '));ax.set_xlabel('Budget fraction when interactions disabled');ax.set_ylim(-.05,1.05);ax.grid(alpha=.2)
    axes[0].set_ylabel('Protected-particle harm fraction' if safety else 'Final target fraction')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2,frameon=False)
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(output/'timing_curve.png',dpi=170);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');plot(p.parse_args().output)
