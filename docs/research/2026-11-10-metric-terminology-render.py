"""Presentation-only views of frozen summaries: zero new experimental runs."""
import hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/figures/2026-11-10-metric-terminology'
FULL='好粒子途中曾明顯偏離初始解的次數'
FONT=Path('/Library/Fonts/Arial Unicode.ttf')


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(FONT)).get_name()
    p5=['fixed_high','fixed_low','cap_all','reserve_best','reserve_random']
    l5=['全高','全低','全體曲率上限','保護當前最佳','隨機選擇保護']
    p8=[h+'_'+v for h in ['high','curve','fixed_ref','fixed_safe'] for v in ['keep','transfer']]
    l8=[h+'・'+v for h in ['全高','曲率高度','固定0.025','固定0.00625'] for v in ['保留動量','轉移動量']]
    views=[]
    jobs=[('collision_curvature_energy','e22_rest.png',p5,l5,False,False),
          ('collision_curvature_energy_kinetic','e22_moving.png',p5,l5,False,False),
          ('collision_energy_generalization','e23.png',p5,l5,True,True),
          ('collision_initial_budget','e24.png',p8,l8,False,True)]
    for folder,name,policies,labels,energy_rows,individual_final in jobs:
        source=ROOT/'output'/folder/'summary.json';data=source.read_bytes();summary=json.loads(data)
        assert all(r['runs']==20 and r['failures']==0 for r in summary)
        columns=[('good_present','particle0_harmed',FULL)]
        if individual_final:columns.append(('good_present','particle0_final_hits','好粒子最後達到目標的次數'))
        columns.append(('bad_only','final_hits','壞井最後達到目標的次數'))
        energies=[0.,1e-5,1e-3] if energy_rows else [None]
        factors=[.5,1.,2.] if energy_rows else [1.,8.,32.]
        fig,axes=plt.subplots(len(energies),len(columns),figsize=(16 if len(columns)==3 else 12,11 if energy_rows else 5.2),squeeze=False)
        width=.8/len(policies);x=np.arange(3)
        for i,energy in enumerate(energies):
            for j,(condition,key,title) in enumerate(columns):
                ax=axes[i,j]
                for t,(policy,label) in enumerate(zip(policies,labels)):
                    selected=[next(r for r in summary if r['condition']==condition and r['policy']==policy
                        and r['longitudinal_scale' if energy_rows else 'stiffness']==f
                        and (not energy_rows or r['incumbent_energy']==energy)) for f in factors]
                    values=[r[key] for r in selected];assert all(0<=n<=20 for n in values)
                    ax.bar(x+(t-(len(policies)-1)/2)*width,values,width,label=label,
                        color=f'C{t//2 if len(policies)==8 else t}',alpha=.45 if len(policies)==8 and t%2==0 else 1.)
                ax.set_title(title+(f'\n好粒子初始動能 K={energy:g}' if energy_rows else ''),fontsize=11)
                ax.set_xticks(x,[str(f) for f in factors]);ax.set_xlabel('縱向地形尺度' if energy_rows else '地形橫向曲率 κ')
                ax.set_ylim(0,21);ax.set_yticks([0,5,10,15,20]);ax.set_ylabel('次數（每種地形20次）')
        handles,legend=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,legend,loc='lower center',ncol=4 if len(policies)==8 else 5,fontsize=9)
        fig.tight_layout(rect=[0,.10 if not energy_rows else .05,1,1]);fig.savefig(OUT/name,dpi=160);plt.close(fig)
        views.append({'source_summary':str(source.relative_to(ROOT)),'source_sha256':hashlib.sha256(data).hexdigest(),
                      'view':str((OUT/name).relative_to(ROOT)),'view_sha256':hashlib.sha256((OUT/name).read_bytes()).hexdigest(),
                      'changed':'Labels and plotting units only: counts out of20; no metric recomputation'})
    (OUT/'views.json').write_text(json.dumps({'new_runs':0,'new_oracle_calls':0,'font':str(FONT),'views':views},indent=2)+'\n')
    print('Created four presentation views from unchanged summaries; zero new runs/oracle calls.')


if __name__=='__main__':main()
