"""Plot E17 saved summaries; never execute or select optimization runs."""

import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(output):
    output=Path(output);rows=json.loads((output/"aggregate.json").read_text())
    cases=("quadratic","double_well","rosenbrock");methods=("independent","elastic","random")
    schedules=("weak","constant","ramp");colors=("#5d7182","#b16428","#17756e")
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharey="row")
    for col,case in enumerate(cases):
        for j,schedule in enumerate(schedules):
            group=[next(r for r in rows if (r["case"],r["method"],r["schedule"])==(case,m,schedule)) for m in methods]
            x=np.arange(3)+(j-1)*.24
            axes[0,col].bar(x,[r["final_hits"]/r["runs"] for r in group],.22,color=colors[j],label=schedule)
            axes[1,col].plot(np.arange(3),[r["median_relative_final_gap"] for r in group],"o-",color=colors[j],label=schedule)
        axes[0,col].set_title(case.replace("_"," "));axes[0,col].set_ylim(0,1.08)
        axes[1,col].set_yscale("log");axes[1,col].grid(axis="y",alpha=.2)
        for ax in axes[:,col]:ax.set_xticks(np.arange(3),methods)
    axes[0,0].set_ylabel("Final target fraction (failures count as misses)")
    axes[1,0].set_ylabel("Median final relative gap (valid runs)")
    axes[0,2].legend(frameon=False)
    fig.suptitle("E17: fixed drag schedules at a shared 6000-query cap | 10 seeds per arm",fontsize=13)
    fig.tight_layout();fig.savefig(output/"cooling_results.png",dpi=170);plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("output");plot(p.parse_args().output)
