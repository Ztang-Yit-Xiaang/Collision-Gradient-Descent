"""Render saved optimizer comparisons; no training or selection occurs here."""
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR","/tmp/ee5239-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME","/tmp/ee5239-cache")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
LABELS={"momentum":"Momentum SGD","adam":"Adam / AdamW (wd=0)","momentum_random":"Momentum + random",
        "momentum_predictive":"Momentum + preview","adam_random":"Adam + random","adam_predictive":"Adam + preview",
        "gravity":"Gravity","gravity_predictive":"Gravity + preview ball",
        "adamw":"AdamW","adamw_random":"AdamW + random","adamw_predictive":"AdamW + preview"}


def synthetic(output):
    cfg=json.loads((output/"config.json").read_text());data=json.loads((output/"aggregate.json").read_text())
    for protocol in ["budget","fixed_updates"]:
        fig,ax=plt.subplots(figsize=(13.8,7.7));values=np.full((len(cfg["methods"]),len(cfg["cases"])),np.nan)
        cells={}
        for i,method in enumerate(cfg["methods"]):
            for j,case in enumerate(cfg["cases"]):
                row=next(r for r in data if r["protocol"]==protocol and r["case"]==case and r["method"]==method)
                cells[i,j]=row
                if row["median_relative_best_gap"] is not None:values[i,j]=np.log10(max(row["median_relative_best_gap"],1e-12))
        im=ax.imshow(values,vmin=-12,vmax=0,cmap="RdYlBu_r",aspect="auto")
        for (i,j),r in cells.items():
            gap=r["median_relative_best_gap"]
            text="no valid runs" if gap is None else f"{r['target_hits']}/{r['runs']} hits\n"+("gap < 1e-12" if gap<1e-12 else f"gap {gap:.2g}")
            if r["failures"]:text+=f"\n{r['failures']} failures"
            ax.text(j,i,text,ha="center",va="center",fontsize=9,color="white" if np.isfinite(values[i,j]) and values[i,j]<-9 else "black")
        ax.set(xticks=range(len(cfg["cases"])),xticklabels=[c.replace("_"," ") for c in cfg["cases"]],
               yticks=range(len(cfg["methods"])),yticklabels=[LABELS[m] for m in cfg["methods"]])
        title="Exact 1,500-query budget" if protocol=="budget" else "500 actual updates, cap 24,000 queries (not equal compute)"
        ax.set_title(title+"\nHeld-out initial states; actual target only; median relative best gap",pad=15)
        fig.colorbar(im,ax=ax,label="log10(relative best gap), lower is better")
        fig.tight_layout();fig.savefig(output/f"synthetic_{protocol}.png",dpi=180);plt.close(fig)


def nanogpt(output):
    data=json.loads((output/"aggregate.json").read_text());runs=json.loads((output/"runs.json").read_text())
    methods=[r["method"] for r in data];labels=["Momentum","Adam","AdamW","Mom + random","Mom + preview","AdamW + random","AdamW + preview"]
    fig,axes=plt.subplots(1,2,figsize=(14.5,5.8))
    for i,row in enumerate(data):
        scores=[r["final_loss"] for r in runs if r["method"]==row["method"] and r["final_loss"] is not None]
        axes[0].scatter(np.linspace(i-.12,i+.12,len(scores)),scores,s=28,color="#087e8b",zorder=3)
        if scores:axes[0].errorbar(i,np.mean(scores),yerr=np.std(scores,ddof=1),fmt="_",markersize=18,capsize=5,color="black")
    axes[0].set(ylabel="Final held-out cross entropy (nats)",title="3 paired seeds: points, mean and sample SD")
    bottom=np.zeros(len(methods))
    for key,label,color in [("real_train_units","Actual training","#087e8b"),("real_probe_units","Diagnostics","#d58519"),
                            ("virtual_train_units","Virtual training","#7557b3"),("virtual_probe_units","Virtual probes","#c07fac")]:
        values=np.array([np.mean([r[key] for r in runs if r["method"]==m]) for m in methods])
        axes[1].bar(range(len(methods)),values,bottom=bottom,color=color,label=label);bottom+=values
    for i,row in enumerate(data):axes[1].text(i,610,f"{row['mean_actual_steps']:.0f}\nupdates",ha="center",color="white",fontsize=9)
    axes[1].set(ylabel="Batch-forward equivalent units",title="Shared cap 1,200 units; all interventions = 0");axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(range(len(methods)),labels,rotation=28,ha="right");ax.grid(axis="y",alpha=.15);ax.spines[["top","right"]].set_visible(False)
    fig.suptitle("Small official NanoGPT, 108,352 parameters | both Adam variants selected weight_decay=0",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(output/"nanogpt_comparison.png",dpi=180);plt.close(fig)


if __name__=="__main__":
    synthetic(ROOT/"output"/"optimizer_comparison")
    nanogpt(ROOT/"output"/"nanogpt_comparison")
