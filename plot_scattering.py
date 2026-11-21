"""Saved-data plots for collision geometry and conserving-random ablation."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(output):
    output=Path(output); aggregate=json.loads((output/"aggregate.json").read_text())
    maps=json.loads((output/"mechanism_maps.json").read_text())
    cases=("quadratic","double_well","rosenbrock"); modes=[("independent",None),("elastic",.12),("random",.12),("elastic",.24),("random",.24)]
    labels=("Independent","Elastic .12","Random .12","Elastic .24","Random .24")
    colors=("#68737c","#248a9c","#d89426","#2961a2","#bb6e24")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(13,4.6),layout="constrained")
    for ax,case in zip(axes,cases):
        group=[next(r for r in aggregate if (r["case"],r["method"],r["rho"])==(case,m,rho)) for m,rho in modes]
        ax.bar(range(5),[r["historical_hits"] for r in group],color=colors,alpha=.25)
        ax.bar(range(5),[r["final_hits"] for r in group],color=colors)
        ax.set_xticks(range(5),labels,rotation=40,ha="right");ax.set_ylim(0,max(r["runs"] for r in group)+1)
        ax.set_title(case.replace("_"," "));ax.set_ylabel("Successful runs (failures count as no success)")
    fig.suptitle("Actual 6,000-query cap; solid = final hit, faded = ever hit")
    fig.savefig(output/"scattering_success.png",dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.9),layout="constrained")
    for ax,center in zip(axes,("zero","parallel","transverse")):
        physical=[r for r in maps if r["center"]==center and r["mode"]=="elastic"]
        angles=[r["angle_degrees"] for r in physical]
        ax.plot(angles,[r["kinetic_gain_i"] for r in physical],"o-",color="#248a9c",label="Physical normal law")
        random=[[r["kinetic_gain_i"] for r in maps if r["center"]==center and r["angle_degrees"]==a and r["mode"]=="random"] for a in angles]
        ax.errorbar(angles,[np.mean(x) for x in random],yerr=[2*np.std(x,ddof=1)/np.sqrt(len(x)) for x in random],fmt="s",color="#d89426",label="Random mean +/- 2 SE")
        ax.plot(angles,[r["random_expected_gain_i"] for r in physical],"--",color="#d89426",label="Random analytic expectation")
        ax.axhline(0,color="#777777",linewidth=.7);ax.set_title("Pair center velocity: "+center)
        ax.set_xlabel("Contact-normal angle (degrees)");ax.set_ylabel("Kinetic change of particle i")
    handles,legend=axes[-1].get_legend_handles_labels();fig.legend(handles,legend,loc="outside lower center",ncols=3,fontsize=8)
    fig.suptitle("Constructed single-impact mechanisms; no optimization success claims")
    fig.savefig(output/"scattering_mechanism.png",dpi=170);plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("output",type=Path)
    main(parser.parse_args().output)
