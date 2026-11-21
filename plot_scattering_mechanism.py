"""Display-only common-axis mechanism plot; do not magnify rounding noise."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(output):
    output=Path(output); maps=json.loads((output/"mechanism_maps.json").read_text())
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True,layout="constrained")
    for ax,center in zip(axes,("zero","parallel","transverse")):
        rows=[r for r in maps if r["center"]==center and r["mode"]=="elastic"]
        angles=[r["angle_degrees"] for r in rows]
        ax.plot(angles,[r["kinetic_gain_i"] for r in rows],"o-",color="#248a9c",label="Physical normal law")
        samples=[[r["kinetic_gain_i"] for r in maps if r["center"]==center and r["angle_degrees"]==a and r["mode"]=="random"] for a in angles]
        ax.errorbar(angles,[np.mean(x) for x in samples],yerr=[2*np.std(x,ddof=1)/np.sqrt(len(x)) for x in samples],fmt="s",color="#d89426",label="Random mean +/- 2 SE")
        ax.plot(angles,[r["random_expected_gain_i"] for r in rows],"--",color="#d89426",label="Random analytic expectation")
        ax.axhline(0,color="#777777",linewidth=.7);ax.set_ylim(-.18,.36)
        ax.set_title("Pair center velocity: "+center);ax.set_xlabel("Contact-normal angle (degrees)")
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Kinetic change of particle i")
    h,l=axes[-1].get_legend_handles_labels();fig.legend(h,l,loc="outside lower center",ncols=3,fontsize=8)
    fig.suptitle("Single-impact diagnostics on a common vertical scale")
    fig.savefig(output/"scattering_mechanism.png",dpi=170);plt.close(fig)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("output",type=Path)
    main(p.parse_args().output)
