from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

work = Path(__file__).resolve().parents[1]
source = work / "results/analysis-shots-two/shot_effects.csv"
output = work / "figures/BBH_144_Shots_and_KV_Updates.png"
with source.open() as file:
    rows = list(csv.DictReader(file))
shots = (5, 10, 20)
retentions = (0.25, 0.50, 0.75, 1.0)
tasks = ("boolean_expressions", "date_understanding")
curves = {}
for row in rows:
    if row["task"] in tasks:
        assert all(int(row[f"n_complete_{n}"]) == 3 for n in shots)
        key = row["task"], float(row["retention"]), int(row["updates"])
        assert key not in curves
        curves[key] = [float(row[f"accuracy_{n}_pct"]) for n in shots]
assert len(curves) == 16
fig = plt.figure(figsize=(15, 8.5), facecolor="white")
plt.rcParams.update({"font.family":"DejaVu Sans", "svg.fonttype":"none"})
ink, muted, teal, gray = "#152c3a", "#4c5e69", "#007a72", "#637683"
def label(x, y, s, size, **k):
    fig.text(x,y,s,fontsize=size, color=k.pop("color", ink), ha=k.pop("ha","left"), va="center", **k)
label(0.05, .942, "After 16 KV updates, 20 shots gave the highest accuracy", 26, weight="bold")
label(.05,.882, "From 5 to 10 shots, accuracy rose in 7 of 8 settings. From 10 to 20, it rose in all 8, by 4.3–11.2 percentage points.",14, color=muted)
styles = {
    0: dict(color=gray, linestyle=(0,(3.6,2.9)), marker="s", markerfacecolor="white", markeredgecolor=gray,
            markeredgewidth=1.4, linewidth=1.8, markersize=7.5),
    16: dict(color=teal, linestyle="-", marker="o", markerfacecolor=teal, markeredgecolor=teal,
             markeredgewidth=.9, linewidth=2.7, markersize=5.3)
}
handles = [Line2D([],[],**styles[16]),Line2D([],[],**styles[0])]
fig.legend(handles, ["16 KV updates", "0 KV updates"], loc="center right", bbox_to_anchor=(.961,.807),ncol=2,
           handlelength=2.8, columnspacing=2.0,frameon=False, fontsize=12, labelcolor=ink, borderaxespad=0)
label(.05,.808,"Demonstration KV retained",14, weight="bold")
left, right, gap = .072, .96, .046
width = (right-left - gap*3)/4
for c, ret in enumerate(retentions):
    x = left+c*(width+gap)
    label(x+width/2,.742, f"{int(ret*100)}%" + (" · full cache" if ret == 1 else ""),15,weight="bold",ha="center")
for r, task in enumerate(tasks):
    title, base, limit, ticks = (
        ("Boolean Expressions", .475, (64.5,90.5), [70,75,80,85]) if r==0
        else ("Date Understanding", .177, (38,64), [40,45,50,55,60])
    )
    label(.05,base+.202,title,15,weight="bold")
    for c, ret in enumerate(retentions):
        x = left+c*(width+gap)
        ax = fig.add_axes([x,base,width,.17])
        ax.set_xlim(3.6,21.4)
        ax.set_ylim(*limit)
        ax.set_axisbelow(True)
        ax.grid(axis="y",color="#dce4e6",lw=.7)
        ax.spines[["left","right","top"]].set_visible(False)
        ax.spines["bottom"].set_color("#becacc")
        ax.tick_params(axis="both", length=0, pad=4, labelsize=11, colors=muted)
        ax.set_xticks(shots)
        ax.set_yticks(ticks)
        if c==0:
            ax.set_yticklabels([f"{n}%" for n in ticks])
        else:
            ax.tick_params(axis="y", labelleft=False)
        for u in [0,16]:
            vals = curves[task,ret,u]
            ax.plot(shots,vals,**styles[u], zorder=2 if u==0 else 3)
        for count, acc in zip(shots,curves[task,ret,16]):
            # Keep Boolean labels below the curves and Date labels above.
            above = r == 1 or (ret == .25 and count == 20)
            ax.annotate(f"{acc:.1f}", (count,acc), xytext=(0,6.3 if above else -7.3),
                        textcoords="offset points",va="bottom" if above else "top",ha="center",
                        fontsize=10.5,color="#005c57",weight="semibold",annotation_clip=False)
label(.515,.11, "Number of demonstrations (shots)",13,weight="bold",ha="center")
fig.add_artist(Line2D([.05,.958],[.084,.084],transform=fig.transFigure,color="#d7e0e2",lw=1))
label(.05,.054,"Mean accuracy across 3 demonstration seeds on the same 200 questions per task. 25–75% uses answer_attention selection.",11.5,color=muted)
fig.savefig(output,dpi=180,facecolor="white",metadata={"Title":"BBH accuracy by demonstration count, cache retention and KV updates"})
