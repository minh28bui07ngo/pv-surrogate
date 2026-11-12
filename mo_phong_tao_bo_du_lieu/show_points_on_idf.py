import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

IDF = r"D:/da2/in191026.idf"   # change path if needed

# ---- your points ----
P = {
 "T13": [32.4394408918354, 284.765261877747, 16],
 "T14": [29.7562875901586, 289.3755500987170, 20],
 "T21": [29.7562875901586, 289.3755500987170, 20],
 "T22": [27.0731342884818, 293.985838319687, 16],
 "T31": [-15.4652105956260, 269.2488417277640, 20],
 "T32": [-10.4578042380461, 272.183117149718, 16],
 "T43": [-20.4726169532059, 266.314566305811, 16],
 "T44": [-15.4652105956260, 269.2488417277640, 20],
 "T53": [10.5740732070205, 321.8269737769870, 16],
 "T54": [7.8909199053437, 326.4372619979570, 20],
 "T61": [7.8909199053437, 326.4372619979570, 20],
 "T62": [5.20776660366693, 331.047550218927, 16],
 "T71": [28.6670260363029, 317.7323311793580, 12.8],
 "T72": [19.8914654697423, 312.697195935236, 8.2],
 "T83": [37.4425866028634, 322.767466423479, 8.2],
 "T84": [28.6670260363029, 317.7323311793580, 12.8],
}
P = {k: np.array(v) for k, v in P.items()}

# ---- minimal IDF parser (no eppy needed) ----
text = open(IDF, encoding="utf-8", errors="ignore").read()
text = re.sub(r"!.*", "", text)                       # strip comments
objs = [o.split(",") for o in text.split(";")]
objs = [[f.strip() for f in o] for o in objs]

def polys(kind, first_vertex_idx):
    out = []
    for o in objs:
        if o[0].lower() == kind.lower():
            nums = [float(x) for x in o[first_vertex_idx:] if x]
            out.append((o[1], np.array(nums).reshape(-1, 3)))
    return out

# BuildingSurface:Detailed: vertices start at field 11 (index 11 after the 10 header fields + Number of Vertices)
walls = polys("BuildingSurface:Detailed", 11)
pvs   = polys("Shading:Building:Detailed", 4)         # name, transm., nverts, then xyz

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection="3d")
ax.add_collection3d(Poly3DCollection([v for _, v in walls], facecolors="lightgrey",
                                     edgecolors="grey", linewidths=0.3, alpha=0.35))
ax.add_collection3d(Poly3DCollection([v for _, v in pvs], facecolors="tab:blue",
                                     edgecolors="navy", alpha=0.6))
for n, v in pvs:
    c = v.mean(0); ax.text(*c, n.split("_")[0], color="navy")

# your points (T14==T21, T44==T31, T54==T61, T71==T84 are coincident -> labels merged)
seen = {}
for k, p in P.items():
    seen.setdefault(tuple(np.round(p, 6)), []).append(k)
for xyz, names in seen.items():
    ax.scatter(*xyz, c="red", s=50, depthshade=False)
    ax.text(*xyz, "/".join(names), color="red", fontsize=8)

allv = np.vstack([v for _, v in walls] + [v for _, v in pvs])
mn, mx = allv.min(0), allv.max(0)
ax.set_xlim(mn[0], mx[0]); ax.set_ylim(mn[1], mx[1]); ax.set_zlim(mn[2], mx[2])
ax.set_box_aspect(mx - mn)                            # true proportions
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
plt.show()
