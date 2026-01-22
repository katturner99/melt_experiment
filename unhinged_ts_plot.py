import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import matplotlib.gridspec as gridspec

from tools.constants import regions
from tools.directories_and_paths import OUTPUT_PATH

# ==========================================================
# User settings
# ==========================================================

RUNS = ["PAS_LENS_2005_average_TS.nc","PAS_MELT_2100_average_TS.nc", "PAS_LENS_2100_average_TS.nc"]
titles = ["CONTROL", "MELT", "LENS"]
YEAR = "210001"

T_bins = np.arange(-2.0, 2.0, 0.05)
S_bins = np.arange(32.5, 35.0, 0.05)

region = "cont_shelf"
lat_rng, lon_rng = regions[region]

Z_MAX = -1000.0  # shallower than 500 m

# ==========================================================
# Helper functions
# ==========================================================

def compute_volume(ds):
    return ds.rA * ds.drF * ds.hFacC


def compute_ts_histogram(run):
    """
    Load data, apply region & depth mask,
    and return volume-weighted TS histogram.
    """
    ds = xr.open_dataset(
        f"{OUTPUT_PATH}/{run}"
    )

    ds = ds.sel(YC=slice(*lat_rng), XC=slice(*lon_rng)).mean("time")

    Z = ds.Z.broadcast_like(ds.THETA)
    mask = (ds.hFacC > 0) & (Z >= Z_MAX)

    T = ds.THETA.where(mask)
    S = ds.SALT.where(mask)
    V = compute_volume(ds).where(mask)

    T_flat = T.values.ravel()
    S_flat = S.values.ravel()
    V_flat = V.values.ravel()

    good = np.isfinite(T_flat) & np.isfinite(S_flat) & np.isfinite(V_flat)

    hist, T_edges, S_edges = np.histogram2d(
        T_flat[good],
        S_flat[good],
        bins=[T_bins, S_bins],
        weights=V_flat[good]
    )

    return hist, T_edges, S_edges


# ==========================================================
# Compute both histograms
# ==========================================================

results = {}
for run in RUNS:
    results[run] = compute_ts_histogram(run)

diff_pairs = [
    ("PAS_MELT_2100_average_TS.nc", "PAS_LENS_2100_average_TS.nc", "MELT − LENS"),
    ("PAS_MELT_2100_average_TS.nc", "PAS_LENS_2005_average_TS.nc", "MELT − CONTROL"),
    ("PAS_LENS_2100_average_TS.nc", "PAS_LENS_2005_average_TS.nc", "LENS − CONTROL"),
]


# Shared color normalization
all_volumes = np.concatenate([results[r][0].ravel() for r in RUNS])
all_volumes = all_volumes[all_volumes > 0]
norm = plt.Normalize(all_volumes.min(), all_volumes.max())

diff_results = []

for run_a, run_b, label in diff_pairs:
    hist_a = results[run_a][0]
    hist_b = results[run_b][0]
    diff_results.append((hist_a - hist_b, label))

# Symmetric normalization around zero
all_diffs = np.concatenate([d[0].ravel() for d in diff_results])
max_abs = 0.21e12
diff_norm = plt.Normalize(-max_abs, max_abs)


# ==========================================================
# Plot
# ==========================================================

fig = plt.figure(figsize=(18, 13))
cmap = cm.PuRd

gs = gridspec.GridSpec(
    nrows=2,
    ncols=4,
    height_ratios=[1.0, 0.8],
    width_ratios=[1, 1, 1, 0.05],
    hspace=0.25,
    wspace=0.15,
)

axes = []

for i, run in enumerate(RUNS):

    hist, T_edges, S_edges = results[run]

    T_centers = 0.5 * (T_edges[:-1] + T_edges[1:])
    S_centers = 0.5 * (S_edges[:-1] + S_edges[1:])

    xpos, ypos = np.meshgrid(T_centers, S_centers, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)

    dx = T_edges[1] - T_edges[0]
    dy = S_edges[1] - S_edges[0]
    dz = hist.ravel()

    nonzero = dz > 0
    xpos = xpos[nonzero]
    ypos = ypos[nonzero]
    zpos = zpos[nonzero]
    dz   = dz[nonzero]

    colors = cmap(norm(dz))

    ax = fig.add_subplot(gs[0, i], projection="3d")
    axes.append(ax)

    ax.bar3d(
        xpos, ypos, zpos,
        dx, dy, dz,
        color=colors,
        shade=True,
        zsort="average"
    )

    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Salinity (PSU)")
    ax.set_zlabel("Volume (m³)")

    ax.invert_xaxis() 
    ax.set_title(titles[i])

diff_cmap = cm.RdBu_r

for i, (diff_hist, label) in enumerate(diff_results):

    hist, T_edges, S_edges = diff_hist, results[RUNS[0]][1], results[RUNS[0]][2]

    ax = fig.add_subplot(gs[1, i])

    pcm = ax.pcolormesh(
        S_edges,
        T_edges,
        hist,
        cmap=diff_cmap,
        norm=diff_norm,
        shading="auto",
    )

    ax.set_xlabel("Salinity (PSU)")
    if i == 0:
        ax.set_ylabel("Temperature (°C)")

    ax.set_title(label)
    ax.invert_xaxis()

# ----------------------------------------------------------
# Colorbars
# ----------------------------------------------------------

# Absolute volume (top row)
cax1 = fig.add_subplot(gs[0, 3])
mappable1 = cm.ScalarMappable(norm=norm, cmap=cmap)
mappable1.set_array([])
cbar1 = fig.colorbar(mappable1, cax=cax1)
cbar1.set_label("Volume (m³)")

# Difference volume (bottom row)
cax2 = fig.add_subplot(gs[1, 3])
mappable2 = cm.ScalarMappable(norm=diff_norm, cmap=diff_cmap)
mappable2.set_array([])
cbar2 = fig.colorbar(mappable2, cax=cax2)
cbar2.set_label("Δ Volume (m³)")


plt.savefig(f"unhinged_TS_{region}.png", dpi=300, bbox_inches="tight")
plt.show()