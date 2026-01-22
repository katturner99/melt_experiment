import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from tools.directories_and_paths import OUTPUT_PATH
from tools.constants import regions

lat_rng, lon_rng = regions["cont_shelf"]

CONTROL_DIR = f"{OUTPUT_PATH}PAS_LENS_2005_average_TS.nc"
MELT_DIR = f"{OUTPUT_PATH}PAS_MELT_2100_average_TS.nc"
LENS_DIR = f"{OUTPUT_PATH}PAS_LENS_2100_average_TS.nc"


def get_profile(var, ds):
    data = ds[var]
    dV = ds["rA"] * ds["drF"] * ds["hFacC"]

    data = data.sel(YC=slice(*lat_rng), XC=slice(*lon_rng))
    num = (data * dV).where(data["hFacC"] > 0).sum(dim=["YC", "XC"])
    den = dV.where(data["hFacC"] > 0).sum(dim=["YC", "XC"])
    return num / den


def seasonal_mean(da, season):
    """
    season: 'annual', 'summer', 'winter'
    """
    if season == "annual":
        return da.mean("time")

    months = {
        "summer": [12, 1, 2],  # DJF
        "winter": [6, 7, 8],   # JJA
    }

    return da.sel(time=da["time.month"].isin(months[season])).mean("time")


def main():
    control = xr.open_dataset(CONTROL_DIR)
    melt = xr.open_dataset(MELT_DIR)
    lens = xr.open_dataset(LENS_DIR)

    # --- profiles ---
    control_theta = get_profile("THETA", control)
    melt_theta = get_profile("THETA", melt)
    lens_theta = get_profile("THETA", lens)

    control_salt = get_profile("SALT", control)
    melt_salt = get_profile("SALT", melt)
    lens_salt = get_profile("SALT", lens)

    seasons = ["annual", "summer", "winter"]
    titles = ["Annual Mean", "Summer (DJF)", "Winter (JJA)"]

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(8, 8),
        sharey=True,
    )

    colors = {
        "Control": "limegreen",
        "MELT": "cornflowerblue",
        "LENS": "orchid",
    }

    lw = 2.5

    for j, season in enumerate(seasons):
        # --- Temperature ---
        ax = axes[0, j]
        ax.plot(
            seasonal_mean(control_theta, season),
            control_theta["Z"],
            label="Control",
            color=colors["Control"],
            lw=lw,
        )
        ax.plot(
            seasonal_mean(melt_theta, season),
            melt_theta["Z"],
            label="MELT",
            color=colors["MELT"],
            lw=lw,
        )
        ax.plot(
            seasonal_mean(lens_theta, season),
            lens_theta["Z"],
            label="LENS",
            color=colors["LENS"],
            lw=lw,
        )
        ax.set_ylim(0, -1000)
        ax.set_title(titles[j])
        ax.invert_yaxis()
        if j == 0:
            ax.set_ylabel("Depth (m)")
        ax.set_xlabel("Temperature (°C)")

        # --- Salinity ---
        ax = axes[1, j]
        ax.plot(
            seasonal_mean(control_salt, season),
            control_salt["Z"],
            label="Control",
            color=colors["Control"],
            lw=lw,
        )
        ax.plot(
            seasonal_mean(melt_salt, season),
            melt_salt["Z"],
            label="MELT",
            color=colors["MELT"],
            lw=lw,
        )
        ax.plot(
            seasonal_mean(lens_salt, season),
            lens_salt["Z"],
            label="LENS",
            color=colors["LENS"],
            lw=lw,
        )
        ax.set_ylim(0, -1000)
        ax.invert_yaxis()
        if j == 0:
            ax.set_ylabel("Depth (m)")
        ax.set_xlabel("Salinity (psu)")

    # Legend (single, shared)
    axes[0, 0].legend(loc="best")

    plt.tight_layout()
    plt.savefig("TS_profiles.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
