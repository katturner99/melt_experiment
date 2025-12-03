import datetime
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import List
from tools.directories_and_paths import OUTPUT_PATH

# === Choose analysis mode ===
# Options: "melt" or "temperature"
MODE = "melt"  # <-- change this to "melt" if needed


def load_datasets(base_path: Path, filenames: List[str]) -> List[xr.Dataset]:
    """Load multiple xarray datasets from a list of filenames."""
    datasets = []
    for f in filenames:
        path = base_path / f
        ds = xr.open_dataset(path)
        datasets.append(ds)
    return datasets


def rolling_slope(da: xr.DataArray, window: int = 60) -> xr.DataArray:
    """Compute rolling slope (best-fit line) over a specified window (in months)."""
    t_years = (da.time.dt.year + (da.time.dt.dayofyear - 1) / 365.0).values
    slopes = np.full(len(da), np.nan)
    half = window // 2

    for i in range(half, len(da) - half):
        y = da.values[i - half:i + half]
        x = t_years[i - half:i + half]
        if np.any(np.isnan(y)):
            continue
        a, _ = np.polyfit(x, y, 1)
        slopes[i] = a

    return xr.DataArray(slopes, coords=da.coords, dims=da.dims, name=f"{da.name}_slope")


def plot_timeseries(
    datasets: List[xr.Dataset],
    labels: List[str],
    vars_to_plot: List[str],
    time: np.ndarray,
    start: datetime.date,
    end: datetime.date,
    window_years: int = 5,
    mode: str = "melt"
):
    """Plot time series and rolling slopes for multiple ensembles."""
    n_vars = len(vars_to_plot)
    fig, axes = plt.subplots(n_vars, 2, figsize=(12, 3 * n_vars), sharex=True)

    if mode == "melt":
        base_cmap = plt.cm.Blues
    else:
        base_cmap = plt.cm.Oranges
    truncated_cmap = mcolors.LinearSegmentedColormap.from_list(
        "trunc_Blues", base_cmap(np.linspace(0.1, 1, 256))
    )
    colors = [truncated_cmap(i) for i in np.linspace(0, 1, len(labels))]

    for row, var in enumerate(vars_to_plot):
        # Collect data arrays for this variable across all ensembles
        all_data = [ds[var] for ds in datasets]
        data_stack = xr.concat(all_data, dim="ensemble")
        ensemble_mean = data_stack.mean(dim="ensemble")

        for ds, label, color in zip(datasets, labels, colors):
            data = ds[var]

            # Plot each ensemble member
            axes[row, 0].plot(time, data, label=label, color=color, linewidth=1.5, alpha=0.6)

            # Rolling mean + slope
            roll = data.rolling(time=12, center=True).mean()
            slope = rolling_slope(roll, window=12 * window_years)
            axes[row, 1].plot(time, slope, label=label, color=color, linewidth=1.5, alpha=0.6)

        # === Add ensemble mean line ===
        axes[row, 0].plot(time, ensemble_mean, color="black", linewidth=3, label="Ensemble Mean")
        roll_mean = ensemble_mean.rolling(time=12, center=True).mean()
        slope_mean = rolling_slope(roll_mean, window=12 * window_years)
        axes[row, 1].plot(time, slope_mean, color="black", linewidth=3, label="Ensemble Mean")

        # === Titles, labels, formatting ===
        if mode == "melt":
            region = var.replace("melt_", "").replace("_", " ").title()
            axes[row, 0].set_title(f"{region} — Melt Timeseries")
            axes[row, 1].set_title(f"{region} — Local Slope ({window_years}-yr window)")
            axes[row, 0].set_ylabel("Melt rate (Gt/yr)")
            axes[row, 1].set_ylabel("Slope (Gt/yr²)")
        else:
            axes[row, 0].set_title("Temperature — θ")
            axes[row, 1].set_title(f"Temperature Trend ({window_years}-yr window)")
            axes[row, 0].set_ylabel("Temperature (°C)")
            axes[row, 1].set_ylabel("Slope (°C/yr)")

        for ax in axes[row]:
            ax.set_xlim([start, end])
            ax.grid(True, alpha=0.3)
            ax.axvspan(datetime.date(2000, 1, 1), datetime.date(2010, 1, 1),
                    color="purple", alpha=0.2, lw=0)
        axes[row, 1].axhline(0, color="grey", lw=1)

        ymin, ymax = axes[row, 1].get_ylim()
        limit = max(abs(ymin), abs(ymax))
        axes[row, 1].set_ylim(-limit, limit)
        #axes[row, 0].set_ylim(0, 1.2)


    axes[-1, 0].set_xlabel("Time")
    axes[-1, 1].set_xlabel("Time")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.subplots_adjust(bottom=0.15)

    fig.savefig(f"{mode}_timeseries.png")
    plt.show()


def main():
    """Main entry point for analysis."""
    base_path = Path(OUTPUT_PATH)
    exp = "LENS"
    # === Dynamically choose file and variable names ===
    if MODE == "melt":
        filenames = [f"{exp}{str(i).zfill(3)}_melt_timeseries.nc" for i in range(1, 11)]
        vars_to_plot = ["melt_pig", "melt_thwaites", "melt_abbot",
                        "melt_dotson_crosson", "melt_getz"]
    elif MODE == "temperature":
        filenames = [f"{exp}{str(i).zfill(3)}_temperature_timeseries.nc" for i in range(2, 11)]
        vars_to_plot = ["theta_cont_shelf", "theta_pig", "theta_thwaites", "theta_abbot",
                        "theta_dotson_crosson", "theta_getz"]
    else:
        raise ValueError("MODE must be either 'melt' or 'temperature'.")

    labels = [f"Ensemble {i}" for i in range(1, 11)]
    datasets = load_datasets(base_path, filenames)
    time = [datetime.datetime(t.year, t.month, t.day) for t in datasets[0].time.values]

    plot_timeseries(
        datasets=datasets,
        labels=labels,
        vars_to_plot=vars_to_plot,
        time=time,
        start=datetime.date(1970, 1, 1),
        end=datetime.date(2030, 1, 1),
        window_years=5,
        mode=MODE,
    )


if __name__ == "__main__":
    main()
