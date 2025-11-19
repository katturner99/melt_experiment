import datetime as dt
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from tools.directories_and_paths import OUTPUT_PATH
from tools.calcs import moving_average


# ----------------------------------------------------------------------------- #
#                                 CONFIGURATION                                 #
# ----------------------------------------------------------------------------- #

SCENARIO_COLORS = {
    "LENS": "palevioletred",
    "MELT_noS": "cornflowerblue",
    "MELT": "seagreen",
}

LINE_STYLES = ['-.', '--', ':']


# ----------------------------------------------------------------------------- #
#                             DATA LOADING UTILITIES                             #
# ----------------------------------------------------------------------------- #

def load_timeseries(
    filepath: Path,
    var_name: str,
    start: dt.datetime,
    end: dt.datetime,
    cutoff_year: int = 2005
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """
    Load a timeseries variable from a NetCDF file and crop to the requested range.

    Parameters
    ----------
    filepath : Path
        Path to the NetCDF file.
    var_name : str
        The dataset variable to load.
    start, end : datetime
        Date range for cropping.
    cutoff_year : int
        Year at which to split the plotted line.

    Returns
    -------
    time : np.ndarray of datetime64
    values : np.ndarray
    index_cutoff : int (index in time array)
    
    Returns None if the file is missing or invalid.
    """

    if not filepath.exists():
        print(f"⚠️ File not found: {filepath}")
        return None

    try:
        ds = xr.open_dataset(filepath)
    except Exception as exc:
        print(f"⚠️ Failed to open dataset {filepath}: {exc}")
        return None

    # Convert time axis to pandas datetime (handles cftime automatically)
    try:
        time = xr.conventions.decode_cf(ds).time.to_pandas()
    except Exception:
        time = xr.DataArray(ds.time).to_pandas()

    mask = (time >= start) & (time <= end)

    if mask.sum() == 0:
        print(f"⚠️ No data in range for {filepath}")
        return None

    values = ds[var_name].sel(time=mask).values
    time = time[mask].to_numpy()

    # Determine cutoff index
    index_cutoff = np.argmax(time.astype("datetime64[Y]").astype(int) + 1970 >= cutoff_year)

    return time, values, index_cutoff


# ----------------------------------------------------------------------------- #
#                                PLOTTING LOGIC                                 #
# ----------------------------------------------------------------------------- #

def plot_comparison(
    scenarios: List[str],
    ens_members: List[int],
    var: str,
    region_var: str,
    start_year: int = 1995,
    end_year: int = 2020,
    cutoff_year: int = 2005,
    window: int = 24,
) -> None:
    """
    Plot the ensemble comparison across scenarios.

    The first portion of each curve (before cutoff_year) is plotted in gray,
    while the remainder uses the scenario color.
    """

    start_date = dt.datetime(start_year, 1, 1)
    end_date = dt.datetime(end_year, 1, 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    for scenario in scenarios:
        for member in ens_members:

            # Build file path
            filename = f"{scenario}00{member}_{var}_timeseries.nc"
            filepath = Path(OUTPUT_PATH) / filename

            ts = load_timeseries(filepath, region_var, start_date, end_date, cutoff_year)

            if ts is None:
                continue

            time, values, idx_cut = ts

            # Smooth if enough data
            smoothed = moving_average(values, window) if len(values) >= window else values

            linestyle = LINE_STYLES[(member - 1) % len(LINE_STYLES)]

            # Pre-cutoff segment (gray)
            ax.plot(
                time[:idx_cut],
                smoothed[:idx_cut],
                color="gray",
                linewidth=2,
                linestyle=linestyle,
                alpha=0.5,
            )

            # Post-cutoff segment (scenario color)
            ax.plot(
                time[idx_cut:],
                smoothed[idx_cut:],
                color=SCENARIO_COLORS.get(scenario, "gray"),
                linewidth=2,
                linestyle=linestyle,
                alpha=0.9,
            )

    # Labels / formatting
    ax.set(
        xlabel="Time",
        ylabel=var,
        title=f"{region_var} Ensemble Comparison"
    )

    ax.grid(True, linestyle="--", alpha=0.2)

    # Legend: only show scenario colors, not each member
    handles = [
        plt.Line2D([0], [0], color=SCENARIO_COLORS[s], lw=3, label=s)
        for s in scenarios
    ]
    ax.legend(handles=handles, title="Scenario")

    plt.tight_layout()

    output_file = Path(f"{region_var}_timeseries.png")
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✅ Saved figure: {output_file}")

    plt.show()


# ----------------------------------------------------------------------------- #
#                                     MAIN                                      #
# ----------------------------------------------------------------------------- #

def main() -> None:
    scenarios = ["LENS", "MELT_noS"]
    ens_members = [2, 3, 4]
    var = "temperature"
    region_var = "theta_cont_shelf"

    plot_comparison(
        scenarios=scenarios,
        ens_members=ens_members,
        var=var,
        region_var=region_var,
        start_year=1990,
        end_year=2100,
    )


if __name__ == "__main__":
    main()
