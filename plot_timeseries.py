import datetime as dt
from pathlib import Path
import argparse
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from tools.directories_and_paths import OUTPUT_PATH
from tools.calcs import moving_average


SCENARIO_COLORS = {
    "LENS": "magenta",
    "MELT": "forestgreen",
    "MELT_noS": "deepskyblue",
    "MELT_old": "orange",
}


def variable_title(var, region_var):
    """Return a human-readable plot title."""
    titles = {
        "temperature": f"Potential Temperature (degC) ({region_var})",
        "salt": f"Salinity ({region_var})",
        "undercurrent": "Undercurrent Speed (m/s)",
        "transport": "Transport through trough at 73S (Sv)"
    }
    return titles.get(var, region_var)


def load_timeseries(
    filepath,
    var_name,
    start,
    end,
):
    """Load and time-crop a NetCDF timeseries."""
    if not filepath.exists():
        print(f"⚠️ Missing file: {filepath}")
        return None

    with xr.open_dataset(filepath) as ds:
        time = [dt.datetime(t.year, t.month, t.day) for t in ds.time.values]
        values = ds[var_name].values

    
    mask = [(start <= t < end) for t in time]
    time = [t for t, keep in zip(time, mask) if keep]
    values = values[mask]

    if var_name == "speed":
        values = -values

    return time, values


def save_figure(fig, filename):
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"✅ Saved figure: {filename}")


def plot_comparison(
    scenarios,
    ens_members,
    var,
    region_var,
    start_year,
    end_year,
    window=24,
):

    start, end = dt.datetime(start_year, 2, 1), dt.datetime(end_year, 1, 1)
    fig, ax = plt.subplots(figsize=(10, 6))

    for scenario in scenarios:
        all_values = []
        time_ref = None

        for member in ens_members:
            filepath = Path(OUTPUT_PATH) / f"{scenario}00{member}_{var}_timeseries.nc"
            ts = load_timeseries(filepath, region_var, start, end)
            if ts is None:
                continue

            time, values = ts


            values = moving_average(values, window)

            if time_ref is None:
                time_ref = time
            else:
                # safety check: skip members with mismatched time axis
                if len(values) != len(time_ref):
                    print("mangos!")
                    # after: time, values = ts

                    # --- check for missing years ---
                    years_present = np.unique([t.year for t in time])
                    expected_years = np.arange(start_year, end_year + 1)

                    missing_years = np.setdiff1d(expected_years, years_present)

                    if len(missing_years) > 0:
                        print(
                            f"Missing years for scenario={scenario}, member={member}: "
                            f"{missing_years.tolist()}"
                        )
                    # --- end check ---

                    continue

            all_values.append(values)

        if len(all_values) == 0:
            continue

        ens = np.vstack(all_values)  # shape: (n_member, n_time)

        ens_min = np.nanmin(ens, axis=0)
        ens_max = np.nanmax(ens, axis=0)
        ens_mean = np.nanmean(ens, axis=0)

        color = SCENARIO_COLORS.get(scenario, "gray")

        # shaded envelope
        ax.fill_between(
            time_ref,
            ens_min,
            ens_max,
            color=color,
            alpha=0.25,
            linewidth=0,
        )

        # bold ensemble mean
        ax.plot(
            time_ref,
            ens_mean,
            color=color,
            linewidth=3,
            label=scenario,
        )

    ax.set(
        title=variable_title(var, region_var),
        xlabel="Time",
        ylabel=var,
    )
    ax.grid(True, linestyle="--", alpha=0.2)

    ax.legend(title="Scenario")
    plt.tight_layout()
    save_figure(fig, f"{region_var}_timeseries.png")



def plot_difference(
    scenarios,
    ens_members,
    var,
    region_var,
    start_year,
    end_year,
    window=48,
):

    start, end = dt.datetime(start_year, 1, 1), dt.datetime(end_year, 1, 1)
    fig, ax = plt.subplots(figsize=(10, 6))

    # only plot differences relative to LENS
    diff_scenarios = [s for s in scenarios if s != "LENS"]

    for scenario in diff_scenarios:
        all_diffs = []
        time_ref = None

        for member in ens_members:
            fp_scn = Path(OUTPUT_PATH) / f"{scenario}00{member}_{var}_timeseries.nc"
            fp_ref = Path(OUTPUT_PATH) / f"LENS00{member}_{var}_timeseries.nc"

            ts_scn = load_timeseries(fp_scn, region_var, start, end)
            ts_ref = load_timeseries(fp_ref, region_var, start, end)
            if ts_scn is None or ts_ref is None:
                continue

            time, scn = ts_scn
            _, ref = ts_ref

            n = min(len(scn), len(ref))
            print(n)
            
            diff = scn[:n] - ref[:n] 
            diff = moving_average(diff, window)

            if time_ref is None:
                time_ref = time[:n]
            else:
                if len(diff) != len(time_ref):
                    continue

            all_diffs.append(diff)

        if len(all_diffs) == 0:
            continue

        ens = np.vstack(all_diffs)

        ens_min = np.nanmin(ens, axis=0)
        ens_max = np.nanmax(ens, axis=0)
        ens_mean = np.nanmean(ens, axis=0)

        color = SCENARIO_COLORS.get(scenario, "gray")

        # shaded min–max envelope
        ax.fill_between(
            time_ref,
            ens_min,
            ens_max,
            color=color,
            alpha=0.25,
            linewidth=0,
        )

        # bold ensemble mean
        ax.plot(
            time_ref,
            ens_mean,
            color=color,
            linewidth=3,
            label=scenario,
        )

    ax.set(
        title=f"Difference MELT - LENS ({region_var})",
        xlabel="Time",
        ylabel=var,
    )
    ax.grid(True, linestyle="--", alpha=0.2)

    ax.legend(title="Scenario")
    plt.tight_layout()
    save_figure(fig, f"{region_var}_timeseries_difference.png")



# ----------------------------------------------------------------------------- #
#                                    MAIN                                       #
# ----------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot ensemble timeseries and differences from MITgcm output."
    )

    parser.add_argument(
        "--var",
        type=str,
        required=True,
        help="Variable name used in filenames (e.g. temperature, salt, undercurrent)",
    )

    parser.add_argument(
        "--region-var",
        type=str,
        required=True,
        help="Variable name inside NetCDF file (e.g. theta_pig)",
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=2006,
        help="Start year for plotting (default: 2006)",
    )

    parser.add_argument(
        "--end-year",
        type=int,
        default=2100,
        help="End year for plotting (default: 2100)",
    )

    args = parser.parse_args()

    scenarios = ["LENS", "MELT", "MELT_noS"]
    ens_members = [2, 3, 4, 5, 6]

    plot_comparison(
        scenarios=scenarios,
        ens_members=ens_members,
        var=args.var,
        region_var=args.region_var,
        start_year=args.start_year,
        end_year=args.end_year,
    )

    plot_difference(
        scenarios=scenarios,
        ens_members=ens_members,
        var=args.var,
        region_var=args.region_var,
        start_year=args.start_year,
        end_year=args.end_year,
    )
