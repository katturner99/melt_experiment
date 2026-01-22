import os
import sys
from pathlib import Path
import xarray as xr
from tools.funcs import (
    get_available_months,
    parse_args,
)
from tools.timeseries_funcs import (
    create_timeseries_3d,
    create_timeseries_2d,
    create_timeseries_vel,
    create_melt,
    create_timeseries_transport,
    create_timeseries_fw,
)
from tools.directories_and_paths import OUTPUT_PATH, get_filepath
from tools.constants import regions


def create_and_save_timeseries(
    sorted_years,
    filepath,
    region,
    out_dataset,
    variable,
):
    """ 
    Function that assigns the correct functions to generate and append 
    timeseries data for a region and variable.

    Parameters
    ----------
    sorted_years : list
                   list containing the years to sort through.
    filepath     : string
                   filepath where we are reading the data from.
    region       : dictionary 
                   region to process (lat and lon and depth limits).
    out_dataset  : xarray
                   xarray containing out timeseries.
    variable     : string
                   variable to process.

    Returns:
    --------
    out_dataset : xarray file containing the tiemseries.
    """

    func_map = {
        "temperature": create_timeseries_3d,
        "salt": create_timeseries_3d,
        "melt": create_melt,
        "etan": create_timeseries_2d,
        "undercurrent": create_timeseries_vel,
        "transport": create_timeseries_transport,
        "si_freezing": create_timeseries_fw,
        "si_melting": create_timeseries_fw,
        "fw_total": create_timeseries_fw,
    }

    attr_map = {
        "temperature": dict(
            standard_name="potential_temperature",
            long_name="Potential Temperature averaged over the {region} region",
            units="degC",
            name="theta_{region}",
        ),
        "salt": dict(
            standard_name="potential_salinity",
            long_name="Salinity averaged over the {region} region",
            units="psu",
            name="salt_{region}",
        ),
        "melt": dict(
            standard_name="total_melt",
            long_name="Melt over area.",
            units="Gt yr-1",
            name="melt_{region}",
        ),
        "etan": dict(
            standard_name="sea_surface_height",
            long_name="Mean sea surface height over the area.",
            units="m",
            name="etan",
        ),
        "undercurrent": dict(
            standard_name="sea_water_speed",
            long_name="Max along slope speed for the undercurrent areas",
            units="m s-1",
            name="speed",
        ),
        "transport": dict(
            standard_name="ocean_volume_y_transport",
            long_name="Southward transport across section",
            units="m3 s-1",
            name="southward_transport",
        ),
        "si_melting": dict(
            long_name="Freshwater Flux from sea ice melting",
            units="m s-1",
            name="si_melting",
        ),
         "si_freezing": dict(
            long_name="Freshwater Flux from sea ice freezing",
            units="m s-1",
            name="si_freezing",
        ),
         "fw_total": dict(
            long_name="Total Freshwater Flux at the Surface",
            units="m s-1",
            name="fw_total",
        ),
    }

    if variable not in func_map:
        raise ValueError(f"Unknown variable '{variable}'")

    func = func_map[variable]
    var = variable if variable != "etan" else "ETAN"
    timeseries_data = func(sorted_years, filepath, var=var, region=region)

    attrs = attr_map[variable]
    timeseries_data.name = attrs["name"].format(region=region)
    timeseries_data.attrs = {
        k: v.format(region=region) if "{region}" in v else v
        for k, v in attrs.items()
        if k != "name"
    }

    out_dataset[timeseries_data.name] = timeseries_data
    return out_dataset


def main():
    """
    Top level function to write a timeseries for a given scenario, 
    ensemble member number, variable.

    Parameters
    ----------
    scenario   : string
                 experiment to calculate (e.g.LENS, MELT, MELT_noS).
    ens_member : int
                 ensemble member number (e.g. 2, 3, 4).
    variable   : string
                 variable to create timeseries about (e.g. temperature).

    Returns:
    --------
    output_timeseries : netcdf file containing the timeseries.
    """

    # --- load data based on scenario, ensemble member number, variable ---
    args = parse_args()
    filepath = get_filepath(args.scenario, args.ens_member)
    output_filename = f"{args.scenario}00{args.ens_member}_{args.variable}_timeseries.nc"
    output_file = Path(OUTPUT_PATH) / output_filename

    # --- look for previous months / months to add to the timeseries ---
    sorted_months = get_available_months(filepath)
    data_dir = filepath / "output"
    old_ds = None

    # --- check for existing dataset ---
    if output_file.exists():
        print("Previous timeseries exists, appending new years...")
        old_ds = xr.open_dataset(output_file)
        last_timestamp = str(old_ds.time.dt.strftime("%Y%m").values[-1])
        print(f"Last timestamp at {last_timestamp}")
        new_months = [m for m in sorted_months if m >= last_timestamp]

        if not new_months:
            print("No new years to process. Timeseries is up-to-date.")
            sys.exit()

        sorted_months = new_months
        print(f"Appending {len(new_months)} new years after {last_timestamp}")
    
    # --- is nothing exists, create a new dataset ---
    else:
        print("No previous timeseries exists, starting from scratch.")

    # --- compute new timeseries ---
    out_dataset = xr.Dataset()
    regions_to_process = regions.keys() if args.variable in ["temperature", "salt", "melt"] else ["total"]

    for region in regions_to_process:
        out_dataset = create_and_save_timeseries(sorted_months, data_dir, region, out_dataset, args.variable)

    # --- combine and save ---
    final_ds = xr.concat([old_ds, out_dataset], dim="time") if old_ds else out_dataset
    if output_file.exists():
        os.remove(output_file)
    
    final_ds.to_netcdf(output_file)
    print(f"Saved updated timeseries to {output_file}")


if __name__ == "__main__":
    main()
