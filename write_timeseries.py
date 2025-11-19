import sys
import logging
from pathlib import Path
import xarray as xr
import argparse
from tools.funcs import (
    get_available_months,
    create_timeseries_3d,
    create_timeseries_2d,
    create_melt,
)
from tools.directories_and_paths import OUTPUT_PATH, get_filepath
from tools.constants import regions


def setup_logging(log_file):
    """Configure logging to save messages to a file."""
    logging.basicConfig(
        filename=log_file,
        filemode="a",   # append mode
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )


def create_and_save_timeseries(
    sorted_months,
    filepath,
    region,
    out_dataset,
    variable,
):
    """Generate and append timeseries data for a region and variable."""

    func_map = {
        "temperature": create_timeseries_3d,
        "salt": create_timeseries_3d,
        "melt": create_melt,
        "etan": create_timeseries_2d,
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
            name="etan_{region}",
        ),
    }

    if variable not in func_map:
        raise ValueError(f"Unknown variable '{variable}'")

    func = func_map[variable]
    var = variable if variable != "etan" else "ETAN"
    timeseries_data = func(sorted_months, filepath, var=var, region=region)

    attrs = attr_map[variable]
    timeseries_data.name = attrs["name"].format(region=region)
    timeseries_data.attrs = {
        k: v.format(region=region) if "{region}" in v else v
        for k, v in attrs.items()
        if k != "name"
    }

    out_dataset[timeseries_data.name] = timeseries_data
    return out_dataset

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create time series for UaMITgcm coupled output."
    )
    parser.add_argument("scenario", help="Scenario (LENS, month, 1year, 5year)")
    parser.add_argument("ens_member", help="Ensemble member (1–9)")
    parser.add_argument(
        "variable", choices=["temperature", "salt", "etan", "melt"], help="Variable to process"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    filepath = get_filepath(args.scenario, args.ens_member)
    #filepath = Path(f"/data/oceans_output/shelf/kaight/archer2_mitgcm/{args.scenario}_LENS001_O")
    output_file = Path(OUTPUT_PATH) / f"{args.scenario}00{args.ens_member}_{args.variable}_timeseries.nc"
    log_file = Path(OUTPUT_PATH) / f"{args.scenario}00{args.ens_member}_{args.variable}_timeseries.log"
    setup_logging(log_file)

    logging.info(
        f"Starting timeseries script for scenario={args.scenario}, "
        f"ens_member={args.ens_member}, variable={args.variable}"
    )

    sorted_months = get_available_months(filepath)
    data_dir = filepath / "output"
    old_ds = None

    # Handle existing dataset
    if output_file.exists():
        print(output_file)
        print("Previous timeseries exists, appending new months...")
        old_ds = xr.open_dataset(output_file)
        last_timestamp = str(old_ds.time.dt.strftime("%Y%m").values[-1])
        new_months = [m for m in sorted_months if m > last_timestamp]

        if not new_months:
            logging.info("No new months to process. Timeseries is up-to-date.")
            sys.exit()

        sorted_months = new_months
        logging.info(f"Appending {len(new_months)} new months after {last_timestamp}")
    else:
        
        logging.info("No previous timeseries exists, starting from scratch.")

    # Compute new data
    out_dataset = xr.Dataset()
    regions_to_process = regions.keys() if args.variable in ["temperature", "salt", "melt"] else ["total"]

    for region in regions_to_process:
        logging.info(f"Processing region: {region}")
        out_dataset = create_and_save_timeseries(sorted_months, data_dir, region, out_dataset, args.variable)

    # Combine and save
    final_ds = xr.concat([old_ds, out_dataset], dim="time") if old_ds else out_dataset
    final_ds.to_netcdf(output_file)
    logging.info(f"Saved updated timeseries to {output_file}")


if __name__ == "__main__":
    main()
