import numpy as np
import sys
import xarray as xr
import re
import os
from .directories_and_paths import *
from .constants import regions, depth_range

def get_available_months(filepath):
    """
    Find all valid months in the given directory that contain output.nc.
    """
    base_output_dir = os.path.join(filepath, "output")
    if not os.path.exists(base_output_dir):
        sys.exit(f"Stopped - Could not find output directory {base_output_dir}")
    
    month_dirs = []
    for entry in os.listdir(base_output_dir):
        full_path = os.path.join(base_output_dir, entry)
        if os.path.isdir(full_path) and re.fullmatch(r"\d{6}", entry):
            nc_file = os.path.join(full_path, "MITgcm", "output.nc")
            if os.path.isfile(nc_file):
                month_dirs.append(entry)
    
    # Sort the months chronologically
    sorted_months = sorted(month_dirs)
    return sorted_months


def create_timeseries_2d(sorted_months, filepath, var = "ETAN"):
    ts_list = []

    for month in sorted_months:
        data = xr.open_dataset(f"{filepath}/{month}/MITgcm/output.nc")
        print(f"{filepath}/{month}/MITgcm/output.nc")
 
        # mask out land
        mask = data["hFacC"][0,...] > 0
        data_masked = data[var].where(mask)
        
        var_mean = data_masked.mean(dim=["XC", "YC"])
        ts_list.append(var_mean[0,...])
        print(var_mean)
        print(ts_list)
        
    return xr.concat(ts_list, dim="time")

def create_melt(sorted_months, filepath, var = "melt", region = "cont_shelf", depth_range = depth_range):
    if var == "melt":
        var = "SHIfwFlx"

    ts_list = []
    lat_range, lon_range = regions[region]

    for month in sorted_months:
        data = xr.open_dataset(f"{filepath}/{month}/MITgcm/output.nc")
        print(f"{filepath}/{month}/MITgcm/output.nc")

        data_selected = data.sel(
            YC=slice(*lat_range),
            XC=slice(*lon_range)
        )

        melt_flux = data_selected[var] * 3600 * 24 * 365
    
        melt_total = 10 ** (-12) * np.sum((-melt_flux) * data.rA, axis=(-2, -1))
        ts_list.append(melt_total)
        
    return xr.concat(ts_list, dim="time")

def create_timeseries_3d(sorted_months, filepath, var = "THETA", region = "cont_shelf", depth_range = depth_range):
    if var == "temperature":
        var = "THETA"
    else:
        var = "SALT"
    ts_list = []
    lat_range, lon_range = regions[region]

    for month in sorted_months:
        data = xr.open_dataset(f"{filepath}/{month}/MITgcm/output.nc")
        print(f"{filepath}/{month}/MITgcm/output.nc")

        dV = (
            data["rA"] *  # (YC,XC)
            data["drF"] *  # (Z)
            data["hFacC"]  # (Z,YC,XC)
        )
        
        # mask out land
        mask = data["hFacC"] > 0
        data_masked = data[var].where(mask)
        print(np.nanmean(data[var]))
        print(np.nanmean(data_masked))

        data_masked = data_masked.sel(
            Z=slice(*depth_range),
            YC=slice(*lat_range),
            XC=slice(*lon_range)
        )

        dV = dV.sel(
            Z=slice(*depth_range),
            YC=slice(*lat_range),
            XC=slice(*lon_range)
        )
        
        # --- Volume-weighted mean ---
        num = (data_masked * dV).sum(dim=["Z", "YC", "XC"])
        den = dV.sum(dim=["Z", "YC", "XC"])
        theta_mean = num / den
        ts_list.append(theta_mean)
        
        
    return xr.concat(ts_list, dim="time")