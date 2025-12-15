from pathlib import Path
import numpy as np
import xarray as xr

from .funcs import load_month_dataset
from .constants import regions, depth_range, depth_limit
from .calcs import calc_density, compute_along_isobath_velocity


def create_timeseries_2d(months, base_dir, var):
    """
    Compute a domain-mean timeseries for a 2-D MITgcm field.

    Each month’s `output.nc` is loaded, the land mask is applied using
    ``hFacC``, and the spatial mean over ``XC`` and ``YC`` is computed.

    Parameters
    ----------
    months : list of str
        List of YYYYMM strings.
    base_dir : str or Path
        Base directory containing the monthly output subdirectories.
    var : str
        Variable name to extract from each dataset.

    Returns
    -------
    xarray.DataArray
        Timeseries of spatial means with dimension ``time``.

    Raises
    ------
    KeyError
        If ``var`` is not found in a dataset.

    Notes
    -----
    Assumes ``ds[var]`` has dimensions ``(YC, XC)`` or ``(time, YC, XC)``.
    """
    ts = []
    base_dir = Path(base_dir)

    for m in months:
        ds = load_month_dataset(base_dir, m)
        mask = ds["hFacC"][0] > 0
        field = ds[var].where(mask)
        mean_val = field.mean(dim=["XC", "YC"])
        ts.append(mean_val.squeeze())

    return xr.concat(ts, dim="time")


def create_timeseries_3d(months, base_dir, var = "THETA", region = "cont_shelf",
                         depth_range = depth_range):
    """
    Compute a volume-weighted mean 3-D tracer timeseries.

    The volume mean is computed over a spatial subset set by the regions 
    and the depth limit defined in constants.py. The mean is weighted
    using ``rA * drF``. 

    Parameters
    ----------
    months : list of str
        List of YYYYMM strings.
    base_dir : str or Path
        Directory containing monthly MITgcm output.
    var : str, optional
        MITgcm variable name or user-friendly alias. One of:
        ``"THETA"``, ``"SALT"``, ``"temperature"``, ``"salt"``.
    region : str, optional
        Named region key in ``constants.regions``.
    depth_range : tuple, optional
        Depth range (min_depth, max_depth) to select.

    Returns
    -------
    xarray.DataArray
        Volume-weighted tracer mean for each month.

    Notes
    -----
    - Uses masking based on ``hFacC`` to remove land.
    - Applies global ``depth_limit`` from constants.

    Potential Issues
    ----------------
    - ``ds.Depth`` is nonstandard; typical MITgcm output uses ``Z``.  
      If ``Depth`` is missing, this will error.
    - ``hFacC`` is 3-D but you apply it on ``field["hFacC"]`` — correct.
    """
    base_dir = Path(base_dir)

    if var == "temperature": var = "THETA"
    elif var == "salt": var = "SALT"

    lat_rng, lon_rng = regions[region]
    ts = []

    for m in months:
        ds = load_month_dataset(base_dir, m)

        dV = ds["rA"] * ds["drF"] * ds["hFacC"]

        field = ds[var]

        field = field.where(ds.Depth < depth_limit).sel(Z=slice(*depth_range), YC=slice(*lat_rng), XC=slice(*lon_rng))
        dV = dV.where(ds.Depth < depth_limit).sel(Z=slice(*depth_range), YC=slice(*lat_rng), XC=slice(*lon_rng))

        num = (field * dV).where(field["hFacC"] > 0).sum(dim=["Z", "YC", "XC"])
        den = dV.where(field["hFacC"] > 0).sum(dim=["Z", "YC", "XC"])
        ts.append(num / den)

    return xr.concat(ts, dim="time")


def create_melt(months, base_dir, var = "melt", region = "cont_shelf"):
    """
    Compute a meltwater flux timeseries (Gt/yr).

    Converts ``SHIfwFlx`` from m³/s to Gt/yr and integrates spatially
    over the chosen region.

    Parameters
    ----------
    months : list of str
        List of YYYYMM strings.
    base_dir : str or Path
        Directory containing MITgcm monthly output.
    var : str, optional
        `"melt"` (alias) or explicit `"SHIfwFlx"`.
    region : str, optional
        Region key defined in ``constants.regions``.

    Returns
    -------
    xarray.DataArray
        Melt flux timeseries in gigatonnes per year.

    Notes
    -----
    - Conversion uses 1e-12 factor to convert m³/s → Gt/yr.
    - Negative SHIfwFlx corresponds to melting.

    Possible Errors
    ---------------
    - Uses ``np.sum`` directly on DataArray; works, but
      ``.sum(dim=...)`` is preferred.
    """
    base_dir = Path(base_dir)
    if var == "melt":
        var = "SHIfwFlx"

    lat_rng, lon_rng = regions[region]
    ts = []

    for m in months:
        ds = load_month_dataset(base_dir, m)

        if region == "cont_shelf":
            sub = ds
        else:
            sub = ds.sel(YC=slice(*lat_rng), XC=slice(*lon_rng))
        melt_flux = (sub[var] * 3600 * 24 * 365)
        area = sub.rA

        melt_total = 1e-12 * ((-melt_flux) * area).sum(dim=("YC", "XC"))
        ts.append(melt_total)

    return xr.concat(ts, dim="time")



def create_timeseries_vel(months, base_dir, var=None, region=None):
    """
    Compute an undercurrent velocity timeseries.

    A density mask, depth mask, and hard-coded geographic mask define the
    region of the undercurrent. The minimum zonal average velocity along
    isobaths is extracted for each month.

    Parameters
    ----------
    months : list of str
        List of YYYYMM strings.
    base_dir : str or Path
        Directory containing MITgcm monthly outputs.
    var : unused
        Placeholder for compatibility; velocity is computed internally.
    region : unused
        Placeholder for future generalization.

    Returns
    -------
    xarray.DataArray
        Undercurrent velocity timeseries.

    Notes
    -----
    - Uses density threshold ``rho >= 1028``.
    - Uses depth mask ``Z <= 800``.
    - Geographic mask polygons are fixed based on prior work.

    Possible Issues
    ---------------
    - ``ds.Depth`` is assumed to exist (not standard).
    - In mask construction, ``lon = ds.XC - 360`` but coords are restored
      as ``XC = lon + 360`` — works, but confusing.
    """
    base_dir = Path(base_dir)
    ts = []

    for m in months:
        ds = load_month_dataset(base_dir, m)

        lat = ds.YC
        lon = ds.XC - 360
        rho = calc_density(ds)
        rho_mask = rho >= 1028
        depth_mask = ds.Z <= 800

        # Geographic mask for the latitudinal band (hard‑coded polygons).
        lat_band_mask = (
            ((ds.Depth > 500) & (ds.Depth < 1500)) &
            ((((lat > -72.75) & (lat < -70)) & ((lon > -124.6) & (lon < -120))) |
             (((lat > -71.90) & (lat < -70)) & ((lon > -118) & (lon < -114.5))) |
             (((lat > -71.35) & (lat < -70)) & ((lon > -112) & (lon < -108))) |
             (((lat > -71.17) & (lat < -70)) & ((lon > -108) & (lon < -107.1)))
            )
        )
        
        mask_loc = xr.DataArray(
            lat_band_mask,
            dims=("YC", "XC"),
            coords={"YC": lat, "XC": lon+360},
        )
        mask = mask_loc & rho_mask & depth_mask
       
        vel = compute_along_isobath_velocity(ds)
        zonal = vel.where(mask).mean(dim="XC")
        undercurrent = zonal.min(dim="Z").mean(dim="YC")
        ts.append(undercurrent)

    return xr.concat(ts, dim="time")


