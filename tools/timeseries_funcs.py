from pathlib import Path
import numpy as np
import xarray as xr

from .funcs import load_month_dataset
from .constants import regions, depth_range, depth_limit, SV
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
    """
    ts = []
    base_dir = Path(base_dir)

    for m in months:
        ds = load_month_dataset(base_dir, m)
        mask = ds["hFacC"][0] > 0
        field = ds[var].where(mask)
        mean_val = field.mean(dim=["XC", "YC"])
        ts.append(mean_val)

    return xr.concat(ts, dim="time")


def create_timeseries_3d(months, base_dir, var = "THETA", region = "cont_shelf",
                         depth_range = depth_range):
    """
    Compute a volume-weighted mean 3-D timeseries.

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

    """
    base_dir = Path(base_dir)

    if var == "temperature": var = "THETA"
    elif var == "salt": var = "SALT"

    lat_rng, lon_rng = regions[region]
    ts = []

    for m in months:
        print(m)
        ds = load_month_dataset(base_dir, m)

        dV = ds["rA"] * ds["drF"] * ds["hFacC"]

        field = ds[var]

        if region == "cont_shelf":
            field = field.where(ds.Depth < depth_limit).sel(Z=slice(*depth_range), YC=slice(*lat_rng), XC=slice(*lon_rng))
            dV = dV.where(ds.Depth < depth_limit).sel(Z=slice(*depth_range), YC=slice(*lat_rng), XC=slice(*lon_rng))
        else: # if region in the cavities then compute the temperature over the entire water column
            field = field.where(ds.Depth < depth_limit).sel(YC=slice(*lat_rng), XC=slice(*lon_rng))
            dV = dV.where(ds.Depth < depth_limit).sel(YC=slice(*lat_rng), XC=slice(*lon_rng))
        
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

def create_timeseries_transport(months, base_dir, var=None, region=None):
    """
    Compute transport through PITT trough timeseries.

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
        Transport through trough timeseries.

    Notes
    -----
    - Geographic mask polygons are fixed based on prior work.
    """
    base_dir = Path(base_dir)
    ts = []

    for m in months:
        ds = load_month_dataset(base_dir, m)

        vel = ds.VVEL.sel(XC=slice(251, 254.5))[:,:,99,:]
        vel = vel.where(vel.hFacS != 0)
        vel = vel.where(vel <= 0)
        transport = SV * np.sum(-vel * vel.dxG * vel.drF, axis=(-2, -1))
        ts.append(transport)

    return xr.concat(ts, dim="time")


def create_timeseries_fw(months, base_dir, var=None, region=None):
    """
    Compute freshwater fluxtimeseries.

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
        Transport through trough timeseries.

    """
    base_dir = Path(base_dir)
    ts = []

    if var == "si_freezing":
        var_name = "SIfwfrz"
    elif var == "si_melting":
        var_name = "SIfwmelt"
    elif var == "fw_total":
        var_name = "oceFWflx"

    for m in months:
        ds = load_month_dataset(base_dir, m)

        data = ds[var_name]
        fwFlx = np.sum(data, axis=(-2, -1)) / 1000 # convert to m s-1
        ts.append(fwFlx)

    return xr.concat(ts, dim="time")