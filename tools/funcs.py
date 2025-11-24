from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import xarray as xr

from .constants import regions, depth_range
from .calcs import calc_density, compute_along_isobath_velocity

# -----------------------------------------------------------------------------
# Generic utilities
# -----------------------------------------------------------------------------

def get_available_months(base_dir: str | Path) -> list[str]:
    """Return sorted list of YYYYMM directories containing MITgcm/output.nc.

    Parameters
    ----------
    base_dir : str or Path
        The simulation root directory containing an `output/` folder.
    """
    base_dir = Path(base_dir)
    output_root = base_dir / "output"

    if not output_root.exists():
        raise FileNotFoundError(f"Output directory not found: {output_root}")

    months = []

    for entry in output_root.iterdir():
        if entry.is_dir() and re.fullmatch(r"\d{6}", entry.name):
            nc_file = entry / "MITgcm" / "output.nc"
            if nc_file.is_file():
                months.append(entry.name)

    return sorted(months)


# -----------------------------------------------------------------------------
# Timeseries utilities (2D, 3D, velocity, melt, etc.)
# -----------------------------------------------------------------------------

def load_month_dataset(base_dir: str | Path, month: str) -> xr.Dataset:
    """Load a MITgcm output.nc file for a given YYYYMM string."""
    path = Path(base_dir) / month / "MITgcm" / "output.nc"
    if not path.is_file():
        raise FileNotFoundError(f"Missing dataset: {path}")
    return xr.open_dataset(path)


def create_timeseries_2d(months: list[str], base_dir: str | Path,
                          var: str = "ETAN") -> xr.DataArray:
    """Return a timeseries (time, ) of domain‑mean 2‑D variable values."""
    ts = []
    base_dir = Path(base_dir)

    for m in months:
        ds = load_month_dataset(base_dir, m)
        mask = ds["hFacC"][0] > 0
        field = ds[var].where(mask)
        mean_val = field.mean(dim=["XC", "YC"])
        ts.append(mean_val.squeeze())

    return xr.concat(ts, dim="time")


def create_melt(months: list[str], base_dir: str | Path,
                var: str = "melt", region: str = "cont_shelf",
                depth_range: tuple[int, int] = depth_range) -> xr.DataArray:
    """Return melt timeseries (time, ).

    Notes
    -----
    - Converts SHIfwFlx from m³/s to Gt/yr (10^-12 factor and sign convention).
    """
    base_dir = Path(base_dir)
    if var == "melt":
        var = "SHIfwFlx"

    lat_rng, lon_rng = regions[region]
    ts = []

    for m in months:
        ds = load_month_dataset(base_dir, m)

        sub = ds.sel(YC=slice(*lat_rng), XC=slice(*lon_rng))
        melt_flux = sub[var] * 3600 * 24 * 365

        # Negative melt flux corresponds to melting; convert to Gt/yr
        melt_total = 1e-12 * np.sum((-melt_flux) * ds.rA, axis=(-2, -1))
        ts.append(melt_total)

    return xr.concat(ts, dim="time")


def create_timeseries_vel(months: list[str], base_dir: str | Path) -> xr.DataArray:
    """Compute undercurrent velocity timeseries.

    The masking criteria follow the original script but could be further
    parameterized if needed.
    """
    base_dir = Path(base_dir)
    ts = []

    for m in months:
        ds = load_month_dataset(base_dir, m)

        lat = ds.YC
        lon = ds.XC
        rho = calc_density(ds)
        rho_mask = rho >= 1028
        depth_mask = ds.Z <= 800

        # Geographic mask for the latitudinal band (hard‑coded polygons).
        lat_band_mask = (
            ((ds.Depth > 500) & (ds.Depth < 1500)) & (
                (((lat > -72.75) & (lat < -70)) & ((lon > -124.6) & (lon < -120))) |
                (((lat > -71.9)  & (lat < -70)) & ((lon > -118)   & (lon < -114.5))) |
                (((lat > -71.35) & (lat < -70)) & ((lon > -112)   & (lon < -108))) |
                (((lat > -71.17) & (lat < -70)) & ((lon > -108)   & (lon < -107.1)))
            )
        )

        mask_loc = xr.DataArray(
            lat_band_mask,
            dims=("YC", "XC"),
            coords={"YC": lat, "XC": lon},
        )
        mask = mask_loc & rho_mask & depth_mask

        vel = compute_along_isobath_velocity(ds)
        zonal = vel.where(mask).mean(dim="XC")
        undercurrent = zonal.min(dim="Z").mean(dim="YC")
        ts.append(undercurrent)

    return xr.concat(ts, dim="time")


def create_timeseries_3d(months: list[str], base_dir: str | Path,
                         var: str = "THETA", region: str = "cont_shelf",
                         depth_range: tuple[int, int] = depth_range) -> xr.DataArray:
    """Volume‑weighted mean 3‑D tracer timeseries.

    Accepts MITgcm variable names or user‑friendly ones like "temperature".
    """
    base_dir = Path(base_dir)

    if var == "temperature":
        var = "THETA"
    elif var == "salinity":
        var = "SALT"

    lat_rng, lon_rng = regions[region]
    ts = []

    for m in months:
        ds = load_month_dataset(base_dir, m)

        dV = ds["rA"] * ds["drF"] * ds["hFacC"]

        field = ds[var].where(ds["hFacC"] > 0)

        field = field.sel(Z=slice(*depth_range), YC=slice(*lat_rng), XC=slice(*lon_rng))
        dV = dV.sel(Z=slice(*depth_range), YC=slice(*lat_rng), XC=slice(*lon_rng))

        num = (field * dV).sum(dim=["Z", "YC", "XC"])
        den = dV.sum(dim=["Z", "YC", "XC"])
        ts.append(num / den)

    return xr.concat(ts, dim="time")
