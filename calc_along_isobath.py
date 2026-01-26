import xarray as xr
from xhistogram.xarray import histogram
import numpy as np

from tools.calcs import calc_density, haversine
from tools.directories_and_paths import OUTPUT_PATH

# open dataset, extract lat and lon and calculate density
ds = xr.open_dataset(f"{OUTPUT_PATH}PAS_LENS_2100_average_TS.nc")
lat = ds.YC
lon = ds.XC - 360
rho = calc_density(ds)
rho = rho.mean(dim="time")
volume = ds.rA * ds.drF * ds.hFacC

# geographic mask for the latitudinal band. (obtained from eyeballing it lol)
region_mask = (
    (
        (((lat > -72.75) & (lat < -70)) & ((lon > -124.6) & (lon < -120)))   |
        (((lat > -71.90) & (lat < -70)) & ((lon > -118)   & (lon < -114.5))) |
        (((lat > -71.35) & (lat < -70)) & ((lon > -112)   & (lon < -108)))   |
        (((lat > -71.17) & (lat < -70)) & ((lon > -108)   & (lon < -107.1)))
    )
)

region_mask_just_lons = (
    (
        ((lon > -124.6) & (lon < -120))   |
        ((lon > -118)   & (lon < -114.5)) |
        ((lon > -112)   & (lon < -108))   |
        ((lon > -108)   & (lon < -107.1))
    )
)

# find the 1000m isobath +-25m
iso_tol = 25 
isobath_mask = (np.abs(ds.Depth - 1000) <= iso_tol) & region_mask # find the 1000m isobath within the region mask
iso_idx = np.where(isobath_mask.values)       # the index for where the isobath is
iso_lats = ds.YC.values[iso_idx[0]]           # lats of the isobath
iso_lons = (ds.XC.values[iso_idx[1]] - 360)   # lons of the isobath

# calculate the distance from the isobath
lat2d = ds.YC.values[:, None]         # [lat, 1]
lon2d = ds.XC.values[None, :] - 360   # [1, lon]

dist_to_iso = np.full(ds.Depth.shape, np.inf) # initialise the mask to be inf values

for lat_i, lon_i in zip(iso_lats, iso_lons): # run through the isobath lat and lons
    # apply haversine to calculate the distance from points
    d = haversine(lat2d, lon2d, lat_i, lon_i)
    # each time save the shortest distances for each point
    dist_to_iso = np.minimum(dist_to_iso, d)

dist_to_iso = xr.DataArray(
    dist_to_iso,
    dims=("YC", "XC"),
    coords={"YC": ds.YC, "XC": ds.XC}
)

# mask out anything above 250km (copying from Haigh et al. 2025)
iso_band_mask = dist_to_iso <= 250

# mask according to the lons and based off of distance from the isobath
final_mask = (
    iso_band_mask &
    region_mask_just_lons
)

hFacC_mask = ds.hFacC > 0

final_mask = hFacC_mask & final_mask 

# save final mask as dataset
mask = xr.DataArray(
    final_mask,
    dims=("Z", "YC", "XC"),
    coords={"Z":ds.Z, "YC": lat, "XC": lon + 360}
)

# mask rho <3
rho_masked = rho.where(mask)
rho_masked = rho_masked.rename("rho_masked")

#___________________________________________________________________________
# now reformat the data so that is is relative to the 1000m isobath
# figure out where this is in relation to the isobath (north or south)
signed_dist = dist_to_iso.copy()
# you already have your mask, but change it so where depth decreases is - and depth increases is +
signed_dist = xr.where(
    ds.Depth < 1000,
    -signed_dist,
    signed_dist
)
signed_dist = signed_dist.broadcast_like(rho_masked)
signed_dist = signed_dist.where(final_mask)
signed_dist = signed_dist.transpose("Z", "YC", "XC")
signed_dist = signed_dist.rename("dist_to_isobath")

volume_masked = volume.where(final_mask)
volume_masked = volume_masked.rename("cell_volume")
volume_masked = volume_masked.transpose("Z", "YC", "XC")
signed_dist = signed_dist.transpose("Z", "YC", "XC")

# create new grid for the data following 
# https://cosima-recipes.readthedocs.io/en/latest/03-Mains/Along_Isobath_Average.html 
bin_width = 5  # km
bins_dist = np.arange(-250, 250 + bin_width, bin_width)

rhoV_sum = histogram(
    signed_dist,
    bins=[bins_dist],
    dim=["YC", "XC"],
    weights=volume_masked * rho_masked,
    block_size=int(1e6)
)

V_sum = histogram(
    signed_dist,
    bins=[bins_dist],
    dim=["YC", "XC"],
    weights=volume_masked,
    block_size=int(1e6)
)

rho_mean = rhoV_sum / V_sum

rho_mean = rho_mean.rename("density_along_isobath")

rho_mean.to_netcdf("LENS_2100_density_along_isobath.nc")

