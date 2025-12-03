import numpy as np 
import xarray as xr 
import matplotlib.pyplot as plt 
import cartopy.crs as ccrs 
import cartopy.feature as cfeature 
from tools.directories_and_paths import OUTPUT_PATH 


def load_dataset(path: str) -> xr.Dataset: 
    """Safely load a NetCDF dataset using xarray.""" 
    return xr.open_dataset(path)
    
def setup_axis(ax, title: str, xc, yc): 
    """Configure map axis with coastlines, land, and proper extents.""" 
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1) 
    ax.coastlines(zorder=2) 
    #ax.set_xlim([-135, -86]) 
    ax.set_ylim([float(yc.min()), -63]) 
    ax.set_title(title) 

def plot_field(ax, ds, field, **kwargs): 
    """Plot a 2D field using pcolormesh.""" 
    print(field.values) 
    return ax.pcolormesh( 
        ds["XC"].values, 
        ds["YC"].values, 
        field.values, 
        transform=ccrs.PlateCarree(), 
        shading="auto", **kwargs ) 
    
def volume_weighted_mean(field, ds, dims):
    """Compute a volume-weighted mean over dims."""
    # full 3D cell volume
    dV = ds["rA"] * ds["drC"] * ds["hFacC"]

    # drop land
    field = field.where(ds["hFacC"] > 0)
    dV = dV.where(ds["hFacC"] > 0)

    return (field * dV).sum(dim=dims) / dV.sum(dim=dims)

def vertical_weighted_mean(field, ds, zrange):
    """Weighted vertical mean over depth using drC * hFacC."""
    field = field.sel(Z=slice(*zrange))
    h = ds["drF"] * ds["hFacC"]
    h = h.sel(Z=slice(*zrange))

    field = field.where(h > 0)
    h = h.where(h > 0)

    return (field * h).sum("Z") / h.sum("Z")


def main(): 
    years = [2015, 2065]
    for start_year in years:
        var = "THETA" 
        # === Filepaths === 
        melt_path = f"{OUTPUT_PATH}/old_timeseries/PAS_MELT_{start_year+1}-{start_year}_average.nc" 
        lens_path = f"{OUTPUT_PATH}/old_timeseries/PAS_LENS_{start_year+1}-{start_year}_average.nc" 
        
        ds_melt = load_dataset(melt_path) 
        ds_lens = load_dataset(lens_path) 
        
        # === Precompute time means === 
        melt_mean = ds_melt[var].mean(dim="time") 
        lens_mean = ds_lens[var].mean(dim="time") 
        diff_mean = lens_mean - melt_mean 
        
        # === Projection === 
        projection = ccrs.Miller(central_longitude=float(ds_lens.XC.mean())) 
        
        fig = plt.figure(figsize=(10, 15), dpi=200) 
        # Shift plots left + allow space for colorbars
        fig.subplots_adjust(
            left=0.05,
            right=0.80,   # More space for colorbars
            top=0.93,     # Allow suptitle to move lower
            bottom=0.05,
            hspace=0.01
        )


        ax1 = fig.add_subplot(3, 1, 1, projection=projection) 
        ax2 = fig.add_subplot(3, 1, 2, projection=projection) 
        ax3 = fig.add_subplot(3, 1, 3, projection=projection) 
        
        axes = [ax1, ax2, ax3] 
        if var == "THETA":
            print(np.nanmean(melt_mean.sel(Z=slice(-200, -700)))) 
            melt_mean = melt_mean.where(melt_mean.hFacC > 0)
            lens_mean = lens_mean.where(lens_mean.hFacC > 0)

            depth_range = (-200, -700)
            melt_mean = vertical_weighted_mean(melt_mean, ds_melt, depth_range)
            lens_mean = vertical_weighted_mean(lens_mean, ds_lens, depth_range)
            
            diff_mean = lens_mean - melt_mean 

            fields = [ 
                ("LENS", lens_mean, 2), 
                ("MELT", melt_mean, 2), 
                ("LENS – MELT", diff_mean, 0.75),
            ]
            
            cmap = "coolwarm"
        else:
            fields = [ 
                ("LENS", lens_mean, 0.0002), 
                ("MELT", melt_mean, 0.0002), 
                ("LENS – MELT", diff_mean, 0.00005),
            ]
            cmap="BrBG", 
        
        cax12 = fig.add_axes([0.87, 0.4, 0.02, 0.5])   # For ax1 + ax2
        cax3  = fig.add_axes([0.87, 0.1, 0.02, 0.24])   # For ax3

        for ax, (title, field, val) in zip(axes, fields):
            print(field.shape)
            mesh = ax.pcolormesh( 
                ds_lens.XC, 
                ds_lens.YC, 
                field, 
                transform=ccrs.PlateCarree(), 
                shading="auto", 
                cmap=cmap, 
                vmin=-val, 
                vmax=val, 
                ) 
            
            # --- Add coastlines and land --- 
            ax.add_feature(cfeature.LAND, facecolor="lightgray") 
            ax.coastlines() 
            
            # --- Set limits --- 
            ax.set_extent( 
                [-135, -86, float(ds_lens.YC.min()), -63], 
                crs=ccrs.PlateCarree(), 
                ) 
            
            # --- Add lat/lon ticks and labels --- 
            gl = ax.gridlines( 
                draw_labels=True, 
                x_inline=False, 
                y_inline=False, 
                linestyle="--", 
                linewidth=0.6, 
                ) 
            
            gl.top_labels = False 
            gl.right_labels = False 
            gl.xlabel_style = {"size": 8} 
            gl.ylabel_style = {"size": 8} 
            ax.set_title(title) 

            if title == "MELT":
                # shared colorbar for LENS+MELT
                cbar12 = fig.colorbar(mesh, cax=cax12)
                cbar12.set_label(f"{var} (degC)")

            if title == "LENS – MELT":
                # separate colorbar for DIFF
                cbar3 = fig.colorbar(mesh, cax=cax3)
                cbar3.set_label(f"{var} (degC)")

        plt.suptitle(f"{var} {start_year-10} - {start_year}", y=0.96, fontsize=20)
        fig.savefig(f"{var}_{start_year}.png")
        #plt.show()

if __name__ == "__main__":
    main()