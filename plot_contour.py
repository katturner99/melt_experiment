import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from tools.directories_and_paths import OUTPUT_PATH, GRID
from mitgcm_python.interpolation import interp_to_depth
from mitgcm_python.plot_utils.latlon import prepare_vel


# ========================
# CONFIG CLASSES: 
# ========================


@dataclass
class PlotConfig:
    """Configuration for plot appearance."""
    cmap: str = "coolwarm"
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    extent: Optional[List[float]] = None
    title: Optional[str] = None
    unit: str = "units"


@dataclass
class VelocityConfig:
    """Configuration for velocity plotting."""
    cmap: str = "cool"
    quiver_step: int = 10
    quiver_scale: float = 0.01
    quiver_width: float = 0.0035
    quiver_color: str = "k"


@dataclass
class ExperimentConfig:
    """Configuration for experiment comparison."""
    exp1_name: str
    exp2_name: str
    members: List[str]
    var_name: str
    depth: float
    output_path: str = OUTPUT_PATH


# ========================
# Data Loading
# ========================

class EnsembleDataLoader:
    """Handles loading and processing of ensemble data."""
    
    def __init__(self, output_path: str = OUTPUT_PATH):
        self.output_path = output_path
        self.sample_data = xr.open_dataset(
            f"{output_path}PAS_MELT002_S/output/210001/MITgcm/output.nc"
        )
    
    def read_mask(self, ds: xr.Dataset, hfac: str) -> xr.Dataset:
        """Apply mask based on hFac field."""
        return ds.where(self.sample_data[hfac] != 0)
    
    def open_ensemble_mean(self, nc_files: List[str]) -> xr.Dataset:
        """Open multiple ensemble netCDFs and return ensemble-mean dataset."""
        dsets = []
        for i, f in enumerate(nc_files):
            ds = xr.open_dataset(f)
            ds = ds.expand_dims(ensemble=[i])
            dsets.append(ds)
        
        ds_ens = xr.concat(dsets, dim="ensemble")
        ds_mean = ds_ens.mean("ensemble")
        return ds_mean
    
    def get_scalar_trend_at_depth(
        self,
        nc_files: List[str],
        depth: float,
        var_name: str = "trend"
    ) -> Tuple[xr.Dataset, xr.DataArray]:
        """Return ensemble-mean trend slice at a specified depth."""
        ds = self.open_ensemble_mean(nc_files)
        data = self.read_mask(ds, "hFacC")
        trend_slice = interp_to_depth(data[var_name], z0=depth, grid=GRID)
        return ds, trend_slice
    
    def get_velocity_trend_at_depth(
        self,
        exp_name: str,
        members: List[str],
        depth: float,
        scale_factor: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load and prepare ensemble-mean velocity data for a given experiment."""
        nc_files_uvel = [
            f"{self.output_path}/{exp_name}{m}_UVEL_trend.nc"
            for m in members
        ]
        nc_files_vvel = [
            f"{self.output_path}/{exp_name}{m}_VVEL_trend.nc"
            for m in members
        ]
        
        ds_uvel = self.open_ensemble_mean(nc_files_uvel)
        ds_vvel = self.open_ensemble_mean(nc_files_vvel)
        
        uvel = self.read_mask(ds_uvel.trend, "hFacW")
        vvel = self.read_mask(ds_vvel.trend, "hFacS")
        
        uvel = np.ma.masked_invalid(uvel.values) * scale_factor
        vvel = np.ma.masked_invalid(vvel.values) * scale_factor
        
        speed, u, v = prepare_vel(uvel, vvel, GRID, vel_option="interp", z0=depth)
        return speed, u, v


# ========================
# Plotting Utilities
# ========================

class MapPlotter:
    """Handles map-based plotting operations."""
    
    @staticmethod
    def setup_map(ax, extent: Optional[List[float]] = None, title: Optional[str] = None):
        """Configure a map axis with coastlines, land, and gridlines."""
        ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
        ax.coastlines(zorder=2)
        
        if extent:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.5,
            color='gray',
            alpha=0.7,
            linestyle='--'
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 10}
        gl.ylabel_style = {"size": 10}
        
        if title:
            ax.set_title(title)
    
    @staticmethod
    def plot_field(
        ax,
        field: xr.DataArray,
        xc: xr.DataArray,
        yc: xr.DataArray,
        config: PlotConfig
    ):
        """Plot a 2D field as pcolormesh on the map."""
        mesh = ax.pcolormesh(
            xc,
            yc,
            field,
            transform=ccrs.PlateCarree(),
            shading="auto",
            cmap=config.cmap,
            vmin=config.vmin,
            vmax=config.vmax,
        )
        return mesh
    
    @staticmethod
    def subsample_quiver(
        X1d: np.ndarray,
        Y1d: np.ndarray,
        U: np.ndarray,
        V: np.ndarray,
        step: int = 8
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Subsample velocity fields for cleaner quiver plots."""
        X, Y = np.meshgrid(X1d, Y1d)
        return (
            X[::step, ::step],
            Y[::step, ::step],
            U[::step, ::step],
            V[::step, ::step],
        )
    
    @staticmethod
    def add_quiver(
        ax,
        xc: xr.DataArray,
        yc: xr.DataArray,
        u: np.ndarray,
        v: np.ndarray,
        config: VelocityConfig
    ):
        """Add quiver plot to axis."""
        Xq, Yq, Uq, Vq = MapPlotter.subsample_quiver(
            xc.values, yc.values, u, v, step=config.quiver_step
        )
        
        ax.quiver(
            Xq, Yq, Uq, Vq,
            transform=ccrs.PlateCarree(),
            color=config.quiver_color,
            scale=config.quiver_scale,
            width=config.quiver_width,
            headwidth=3,
            headlength=4,
            headaxislength=3.5,
            zorder=2,
        )


# ========================
# Comparison Plotter
# ========================

class ExperimentComparisonPlotter:
    """Creates comparison plots between two experiments."""
    
    def __init__(self, data_loader: EnsembleDataLoader):
        self.data_loader = data_loader
        self.map_plotter = MapPlotter()
    
    def _compute_color_limits(
        self,
        data1: xr.DataArray,
        data2: xr.DataArray,
        vlim: Optional[float] = None,
        vlim_diff: Optional[float] = None,
        symmetric: bool = True
    ) -> Tuple[float, float]:
        """Compute appropriate color limits for main and difference plots."""
        if vlim is None:
            vlim = float(np.nanmax(np.abs(xr.concat([data1, data2], dim="panel"))))
        
        diff = data1 - data2
        if vlim_diff is None:
            vlim_diff = float(np.nanmax(np.abs(diff)))
        
        return vlim, vlim_diff
    
    def plot_scalar_comparison(
        self,
        exp_config: ExperimentConfig,
        plot_config: PlotConfig,
        save_path: str = "scalar_comparison.png"
    ) -> Tuple:
        """Plot scalar field comparison between two experiments."""
        # Load data
        nc_files_exp1 = [
            f"{exp_config.output_path}/{exp_config.exp1_name}{m}_{exp_config.var_name}_trend.nc"
            for m in exp_config.members
        ]
        nc_files_exp2 = [
            f"{exp_config.output_path}/{exp_config.exp2_name}{m}_{exp_config.var_name}_trend.nc"
            for m in exp_config.members
        ]
        
        ds1, data1 = self.data_loader.get_scalar_trend_at_depth(
            nc_files_exp1, exp_config.depth
        )
        ds2, data2 = self.data_loader.get_scalar_trend_at_depth(
            nc_files_exp2, exp_config.depth
        )
        
        diff = data1 - data2
        
        # Compute color limits
        vlim_main, vlim_diff = self._compute_color_limits(
            data1, data2, plot_config.vmin, plot_config.vmax
        )
        
        # Create figure
        projection = ccrs.Miller(central_longitude=float(ds1.XC.mean()))
        fig, axes = plt.subplots(
            1, 3, figsize=(18, 5),
            subplot_kw={"projection": projection},
            dpi=200,
            constrained_layout=True
        )
        
        # Plot exp1
        config1 = PlotConfig(
            cmap=plot_config.cmap,
            vmin=-vlim_main,
            vmax=vlim_main,
            extent=plot_config.extent,
            unit=plot_config.unit
        )
        mesh1 = self.map_plotter.plot_field(axes[0], data1, ds1.XC, ds1.YC, config1)
        self.map_plotter.setup_map(
            axes[0],
            extent=plot_config.extent,
            title=f"{exp_config.exp1_name} {exp_config.var_name} at {exp_config.depth} m"
        )
        
        # Plot exp2
        mesh2 = self.map_plotter.plot_field(axes[1], data2, ds1.XC, ds1.YC, config1)
        self.map_plotter.setup_map(
            axes[1],
            extent=plot_config.extent,
            title=f"{exp_config.exp2_name} {exp_config.var_name} at {exp_config.depth} m"
        )
        
        # Plot difference
        config_diff = PlotConfig(
            cmap=plot_config.cmap,
            vmin=-vlim_diff,
            vmax=vlim_diff,
            extent=plot_config.extent,
            unit=plot_config.unit
        )
        mesh_diff = self.map_plotter.plot_field(axes[2], diff, ds1.XC, ds1.YC, config_diff)
        self.map_plotter.setup_map(
            axes[2],
            extent=plot_config.extent,
            title=f"{exp_config.exp1_name} − {exp_config.exp2_name} at {exp_config.depth} m"
        )
        
        # Add colorbars
        cbar_main = fig.colorbar(mesh1, ax=axes[:2], fraction=0.03, pad=0.03)
        cbar_main.set_label(plot_config.unit)
        
        cbar_diff = fig.colorbar(mesh_diff, ax=axes[2], fraction=0.03, pad=0.03)
        cbar_diff.set_label(plot_config.unit)
        
        plt.savefig(save_path)
        return fig, axes, data1, data2, diff
    
    def plot_combined_comparison(
        self,
        exp_config: ExperimentConfig,
        scalar_config: PlotConfig,
        velocity_config: VelocityConfig,
        vel_vlim_main: float = 0.001,
        vel_vlim_diff: float = 0.0005,
        save_path: str = "combined_comparison.png"
    ) -> Tuple:
        """Plot combined scalar and velocity comparison."""
        # Load scalar data
        nc_files_exp1_scalar = [
            f"{exp_config.output_path}/{exp_config.exp1_name}{m}_{exp_config.var_name}_trend.nc"
            for m in exp_config.members
        ]
        nc_files_exp2_scalar = [
            f"{exp_config.output_path}/{exp_config.exp2_name}{m}_{exp_config.var_name}_trend.nc"
            for m in exp_config.members
        ]
        
        ds1, scalar1 = self.data_loader.get_scalar_trend_at_depth(
            nc_files_exp1_scalar, exp_config.depth
        )
        ds2, scalar2 = self.data_loader.get_scalar_trend_at_depth(
            nc_files_exp2_scalar, exp_config.depth
        )
        scalar_diff = scalar1 - scalar2
        
        # Load velocity data
        speed1, u1, v1 = self.data_loader.get_velocity_trend_at_depth(
            exp_config.exp1_name, exp_config.members, exp_config.depth
        )
        speed2, u2, v2 = self.data_loader.get_velocity_trend_at_depth(
            exp_config.exp2_name, exp_config.members, exp_config.depth
        )
        
        speed_diff = speed1 - speed2
        u_diff = u1 - u2
        v_diff = v1 - v2
        
        # Compute color limits
        vlim_scalar_main, vlim_scalar_diff = self._compute_color_limits(
            scalar1, scalar2, scalar_config.vmin, scalar_config.vmax
        )
        
        # Create figure
        projection = ccrs.Miller(central_longitude=float(ds1.XC.mean()))
        fig, axes = plt.subplots(
            2, 3, figsize=(18, 8),
            subplot_kw={"projection": projection},
            dpi=200,
            constrained_layout=True
        )
        
        # ========== ROW 1: SCALAR FIELDS ==========
        
        # Exp1 scalar
        config_main = PlotConfig(
            cmap=scalar_config.cmap,
            vmin=-vlim_scalar_main,
            vmax=vlim_scalar_main,
            extent=scalar_config.extent
        )
        mesh_s1 = self.map_plotter.plot_field(axes[0, 0], scalar1, ds1.XC, ds1.YC, config_main)
        self.map_plotter.setup_map(
            axes[0, 0],
            extent=scalar_config.extent,
            title=f"{exp_config.exp1_name} {exp_config.var_name} at {exp_config.depth} m"
        )
        
        # Exp2 scalar
        mesh_s2 = self.map_plotter.plot_field(axes[0, 1], scalar2, ds1.XC, ds1.YC, config_main)
        self.map_plotter.setup_map(
            axes[0, 1],
            extent=scalar_config.extent,
            title=f"{exp_config.exp2_name} {exp_config.var_name} at {exp_config.depth} m"
        )
        
        # Scalar difference
        config_diff = PlotConfig(
            cmap=scalar_config.cmap,
            vmin=-vlim_scalar_diff,
            vmax=vlim_scalar_diff,
            extent=scalar_config.extent
        )
        mesh_s_diff = self.map_plotter.plot_field(axes[0, 2], scalar_diff, ds1.XC, ds1.YC, config_diff)
        self.map_plotter.setup_map(
            axes[0, 2],
            extent=scalar_config.extent,
            title=f"{exp_config.exp1_name} − {exp_config.exp2_name} at {exp_config.depth} m"
        )
        
        # Colorbars for scalar row
        cbar_s_main = fig.colorbar(mesh_s1, ax=axes[0, :2], fraction=0.03, pad=0.03)
        cbar_s_main.set_label(scalar_config.unit)
        cbar_s_diff = fig.colorbar(mesh_s_diff, ax=axes[0, 2], fraction=0.03, pad=0.03)
        cbar_s_diff.set_label(scalar_config.unit)
        
        # ========== ROW 2: VELOCITY FIELDS ==========
        
        # Exp1 velocity
        config_vel_main = PlotConfig(
            cmap=velocity_config.cmap,
            vmin=0,
            vmax=vel_vlim_main,
            extent=scalar_config.extent
        )
        mesh_v1 = self.map_plotter.plot_field(axes[1, 0], speed1, ds1.XC, ds1.YC, config_vel_main)
        self.map_plotter.add_quiver(axes[1, 0], ds1.XC, ds1.YC, u1, v1, velocity_config)
        self.map_plotter.setup_map(
            axes[1, 0],
            extent=scalar_config.extent,
            title=f"{exp_config.exp1_name} velocity at {exp_config.depth} m"
        )
        
        # Exp2 velocity
        mesh_v2 = self.map_plotter.plot_field(axes[1, 1], speed2, ds1.XC, ds1.YC, config_vel_main)
        self.map_plotter.add_quiver(axes[1, 1], ds1.XC, ds1.YC, u2, v2, velocity_config)
        self.map_plotter.setup_map(
            axes[1, 1],
            extent=scalar_config.extent,
            title=f"{exp_config.exp2_name} velocity at {exp_config.depth} m"
        )
        
        # Velocity difference
        config_vel_diff = PlotConfig(
            cmap="PiYG",
            vmin=-vel_vlim_diff,
            vmax=vel_vlim_diff,
            extent=scalar_config.extent
        )
        mesh_v_diff = self.map_plotter.plot_field(axes[1, 2], speed_diff, ds1.XC, ds1.YC, config_vel_diff)
        self.map_plotter.add_quiver(axes[1, 2], ds1.XC, ds1.YC, u_diff, v_diff, velocity_config)
        self.map_plotter.setup_map(
            axes[1, 2],
            extent=scalar_config.extent,
            title=f"{exp_config.exp1_name} − {exp_config.exp2_name} velocity at {exp_config.depth} m"
        )
        
        # Colorbars for velocity row
        cbar_v_main = fig.colorbar(mesh_v1, ax=axes[1, :2], fraction=0.03, pad=0.03)
        cbar_v_main.set_label("m/s/century")
        cbar_v_diff = fig.colorbar(mesh_v_diff, ax=axes[1, 2], fraction=0.03, pad=0.03)
        cbar_v_diff.set_label("m/s/century")
        
        plt.savefig(save_path)
        
        return fig, axes, scalar1, scalar2, scalar_diff, speed1, speed2, speed_diff


# ========================
# Convenience Functions
# ========================

def vertical_weighted_mean(field: xr.DataArray, ds: xr.Dataset, zrange: Tuple[float, float]) -> xr.DataArray:
    """Compute weighted vertical mean over depth using drF * hFacC."""
    field_sel = field.sel(Z=slice(*zrange))
    h = ds["drF"] * ds["hFacC"]
    h_sel = h.sel(Z=slice(*zrange))
    
    field_sel = field_sel.where(h_sel > 0)
    h_sel = h_sel.where(h_sel > 0)
    
    return (field_sel * h_sel).sum("Z") / h_sel.sum("Z")


def reproduce_original_plot():
    """Reproduce the original MELT vs LENS comparison plot."""
    members = ["002", "003", "004", "005", "006"]
    
    exp_config = ExperimentConfig(
        exp1_name="MELT",
        exp2_name="LENS",
        members=members,
        var_name="THETA",
        depth=-100
    )
    
    scalar_config = PlotConfig(
        cmap="coolwarm",
        vmin=0.04,  # Will be used as vlim_main (symmetric)
        vmax=0.01,  # Will be used as vlim_diff
        extent=[-120, -95, -77, -66],
        unit="degC/yr"
    )
    
    velocity_config = VelocityConfig(
        cmap="cool",
        quiver_step=10,
        quiver_scale=0.01,
        quiver_width=0.0035
    )
    
    data_loader = EnsembleDataLoader()
    plotter = ExperimentComparisonPlotter(data_loader)
    
    return plotter.plot_combined_comparison(
        exp_config=exp_config,
        scalar_config=scalar_config,
        velocity_config=velocity_config,
        vel_vlim_main=0.001,
        vel_vlim_diff=0.0005,
        save_path="ensemble_mean_trend_MELT_LENS_diff_with_velocity.png"
    )


def main():
    """Main function - reproduce original plot."""
    reproduce_original_plot()
    plt.show()


if __name__ == "__main__":
    main()