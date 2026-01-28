import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from dataclasses import dataclass
from typing import List, Optional, Tuple
from tools.directories_and_paths import OUTPUT_PATH


# ========================
# Configuration Classes
# ========================

@dataclass
class VariableConfig:
    """Configuration for different variables."""
    name: str
    cmap: str
    vlim_main: Tuple[float, float]
    vlim_diff: Tuple[float, float]
    label: str
    
    @classmethod
    def from_name(cls, var_name: str):
        """Create config from variable name."""
        configs = {
            "THETA": cls(
                name="THETA",
                cmap="coolwarm",
                vlim_main=(-2.5, 2.5),
                vlim_diff=(-0.2, 0.2),
                label="Temperature (°C)"
            ),
            "SALT": cls(
                name="SALT",
                cmap="BrBG",
                vlim_main=(33, 35),
                vlim_diff=(-0.3, 0.3),
                label="Salinity (PSU)"
            ),
            "UVEL": cls(
                name="UVEL",
                cmap="PiYG",
                vlim_main=(-0.1, 0.1),
                vlim_diff=(-0.05, 0.05),
                label="Zonal velocity (m/s)"
            ),
            "VVEL": cls(
                name="VVEL",
                cmap="PiYG",
                vlim_main=(-0.001, 0.001),
                vlim_diff=(-0.0005, 0.0005),
                label="Meridional velocity (m/s)"
            ),
        }
        return configs.get(var_name, configs["THETA"])


@dataclass
class TransectConfig:
    """Configuration for transect location."""
    orientation: str  # "zonal" or "meridional" - Cut along lat or lon
    position: float  # Longitude (for zonal) or latitude (for meridional)
    lat_range: Optional[Tuple[float, float]] = None  # For zonal transects
    lon_range: Optional[Tuple[float, float]] = None  # For meridional transects
    depth_max: float = -2000  # Maximum depth to show (negative)
    
    def get_position_label(self) -> str:
        """Get label for transect position."""
        if self.orientation == "zonal":
            # Convert to degrees West if negative
            lon_label = abs(self.position) if self.position < 0 else 360 - self.position
            return f"{lon_label:.1f}°W"
        else:
            lat_label = abs(self.position)
            hemisphere = "S" if self.position < 0 else "N"
            return f"{lat_label:.1f}°{hemisphere}"


@dataclass
class ExperimentConfig:
    """Configuration for experiment comparison."""
    exp1_name: str
    exp2_name: str
    exp1_label: str
    exp2_label: str
    members: List[str]
    output_path: str = OUTPUT_PATH


@dataclass
class PlotConfig:
    """Configuration for plot appearance."""
    figsize: Tuple[float, float] = (6, 10)
    dpi: int = 200
    show_inset_map: bool = True
    inset_extent: List[float] = None  # [lon_min, lon_max, lat_min, lat_max]
    
    def __post_init__(self):
        if self.inset_extent is None:
            self.inset_extent = [-120, -95, -78, -68]  # Amundsen Sea default


# ========================
# Data Loading
# ========================

class TransectDataLoader:
    """Handles loading and processing of transect data."""
    
    def __init__(self, output_path: str = OUTPUT_PATH):
        self.output_path = output_path
        # Load reference grid
        ref_path = f"{output_path}/PAS_MELT003_S/output/210001/MITgcm/output.nc"
        self.ds_ref = xr.open_dataset(ref_path)
    
    def get_hfac_name(self, var_name: str) -> str:
        """Get appropriate hFac field for variable."""
        if var_name == "UVEL":
            return "hFacW"
        elif var_name == "VVEL":
            return "hFacS"
        else:
            return "hFacC"
    
    def get_coordinate_names(self, var_name: str) -> Tuple[str, str]:
        """Get appropriate X and Y coordinate names for variable.
        
        Returns:
            (x_coord_name, y_coord_name)
        """
        x_coord = "XC"
        y_coord = "YC"
        
        if var_name == "UVEL":
            x_coord = "XG"
        elif var_name == "VVEL":
            y_coord = "YG"
        
        return x_coord, y_coord
    
    def find_nearest_index(self, coord_array: xr.DataArray, value: float) -> int:
        """Find index of nearest coordinate value."""
        return int(np.argmin(np.abs(coord_array.values - value)))
    
    def get_transect_indices(
        self,
        transect_config: TransectConfig,
        var_name: str
    ) -> Tuple[int, slice, slice]:
        """
        Convert lat/lon coordinates to array indices.
        
        Args:
            transect_config: Transect configuration
            var_name: Variable name (needed for correct coordinate grid)
        
        Returns:
            position_idx: Index along the transect direction
            lat_slice: Slice for latitude dimension
            lon_slice: Slice for longitude dimension
        """
        # Get correct coordinate names for this variable
        x_coord, y_coord = self.get_coordinate_names(var_name)
        
        # For finding indices, we always use the center coordinates (XC, YC)
        # from the reference dataset, then apply to the staggered grid
        if transect_config.orientation == "zonal":
            # Transect is along latitude at fixed longitude
            # Find longitude index using XC
            xc_1d = self.ds_ref.XC
            lon_idx = self.find_nearest_index(xc_1d, transect_config.position)
            
            if transect_config.lat_range:
                # Find latitude indices using YC
                yc_1d = self.ds_ref.YC
                lat_start = self.find_nearest_index(yc_1d, transect_config.lat_range[0])
                lat_end = self.find_nearest_index(yc_1d, transect_config.lat_range[1])
                lat_slice = slice(lat_start, lat_end)
            else:
                lat_slice = slice(None)
            
            lon_slice = lon_idx
            position_idx = lon_idx
            
        else:  # meridional
            # Transect is along longitude at fixed latitude
            # Find latitude index using YC
            yc_1d = self.ds_ref.YC
            lat_idx = self.find_nearest_index(yc_1d, transect_config.position)
            
            if transect_config.lon_range:
                # Find longitude indices using XC
                xc_1d = self.ds_ref.XC
                lon_start = self.find_nearest_index(xc_1d, transect_config.lon_range[0])
                lon_end = self.find_nearest_index(xc_1d, transect_config.lon_range[1])
                lon_slice = slice(lon_start, lon_end)
            else:
                lon_slice = slice(None)
            
            lat_slice = lat_idx
            position_idx = lat_idx
        
        return position_idx, lat_slice, lon_slice
    
    def get_depth_indices(self, depth_max: float) -> slice:
        """Get depth slice up to maximum depth."""
        # Find index where depth is closest to depth_max
        z_vals = self.ds_ref.Z.values
        depth_idx = np.argmin(np.abs(z_vals - depth_max))
        return slice(0, depth_idx + 1)
    
    def load_ensemble_mean(
        self,
        var_name: str,
        members: List[str],
        path_template: str,
        year: Optional[int] = None,
        is_trend: bool = False
    ) -> xr.DataArray:
        """Load ensemble mean for a variable.
        
        Args:
            var_name: Physical variable name (THETA, SALT, etc.)
            members: List of ensemble member IDs
            path_template: Path template with {mem} and optionally {year}
            year: Year for timeseries data (None for trend data)
            is_trend: If True, variable in file is named "trend" instead of var_name
        """
        fields = []
        
        for mem in members:
            path = path_template.format(mem=mem, year=year) if year else path_template.format(mem=mem)
            ds = xr.open_dataset(path)
            
            # For trend files, variable is always called "trend"
            file_var_name = "trend" if is_trend else var_name
            
            # Handle time dimension
            if "time" in ds[file_var_name].dims:
                field = ds[file_var_name].mean(dim="time")
            else:
                field = ds[file_var_name]
            
            fields.append(field)
        
        return xr.concat(fields, dim="member").mean(dim="member")
    
    def extract_transect(
        self,
        field: xr.DataArray,
        transect_config: TransectConfig,
        var_name: str
    ) -> Tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
        """
        Extract transect from 3D field.
        
        Returns:
            transect_data: 2D slice [Z, along-transect]
            coord_along: Coordinate along transect (lat or lon) - uses correct staggered grid
            mask: Boolean mask for valid data
        """
        position_idx, lat_slice, lon_slice = self.get_transect_indices(transect_config, var_name)
        depth_slice = self.get_depth_indices(transect_config.depth_max)
        
        # Get appropriate hFac for masking
        hfac_name = self.get_hfac_name(var_name)
        hfac = self.ds_ref[hfac_name]
        
        # Get correct coordinate names for this variable (XC/XG, YC/YG)
        x_coord, y_coord = self.get_coordinate_names(var_name)
        
        # Extract transect
        if transect_config.orientation == "zonal":
            transect_data = field[depth_slice, lat_slice, lon_slice]
            mask = hfac[depth_slice, lat_slice, lon_slice] > 0
            # For zonal transect, coordinate varies along Y
            # Extract 1D Y coordinate (either YC or YG depending on variable)
            if y_coord == "YG":
                coord_along = self.ds_ref.YG[lat_slice]
            else:
                coord_along = self.ds_ref.YC[lat_slice]
        else:  # meridional
            transect_data = field[depth_slice, lat_slice, lon_slice]
            mask = hfac[depth_slice, lat_slice, lon_slice] > 0
            # For meridional transect, coordinate varies along X
            # Extract 1D X coordinate (either XC or XG depending on variable)
            if x_coord == "XG":
                coord_along = self.ds_ref.XG[lon_slice]
            else:
                coord_along = self.ds_ref.XC[lon_slice]
        
        # Apply mask
        transect_data = transect_data.where(mask)
        
        return transect_data, coord_along, mask


# ========================
# Plotting
# ========================

class TransectPlotter:
    """Handles plotting of transects."""
    
    def __init__(self, data_loader: TransectDataLoader):
        self.data_loader = data_loader
    
    def add_location_inset(
        self,
        fig: plt.Figure,
        transect_config: TransectConfig,
        plot_config: PlotConfig
    ):
        """Add small map showing transect location."""
        # Create inset axis in upper left
        ax_inset = fig.add_axes([0.005, 0.83, 0.3, 0.3], projection=ccrs.PlateCarree())
        
        # Add features
        ax_inset.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black', linewidth=0.5)
        ax_inset.coastlines(linewidth=0.5)
        ax_inset.set_extent(plot_config.inset_extent, crs=ccrs.PlateCarree())
        
        # Add gridlines
        gl = ax_inset.gridlines(draw_labels=False, linewidth=0.3, color='gray', alpha=0.5)
        
        # Draw transect line in red
        if transect_config.orientation == "zonal":
            # Vertical line at fixed longitude
            lon = transect_config.position
            if transect_config.lat_range:
                lat_min, lat_max = transect_config.lat_range
            else:
                lat_min, lat_max = plot_config.inset_extent[2], plot_config.inset_extent[3]
            
            ax_inset.plot(
                [lon, lon],
                [lat_min, lat_max],
                'r-',
                linewidth=3,
                transform=ccrs.PlateCarree(),
            )
        else:  # meridional
            # Horizontal line at fixed latitude
            lat = transect_config.position
            if transect_config.lon_range:
                lon_min, lon_max = transect_config.lon_range
            else:
                lon_min, lon_max = plot_config.inset_extent[0], plot_config.inset_extent[1]
            
            ax_inset.plot(
                [lon_min, lon_max],
                [lat, lat],
                'r-',
                linewidth=3,
                transform=ccrs.PlateCarree(),
                label='Transect'
            )
    
    def plot_static_comparison(
        self,
        var_config: VariableConfig,
        exp_config: ExperimentConfig,
        transect_config: TransectConfig,
        plot_config: PlotConfig,
        path_template_exp1: str,
        path_template_exp2: str,
        year: Optional[int] = None,
        is_trend: bool = False,
        save_path: str = "transect_comparison.png"
    ):
        """Create static comparison plot of two experiments.
        
        Args:
            is_trend: If True, assumes files contain variable named "trend"
        """
        # Load data
        field_exp1 = self.data_loader.load_ensemble_mean(
            var_config.name,
            exp_config.members,
            path_template_exp1,
            year,
            is_trend=is_trend
        )
        field_exp2 = self.data_loader.load_ensemble_mean(
            var_config.name,
            exp_config.members,
            path_template_exp2,
            year,
            is_trend=is_trend
        )
        
        # Extract transects
        transect_exp1, coord_along, _ = self.data_loader.extract_transect(
            field_exp1, transect_config, var_config.name
        )
        transect_exp2, _, _ = self.data_loader.extract_transect(
            field_exp2, transect_config, var_config.name
        )
        
        transect_diff = transect_exp2 - transect_exp1
        
        # Get depth coordinate
        depth_slice = self.data_loader.get_depth_indices(transect_config.depth_max)
        z_coord = self.data_loader.ds_ref.Z[depth_slice]
        
        # Create figure
        fig = plt.figure(figsize=plot_config.figsize, dpi=plot_config.dpi)
        fig.subplots_adjust(left=0.1, right=0.85, top=0.92, bottom=0.08, hspace=0.3)
        
        axes = [
            fig.add_subplot(3, 1, 1),
            fig.add_subplot(3, 1, 2),
            fig.add_subplot(3, 1, 3),
        ]
        
        # Colorbars
        cax_main = fig.add_axes([0.88, 0.4, 0.02, 0.5])
        cax_diff = fig.add_axes([0.88, 0.1, 0.02, 0.24])
        
        # Plot data
        data_list = [transect_exp2, transect_exp1, transect_diff]
        titles = [
            f"{exp_config.exp2_label}",
            f"{exp_config.exp1_label}",
            f"{exp_config.exp2_label} − {exp_config.exp1_label}"
        ]
        vlims = [var_config.vlim_main, var_config.vlim_main, var_config.vlim_diff]
        
        meshes = []
        for ax, data, title, vlim in zip(axes, data_list, titles, vlims):
            ax.set_facecolor("lightgrey")
            mesh = ax.pcolormesh(
                coord_along,
                z_coord,
                data.values,
                shading="auto",
                cmap=var_config.cmap,
                vmin=vlim[0],
                vmax=vlim[1],
            )
            
            ax.set_title(title, fontsize=12)
            ax.set_ylim(z_coord[-1], z_coord[0])
            ax.set_xlim(coord_along[0], coord_along[-1])
            
            # Labels
            if transect_config.orientation == "zonal":
                ax.set_xlabel("Latitude (°S)" if coord_along[0] < 0 else "Latitude (°N)")
            else:
                ax.set_xlabel("Longitude")
            ax.set_ylabel("Depth (m)")
            
            meshes.append(mesh)
        
        # Add colorbars
        cb_main = fig.colorbar(meshes[0], cax=cax_main)
        cb_main.set_label(var_config.label, fontsize=10)
        
        cb_diff = fig.colorbar(meshes[2], cax=cax_diff)
        cb_diff.set_label(var_config.label, fontsize=10)
        
        # Main title
        year_str = f" ({year})" if year else ""
        fig.suptitle(
            f"{var_config.name} transect at {transect_config.get_position_label()}{year_str}",
            fontsize=14,
            y=0.96
        )
        
        # Add location inset
        if plot_config.show_inset_map:
            self.add_location_inset(fig, transect_config, plot_config)
        
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
        
        return fig, axes, meshes
    
    def create_animation(
        self,
        var_config: VariableConfig,
        exp_config: ExperimentConfig,
        transect_config: TransectConfig,
        plot_config: PlotConfig,
        path_template_exp1: str,
        path_template_exp2: str,
        years: List[int],
        interval: int = 300,
        is_trend: bool = False,
        save_path: str = "transect_animation.gif"
    ):
        """Create animated comparison over multiple years.
        
        Args:
            is_trend: If True, assumes files contain variable named "trend"
        """
        # Create figure
        fig = plt.figure(figsize=plot_config.figsize, dpi=150)
        fig.subplots_adjust(left=0.1, right=0.85, top=0.92, bottom=0.08, hspace=0.3)
        
        axes = [
            fig.add_subplot(3, 1, 1),
            fig.add_subplot(3, 1, 2),
            fig.add_subplot(3, 1, 3),
        ]
        
        # Colorbars
        cax_main = fig.add_axes([0.88, 0.4, 0.02, 0.5])
        cax_diff = fig.add_axes([0.88, 0.1, 0.02, 0.24])
        
        # Get coordinate info (doesn't change between years)
        field_init = self.data_loader.load_ensemble_mean(
            var_config.name,
            exp_config.members,
            path_template_exp1,
            years[0],
            is_trend=is_trend
        )
        _, coord_along, _ = self.data_loader.extract_transect(
            field_init, transect_config, var_config.name
        )
        depth_slice = self.data_loader.get_depth_indices(transect_config.depth_max)
        z_coord = self.data_loader.ds_ref.Z[depth_slice]
        
        # Initialize empty plots
        meshes = []
        titles = [
            f"{exp_config.exp2_label}",
            f"{exp_config.exp1_label}",
            f"{exp_config.exp2_label} − {exp_config.exp1_label}"
        ]
        vlims = [var_config.vlim_main, var_config.vlim_main, var_config.vlim_diff]
        
        for ax, title, vlim in zip(axes, titles, vlims):
            ax.set_facecolor("lightgrey")
            print(vlim)
            mesh = ax.pcolormesh(
                coord_along,
                z_coord,
                np.zeros((len(z_coord), len(coord_along))),
                shading="auto",
                cmap=var_config.cmap,
                vmin=vlim[0],
                vmax=vlim[1],
            )
            
            ax.set_title(title, fontsize=12)
            ax.set_ylim(z_coord[-1], z_coord[0])
            ax.set_xlim(coord_along[0], coord_along[-1])
            
            if transect_config.orientation == "zonal":
                ax.set_xlabel("Latitude (°S)" if coord_along[0] < 0 else "Latitude (°N)")
            else:
                ax.set_xlabel("Longitude (°E)")
            ax.set_ylabel("Depth (m)")
            
            meshes.append(mesh)
        
        # Add colorbars
        cb_main = fig.colorbar(meshes[0], cax=cax_main)
        cb_main.set_label(var_config.label, fontsize=10)
        
        cb_diff = fig.colorbar(meshes[2], cax=cax_diff)
        cb_diff.set_label(var_config.label, fontsize=10)
        
        # Add location inset
        if plot_config.show_inset_map:
            self.add_location_inset(fig, transect_config, plot_config)
        
        # Update function
        def update(year):
            print(f"Processing year {year}")
            
            # Load data
            field_exp1 = self.data_loader.load_ensemble_mean(
                var_config.name,
                exp_config.members,
                path_template_exp1,
                year,
                is_trend=is_trend
            )
            field_exp2 = self.data_loader.load_ensemble_mean(
                var_config.name,
                exp_config.members,
                path_template_exp2,
                year,
                is_trend=is_trend
            )
            
            # Extract transects
            transect_exp1, _, _ = self.data_loader.extract_transect(
                field_exp1, transect_config, var_config.name
            )
            transect_exp2, _, _ = self.data_loader.extract_transect(
                field_exp2, transect_config, var_config.name
            )
            
            transect_diff = transect_exp2 - transect_exp1
            
            # Update meshes
            data_list = [transect_exp2, transect_exp1, transect_diff]
            for mesh, data in zip(meshes, data_list):
                mesh.set_array(data.values.ravel())
            
            # Update title
            fig.suptitle(
                f"{var_config.name} transect at {transect_config.get_position_label()} ({year})",
                fontsize=14,
                y=0.96
            )
            
            return meshes
        
        # Create animation
        ani = FuncAnimation(
            fig,
            update,
            frames=years,
            interval=interval,
            blit=False,
        )
        
        ani.save(save_path, writer="pillow", dpi=150)
        print(f"Saved animation: {save_path}")
        plt.close(fig)
        
        return ani


# ========================
# Convenience Functions
# ========================

def plot_trend_comparison(
    var_name: str = "THETA",
    exp1_name: str = "MELT",
    exp2_name: str = "LENS",
    orientation: str = "zonal",  # "zonal" or "meridional"
    position: float = -106,  # Longitude or latitude
    lat_range: Optional[Tuple[float, float]] = None,
    lon_range: Optional[Tuple[float, float]] = None,
    depth_max: float = -2000,
    members: List[str] = None,
    show_inset: bool = True,
    is_trend: bool = True,
    save_path: str = None
):
    """
    Convenience function to plot trend comparison.
    
    Args:
        var_name: Variable to plot (THETA, SALT, UVEL, VVEL)
        exp1_name: Name of first experiment
        exp2_name: Name of second experiment
        orientation: "zonal" (lat transect at fixed lon) or "meridional" (lon transect at fixed lat)
        position: Longitude (for zonal) or latitude (for meridional)
        lat_range: (min, max) latitude range for zonal transects
        lon_range: (min, max) longitude range for meridional transects
        depth_max: Maximum depth to show (negative value)
        members: List of ensemble members
        show_inset: Whether to show location map
        is_trend: If True, variable in files is named "trend" (default for trend files)
        save_path: Output filename
    """
    if members is None:
        members = ["002", "003", "004", "005", "006"]
    
    if save_path is None:
        save_path = f"transect_{var_name}_{exp1_name}_{exp2_name}_{orientation}.png"
    
    # Create configurations
    var_config = VariableConfig.from_name(var_name)
    
    exp_config = ExperimentConfig(
        exp1_name=exp1_name,
        exp2_name=exp2_name,
        exp1_label=exp1_name,
        exp2_label=exp2_name,
        members=members
    )
    
    transect_config = TransectConfig(
        orientation=orientation,
        position=position,
        lat_range=lat_range,
        lon_range=lon_range,
        depth_max=depth_max
    )
    
    plot_config = PlotConfig(
        show_inset_map=show_inset
    )
    
    # Setup data loader and plotter
    data_loader = TransectDataLoader()
    plotter = TransectPlotter(data_loader)
    
    # Create path templates for trend files
    path_template_exp1 = f"{OUTPUT_PATH}/{exp1_name}{{mem}}_{var_name}_trend.nc"
    path_template_exp2 = f"{OUTPUT_PATH}/{exp2_name}{{mem}}_{var_name}_trend.nc"
    
    # Create plot
    return plotter.plot_static_comparison(
        var_config=var_config,
        exp_config=exp_config,
        transect_config=transect_config,
        plot_config=plot_config,
        path_template_exp1=path_template_exp1,
        path_template_exp2=path_template_exp2,
        is_trend=is_trend,
        save_path=save_path
    )


def create_timeseries_animation(
    var_name: str = "THETA",
    exp1_name: str = "MELT",
    exp2_name: str = "LENS",
    orientation: str = "zonal",  # "zonal" or "meridional"
    position: float = -106,
    lat_range: Optional[Tuple[float, float]] = None,
    lon_range: Optional[Tuple[float, float]] = None,
    depth_max: float = -2000,
    years: List[int] = None,
    members: List[str] = None,
    interval: int = 300,
    show_inset: bool = True,
    is_trend: bool = False,
    save_path: str = None
):
    """
    Convenience function to create animation over time.
    
    Args:
        var_name: Variable to plot
        exp1_name: Name of first experiment
        exp2_name: Name of second experiment
        orientation: "zonal" or "meridional"
        position: Longitude (for zonal) or latitude (for meridional)
        lat_range: (min, max) latitude range for zonal transects
        lon_range: (min, max) longitude range for meridional transects
        depth_max: Maximum depth to show (negative value)
        years: List of years to animate
        members: List of ensemble members
        interval: Milliseconds between frames
        show_inset: Whether to show location map
        is_trend: If True, variable in files is named "trend"
        save_path: Output filename
    """
    if members is None:
        members = ["002", "003", "004", "005", "006"]
    
    if years is None:
        years = list(range(2005, 2101))
    
    if save_path is None:
        save_path = f"transect_{var_name}_{exp1_name}_{exp2_name}_{orientation}_animation.gif"
    
    # Create configurations
    var_config = VariableConfig.from_name(var_name)
    
    exp_config = ExperimentConfig(
        exp1_name=exp1_name,
        exp2_name=exp2_name,
        exp1_label=exp1_name,
        exp2_label=exp2_name,
        members=members
    )
    
    transect_config = TransectConfig(
        orientation=orientation,
        position=position,
        lat_range=lat_range,
        lon_range=lon_range,
        depth_max=depth_max
    )
    
    plot_config = PlotConfig(
        show_inset_map=show_inset
    )
    
    # Setup data loader and plotter
    data_loader = TransectDataLoader()
    plotter = TransectPlotter(data_loader)
    
    # Determine path templates based on exp names
    if "MELT" in exp1_name:
        path_template_exp1 = f"{OUTPUT_PATH}/PAS_{exp1_name}{{mem}}_S/output/{{year}}01/MITgcm/output.nc"
    else:
        path_template_exp1 = f"{OUTPUT_PATH}/old_PAS/PAS_{exp1_name}{{mem}}_O/output/{{year}}01/MITgcm/output.nc"
    
    if "MELT" in exp2_name:
        path_template_exp2 = f"{OUTPUT_PATH}/PAS_{exp2_name}{{mem}}_S/output/{{year}}01/MITgcm/output.nc"
    else:
        path_template_exp2 = f"{OUTPUT_PATH}/old_PAS/PAS_{exp2_name}{{mem}}_O/output/{{year}}01/MITgcm/output.nc"
    
    # Create animation
    return plotter.create_animation(
        var_config=var_config,
        exp_config=exp_config,
        transect_config=transect_config,
        plot_config=plot_config,
        path_template_exp1=path_template_exp1,
        path_template_exp2=path_template_exp2,
        years=years,
        interval=interval,
        is_trend=is_trend,
        save_path=save_path
    )


def main():
    """Example usage: plot trend comparison."""
    # Example 1: Zonal transect (along latitude at fixed longitude)
    # plot_trend_comparison(
    #     var_name="THETA",
    #     exp1_name="MELT",
    #     exp2_name="LENS",
    #     orientation="zonal",
    #     position=-106,  # 106°W
    #     lat_range=(-76, -70),  # Between 76°S and 70°S
    #     depth_max=-2000,
    #     save_path="theta_zonal_transect.png"
    # )
    
    # Example 2: Meridional transect (along longitude at fixed latitude)
    plot_trend_comparison(
        var_name="VVEL",
        exp1_name="MELT",
        exp2_name="LENS",
        orientation="meridional",
        position=-73.5,  # 73°S
        lon_range=(250, 255),  # Between 120°W and 100°W
        depth_max=-800,
        save_path="vvel_meridional_transect.png"
    )
    
    # Example 3: Create animation (uncomment to use)
    # create_timeseries_animation(
    #     var_name="THETA",
    #     exp1_name="MELT",
    #     exp2_name="LENS",
    #     orientation="zonal",
    #     position=-106,
    #     lat_range=(-76, -70),
    #     years=list(range(2005, 2101, 5)),  # Every 5 years
    #     interval=500,
    #     save_path="theta_animation.gif"
    # )


if __name__ == "__main__":
    main()