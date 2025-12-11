import numpy as np
from mitgcm_python.grid import Grid


def pretty_labels(ax, both="all"):
    from mitgcm_python.plot_utils.labels import lat_label, lon_label
    """
    Adjusts the labels on the given matplotlib axis `ax` to display longitude and latitude
    values in a more readable format.

    Parameters:
    - ax (matplotlib.axis): The axis for which to adjust the labels.

    Returns:
    - None
    """
    if both == "all" or both =="lon":
        lon_ticks = ax.get_xticks() - 360
        lon_labels = []
        for x in lon_ticks:
            lon_labels.append(lon_label(x, 2))
        ax.set_xticklabels(lon_labels[:-1], size = 12)
        ax.tick_params(axis="x")
        
    if both == "all" or both =="lat":
        ax.locator_params(axis='y', nbins=6)
        lat_ticks = ax.get_yticks()
        lat_labels = []
        for y in lat_ticks:
            lat_labels.append(lat_label(y, 2))
        ax.set_yticklabels(lat_labels, size = 12)


def create_mask(depth, ice_mask):
    """
    Creates masks for land, ice shelf, and continental shelf based on depth and ice coverage.

    Parameters:
        depth (numpy.ndarray): Array representing depth information.
        ice_mask (numpy.ndarray): Array representing ice coverage.

    Returns:
        tuple: A tuple containing:
            - land_mask (numpy.ndarray): Mask for land areas.
            - mask (numpy.ndarray): Combined mask for ice shelf and continental shelf.
            - colors (list): List of RGBA colors for plotting.
    """
    land_mask = np.zeros(np.shape(depth))
    land_mask[depth == 0] = 1

    # apply mask over the ice shelf (determiend by the ice-mask) and the continental shelf (roughly where the depth is less than 1500m)
    mask = np.zeros(np.shape(depth))
    mask[depth < 1500] = 1
    mask[ice_mask == 0] = 2

    # set the colors to block the continent (set to grey)
    colors = [(1.0, 1.0, 1.0, 0), (0.7, 0.7, 0.7, 1), (0.6, 0.6, 0.6, 1)]
    return land_mask, mask, colors


def read_u_and_v(input_data, option="avg"):
    """
    Reads and processes u and v velocity components from input data.

    Parameters:
        input_data (xarray.Dataset): Dataset containing velocity data.
        option (str, optional): Velocity option for plotting. Defaults to 'avg'.

    Returns:
        tuple: A tuple containing:
            - speed (numpy.ndarray): Array representing speed of velocity vectors.
            - u_plot (numpy.ndarray): Array representing processed u velocity component for plotting.
            - v_plot (numpy.ndarray): Array representing processed v velocity component for plotting.
    """
    from mitgcm_python.utils import mask_3d
    from mitgcm_python.plot_utils.latlon import prepare_vel
    from constants import DAYS_IN_MONTH
    from tools.directories_and_paths import grid_filepath

    vvel = input_data.VVEL.values
    uvel = input_data.UVEL.values
    grid = Grid(grid_filepath)

    uvel = mask_3d(uvel, grid, gtype="u", time_dependent=True)
    vvel = mask_3d(vvel, grid, gtype="v", time_dependent=True)

    u = np.average(uvel, axis=0, weights=DAYS_IN_MONTH)
    v = np.average(vvel, axis=0, weights=DAYS_IN_MONTH)

    speed, u_plot, v_plot = prepare_vel(u, v, grid, vel_option=option)
    return speed, u_plot, v_plot


def zoom_shelf(ax, zoom):
    if zoom == "ice_shelf":
        ax.set_ylim([-75.6, -73])
        ax.set_xlim([245, 262])
    elif zoom == "cont_shelf":
        ax.set_xlim([230, 265])
        ax.set_ylim([-75.5, -68])


def read_mask(input_data=None, cut=None, lat_range=None, lon_range=None):
    """
    Reads and processes mask data from input data.

    Parameters:
        input_data (xarray.Dataset): Dataset containing mask data.
        cut (string): None or lat, if lat it takes a latitudinal cut using the lat and lon range

    Returns:
        dict: Dictionary containing setup information for plotting including latitude, longitude, depth, ice mask, land mask, combined mask, colors, X, and Y coordinates.
    """
    import xarray as xr
    from tools.directories_and_paths import output_path

    if input_data is None:
        input_data = xr.open_dataset(
            f"{output_path}CTRL_ens01_noOBC/output/192001/MITgcm/output.nc", decode_times=False
        )

    if cut is None:
        [lat, lon, ice_mask_temp, depth] = [
            input_data[param].values for param in ["YC", "XC", "maskC", "Depth"]
        ]
        ice_mask = ice_mask_temp[0, :, :]
        [land_mask, mask, colors] = create_mask(depth, ice_mask)
        [X, Y] = np.meshgrid(lon, lat)

        set_up = {
            "lat": lat,
            "lon": lon,
            "depth": depth,
            "ice_mask": ice_mask,
            "land_mask": land_mask,
            "mask": mask,
            "colors": colors,
            "X": X,
            "Y": Y,
        }

    elif cut == "lat":
        z = input_data.Z.values
        lat = input_data["YC"][lat_range[0] : lat_range[1]].values
        lon = input_data["XC"][lon_range]
        ice_mask = input_data.maskC.values[:, lat_range[0] : lat_range[1], lon_range]

        set_up = {
            "lat": lat,
            "lon": lon,
            "z": z,
            "ice_mask": ice_mask,
        }

    return set_up
