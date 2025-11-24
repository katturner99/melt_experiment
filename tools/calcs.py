import numpy as np
import xarray as xr
from .constants import ATM_PRESS, RHO_REF, G
from mitgcm_python.diagnostics import density

def moving_average(a, n=3):
    """calculate a centred moving average"""
    if n < 2:
        raise ValueError(
            "Window size (n) must be at least 2 for a centered moving average."
        )

    data = np.empty_like(a, dtype=float)
    data.fill(np.nan)

    # Calculate the cumulative sum
    cumsum = np.cumsum(np.insert(a, 0, 0))

    # Calculate the centered moving average
    half_n = n // 2
    if n % 2 == 0:
        data[half_n - 1 : -half_n] = (cumsum[n:] - cumsum[:-n]) / n
    else:
        data[half_n:-half_n] = (cumsum[n:] - cumsum[:-n]) / n

    return data

def calc_density(ds):
    pressure = np.zeros((len(ds.time.values),len(ds.Z.values), len(ds.YC.values), len(ds.XC.values)))

    for t in range(len(ds.time.values)):
        for z in range(len(ds.Z.values)):
            h = -ds.Z[z]
            pressure[t, z, :, :] = ATM_PRESS + RHO_REF * G * h

    salt = ds.SALT
    theta = ds.THETA
    rho_vals = density('MDJWF', salt, theta, pressure/10000)
    
    rho = xr.DataArray(
        rho_vals,
        dims=("time", "Z", "YC", "XC"),
        coords={"time": ds.time, "Z": ds.Z, "YC": ds.YC, "XC": ds.XC}
    )
    
    return rho

def compute_along_isobath_velocity(ds):
    """
    Compute velocity along the isobath (bathymetry contour) using
    MITgcm C-grid geometry with uneven grid spacing.
    """

    # 1. --- Extract needed fields ---
    Depth = ds.Depth  # (YC, XC)
    dxC = ds.dxC      # (YC, XG) - spacing for UVEL
    dyC = ds.dyC      # (YG, XC) - spacing for VVEL
    U = ds.UVEL       # (time, Z, YC, XG)
    V = ds.VVEL       # (time, Z, YG, XC)

    # -------------------------------------------
    # 2. --- Find Depth field and compute gradients (nonuniform grid) ---
    # -------------------------------------------

    def nonuniform_gradient_depth(D, dxC, dyC):
        """
        D:  (YC, XC)
        dxC: (YC, XG)  spacing for UVEL
        dyC: (YG, XC)  spacing for VVEL
        """

        # Map staggered metrics to cell centers
        dx_center = dxC.interp(XG=D.XC)
        dy_center = dyC.interp(YG=D.YC)

        # X-derivative
        dDdx = D / dx_center

        # Y-derivative
        dDdy = D / dy_center

        return dDdx, dDdy

    D_x, D_y = nonuniform_gradient_depth(Depth, dxC, dyC)

    # -------------------------------------------
    # 3. --- Construct tangent direction along the isobath ---
    # -------------------------------------------

    # Gradient vector normal to the isobath:
    #   n = (D_x, D_y)
    # Tangent is 90° left rotation:
    #   t = (-D_y, D_x)

    t_x = -D_y
    t_y = D_x

    # Normalize tangent vector
    mag = np.sqrt(t_x**2 + t_y**2)
    t_x = t_x / mag
    t_y = t_y / mag

    # -------------------------------------------
    # 4. --- Interpolate U and V to tracer points (YC, XC) ---
    # -------------------------------------------

    # UVEL is at (YC, XG): average to XC
    U_c = U.interp(XG=ds.XC)

    # VVEL is at (YG, XC): average to YC
    V_c = V.interp(YG=ds.YC)

    V_along_slope = U_c * t_x + V_c * t_y

    return V_along_slope