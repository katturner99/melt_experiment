"""
Plot along-isobath density sections and differences.

Creates a 3x2 subplot showing:
- Top row: Density sections for CONTROL, LENS, MELT
- Bottom row: Differences (CONTROL - MELT, CONTROL - LENS, LENS - MELT)
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from tools.directories_and_paths import OUTPUT_PATH

# Load datasets
lens_2005 = xr.open_dataset(f"{OUTPUT_PATH}LENS_2005_density_along_isobath.nc")
lens_2100 = xr.open_dataset(f"{OUTPUT_PATH}LENS_2100_density_along_isobath.nc")
melt_2100 = xr.open_dataset(f"{OUTPUT_PATH}MELT_2100_density_along_isobath.nc")

# Extract density variable (adjust variable name if needed)
# Assuming the variable is named 'rho_mean' or similar
density_var = list(lens_2005.data_vars)[0]  # Get first data variable
rho_2005 = lens_2005[density_var]
rho_2100_lens = lens_2100[density_var]
rho_2100_melt = melt_2100[density_var]

# Calculate differences
diff_2005_melt = rho_2005 - rho_2100_melt
diff_2005_2100 = rho_2005 - rho_2100_lens
diff_lens_melt = rho_2100_lens - rho_2100_melt

# Create figure with 3x2 subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

# Common parameters for plotting
cmap_density = 'PuRd'
cmap_diff = 'PiYG_r'

# Determine global color limits for density plots
vmin_density = 1027
vmax_density = 1032

# Determine symmetric color limits for difference plots
vmax_diff = 0.2
vmin_diff = -vmax_diff

# Top row: Density sections
# Plot 1: CONTROL
im1 = axes[0, 0].pcolormesh(
    rho_2005.dist_to_isobath_bin,
    rho_2005.Z,
    rho_2005,
    cmap=cmap_density,
    vmin=vmin_density,
    vmax=vmax_density,
    shading='auto'
)
axes[0, 0].set_title('CONTROL', fontsize=14, fontweight='bold')
axes[0, 0].set_ylabel('Depth (m)', fontsize=12)
axes[0, 0].set_ylim(-1200, 0)
axes[0, 0].set_xlim(-125, 125)
axes[0, 0].set_xlabel('Distance from 1000m isobath (km)', fontsize=12)
axes[0, 0].set_facecolor("lightgrey")
fig.colorbar(im1, ax=axes[0, 0], label='Density (kg/m³)')

# Plot 2: LENS
im2 = axes[0, 1].pcolormesh(
    rho_2100_lens.dist_to_isobath_bin,
    rho_2100_lens.Z,
    rho_2100_lens,
    cmap=cmap_density,
    vmin=vmin_density,
    vmax=vmax_density,
    shading='auto'
)
axes[0, 1].set_title('LENS', fontsize=14, fontweight='bold')
axes[0, 1].set_ylabel('Depth (m)', fontsize=12)
axes[0, 1].set_ylim(-1200, 0)
axes[0, 1].set_xlim(-125, 125)
axes[0, 1].set_xlabel('Distance from 1000m isobath (km)', fontsize=12)
axes[0, 1].set_facecolor("lightgrey")
fig.colorbar(im2, ax=axes[0, 1], label='Density (kg/m³)')

# Plot 3: MELT
im3 = axes[0, 2].pcolormesh(
    rho_2100_melt.dist_to_isobath_bin,
    rho_2100_melt.Z,
    rho_2100_melt,
    cmap=cmap_density,
    vmin=vmin_density,
    vmax=vmax_density,
    shading='auto'
)
axes[0, 2].set_title('MELT', fontsize=14, fontweight='bold')
axes[0, 2].set_ylabel('Depth (m)', fontsize=12)
axes[0, 2].set_ylim(-1200, 0)
axes[0, 2].set_xlim(-125, 125)
axes[0, 2].set_xlabel('Distance from 1000m isobath (km)', fontsize=12)
axes[0, 2].set_facecolor("lightgrey")
fig.colorbar(im3, ax=axes[0, 2], label='Density (kg/m³)')

# Bottom row: Differences
# Plot 4: CONTROL - MELT
im4 = axes[1, 0].pcolormesh(
    diff_2005_melt.dist_to_isobath_bin,
    diff_2005_melt.Z,
    diff_2005_melt,
    cmap=cmap_diff,
    vmin=vmin_diff,
    vmax=vmax_diff,
    shading='auto'
)
axes[1, 0].set_title('CONTROL - MELT', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Depth (m)', fontsize=12)
axes[1, 0].set_ylim(-1200, 0)
axes[1, 0].set_xlim(-125, 125)
axes[1, 0].set_xlabel('Distance from 1000m isobath (km)', fontsize=12)
axes[1, 0].set_facecolor("lightgrey")
fig.colorbar(im4, ax=axes[1, 0], label='Density difference (kg/m³)')

# Plot 5: CONTROL - LENS
im5 = axes[1, 1].pcolormesh(
    diff_2005_2100.dist_to_isobath_bin,
    diff_2005_2100.Z,
    diff_2005_2100,
    cmap=cmap_diff,
    vmin=vmin_diff,
    vmax=vmax_diff,
    shading='auto'
)
axes[1, 1].set_title('CONTROL - LENS', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Depth (m)', fontsize=12)
axes[1, 1].set_ylim(-1200, 0)
axes[1, 1].set_xlim(-125, 125)
axes[1, 1].set_xlabel('Distance from 1000m isobath (km)', fontsize=12)
axes[1, 1].set_facecolor("lightgrey")
fig.colorbar(im5, ax=axes[1, 1], label='Density difference (kg/m³)')

# Plot 6: LENS - MELT
im6 = axes[1, 2].pcolormesh(
    diff_lens_melt.dist_to_isobath_bin,
    diff_lens_melt.Z,
    diff_lens_melt,
    cmap=cmap_diff,
    vmin=vmin_diff,
    vmax=vmax_diff,
    shading='auto'
)
axes[1, 2].set_title('LENS - MELT', fontsize=14, fontweight='bold')
axes[1, 2].set_ylabel('Depth (m)', fontsize=12)
axes[1, 2].set_ylim(-1200, 0)
axes[1, 2].set_xlim(-125, 125)
axes[1, 2].set_xlabel('Distance from 1000m isobath (km)', fontsize=12)
axes[1, 2].set_facecolor("lightgrey")
fig.colorbar(im6, ax=axes[1, 2], label='Density difference (kg/m³)')

# Save figure
plt.savefig(f"density_along_isobath_comparison.png", 
            dpi=300, bbox_inches='tight')
plt.show()

print("Figure saved successfully!")