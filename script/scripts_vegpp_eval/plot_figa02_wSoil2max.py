'''
'''
# =============================================================================
# load libraries and functions
# =============================================================================

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# =============================================================================
# load data
# =============================================================================

path = '/Net/Groups/BGI/people/hlee/sindbad/data/output/VEGPP2pool1519_studyArea_10k_RD4wtnanmean_1_20230712/VEGPP2pool1519_studyArea_10k_RD4wtnanmean_1_p_wSoilBase_wSat2_3dim_fullPixel.nc'
nc = xr.open_dataset(path).isel(time=0)
soil_cap = np.flip(nc.p_wSoilBase_wSat2.values, axis=0)

path_common_nonnan = '/Net/Groups/BGI/people/hlee/to_others/sindbad_h2m_for_2001_2019/common_nonnan_pixels.nc'
common_nonnan = xr.open_dataset(path_common_nonnan).common_nonnan_pixels.values
soil_cap = np.where(common_nonnan, soil_cap, np.nan)

# =============================================================================
# plot
# =============================================================================
extent = [-179.5 , 179.5, -59.5 , 89.5]
vmin=0
vmax=2000
cmap='viridis'
alphabets_coords = (-0.09, 1.07)

fig = plt.figure(figsize=(9, 3))
gs = fig.add_gridspec(nrows=1, ncols=2, figure=fig, width_ratios=[1, 0.33])
                      
ax = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson(central_longitude=0), frameon=False)
ax.add_feature(cfeature.LAKES, alpha=0.1, color='black')
ax.add_feature(cfeature.RIVERS, color='black')
ax.coastlines()
im = ax.imshow(soil_cap[:150, :], extent=extent, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())

ax_cb = fig.add_axes([0.17, 0.03, 0.5, 0.025])
cb = plt.colorbar(im, cax=ax_cb, extend='max', orientation='horizontal')
cb.ax.set_xlabel('Soil water storage capacity (mm)')
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.annotate(f'(a)', xy=(0.1, 1.00), xycoords='axes fraction', fontsize=14, weight='bold')

ax_box = [0.84, 0.03, 0.25, 0.97]
ax = plt.axes(ax_box, frameon=True)
ax.hist(soil_cap.ravel(), bins=100, density=True, alpha=0.5, color='green')
ax.set_xlabel('Soil water storage capacity (mm)')
ax.set_ylabel('Density')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.annotate(f'(b)', xy=(-0.45, 0.885), xycoords='axes fraction', fontsize=14, weight='bold')

fig.tight_layout()

save_name = '/Net/Groups/BGI/people/hlee/hlee_vegpp_eval/figures/figa02_wSoil2max.png'
fig.savefig(save_name,
            dpi=600,
            bbox_inches='tight', 
            facecolor='w',
            transparent=False)