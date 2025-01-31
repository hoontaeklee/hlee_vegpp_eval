'''
check how many heavily deforested and cultivated grid cells the opti904 mask includes...

- if a lot (?) --> sindbad may know about deforestation and cultivation
- if yes --> doesn't know
'''

import sys
import os
path_bgi = '/Net/Groups/BGI'
if os.path.join(path_bgi, 'people/hlee/scripts/utils') not in sys.path:
    sys.path.insert(1, os.path.join(path_bgi, 'people/hlee/scripts/utils'))  # https://stackoverflow.com/a/4383597/7578494
if os.path.join(path_bgi, 'people/hlee/scripts/diagnose_tws_nee') not in sys.path:
    sys.path.insert(1, os.path.join(path_bgi, 'people/hlee/scripts/diagnose_tws_nee'))  # https://stackoverflow.com/a/4383597/7578494
import numpy as np
import copy
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs

#%% load masks

# load study area common nonnan mask of forcing
path_opti904 = os.path.join(path_bgi, 'work_3/sindbad/project/twsnee_hlee/tina_veg/data/tina_veg_opti904_grids.nc')
da_opti904 = xr.open_dataarray(path_opti904)
da_opti904 = da_opti904.sortby('lat', ascending=False)
da_opti904 = da_opti904.sortby('lon', ascending=True)
ar_opti904 = da_opti904.data

# load deforested grid cells
path_fcc = os.path.join(path_bgi, 'people/hlee/data/hansen_forest_cover/treecover/treecover.2001-2019.180.360.nc')
ds_fcc = xr.open_dataset(path_fcc)
p_threshold = 10
ds_fcc['treecoverchange'].data = np.where(ds_fcc['treecoverchange'].data<np.nanpercentile(ds_fcc.treecoverchange, p_threshold), True, False)
ar_fcc = ds_fcc['treecoverchange'].data

# load cultivated grid cells
path_lc = os.path.join(path_bgi, 'people/hlee/data/MODIS/MCD12Q1_006_IGBP_fraction.360.180.mean.2001-2019.nc')
ds_lc = xr.open_dataset(path_lc).isel(classes=11)  # class 11: cropland
frac_threshold = 0.5
ds_lc.landcover.data = np.where(ds_lc.landcover.data>frac_threshold, True, False)
ar_lc = ds_lc['landcover'].data

#%% plot: a map of top-N% grid cells in tree cover change
alphabets = [chr(i) for i in range(ord('a'),ord('z')+1)]
alphabets_coords = (-0.09, 1.07)    

fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(2, 1)

ax1 = fig.add_subplot(211, projection=ccrs.Robinson())
ax1.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
ax1.coastlines(resolution='auto', color='k')
gl = ax1.gridlines(color='lightgrey', linestyle='--', draw_labels=True)
gl.top_labels = False
gl.right_labels = False

ar_plot = np.where(ar_opti904*ar_fcc, ar_opti904*ar_fcc, np.nan)
im = ax1.imshow(ar_plot, interpolation='none', cmap='coolwarm_r', transform=ccrs.PlateCarree())
# cbar = fig.colorbar(im)

ratio_intersection = ((100 * ar_opti904*ar_fcc).sum() / ar_opti904.sum()).round(2)
ax1.annotate(f'{(ar_opti904*ar_fcc).sum()} grids ({ratio_intersection}%)', xy=(0.01, 0.25), xycoords='axes fraction', fontsize=11)
ax1.annotate(f'({alphabets[0]})', xy=alphabets_coords, xycoords='axes fraction', fontsize=13, weight='bold')
ax1.set_title('Heavily deforested grid cells used for parameter calibration')

ax2 = fig.add_subplot(212, projection=ccrs.Robinson())
ax2.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
ax2.coastlines(resolution='auto', color='k')
gl = ax2.gridlines(color='lightgrey', linestyle='--', draw_labels=True)
gl.top_labels = False
gl.right_labels = False

ar_plot = np.where(ar_opti904*ar_lc, ar_opti904*ar_lc, np.nan)
im = ax2.imshow(ar_plot, interpolation='none', cmap='coolwarm_r', transform=ccrs.PlateCarree())
# cbar = fig.colorbar(im)

ratio_intersection = (100 * (ar_opti904*ar_lc).sum() / ar_opti904.sum()).round(2)
ax2.annotate(f'{(ar_opti904*ar_lc).sum()} grids ({ratio_intersection}%)', xy=(0.01, 0.25), xycoords='axes fraction', fontsize=11)
ax2.annotate(f'({alphabets[1]})', xy=alphabets_coords, xycoords='axes fraction', fontsize=13, weight='bold')
ax2.set_title('Heavily cultivated grid cells used for parameter calibration')

fig.tight_layout()

fig.savefig(
    f'/Net/Groups/BGI/people/hlee/hlee_vegpp_eval/figures/overlap_opti904_deforestation_cultivation.png',
    dpi=600,
    transparent=False,
    facecolor='white',
    bbox_inches='tight'
)

# %%