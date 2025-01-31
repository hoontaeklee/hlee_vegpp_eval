
path_bgi='/Net/Groups/BGI'
import os
import sys
if os.path.join(path_bgi, 'people/hlee/scripts/utils') not in sys.path:
    sys.path.insert(1, os.path.join(path_bgi, 'people/hlee/scripts/utils'))  # https://stackoverflow.com/a/4383597/7578494
if os.path.join(path_bgi, 'people/hlee/scripts/diagnose_tws_nee') not in sys.path:
    sys.path.insert(1, os.path.join(path_bgi, 'people/hlee/scripts/diagnose_tws_nee'))  # https://stackoverflow.com/a/4383597/7578494
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import colors
import json
import ast

#%% load data

# load study area common nonnan mask of forcing
path_common_nonnan = os.path.join(path_bgi, 'people/hlee/to_others/sindbad_h2m_for_2001_2019/common_nonnan_pixels.nc')
common_nonnan = xr.open_dataset(path_common_nonnan)
common_nonnan = common_nonnan.sortby('lat', ascending=False)
common_nonnan = common_nonnan.sortby('lon', ascending=True)
common_nonnan = common_nonnan.common_nonnan_pixels.values

#%% plot - studyareas
alphabets_coords = (-0.09, 0.95)
fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(2, 1)

# a) koeppen-geiger5
path_rm = os.path.join(path_bgi, 'people/hlee/data/koeppengeiger/kg5_regions_oneDeg.nc')
ds_rm = xr.open_dataset(path_rm)
rcnt = 5
rnames_rm = list(json.loads(ds_rm.attrs['Legends'])['meaning'].values())
rnames_short = list(json.loads(ds_rm.attrs['Legends'])['meaning'].values())
rnames_short = [e.replace('Tropic', 'Tropical') for e in rnames_short]
colrm = ['royalblue', 'skyblue', 'sienna', 'forestgreen', 'rebeccapurple']
ar_rm = ds_rm.kgRegions.values
dict_rm = dict(zip(np.arange(1, len(rnames_rm)+1), rnames_rm))
msk = ar_rm

list_rorder = [1, 2, 3, 4, 5]

ar_plot = np.ones_like(ar_rm) * np.nan
for e in range(rcnt):  # reorder region index
    ar_plot = np.where(ar_rm==list_rorder[e], e+1, ar_plot)
ar_plot_studyarea = np.where(common_nonnan, ar_plot, np.nan)

cmap = colors.ListedColormap([colrm[e-1] for e in list_rorder])
bounds = np.arange(0.5, rcnt+1.5, 1)
norm = colors.BoundaryNorm(bounds, cmap.N)

ax1 = fig.add_subplot(211, projection=ccrs.Robinson())
ax1.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
ax1.coastlines(resolution='auto', color='k')
gl = ax1.gridlines(color='lightgrey', linestyle='--', draw_labels=True)
gl.top_labels = False
gl.right_labels = False
im = ax1.imshow(ar_plot_studyarea, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

cbar = fig.colorbar(im)
cbar.set_ticks(np.arange(1, rcnt+1, 1))
cbar.set_ticklabels([rnames_short[e-1] for e in list_rorder])
ax1.annotate(f'(a)', xy=alphabets_coords, xycoords='axes fraction', fontsize=13, weight='bold')

# b) koeppen-geiger5 ns
path_rm = os.path.join(path_bgi, 'people/hlee/data/koeppengeiger/kg5ns_regions_oneDeg.nc')
ds_rm = xr.open_dataset(path_rm)
rcnt = 9
rnames_rm = list(ast.literal_eval(ds_rm.attrs['Legends'])['meaning'].values())
rnames_short = list(ast.literal_eval(ds_rm.attrs['Legends'])['meaning'].values())
colrm = ['blue', 'royalblue', 'cadetblue', 'skyblue', 'brown', 'sienna', 'darkgreen', 'forestgreen', 'rebeccapurple']
ar_rm = ds_rm.kgRegions.values
dict_rm = dict(zip(np.arange(1, len(rnames_rm)+1), rnames_rm))
msk = ar_rm

list_rorder = list(np.arange(len(rnames_rm))+1)

ar_plot = np.ones_like(ar_rm) * np.nan
for e in range(rcnt):  # reorder region index
    ar_plot = np.where(ar_rm==list_rorder[e], e+1, ar_plot)
ar_plot_studyarea = np.where(common_nonnan, ar_plot, np.nan)

cmap = colors.ListedColormap([colrm[e-1] for e in list_rorder])
bounds = np.arange(0.5, rcnt+1.5, 1)
norm = colors.BoundaryNorm(bounds, cmap.N)

ax2 = fig.add_subplot(212, projection=ccrs.Robinson())
ax2.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
ax2.coastlines(resolution='auto', color='k')
gl = ax2.gridlines(color='lightgrey', linestyle='--', draw_labels=True)
gl.top_labels = False
gl.right_labels = False
im = ax2.imshow(ar_plot_studyarea, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())

cbar = fig.colorbar(im)
cbar.set_ticks(np.arange(1, rcnt+1, 1))
cbar.set_ticklabels([rnames_short[e-1] for e in list_rorder])
ax2.annotate(f'(b)', xy=alphabets_coords, xycoords='axes fraction', fontsize=13, weight='bold')

fig.tight_layout()

fig.savefig(
    '/Net/Groups/BGI/people/hlee/hlee_vegpp_eval/figures/figa10_regions_koeppengeiger5_koeppengeiger5ns.png',
    dpi=600,
    transparent=False,
    facecolor='white',
    bbox_inches='tight'
)

plt.clf()

