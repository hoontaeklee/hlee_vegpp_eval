'''

'''

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import os

path_fcc = '/Net/Groups/BGI/people/hlee/data/hansen_forest_cover/treecover'
fnames = []
for yr in np.arange(2001, 2020, 1):
    fnames.append(os.path.join(path_fcc, f'treecover.360.180.{yr}.nc'))
ds_fcc_temp = xr.open_mfdataset(fnames)

# frame for final ds
res = 1.0  # target spatial resoluion in degree
ar_lon = np.arange(-180+res*0.5, 180-res*0.5+res, res)
ar_lat = np.arange(-90+res*0.5, 90-res*0.5+res, res)
date_start = '2001-01-01'
date_end= '2019-12-31'
ar_time = pd.date_range(date_start, date_end, freq='Y')

shp = (len(ar_time), len(ar_lat), len(ar_lon))
ar_nan = np.ones(shp) * np.nan
da_frame = xr.DataArray(
    data=ar_nan,
    dims=["time", "lat", "lon"],
    coords=dict(
        time=ar_time,
        lat=ar_lat,
        lon=ar_lon        
    )
)
ds_frame = xr.Dataset({'treecover':da_frame})

# add -90 ~ -60

ds_fcc = ds_frame.copy(deep=True)
ds_fcc['treecover'][:, 30:170, :] = ds_fcc_temp['treecover'].data
ds_fcc['treecoverchange'] = ds_fcc['treecover'][-1, :, :] - ds_fcc['treecover'][0, :, :]
ds_fcc = ds_fcc.sortby('lat', ascending=False)
ds_fcc = ds_fcc.sortby('lon', ascending=True)
ar_fcc = ds_fcc['treecoverchange'].data
ds_fcc.attrs = {
    'treecover': 'treecover2000, forest cover in %',
    'treecoverchange': f'the change in treecover from the start of period ({date_start}) to the end of period ({date_end}) (i.e., end - start)'
}
# ds_fcc.to_netcdf('/Net/Groups/BGI/people/hlee/data/hansen_forest_cover/treecover/treecover.2001-2019.180.360.nc')

#%% plot: a full map of tree cover change
fig = plt.figure(figsize=(8, 3))
gs = fig.add_gridspec(1, 1)

ax1 = fig.add_subplot(111, projection=ccrs.Robinson())
ax1.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
ax1.coastlines(resolution='auto', color='k')
gl = ax1.gridlines(color='lightgrey', linestyle='--', draw_labels=True)
gl.top_labels = False
gl.right_labels = False

p_threshold = 100
ar_plot = np.where(ar_fcc<np.nanpercentile(ds_fcc.treecoverchange, p_threshold), ar_fcc * 0.01, np.nan)
im = ax1.imshow(ar_plot, cmap='inferno', transform=ccrs.PlateCarree())
cbar = fig.colorbar(im).set_label(label='Changes in forest cover fraction (-)')

axins1 = ax1.inset_axes([0.05, 0.15, 0.21, 0.21])
_data_hist = ar_plot.ravel()
_data_hist = _data_hist[~np.isnan(_data_hist)]
sns.histplot(_data_hist, stat='probability', bins=30, kde=True,
        ax=axins1, alpha=0.3, color='grey')
axins1.vlines(x=np.nanpercentile(ds_fcc.treecoverchange, p_threshold), ymin=0, ymax=0.6, color='red', linestyles='dashed')
# axins1.set_xlim(-1.0, 1.0)
axins1.set_ylabel('')
axins1.set_yticklabels([])
axins1.spines['right'].set_visible(False)
axins1.spines['top'].set_visible(False)

fig.savefig(
    '/Net/Groups/BGI/people/hlee/data/hansen_forest_cover/figures/treecoverchange_2001-2019.png',
    dpi=600,
    transparent=False,
    facecolor='white',
    bbox_inches='tight'
)

#%% plot: a map of top-N% grid cells in tree cover change
fig = plt.figure(figsize=(8, 3))
gs = fig.add_gridspec(1, 1)

ax1 = fig.add_subplot(111, projection=ccrs.Robinson())
ax1.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
ax1.coastlines(resolution='auto', color='k')
gl = ax1.gridlines(color='lightgrey', linestyle='--', draw_labels=True)
gl.top_labels = False
gl.right_labels = False

p_threshold = 10
ar_plot = np.where(ar_fcc<np.nanpercentile(ds_fcc.treecoverchange, p_threshold), ar_fcc * 0.01, np.nan)
im = ax1.imshow(ar_plot, cmap='inferno', transform=ccrs.PlateCarree())
cbar = fig.colorbar(im).set_label(label='Changes in forest cover fraction (-)')

axins1 = ax1.inset_axes([0.05, 0.15, 0.21, 0.21])
_data_hist = np.where(ar_fcc<np.nanpercentile(ds_fcc.treecoverchange, 100), ar_fcc, np.nan).ravel()
_data_hist = _data_hist[~np.isnan(_data_hist)]
sns.histplot(_data_hist, stat='probability', bins=30, kde=True,
        ax=axins1, alpha=0.3, color='grey')
axins1.vlines(x=np.nanpercentile(ds_fcc.treecoverchange, p_threshold), ymin=0, ymax=0.6, color='red', linestyles='dashed')
# axins1.vlines(x=np.nanpercentile(ds_fcc.treecoverchange, 0), ymin=0, ymax=0.6, color='red', linestyles='dashed')
axins1.annotate(
    "",
    xy=(np.nanpercentile(ds_fcc.treecoverchange, p_threshold), 0.1),
    xytext=(np.nanpercentile(ds_fcc.treecoverchange, 0), 0.1),
    arrowprops=dict(arrowstyle="<->", color='red'))
axins1.annotate(
    'mapped',
    xy=(0.2, 0.3),
    xycoords='axes fraction'
)
# axins1.set_xlim(-1.0, 1.0)
axins1.set_ylabel('')
axins1.set_yticklabels([])
axins1.spines['right'].set_visible(False)
axins1.spines['top'].set_visible(False)

fig.savefig(
    f'/Net/Groups/BGI/people/hlee/data/hansen_forest_cover/figures/treecoverchange_2001-2019_{p_threshold}percentile.png',
    dpi=600,
    transparent=False,
    facecolor='white',
    bbox_inches='tight'
)
# %%