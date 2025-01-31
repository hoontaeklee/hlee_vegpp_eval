'''
'''

import os
import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


path_lc = '/Net/Groups/BGI/data/DataStructureMDI/DATA/grid/Global/0d25_annual/MCD12Q1/V006/Data'
fnames = []
for yr in np.arange(2001, 2020, 1):
    fnames.append(os.path.join(path_lc, f'MCD12Q1_006_IGBP_fraction.percent.1440.720.{yr}.nc'))

ar_lc_stack = np.ones((len(fnames), 17, 720, 1440)) * np.nan
for i in range(len(fnames)):
    ds_temp = xr.open_dataset(fnames[i])
    ar_lc_stack[i, :, :] = ds_temp['MCD12Q1_006_IGBP_fraction'].data

# frame for final ds
res = 0.25  # target spatial resoluion in degree
ar_lon = np.arange(-180+res*0.5, 180-res*0.5+res, res)
ar_lat = np.flip(np.arange(-90+res*0.5, 90-res*0.5+res, res))
date_start = '2001-01-01'
date_end= '2019-12-31'
ar_time = pd.date_range(date_start, date_end, freq='Y')
ar_classes = np.arange(17)

shp = ar_lc_stack.shape
ar_nan = np.ones(shp) * np.nan
da_frame = xr.DataArray(
    data=ar_nan,
    dims=["time", 'classes', "lat", "lon"],
    coords=dict(
        time=ar_time,
        classes=ar_classes,
        lat=ar_lat,
        lon=ar_lon
    )
)
ds_frame = xr.Dataset({'landcover':da_frame})

# assign stacked annual landcover fraction
ds_lc = ds_frame.copy(deep=True)

# replace nan with 0.0 
# because the sum of land cover types of a grid cell can be larger than 1.0
# as np.nanmean() excludes np.nan from the calculation,
# the denominator for each class when do the mean becomes different to each other,
# and to make the denominators the same, different scalars are used for nominators,
# and the sum of nominators for each grid cell becomes larger than 1.0
# ...hard to explain clearly in text :|
ds_lc['landcover'].data = np.where(np.isnan(ar_lc_stack), 0.0, ar_lc_stack)  
# ds_lc = ds_lc.sortby('lat', ascending=False)

ds_lc_1deg = ds_lc.coarsen(lat=4, lon=4).reduce(np.nanmean)

# same the mean across years (mean LC fractions of the study period, 2001--2019)
ds_lc_1deg_mean = ds_lc_1deg.mean(dim='time')
ds_lc_1deg_mean.attrs = {'Documentation': 'https://www.bu.edu/lcsc/data-documentation/'}
ds_lc_1deg_mean.to_netcdf('/Net/Groups/BGI/people/hlee/data/MODIS/MCD12Q1_006_IGBP_fraction.360.180.mean.2001-2019.nc')

#%% plot: class 12 croplands
# load study area common nonnan mask of forcing
path_common_nonnan = '/Net/Groups/BGI/people/hlee/to_others/sindbad_h2m_for_2001_2019/common_nonnan_pixels.nc'
ds_common_nonnan = xr.open_dataset(path_common_nonnan)
ds_common_nonnan = ds_common_nonnan.sortby('lat', ascending=False)
ds_common_nonnan = ds_common_nonnan.sortby('lon', ascending=True)
common_nonnan = ds_common_nonnan.common_nonnan_pixels.values

fig = plt.figure(figsize=(8, 3))
gs = fig.add_gridspec(1, 1)

ax1 = fig.add_subplot(111, projection=ccrs.Robinson())
ax1.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
ax1.coastlines(resolution='auto', color='k')
gl = ax1.gridlines(color='lightgrey', linestyle='--', draw_labels=True)
gl.top_labels = False
gl.right_labels = False

ar_plot = np.where(common_nonnan, ds_lc_1deg_mean.landcover.isel(classes=11).data, np.nan)
im = ax1.imshow(ar_plot, cmap='inferno', transform=ccrs.PlateCarree())
cbar = fig.colorbar(im).set_label(label='Cropland cover fraction (-)')

axins1 = ax1.inset_axes([0.05, 0.15, 0.21, 0.21])
_data_hist = ar_plot.ravel()
_data_hist = _data_hist[~np.isnan(_data_hist)]
sns.histplot(_data_hist, stat='probability', bins=30, kde=True,
        ax=axins1, alpha=0.3, color='grey')
# axins1.set_xlim(-1.0, 1.0)
axins1.set_ylabel('')
axins1.set_yticklabels([])
axins1.spines['right'].set_visible(False)
axins1.spines['top'].set_visible(False)

fig.savefig(
    '/Net/Groups/BGI/people/hlee/data/MODIS/MCD12Q1_006_IGBP_fraction_classes_12_croplands_mean_2001-2019.png',
    dpi=600,
    transparent=False,
    facecolor='white',
    bbox_inches='tight'
)

#%% plot: class 12 croplands > 50%
# load study area common nonnan mask of forcing
path_common_nonnan = '/Net/Groups/BGI/people/hlee/to_others/sindbad_h2m_for_2001_2019/common_nonnan_pixels.nc'
ds_common_nonnan = xr.open_dataset(path_common_nonnan)
ds_common_nonnan = ds_common_nonnan.sortby('lat', ascending=False)
ds_common_nonnan = ds_common_nonnan.sortby('lon', ascending=True)
common_nonnan = ds_common_nonnan.common_nonnan_pixels.values

fig = plt.figure(figsize=(8, 3))
gs = fig.add_gridspec(1, 1)

ax1 = fig.add_subplot(111, projection=ccrs.Robinson())
ax1.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
ax1.coastlines(resolution='auto', color='k')
gl = ax1.gridlines(color='lightgrey', linestyle='--', draw_labels=True)
gl.top_labels = False
gl.right_labels = False

ar_plot = np.where(common_nonnan, ds_lc_1deg_mean.landcover.isel(classes=11).where(ds_lc_1deg_mean.landcover.isel(classes=11)>0.5).data, np.nan)

im = ax1.imshow(ar_plot, cmap='inferno', transform=ccrs.PlateCarree())
cbar = fig.colorbar(im).set_label(label='Cropland cover fraction (-)')

axins1 = ax1.inset_axes([0.05, 0.15, 0.21, 0.21])
_data_hist = ar_plot.ravel()
_data_hist = _data_hist[~np.isnan(_data_hist)]
sns.histplot(_data_hist, stat='probability', bins=30, kde=True,
        ax=axins1, alpha=0.3, color='grey')
# axins1.set_xlim(-1.0, 1.0)
axins1.set_ylabel('')
axins1.set_yticklabels([])
axins1.spines['right'].set_visible(False)
axins1.spines['top'].set_visible(False)

fig.savefig(
    '/Net/Groups/BGI/people/hlee/data/MODIS/MCD12Q1_006_IGBP_fraction_classes_12_croplands_mean_2001-2019_largerthan50.png',
    dpi=600,
    transparent=False,
    facecolor='white',
    bbox_inches='tight'
)

