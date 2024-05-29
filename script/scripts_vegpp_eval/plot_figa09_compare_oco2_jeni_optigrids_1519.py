'''
compare timeseries of OCO-2 NEE and Jena Inversion NEE for 2015-2019, opti904 grids
'''
import sys
if '/Net/Groups/BGI/people/hlee/scripts/utils' not in sys.path:
    sys.path.insert(1, '/Net/Groups/BGI/people/hlee/scripts/utils')  # https://stackoverflow.com/a/4383597/7578494
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from apply_transcom_mask import apply_transcom_mask
from calc_land_area_weighted_mean import calc_land_area_weighted_mean

# load nee
path_oco2 = '/Net/Groups/BGI/people/hlee/sindbad/data/input/VEGPP_1519/opti904/VEGPP_1519.NEEoco2EnvMedianGFED4FireRemoved.3dim.opti904.nc'
path_jeni = '/Net/Groups/BGI/people/hlee/sindbad/data/input/VEGPP_1519/opti904/VEGPP_1519.NEEjenainvGFED4FireRemoved.3dim.opti904.nc'

ds_oco2 = xr.open_dataset(path_oco2)
ds_jeni = xr.open_dataset(path_jeni)
time = ds_oco2.time.data
ar_oco2 = ds_oco2.net_med_nf.data
ar_jeni = ds_jeni.NEE.data

# load transcom regions
ds_tc = xr.open_dataset('/Net/Groups/BGI/people/hlee/data/transcom/TranscomRegions.360.180.nc')
rnames_tc = [x.decode('UTF-8') for x in ds_tc.Legend.values[0, :]]
# rnames_tc.insert(0, 'Excluded')
rnames_tc = rnames_tc[:11]  # exclude oceans
dict_tc = dict(zip(np.arange(1, len(rnames_tc)+1), rnames_tc))

# load opti904 mask
path_opti904_mask = '/Net/Groups/BGI/work_3/sindbad/project/twsnee_hlee/tina_veg/data/tina_veg_opti904_grids.nc'
ds_opti904_mask = xr.open_dataset(path_opti904_mask)
ds_opti904_mask = ds_opti904_mask.rename({'__xarray_dataarray_variable__':'opti904'})
ds_opti904_mask = ds_opti904_mask.sortby('lat', ascending=False)
ds_opti904_mask = ds_opti904_mask.sortby('lon', ascending=True)
opti904_mask = ds_opti904_mask.opti904.values

# grid area
area = np.load('/Net/Groups/BGI/people/hlee/data/gridAreaAndLandFraction/gridAreaInKm2_180_360.npz')['area']
area_msk = np.where(opti904_mask, area, np.nan)
area_tc = apply_transcom_mask(
    dsin=ds_opti904_mask.expand_dims(dim={'time': 1}),
    pathlf='ones',
    varlf='landfraction',
    faclf='1.0',
    path_tc='/Net/Groups/BGI/people/hlee/data/transcom/TranscomRegions.360.180.nc',
    func_aggr='sum',
    tosave=False,
    toplot=False
)['opti904'].isel(time=0)

# calc. regional mean
ds_oco2_tc = apply_transcom_mask(
    dsin=ds_oco2[['net_med_nf']],
    pathlf='ones',
    varlf='landfraction',
    faclf='1.0',
    path_tc='/Net/Groups/BGI/people/hlee/data/transcom/TranscomRegions.360.180.nc',
    p_truncate=1.0,
    tosave=False,
    toplot=False)
ds_oco2_tc = ds_oco2_tc['net_med_nf']

ds_jeni_tc = apply_transcom_mask(
    dsin=ds_jeni[['NEE']],
    pathlf='ones',
    varlf='landfraction',
    faclf='1.0',
    path_tc='/Net/Groups/BGI/people/hlee/data/transcom/TranscomRegions.360.180.nc',
    p_truncate=1.0,
    tosave=False,
    toplot=False)
ds_jeni_tc = ds_jeni_tc['NEE']

# calculate land-area-weighted global mean
_ar_wtm_temp = ds_oco2_tc['net_med_nf'].values.T
ar_oco2_glo = calc_land_area_weighted_mean(
    arr_data=_ar_wtm_temp,
    arr_area=area_tc.opti904.values
)

_ar_wtm_temp = ds_jeni_tc['NEE'].values.T
ar_jeni_glo = calc_land_area_weighted_mean(
    arr_data=_ar_wtm_temp,
    arr_area=area_tc.opti904.values
)

# plot

fsize_legend = 18
fsize_suplab = 18
fsize_text = 10
alphabets = [chr(i) for i in range(ord('a'),ord('z')+1)]
alphabets_coords = (-0.12, 1.07)

legend_handles = []
nrow = 3
ncol = 4
fig, axes = plt.subplots(
    nrow, ncol,
    figsize=(16*ncol*0.2, 9*nrow*0.2),
    gridspec_kw={
        'height_ratios': [1, 1, 1]
    }
)
axes = axes.flatten()

i=0
ax = axes[i]
l1, = ax.plot(time, ar_oco2_glo/365, color='black', label='OCO2')
l2, = ax.plot(time, ar_jeni_glo, color='red', label='JENI')
ax.annotate(f'({alphabets[i]})', xy=alphabets_coords, xycoords='axes fraction', fontsize=fsize_text, weight='bold')
ax.set_ylim(-1.0, 1.0)
ax.set_title('Globe')
legend_handles.append(l1)
legend_handles.append(l2)

for i in range(1, 12):
    ax = axes[i]
    ax.plot(time, ds_oco2_tc.sel(region=i).net_med_nf.data/365, color='black', label='oco2')
    ax.plot(time, ds_jeni_tc.sel(region=i).NEE.data, color='red', label='jeni')

    ax.set_ylim(-2.5, 2.5)

    ax.set_title(f'{i}: {dict_tc[i]}')
    ax.annotate(f'({alphabets[i]})', xy=alphabets_coords, xycoords='axes fraction', fontsize=fsize_text, weight='bold')

axes[0].legend(
    handles=legend_handles,  # [l1, l2, l3, l4],
    loc='upper center', bbox_to_anchor=(0.5, 1.15), bbox_transform=fig.transFigure,
    fontsize=fsize_legend, fancybox=False, ncol=2, frameon=False
)

fig.text(0.5, -0.10, 'Time', ha='center', fontsize=fsize_suplab)
fig.text(-0.05, 0.6, 'NEE (gC m-2 day-1)', va='center', rotation='vertical', fontsize=fsize_suplab)

fig.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98,
                    wspace=0.23, hspace=0.45)
fig.savefig(
    '/Net/Groups/BGI/people/hlee/hlee_vegpp_eval/figures/figa09_compare_oco2_jeni_optigrids_1519.png',
    dpi=600,
    transparent=False,
    bbox_inches='tight'
)
plt.clf()

