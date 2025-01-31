'''
- calculate the uncertainty of oco2 ensemble across koeppen-geiger regions

- For each member, detrend land-area-weightded mean regional time series to get MSC and IAV
- Do the median across members to get ensemble median regional MSC or IAV

'''

import sys
if '/Net/Groups/BGI/people/hlee/scripts/utils' not in sys.path:
    sys.path.insert(1, '/Net/Groups/BGI/people/hlee/scripts/utils')  # https://stackoverflow.com/a/4383597/7578494
if '/Net/Groups/BGI/people/hlee/scripts/diagnose_tws_nee' not in sys.path:
    sys.path.insert(1, '/Net/Groups/BGI/people/hlee/scripts/diagnose_tws_nee')  # https://stackoverflow.com/a/4383597/7578494
import numpy as np
import copy
import os
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from apply_koeppengeiger5_mask import apply_koeppengeiger5_mask
from sys import argv
from calc_land_area_weighted_mean import calc_land_area_weighted_mean
from calc_metrics import calc_mad
from running_mean import running_mean

def truncate_array(ar, p):
    '''
    ar: numpy array
    p: percentile for thresholds
    return: array in which p percentile outliers are replaces with NaN for each time step
    process:
      - set range as [1/2 p percentile, (100 - 1/2 p) percentile]
      - assign NaN values out of the range
    '''

    uplim = np.nanpercentile(ar, 100-0.5*p)
    dnlim = np.nanpercentile(ar, 0.5*p)
    ar = np.where(ar > uplim, np.nan, ar)
    ar = np.where(ar < dnlim, np.nan, ar)
    
    return ar

date_start = '2015-01-01'
date_end = '2019-12-31'
nregions = 5

## fire emission - load and convert to yearly
path_gfed = '/Net/Groups/BGI/people/hlee/data/GFED/v4/C_Emissions.2015-2020.1deg.3dim.nc'
ds_fire = xr.open_dataset(path_gfed).sel(time=slice(date_start, date_end))
ds_fire = ds_fire.sortby('lat', ascending=False)
ds_fire = ds_fire.sortby('lon', ascending=True)
ts_1520 = pd.date_range('2015-01-01', '2019-12-31', freq='1y')
days_in_year = np.where(ts_1520.is_leap_year, 366, 365)
days_in_year = np.repeat(days_in_year, 12, axis=0)
days_in_year_3d = np.tile(days_in_year, (180, 360, 1)).transpose(2, 0, 1)

path = '/Net/Groups/BGI/people/hlee/data/oco2mipv10/detrended/members'
files = os.listdir(os.path.join(path))
target_exp_label = 'LNLGIS'
member_to_exclude = ['LoFI', 'JHU', 'CMS-Flux', 'EnsMedian', 'EnsMAD', 'Region']
files = [f for f in files if f.endswith('.nc') and target_exp_label in f and all(m not in f for m in member_to_exclude)]
files = [f for f in files if not f.__contains__('GFED')]
list_member_label = [e.split('_')[0] for e in files]

path_out = '/Net/Groups/BGI/people/hlee/data/oco2mipv10/detrended'
nmonths = len(pd.date_range(date_start, date_end, freq='1M'))
net_iav_stack_global = np.ones((len(files), nmonths, 180, 360)) * np.nan  # (member, time, lat, lon)
net_msc_stack_global = np.ones((len(files), nmonths, 180, 360)) * np.nan  # (member, time, lat, lon)
net_iav_stack_regions = np.ones((len(files), nregions, nmonths)) * np.nan  # (member, region, time)
net_msc_stack_regions = np.ones((len(files), nregions, nmonths)) * np.nan  # (member, region, time)

# load study area common nonnan mask of forcing
path_common_nonnan = '/Net/Groups/BGI/people/hlee/to_others/sindbad_h2m_for_2001_2019/common_nonnan_pixels.nc'
ds_common_nonnan = xr.open_dataset(path_common_nonnan)
ds_common_nonnan = ds_common_nonnan.sortby('lat', ascending=False)
ds_common_nonnan = ds_common_nonnan.sortby('lon', ascending=True)
da_common_nonnan = ds_common_nonnan.common_nonnan_pixels

# grid area
area = np.load('/Net/Groups/BGI/people/hlee/data/gridAreaAndLandFraction/gridAreaInKm2_180_360.npz')['area']
area_msk = np.where(da_common_nonnan.values, area, np.nan)
area_tc = apply_koeppengeiger5_mask(
    dsin=ds_common_nonnan.expand_dims(dim={'time': 1}),
    pathlf='ones',
    varlf='landfraction',
    faclf='1.0',
    path_rm='/Net/Groups/BGI/people/hlee/data/koeppengeiger/kg5_regions_oneDeg.nc',
    func_aggr='sum',
    tosave=False,
    toplot=False
)['common_nonnan_pixels'].isel(time=0)

# land fraction
ds_lf = xr.open_dataset('/Net/Groups/BGI/people/hlee/data/oco2mipv10/area.ESACCI.360.180.nc')
lf_msk = np.where(da_common_nonnan.values, ds_lf.fraction.values * 0.01, np.nan)

lf_tc = apply_koeppengeiger5_mask(
    dsin=ds_lf.expand_dims(dim={'time': 1})[['fraction']] * 0.01,
    pathlf='ones',
    varlf='landfraction',
    faclf='1.0',
    path_rm='/Net/Groups/BGI/people/hlee/data/koeppengeiger/kg5_regions_oneDeg.nc',
    func_aggr='mean',
    tosave=False,
    toplot=False
)['fraction'].isel(time=0)

# calculate land-area-weighted regional time series
mask_to_use = da_common_nonnan
for f in range(len(files)):
    member_label = list_member_label[f]
    print(f'processing {member_label}, {f+1} / {len(files)}')

    ds = xr.open_dataset(os.path.join(path, files[f])).sortby('lat', ascending=False)
    ds['time'] = ds_fire['time']

    # ds = ds[['net']]
    ds_nf = ds.copy(deep=True)
    ds_nf_msk = ds_nf.where(mask_to_use)

    net_iav_stack_global[f, :, :, :] = ds_nf_msk.net_det.values
    net_msc_stack_global[f, :, :, :] = ds_nf_msk.net_msc.values
    
    # aggregate to regions
    ds_nf_tc = apply_koeppengeiger5_mask(
        dsin=ds_nf_msk[['net_det', 'net_msc']],
        pathlf='/Net/Groups/BGI/people/hlee/data/oco2mipv10/area.ESACCI.360.180.nc',
        varlf='fraction',
        faclf=0.01,
        path_rm='/Net/Groups/BGI/people/hlee/data/koeppengeiger/kg5_regions_oneDeg.nc',
        p_truncate=1.0,  # for each time step, exclude 0.5 p from each tail, before calculating regional mean
        tosave=False,
        toplot=False)
    ds_det_tc = ds_nf_tc['net_det']
    ds_msc_tc = ds_nf_tc['net_msc']

    net_iav_stack_regions[f, :, :] = ds_det_tc.net_det.values
    net_msc_stack_regions[f, :, :] = ds_msc_tc.net_msc.values

# # exclude 0.5% outliers from each tail  <-- already applied above during regional aggregation by apply_koeppengeiger5_mask()
# net_iav_stack_global_trunc = np.apply_along_axis(truncate_array, axis=2, arr=net_iav_stack_global.reshape(11, 60, -1), p=1).reshape(net_iav_stack_global.shape)  # truncate 1%
# net_msc_stack_global_trunc = np.apply_along_axis(truncate_array, axis=2, arr=net_msc_stack_global.reshape(11, 60, -1), p=1).reshape(net_msc_stack_global.shape)  # truncate 1%

# 12-month running mean for IAV
net_iav_stack_regions_rm = np.apply_along_axis(running_mean, axis=2, arr=net_iav_stack_regions, N=12)  # 12-months running mean

#%% regional, 12-month running-mean regional time series: IAV
# calculate uncertainty metrics...
unc_net = calc_mad(net_iav_stack_regions_rm, axis=0)
ds_unc = ds_det_tc.copy(deep=True)
ds_unc = ds_unc.rename({'net_det':'net_mad_nf_det'})
ds_unc['net_mad_nf_det'].values = unc_net
ds_unc.attrs['description'] = 'mean absolute deviation for each region and timestep calculated over area-weighted and "12-months running mean" regional time series of ensemble members; for each timestep, 0.5p of each tail was excluded from calculating the regional mean time series'
ds_unc.to_netcdf(os.path.join(path_out, 'EnsMAD_LNLGIS_Koeppengeiger5Regions_2015_2019_studyArea_det_fromRunningMean.nc'))

# calculate median
ar_median = np.nanmedian(net_iav_stack_regions_rm, axis=0)
ds_med = ds_det_tc.copy(deep=True)
ds_med = ds_med.rename({'net_det':'net_med_nf_det'})
ds_med['net_med_nf_det'].values = ar_median
ds_med.attrs['description'] = 'median for each region and timestep calculated over area-weighted and "12-months running mean" regional time series of ensemble members; for each timestep, 0.5p of each tail was excluded from calculating the regional mean time series'
ds_med.to_netcdf(os.path.join(path_out, 'EnsMedian_LNLGIS_Koeppengeiger5Regions_2015_2019_studyArea_det_fromRunningMean.nc'))

# #%% regional: calculations for IAV
# # calculate uncertainty metrics...
# unc_net = calc_mad(net_iav_stack_regions, axis=0)
# ds_unc = ds_det_tc.copy(deep=True)
# ds_unc = ds_unc.rename({'net_det':'net_mad_nf_det'})
# ds_unc['net_mad_nf_det'].values = unc_net
# ds_unc.attrs['description'] = 'mean absolute deviation for each region and timestep calculated over area-weighted mean regional time series of ensemble members; for each timestep, 0.5p of each tail was excluded from calculating the regional mean time series'
# ds_unc.to_netcdf(os.path.join(path_out, 'EnsMAD_LNLGIS_TranscomRegions_2015_2019_studyArea_det.nc'))

# # calculate median
# ar_median = np.nanmedian(net_iav_stack_regions, axis=0)
# ds_med = ds_det_tc.copy(deep=True)
# ds_med = ds_med.rename({'net_det':'net_med_nf_det'})
# ds_med['net_med_nf_det'].values = ar_median
# ds_med.attrs['description'] = 'median for each region and timestep calculated over area-weighted mean regional time series of ensemble members; for each timestep, 0.5p of each tail was excluded from calculating the regional mean time series'
# ds_med.to_netcdf(os.path.join(path_out, 'EnsMedian_LNLGIS_TranscomRegions_2015_2019_studyArea_det.nc'))

#%% regional: calculations for MSC
# calculate uncertainty metrics...
unc_net = calc_mad(net_msc_stack_regions, axis=0)
ds_unc = ds_det_tc.copy(deep=True)
ds_unc = ds_unc.rename({'net_det':'net_mad_nf_msc'})
ds_unc['net_mad_nf_msc'].values = unc_net
ds_unc.attrs['description'] = 'mean absolute deviation for each region and timestep calculated over area-weighted mean regional time series of ensemble members; for each timestep, 0.5p of each tail was excluded from calculating the regional mean time series'
ds_unc.to_netcdf(os.path.join(path_out, 'EnsMAD_LNLGIS_Koeppengeiger5Regions_2015_2019_studyArea_msc.nc'))

# calculate median
ar_median = np.nanmedian(net_msc_stack_regions, axis=0)
ds_med = ds_det_tc.copy(deep=True)
ds_med = ds_med.rename({'net_det':'net_med_nf_msc'})
ds_med['net_med_nf_msc'].values = ar_median
ds_med.attrs['description'] = 'median for each region and timestep calculated over area-weighted mean regional time series of ensemble members; for each timestep, 0.5p of each tail was excluded from calculating the regional mean time series'
ds_med.to_netcdf(os.path.join(path_out, 'EnsMedian_LNLGIS_Koeppengeiger5Regions_2015_2019_studyArea_msc.nc'))

# #%% global:calculations for IAV

# wtm_stack = np.zeros(net_iav_stack_global.shape[:2]) * np.nan
# for m in range(11):
#     wtm_m = calc_land_area_weighted_mean(
#         arr_data=net_iav_stack_global[m],
#         arr_area=area_msk,
#         arr_lf=lf_msk
#     )
#     wtm_stack[m] = wtm_m

# # calculate uncertainty metrics...
# unc_net = calc_mad(wtm_stack, axis=0)
# ds_unc = ds_det_tc.copy(deep=True)
# ds_unc = ds_unc.drop_vars(['net_det', 'legends']).drop_dims('region')
# ds_unc['net_mad_nf_det'] = ('time', unc_net)
# ds_unc.attrs['description'] = 'mean absolute deviation for each timestep calculated over area-weighted mean global time series of ensemble members; for each timestep, 0.5p of each tail was excluded from calculating the global mean time series'
# ds_unc.to_netcdf(os.path.join(path_out, 'EnsMAD_global_mean_fluxes_LNLGIS_2015_2019_studyArea_det.nc'))

# # calculate median
# ar_median = np.nanmedian(wtm_stack, axis=0)
# ds_med = ds_det_tc.copy(deep=True)
# ds_med = ds_med.drop_vars(['net_det', 'legends']).drop_dims('region')
# ds_med['net_med_nf_det'] = ('time', ar_median)
# ds_med.attrs['description'] = 'median for each timestep calculated over area-weighted mean global time series of ensemble members; for each timestep, 0.5p of each tail was excluded from calculating the global mean time series'
# ds_med.to_netcdf(os.path.join(path_out, 'EnsMedian_global_mean_fluxes_LNLGIS_2015_2019_studyArea_det.nc'))

#%% global:calculations for MSC
wtm_stack = np.zeros(net_msc_stack_global.shape[:2]) * np.nan
for m in range(nregions):
    wtm_m = calc_land_area_weighted_mean(
        arr_data=net_msc_stack_regions[m].T,
        arr_area=area_tc,
        arr_lf=lf_tc
    )
    wtm_stack[m] = wtm_m

# calculate uncertainty metrics...
unc_net = calc_mad(wtm_stack, axis=0)
ds_unc = ds_det_tc.copy(deep=True)
ds_unc = ds_unc.drop_vars(['net_det', 'legends']).drop_dims('region')
ds_unc['net_mad_nf_msc'] = ('time', unc_net)
ds_unc.attrs['description'] = 'mean absolute deviation for each timestep calculated over area-weighted mean global time series of ensemble members; for each timestep, 0.5p of each tail was excluded from calculating the regional mean time series; for each member, regional time series was aggregated to the globe'
ds_unc.to_netcdf(os.path.join(path_out, 'EnsMAD_global_mean_fluxes_LNLGIS_Koeppengeiger5Regions_2015_2019_studyArea_msc.nc'))

# calculate median
ar_median = np.nanmedian(wtm_stack, axis=0)
ds_med = ds_det_tc.copy(deep=True)
ds_med = ds_med.drop_vars(['net_det', 'legends']).drop_dims('region')
ds_med['net_med_nf_msc'] = ('time', ar_median)
ds_med.attrs['description'] = 'median for each timestep calculated over area-weighted mean global time series of ensemble members; for each timestep, 0.5p of each tail was excluded from calculating the regional mean time series; for each member, regional time series was aggregated to the globe'
ds_med.to_netcdf(os.path.join(path_out, 'EnsMedian_global_mean_fluxes_LNLGIS_Koeppengeiger5Regions_2015_2019_studyArea_msc.nc'))

#%% global, aggregate of 12-month running-mean regional time series: IAV

wtm_stack = np.zeros(net_iav_stack_regions_rm.shape[1:]) * np.nan
for m in range(nregions):
    wtm_m = calc_land_area_weighted_mean(
        arr_data=net_iav_stack_regions_rm[m].T,
        arr_area=area_tc.common_nonnan_pixels.values,
        arr_lf=lf_tc.fraction.values
    )
    wtm_stack[m] = wtm_m

# calculate uncertainty metrics...
unc_net = calc_mad(wtm_stack, axis=0)
ds_unc = ds_det_tc.copy(deep=True)
ds_unc = ds_unc.drop_vars(['net_det', 'legends']).drop_dims('region')
ds_unc['net_mad_nf_det'] = ('time', unc_net)
ds_unc.attrs['description'] = 'mean absolute deviation for each timestep calculated over area-weighted and "12-months running mean" global time series of ensemble members; for each timestep, 0.5p of each tail was excluded from calculating the global mean time series'
ds_unc.to_netcdf(os.path.join(path_out, 'EnsMAD_global_mean_fluxes_LNLGIS_Koeppengeiger5Regions_2015_2019_studyArea_det_fromRunningMean.nc'))

# calculate median
ar_median = np.nanmedian(wtm_stack, axis=0)
ds_med = ds_det_tc.copy(deep=True)
ds_med = ds_med.drop_vars(['net_det', 'legends']).drop_dims('region')
ds_med['net_med_nf_det'] = ('time', ar_median)
ds_med.attrs['description'] = 'median for each timestep calculated over area-weighted and "12-months running mean" global time series of ensemble members; for each timestep, 0.5p of each tail was excluded from calculating the global mean time series'
ds_med.to_netcdf(os.path.join(path_out, 'EnsMedian_global_mean_fluxes_LNLGIS_Koeppengeiger5Regions_2015_2019_studyArea_det_fromRunningMean.nc'))