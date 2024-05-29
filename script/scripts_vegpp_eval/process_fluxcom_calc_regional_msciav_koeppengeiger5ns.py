'''
- calculate the uncertainty of oco2 ensemble across koeppen-geiger regions

- For each member, detrend land-area-weightded mean regional time series to get MSC and IAV
- Do the median across members to get ensemble median regional MSC or IAV

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
from apply_koeppengeiger5ns_mask import apply_koeppengeiger5ns_mask
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

dict_src = {
    'GPP': {
        'period': ('2002-01-01', '2015-12-31')
        },
    'TER': {
        'period': ('2002-01-01', '2015-12-31')
        },
    'NEE': {
        'period': ('2002-01-01', '2015-12-31')
        }
}

path_rm = os.path.join(path_bgi, 'people/hlee/data/koeppengeiger/kg5ns_regions_oneDeg.nc')
path_in = os.path.join(path_bgi, 'people/hlee/data/FLUXCOM/members_detrended')
path_out = os.path.join(path_bgi, 'people/hlee/data/FLUXCOM/members_detrended/region_koeppengeiger5ns')

# load study area common nonnan mask of forcing
path_common_nonnan = os.path.join(path_bgi, 'people/hlee/to_others/sindbad_h2m_for_2001_2019/common_nonnan_pixels.nc')
ds_common_nonnan = xr.open_dataset(path_common_nonnan)
ds_common_nonnan = ds_common_nonnan.sortby('lat', ascending=False)
ds_common_nonnan = ds_common_nonnan.sortby('lon', ascending=True)
da_common_nonnan = ds_common_nonnan.common_nonnan_pixels

# grid area
area = np.load(os.path.join(path_bgi, 'people/hlee/data/gridAreaAndLandFraction/gridAreaInKm2_180_360.npz'))['area']
area_msk = np.where(da_common_nonnan.values, area, np.nan)
area_tc = apply_koeppengeiger5ns_mask(
    dsin=ds_common_nonnan.expand_dims(dim={'time': 1}),
    pathlf='ones',
    varlf='landfraction',
    faclf='1.0',
    path_rm=path_rm,
    func_aggr='sum',
    tosave=False,
    toplot=False
)['common_nonnan_pixels'].isel(time=0)

# land fraction
path_lf = os.path.join(path_bgi, 'people/hlee/data/FLUXCOM/landfraction.360_180.nc')
ds_lf = xr.open_dataset(path_lf)
vname_lf = 'landfraction'
fac_lf = 1.0
lf_msk = np.where(da_common_nonnan.values, ds_lf[vname_lf].values * fac_lf, np.nan)

lf_tc = apply_koeppengeiger5ns_mask(
    dsin=ds_lf.expand_dims(dim={'time': 1})[[vname_lf]] * fac_lf,
    pathlf='ones',
    varlf=vname_lf,
    faclf='1.0',
    path_rm=path_rm,
    func_aggr='mean',
    tosave=False,
    toplot=False
)[vname_lf].isel(time=0)

# calculate land-area-weighted regional time series
mask_to_use = da_common_nonnan

rcnt = 9  # transcom 11; bioclimatic 11; trautmann 5; koeppengeiger 8
for v in range(len(list(dict_src.keys()))):
    vname = list(dict_src.keys())[v]

    print(f'Processing {vname}, {v+1}/{len(list(dict_src.keys()))}', flush=True)

    files = os.listdir(os.path.join(path_in))
    files = [e for e in files if vname in e]
    files.sort()

    net_msc_stack_global = []
    net_iav_stack_global = []
    net_msc_stack_regions = []
    net_iav_stack_regions = []
    for f in range(len(files)):
        date_label = files[f].split('.')[-2]
        print(f'processing {vname} files... {f+1} / {len(files)}', flush=True)

        ds = xr.open_dataset(os.path.join(path_in, files[f]))
        ds = ds.sortby('lat', ascending=False)
        ds = ds.sortby('lon', ascending=True)
        ds_msk = ds.where(mask_to_use)

        net_msc_stack_global.append(ds_msk[f'{vname}_msc'].values)
        net_iav_stack_global.append(ds_msk[f'{vname}_det'].values)
        
        # aggregate to regions
        ds_nf_tc = apply_koeppengeiger5ns_mask(
            dsin=ds_msk[[f'{vname}_msc', f'{vname}_det']],
            pathlf=path_lf,
            varlf=vname_lf,
            faclf=fac_lf,
            path_rm=path_rm,
            p_truncate=1.0,  # for each time step, exclude 0.5 p from each tail, before calculating regional mean
            tosave=False,
            toplot=False)
        ds_msc_tc = ds_nf_tc[f'{vname}_msc']
        ds_det_tc = ds_nf_tc[f'{vname}_det']

        net_msc_stack_regions.append(ds_msc_tc[f'{vname}_msc'].values)
        net_iav_stack_regions.append(ds_det_tc[f'{vname}_det'].values)

    net_msc_stack_global = np.array(net_msc_stack_global)  # (member, time, lat, lon)
    net_iav_stack_global = np.array(net_iav_stack_global)  # (member, time, lat, lon)
    net_msc_stack_regions = np.array(net_msc_stack_regions)  # (member, region, time)
    net_iav_stack_regions = np.array(net_iav_stack_regions)  # (member, region, time)

    # 12-month running mean for IAV
    net_iav_stack_regions_rm = np.apply_along_axis(running_mean, axis=2, arr=net_iav_stack_regions, N=12)  # 12-months running mean

    # save netcdf of all members
    legend_members = ['.'.join(e.split('.')[1:5]) for e in files]
    ds_tc_aggr = ds_msc_tc.copy(deep=True)
    ds_tc_aggr[f'{vname}_msc'] = ds_tc_aggr[f'{vname}_msc'].expand_dims(dim={'model': np.arange(1, len(files)+1, 1)}, axis=0)
    ds_tc_aggr[f'{vname}_msc'].values = net_msc_stack_regions
    ds_tc_aggr[f'{vname}_det'] = ds_tc_aggr[f'{vname}_msc'].copy(deep=True)
    ds_tc_aggr[f'{vname}_det'].values = net_iav_stack_regions
    ds_tc_aggr['legends_model'] = (['model'], legend_members)
    ds_tc_aggr.attrs['description'] = 'mean absolute deviation for each region and timestep calculated over area-weighted; for _det, "12-months running mean" regional time series of ensemble members; for each timestep, 0.5p of each tail was excluded from calculating the regional mean time series'
    ds_tc_aggr.to_netcdf(os.path.join(path_out, f'{vname}_RS_METEO-NONE_Koeppengeiger5nsRegions_{date_label}_studyArea_members.nc'))

    #%% regional, 12-month running-mean regional time series: IAV
    # calculate uncertainty metrics...
    unc_net = calc_mad(net_iav_stack_regions_rm, axis=0)
    ds_unc = ds_det_tc.copy(deep=True)
    ds_unc = ds_unc.rename({f'{vname}_det':f'{vname}_mad_det'})
    ds_unc[f'{vname}_mad_det'].values = unc_net
    ds_unc.attrs['description'] = 'mean absolute deviation for each region and timestep calculated over area-weighted and "12-months running mean" regional time series of ensemble members; for each timestep, 0.5p of each tail was excluded from calculating the regional mean time series'
    ds_unc.to_netcdf(os.path.join(path_out, f'EnsMAD_{vname}_RS_METEO-NONE_Koeppengeiger5nsRegions_{date_label}_studyArea_det_fromRunningMean.nc'))

    # calculate median
    ar_median = np.nanmedian(net_iav_stack_regions_rm, axis=0)
    ds_med = ds_det_tc.copy(deep=True)
    ds_med = ds_med.rename({f'{vname}_det':f'{vname}_med_det'})
    ds_med[f'{vname}_med_det'].values = ar_median
    ds_med.attrs['description'] = 'median for each region and timestep calculated over area-weighted and "12-months running mean" regional time series of ensemble members; for each timestep, 0.5p of each tail was excluded from calculating the regional mean time series'
    ds_med.to_netcdf(os.path.join(path_out, f'EnsMedian_{vname}_RS_METEO-NONE_Koeppengeiger5nsRegions_{date_label}_studyArea_det_fromRunningMean.nc'))

    #%% regional: calculations for MSC
    # calculate uncertainty metrics...
    unc_net = calc_mad(net_msc_stack_regions, axis=0)
    ds_unc = ds_det_tc.copy(deep=True)
    ds_unc = ds_unc.rename({f'{vname}_det':f'{vname}_mad_msc'})
    ds_unc[f'{vname}_mad_msc'].values = unc_net
    ds_unc.attrs['description'] = 'mean absolute deviation for each region and timestep calculated over area-weighted mean regional time series of ensemble members; for each timestep, 0.5p of each tail was excluded from calculating the regional mean time series'
    ds_unc.to_netcdf(os.path.join(path_out, f'EnsMAD_{vname}_RS_METEO-NONE_Koeppengeiger5nsRegions_{date_label}_studyArea_msc.nc'))

    # calculate median
    ar_median = np.nanmedian(net_msc_stack_regions, axis=0)
    ds_med = ds_det_tc.copy(deep=True)
    ds_med = ds_med.rename({f'{vname}_det':f'{vname}_med_msc'})
    ds_med[f'{vname}_med_msc'].values = ar_median
    ds_med.attrs['description'] = 'median for each region and timestep calculated over area-weighted mean regional time series of ensemble members; for each timestep, 0.5p of each tail was excluded from calculating the regional mean time series'
    ds_med.to_netcdf(os.path.join(path_out, f'EnsMedian_{vname}_RS_METEO-NONE_Koeppengeiger5nsRegions_{date_label}_studyArea_msc.nc'))

    #%% global:calculations for MSC
    wtm_stack = np.zeros(net_msc_stack_global.shape[:2]) * np.nan
    for m in range(len(files)):
        wtm_m = calc_land_area_weighted_mean(
            arr_data=net_msc_stack_regions[m].T,
            arr_area=area_tc,
            arr_lf=lf_tc[vname_lf].values
        )
        wtm_stack[m] = wtm_m

    # calculate uncertainty metrics...
    unc_net = calc_mad(wtm_stack, axis=0)
    ds_unc = ds_det_tc.copy(deep=True)
    ds_unc = ds_unc.drop_vars([f'{vname}_det', 'legends']).drop_dims('region')
    ds_unc[f'{vname}_mad_msc'] = ('time', unc_net)
    ds_unc.attrs['description'] = 'mean absolute deviation for each timestep calculated over area-weighted mean global time series of ensemble members; for each timestep, 0.5p of each tail was excluded from calculating the regional mean time series; for each member, regional time series was aggregated to the globe'
    ds_unc.to_netcdf(os.path.join(path_out, f'EnsMAD_{vname}_RS_METEO-NONE_global_mean_fluxes_Koeppengeiger5nsRegions_{date_label}_studyArea_msc.nc'))

    # calculate median
    ar_median = np.nanmedian(wtm_stack, axis=0)
    ds_med = ds_det_tc.copy(deep=True)
    ds_med = ds_med.drop_vars([f'{vname}_det', 'legends']).drop_dims('region')
    ds_med[f'{vname}_med_msc'] = ('time', ar_median)
    ds_med.attrs['description'] = 'median for each timestep calculated over area-weighted mean global time series of ensemble members; for each timestep, 0.5p of each tail was excluded from calculating the regional mean time series; for each member, regional time series was aggregated to the globe'
    ds_med.to_netcdf(os.path.join(path_out, f'EnsMedian_{vname}_RS_METEO-NONE_global_mean_fluxes_Koeppengeiger5nsRegions_{date_label}_studyArea_msc.nc'))

    #%% global, aggregate of 12-month running-mean regional time series: IAV

    wtm_stack = np.zeros(net_msc_stack_global.shape[:2]) * np.nan
    for m in range(len(files)):
        wtm_m = calc_land_area_weighted_mean(
            arr_data=net_iav_stack_regions_rm[m].T,
            arr_area=area_tc.common_nonnan_pixels.values,
            arr_lf=lf_tc[vname_lf].values
        )
        wtm_stack[m] = wtm_m

    # calculate uncertainty metrics...
    unc_net = calc_mad(wtm_stack, axis=0)
    ds_unc = ds_det_tc.copy(deep=True)
    ds_unc = ds_unc.drop_vars([f'{vname}_det', 'legends']).drop_dims('region')
    ds_unc[f'{vname}_mad_det'] = ('time', unc_net)
    ds_unc.attrs['description'] = 'mean absolute deviation for each timestep calculated over area-weighted and "12-months running mean" global time series of ensemble members; for each timestep, 0.5p of each tail was excluded from calculating the global mean time series'
    ds_unc.to_netcdf(os.path.join(path_out, f'EnsMAD_{vname}_RS_METEO-NONE_global_mean_fluxes_{date_label}_studyArea_det_fromKoeppengeiger5nsRegions_fromRunningMean.nc'))

    # calculate median
    ar_median = np.nanmedian(wtm_stack, axis=0)
    ds_med = ds_det_tc.copy(deep=True)
    ds_med = ds_med.drop_vars([f'{vname}_det', 'legends']).drop_dims('region')
    ds_med[f'{vname}_med_det'] = ('time', ar_median)
    ds_med.attrs['description'] = 'median for each timestep calculated over area-weighted and "12-months running mean" global time series of ensemble members; for each timestep, 0.5p of each tail was excluded from calculating the global mean time series'
    ds_med.to_netcdf(os.path.join(path_out, f'EnsMedian_{vname}_RS_METEO-NONE_global_mean_fluxes_{date_label}_studyArea_det_fromKoeppengeiger5nsRegions_fromRunningMean.nc'))