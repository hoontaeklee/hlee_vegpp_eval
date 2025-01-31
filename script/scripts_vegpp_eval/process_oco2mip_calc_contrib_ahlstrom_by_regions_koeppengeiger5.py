'''
- apply the covariance metrix analysis to VEGPP NEE residuals
- koeppen-geiger regions

- for each ensemble member and for the ensemble median

'''

import sys
import os
path_bgi = '/Net/Groups/BGI'
if os.path.join(path_bgi, 'people/hlee/scripts/utils') not in sys.path:
    sys.path.insert(1, os.path.join(path_bgi, 'people/hlee/scripts/utils'))  # https://stackoverflow.com/a/4383597/7578494
if os.path.join(path_bgi, 'people/hlee/scripts/diagnose_tws_nee') not in sys.path:
    sys.path.insert(1, os.path.join(path_bgi, 'people/hlee/scripts/diagnose_tws_nee'))  # https://stackoverflow.com/a/4383597/7578494
import numpy as np
from calc_contrib_ahlstrom2015 import calc_contrib_ahlstrom2015
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from apply_koeppengeiger5_mask import apply_koeppengeiger5_mask
from calc_land_area_weighted_mean import calc_land_area_weighted_mean
from sys import argv
from running_mean import running_mean

def process_oco2inv_members_contrib_ahlstrom_by_regions_koeppengeiger5():
    '''
    '''

    # load region mask
    # path_rm = os.path.join(path_bgi, 'data/DataStructureMDI/DATA/grid/Global/1d00_static/eco_hydro_regions/dm01/Data/class.bioclimatic.nc')
    # ds_rm = xr.open_dataset(path_rm)
    # ds_rm = ds_rm.rename({
    #     'latitude': 'lat',
    #     'longitude': 'lon',
    #     'class': 'region'
    #     })
    rcnt = 5
    # rnames = ds_rm.region.units.split(',')
    # rnames_short = ['BrWT', 'BrW', 'BrT', 'BrE', 'TsW', 'StW', 'MdT', 'MdW', 'StE', 'Tp', 'TsE']
    # colrm = ['rebeccapurple', 'mediumpurple', 'violet', 'indigo', 'palegreen', 'orange', 'green', 'sienna', 'darkgreen', 'royalblue', 'yellowgreen']
    # ar_r = ds_rm.region.values

    # load ensemble median msc and det
    path_oco2det = os.path.join(path_bgi, 'people/hlee/data/oco2mipv10/detrended')
    path_oco2median_msc = os.path.join(path_oco2det, 'EnsMedian_LNLGIS_GFED4FireRemoved_Koeppengeiger5Regions_2015_2019_studyArea_msc.nc')
    path_oco2median_det = os.path.join(path_oco2det, 'EnsMedian_LNLGIS_GFED4FireRemoved_Koeppengeiger5Regions_2015_2019_studyArea_det_fromRunningMean.nc')

    # load landmask
    ds_lf = xr.open_dataset(os.path.join(os.path.dirname(path_oco2det), 'area.ESACCI.360.180.nc'))

    # load studyarea mask    
    path_common_nonnan = os.path.join(path_bgi, 'people/hlee/to_others/sindbad_h2m_for_2001_2019/common_nonnan_pixels.nc')
    ds_common_nonnan = xr.open_dataset(path_common_nonnan)
    ds_common_nonnan = ds_common_nonnan.sortby('lat', ascending=False)
    ds_common_nonnan = ds_common_nonnan.sortby('lon', ascending=True)

    area_region = apply_koeppengeiger5_mask(
        dsin=ds_common_nonnan.expand_dims(dim={'time': 1}),
        pathlf='ones',
        varlf='landfraction',
        faclf=1.0,
        path_rm='/Net/Groups/BGI/people/hlee/data/koeppengeiger/kg5_regions_oneDeg.nc',
        func_aggr='sum',
        tosave=False,
        toplot=False
    )['common_nonnan_pixels'].isel(time=0)

    ds_lf = ds_lf.expand_dims(dim={'time': 1}) if 'time' not in list(ds_lf.dims) else ds_lf
    lf_region = apply_koeppengeiger5_mask(
        dsin=ds_lf[['fraction']],
        pathlf='ones',
        varlf='landfraction',
        faclf=1.0,
        path_rm='/Net/Groups/BGI/people/hlee/data/koeppengeiger/kg5_regions_oneDeg.nc',
        func_aggr='mean',
        tosave=False,
        toplot=False
    )['fraction'].isel(time=0)

    #%% calc for the ensemble median

    # calculate land-area-weighted global mean
    ds_med_msc = xr.open_dataset(path_oco2median_msc)
    ds_med_det = xr.open_dataset(path_oco2median_det)

    ar_med_msc = ds_med_msc['net_med_nf_msc'].values
    ar_med_det = ds_med_det['net_med_nf_det'].values

    # calculate anomalies for MSC (Ahlstrom method needs anomalies as input)
    ar_med_msc = (ar_med_msc.T - np.nanmean(ar_med_msc, axis=1)).T
    
    _ar_wtm_temp = ar_med_msc
    ar_med_msc_glo = calc_land_area_weighted_mean(
        arr_data=_ar_wtm_temp.T,
        arr_area=area_region.common_nonnan_pixels.values,
        arr_lf=lf_region.fraction.values
    )

    _ar_wtm_temp = ar_med_det
    ar_med_det_glo = calc_land_area_weighted_mean(
        arr_data=_ar_wtm_temp.T,
        arr_area=area_region.common_nonnan_pixels.values,
        arr_lf=lf_region.fraction.values
    )
    
    # apply the ahlstrom method
    _ar_x = (ar_med_msc.T * area_region.common_nonnan_pixels.values * lf_region.fraction.values)/np.nansum(area_region.common_nonnan_pixels.values * lf_region.fraction.values)
    ar_contrib_msc_med = calc_contrib_ahlstrom2015(ar_x=_ar_x, ar_X=ar_med_msc_glo)
    _ar_x = (ar_med_det.T * area_region.common_nonnan_pixels.values * lf_region.fraction.values)/np.nansum(area_region.common_nonnan_pixels.values * lf_region.fraction.values)
    ar_contrib_det_med = calc_contrib_ahlstrom2015(ar_x=_ar_x, ar_X=ar_med_det_glo)

    np.savez(
            os.path.join(path_oco2det, 'contrib_ahlstrom', f'koeppengeiger5_region_contrib_ahlstrom_EnsMedian.npz'),
            ar_contrib_msc=ar_contrib_msc_med,
            ar_contrib_det=ar_contrib_det_med
        )

    #%% calc for each ensemble member
    list_files_det = os.listdir(os.path.join(path_oco2det, 'members'))
    list_files_det.sort()
    nmembers = len(list_files_det)
    ar_contrib_msc_mem_stack = np.ones((nmembers, rcnt)) * np.nan
    ar_contrib_det_mem_stack = np.ones((nmembers, rcnt)) * np.nan

    for f in range(nmembers):
        fname = list_files_det[f]
        membername = fname.split('_')[0]

        print(f'processing {membername}, {f+1} / {nmembers}', flush=True)

        ds_det = xr.open_dataset(os.path.join(path_oco2det, 'members', fname))[['net_msc', 'net_det']]
        ds_det = ds_det.sortby('lat', ascending=False)
        ds_det = ds_det.sortby('lon', ascending=True)
        ds_det_msk = ds_det.where(ds_common_nonnan.common_nonnan_pixels)

        ds_det_tc = apply_koeppengeiger5_mask(
            dsin=ds_det_msk,
            pathlf=os.path.join(path_bgi, 'people/hlee/data/oco2mipv10/area.ESACCI.360.180.nc'),
            varlf='fraction',
            faclf=0.01,
            path_rm=os.path.join(path_bgi, 'people/hlee/data/koeppengeiger/kg5_regions_oneDeg.nc'),
            func_aggr='mean',
            p_truncate=1.0,
            tosave=False,
            toplot=False
        )

        # apply the covariance matrix method
        ar_nee_msc = ds_det_tc['net_msc']['net_msc'].values
        ar_nee_msc = (ar_nee_msc.T - np.nanmean(ar_nee_msc, axis=1)).T
    
        ar_nee_det = ds_det_tc['net_det']['net_det'].values
        ar_nee_det = np.apply_along_axis(running_mean, axis=1, arr=ar_nee_det, N=12)  # 12-months running mean

        _ar_wtm_temp = ar_nee_msc
        ar_nee_msc_glo = calc_land_area_weighted_mean(
            arr_data=_ar_wtm_temp.T,
            arr_area=area_region.common_nonnan_pixels.values,
            arr_lf=lf_region.fraction.values
        )

        _ar_wtm_temp = ar_nee_det
        ar_nee_det_glo = calc_land_area_weighted_mean(
            arr_data=_ar_wtm_temp.T,
            arr_area=area_region.common_nonnan_pixels.values,
            arr_lf=lf_region.fraction.values
        )

        _ar_x = (ar_nee_msc.T * area_region.common_nonnan_pixels.values * lf_region.fraction.values)/np.nansum(area_region.common_nonnan_pixels.values * lf_region.fraction.values)
        ar_contrib_msc = calc_contrib_ahlstrom2015(ar_x=_ar_x, ar_X=ar_nee_msc_glo)
        _ar_x = (ar_nee_det.T * area_region.common_nonnan_pixels.values * lf_region.fraction.values)/np.nansum(area_region.common_nonnan_pixels.values * lf_region.fraction.values)
        ar_contrib_det = calc_contrib_ahlstrom2015(ar_x=_ar_x, ar_X=ar_nee_det_glo)

        ar_contrib_msc_mem_stack[f, :] = ar_contrib_msc
        ar_contrib_det_mem_stack[f, :] = ar_contrib_det

    np.savez(
        os.path.join(path_oco2det, 'contrib_ahlstrom', f'koeppengeiger5_region_contrib_ahlstrom_members.npz'),
        ar_contrib_msc=ar_contrib_msc_mem_stack,
        ar_contrib_det=ar_contrib_det_mem_stack
    )

if __name__ == '__main__':
    process_oco2inv_members_contrib_ahlstrom_by_regions_koeppengeiger5()