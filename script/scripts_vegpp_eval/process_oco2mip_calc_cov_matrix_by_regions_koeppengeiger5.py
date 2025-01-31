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
import copy
from calc_cov_matrix_contributions import calc_cov_matrix_contributions
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from apply_koeppengeiger5_mask import apply_koeppengeiger5_mask
from sys import argv
from running_mean import running_mean

def process_oco2inv_members_cov_matrix_by_regions_koeppengeiger5():
    '''
    '''

    # load studyarea mask    
    path_common_nonnan = os.path.join(path_bgi, 'people/hlee/to_others/sindbad_h2m_for_2001_2019/common_nonnan_pixels.nc')
    ds_common_nonnan = xr.open_dataset(path_common_nonnan)
    ds_common_nonnan = ds_common_nonnan.sortby('lat', ascending=False)
    ds_common_nonnan = ds_common_nonnan.sortby('lon', ascending=True)

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


    #%% calc for the ensemble median
    ds_med_msc = xr.open_dataset(path_oco2median_msc)
    ds_med_det = xr.open_dataset(path_oco2median_det)
    
    # apply the covariance matrix method
    ar_med_msc = ds_med_msc['net_med_nf_msc'].values
    ar_med_det = ds_med_det['net_med_nf_det'].values

    dict_cov_msc_med = calc_cov_matrix_contributions(ar=ar_med_msc)
    dict_cov_det_med = calc_cov_matrix_contributions(ar=ar_med_det)

    np.savez(
            os.path.join(path_oco2det, 'cov_norm', f'koeppengeiger5_region_cov_norm_EnsMedian.npz'),
            ar_cov_msc=dict_cov_msc_med['cov_norm'].data,
            ar_cov_det=dict_cov_det_med['cov_norm'].data,
            ar_cov_var_msc=dict_cov_msc_med['cov_norm_var'].data,
            ar_cov_var_det=dict_cov_det_med['cov_norm_var'].data,
            ar_cov_cov_msc=dict_cov_msc_med['cov_norm_cov'].data,
            ar_cov_cov_det=dict_cov_det_med['cov_norm_cov'].data,
            sum_cov_minus_covt_msc=dict_cov_msc_med['sum_cov_minus_covt'].data,
            sum_cov_minus_covt_det=dict_cov_det_med['sum_cov_minus_covt'].data
        )

    #%% calc for each ensemble member
    list_files_det = os.listdir(os.path.join(path_oco2det, 'members'))
    list_files_det.sort()
    nmembers = len(list_files_det)
    ar_cov_norm_msc_mem_stack = np.ones((nmembers, rcnt)) * np.nan
    ar_cov_norm_det_mem_stack = np.ones((nmembers, rcnt)) * np.nan

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
        ar_nee_det = ds_det_tc['net_det']['net_det'].values
        ar_nee_det = np.apply_along_axis(running_mean, axis=1, arr=ar_nee_det, N=12)  # 12-months running mean

        dict_cov_msc = calc_cov_matrix_contributions(ar=ar_nee_msc)
        dict_cov_det = calc_cov_matrix_contributions(ar=ar_nee_det)

        ar_cov_norm_msc_mem_stack[f, :] = dict_cov_msc['cov_norm'].data
        ar_cov_norm_det_mem_stack[f, :] = dict_cov_det['cov_norm'].data

    np.savez(
        os.path.join(path_oco2det, 'cov_norm', f'koeppengeiger5_region_cov_norm_members.npz'),
        ar_cov_msc=ar_cov_norm_msc_mem_stack,
        ar_cov_det=ar_cov_norm_det_mem_stack
    )

if __name__ == '__main__':
    process_oco2inv_members_cov_matrix_by_regions_koeppengeiger5()