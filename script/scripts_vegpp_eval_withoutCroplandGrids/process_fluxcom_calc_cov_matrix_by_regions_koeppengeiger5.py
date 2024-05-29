'''
- apply the covariance metrix analysis to VEGPP NEE residuals
- by the bioclimatic regions by Papagiannopoulus et al. (2018)
- calculate contriubions of groups of regions (trdw - tropical/extratropical * dry/wet)

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
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from calc_cov_matrix_contributions import calc_cov_matrix_contributions
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from apply_koeppengeiger5_mask import apply_koeppengeiger5_mask
from sys import argv
import pingouin as pg
from running_mean import running_mean

def process_fluxcom_calc_cov_matrix_by_regions_koeppengeiger5():
    '''
    '''
    vname = 'GPP'  # GPP, TER, NEE; the flux variable to process
    path_out = os.path.join(path_bgi, 'people/hlee/data/FLUXCOM/cov_norm')

    path_lf = os.path.join(path_bgi, 'people/hlee/data/FLUXCOM/landfraction.360_180.nc')
    ds_lf = xr.open_dataset(path_lf)
    vname_lf = 'landfraction'
    fac_lf = 1.0

    # load studyarea mask    
    path_common_nonnan = os.path.join(path_bgi, 'people/hlee/to_others/sindbad_h2m_for_2001_2019/common_nonnan_pixels.nc')
    ds_common_nonnan = xr.open_dataset(path_common_nonnan)
    ds_common_nonnan = ds_common_nonnan.sortby('lat', ascending=False)
    ds_common_nonnan = ds_common_nonnan.sortby('lon', ascending=True)

    # load cropland mask
    path_lc = '/Net/Groups/BGI/people/hlee/data/MODIS/MCD12Q1_006_IGBP_fraction.360.180.mean.2001-2019.nc'
    ds_lc = xr.open_dataset(path_lc).isel(classes=11)  # class 11: cropland
    frac_threshold = 0.5
    ds_lc.landcover.data = np.where(ds_lc.landcover.data>frac_threshold, True, False)

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
    path_det_ens = os.path.join(path_bgi, 'people/hlee/data/FLUXCOM/members_detrended/region_koeppengeiger5')
    path_median_msc = os.path.join(path_det_ens, f'EnsMedian_{vname}_RS_METEO-NONE_Koeppengeiger5Regions_withoutCroplandGrids_2002-2015_studyArea_msc.nc')
    path_median_det = os.path.join(path_det_ens, f'EnsMedian_{vname}_RS_METEO-NONE_Koeppengeiger5Regions_withoutCroplandGrids_2002-2015_studyArea_det_fromRunningMean.nc')
    

    #%% calc for the ensemble median
    ds_med_msc = xr.open_dataset(path_median_msc)
    ds_med_det = xr.open_dataset(path_median_det)
    
    # apply the covariance matrix method
    ar_med_msc = ds_med_msc[f'{vname}_med_msc'].values
    ar_med_det = ds_med_det[f'{vname}_med_det'].values

    dict_cov_msc_med = calc_cov_matrix_contributions(ar=ar_med_msc)
    dict_cov_det_med = calc_cov_matrix_contributions(ar=ar_med_det)

    np.savez(
            os.path.join(path_out, f'koeppengeiger5_region_cov_norm_{vname}_EnsMedian_withoutCroplandGrids.npz'),
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
    path_det = os.path.join(path_bgi, 'people/hlee/data/FLUXCOM/members_detrended')
    list_files_det = os.listdir(path_det)
    list_files_det = [e for e in list_files_det if vname in e]
    list_files_det.sort()
    nmembers = len(list_files_det)
    ar_cov_norm_msc_mem_stack = np.ones((nmembers, rcnt)) * np.nan
    ar_cov_norm_det_mem_stack = np.ones((nmembers, rcnt)) * np.nan

    for f in range(nmembers):
        fname = list_files_det[f]
        fpname = fname.split('.')[2].split('-')[1]
        mlname = fname.split('.')[3].split('-')[1]
        membername = fpname+'-'+mlname

        print(f'processing {vname}, {membername}, {f+1} / {nmembers}', flush=True)

        ds_det = xr.open_dataset(os.path.join(path_det, fname))[[f'{vname}_msc', f'{vname}_det']]
        ds_det = ds_det.sortby('lat', ascending=False)
        ds_det = ds_det.sortby('lon', ascending=True)
        ds_det_msk = ds_det.where(ds_common_nonnan.common_nonnan_pixels)
        ds_det_msk = ds_det.where(~ds_lc.landcover)

        ds_det_tc = apply_koeppengeiger5_mask(
            dsin=ds_det_msk,
            pathlf=path_lf,
            varlf=vname_lf,
            faclf=fac_lf,
            path_rm=os.path.join(path_bgi, 'people/hlee/data/koeppengeiger/kg5_regions_oneDeg.nc'),
            func_aggr='mean',
            p_truncate=1.0,
            tosave=False,
            toplot=False
        )

        # apply the covariance matrix method
        ar_msc = ds_det_tc[f'{vname}_msc'][f'{vname}_msc'].values
        ar_det = ds_det_tc[f'{vname}_det'][f'{vname}_det'].values
        ar_det = np.apply_along_axis(running_mean, axis=1, arr=ar_det, N=12)  # 12-months running mean

        dict_cov_msc = calc_cov_matrix_contributions(ar=ar_msc)
        dict_cov_det = calc_cov_matrix_contributions(ar=ar_det)

        np.savez(
            os.path.join(path_out, f'koeppengeiger5_region_cov_norm_{vname}_{membername}_withoutCroplandGrids.npz'),
            ar_cov_msc=dict_cov_msc['cov_norm'].data,
            ar_cov_det=dict_cov_det['cov_norm'].data,
            ar_cov_var_msc=dict_cov_msc['cov_norm_var'].data,
            ar_cov_var_det=dict_cov_det['cov_norm_var'].data,
            ar_cov_cov_msc=dict_cov_msc['cov_norm_cov'].data,
            ar_cov_cov_det=dict_cov_det['cov_norm_cov'].data,
            sum_cov_minus_covt_msc=dict_cov_msc['sum_cov_minus_covt'].data,
            sum_cov_minus_covt_det=dict_cov_det['sum_cov_minus_covt'].data
        )

        ar_cov_norm_msc_mem_stack[f, :] = dict_cov_msc['cov_norm'].data
        ar_cov_norm_det_mem_stack[f, :] = dict_cov_det['cov_norm'].data

    np.savez(
        os.path.join(path_out, f'koeppengeiger5_region_cov_norm_{vname}_members_withoutCroplandGrids.npz'),
        ar_cov_msc=ar_cov_norm_msc_mem_stack,
        ar_cov_det=ar_cov_norm_det_mem_stack
    )

if __name__ == '__main__':
    process_fluxcom_calc_cov_matrix_by_regions_koeppengeiger5()