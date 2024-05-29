'''
- apply the covariance metrix analysis to VEGPP NEE residuals
- by the regions by Trautmann et al. (2022)

'''

import sys
if '/Net/Groups/BGI/people/hlee/scripts/utils' not in sys.path:
    sys.path.insert(1, '/Net/Groups/BGI/people/hlee/scripts/utils')  # https://stackoverflow.com/a/4383597/7578494
if '/Net/Groups/BGI/people/hlee/scripts/diagnose_tws_nee' not in sys.path:
    sys.path.insert(1, '/Net/Groups/BGI/people/hlee/scripts/diagnose_tws_nee')  # https://stackoverflow.com/a/4383597/7578494
import numpy as np
import copy
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from calc_cov_matrix_contributions import calc_cov_matrix_contributions
import os
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from apply_koeppengeiger5_mask import apply_koeppengeiger5_mask
from sys import argv
import pingouin as pg

def calc_vegpp_cov_matrix_by_regions_koeppengeiger5(path_expOutput):
    '''
    path_expOutput: string, full path to a sindbad output (**not detrended**)
    '''
    # path_expOutput = '/Net/Groups/BGI/people/hlee/sindbad/data/output/VEGPP2pool1519_studyArea_10k_RD4wtnanmean_1_20230712/detrended'
    # path_expOutput = '/Net/Groups/BGI/people/hlee/sindbad/data/output/VEGPP2pool1519_studyArea_10k_RD4wtnanmean_jeni_1_20240314'
    exp_name = '_'.join(path_expOutput.split('/')[-1].split('_')[:-1])
    vars_to_diag = ['NEE', 'cRECO', 'gpp', 'wTotal', 'evapTotal', 'roTotal', 'wSnow']

    #%% some paths... landfraction, observations raw & det

    path_common_nonnan = '/Net/Groups/BGI/people/hlee/to_others/sindbad_h2m_for_2001_2019/common_nonnan_pixels.nc'
    common_nonnan = xr.open_dataset(path_common_nonnan)
    common_nonnan = common_nonnan.sortby('lat', ascending=False)
    common_nonnan = common_nonnan.sortby('lon', ascending=True)
    common_nonnan = common_nonnan.common_nonnan_pixels.values

    # grid area
    area = np.load('/Net/Groups/BGI/people/hlee/data/gridAreaAndLandFraction/gridAreaInKm2_180_360.npz')['area']
    area_msk = np.where(common_nonnan, area, np.nan)

    files_lf = {
        'evapTotal': ['/Net/Groups/BGI/people/hlee/data/FLUXCOM/landfraction.360_180.nc', 'landfraction', 1.0],
        'gpp': ['/Net/Groups/BGI/people/hlee/data/FLUXCOM/landfraction.360_180.nc', 'landfraction', 1.0],
        'cRECO': ['/Net/Groups/BGI/people/hlee/data/FLUXCOM/landfraction.360_180.nc', 'landfraction', 1.0],
        'NEE': ['/Net/Groups/BGI/people/hlee/data/oco2mipv10/area.ESACCI.360.180.nc', 'fraction', 0.01]
    }

    for i, var_to_diag in enumerate(vars_to_diag):
        print(f'processing {var_to_diag}, {i+1} / {len(vars_to_diag)}', flush=True)

        #%% load data
        nee_label = '20152019' if var_to_diag=='NEE' else ''
        dir_det_sin = path_expOutput
        file_det_sin = path_expOutput.split('/')[-2] + '_' + var_to_diag + nee_label + '.nc'
        path_det_sin = os.path.join(dir_det_sin, file_det_sin)
        if file_det_sin in os.listdir(dir_det_sin):
            ds_det_sin = xr.open_dataset(path_det_sin)[[var_to_diag+'_det']]
            ds_det_sin = ds_det_sin.sortby('lat', ascending=False)
            ds_det_sin = ds_det_sin.sortby('lon', ascending=True)
            
            ds_msc_sin = xr.open_dataset(path_det_sin)[[var_to_diag+'_msc']]
            ds_msc_sin = ds_msc_sin.sortby('lat', ascending=False)
            ds_msc_sin = ds_msc_sin.sortby('lon', ascending=True)
        else:
            msg = [
                    f'The netcdf for observations of desired variable does not exist.'
            ]
            exit(print('\n'.join(msg)))


        #%% apply bioclimatic mask

        ds_det_sin_tc = apply_koeppengeiger5_mask(
            dsin=ds_det_sin,
            pathlf=files_lf[var_to_diag][0] if var_to_diag in files_lf.keys() else 'ones',
            varlf=files_lf[var_to_diag][1] if var_to_diag in files_lf.keys() else 'landfraction',
            faclf=files_lf[var_to_diag][2] if var_to_diag in files_lf.keys() else '1.0',
            path_rm='/Net/Groups/BGI/people/hlee/data/koeppengeiger/kg5_regions_oneDeg.nc',
            p_truncate=1.0,
            func_aggr='mean',
            tosave=False,
            toplot=False
        )
        ds_det_sin_tc = ds_det_sin_tc[var_to_diag+'_det']

        ds_msc_sin_tc = apply_koeppengeiger5_mask(
            dsin=ds_msc_sin,
            pathlf=files_lf[var_to_diag][0] if var_to_diag in files_lf.keys() else 'ones',
            varlf=files_lf[var_to_diag][1] if var_to_diag in files_lf.keys() else 'landfraction',
            faclf=files_lf[var_to_diag][2] if var_to_diag in files_lf.keys() else '1.0',
            path_rm='/Net/Groups/BGI/people/hlee/data/koeppengeiger/kg5_regions_oneDeg.nc',
            p_truncate=1.0,
            func_aggr='mean',
            tosave=False,
            toplot=False
        )
        ds_msc_sin_tc = ds_msc_sin_tc[var_to_diag+'_msc']

        #%% apply the covariance matrix method
        ar_cov_msc_sin = calc_cov_matrix_contributions(ar=ds_msc_sin_tc[var_to_diag+'_msc'].values)['cov_norm'].data
        ar_cov_det_sin = calc_cov_matrix_contributions(ar=ds_det_sin_tc[var_to_diag+'_det'].values)['cov_norm'].data

        ar_cov_var_msc_sin = calc_cov_matrix_contributions(ar=ds_msc_sin_tc[var_to_diag+'_msc'].values)['cov_norm_var'].data
        ar_cov_var_det_sin = calc_cov_matrix_contributions(ar=ds_det_sin_tc[var_to_diag+'_det'].values)['cov_norm_var'].data

        ar_cov_cov_msc_sin = calc_cov_matrix_contributions(ar=ds_msc_sin_tc[var_to_diag+'_msc'].values)['cov_norm_cov'].data
        ar_cov_cov_det_sin = calc_cov_matrix_contributions(ar=ds_det_sin_tc[var_to_diag+'_det'].values)['cov_norm_cov'].data

        np.savez(
            os.path.join(path_expOutput, f'koeppengeiger5_region_cov_norm_{var_to_diag}.npz'),
            ar_cov_msc_sin=ar_cov_msc_sin,
            ar_cov_det_sin=ar_cov_det_sin,
            ar_cov_var_msc_sin=ar_cov_var_msc_sin,
            ar_cov_var_det_sin=ar_cov_var_det_sin,
            ar_cov_cov_msc_sin=ar_cov_cov_msc_sin,
            ar_cov_cov_det_sin=ar_cov_cov_det_sin
        )

        for d in [v for v in dir() if v[:2]=='ds']:
            print(f'd={d}')
            exec(d+'.close()')

if __name__ == '__main__':
    calc_vegpp_cov_matrix_by_regions_koeppengeiger5(path_expOutput=argv[1])