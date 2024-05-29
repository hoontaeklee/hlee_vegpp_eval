'''
- apply the covariance metrix analysis to VEGPP variables (NEE, GPP, TWS, SWC) (RECO?, RH?)

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
from apply_transcom_mask import apply_transcom_mask
from sys import argv
import pingouin as pg

def calc_vegpp_cov_matrix_by_grids(path_expOutput):
    '''
    path_expOutput: string, full path to a sindbad output (**not detrended**)
    '''
    # path_expOutput = '/Net/Groups/BGI/people/hlee/sindbad/data/output/VEGPP2pool1519_studyArea_10k_RD4wtnanmean_1_20230712'
    # path_expOutput = '/Net/Groups/BGI/people/hlee/sindbad/data/output/VEGPP2pool1519_studyArea_10k_RD4wtnanmean_esviav_1_20240503/detrended'
    exp_name = '_'.join(path_expOutput.split('/')[-1].split('_')[:-1])
    # vars_to_diag = ['NEE', 'cRECO', 'cRH', 'gpp', 'wTotal', 'wSoil']

    #%% some paths... landfraction, observations raw & det

    path_common_nonnan = '/Net/Groups/BGI/people/hlee/to_others/sindbad_h2m_for_2001_2019/common_nonnan_pixels.nc'
    common_nonnan = xr.open_dataset(path_common_nonnan)
    common_nonnan = common_nonnan.sortby('lat', ascending=False)
    common_nonnan = common_nonnan.sortby('lon', ascending=True)
    common_nonnan = common_nonnan.common_nonnan_pixels.values

    # grid area
    area = np.load('/Net/Groups/BGI/people/hlee/data/gridAreaAndLandFraction/gridAreaInKm2_180_360.npz')['area']
    area_msk = np.where(common_nonnan, area, np.nan)

    dict_input = {
        'NEE_mod_det': {
            'isModel': True,
            'varNameInFileName': 'NEE20152019',
            'varNameInFile':'NEE_det',
            'unit': 'gC m-2 day-1',
            'period': ('2015-01-01', '2019-12-31'),
            'convFac': 1.0,
            'path_data': '',
            'path_lf': '',
            'lf_varname': '',
            'convFacLF': ''},
        'NEE_mod_msc': {
            'isModel': True,
            'varNameInFileName': 'NEE20152019',
            'varNameInFile':'NEE_msc',
            'unit': 'gC m-2 day-1',
            'period': ('2015-01-01', '2019-12-31'),
            'convFac': 1.0,
            'path_data': '',
            'path_lf': '',
            'lf_varname': '',
            'convFacLF': ''},
        'gpp_mod_det': {
            'isModel': True,
            'varNameInFileName': 'gpp',
            'varNameInFile': 'gpp_det',
            'unit': 'gC m-2 day-1',
            'period': ('2001-01-01', '2019-12-31'),
            'convFac': 1.0,
            'path_data': '',
            'path_lf': '',
            'lf_varname': '',
            'convFacLF': ''},
        'gpp_mod_msc': {
            'isModel': True,
            'varNameInFileName': 'gpp',
            'varNameInFile': 'gpp_msc',
            'unit': 'gC m-2 day-1',
            'period': ('2001-01-01', '2019-12-31'),
            'convFac': 1.0,
            'path_data': '',
            'path_lf': '',
            'lf_varname': '',
            'convFacLF': ''},
        'cRECO_mod_det': {
            'isModel': True,
            'varNameInFileName': 'cRECO',
            'varNameInFile': 'cRECO_det',
            'unit': 'gC m-2 day-1',
            'period': ('2001-01-01', '2019-12-31'),
            'convFac': 1.0,
            'path_data': '',
            'path_lf': '',
            'lf_varname': '',
            'convFacLF': ''},
        'cRECO_mod_msc': {
            'isModel': True,
            'varNameInFileName': 'cRECO',
            'varNameInFile': 'cRECO_msc',
            'unit': 'gC m-2 day-1',
            'period': ('2001-01-01', '2019-12-31'),
            'convFac': 1.0,
            'path_data': '',
            'path_lf': '',
            'lf_varname': '',
            'convFacLF': ''}
    }

    for i, k in enumerate(dict_input.keys()):
        print(f'processing {k}, {i+1} / {len(dict_input)}', flush=True)

        #%% load data of raw signal
        
        if dict_input[k]['isModel']:
            if 'det' in k or 'msc' in k:
                dir_det_sin = os.path.join(path_expOutput)
                file_det_sin = path_expOutput.split('/')[-2] + '_' + dict_input[k]['varNameInFileName'] + '.nc'
                path_in = os.path.join(dir_det_sin, file_det_sin)
            else:
                path_in = os.path.join(path_expOutput, exp_name + '_' + dict_input[k]['varNameInFileName'] + '_3dim_fullPixel.nc')
        else:
            path_in = dict_input[k]['path_data']

        print(f'path_in: {path_in}', flush=True)

        if os.path.exists(path_in):
            ds_in = xr.open_dataset(path_in)
        else:
            print(f'File for {k} does not exist. Move to the next variable', flush=True)
            continue

        if 'msc' in dict_input[k]['varNameInFile']:
            ds_in = ds_in.isel(time=range(12))
        ds_in = ds_in.resample(time='1M').reduce(np.nanmean)
        ds_in = ds_in.sortby('lat', ascending=False)
        ds_in = ds_in.sortby('lon', ascending=True)
        
        ar_in = ds_in[dict_input[k]['varNameInFile']].values
        ar_in = np.where(common_nonnan, ar_in, np.nan)

        if dict_input[k]['path_lf']=='':
            ar_lf = np.ones_like(ar_in)  # no LF info --> 100% land
        else:
            ar_lf = xr.open_dataset(dict_input[k]['path_lf'])[dict_input[k]['lf_varname']].values
            ar_lf = ar_lf * dict_input[k]['convFacLF']  # convert to fractional unit
        ar_lf = np.where(common_nonnan, ar_lf, np.nan)

        ar_in = ar_in * area_msk * ar_lf


        #%% apply the covariance matrix method
        dict_cov = calc_cov_matrix_contributions(ar=ar_in)

        out_name = os.path.join(path_expOutput, f'cov_norm_{k}.npz')
        np.savez(
            out_name,
            cov_norm=dict_cov['cov_norm'],
            cov_norm_var=dict_cov['cov_norm_var'],
            cov_norm_cov=dict_cov['cov_norm_cov'],
            sum_cov_minus_covt=dict_cov['sum_cov_minus_covt']
        )

        del [dict_cov, ar_in] 
        ds_in.close()


if __name__ == '__main__':
    calc_vegpp_cov_matrix_by_grids(path_expOutput=argv[1])