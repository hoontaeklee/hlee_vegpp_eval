import sys
if '/Net/Groups/BGI/people/hlee/scripts/utils' not in sys.path:
    sys.path.insert(1, '/Net/Groups/BGI/people/hlee/scripts/utils')  # https://stackoverflow.com/a/4383597/7578494
if '/Net/Groups/BGI/people/hlee/scripts/diagnose_tws_nee' not in sys.path:
    sys.path.insert(1, '/Net/Groups/BGI/people/hlee/scripts/diagnose_tws_nee')  # https://stackoverflow.com/a/4383597/7578494
import numpy as np
from calc_land_area_weighted_mean import calc_land_area_weighted_mean
from calc_contrib_ahlstrom2015 import calc_contrib_ahlstrom2015
from running_mean import running_mean
import os
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from sys import argv

path_expOutput = '/Net/Groups/BGI/people/hlee/sindbad/data/output/VEGPP2pool1519_studyArea_10k_RD4wtnanmean_1_20230712/detrended'
exp_name = '_'.join(path_expOutput.split('/')[-2].split('_')[:-1])

#%% some paths... landfraction, observations raw & det

path_common_nonnan = '/Net/Groups/BGI/people/hlee/to_others/sindbad_h2m_for_2001_2019/common_nonnan_pixels.nc'
ds_common_nonnan = xr.open_dataset(path_common_nonnan)
ds_common_nonnan = ds_common_nonnan.sortby('lat', ascending=False)
ds_common_nonnan = ds_common_nonnan.sortby('lon', ascending=True)
common_nonnan = ds_common_nonnan.common_nonnan_pixels.values

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

    x_years = pd.date_range(dict_input[k]['period'][0], dict_input[k]['period'][1], freq='1MS')

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

    # load sindbad results
    if 'msc' in dict_input[k]['varNameInFile']:
        ds_in = ds_in.isel(time=range(12))
    ds_in = ds_in.resample(time='1M').reduce(np.nanmean)
    ds_in = ds_in.sortby('lat', ascending=False)
    ds_in = ds_in.sortby('lon', ascending=True)
    
    ar_in = ds_in[dict_input[k]['varNameInFile']].values
    ar_in = np.where(common_nonnan, ar_in, np.nan)

    if 'msc' in dict_input[k]['varNameInFile']:
        ar_in = ar_in - np.nanmean(ar_in, axis=0)

    # apply running mean, for det
    if 'det' in dict_input[k]['varNameInFile']:
        ar_in = np.apply_along_axis(running_mean, 0, ar_in, N=12)

    # set landfraction
    if dict_input[k]['path_lf']=='':
        ar_lf = np.ones_like(ar_in)  # no LF info --> 100% land
    else:
        ar_lf = xr.open_dataset(dict_input[k]['path_lf'])[dict_input[k]['lf_varname']].values
        ar_lf = ar_lf * dict_input[k]['convFacLF']  # convert to fractional unit

    ar_lf = np.ones_like(ar_in)  # no LF info --> 100% land
    ar_lf = np.where(common_nonnan, ar_lf, np.nan)

    _shp_lf = (len(x_years), ) + area.shape
    lf = np.ones(_shp_lf)
    _vlf = 'lf'
    _ds_lf = xr.Dataset({_vlf: (['time', 'lat', 'lon'],  lf)},
                            coords={'time': (['time'], x_years),
                                    'lat': (['lat'], np.arange(89.5, -89.5-1.0, -1)),
                                    'lon': (['lon'], np.arange(-179.5, 179.5+1.0, 1))})
    lf_msk = np.where(common_nonnan, lf, np.nan)

    # scale ar_in with grid area and land fraction to observe sum_j_(xt)=Xt for the Ahlstrom method
    ar_in_arealf = ar_in * area_msk * lf_msk[0] / np.nansum(area_msk*lf_msk[0])

    # calculate land-area-weighted global mean
    _ar_wtm_temp = ar_in
    ar_in_glo = calc_land_area_weighted_mean(
        arr_data=_ar_wtm_temp,
        arr_area=area_msk,
        arr_lf=lf_msk[0]
    )

    #%% apply the Ahlstrom 2015 method
    ar_fj = calc_contrib_ahlstrom2015(ar_x=ar_in_arealf, ar_X=ar_in_glo)
    ar_fj = np.where(common_nonnan, ar_fj, np.nan)
    print(f'The sum of contributions: {np.nansum(ar_fj).round(2)}', flush=True)
    
    out_name = os.path.join(os.path.dirname(path_expOutput), f'contrib_ahlstrom_{k}.npz')
    np.savez(
        out_name,
        contrib=ar_fj
    )

    ds_in.close()