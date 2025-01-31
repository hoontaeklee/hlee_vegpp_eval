'''
- apply the spatial attribution analysis by Ahlstrom et al (2015) to VEGPP NEE residuals
- by the regions by Trautmann et al. (2022)

'''

import sys
if '/Net/Groups/BGI/people/hlee/scripts/utils' not in sys.path:
    sys.path.insert(1, '/Net/Groups/BGI/people/hlee/scripts/utils')  # https://stackoverflow.com/a/4383597/7578494
if '/Net/Groups/BGI/people/hlee/scripts/diagnose_tws_nee' not in sys.path:
    sys.path.insert(1, '/Net/Groups/BGI/people/hlee/scripts/diagnose_tws_nee')  # https://stackoverflow.com/a/4383597/7578494
import numpy as np
from calc_land_area_weighted_mean import calc_land_area_weighted_mean
from calc_contrib_ahlstrom2015 import calc_contrib_ahlstrom2015
import os
import xarray as xr
import pandas as pd
from apply_koeppengeiger5_mask import apply_koeppengeiger5_mask
from sys import argv

def calc_vegpp_contrib_ahlstrom2015_koeppengeiger5(path_expOutput):
    '''
    path_expOutput: string, full path to a sindbad output (**not detrended**)
    '''
    # path_expOutput = '/Net/Groups/BGI/people/hlee/sindbad/data/output/VEGPP2pool1519_studyArea_10k_RD4wtnanmean_1_20230712/detrended'
    # path_expOutput = '/Net/Groups/BGI/people/hlee/sindbad/data/output/VEGPP2pool1519_studyArea_10k_RD4wtnanmean_jeni_1_20240314'
    exp_name = '_'.join(path_expOutput.split('/')[-2].split('_')[:-1])
    vars_to_diag = ['NEE', 'cRECO', 'gpp']

    #%% some paths... landfraction, observations raw & det

    path_common_nonnan = '/Net/Groups/BGI/people/hlee/to_others/sindbad_h2m_for_2001_2019/common_nonnan_pixels.nc'
    ds_common_nonnan = xr.open_dataset(path_common_nonnan)
    ds_common_nonnan = ds_common_nonnan.sortby('lat', ascending=False)
    ds_common_nonnan = ds_common_nonnan.sortby('lon', ascending=True)
    common_nonnan = ds_common_nonnan.common_nonnan_pixels.values

    # grid area
    area = np.load('/Net/Groups/BGI/people/hlee/data/gridAreaAndLandFraction/gridAreaInKm2_180_360.npz')['area']
    area_msk = np.where(common_nonnan, area, np.nan)

    files_lf = {
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
            ds_det_sin = ds_det_sin.where(ds_common_nonnan.common_nonnan_pixels)
            
            ds_msc_sin = xr.open_dataset(path_det_sin)[[var_to_diag+'_msc']]
            ds_msc_sin = ds_msc_sin.sortby('lat', ascending=False)
            ds_msc_sin = ds_msc_sin.sortby('lon', ascending=True)
            ds_msc_sin = ds_msc_sin.where(ds_common_nonnan.common_nonnan_pixels)

            # # calculate anomalies for MSC (Ahlstrom method needs anomalies as input)
            # ds_msc_sin[var_to_diag+'_msc'] = (ds_msc_sin[var_to_diag+'_msc'] - ds_msc_sin[var_to_diag+'_msc'].mean('time', skipna=True))
        else:
            msg = [
                    f'The netcdf for observations of desired variable does not exist.'
            ]
            exit(print('\n'.join(msg)))

        #%% apply region mask

        ds_det_sin_region = apply_koeppengeiger5_mask(
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
        ds_det_sin_region = ds_det_sin_region[var_to_diag+'_det']

        ds_msc_sin_region = apply_koeppengeiger5_mask(
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
        ds_msc_sin_region = ds_msc_sin_region[var_to_diag+'_msc']

        # calculate anomalies for MSC (Ahlstrom method needs anomalies as input)
        ds_msc_sin_region[var_to_diag+'_msc'] = (ds_msc_sin_region[var_to_diag+'_msc'] - ds_msc_sin_region[var_to_diag+'_msc'].mean('time', skipna=True))

        #%% calc. regional area and landfraction
        # area
        area_region = apply_koeppengeiger5_mask(
            dsin=ds_common_nonnan.expand_dims(dim={'time': 1}),
            pathlf=files_lf[var_to_diag][0] if var_to_diag in files_lf.keys() else 'ones',
            varlf=files_lf[var_to_diag][1] if var_to_diag in files_lf.keys() else 'landfraction',
            faclf=files_lf[var_to_diag][2] if var_to_diag in files_lf.keys() else '1.0',
            path_rm='/Net/Groups/BGI/people/hlee/data/koeppengeiger/kg5_regions_oneDeg.nc',
            func_aggr='sum',
            tosave=False,
            toplot=False
        )['common_nonnan_pixels'].isel(time=0)

        # landfraction
        if var_to_diag in files_lf.keys():
            _plf = files_lf[var_to_diag][0]  # path to the land fraction file
            _vlf = files_lf[var_to_diag][1]  # var name of land fraction
            _flf = files_lf[var_to_diag][2]  # conversion factor to fraction
            _ds_lf = xr.open_dataset(_plf)
            lf = _ds_lf[_vlf].values * _flf
        else:
            lf = np.ones_like(area)
            _vlf = 'lf'
            date_start = pd.to_datetime(str(ds_det_sin.time.values[0])).strftime('%Y-%m')
            date_end = pd.to_datetime(str(ds_det_sin.time.values[-1])).strftime('%Y-%m')
            x_years = pd.date_range(date_start, date_end, freq='1MS')
            _ds_lf = xr.Dataset({_vlf: (['time', 'lat', 'lon'],  lf)},
                                 coords={'time': (['time'], x_years),
                                         'lat': (['lat'], np.arange(89.5, -89.5-1.0, -1)),
                                         'lon': (['lon'], np.arange(-179.5, 179.5+1.0, 1))})
        lf_msk = np.where(common_nonnan, lf, np.nan)

        _vlf = files_lf[var_to_diag][1] if var_to_diag in files_lf.keys() else 'landfraction'
        _ds_lf = _ds_lf.expand_dims(dim={'time': 1}) if 'time' not in list(_ds_lf.dims) else _ds_lf
        lf_region = apply_koeppengeiger5_mask(
            dsin=_ds_lf[[_vlf]],
            pathlf=files_lf[var_to_diag][0] if var_to_diag in files_lf.keys() else 'ones',
            varlf=files_lf[var_to_diag][1] if var_to_diag in files_lf.keys() else 'landfraction',
            faclf=files_lf[var_to_diag][2] if var_to_diag in files_lf.keys() else '1.0',
            path_rm='/Net/Groups/BGI/people/hlee/data/koeppengeiger/kg5_regions_oneDeg.nc',
            func_aggr='mean',
            tosave=False,
            toplot=False
        )[_vlf].isel(time=0)

        #%% calculate the global time series
        _ar_wtm_temp = ds_det_sin_region[var_to_diag+'_det'].values.T
        ar_det_sin_glo = calc_land_area_weighted_mean(
            arr_data=_ar_wtm_temp,
            arr_area=area_region.common_nonnan_pixels.values,
            arr_lf=lf_region[_vlf].values
        )

        _ar_wtm_temp = ds_msc_sin_region[var_to_diag+'_msc'].values.T
        ar_msc_sin_glo = calc_land_area_weighted_mean(
            arr_data=_ar_wtm_temp,
            arr_area=area_region.common_nonnan_pixels.values,
            arr_lf=lf_region[_vlf].values
        )

        #%% apply the spatial attribution method
        ar_in_det = (ds_det_sin_region[var_to_diag+'_det'].values.T * lf_region[_vlf].values * area_region.common_nonnan_pixels.values) / np.nansum(lf_region[_vlf].values * area_region.common_nonnan_pixels.values)
        ar_fj_det = calc_contrib_ahlstrom2015(ar_x=ar_in_det, ar_X=ar_det_sin_glo)

        ar_in_msc = (ds_msc_sin_region[var_to_diag+'_msc'].values.T * lf_region[_vlf].values * area_region.common_nonnan_pixels.values) / np.nansum(lf_region[_vlf].values * area_region.common_nonnan_pixels.values)
        ar_fj_msc = calc_contrib_ahlstrom2015(ar_x=ar_in_msc, ar_X=ar_msc_sin_glo)

        np.savez(
            os.path.join(os.path.dirname(path_expOutput), f'koeppengeiger5_region_contrib_ahlstrom_{var_to_diag}.npz'),
            ar_contrib_msc_sin=ar_fj_msc,
            ar_contrib_det_sin=ar_fj_det
        )

        for d in [v for v in dir() if v[:2]=='ds']:
            print(f'd={d}')
            exec(d+'.close()')

if __name__ == '__main__':
    calc_vegpp_contrib_ahlstrom2015_koeppengeiger5(path_expOutput=argv[1])