'''
- calculate regional contributions to the global variance in co2 fluxes (NEE, RECO, and GPP)
- by the bioclimatic regions by Papagiannopoulus et al. (2018)

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
from apply_koeppengeiger5_mask import apply_koeppengeiger5_mask
from calc_land_area_weighted_mean import calc_land_area_weighted_mean
import xarray as xr
import matplotlib.pyplot as plt
from running_mean import running_mean

def process_trendy_members_contrib_ahlstrom_by_regions_koeppengeiger5():
    '''
    '''

    path_lf = os.path.join(path_bgi, 'people/hlee/data/FLUXCOM/landfraction.360_180.nc')
    ds_lf = xr.open_dataset(path_lf)
    vname_lf = 'landfraction'
    fac_lf = 1.0

    # load studyarea mask    
    path_common_nonnan = os.path.join(path_bgi, 'people/hlee/to_others/sindbad_h2m_for_2001_2019/common_nonnan_pixels.nc')
    ds_common_nonnan = xr.open_dataset(path_common_nonnan)
    ds_common_nonnan = ds_common_nonnan.sortby('lat', ascending=False)
    ds_common_nonnan = ds_common_nonnan.sortby('lon', ascending=True)

    area_region = apply_koeppengeiger5_mask(
        dsin=ds_common_nonnan.expand_dims(dim={'time': 1}),
        pathlf='ones',
        varlf=vname_lf,
        faclf=1.0,
        path_rm='/Net/Groups/BGI/people/hlee/data/koeppengeiger/kg5_regions_oneDeg.nc',
        func_aggr='sum',
        tosave=False,
        toplot=False
    )['common_nonnan_pixels'].isel(time=0)

    ds_lf = ds_lf.expand_dims(dim={'time': 1}) if 'time' not in list(ds_lf.dims) else ds_lf
    lf_region = apply_koeppengeiger5_mask(
        dsin=ds_lf[[vname_lf]],
        pathlf='ones',
        varlf='landfraction',
        faclf=1.0,
        path_rm='/Net/Groups/BGI/people/hlee/data/koeppengeiger/kg5_regions_oneDeg.nc',
        func_aggr='mean',
        tosave=False,
        toplot=False
    )[vname_lf].isel(time=0)

    # # load region mask
    # path_rm = os.path.join(path_bgi, 'data/DataStructureMDI/DATA/grid/Global/1d00_static/eco_hydro_regions/dm01/Data/class.bioclimatic.nc')
    # ds_rm = xr.open_dataset(path_rm)
    # ds_rm = ds_rm.rename({
    #     'latitude': 'lat',
    #     'longitude': 'lon',
    #     'class': 'region'
    #     })
    # rcnt = 11
    # rnames = ds_rm.region.units.split(',')
    # rnames_short = ['BrWT', 'BrW', 'BrT', 'BrE', 'TsW', 'StW', 'MdT', 'MdW', 'StE', 'Tp', 'TsE']
    # colrm = ['rebeccapurple', 'mediumpurple', 'violet', 'indigo', 'palegreen', 'orange', 'green', 'sienna', 'darkgreen', 'royalblue', 'yellowgreen']
    # ar_r = ds_rm.region.values

    # load ensemble median msc and det
    path_trddet = os.path.join(path_bgi, 'people/hlee/data/trendy/v9/regridded_1deg')

    dict_var = {
        'nee': {
            'period': '2015-2019'
        },
        'gpp': {
            'period': '2002-2015'
        },
        'reco': {
            'period': '2002-2015'
        }
    }
    
    for v in range(len(list(dict_var.keys()))):
        
        vname = list(dict_var.keys())[v]
        vperiod = dict_var[vname]['period']
        path_trd_var = os.path.join(path_trddet, f'trendyv9_S2_{vname}-regionalMSCIAV_{vperiod}_koeppengeiger5.nc')

        print(f'processing {vname}, {v+1} / {len(list(dict_var.keys()))}', flush=True)

        #%% calc for the ensemble median
        ds_trd = xr.open_dataset(path_trd_var)

        # remove lpj-guess (model index 8) as it doesn't have rh (v7)
        # remove jules-es-1p0 (model index 6), lpj-guess (model index 7), ocn (model index 9) (v9)
        idx_model2exclude = [6, 7, 9]
        ar_temp = ds_trd[f'{vname}_msc'].data
        ar_temp[idx_model2exclude, :, :] = np.nan
        ds_trd[f'{vname}_msc'].data = ar_temp

        ar_temp = ds_trd[f'{vname}_det'].data
        ar_temp[idx_model2exclude, :, :] = np.nan
        ds_trd[f'{vname}_det'].data = ar_temp

        # do running mean for det values
        ds_trd[f'{vname}_det'].data = np.apply_along_axis(running_mean, 2, ds_trd[f'{vname}_det'], N=12)

        # apply the covariance matrix method
        ar_mem_msc = ds_trd[f'{vname}_msc'].values
        ar_mem_det = ds_trd[f'{vname}_det'].values

        ar_mem_msc = (ar_mem_msc.reshape(-1, ar_mem_msc.shape[-1]).T - np.nanmean(ar_mem_msc, axis=2).reshape(-1)).T.reshape(ar_mem_msc.shape)

        ar_med_msc = np.nanmedian(ar_mem_msc, axis=0)
        ar_med_det = np.nanmedian(ar_mem_det, axis=0)

        _ar_wtm_temp = ar_med_msc
        ar_med_msc_glo = calc_land_area_weighted_mean(
            arr_data=_ar_wtm_temp.T,
            arr_area=area_region.common_nonnan_pixels.values,
            arr_lf=lf_region[vname_lf].values
        )

        _ar_wtm_temp = ar_med_det
        ar_med_det_glo = calc_land_area_weighted_mean(
            arr_data=_ar_wtm_temp.T,
            arr_area=area_region.common_nonnan_pixels.values,
            arr_lf=lf_region[vname_lf].values
        )

        _ar_x = (ar_med_msc.T * area_region.common_nonnan_pixels.values * lf_region[vname_lf].values)/np.nansum(area_region.common_nonnan_pixels.values * lf_region[vname_lf].values)
        ar_contrib_msc_med = calc_contrib_ahlstrom2015(ar_x=_ar_x, ar_X=ar_med_msc_glo)
        _ar_x = (ar_med_det.T * area_region.common_nonnan_pixels.values * lf_region[vname_lf].values)/np.nansum(area_region.common_nonnan_pixels.values * lf_region[vname_lf].values)
        ar_contrib_det_med = calc_contrib_ahlstrom2015(ar_x=_ar_x, ar_X=ar_med_det_glo)

        np.savez(
                os.path.join(path_trddet, 'contrib_ahlstrom', f'koeppengeiger5_region_contrib_ahlstrom_{vname}_{vperiod}_EnsMedian.npz'),
                ar_contrib_msc=ar_contrib_msc_med,
                ar_contrib_det=ar_contrib_det_med
            )

        #%% calc for each ensemble member
        nmembers = 16
        nregions = 5
        ar_contrib_ahlstrom_mem_msc = np.ones((nmembers, nregions)) * np.nan
        ar_contrib_ahlstrom_mem_det = np.ones((nmembers, nregions)) * np.nan
        
        for m in range(nmembers):

            # apply the covariance matrix method

            _ar_wtm_temp = ar_mem_msc[m]
            ar_mem_msc_glo = calc_land_area_weighted_mean(
                arr_data=_ar_wtm_temp.T,
                arr_area=area_region.common_nonnan_pixels.values,
                arr_lf=lf_region[vname_lf].values
            )

            _ar_wtm_temp = ar_mem_det[m]
            ar_mem_det_glo = calc_land_area_weighted_mean(
                arr_data=_ar_wtm_temp.T,
                arr_area=area_region.common_nonnan_pixels.values,
                arr_lf=lf_region[vname_lf].values
            )

            _ar_x = (ar_mem_msc[m].T * area_region.common_nonnan_pixels.values * lf_region[vname_lf].values)/np.nansum(area_region.common_nonnan_pixels.values * lf_region[vname_lf].values)
            ar_contrib_msc_med = calc_contrib_ahlstrom2015(ar_x=_ar_x, ar_X=ar_mem_msc_glo)
            _ar_x = (ar_mem_det[m].T * area_region.common_nonnan_pixels.values * lf_region[vname_lf].values)/np.nansum(area_region.common_nonnan_pixels.values * lf_region[vname_lf].values)
            ar_contrib_det_med = calc_contrib_ahlstrom2015(ar_x=_ar_x, ar_X=ar_mem_det_glo)

            ar_contrib_ahlstrom_mem_msc[m] = ar_contrib_msc_med
            ar_contrib_ahlstrom_mem_det[m] = ar_contrib_det_med

        np.savez(
            os.path.join(path_trddet, 'contrib_ahlstrom', f'koeppengeiger5_region_contrib_ahlstrom_{vname}_{vperiod}_members.npz'),
            ar_contrib_msc=ar_contrib_ahlstrom_mem_msc,
            ar_contrib_det=ar_contrib_ahlstrom_mem_det
        )

if __name__ == '__main__':
    process_trendy_members_contrib_ahlstrom_by_regions_koeppengeiger5()