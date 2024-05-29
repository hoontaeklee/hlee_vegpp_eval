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
from calc_cov_matrix_contributions import calc_cov_matrix_contributions
import xarray as xr
from running_mean import running_mean

def process_trendy_members_cov_matrix_by_regions_koeppengeiger5():
    '''
    '''

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
        path_trd_var = os.path.join(path_trddet, f'trendyv9_S2_{vname}-regionalMSCIAV_{vperiod}_koeppengeiger5_withoutDeforestedGrids.nc')

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

        ar_med_msc = np.nanmedian(ar_mem_msc, axis=0)
        ar_med_det = np.nanmedian(ar_mem_det, axis=0)

        dict_cov_msc_med = calc_cov_matrix_contributions(ar=ar_med_msc)
        dict_cov_det_med = calc_cov_matrix_contributions(ar=ar_med_det)

        np.savez(
                os.path.join(path_trddet, 'cov_norm', f'koeppengeiger5_region_cov_norm_{vname}_{vperiod}_EnsMedian_withoutDeforestedGrids.npz'),
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
        nmembers = 16
        nregions = 5
        ar_cov_norm_msc = np.ones((nmembers, nregions)) * np.nan
        ar_cov_norm_var_msc = np.ones((nmembers, nregions)) * np.nan
        ar_cov_norm_cov_msc = np.ones((nmembers, nregions)) * np.nan
        ar_sum_cov_minus_covt_msc = np.ones((nmembers, nregions)) * np.nan
        ar_cov_norm_det = np.ones((nmembers, nregions)) * np.nan
        ar_cov_norm_var_det = np.ones((nmembers, nregions)) * np.nan
        ar_cov_norm_cov_det = np.ones((nmembers, nregions)) * np.nan
        ar_sum_cov_minus_covt_det = np.ones((nmembers, nregions)) * np.nan

        for m in range(nmembers):

            # apply the covariance matrix method
            dict_cov_msc = calc_cov_matrix_contributions(ar=ar_mem_msc[m])
            dict_cov_det = calc_cov_matrix_contributions(ar=ar_mem_det[m])

            ar_cov_norm_msc[m] = dict_cov_msc['cov_norm']
            ar_cov_norm_var_msc[m] = dict_cov_msc['cov_norm_var']
            ar_cov_norm_cov_msc[m] = dict_cov_msc['cov_norm_cov']
            ar_sum_cov_minus_covt_msc[m] = dict_cov_msc['sum_cov_minus_covt']

            ar_cov_norm_det[m] = dict_cov_det['cov_norm']
            ar_cov_norm_var_det[m] = dict_cov_det['cov_norm_var']
            ar_cov_norm_cov_det[m] = dict_cov_det['cov_norm_cov']
            ar_sum_cov_minus_covt_det[m] = dict_cov_det['sum_cov_minus_covt']

            # # calc. trdw groups
            # ar_cov_norm_msc_mem = dict_cov_msc['cov_norm'].data
            # ar_cov_norm_det_mem = dict_cov_det['cov_norm'].data
            
            # ar_rsum_trdw_msc_mem[m, 0] = np.where(np.sum(np.isnan(ar_cov_norm_msc_mem[[9]]), axis=0)==1, np.nan, np.nansum(ar_cov_norm_msc_mem[[9]], axis=0))
            # ar_rsum_trdw_msc_mem[m, 1] = np.where(np.sum(np.isnan(ar_cov_norm_msc_mem[[4, 5, 8, 10]]), axis=0)==4, np.nan, np.nansum(ar_cov_norm_msc_mem[[4, 5, 8, 10]], axis=0))
            # ar_rsum_trdw_msc_mem[m, 2] = np.where(np.sum(np.isnan(ar_cov_norm_msc_mem[[0, 2, 3, 6]]), axis=0)==4, np.nan, np.nansum(ar_cov_norm_msc_mem[[0, 2, 3, 6]], axis=0))
            # ar_rsum_trdw_msc_mem[m, 3] = np.where(np.sum(np.isnan(ar_cov_norm_msc_mem[[1, 7]]), axis=0)==2, np.nan, np.nansum(ar_cov_norm_msc_mem[[1, 7]], axis=0))

            # ar_rsum_trdw_det_mem[m, 0] = np.where(np.sum(np.isnan(ar_cov_norm_det_mem[[9]]), axis=0)==1, np.nan, np.nansum(ar_cov_norm_det_mem[[9]], axis=0))
            # ar_rsum_trdw_det_mem[m, 1] = np.where(np.sum(np.isnan(ar_cov_norm_det_mem[[4, 5, 8, 10]]), axis=0)==4, np.nan, np.nansum(ar_cov_norm_det_mem[[4, 5, 8, 10]], axis=0))
            # ar_rsum_trdw_det_mem[m, 2] = np.where(np.sum(np.isnan(ar_cov_norm_det_mem[[0, 2, 3, 6]]), axis=0)==4, np.nan, np.nansum(ar_cov_norm_det_mem[[0, 2, 3, 6]], axis=0))
            # ar_rsum_trdw_det_mem[m, 3] = np.where(np.sum(np.isnan(ar_cov_norm_det_mem[[1, 7]]), axis=0)==2, np.nan, np.nansum(ar_cov_norm_det_mem[[1, 7]], axis=0))

        np.savez(
            os.path.join(path_trddet, 'cov_norm', f'koeppengeiger5_region_cov_norm_{vname}_{vperiod}_members_withoutDeforestedGrids.npz'),
            ar_cov_msc=ar_cov_norm_msc,
            ar_cov_det=ar_cov_norm_det,
            ar_cov_var_msc=ar_cov_norm_var_msc,
            ar_cov_var_det=ar_cov_norm_var_det,
            ar_cov_cov_msc=ar_cov_norm_cov_msc,
            ar_cov_cov_det=ar_cov_norm_cov_det,
            sum_cov_minus_covt_msc=ar_sum_cov_minus_covt_msc,
            sum_cov_minus_covt_det=ar_sum_cov_minus_covt_det
        )

        # np.savez(
        #     os.path.join(path_trddet, 'cov_norm', f'trdw_region_fromTranscom_cov_norm_{vname}_{vperiod}_members.npz'),
        #     ar_cov_msc=ar_rsum_trdw_msc_mem,
        #     ar_cov_det=ar_rsum_trdw_det_mem
        # )

if __name__ == '__main__':
    process_trendy_members_cov_matrix_by_regions_koeppengeiger5()