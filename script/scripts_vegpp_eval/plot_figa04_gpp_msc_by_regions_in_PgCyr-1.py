'''
figure a04: gpp seasonal variability across regions
'''

#%% libraries and functions
import sys
import os
path_bgi = '/Net/Groups/BGI'
if os.path.join(path_bgi, 'people/hlee/scripts/utils') not in sys.path:
    sys.path.insert(1, os.path.join(path_bgi, 'people/hlee/scripts/utils'))  # https://stackoverflow.com/a/4383597/7578494
if os.path.join(path_bgi, 'people/hlee/scripts/diagnose_tws_nee') not in sys.path:
    sys.path.insert(1, os.path.join(path_bgi, 'people/hlee/scripts/diagnose_tws_nee'))  # https://stackoverflow.com/a/4383597/7578494
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from apply_transcom_mask import apply_transcom_mask
from sys import argv
from calc_land_area_weighted_mean import calc_land_area_weighted_mean
from calc_metrics import calc_rsq, calc_rae, calc_mad

def find_mode(window, axis=None, **kwargs):
    # find the mode over all axes 
    uniq = np.unique(window, return_counts=True)
    
    ret = uniq[0][np.argmax(uniq[1])]
    ret = np.atleast_2d(ret)

    return ret

def reduce_chunks(chunks, reduce_func):
    chunks_tr = np.vstack(chunks.transpose(0, 2, 1, 3))
    ret = np.array([reduce_func(x) for x in chunks_tr]).reshape(chunks.shape[0], chunks.shape[2])

    return ret

def plot_figa04_gpp_msc_by_regions_in_PgCperyr(path_expOutput):
    
    _tsc = 'msc'  # 'det' or 'msc'; time scale

    isdet = path_expOutput.split('/')[-1]=='detrended'

    #%% load data
    ds_tc = xr.open_dataset(os.path.join(path_bgi, 'people/hlee/data/transcom/TranscomRegions.360.180.nc'))
    rnames_tc = [x.decode('UTF-8') for x in ds_tc.Legend.values[0, :]]
    # rnames_tc.insert(0, 'Excluded')
    rnames_tc = rnames_tc[:11]  # exclude oceans
    dict_tc = dict(zip(np.arange(1, len(rnames_tc)+1), rnames_tc))

    # path_expOutput = os.path.join(path_bgi, 'people/hlee/sindbad/data/output/VEGPP2pool1519_studyArea_10k_RD4wtnanmean_1_20230712/detrended')

    # load study area common nonnan mask of forcing
    path_common_nonnan = os.path.join(path_bgi, 'people/hlee/to_others/sindbad_h2m_for_2001_2019/common_nonnan_pixels.nc')
    ds_common_nonnan = xr.open_dataset(path_common_nonnan)
    ds_common_nonnan = ds_common_nonnan.sortby('lat', ascending=False)
    ds_common_nonnan = ds_common_nonnan.sortby('lon', ascending=True)
    common_nonnan = ds_common_nonnan.common_nonnan_pixels.values

    da_common_nonnan_chunks = ds_common_nonnan.common_nonnan_pixels.coarsen(lat=4, lon=5).construct(
        lat=("lat_coarse", "lat_window"),
        lon=("lon_coarse", "lon_window")
    )
    ngrid_n = 1  # ngrids to remove from the north pole
    ngrid_s = 9  #  ngrids to remove from the south pole
    ar_common_nonnan_4by5 = reduce_chunks(da_common_nonnan_chunks.values, reduce_func=find_mode)
    ar_common_nonnan_4by5 = ar_common_nonnan_4by5[ngrid_n:(-ngrid_s)]
    
    if isdet:
        exp_name = '_'.join(path_expOutput.split('/')[-2].split('_')[:-1])
    else:
        exp_name = '_'.join(path_expOutput.split('/')[-1].split('_')[:-1])

    # out-path for figures
    path_out = os.path.join(path_bgi, 'people/hlee/hlee_vegpp_eval/figures')
    if os.path.exists(path_out) == False:
        os.mkdir(path_out)

    # read nc files of variables
    files = os.listdir(os.path.join(path_expOutput))
    files_var = [f for f in files if f.endswith('.nc')]
    files_var = [f for f in files_var if not f.__contains__('esv-iav')]

    if isdet:
        list_var = [v.replace(path_expOutput.split('/')[-2]+'_', '').replace('.nc', '') for v in files_var]
    else:
        list_var = ['_'.join(v.replace(exp_name+'_', '').split('_')[:-2]) for v in files_var]

    list_target_var = [
        'gpp'
    ]
    
    ##
    # load fluxcom members for the uncertainty
    path_flx = os.path.join(path_bgi, 'people/hlee/data/FLUXCOM/members_detrended/region_transcom')
    ds_flx_reg = xr.open_dataset(os.path.join(path_flx, 'GPP_RS_METEO-NONE_TranscomRegions_2002-2015_studyArea_members.nc'))
    ar_flx_reg_mem = ds_flx_reg[f'GPP_{_tsc}'].isel(time=range(12)).data
    #

    ##
    # load trendy
    path_trendy = os.path.join(path_bgi, 'people/hlee/data/trendy/v9/regridded_1deg')
    ds_trd_reg = xr.open_dataset(os.path.join(path_trendy, 'trendyv9_S2_gpp-regionalMSCIAV_2002-2015_transcom.nc'))
    ds_trd_reg = ds_trd_reg.isel(time=range(12))

    # remove lpj-guess (model index 8) as it doesn't have rh (v7)
    # remove jules-es-1p0 (model index 6), lpj-guess (model index 7), ocn (model index 9) (v9)
    idx_model2exclude = [6, 7, 9]
    ar_temp = ds_trd_reg[f'gpp_{_tsc}'].data
    ar_temp[idx_model2exclude, :, :] = np.nan
    ds_trd_reg[f'gpp_{_tsc}'].data = ar_temp
    
    ##

    list_var = [ele for ele in list_var if ele in list_target_var]

    # keep files only for target variables
    if isdet:
        files_var = [ele for ele in files_var if ele.replace(path_expOutput.split('/')[-2]+'_', '').replace('.nc', '') in list_var]
    else:
        files_var = [ele for ele in files_var if '_'.join(ele.replace(exp_name+'_', '').split('_')[:-2]) in list_var]

    list_var.sort()
    files_var.sort()
    
    # grid area
    area = np.load(os.path.join(path_bgi, 'people/hlee/data/gridAreaAndLandFraction/gridAreaInKm2_180_360.npz'))['area']
    area_msk = np.where(common_nonnan, area, np.nan)
    area_tc = apply_transcom_mask(
        dsin=ds_common_nonnan.expand_dims(dim={'time': 1}),
        pathlf='ones',
        varlf='landfraction',
        faclf='1.0',
        path_tc=os.path.join(path_bgi, 'people/hlee/data/transcom/TranscomRegions.360.180.nc'),
        func_aggr='sum',
        tosave=False,
        toplot=False
    )['common_nonnan_pixels'].isel(time=0)

    # landfraction
    # path to corresponding landfraction data for each variable
    # [path, variable name, factor_to_fraction]
    files_lf = {
        'gpp': [os.path.join(path_bgi, 'people/hlee/data/FLUXCOM/landfraction.360_180.nc'), 'landfraction', 1.0],
        'NEE20152019': [os.path.join(path_bgi, 'people/hlee/data/oco2mipv10/area.ESACCI.360.180.nc'), 'fraction', 0.01]
    }

    # set plotting variables 
    if _tsc=='msc':
        x_years = np.arange(1, 13)
    arr_rsq_sin = np.arange(len(list_var)*12).reshape(len(list_var), 12) * np.nan
    arr_rae_sin = np.arange(len(list_var)*12).reshape(len(list_var), 12) * np.nan
    arr_rsq_trd = np.arange(len(list_var)*12).reshape(len(list_var), 12) * np.nan
    arr_rae_trd = np.arange(len(list_var)*12).reshape(len(list_var), 12) * np.nan
    
    ylim_rng = [-10, 0.9]
    fsize_legend = 18
    fsize_suplab = 18
    fsize_text = 10
    line_alpha = 0.8
    area_alpha = 0.1
    alphabets = [chr(i) for i in range(ord('a'),ord('z')+1)]
    alphabets_coords = (-0.09, 1.07)
    dict_colors = {  # data_name: [col_var1, col_var2, ...]
        'obs': ['black', 'black'],
        'sin': ['#BBAF52', '#BBAF52'],
        'trd': ['#3C7EA2', '#3C7EA2']
    }
    dict_linestyle = {  # data_name: [ls_var1, ls_var2, ...]
        'obs': ['solid', 'dashed'],
        'sin': ['solid', 'dashed'],
        'trd': ['solid', 'dashed']
    }
    legend_handles = []
    nrow = 3
    ncol = 4
    fig, axes = plt.subplots(
        nrow, ncol,
        figsize=(16*ncol*0.2, 9*nrow*0.2),
        gridspec_kw={
            'height_ratios': [1, 1, 1]
        }
    )
    axes = axes.flatten()

    for i in range(len(list_var)):

        # get variable name
        _vname = list_var[i]
        _vname_in_sin = _vname if not isdet else f'{_vname[:3]}_{_tsc}'

        print(f'processing {_vname}, {i+1} / {len(list_var)}')

        # load simulated variable
        if isdet:
            _nc_sin = xr.open_dataset(os.path.join(path_expOutput, files_var[i]))
        else:
            _nc_sin = xr.open_dataset(os.path.join(
                path_expOutput, files_var[i])).resample(time='1M').reduce(np.nanmean)
            _nc_sin = _nc_sin.sortby('lat', ascending=False)
        _nc_sin = _nc_sin[[_vname_in_sin]]
        _nc_sin = _nc_sin.sortby('lat', ascending=False)
        _nc_sin = _nc_sin.sortby('lon', ascending=True)
        if _tsc=='msc':
            _nc_sin = _nc_sin.isel(time=range(12))
        
        # set landfraction
        if _vname in files_lf.keys():
            _plf = files_lf[_vname][0]  # path to the land fraction file
            _vlf = files_lf[_vname][1]  # var name of land fraction
            _flf = files_lf[_vname][2]  # conversion factor to fraction
            _ds_lf = xr.open_dataset(_plf)
            lf = _ds_lf[_vlf].values * _flf
        else:
            lf = np.ones_like(area)
            _vlf = 'lf'
            _ds_lf = xr.Dataset({_vlf: (['time', 'lat', 'lon'],  lf)},
                                 coords={'time': (['time'], x_years),
                                         'lat': (['lat'], np.arange(89.5, -89.5-1.0, -1)),
                                         'lon': (['lon'], np.arange(-179.5, 179.5+1.0, 1))})
        lf_msk = np.where(common_nonnan, lf, np.nan)
        
        _ds_lf = _ds_lf.expand_dims(dim={'time': 1}) if 'time' not in list(_ds_lf.dims) else _ds_lf
        lf_tc = apply_transcom_mask(
            dsin=_ds_lf[[_vlf]],
            pathlf='ones',
            varlf='landfraction',
            faclf='1.0',
            path_tc=os.path.join(path_bgi, 'people/hlee/data/transcom/TranscomRegions.360.180.nc'),
            func_aggr='mean',
            tosave=False,
            toplot=False
        )[_vlf].isel(time=0)

        _nc_sin_tc = apply_transcom_mask(
            dsin=_nc_sin,
            pathlf=files_lf[_vname][0] if _vname in files_lf.keys() else 'ones',
            varlf=files_lf[_vname][1] if _vname in files_lf.keys() else 'landfraction',
            faclf=files_lf[_vname][2] if _vname in files_lf.keys() else '1.0',
            path_tc=os.path.join(path_bgi, 'people/hlee/data/transcom/TranscomRegions.360.180.nc'),
            p_truncate=1.0,
            tosave=False,
            toplot=False)
        _nc_sin_tc = _nc_sin_tc[_vname_in_sin]

        # calculate land-area-weighted global mean
        _ar_wtm_temp = _nc_sin_tc[_vname_in_sin].values.T
        _sin = calc_land_area_weighted_mean(
            arr_data=_ar_wtm_temp,
            arr_area=area_tc.common_nonnan_pixels.values,
            arr_lf=lf_tc[_vlf].values
        )

        ar_flx_glo = np.ones((18, 12)) * np.nan  # (model, time)
        for m in range(18):
            ar_flx_glo[m, :] = calc_land_area_weighted_mean(
                arr_data=ar_flx_reg_mem[m].T,
                arr_area=area_tc.common_nonnan_pixels.values,
                arr_lf=lf_tc[_vlf].values
        )

        ar_trd_glo = np.ones((16, 12)) * np.nan  # (model, time)
        for m in range(16):
            ar_trd_glo[m, :] = calc_land_area_weighted_mean(
                arr_data=ds_trd_reg[f'gpp_{_tsc}'].values[m].T,
                arr_area=area_tc.common_nonnan_pixels.values,
                arr_lf=lf_tc[_vlf].values
        )

        # calc. median and robust std.
        ar_flx_glo_val = np.nanmedian(ar_flx_glo, axis=0)
        ar_flx_glo_unc = calc_mad(ar_flx_glo, axis=0) * 1.25 # mad to std (*1.25)

        ar_flx_reg_val = np.nanmedian(ar_flx_reg_mem, axis=0)
        ar_flx_reg_unc = calc_mad(ar_flx_reg_mem, axis=0) * 1.25 # mad to std (*1.25)
        
        ar_trd_glo_val = np.nanmedian(ar_trd_glo, axis=0) * 1000 * 86400  # kgC m-2 s-1 to gC m-2 d-1
        ar_trd_glo_unc = calc_mad(ar_trd_glo, axis=0) * 1000 * 86400 * 1.25 # kgC m-2 s-1 to gC m-2 d-1; mad to std (*1.25)
        
        ar_trd_reg_val = np.nanmedian(ds_trd_reg[f'gpp_{_tsc}'].values, axis=0) * 1000 * 86400  # kgC m-2 s-1 to gC m-2 d-1
        ar_trd_reg_unc = calc_mad(ds_trd_reg[f'gpp_{_tsc}'].values, axis=0) * 1000 * 86400 * 1.25 # kgC m-2 s-1 to gC m-2 d-1; mad to std (*1.25)

        # convert unit from gCm-2d-1 to PgCyr-1
        # ndaysperyear = np.array([pd.Timestamp(e, 12, 31).dayofyear for e in np.arange(2002, 2015+1)])
        _sin = _sin * np.nansum(area_msk) * 1000 * 1000 * 365 * 1e-15
        _nc_sin_tc[_vname_in_sin] = _nc_sin_tc[_vname_in_sin] * area_tc['common_nonnan_pixels'] * 1000 * 1000 * 365 * 1e-15
        ar_flx_glo_val = ar_flx_glo_val * np.nansum(area_msk) * 1000 * 1000 * 365 * 1e-15
        ar_flx_glo_unc = ar_flx_glo_unc * np.nansum(area_msk) * 1000 * 1000 * 365 * 1e-15
        ar_flx_reg_val = (ar_flx_reg_val.T * area_tc['common_nonnan_pixels'].data * 1000 * 1000 * 365 * 1e-15).T
        ar_flx_reg_unc = (ar_flx_reg_unc.T * area_tc['common_nonnan_pixels'].data * 1000 * 1000 * 365 * 1e-15).T
        ar_trd_glo_val = ar_trd_glo_val * np.nansum(area_msk) * 1000 * 1000 * 365 * 1e-15
        ar_trd_glo_unc = ar_trd_glo_unc * np.nansum(area_msk) * 1000 * 1000 * 365 * 1e-15
        ar_trd_reg_val = (ar_trd_reg_val.T * area_tc['common_nonnan_pixels'].data * 1000 * 1000 * 365 * 1e-15).T
        ar_trd_reg_unc = (ar_trd_reg_unc.T * area_tc['common_nonnan_pixels'].data * 1000 * 1000 * 365 * 1e-15).T

        # set y bounds
        sc_yrng = 1.3
        ylim_rng[0] = np.nanmin(
            [ylim_rng[0],
            np.nanmin(_nc_sin_tc[_vname_in_sin]) * sc_yrng,
            np.nanmin(ar_flx_reg_val) * sc_yrng,
            np.nanmin(ar_trd_reg_val) * sc_yrng
            ]
            ).round(1)
        ylim_rng[1] = np.nanmax(
            [ylim_rng[1],
            np.nanmax(_nc_sin_tc[_vname_in_sin]) * sc_yrng,
            np.nanmax(ar_flx_reg_val) * sc_yrng,
            np.nanmax(ar_trd_reg_val) * sc_yrng
            ]
            ).round(1)

        # globe
        r=0
        _obs = ar_flx_glo_val
        _obs_unc = ar_flx_glo_unc
        _trd = ar_trd_glo_val
        _trd_unc = ar_trd_glo_unc

        arr_rsq_sin[i, r] = calc_rsq(d1=_obs, d2=_sin).round(2)
        arr_rae_sin[i, r] = calc_rae(obs=_obs, est=_sin).round(2)
        # arr_rsq_car[i, r] = calc_rsq(d1=_obs, d2=_car).round(2)
        # arr_rae_car[i, r] = calc_rae(obs=_obs, est=_car).round(2)
        arr_rsq_trd[i, r] = calc_rsq(d1=_obs, d2=_trd).round(2)
        arr_rae_trd[i, r] = calc_rae(obs=_obs, est=_trd).round(2)

        ax = axes[r]

        if i==0:
            ax.annotate(f'({alphabets[r]})', xy=alphabets_coords, xycoords='axes fraction', fontsize=13, weight='bold')
        
        l1, = ax.plot(x_years, _obs, linestyle=dict_linestyle['obs'][i], color=dict_colors['obs'][i], alpha=line_alpha, label='FLUXCOM')
        l2, = ax.plot(x_years, _sin, linestyle=dict_linestyle['sin'][i], color=dict_colors['sin'][i], alpha=line_alpha, label='SINDBAD')
        l4, = ax.plot(x_years, _trd, linestyle=dict_linestyle['trd'][i], color=dict_colors['trd'][i], alpha=line_alpha, label='TRENDY')
        ax.fill_between(x_years, _obs-_obs_unc, _obs+_obs_unc, linestyle=dict_linestyle['obs'][i], color=dict_colors['obs'][i], alpha=area_alpha)
        ax.fill_between(x_years, _trd-_trd_unc, _trd+_trd_unc, linestyle=dict_linestyle['trd'][i], color=dict_colors['trd'][i], alpha=area_alpha)
        ax.set_ylim(0, 200)
        ax.set_title('Globe')
        legend_handles.append(l1)
        legend_handles.append(l2)
        legend_handles.append(l4)
            
        # transcom regions
        for r in np.arange(1, 12):
            _sin = _nc_sin_tc.sel(region=r)[_vname_in_sin].values
            _obs = ar_flx_reg_val[(r-1)]  # _nc_obs_tc.sel(region=r)[_vname_in_obs].values
            _obs_unc = ar_flx_reg_unc[(r-1)]
            _trd = ar_trd_reg_val[(r-1)]
            _trd_unc = ar_trd_reg_unc[(r-1)]

            arr_rsq_sin[i, r] = calc_rsq(d1=_obs, d2=_sin).round(2)
            arr_rae_sin[i, r] = calc_rae(obs=_obs, est=_sin).round(2)
            arr_rsq_trd[i, r] = calc_rsq(d1=_obs, d2=_trd).round(2)
            arr_rae_trd[i, r] = calc_rae(obs=_obs, est=_trd).round(2)
                
            ax = axes[r]

            p1 = ax.plot(x_years, _obs, linestyle=dict_linestyle['obs'][i], color=dict_colors['obs'][i], alpha=line_alpha, label='FLUXCOM')
            p2 = ax.plot(x_years, _sin, linestyle=dict_linestyle['sin'][i], color=dict_colors['sin'][i], alpha=line_alpha, label='SINDBAD')
            p4 = ax.plot(x_years, _trd, linestyle=dict_linestyle['trd'][i], color=dict_colors['trd'][i], alpha=line_alpha, label='TRENDY')
            ax.fill_between(x_years, _obs-_obs_unc, _obs+_obs_unc, color=dict_colors['obs'][i], alpha=area_alpha)
            ax.fill_between(x_years, _trd-_trd_unc, _trd+_trd_unc, color=dict_colors['trd'][i], alpha=area_alpha)
            ax.set_ylim(ylim_rng[0], ylim_rng[1])

            ax.set_title(f'{r}: {dict_tc[r]}')
            ax.annotate(f'({alphabets[r]})', xy=alphabets_coords, xycoords='axes fraction', fontsize=13, weight='bold')

    fig.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98,
                        wspace=0.23, hspace=0.40)

    fig.text(0.5, -0.10, 'Months', ha='center', fontsize=fsize_suplab)
    fig.text(-0.04, 0.51, 'GPP (PgC yr-1)', va='center', rotation='vertical', fontsize=fsize_suplab)

    # add legends and alphabets
    axes[0].legend(
                handles=legend_handles,  # [l1, l2, l3, l4],
                loc='upper center', bbox_to_anchor=(0.5, 1.17), bbox_transform=fig.transFigure,
                fontsize=fsize_legend, fancybox=False, ncol=4, frameon=False
            )
    
    # iterate over subplots: add metrics, xticklabels, ...
    for i in range(len(list_var)):
        for r in range(12):
            if list_var[i]=='gpp':  # NEP
                axes[r].annotate(
                    f'R$^2$: ' + str(arr_rsq_sin[i, r]),
                    xy=(0.01, 0.15), xycoords='axes fraction',
                    fontsize=fsize_text, color=dict_colors['sin'][0]
                    )
                axes[r].annotate(
                    f'RAE: ' + str(arr_rae_sin[i, r]),
                    xy=(0.01, 0.05), xycoords='axes fraction',
                    fontsize=fsize_text, color=dict_colors['sin'][0]
                    )
                axes[r].annotate(
                    f'R$^2$: ' + str(arr_rsq_trd[i, r]),
                    xy=(0.73, 0.15), xycoords='axes fraction',
                    fontsize=fsize_text, color=dict_colors['trd'][0]
                    )
                axes[r].annotate(
                    f'RAE: ' + str(arr_rae_trd[i, r]),
                    xy=(0.73, 0.05), xycoords='axes fraction',
                    fontsize=fsize_text, color=dict_colors['trd'][0]
                    )

    for r in range(12):
        axes[r].set_xticks(np.arange(1, 13, 2))
        axes[r].set_xticklabels(np.arange(1, 13, 2), fontsize=11)

    fig.savefig(
        os.path.join(path_out, f"figa04_gpp_msc_by_regions_in_PgCyr-1.png"),
        dpi=600,
        transparent=False,
        bbox_inches='tight'
    )
    plt.clf()
    
    # sys.exit("done!")  # doesn't work... keep running somehow
    print('Done!', flush=True)
    os.system(f"pkill -f {os.path.basename(__file__)}")  # this works...

if __name__ == '__main__':
    plot_figa04_gpp_msc_by_regions_in_PgCperyr(path_expOutput=argv[1])
# %%
