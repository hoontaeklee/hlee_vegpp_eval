'''
compare to methods to calculate MSC

1) mean(a month across years)
2) mean(x_mn,i,raw - IAV_mn,i) = mean(x_mn,i,raw - fit(x_mn,i,raw))
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
from calc_metrics import calc_rsq, calc_rae

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

def plot_figa99_compare_gpp_msc_esv_transcom(path_expOutput):
    
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
    files_var = [f for f in files_var if f.__contains__('_esv-iav')]

    if isdet:
        list_var = [v.replace(path_expOutput.split('/')[-2]+'_', '').replace('_esv-iav.nc', '') for v in files_var]
    else:
        list_var = ['_'.join(v.replace(exp_name+'_', '').split('_')[:-2]) for v in files_var]

    list_target_var = [
        'gpp'
    ]

    list_var = [ele for ele in list_var if ele in list_target_var]

    # keep files only for target variables
    if isdet:
        files_var = [ele for ele in files_var if ele.replace(path_expOutput.split('/')[-2]+'_', '').replace('_esv-iav.nc', '') in list_var]
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
        
    ylim_rng = [-0.5, 0.9]
    fsize_legend = 18
    fsize_suplab = 18
    fsize_text = 10
    line_alpha = 0.8
    area_alpha = 0.1
    alphabets = [chr(i) for i in range(ord('a'),ord('z')+1)]
    alphabets_coords = (-0.09, 1.07)
    dict_colors = {  # data_name: [col_var1, col_var2, ...]
        'sin': ['firebrick', 'firebrick'],
        'sin_sim': ['black', 'black']
    }
    dict_linestyle = {  # data_name: [ls_var1, ls_var2, ...]
        'sin': ['solid', 'dashed'],
        'sin_sim': ['solid', 'dashed']
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

        # load msc with a simpler calculation...
        _nc_sin_sim = xr.open_dataset(os.path.join(path_expOutput, 'etc', 'VEGPP2pool1519_studyArea_10k_RD4wtnanmean_1_20230712_gpp_simpleMSC.nc'))
        _nc_sin_sim = _nc_sin_sim[['gpp_msc']]
        _nc_sin_sim = _nc_sin_sim.sortby('lat', ascending=False)
        _nc_sin_sim = _nc_sin_sim.sortby('lon', ascending=True)
        _nc_sin_sim = _nc_sin_sim.rename({'month': 'time'})
        _nc_sin_sim['time'] = _nc_sin['time']
        
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

        _nc_sin_sim_tc = apply_transcom_mask(
            dsin=_nc_sin_sim,
            pathlf=files_lf[_vname][0] if _vname in files_lf.keys() else 'ones',
            varlf=files_lf[_vname][1] if _vname in files_lf.keys() else 'landfraction',
            faclf=files_lf[_vname][2] if _vname in files_lf.keys() else '1.0',
            path_tc=os.path.join(path_bgi, 'people/hlee/data/transcom/TranscomRegions.360.180.nc'),
            p_truncate=1.0,
            tosave=False,
            toplot=False)
        _nc_sin_sim_tc = _nc_sin_sim_tc['gpp_msc']

        # calculate land-area-weighted global mean
        _ar_wtm_temp = _nc_sin_tc[_vname_in_sin].values.T
        _sin = calc_land_area_weighted_mean(
            arr_data=_ar_wtm_temp,
            arr_area=area_tc.common_nonnan_pixels.values,
            arr_lf=lf_tc[_vlf].values
        )

        # calculate land-area-weighted global mean
        _ar_wtm_temp = _nc_sin_sim_tc['gpp_msc'].values.T
        _sin_sim = calc_land_area_weighted_mean(
            arr_data=_ar_wtm_temp,
            arr_area=area_tc.common_nonnan_pixels.values,
            arr_lf=lf_tc[_vlf].values
        )

        # set y bounds
        sc_yrng = 1.05
        ylim_rng[0] = np.nanmin(
            [ylim_rng[0],
            np.nanmin(_nc_sin_tc[_vname_in_sin]) * sc_yrng,
            np.nanmin(_nc_sin_sim_tc['gpp_msc']) * sc_yrng,
            np.nanmin(_sin) * sc_yrng,
            np.nanmin(_sin_sim) * sc_yrng
            ]
            ).round(1)
        ylim_rng[1] = np.nanmax(
            [ylim_rng[1],
            np.nanmax(_nc_sin_tc[_vname_in_sin]) * sc_yrng,
            np.nanmax(_nc_sin_sim_tc['gpp_msc']) * sc_yrng,
            np.nanmax(_sin) * sc_yrng,
            np.nanmax(_sin_sim) * sc_yrng
            ]
            ).round(1)

        # globe
        r=0
        # if _vname not in ['NEE_mix']:
            # arr_rsq_sin[i, r] = calc_rsq(d1=_obs, d2=_sin).round(2)
            # arr_rae_sin[i, r] = calc_rae(obs=_obs, est=_sin).round(2)
        ax = axes[r]

        if i==0:
            ax.annotate(f'({alphabets[r]})', xy=alphabets_coords, xycoords='axes fraction', fontsize=13, weight='bold')
        
        if _vname=='gpp':
            l1, = ax.plot(x_years, _sin, linestyle=dict_linestyle['sin'][i], color=dict_colors['sin'][i], alpha=line_alpha, label='ESV')
            l2, = ax.plot(x_years, _sin_sim, linestyle=dict_linestyle['sin_sim'][i], color=dict_colors['sin_sim'][i], alpha=line_alpha, label='MSC')
            ax.set_ylim(1, 4.5)
            ax.set_title('Globe')
            legend_handles.append(l2)
            legend_handles.append(l1)
            
        # transcom regions
        for r in np.arange(1, 12):
            _sin = _nc_sin_tc.sel(region=r)[_vname_in_sin].values
            _sin_sim = _nc_sin_sim_tc.sel(region=r)['gpp_msc'].values

            # if _vname not in ['NEE_mix']:
                # arr_rsq_sin[i, r] = calc_rsq(d1=_obs, d2=_sin).round(2)
                # arr_rae_sin[i, r] = calc_rae(obs=_obs, est=_sin).round(2)

            ax = axes[r]

            # axis for NEP
            if _vname=='gpp':
                p1 = ax.plot(x_years, _sin, linestyle=dict_linestyle['sin'][i], color=dict_colors['sin'][i], alpha=line_alpha, label='MSC1')
                p2 = ax.plot(x_years, _sin_sim, linestyle=dict_linestyle['sin_sim'][i], color=dict_colors['sin_sim'][i], alpha=line_alpha, label='MSC2')
                ax.set_ylim(ylim_rng[0], ylim_rng[1])

            ax.set_title(f'{r}: {dict_tc[r]}')
            ax.annotate(f'({alphabets[r]})', xy=alphabets_coords, xycoords='axes fraction', fontsize=13, weight='bold')

    fig.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98,
                        wspace=0.23, hspace=0.40)

    fig.text(0.5, -0.10, 'Months', ha='center', fontsize=fsize_suplab)
    fig.text(-0.04, 0.51, 'GPP (gC m-2 day-1)', va='center', rotation='vertical', fontsize=fsize_suplab)

    # add legends and alphabets
    axes[0].legend(
                handles=legend_handles,  # [l1, l2, l3, l4],
                loc='upper center', bbox_to_anchor=(0.5, 1.17), bbox_transform=fig.transFigure,
                fontsize=fsize_legend, fancybox=False, ncol=4, frameon=False
            )

    for r in range(12):
        axes[r].set_xticks(np.arange(1, 13, 2))
        axes[r].set_xticklabels(np.arange(1, 13, 2), fontsize=11)

    fig.savefig(
        os.path.join(path_out, f"figa99_compare_gpp_msc_esv_transcom.png"),
        dpi=600,
        transparent=False,
        bbox_inches='tight'
    )
    plt.clf()
    
    # sys.exit("done!")  # doesn't work... keep running somehow
    print('Done!', flush=True)
    os.system(f"pkill -f {os.path.basename(__file__)}")  # this works...

if __name__ == '__main__':
    plot_figa99_compare_gpp_msc_esv_transcom(path_expOutput=argv[1])
# %%
