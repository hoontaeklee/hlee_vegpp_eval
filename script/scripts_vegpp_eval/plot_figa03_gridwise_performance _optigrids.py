'''
1. in calc_vegpp_gridwise_performance.py
calculate performance metric between SINDBAD and constraints (in the form as used in the calibration)
1. for studyarea and study period (available period of constraints)
2. for calibration grids (i.e., 904) and period (2015-2019)

save arrays of gridwise performance (180x360) and
regional performance (nregions)

2. in plot_vegpp_gridwise_performance.py
boxplot for each variable (spread of gridwise metrics within each region + symbols for regional metrics)
subplots for TRANSCOM regions and global
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
import seaborn as sns
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

def plot_vegpp_gridwise_performance_opti904(path_expOutput):

    # path_expOutput = os.path.join(path_bgi, 'people/hlee/sindbad/data/output/VEGPP2pool1519_studyArea_10k_RD4wtnanmean_1_20230712')
    path_src = os.path.join(path_bgi, 'people/hlee/hlee_vegpp_eval')

    #%% load data

    # load transcom mask
    ds_tc = xr.open_dataset(os.path.join(path_bgi, 'people/hlee/data/transcom/TranscomRegions.360.180.nc'))
    ds_tc = ds_tc.sortby('lat', ascending=False)
    ds_tc = ds_tc.sortby('lon', ascending=True)
    ar_tc = ds_tc.TranscomRegions.data
    rnames_tc = [x.decode('UTF-8') for x in ds_tc.Legend.values[0, :]]
    # rnames_tc.insert(0, 'Excluded')
    rnames_tc = rnames_tc[:11]  # exclude oceans
    dict_tc = dict(zip(np.arange(1, len(rnames_tc)+1), rnames_tc))

    # load study area common nonnan mask of forcing
    path_common_nonnan = os.path.join(path_bgi, 'people/hlee/to_others/sindbad_h2m_for_2001_2019/common_nonnan_pixels.nc')
    ds_common_nonnan = xr.open_dataset(path_common_nonnan)
    ds_common_nonnan = ds_common_nonnan.sortby('lat', ascending=False)
    ds_common_nonnan = ds_common_nonnan.sortby('lon', ascending=True)
    common_nonnan = ds_common_nonnan.common_nonnan_pixels.values

    # load opti904 mask
    path_opti904_mask = os.path.join(path_bgi, 'work_3/sindbad/project/twsnee_hlee/tina_veg/data/tina_veg_opti904_grids.nc')
    ds_opti904_mask = xr.open_dataset(path_opti904_mask)
    ds_opti904_mask = ds_opti904_mask.rename({'__xarray_dataarray_variable__':'opti904'})
    ds_opti904_mask = ds_opti904_mask.sortby('lat', ascending=False)
    ds_opti904_mask = ds_opti904_mask.sortby('lon', ascending=True)
    opti904_mask = ds_opti904_mask.opti904.values

    # load metrics array

    list_vname = [
    'wTotal', 'wSnow', 'evapTotal', 'roTotal', 'gpp', 'NEE'
    ]
    list_vname_short = [
        'TWSA', 'SWE', 'ET', 'Q', 'GPP', 'NEE'
    ]
    
    dict_metrics = {k:{'rsq_grid':0, 'rae_grid':0, 'rsq_region':0, 'rae_region':0, 'rsq_global':0, 'rae_global':0} for k in list_vname}
    
    for vname in list_vname:
        path_metrics_temp = os.path.join(path_src, f'data/performance_metrics_gridwise_transcomRegional_optigrids_{vname}.npz')
        
        dict_metrics[vname]['rsq_grid'] = np.load(path_metrics_temp)['rsq_grid']
        dict_metrics[vname]['rae_grid'] = np.load(path_metrics_temp)['rae_grid']
        dict_metrics[vname]['rsq_region'] = np.load(path_metrics_temp)['rsq_region']
        dict_metrics[vname]['rae_region'] = np.load(path_metrics_temp)['rae_region']
        dict_metrics[vname]['rsq_global'] = np.load(path_metrics_temp)['rsq_global']
        dict_metrics[vname]['rae_global'] = np.load(path_metrics_temp)['rae_global']

    list_rsq_grid = []
    list_rae_grid = []
    list_rsq_global = []
    list_rae_global = []
    for vname in list_vname:
        list_rsq_grid.append(dict_metrics[vname]['rsq_grid'].ravel())
        list_rae_grid.append(dict_metrics[vname]['rae_grid'].ravel())
        list_rsq_global.append(dict_metrics[vname]['rsq_global'])
        list_rae_global.append(dict_metrics[vname]['rae_global'])

    ar_value_1d = np.concatenate([list_rsq_grid, list_rae_grid]).ravel()
    ar_variable_1d = np.concatenate([np.repeat(list_vname, 64800), np.repeat(list_vname, 64800)])
    ar_metric_1d = ['rsq']*64800*6 + ['rae']*64800*6
    ar_region_1d = np.tile(ar_tc.ravel(), 2*6)
    
    df_metrics_grid = pd.DataFrame({
        'value': ar_value_1d,
        'variable': ar_variable_1d,
        'metric': ar_metric_1d,
        'region': ar_region_1d
    })

    # grid area
    area = np.load(os.path.join(path_bgi, 'people/hlee/data/gridAreaAndLandFraction/gridAreaInKm2_180_360.npz'))['area']
    area_msk = np.where(opti904_mask, area, np.nan)
    area_tc = apply_transcom_mask(
        dsin=ds_opti904_mask.expand_dims(dim={'time': 1}),
        pathlf='ones',
        varlf='landfraction',
        faclf='1.0',
        path_tc=os.path.join(path_bgi, 'people/hlee/data/transcom/TranscomRegions.360.180.nc'),
        func_aggr='sum',
        tosave=False,
        toplot=False
    )['opti904'].isel(time=0)

    # out-path for figures
    path_out = os.path.join(path_src, 'figures')
    if os.path.exists(path_out) == False:
        os.mkdir(path_out)

    #%% plot rsq
    # set plotting variables
    fsize_legend = 18
    fsize_suplab = 18
    fsize_text = 9
    alphabets = [chr(i) for i in range(ord('a'),ord('z')+1)]
    alphabets_coords = (-0.17, 1.07)
    jit_size = 15
    width_box = 0.8
    nrow = 3
    ncol = 4
    fig, axes = plt.subplots(
        nrow, ncol,
        figsize=(16*ncol*0.15, 9*nrow*0.2),
        gridspec_kw={
            'height_ratios': [1, 1, 1]
        }
    )
    axes = axes.flatten()

    # global
    ngroups = 1
    i=0
    # global

    ax = axes[i]
    bx_bc = sns.boxplot(
        ax=ax,
        data=df_metrics_grid.loc[df_metrics_grid['metric']=='rsq'],
        x='variable',
        y='value',
        color='grey',
        order=list_vname,
        dodge=True,
        width=width_box,
        showfliers=False,
        fliersize=0,
        zorder=1
        )
    # jt_bc = sns.stripplot(
    #     ax=ax,
    #     data=df_metrics_grid.loc[df_metrics_grid['metric']=='rsq'],
    #     x='variable',
    #     y='value',
    #     color='grey',
    #     order=list_vname,
    #     dodge=True,
    #     size=0.2,
    #     alpha=0.3,
    #     zorder=2
    #     )
    ax.scatter(
        x=np.arange(len(list_vname)),
        y=[list_rsq_global],
        s=jit_size, marker='d', facecolors='red',
        # color=dict_colors['OCO2'],
        # linewidth=2,
        zorder=3
    )

    ax.set_ylim(-0.05, 1.05)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend('', frameon=False)
    ax.set_title(f'Globe', fontsize=fsize_text)
    ax.set_xticklabels(list_vname_short, fontsize=8, rotation=30)
    ax.annotate(f'({alphabets[0]})', xy=alphabets_coords, xycoords='axes fraction', fontsize=11, weight='bold')
            
    # transcom regions
    for r in np.arange(1, 12):
        list_rsq_region = []
        list_rae_region = []
        for vname in list_vname:
            list_rsq_region.append(dict_metrics[vname]['rsq_region'][r-1])
            list_rae_region.append(dict_metrics[vname]['rae_region'][r-1])

        ax = axes[r]
        bx_bc = sns.boxplot(
            ax=ax,
            data=df_metrics_grid.loc[(df_metrics_grid['metric']=='rsq') & (df_metrics_grid['region']==r)],
            x='variable',
            y='value',
            color='grey',
            order=list_vname,
            dodge=True,
            width=width_box,
            showfliers=False,
            fliersize=0,
            zorder=1
            )
        # jt_bc = sns.stripplot(
        #     ax=ax,
        #     data=df_metrics_grid.loc[(df_metrics_grid['metric']=='rsq') & (df_metrics_grid['region']==r)],
        #     x='variable',
        #     y='value',
        #     color='grey',
        #     order=list_vname,
        #     dodge=True,
        #     size=0.2,
        #     alpha=0.3,
        #     zorder=2
        #     )
        ax.scatter(
            x=np.arange(len(list_vname)),
            y=[list_rsq_region],
            s=jit_size, marker='d', facecolors='red',
            # color=dict_colors['OCO2'],
            # linewidth=2,
            zorder=3
        )

        ax.set_ylim(-0.05, 1.05)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.legend('', frameon=False)
        ax.set_title(f'{r}: {dict_tc[r]}', fontsize=fsize_text)
        ax.set_xticklabels(list_vname_short, fontsize=8, rotation=30)
        ax.annotate(f'({alphabets[r]})', xy=alphabets_coords, xycoords='axes fraction', fontsize=11, weight='bold')

    fig.tight_layout()

    fig.text(0.5, -0.02, 'Variable', ha='center', fontsize=fsize_suplab)
    fig.text(-0.02, 0.51, r'$R^2$', va='center', rotation='vertical', fontsize=fsize_suplab)

    fig.savefig(
        os.path.join(path_out, f"figa03_gridwise_performance_rsq_transcom_optigrids.png"),
        dpi=600,
        transparent=False,
        bbox_inches='tight'
    )
    plt.clf()
    
    # sys.exit("done!")  # doesn't work... keep running somehow
    print('Done!', flush=True)
    os.system(f"pkill -f {os.path.basename(__file__)}")  # this works...

if __name__ == '__main__':
    plot_vegpp_gridwise_performance_opti904(path_expOutput=argv[1])
# %%
