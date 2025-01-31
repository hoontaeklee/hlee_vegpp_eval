'''
- visualize covariance matrices by calc_vegpp_cov_matrix_by_grids.py

- plot many maps of spatial contributions in a figure
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
import glob
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from apply_transcom_mask import apply_transcom_mask
from sys import argv
import pingouin as pg
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcbar
import matplotlib.cm as mcm

def plot_trendy_cov_matrix_by_grids_altogether():
    '''
    '''
    # vars_to_diag = ['NEE', 'cRECO', 'cRH', 'gpp', 'wTotal', 'wSoil']

    #%% some paths... landfraction, observations raw & det

    path_common_nonnan = os.path.join(path_bgi, 'people/hlee/to_others/sindbad_h2m_for_2001_2019/common_nonnan_pixels.nc')
    common_nonnan = xr.open_dataset(path_common_nonnan)
    common_nonnan = common_nonnan.sortby('lat', ascending=False)
    common_nonnan = common_nonnan.sortby('lon', ascending=True)
    common_nonnan = common_nonnan.common_nonnan_pixels.values

    # grid area
    area = np.load(os.path.join(path_bgi, 'people/hlee/data/gridAreaAndLandFraction/gridAreaInKm2_180_360.npz'))['area']
    area_msk = np.where(common_nonnan, area, np.nan)


    path_trd = os.path.join(path_bgi, 'people/hlee/data/trendy/v9/regridded_1deg/cov_norm')
    
    list_npz = glob.glob(path_trd + '/*.npz')
    list_npz = [e for e in list_npz if 'region' not in e]
    list_keys = [e.split('/')[-1].split('.')[0].split('_')[2] for e in list_npz]
    list_keys.sort()
    list_npz.sort()
    dict_npz = dict(zip(list_keys, list_npz))

    res = 1.0
    ar_lat = np.arange(90-res/2, -90+res/2-res, -res)
    ar_lon = np.arange(-180+res/2, 180-res/2+res, res)

    keys_to_use = ['gpp', 'reco', 'nee']

    alphabets = [chr(i) for i in range(ord('a'),ord('z')+1)]
    alphabets_coords = (-0.09, 1.07)
    ncol=4
    nrow=2
    fig = plt.figure(figsize=(16*1.1*0.35*3, 9*0.35*nrow))
    gs = fig.add_gridspec(nrow, ncol, width_ratios=[1, 1, 1, 0.05])

    for i, k in enumerate(keys_to_use):  
        print(f'processing {k}, {i+1} / {len(keys_to_use)}', flush=True)

        #%% load data of raw signal
        npz = np.load(dict_npz[k])
        sum_cov_minus_covt = npz['sum_cov_minus_covt_msc']
        print(f'{k}, sum_cov_minus_covt_msc={sum_cov_minus_covt}', flush=True)
        sum_cov_minus_covt = npz['sum_cov_minus_covt_det']
        print(f'{k}, sum_cov_minus_covt_det={sum_cov_minus_covt}', flush=True)

        cov_norm_msc = npz['cov_norm_msc'].reshape(180, 360)
        cov_norm_msc = np.where(common_nonnan, cov_norm_msc, np.nan)
        cov_norm_det = npz['cov_norm_det'].reshape(180, 360)
        cov_norm_det = np.where(common_nonnan, cov_norm_det, np.nan)

        # plot msc
        gs_ridx = 0
        gs_cidx = i
        ax1 = fig.add_subplot(gs[gs_ridx, gs_cidx], projection=ccrs.Robinson())
        ax1.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
        ax1.coastlines(resolution='auto', color='k')
        gl = ax1.gridlines(color='lightgrey', linestyle='--', draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        im = ax1.imshow(cov_norm_msc, transform=ccrs.PlateCarree(), vmin=-0.001, vmax=0.001, cmap='coolwarm', aspect='auto')
        # plt.colorbar(im, ax=ax1, extend='both')
        # ax1.scatter(x=hotspots_latlon[:, 1], y=hotspots_latlon[:, 0], c='indigo', marker='+', alpha=0.1, transform=ccrs.PlateCarree())
        ax1.set_title(f"Grid-wise contributions: {k.upper()}_ESV")
        ax1.annotate(f'({alphabets[i]})', xy=alphabets_coords, xycoords='axes fraction', fontsize=13, weight='bold')

        # plot det
        gs_ridx = 1
        gs_cidx = i
        ax1 = fig.add_subplot(gs[gs_ridx, gs_cidx], projection=ccrs.Robinson())
        ax1.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
        ax1.coastlines(resolution='auto', color='k')
        gl = ax1.gridlines(color='lightgrey', linestyle='--', draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        im = ax1.imshow(cov_norm_det, transform=ccrs.PlateCarree(), vmin=-0.001, vmax=0.001, cmap='coolwarm', aspect='auto')
        # plt.colorbar(im, ax=ax1, extend='both')
        # ax1.scatter(x=hotspots_latlon[:, 1], y=hotspots_latlon[:, 0], c='indigo', marker='+', alpha=0.1, transform=ccrs.PlateCarree())
        ax1.set_title(f"Grid-wise contributions: {k.upper()}_IAV")
        ax1.annotate(f'({alphabets[i]})', xy=alphabets_coords, xycoords='axes fraction', fontsize=13, weight='bold')

        # ax2 = fig.add_subplot(gs[1])
        # ax2.hist(cov_norm.ravel(), weights=np.ones_like(cov_norm.ravel()) / cov_norm.size, bins=20, edgecolor='white')
        # ax2. set_xlabel('Contributions (-)')
       # ax2.set_ylabel('Fraction (-)')
        # ax2.axvline(x=cue_mean, color='r')
    fig.tight_layout()

    # add colorbar
    cax = fig.add_subplot(gs[:, -1])
    cb = mcbar.ColorbarBase(
        cax,
        cmap=mcm.get_cmap('coolwarm'),
        norm=mcolors.Normalize(vmin=-0.001, vmax=0.001),
        orientation='vertical',
        extend='both'
    )
    cb.ax.set_ylabel('Contribution (-)', fontsize=13)
    cb.formatter.set_powerlimits((0, 0))
    cb.formatter._useMathText = True
        
    dir_exp_out = os.path.join(path_bgi, 'people/hlee/data/trendy/v9/regridded_1deg/figures')
    save_name = os.path.join(dir_exp_out, f'cov_map_hist_cflux_altogether_trendyv9_med.png')
    fig.savefig(
        save_name,
        dpi=600,
        bbox_inches='tight',
        facecolor='w',
        transparent=False
    )

if __name__ == '__main__':
    plot_trendy_cov_matrix_by_grids_altogether()
