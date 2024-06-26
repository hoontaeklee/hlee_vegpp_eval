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

def plot_fig07_cov_matrix_by_grids(path_expOutput):
    '''
    path_expOutput: string, full path to a sindbad output (**not detrended**)
    '''
    # path_expOutput = os.path.join(path_bgi, 'people/hlee/sindbad/data/output/VEGPP2pool1519_studyArea_10k_RD4wtnanmean_1_20230712')

    #%% some paths... landfraction, observations raw & det

    path_common_nonnan = os.path.join(path_bgi, 'people/hlee/to_others/sindbad_h2m_for_2001_2019/common_nonnan_pixels.nc')
    common_nonnan = xr.open_dataset(path_common_nonnan)
    common_nonnan = common_nonnan.sortby('lat', ascending=False)
    common_nonnan = common_nonnan.sortby('lon', ascending=True)
    common_nonnan = common_nonnan.common_nonnan_pixels.values

    # grid area
    area = np.load(os.path.join(path_bgi, 'people/hlee/data/gridAreaAndLandFraction/gridAreaInKm2_180_360.npz'))['area']
    area_msk = np.where(common_nonnan, area, np.nan)

    list_npz = glob.glob(path_expOutput + '/*.npz')
    list_npz = [e for e in list_npz if 'region' not in e]
    list_keys = [e.split('/')[-1].split('.')[0].replace('cov_norm_', '') for e in list_npz]
    list_keys.sort()
    list_npz.sort()
    dict_npz = dict(zip(list_keys, list_npz))

    res = 1.0
    ar_lat = np.arange(90-res/2, -90+res/2-res, -res)
    ar_lon = np.arange(-180+res/2, 180-res/2+res, res)

    keys_to_use = ['gpp_mod_msc', 'gpp_mod_det', 'cRECO_mod_msc', 'cRECO_mod_det', 'NEE_mod_msc', 'NEE_mod_det']
    # keys_to_use = [e + '_esv-iav' for e in keys_to_use]

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
        sum_cov_minus_covt = npz['sum_cov_minus_covt']
        print(f'{k}, sum_cov_minus_covt={sum_cov_minus_covt}', flush=True)

        cov_norm = npz['cov_norm'].reshape(180, 360)
        cov_norm = np.where(common_nonnan, cov_norm, np.nan)
        
        gs_ridx = i%2
        gs_cidx = i//2
        ax1 = fig.add_subplot(gs[gs_ridx, gs_cidx], projection=ccrs.Robinson())
        ax1.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
        ax1.coastlines(resolution='auto', color='k')
        gl = ax1.gridlines(color='lightgrey', linestyle='--', draw_labels=True)
        gl.top_labels = False
        gl.right_labels = False
        im = ax1.imshow(cov_norm, transform=ccrs.PlateCarree(), vmin=-0.001, vmax=0.001, cmap='coolwarm', aspect='auto')
        ax1.set_title(f"Grid-wise contributions: {k.replace('cRECO', 'RECO').replace('det', 'iav').replace('msc', 'esv').replace('mod_', '').upper()}")
        ax1.annotate(f'({alphabets[i]})', xy=alphabets_coords, xycoords='axes fraction', fontsize=13, weight='bold')

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
        
    path_out = os.path.join(path_bgi, 'people/hlee/hlee_vegpp_eval/figures')
    save_name = os.path.join(path_out, f"fig07_cov_matrix_by_grids.png")
    fig.savefig(
        save_name,
        dpi=600,
        bbox_inches='tight',
        facecolor='w',
        transparent=False
    )

if __name__ == '__main__':
    plot_fig07_cov_matrix_by_grids(path_expOutput=argv[1])
