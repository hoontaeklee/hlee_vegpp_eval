'''
- boxplots of regional contributions across koeppen-geiger regions
- SINDBAD, TRENDYv9, CARDAMOM, and EO (OCO2 for NEE and FLUXCOM for GPP)

# note:
# 
'''
import sys
import os
import glob
path_bgi = '/Net/Groups/BGI'
if os.path.join(path_bgi, 'people/hlee/scripts/utils') not in sys.path:
    sys.path.insert(1, os.path.join(path_bgi, 'people/hlee/scripts/utils'))  # https://stackoverflow.com/a/4383597/7578494
if os.path.join(path_bgi, 'people/hlee/scripts/diagnose_tws_nee') not in sys.path:
    sys.path.insert(1, os.path.join(path_bgi, 'people/hlee/scripts/diagnose_tws_nee'))  # https://stackoverflow.com/a/4383597/7578494
import numpy as np
import copy
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sys import argv
import json

def plot_fig_iav_contrib_var_cov_by_regions_koeppengeiger5(path_expOutput):

    # load region mask
    path_rm = os.path.join(path_bgi, 'people/hlee/data/koeppengeiger/kg5_regions_oneDeg.nc')
    ds_rm = xr.open_dataset(path_rm)
    rcnt = 5
    rnames = list(json.loads(ds_rm.attrs['Legends'])['meaning'].values())
    rnames_short = list(json.loads(ds_rm.attrs['Legends'])['meaning'].values())
    rnames_short = [e.replace('Tropic', 'Tropical') for e in rnames_short]
    colrm = ['royalblue', 'sienna', 'forestgreen', 'rebeccapurple', 'lightslategrey']
    ar_r = ds_rm.kgRegions.values
    color_bc = {k:v for (k,v) in zip(rnames_short, colrm)}

    # I want to draw in the order as in the paper...
    list_rorder = [1, 2, 3, 4, 5]

    #%% load cov. norm. of SINDBAD
    # path_expOutput = os.path.join(path_bgi, 'people/hlee/sindbad/data/output/VEGPP2pool1519_studyArea_10k_RD4wtnanmean_1_20230712/detrended')
    path_npz_mem_sin = os.path.join(os.path.dirname(path_expOutput), 'koeppengeiger5_region_cov_norm_gpp.npz')
    ar_cov_gpp_det_kg_mem_sin = np.load(path_npz_mem_sin)['ar_cov_det_sin']

    path_npz_mem_sin = os.path.join(os.path.dirname(path_expOutput), 'koeppengeiger5_region_cov_norm_cRECO.npz')
    ar_cov_reco_det_kg_mem_sin = np.load(path_npz_mem_sin)['ar_cov_det_sin']

    path_npz_mem_sin = os.path.join(os.path.dirname(path_expOutput), 'koeppengeiger5_region_cov_norm_NEE.npz')
    ar_cov_nee_det_kg_mem_sin = np.load(path_npz_mem_sin)['ar_cov_det_sin']
    
    ar_cov_var_nee_det_kg_mem_sin = np.load(path_npz_mem_sin)['ar_cov_var_det_sin']
    ar_cov_cov_nee_det_kg_mem_sin = np.load(path_npz_mem_sin)['ar_cov_cov_det_sin']

    # plot
    palette = sns.color_palette('colorblind')
    fig, ax = plt.subplots()
    ax.plot(np.arange(5), ar_cov_var_nee_det_kg_mem_sin, 'o', color=palette[0], label='Variance')
    ax.plot(np.arange(5), ar_cov_cov_nee_det_kg_mem_sin, 'o', color=palette[1], label='Covariance')
    ax.plot(np.arange(5), ar_cov_nee_det_kg_mem_sin, 'o', color=palette[2], label='Variance+Covariance')
    ax.legend(frameon=False)
    ax.axhline(
            y=0,
            color='black',
            linewidth=1,
            linestyle='dashed'
        )

    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(rnames_short, fontsize=11, rotation=35)
    ax.set_ylim(-0.25, 0.85)
    ax.set_xlabel('Regions', x=0.50, y=-0.2, fontsize=16)
    ax.set_ylabel('Contribution (-)', x=-0.12, fontsize=16)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    path_out = os.path.join(path_bgi, 'people/hlee/hlee_vegpp_eval/figures_revision1')
    save_name = os.path.join(path_out, f"fig_iav_contrib_var_cov_by_regions_koeppengeiger5.png")
    fig.savefig(
        save_name,
        dpi=600,
        bbox_inches='tight',
        facecolor='w',
        transparent=False
    )

    plt.clf()

if __name__ == '__main__':
    plot_fig_iav_contrib_var_cov_by_regions_koeppengeiger5(path_expOutput=argv[1])

# %%
