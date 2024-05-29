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
import numpy.ma as ma
import json
import ast

def plot_figa12_iav_contrib_by_regions_koeppengeiger5ns(path_expOutput):

    # load region mask
    path_rm = os.path.join(path_bgi, 'people/hlee/data/koeppengeiger/kg5ns_regions_oneDeg.nc')
    ds_rm = xr.open_dataset(path_rm)
    rcnt = 9
    rnames_rm = list(ast.literal_eval(ds_rm.attrs['Legends'])['meaning'].values())
    rnames_short = list(ast.literal_eval(ds_rm.attrs['Legends'])['meaning'].values())
    colrm = ['blue', 'royalblue', 'cadetblue', 'skyblue', 'brown', 'sienna', 'darkgreen', 'forestgreen', 'rebeccapurple']
    ar_r = ds_rm.kgRegions.values
    color_bc = {k:v for (k,v) in zip(rnames_short, colrm)}

    # I want to draw in the order as in the paper...
    list_rorder = list(np.arange(len(rnames_rm))+1)
    
    #%% load cov. norm. of oco2
    path_oco2 = os.path.join(path_bgi, 'people/hlee/data/oco2mipv10/detrended/cov_norm')

    # the ensemble median
    path_npz_med_oco2 = os.path.join(path_oco2, 'koeppengeiger5ns_region_cov_norm_EnsMedian.npz')
    ar_cov_nee_det_kg_med_oco2 = np.load(path_npz_med_oco2)['ar_cov_det']

    # members
    path_npz_mem_oco2 = os.path.join(path_oco2, 'koeppengeiger5ns_region_cov_norm_members.npz')
    ar_cov_nee_det_kg_mem_oco2 = np.load(path_npz_mem_oco2)['ar_cov_det']

    #%% load cov. norm. of SINDBAD
    # path_expOutput = os.path.join(path_bgi, 'people/hlee/sindbad/data/output/VEGPP2pool1519_studyArea_10k_RD4wtnanmean_1_20230712/detrended')
    path_npz_mem_sin = os.path.join(os.path.dirname(path_expOutput), 'koeppengeiger5ns_region_cov_norm_gpp.npz')
    ar_cov_gpp_det_kg_mem_sin = np.load(path_npz_mem_sin)['ar_cov_det_sin']

    path_npz_mem_sin = os.path.join(os.path.dirname(path_expOutput), 'koeppengeiger5ns_region_cov_norm_cRECO.npz')
    ar_cov_reco_det_kg_mem_sin = np.load(path_npz_mem_sin)['ar_cov_det_sin']

    path_npz_mem_sin = os.path.join(os.path.dirname(path_expOutput), 'koeppengeiger5ns_region_cov_norm_NEE.npz')
    ar_cov_nee_det_kg_mem_sin = np.load(path_npz_mem_sin)['ar_cov_det_sin']

    #%% load cov. norm. of TRENDY
    path_trd = os.path.join(path_bgi, 'people/hlee/data/trendy/v9/regridded_1deg/cov_norm')

    # the ensemble median
    path_npz_med_trd = os.path.join(path_trd, 'koeppengeiger5ns_region_cov_norm_gpp_2002-2015_EnsMedian.npz')
    ar_cov_gpp_det_kg_med_trd = np.load(path_npz_med_trd)['ar_cov_det']

    path_npz_med_trd = os.path.join(path_trd, 'koeppengeiger5ns_region_cov_norm_reco_2002-2015_EnsMedian.npz')
    ar_cov_reco_det_kg_med_trd = np.load(path_npz_med_trd)['ar_cov_det']

    path_npz_med_trd = os.path.join(path_trd, 'koeppengeiger5ns_region_cov_norm_nee_2015-2019_EnsMedian.npz')
    ar_cov_nee_det_kg_med_trd = np.load(path_npz_med_trd)['ar_cov_det']

    # members
    idx_model2exclude = [6, 7, 9]
    path_npz_mem_trd = os.path.join(path_trd, 'koeppengeiger5ns_region_cov_norm_gpp_2002-2015_members.npz')
    ar_cov_gpp_det_kg_mem_trd = np.load(path_npz_mem_trd)['ar_cov_det']
    ar_cov_gpp_det_kg_mem_trd[idx_model2exclude, :] = np.nan  # remove jules-es-1p0, lpj-guess, ocn

    path_npz_mem_trd = os.path.join(path_trd, 'koeppengeiger5ns_region_cov_norm_reco_2002-2015_members.npz')
    ar_cov_reco_det_kg_mem_trd = np.load(path_npz_mem_trd)['ar_cov_det']
    ar_cov_reco_det_kg_mem_trd[idx_model2exclude, :] = np.nan  # remove jules-es-1p0, lpj-guess, ocn

    path_npz_mem_trd = os.path.join(path_trd, 'koeppengeiger5ns_region_cov_norm_nee_2015-2019_members.npz')
    ar_cov_nee_det_kg_mem_trd = np.load(path_npz_mem_trd)['ar_cov_det']
    ar_cov_nee_det_kg_mem_trd[idx_model2exclude, :] = np.nan  # remove jules-es-1p0, lpj-guess, ocn

    #%% create dataframe for boxplot

    # gpp
    df_mem_gpp_kg_trd = pd.DataFrame(
        {
            'region_group':np.repeat('kg5', ar_cov_gpp_det_kg_mem_trd.size),
            'region':rnames_short*ar_cov_gpp_det_kg_mem_trd.shape[0],
            'product':np.repeat('TRENDY', ar_cov_gpp_det_kg_mem_trd.size),
            'contrib':ar_cov_gpp_det_kg_mem_trd.ravel()
        },
        columns=['region_group', 'region', 'product', 'contrib']
    )
    df_mem_gpp = pd.concat([df_mem_gpp_kg_trd])

    # reco
    df_mem_reco_kg_trd = pd.DataFrame(
        {
            'region_group':np.repeat('kg5', ar_cov_reco_det_kg_mem_trd.size),
            'region':rnames_short*ar_cov_reco_det_kg_mem_trd.shape[0],
            'product':np.repeat('TRENDY', ar_cov_reco_det_kg_mem_trd.size),
            'contrib':ar_cov_reco_det_kg_mem_trd.ravel()
        },
        columns=['region_group', 'region', 'product', 'contrib']
    )
    df_mem_reco = pd.concat([df_mem_reco_kg_trd])

    # nee
    df_mem_nee_kg_oco2 = pd.DataFrame(
        {
            'region_group':np.repeat('kg5', ar_cov_nee_det_kg_mem_oco2.size),
            'region':rnames_short*ar_cov_nee_det_kg_mem_oco2.shape[0],
            'product':np.repeat('OCO2', ar_cov_nee_det_kg_mem_oco2.size),
            'contrib':ar_cov_nee_det_kg_mem_oco2.ravel()
        },
        columns=['region_group', 'region', 'product', 'contrib']
    )
    df_mem_nee_kg_trd = pd.DataFrame(
        {
            'region_group':np.repeat('kg5', ar_cov_nee_det_kg_mem_trd.size),
            'region':rnames_short*ar_cov_nee_det_kg_mem_trd.shape[0],
            'product':np.repeat('TRENDY', ar_cov_nee_det_kg_mem_trd.size),
            'contrib':ar_cov_nee_det_kg_mem_trd.ravel()
        },
        columns=['region_group', 'region', 'product', 'contrib']
    )
    df_mem_nee = pd.concat([df_mem_nee_kg_oco2, df_mem_nee_kg_trd])

    #%% plot 
    dict_colors = {  # data_name: [col_var1, col_var2, ...]
        'OCO2': '#4E4E4E',
        'SINDBAD': '#BBAF52',
        'CARDAMOM': '#9A360E',
        'TRENDY': '#3C7EA2'
    }
    alphabets = [chr(i) for i in range(ord('a'),ord('z')+1)]
    alphabets_coords = (-0.085, 1.07)
    jit_size = 3
    fig = plt.figure(figsize=(16*0.3*1, 9*0.1375*4))
    gs = fig.add_gridspec(nrows=3, ncols=1, figure=fig)
    plt.rcParams.update({'mathtext.default':  'regular' })

    # gpp
    ngroups = 1
    ax = fig.add_subplot(gs[0])
    bx_kg = sns.boxplot(
        ax=ax,
        data=df_mem_gpp.loc[df_mem_gpp['region_group']=='kg5'].reset_index(),
        x='region',
        y='contrib',
        hue='product',
        order=[rnames_short[e-1] for e in list_rorder],
        dodge=True,
        palette=dict_colors,
        fliersize=0,
        width=0.6,
        zorder=1
        )
    bx_kg.set_xticklabels('')
    jt_kg = sns.stripplot(
        ax=ax,
        data=df_mem_gpp.loc[df_mem_gpp['region_group']=='kg5'].reset_index(),
        x='region',
        y='contrib',
        hue='product',
        order=[rnames_short[e-1] for e in list_rorder],
        dodge=True,
        palette=dict_colors,
        alpha=0.3,
        zorder=2,
        legend=False
        )

    ax.hlines(
        y=[ar_cov_gpp_det_kg_mem_sin[e-1] for e in list_rorder],
        xmin=np.arange(rcnt)-1/4,
        xmax=np.arange(rcnt)+1/4,
        color=dict_colors['SINDBAD'],
        linewidth=2,
        zorder=4
    )

    ax.scatter(
        x=np.arange(rcnt),
        y=[ar_cov_gpp_det_kg_med_trd[e-1] for e in list_rorder],
        s=jit_size*2.25, marker='d', facecolors='red',
        # color=dict_colors['OCO2'],
        # linewidth=2,
        zorder=3
    )

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend('', frameon=False)

    # reco
    ngroups = 1
    ax = fig.add_subplot(gs[1, 0])
    bx_kg = sns.boxplot(
        ax=ax,
        data=df_mem_reco.loc[df_mem_reco['region_group']=='kg5'].reset_index(),
        x='region',
        y='contrib',
        hue='product',
        order=[rnames_short[e-1] for e in list_rorder],
        dodge=True,
        palette=dict_colors,
        fliersize=0,
        width=0.6,
        zorder=1
        )
    bx_kg.set_xticklabels('')
    jt_kg = sns.stripplot(
        ax=ax,
        data=df_mem_reco.loc[df_mem_reco['region_group']=='kg5'].reset_index(),
        x='region',
        y='contrib',
        hue='product',
        order=[rnames_short[e-1] for e in list_rorder],
        dodge=True,
        palette=dict_colors,
        alpha=0.3,
        zorder=2,
        legend=False
        )

    ax.hlines(
        y=[ar_cov_reco_det_kg_mem_sin[e-1] for e in list_rorder],
        xmin=np.arange(rcnt)-1/4,
        xmax=np.arange(rcnt)+1/4,
        color=dict_colors['SINDBAD'],
        linewidth=2,
        zorder=4
    )

    ax.scatter(
        x=np.arange(rcnt),
        y=[ar_cov_reco_det_kg_med_trd[e-1] for e in list_rorder],
        s=jit_size*2.25, marker='d', facecolors='red',
        # color=dict_colors['OCO2'],
        # linewidth=2,
        zorder=3
    )

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend('', frameon=False)

    # nee
    ngroups = 2
    # det, 11 hydro-climatic regions
    ax = fig.add_subplot(gs[2, 0])
    bx_kg = sns.boxplot(
        ax=ax,
        data=df_mem_nee.loc[df_mem_nee['region_group']=='kg5'].reset_index(),
        x='region',
        y='contrib',
        hue='product',
        order=[rnames_short[e-1] for e in list_rorder],
        dodge=True,
        palette=dict_colors,
        fliersize=0,
        zorder=1
        )
    bx_kg.set_xticklabels(bx_kg.get_xticklabels(),rotation=90)
    jt_kg = sns.stripplot(
        ax=ax,
        data=df_mem_nee.loc[df_mem_nee['region_group']=='kg5'].reset_index(),
        x='region',
        y='contrib',
        hue='product',
        order=[rnames_short[e-1] for e in list_rorder],
        dodge=True,
        palette=dict_colors,
        alpha=0.3,
        zorder=2
        )

    ax.hlines(
        y=[ar_cov_nee_det_kg_mem_sin[e-1] for e in list_rorder],
        xmin=np.arange(rcnt)-1/4,
        xmax=np.arange(rcnt)+1/4,
        color=dict_colors['SINDBAD'],
        linewidth=2,
        zorder=4,
        label='SINDBAD (this study)'
    )

    ax.scatter(
        x=np.arange(rcnt)-ngroups/8,
        y=[ar_cov_nee_det_kg_med_oco2[e-1] for e in list_rorder],
        s=jit_size*2.25, marker='d', facecolors='red',
        # color=dict_colors['OCO2'],
        # linewidth=2,
        zorder=3
    )

    ax.scatter(
        x=np.arange(rcnt)+ngroups/8,
        y=[ar_cov_nee_det_kg_med_trd[e-1] for e in list_rorder],
        s=jit_size*2.25, marker='d', facecolors='red',
        # color=dict_colors['OCO2'],
        # linewidth=2,
        zorder=3
    )

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.legend('', frameon=False)

    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[e] for e in np.arange(len(handles)) if e in [0, 1, 4]]
    labels = [labels[e] for e in np.arange(len(labels)) if e in [0, 1, 4]]
    ax.legend('', frameon=False)
    fig.legend(
        handles, labels,
        # bbox_to_anchor=(0.13, 1.03, 0.74, 0.01),
        bbox_to_anchor=(0.10, 1.03, 0.88, 0.01),
        loc="upper center",
        mode="expand",
        borderaxespad=0,
        ncol=3,
        frameon=False
    )

    for i, ax in enumerate(fig.axes):
        ax.annotate(f'({alphabets[i]})'+' '+['GPP', 'RECO', 'NEE'][i], xy=alphabets_coords, xycoords='axes fraction', weight='bold', fontsize=12)
        ax.set_ylim(-0.25, 0.85)
        ax.set_yticks(np.arange(0.0, 1.0, 0.25))

    # fig.suptitle(f"Contributions to the global NEE", x=0.50, y=1.1, fontsize=16)
    fig.supxlabel('Regions', x=0.50, y=-0.32, fontsize=16)
    fig.supylabel('Contribution (-)', x=-0.12, fontsize=16)
    fig.subplots_adjust(
        left=0.02,
        bottom=0.02,
        right=0.98,
        top=0.98,
        wspace=0.15,
        hspace=0.3
        )
    
    path_out = os.path.join(path_bgi, 'people/hlee/hlee_vegpp_eval/figures')
    save_name = os.path.join(path_out, f"figa12_cov_iav_region_koeppengeiger5ns.png")
    fig.savefig(
        save_name,
        dpi=600,
        bbox_inches='tight',
        facecolor='w',
        transparent=False
    )
    
    plt.clf()

if __name__ == '__main__':
    plot_figa12_iav_contrib_by_regions_koeppengeiger5ns(path_expOutput=argv[1])

# %%
