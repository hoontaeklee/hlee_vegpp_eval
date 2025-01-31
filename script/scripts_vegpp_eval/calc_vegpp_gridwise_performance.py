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

path_bgi='/Net/Groups/BGI'  # path to BGI
import os
import sys
if os.path.join(path_bgi, 'people/hlee/scripts/utils') not in sys.path:
    sys.path.insert(1, os.path.join(path_bgi, 'people/hlee/scripts/utils'))  # https://stackoverflow.com/a/4383597/7578494
if os.path.join(path_bgi, 'people/hlee/scripts/diagnose_tws_nee') not in sys.path:
    sys.path.insert(1, os.path.join(path_bgi, 'people/hlee/scripts/diagnose_tws_nee'))  # https://stackoverflow.com/a/4383597/7578494
import numpy as np
import xarray as xr
import pandas as pd
from sys import argv
from calc_metrics import calc_rsq, calc_rae
from apply_transcom_mask import apply_transcom_mask
from calc_land_area_weighted_mean import calc_land_area_weighted_mean
import matplotlib.pyplot as plt

list_vname = [
    'wTotal', 'wSnow', 'evapTotal', 'roTotal', 'gpp', 'NEE'
]

dict_constraints = {
    'wTotal': {
        'form': 'raw',
        'path': os.path.join(path_bgi, 'people/hlee/data/GRACE/GRCTellus.JPL.200204_202110.GLO.RL06M.MSCNv02CRI.scaleFactorApplied.areaWeighted.200101_201912.nc'),
        'vname': 'graceTWS',
        'resample': 'mean',
        'path_landfrac': 'ones',  # no landfractions data used...
        'vname_landfrac': 'landfraction',
        'fac_landfrac': 1.0
        },
    'wSnow': {
        'form': 'raw',
        'path': os.path.join(path_bgi, 'people/hlee/data/GlobSnow/v3/processed/globsnow.v3.360.180.2001.2018.swe.gap-filled.with_NaNs_for_2001_2019.masked.nc'),
        'vname': 'swe',
        'resample': 'mean',
        'path_landfrac': os.path.join(path_bgi, 'people/hlee/data/oco2mipv10/area.ESACCI.360.180.nc'),
        'vname_landfrac': 'fraction',
        'fac_landfrac': 0.01
    },
    'evapTotal': {
        'form': 'msc',
        'path': os.path.join(path_bgi, 'people/hlee/data/FLUXCOM/ET.MSC.RS.FP-ALL.MLM-ALL.METEO-NONE.360_180.monthly.2001-2019.3dim.nc'),
        'vname': 'ET',
        'resample': 'sum',
        'path_landfrac': os.path.join(path_bgi, 'people/hlee/data/FLUXCOM/landfraction.360_180.nc'),
        'vname_landfrac': 'landfraction',
        'fac_landfrac': 1.0
    },
    'roTotal': {
        'form': 'msc',
        'path': os.path.join(path_bgi, 'people/hlee/data/GRUN/multi_years/Runoff.G-RUN_ENSEMBLE_MMM_v1.360.180.200201.201712.masked.det.nc'),
        'vname': 'Runoff_msc',
        'resample': 'sum',
        'path_landfrac': os.path.join(path_bgi, 'people/hlee/data/oco2mipv10/area.ESACCI.360.180.nc'),  # as many of ERA5 data was used to construct GRUN...
        'vname_landfrac': 'fraction',
        'fac_landfrac': 0.01
    },
    'gpp': {
        'form': 'msc',
        'path': os.path.join(path_bgi, 'people/hlee/data/FLUXCOM/GPP.RS.FP-ALL.MLM-ALL.METEO-NONE.360_180.monthly.2001-2019.3dim.det.nc'),
        'vname': 'GPP_msc',
        'resample': 'sum',
        'path_landfrac': os.path.join(path_bgi, 'people/hlee/data/FLUXCOM/landfraction.360_180.nc'),
        'vname_landfrac': 'landfraction',
        'fac_landfrac': 1.0
    },
    'NEE': {
        'form': 'raw',
        'path': os.path.join(path_bgi, 'people/hlee/data/oco2mipv10/EnsMedian_gridded_fluxes_LNLGIS_GFED4FireRemoved_2001_2019.nc'),
        'vname': 'net_med_nf',
        'resample': 'sum',
        'path_landfrac': os.path.join(path_bgi, 'people/hlee/data/oco2mipv10/area.ESACCI.360.180.nc'),
        'vname_landfrac': 'fraction',
        'fac_landfrac': 0.01
    }
}

# load study area common nonnan mask of forcing
path_common_nonnan = os.path.join(path_bgi, 'people/hlee/to_others/sindbad_h2m_for_2001_2019/common_nonnan_pixels.nc')
ds_common_nonnan = xr.open_dataset(path_common_nonnan)
ds_common_nonnan = ds_common_nonnan.sortby('lat', ascending=False)
ds_common_nonnan = ds_common_nonnan.sortby('lon', ascending=True)
common_nonnan = ds_common_nonnan.common_nonnan_pixels.values

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

for i in range(len(list_vname)):
    vname = list_vname[i]
    vname_obs = dict_constraints[vname]['vname']

    print(f'processing {vname}, {i+1} / {len(list_vname)}', flush=True)

    # load sindbad output
    if dict_constraints[vname]['form']=='raw':
        path_expOutput = os.path.join(path_bgi, 'people/hlee/sindbad/data/output/VEGPP2pool1519_studyArea_10k_RD4wtnanmean_1_20230712')
        fname_sin = f'VEGPP2pool1519_studyArea_10k_RD4wtnanmean_1_{vname}_3dim_fullPixel.nc'
        ds_sin = xr.open_dataset(os.path.join(path_expOutput, fname_sin))
        vname_ds_sin = vname
        vname_ds_obs = vname_obs
    else:  # in the case of msc
        path_expOutput = os.path.join(path_bgi, 'people/hlee/sindbad/data/output/VEGPP2pool1519_studyArea_10k_RD4wtnanmean_1_20230712/detrended')
        fname_sin = f'VEGPP2pool1519_studyArea_10k_RD4wtnanmean_1_20230712_{vname}.nc'
        ds_sin = xr.open_dataset(os.path.join(path_expOutput, fname_sin))
        vname_ds_sin = vname + '_msc'
        vname_ds_obs = vname_obs
    ds_sin = ds_sin.sortby('lat', ascending=False)
    ds_sin = ds_sin.sortby('lon', ascending=True)

    # aggregate daily results to monthly
    if dict_constraints[vname]['resample']=='mean':
        ds_sin = ds_sin.resample(time='1MS').reduce(np.nanmean)
    
    if dict_constraints[vname]['resample']=='sum':
        ds_sin = ds_sin.resample(time='1MS').reduce(np.nansum)

    date_start_sin = ds_sin.time[0].data
    date_end_sin = ds_sin.time[-1].data
    nts = ds_sin.time.size

    # load constraint
    ds_obs = xr.open_dataset(dict_constraints[vname]['path'])
    if vname=='wSnow':
        ds_obs = ds_obs.resample(time='1MS').reduce(np.nanmean)
    ds_obs = ds_obs.sel(time=slice(date_start_sin, date_end_sin))

    # calculate gridwise performance metrics
    ar_rsq_grid = np.ones(ds_sin[vname_ds_sin].shape[1:]) * np.nan
    ar_rae_grid = np.ones(ds_sin[vname_ds_sin].shape[1:]) * np.nan

    ar_corr_1 = ds_obs[vname_ds_obs].values.reshape(nts, -1).transpose(1, 0)
    ar_corr_2 = ds_sin[vname_ds_sin].values.reshape(nts, -1).transpose(1, 0)

    ar_rsq_grid = np.array([
        calc_rsq(d1=e1, d2=e2, ndigit=2) if np.sum(~np.isnan(e1) * ~np.isnan(e2)) > 3 else np.nan \
        for e1, e2 in zip(ar_corr_1, ar_corr_2)
    ]).reshape(180, 360)

    ar_rae_grid = np.array([
        calc_rae(obs=e1, est=e2) if np.sum(~np.isnan(e1) * ~np.isnan(e2)) > 3 else np.nan \
        for e1, e2 in zip(ar_corr_1, ar_corr_2)
    ]).reshape(180, 360)

    # aggregate into transcom regions
    ds_obs_r = apply_transcom_mask(
            dsin=ds_obs[[vname_ds_obs]],
            pathlf=dict_constraints[vname]['path_landfrac'],
            varlf=dict_constraints[vname]['vname_landfrac'],
            faclf=dict_constraints[vname]['fac_landfrac'],
            path_tc=os.path.join(path_bgi, 'people/hlee/data/transcom/TranscomRegions.360.180.nc'),
            func_aggr='mean',
            p_truncate='',
            tosave=False,
            toplot=False
        )
    ds_obs_r = ds_obs_r[vname_ds_obs]

    ds_sin_r = apply_transcom_mask(
        dsin=ds_sin[[vname_ds_sin]],
        pathlf=dict_constraints[vname]['path_landfrac'],
        varlf=dict_constraints[vname]['vname_landfrac'],
        faclf=dict_constraints[vname]['fac_landfrac'],
        path_tc=os.path.join(path_bgi, 'people/hlee/data/transcom/TranscomRegions.360.180.nc'),
        func_aggr='mean',
        p_truncate='',
        tosave=False,
        toplot=False
    )
    ds_sin_r = ds_sin_r[vname_ds_sin]

    # calculate regional performance metrics
    ar_rsq_r = np.ones(ds_sin_r[vname_ds_sin].shape[0]) * np.nan
    ar_rae_r = np.ones(ds_sin_r[vname_ds_sin].shape[0]) * np.nan

    ar_corr_1 = ds_obs_r[vname_ds_obs].values
    ar_corr_2 = ds_sin_r[vname_ds_sin].values

    ar_rsq_r = np.array([
        calc_rsq(d1=e1, d2=e2, ndigit=2) if np.sum(~np.isnan(e1) * ~np.isnan(e2)) > 3 else np.nan \
        for e1, e2 in zip(ar_corr_1, ar_corr_2)
    ])

    ar_rae_r = np.array([
        calc_rae(obs=e1, est=e2) if np.sum(~np.isnan(e1) * ~np.isnan(e2)) > 3 else np.nan \
        for e1, e2 in zip(ar_corr_1, ar_corr_2)
    ])

    # calculate landfractions across regions
    _plf = dict_constraints[vname]['path_landfrac']
    _vlf = dict_constraints[vname]['vname_landfrac']
    _flf = dict_constraints[vname]['fac_landfrac']
    if _plf=='ones':
        lf = np.ones_like(area)
        _ds_lf = xr.Dataset({_vlf: (['lat', 'lon'],  lf)},
                            coords={'lat': (['lat'], ds_sin.lat.data),
                                    'lon': (['lon'], ds_sin.lon.data)})
    else:
        _ds_lf = xr.open_dataset(_plf)

    lf = _ds_lf[_vlf].values * _flf
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
    
    # aggregate into global
    _ar_wtm_temp = ds_obs_r[vname_ds_obs].values.T
    ar_obs_gl = calc_land_area_weighted_mean(
        arr_data=_ar_wtm_temp,
        arr_area=area_tc.common_nonnan_pixels.values,
        arr_lf=lf_tc[_vlf].values
    )

    _ar_wtm_temp = ds_sin_r[vname_ds_sin].values.T
    ar_sin_gl = calc_land_area_weighted_mean(
        arr_data=_ar_wtm_temp,
        arr_area=area_tc.common_nonnan_pixels.values,
        arr_lf=lf_tc[_vlf].values
    )

    # calculate global performance metrics
    ar_rsq_gl = np.ones(1) * np.nan
    ar_rae_gl = np.ones(1) * np.nan

    ar_rsq_gl = calc_rsq(d1=ar_obs_gl, d2=ar_sin_gl, ndigit=2)
    ar_rae_gl = calc_rae(obs=ar_obs_gl, est=ar_sin_gl)

    # save
    np.savez(
        f'/Net/Groups/BGI/people/hlee/hlee_vegpp_eval/data/performance_metrics_gridwise_transcomRegional_{vname}.npz',
        rsq_grid=ar_rsq_grid,
        rae_grid=ar_rae_grid,
        rsq_region=ar_rsq_r,
        rae_region=ar_rae_r,
        rsq_global=ar_rsq_gl,
        rae_global=ar_rae_gl
    )
