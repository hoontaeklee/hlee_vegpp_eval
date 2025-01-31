'''
- detrend output variables of a sindbad run constrained with Jena Inversion, not OCO-2 MIP
'''

#%% load libraries
import os
import glob as glob
import copy
import xarray as xr
import pandas as pd
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import scipy.io as scio
from sys import argv

#%% load functions
def remove_pix_mean(dat_full, idx_start, idx_end):
    # dat_full: numpy array with the dimension of (time, lat, lon)
    dt = np.ones_like(dat_full) * np.nan
    yrs = np.arange(1, len(dt) + 1)
    for i in range(180):
        for j in range(360):
            dat_p = dat_full[:,i,j]
            nanc=np.sum(~np.isnan(dat_p))
            if nanc > 3:
                dt[:,i,j] = dat_p - np.nanmean(dat_p[idx_start:(idx_end+1)])
    return dt    

def detrend_monthly(dat_full, nlat=180, nlon=360):
    '''
    data - trend
    '''
    # dat_full: numpy array with the dimension of (time, lat, lon)
    dat_yr_mon = dat_full.reshape(-1,12,nlat,nlon)
    dt = np.ones_like(dat_yr_mon) * np.nan
    yrs = np.arange(1, len(dt) + 1)
    for mn in range(12):
        dat_mon = dat_yr_mon[:,mn,:,:]
        for i in range(nlat):
            for j in range(nlon):
                dat_p = dat_mon[:,i,j]
                nanc=np.sum(~np.isnan(dat_p))
                non_nan_ind = ~np.isnan(dat_p)
                if nanc >= 3:
                    dat_p_sel = dat_p[non_nan_ind]
                    yr_sel = yrs[non_nan_ind]
                    p = np.polyfit(yr_sel, dat_p_sel, 1)  # can be replaced with RLM
                    tr = yrs * p[0] + p[1]
                    dt[:,mn,i,j] = dat_p - tr
    dt=dt.reshape(-1,nlat,nlon)
    return dt

def get_monthly_msc(dat_full, nlat=180, nlon=360):
    # dat_full: numpy array with the dimension of (time, lat, lon)
    dat_yr_mon = dat_full.reshape(-1,12,nlat,nlon)
    dt = np.ones_like(dat_yr_mon) * np.nan
    yrs = np.arange(1, len(dt) + 1)
    for mn in range(12):
        dat_mon = dat_yr_mon[:,mn,:,:]
        for i in range(nlat):
            for j in range(nlon):
                dat_p = dat_mon[:,i,j]
                nanc=np.sum(~np.isnan(dat_p))
                non_nan_ind = ~np.isnan(dat_p)
                if nanc >= 3:
                    dat_p_sel = dat_p[non_nan_ind] - np.nanmean(dat_p[non_nan_ind])
                    yr_sel = yrs[non_nan_ind]
                    p = np.polyfit(yr_sel, dat_p_sel,1)  # can be replaced with RLM
                    tr = yrs * p[0] + p[1]
                    dt[:,mn,i,j] = np.nanmean(dat_p - tr)  # np.nanmean(dat_p)
    dt=dt.reshape(-1,nlat,nlon)
    return dt

def get_trends(dat_full):
    # dat_full: numpy array with the dimension of (time, lat, lon)
    # get linear trend (slope) of full-time series of each pixel
    dt_slp = np.ones(dat_full.shape[1:]) * np.nan
    dt_idx = np.arange(dat_full.shape[0])
    for i in range(180):
        for j in range(360):
            dat_p = dat_full[:,i,j]
            nanc=np.sum(~np.isnan(dat_p))
            non_nan_ind = ~np.isnan(dat_p)
            if nanc > 3:
                dat_p_sel = dat_p[non_nan_ind]
                dt_idx_sel = dt_idx[non_nan_ind]
                p = np.polyfit(dt_idx_sel, dat_p_sel, 1)  # can be replaced with RLM
                dt_slp[i, j] = p[0]
    return dt_slp    

def detrend_dataset(ds_orig, idx_start, idx_end, list_var=[], use_anomaly=False, nlat=180, nlon=360):
    '''
    ds_orig: xarray dataset
    list_var: list of variable names to detrend. If empty, detrend all variables
    '''

    if len(list_var)==0:
        list_var = list(ds_orig.data_vars)  # list of variables in the original xr.dataset
    list_dims_temp = list(ds_orig.dims.keys())  # ['time', 'lat', 'lon'] 
    if 'time' in list_dims_temp:
        list_dims = ['time' , 'lat', 'lon']  # may need to generalize...
    else:
        list_dims = ['lat', 'lon']
    list_coords = []  # coordinations for dimensions of the xr.dataset
    for i in range(len(list_dims)):
        list_coords.append(ds_orig[list_dims[i]])
    dic_ds = {}  # dictionary for creating the final xr.dataset
    list_label_key = [
        'det',
        'msc',
        'tr',
        'msc_amp',
        'det_std'
    ]
    # add dictionary items of each variable
    for i, var in enumerate(list_var):
        if use_anomaly:
            arr_var = remove_pix_mean(ds_orig[var].values, idx_start=idx_start, idx_end=idx_end, nlat=nlat, nlon=nlon)
        else:
            arr_var = ds_orig[var].values
        arr_det = detrend_monthly(arr_var, nlat=nlat, nlon=nlon)
        arr_msc = get_monthly_msc(arr_var, nlat=nlat, nlon=nlon)
        arr_tr = get_trends(arr_var, nlat=nlat, nlon=nlon)
        arr_msc_amp = np.nanmax(arr_msc, axis=0) - np.nanmin(arr_msc, axis=0)
        arr_det_std = np.nanstd(arr_det, axis=0)
        list_arr = [arr_det, arr_msc, arr_tr, arr_msc_amp, arr_det_std]
        for k in range(len(list_label_key)):
            if k <= 1:
                new_da = xr.DataArray(list_arr[k], coords=list_coords, dims=list_dims)  # w/ time
            else:
                new_da = xr.DataArray(list_arr[k], coords=list_coords[1:], dims=list_dims[1:])  # w/o time
            new_key = var+'_'+list_label_key[k]
            dic_ds[new_key] = new_da
    ds_det = xr.Dataset(data_vars=dic_ds)
    return ds_det

def detrend_sindbad_output(path_expOutput):
    # path_expOutput = '/Net/Groups/BGI/people/hlee/sindbad/data/output/VEGPP2pool1519_studyArea_10k_RD4wtnanmean_jeni_1_20240314'
    
    #%%
    # detrend SINDBAD VEGPP

    # load the original .nc
    print(f'path_expOutput={path_expOutput}', flush=True)

    runname = path_expOutput.split('/')[-1]
    files = os.listdir(os.path.join(path_expOutput))
    files_var = [f for f in files if f.endswith('.nc')]
    dict_target_var = {  # varName: (period); period is for the obs. if exists, otherwise simulation period.
        # 'evapTotal': ('2002-01-01', '2015-12-31'),
        # 'roTotal': ('2002-01-01', '2017-12-31'),
        # 'wGW': ('2001-01-01', '2019-12-31'),
        # 'wSnow': ('2001-01-01', '2018-12-31'),
        # 'wSoil': ('2001-01-01', '2019-12-31'),
        # 'wSurf': ('2001-01-01', '2019-12-31'),
        # 'wTotal': ('2002-01-01', '2017-12-31'),
        # 'gpp': ('2002-01-01', '2015-12-31'),
        # 'npp': ('2002-01-01', '2015-12-31'),
        # 'cRECO': ('2002-01-01', '2015-12-31'),
        'NEE': [('2015-01-01', '2019-12-31'), ('2001-01-01', '2019-12-31')]  # for oco2 and carboscope, respectively
        # 'tranAct': ('2002-01-01', '2015-12-31'),
        # 'AoE': ('2002-01-01', '2015-12-31'),
        # 'vegFrac': ('2002-01-01', '2015-12-31'), 
        # 'cEco_cLit': ('2001-01-01', '2019-12-31'),
        # 'cVeg2cLit': ('2001-01-01', '2019-12-31'),
        # 'cRA': ('2002-01-01', '2015-12-31'),
        # 'cRH': ('2002-01-01', '2015-12-31'),
        # 'cRHcLit': ('2001-01-01', '2019-12-31'),
        # 'cRHcSoil': ('2001-01-01', '2019-12-31'),
        # 'TairSccTau': ('2002-01-01', '2015-12-31'),
        # 'TairScRA': ('2002-01-01', '2015-12-31'),
        # 'fwSoil': ('2001-01-01', '2019-12-31'),
        # 'p_cTaufwSoil_fwSoil_cLit': ('2001-01-01', '2019-12-31')
    }

    # keep files only for target variables
    files_var = [ele for ele in files_var if ele.replace('_'.join(runname.split('_')[:-1])+'_', '').replace('_3dim_fullPixel.nc', '') in dict_target_var.keys()]

    for f in np.arange(0, len(files_var)):  # range(len(files_var)):
        # remain only the substring for variable name
        _vname = files_var[f].replace('_'.join(runname.split('_')[:-1])+'_', '').replace('_3dim_fullPixel.nc', '')
        
        print(f'processing {_vname}, {f+1} / {len(files_var)}', flush=True)

        # if _vname not in ['npp']:
        #     continue

        if type(dict_target_var[_vname])==list:  # in case the variable has two different files and time periods
            for i in range(len(dict_target_var[_vname])):
                date_start = dict_target_var[_vname][i][0]
                date_end = dict_target_var[_vname][i][1]
                path_ds = os.path.join(path_expOutput, files_var[f])
                ds = xr.open_dataset(path_ds).sel(time=slice(date_start, date_end)).resample(time='1M').reduce(np.nanmean)

                # detrend
                use_anomaly = False

                ds_det = detrend_dataset(ds_orig=ds, idx_start=0, idx_end=ds.dims['time'], use_anomaly=use_anomaly)

                path_out = os.path.join(path_expOutput, 'detrended')
                if os.path.exists(path_out)==False:
                    os.mkdir(path_out)

                save_name = os.path.join(path_out, f'{runname}_{_vname+date_start[:4]+date_end[:4]}.nc')
                ds_det.to_netcdf(save_name)

                # close nc
                ds.close()
                ds_det.close()


if __name__ == '__main__':
    detrend_sindbad_output(path_expOutput=argv[1])