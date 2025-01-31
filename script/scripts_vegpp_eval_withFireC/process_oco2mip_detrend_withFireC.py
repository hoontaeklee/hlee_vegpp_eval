'''
save detrended data of each ensemble member
--> and then they will be used to calculate MSC and IAV, to aggregate to regional MSC and IAV time series
'''

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from sys import path
if '/Net/Groups/BGI/people/hlee/scripts/utils' not in path:
    path.insert(1, '/Net/Groups/BGI/people/hlee/scripts/utils')  # https://stackoverflow.com/a/4383597/7578494    
from detrend_dataset import detrend_dataset

#%% for 2015-2019

## time - target period
date_start = '2015-01-01'
date_end = '2019-12-31'
date_out = pd.date_range(date_start, date_end, freq='1MS')

## fire emission - load and convert the unit from daily to yearly considering the leaf year
# gC m-2 month-1
path_gfed = '/Net/Groups/BGI/people/hlee/data/GFED/v4/C_Emissions.2015-2020.1deg.3dim.nc'
fire = xr.open_dataset(path_gfed).sel(time=slice(date_start, date_end))
ts_1519 = pd.date_range('2015-01-01', '2019-12-31', freq='1y')
days_in_year = np.where(ts_1519.is_leap_year, 366, 365)
days_in_year = np.repeat(days_in_year, 12, axis=0)
days_in_year_3d = np.tile(days_in_year, (180, 360, 1)).transpose(2, 0, 1)
fire['C_Emissions'] = fire['C_Emissions'] * days_in_year_3d

path_in = '/Net/Groups/BGI/people/mjung/FLUXCOM/_ANALYSIS/OCO2InversionMIPv10/raw/flux_mip/gridded_fluxes'
files = os.listdir(os.path.join(path_in))
target_exp_label = 'LNLGIS'
member_to_exclude = ['LoFI', 'JHU', 'CMS-Flux', 'EnsMean', 'EnsStd']
files = [f for f in files if f.endswith('.nc4') and target_exp_label in f and all(m not in f for m in member_to_exclude)]
path_out = '/Net/Groups/BGI/people/hlee/data/oco2mipv10/detrended/members'

for f in range(len(files)):
    member_label = files[f].split('_')[0]
    print(f'processing {member_label}, {f+1} / {len(files)}', flush=True)

    ds = xr.open_dataset(os.path.join(path_in, files[f])).sortby('latitude', ascending=False)
    ds = ds.isel(time=np.arange(60))
    # ds_nf = ds.copy(deep=True)
    ds['time'] = fire['time']
    ds = ds.rename({'latitude':'lat', 'longitude':'lon'})
    # ds_nf['net'] = xr.where(np.isnan(fire['C_Emissions']), ds_nf['net'], ds_nf['net'] - fire['C_Emissions'])  # subtract fire emission

    ds_det = detrend_dataset(ds_orig=ds, list_var=['net'], use_anomaly=False)
    ds_det.to_netcdf(os.path.join(path_out, files[f]).replace('.nc4', '_det_2015_2019.nc'))