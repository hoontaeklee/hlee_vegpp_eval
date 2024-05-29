'''
save detrended data of each ensemble member
--> and then they will be used to calculate MSC and IAV, to aggregate to regional MSC and IAV time series
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
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from detrend_dataset import detrend_dataset
from sys import argv

#%% for 2015-2019

## time - target period
dict_src = {
    'GPP': {
        'period': ('2002-01-01', '2015-12-31')
        },
    'TER': {
        'period': ('2002-01-01', '2015-12-31')
        },
    'NEE': {
        'period': ('2002-01-01', '2015-12-31')
        }
}
path_in = os.path.join(path_bgi, 'people/hlee/data/FLUXCOM/members')
path_out = os.path.join(path_bgi, 'people/hlee/data/FLUXCOM/members_detrended')

for v in range(len(list(dict_src.keys()))):

    vname = list(dict_src.keys())[v]
    files = os.listdir(os.path.join(path_in))
    files = [f for f in files if vname in f]
    files.sort()
    date_start = dict_src[vname]['period'][0]
    date_end = dict_src[vname]['period'][1]
    date_label = date_start.split('-')[0]+'-'+date_end.split('-')[0]
    date_out = pd.date_range(date_start, date_end, freq='1MS')

    for f in range(len(files)):
        print(f'processing {vname}, {f+1} / {len(files)}', flush=True)

        ds = xr.open_dataset(os.path.join(path_in, files[f]))
        ds = ds.sortby('lat', ascending=False)
        ds = ds.sortby('lon', ascending=True)
        ds = ds.sel(time=slice(date_start, date_end))

        ds_det = detrend_dataset(ds_orig=ds, list_var=[], use_anomaly=False)
        ds_det.to_netcdf(os.path.join(path_out, files[f]).replace('.nc', f'.det.{date_label}.nc'))