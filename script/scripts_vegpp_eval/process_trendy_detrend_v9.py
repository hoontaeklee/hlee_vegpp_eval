'''
detrend trendy v9 S2

- excluded
JULES-ES-1p0 (no ra)
LPJ-GUESS (no rh)
OCN (no ra and rh)
DLEM (no gpp)
'''

path_bgi = '/Net/Groups/BGI'
import os
import sys
import glob
if os.path.join(path_bgi, 'people/hlee/scripts/utils') not in sys.path:
    sys.path.insert(1, os.path.join(path_bgi, 'people/hlee/scripts/utils'))  # https://stackoverflow.com/a/4383597/7578494
if os.path.join(path_bgi, 'people/hlee/scripts/diagnose_tws_nee') not in sys.path:
    sys.path.insert(1, os.path.join(path_bgi, 'people/hlee/scripts/diagnose_tws_nee'))  # https://stackoverflow.com/a/4383597/7578494
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import detrend_dataset
from sys import argv

# 2002-01-01 ~ 2015-12-31 for evaluating GPP MSC (and NEE MSC)
# 2001-01-01 ~ 2019-12-31 for evaluating NEE IAV
def process_trendy_detrend_v9():
    
    dict_src = {
        'gpp': os.path.join(path_bgi, 'people/hlee/data/trendy/v9/regridded_1deg/trendyv9_S2_gpp_2001-2019.nc'),
        'reco': os.path.join(path_bgi, 'people/hlee/data/trendy/v9/regridded_1deg/trendyv9_S2_reco_2001-2019.nc'),
        'nee': os.path.join(path_bgi, 'people/hlee/data/trendy/v9/regridded_1deg/trendyv9_S2_nee_2001-2019.nc'),
        'nbp': os.path.join(path_bgi, 'people/hlee/data/trendy/v9/regridded_1deg/trendyv9_S2_nbp_2001-2019.nc')
    }

    v=0
    i=0

    dict_period = {  # date_start and date_end for each variable
        'date_start': ['2001-01-01', '2002-01-01', '2015-01-01'],
        'date_end': ['2019-12-31', '2015-12-31', '2019-12-31']
    }

    for p in range(len(dict_period['date_start'])):

        # if p in [0, 1]:
        #     continue

        date_start = dict_period['date_start'][p]
        date_end = dict_period['date_end'][p]
        date_label = date_start.split('-')[0]+'-'+date_end.split('-')[0]
        print(f'detrending over the period of {date_start} ~ {date_end}', flush=True)

        for v in range(len(dict_src)):
            vname = list(dict_src.keys())[v]
            print(f'reading {vname}, {v+1} / {len(dict_src)}', flush=True)

            if v in [0, 1, 2]:
                continue

            path_src = dict_src[vname]
            ds = xr.open_dataset(path_src).sel(time=slice(date_start, date_end))
            ds = ds.sortby('lat', ascending=False)
            ds = ds.sortby('lon', ascending=True)

            # ##
            # # two records for 2012-01 (2012-01-01, 2012-01-31) --> use 2012-01-31
            # idx_20120101 = np.argwhere(ds_temp.time.dt.strftime('%Y-%m-%d').values=='2012-01-01')
            # ds_temp = ds_temp.isel(time=np.delete(np.arange(len(ds_temp.time)), idx_20120101))


            # # 2017-12 is missed --> assign mean of other decembers
            # # if leave it as nan, dec. for 2015-2017 will have only two values, being impossible to detrend
            # date_2017 = pd.to_datetime('2017-01-01')
            # if date_2017>=pd.to_datetime(date_start) and date_2017<=pd.to_datetime(date_end):
            #     time_with_201712 = np.insert(ds_temp.time.values, len(ds_temp.time.values), '2017-12-30T12:00:00.000000000')
            #     ds = ds_temp.reindex(time=time_with_201712, fill_value=np.nan)
            #     ar_dec_mean = ds.sel(time=ds.time.dt.month==12).mean(dim='time')[vname].values  # https://stackoverflow.com/a/70837245/7578494
            #     ds[vname][:, -1, :, :] = ar_dec_mean
            # ##

            list_ds_det = []
            for i in range(ds.model.shape[0]):
                print(f'detrending {vname} model {i+1} / {ds.model.shape[0]}', flush=True)

                ds_det_temp = detrend_dataset.detrend_dataset(ds.isel(model=i).drop_vars('model'), list_var=[], use_anomaly=False)
                list_ds_det.append(ds_det_temp)

            # a variable for each region
            print(f'saving {vname}, {v+1} / {len(dict_src)}', flush=True)

            ds_det_aggr = xr.concat(list_ds_det, pd.Index(np.arange(ds.model.shape[0]), name='model'))
            dir_out = os.path.dirname(path_src)
            path_out = os.path.join(dir_out, path_src.split('/')[-1].replace('_2001', '-det_2001').replace('2001-2019', date_label))
            ds_det_aggr.to_netcdf(path_out)

if __name__ == '__main__':
    process_trendy_detrend_v9()



