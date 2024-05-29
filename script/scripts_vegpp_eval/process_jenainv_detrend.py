'''
- aggregate into a ncfile
- time slice (2001-2019)
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
from sys import argv

def process_jenainv_detrend():
    ## time - target period
    date_start = '2001-01-01'
    date_end = '2019-12-31'
    date_out = pd.date_range(date_start, date_end, freq='1MS')
    date_label = f"{date_start.split('-')[0]}-{date_end.split('-')[0]}"

    print(f'detrending for {date_label}', flush=True)

    ## fire emission - load and convert the unit from daily to yearly considering the leaf year
    path_gfed = '/Net/Groups/BGI/people/hlee/data/GFED/v4/C_Emissions.1997-2021.1deg.3dim.nc'
    fire = xr.open_dataset(path_gfed).sel(time=slice(date_start, date_end))  # gC m-2 month-1 at monthly steps
    ts_m = pd.date_range(date_start, date_end, freq='1M')
    days_in_month = ts_m.days_in_month.to_numpy()
    days_in_month_3d = np.tile(days_in_month, (180, 360, 1)).transpose(2, 0, 1)
    fire['C_Emissions'] = fire['C_Emissions'] / days_in_month_3d  # gC m-2 month-1 to gC m-2 day-1 at monthly steps

    ## jena inversion
    path_jeninv = '/Net/Groups/BGI/people/hlee/data/jenainv/s99oc_v2022_daily'
    path_in = os.path.join(path_jeninv, 'NEE.daily.2001-2019.1deg.3dim.global.nc')
    path_out = path_in.replace('.nc', '.det.nc').replace('2001-2019', date_label)

    print(f'loading Jena Inversion and aggreagting to monthly values', flush=True)
    ds = xr.open_dataset(path_in).sel(time=slice(date_start, date_end)).resample(time='1M').reduce(np.nanmean)  # still gC m-2 d-1 after meaning at monthly steps
    ds = ds.sortby('lat', ascending=False)
    ds = ds.sortby('lon', ascending=True)

    # remove fire emission from inversion nee
    ds_nf = ds.copy(deep=True)
    ds_nf['time'] = fire['time']
    vname_nee = 'NEE'
    ds_nf[vname_nee] = xr.where(np.isnan(fire['C_Emissions']), ds_nf[vname_nee], ds_nf[vname_nee] - fire['C_Emissions'])  # subtract fire emission

    # detrend
    print(f'detrending Jena Inversion', flush=True)
    ds_det = detrend_dataset(ds_orig=ds_nf, list_var=[vname_nee], use_anomaly=False)

    ds_det.NEE_det.mean(dim=['lat', 'lon']).plot()
    plt.savefig(f'/Net/Groups/BGI/people/hlee/jeninv_det_{date_label}.png')

    # save
    print(f'saving Jena Inversion', flush=True)
    ds_det.to_netcdf(path_out)
    
if __name__ == '__main__':
    process_jenainv_detrend()
