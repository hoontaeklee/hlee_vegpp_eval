'''
- calculate global and regional (TRANSCOM regions) MSC and IAV
- median and robust std. (i.e. std. * 1.25)
  --> calculate over global or regional n time series by n members
- desired results before calculating median and robust std.
ds of (model, time) for global median and unc.
ds of (model, region, time) for regional median and unc.

gpp: 2002-2015
nee: 2001-2017 (2015-2017 looks too short, 2002-2015 or 2017 looks not too different from 2001-2017, esp. for MSC)

2002-2015: GPP, RECO
2015-2019: NEE_OCO2

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
from truncate_array import truncate_array as truncate_array 
from apply_transcom_mask import apply_transcom_mask
import matplotlib.pyplot as plt
from sys import argv

def process_jenainv_calc_regional_msciav_transcom():
    path_jeni = os.path.join(path_bgi, f'people/hlee/data/jenainv/s99oc_v2022_daily/NEE.monthly.2001-2019.1deg.3dim.global.det.nc')
    date_start = '2001-01-01'
    date_end = '2019-12-31'
    date_label = date_start.split('-')[0]+'-'+date_end.split('-')[0]
    nmonths = len(pd.date_range(date_start, date_end, freq='1M'))
    
    # study area mask
    path_common_nonnan = os.path.join(path_bgi, 'people/hlee/to_others/sindbad_h2m_for_2001_2019/common_nonnan_pixels.nc')
    ds_studyarea_mask = xr.open_dataset(path_common_nonnan)
    ds_studyarea_mask = ds_studyarea_mask.sortby('lat', ascending=False)
    ds_studyarea_mask = ds_studyarea_mask.sortby('lon', ascending=True)
    da_studyarea_mask = ds_studyarea_mask.common_nonnan_pixels
    ar_studyarea_mask = da_studyarea_mask.values

    # grid area
    area = np.load(os.path.join(path_bgi, 'people/hlee/data/gridAreaAndLandFraction/gridAreaInKm2_180_360.npz'))['area']
    ar_area_msk = np.where(ar_studyarea_mask, area, np.nan)

    _vname = 'NEE'
    ds_temp = xr.open_dataset(path_jeni).sel(time=slice(date_start, date_end))
    ds_temp = ds_temp.sortby('lat', ascending=False)
    ds_temp = ds_temp.sortby('lon', ascending=True)
    ds_temp = ds_temp.where(ar_studyarea_mask)

    # calculate land-area-weighted regional time series
    ds_tc = ds_temp[[f'{_vname}_msc', f'{_vname}_det']].mean(dim=['lat', 'lon']).copy(deep=True)
    ds_tc = ds_tc.expand_dims(dim={'region': np.arange(1, 12, 1)}, axis=0)
    ar_bowl_msc = np.ones_like(ds_tc[f'{_vname}_msc'].data) * np.nan
    ar_bowl_iav = np.ones_like(ds_tc[f'{_vname}_det'].data) * np.nan
        
    # aggregate to regions
    ds_tc_temp = apply_transcom_mask(
        dsin=ds_temp[[f'{_vname}_msc', f'{_vname}_det']],
        pathlf='ones',
        path_tc=os.path.join(path_bgi, 'people/hlee/data/transcom/TranscomRegions.360.180.nc'),
        p_truncate=1.0,  # for each time step, exclude 0.5 p from each tail, before calculating regional mean
        tosave=False,
        toplot=False)
    ar_bowl_msc = ds_tc_temp[f'{_vname}_msc'][f'{_vname}_msc'].values
    ar_bowl_iav = ds_tc_temp[f'{_vname}_det'][f'{_vname}_det'].values

    ds_tc[f'{_vname}_msc'].data = ar_bowl_msc
    ds_tc[f'{_vname}_det'].data = ar_bowl_iav
    ds_tc['legends'] = ds_tc_temp[f'{_vname}_msc'].legends

    # save
    file_out = os.path.join(os.path.dirname(path_jeni), f's99oc_v2022_{_vname}_regionalMSCIAV_{date_label}_transcom.nc')
    ds_tc.to_netcdf(file_out)

    print(f'saved: {file_out}', flush=True)

if __name__ == '__main__':
    process_jenainv_calc_regional_msciav_transcom()