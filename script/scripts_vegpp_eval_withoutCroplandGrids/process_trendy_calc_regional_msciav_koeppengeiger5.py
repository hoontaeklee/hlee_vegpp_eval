'''
- calculate global and regional (Koeppen-Geiger regions) MSC and IAV
- median and robust std. (i.e. std. * 1.25)
  --> calculate over global or regional n time series by n members
- desired results before calculating median and robust std.
ds of (model, time) for global median and unc.
ds of (model, region, time) for regional median and unc.

gpp: 2002-2015
nee: 2001-2017 (2015-2017 looks too short, 2002-2015 or 2017 looks not too different from 2001-2017, esp. for MSC)

2002-2015: GPP, RECO
2015-2019: NEE_OCO2
2001-2019: NEE_JENI

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
from apply_koeppengeiger5_mask import apply_koeppengeiger5_mask
import matplotlib.pyplot as plt
from sys import argv

def process_trendy_calc_regional_msciav_koeppengeiger5():
    path_trendy = os.path.join(path_bgi, f'people/hlee/data/trendy/v9/regridded_1deg')

    dict_period = {
        'date_start': ['2001-01-01', '2002-01-01', '2015-01-01'],
        'date_end': ['2019-12-31', '2015-12-31', '2019-12-31']
    }

    # study area mask
    path_common_nonnan = os.path.join(path_bgi, 'people/hlee/to_others/sindbad_h2m_for_2001_2019/common_nonnan_pixels.nc')
    ds_studyarea_mask = xr.open_dataset(path_common_nonnan)
    ds_studyarea_mask = ds_studyarea_mask.sortby('lat', ascending=False)
    ds_studyarea_mask = ds_studyarea_mask.sortby('lon', ascending=True)
    da_studyarea_mask = ds_studyarea_mask.common_nonnan_pixels
    ar_studyarea_mask = da_studyarea_mask.values

    # load cropland mask
    path_lc = '/Net/Groups/BGI/people/hlee/data/MODIS/MCD12Q1_006_IGBP_fraction.360.180.mean.2001-2019.nc'
    ds_lc = xr.open_dataset(path_lc).isel(classes=11)  # class 11: cropland
    frac_threshold = 0.5
    ds_lc.landcover.data = np.where(ds_lc.landcover.data>frac_threshold, True, False)
    ar_lc = ds_lc['landcover'].data
    period_lc = path_lc.split('/')[-1].split('.')[-2]

    # grid area
    area = np.load(os.path.join(path_bgi, 'people/hlee/data/gridAreaAndLandFraction/gridAreaInKm2_180_360.npz'))['area']
    ar_area_msk = np.where(ar_studyarea_mask, area, np.nan)
    ar_area_msk = np.where(ar_lc, np.nan, ar_area_msk)

    for p in range(len(dict_period['date_start'])):
        date_start = dict_period['date_start'][p]
        date_end = dict_period['date_end'][p]
        date_label = date_start.split('-')[0]+'-'+date_end.split('-')[0]
        nmonths = len(pd.date_range(date_start, date_end, freq='1M'))
    
        print(f'calculating for the period {date_label}', flush=True)

        dict_src = {
            'gpp': {
                'path': os.path.join(path_trendy, f'trendyv9_S2_gpp-det_{date_label}.nc'),
                'period': (date_start, date_end)
            },
            'nee': {
                'path': os.path.join(path_trendy, f'trendyv9_S2_nee-det_{date_label}.nc'),
                'period': (date_start, date_end)
            },
            'reco': {
                'path': os.path.join(path_trendy, f'trendyv9_S2_reco-det_{date_label}.nc'),
                'period': (date_start, date_end)
            }
        }

        for v in range(len(dict_src)):
            _vname = list(dict_src.keys())[v]
            print(f'reading {_vname}, {v+1} / {len(dict_src)}', flush=True)

            path_src = dict_src[_vname]['path']
            ds_temp = xr.open_dataset(path_src).sel(time=slice(date_start, date_end))
            ds_temp = ds_temp.sortby('lat', ascending=False)
            ds_temp = ds_temp.sortby('lon', ascending=True)
            # ds_temp['lon'] = ds_temp['lon'] + 0.5  # to align with the lon coords to the center of each grid
            ds_temp = ds_temp.where(ar_studyarea_mask)
            ds_temp = ds_temp.where(~ds_lc.landcover)
            nmembers = len(ds_temp.model)

            # calculate land-area-weighted regional time series
            nregions = 5
            ds_rm = ds_temp[[f'{_vname}_msc', f'{_vname}_det']].mean(dim=['lat', 'lon']).copy(deep=True)
            ds_rm = ds_rm.expand_dims(dim={'region': np.arange(1, nregions+1, 1)}, axis=1)
            ar_bowl_msc = np.ones_like(ds_rm[f'{_vname}_msc'].data) * np.nan
            ar_bowl_iav = np.ones_like(ds_rm[f'{_vname}_det'].data) * np.nan
            for m in range(nmembers):
                print(f'processing model {m+1} / {nmembers}', flush=True)
                
                # aggregate to regions
                ds_rm_temp = apply_koeppengeiger5_mask(
                    dsin=ds_temp[[f'{_vname}_msc', f'{_vname}_det']].isel(model=m).drop_vars('model'),
                    pathlf='ones',
                    path_rm=os.path.join(path_bgi, 'people/hlee/data/koeppengeiger/kg5_regions_oneDeg.nc'),
                    p_truncate=1.0,  # for each time step, exclude 0.5 p from each tail, before calculating regional mean
                    tosave=False,
                    toplot=False)
                ar_bowl_msc[m, :, :] = ds_rm_temp[f'{_vname}_msc'][f'{_vname}_msc'].values
                ar_bowl_iav[m, :, :] = ds_rm_temp[f'{_vname}_det'][f'{_vname}_det'].values

            ds_rm[f'{_vname}_msc'].data = ar_bowl_msc
            ds_rm[f'{_vname}_det'].data = ar_bowl_iav
            ds_rm['legends'] = ds_rm_temp[f'{_vname}_msc'].legends

            # save
            file_out = os.path.join(path_trendy, f'trendyv9_S2_{_vname}-regionalMSCIAV_{date_label}_koeppengeiger5_withoutCroplandGrids.nc')
            ds_rm.to_netcdf(file_out)

            print(f'saved: {file_out}', flush=True)

if __name__ == '__main__':
    process_trendy_calc_regional_msciav_koeppengeiger5()