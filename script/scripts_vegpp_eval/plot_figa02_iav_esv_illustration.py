'''
'''
import os
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.dates import DateFormatter, MonthLocator
mpl.rcParams['mathtext.default'] = 'regular'

#%% load data
# path_expOutput = '/Net/Groups/BGI/people/hlee/sindbad/data/output/VEGPP2pool1519_studyArea_10k_RD4wtnanmean_1_20230712'

# ds_sin = xr.open_dataset(os.path.join(path_expOutput, 'VEGPP2pool1519_studyArea_10k_RD4wtnanmean_1_wTotal_3dim_fullPixel.nc'))
# ds_sin = ds_sin.resample(time='1M').reduce(np.nanmean)
# ds_sin = ds_sin.sortby('lat', ascending=False)
# ds_sin = ds_sin.sortby('lon', ascending=True)

# time_series = ds_sin['wTotal'].mean(dim=['lat', 'lon']).values
# time_series_jan = ds_sin['wTotal'].sel(time=ds_sin.time.dt.month==1).mean(dim=['lat', 'lon']).values
# time_series_may = ds_sin['wTotal'].sel(time=ds_sin.time.dt.month==7).mean(dim=['lat', 'lon']).values

## generate time series with seasonal cycle and linear trend

# Parameters
n_years = 19  # Number of years
points_per_year = 12  # Number of time points per year (e.g., monthly data)
total_points = n_years * points_per_year

# Time axis
time = np.arange(total_points) / points_per_year  # Time in years

# Dates for monthly identification
dates = pd.date_range(start="2000-01-01", periods=total_points, freq="M")
months = dates.month

# Define monthly weights for the trend (e.g., stronger slope in July)
monthly_trend_weights = {
    1: -0.4,  2: 0.2,  3: 0.3,  4: 0.4,  5: 0.7,  6: 0.6,
    7: 2.0,  8: 0.8,  9: 0.9, 10: 1.0, 11: 1.1, 12: 1.2
}


# Generate the trend component with varying slopes
trend_slope = 0.05  # Base slope
trend = np.array([monthly_trend_weights[m] for m in months]) * trend_slope * time

# Seasonal cycle (sinusoidal pattern)
amplitude = 1.0  # Amplitude of the seasonal cycle
seasonal_cycle = amplitude * np.sin(2 * np.pi * time)

# Random noise
np.random.seed(111)
noise_amplitude = 0.2  # Amplitude of the noise
noise = noise_amplitude * np.random.randn(total_points)

# Combine components
time_series = seasonal_cycle + trend + noise

# # Seasonal cycle (sinusoidal pattern)
# amplitude = 1.0  # Amplitude of the seasonal cycle
# seasonal_cycle = amplitude * np.sin(2 * np.pi * time)

# # Interannual trend (linear or nonlinear)
# trend_slope = 0.1  # Slope of the trend (linear increase per year)
# trend = trend_slope * time

# # Random noise
# noise_amplitude = 0.2  # Amplitude of the noise
# noise = noise_amplitude * np.random.randn(total_points)

# # Combine components
# time_series = seasonal_cycle + trend + noise

# time series of january and july
idx_mn_test = 4  # e.g., to use may as a test month-->4
idx_full = np.arange(n_years*points_per_year)
idx_jan = np.arange(0, 12*n_years, 12)
idx_may = np.arange(idx_mn_test, 12*n_years, 12)
time_series_jan = time_series[idx_jan]
time_series_may = time_series[idx_may]

# time_series = ds_sin['wTotal'].mean(dim=['lat', 'lon']).values
# time_series_jan = ds_sin['wTotal'].sel(time=ds_sin.time.dt.month==1).mean(dim=['lat', 'lon']).values
# time_series_may = ds_sin['wTotal'].sel(time=ds_sin.time.dt.month==7).mean(dim=['lat', 'lon']).values

#%% fit regression for two example months
dates = pd.date_range('2001-01-01', '2019-12-31', freq='1MS')
idx_jan = dates.month==1
idx_may = dates.month==(idx_mn_test+1)

# # time = ds_sin.time.values
# x_jan = np.array([np.argwhere(time==e)[0, 0] for e in time if e.astype('datetime64[D]').astype(object).month==1])
# x_may = np.array([np.argwhere(time==e)[0, 0] for e in time if e.astype('datetime64[D]').astype(object).month==7])
# x_time = np.array([e.astype('datetime64[Y]').astype(object).year for e in time])

idx_valid = ~np.isnan(time_series_jan)
fit_jan = np.polyfit(dates[idx_jan].values.astype('float'), time_series_jan[idx_valid], 1)
fit_jan_without_mean = np.polyfit(dates[idx_jan].values.astype('float'), time_series_jan[idx_valid]-np.nanmean(time_series_jan[idx_valid]), 1)
idx_valid = ~np.isnan(time_series_may)
fit_may = np.polyfit(dates[idx_may].values.astype('float'), time_series_may[idx_valid], 1)
fit_may_without_mean = np.polyfit(dates[idx_may].values.astype('float'), time_series_may[idx_valid]-np.nanmean(time_series_may[idx_valid]), 1)

tr_x_jan = np.linspace(dates[idx_jan].values.astype('float')[0], dates[idx_jan].values.astype('float')[-1], 100)
tr_jan = tr_x_jan * fit_jan[0] + fit_jan[1]
tr_jan_without_mean = tr_x_jan * fit_jan_without_mean[0] + fit_jan_without_mean[1]
tr_x_may = np.linspace(dates[idx_may].values.astype('float')[0], dates[idx_may].values.astype('float')[-1], 100)
tr_may = tr_x_may * fit_may[0] + fit_may[1]
tr_may_without_mean = tr_x_may * fit_may_without_mean[0] + fit_may_without_mean[1]

iav_glo_jan = time_series[idx_jan] - (dates[idx_jan].values.astype('float')*fit_jan[0]+fit_jan[1])
iav_glo_may = time_series[idx_may] - (dates[idx_may].values.astype('float')*fit_may[0]+fit_may[1])

esv_glo_jan_without_mean = np.nanmean(time_series[idx_jan] - (dates[idx_jan].values.astype('float')*fit_jan[0]+fit_jan[1]))
esv_glo_may_without_mean = np.nanmean(time_series[idx_may] - (dates[idx_may].values.astype('float')*fit_may[0]+fit_may[1]))

esv_glo_jan_with_mean = np.nanmean(time_series[idx_jan] - (dates[idx_jan].values.astype('float')*fit_jan_without_mean[0]+fit_jan_without_mean[1]))
esv_glo_may_with_mean = np.nanmean(time_series[idx_may] - (dates[idx_may].values.astype('float')*fit_may_without_mean[0]+fit_may_without_mean[1]))

# calc. esv for each month
ar_glo_iav_all_months = np.arange(total_points) * np.nan
ar_glo_esv_all_months_with_mean = np.arange(12) * np.nan
for mn in range(12):
    idx_mn = dates.month==(mn+1)

    time_series_mn = time_series[idx_mn]
    idx_valid = ~np.isnan(time_series_mn)

    # iav
    fit_mn = np.polyfit(dates[idx_mn].values.astype('float'), time_series_mn[idx_valid], 1)
    ar_glo_iav_all_months[idx_mn] = time_series_mn - (dates[idx_mn].values.astype('float')*fit_mn[0]+fit_mn[1])

    # esv
    fit_mn_without_mean = np.polyfit(dates[idx_mn].values.astype('float'), time_series_mn[idx_valid]-np.nanmean(time_series_mn[idx_valid]), 1)
    esv_with_mean = np.nanmean(time_series[idx_mn] - (dates[idx_mn].values.astype('float')*fit_mn_without_mean[0]+fit_mn_without_mean[1]))

    ar_glo_esv_all_months_with_mean[mn] = esv_with_mean

#%% plot
color_month1 = '#6412D0'
color_month2 = '#F58726'
size_marker = 5
alphabets = [chr(i) for i in range(ord('a'),ord('z')+1)]
alphabets_coords = (0.03, 1.0)
legend_handles = []
label_ax = ['Regression fit for each month', 'IAV', 'ESV']
size_scaler = 0.4
fig = plt.figure(figsize=(16*size_scaler, 9*size_scaler))
gs = fig.add_gridspec(nrows=2, ncols=2, figure=fig, width_ratios=[3, 1], height_ratios=[3, 1])

## raw and regression
ax = fig.add_subplot(gs[0, 0])
ax.plot(dates, time_series, color='grey')
l1, = ax.plot(dates[idx_jan], time_series[idx_jan], color=color_month1, marker='o', markersize=size_marker, linestyle='', label='January')
l2, = ax.plot(dates[idx_may], time_series[idx_may], color=color_month2, marker='o', markersize=size_marker, linestyle='', label='May')
l3, = ax.plot(pd.to_datetime(tr_x_jan), tr_jan, color=color_month1, label='Fitted line for January')
l4, = ax.plot(pd.to_datetime(tr_x_may), tr_may, color=color_month2, label='Fitted line for May')

legend_handles.append(l1)
legend_handles.append(l2)
legend_handles.append(l3)
legend_handles.append(l4)

ax.set_ylim(-1.5, 2)
ax.set_xlim(pd.to_datetime('2000-09-01'), pd.to_datetime('2021-04-01'))
ax.set_xlabel('Year')
ax.set_xticks(dates[np.arange(0, len(dates), 60)])
ax.set_xticklabels(dates[np.arange(0, len(dates), 60)])

ax.xaxis.set_major_formatter(DateFormatter('%Y'))
# ax.xaxis.set_major_locator(MonthLocator(interval=60))

## iav
ax = fig.add_subplot(gs[1, 0])

ax.plot(dates, ar_glo_iav_all_months, color='grey', linestyle='-', linewidth=1)

ax.plot(dates[idx_jan], iav_glo_jan, color=color_month1, marker='o', markersize=size_marker, linestyle='', label='Regression fit for January')
ax.plot(dates[idx_may], iav_glo_may, color=color_month2, marker='o', markersize=size_marker, linestyle='', label='Regression fit for July')
ax.axhline(y=0, linewidth=1, linestyle='dashed', color='black')

ax.set_ylim(-0.6, 0.6)
ax.set_xlim(pd.to_datetime('2000-09-01'), pd.to_datetime('2021-04-01'))
ax.set_xlabel('Year')
ax.set_xticks(dates[np.arange(0, len(dates), 60)])
ax.set_xticklabels(dates[np.arange(0, len(dates), 60)])

ax.xaxis.set_major_formatter(DateFormatter('%Y'))

## esv
ax = fig.add_subplot(gs[0, 1])

ax.plot(np.arange(1, 13), ar_glo_esv_all_months_with_mean, color='grey', marker='o', markersize=size_marker, linestyle='-')

# ar_esv_frame = np.arange(12) * np.nan
# ar_esv_frame[0] = esv_glo_jan_without_mean
# ax.plot(np.arange(1, 13), ar_esv_frame, color=color_month1, marker='x', markersize=size_marker, linestyle='')
# ar_esv_frame = np.arange(12) * np.nan
# ar_esv_frame[idx_mn_test] = esv_glo_may_without_mean
# ax.plot(np.arange(1, 13), ar_esv_frame, color=color_month2, marker='x', markersize=size_marker, linestyle='')

ar_esv_frame = np.arange(12) * np.nan
ar_esv_frame[0] = esv_glo_jan_with_mean
ax.plot(np.arange(1, 13), ar_esv_frame, color=color_month1, marker='o', markersize=size_marker, linestyle='')
ar_esv_frame = np.arange(12) * np.nan
ar_esv_frame[idx_mn_test] = esv_glo_may_with_mean
ax.plot(np.arange(1, 13), ar_esv_frame, color=color_month2, marker='o', markersize=size_marker, linestyle='')

ax.set_xlim(0, 13)
ax.set_ylim(-1.5, 2)

ax.set_xlabel('Months')
ax.set_xticks(np.arange(1, 13, 6))
ax.set_xticklabels(np.arange(1, 13, 6))

for i, ax in enumerate(fig.axes):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.annotate(f'({alphabets[i]}) {label_ax[i]}', xy=alphabets_coords, xycoords='axes fraction', weight='bold', fontsize=12)
    # ax.set_xticklabels('')

## legend
ax = fig.add_subplot(gs[1, 1])
ax.axis('off')
ax.legend(
    handles=legend_handles,  # [l1, l2, l3, l4],
    loc='upper center', bbox_to_anchor=(0.4, 0.9), bbox_transform=ax.transAxes,
    fontsize=7, fancybox=False, ncol=1, frameon=False
)

# fig.supxlabel('Year or Month', y=-0.05)
fig.supylabel('Variable', x=0.04, fontsize=12)

fig.subplots_adjust(hspace=0.5, wspace=0.15)
# plt.legend(loc='upper center', frameon=False)

# %% save
save_name = '/Net/Groups/BGI/people/hlee/hlee_vegpp_eval/figures_revision1/fig_iav_esv_illustration.png'

fig.savefig(
    save_name,
    dpi=600,
    bbox_inches='tight',
    facecolor='w',
    transparent=False
)

#%% x_i,mn - x_i,mn,fit




#%% mean(x_i,mn - x_i,mn,fit)

#%% mean(x_i,mn - x_i,mn,fit,w/o_mean)





import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Create a DataFrame for better visualization
dates = pd.date_range(start="2000-01-01", periods=total_points, freq="M")
df = pd.DataFrame({"Date": dates, "Value": time_series})

# Plot the time series
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Value"], label="Time Series", color="blue")
plt.plot(df["Date"], seasonal_cycle + trend, label="Signal (No Noise)", color="orange", linestyle="--")
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Time Series with Seasonal Cycle and Interannual Trend")
plt.legend()
plt.grid()
plt.show()
