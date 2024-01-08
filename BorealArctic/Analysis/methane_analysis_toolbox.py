import pandas as pd
import numpy as np
from netCDF4 import Dataset
from scipy import signal
import pingouin as pg
import statsmodels.api as sm
import copy
def FCH4Data_load(dir_fch4,max_lat_idx):#load upscaled FCH4 dataset
    nc = Dataset(dir_fch4, 'r')
    fch4_var = nc.variables['FCH4'][-1, 52:, :max_lat_idx].astype(np.float32)
    lats = nc.variables['latitude'][:max_lat_idx]
    lons = nc.variables['longitude'][:]
    nc.close()
    return fch4_var,lats,lons

def FCH4_anomaly_calculation(fch4_var,lats,lons):#calculate FCH4 anomaly
    fch4_var = fch4_var.reshape(-1, 52, len(lats), len(lons))
    fch4_var_mean = np.nanmean(fch4_var, axis=0, keepdims=True)
    fch4_var_anomaly = fch4_var - fch4_var_mean
    fch4_var_anomaly = fch4_var_anomaly.reshape(-1, len(lats), len(lons))
    return fch4_var_anomaly

def FCH4_anomaly_yearly(fch4_var,lats,lons):#calculate FCH4 anomaly in yearly scale
    fch4_var = fch4_var.reshape(-1, 52, len(lats), len(lons))
    fch4_var = np.nanmean(fch4_var, axis=1)
    fch4_var_mean = np.nanmean(fch4_var, axis=0, keepdims=True)
    fch4_var_anomaly = fch4_var - fch4_var_mean
    return fch4_var_anomaly

def ForcingData_load(forcing_file_list,forcing_vars,dir_forcing,max_lat_idx,lats,lons):#obtain forcing datasets including TS, TA, SC, PA, P, SWC, WS, and GPP
    start = True
    for file in forcing_file_list:# load each nc file of different input forcing
        forcing_var = forcing_vars[forcing_file_list.index(file)]
        nc_file = dir_forcing + f'ecmwf_weekly_0.5degree_unified_2000-2021_{file}.nc'
        nc = Dataset(nc_file, 'r')
        temp_var = nc.variables[forcing_var][52 * 2:, :max_lat_idx].astype(np.float32)  # (start from the year of 2002)
        nc.close()
        temp_var = temp_var[np.newaxis]
        if start:
            all_forcing_vars = temp_var
            start = False
        else:
            all_forcing_vars = np.concatenate((all_forcing_vars, temp_var), axis=0)
        print('loading', file)
    # obatain datsets of GPP
    nc_file = dir_forcing + f'GOSIF_GPP_05degree_weekly_2001-2021.nc'
    nc = Dataset(nc_file, 'r')
    temp_var = nc.variables['GPP'][52:, :max_lat_idx].astype(np.float32)# (start from the year of 2002)
    nc.close()
    temp_var = temp_var[np.newaxis]
    # combine GPP with other variables
    all_forcing_vars = np.concatenate((all_forcing_vars, temp_var), axis=0)
    all_forcing_vars = all_forcing_vars.reshape(all_forcing_vars.shape[0], -1, 52, len(lats), len(lons))
    return all_forcing_vars

def Forcing_anomaly_calculation(all_forcing_vars,lats,lons):#calculate forcing anomaly, including TS, TA, SC, PA, P, SWC, WS, and GPP
    all_forcing_vars_mean = np.nanmean(all_forcing_vars, axis=1, keepdims=True)
    all_forcing_vars_anomaly = all_forcing_vars - all_forcing_vars_mean
    all_forcing_vars_anomaly = all_forcing_vars_anomaly.reshape(all_forcing_vars.shape[0], -1, len(lats), len(lons))
    return all_forcing_vars_anomaly

def Forcing_anomaly_yearly(all_forcing_vars,lats,lons):#calculate forcing anomaly in yearly scale, including TS, TA, SC, PA, P, SWC, WS, and GPP
    all_forcing_vars = np.nanmean(all_forcing_vars, axis=2)
    all_forcing_vars_mean = np.nanmean(all_forcing_vars, axis=1, keepdims=True)
    all_forcing_vars_anomaly = all_forcing_vars - all_forcing_vars_mean
    return all_forcing_vars_anomaly

def Par_cor(forcing_vars,lats,lons,all_forcing_vars_anomaly,fch4_var_anomaly,all_vars,target_var): #calculate partial correlation between FCH4 and its drivers
    partial_result = np.full((len(forcing_vars), len(lats), len(lons)), np.nan)
    for lat_idx in range(len(lats)):
        for lon_idx in range(len(lons)):
            print(lat_idx, lon_idx)
            if np.sum(np.isnan(all_forcing_vars_anomaly[:, :, lat_idx, lon_idx].reshape(-1))) == 0:
                if np.sum(np.isnan(fch4_var_anomaly[:, lat_idx, lon_idx])) == 0:
                    input_series = all_forcing_vars_anomaly[:, :, lat_idx, lon_idx]
                    target_series = fch4_var_anomaly[:, lat_idx, lon_idx]
                    target_series = target_series[np.newaxis]
                    data = np.concatenate((input_series, target_series), axis=0)
                    data = data.transpose((1, 0))
                    data = signal.detrend(data, axis=0)
                    if np.sum(np.isnan(data.reshape(-1))) == 0:
                        df = pd.DataFrame(data, columns=all_vars)
                        for temp_var in forcing_vars:
                            source_var = temp_var
                            condition_vars = list(set(forcing_vars).difference(set([source_var])))
                            res = pg.partial_corr(data=df, x=source_var, y=target_var, covar=condition_vars).round(3)
                            r = res[['r', 'p-val']].values
                            r, p = r[0, 0], r[0, 1]
                            partial_result[forcing_vars.index(temp_var), lat_idx, lon_idx] = r if p < 0.05 else np.nan
    return partial_result

def LinearReg_model(forcing_vars,lats,lons,fch4_var_anomaly,all_forcing_vars_anomaly,all_vars,forcing_group):#build linear regression models to calculate FCH4 driven by different variables in each grid cell
    corr_result = np.full((len(forcing_vars), len(lats), len(lons)), np.nan)
    prediction_result = np.full((fch4_var_anomaly.shape[0], len(lats), len(lons)), np.nan)
    # calculate the predicted FCH4 in each grid
    for lat_idx in range(len(lats)):
        for lon_idx in range(len(lons)):
            print(lat_idx, lon_idx)
            if np.sum(np.isnan(all_forcing_vars_anomaly[:, :, lat_idx, lon_idx].reshape(-1))) == 0:
                input_series = all_forcing_vars_anomaly[:, :, lat_idx, lon_idx]
                target_series = fch4_var_anomaly[:, lat_idx, lon_idx]
                target_series = target_series[np.newaxis]
                data = np.concatenate((input_series, target_series), axis=0)
                data = data.transpose((1, 0))
                if np.sum(np.isnan(data.reshape(-1))) == 0:
                    df = pd.DataFrame(data, columns=all_vars)
                    input_cols = all_vars[:-1]
                    x = df[input_cols]
                    y = df['fch4']
                    x = sm.add_constant(x)
                    model = sm.OLS(y, x).fit()
                    if forcing_group == 'temperature':
                        input_cols = ['stl1', 't2m']
                    elif forcing_group == 'gpp':
                        input_cols = ['gpp']
                    else:
                        input_cols = ['tp', 'swvl1']
                    df_temp = copy.deepcopy(x)
                    first_row_value = df_temp[input_cols].values[0]
                    first_row_value = first_row_value[np.newaxis]
                    first_row_value = np.repeat(first_row_value, len(df_temp), axis=0)
                    df_temp[input_cols] = first_row_value
                    predictions = model.predict(df_temp)
                    f_pvalue = model.f_pvalue
                    if f_pvalue < 0.1:
                        prediction_result[:, lat_idx, lon_idx] = predictions
                    for i in input_cols:
                        p = model.pvalues[i]
                        r = model.params[i]
                        corr_result[input_cols.index(i), lat_idx, lon_idx] = r if p < 0.1 else np.nan
    return corr_result,prediction_result
