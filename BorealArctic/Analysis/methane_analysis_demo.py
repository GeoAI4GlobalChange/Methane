from netCDF4 import Dataset
import copy
from methane_analysis_toolbox import FCH4_anomaly_calculation,Forcing_anomaly_calculation,Par_cor,LinearReg_model,ForcingData_load,FCH4Data_load,FCH4_anomaly_yearly,Forcing_anomaly_yearly

def grid_variation_analysis(max_lat_idx,dir_forcing,dir_fch4,save_file,forcing_file_list,forcing_vars,target_var):
    #load FCH4 upscaled data (start from the year of 2002)
    fch4_var,lats,lons=FCH4Data_load(dir_fch4,max_lat_idx)
    #obtain FCH4 anomaly
    fch4_var_anomaly=FCH4_anomaly_calculation(fch4_var,lats,lons)
    #obatain datsets of input forcing (TS, TA, SC, PA, P, SWC, WS, and GPP)
    all_forcing_vars=ForcingData_load(forcing_file_list,forcing_vars,dir_forcing,max_lat_idx,lats,lons)
    #obtain forcing anomaly
    all_forcing_vars_anomaly=Forcing_anomaly_calculation(all_forcing_vars,lats,lons)

    #update forcing_var and all_vars
    forcing_vars.extend(['gpp'])
    all_vars=copy.deepcopy(forcing_vars)
    all_vars.extend(['fch4'])

    #calculate partial correlation between FCH4 and its drivers in each grid
    partial_result=Par_cor(forcing_vars,lats,lons,all_forcing_vars_anomaly,fch4_var_anomaly,all_vars,target_var)

    #obtain and save the results
    nc_fid2 = Dataset(save_file, 'w', format="NETCDF4")
    nc_fid2.createDimension('latitude', len(lats))
    nc_fid2.createDimension('longitude', len(lons))
    nc_fid2.createDimension('forcing_id', partial_result.shape[0])
    latitudes = nc_fid2.createVariable('latitude', 'f4', ('latitude',))
    longitudes = nc_fid2.createVariable('longitude', 'f4', ('longitude',))
    forcing_var_id = nc_fid2.createVariable("forcing_id", "f4", ("forcing_id",))
    par_cor_results = nc_fid2.createVariable('par_r', "f4", ("forcing_id","latitude", "longitude",), zlib=True)
    forcing_var_id[:] = range(partial_result.shape[0])
    forcing_var_id.units='_'.join(forcing_vars)
    latitudes[:] = lats[:]
    longitudes[:] = lons[:]
    par_cor_results[:] = partial_result[:]
    nc_fid2.close()


def grid_trend_analysis(forcing_group,max_lat_idx,dir_forcing,dir_fch4,forcing_file_list,forcing_vars):
    # load FCH4 upscaled data
    fch4_var, lats, lons = FCH4Data_load(dir_fch4, max_lat_idx)
    #calculate yearly FCH4 anomaly
    fch4_var_anomaly = FCH4_anomaly_yearly(fch4_var, lats, lons)
    # obatain datsets of input forcing (TS, TA, SC, PA, P, SWC, WS, and GPP)
    all_forcing_vars = ForcingData_load(forcing_file_list, forcing_vars, dir_forcing, max_lat_idx, lats, lons)
    # obtain forcing anomaly
    all_forcing_vars_anomaly = Forcing_anomaly_yearly(all_forcing_vars, lats, lons)

    #update forcing_vars and all_vars
    forcing_vars.extend(['gpp'])
    all_vars = copy.deepcopy(forcing_vars)
    all_vars.extend(['fch4'])

    #Quantify the controls from drivers on the trend of FCH4 using a statistical linear regression model
    corr_result,prediction_result=LinearReg_model(forcing_vars,lats,lons,fch4_var_anomaly,all_forcing_vars_anomaly,all_vars,forcing_group)

    #obtain and save the results
    save_file = f'./results/FCH4_trend_explanation_{forcing_group}.nc'  # path to save the results
    nc_fid2 = Dataset(save_file, 'w', format="NETCDF4")
    nc_fid2.createDimension('latitude', len(lats))
    nc_fid2.createDimension('longitude', len(lons))
    nc_fid2.createDimension('forcing_id', corr_result.shape[0])
    nc_fid2.createDimension('time', prediction_result.shape[0])
    latitudes = nc_fid2.createVariable('latitude', 'f4', ('latitude',))
    longitudes = nc_fid2.createVariable('longitude', 'f4', ('longitude',))
    wetland_type_var = nc_fid2.createVariable("forcing_id", "f4", ("forcing_id",))
    time_var = nc_fid2.createVariable("time", "f4", ("time",))
    corr_var = nc_fid2.createVariable('corr', "f4", ("forcing_id", "latitude", "longitude",), zlib=True)
    Prediction_var = nc_fid2.createVariable('Prediction', "f4", ("time", "latitude", "longitude",), zlib=True)
    wetland_type_var[:] = range(corr_result.shape[0])
    wetland_type_var.units = '_'.join(forcing_vars)
    latitudes[:] = lats[:]
    longitudes[:] = lons[:]
    corr_var[:] = corr_result[:]
    time_var[:]=range(prediction_result.shape[0])
    Prediction_var[:]=prediction_result[:]
    nc_fid2.close()

if __name__ == '__main__':
    max_lat_idx = 90
    target_var = 'fch4'
    forcing_group = 'temperature'  # select variable groups that are included for fch4 prediction. Three options: 'temperature','gpp','water'
    dir_forcing = './data/forcing/'  # path of input forcing datasets
    dir_fch4_1 = './results/wetland_FCH4_2002-2021_upscale_result1.nc'  # path of upscaled methane dataset
    dir_fch4_2 = './results/wetland_FCH4_2002-2021_upscale_result2.nc'  # path of upscaled methane dataset
    save_file = f'./results/wetland_FCH4_partial_corr.nc'  # path to save the partial correlation results

    forcing_file_list = ['soil_temperature_level_1', '2m_temperature', 'snow_cover', 'surface_pressure',
                         'total_precipitation', 'volumetric_soil_water_layer_1',
                         'windspeed']  # the file names of input forcing
    forcing_vars = ['stl1', 't2m', 'snowc', 'sp',
                    'tp', 'swvl1', 'ws']  # the variable names of the input forcing

    #Calculate partial correlation between FCH4 and its drivers in each grid cell
    grid_variation_analysis(max_lat_idx,dir_forcing,dir_fch4_1,save_file,forcing_file_list,forcing_vars,target_var)
    #Calculate CH4 driven by different variables using statistical linear regression models
    grid_trend_analysis(forcing_group, max_lat_idx, dir_forcing, dir_fch4_2, forcing_file_list, forcing_vars)
