# Upscaling and analyzing wetland methane dynamics in the boreal arctic area
The repository contains exemplary studies and demo codes of upscaling and analyzing wetland methane emissions in the boreal arctic area. In this study, we proposed a causality guided machine learning model that incorporated the hysteretic relationships between CH4 emission and its drivers (including soil temperature, air temperature, gross primary productivity, air pressure, precipitation, wind speed, snow cover, and soil water content) to generate an upscaled data product of boreal arctic wetland CH4 emission from 2002 to 2021. We then identified dominant controls on wetland CH4 emission variability and trends using partial correlation method and a statistical linear regression model respectively.  
## Code introduction
1)	Causality-guided machine learning model  
In the”model_training_demo.py” file, run the “model_training” function.  
2) Identifying dominant controls on wetland CH4 emission variability  
In the”methane_analysis_demo.py” file, run the “grid_variation_analysis” function.  
3) Quantifying dominant controls on wetland CH4 emission trend  
In the”methane_analysis_demo.py” file, run the “grid_trend_analysis” function.  

## Data availability
The original eddy covariance methane flux datasets are from FLUXNET-CH4, available at https://fluxnet.org/data/fluxnet-ch4-community-product/. The chamber datasets are from the BAWLD-CH4 dataset (https://doi.org/10.5194/essd-13-5151-2021), and the dataset in Bao’s study (https://doi.org/10.1021/acs.est.1c01616).  
For the input drivers, GPP was obtained from the GOSIF dataset, which is available at https://globalecology.unh.edu/data/GOSIF-GPP.html. Other variables (soil temperature, air temperature, air pressure, precipitation, wind speed, snow cover, and soil water content) were obtained from ERA5-land datasets, which are available at (https://www.ecmwf.int/en/forecasts/datasets/reanalysis-datasets/era5).  
For more datasets used for uncertainty analysis, please see the correponding paper.  
## References
Details of the research will be seen in the upcoming paper “Two decades of boreal arctic wetland methane emissions modulated by warming and vegetation activity” (The manuscript will be online available very soon). If you have any questions about the code or dataset, please contact yuankunxiaojia@gmail.com or lifa.lbnl@gmail.com.

