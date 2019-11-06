%matplotlib inline
import xarray as xr
import glob
import os

path = "/scratch/globc/dcom/ARPEGE6_TUNE"

da_model=xr.open_mfdataset(path+'/PRE623TUN010*.nc',concat_dim='ensemble')
da_model_concat = xr.concat(da_model, dim='ensemble')
