# 
#
###############################################################
#	File to define functions
###############################################################
#
# Import MODULES 
#
# Computational modules 
import xarray as xr
import glob
import os
import numpy as np
import netCDF4
from netCDF4 import Dataset
import pandas as pd
import re
from array import array
from pylab import *
#
# Plotting modules 
import matplotlib.pyplot as plt
#from mpl_toolkits.basemap import Basemap
import pandas.plotting
import matplotlib.ticker as ticker
import seaborn as sns
#
#...................................................................
# GLOBAL ANNUAL MEAN WEIGHTED WITH COS(LAT)

import xarray as xr
import numpy as np
import pandas as pd


def wavg(df, start_yr, variables):
    # First, we need to convert latitude to radians and the time into year
    df1=df.reset_index(level=['time', 'lat'])
    df1['latr'] = np.deg2rad(df1['lat']) # lat in radians 
    df1['year']=pd.DatetimeIndex(df1['time']).year # time in years 


    # Then, we find the zonal mean by averaging along the latitude circles
    df2=df1.groupby(['year', 'lat']).mean()

    # Finally, we use the cosine of the converted latitudes as weights for the average
    df2['weights'] = np.cos(df2['latr'])
    result=pd.DataFrame()
    df2_tmp=df2.groupby(['year']).mean()
    result[variables]=df2_tmp[variables]
    N=len(result)
    YR=start_yr
    i=0
    while i < N:
        yr=YR+i
        tmp=df2.loc[yr]
        n=len(variables)
        j=0
        while j < n:
            t = (tmp[variables[j]]*tmp['weights']).sum()/tmp['weights'].sum()
            result[variables[j]].iloc[i]=t
            j = j + 1
        i = i + 1
        
    return result
#
#............................................................................
# Function to convert the monthly values into yearly global weighted mean, to compute the radiative budget and 
# to put them into a dataframe adapted for the plotting part

import xarray as xr
import numpy as np
import pandas as pd

def get_wavg_budget_df(path, filename, variables, start_yr ,drop, year_list):
#    “”"
#    This funciton read the netCDF file of monthly data, compute the radiative budget, perform a yearly mean and 
#    return a dataframe
#    “”"
    # First step : download the data into dataframe
    file = xr.open_mfdataset(path+filename,combine='by_coords')
    df=file[variables].to_dataframe().drop('height',axis=1)
    #
    # Second step : compute the annual global average weighted by cos(lat)
    df2=wavg(df, start_yr, variables)
    #
    # Compute radiative budget 
    df2['F']=df2['rsdt']
    df2['H']=df2['rsut']+df2['rlut']
    df2['N']=df2['F']-df2['H']
    #
    # reshape
    if drop == True:
        df3=df2.drop(year_list).reset_index(level=['year']).drop(axis=1,columns='year')
    else:
        df3=df2.reset_index(level=['year']).drop(axis=1,columns='year')
    #
    return df3
#
#...............................................................................
# PLOT A LINE GRAPH FROM DIFFERENT DATAFRAMES 
import matplotlib.pyplot as plt

def plotlines_Xdf(df, y, title, colors, linewidth, xlabel, xmin, xmax, ymin, ymax, legend):
    ax_model=plt.gca()
    N=len(df)
    i=0
    while i < N:
        df[i].plot(y=y,kind='line',title=title,legend=True, color=colors[i],linewidth=linewidth[i],ax=ax_model)
        i = (i+1)
        
    plt.xlabel(xlabel)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    ax_model.legend(legend, loc='center left', bbox_to_anchor=(1, 0.5))
#
#...................................................................................
# PLOT A LINE GRAPH FROM A UNIQUE DATA FRAME 
import matplotlib.pyplot as plt

def plotlines_1df(df, y , xmin, xmax, title, colors, xlabel, legend ):
    ax_model=plt.gca()
    N=len(y)
    i=0
    while i<N:
            df.plot(y=y[i],kind='line',title=title,legend=True, color=colors[i],ax=ax_model)
            i = (i+1)
            
    plt.xlim(xmin, xmax)
    plt.xlabel(xlabel)
    ax_model.legend(legend, loc='center left', bbox_to_anchor=(1, 0.5))
#
#.....................................................................................
# COMPUTE Delta(TOA), Delta(tas), Lambda

def Deltas_Lambda(result, df_CTL, df, expe_name, n):
    i=1

    Lbda=[]
    DN=[]
    Dtas=[]
    N = (n+1)

    while i<N:
    
        df_CTL_tmp=df_CTL.iloc[0:i,:]
        df_tmp=df.iloc[0:i,:]
        Delta_N=(df_tmp['N']-df_CTL_tmp['N']).mean()
        Delta_tas=(df_tmp['tas']-df_CTL_tmp['tas']).mean()
        Lambda=Delta_N/Delta_tas
        Lbda.append(Lambda)
        DN.append(Delta_N)
        Dtas.append(Delta_tas)
    
        i=i+1
 
    result['Delta_N_'+expe_name]=DN
    result['Delta_tas_'+expe_name]=Dtas
    result['Lambda_'+expe_name]=Lbda

    return result
#......................................................................................
#
