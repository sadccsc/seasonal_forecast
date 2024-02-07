#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
#import cartopy.crs as crs
#import cartopy.feature as cfeature
import requests
import os,sys
from mpl_toolkits.basemap import Basemap, cm, shiftgrid, addcyclic
import glob
import statsmodels.api as sm
import pymannkendall as mk
#import xskillscore


def get_linear(_y, what="slope"):
    # receives 
    # data need to be regularly spaced
    # returns slope in units of _y per unit of _x, intercept in units of _y or pvalue
    # pvalue is analytical, perhaps one day I will implement bootstrap 
    #
    # need to add constant for OLS does not include intercept by default 
    if not np.ma.is_masked(_y):
        _x = sm.add_constant(np.arange(len(_y)))
        res=sm.OLS(_y, _x, missing='none').fit()
        #res=sm.OLS(_y, _x, missing='drop').fit()
        if what=="slope":
            return res.params[1]
        elif what=="pval":
            return res.pvalues[1]    # results from the t_test, checking if the data follows a 
        elif what=="intercept":
            return res.params[0]
    else:
        return np.nan
    
def get_linear_mk(_y, what="slope"):

    try:
        res = mk.original_test(_y)
        if what=="slope":
            return res.slope
        elif what=="pval":
            return res.p    # results from the t_test, checking if the data follows a 
        elif what=="intercept":
            return res.params

    except:
        return np.nan

def plot_functions(evaluation_metric, var, reference_dataset, target_dataset, start_year, end_year):

    if evaluation_metric == 'bias':
        results = reference_dataset[var].sel(time=slice(start_year,end_year)).mean('time') - target_dataset[var].sel(time=slice(start_year,end_year)).mean('time')
        colorbar_label = 'Precipitation Bias'
        cmap = 'RdBu'
        levs = np.arange(-350, 400, 100)
        plotted_start_year = target_dataset[var].sel(time=slice(start_year,end_year)).time.min().dt.year.values
        plotted_end_year = target_dataset[var].sel(time=slice(start_year,end_year)).time.max().dt.year.values
        
    if evaluation_metric == 'climatology':
        results = target_dataset[var].sel(time=slice(start_year,end_year)).mean('time')
        colorbar_label = 'Total Precipitation (mm)'
        cmap = 'Blues'
        levs = np.arange( 100, 1000, 100)
        plotted_start_year = target_dataset[var].sel(time=slice(start_year,end_year)).time.min().dt.year.values
        plotted_end_year = target_dataset[var].sel(time=slice(start_year,end_year)).time.max().dt.year.values
        
    if evaluation_metric == 'nrmse':
        results = xskillscore.rmse(reference_dataset[var].resample(time="1Y").mean(),
                                         target_dataset[var].resample(time="1Y").mean(),dim='time', skipna=True)
        results = results/(target_dataset[var].max('time') - target_dataset[var].min('time'))
        colorbar_label = 'Normalised RMSE'
        cmap = 'coolwarm'
        levs = np.arange( 0.2, 1.5, 0.1)
        plotted_start_year = target_dataset[var].sel(time=slice(start_year,end_year)).time.min().dt.year.values
        plotted_end_year = target_dataset[var].sel(time=slice(start_year,end_year)).time.max().dt.year.values
        
    if evaluation_metric == 'temporal_correlation':
        results = xr.corr(reference_dataset[var].resample(time="1Y").mean(),
                                target_dataset[var].resample(time="1Y").mean(),dim='time')
        colorbar_label = 'Correlation - Coefficient'
        cmap = 'RdBu_r'
        levs = np.arange(-.7, .9, .2)
        plotted_start_year = target_dataset[var].sel(time=slice(start_year,end_year)).time.min().dt.year.values
        plotted_end_year = target_dataset[var].sel(time=slice(start_year,end_year)).time.max().dt.year.values
        
    if evaluation_metric == 'mk_trend':
        lons = target_dataset.variables['longitude'][:]
        lats = target_dataset.variables['latitude'][:]
        slopes = np.ma.apply_along_axis(get_linear_mk, 0, target_dataset[var], what='slope')
        slopes_p= np.ma.apply_along_axis(get_linear_mk, 0, target_dataset[var], what='pval')
    
        slopes = xr.DataArray(slopes, dims = ('latitude', 'longitude' ), coords= {'longitude':lons, 'latitude':lats})    
        slopes_p = xr.DataArray(slopes_p, dims = ('latitude', 'longitude' ), coords= {'longitude':lons, 'latitude':lats})
        
        results=[slopes, slopes_p]
        colorbar_label = 'Gradient (mm/yr)'
        cmap = 'RdBu'
        levs = np.arange(-9, 11, 2)
        plotted_start_year = target_dataset[var].sel(time=slice(start_year,end_year)).time.min().dt.year.values
        plotted_end_year = target_dataset[var].sel(time=slice(start_year,end_year)).time.max().dt.year.values
    
    return results, colorbar_label, cmap, levs, plotted_start_year, plotted_end_year


def fig_format(num_datasets):
    
    no_labels = [0,0,0,0]
    left_labels = [1, 0, 0, 0]
    bottom_labels = [0, 0, 0, 1]
    
    if num_datasets == 1:
        fig = plt.figure(figsize=(9,4))
        fig_array = [1, 1]
        label_parallels = [left_labels]
        label_meridians = [bottom_labels]
    
    if num_datasets == 2:
        fig = plt.figure(figsize=(9,4))
        fig_array = [1, 2]
        label_parallels = [left_labels, no_labels]
        label_meridians = [bottom_labels, bottom_labels]
        
    if num_datasets == 3:
        fig = plt.figure(figsize=(9,4))
        fig_array = [1, 3]
        label_parallels = [left_labels, no_labels, no_labels]
        label_meridians = [bottom_labels, bottom_labels, bottom_labels]
         
    if num_datasets == 4:
        fig = plt.figure(figsize=(9,4))
        fig_array = [2, 2]
        label_parallels = [left_labels, no_labels, left_labels, no_labels]
        label_meridians = [no_labels, no_labels, bottom_labels, bottom_labels]
        
    if num_datasets == 5:
        fig = plt.figure(figsize=(9,4))
        fig_array = [2, 3]
        label_parallels = [left_labels, no_labels, no_labels, left_labels, no_labels]
        label_meridians = [no_labels, no_labels, no_labels, bottom_labels, bottom_labels, bottom_labels]
          
    if num_datasets == 6:
        fig = plt.figure(figsize=(9,4))
        fig_array = [2, 3]
        label_parallels = [left_labels, no_labels, no_labels, left_labels, no_labels, no_labels]
        label_meridians = [no_labels, no_labels, no_labels, bottom_labels, bottom_labels, bottom_labels]
        
    if num_datasets == 7:
        fig = plt.figure(figsize=(9,4))
        fig_array = [2, 4]
        label_parallels = [left_labels, no_labels, no_labels, no_labels, left_labels, no_labels, no_labels]
        label_meridians = [no_labels, no_labels, no_labels, bottom_labels, bottom_labels, bottom_labels, bottom_labels]
        
    if num_datasets == 8:
        figsize=[9,4]
        fig_array = [2, 4]
        label_parallels = [left_labels, no_labels, no_labels, no_labels, left_labels, no_labels, no_labels, no_labels]
        label_meridians = [no_labels, no_labels, no_labels, no_labels, bottom_labels, bottom_labels, bottom_labels, bottom_labels]
         
    if num_datasets == 9:
        fig = plt.figure(figsize=(9,4))
        fig_array = [3, 3]
        label_parallels = [left_labels, no_labels, no_labels, left_labels, no_labels, no_labels, left_labels, no_labels, no_labels]
        label_meridians = [no_labels, no_labels, no_labels, no_labels, no_labels, no_labels, bottom_labels, bottom_labels, bottom_labels]

    if num_datasets == 10:
        fig = plt.figure(figsize=(9,4))
        fig_array = [2, 5]
        label_parallels = [left_labels, no_labels, no_labels, no_labels, no_labels, 
                           left_labels, no_labels, no_labels, no_labels, no_labels]
        label_meridians = [no_labels, no_labels, no_labels, no_labels, no_labels, 
                           bottom_labels, bottom_labels, bottom_labels, bottom_labels, bottom_labels]

    if num_datasets == 11:
        fig = plt.figure(figsize=(9,4))
        fig_array = [3, 4]
        label_parallels = [left_labels, no_labels, no_labels, no_labels,
                           left_labels, no_labels, no_labels, no_labels, 
                           left_labels, no_labels, no_labels, no_labels]
        label_meridians = [no_labels, no_labels, no_labels, no_labels, 
                           no_labels, no_labels, no_labels, bottom_labels, 
                           bottom_labels, bottom_labels, bottom_labels, bottom_labels]

    if num_datasets == 12:
        fig = plt.figure(figsize=(22,14))
        fig_array = [3, 4]
        label_parallels = [left_labels, no_labels, no_labels, no_labels,
                           left_labels, no_labels, no_labels, no_labels, 
                           left_labels, no_labels, no_labels, no_labels]
        label_meridians = [no_labels, no_labels, no_labels, no_labels, 
                           no_labels, no_labels, no_labels, no_labels, 
                           bottom_labels, bottom_labels, bottom_labels, bottom_labels]

    if num_datasets == 13:
        fig = plt.figure(figsize=(22,12))
        fig_array = [3, 5]
        label_parallels = [left_labels, no_labels, no_labels,  no_labels, no_labels,
                           left_labels, no_labels, no_labels,  no_labels, no_labels,
                           left_labels, no_labels, no_labels,  no_labels, no_labels]
        label_meridians = [no_labels, no_labels, no_labels, no_labels,  no_labels,
                           no_labels, no_labels, no_labels,  bottom_labels, bottom_labels,
                           bottom_labels, bottom_labels, bottom_labels, bottom_labels, bottom_labels]

    if num_datasets == 14:
        fig = plt.figure(figsize=(22,14))
        fig_array = [3, 5]
        label_parallels = [left_labels, no_labels, no_labels, left_labels, no_labels, no_labels, left_labels, no_labels, no_labels]
        label_meridians = [no_labels, no_labels, no_labels, no_labels, no_labels, no_labels, bottom_labels, bottom_labels, bottom_labels]

    if num_datasets == 15:
        fig = plt.figure(figsize=(22,14))
        fig_array = [3, 5]
        label_parallels = [left_labels, no_labels, no_labels, left_labels, no_labels, no_labels, left_labels, no_labels, no_labels]
        label_meridians = [no_labels, no_labels, no_labels, no_labels, no_labels, no_labels, bottom_labels, bottom_labels, bottom_labels]

    if num_datasets == 16:
        fig = plt.figure(figsize=(22,18))
        fig_array = [4, 4]
        label_parallels = [left_labels, no_labels, no_labels, left_labels, no_labels, no_labels, left_labels, no_labels, no_labels]
        label_meridians = [no_labels, no_labels, no_labels, no_labels, no_labels, no_labels, bottom_labels, bottom_labels, bottom_labels]
 
 
 
    return fig, fig_array, label_parallels, label_meridians