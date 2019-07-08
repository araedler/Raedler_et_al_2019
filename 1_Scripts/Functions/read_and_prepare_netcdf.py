'''
Created on 15.02.2017

@author: Anja T. Raedler

Purpose: reading and preparing netcdf datasets
'''
import logging as l
import imp            
try:
    imp.find_module('xray')
    foundxray = True
except ImportError:
    foundxray = False
try:
    imp.find_module('xarray')
    foundxarray = True
except ImportError:
    foundxarray = False

if foundxarray==True:
    import xarray as xr
else:
    if foundxray ==True:
        import xray as xr
    else:
        l.error('Neither xray nor xarray is available.')
import numpy as np


    
def read_netcdf(filename,ncvariable,domainCoords,selyearrange,verbose=True):   
    try:
        data = xr.open_dataset(filename).sel(time=slice(str(selyearrange.start)+'-01-01',str(selyearrange.end)+'-12-31')) 
        variables = data.data_vars.keys()
    except: 
        raise UserWarning('cannot open file ' + filename)     
    try:
        variables.index(ncvariable)
    except:
        print ncvariable, ' variable not found in netcdf file'    
    if verbose:
        l.debug('shape of initial file = '+ str(np.shape(data[ncvariable])) )    
    if 'lat' in data.dims.keys():
        latvar = 'lat'
    if 'latitude' in data.dims.keys():
        latvar = 'latitude'    
    if data[latvar][0] > data[latvar][1]:   
        startlat = domainCoords.latmax
        endlat   = domainCoords.latmin
    elif data[latvar][0] < data[latvar][1]:
        startlat = domainCoords.latmin
        endlat   = domainCoords.latmax        
    datasel = data[ncvariable].loc[:,startlat:endlat,domainCoords.lonmin:domainCoords.lonmax].to_dataset()     
    return datasel

def read_netcdf_all_domain(filename,ncvariable,selyearrange,verbose=True):   
    try:
        data = xr.open_dataset(filename).sel(time=slice(str(selyearrange.start)+'-01-01',str(selyearrange.end)+'-12-31')) 
        variables = data.data_vars.keys()
    except: 
        raise UserWarning('cannot open file ' + filename)     
    try:
        variables.index(ncvariable)
    except:
        print ncvariable, ' variable not found in netcdf file'    
    if verbose:
        l.debug('shape of initial file = '+ str(np.shape(data[ncvariable])) )      
    datasel = data[ncvariable].to_dataset()     
    return datasel


def read_netcdf_flat(filename,ncvariable,domainCoords,selyearrange,verbose=True):    
    try:
        data      = xr.open_dataset(filename).sel(time=slice(str(selyearrange.start)+'-01-01',str(selyearrange.end)+'-12-31'))          
        variables = data.data_vars.keys()
    except: 
        raise UserWarning('cannot open file ' +filename) 
    try:
        l.debug('variable found:' + str(variables[variables.index(ncvariable)]))        
        ncvariable = variables[variables.index(ncvariable)]
    except:        
        UserWarning(ncvariable+ ' variable not found in netcdf file '  + filename)       
    # check if latitude values start from high value or from low value
    if 'lat' in data.dims.keys():
        latvar = 'lat'
    if 'latitude' in data.dims.keys():
        latvar = 'latitude'  
    if data[latvar][0] > data[latvar][1]:   
        startlat = domainCoords.latmax
        endlat   = domainCoords.latmin
    elif data[latvar][0] < data[latvar][1]:
        startlat = domainCoords.latmin
        endlat   = domainCoords.latmax        
    datasel = data[ncvariable].loc[:,startlat:endlat,domainCoords.lonmin:domainCoords.lonmax].to_dataset()         
    if verbose:        
        l.debug('shape of initial file = '+ str(np.shape(data[ncvariable])) )    
    # flatten data for comparison    
    datasel_flat = datasel[ncvariable].values.flatten()     
    if verbose:
        l.debug('shape after flatten selected domain = '+ str(datasel_flat.shape) )        
    return datasel_flat


def get_ncvariable(path,filename,verbose=True):  
    try: 
        data = xr.open_dataset(path+filename+'.nc')  
        variables = data.data_vars.keys()
    except:
        raise UserWarning('Netcdf file not found '+ path+filename+'.nc')    
    if len(variables) != 1:
        raise UserWarning('Netcdf file contains either none or more than one variable: ', variables)    
    return variables[0]
    
    

    