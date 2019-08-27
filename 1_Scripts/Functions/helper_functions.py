'''
Created on 17.02.2017

@author: Anja T. Rädler
'''

import os
import errno
import numpy as np
import numpy as np
import re
import glob
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


from matplotlib import cbook
from matplotlib.colors import Normalize
from numpy import ma

class MidPointNorm(Normalize):    
    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self,vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin < midpoint < vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")       
        elif vmin == vmax:
            result.fill(0) 
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            #First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint            
            resdat[resdat>0] /= abs(vmax - midpoint)            
            resdat[resdat<0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = ma.array(resdat, mask=result.mask, copy=False)                

        if is_scalar:
            result = result[0]            
        return result
          
def find_ncvariable_names(data):
    ncvarnames=[]
    for i in range(0,len(data.data_vars.keys())):        
        if ((data.data_vars.keys()[i]!='lon') and (data.data_vars.keys()[i]!='lat') and (data.data_vars.keys()[i]!='longitude') and (data.data_vars.keys()[i]!='latitude') and (data.data_vars.keys()[i]!='height')and (data.data_vars.keys()[i]!='rotated_pole') and (data.data_vars.keys()[i]!='standard_error')):
            ncvarnames.append(data.data_vars.keys()[i])
            l.debug('ncvarname = '+str(ncvarnames))
        else: 
            l.debug('Found extra variables for lon/lat/height/rotated_pole: '+str(data.data_vars.keys()[i]))
    if len(ncvarnames) == 1:
        return ncvarnames[0]
    elif len(ncvarnames) == 0:
        l.critical('No variable found!')
        raise UserWarning('No ncvariable found while reading data.')        
    else:
        l.error('More than one variable found! '+str(ncvarnames))    
            
def find_underlines(name):
    # finds position of _ in string
    underlineendis = []
    for m in re.finditer('_', name):
        underlineendis.append(m.end()) 
    return underlineendis
    
def find_slash(name):
    # finds position of _ in string
    slashes = []
    for m in re.finditer('/', name):
        slashes.append(m.end()) 
    return slashes
 
def find_between( s, first, last ):
    # finds content between two strings in string
    try:
        start = s.index( first ) + len( first )
        end = s.index( last, start )
        return s[start:end]
    except ValueError:
        return ""   
     
def extract_filename(filename):
        return os.path.splitext(os.path.split(filename)[1])[0]         
    
def extract_filepath(filename):
        return os.path.splitext(os.path.split(filename)[0])[0] +'/'     
          

def custom_div_cmap(numcolors=11, name='custom_div_cmap',
                    mincol='blue', midcol='white', maxcol='red'):
    """ Create a custom diverging colormap with three colors
    
    Default is blue to white to red with 11 colors.  Colors can be specified
    in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    """

    from matplotlib.colors import LinearSegmentedColormap 
    
    cmap = LinearSegmentedColormap.from_list(name=name, 
                                             colors =[mincol, midcol, maxcol],
                                             N=numcolors)
    return cmap
