'''
Created on 01.07.2017

@author: Anja Raedler 
'''

import os
import numpy as np
import pylab as pylab
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

from helper_functions import *
       
class ENSEMBLE_OBJ:
    '''
    classdocs
    '''
        
    def __init__(self, __ncfilename,dummy_coord_file=None):
        '''
        Constructor
        '''
        self.ncfilename  = __ncfilename                
        self.data        = xr.open_dataset(self.ncfilename)
        
        ### find netcdf variable other than longitude or latitude
        self.ncvariable = find_ncvariable_names(self.data) 
                        
        # check if past ensemble object and matching object contains parameter data or predictions        
        if 'statistical_measure' in self.data.attrs:
            l.debug('statistical_measure found in attributes. Assuming parameter ensemble.')
            self.ensembletype = 'parameter'
        else:
            l.debug('statistical_measure not found in attributes. Assuming prediction ensemble.')
            self.ensembletype = 'prediction'
        
        self.filename = self.extract_filename()
        self.filepath = self.extract_filepath()
        
        underlines = find_underlines(self.extract_filename())      
        
        if self.ensembletype=='prediction':  
            self.paramname   = None
            self.stat_measure= None
            self.datasetname = self.extract_filename()[0:underlines[0]-1]
            self.scenario    = self.extract_filename()[underlines[3]:underlines[4]-1]       
            self.hazard      = self.extract_filename()[underlines[1]:underlines[2]-1]
            self.severity    = self.extract_filename()[underlines[2]:underlines[3]-1]            
            self.years       = self.extract_filename()[underlines[4]:underlines[4]+9]

        elif self.ensembletype=='parameter':            
            self.paramname   = self.extract_filename()[0:underlines[0]-1]
            self.stat_measure= self.extract_filename()[underlines[0]:underlines[1]-1]
            self.scenario    = self.extract_filename()[underlines[1]:underlines[2]-1]    
            self.hazard      = None
            self.severity    = None           
            self.years       = self.extract_filename()[underlines[2]:underlines[3]-1] +'-'+self.extract_filename()[underlines[3]:underlines[4]-1]            
               
        if self.ensembletype=='prediction':
            self.varname_unit  = find_ncvariable_names(self.data)
        elif self.ensembletype=='parameter':
            if self.paramname=='LI':
                self.varname_unit  = r'LI (K)'
            elif self.paramname=='RH':
                self.varname_unit  = r'RH (%)'
            elif self.paramname=='DLS':
                self.varname_unit = r'DLS (m s$^{-1}$)'
            else:
                self.varname_unit  = find_ncvariable_names(self.data)
            
        l.info('varname_unit = '+str(self.varname_unit))
        
        l.debug('self.data.dims.keys() = '+str(self.data.dims.keys()))
              
        if 'lat' in self.data.dims.keys():
            self.latvar = 'lat'
        elif 'latitude' in self.data.dims.keys():
            self.latvar = 'latitude'    
        elif 'rlat' in  self.data.dims.keys():
            self.latvar = 'rlat'
        else: 
            raise UserWarning('latvar is neighter lat nor latitude nor rlat')            
        
            
        if 'lon' in self.data.dims.keys():
            self.lonvar = 'lon'
        elif 'longitude' in self.data.dims.keys():
            self.lonvar = 'longitude'

        elif 'rlon' in  self.data.dims.keys():
            self.lonvar = 'rlon'            
        else: 
            raise UserWarning('lonvar is neighter lon nor longitude nor rlon')            
        
        self.dummy_coord_file = dummy_coord_file
        
        
        if dummy_coord_file!=None:
            dummy_coord = xr.open_dataset(dummy_coord_file)            
            if ('lat' in dummy_coord.data_vars.keys()) and ('lon' in dummy_coord.data_vars.keys()):            
                self.data['lat'] = dummy_coord.lat
                self.data['lon'] = dummy_coord.lon
        
        self.dpi = 400    
        
    def extract_filename(self):
        return os.path.splitext(os.path.split(self.ncfilename)[1])[0]    
    
    def extract_filepath(self):
        return os.path.splitext(os.path.split(self.ncfilename)[0])[0] +'/'         
       
    def return_domain_indices(self,domain):
        if self.lonvar == 'rlon':                      
            latindex1 = (self.data.lat.values >= domain.latmin)
            latindex2 = (self.data.lat.values <=domain.latmax)
            latindex = latindex1*latindex2            
            lonindex1 = (self.data.lon.values >= domain.lonmin)
            lonindex2 = (self.data.lon.values <= domain.lonmax)
            lonindex = lonindex1*lonindex2                        
            finalindex = latindex*lonindex            
            rloni,rlati = np.where(finalindex)       
                 
            return finalindex
        else:
            raise UserWarning('lonvar is not rlon')
     
    def sel_data_domain(self,domain):      
        if self.lonvar == 'rlon':                      
            latindex1 = (self.data.lat.values >= domain.latmin)
            latindex2 = (self.data.lat.values <=domain.latmax)
            latindex = latindex1*latindex2            
            lonindex1 = (self.data.lon.values >= domain.lonmin)
            lonindex2 = (self.data.lon.values <= domain.lonmax)
            lonindex = lonindex1*lonindex2                        
            finalindex = latindex*lonindex            
            rloni,rlati = np.where(finalindex)                
            return self.data.isel_points(rlat=rlati,rlon=rloni)
        else:
            raise UserWarning('lonvar is not rlon')
    
    def calculate_robustness_of_change_1std(self,historical_ensemble_obj):        
        difference =  self.data - historical_ensemble_obj.return_rearanged_data(self)        
        diff_mean  = difference.mean(dim='runs')
        diff_std   = difference.std(dim='runs')            
        meshrlon,meshrlat = np.meshgrid(diff_mean[find_ncvariable_names(diff_mean)].rlon,diff_mean[find_ncvariable_names(diff_mean)].rlat)        
        meshrlon = meshrlon[np.abs(diff_mean[find_ncvariable_names(diff_mean)].values)>1*diff_std[find_ncvariable_names(diff_mean)].values]
        meshrlat = meshrlat[np.abs(diff_mean[find_ncvariable_names(diff_mean)].values)>1*diff_std[find_ncvariable_names(diff_mean)].values]
        return [meshrlon,meshrlat]
    
    def calculate_robustness_of_change_2std(self,historical_ensemble_obj):        
        difference =  self.data - historical_ensemble_obj.return_rearanged_data(self)        
        diff_mean = difference.mean(dim='runs')
        diff_std  = difference.std(dim='runs')   
        meshrlon,meshrlat = np.meshgrid(diff_mean[find_ncvariable_names(diff_mean)].rlon,diff_mean[find_ncvariable_names(diff_mean)].rlat)        
        meshrlon = meshrlon[np.abs(diff_mean[find_ncvariable_names(diff_mean)].values)>2*diff_std[find_ncvariable_names(diff_mean)].values]
        meshrlat = meshrlat[np.abs(diff_mean[find_ncvariable_names(diff_mean)].values)>2*diff_std[find_ncvariable_names(diff_mean)].values]
        return [meshrlon,meshrlat]
       
    def return_rearanged_data(self,matching_obj,domain=None):       
        l.debug('-----------------------------')
        l.debug('self.data.attrs = '+str(self.data.attrs))
        l.debug('matching_obj.attrs    = '+str(matching_obj.data.attrs))
        number_runs = matching_obj.data.dims['runs']
        l.debug('number_runs_self         = '+str(self.data.dims['runs']))
        l.debug('number_runs_matching_obj = '+str(number_runs))        
        # initialize list 
        matchlist = []
        for future_i in range(0,number_runs):
            l.debug('#####################')
            l.debug('future_i = '+str(future_i))
            attrstr_future =  matching_obj.data.attrs['Run '+str(future_i)][0:matching_obj.data.attrs['Run '+str(future_i)].find('.nc')]            
            if self.ensembletype=='prediction':
                instname_future = attrstr_future[find_underlines(attrstr_future)[-4]:find_underlines(attrstr_future)[-3]-1]
                runname_future  = attrstr_future[find_underlines(attrstr_future)[-3]:find_underlines(attrstr_future)[-2]-1]  
            elif self.ensembletype=='parameter':                
                slashlist_future = find_slash(attrstr_future)    
                l.debug('slashlist_future='+str(slashlist_future))            
                instname_future = attrstr_future[slashlist_future[-4]:slashlist_future[-3]-1]
                runname_future  = attrstr_future[slashlist_future[-3]:slashlist_future[-2]-1]                      
            l.debug('instname_future = '+str(instname_future))
            l.debug('runname_future = '+str(runname_future))
            l.debug('future: '+str(instname_future)+ ' '+str(runname_future))            
            # go through all historical runs and find match
            reduced_haz_obj_hist_data = self.data            
            for hist_i in range(0,self.data.dims['runs']):                
                l.debug('********************')
                l.debug('hist_i = '+str(hist_i))
                attrstr_hist   = self.data.attrs['Run '+str(hist_i)][0:self.data.attrs['Run '+str(hist_i)].find('.nc')]
                if self.ensembletype=='prediction':
                    instname_hist = attrstr_hist[find_underlines(attrstr_hist)[-4]:find_underlines(attrstr_hist)[-3]-1]
                    runname_hist  = attrstr_hist[find_underlines(attrstr_hist)[-3]:find_underlines(attrstr_hist)[-2]-1]     
                elif self.ensembletype=='parameter':
                    slashlist_hist = find_slash(attrstr_hist)  
                    instname_hist = attrstr_hist[slashlist_hist[-4]:slashlist_hist[-3]-1]
                    runname_hist  = attrstr_hist[slashlist_hist[-3]:slashlist_hist[-2]-1]                             
                l.debug('instname_hist = '+str(instname_hist))
                l.debug('runname_hist = '+str(runname_hist)) 
                l.debug('hist: '+str(instname_hist)+ ' '+str(runname_hist))                
                if (instname_future == instname_hist) and (runname_future == runname_hist): 
                    matchlist.append(hist_i)
                    l.debug('future: '+str(instname_future)+ ' '+str(runname_future)+' = hist: '+ str(instname_hist)+' '+str(runname_hist))
                    l.debug('Hist   run '+str(hist_i) + ' '+self.data.attrs['Run '+str(hist_i)])
                    l.debug('Future run '+str(future_i) + ' '+matching_obj.data.attrs['Run '+str(future_i)])
                    l.debug('future: '+str(instname_future)+ ' '+str(runname_future)+' = hist: '+ str(instname_hist)+' '+str(runname_hist))
                    break                    
                else: 
                    l.debug('future: '+str(instname_future)+ ' '+str(runname_future)+' = hist: '+ str(instname_hist)+' '+str(runname_hist))              
                    continue                
        if len(matchlist) !=   number_runs:
            l.error('Matchlist dimension does not match number of future runs')
            raise UserWarning('ERROR!')            
        else:
            l.debug('Matchlist = '+str(matchlist))        
        # return rearanged historical dataset for given future dataset with same dimensions
        ens_data_rearanged = self.data[find_ncvariable_names(self.data)].sel(runs=tuple(matchlist)).to_dataset()
        ens_data_rearanged.runs.values = range(0,number_runs)        
        attrs_new = {}              
        for i in range(0,len(matchlist)):
            attrs_new['Run '+str(i)] = self.data.attrs['Run '+str(matchlist[i])]        
            ens_data_rearanged.attrs = attrs_new
            l.debug('ens_data_rearanged = '+str(ens_data_rearanged))        
        # add lat lon variable to rearanged dataset
        if self.dummy_coord_file !=None:
            dummy_coord = xr.open_dataset( self.dummy_coord_file)            
            if ('lat' in dummy_coord.data_vars.keys()) and ('lon' in dummy_coord.data_vars.keys()):            
                ens_data_rearanged['lat'] = dummy_coord.lat
                ens_data_rearanged['lon'] = dummy_coord.lon            
        if domain!=None:
            rlati,rloni =  np.where(self.return_domain_indices(domain))
            return ens_data_rearanged.isel_points(rlat=rlati,rlon=rloni)            
        else:
            return ens_data_rearanged

   