# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:51:08 2016

@author: jlarsch
"""
import numpy as np
import scipy.stats as sta
import models.geometry as geometry
import matplotlib.pyplot as plt
import functions.peakdet as peakdet
import random
from models.AnimalTimeSeriesCollection import AnimalTimeSeriesCollection

class Animal(object):
    def __init__(self,ID=0):
        self.ID=ID
        self.shiftIndex=[]
        self.pair=[]
        self.ts=[]
        self.shiftIndex=[]
        self.neighbor=[]

    def joinPair(self,pair):
        self.pair=pair
        pair.addAnimal(self)
        
    def add_TimeSeriesCollection(self,ts):
        self.ts=ts
        return ts
        
    def wakeUp(self):
        
        self.shiftIndex=self.get_shiftIndex()
        AnimalTimeSeriesCollection().linkAnimal(self)
        self.neighbor=self.pair.animals[1-self.ID]
        
                
    def get_shiftIndex(self):
        shift=self.pair.shift
        minShift=self.pair.experiment.expInfo.minShift
        numFrames=self.pair.experiment.expInfo.numFrames
        if shift and self.ID==1:
            return int(random.uniform(minShift,numFrames-minShift))
        else:
            return 0
        
    
    def at_bottom_of_dish_stack(self):
              
        out_venture=self.ts.Pol_n()
        x=out_venture[-2]==0
        return x
         
    

    def bout_analysis(self):
        self.bout_start=peakdet.detect_peaks(self.tailCurvature_mean,1,8)
        rangeVector_before=np.nanmedian([self.rangeVector[b-6:b-3] for b in self.bout_start[1:-2]],axis=1)
        heading_before=np.nanmedian([self.fish_orientation_elipse[b-6:b-3] for b in self.bout_start[1:-2]],axis=1)
        heading_after=np.nanmedian([self.fish_orientation_elipse[b+3:b+6] for b in self.bout_start[1:-2]],axis=1)
        #error(angle) is difference baseline-heading    
        error_before=geometry.smallest_angle_difference_degrees(heading_before,rangeVector_before) 
        error_after=geometry.smallest_angle_difference_degrees(heading_after,rangeVector_before) 
        d_error_inst=np.diff(np.abs([error_before,error_after]).T,axis=1)
#        d_heading=geometry.smallest_angle_difference_degrees(heading_before,heading_after)
        d_heading=np.squeeze(np.diff([heading_before,heading_after],axis=0).T)
        
#        interBoutSD=np.nanstd([self.fish_orientation_elipse[self.bout_start[i-1]+3:self.bout_start[i]-3] for i in np.arange(1,self.bout_start.shape[0]-2,1)],axis=1)   
#        intraBoutSD=np.nanstd([self.fish_orientation_elipse[self.bout_start[i]-2:self.bout_start[i]+2] for i in np.arange(1,self.bout_start.shape[0]-2,1)],axis=1)
        
        #plt.figure()  
        if self.ID==0:
            col=(0,0,1)
        else:
            col=(0,1,0)
        
        if self.noShiftAnimal:
            lw=3
        else:
            lw=1
            col=(.5,.5,.5)

        m1=np.logical_and(np.isfinite(error_before)==True,np.isfinite(d_heading)==True)
        m2=np.abs(error_before)<90
        m=np.logical_and(m1,m2)
        self.corr_edh=sta.pearsonr(error_before[m],d_heading[m])
        if self.shiftRun<1:
#            print self.corr_edh
#            plt.figure()
#            bins=np.arange(-180,180,5)
#            plt.hist2d(error_before[m],d_heading[m],bins=bins)
            plt.figure(0)
            n1, bins, patches =plt.hist(d_error_inst,bins=range(-100,100,1), normed=1, histtype='step',lw=lw)    
            plt.plot([0,0], [0, np.max(n1)], 'k:', lw=3)
            plt.title('change in error angle before vs after bout (instantaneous, all frames)')

            dist_at_bout=self.dist[self.bout_start[1:-2]]
            near=dist_at_bout<260
            
            plt.figure(1+self.ID)

            n1, bins, patches =plt.hist(error_before[near],bins=range(-180,180,5), normed=1, histtype='step',lw=lw,color=tuple(np.array(col)/2.0))  
            n1, bins, patches =plt.hist(error_after[near],bins=range(-180,180,5), normed=1, histtype='step',lw=lw,color=col)    
            plt.title(['animal '+str(self.ID)+' error angle before vs after bout (dist<30 mm)'])

            plt.figure(5+self.ID)
            n1, bins, patches =plt.hist(error_before,bins=range(-180,180,5), normed=1, histtype='step',lw=lw,color=tuple(np.array(col)/2.0))  
            n1, bins, patches =plt.hist(error_after,bins=range(-180,180,5), normed=1, histtype='step',lw=lw,color=col)    
            plt.title(['animal '+str(self.ID)+' error angle before vs after bout (all dist)'])


    def plot_errorAngle(self,maxCenterDistance=np.inf):
        p1=self.ts.centroidContour().xy
        p2=self.neighbor.ts.centroidContour().xy
        borderMask=np.logical_and(np.abs(p1)<maxCenterDistance, np.abs(p2)<maxCenterDistance)
        borderMask=np.logical_and(borderMask[:,0],borderMask[:,1])

        if self.ID==0:
            col=(0,0,1)
        else:
            col=(0,1,0)
        
        if not self.pair.shift:
            lw=3
        else:
            lw=1
            col=(.5,.5,.5)

            
        plt.figure(10)
        n, bins, patches =plt.hist(self.ts.errorAngle()[borderMask], bins=range(-180,180,5),normed=1, histtype='step',lw=lw, color=col)