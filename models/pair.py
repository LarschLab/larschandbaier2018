# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:57:38 2016

@author: jlarsch
"""


import numpy as np
import random
from models.geometry import Trajectory
import functions.matrixUtilities_joh as mu
from models.animal import Animal

class Pair(object):
    #Class for trajectories of one pair of animals.
    #Calculation and storage of dependent time series: speed, heading, etc.
    def __init__(self, trajectories, expInfo,shiftIndex=0,shiftRun=0):
        self.animals=[]
        self.animals.append(Animal(0,trajectories[:,0,:],expInfo,shiftRun=shiftRun))
        self.animals.append(Animal(1,trajectories[:,1,:],expInfo,shiftIndex,shiftRun=shiftRun))
        self.get_stack_order()
        self.get_IAD()
        
        self.animals[0].analyze_neighbor_interactions(self.animals[1])
        self.animals[1].analyze_neighbor_interactions(self.animals[0])

        
    def get_stack_order(self):
        for x in self.animals:
            x.get_outward_venture(self.max_out_venture())
        
        out_venture_all=[x.Pol_n for x in self.animals]
        #figure out who is on top. Animal in top arena can go outwards further
        if out_venture_all[0][-2]==0: #second last bin of animal0 is empty, meaning animal1 went out further ->was on top
            self.StackTopAnimal=np.array([0,1])
        elif out_venture_all[1][-2]==0:
            self.StackTopAnimal=np.array([1,0])
        else: #no animal went out more than one bin further than the other -> likely no stack experiment
            self.StackTopAnimal=np.array([0,0])   
            

            
            
    def get_IAD(self):
        dist=Trajectory()
        dist.xy=self.animals[0].position.xy-self.animals[1].position.xy
#        dist.xy=self.animals[0].centroidContour-self.animals[1].centroidContour
        self.IAD = np.sqrt(dist.x()**2 + dist.y()**2) 
    
        #absolute inter animal distance IAD
        
        self.IAD_m=np.nanmean(self.IAD)
        histBins=np.arange(100)
        self.IADhist =np.histogram(self.IAD,bins=histBins,normed=1)[0]
        


    def getShoalDwellTimes(IAD):
        IADsm=mu.runningMean(IAD,30)
        lowThreshold=np.nanmax(IADsm)/1.5
        
        lowIAD=(np.less(IAD,lowThreshold)).astype(int)
        hiIAD=(np.greater(IAD,lowThreshold)).astype(int)
        
        #transitions from low to high inter animal distance and vice versa
        LowToHigh=np.where(np.equal((hiIAD[:-1]-hiIAD[1:]),-1))[0]
        HighToLow=np.where(np.equal((lowIAD[:-1]-lowIAD[1:]),-1))[0]
        
        #number of transitions to use below
        maxL=np.min([np.shape(HighToLow)[0],np.shape(LowToHigh)[0]])-2
        
        #How long are High and low dwell times?
        #calculate from transition times. Order depends on starting state of data
        
        if HighToLow[0]>LowToHigh[0]: #meaning starting low
            HiDwell=HighToLow[0:maxL]-LowToHigh[0:maxL]
            LowDwell=LowToHigh[1:maxL]-HighToLow[0:maxL-1]
        else: #meaning starting high
            HiDwell=HighToLow[1:maxL]-LowToHigh[0:maxL-1]
            LowDwell=LowToHigh[0:maxL]-HighToLow[0:maxL]
    
        return HiDwell,LowDwell,HighToLow,LowToHigh

        
    def avgSpeed(self):
        a1=np.nanmean(self.animals[0].speed)
        a2=np.nanmean(self.animals[1].speed)
        return np.array([a1,a2])
        
    def max_out_venture(self):
        mov=[]
        mov.append([x.positionPol.y() for x in self.animals])
        return np.nanmax(mov)

    def get_var_from_all_animals(self, var):
        #this function returns a specified variable from all animals as a matrix.
        #animal number will be second dimension
        
        tmp=self.animals
        for i in range(len(var)):
            tmp=[getattr(x, var[i]) for x in tmp]
        return np.stack(tmp,axis=1)

class shiftedPair(object):
    #Class to calculate null hypothesis time series using shifted pairs of animals
    def __init__(self, pair,expInfo):
        self.nRuns=10
        self.minShift=5*60*expInfo.fps
        self.sPair=[]
        self.shiftIndex=[]
        #generate nRuns instances of Pair class with one animal time shifted against the other
        for i in range(self.nRuns):
            tra=pair.get_var_from_all_animals(['rawTra','xy'])
            
            shiftIndex=int(random.uniform(self.minShift,pair.IAD.shape[0]-self.minShift))
            #time-rotate animal 0, keep animal 1 as is
            #traShift[:,0,:]=np.roll(traShift[:,0,:],shiftIndex,axis=0)
            self.sPair.append(Pair(tra,expInfo,shiftIndex=shiftIndex,shiftRun=i))
            self.shiftIndex.append(shiftIndex)
            
        #calculate mean and std IAD for the goup of shifted instances
        self.spIAD_m = np.nanmean([x.IAD_m for x in self.sPair])
        self.spIAD_std = np.nanstd([x.IAD_m for x in self.sPair])

def shiftedPairSystematic(tra,expInfo,nRuns):
    #function to test systematically the effect of the extent of the time shift
    shiftInterval=expInfo.fps
    sIADList=[]
    #generate nRuns instances of Pair class with one animal time shifted against the other
    for i in range(nRuns):
        traShift=tra.positionPx.copy()
        shiftIndex=i*shiftInterval
        #time-rotate animal 0, keep animal 1 as is
        traShift[:,0,:]=np.roll(traShift[:,0,:],shiftIndex,axis=0)
        tmpPair=Pair(traShift,expInfo)
        sIADList.append(tmpPair.IAD_m)
    return sIADList