# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:57:38 2016

@author: jlarsch
"""


import numpy as np
from models.geometry import Trajectory
from models.animal import Animal

class Pair(object):
    #Class for trajectories of one pair of animals.
    #Calculation and storage of dependent time series: speed, heading, etc.
    def __init__(self, shift=False,nRun=0):

        self.animals=[]
        self.shift=shift
        self.nRun=nRun
        
        
#        self.animals.append(Animal(0,trajectories[:,0,:],expInfo,shiftRun=shiftRun))
#        self.animals.append(Animal(1,trajectories[:,1,:],expInfo,shiftIndex,shiftRun=shiftRun))
#        self.get_stack_order()
#        self.get_IAD()
#        
#        self.animals[0].analyze_neighbor_interactions(self.animals[1])
#        self.animals[1].analyze_neighbor_interactions(self.animals[0])

    def addAnimal(self,animal):
        self.animals.append(animal)
        return self.animals[animal.ID]

    def linkExperiment(self,experiment):
        self.experiment=experiment
        experiment.addPair(self)
    
        for i in range(2):
            Animal(i).joinPair(self)
        
        for i in range(2):
            self.animals[i].wakeUp()
            
            
    def IAD(self):
        dist=Trajectory()
        dist.xy=self.animals[0].ts.position().xy-self.animals[1].ts.position().xy
        x = np.sqrt(dist.x()**2 + dist.y()**2) 
        return x


    def IADhist(self):
        histBins=np.arange(100)
        try:
            x=np.histogram(self.IAD()[np.isfinite(self.IAD())],bins=histBins,normed=1)[0]
        except:
            print 'problem'
            x=0
        return x
    


#    def getShoalDwellTimes(IAD):
#        IADsm=mu.runningMean(IAD,30)
#        lowThreshold=np.nanmax(IADsm)/1.5
#        
#        lowIAD=(np.less(IAD,lowThreshold)).astype(int)
#        hiIAD=(np.greater(IAD,lowThreshold)).astype(int)
#        
#        #transitions from low to high inter animal distance and vice versa
#        LowToHigh=np.where(np.equal((hiIAD[:-1]-hiIAD[1:]),-1))[0]
#        HighToLow=np.where(np.equal((lowIAD[:-1]-lowIAD[1:]),-1))[0]
#        
#        #number of transitions to use below
#        maxL=np.min([np.shape(HighToLow)[0],np.shape(LowToHigh)[0]])-2
#        
#        #How long are High and low dwell times?
#        #calculate from transition times. Order depends on starting state of data
#        
#        if HighToLow[0]>LowToHigh[0]: #meaning starting low
#            HiDwell=HighToLow[0:maxL]-LowToHigh[0:maxL]
#            LowDwell=LowToHigh[1:maxL]-HighToLow[0:maxL-1]
#        else: #meaning starting high
#            HiDwell=HighToLow[1:maxL]-LowToHigh[0:maxL-1]
#            LowDwell=LowToHigh[0:maxL]-HighToLow[0:maxL]
#    
#        return HiDwell,LowDwell,HighToLow,LowToHigh

        
    def avgSpeed(self):
        a1=np.nanmean(self.animals[0].ts.speed())
        a2=np.nanmean(self.animals[1].ts.speed())
        return np.array([a1,a2])

    def avgSpeed_smooth(self):
        a1=np.nanmean(self.animals[0].ts.speed_smooth())
        a2=np.nanmean(self.animals[1].ts.speed_smooth())
        return np.array([a1,a2])
        
    def max_out_venture(self):
        mov=[]
        mov.append([x.ts.positionPol().y() for x in self.animals])
        return np.nanmax(mov)

    def get_var_from_all_animals(self, var):
        #this function returns a specified variable from all animals as a matrix.
        #animal number will be second dimension
        
        tmp=self.animals
        for i in range(len(var)):
            tmp=[getattr(x, var[i]) for x in tmp]
        return np.stack(tmp,axis=1)

       
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