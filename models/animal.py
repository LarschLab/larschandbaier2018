# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:51:08 2016

@author: jlarsch
"""
import numpy as np
from models.geometry import Trajectory
import functions.matrixUtilities_joh as mu
import scipy.stats as sta


class Animal(object):
    def __init__(self,ID=0,rawTra=[],expInfo=[]):
        self.rawTra=Trajectory()
        self.expInfo=expInfo
        self.ID=ID
        self.position=Trajectory()
        self.positionPol=Trajectory() #x= theta, y=rho
        self.d_position=Trajectory()
        self.dd_position=Trajectory()
        
        #position of neighbor
        self.N_relPos=Trajectory()
        self.N_relPosPolRotCart=Trajectory()
        
        self.travel=np.array([])
        self.totalTravel=np.array([])
        self.speed=np.array([])
        self.accel=np.array([])
        self.heading=np.array([])
        self.d_heading=np.array([])
        
        self.rawTra.xy=rawTra
        self.position.xy=(self.rawTra.xy-self.expInfo.arenaCenterPx) / self.expInfo.pxPmm
        
        self.update_trajectory_transforms()
        
        
        #how far did the animal go out towards center?
        #this can be used to determine upper/lower animal in stacks
    
    def update_trajectory_transforms(self):
        posPol=[mu.cart2pol(*self.position.xy.T)]
        self.positionPol.xy=np.squeeze(np.array(posPol).T)
        
        #change in position for x and y
        self.d_position.xy =np.diff(self.position.xy,axis=0)
        
        self.dd_position.xy =np.diff(self.d_position.xy,axis=0)
        
        #travel distance (cartesian displacement)
        self.travel=mu.distance(*self.d_position.xy.T)
        
        #tavel speed
        self.speed=self.travel*self.expInfo.fps
        self.totalTravel=np.nansum(np.abs(self.travel))
        
        #travel acceleration
        self.accel=np.diff(self.speed)
        self.heading=mu.cart2pol(*self.d_position.xy.T)[0] #heading[0] = heading, heading[1] = speed
        #heading: pointing right = 0, pointung up = pi/2      
        self.d_heading=np.diff(self.heading)
    
        
    def get_neighbor_position(self, neighbor_animal):
        #position of neighbor relative to current animal. (where current animal has a neighbor)
        
        self.N_relPos.xy=neighbor_animal.position.xy-self.position.xy
        relPosPol=[mu.cart2pol(*self.N_relPos.xy.T)]
        relPosPolRot=np.squeeze(np.array(relPosPol).T)
        relPosPolRot=relPosPolRot[1:,:]
        relPosPolRot[:,0]=relPosPolRot[:,0]-self.heading
        tmp=[mu.pol2cart(relPosPolRot[:,0],relPosPolRot[:,1])]
        self.N_relPosPolRotCart.xy=np.squeeze(np.array(tmp).T)
        
        mapBins=np.arange(-31,32)
        self.neighborMat=np.zeros([62,62])
        #creates the neighbormat for current animal (where neighbor was)
        #this map seems flipped both horizontally and vertically! vertical is corrected at plotting.
        self.neighborMat=np.histogramdd(self.N_relPosPolRotCart.xy,bins=[mapBins,mapBins])[0]
        self.leadership()
    
    def getNeighborForce(self):
        #position here should be transformed (rotated, relative). [:,0,:] is the position of 
        mapBins=np.arange(-31,32)
        self.ForceMat=sta.binned_statistic_2d(self.N_relPosPolRotCart.x()[1:],self.N_relPosPolRotCart.y()[1:],self.accel,bins=[mapBins,mapBins])[0]
        
    def analyze_neighbor_interactions(self,neighbor_animal):
        self.get_neighbor_position(neighbor_animal)
        self.getNeighborForce()

        
        #self.dwellH,self.dwellL,self.dwellHTL,self.dwellLTH=getShoalDwellTimes(self.IAD)

    def get_outward_venture(self,maxCenterDistance=50):
        #distance from center
        histData=self.positionPol.y()
        histBins=np.linspace(0,maxCenterDistance,100)   
        self.PolhistBins=histBins
        
        self.Pol_n =np.histogram(histData[~np.isnan(histData)],bins=histBins,normed=1)[0]    
        
    def leadership(self):
        PosMat=self.neighborMat
        front=np.sum(np.sum(PosMat[:31,:],axis=1),axis=0)
        back =np.sum(np.sum(PosMat[32:,:],axis=1),axis=0)
        self.LeadershipIndex=(front-back)/(front+back)