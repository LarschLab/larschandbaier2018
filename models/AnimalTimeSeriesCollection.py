# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:22:22 2016

@author: jlarsch
"""
import numpy as np
from models.geometry import Trajectory
import functions.matrixUtilities_joh as mu
import scipy.stats as sta
import models.geometry as geometry


class AnimalTimeSeriesCollection:
    def __init__(self):
        self.dummy=0
    
    def linkAnimal(self,animal):
        self.animal=animal
        self.shiftIndex=self.animal.shiftIndex
        self.ID=animal.ID
        animal.add_TimeSeriesCollection(self)
        
    def timeShift(self,x):
        return np.roll(x,self.shiftIndex,axis=0)
        
    def rawTra(self):
        x=self.animal.pair.experiment.rawTra[:,self.ID,:]
        return Trajectory(self.timeShift(x))
    
    def tailCurvature_mean(self):
        x=self.animal.pair.experiment.tailCurvature_mean[:,self.ID]
        return self.timeShift(x)
        
    def fish_orientation_elipse(self):
        x=self.animal.pair.experiment.fish_orientation_elipse[self.ID,:]
        return self.timeShift(x)

    def centroidContour(self):
        x=self.animal.pair.experiment.centroidContour[:,:,self.ID]
        return Trajectory(self.timeShift(x))
        
    def position(self):
        arenaCenterPx=self.animal.pair.experiment.expInfo.arenaCenterPx
        pxPmm=self.animal.pair.experiment.expInfo.pxPmm
        x=(self.rawTra().xy-arenaCenterPx) / pxPmm
        return Trajectory(x)
        
    def positionPol(self):
        x=[mu.cart2pol(*self.position().xy.T)]
        x=np.squeeze(np.array(x).T)
        return Trajectory(x)
        
    def d_position(self):
        x=np.diff(self.position().xy,axis=0)
        return Trajectory(x)
        
    def dd_position(self):
        x=np.diff(self.d_position().xy,axis=0)
        return Trajectory(x)
        
    def travel(self):
        x=mu.distance(*self.d_position().xy.T)
        return x
        
    def speed(self):
        fps=self.animal.pair.experiment.expInfo.fps
        return self.travel() *fps
        
    def totalTravel(self):
        return np.nansum(np.abs(self.travel()))
        
    def accel(self):
        return np.diff(self.speed())
        
    def heading(self):
        return mu.cart2pol(*self.d_position().xy.T)[0] #heading[0] = heading, heading[1] = speed
        #heading: pointing right = 0, pointung up = pi/2  
    def d_heading(self):
        return np.diff(self.heading)
        
    def position_relative_to_neighbor(self):
        x= self.animal.neighbor.ts.position().xy-self.position().xy
        return Trajectory(x)
        
    def position_relative_to_neighbor_rot(self):
        relPosPol=[mu.cart2pol(*self.position_relative_to_neighbor().xy.T)]
        relPosPolRot=np.squeeze(np.array(relPosPol).T)
        relPosPolRot=relPosPolRot[1:,:]
        relPosPolRot[:,0]=relPosPolRot[:,0]-self.heading()
        x=[mu.pol2cart(relPosPolRot[:,0],relPosPolRot[:,1])]
        x=np.squeeze(np.array(x).T)
        return Trajectory(x)
        
    def dd_pos_pol(self):
        x=[mu.cart2pol(*self.dd_position().xy.T)]
        x=np.squeeze(np.array(x)).T
        return Trajectory(x)
        
    def dd_pos_pol_rot(self):
        x_rot =self.dd_pos_pol().xy
        x_rot[:,0]=x_rot[:,0]-self.heading()[:-1]
        x_rot_cart=[mu.pol2cart(x_rot[:,0],x_rot[:,1])]
        x_rot_cart=np.squeeze(np.array(x_rot_cart)).T
        return Trajectory(x_rot_cart)
    
    def neighborMat(self):
        mapBins=np.arange(-31,32)
        neighborMat=np.zeros([62,62])
        #creates the neighbormat for current animal (where neighbor was)
        #this map seems flipped both horizontally and vertically! vertical is corrected at plotting.
        neighborMat=np.histogramdd(self.position_relative_to_neighbor_rot().xy,bins=[mapBins,mapBins])[0]
        return neighborMat
        
    def ForceMat_speedAndTurn(self):
        mapBins=np.arange(-31,32)
        ForceMat=sta.binned_statistic_2d(self.position_relative_to_neighbor_rot().x()[1:],self.position_relative_to_neighbor_rot().y()[1:],self.accel(),bins=[mapBins,mapBins])[0]
        return ForceMat
        
    def ForceMat_speed(self):
        mapBins=np.arange(-31,32)
        sta.binned_statistic_2d(self.position_relative_to_neighbor_rot.x()[1:],self.position_relative_to_neighbor_rot.y()[1:],self.dd_pos_pol_rot().xy[:,0],bins=[mapBins,mapBins])[0]
        
    def ForceMat_turn(self):
        mapBins=np.arange(-31,32)
        sta.binned_statistic_2d(self.position_relative_to_neighbor_rot.x()[1:],self.position_relative_to_neighbor_rot.y()[1:],self.dd_pos_pol_rot().xy[:,1],bins=[mapBins,mapBins])[0]
        
    def FrontnessIndex(self):
        PosMat=self.neighborMat()
        front=np.sum(np.sum(PosMat[:31,:],axis=1),axis=0)
        back =np.sum(np.sum(PosMat[32:,:],axis=1),axis=0)
        return (front-back)/(front+back)
        
    def rangeVector(self):
        p1=self.centroidContour().xy
        p2=self.animal.neighbor.ts.centroidContour().xy
        angle_centroid_connect=geometry.get_angle_list(p1,p2)
        x=np.mod(180-angle_centroid_connect,360)
        return x
    
    def errorAngle(self):
        x=geometry.smallest_angle_difference_degrees(self.fish_orientation_elipse(),self.rangeVector())
        return x
        
    def PolhistBins(self):
        x=np.linspace(0,self.animal.pair.max_out_venture(),100)   
        return x
        
    def Pol_n(self):
        histData=self.positionPol().y()
        histData=histData[~np.isnan(histData)]
        x=np.histogram(histData,bins=self.PolhistBins(),normed=1)[0]
        return x
        
