# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 11:22:22 2016

@author: jlarsch

-This class contains 'fundamental' and 'derived' time series belonging to a specific animal

-'fundamental' time series are extracted directly from the video
    -fundamentals are stored in the parent experiment class for all animal siblings
    -fundamentals get loaded into memory from csv files
-'derived' time series are transformed fundamentals, e.g. speed, angles, heading
    -derived time series are computed during run time at each call

-this class has one parent animal. 
-Using class composition to link the parent animal into this class (self.animal)
-for derived time series related to neighbor animal, self.animal is the 'focal' animal
-the neighbor can be found in self.animal.neighbor
-each animal has one class of this type

"""
import numpy as np
from models.geometry import Trajectory
import functions.matrixUtilities_joh as mu
import scipy.stats as sta
import models.geometry as geometry


class AnimalTimeSeriesCollection:
    def __init__(self):
        self.dummy=0
    
    # Call this function from an animal class to link it with this class
    def linkAnimal(self,animal):
        self.animal=animal
        self.shiftIndex=self.animal.shiftIndex
        self.ID=animal.ID
        animal.add_TimeSeriesCollection(self)
    
    # function to shift fundamental time series upon loading to generate mock control data    
    def timeShift(self,x):
        return np.roll(x,self.shiftIndex,axis=0)
    
#--------------------------
#fundamental time series
#--------------------------
    
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

#--------------------------
#derived time series
#--------------------------
        
    #convert pixels to mm
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
        
    #currently, this is the position of the neighbor, relative to focal animal name is misleading...
    def position_relative_to_neighbor(self):
        x= self.animal.neighbor.ts.position().xy-self.position().xy
        return Trajectory(x)
        
    #rotate self to face up in order to map neighbor position relative to self
    def position_relative_to_neighbor_rot(self):
        relPosPol=[mu.cart2pol(*self.position_relative_to_neighbor().xy.T)]
        relPosPolRot=np.squeeze(np.array(relPosPol).T)
        relPosPolRot=relPosPolRot[1:,:]
        relPosPolRot[:,0]=relPosPolRot[:,0]-self.heading()
        x=[mu.pol2cart(relPosPolRot[:,0],relPosPolRot[:,1])]
        x=np.squeeze(np.array(x).T)
        return Trajectory(x)
    
    #acceleration using rotation corrected data
    #effectively splits acceleration into speeding [0] and turning [1]          
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
    
    
    #creates the neighbormat for current animal (where neighbor was)
    #this map seems flipped both horizontally and vertically! vertical is corrected at plotting.    
    def neighborMat(self):
        mapBins=np.arange(-31,32)
        neighborMat=np.zeros([62,62])

        neighborMat=np.histogramdd(self.position_relative_to_neighbor_rot().xy,bins=[mapBins,mapBins])[0]
        return neighborMat
    
#------Force matrices-------
#creates force matrix (how did focal animal accelerate depending on neighbor position)
        
    #speedAndTurn - using total acceleration
    def ForceMat_speedAndTurn(self):
        mapBins=np.arange(-31,32)
        ForceMat=sta.binned_statistic_2d(self.position_relative_to_neighbor_rot().x()[1:],self.position_relative_to_neighbor_rot().y()[1:],self.accel(),bins=[mapBins,mapBins])[0]
        return ForceMat
        
    #speed - using only acceleration component aligned with heading
    def ForceMat_speed(self):
        mapBins=np.arange(-31,32)
        sta.binned_statistic_2d(self.position_relative_to_neighbor_rot.x()[1:],self.position_relative_to_neighbor_rot.y()[1:],self.dd_pos_pol_rot().xy[:,0],bins=[mapBins,mapBins])[0]
    
    #turn - using only acceleration component perpendicular to heading   
    def ForceMat_turn(self):
        mapBins=np.arange(-31,32)
        sta.binned_statistic_2d(self.position_relative_to_neighbor_rot.x()[1:],self.position_relative_to_neighbor_rot.y()[1:],self.dd_pos_pol_rot().xy[:,1],bins=[mapBins,mapBins])[0]
    
    #percentage of time the neighbor animal was in front vs. behind focal animal
    def FrontnessIndex(self):
        PosMat=self.neighborMat()
        front=np.sum(np.sum(PosMat[:31,:],axis=1),axis=0)
        back =np.sum(np.sum(PosMat[32:,:],axis=1),axis=0)
        return (front-back)/(front+back)
    
    #angle connecting self to neighbor animal    
    def rangeVector(self):
        p1=self.centroidContour().xy
        p2=self.animal.neighbor.ts.centroidContour().xy
        angle_centroid_connect=geometry.get_angle_list(p1,p2)
        x=np.mod(180-angle_centroid_connect,360)
        return x
    
    #difference between heading and range vector
    def errorAngle(self):
        x=geometry.smallest_angle_difference_degrees(self.fish_orientation_elipse(),self.rangeVector())
        return x
    
    #generate a histogram of position over disctance from arena center
    #used in a hacky way to determine stacking order of the two animals in experiments
    #where two dishes are stacked on top of each other.
    def PolhistBins(self):
        x=np.linspace(0,self.animal.pair.max_out_venture(),100)   
        return x
        
    def Pol_n(self):
        histData=self.positionPol().y()
        histData=histData[~np.isnan(histData)]
        x=np.histogram(histData,bins=self.PolhistBins(),normed=1)[0]
        return x
        
