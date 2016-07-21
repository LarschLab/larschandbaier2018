# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:51:08 2016

@author: jlarsch
"""
import numpy as np
from models.geometry import Trajectory
import functions.matrixUtilities_joh as mu
import scipy.stats as sta
import models.geometry as geometry
import matplotlib.pyplot as plt
import functions.peakdet as peakdet
import os


class Animal(object):
    def __init__(self,ID=0,rawTra=[],expInfo=[],shiftIndex=0,shiftRun=0):
        self.rawTra=Trajectory()
        self.expInfo=expInfo
        self.ID=ID
        self.shiftIndex=shiftIndex
        self.shiftRun=shiftRun
        self.noShiftAnimal=[]
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
        
        self.rawTra.xy=np.roll(rawTra,shiftIndex,axis=0)
        self.position.xy=(self.rawTra.xy-self.expInfo.arenaCenterPx) / self.expInfo.pxPmm
        
        if np.equal(~os.path.isfile(expInfo.aspPath),-2):
            npzfile = np.load(expInfo.aspPath)
            self.tailCurvature_mean=np.roll(npzfile['tailCurvature_mean'][:,ID],shiftIndex,axis=0)
            self.fish_orientation_elipse=np.roll(npzfile['ori'][ID,:],shiftIndex,axis=0)
            self.centroidContour=np.roll(npzfile['centroidContour'][:,:,ID],shiftIndex,axis=0)
            
            
            
            
        
        self.update_trajectory_transforms()
        
        
        #how far did the animal go out towards center?
        #this can be used to determine upper/lower animal in stacks
        
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
        self.d_pos_pol=Trajectory()
        self.d_pos_pol_rot=Trajectory()
        self.d_pos_pol_rot_cart=Trajectory()
        self.dd_pos_rot=Trajectory()
        
        self.dd_pos_pol=Trajectory()
        self.dd_pos_pol_rot=Trajectory()
        self.dd_pos_pol_rot_cart=Trajectory()
#        self.d_pos_pol.xy=np.squeeze(np.array([mu.cart2pol(*self.d_position.xy.T)])).T
#        self.d_pos_pol_rot.xy=self.d_pos_pol.xy.copy()
#        self.d_pos_pol_rot.xy=self.d_pos_pol_rot.xy[1:,:]
#        self.d_pos_pol_rot.xy[:,0]=self.d_pos_pol_rot.xy[:,0]-self.heading[:-1]
#        self.d_pos_pol_rot_cart.xy=np.squeeze(np.array([mu.pol2cart(self.d_pos_pol_rot.xy[:,0],self.d_pos_pol_rot.xy[:,1])])).T
#        self.dd_pos_rot.xy =np.diff(self.d_pos_pol_rot_cart.xy,axis=0)
        
        self.dd_pos_pol.xy=np.squeeze(np.array([mu.cart2pol(*self.dd_position.xy.T)])).T
        self.dd_pos_pol_rot.xy =self.dd_pos_pol.xy.copy()
        self.dd_pos_pol_rot.xy[:,0]=self.dd_pos_pol_rot.xy[:,0]-self.heading[:-1]
        self.dd_pos_pol_rot_cart.xy=np.squeeze(np.array([mu.pol2cart(self.dd_pos_pol_rot.xy[:,0],self.dd_pos_pol_rot.xy[:,1])])).T
        
        mapBins=np.arange(-31,32)
        self.neighborMat=np.zeros([62,62])
        #creates the neighbormat for current animal (where neighbor was)
        #this map seems flipped both horizontally and vertically! vertical is corrected at plotting.
        self.neighborMat=np.histogramdd(self.N_relPosPolRotCart.xy,bins=[mapBins,mapBins])[0]
        self.leadership()
        
        animal_dif=self.centroidContour - neighbor_animal.centroidContour
        self.dist=np.sqrt(animal_dif[:,0]**2 + animal_dif[:,1]**2)
    
    
    def getNeighborForce(self):
        #position here should be transformed (rotated, relative). [:,0,:] is the position of 
        mapBins=np.arange(-31,32)
        self.ForceMat=sta.binned_statistic_2d(self.N_relPosPolRotCart.x()[1:],self.N_relPosPolRotCart.y()[1:],self.accel,bins=[mapBins,mapBins])[0]
        self.ForceMat_speed=sta.binned_statistic_2d(self.N_relPosPolRotCart.x()[1:],self.N_relPosPolRotCart.y()[1:],self.dd_pos_pol_rot_cart.xy[:,0],bins=[mapBins,mapBins])[0]
        self.ForceMat_turn=sta.binned_statistic_2d(self.N_relPosPolRotCart.x()[1:],self.N_relPosPolRotCart.y()[1:],self.dd_pos_pol_rot_cart.xy[:,1],bins=[mapBins,mapBins])[0]

    def getRangeVector(self,neighbor_animal):
        p1=self.centroidContour
        p2=neighbor_animal.centroidContour
        angle_centroid_connect=geometry.get_angle_list(p1,p2)
        self.rangeVector=np.mod(180-angle_centroid_connect,360)
        #self.d_rangeVector=np.diff()
        #print self.centroidContour.shape
        #print self.fish_orientation_elipse.shape
        self.errorAngle=geometry.smallest_angle_difference_degrees(self.fish_orientation_elipse,self.rangeVector)
        borderMask=np.logical_and(np.abs(p1)<70, np.abs(p2)<70)
        borderMask=np.logical_and(borderMask[:,0],borderMask[:,1])
        #print np.sum(borderMask)
#        if self.noShiftAnimal:
        if self.ID==0:
            col=(0,0,1)
        else:
            col=(0,1,0)
        
        if self.noShiftAnimal:
            lw=3
        else:
            lw=1
            col=(.5,.5,.5)

            
        plt.figure(10)
        n, bins, patches =plt.hist(self.errorAngle, bins=range(-180,180,5),normed=1, histtype='step',lw=lw, color=col)
#            n, bins, patches =plt.hist(self.errorAngle[borderMask], bins=range(-180,180,1),normed=1, histtype='step',lw=2)

    
    def analyze_neighbor_interactions(self,neighbor_animal):
        self.noShiftAnimal=np.logical_and(self.shiftIndex==0,neighbor_animal.shiftIndex==0)

        self.get_neighbor_position(neighbor_animal)
        self.getNeighborForce()
        if np.equal(~os.path.isfile(self.expInfo.aspPath),-2):
            self.getRangeVector(neighbor_animal)
            self.bout_analysis()

        
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