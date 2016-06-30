# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:29:56 2016

@author: jlarsch
"""

import AnimalShapeParameters
import tkFileDialog
import os
from tifffile import imsave
import numpy as np
import models.experiment as xp
import matplotlib.pyplot as plt
import models.geometry as geometry
from scipy import stats
from scipy import signal
import functions.ImageProcessor as ImageProcessor
import functions.peakdet as peakdet


import random

if 'adf' in globals():
    print af
else:
    af = tkFileDialog.askopenfilename(initialdir=os.path.normpath('d:/data/b/2016/20160314_TLStacks/1/a/1/'))
    head, tail = os.path.split(af)
    head=os.path.normpath(head)
    rotVideoPath = os.path.join(head,tail[:-4]+'_rotateBoth_mask.avi')
    csvPath = os.path.join(head,tail[:-4]+'_contour_orientation.csv')

    experiment=xp.experiment(af)
    t=experiment.Pair.get_var_from_all_animals(['rawTra','xy'])
    
#    asp2=AnimalShapeParameters.AnimalShapeParameters(af,t[:,1,:],t.shape[0])
#    asp1=AnimalShapeParameters.AnimalShapeParameters(af,t[:,0,:],t.shape[0])
    asp2=AnimalShapeParameters.AnimalShapeParameters(af,t[:,1,:],1000)
    asp1=AnimalShapeParameters.AnimalShapeParameters(af,t[:,0,:],1000)
    
    a=asp1.fish_orientation_elipse_all
    b=asp2.fish_orientation_elipse_all

    #distance between centroids
    #use as mask to flag collisions
    animal_dif=asp1.centroidContour - asp2.centroidContour
    dist=np.sqrt(animal_dif[:,0]**2 + animal_dif[:,1]**2)
    collision_frames=dist<1
    
    a[collision_frames]=np.nan
    b[collision_frames]=np.nan
    
    #compute delta heading as difference of heading to angle between centroids of the two animals
    angle_centroid_connect=get_angle_list(asp1.centroidContour,asp2.centroidContour)
    
    #flip angles along x axis for consistent angle reference
    acc_1to2=np.mod(180-angle_centroid_connect,360)
    acc_2to1=np.mod(180+acc_1to2,360)
    
    #deviation of an1 from an2 centroid
    an2_deviation=geometry.smallest_angle_difference_degrees(b,acc_2to1)  
    an1_deviation=geometry.smallest_angle_difference_degrees(a,acc_1to2)
    #mr=np.transpose(np.concatenate((np.array(asp1.frAll_rot),np.array(asp2.frAll_rot)),axis=1),[2,0,1])
    
    plt.figure()
    n, bins, patches =plt.hist(an1_deviation,bins=range(-180,180,10), normed=1, histtype='step')
    n, bins, patches =plt.hist(an2_deviation,bins=range(-180,180,10), normed=1, histtype='step')
    
    

    #imsave(rotVideoPath, mr)
    
    ang2=np.array(asp2.spineAngles)
    ang3=np.mod(ang2,180).T
    ang3[:,collision_frames]=np.nan
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(ang3)
    ax.set_aspect('auto')
        
def get_angle_list(a,b):
    angle_list=np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        p1=geometry.Vector(*a[i,:])
        p2=geometry.Vector(*b[i,:])
        angle_list[i]=geometry.Vector.get_angle(p1,p2)
    return angle_list   
        
def distance_point_line(p,l1,l2):
    distance=np.abs((l2[1]-l1[1])*p[0]-(l2[0]-l1[0])*p[1]+l2[0]*l1[1]-l2[1]*l1[0])/np.sqrt((l2[1]-l1[1])**2+(l2[0]-l1[0])**2)
    return distance
    
allDist=np.zeros(ang3.shape)
for i in range(allDist.shape[1]):
    for j in range(allDist.shape[0]):
        try:
            allDist[j,i]=distance_point_line(asp2.skel_smooth_all[i][j],asp2.skel_smooth_all[i][0],asp2.skel_smooth_all[i][-1])
        except:
            allDist[j,1]=np.nan
allDist[:,collision_frames]=np.nan

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(allDist)
ax.set_aspect('auto')



plt.figure()
plt.plot(np.nanmean(allDist,axis=0),'o-')

#detect bouts

bout_start=peakdet.detect_peaks(np.abs(d_orientation_b[:1000]),5,8)

plt.plot(xxx,np.abs(d_orientation_b)[xxx],'o')
plt.figure()
plt.plot(np.mod(np.abs(d_orientation_b),180))
    

#contour_orientation = np.array([a, b]).T
#contour_orientation_rad=np.deg2rad(contour_orientation-180)
#np.savetxt(csvPath,contour_orientation)
#    
angle_diff=geometry.smallest_angle_difference_degrees(a,b)

angle_diff_shift=[]
for i in range(10):
    bs=b.copy()
    minShift=5*6*30
    shiftIndex=int(random.uniform(minShift,bs.shape[0]-minShift))
    bs=np.roll(bs,shiftIndex)
    angle_diff_shift.append(geometry.smallest_angle_difference_degrees(a,bs))
#
#    
#    aa=experiment.Pair.heading[:,0,0]
#    bb=experiment.Pair.heading[:,1,0]
#    
#    #plt.plot(asp1.fish_orientation_elipse_all[1:4000])
#    #plt.plot(asp2.fish_orientation_elipse_all[1:4000])
#    #plt.plot(np.degrees(aa)+180)
#    #plt.scatter(np.degrees(aa[:-1])+180,asp1.fish_orientation_elipse_all[1:])
#    ac=a.copy()
#    ac[np.isnan(ac)]=0
#    bc=b.copy()
#    bc[np.isnan(bc)]=0
#    c=np.correlate(bc,ac,mode='full')
#    plt.figure()
#    plt.plot(c)
#    c=np.correlate(ac,bc,mode='full')
#    plt.plot(c)
#    plt.figure()
#    c=np.correlate(bb,aa,mode='full')
#    plt.plot(c)
#    c=np.correlate(aa,bb,mode='full')
#    plt.plot(c)
#
#chunkLength=60*30
#chunk_lag=[]
#chunk_lead=[]
#chunk_IADm=[]
#for i in range(2,np.int((a.shape[0])/chunkLength)-2):
#    c_range1=range((i-1)*chunkLength,(i+2)*chunkLength-1)
#    c_range2=range(i*chunkLength,(i+1)*chunkLength-1)
#    c_corr=np.correlate(bc[c_range1],ac[c_range2],mode='same')
#    c_lag=np.argmax(c_corr)-np.shape(c_range2)[0]
#    chunk_lag.append(c_lag)
#    c_neighborMat,c_relPosRot,c_relPos=joFishHelper.getRelativeNeighborPositions(experiment.Pair.position[c_range2,:,:],contour_orientation_rad[c_range2[1:]])
#    c_lead=joFishHelper.leadership(c_neighborMat)
#    chunk_lead.append(c_lead)
#    chunk_IADm.append(np.nanmean(experiment.Pair.IAD[c_range2]))
#    
#leadReal=joFishHelper.leadership(experiment.Pair.neighborMat)
#lead_shifted=[]
#lead_shifted.append([joFishHelper.leadership(y) for y in ([x.neighborMat for x in experiment.sPair.sPair])])
#
#
##plt.scatter(np.array(chunk_lead)[:,0],np.array(chunk_lag))
##plt.plot([0, 0], [-300, 300], 'k:', lw=1)
##plt.plot([-1, 1], [0, 0], 'k:', lw=1)
##plt.figure()
##plt.plot(chunk_lag,'o')
#plt.figure()
#plt.plot(chunk_lead)
#
#plt.figure()
#plt.plot(chunk_IADm)
#
d_orientation_a=np.diff(a)
d_orientation_b=np.diff(b)
plt.figure()
plt.plot(np.abs(d_orientation_b))




