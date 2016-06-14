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
import joFishHelper
import matplotlib.pyplot as plt
import geometry
from scipy import stats
from scipy import signal
import ImageProcessor
import matplotlib.pyplot as plt


import random

if 'adf' in globals():
    print af
else:
    af = tkFileDialog.askopenfilename(initialdir=os.path.normpath('d:/data/b/2016/20160314_TLStacks/1/a/1/'))
    head, tail = os.path.split(af)
    head=os.path.normpath(head)
    rotVideoPath = os.path.join(head,tail[:-4]+'_rotateBoth_mask.avi')
    csvPath = os.path.join(head,tail[:-4]+'_contour_orientation.csv')

    experiment=joFishHelper.experiment(af)
    t=experiment.Pair.positionPx
    
#    asp2=AnimalShapeParameters.AnimalShapeParameters(af,t[:,1,:],t.shape[0])
#    asp1=AnimalShapeParameters.AnimalShapeParameters(af,t[:,0,:],t.shape[0])
    asp2=AnimalShapeParameters.AnimalShapeParameters(af,t[:,1,:],200)
    asp1=AnimalShapeParameters.AnimalShapeParameters(af,t[:,0,:],200)
    
    
    
    mr=np.concatenate((np.array(asp1.frAll_rot),np.array(asp2.frAll_rot)),axis=1)
    
    
    imsave(rotVideoPath, mr)
    
    ang2=np.array(asp2.spineAngles)
    ang3=np.mod(ang2,180).T
    plt.imshow(ang3)
    
    #for i in ang3.shape[1]:
        
        
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
plt.figure()
plt.imshow(allDist)
plt.figure()
plt.plot(np.nanmean(allDist,axis=0),'o-')
    
a=asp1.fish_orientation_elipse_all
b=asp2.fish_orientation_elipse_all
contour_orientation = np.array([a, b]).T
contour_orientation_rad=np.deg2rad(contour_orientation-180)
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
xxx=detect_peaks(np.abs(d_orientation_b[:1000]),5,8)
plt.plot(xxx,np.abs(d_orientation_b)[xxx],'o')