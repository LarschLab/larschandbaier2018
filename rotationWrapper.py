# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:29:56 2016

@author: jlarsch
"""

import models.AnimalShapeParameters as AnimalShapeParameters
import tkFileDialog
import os
#from tifffile import imsave
import numpy as np
import models.experiment as xp
import matplotlib.pyplot as plt
import models.geometry as geometry
import multiprocessing as mp
import functions.peakdet as peakdet
import pickle
import scipy.stats as sta

if __name__ == '__main__':
    mp.freeze_support()
    reread=1
    numframes=5000 #0 = all

         
    if reread:
        af = tkFileDialog.askopenfilename(initialdir=os.path.normpath('D:/data/b/2016/20160311_arenaSize/b/1/'))
        head, tail = os.path.split(af)
        head=os.path.normpath(head)
        rotVideoPath = os.path.join(head,tail[:-4]+'_rotateBoth_mask.avi')
        csvPath = os.path.join(head,tail[:-4]+'_contour_orientation.csv')
        
        experiment=xp.experiment(af)
        pair_trajectories=experiment.Pair.get_var_from_all_animals(['rawTra','xy'])
        asp=AnimalShapeParameters.vidSplit(af,pair_trajectories,numframes,1)
        
    else:
        pf = tkFileDialog.askopenfilename(initialdir=os.path.normpath('D:/data/b/2016/20160311_arenaSize/b/1/'))
        f=open(pf,'r')
        asp_f=pickle.load(f)
        if type(asp_f[0]) is list: #this collection still contains individual animal shape objects for each frame
            asp=[]            
            asp.append(AnimalShapeParameters.AnimalShapeParameters(asp_f,0))
            asp.append(AnimalShapeParameters.AnimalShapeParameters(asp_f,1))
            asp=AnimalShapeParameters.asp_cleanUp(asp)
            
            pickleFile=pf[:-7]+'_+'.pickle
            AnimalShapeParameters.save_asp(pickleFile,asp)
        else:
            asp=asp_f
#b=hhh[0] 
#
    plt.figure()
    n, bins, patches =plt.hist(asp[0].deviation,bins=range(-180,180,1), normed=1, histtype='step')
    n, bins, patches =plt.hist(asp[1].deviation,bins=range(-180,180,1), normed=1, histtype='step')
    
    animal_dif=asp[0].centroidContour - asp[1].centroidContour
    dist=np.sqrt(animal_dif[:,0]**2 + animal_dif[:,1]**2)
    near=dist<260
    
    
    plt.figure()

    n1, bins, patches =plt.hist(asp[0].deviation[near],bins=range(-180,180,1), normed=1, histtype='step')
    n2, bins, patches =plt.hist(asp[1].deviation[near],bins=range(-180,180,1), normed=1, histtype='step')
    
    
    ad=asp[0].allDist.T
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(ad)
    ax.set_aspect('auto')
    
    plt.figure()
    tail_curve_1=np.nanmean(asp[0].allDist.T,axis=0)
    plt.plot(tail_curve_1,'-') 
    #detect bouts
    bout_start=peakdet.detect_peaks(tail_curve_1,1,8)
    plt.plot(bout_start,tail_curve_1[bout_start],'o')

    d_head_sm=geometry.smallest_angle_difference_degrees(asp[0].fish_orientation_elipse[:-1],asp[0].fish_orientation_elipse[1:])
    d_head_sm=np.abs(d_head_sm)
    d_heading_norm=d_head_sm/np.nanmax(d_head_sm)*np.nanmax(tail_curve_1)
    plt.plot(d_heading_norm,'-') 

    plt.figure()
    plt.hist(d_head_sm,bins=range(360))

    


    
    ##bout analysis
    i=0
    orientation_before=np.nanmedian([asp[0].deviation[b-6:b-3] for b in bout_start[1:-2]],axis=1)
    orientation_after=np.nanmedian([asp[0].deviation[b+3:b+6] for b in bout_start[1:-2]],axis=1)
    d_ori=np.diff(np.abs([orientation_before,orientation_after]).T,axis=1)
    #d_ori=geometry.smallest_angle_difference_degrees(orientation_before,orientation_after)
    ##alternative d_ori 
    #considering instantaneous positions of both animals during the bout.    
    plt.figure()   
    n1, bins, patches =plt.hist(d_ori,bins=range(-180,180,1), normed=1, histtype='step')    
    asp=AnimalShapeParameters.asp_cleanUp(asp)
    baseline1=[asp[0].baseline[b] for b in bout_start]
    heading_before=np.nanmedian([asp[0].fish_orientation_elipse[b-6:b-3] for b in bout_start[1:-2]],axis=1)
    heading_after=np.nanmedian([asp[0].fish_orientation_elipse[b+3:b+6] for b in bout_start[1:-2]],axis=1)
    bearing_before=geometry.smallest_angle_difference_degrees(heading_before,baseline1[1:-2]) 
    bearing_after=geometry.smallest_angle_difference_degrees(heading_after,baseline1[1:-2]) 
    d_ori_inst=np.diff(np.abs([bearing_before,bearing_after]).T,axis=1)
    
    plt.figure()
    n1, bins, patches =plt.hist(bearing_before,bins=range(-180,180,10), normed=1, histtype='step')    
    n1, bins, patches =plt.hist(bearing_after,bins=range(-180,180,10), normed=1, histtype='step')    

    dist_at_bout=dist[bout_start]
    near=dist_at_bout<260
    
    plt.figure()
    n1, bins, patches =plt.hist(bearing_before[near],bins=range(-180,180,10), normed=1, histtype='step')    
    n1, bins, patches =plt.hist(bearing_after[near],bins=range(-180,180,10), normed=1, histtype='step')    

    
    plt.figure()   
    n1, bins, patches =plt.hist(d_ori_inst,bins=range(-180,180,1), normed=1, histtype='step')    
    plt.plot([0,0], [0, np.max(n1)], 'k:', lw=3)
    
    plt.figure()   
    n1, bins, patches =plt.hist(d_ori,bins=range(-180,180,1), normed=1, histtype='step')    
    plt.plot([0,0], [0, np.max(n1)], 'k:', lw=3)
    
    #is bout-directed-ness modulated with animal distance?    
    dist_at_bout=dist[bout_start[1:-2]]
        
    plt.figure()
    plt.scatter(dist_at_bout,d_ori_inst)
    
    plt.figure()   
    n1, bins, patches =plt.hist(d_ori_inst[dist_at_bout<260],bins=range(-180,180,1), normed=1, histtype='step')   
    plt.plot([0,0], [0, np.max(n1)], 'k:', lw=3)

    mapBinsx=np.arange(0,1500,2)
    mapBinsy=np.arange(-120,120,10)

    bm=sta.binned_statistic(dist_at_bout,d_ori_inst[:,0],bins=mapBinsx,statistic='mean')[0]
    plt.figure()
    plt.plot(mapBinsx[:-1]+5,bm)
    plt.plot([mapBinsx[0],mapBinsx[-1]], [0, 0], 'k:', lw=3)

    bma=np.zeros((mapBinsx.shape[0]-1,mapBinsy.shape[0]-1))

    for i in range(mapBinsx.shape[0]-1):
        mask1=np.multiply(dist_at_bout[:-1]>mapBinsx[i] , dist_at_bout[:-1]<mapBinsx[i+1])
        
        bma[i,:]=np.histogram(d_ori_inst[mask1,0],bins=mapBinsy,normed=1)[0]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ax.imshow(bma.T, extent=[mapBinsx[0],mapBinsx[-1],mapBinsy[0],mapBinsy[-1]],origin='lower')
    ax.set_aspect('auto')
    plt.plot([mapBinsx[0],mapBinsx[-1]], [0, 0], 'k:', lw=3)
    
    def movingaverage(interval, window_size):
        window = np.ones(int(window_size))/float(window_size)
        return np.convolve(interval, window, 'same')
    
    bm_av=movingaverage(bm,5)
    plt.plot(mapBinsx[:-1]+5,bm_av,'r', lw=4)
    plt.ylim([mapBinsy[0],mapBinsy[-1]])
    plt.xlim([mapBinsx[0],mapBinsx[-1]])
    
    #is bout interval modulated with animal distance? -->seems not!    
    bout_interval=np.diff(bout_start)
    dist_at_bout=dist[bout_start]
        
    plt.figure()
    plt.scatter(dist_at_bout[:-1],bout_interval)
    
    mapBins=np.arange(0,2000,50)
    bm=sta.binned_statistic(dist_at_bout[:-1],bout_interval,bins=mapBins)[0]
    plt.figure()
    plt.plot(bm)
    
    mapBinsx=np.arange(0,2000,50)
    mapBinsy=np.arange(0,100,5)
    
    bma=np.zeros((mapBinsx.shape[0]-1,mapBinsy.shape[0]-1))

    for i in range(mapBinsx.shape[0]-1):
        mask=np.multiply(dist_at_bout[:-1]>mapBinsx[i] , dist_at_bout[:-1]<mapBinsx[i+1])
        bma[i,:]=np.histogram(bout_interval[mask],bins=mapBinsy,normed=1)[0]
    plt.figure()
    plt.imshow(bma.T)
    
    #is bout interval modulated with relative orientation? (only considering 'close animals')
    #--> maybe
    dist_at_bout=dist[bout_start]
    bout_interval=np.diff(bout_start)
    dev_at_bout=asp[0].deviation[bout_start]
    distLim=260
    distMask=dist_at_bout<distLim
    dev_at_bout=dev_at_bout[distMask]
    bout_interval=bout_interval[distMask[:-1]]
        
    plt.figure()
    plt.scatter(dev_at_bout,bout_interval)
    
    mapBinsx=np.arange(-180,180,20)
    mapBinsy=np.arange(0,40,1)

    bm=sta.binned_statistic(dev_at_bout,bout_interval,bins=mapBinsx)[0]
    plt.figure()
    plt.plot(mapBinsx[:-1]+5,bm)
    

    
    bma=np.zeros((mapBinsx.shape[0]-1,mapBinsy.shape[0]-1))

    for i in range(mapBinsx.shape[0]-1):
        mask1=np.multiply(dev_at_bout[:-1]>mapBinsx[i] , dev_at_bout[:-1]<mapBinsx[i+1])
        
        bma[i,:]=np.histogram(bout_interval[mask1],bins=mapBinsy,normed=1)[0]
    plt.figure()
    plt.imshow(bma.T)
    
    
    
    
    #
#    
#
##contour_orientation = np.array([a, b]).T
##contour_orientation_rad=np.deg2rad(contour_orientation-180)
##np.savetxt(csvPath,contour_orientation)
##    
#angle_diff=geometry.smallest_angle_difference_degrees(a,b)
#
#angle_diff_shift=[]
#for i in range(10):
#    bs=b.copy()
#    minShift=5*6*30
#    shiftIndex=int(random.uniform(minShift,bs.shape[0]-minShift))
#    bs=np.roll(bs,shiftIndex)
#    angle_diff_shift.append(geometry.smallest_angle_difference_degrees(a,bs))
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