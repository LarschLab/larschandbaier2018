import numpy as np
#import cv2
import subprocess
import os
import matrixUtilities_joh as mu
import random
import scipy.io
import matplotlib.pyplot as plt


class ExperimentMeta(object):
    #This class collects paths, arena and video parameters
    def __init__(self,path,arenaDiameter_mm=100):
        self.arenaDiameter_mm = arenaDiameter_mm
        self.arenaCenterPx = [0,0] #assign later from class 'Pair'
        self.pxPmm = 0 #assign later from class 'Pair'
        
        #If a video file name is passed, collect video parameters
        if path.endswith('.avi'):
            self.aviPath = path
            #get video meta data
            vp=getVideoProperties(path) #video properties
            self.ffmpeginfo = vp
            self.videoDims = [vp['width'] , vp['height']]
            self.numFrames=vp['nb_frames']
            self.fps=vp['fps']
            self.date=vp['TAG:date']
        else:
            self.fps=30
        
        #concatenate dependent file paths (trajectories, pre-analysis)
        head, tail = os.path.split(path)
        head=os.path.normpath(head)
        self.trajectoryPath = os.path.join(head,'trajectories_nogaps.mat')
        self.dataPath = os.path.join(head,'analysisData.mat')
        
def getVideoProperties(aviPath):
    #read video metadata via ffprobe and parse output
    #can't use openCV because it reports tbr instead of fps (frames per second)
    cmnd = ['c:/ffmpeg/bin/ffprobe', '-show_format', '-show_streams', '-pretty', '-loglevel', 'quiet', aviPath]
    p = subprocess.Popen(cmnd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err =  p.communicate()
    decoder_configuration = {}
    for line in out.splitlines():
        if '=' in line:
            key, value = line.split('=')
            decoder_configuration[key] = value
    
    #frame rate needs special treatment. calculate from parsed str argument        
    nominator,denominator=decoder_configuration['avg_frame_rate'].split('/')
    decoder_configuration['fps']=int(float(nominator) / float(denominator))
    return decoder_configuration
    

class Pair(object):
    #Class for trajectories of one pair of animals.
    #Calculation and storage of dependent time series: speed, heading, etc.
    def __init__(self, trajectories, expInfo):
        self.positionPx=trajectories
        maxPixel=np.nanmax(self.positionPx,0)
        minPixel=np.nanmin(self.positionPx,0)
        arenaDiameterPx=np.mean(maxPixel-minPixel)
        expInfo.pxPmm=arenaDiameterPx/expInfo.arenaDiameter_mm
        expInfo.arenaCenterPx=np.mean(maxPixel-(arenaDiameterPx/2),axis=0)
        
        self.position=(self.positionPx-expInfo.arenaCenterPx) / expInfo.pxPmm
        self.d_position=np.diff(self.position,axis=0)
        self.dd_position=np.diff(self.d_position,axis=0)
        self.speed=np.sqrt(self.d_position[:,:,0]**2 + self.d_position[:,:,1]**2)
        self.accel=np.sqrt(self.dd_position[:,:,0]**2 + self.dd_position[:,:,1]**2)
        self.heading=np.transpose(mu.cart2pol(self.d_position[:,:,0],self.d_position[:,:,1]),[1,2,0]) #heading[0] = heading, heading[1] = speed
        self.d_heading=np.diff(self.heading[0],axis=0)
        
        #absolute inter animal distance IAD
        dist=self.position[:,0,:]-self.position[:,1,:]
        self.IAD = np.sqrt(dist[:,0]**2 + dist[:,1]**2)
        self.IAD_m=np.nanmean(self.IAD)
        
        #relative distance between animals
        self.neighborMat,self.relPosPolRotCart,self.relPosPolRot,self.relPos,self.relPosPol = getRelativeNeighborPositions(self.position,self.heading)

def getRelativeNeighborPositions(trajectory,heading):
    tra=trajectory[1:,:,:].copy() #using *1 to obtain a new copy rather than reference
    relPos2=tra[:,1,:]-tra[:,0,:]  
    relPos1=tra[:,0,:]-tra[:,1,:]
    relPos=np.transpose([relPos1,relPos2],[1,0,2])
    
    relPosPol=np.transpose(mu.cart2pol(relPos[:,:,0],relPos[:,:,1]),[1,2,0])
    relPosPolRot=relPosPol.copy()
    relPosPolRot[:,0,0]=relPosPol[:,0,0]-(heading[:,0,0])
    relPosPolRot[:,1,0]=relPosPol[:,1,0]-(heading[:,1,0])
    relPosPolRotCart=mu.pol2cart(relPosPolRot[:,:,0],relPosPolRot[:,:,1])
    relPosPolRotCart=np.transpose(relPosPolRotCart,[1,2,0])
    
    neighborMat=np.histogramdd(relPosPolRotCart[:,0,:],bins=[np.arange(-31,32),np.arange(-31,32)])
    #AccPol=mu.cart2pol(tra.d_position)
    
    return neighborMat,relPosPolRotCart,relPosPolRot,relPos,relPosPol



class shiftedPair(object):
    #Class to calculate null hypothesis time series using shifted pairs of animals
    def __init__(self, tra,expInfo):
        self.nRuns=10
        self.minShift=5*60*expInfo.fps
        self.sPair=[]
        #generate nRuns instances of Pair class with one animal time shifted against the other
        for i in range(self.nRuns):
            traShift=tra.positionPx
            shiftIndex=int(random.uniform(self.minShift,traShift.shape[0]-self.minShift))
            #time-rotate animal 0, keep animal 1 as is
            traShift[:,0,:]=np.roll(traShift[:,0,:],shiftIndex,axis=0)
            self.sPair.append(Pair(traShift,expInfo))
        #calculate mean and std IAD for the goup of shifted instances
        self.spIAD_m = np.nanmean([x.IAD_m for x in self.sPair])
        self.spIAD_std = np.nanstd([x.IAD_m for x in self.sPair])
       
class experiment(object):
    #Class to collect, store and plot data belonging to one experiment
    def __init__(self,path):
        self.expInfo=ExperimentMeta(path)
        mat=scipy.io.loadmat(self.expInfo.trajectoryPath)
        rawTra=mat['trajectories']
        #take out nan in the beginnin
        nanInd=np.max(np.where(np.isnan(rawTra)))+1
        rawTra=rawTra[nanInd:,:,:]
        self.Pair=Pair(rawTra,self.expInfo)
        
        #generate shifted control 'mock' pairs
        self.sPair=shiftedPair(self.Pair,self.expInfo)        
        
        plt.subplot(331)
        plt.cla()
        plt.plot(self.Pair.position[:,0,0],self.Pair.position[:,0,1],'b.',markersize=1,alpha=0.1)
        plt.plot(self.Pair.position[:,1,0],self.Pair.position[:,1,1],'r.',markersize=1,alpha=0.1)
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        plt.title('raw trajectories')     
        
        #plot IAD time series
        x=np.arange(float(np.shape(self.Pair.IAD)[0]))/(self.expInfo.fps*60)
        plt.subplot(312)
        plt.cla()
        plt.plot(x,self.Pair.IAD,'b.',markersize=0.2)
        plt.xlim([0, 90]) 
        plt.xlabel('time [minutes]')
        plt.ylabel('IAD [mm]')
        plt.title('Inter Animal Distance over time')
        
        plt.subplot(332)
        plt.cla()
        #get rid of nan
        IAD=self.Pair.IAD
        IAD=IAD[~np.isnan(IAD)]
        n, bins, patches = plt.hist(IAD, 30, normed=1, histtype='stepfilled') 
        plt.xlabel('IAD [mm]')
        plt.ylabel('p')
        plt.title('IAD')
        
        plt.subplot(333)
        plt.cla()
        x=[1,2]
        y=[self.Pair.IAD_m,self.sPair.spIAD_m]
        yerr=[0,self.sPair.spIAD_std]
        plt.bar(x, y, yerr=yerr, width=0.5,color='k')
        lims = plt.ylim()
        plt.ylim([0, lims[1]]) 
        plt.xlim([0.5, 3]) 
        plt.ylabel('[mm]')
        plt.xticks([1.25,2.25],['raw','shift'])
        plt.tight_layout()
        plt.title('mean IAD')
        plt.show()
        
        plt.subplot(337)
        plt.cla()
        plt.imshow(self.Pair.neighborMat[0],interpolation='none', extent=[-31,31,-31,31])
        #plt.ylim([-150, 150]) 
        #plt.xlim([-150, 150]) 