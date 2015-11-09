import numpy as np
#import cv2
import subprocess
import os
import matrixUtilities_joh as mu
import random

class ExperimentMeta(object):
    def __init__(self,path,arenaDiameter_mm=100):
        self.arenaDiameter_mm = arenaDiameter_mm
        self.arenaCenterPx = [0,0] #assign later from class 'Pair'
        self.pxPmm = 0 #assign later from class 'Pair'
        
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
        
        #concatenate dependent file paths
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
    def __init__(self, trajectories, expInfo):
        

        self.positionPx=trajectories
        maxPixel=np.nanmax(self.positionPx,0)
        minPixel=np.nanmin(self.positionPx,0)
        arenaDiameterPx=np.mean(maxPixel-minPixel)
        expInfo.pxPmm=arenaDiameterPx/expInfo.arenaDiameter_mm
        expInfo.arenaCenterPx=maxPixel-(arenaDiameterPx/2)
        
        self.position=(self.positionPx-expInfo.arenaCenterPx) / expInfo.pxPmm
        self.d_position=np.diff(self.position,axis=0)
        self.dd_position=np.diff(self.d_position,axis=0)
        self.speed=np.sqrt(self.d_position[:,:,0]**2 + self.d_position[:,:,1]**2)
        self.accel=np.sqrt(self.dd_position[:,:,0]**2 + self.dd_position[:,:,1]**2)
        self.heading=mu.cart2pol(self.d_position[:,:,0],self.d_position[:,:,1])
        self.d_heading=np.diff(self.heading[0],axis=0)
        
        #inter animal distance IAD
        dist=self.position[:,0,:]-self.position[:,1,:]
        self.IAD = np.sqrt(dist[:,0]**2 + dist[:,1]**2)
        self.IAD_m=np.nanmean(self.IAD)


class shiftedPair(object):
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
       