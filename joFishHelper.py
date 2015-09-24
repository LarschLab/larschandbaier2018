import numpy as np
#import cv2
import Tkinter
import tkFileDialog
import subprocess
import os
import matrixUtilities_joh as mu

class ExperimentMeta(object):
    def __init__(self,path,arenaDiameter=100):
        self.arenaDiameter = arenaDiameter
        self.arenaCenter = [0,0] #assign later
        self.pxPmm = 0 #assign later
        
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
    def __init__(self, positionPx, videoproperties):
        self.positionPx=positionPx
        
        self.position=self.positionPx / videoproperties.pxPmm
        self.d_position=np.diff(self.position,axis=0)
        self.dd_position=np.diff(self.d_position,axis=0)
        self.speed=np.sqrt(self.d_position[:,:,0]**2 + self.d_position[:,:,1]**2)
        self.accel=np.sqrt(self.dd_position[:,:,0]**2 + self.dd_position[:,:,1]**2)
        self.heading=mu.cart2pol(self.d_position[:,:,0],self.d_position[:,:,1])
        self.d_heading=np.diff(self.heading[0],axis=0)
        
        #inter animal distance IAD
        dist=self.position[:,0,:]-self.position[:,1,:]
        self.IAD = np.sqrt(dist[:,0]**2 + dist[:,1]**2)


       
    #calculate dependent trajectories, speed, heading etc.    
        

# while(cap.isOpened()):
#     ret, frame = cap.read()
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()