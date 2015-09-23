import numpy as np
import cv2
import Tkinter
import tkFileDialog
import subprocess
import os

class ExperimentMeta:
    def __init__(self,aviPath,arenaDiameter=100):
        self.aviPath = aviPath
        self.arenaDiameter = arenaDiameter
        self.arenaCenter = [0,0] #assign later
        self.pxPmm = 0 #assign later
        
        #get video meta data
        vp=getVideoProperties(self.aviPath) #video properties
        self.ffmpeginfo = vp
        self.videoDims = [vp['width'] , vp['height']]
        self.numFrames=vp['nb_frames']
        self.fps=vp['fps']
        self.date=vp['TAG:date']
        
        #concatenate dependent file paths
        head, tail = os.path.split(self.aviPath)
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
    
class Trajectories:
    def __init__(self, rawPx):
        self.rawPx=rawPx
        
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