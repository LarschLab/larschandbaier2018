# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 15:34:22 2016

@author: jlarsch
"""

import numpy as np
import subprocess
import os
import cv2



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
    
    
def getMedVideo(aviPath,FramesToAvg,saveFile):
    cap = cv2.VideoCapture(aviPath)
    head, tail = os.path.split(aviPath)
    vp=getVideoProperties(aviPath)
    videoDims = tuple([int(vp['width']) , int(vp['height'])])
    print videoDims
    #numFrames=int(vp['nb_frames'])
    numFrames=50000
    img1=cap.read()
    img1=cap.read()
    img1=cap.read()
    img1=cap.read()
    gray = cv2.cvtColor(img1[1], cv2.COLOR_BGR2GRAY)
    allMed=gray.copy()
    for i in range(10,numFrames-2,np.round(numFrames/FramesToAvg)): #use FramesToAvg images to calculate median
        cap.set(cv2.CAP_PROP_POS_FRAMES,i)
        image=cap.read()
        print i
        gray = cv2.cvtColor(image[1], cv2.COLOR_BGR2GRAY)  
        allMed=np.dstack((allMed,gray))
        
    vidMed=np.median(allMed,axis=2)

    if saveFile:
        ImOutFile=(head+'/bgMed.tif')
        cv2.imwrite(ImOutFile,vidMed)
        return 1
    else:
        return vidMed