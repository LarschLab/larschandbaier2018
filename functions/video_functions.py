# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 15:34:22 2016

@author: jlarsch
"""

import numpy as np
import subprocess
import os
import cv2
import tkFileDialog
import functions.gui_circle as gc

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
    

    
    
def get_pixel_scaling(aviPath,forceCorrectPixelScaling=0,forceInput=0):
    #pixel scaling file will typically reside in parent directory where the raw video file lives
    #forceCorrectPixelScaling=0 - force user iput if no previous data exists
    #forceInput=0 - force user input even if data exists - overwrite
    head, tail = os.path.split(aviPath)
    head=os.path.normpath(head)
    parentDir = os.path.dirname(head)
    scaleFile = os.path.join(parentDir,'bgMed_scale.csv')
    
    if np.equal(~os.path.isfile(scaleFile),-2) or forceCorrectPixelScaling:
        scaleData=np.array(np.loadtxt(scaleFile, skiprows=1,dtype=float))
        
    elif forceInput or (np.equal(~os.path.isfile(scaleFile),-1) and  forceCorrectPixelScaling):
        aviPath = tkFileDialog.askopenfilename(initialdir=parentDir,title='select video to generate median for scale information')
        bg_file=getMedVideo(aviPath)[1]
        scaleData=gc.get_circle_rois(bg_file,'_scale',forceInput)[0]
        pxPmm=scaleData['circle radius']/scaleData['arena size']
        return pxPmm
    else:
        return defaultScaling
        
    
def getMedVideo(aviPath,FramesToAvg=9,saveFile=1,forceInput=0):
    
    head, tail = os.path.split(aviPath)
    bg_file=(head+'/bgMed.tif')
    
    if np.equal(~os.path.isfile(bg_file),-2) and not forceInput:
        bg=cv2.imread(bg_file)
        try:
            bg = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)
        except:
            pass
        return bg
        
    else:
        print 'calculating median video'
        cap = cv2.VideoCapture(aviPath)
        vp=getVideoProperties(aviPath)
        videoDims = tuple([int(vp['width']) , int(vp['height'])])
        print videoDims
        #numFrames=int(vp['nb_frames'])
        numFrames=np.min([50000,int(vp['nb_frames'])])
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
            cv2.imwrite(bg_file,vidMed)

        return vidMed,bg_file