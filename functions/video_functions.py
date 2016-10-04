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
            


        def clip8bit(a):
            a[a>255]=255
            a[a<0]=0
            a[~np.isfinite(a)]=255
            return a.astype('uint8')
            
        def bgDiv8bit(a,b):
            
            tmp=255*(a.astype('float')/b.astype('float'))
            return clip8bit(tmp)
        
        def stretchNorm(x):
            return (x-x.min()).astype('float')/(x.max()-x.min())
            
        def stretchRange(x,mi,ma):
            return (x-mi).astype('float')/(ma-mi)
            
        def norm(x):
            return x.astype('float')/x.max()
            
        def stretchDirect(x,mi,ma):
            xf=x.astype('float')
            mif=float(mi)
            maf=float(ma)
            return clip8bit((xf-mif)*(maf/(maf-mif)))
        
        vidMed=(np.median(allMed,axis=2)).astype('uint8')
        cv2.imwrite(bg_file,vidMed)
        vidMed=cv2.imread(bg_file)
        vidMed = cv2.cvtColor(vidMed, cv2.COLOR_RGB2GRAY)
        vidMed=np.expand_dims(vidMed,axis=2)
        
        allMed_bgSub=bgDiv8bit(allMed,vidMed)
#        cv2.imshow('allMed_bgSub',allMed_bgSub[:,:,0])        
#        allMed_bgSub_norm=(norm(allMed_bgSub)*255).astype('uint8')
        
#        allMed_bgSub_norm=allMed_bgSub.copy()
#        allMed_bgSub_norm[allMed_bgSub_norm>1]=1
#        allMed_bgSub_norm=norm(allMed_bgSub.copy())
#        allMed_bgSub_norm=(allMed_bgSub_norm*255)
        
        #cv2.imshow('allMed_bgSub_norm',allMed_bgSub_norm[:,:,0])
        
        minval2=np.min(allMed_bgSub)-5 #allow for some buffer to the range
#        minval2=np.min(allMed) 
        #maxval2=np.max(allMed_bgSub)
        
#        allMedStretch=(stretchRange(allMed,minval2,maxval2)*255).astype('uint8')
        allMedStretch=stretchDirect(allMed,minval2,255)
        #allMedStretch[allMedStretch<1]=1
#        cv2.imshow('allMedStretch',allMedStretch[:,:,0])
        
#        vidMedStretch=(np.median(allMedStretch,axis=2)).astype('uint8')
        vidMedStretch=stretchDirect(vidMed,minval2,250)
#        vidMedStretch=np.expand_dims(vidMedStretch,axis=2)
        
        
#        vidMedStretch[vidMedStretch<1]=1
#        cv2.imshow('vidMedStretch',vidMedStretch)
        
        allMedStretch_bgSub=bgDiv8bit(allMedStretch,vidMedStretch)
#        allMedStretch_bgSub[allMedStretch_bgSub>1]=1
#        allMedStretch_bgSub=norm(allMedStretch_bgSub)*255
#        allMedStretch_bgSub[allMedStretch_bgSub<1]=1
        
        print type(allMedStretch_bgSub)
        
        
        if saveFile:
            
            bg_file=(head+'/bgMed_stretch.tif')
            cv2.imwrite(bg_file,vidMedStretch)
            av_file=(head+'/bgcorrect.avi')
            fourcc=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
            writer = cv2.VideoWriter(av_file, fourcc, 30, (vidMed.shape[0],vidMed.shape[1]))

            for i in range(allMedStretch_bgSub.shape[2]):
                print i
                x=(np.squeeze(allMedStretch_bgSub[:,:,i])).astype('uint8')
                writer.write(x)

#            cv2.imshow('allMedStretch_bgSub',x)
            bg_file=(head+'/bgMed_correctedframe.tif')
            cv2.imwrite(bg_file,x)
            writer.release()

        return vidMed,bg_file,minval2