# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 10:55:23 2017

@author: jlarsch
"""

import cv2
import tkFileDialog
import os
import matplotlib.pyplot as plt
import models.experiment as xp
import glob
import numpy as np
import pandas as pd
#import datetime

class settings(object):

    def __init__(self, startFrame=0,
                 endFrame=10):
        self.startFrame=startFrame
        self.endFrame=endFrame      
        self.currFrame=5000
        self.run=False
        self.vidRec=False
        self.haveVid=False

class vidGui(object):

    window_name = "vidGUI"
    def __init__(self, path,anMat,sMat=[],df_roi=[]):

        self.settings=settings()
        #self.df=df
        #self.skipNan=e1.skipNanInd
        self.path = path
        self.cap = cv2.VideoCapture(path)
        self.im=[]
        self.df_roi=df_roi
        self.anMat=anMat
        self.sMat=sMat
        #self.t1=e1.rawTra[:,0,:].copy()
        #self.t2=e2.rawTra[:,0,:].copy()
        #self.t1b=e1.rawTra[:,1,:].copy()
        #self.t2b=e2.rawTra[:,1,:].copy()
        #self.t1[:,0]=self.t1[:,0]+512        
        #self.t1b[:,0]=self.t1b[:,0]+512  
        cv2.namedWindow(self.window_name,cv2.WINDOW_AUTOSIZE)
        self.desiredWidth=1024
        self.desiredheight=1024
        
        #cv2.createTrackbar("Min", "Threshold", &threshMin, 255, on_trackbar);

        #cv2.namedWindow(self.window_name,cv2.WINDOW_AUTOSIZE)
        length = 30000#int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cv2.setMouseCallback(self.window_name, self.startStop)
        cv2.createTrackbar("startFrame", self.window_name,
                           self.settings.startFrame, 1000,
                           self.set_startFrame)
        cv2.createTrackbar("endFrame", self.window_name,
                           self.settings.endFrame, 1000,
                           self.set_endFrame)
        cv2.createTrackbar("currFrame", self.window_name,
                           self.settings.currFrame, length,
                           self.set_currFrame)
        #cv2.resizeWindow(self.window_name, self.desiredWidth,self.desiredheight)               
    def startStop(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.settings.run=True
            while self.settings.run:
                f=self.settings.currFrame
                f+=1
                
            
                key = cv2.waitKey(1) & 0xFF
                 
                if key == ord("v"):
                  
                    self.settings.vidRec = not self.settings.vidRec
                    
                    if self.settings.vidRec:
                        if not self.settings.haveVid:
                                             
                            p, tail = os.path.split(self.path)
                            #fn=p+"\\episodeVid_frame_%07d_%s.avi"%(f,self.currEpi)
                            fn=p+"\\episodeVid_frame_%07d.avi"%(f)
                            fn=os.path.normpath(fn)
                            fourcc=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
                            fps=30
                            self.vOut = cv2.VideoWriter(fn,fourcc,fps,self.im.transpose(1,0,2).shape[0:2],True)
                            self.settings.haveVid=True
                    else:
                        if self.settings.haveVid:
                            self.vOut.release()
                            self.settings.haveVid=False
     
                if key == ord("s"):
                    self.settings.run=False
                cv2.setTrackbarPos("currFrame", self.window_name,f)
                
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.settings.run=False
                      
    def set_startFrame(self, f1):
        self.settings.startFrame = f1
        self.updateFrame()

    def set_endFrame(self, f2):
        self.settings.endFrame = f2
        self.updateFrame()

    def set_currFrame(self, f3):
        
        self.settings.currFrame = f3
        self.updateFrame()
    def updateFrame(self):
        
         
        tail=1
        tailStep=1.0
        #dotScale=1
        pathDotSize=4
        stimDotSize=4
        stimDotSize2=4
        self.cap.set(cv2.CAP_PROP_POS_FRAMES,self.settings.currFrame)
        self.im=255-self.cap.read()[1]
        fs= self.settings.currFrame-np.mod(self.settings.currFrame,tailStep)
        #self.currEpi=self.df['episode'].ix[np.where(self.df['epStart']>fs)[0][0]-1][1:]
        r=np.arange(fs-tail,fs,tailStep).astype('int')
        s=sMat[0,self.settings.currFrame,0]
        
        if ~np.isnan(s):
            stimDotSize=int(s)
            stimDotSize2=int(sMat[0,self.settings.currFrame,1])
        #DRAW path history
        #for f in r:
        for f in [self.settings.currFrame]:
            opacity=((fs-f)/float(tail))
            
            for an in range(4):
                center=tuple(self.anMat[an,f,[0,1]].astype('int'))
                cv2.circle(self.im, center,pathDotSize , (opacity*255,opacity*255,255), -1) 
                #center=tuple(self.anMat[an,f,[2,3]].astype('int'))
                #cv2.circle(self.im, center, stimDotSize, (opacity*255,opacity*255,opacity*255), -1) 
                #center=tuple(self.anMat[an,f,[4,5]].astype('int'))
                #cv2.circle(self.im, center, stimDotSize2, (opacity*255,opacity*255,opacity*255), -1)              

        
        #DRAW Current frame animal positions 
        for an in range(4):
            opacity=0
            center=tuple(self.anMat[an,f,[0,1]].astype('int'))
            cv2.circle(self.im, center, 6, (0,1,0), -1)
            center=tuple(self.anMat[an,f,[2,3]].astype('int'))
            cv2.circle(self.im, center, stimDotSize, (opacity*255,opacity*255,opacity*255), -1) 
            center=tuple(self.anMat[an,f,[4,5]].astype('int'))
            cv2.circle(self.im, center, stimDotSize2, (opacity*255,opacity*255,opacity*255), -1)  
           

        #if 'skype' in self.currEpi:
        #    cv2.circle(self.im, center, 6, (255,opacity*255,opacity*255), 1)
        

        
        #DRAW DISH BOUNDARIES
        for index, row in self.df_roi.iterrows():
            center=tuple((row.x_center,row.y_center))
            rad=row.radius
            cv2.circle(self.im, center, rad, (0,0,0), 2)
        #center=tuple((256+512,256))
        #cv2.circle(self.im, center, 240, (0,0,0), 2)
        #center=tuple((256+2*512,256))
        #cv2.circle(self.im, center, 240, (0,0,0), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(self.im,self.currEpi,(450,20), font, 0.6,(0,0,0),2)  
        
        frInfo='Frame #: '+str(self.settings.currFrame)
        cv2.putText(self.im,frInfo,(450,40), font, 0.4,(0,0,0),2)
        
        miliseconds=self.settings.currFrame*(100/3.0)
        seconds,ms=divmod(miliseconds, 1000)
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        timeInfo= "%02d:%02d:%02d:%03d" % (h, m, s,ms)
        cv2.putText(self.im,timeInfo,(450,60), font, 0.4,(0,0,0),2)
        timeInfo2='(hh:mm:ss:ms)'
        cv2.putText(self.im,timeInfo2,(450,80), font, 0.4,(0,0,0),2)

        cv2.line(self.im, tuple((1024,0)), tuple((1024,512)), (0,0,0))        
        
        #cv2.putText(self.im,"Arena 1",(220,60), font, 0.6,(0,0,0),2)
        #cv2.putText(self.im,"Arena 2",(220+512,60), font, 0.6,(0,0,0),2)
        cv2.putText(self.im,str(stimDotSize),(200+1024,60), font, 0.6,(0,0,0),2)
        cv2.putText(self.im,str(stimDotSize2),(200+1024,80), font, 0.6,(0,0,0),2)
        #cv2.resizeWindow(self.window_name, self.desiredWidth,self.desiredheight)
        #newWidth=1024
        #r = newWidth / self.im.shape[1]
        #dim = (newWidth, int(self.im.shape[0] * r))
 
        # perform the actual resizing of the image and show it
        dim=(1024,1024)
        resized = cv2.resize(self.im, dim)
         
        cv2.imshow(self.window_name, resized)
                
        if self.settings.vidRec:
           
            wr=self.im.astype('uint8')

            self.vOut.write(wr)

        
#p='D:\\data\\b\\2017\\20170131_VR_skypeVsTrefoil\\01_skypeVsTrefoil_blackDisk002\\'

   #D:\\data\\b\\2017\\20170420_miguel_competitionTest\\1\\PositionTxt_allROI2017-04-20T09_22_51.txt 

rereadTxt=0


if rereadTxt:
    lines=32
    empty=np.repeat('NaN',lines)
    empty=' '.join(empty)
    p='D:\\data\\b\\2017\\20170420_miguel_competitionTest\\1\\'
    avi_path = tkFileDialog.askopenfilename(initialdir=os.path.normpath(p))    
    p, tail = os.path.split(avi_path)
    txt_path=glob.glob(p+'\\Position*.txt')[0]
    roi_path=glob.glob(p+'\\ROI*.csv')[0]
    
    df_roi=pd.read_csv(roi_path,
        names=['x_topLeft',
        'y_topLeft',
        'widthHeight',
        'x_center',
        'y_center',
        'radius'],
        delim_whitespace=True)
    
    with open(txt_path) as f:
        mat=np.loadtxt((x if len(x)>6 else empty for x in f ),delimiter=' ')
    #epiLen=int(np.median(np.diff(np.where(mat[:-1,-1]!=mat[1:,-1]))))
    epiLen=int(np.median(np.diff(np.where((mat[:-1,-1]!=mat[1:,-1]) * ((mat[:-1,-1])==0)))))
    
    #ind=np.arange(149+epiLen,mat.shape[0],epiLen)
    #fixCol=[3,4,5,6,10,11,12,13,17,18,19,20,24,25,26,27,28,29]
    #for fc in fixCol:
    #    mat[ind,fc]=mat[ind+1,fc]
    
    anMat=[]
    sMat=[]
    for an in range(4):
        tmp=np.array(mat[:,an*7+np.array([0,1,3,4,5,6])])
        tmp[:,[0,2,4]]=tmp[:,[0,2,4]]+df_roi.x_topLeft[an]
        tmp[:,[1,3,5]]=tmp[:,[1,3,5]]+df_roi.y_topLeft[an]
        anMat.append(tmp)
        tmp=np.array(mat[:,[28,29]])
        sMat.append(tmp)
        #df['episode']=np.repeat(np.arange(mat.shape[0]/float(epiLen)),epiLen)
        #dfAll.append(df.copy())
    anMat=np.array(anMat)
    sMat=np.array(sMat)
    #e1=xp.experiment(avi_path,tp[0])
    #e2=xp.experiment(avi_path,tp[1])    
    
#csvFileOut=tp[0][:-4]+'_siSummary_epi'+str(10)+'.csv'
#df=pd.read_csv(csvFileOut,index_col=0,sep=',')[['epStart','episode']]
    
a=vidGui(avi_path,anMat,sMat,df_roi)