# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 09:56:15 2016

@author: jlarsch
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


def processing_video(asp,path,frames,start=0):
    
    #read video frame
    outSize=200
    video = cv2.VideoCapture(path)
    img_frame_first = video.read()[1]
    height_ori, width_ori = img_frame_first.shape[:2]
    resize_factor=height_ori/outSize
    
    frameList=np.arange(start,frames,1)
    fourcc=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    #fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    videoOut1 = cv2.VideoWriter('d:/VidProc.avi',fourcc,5,(outSize ,3*outSize),True)
    try:
        buffSize=50
        posBuffer=np.zeros((buffSize,2,2))
        buff=0
        
        
        for i in frameList:
            cont=np.zeros((3 * outSize,outSize,3),dtype='uint8')
            video.set(cv2.CAP_PROP_POS_FRAMES,i)
            img_frame_original = video.read()[1]
            img=cv2.resize(img_frame_original,(outSize,outSize))
            #img_c_old=img.copy()
            
            
            for a in range(2):
                
                if np.mod(i,2)==0:
                    pos=np.floor(asp[a].trajectory[i,:]/resize_factor)
                    posBuffer=np.roll(posBuffer,1,axis=0)
                    posBuffer[0,a,:]=pos
                    buff +=1
                
    #            for b in np.arange(np.min([buff,100])-2,-1,-1):
                #img_c_new=img_c_old.copy()            
                for b in np.arange(0,np.min([buff,buffSize])-1,1):
    
                    opa=1-float(b)/float(buffSize)
                    img=draw_alpha(img,tuple(posBuffer[b,a,:].astype('int')),5,(255*(a==0),0,(a==1)*255),1,opa)        
    
                    
                crop=asp[a].frAll_rot[i,50:250,50:250]
                cropCol=cv2.cvtColor(crop,cv2.COLOR_GRAY2BGR)
                cont[(a+1)*outSize:(a+2)*outSize,:outSize,:]=cropCol
                
            cont[:outSize,:outSize,:]=img
            videoOut1.write(cont)
    
        videoOut1.release()
    except:
        videoOut1.release()

processing_video(asp,af,200)

def draw_alpha(img,center,rad,col,thick,opacity):
    
    overlay = img.copy()
    # (2) draw shapes:
    cv2.circle(overlay, center, rad, col, thick)
    # (3) blend with the original:
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
    return img