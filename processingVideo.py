# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 09:56:15 2016

@author: jlarsch
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import models.geometry as geometry


def processing_video(asp,path,frames,start=0):
    
    #read video frame
    outSize=300
    video = cv2.VideoCapture(path)
    img_frame_first = video.read()[1]
    height_ori, width_ori = img_frame_first.shape[:2]
    resize_factor=height_ori/float(outSize)
    
    frameList=np.arange(start,frames,1)
    fourcc=cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    fps=30
    videoOut1 = cv2.VideoWriter('d:/VidProc1.avi',fourcc,fps,(3*outSize ,2*outSize),True)
    try:
        buffSize=50
        posBuffer=np.zeros((buffSize,2,2))
        buff=0
        
        for i in frameList:
            cont=np.zeros((2 * outSize,3*outSize,3),dtype='uint8')
            video.set(cv2.CAP_PROP_POS_FRAMES,i)
            img_frame_original = video.read()[1]
            img=cv2.resize(img_frame_original,(2*outSize,2*outSize))
            #img_c_old=img.copy()
            
            mask=img.copy()*0
            
            for a in range(2):
                pos=2*np.floor(asp[a].trajectory[i,:]/resize_factor)
                cv2.circle(mask, tuple(pos.astype('int')), 6, (255,255,255), -1)
            
            mask=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
        
            img_an = cv2.bitwise_and(img,img,mask = mask)
            
            img_pos=255+img.copy()*0
            
            for a in range(2):
                
                if np.mod(i,1)==0:
                    pos=2*asp[a].trajectory[i,:]/resize_factor
                    posBuffer[0,a,:]=pos
                    buff +=1
                
         
                for b in np.arange(0,np.min([buff/2,buffSize]),1):
    
                    opa=1-float(b)/float(buffSize)
                    img=draw_alpha(img_pos,tuple(posBuffer[b,a,:].astype('int')),5,(255*(a==0),0,(a==1)*255),1,opa)
                                
                
                cv2.circle(img_pos, tuple(pos.astype('int')), 4, (255,255,255), -1)
                cv2.circle(img_pos, tuple(pos.astype('int')), 6, (0,0,0), 2)
                cv2.addWeighted(img_pos, 0.5, img, 0.5, 0, img)
                
                curr_ori=asp[a].fish_orientation_elipse[i]          
                curr_ori=np.mod(180-curr_ori,360)+90
                oriVec=geometry.Vector.new_vector_from_angle(curr_ori)*30
                
                oriVec.draw(img,geometry.Vector(*posBuffer[0,a,:]))
                
                crop=asp[a].frAll_rot[i,50:250,50:250]
                cropCol=cv2.cvtColor(crop,cv2.COLOR_GRAY2BGR)
                cont[a*outSize:a*outSize+cropCol.shape[0],:cropCol.shape[0],:]=cropCol
                
            img=255-cv2.bitwise_and(255-img,255-img_an)
            cv2.line(img,tuple(posBuffer[0,0,:].astype('int')),tuple(posBuffer[0,1,:].astype('int')),(0,255,0))
            cont[:,outSize:,:]=img
            videoOut1.write(cont)
            posBuffer=np.roll(posBuffer,1,axis=0)

    
        videoOut1.release()
    except:
        print 'error'
        videoOut1.release()

processing_video(asp,af,999)

def draw_alpha(img,center,rad,col,thick,opacity):
    
    overlay = img.copy()
    # (2) draw shapes:
    cv2.circle(overlay, center, rad, col, thick)
    # (3) blend with the original:
    cv2.addWeighted(overlay, opacity, img, 1 - opacity, 0, img)
    return img