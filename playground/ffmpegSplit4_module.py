# -*- coding: utf-8 -*-
"""
Created on Fri Dec 04 14:16:45 2015

@author: jlarsch
"""
# split video into tiles based on tileList (width, height, x, y)
import subprocess as sp
import os
import numpy as np

def videoSplit(aviP,tileList):
    
    # path to ffmpeg bin
    #FFMPEG_PATH = 'c:/ffmpeg/bin/ffmpeg.exe'
    FFMPEG_PATH = 'ffmpeg.exe'
    
    head, tail = os.path.split(aviP)
    
    #assemble ffmpeg command
    
    fc='' #start with empty filter command
    mc=[''] #start with empty map command
    for i in range(np.shape(tileList)[0]):    
        fcNew=['[0:v]crop=',str(tileList[i][0]),':',str(tileList[i][1]),':',str(tileList[i][2]),':',str(tileList[i][3]),':','[out',str(i+1),'];']
        #print fcNew
        fc=fc+''.join(str(w) for w in fcNew)
        
        mcn1='[out'+str(i+1)+']'
        mcn2='out'+str(i+1)+'.mp4'
        mcNew=['-map',mcn1,mcn2]
        print mcNew
        mc.extend(mcNew)
    
    
    cmd=[FFMPEG_PATH,
    '-i', aviP,
    '-filter_complex', fc[:-1]]

    cmd.extend(mc[1:])    
    
    sp.check_output(cmd)
    

