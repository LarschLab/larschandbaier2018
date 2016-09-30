# -*- coding: utf-8 -*-
"""
Created on Fri Dec 04 14:16:45 2015

split video into tiles based on tileList (width, height, x, y)
Ask user to select circular arena(s) on median video frame
Start ffmpeg in sub process to perform splitting and background division of each frame
ffmpeg comand is assembled as a string - that part is not pretty...

@author: jlarsch
"""
import subprocess as sp
import os
import numpy as np
import functions.video_functions as vf
import tkFileDialog
import functions.gui_circle as gc
import cv2

avi_path = tkFileDialog.askopenfilename(initialdir='c:/test/')
print avi_path

def videoSplit(aviP):
    
    # path to ffmpeg bin
    FFMPEG_PATH = 'c:/ffmpeg/bin/ffmpeg.exe'
    
    head, tail = os.path.split(aviP)
    
    #get median background image for background division
    vidMed,bg_file,minval2=vf.getMedVideo(aviP,9,1)
    bgPath=(head+'/bgMed.tif')

    
    
    print 'background generated'
    
    #ask user for arenas
    scaleData=gc.get_circle_rois(bgPath,'_scale',0)[0]
    tileList=np.array(scaleData.ix[:,3:7].values,dtype='int64')
    numTiles=np.shape(tileList)[0]
    
    vp=vf.getVideoProperties(avi_path)
    fps_s=str(vp['fps'])
    
    bg_file_mask=(head+'/bgMed_mask.tif')
    bg=cv2.imread(bgPath)
    bg = cv2.cvtColor(bg, cv2.COLOR_RGB2GRAY)
    circleList=np.array(scaleData.ix[:,0:3].values,dtype='int64')
    
    def circleMask(a,b,r,n):
        y,x = np.ogrid[-a:n-a, -b:n-b]
        return (x*x + y*y <= r*r)
    
    for i in range(numTiles):
        bg[~circleMask(circleList[i,1],circleList[i,0],circleList[i,2],bg.shape[0])]=minval2
        
    cv2.imwrite(bg_file_mask,bg)
    
    
    
    #assemble ffmpeg command
    
    fc='' #start with empty filter command
    mc=[''] #start with empty map command
    spc='' #start with empty split command
    
    for i in range(numTiles):
        #create subdirectories for split output

        directory=head+'/'+ str(i+1)+'/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        fcNew=('[int{4}]crop={0:.0f}:{1:.0f}:{2:.0f}:{3:.0f}:[out{4}];').format(*np.append(tileList[i],i+1)) #e.g. [0:v]crop=1024:1024:0:0[out1];
        fc=fc+''.join(str(w) for w in fcNew)
        mcn1='[out'+str(i+1)+']'
        mcn2=directory+'split_'+str(i+1)+'_'+tail+''
        mcNew=['-map',mcn1,'-c:v','libxvid','-q:v','4','-g','10',mcn2]
        mc.extend(mcNew)
        spcNew=('[int{0}]').format(i+1)
        spc=spc+spcNew

    #command string for background subtraction
    cmdBG=('[1:0] setsar=sar=1,format=gray,lutyuv=y=(val-'+str(minval2)+')*(255/(255-'+str(minval2)+')) [1scaled]; [0:0]format=gray,lutyuv=y=(val-'+str(minval2)+')*(255/(255-'+str(minval2)+')) [0scaled];[0scaled][1scaled] blend=all_mode=\'divide\':repeatlast=1,format=gray,split={0} ').format(numTiles)

    fc=cmdBG+spc+';'+''.join(str(w) for w in fc)

    cmd=[FFMPEG_PATH,
    '-i', aviP,
    '-i', bg_file_mask,
    '-y',
    #'-c:v', 'libx264',
    #'-maxrate', '5M',
    #'-minrate', '5M',
    #'-threads', '0',
    #'-c:v', 'libxvid',
    #'-q:v', '5',
    #'-g', '10',
    #'-keyint_min','10',
    '-r',fps_s,
    '-filter_complex', fc[:-1]]

    cmd.extend(mc[1:])    
    print cmd
    print 'starting ffmpeg for video processing...'
    sp.Popen(cmd)
    

videoSplit(avi_path)