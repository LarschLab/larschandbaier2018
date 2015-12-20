# -*- coding: utf-8 -*-
"""
Created on Fri Dec 04 14:16:45 2015

@author: jlarsch
"""
# split video into tiles based on tileList (width, height, x, y)
import subprocess as sp
import os
import numpy as np
import joFishHelper
import unTileVideo_b
import tkFileDialog

avi_path = tkFileDialog.askopenfilename(initialdir=os.path.normpath('c:/test/'))
print avi_path
Scl=unTileVideo_b.UnTileArenaVideo(avi_path)


def videoSplit(aviP,tileList):
    
    # path to ffmpeg bin
    #FFMPEG_PATH = 'c:/ffmpeg/bin/ffmpeg.exe'
    FFMPEG_PATH = 'ffmpeg.exe'
    
    head, tail = os.path.split(aviP)
    
    #calculate median background image for background division
    joFishHelper.getMedVideo(aviP,9,1)
    bgPath=(head+'/bgMed.tif')
    print 'background generated'
    #assemble ffmpeg command
    
    fc='' #start with empty filter command
    mc=[''] #start with empty map command
    spc='' #start with empty split command
    numTiles=np.shape(tileList)[0]
    for i in range(numTiles):
        print tileList
        print np.shape(tileList)[0]
        print i
        print tileList[i]
        print np.append(tileList[i],i+1)
        #create subdirectories for split output
        directory=head+'/'+ str(i+1)+'/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        fcNew=('[int{4}]crop={0:.0f}:{1:.0f}:{2:.0f}:{3:.0f}:[out{4}];').format(*np.append(tileList[i],i+1)) #e.g. [0:v]crop=1024:1024:0:0[out1];
        fc=fc+''.join(str(w) for w in fcNew)
        mcn1='[out'+str(i+1)+']'
        mcn2=directory+'split_'+str(i+1)+'_'+tail+'.mp4'
        mcNew=['-map',mcn1,mcn2]
        mc.extend(mcNew)
        spcNew=('[int{0}]').format(i+1)
        spc=spc+spcNew

    #command string for background subtraction
    #cmdBG=('[1:0] setsar=sar=1,format=rgba [1sared]; [0:0]format=rgba [0rgbd];[0rgbd] [1sared] blend=all_mode=\'divide\':repeatlast=1,format=yuva422p10le,split={0} ').format(numTiles)
    #cmdBG=('[1:0] setsar=sar=1 [1sared]; [0:0][1sared] blend=all_mode=\'divide\':repeatlast=1,split={0} ').format(numTiles)
    cmdBG=('[1:0] setsar=sar=1 [1sared]; [0:0][1sared] blend=all_mode=\'divide\':repeatlast=1,format=gray,split={0} ').format(numTiles)

   #fc=cmdBG+fc[:-1]
    fc=cmdBG+spc+';'+''.join(str(w) for w in fc)
    print fc
    #print cmdBG
    cmd=[FFMPEG_PATH,
    '-i', aviP,
    '-i', bgPath,
    '-maxrate', '5M',
    '-minrate', '5M',
    '-filter_complex', fc[:-1]]

    
    cmd.extend(mc[1:])    
    print cmd
    print 'starting ffmpeg for video processing...'
    sp.Popen(cmd)
    

videoSplit(avi_path,Scl.roiSq)