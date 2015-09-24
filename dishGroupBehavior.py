__author__ = 'jlarsch'

import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import Tkinter
import tkFileDialog
import joFishHelper
import matrixUtilities_joh as mu 
import random
        
avi_path = tkFileDialog.askopenfilename()
#Tkinter.Tk().withdraw() # Close the root window - not working?

expInfo=joFishHelper.ExperimentMeta(avi_path)

mat=scipy.io.loadmat(expInfo.trajectoryPath)
trajectories=mat['trajectories']
maxPixel=np.max(np.max(np.nanmax(trajectories,0)))
minPixel=np.min(np.min(np.nanmin(trajectories,0)))
arenaDiameterPx=maxPixel-minPixel
expInfo.pxPmm=arenaDiameterPx/expInfo.arenaDiameter
tra=joFishHelper.Pair(trajectories,expInfo)
plt.plot(trajectories[:,0,0],trajectories[:,0,1],'b.',markersize=1)
plt.show()

#generate shifted control 'mock' pairs
nRuns=10
minShift=5*60*expInfo.fps

def shiftPairTrajectory(tra,nRuns,minShift,expInfo):
    #s=np.shape(tra)
    sPair=[]
    #traShiftAll=np.zeros([s[0],s[1],s[2],nRuns])
    for i in range(10):
        traShift=tra
        shiftIndex=int(random.uniform(minShift,tra.shape[0]-minShift))
        #shift animal 0
        traShift[:,0,:]=np.roll(tra[:,0,:],shiftIndex,axis=0)
        #traShiftAll[:,:,:,i]=traShift
        sPair.append(joFishHelper.Pair(traShift,expInfo))
    return sPair
        

#vp=joFishHelper.getVideoProperties(avi_Path)