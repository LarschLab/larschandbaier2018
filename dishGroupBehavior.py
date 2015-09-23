__author__ = 'jlarsch'

import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import Tkinter
import tkFileDialog
import joFishHelper 
        
avi_path = tkFileDialog.askopenfilename()
#Tkinter.Tk().withdraw() # Close the root window - not working?

expInfo=joFishHelper.ExperimentMeta(avi_path)

mat=scipy.io.loadmat(expInfo.trajectoryPath)
trajectories=mat['trajectories']
maxPixel=np.max(np.max(np.nanmax(trajectories,0)))
minPixel=np.min(np.min(np.nanmin(trajectories,0)))
arenaDiameterPx=maxPixel-minPixel
expInfo.pxPmm=arenaDiameterPx/expInfo.arenaDiameter
plt.plot(trajectories[:,0,0],trajectories[:,0,1],'b.',markersize=1)
plt.show()
vp=joFishHelper.getVideoProperties(avi_Path)