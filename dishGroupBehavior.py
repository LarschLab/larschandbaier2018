__author__ = 'jlarsch'


import matplotlib.pyplot as plt
import numpy as np
import Tkinter
import tkFileDialog
import joFishHelper

import scipy.io
      
avi_path = tkFileDialog.askopenfilename()
#Tkinter.Tk().withdraw() # Close the root window - not working?

expInfo=joFishHelper.ExperimentMeta(avi_path)
mat=scipy.io.loadmat(expInfo.trajectoryPath)

        
tra=joFishHelper.Pair(mat['trajectories'],expInfo)

plt.plot(tra.position[:,0,0],tra.position[:,0,1],'b.',markersize=1)
plt.show()

#generate shifted control 'mock' pairs
test=joFishHelper.shiftedPair(tra,expInfo)
