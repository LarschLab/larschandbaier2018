# -*- coding: utf-8 -*-
"""
Created on Tue May 31 14:29:56 2016

@author: jlarsch
"""

import AnimalShapeParameters
import tkFileDialog
import os
from tifffile import imsave
import numpy as np
import joFishHelper
import matplotlib.pyplot as plt


af = tkFileDialog.askopenfilename(initialdir=os.path.normpath('d:/data/b'))
experiment=joFishHelper.experiment(af)
t=experiment.Pair.positionPx

asp1=AnimalShapeParameters.AnimalShapeParameters(af,t[:,0,:])
asp2=AnimalShapeParameters.AnimalShapeParameters(af,t[:,1,:])
plt.plot(asp1.fish_orientation_elipse_all)
plt.plot(asp2.fish_orientation_elipse_all)

mr=np.concatenate(np.array(asp1.frAll_rot),np.array(asp2.frAll_rot),axis=1)

head, tail = os.path.split(af)
head=os.path.normpath(head)
rotVideoPath = os.path.join('d:\\',tail[:-4]+'_rotateBoth.avi')
imsave(rotVideoPath, mr)
