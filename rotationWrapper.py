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

asp=AnimalShapeParameters.AnimalShapeParameters(af,t)
plt.plot(asp.fish_orientation_elipse_all)
mr=np.array(asp.frAll_rot)

head, tail = os.path.split(af)
head=os.path.normpath(head)
rotVideoPath = os.path.join('d:\\',tail[:-4]+'_rotate.avi')
imsave(rotVideoPath, mr)
