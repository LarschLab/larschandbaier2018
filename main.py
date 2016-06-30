__author__ = 'jlarsch'

#import Tkinter
import tkFileDialog
import models.experiment as xp
import models.experiment_set as es
import os


batchmode=0
timeStim=0
sizePlot=0
systShift=0

if batchmode:
    
    expSet=es.experiment_set(systShift,timeStim,sizePlot)
            
else:    
    avi_path = tkFileDialog.askopenfilename(initialdir=os.path.normpath('d:/data/b/2016/20160314_TLStacks/1/a/1'))
    experiment=xp.experiment(avi_path)
    experiment.plotOverview()


    
    