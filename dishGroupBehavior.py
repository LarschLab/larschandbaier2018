__author__ = 'jlarsch'

#import Tkinter
import tkFileDialog
import joFishHelper
import os

avi_path = tkFileDialog.askopenfilename(initialdir=os.path.normpath('d:/data/b'))
#Tkinter.Tk().withdraw() # Close the root window - not working?

experiment=joFishHelper.experiment(avi_path)
