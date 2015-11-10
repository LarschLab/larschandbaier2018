__author__ = 'jlarsch'

#import Tkinter
import tkFileDialog
import joFishHelper

avi_path = tkFileDialog.askopenfilename()
#Tkinter.Tk().withdraw() # Close the root window - not working?

experiment=joFishHelper.experiment(avi_path)
