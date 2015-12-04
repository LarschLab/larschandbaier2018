
import matplotlib.pyplot as plt
import numpy as np
import cv2
import tkFileDialog
import wx
import os

avi_path = tkFileDialog.askopenfilename(initialdir=os.path.normpath('d:/data/b'))
#avi_path = 'D:/data/b/2FishSeries_2/20151125_isolatedVsGroup/expStream2015-11-25T16_45_05_isolatedVsGroup.avi'

capture = cv2.VideoCapture(avi_path)
capture.set(cv2.CAP_PROP_POS_FRAMES,1)


# load the image, clone it for output, and then convert it to grayscale
image = capture.read()
output = image[1].copy()
gray = cv2.cvtColor(image[1], cv2.COLOR_BGR2GRAY)


class MyClickableImage(object):
    def __init__(self,frame):
        
        def ask(parent=None, message='', default_value=''):
            dlg = wx.TextEntryDialog(parent, message, defaultValue=default_value)
            dlg.ShowModal()
            result = dlg.GetValue()
            dlg.Destroy()
            return result

        self.frame = frame
        self.fig = plt.figure(figsize=(6,9))
        self.ax = self.fig.add_subplot(111)
        xaxis = self.frame.shape[1]
        yaxis = self.frame.shape[0]
        self.im = self.ax.imshow(self.frame[::-1,:], cmap='jet', extent=(0,xaxis,0,yaxis), picker=5)

        line, = self.ax.plot([0], [0])  # empty line
        self.line = line
        self.fig.canvas.mpl_connect('button_press_event', self.onpick1)
        #button_press_event
        self.fig.canvas.draw()
        
        self.ClickCount=0
        # Initialize wx App
        app = wx.App()
        app.MainLoop()

        # Call Dialog
        self.numPicks = np.int(ask(message = 'How Many Arenas?'))
        self.PicksDone=0
        self.x = np.zeros([self.numPicks,4])
        self.y = np.zeros([self.numPicks,4])
        self.roiAll=[]

    def onpick1(self,event):
        if np.less(self.PicksDone,self.numPicks):
        
            if np.less(self.ClickCount,3):
                self.x[self.PicksDone,self.ClickCount]=event.xdata
                self.y[self.PicksDone,self.ClickCount]=event.ydata
                self.ClickCount += 1
                print self.x, self.ClickCount, self.PicksDone
            else:
                self.x[self.PicksDone,self.ClickCount]=event.xdata
                self.y[self.PicksDone,self.ClickCount]=event.ydata
                self.line.set_data(self.x, self.y)
                self.line.figure.canvas.draw()
                self.ClickCount=0
                A=np.array([self.x[self.PicksDone,:],self.y[self.PicksDone,:],np.ones(4)])
                
                A=A.transpose()
                print A
                b=-self.x[self.PicksDone,:]**2-self.y[self.PicksDone,:]**2
                coefs=np.linalg.lstsq(A,b)
                roi=np.zeros(3)
                roi[0]=-coefs[0][0]/2
                roi[1]=-coefs[0][1]/2
                roi[2]=np.sqrt(roi[0]**2+roi[1]**2-coefs[0][2])
                circle=plt.Circle((roi[0],roi[1]),roi[2],color='b',fill=False)
                event.canvas.figure.gca().add_artist(circle)
                event.canvas.draw()
                self.PicksDone +=1
                print 'ahllo'
                #print self.PicksDone
                self.roiAll[self.PicksDone]=roi
        else:
            print 'all circles done'
            
cl=MyClickableImage(gray)