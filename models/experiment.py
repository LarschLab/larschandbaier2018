import numpy as np
import os
import scipy.io
import datetime

import functions.plotFunctions_joh as johPlt
import functions.randomDotsOnCircle as randSpacing
import functions.video_functions as vf
from models.pair import Pair

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages


class ExperimentMeta(object):
    #This class collects paths, arena and video parameters
    def __init__(self,path,arenaDiameter_mm=100):
        self.arenaDiameter_mm = arenaDiameter_mm
        self.arenaCenterPx = [0,0] #assign later from class 'Pair'
        self.pxPmm = 0 #assign later from class 'Pair'
        
        
        #If a video file name is passed, collect video parameters
        if path.endswith('.avi') or path.endswith('.mp4'):
            self.aviPath = path
            #get video meta data
            vp=vf.getVideoProperties(path) #video properties
            self.ffmpeginfo = vp
            self.videoDims = [vp['width'] , vp['height']]
            self.numFrames=vp['nb_frames']
            self.fps=vp['fps']
            #self.date=vp['TAG:date']
        else:
            self.fps=30
        
        self.minShift=5*60*self.fps
        #concatenate dependent file paths (trajectories, pre-analysis)
        head, tail = os.path.split(path)
        head=os.path.normpath(head)
        
        trajectoryPath = os.path.join(head,'trajectories_nogaps.mat')
        if np.equal(~os.path.isfile(trajectoryPath),-2):
            self.trajectoryPath = trajectoryPath
        else:
            trajectoryPath = os.path.join(head,'trajectories.mat')
            if np.equal(~os.path.isfile(trajectoryPath),-2):
                self.trajectoryPath = trajectoryPath
                
        AnSizeFilePath = os.path.join(head,'animalSize.txt')
        if np.equal(~os.path.isfile(AnSizeFilePath),-2):
            self.AnSizeFilePath = AnSizeFilePath
        
        self.dataPath = os.path.join(head,'analysisData.mat')
        self.aspPath=os.path.join(head,tail[:-4]+'_asp.npz')


        
class experiment(object):
    #Class to collect, store and plot data belonging to one experiment
    def __init__(self,path):
        self.n_shift_Runs=10
        self.sPair=[]
        self.expInfo=ExperimentMeta(path)
        self.AnSize=np.array(np.loadtxt(self.expInfo.AnSizeFilePath, skiprows=1,dtype=int))
        mat=scipy.io.loadmat(self.expInfo.trajectoryPath)
        self.rawTra=mat['trajectories']
        #some mat files begin with some number of nan entries for position. set those to zero
        nanInd=np.where(np.isnan(self.rawTra))
        if np.equal(np.shape(nanInd)[1],0) or np.greater(np.max(nanInd),1000):
            LastNan=0
        else:
            LastNan=np.max(nanInd)+1
        
        self.rawTra[:LastNan,:,:]=0
        self.skipNanInd=LastNan

        self.maxPixel=np.nanmax(self.rawTra,0)
        self.minPixel=np.nanmin(self.rawTra,0)
        self.expInfo.trajectoryDiameterPx=np.mean(self.maxPixel-self.minPixel)
        #expInfo.pxPmm=expInfo.trajectoryDiameterPx/expInfo.arenaDiameter_mm
        self.expInfo.pxPmm=8.6
        self.expInfo.arenaCenterPx=np.mean(self.maxPixel-(self.expInfo.trajectoryDiameterPx/2),axis=0)
        self.expInfo.numFrames=self.rawTra.shape[0]

        self.load_animalShapeParameters()
        
        #proceed with animal-pair analysis if there is more than one trajectory
        if np.shape(self.rawTra)[1]>1:
            Pair(shift=False).linkExperiment(self)
        
            #self.AnShape=AnimalShapeParameters()
            
            #generate shifted control 'mock' pairs
            #generate nRuns instances of Pair class with one animal time shifted against the other
            for i in range(self.n_shift_Runs):
                Pair(shift=True,nRun=i).linkExperiment(self)
            
            #calculate mean IAD histogram for shifted pairs
            IADHistall=[]
            IADHistall.append([x.IADhist()[0:30*60*90] for x in self.sPair])
            self.spIADhist_m=np.nanmean(IADHistall,axis=1)            
            
            self.ShoalIndex=(self.spIAD_m()-np.nanmean(self.pair.IAD()))/self.spIAD_m()
            #self.totalPairTravel=sum(self.Pair.totalTravel)
            self.avgSpeed=self.pair.avgSpeed
            probTra=mat['probtrajectories']
            self.idQuality=np.mean(probTra[probTra>=0])*100

    def addPair(self,pair):
        
        if pair.shift:
            self.sPair.append(pair)
            return self.sPair[-1]
        else:
            self.pair=pair
            return self.pair
        
    def spIAD_m(self):
        x=np.nanmean([np.nanmean(x.IAD()) for x in self.sPair])
        return x
        
    def spIAD_std(self):
        x=np.nanstd([np.nanmean(x.IAD()) for x in self.sPair])
        return x
           
    def load_animalShapeParameters(self):
        
        aspPath=self.expInfo.aspPath
        self.haveASP=np.equal(~os.path.isfile(aspPath),-2)
        
        if self.haveASP:
            npzfile = np.load(aspPath)
            self.tailCurvature_mean=npzfile['tailCurvature_mean']
            self.fish_orientation_elipse=npzfile['ori']
            self.centroidContour=npzfile['centroidContour']
    
    def plotOverview(self,condition='notDefined'):
        
#        for i in range(2):
#            self.pair.animals[i].plot_errorAngle()
#            self.pair.animals[i].plot_d_error_over_d_IAD()
#            
#        for j in range(10):
#            for i in range(2):
#                self.sPair[j].animals[i].plot_errorAngle()
#                self.sPair[j].animals[i].plot_d_error_over_d_IAD()
            
        outer_grid = gridspec.GridSpec(4, 4)        
        plt.figure(figsize=(8, 8))   
        #plt.text(0,0.95,path)
        plt.figtext(0,.01,self.expInfo.aviPath)
        plt.figtext(0,.03,condition)
        plt.figtext(0,.05,self.idQuality)
        
        plt.subplot(4,4,1,rasterized=True)
        plt.cla()
        plt.plot(self.pair.animals[0].ts.position().x(),self.pair.animals[0].ts.position().y(),'b.',markersize=1,alpha=0.1)
        plt.plot(self.pair.animals[1].ts.position().x(),self.pair.animals[1].ts.position().y(),'r.',markersize=1,alpha=0.1)
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        plt.title('raw trajectories')  
        
        plt.subplot(4,4,3)
        plt.cla()
        PolhistBins=self.pair.animals[0].ts.PolhistBins()
        plt.step(PolhistBins[:-1], self.pair.animals[0].ts.Pol_n(),'b',lw=2,where='mid')
        plt.step(PolhistBins[:-1], self.pair.animals[1].ts.Pol_n(),'r',lw=2,where='mid')
        plt.ylim([0, .1])

        
        
        #plot IAD time series
        x=np.arange(float(np.shape(self.pair.IAD())[0]))/(self.expInfo.fps*60)
        plt.subplot(4,1,2,rasterized=True)
        plt.cla()
        plt.plot(x,self.pair.IAD(),'b.',markersize=1)
        plt.xlim([0, 90]) 
        plt.xlabel('time [minutes]')
        plt.ylabel('IAD [mm]')
        plt.title('Inter Animal Distance over time')
        
        #IAD histogram for raw and shifted data and simulated random dots
        plt.subplot(4,4,2)
        plt.cla()
        #get rid of nan
        IAD=self.pair.IAD()
        IAD=IAD[~np.isnan(IAD)]
        histBins=np.arange(100)
        n, bins, patches = plt.hist(IAD, bins=histBins, normed=1, histtype='stepfilled')
        plt.step(histBins[:-1],self.spIADhist_m.T,'k',lw=1)
        #simulate random uniform spacing
        rad=(self.expInfo.trajectoryDiameterPx/self.expInfo.pxPmm)/2
        num=1000
        dotList,dotND = randSpacing.randomDotsOnCircle(rad,num)
        n, bins, patches = plt.hist(dotND, bins=histBins, normed=1, histtype='step')
        plt.xlabel('IAD [mm]')
        plt.ylabel('p')
        plt.title('IAD')
        plt.ylim([0, .05])
        
        plt.subplot(4,4,4)
        plt.cla()
        x=[1,2]
        y=[np.nanmean(IAD),self.spIAD_m()]
        yerr=[0,self.spIAD_std()]
        plt.bar(x, y, yerr=yerr, width=0.5,color='k')
        lims = plt.ylim()
        plt.ylim([0, lims[1]]) 
        plt.xlim([0.5, 3]) 
        plt.ylabel('[mm]')
        plt.xticks([1.25,2.25],['raw','shift'])
        plt.title('mean IAD')
        
        plt.subplot(4,4,9)
        plt.cla()
        a1=self.pair.animals[0].ts.neighborMat()
        a2=self.pair.animals[1].ts.neighborMat()
        
        meanPosMat=np.nanmean(np.stack([a1,a2],-1),axis=2)
        plt.imshow(meanPosMat,interpolation='none', extent=[-31,31,-31,31],clim=[0,100],origin='lower')
        plt.title('mean neighbor position')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        
        #neighborhood matrix for animal0
        #the vertial orientation was confirmed correct in march 2016
        plt.subplot(4,4,13)
        plt.cla()
        PosMat=a1
        plt.imshow(PosMat,interpolation='none', extent=[-31,31,-31,31],clim=[0,200],origin='lower')
        plt.title('mean neighbor position')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        
        plt.subplot(4,4,14)
        plt.cla()
        PosMat=a2
        plt.imshow(PosMat,interpolation='none', extent=[-31,31,-31,31],clim=[0,200],origin='lower')
        plt.title('mean neighbor position')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        
        #LEADERSHIP
        plt.subplot(4,4,15)
        plt.cla()
        self.LeadershipIndex=np.array([a.ts.FrontnessIndex() for a in self.pair.animals])
        x=[1,2]
        barlist=plt.bar(x,self.LeadershipIndex, width=0.5,color='b')
        barlist[1].set_color('r')
        plt.title('Leadership')
        plt.ylabel('index')
        plt.ylim([-.5, .5])
        plt.xlim([0.5, 3])
        
        tmp=[an.at_bottom_of_dish_stack() for an in self.pair.animals]
        if np.sum(tmp)==0:
            plt.xticks([1.25,2.25],['same','dish'])

        else:
            xtickList=[int(np.where(self.Pair.StackTopAnimal==1)[0])+1.25,int(np.where(self.Pair.StackTopAnimal==0)[0])+1.25]
            plt.xticks(xtickList,['top','bottom'])            
            
        plt.subplot(4,4,16)
        plt.cla()
        x=[1,2]
        barlist=plt.bar(x,self.AnSize[:,1], width=0.5,color='b')
        barlist[1].set_color('r')
        plt.xlim([0.5, 3])
        plt.title('Body size')
        plt.ylabel('area [px]')
        

        plt.subplot(4,4,10)
        plt.title('accel=f(pos_n)')
        a1=self.pair.animals[0].ts.ForceMat_speedAndTurn()
        a2=self.pair.animals[1].ts.ForceMat_speedAndTurn()
        meanForceMat=np.nanmean(np.stack([a1,a2],-1),axis=2)
        johPlt.plotMapWithXYprojections(meanForceMat,3,outer_grid[9],31,10)
        
        
        
        
        plt.subplot(4,4,11)
        plt.cla()
        plt.bar([1,2],self.pair.avgSpeed(),width=0.5)
        plt.title('avgSpeed')
   
        plt.tight_layout()
        plt.show()
        
        currentTime=datetime.datetime.now()
        self.pdfPath=self.expInfo.aviPath[:-4]+'_'+currentTime.strftime('%Y%m%d%H%M%S')+'.pdf'
        with PdfPages(self.pdfPath) as pdf:
            pdf.savefig()
            
#        plt.figure()
#        a1=self.Pair.animals[0].ForceMat
#        a2=self.Pair.animals[1].ForceMat
#        meanForceMat=np.nanmean(np.stack([a1,a2],-1),axis=2)
#        outer_grid = gridspec.GridSpec(1, 1)
#        johPlt.plotMapWithXYprojections(meanForceMat,3,outer_grid[0],31,10)
#        
#        plt.figure()
#        a1=self.Pair.animals[0].ForceMat_speed
#        a2=self.Pair.animals[1].ForceMat_speed
#        meanForceMat=np.nanmean(np.stack([a1,a2],-1),axis=2)
#        outer_grid = gridspec.GridSpec(1, 1)
#        johPlt.plotMapWithXYprojections(meanForceMat,3,outer_grid[0],31,.1)
#        
#        plt.figure()
#        a1=self.Pair.animals[0].ForceMat_turn
#        a2=self.Pair.animals[1].ForceMat_turn
#        meanForceMat=np.nanmean(np.stack([a1,a2],-1),axis=2)
#        outer_grid = gridspec.GridSpec(1, 1)
#        johPlt.plotMapWithXYprojections(meanForceMat,3,outer_grid[0],31,.1)
        
        
        