import numpy as np
import os
import datetime
import glob
import functions.plotFunctions_joh as johPlt
import functions.randomDotsOnCircle as randSpacing
import functions.video_functions as vf
from models.pair import Pair
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from functions.getAnimalSizeFromVideo import getAnimalSizeFromVideo
import pandas as pd
from datetime import timedelta
from datetime import datetime
import random

#
# experiment class stores raw animal trajectory data and meta information for one video recording
# experiments contain 1 or more animal-pair-episodes (ape)
# 1 ape represents a period of animal-stimulus pairing used to compute aggregate stats such  as inter animal distance

class ExperimentMeta(object):
    # ExperimentMeta class collects file paths, arena and video parameters for one experiment
    def __init__(self, trajectoryPath):

        self.trajectoryPath = trajectoryPath

        self.aviPath = None
        self.aspPath = None
        self.anSizeFile = None
        self.roiPath = None
        self.pairListFn = None

        self.animalSet = None
        self.inDishTime = None
        self.minShift = None
        self.episodeDur = None
        self.episodes = None
        self.episodePLcode = None
        self.pairListAll = None
        self.nShiftRuns = None
        self.birthDayAll = None
        self.SaveNeighborhoodMaps = None

        self.numPairs = None
        self.txtTrajectories = None
        self.rois = None
        self.videoDims = None
        self.numFrames = None
        self.fps = None
        self.pxPmm = None

        #Parse arguments from expinfo or use defaults with a notification.

    def fillCompleteMeta(self, expinfo):
        try:
            self.episodeDur = float(expinfo['epiDur'])
        except KeyError:
            print('episode duration required')
            raise

        try:
            self.aviPath = expinfo['aviPath']
            tmp = open(self.aviPath, 'r')
            tmp.close()
        except KeyError as error:
            print('No video file specified. Will resort to defaults.', error)
            self.aviPath = None

        except FileNotFoundError as error:
            print('Specified video file not found.', error)
            self.aviPath = None
            raise

        try:
            self.pairListFn = expinfo['pairList']
            self.pairListAll = np.loadtxt(self.pairListFn, dtype='int')
        except FileNotFoundError:
            print('Pair list not found. Using default...')
            self.pairListAll = np.diag((-1, -1)) + 1

        try:
            self.episodePLcode = expinfo['episodePLcode']
        except KeyError:
            print('Use of PLcode not specified. Not using (default).')
            self.episodePLcode = 0

        if self.episodePLcode:
            self.numPairs = self.pairListAll.shape[1]-1
        else:
            self.numPairs = self.pairListAll.sum()

        try:
            self.arenaDiameter_mm = expinfo['arenaDiameter_mm']
        except KeyError:
            print('arenaDiameter_mm not specified. Using 100 mm (default).')
            self.arenaDiameter_mm = 100

        try:
            self.pxPmm, self.roiPath, self.rois = self.PxPmmFromRois()
        except NameError:
            # try reading from video
            print('pxPmm not yet computed. Trying to re compute.')
            self.pxPmm = vf.get_pixel_scaling(path, forceCorrectPixelScaling=1, forceInput=0)
            #
            #self.pxPmm = 8

        # If a video file name is passed, collect video parameters
        if self.aviPath:
            # get video meta data
            vp = vf.getVideoProperties(self.aviPath)  # video properties
            # self.ffmpeginfo = vp
            self.videoDims = [int(vp['width']), int(vp['height'])]
            self.numFrames = int(vp['nb_frames'])
            self.fps = int(vp['fps'])
            # self.videoDims = np.array([2048, 1280])
            # self.numFrames = 558135
            # self.fps = 30
        else:  # otherwise, use defaults
            self.videoDims = np.array([2048, 1280])
            self.numFrames = 558135
            self.fps = 30
            print('Attention! Using default video parameters: ', self.videoDims, self.numFrames, self.fps)

        try:
            self.minShift = expinfo['minShift']
        except KeyError:
            print('minShift not specified. Using 60 seconds (default).')
            self.minShift = 60 + self.fps

        try:
            self.nShiftRuns = expinfo['nShiftRuns']
        except KeyError:
            print('nShiftRuns not specified. Using default: 10')
            self.nShiftRuns = 10


        try:
            self.anSizeFile = expinfo['AnSizeFile']
            tmp = open(self.anSizeFile, 'r')
            tmp.close()
        except KeyError:
            self.anSizeFile = self.trajectoryPath[:-4] + '_anSize.csv'
            print('AnSizeFile not specified. Using default.',self.anSizeFile)
        except FileNotFoundError:
            print('AnSizeFile specified but not found.')
            self.anSizeFile = -1

        try:
            self.episodes = expinfo['episodes']
            if self.episodes == -1:
                self.episodes = int(np.floor(((self.numFrames / self.fps) / 60) / self.episodeDur))
        except KeyError:
            print('episode number not specified. Using default: all')
            self.episodes = int(np.floor(((self.numFrames / self.fps) / 60) / self.episodeDur))

        print('Using: ', self.episodes, ' episodes. Episode duration (minutes): ', self.episodeDur)


        try:
            self.animalSet = expinfo['set']
        except KeyError:
            print('animalSet not specified. Using default: 0')
            self.animalSet = 0

        try:
            self.inDishTime = float(expinfo['inDish'])
        except KeyError:
            print('inDishTime not specified. Using default: 0')
            self.inDishTime = 0

        try:
            self.SaveNeighborhoodMaps = float(expinfo['SaveNeighborhoodMaps'])
        except KeyError:
            print('SaveNeighborhoodMaps not specified. Using default: True')
            self.SaveNeighborhoodMaps = True

        try:
            self.birthDayAll = expinfo['birthDayAll'].split()
        except KeyError:
            defBD = '2018-04-14-09-00'
            self.birthDayAll = np.repeat(defBD,self.numPairs)
            print('birthDayAll not specified. Using default: ', defBD)

            # concatenate dependent file paths (trajectories, pre-analysis)
        #head, tail = os.path.split(path)
        #head = os.path.normpath(head)

        #self.AnSizeFilePath = os.path.join(head, 'animalSize.txt')


    def PxPmmFromRois(self):
        #
        # calculate pixel scaling based on dish ROI file
        # bonsai VR experiments produce a ROIdef* file
        # idTracker experiment pipeline produces bgMed_scale* file with ROI information

        startDir, tail = os.path.split(self.trajectoryPath)
        ROIpath = glob.glob(startDir + '\\ROIdef*')  # skype experiments
        if len(ROIpath) < 1: # no skype rois, try idTracker pipeline output
            print('No skype roi found.')
            ROIpath = glob.glob(startDir + '\\bgMed_scale*')  # idTracker pipeline output
            if len(ROIpath) < 1:
                print('No idTracker pipeline ROI found.')
                ROIpath = None
                raise NameError('no ROIs found')
            else:
                print('idTracker pipeline ROI found.')
                rois = np.loadtxt(ROIpath[0], skiprows=1, delimiter=',')
                r_px = rois.mean(axis=0)[3]

        else:
            print('skype roi found')
            rois = np.loadtxt(ROIpath[0])
            r_px = rois.mean(axis=0)[-1]
        roiPath = ROIpath[0]
        pxPmm = 2 * r_px / self.arenaDiameter_mm
        return pxPmm, roiPath, rois

class experiment(object):
    #Class to collect, store and plot data belonging to one experiment
    def __init__(self, expDef):

        self.expInfo = None
        self.rawTra = None
        self.episodeAll = None
        self.anSize = None
        self.pair = []
        self.shiftList = None

        if type(expDef) == str:
            self.expInfo = ExperimentMeta(expDef)
            print('loading data', end="\r", flush=True)
            self.rawTra, self.episodeAll = self.loadData()
            print(' ...done. Shape: ', self.rawTra.shape)

        elif type(expDef) == pd.Series:

            self.expInfo = ExperimentMeta(expDef['txtPath'])
            self.expInfo.fillCompleteMeta(expDef)
            print('loading data', end="\r", flush=True)
            self.rawTra, self.episodeAll = self.loadData()
            print(' ...done. Shape: ', self.rawTra.shape)
            self.anSize = self.getAnimalSize()
            # self.load_animalShapeParameters()
            self.shiftList = [int(random.uniform(self.expInfo.minShift,
                                                 self.expInfo.episodeDur*self.expInfo.fps*60 - self.expInfo.minShift))
                              for x in range(self.expInfo.nShiftRuns)]
            self.splitToPairs()
            self.saveExpData()

        else:
            print('Wrong experiment definition argument. Provide TxtPath or pd.Series')

    def splitToPairs(self):

        for p in range(self.expInfo.numPairs):
            for i in range(self.expInfo.episodes):

                fps = self.expInfo.fps
                episodeDur = self.expInfo.episodeDur  # expected in minutes
                episodeFrames = fps * episodeDur * 60
                rng = np.array([i * episodeFrames, (i + 1) * episodeFrames]).astype('int')
                epCurr = self.episodeAll.loc[rng[0]]

                if self.expInfo.episodePLcode:
                    pairListNr = int(epCurr[:2])
                    pairList = self.expInfo.pairListAll[pairListNr * 16:(pairListNr + 1) * 16]
                else:
                    pairList = self.expInfo.pairListAll

                currPartnerAll = np.where(pairList[:, p])[0]

                for mp in range(currPartnerAll.shape[0]):
                    currPartner = currPartnerAll[mp]
                    Pair(shift=0, animalIDs=[p, currPartner], epiNr=i).linkExperiment(self)

    def saveExpData(self):

        # collect summary statistics for each animal-pair-episode
        # produce pandas df where each row is one episode
        # ordering is same as self.pair list

        animalSet = self.expInfo.animalSet
        episodeStartFrame = np.array([x.rng[0] for x in self.pair])
        inDishTime = (episodeStartFrame / (30 * 60)) + self.expInfo.inDishTime
        AnimalIndex = np.array([x.animalIDs[0] for x in self.pair])
        cp = np.array([x.animalIDs[1] for x in self.pair])  # current partner
        episodeName = self.episodeAll[episodeStartFrame]
        epiNr = np.array([x.epiNr for x in self.pair])
        bd = self.expInfo.birthDayAll[AnimalIndex]
        anSize = self.anSize[AnimalIndex]

        si = np.array([x.ShoalIndex() for x in self.pair])
        avgSpeed = np.array([x.avgSpeed()[0] for x in self.pair])
        avgSpeed_smooth = np.array([x.avgSpeed_smooth()[0] for x in self.pair])
        thigmoIndex = np.array([x.thigmoIndex()[0] for x in self.pair])
        boutDur = np.array([x.medBoutDur()[0] for x in self.pair])
        leadershipIndex = np.array([x.LeadershipIndex()[0] for x in self.pair])


        head, tail = os.path.split(self.expInfo.aviPath)

        try:
            datetime_object = datetime.strptime(tail[-18:-4], '%Y%m%d%H%M%S')
            tRun = np.array([datetime_object + timedelta(minutes=x) for x in inDishTime])
        except:
            print('avi file name does not provide valid experiment dateTime. Using default inDishTime')
            tRun = np.array(inDishTime)

        try:
            ageAll = np.array([(datetime_object - x).days for x in bd])
        except:
            print('Could not compute animal age at experiment date. Using default: 0')
            ageAll = np.zeros(avgSpeed.shape)

        df = pd.DataFrame(
            {'animalSet': animalSet,
             'animalIndex': AnimalIndex,
             'CurrentPartner': cp,
             'si': si,
             'episode': episodeName.squeeze(),
             'epStart': episodeStartFrame,
             'inDishTime': inDishTime,
             'epiNr': epiNr,
             'time': tRun,
             'birthDay': bd,
             'age': ageAll})

        df['avgSpeed'] = avgSpeed
        df['avgSpeed_smooth'] = avgSpeed_smooth
        df['anSize'] = anSize
        df['thigmoIndex'] = thigmoIndex
        df['boutDur'] = boutDur
        df['leadershipIndex'] = leadershipIndex

        csvFileOut = self.expInfo.trajectoryPath[:-4] + '_siSummary_epi' + str(self.expInfo.episodeDur) + '.csv'
        df.to_csv(csvFileOut, encoding='utf-8')

        if self.expInfo.SaveNeighborhoodMaps:
            numepiAll = self.expInfo.episodes * self.expInfo.numPairs
            nmAll = np.zeros((numepiAll, 3, 2, 62, 62))  # animal,[neighbor,speed,turn],[data,shuffle0],[mapDims]
            print('Computing neighborhood maps... ', end="\r", flush=True)
            for i in range(numepiAll):
                nmAll[i, 0, 0, :, :] = self.pair[i].animals[0].ts.neighborMat()
                nmAll[i, 1, 0, :, :] = self.pair[i].animals[0].ts.ForceMat_speed()
                nmAll[i, 2, 0, :, :] = self.pair[i].animals[0].ts.ForceMat_turn()
                self.pair[i].shift = self.shiftList[0]
                nmAll[i, 0, 1, :, :] = self.pair[i].animals[0].ts.neighborMat()
                nmAll[i, 1, 1, :, :] = self.pair[i].animals[0].ts.ForceMat_speed()
                nmAll[i, 2, 1, :, :] = self.pair[i].animals[0].ts.ForceMat_turn()
                self.pair[i].shift = 0
            print(' done. Saving maps...', end="\r", flush=True)
            npyFileOut = self.expInfo.trajectoryPath[:-4] + 'MapData.npy'
            np.save(npyFileOut, nmAll)
            print(' done.')

    def loadData(self):
        # read data for current experiment or many-dish-set
        # begin by reading first line to determine format
        print(' ', self.expInfo.trajectoryPath, end="\r", flush=True)
        firstLine = pd.read_csv(self.expInfo.trajectoryPath,
                                header=None,
                                nrows=1,
                                sep=':')

        if firstLine.values[0][0][0] == '(':
            rawData = pd.read_csv(self.expInfo.trajectoryPath,
                                  sep=',|\)|\(',
                                  engine='python',
                                  index_col=None,
                                  header=None,
                                  skipfooter=1,
                                  usecols=[2, 3, 6, 7, 10],
                                  names=np.arange(5))
            episodeAll = rawData[4]
            rawData.drop(rawData.columns[[4]], axis=1, inplace=True)

            # rawData= mat.reshape((mat.shape[0],2,2))

        elif firstLine.values[0][0][0] == 'X':
            rawData = pd.read_csv(self.expInfo.trajectoryPath,
                                  header=None,
                                  delim_whitespace=True,
                                  skiprows=1)

            episodeAll = pd.DataFrame(np.zeros(rawData.shape[0]))

        else:  # default for all bonsai VR experiments since mid 2017
            rawData = pd.read_csv(self.expInfo.trajectoryPath,
                                  header=None,
                                  delim_whitespace=True)
            episodeAll = rawData[rawData.columns[-1]]

        return rawData, episodeAll

    def getAnimalSize(self):

        if self.expInfo.anSizeFile == -1:
            anSize = getAnimalSizeFromVideo(currAvi=self.expInfo.aviPath,
                                            rawData=self.rawTra,
                                            numPairs=self.expInfo.numPairs,
                                            roiPath=self.expInfo.roiPath)

            print('saving anSize to', self.expInfo.anSizeFile)
            np.savetxt(self.expInfo.anSizeFile, anSize)
            print('Animal Size saved.')
        else:
            if np.equal(~os.path.isfile(self.expInfo.anSizeFile), -1):
                anSize = np.zeros(15)
            else:
                anSize = np.loadtxt(self.expInfo.anSizeFile)
        return anSize


    def addPair(self,pair):

        self.pair.append(pair)
        return self.pair[-1]
      
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
        plt.figtext(0,.01,self.expInfo.trajectoryPath)
        plt.figtext(0,.03,condition)
        plt.figtext(0,.05,self.idQuality)
        
        plt.subplot(4,4,1,rasterized=True)
        plt.cla()
        #plt.plot(self.pair.animals[0].ts.position().x(),self.pair.animals[0].ts.position().y(),'b.',markersize=1,alpha=0.1)
        #plt.plot(self.pair.animals[1].ts.position().x(),self.pair.animals[1].ts.position().y(),'r.',markersize=1,alpha=0.1)
        
        plt.plot(self.pair.animals[0].ts.position().x(),self.pair.animals[0].ts.position().y(),'.',markersize=1,alpha=0.1)
        plt.plot(self.pair.animals[1].ts.position().x(),self.pair.animals[1].ts.position().y(),'.',markersize=1,alpha=0.1)
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        plt.title('raw trajectories')  
        
        plt.subplot(4,4,3)
        plt.cla()
        PolhistBins=self.pair.animals[0].ts.PolhistBins()
        plt.step(PolhistBins[:-1], self.pair.animals[0].ts.Pol_n(),lw=2,where='mid')
        plt.step(PolhistBins[:-1], self.pair.animals[1].ts.Pol_n(),lw=2,where='mid')
        plt.axhline(1,ls=':',color='k')

        plt.xlabel('dist from center [mm]')
        plt.ylabel('p')
        plt.title('thigmotaxis') 
        #plt.ylim([0, .1])

        
        
        #plot IAD time series
        IAD=self.pair.IAD()
        x=np.arange(float(np.shape(IAD)[0]))/(self.expInfo.fps*60)
        plt.subplot(4,1,2,rasterized=True)
        plt.cla()
        plt.plot(x,IAD,'k.',markersize=1)
        plt.xlim([0, 90]) 
        plt.xlabel('time [minutes]')
        plt.ylabel('IAD [mm]')
        plt.title('Inter Animal Distance over time')
        
        #IAD histogram for raw and shifted data and simulated random dots
        plt.subplot(4,4,2)
        plt.cla()
        #get rid of nan
        #IAD=self.pair.IAD()
        IAD=IAD[~np.isnan(IAD)]
        histBins=np.arange(100)
        n, bins, patches = plt.hist(IAD, bins=histBins, normed=1, histtype='stepfilled',color='k')
        plt.step(histBins[:-1],self.spIADhist_m.T,lw=1,color=[.5,.5,.5])
        #simulate random uniform spacing
        rad=(self.expInfo.trajectoryDiameterPx/self.expInfo.pxPmm)/2
        num=1000
        dotList,dotND = randSpacing.randomDotsOnCircle(rad,num)
        n, bins, patches = plt.hist(dotND, bins=histBins, normed=1, histtype='step',color='r',linestyle='--')
        plt.xlabel('IAD [mm]')
        plt.ylabel('p')
        plt.title('IAD')
        plt.ylim([0, .05])
        
        plt.subplot(4,4,4)
        plt.cla()
        x=[1,2]
        y=[np.nanmean(IAD),self.spIAD_m([])]
        yerr=[0,self.spIAD_std()]
        barlist=plt.bar(x, y, yerr=yerr, width=0.5,color='k')
        barlist[1].set_color([.5,.5,.5])
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
        clim=[meanPosMat.min(),meanPosMat.max()]
        plt.imshow(meanPosMat,interpolation='none', extent=[-31,31,-31,31],clim=clim,origin='lower')
        plt.title('mean neighbor position')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        
        def pltArrow(size,color='m'):
            w=size/6.0        
            plt.arrow(
            0,            # x
            -size/2.0,            # y
            0,            # dx
            size,            # dy
            width=w,       # optional - defaults to 1.0
            length_includes_head=True,
            head_width=w*2,
            color=color
            )
            
        #neighborhood matrix for animal0
        #the vertial orientation was confirmed correct in march 2016
        cp=sns.color_palette()
        plt.subplot(4,4,13)
        plt.cla()
        PosMat=a1
        plt.imshow(PosMat,interpolation='none', extent=[-31,31,-31,31],clim=clim,origin='lower')
        pltArrow(self.AnSize[0,1],cp[0])
        plt.title('Animal 0 neighbor position')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        
        plt.subplot(4,4,14)
        plt.cla()
        PosMat=a2
        plt.imshow(PosMat,interpolation='none', extent=[-31,31,-31,31],clim=clim,origin='lower')
        pltArrow(self.AnSize[1,1],cp[1])
        plt.title('Animal 1 neighbor position')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        
        #LEADERSHIP
        
        plt.subplot(4,4,15)
        plt.cla()
        
        x=[1,2]
        barlist=plt.bar(x,self.LeadershipIndex, width=0.5,color=cp[0])
        barlist[1].set_color(cp[1])
        plt.title('Leadership')
        plt.ylabel('index')
        plt.ylim([-1, 1])
        plt.xlim([0.5, 3])
        
        tmp=[an.at_bottom_of_dish_stack() for an in self.pair.animals]
        #if np.sum(tmp)==0 or self.expInfo.txtTrajectories:
        plt.xticks([1.25,2.25],['same','dish'])

       # else:
       #     xtickList=[int(np.where(self.pair.StackTopAnimal==1)[0])+1.25,int(np.where(self.pair.StackTopAnimal==0)[0])+1.25]
       #     plt.xticks(xtickList,['top','bottom'])            
        
        #BODY SIZE
        plt.subplot(4,4,16)
        plt.cla()
        x=[1,2]
        barlist=plt.bar(x,self.AnSize[:,1], width=0.5,color=cp[0])
        barlist[1].set_color(cp[1])
        plt.xlim([0.5, 3])
        plt.xticks([1.25,2.25],['an0','an1'])
        
        plt.title('Body Size')
        plt.ylabel('length [mm]')
        

        plt.subplot(4,4,10)
        plt.title('accel=f(pos_n)')
        a1=self.pair.animals[0].ts.ForceMat_speedAndTurn()
        a2=self.pair.animals[1].ts.ForceMat_speedAndTurn()
        meanForceMat=np.nanmean(np.stack([a1,a2],-1),axis=2)
        johPlt.plotMapWithXYprojections(meanForceMat,3,outer_grid[9],31,10)
        
        
        
        
        plt.subplot(4,4,11)
        plt.cla()
        barlist=plt.bar([1,2],self.pair.avgSpeed(),width=0.5)
        barlist[1].set_color(cp[1])
        plt.title('avgSpeed')
        plt.xlim([0.5, 3])
        plt.xticks([1.25,2.25],['an0','an1'])
        try:
            plt.tight_layout()
        except:
            pass
                
        plt.show()
        
        mpl_is_inline = 'inline' in matplotlib.get_backend()
        if not mpl_is_inline:
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
        
        
        