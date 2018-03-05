# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 10:11:29 2016

@author: jlarsch
"""
import tkFileDialog
from models.experiment import experiment
from models.pair import shiftedPairSystematic
import functions.video_functions as vf

import os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from datetime import timedelta
import numpy as np
import seaborn as sns
import glob

class experiment_set(object):
    def __init__(self,csvFile,
                 systShift=0,
                 timeStim=0,
                 sizePlot=0,
                 episodes=1,
                 arenaDiameter_mm=100,
                 recomputeAnimalSize=False,
                 birthDay=[],
                episodePLcode=False):
        
        self.systShift=systShift
        self.timeStim=timeStim
        self.sizePlot=sizePlot
        self.episodes=episodes
        self.csvFile=csvFile
        self.arenaDiameter_mm=arenaDiameter_mm
        self.recomputeAnimalSize=recomputeAnimalSize     
        self.birthDay=birthDay
        self.episodePLcode=episodePLcode
        self.process_csv_experiment_list()
        
        #self.plot_group_summaries()
        
        if timeStim:
            self.plot_IAD_overTime()
        
        if sizePlot:
            self.plot_shoaling_vs_size()
            
        if self.systShift:
            self.plot_syst_shift()
            
        mpl_is_inline = 'inline' in matplotlib.get_backend()
        if not mpl_is_inline:
            self.pdf.close()
        
    def process_csv_experiment_list(self):
    
        if self.csvFile==[]:
            self.csvFile=tkFileDialog.askopenfilename(initialdir=os.path.normpath('d:/data/b'))
        #save pdf summary in same foldera as csv
        currentTime=datetime.now()
        self.PdfFile=self.csvFile[:-4]+'_'+currentTime.strftime('%Y%m%d%H%M%S')+'.pdf'
        self.df=pd.read_csv(self.csvFile,sep=',')
        
        
        self.experiments=[]
        self.ee=[] #experiments chopped into episodes
        self.systShiftAll=[]

        
        mpl_is_inline = 'inline' in matplotlib.get_backend()
        if not mpl_is_inline:
            self.pdf = PdfPages(self.PdfFile)
                 
        for index, row in self.df.iterrows():
            
            try:                
                pairListFile=row['pairList']
                pairListAll=np.loadtxt(pairListFile,dtype='int')
            except:
                pairListAll=np.diag((-1,-1))+1
                
            if self.episodePLcode:
                numPairs=15
            else:
                numPairs=pairListAll.sum()
            print 'processing: ', row['aviPath']
            currAvi=row['aviPath']
            currTxt=row['txtPath']
            head, tail = os.path.split(currAvi)
            if (currTxt == []) or (currTxt=='none'):
                print 'reading: ',currTxt
                currTxt = os.path.join(head,'trajectories_nogaps.mat')
                mat=scipy.io.loadmat(currTxt)
                rawData= mat['trajectories']
                self.pxPmm=[]
                episodeAll=pd.DataFrame(np.zeros(rawData.shape[0]))
                
            else:
                # determine pixel scaling from arena ROI definition file if exists
                try:
                    startDir, tail = os.path.split(currTxt)
    
                    ROIpath=    glob.glob(startDir+'\\ROIdef*') #skype experiments
                    if len(ROIpath)<1:
                        ROIpath=    glob.glob(startDir+'\\bgMed_scale*') #skype experiments
                        print 'no skype roi found'
                        if len(ROIpath)<1:
                            print 'no DishPair roi found'
                            ROIpath=[]
                        else:
                            print 'DishPair roi found'
                            rois=np.loadtxt(ROIpath[0],skiprows=1,delimiter=',')
                            r_px=rois.mean(axis=0)[3]
                        
                    else:
                        print 'skype roi found'
                        rois=np.loadtxt(ROIpath[0])
                        r_px=rois.mean(axis=0)[-1]
                        
                    self.roiPath=ROIpath[0]
                    self.pxPmm=2*r_px/self.arenaDiameter_mm
    
    
                except:
                    self.pxPmm=[]
                    self.roiPath=[]
                print 'pxPmm:',self.pxPmm
                print 'ROIpath:',self.roiPath
                #read data for current experiment or many-dish-set
                #begin by reading first line to determine format
                print 'reading: ',currTxt
                firstLine=pd.read_csv(currTxt,header=None,nrows=1,sep=':')
                
                if firstLine.values[0][0][0]=='(':
                    rawData = pd.read_csv(currTxt, sep=',|\)|\(', engine='python',index_col=None,header=None,skipfooter=1,usecols=[2,3,6,7,10],names=np.arange(5))
                    episodeAll=rawData[4]
                    rawData.drop(rawData.columns[[4]], axis=1,inplace=True)
                    
                    #rawData= mat.reshape((mat.shape[0],2,2))
                elif firstLine.values[0][0][0]=='X':
                    rawData=pd.read_csv(currTxt,header=None,delim_whitespace=True,skiprows =1)
                    episodeAll=pd.DataFrame(np.zeros(rawData.shape[0]))
                else:
                    
                    rawData=pd.read_csv(currTxt,header=None,delim_whitespace=True)
                    episodeAll=rawData[rawData.columns[-1]]
                #print rawData.head()

                
            ee_StartFrames=[]
            ee_AnimalIndex=[]
            ee_AnimalSet=[]
            ee_inDishTime=[]
            ee_birthDay=[]
            ee_epiName=[]
            ee_si=[]
            ee_epiNr=[]
        
            
            try:                
                self.episodes=row['episodes']
            except:
                pass
                
            try:                
                episodeDur=row['epiDur']
            except:
                episodeDur=60
            
            print episodeDur
            try:                
                currAnimalSet=row['set']
            except:
                currAnimalSet=0
            
            try:                
                currInDishTime=float(row['inDish'])
            except:
                currInDishTime=0
                
            #get Animal Size
            #try:
            AnSizeFileOut=currTxt[:-4]+'_anSize.csv'
            
            if self.recomputeAnimalSize:
               anSize=self.getAnimalSizeFromVideo(currAvi,rawData,numPairs=numPairs)
               print 'saving anSize to', AnSizeFileOut
               np.savetxt(AnSizeFileOut,anSize)
               print 'Animal Size saved.'
            else:
                if np.equal(~os.path.isfile(AnSizeFileOut),-1):
                    anSize=np.zeros(15)
                else:
                    anSize= np.loadtxt(AnSizeFileOut)
            #except:
            #    print 'no animal size'
               
            #cycle through all pairs in this data
            nmp=0
            for p in range(numPairs):
                print p
                currAnimal=p
                currPartner=np.where(pairListAll[:,p])[0][0]
                
                if currPartner <= numPairs:
                    ps=0
                else:
                    ps=anSize[currPartner]
                    
                currSize=[anSize[p],ps]
                
                currCols=[p*3,p*3+1,p*3+2,currPartner*3,currPartner*3+1,currPartner*3+2]
                try:                
                    currDf=rawData[rawData.columns[currCols]]
                #p1=df[[6,7,9,10]].values[:30*60*120].reshape(-1,2,2)
                    data=currDf.values.reshape(-1,2,3) #(time,animal,(x,y,ori))
                    currCenterPx=rois[p,-1]+2
                    currCenterPx=np.array([currCenterPx,currCenterPx])               
                except:
                    data=rawData
                    currCenterPx=[]

                
                print 'animalSize:',currSize
                self.experiments.append(experiment(currAvi, currTxt,
                                                   data=data,
                                                   pxPmm=self.pxPmm,
                                                   anSize=currSize,
                                                   arenaCenterPx=currCenterPx,
                                                   episodeMarker=episodeAll.values))
                
                numFrames=self.experiments[-1].expInfo.numFrames
                fps=self.experiments[-1].expInfo.fps
                
                
                #vp=vf.getVideoProperties(currAvi) #video properties  
                #numFrames=vp['nb_frames']
                #fps=vp['fps']
                
                
                if self.episodes==-1:
                    numEpi=int(np.floor(((numFrames/fps)/60) /episodeDur))
                    print 'new episode number',numEpi
                else:
                    numEpi=int(self.episodes)
                    print 'using episode number',numEpi, episodeDur
                    
                if self.birthDay==[]:
                    currBirthDay=np.nan
                else:
                    currBirthDay=self.birthDay[p]
                

                #try:
                #    episodeAll = pd.read_csv(currTxt, names=['episode'], sep=',|\)', engine='python',usecols=[7],index_col=None,header=None)
                #    print('read episode names')
                #except:
                #    episodeAll=[]
                
    #            mpl_is_inline = 'inline' in matplotlib.get_backend()
    #            if not mpl_is_inline:
    #                try:
    #                    self.experiments[index].plotOverview(row['condition'])
    #                    self.pdf.savefig()  # saves the current figure as pdf page
    #                    plt.close()
    #                except:
    #                    pass
                
               # print numEpi
                #print 'data:',self.experiments[-1].rawTra.shape
                for i in range(numEpi):
                    episodeFrames=self.experiments[-1].expInfo.fps*episodeDur*60
                    rng=np.arange(i*episodeFrames,(i+1)*episodeFrames).astype('int')

                    try:
                        epCurr=episodeAll.loc[rng[0]]
                    except:
                        epCurr='n'
                        
                    if self.episodePLcode:
                        pairListNr=int(epCurr[:2])
                        pairList=pairListAll[pairListNr*16:(pairListNr+1)*16]
                    else:
                        pairList=pairListAll
                        

                    currPartnerAll=np.where(pairList[:,p])[0]
                    #currPartner=np.where(pairList[:,p])[0][0]
                    
                    for mp in range(currPartnerAll.shape[0]):
                        
                        ee_epiNr.append(i)
                        ee_epiName.append(epCurr)
                        ee_StartFrames.append(rng[0])
                        ee_AnimalIndex.append(currAnimal)
                        ee_AnimalSet.append(currAnimalSet)
                        ee_inDishTime.append((rng[0]/(30*60))+currInDishTime)
                        ee_birthDay.append(currBirthDay)
                        
                        currPartner=currPartnerAll[mp]
                        if currPartner >= numPairs:
                            ps=0
                        else:
                            ps=anSize[currPartner]
                            
                        currSize=[anSize[p],ps]
                        
                        currCols=[p*3,p*3+1,p*3+2,currPartner*3,currPartner*3+1,currPartner*3+2]
                        try:                        
                            currDf=rawData[rawData.columns[currCols]].iloc[rng]
                            data=currDf.values.reshape(-1,2,3) #(time,animal,(x,y,ori))
                            currCenterPx=rois[p,-1]+2
                            currCenterPx=np.array([currCenterPx,currCenterPx])
                        except:
                            
                            print 'using all data'
                            data=rawData[rng,:,:]
                            #data=currDf.values.reshape(-1,2,3) #(time,animal,(x,y,ori))
                            currCenterPx=[]
                            currCenterPx=[]
                        #    data=rawData
                        #    currCenterPx=[]
                    
                        nmp+=1
                        self.ee.append(experiment(currAvi,currTxt,
                                                  #rng=rng,
                                                  data=data,
                                                  pxPmm=self.pxPmm,
                                                  anSize=currSize,
                                                  arenaCenterPx=currCenterPx,
                                                  episodeMarker=episodeAll.values[rng]))
                        self.ee[-1].cp=currPartner
                        
                        #ee_si.append(self.ee[-1].ShoalIndex())
                        
                        mpl_is_inline = 'inline' in matplotlib.get_backend()
                        if not mpl_is_inline:
                            try:
                                self.ee[(numEpi*index)+i].plotOverview(row['condition'])
                                self.pdf.savefig()  # saves the current figure as pdf page
                                plt.close()
                            except:
                                pass
            
            if self.systShift:
                self.systShiftAll.append(shiftedPairSystematic(self.experiments[index].Pair, self.experiments[index].expInfo, 60))
             
             
            #save results for each file
            
    #            txtFile=np.array([x.expInfo.trajectoryPath for x in self.ee])
            animalSet=np.array(ee_AnimalSet)
            episodeStartFrame=np.array(ee_StartFrames)
            AnimalIndex=np.array(ee_AnimalIndex)
            inDishTime=np.array(ee_inDishTime)
            episodeName=np.array(ee_epiName)
            epiNr=np.array(ee_epiNr)
            bd=np.array(ee_birthDay)

            si=np.array([x.ShoalIndex() for x in self.ee])[-nmp:]
            cp=np.array([x.cp for x in self.ee])[-nmp:]
            avgSpeed=np.array([x.avgSpeed()[0] for x in self.ee])[-nmp:]
            avgSpeed_smooth=np.array([x.avgSpeed_smooth()[0] for x in self.ee])[-nmp:]
            thigmoIndex=np.array([x.thigmoIndex[0] for x in self.ee])[-nmp:]
            boutDur=np.array([x.medBoutDur[0] for x in self.ee])[-nmp:]
            leadershipIndex=np.array([x.LeadershipIndex[0] for x in self.ee])[-nmp:]
            anSize=np.array([x.AnSize[0] for x in self.ee])[-nmp:]
            
            head, tail = os.path.split(currAvi)
            
            try:
                
                datetime_object = datetime.strptime(tail[-18:-4], '%Y%m%d%H%M%S')
                tRun=np.array([datetime_object + timedelta(minutes=x) for x in inDishTime])
            except:
                tRun=np.array(inDishTime)
            
            try:
                ageAll=np.array([(datetime_object-x).days for x in bd])
            except:
                ageAll=np.zeros(avgSpeed.shape)
#            print animalSet.shape
#            print cp.shape
#            print ageAll.shape
#            print tRun.shape
#            print animalSet.shape
#            print AnimalIndex.shape
#            print si.shape
#            print episodeName.squeeze().shape
            
            df=pd.DataFrame(
            {'animalSet':animalSet,'animalIndex':AnimalIndex,'CurrentPartner':cp,
            'si':si,'episode':episodeName.squeeze(),'epStart':episodeStartFrame,
            'inDishTime':inDishTime,'epiNr':epiNr,'time':tRun,'birthDay':bd,'age':ageAll})
            

            df['avgSpeed']=avgSpeed
            df['avgSpeed_smooth']=avgSpeed_smooth
            df['anSize']=anSize
            df['thigmoIndex']=thigmoIndex
            df['boutDur']=boutDur
            df['leadershipIndex']=leadershipIndex
            if currTxt=='none':
                csvFileOut=head+'_siSummary_epi'+str(episodeDur)+'.csv'
            else:
                csvFileOut=currTxt[:-4]+'_siSummary_epi'+str(episodeDur)+'.csv'
            df.to_csv(csvFileOut,encoding='utf-8')
            
            numepiAll=numEpi*numPairs
            nmAll=np.zeros((numepiAll,3,2,62,62)) #animal,[neighbor,speed,turn],[data,shuffle0],[mapDims]
            for i in range(numepiAll):
                nmAll[i,0,0,:,:]=self.ee[i].pair.animals[0].ts.neighborMat()
                nmAll[i,0,1,:,:]=self.ee[i].sPair[0].animals[0].ts.neighborMat()
                nmAll[i,1,0,:,:]=self.ee[i].pair.animals[0].ts.ForceMat_speed()
                nmAll[i,1,1,:,:]=self.ee[i].sPair[0].animals[0].ts.ForceMat_speed()
                nmAll[i,2,0,:,:]=self.ee[i].pair.animals[0].ts.ForceMat_turn()
                nmAll[i,2,1,:,:]=self.ee[i].sPair[0].animals[0].ts.ForceMat_turn()
                
            npyFileOut=currTxt[:-4]+'MapData.npy'
            np.save(npyFileOut,nmAll)
    
    def getAnimalSizeFromVideo(self,currAvi,rawData,sizePercentile=40,numPairs=15):           
        xMax=2048.0 #relevant for openGL scaling
        numFrames=2000#2000
        maxFrames=100000
        boxSize=200
        head, tail = os.path.split(currAvi)
        print 'processing animals size:', currAvi
        ROIPath=self.roiPath#glob.glob(head+'\\ROI*.csv')
        
        head, tail = os.path.split(ROIPath)
        #print tail
        if tail[:3]=='ROI':
            rois=np.loadtxt(ROIPath)
            correct=True
            
            #for virtual pairing, can use random frames, no need to avoid collisions
            traAll=np.zeros((numFrames,numPairs,2))
            frames=np.random.randint(1000,20000,numFrames)
            for i in range(numPairs):
                currCols=[i*3,i*3+1]
                rawTra=rawData[rawData.columns[currCols]].values
                tra=rawTra.copy()
                xx=rawTra[:,0]
                yy=rawTra[:,1]
                xoff=rois[i,0]
                yoff=rois[i,1]
                xx,yy=self.correctFish(xx,yy,xoff,yoff,xMax,53.)
                tra[:,0]=xx+xoff
                tra[:,1]=yy+yoff
                traAll[:,i,:]=tra[frames,:].copy()           
            
        else:
            currCols=[0,1,3,4]
            rawTra=rawData[rawData.columns[currCols]].values.reshape(-1,2,2)
            rois=np.loadtxt(ROIPath,skiprows=1,delimiter=',')
            correct=False
        
            haveFrames=0
            frames=np.zeros(numFrames).astype('int')
            dist=np.zeros(numFrames)
            
            triedFr=[]
            triedD=[]
            print 'determining 2000 frames to read animal size...'
            while haveFrames<numFrames:
                tryFrame=np.random.randint(1000,maxFrames,1)
                minDist=np.max(np.abs(np.diff(rawTra[tryFrame,:,:],axis=1)))
                if minDist>boxSize:
                    frames[haveFrames]=int(tryFrame)
                    dist[haveFrames]=minDist
                    haveFrames += 1
                else:
                    triedFr.append(tryFrame)
                    triedD.append(minDist)
            
            print 'done. tried',len(triedFr),'frames'
            traAll=np.zeros((numFrames,numPairs,2))
            for i in range(numPairs):
                currCols=[i*3,i*3+1]
                rawTra=rawData[rawData.columns[currCols]].values
                tra=rawTra.copy()
                traAll[:,i,:]=tra[frames,:].copy()           



        invert=(not correct)
        print 'inverting video',invert
        tmp=vf.getAnimalLength(currAvi,frames,traAll,threshold=5,invert=invert)

        anSize=[]
        MA=[]
        for i in range(numPairs):
            MA.append(np.max(tmp[:,i,2:4],axis=1))
            anSize.append(np.nanpercentile(MA[i],sizePercentile))
        
        fnSizeAll=head+'\\anSizeAll.csv'
        #print 'writing size to',fnSizeAll
        with open(fnSizeAll,'wb') as f:
            np.savetxt(f,np.array(MA),fmt='%.5f')
            
        return np.array(anSize)/self.pxPmm
               
    def plot_group_summaries(self):
        self.df['ShoalIndex']=pd.DataFrame([f.ShoalIndex([]) for f in self.experiments])
#        self.df['totalTravel']=pd.DataFrame([f.totalPairTravel for f in self.experiments])
        self.df['avgSpeed']=pd.DataFrame([f.avgSpeed for f in self.experiments])
        fig, axes = plt.subplots(nrows=1, ncols=2)
        
        sns.boxplot(ax=axes[0],x='condition',y='ShoalIndex',data=self.df,width=0.2)
        sns.stripplot(ax=axes[0],x='condition',y='ShoalIndex',data=self.df, size=4,jitter=True,edgecolor='gray')

        #sns.boxplot(ax=axes[1],x='condition',y='avgSpeed',data=self.df,width=0.2)
        #sns.stripplot(ax=axes[1],x='condition',y='avgSpeed',data=self.df, size=4,jitter=True,edgecolor='gray')
        
        axes[0].set_ylim(-0.1,1)
        axes[1].set_ylim(0,)
        mpl_is_inline = 'inline' in matplotlib.get_backend()
        if not mpl_is_inline:
            self.pdf.savefig()
            
    def plot_IAD_overTime(self):
        plt.figure(figsize=(8, 2))
        IADall=[]
        IADall.append([x.pair.IAD()[0:30*60*50] for x in self.experiments])
        t=np.nanmean(IADall,axis=1)
        smIAD_mAll=np.nanmean([f.spIAD_m() for f in self.experiments])
        x=np.arange(float(np.shape(t[0])[0]))/(self.experiments[0].expInfo.fps*60)
        #stim=np.arange(float(np.shape(t[0])[0]))/(experiment[0].expInfo.fps*60)
        plt.plot(x,t[0])
        plt.plot([0, np.shape(t[0])[0]], [smIAD_mAll, smIAD_mAll], 'r:', lw=1)
        plt.xlim([0, 100])
        plt.ylim([0, 100])
        plt.xlabel('time [minutes]')
        plt.ylabel('IAD [mm]')
        plt.title('Inter Animal Distance over time')
        self.IAD_overTime=t
        
        mpl_is_inline = 'inline' in matplotlib.get_backend()
        if not mpl_is_inline:
            self.pdf.savefig()
        
            
    def plot_shoaling_vs_size(self):
        AnS=np.sqrt(np.sort(np.array([f.AnSize[0] for f in self.experiments]),axis=0))/3
        AnSm=np.mean(AnS,axis=0)
        si=([f.ShoalIndex([]) for f in self.experiments])
        xErr=np.abs(AnS[:,0]-AnS[:,1])/2
        plt.figure()
        ebx=plt.errorbar(AnSm,si,xerr=xErr,ecolor='k',fmt=' ')
        ebx[-1][0].set_linestyle(':')
        eby=plt.errorbar(AnSm,si,yerr=0.01,ecolor='gray',fmt=' ')
        plt.scatter(AnS[:,0],si,s=50,marker='d',c='m')
        plt.scatter(AnS[:,1],si,s=50,marker='d',c='g')
        self.pdf.savefig()
        
        leadershipIndex=np.array([f.LeadershipIndex for f in self.experiments])
        AnSd=np.zeros(np.shape(AnS))
        AnSd[:,0]=AnS[:,1]-AnS[:,0]
        AnSd[:,1]=AnS[:,0]-AnS[:,1]
        plt.figure()
        plt.scatter(AnSd[:],leadershipIndex[:])
        plt.xlabel('size difference [mm]')
        plt.ylabel('leadership index')
        mpl_is_inline = 'inline' in matplotlib.get_backend()
        if not mpl_is_inline:
            self.pdf.savefig()
        
    def plot_syst_shift(self):
        a=np.array(self.systShiftAll)
        m=np.mean(a,axis=0)
        e=np.std(a,axis=0)
        plt.figure()
        plt.plot(a.T,color=[.7,.7,.7],zorder=1)
        plt.errorbar(range(60),m,yerr=e,color='k',zorder=2)
        plt.plot(m,lw=5,color='k')
        plt.xlim([-1,60])
        plt.xlabel('time shift [s]')
        plt.ylabel('inter animal distance [mm]')
        mpl_is_inline = 'inline' in matplotlib.get_backend()
        if not mpl_is_inline:
            self.pdf.savefig()
            


    def correctFish(self,x,y,xOff,yOff,xMax,xMaxCm):
      yMax=1280.0
      scale=xMax/xMaxCm
      xR=x+xOff
      yR=y+yOff
    
      xR=xR+(scale*(xR-xMax/2.)/(79*scale))
      yR=yR+(scale*(yR-yMax/2.)/(79*scale))
    
      return xR-xOff,yR-yOff