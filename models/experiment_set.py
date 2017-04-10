# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 10:11:29 2016

@author: jlarsch
"""
import tkFileDialog
from models.experiment import experiment
from models.pair import shiftedPairSystematic
import os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from datetime import timedelta
import numpy as np
import seaborn as sns

class experiment_set(object):
    def __init__(self,csvFile,systShift=0, timeStim=0,sizePlot=0,episodes=1):
        
        self.systShift=systShift
        self.timeStim=timeStim
        self.sizePlot=sizePlot
        self.episodes=episodes
        self.csvFile=csvFile
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
                pairList=np.loadtxt(pairListFile,dtype='int')
            except:
                pairList=np.diag((-1,-1))+1
                
            numPairs=pairList.sum()
            
            print 'processing: ', row['aviPath']
            currAvi=row['aviPath']
            currTxt=row['txtPath']

            #read data for current experiment or many-dish-set
            #begin by reading first line to determine format
            firstLine=pd.read_csv(currTxt,header=None,nrows=1,sep=':')
            
            if firstLine.values[0][0][0]=='(':
                rawData = pd.read_csv(currTxt, sep=',|\)|\(', engine='python',index_col=None,header=None,skipfooter=1,usecols=[2,3,6,7,10],names=np.arange(5))
                episodeAll=rawData[4]
                rawData.drop(rawData.columns[[4]], axis=1,inplace=True)
                
                #rawData= mat.reshape((mat.shape[0],2,2))
            else:
                
                rawData=pd.read_csv(currTxt,header=None,delim_whitespace=True)
                episodeAll=rawData[rawData.columns[-1]]
                

                
            ee_StartFrames=[]
            ee_AnimalIndex=[]
            ee_AnimalSet=[]
            ee_inDishTime=[]
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
                
                
            try:                
                currAnimalSet=row['set']
            except:
                currAnimalSet=0
            
            try:                
                currInDishTime=float(row['inDish'])
            except:
                currInDishTime=0
                
                
            #cycle through all pairs in this data
            for p in range(numPairs):
                print p
                currAnimal=p
                currPartner=np.where(pairList[:,p])[0][0]
                
                currCols=[p*3,p*3+1,currPartner*3,currPartner*3+1]
                currDf=rawData[rawData.columns[currCols]]
                #p1=df[[6,7,9,10]].values[:30*60*120].reshape(-1,2,2)
                data=currDf.values.reshape(-1,2,2)
                
                self.experiments.append(experiment(currAvi,currTxt,data=data,forceCorrectPixelScaling=False,anSize=np.array([[5.,5.],[5,5]])))
                
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
                for i in range(numEpi):
                    ee_epiNr.append(i)
                    episodeFrames=self.experiments[-1].expInfo.fps*episodeDur*60
                    rng=np.arange(i*episodeFrames,(i+1)*episodeFrames).astype('int')
                    ee_StartFrames.append(rng[0])
                    ee_AnimalIndex.append(currAnimal)
                    ee_AnimalSet.append(currAnimalSet)
                    ee_inDishTime.append((rng[0]/(30*60))+currInDishTime)
                    try:
                        ee_epiName.append(episodeAll.loc[rng[0]])
                    except:
                        ee_epiName.append('n')
                    #print('episode: ', i, ' , ', rng[0],rng[-1])
                    self.ee.append(experiment(currAvi,currTxt,rng=rng,data=self.experiments[-1].rawTra,forceCorrectPixelScaling=False,anSize=np.array([[5.,5.],[5,5]])))
                    
                    ee_si.append(self.ee[-1].ShoalIndex())
                    
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
            si=np.array(ee_si)
    #            txtFile=np.array([x.expInfo.trajectoryPath for x in self.ee])
            animalSet=np.array(ee_AnimalSet)
            episodeStartFrame=np.array(ee_StartFrames)
            AnimalIndex=np.array(ee_AnimalIndex)
            inDishTime=np.array(ee_inDishTime)
            episodeName=np.array(ee_epiName)
            epiNr=np.array(ee_epiNr)
            
            head, tail = os.path.split(currAvi)
            try:
                datetime_object = datetime.strptime(tail[-18:-4], '%Y%m%d%H%M%S')

                tRun=[datetime_object + timedelta(minutes=x) for x in inDishTime]
            except:
                tRun=inDishTime
            
            
            df=pd.DataFrame(
            {'animalSet':animalSet,'animalIndex':AnimalIndex,
            'si':si,'episode':episodeName,'epStart':episodeStartFrame,
            'inDishTime':inDishTime,'epiNr':epiNr,'time':tRun})
            if currTxt=='none':
                csvFileOut=head+'_siSummary_epi'+str(episodeDur)+'.csv'
            else:
                csvFileOut=currTxt[:-4]+'_siSummary_epi'+str(episodeDur)+'.csv'
            df.to_csv(csvFileOut,encoding='utf-8')
            
            
               
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
        AnS=np.sqrt(np.sort(np.array([f.AnSize[:,1] for f in self.experiments]),axis=1))/3
        AnSm=np.mean(AnS,axis=1)
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