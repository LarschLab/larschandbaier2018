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
import datetime
import numpy as np
import seaborn as sns

class experiment_set(object):
    def __init__(self,systShift=0, timeStim=1,sizePlot=0):
        
        self.systShift=systShift
        self.timeStim=timeStim
        self.sizePlot=sizePlot
        
        self.process_csv_experiment_list()
        self.plot_group_summaries()
        
        if timeStim:
            self.plot_IAD_overTime()
        
        if sizePlot:
            self.plot_shoaling_vs_size()
            
        if self.systShift:
            self.plot_syst_shift()
            
            
        self.pdf.close()
        
    def process_csv_experiment_list(self):
    
        csvFile=tkFileDialog.askopenfilename(initialdir=os.path.normpath('d:/data/b'))
        #save pdf summary in same foldera as csv
        currentTime=datetime.datetime.now()
        self.PdfFile=csvFile[:-4]+'_'+currentTime.strftime('%Y%m%d%H%M%S')+'.pdf'
        self.df=pd.read_csv(csvFile,sep=',')
        self.experiments=[]
        self.systShiftAll=[]
        
        self.pdf = PdfPages(self.PdfFile)
            
        for index, row in self.df.iterrows():
            print 'processing: ', row['aviPath']
            currAvi=row['aviPath']
            self.experiments.append(experiment(currAvi))
            self.experiments[index].plotOverview(row['condition'])
            self.pdf.savefig()  # saves the current figure as pdf page
            plt.close()
            
            if self.systShift:
                self.systShiftAll.append(shiftedPairSystematic(self.experiments[index].Pair, self.experiments[index].expInfo, 60))
    
    def plot_group_summaries(self):
        self.df['ShoalIndex']=pd.DataFrame([f.ShoalIndex for f in self.experiments])
        self.df['totalTravel']=pd.DataFrame([f.totalPairTravel for f in self.experiments])
        self.df['avgSpeed']=pd.DataFrame([f.avgSpeed for f in self.experiments])
        fig, axes = plt.subplots(nrows=1, ncols=2)
        
        sns.boxplot(ax=axes[0],x='condition',y='ShoalIndex',data=self.df,width=0.2)
        sns.stripplot(ax=axes[0],x='condition',y='ShoalIndex',data=self.df, size=4,jitter=True,edgecolor='gray')

        sns.boxplot(ax=axes[1],x='condition',y='avgSpeed',data=self.df,width=0.2)
        sns.stripplot(ax=axes[1],x='condition',y='avgSpeed',data=self.df, size=4,jitter=True,edgecolor='gray')
        
        axes[0].set_ylim(-0.1,1)
        axes[1].set_ylim(0,)
        
        self.pdf.savefig()
            
    def plot_IAD_overTime(self):
        plt.figure(figsize=(8, 2))
        IADall=[]
        IADall.append([x.Pair.IAD[0:30*60*90] for x in self.experiments])
        t=np.nanmean(IADall,axis=1)
        smIAD_mAll=np.nanmean([f.sPair.spIAD_m for f in self.experiments])
        x=np.arange(float(np.shape(t[0])[0]))/(self.experiments[0].expInfo.fps*60)
        #stim=np.arange(float(np.shape(t[0])[0]))/(experiment[0].expInfo.fps*60)
        plt.plot(x,t[0])
        plt.plot([0, np.shape(t[0])[0]], [smIAD_mAll, smIAD_mAll], 'r:', lw=1)
        plt.xlim([0, 100])
        plt.ylim([0, 100])
        plt.xlabel('time [minutes]')
        plt.ylabel('IAD [mm]')
        plt.title('Inter Animal Distance over time')
        
        self.pdf.savefig()
        
            
    def plot_shoaling_vs_size(self):
        AnS=np.sqrt(np.sort(np.array([f.AnSize[:,1] for f in self.experiments]),axis=1))/3
        AnSm=np.mean(AnS,axis=1)
        si=([f.ShoalIndex for f in self.experiments])
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
        self.pdf.savefig()