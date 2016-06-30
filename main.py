__author__ = 'jlarsch'

#import Tkinter
import tkFileDialog
import sys
sys.path.append('C:/Users/johannes/Dropbox/python/zFishBehavior/dishGroupBehavior/')
import experiment as xp
import os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

batchmode=0
timeStim=0
sizePlot=0
systShift=0

if batchmode:
    
    def process_csv_experiment_list():
        
        csvFile=tkFileDialog.askopenfilename(initialdir=os.path.normpath('d:/data/b'))
        PdfFile=csvFile[:-4]+'.pdf'
        df=pd.read_csv(csvFile,sep=',')
        experiment=[]
        systShiftAll=[]
        with PdfPages(PdfFile) as pdf:
            
            for index, row in df.iterrows():
                print 'processing: ', row['aviPath']
                currAvi=row['aviPath']
                experiment.append(xp.experiment(currAvi))
                experiment[index].plotOverview(row['condition'])
                pdf.savefig()  # saves the current figure as pdf page
                plt.close()
                
                if systShift:
                    systShiftAll.append(xp.shiftedPairSystematic(experiment[index].Pair, experiment[index].expInfo, 60))
    def 
            df['ShoalIndex']=pd.DataFrame([f.ShoalIndex for f in experiment])
            df['totalTravel']=pd.DataFrame([f.totalPairTravel for f in experiment])
            df['avgSpeed']=pd.DataFrame([f.avgSpeed for f in experiment])
            IADall=[]
            IADall.append([x.Pair.IAD[0:30*60*90] for x in experiment])
            t=np.nanmean(IADall,axis=1)
            smIAD_mAll=np.nanmean([f.sPair.spIAD_m for f in experiment])
            
            fig, axes = plt.subplots(nrows=1, ncols=2)
            
            sns.boxplot(ax=axes[0],x='condition',y='ShoalIndex',data=df,width=0.2)
            sns.stripplot(ax=axes[0],x='condition',y='ShoalIndex',data=df, size=4,jitter=True,edgecolor='gray')
    
            sns.boxplot(ax=axes[1],x='condition',y='avgSpeed',data=df,width=0.2)
            sns.stripplot(ax=axes[1],x='condition',y='avgSpeed',data=df, size=4,jitter=True,edgecolor='gray')
            
            axes[0].set_ylim(-0.1,1)
            axes[1].set_ylim(0,)
            pdf.savefig()
        
        if timeStim:
            fig=plt.figure(figsize=(8, 2))
            x=np.arange(float(np.shape(t[0])[0]))/(experiment[0].expInfo.fps*60)
            stim=np.arange(float(np.shape(t[0])[0]))/(experiment[0].expInfo.fps*60)
            plt.plot(x,t[0])
            plt.plot([0, np.shape(t[0])[0]], [smIAD_mAll, smIAD_mAll], 'r:', lw=1)
            plt.xlim([0, 100])
            plt.ylim([0, 100])
            plt.xlabel('time [minutes]')
            plt.ylabel('IAD [mm]')
            plt.title('Inter Animal Distance over time')
        
        pdf.savefig()
        
        if sizePlot:
            AnS=np.sqrt(np.sort(np.array([f.AnSize[:,1] for f in experiment]),axis=1))/3
            AnSm=np.mean(AnS,axis=1)
            si=([f.ShoalIndex for f in experiment])
            xErr=np.abs(AnS[:,0]-AnS[:,1])/2
            plt.figure()
            ebx=plt.errorbar(AnSm,si,xerr=xErr,ecolor='k',fmt=' ')
            ebx[-1][0].set_linestyle(':')
            eby=plt.errorbar(AnSm,si,yerr=0.01,ecolor='gray',fmt=' ')
            plt.scatter(AnS[:,0],si,s=50,marker='d',c='m')
            plt.scatter(AnS[:,1],si,s=50,marker='d',c='g')
            pdf.savefig()
            
            leadershipIndex=np.array([f.LeadershipIndex for f in experiment])
            AnSd=np.zeros(np.shape(AnS))
            AnSd[:,0]=AnS[:,1]-AnS[:,0]
            AnSd[:,1]=AnS[:,0]-AnS[:,1]
            plt.figure()
            plt.scatter(AnSd[:],leadershipIndex[:])
            plt.xlabel('size difference [mm]')
            plt.ylabel('leadership index')
            pdf.savefig()
        
        if systShift:
            a=np.array(systShiftAll)
            m=np.mean(a,axis=0)
            e=np.std(a,axis=0)
            plt.figure()
            plt.plot(a.T,color=[.7,.7,.7],zorder=1)
            plt.errorbar(range(60),m,yerr=e,color='k',zorder=2)
            plt.plot(m,lw=5,color='k')
            plt.xlim([-1,60])
            plt.xlabel('time shift [s]')
            plt.ylabel('inter animal distance [mm]')
            pdf.savefig()
            

            
else:    
    avi_path = tkFileDialog.askopenfilename(initialdir=os.path.normpath('d:/data/b/2016/20160314_TLStacks/1/a/1'))
    experiment=xp.experiment(avi_path)
    experiment.plotOverview()


    
    