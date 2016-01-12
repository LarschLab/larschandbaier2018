__author__ = 'jlarsch'

#import Tkinter
#import tkFileDialog
import joFishHelper
#import os
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

#df=pd.read_csv('d:/data/b/GRvsHet_pairs_10cmDish.csv',sep=',')
df=pd.read_csv('d:/data/b/GRvsHet_pairs_10cmDish_20160107.csv',sep=',')

experiment=[];

with PdfPages('d:/data/b/GR2016all.pdf') as pdf:

    for index, row in df.iterrows():
        print 'processing: ', row['aviPath']
        currAvi=row['aviPath']
        #avi_path = tkFileDialog.askopenfilename(initialdir=os.path.normpath('d:/data/b'))
        #Tkinter.Tk().withdraw() # Close the root window - not working?
        experiment.append(joFishHelper.experiment(currAvi))
        experiment[index].plotOverview()
        pdf.savefig()  # saves the current figure into a pdf page
        plt.close()
        
        
    df['ShoalIndex']=pd.DataFrame([f.ShoalIndex for f in experiment])
    bp=df.boxplot(column=['ShoalIndex'],by=['condition'])
    plt.ylim([0,1])
    pdf.savefig()