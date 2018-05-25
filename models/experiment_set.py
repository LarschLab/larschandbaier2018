# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 10:11:29 2016

@author: jlarsch
"""

from tkinter import filedialog as tkFileDialog
from models.experiment import experiment
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime


class experiment_set(object):
    def __init__(self, csvFile=[]):
        
        self.csvFile = csvFile
        if not self.csvFile:
            self.csvFile = tkFileDialog.askopenfilename()

        self.df = None
        self.experiments = None
        self.pdf = None

        self.process_csv_experiment_list()

    def process_csv_experiment_list(self):

        self.df = pd.read_csv(self.csvFile, sep=',')
        self.experiments = []

        for index, row in self.df.iterrows():

            print('processing: ', row['aviPath'])
            self.experiments.append(experiment(row))

    def saveExperimentOverviewPDF(self, experiment, label):
        # save pdf summary in same foldera as csv
        mpl_is_inline = 'inline' in matplotlib.get_backend()
        if not mpl_is_inline:
            currentTime = datetime.now()
            experiment.PdfFile=experiment.csvFile[:-4]+'_'+currentTime.strftime('%Y%m%d%H%M%S')+'.pdf'
            try:
                self.pdf = PdfPages(self.PdfFile)
                experiment.plotOverview(label)
                experiment.pdf.savefig()  # saves the current figure as pdf page
                plt.close()
            except:
                print('PDF could not be created')
                pass
