# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:38:07 2017

@author: jlarsch
"""



import seaborn as sns
import matplotlib as mpl
import pylab
inToCm=2.54

def paper():

    
    print('using plot settings for paper plots')
    inToCm=2.54
    sns.set()
    sns.set_style("ticks")
    sns.set_context("paper")
    
    mpl.rcParams['lines.linewidth'] = 1
    pylab.rcParams['figure.autolayout'] = False
    mpl.rcParams['axes.labelsize'] = 10
    mpl.rcParams['axes.titlesize'] = 10
    mpl.rcParams['figure.titlesize'] = 10
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    pylab.rcParams['font.size'] = 10
    pylab.rcParams['lines.linewidth'] = 1.0
    pylab.rcParams['lines.markersize'] = 10
    pylab.rcParams['legend.fontsize'] = 8
    pylab.rcParams['figure.dpi'] = 100
    pylab.rcParams['savefig.dpi'] = 100
    pylab.rcParams['figure.figsize'] = (18/inToCm, 6/inToCm)