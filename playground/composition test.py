# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 22:54:37 2016

@author: johannes
"""
import numpy as np

class experiment:
    def __init__(self,fn):
        self.fn=fn
        self.numAn=2
        self.pair=[]
        
        self.tra=np.zeros((1000,2,2))
        
        pair(2).linkExperiment(self)
        
    def addPair(self,pair):
        self.pair=pair
        return self.pair
        

class pair:
    def __init__(self,numAn):
        self.animals={}
        self.tra=[]
        self.numAn=numAn
        

        
        
    def addAnimal(self,animal):
        self.animals[animal.ID]=animal
        return self.animals[animal.ID]
    
    def linkExperiment(self,experiment):
        self.experiment=experiment
        experiment.addPair(self)
        self.tra=experiment.tra
        
        for i in range(self.numAn):
            animal(i).joinPair(self)
        
      
      
class animal:
    def __init__(self,ID):
        self.ID=ID
        self.tra=[]
        self.pair=[]

    def joinPair(self,pair):
        self.pair=pair
        pair.addAnimal(self)
        self.tra=pair.tra[:5,:,:]+1
        