# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:51:08 2016

@author: jlarsch
"""

import random
from models.AnimalTimeSeriesCollection import AnimalTimeSeriesCollection


class Animal(object):
    def __init__(self, ID=0):
        self.ID = ID
        self.pair = []
        self.ts = []
        self.neighbor = None

    def joinPair(self, pair):
        self.pair = pair
        pair.addAnimal(self)

    def add_TimeSeriesCollection(self, ts):
        self.ts = ts
        return ts

    def add_BoutSeriesCollection(self, bs):
        self.bs = bs
        return bs

    def wakeUp(self):
        AnimalTimeSeriesCollection().linkAnimal(self)
        self.neighbor = self.pair.animals[1-self.ID]


