# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 20:52:49 2015

@author: johannes
"""
import numpy as np

def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho
    
def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y
    
def runningMean(x, N):
    return np.convolve(x, np.ones((N,))/N,mode='valid')[(N-1):]

def distance(x,y):
    return np.sqrt(x**2 + y**2)
    
def equalizePath(x,y,precision=2):
    

    M = len(x)*100
    t = np.linspace(0, len(x), M)
    xi = np.interp(t, np.arange(len(x)), x)
    yi = np.interp(t, np.arange(len(y)), y)
    
    
    i, idx = 0, [0]
    while i < len(xi)-1:
        total_dist = 0
        for j in range(i+1, len(xi)):
            total_dist = np.sqrt((xi[j]-xi[i])**2 + (yi[j]-yi[i])**2)
            if total_dist > precision:
                idx.append(j)
                break
        i = j+1
    
    xn = xi[idx]
    yn = yi[idx]
    
    # Interpolate values for x and y.
    t = np.arange(len(xn))
    t2 = np.linspace(0, len(xn), len(x))
    # One-dimensional linear interpolation.
    xnn = np.interp(t2, t, xn)
    ynn = np.interp(t2, t, yn)
    return xnn,ynn