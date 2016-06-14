# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 08:52:58 2016

@author: jlarsch
"""

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import networkx as nx
import numpy as np
from scipy import interpolate
import geometry
import ImageProcessor
import matplotlib.pyplot as plt

#img_binary = ImageProcessor.to_binary(img_crop.copy(), 240)

def return_skeleton(img_binary):
#skeletonize image and return an ordered list that represents the shortest path through the skeleton points
#this is going to be trouble when there is significant branching
    #img_bin_crop=img_binary[70:130,70:130]
    img_bin_crop=img_binary
    skel=ImageProcessor.to_skeleton(img_bin_crop)
    skel_smooth,skel_list=get_skeleton_list(skel,30)
    if skel_smooth is not None:
        skel_angles=get_line_angles(skel_smooth)
    else:
        skel_angles=np.zeros(29).tolist()
    return skel_angles, skel_smooth,skel
    
def get_skeleton_list(skel,number_points):
    skel_list=np.where(skel>0)
    points=np.c_[skel_list[0],skel_list[1]]
    ep=findEndpoints(skel)
#    clf = NearestNeighbors(2).fit(points,2)
#    G = clf.kneighbors_graph()
#    T = nx.from_scipy_sparse_matrix(G)
    
    G=NN_graph(points)
    plt.imshow(skel)
    try:
        sp=nx.shortest_path(G,source=(ep[0][0],ep[0][1]),target=(ep[1][0],ep[1][1]))
        spa=np.array(sp)
        x=spa[:,0]
        y=spa[:,1]
        s=5
        k=1
        nest=-1
        tckp,u=interpolate.splprep([x,y],s=s,k=k,nest=nest)
        xnew,ynew=interpolate.splev(np.linspace(0,1,number_points),tckp)
        return np.c_[xnew,ynew], spa
    except:
        print 'endpoints'
        print ep
        #print 'points'
        #print points
        print 'edges'
        print G.edges()
        
        #sp=nx.shortest_path(G,source=(ep[1][0],ep[1][1]),target=(ep[0][0],ep[0][1]))
        return None, None

    
#plt.figure()
#data,=pylab.plot(x,y,'bo-',label='data')
#fit,=pylab.plot(xnew,ynew,'ro-',label='fit')
#pylab.xlabel('x')
#pylab.ylabel('y')

def NN_graph(points):
    G = nx.Graph()  # A graph to hold the nearest neighbours
    xyl=tuple(map(tuple,points))
    tree = KDTree(points, leaf_size=2, metric='euclidean')  # Create a distance tree
    #for p in xyl[1:-1]:  # Skip first and last items in list
    for p in xyl:
        dist, ind = tree.query(np.reshape(p,(1,-1)), k=3)

    
        # ind Indexes represent nodes on a graph
        # Two nearest points are at indexes 1 and 2. 
        # Use these to form edges on graph
        # p is the current point in the list
        G.add_node(p)
        n1 = xyl[ind[0][1]]  # The next nearest point
        n2 = xyl[ind[0][2]]  # The following nearest point
        G.add_edge(p, n1)
        G.add_edge(p, n2)
    return G
    

def findEndpoints(img_skel):
    endPoints=[]
    sb_list=np.where(img_skel>0)
    for i in range(np.shape(sb_list)[1]):
        xmin=np.max([0,sb_list[0][i]-1])
        xmax=np.min([sb_list[0][i]+2,img_skel.shape[0]])
        ymin=np.max([0,sb_list[1][i]-1])
        ymax=np.min([sb_list[1][i]+2,img_skel.shape[1]])
        n=img_skel[xmin:xmax,ymin:ymax]
        nn=np.sum(n>0)
        if nn<3:
            endPoints.append([sb_list[0][i],sb_list[1][i]])
    return endPoints
        
def get_contour_inner_angles(points):
    #for each point, get angle of vectors pointing away from point towards neighbors
    contour=points[1:-1]
    contour_roll_forward=points[2:]
    contour_roll_backward=points[:-2]
    vectors_forward=contour_roll_forward-contour
    vectors_backward=contour_roll_backward-contour
    contour_angles=[]
    #calculate polygon angles
    #angle between lines defined by 3 adjacent polygon points
    for j in range(contour.shape[0]):
        v1=geometry.Vector(*vectors_backward[j])
        v2=geometry.Vector(*vectors_forward[j])                    
        contour_angles.append(v1.get_angleb(v2))
        #contour_angles.append(geometry.Vector.get_angle(v1,v2))
    return contour_angles    

def get_line_angles(points):
    points_base=points[:-1]
    points_fwd=points[1:]
    line_angles=[]
    for j in range(points_base.shape[0]):
        v1=geometry.Vector(*points_base[j])
        v2=geometry.Vector(*points_fwd[j])      
        line_angles.append(geometry.Vector.get_angle(v1,v2))
    return line_angles

