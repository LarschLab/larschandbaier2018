# -*- coding: utf-8 -*-
"""
Created on Fri May 06 00:03:18 2016

@author: jlarsch
"""

import numpy as np
import cv2
import geometry
import ImageProcessor

class AnimalShapeParameters(object):
    #obtain descriptive shape parameters of animals
    #typically starting from a video tracked (by idTracker), return to the video to extract further shape information
    
    #Currently sort of working:
    #extract orientation from a tiff stack of single isolated animal
    #-direction based on iterative contour analysis: elipse fit & tail detection
    
    #Planning to do:
    #-input an experiment with multiple animals - need to crop and id each animal based on idTracker position info    
    #-skeleton
    #--store extracted parameters and re-load if possible instead re-generating from video
    
    #features to extract:
    #   position between eyes: center_eyes
    #   center of chest&eyes: center_chest
    #   Line between center_eyes&center_chest: headVector
    #   Angle of headVector: headAngle
    #   Skeleton:listOfPoints
    #       tailStraight-ness

    def __init__(self,path,trajectory='none'):
        self.path=path
        self.trajectory=trajectory
        #self.expInfo=expInfo
        # region User variables

        self.fish_orientation_elipse_all, self.frAll_rot=self.crawlVideo(path,trajectory)
        
    def cropAnimal(self,fullFrame,xy_point,cropSize=200):
        #crop full Frame around animal
        
        region_of_interest = geometry.Region(cropSize, cropSize, geometry.Vector(50, 50))
        region_of_interest.reposition_around_center(xy_point)
        
        # Clip the frame according to the region of interest
        img_frame_original = fullFrame[region_of_interest.top_left.y:region_of_interest.bottom_right.y,
                             region_of_interest.top_left.x:region_of_interest.bottom_right.x]

        # copy the clipped frame to keep it away from processing
        img_input = img_frame_original.copy()
    
        # change the input image into grayscal (gray)
        img_gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
        
        return img_gray
        
    #process sub region around fish
    def crawlVideo(self,path,trajectory='none'):
        
        threshold_elipse = 185
        
        video = cv2.VideoCapture(path)
        nframes=4000
        fish_orientation_elipse_all=np.zeros(nframes)
        frAll_rot=[]
        
        for i in range(nframes):
             # Get the frame
            video.set(cv2.CAP_PROP_POS_FRAMES,i)
            ret, img_frame_original = video.read()
            
            if trajectory !='none':
                currCenter=geometry.Vector(*trajectory[i,:])
                img_crop=self.cropAnimal(img_frame_original,currCenter)
            else:
                img_crop=img_frame_original
            
            #currently, best strategy for orientation seems to be a sequential process:
            #   1: Use a robust 0-180 degree measure to find major orientation of the fish shape
            #       *-a) fit ellipse on (low) rigid body threshold contour, ellipse angle is robust chest-eye orientation
            #       -b) use image moments
            #   2: use second approach to distinguish head-tail
            #       *-a) find tail in polygonFit on low threshold contour, find tail-center direction
            #       -b)  use hull contour instead of polygon to find tail tip [NOT robust when tail bends - deleted]
            #       -c) find darkest pixel, assume it is eye, find center-eye direction [NOT robust, sometimes chest darker than eye]
            #   3: (not yet implemented) refine elipse orientation using eye position
            
            #   Alternative: go diretly to chest-eye detection
            
            
            # find robust 0-180 major axis orientation
            # use a low threshold contour that includes the rigid fish body but not the flexible tail

            contour_fish_dark, centerOfMass = self.get_fish_contour(img_crop.copy(),threshold_elipse)
            fish_orientation_elipse = self.get_fish_ellipse_angle(contour_fish_dark)
            #fish_orientation_moment = self.get_fish_moment_angle(contour_fish_dark)
            
            # orient major axis using head-tail differences
            
            #find tail tip using polygon fit
            tail_tip_polygon=self.get_tail_tip_polygon(contour_fish_dark)
            
            #get angle from tail to center of mass            
            tail_to_center_angle= np.mod(180-geometry.Vector.get_angle(geometry.Vector(*tail_tip_polygon[0]),centerOfMass),360)

            #use elipse orientation, flip to approximately match tail-center angle (within 60 degrees)
            #tail-center angle and elipse angle should be off by only a couple of degrees or swapped by 180 degrees
            if np.abs(tail_to_center_angle-fish_orientation_elipse)>60:
                if np.abs(tail_to_center_angle-fish_orientation_elipse)<300:
                    fish_orientation_elipse=np.mod(fish_orientation_elipse+180,360)
                    

                
            fish_orientation_elipse_all[i]=fish_orientation_elipse
            
            #generate rotation cancelled image
            #translate center of mass to image center
              
            M_trans = np.float32([[1,0,50-centerOfMass[0]],[0,1,50-centerOfMass[1]]])
            img_trans = cv2.warpAffine(255-img_crop,M_trans,img_crop.shape)
            
            M_rot = cv2.getRotationMatrix2D((50,50),fish_orientation_elipse,1)
            img_rot = 255-cv2.warpAffine(255-img_trans,M_rot,img_trans.shape)
            
            frAll_rot.append(img_rot)
        return fish_orientation_elipse_all, frAll_rot
            

    def get_tail_tip_polygon(self,contour,epsilonFactor=0.02):         
    
        #use polygon around contour to find tail tip as sharpest angle      
        epsilon = epsilonFactor*cv2.arcLength(contour[0],True)
        poly = cv2.approxPolyDP(contour[0],epsilon,True)
        
        #get inner angles of all contour points
        poly_angles=self.get_contour_inner_angles(poly)
        tail_tip=poly[np.argmin(poly_angles)]
        
        return tail_tip
    
    
    def get_contour_inner_angles(self,contour_in):
        #for each point, get angle of vectors pointing away from point towards neighbors
        contour=np.squeeze(contour_in)
        contour_roll_forward=np.roll(contour,1,axis=0)
        contour_roll_backward=np.roll(contour,-1,axis=0)
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

    
    
    
    def get_fish_ellipse_angle(self,contour):
        elipse_fit=cv2.fitEllipse(contour[0])
        elipse_orientation=np.mod(90+elipse_fit[2],180)
        return elipse_orientation
        
    def get_fish_moment_angle(self,contour):              
        mom = cv2.moments(contour[0])
        ori_mom=0.5*np.arctan2(2*mom['m21'],(mom['m20']-mom['m02']))
        ori_mom_degrees=np.degrees(ori_mom)     
        return ori_mom_degrees
        
    def get_fish_contour(self,img,threshold):
        img_binary = ImageProcessor.to_binary(img.copy(), threshold)            
        im_tmp2,contours, hierarchy = cv2.findContours(img_binary.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour = ImageProcessor.get_biggest_n_contours(contours, 1)
        centerOfMass = ImageProcessor.get_contours_centroid(contour)
        return contour, centerOfMass
            

            
            

    

                
#            contours_eyes,good_eyes=find_good_eyes(img_crop,threshold_eyes,max_eye_distance,threshold_step,threshold_steps)
#            
#            if good_eyes:
#                
#                # Get the center point of these contours (the center point between the eyes)
#                center_eyes = ImageProcessor.get_contours_centroid(contours_eyes)
#                
#                # ***************************************************************
#                #                       Detecting the chest                     *
#                # ***************************************************************
#                #
#                # The next steps tries to detect the chest of the fish
#                # This step can only be applied if the eyes were detected.
#
#                # get the sharped image of the original frame (don't use the previous processed image because the threshold differs between the eyes and the chest)
#                img_finding_chest = img_crop.copy()
#        
#                # Convert the image into a binary image (black and white) using the threshold of the chest.
#                img_binary_chest = ImageProcessor.to_binary(img_finding_chest, threshold_chest)
#        
#                # Dilate the image (increase the white pixels, so the spaces between them will be reduced).
#                img_dilated_chest = cv2.dilate(img_binary_chest, kernel, dilate_iteration_chest)
#                #img_dilated_chest=img_binary_chest.copy()
#                
#                # Get all the contours in the image
#                im_tmp2,contours_chest, hierarchy = cv2.findContours(img_dilated_chest, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#        
#                # Keep the biggest two contours (they represent the chest and both eyes combined in one blob)
#                contours_chest = ImageProcessor.get_biggest_n_contours(contours_chest, 2)
#        
#                # Get the center point of these contours (the center point between the chest and the eyes)
#                center_chest = ImageProcessor.get_contours_centroid(contours_chest)
#        
#                # ***************************************************************
#                #                Calculate the direction of the fish.           *
#                # ***************************************************************
#                #
#                # The direction is the vector from the chest to the eyes.
#        
#                # Calculate the direction of the fish (from the chest toward the center of the eyes)
#                headVector = center_eyes.__sub__(center_chest)      
#                fish_angle = geometry.Vector.get_angle(center_chest, center_eyes)
#                dirAll[i]=fish_angle
#                
#                
#                if center_eyes:
#                    #cv2.circle(img_frame_original, (center_eyes.x + fish_region.top_left.x, center_eyes.y + fish_region.top_left.y), 5,
#                    #   (0, 0, 255), 2)
#                    cv2.circle(img_frame_original, (center_eyes.x, center_eyes.y), 2,(0, 0, 255), 1)
#                    cv2.circle(img_frame_original, (center_chest.x, center_chest.y), 2,(0, 0, 255), 1)
#                    cv2.imshow('Frame', img_frame_original)
#                    frAll.append(img_frame_original)
                
                

    def find_good_eyes(img,start_threshold,max_eye_distance=5,threshold_step=0.1,threshold_steps=10):
        #want:
        #2 blobs
        #appropriate distance
        good_eyes=False
        threshold_eyes=start_threshold
        contours_eyes,hierarchy,good_eyes=find_eyes(img.copy(),threshold_eyes,max_eye_distance)
        #expect two contours close to each other (eye distance)
          
        if contours_eyes.__len__() >2 or ((contours_eyes.__len__() == 2) and (good_eyes==False)):
            step=0
            while contours_eyes.__len__() >2 or ((contours_eyes.__len__() == 2) and (good_eyes==False)):
                #threshold too low, already including chest.
                #--> increase threshold by threshold_step
                threshold_eyes=int(threshold_eyes - threshold_eyes*threshold_step)
                step += 1
                if step > threshold_steps:
                    break
                contours_eyes,hierarchy,good_eyes=find_eyes(img.copy(),threshold_eyes,max_eye_distance)
        
        if contours_eyes.__len__() <2:
            step=0
            while contours_eyes.__len__() <2:
                #threshold too high, only getting one eye.
                #--> lower threshold by threshold_step
                threshold_eyes=int(threshold_eyes + threshold_eyes*threshold_step)
                step += 1
                if step > threshold_steps:
                    break
                contours_eyes,hierarchy,good_eyes=find_eyes(img.copy(),threshold_eyes,max_eye_distance)
         
        return contours_eyes,good_eyes
            
    
        
    def find_eyes(img,start_threshold,max_eye_distance):
        img_binary_eyes = ImageProcessor.to_binary(img.copy(), start_threshold)
        good_eyes = False
        # Dilate the image (increase the white pixels, so the spaces between them will be reduced).
        #img_dilated_eyes = cv2.erode(img_binary_eyes, kernel, dilate_iteration_eyes)
        #img_dilated_eyes=img_binary_eyes.copy()
        # Get all the contours in the image
        im_tmp,contours_eyes, hierarchy = cv2.findContours(img_binary_eyes.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        #remove very small contours (noise)            
        contours_eyes=remove_contours_smaller_than(contours_eyes,0.1)
        
        #if finding 3 contours, chest probably included.
        #select the 2 that are closest to each other
        
        
        
        if contours_eyes.__len__() ==2:
            #find centroids of each contour
            center_each_eye=get_each_contours_centroid(contours_eyes)
            #calculate distance of the contours
            center_distance=center_each_eye[0].__sub__(center_each_eye[1]).norm()
            #distance should be within bounds [this should be rather robust]
            if center_distance < max_eye_distance:
                good_eyes = True
                
            #else:
                
                #for now: give up on this frame
                
                #eyes might have been combined into one blob and chest was the other blob?
                #--> reduce threshold to separate the eyes
            
                #OR one eye was below threshold?
        
        
        return contours_eyes,hierarchy,good_eyes