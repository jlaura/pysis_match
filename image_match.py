#!/usr/bin/env python
"""
  Find match points between an image pair using OpenCV's python libraries.

  @image_match
  @Author --Moses Milazzo
  @Date -- 19 November 2014
  @Version 0.01

"""

# This is the Astrogeology Science Center, math is
# a given.
from math import *

import time

import cubehelix_array as ch
# Need to be able to pass arguments to this code.
import sys
import glob
import os
import argparse

# May be fiddling with OpenCV in this code, but probably not.
import cv2 as cv

#Obviously need HDF5 for HDF5 reading/writing/constructing.
import h5py as h5

# regular expressions
import re

# Numpy and Tables are awesome.
import tables as tb
import numpy as np
from PIL import Image

# May be displaying images.
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Need Pandas' fast array abilities.
import pandas as pd

# Need GDAL to open ISIS3 cubes.
from osgeo import gdal
from gdalconst import *

import cnet_hdf5 as cnet



"""
    Iterate through the list of images:
    Detect a set of features (keypoints) on each image, 
    Then iterate through each image and match to the other
    images in the list. Pop that image off of the list and
    make the next image our match template. Continue through
    the list of images. This can get expensive for disk I/O
    if we're loading the images multiple times, or it can
    be expensive for memory if we're trying to keep all of the
    images in memory.
"""

def histeq(im,nbr_bins=256):
    """
    Normalize image histogram.
    """
    immin = im.min()
    immax = im.max()
    imhist, bins = np.histogram(im.flatten(),nbr_bins,range=(immin,immax), normed=True)
    cdf = imhist.cumsum() # cumulative distribution function.
    cdf = 255*cdf/cdf[-1] # normalize the cdf.
    #Use linear interpolation of CDF to find new pixel values.
    im2 = np.interp(im.flatten(),bins[:-1],cdf)
    return im2.reshape(im.shape), cdf


def read_list(flist):
    """
        Read in the list of images, make sure all images exist.
    """
    with open(flist, 'r') as fl:
        read_data = fl.read()
    read_data = read_data.strip()
    is_csv = ',' in read_data
    if is_csv:
       file_list = read_data.split(",")
    else: 
       file_list = read_data.split("\n")
    
    # Check to see if the files exist on disk. Probably too expensive for
    # large lists.
    for fi in file_list:
        exists = os.path.isfile(fi)
        if (exists == False):
            print "File "+fi+" does not appear to exist. Exiting."
            exit()
           
    return(file_list)

def open_clean_image(fname):
    """
    Open and clean, normalize image
    """
    #print "Opening and cleaning the image: "+fname 
    image = gdal.Open(fname)
    if image is None:
        print "Image, "+fname+" could not be opened. Moving on to next image."
        return None
    npimage = np.array(image.ReadAsArray())
    imshape = npimage.shape
    if len(imshape) != 2:
        print "Image "+fname+" has more than one band. Moving on to next image."
        return None
    # Don't allow negative values for now.
    npimage[npimage < -0.] = 0.
    npimage, cdf = histeq(npimage)
    npimage = npimage.astype('uint8')
    #print "Opened and stretched image:"+fname
    return(npimage)
   

def find_features(files, feat):
    """
    Iterate through the images and find keypoints. 
    We're using ORB for now because it's a replacement for SIFT and SURF and seems to work well.
 
    The feature extraction algorithms are all described here:
    http://docs.opencv.org/trunk/modules/features2d/doc/feature_detection_and_description.html#fast"

    If files is a file handler, then we've already opened the image, so 
    skip the open part and move on to the finding keypoints.
    """
    my_key_points = {}
    my_descriptors = {}
    orb = cv.ORB()
    fname = ""
    if (type(files) is list):
        temp_files = list(files)
        for i in files:
            npimage = open_clean_image(i)
            if npimage is None:
                temp_files.remove(i)
                continue
            kp, des = orb.detectAndCompute(npimage,None)
            my_key_points[i] = kp
            my_descriptors[i] = des
        files = temp_files
        return(my_key_points, my_descriptors, files)
    else:
        npimage = files
        fname = str(npimage)
        kp, des = orb.detectAndCompute(npimage,None)
        return(kp, des, files)
            

def find_matches(keypoints, descriptors, files):
    """
    Find matches between images.

    Iterate through all images, finding matches between the pairs of
    images. Example: Assume five images in the list. First, use 
    find_features on each image. Then match:
    image 1 -> image 2
    image 1 -> image 3
    image 1 -> image 4
    image 1 -> image 5
    image 2 -> image 3
    image 2 -> image 4
    image 2 -> image 5
    image 3 -> image 4
    image 3 -> image 5
    image 4 -> image 5
    This gives the full set of matches. Accomplish this by removing image 1 from the list
    after the first iteration, then image 2, and so on.
    """
    
    temp_files = list(files)
    files2 = list(files)
    matches = {}
    for i in files:
        image1 = open_clean_image(i)
        if image1 is None:
            temp_files.remove(i)
            continue
        kps1, dsc1, image1 = find_features(image1,'orb')
        keypoints[i] = kps1
        descriptors[i] = dsc1
        files2.remove(i)
        for j in files2:
            image2 = open_clean_image(j)
            if image2 is None:
                files2.remove(j)
                continue
            kps2, dsc2, image2 = find_features(image2,'orb')
            keypoints[j] = kps2
            descriptors[j] = dsc2
            match_key=i +"___"+ j
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
            matched = bf.match(dsc1,dsc2)
            #matched = [m for m in matched if m.distance < 30]
            matched = sorted(matched, key = lambda x:x.distance)
            #matches[match_key] = matched[:20]
            matches[match_key] = matched

    files = temp_files
    return matches, keypoints, descriptors, files
 


if __name__ == "__main__":


    """
    Parse the arguments.
    """
    t1 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("-l", "--flist", dest='flist', required=True, type=str,
        help="Image list either in one-per-line or csv format.")
    ap.add_argument("-f", "--feature_method", dest='feat', required=False, type=str,
        help="Feature Detection Method.")
    ap.add_argument("-m", "--match_method", dest='match', required=False, type=str,
        help="Image Matching Method.")
    ap.add_argument("-o", "--output", required=False, dest='outnet', type=str,
        help="Output Control Network file (hdf5 format).")

    ap.set_defaults(feat="ORB", match="HAM")
    args = vars(ap.parse_args())

    feat = args['feat'].upper()
    #feat = feat.upper()
    flist = args['flist']

    files = read_list(flist)
    
    # Feature Detectors allowed
    #allowed_feature_detectors = ["MSER", "ORB", "STAR", "FAST", "HARRIS", "Dense", "SimpleBlob", "GFTT"]
    allowed_feature_detectors = ["ORB"] 
    # Verify that the feature detection method is one that we allow.
    if (feat in allowed_feature_detectors):
        # Call the keypoint detection function, which uses the 
        # correct feature detection algorithm.
        #kps, desc,files = find_features(files, feat)
        kps = {}
        desc = {}
        matches,kps,desc,files = find_matches(kps, desc, files)
        outfile = "test_network.h5"
        cnet.control_network_hdf5(outfile, files, matches, kps)
        """
        for key in matches:
            im1str, im2str = key.split("_")
            print im1str, im2str
            matchvals = matches[key]
            if len(matchvals) > 0:
                im1 = open_clean_image(im1str)
                im2 = open_clean_image(im2str)
                h1, w1 = im1.shape[:2]
                h2, w2 = im2.shape[:2]
                im3 = np.hstack((im1,im2))
                im4 = np.dstack((im3,im3))
                im4 = np.dstack((im4,im3))
                for m in matchvals:
                    cdist = int(255*(m.distance/30))
                    color = ch.cubehelix(cdist)
                    cv.line(im4, (int(kps[im1str][m.queryIdx].pt[0]), int(kps[im1str][m.queryIdx].pt[1])), 
                                 (int(kps[im2str][m.trainIdx].pt[0]+w1), int(kps[im2str][m.trainIdx].pt[1])),                              color)
                cv.imshow("Result", im4)
                cv.waitKey()
        """
    else:
        print "Your feature detection algorithm "+feat+" is not incorporated into this software."
        print "Please check your options or re-read the help."
        exit()
    t2 = time.time()
    tottime = t2-t1
    print "This entire program took: ",tottime," to complete."
