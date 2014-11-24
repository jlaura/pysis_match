#!/usr/bin/env python
"""
  Find match points between an image pair using OpenCV's python libraries.

  @image_match
  @Author --Moses Milazzo
  @Date -- 19 November 2014
  @Version 0.01

"""

from itertools import combinations
import json

# This is the Astrogeology Science Center, math is
# a given.
from math import *

import time

# Need to be able to pass arguments to this code.
import os
import argparse


# May be fiddling with OpenCV in this code, but probably not.
import cv2 as cv

import numpy as np

from osgeo import gdal
import osr

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


def getextent(gt,cols,rows):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]

    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
            print x,y
        yarr.reverse()
    return ext

def reproject_coords(coords,src_srs,tgt_srs):
    ''' Reproject a list of x,y coordinates.

        @type geom:     C{tuple/list}
        @param geom:    List of [[x,y],...[x,y]] coordinates
        @type src_srs:  C{osr.SpatialReference}
        @param src_srs: OSR SpatialReference object
        @type tgt_srs:  C{osr.SpatialReference}
        @param tgt_srs: OSR SpatialReference object
        @rtype:         C{tuple/list}
        @return:        List of transformed [[x,y],...[x,y]] coordinates
    '''
    trans_coords=[]
    transform = osr.CoordinateTransformation( src_srs, tgt_srs)
    for x,y in coords:
        x,y,z = transform.TransformPoint(x,y)
        trans_coords.append([x,y])
    return trans_coords

def open_clean_image(fname):
    """
    Open and clean, normalize image

    Parameters
    -----------
    fname       str PATH to the input image
    """
    #print "Opening and cleaning the image: "+fname
    try:
        image = gdal.Open(fname)
    except:
        print "Image, "+fname+" could not be opened. Moving on to next image."
        return None
    npimage = image.ReadAsArray()
    imshape = npimage.shape

    if len(imshape) != 2:
        print "Image "+fname+" has more than one band. Moving on to next image."
        return None
    # Don't allow negative values for now.
    npimage[npimage < -0.] = 0.
    npimage, cdf = histeq(npimage)
    npimage = npimage.astype('uint8')
    #print "Opened and stretched image:"+fname

    '''
    gt = image.GetGeoTransform()
    cols = image.RasterXSize
    rows = image.RasterYSize
    ext = getextent(gt, cols, rows)

    src_srs=osr.SpatialReference()
    src_srs.ImportFromWkt(image.GetProjection())
    print image.GetProjection()
    #tgt_srs=osr.SpatialReference()
    #tgt_srs.ImportFromEPSG(4326)
    tgt_srs = src_srs.CloneGeogCS()
    geo_ext=reproject_coords(ext,src_srs,tgt_srs)
    print geo_ext
    '''
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


@profile
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

    Parameters
    -----------
    keypoints       dict    Desc....
    descriptors     dict
    files           list

    Returns
    --------
    matches
    keypoints
    descriptors
    files
    """

    matches = {}
    cachedimage = None
    for i in combinations(files, 2):
        print i[0], i[1]
        if cachedimage == None or cachedimage != i[0]:
            cachedimage = i[0]
            image1 = open_clean_image(i[0])
            kps1, dsc1, image1 = find_features(image1,'orb')
            keypoints[i[0]] = kps1
            descriptors[i[0]] = dsc1

        if i[1] not in keypoints.keys():
            image2 = open_clean_image(i[1])
            kps2, dsc2, image2 = find_features(image2,'orb')
            keypoints[i[1]] = kps2
            descriptors[i[1]] = dsc2

        #This could be a hash as well
        #b1 = os.path.basename(i[0].split('.')[0])
        #b2 = os.path.basename(i[1].split('.')[0])
        match_key = i[0] +"___"+ i[1]

        bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
        matched = bf.match(descriptors[i[0]], descriptors[i[1]])

        #matched = [m for m in matched if m.distance < 30]
        matched = sorted(matched, key = lambda x:x.distance)
        #matches[match_key] = matched[:20]
        matches[match_key] = matched

    return matches, keypoints, descriptors


def main(args):
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
        matches, kps, desc = find_matches(kps, desc, files)
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

    return

if __name__ == "__main__":


    #Parse the arguments.
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

    main(args)
