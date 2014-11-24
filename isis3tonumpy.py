#!/usr/bin/env python
"""
  Use gdal to import ISIS3 images into numpy, but the file size can become too large to
    effeciently deal with in numpy. How to fix?
    Will HDF5 rescue us, or is that just not the right path?

  @isis3tonumpy
  @Author --Moses Milazzo
  @Date -- 19 November 2014
  @Version 0.01

"""

# This is the Astrogeology Science Center, math is
# a given.
from math import *

# Need to be able to pass arguments to this code.
import sys
import glob
import os
import argparse

# May be fiddling with OpenCV in this code, but probably not.
#import cv2 as cv

#Obviously need HDF5 for HDF5 reading/writing/constructing.
import h5py as h5

# Numpy and Tables are awesome.
import tables as tb
import numpy as np

# May be displaying images.
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Need Pandas' fast array abilities.
import pandas as pd

# Need GDAL to open ISIS3 cubes.
from osgeo import gdal


"""
# Let's build a function to read ISIS3 images. This 
# could be as simple as running gdal, but I think
# we need to make sure we're verifying the data
# and metadata integrity.
"""

def read_isis3(filename):
    """
    Reads an ISIS3 cube from disk and return a hdf5 data structure.

    Read an ISIS3 cube from disk, verify its contents, set up the proper 
    HDF5 data structure, including the appropriate attributes, and 
    return the HDF5 data structure. We do not want to generalize this
    to read any GDAL-supported file type because that's what this program
    will eventually be generalized to.
    """

    gdimg1 = gdal.Open(filename)
    nband = gdimg1.RasterCount
    nsamp = gdimg1.RasterXSize
    nline = gdimg1.RasterYSize 
    img1 = np.array(gdimg1.ReadAsArray())


def astro_hdf5()
    """
    Builds the HDF5 data structure specific to Astrogeology Science Center image cubes.

    The ASC HDF5 Image format is specified below and is based on two formats:
    The KNMI HDF5 image format: http://www.knmi.nl/publications/fulltexts/hdftag35.pdf
    and
    The PDS4 design document: 

    The tags in parentheses () specify whether an attribute is mandatory (M)
    or repeated (R). Subgroups are designated as [subgroup].
    Groups:
        overview (M)
        geographic (M)
            [map projection]
        image (R)
            [calibration]
            [statistics]
            [satellite]
            
      
    """	

isis_hdf = h5.File('EWstack.h5', 'w')
print type(isis_hdf)

