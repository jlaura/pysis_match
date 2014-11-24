#!/usr/bin/env python

from math import *

import time

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

#import cnet_hdf5 as cnet


filename = 'I05075015RDR.cub'
dataset = gdal.Open('I05075015RDR.cub', GA_ReadOnly)

print 'Driver:', dataset.GetDriver().ShortName, '/', \
    dataset.GetDriver().LongName
print 'Size is ',dataset.RasterXSize,'x',dataset.RasterYSize, \
      'x',dataset.RasterCount
print 'Projection is ',dataset.GetProjection()
    
fileend = re.compile(r"\bEnd\b", re.IGNORECASE)
objstart = re.compile(r"\s*\bObject\b")
objend = re.compile(r"\s*\bEndObject\b")
groupstart = re.compile(r"\s*Group\b")
groupend = re.compile(r"\s*End_Group\b")
comment = re.compile(r"\s*#")
altcomment = re.compile(r"\s*\/*")
whitespace = re.compile(r"^\s$")

with open(filename, 'r') as f:
    for i, line in enumerate(f):
        
