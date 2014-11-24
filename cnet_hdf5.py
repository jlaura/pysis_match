from ast import literal_eval
from collections import OrderedDict, defaultdict
from itertools import islice
import json
import re

import numpy as np
import tables as tb

import pandas as pd

class Image(tb.IsDescription):
    """
    Base class for Image tables in the new scheme.

    Each Instrument will add additional columns to this table.
    """
    Samples = tb.IntCol()
    Lines = tb.IntCol()
    Bands = tb.IntCol()
    Instrument_Name = tb.StringCol(128)
    Instrument_Id = tb.StringCol(32)
    Target_Name = tb.StringCol(32)
    Start_Time = tb.TimeCol()
    Stop_Time = tb.TimeCol()
    Spacecraft_Clock_Count = tb.StringCol(128)
    Target_Center_Distance = tb.FloatCol()
    SC_Target_Position_Vector = (tb.FloatCol(), tb.FloatCol(), tb.FloatCol())
    Slant_Distance = tb.FloatCol()
    Center_Latitude = tb.FloatCol()
    Center_Longitude = tb.FloatCol()
    Horizontal_Pixel_Scale = tb.FloatCol()
    Smear_Magnitude = tb.FloatCol()
    Smear_Azimuth = tb.FloatCol()
    North_Azimuth = tb.FloatCol()
    Reticle_Point_Latitude = tb.FloatCol()
    Reticle_Point_Longitude = tb.FloatCol()
    Sub_Spacecraft_Latitude = tb.FloatCol()
    Sub_Spacecraft_Longitude = tb.FloatCol()
    Spacecraft_Altitude = tb.FloatCol()
    Sub_Spacecraft_Azimuth = tb.FloatCol()
    Spacecraft_Solar_Distance = tb.FloatCol()
    SC_Sun_Position_Vector = tb.FloatCol()
    SC_Sun_Velocity_Vector = tb.FloatCol()
    Solar_Distance = tb.FloatCol()
    Sub_Solar_Azimuth = tb.FloatCol()
    Sub_Solar_Latitude = tb.FloatCol()
    Sub_Solar_Longitude = tb.FloatCol()
    Incidence_Angle = tb.FloatCol()
    Phase_Angle = tb.FloatCol()
    Emission_Angle = tb.FloatCol()
    Local_Hour_Angle = tb.FloatCol()
    PixelType = tb.StringCol(12)
    Base = tb.FloatCol()
    Mult = tb.FloatCol()
    ExposureDuration = tb.FloatCol()
    ExposureType = tb.StringCol(12)
    Naif_Frame_Code             = tb.FloatCol()
    Leap_Second                = tb.StringCol(128)
    Target_Attitude_Shape       = tb.StringCol(128)
    Target_Position            = (tb.StringCol(32), tb.StringCol(128))
    Instrument_Pointing        = (tb.StringCol(32), tb.StringCol(128), tb.StringCol(128))
    Instrument                = tb.StringCol(32)
    Spacecraft_Clock           = tb.StringCol(12)
    Instrument_Position        = (tb.StringCol(12), tb.StringCol(128))
    Instrument_Addendum        = tb.StringCol(128)
    ShapeModel                = tb.StringCol(128)
    Instrument_Position_Quality = tb.StringCol(32)
    Instrument_Pointing_Quality = tb.StringCol(32)
    Camera_Version             = tb.FloatCol()
    Instrument_Pointing_Time_Dependent_Frames = (tb.FloatCol(), tb.IntCol(), tb.IntCol())
    Instrument_Pointing_Constant_Frames      = (tb.IntCol(), tb.IntCol(), tb.IntCol())
    Instrument_Pointing_Constant_Rotation    = (tb.FloatCol(), tb.FloatCol(),
                         tb.FloatCol(), tb.FloatCol(), tb.FloatCol(),
                         tb.FloatCol(), tb.FloatCol(),
                         tb.FloatCol(), tb.FloatCol())
    Instrument_Pointing_Ck_Table_Start_Time    = tb.FloatCol()
    Instrument_Pointing_Ck_Table_End_Time      = tb.FloatCol()
    Instrument_Pointing_Ck_Table_Original_Size = tb.IntCol()
    Instrument_Pointing_Ck_Description         = tb.StringCol(64)
    Instrument_Pointing_Ck_Kernels             = (tb.StringCol(128), tb.StringCol(128))
    Instrument_Position_Spk_Cache_Type = tb.StringCol(32)
    Instrument_Position_Spk_Table_Start_Time = tb.FloatCol()
    Instrument_Position_Spk_Table_End_Time = tb.FloatCol()
    Instrument_Position_Spk_Table_Original_Size = tb.FloatCol()
    Instrument_Position_Spk_Description = tb.StringCol(32)
    Instrument_Position_Spk_Kernels = tb.StringCol(128)
    Body_Rotation_Time_Dependent_Frames = (tb.FloatCol(), tb.FloatCol())
    Body_Rotation_Ck_Table_Start_Time    = tb.FloatCol()
    Body_Rotation_Ck_Table_End_Time      = tb.FloatCol()
    Body_Rotation_Ck_Table_Original_Size = tb.IntCol()
    Body_Rotation_Description         = tb.StringCol(128)
    Body_Rotation_Kernels             = (tb.StringCol(128), tb.StringCol(64))
    Body_Rotation_Solar_Longitude      = tb.FloatCol()
    Sun_Position_Cache_Type            = tb.StringCol(32)
    Sun_Position_Spk_Table_Start_Time    = tb.FloatCol()
    Sun_Position_Spk_Table_End_Time      = tb.FloatCol()
    Sun_Position_Spk_Table_Original_Size = tb.FloatCol()
    Sun_Position_Description          = tb.StringCol(128)
    Sun_Position_Kernels              = tb.StringCol(128)
    Naif_Keywords_Body499_Radii = (tb.FloatCol(), tb.FloatCol(), tb.FloatCol())
    Naif_Keywords_Body_Frame_Code = tb.IntCol()
    Polygon_Footprint = tb.StringCol(256)


"""
    MissionPhaseName = tb.StringCol()
    DetectorTemperature = tb.FloatCol()
    FilterTemperature = tb.FloatCol()
    FocalPlaneTemperature = tb.FloatCol()
    OpticsTemperature = tb.FloatCol()
    AttitudeQuality = tb.StringCol()
"""





class ControlPoint(tb.IsDescription):
    pointid = tb.StringCol(64)  #Why string?  #Int or float?  In the latter case,

    pointtype = tb.StringCol(12)
    choosername = tb.StringCol(36)
    datetime = tb.StringCol(36)  #Should be datetime
    referenceindex = tb.Int32Col()
    apriorisurface_pointsource = tb.StringCol(32)
    apriorisurface_pointfile = tb.StringCol(128)
    aprioriradius_pointsource = tb.StringCol(32)
    aprioriradius_pointfile = tb.StringCol(128)

    latitudeconstrained = tb.BoolCol()
    longitudeconstrained = tb.BoolCol()
    radiusconstrained = tb.BoolCol()

    #apriorilatitude = tb.FloatCol()
    apriorix = tb.FloatCol()
    #apriorilongitude = tb.FloatCol()
    aprioriy = tb.FloatCol()
    #aprioriradius = tb.FloatCol()
    aprioriz = tb.FloatCol()

    #adjustedlatitude = tb.FloatCol()
    adjustedx = tb.FloatCol()
    #adjustedlongitude = tb.FloatCol()
    adjustedy = tb.FloatCol()
    #adjustedradius = tb.FloatCol()
    adjustedz = tb.FloatCol()

    #aprioricovar = tb.Float32Col(shape=6)
    #adjustedcovar = tb.Float32Col(shape=6)

    #Measure
    serialnumber = tb.StringCol(256)
    measuretype = tb.StringCol(32)
    choosername = tb.StringCol(32)
    datetime = tb.StringCol(64)  #should be datetime
    editlock = tb.BoolCol()
    ignore = tb.BoolCol()
    jigsawrejected = tb.BoolCol()
    diameter = tb.FloatCol()
    sample = tb.FloatCol()
    line = tb.FloatCol()

    samplesigma = tb.FloatCol()
    linesigma = tb.FloatCol()

    apriorisample = tb.FloatCol()
    aprioriline = tb.FloatCol()
    sampleresidual = tb.FloatCol()
    lineresidual = tb.FloatCol()
    #goodnessoffit = tb.FloatCol()
    #reference = tb.BoolCol()

def image_table(outfile, imfile):
    """
    Add the image metadata to an HDF5 table.

    I should be creating a relatively small table and attaching
    subgroups and attributes to it, but I'm still learning hdf5... :/
    """
    h5file = tb.open_file(outfile, mode='w', title='cnet')
    img = h5file.create_group('/', 'Images', 'Image List')
    imgtable = h5file.create_table(img, 'images', Image, 'Lookup table of images')
    imgrow = imgtable.row




def control_network_hdf5(outfile, infiles, matches, kps):
    """
    Build an HDF5 control network out of the match points found by using OpenCV.

    matches is a dictionary with the image1_image2 as the key and the matchpoints list
        as the value.
        The matchpoints are structured as follows:
        int queryIdx // query-image descriptor index
        int trainIdx // train-image descriptor index
        int imgIdx // train image index
        float distance // Quality of the match (smaller is better).

    kps is the keypoints dictionary with image as the key and keypoint object as the value.
        The keypoint objects are structured as follows:
            float angle // orientation of the keypoint. (-1 if not applicable)
            int class_id // Object ID. Can be used to cluster keypoints by an object they belong to.
            int octave // Octave or pyramid layer from which the keypoint has been extracted.
            Point pt // Coordinates of the keypoint.
            float response // The response, by which the strongest keypoints have been selected.
            float size // Diameter of the useful keypoint adjacent area.
    """


    h5file = tb.open_file(outfile, mode='w', title='cnet')
    free_group = h5file.create_group('/', 'Free', 'Points of type free')
    freetable = h5file.create_table(free_group, 'controlmeasure', ControlPoint  , 'Free Control Points')
    freerow = freetable.row



    """
    The description of the incoming parameters to this function
    """

    """
    Build the hdf5 match point table now.

    Use the keypoint.class_id to as the pointid.
    """
    for key in matches:
        im1str, im2str = key.split("___")
        matchvals = matches[key]
        if (len(matchvals) > 0):
            for m in matchvals:
                pid = str(kps[im1str]).split(" ")[1].split(">")[0]
                pid = int(pid.encode('hex'), 16)
                freerow['pointid'] = str(pid)
                freerow['pointtype'] = 'Free'
                freerow['choosername'] = 'PyImageMatch'
                freerow['referenceindex'] = m.queryIdx
                freerow['apriorisurface_pointsource'] = "NULL"
                freerow['apriorisurface_pointfile'] = "NULL"
                freerow['aprioriradius_pointsource'] = "NULL"
                freerow['aprioriradius_pointfile'] = "NULL"
                freerow['latitudeconstrained'] = False
                freerow['longitudeconstrained'] = False
                freerow['radiusconstrained'] = False
                freerow['apriorix'] = 0.0
                freerow['aprioriy'] = 0.0
                freerow['aprioriz'] = 0.0
                freerow['adjustedx'] = 0.0
                freerow['adjustedy'] = 0.0
                freerow['adjustedz'] = 0.0
                freerow['serialnumber'] = im1str+"/"+str(m.queryIdx)+'/'+str(m.trainIdx) +'/'+str(m.imgIdx)+'/'+str(m.distance)
                freerow['measuretype'] = 'Free'
                freerow['editlock'] = False
                freerow['ignore'] = False
                freerow['jigsawrejected'] = False
                freerow['diameter'] = kps[im1str][m.queryIdx].size
                freerow['sample'] = kps[im1str][m.queryIdx].pt[0]
                freerow['line'] = kps[im1str][m.queryIdx].pt[1]
                freerow['samplesigma'] = kps[im1str][m.queryIdx].size
                freerow['linesigma'] = kps[im1str][m.queryIdx].size
                freerow['apriorisample'] = kps[im1str][m.queryIdx].pt[0]
                freerow['aprioriline'] = kps[im1str][m.queryIdx].pt[1]
                freerow.append()
        freetable.flush()


#    def gather_image_metadata(fname):
        """
        Gather metadata from the image. Add it to the image_group of the control network.

        The image could have metadata at the footer of the image as well as at the header.
        The header contains a StartByte that tells when the image data start.
        """

#        with open('fname', 'r') as f:
