import sys
import glob
import os
import subprocess
import math
import time
import pickle
import doppler
import numpy as np
from pathlib import Path
from astropy.io import fits, ascii
from astropy.table import Table
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.lib.recfunctions import append_fields, merge_arrays
from astroplan import moon_illumination
from astropy.coordinates import SkyCoord, get_moon
from astropy import units as astropyUnits
from apogee_drp.utils import plan,apload,yanny,plugmap,platedata,bitmask,peakfit
from apogee_drp.apred import wave
from apogee_drp.database import apogeedb
from dlnpyutils import utils as dln
from sdss_access.path import path
import pdb
import matplotlib.pyplot as plt
import matplotlib
from astropy.convolution import convolve, Box1DKernel
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator
import matplotlib.ticker as ticker
import matplotlib.colors as mplcolors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from scipy.signal import medfilt2d as ScipyMedfilt2D
from scipy.signal import medfilt, convolve, boxcar, argrelextrema, find_peaks
from scipy.optimize import curve_fit
import datetime

###################################################################################################
''' makeSkyHTML: make an html page showing ALL sky plots '''
def makeSkyHTML(mjdstart='59146', observatory='apo', apred='daily'):

    # Establish telescope
    telescope = observatory + '25m'
    load = apload.ApLoad(apred=apred, telescope=telescope)

    # HTML header background color
    thcolor = '#DCDCDC'

    # Find all the plate-mjds
    apodir = os.environ.get('APOGEE_REDUX') + '/'
    mjdDirs = np.array(glob.glob(apodir + apred + '/visit/' + telescope + '/*/*/*'))
    ndirs = len(mjdDirs)
    allmjd = np.empty(ndirs).astype(str)
    allplate = np.empty(ndirs).astype(str)
    allfield = np.empty(ndirs).astype(str)
    for i in range(ndirs): 
        tmp = mjdDirs[i].split(telescope + '/')
        allfield[i] = tmp[1].split('/')[0]
        allplate[i] = tmp[1].split('/')[1]
        allmjd[i] = tmp[1].split('/')[2]
    gd, = np.where(allmjd != 'plots')
    allfield = allfield[gd]
    allplate = allplate[gd]
    allmjd = allmjd[gd]
    nplates = len(allplate)    

    # Loop over the plates, adding sky plots to skyhtml page
    html = open(apodir + apred + '/qa/sky-' + observatory + '.html', 'w')
    html.write('<HTML>\n')
    html.write('<HEAD><script src="sorttable.js"></script><title>APOGEE-N Sky Fiber plots</title></head>\n')
    html.write('<BODY>\n')
    html.write('<H1>APOGEE-N Sky Fiber plots</H1><HR>\n')
    html.write('<TABLE BORDER=2 CLASS="sortable">\n')
    html.write('<TR bgcolor="' + thcolor + '"><TH>FIELD <TH>PLATE-MJD-FIBER <TH>RA <TH>DEC <TH>apVisit Plot\n')
    for iplate in range(nplates):
        plate = allplate[iplate]
        mjd = allmjd[iplate]
        field = allfield[iplate]

        # Load in the apPlate file and restrict to sky fibers
        apPlate = load.apPlate(int(plate), mjd)
        import pdb; pdb.set_trace()
        apPlateFile = load.filename(plate=int(plate), mjd=mjd, chips=True)
        if os.path.exists(apPlateFile):
            print("Doing " + field + ", plate " + plate + ", mjd " + mjd) 
            apPlate = load.apPlate(int(plate), mjd)
            data = apPlate['a'][11].data[::-1]
            sky, = np.where((data['FIBERID'] > 0) & (data['OBJTYPE'] == 'SKY'))
            data = data[sky]
            nsky = len(sky)
            # Loop over the fibers
            for j in range(nsky):
                cfiber = str(data['FIBERID'][j]).zfill(3)
                cra = str("%.5f" % round(data['RA'][j], 5))
                cdec = str("%.5f" % round(data['DEC'][j], 5))
                pmf = plate + '-' + mjd + '-' + cfiber
                visplot = '../visit/' + telescope + '/' + field + '/' + plate + '/' + mjd + '/plots/apPlate-' + pmf + '.png'
                html.write('<TR><TD>' + field + '<TD>' + pmf + '<TD ALIGN="RIGHT">' + cra + '<TD ALIGN="RIGHT">' + cdec)
                html.write('<TD><A HREF=' + visplot + ' target="_blank"><IMG SRC=' + visplot + ' WIDTH=500></A>\n')
        html.close()
    
