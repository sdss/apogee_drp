import sys
import glob
import os
import subprocess
import math
import time
import numpy as np
from pathlib import Path
from astropy.io import fits, ascii
from astropy.table import Table
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.lib.recfunctions import append_fields, merge_arrays
from astropy import units as astropyUnits
from scipy.signal import medfilt2d as ScipyMedfilt2D
from apogee_drp.utils import plan,apload,yanny,plugmap,platedata,bitmask
from apogee_drp.apred import wave
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

'''PlotFlats: overplot some dome flats '''
def PlotFlats(apred='daily', telescope='apo25m',sep=50):
    load = apload.ApLoad(apred=apred, telescope=telescope)

    visitDir = os.environ.get('APOGEE_REDUX')+'/'+apred+'/visit/'+telescope+'/'
    planfiles = glob.glob(visitDir+'*/*/*/apPlan*yaml')
    planfiles.sort()
    planfiles = np.array(planfiles)
    nplans = len(planfiles)
    print(str(nplans) + ' planfiles found')

    psfid = np.empty(nplans).astype(str)
    for i in range(nplans):
        planstr = plan.load(planfiles[i], np=True)
        psfid[i] = planstr['psfid']

    print(psfid)

    colors = np.array(['r','b','g','k','purple','orange','cyan'])
    ncolors = len(colors)

    twod = load.ap2D(int(psfid[0]))
    data = twod['b'][1].data
    tot = np.median(data[:,900:1100], axis=1)
    plt.clf()
    plt.plot(tot+sep*0, color=colors[0%ncolors])
    plt.xlim(100,200)

    for i in range(nplans):
        twod = load.ap2D(int(psfid[i]))
        data = twod['b'][1].data
        tot = np.median(data[:,900:1100], axis=1)
        plt.plot(tot+sep*i, color=colors[i%ncolors])

    return psfid
