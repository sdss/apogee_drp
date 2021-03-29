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
from apogee_drp.utils import plan,apload,yanny,plugmap,platedata,bitmask,peakfit
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
from scipy.signal import medfilt, convolve, boxcar, argrelextrema, find_peaks

###################################################################################################
def FindAllPeaks(apred='daily', telescope='apo25m',sep=50):

    chips = np.array(['a','b','c'])
    nchips = len(chips)
    nfiber = 300
    npix = 2048

    instrument = 'apogee-n'
    refpix = ascii.read('/uufs/chpc.utah.edu/common/home/u0955897/refpixN.dat')
    
    if telescope == 'lco25m':
        instrument = 'apogee-s'
        refpix = ascii.read('/uufs/chpc.utah.edu/common/home/u0955897/refpixS.dat')

    apodir = os.environ.get('APOGEE_REDUX') + '/'

    outfile = apodir + apred + '/monitor/' + instrument + 'DomeFlatTrace.fits'

    load = apload.ApLoad(apred=apred, telescope=telescope)

    visitDir = apodir + '/' + apred + '/visit/' + telescope + '/'
    planfiles = glob.glob(visitDir + '*/*/*/apPlan*yaml')
    planfiles.sort()
    planfiles = np.array(planfiles)
    nplans = len(planfiles)
    print(str(nplans) + ' planfiles found')

    # Lookup table structure.
    dt = np.dtype([('PSFID',    np.str, 9),
                   ('PLATEID',  np.int32),
                   ('CARTID',   np.int16),
                   ('NAME',     np.str, 14),
                   ('DATE-OBS', np.str, 23),
                   ('MJD',      np.float64),
                   ('CENT',     np.float64, (nchips, nfiber)),
                   ('HEIGHT',   np.float64, (nchips, nfiber)),
                   ('FLUX',     np.float64, (nchips, nfiber)),
                   ('SUCCESS',  np.int16,   (nchips, nfiber))])

    outstr = np.zeros(nplans,dtype=dt)

    # Loop over the plan files
    #for i in range(nplans):
    for i in range(5):
        planstr = plan.load(planfiles[i], np=True)
        psfid = planstr['psfid']
        twod = load.ap2D(int(psfid))
        header = twod['a'][0].header

        outstr['PSFID'][i] = psfid
        outstr['PLATEID'][i] = header['PLATEID']
        outstr['CARTID'][i] = header['CARTID']
        outstr['NAME'][i] = header['NAME']
        outstr['DATE-OBS'][i] = header['DATE-OBS']
        t = Time(header['DATE-OBS'], format='fits')
        outstr['MJD'][i] = t.mjd

        # Loop over the chips
        for ichip in range(nchips):
            flux = twod[chips[ichip]][1].data
            error = twod[chips[ichip]][2].data

            totflux = np.nanmedian(flux[:, (npix//2) - 200:(npix//2) + 200], axis=1)
            toterror = np.sqrt(np.nanmedian(error[:, (npix//2) - 200:(npix//2) + 200]**2, axis=1))
            pix0 = np.array(refpix[chips[ichip]])
            gpeaks = peakfit.peakfit(totflux, sigma=toterror, pix0=pix0)

            outstr['SUCCESS'][ichip, :] = 1
            failed, = np.where(gpeaks['success'] == False)
            nfailed = len(failed)
            if nfailed > 0: outstr['SUCCESS'][i, ichip, failed] = 0

            print('   ' + str(len(gpeaks)) + ' elements in gpeaks; ' + str(300 - nfailed) + ' successful Gaussian fits')

            outstr['CENT'][i, ichip, :] =    gpeaks['pars'][:, 1]
            outstr['HEIGHT'][i, ichip, :] =  gpeaks['pars'][:, 0]
            outstr['FLUX'][i, ichip, :] =    gpeaks['sumflux']

    Table(outstr).write(outfile, overwrite=True)

    return

