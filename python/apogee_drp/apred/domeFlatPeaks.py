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
from scipy.signal import find_peaks

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
    mjd = np.empty(nplans).astype(str)
    for i in range(nplans):
        planstr = plan.load(planfiles[i], np=True)
        psfid[i] = planstr['psfid']
        mjd[i] = planstr['mjd']

    print(psfid)

    #colors = np.array(['r','b','violet','g','k','darkorange','cyan'])
    #ncolors = len(colors)

    cmap=plt.get_cmap('hot')
    colors = [cmap(k) for k in np.linspace(0,0.65,nplans)]
    #colors=colors[::-1]

    twod = load.ap2D(int(psfid[0]))
    data = twod['b'][1].data
    tot = np.median(data[:,900:1100], axis=1)

    peaks,_ = find_peaks(tot, height=100, distance=4)

    stot = convolve(tot, Box1DKernel(5))
    speaks,_ = find_peaks(stot, height=50, distance=4)

    plt.clf()
    plt.plot(tot, color=colors[0])
    #plt.xlim(750,900)
    #plt.scatter(peaks,tot[peaks], marker='x', color='r')

    #plt.plot(stot, color='k')
    #plt.scatter(speaks, stot[speaks], marker='x', color='g')


    for i in range(nplans):
        twod = load.ap2D(int(psfid[i]))
        data = twod['b'][1].data
        tot = np.median(data[:,900:1100], axis=1)
        plt.plot(tot+sep*i, color=colors[i])

    return planstr

def FindAllPeaks(apred='daily', telescope='apo25m',sep=50):
    load = apload.ApLoad(apred=apred, telescope=telescope)

    nfiber = 300
    mediansep = 6.65
    pixstart = 28

    visitDir = os.environ.get('APOGEE_REDUX')+'/'+apred+'/visit/'+telescope+'/'
    planfiles = glob.glob(visitDir+'*/*/*/apPlan*yaml')
    planfiles.sort()
    planfiles = np.array(planfiles)
    nplans = len(planfiles)
    print(str(nplans) + ' planfiles found')

    # FITS table structure.
    dt = np.dtype([('PSFID',  np.str, 9),
                   ('MJD',    np.float64),
                   ('XPEAK',  np.float64, nfiber),
                   ('YPEAK',  np.float64, nfiber)])
    peakstruct = np.zeros(nplans,dtype=dt)

    for i in range(nplans):
        planstr = plan.load(planfiles[i], np=True)
        psfid = planstr['psfid']
        twod = load.ap2D(int(psfid))
        gdata = twod['b'][1].data
        header = twod['b'][0].header
        t = Time(header['DATE-OBS'], format='fits')
        peakstruct['PSFID'][i] = psfid
        peakstruct['MJD'][i] = t.mjd

        tot = np.median(gdata[:,1024-100:1024+100], axis=1)
        #import pdb; pdb.set_trace()


        peaks,_ = find_peaks(tot, height=80)
        outfile = '/uufs/chpc.utah.edu/common/home/u0955897/dflat/'+str(psfid)+'_'+str("%.3f" % round(t.mjd,3))+'.dat'
        ascii.write([peaks, tot[peaks]], outfile, names=['x', 'y'], overwrite=True)
        print(len(peaks))

        #for j in range(nfiber):
        #    if j == 0: 
        #        cent = pixstart
        #    else:
        #        cent = peakstruct['XPEAK'][i,j-1] + mediansep
        #    pstart = int(round(np.floor(cent - (mediansep/2.) + 1)))
        #    pstop = int(round(np.ceil(cent + (mediansep/2.) - 1)))
        #    ptot = tot[pstart:pstop]
        #    peaks,_ = find_peaks(ptot, height=80)
        #    peakstruct['XPEAK'][i,j] = cent
        #    peakstruct['YPEAK'][i,j] = 0.0
        #    if len(peaks) != 0:
        #        peakstruct['YPEAK'][i,j] = ptot[peaks][0]

    return tot

def FindAllPeaks2(apred='daily', telescope='apo25m'):
    load = apload.ApLoad(apred=apred, telescope=telescope)

    nfiber = 300

    d = ascii.read('/uufs/chpc.utah.edu/common/home/u0955897/dflat/dflat_xposref.dat')
    xref = np.array(d['col1'])

    files = glob.glob('/uufs/chpc.utah.edu/common/home/u0955897/dflat/3*dat')
    files.sort()
    files=np.array(files)
    nfiles=len(files)

    # FITS table structure.
    dt = np.dtype([('PSFID',  np.str, 9),
                   ('MJD',    np.float64),
                   ('XPEAK',  np.float64, nfiber),
                   ('YPEAK',  np.float64, nfiber)])
    peakstruct = np.zeros(nfiles, dtype=dt)

    for i in range(nfiles):
        d = ascii.read(files[i])
        x = np.array(d['x'])
        y = np.array(d['y'])

        tmp = files[i].split('_')
        peakstruct['PSFID'][i] = tmp[0]
        tmp1 = tmp[1].split('.da')
        peakstruct['MJD'][i] = tmp1[0]

        for j in range(nfiber):
            peakstruct['XPEAK'][i,j] = xref[j]
            peakstruct['YPEAK'][i,j] = 0.0
            dif = np.absolute(x - xref[j])
            gd, = np.where(dif < 2)
            if len(gd) > 0: peakstruct['YPEAK'][i,j] = y[gd]

    return peakstruct

