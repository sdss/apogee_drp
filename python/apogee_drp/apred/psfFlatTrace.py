import sys
import pdb
import glob
import os
import subprocess
import math
import time
import pdb
import fitsio
import numpy as np
from pathlib import Path
from astropy.io import fits, ascii
from astropy.table import Table,vstack
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
# Program for matching a sequence of exposures to the PSF flat lookup table.
# Calls "findBestFlatExposure" to do the matching.
#
# Inputs can be:
#     (1) an array of exposure numbers, e.g. [36880094, 36880095], or [36880094] for single exposure
#     (2) a planfile
#     (3) a plate + MJD pair
#
# Output can be:
#     (1) an array of dome/quartz flat exposure numbers, one per input exposure number (default)
#     (2) a single dome/quartz flat exposure number (if the "single" keyword is set)
#
###################################################################################################
def findBestFlatSequence(ims=None, imtype='QuartzFlat', libFile=None, planfile=None, 
                         mjdplate='59146-15000', observatory='apo', medianrad=100, apred='daily', 
                         apred_exp=None, single=False, highfluxfrac=None, minflux=None, silent=True):

    start_time = time.time()
    #print("Finding best dome flats for plate " + plate + ", MJD " + mjd)

    mjd = mjdplate.split('-')[0]
    plate = mjdplate.split('-')[1]

    # Option to use a different apred to find the science exposures
    if apred_exp is None: apred_exp = apred

    # Get telescope/instrument info
    instrument = 'apogee-n'
    inst = 'N'
    telescope = 'apo25m'
    if observatory == 'lco':
        instrument = 'apogee-s'
        inst = 'S'
        telescope = 'lco25m'

    # Set up apload
    load = apload.ApLoad(apred=apred, telescope=telescope)
    load_exp = apload.ApLoad(apred=apred_exp, telescope=telescope)

    # Find the reference pixel file
    fil = os.path.abspath(__file__)
    codedir = os.path.dirname(fil)
    datadir = os.path.dirname(os.path.dirname(os.path.dirname(codedir))) + '/data/domeflat/'

    # Need to make pixel reference file for lco25m
    if inst == 'S': sys.exit('Problem! Reference pixel file for LCO 2.5m does not exist yet')
    refpix = ascii.read(datadir + 'refpix' + inst + '.dat')

    # Establish directories.
    datadir = {'apo25m':os.environ['APOGEE_DATA_N'], 'apo1m':os.environ['APOGEE_DATA_N'],
               'lco25m':os.environ['APOGEE_DATA_S']}[telescope]
    apodir = os.environ.get('APOGEE_REDUX') + '/' + apred + '/'
    mdir = apodir + 'monitor/'

    # Read in the dome flat lookup table and master exposure table
    flatTable = fits.getdata(mdir + instrument + imtype + 'Trace-all.fits')
    if libFile is not None: flatTable = fits.getdata(mdir + libFile)
    expTable = fits.getdata(mdir + instrument + 'Exp.fits')

    # Restrict to valid data with at least nchips*290 Gaussian fits
    gd, = np.where((flatTable['MJD'] > 0) & (np.sum(flatTable['GAUSS_NPEAKS'], axis=1) > 870))
    flatTable = flatTable[gd]
    # separate into plates and FPS era
    gdplates = np.where(flatTable['MJD']<59556)
    flatTablePlates = flatTable[gdplates]
    gdfps = np.where(flatTable['MJD']>=59556)
    flatTableFPS = flatTable[gdfps]
    #if medianrad != 100: flatTable = fits.getdata(mdir + instrument + 'DomeFlatTrace-all_medrad' + str(medianrad) + '.fits')

    if ims is None:
        # Load planfile into structure.
        if planfile is None: planfile = load.filename('Plan', plate=int(plate), mjd=mjd)
        planstr = plan.load(planfile, np=True)

        # Get array of object exposures and find out how many are objects.
        gd, = np.where(planstr['APEXP']['flavor'] == 'object')
        ims = planstr['APEXP']['name'][gd]
        mjd = str(planstr['mjd'])
    else:
        # Construct MJD from exposure number
        num = (ims[0] - ims[0] % 10000 ) / 10000
        mjd = '{:05d}'.format(int(num) + 55562)

    n_ims = len(ims)

    print(str(int(round(n_ims))) + " exposures\n")

    flatnums = np.empty(n_ims).astype(int)
    flatmjds = np.empty(n_ims).astype(int)
    rms =      np.empty(n_ims)

    for i in range(n_ims):
        # Find the ap2D files for this exposure
        file2d = load_exp.filename('2D', mjd=mjd, num=ims[i], chips='c')
        twodfiles = np.array([file2d.replace('2D-', '2D-a-'),
                              file2d.replace('2D-', '2D-b-'),
                              file2d.replace('2D-', '2D-c-')])

        # Use plate-era flats for plate exposures and and FPS-era flats for FPS exposures
        mjd = int(load.cmjd(ims[i]))
        if mjd>=59556:
            flatTable1 = flatTableFPS
        else:
            flatTable1 = flatTablePlates
        
        # Run findBestFlatExposure on this exposure
        flatnums[i], flatmjds[i], rms[i] = findBestFlatExposure(flatTable=flatTable1, imtype=imtype, refpix=refpix, 
                                                                twodfiles=twodfiles, medianrad=medianrad,
                                                                minflux=minflux, highfluxfrac=highfluxfrac, silent=silent)
        # Print info about this exposure and the matching dome flat
        pflat, = np.where(flatnums[i] == flatTable1['PSFID'])
        psci, = np.where(ims[i] == expTable['NUM'])
        if len(psci) > 0:
            p1 = '(' + str(i+1).rjust(2) + ') sci exposure ' + str(ims[i]) + ' ----> ' + imtype + ' ' + str(int(round(flatnums[i]))) + ' (MJD ' + str(int(round(flatmjds[i]))) + '),  '
            #p2 = 'alt [' + str("%.3f" % round(expTable['ALT'][psci][0],3)) + ', ' + str("%.3f" % round(flatTable['ALT'][pflat][0],3)) + '],  '
            p3 = 'ln2level [' + str("%.3f" % round(expTable['LN2LEVEL'][psci][0],3)) + ', ' + str("%.3f" % round(flatTable1['LN2LEVEL'][pflat][0],3)) + '],   '
            p4 = 'rms = ' + str("%.4f" % round(rms[i],4))
            print(p1 + p3 + p4)
        else:
            p1 = '(' + str(i+1).rjust(2) + ') sci exposure ' + str(ims[i]) + ' ----> ' + imtype + ' ' + str(int(round(flatnums[i]))) + ' (MJD ' + str(int(round(flatmjds[i]))) + '),  '
            p4 = 'rms = ' + str("%.4f" % round(rms[i],4))
            print(p1 + p4)

    # Get the unique dome flat exposure numbers
    uniqflatnums = np.unique(flatnums)
    nflats = len(uniqflatnums)
    print('\n' + str(int(round(nflats))) + ' ' + imtype)

    # Check on how many time each unique dome flat was selected
    nrepeats = np.empty(nflats)
    for i in range(nflats):
        tmp, = np.where(uniqflatnums[i] == flatnums)
        nrepeats[i] = len(tmp)
        print(str(int(round(uniqflatnums[i]))) + ':  ' + str(int(round(nrepeats[i]))).rjust(2) + ' matches')

    # Option to retun a single dome flat rather than an array of them
    if single is True:
        maxrepeats = np.max(nrepeats)
        maxind, = np.where(nrepeats == maxrepeats)
        uniqflatnums = uniqflatnums[maxind]
        if len(maxind) == 1:
            flatnums = uniqflatnums[0]
        else:
            # If more than one dome flat have maxrepeats, decide based on rms
            urms = np.empty(len(uniqflatnums))
            for j in range(len(uniqflatnums)):
                gd, = np.where(uniqflatnums[j] == flatnums)
                urms[j] = np.sum(rms[gd])
            minrmsind, = np.where(urms == np.min(urms))
            flatnums = uniqflatnums[minrmsind][0]
        print("\nSingle keyword set: going with " + imtype + " " + str(flatnums) + " for all exposures.")

    runtime = str("%.2f" % (time.time() - start_time))
    print("\nDone in " + runtime + " seconds.")

    return flatnums,ims


###################################################################################################
# Function for matching an exposure to the dome flat lookup table
# Calls "gaussFitAll" to do the Gaussian fitting
#
# Only run by calling findBestFlatSequence, which can be a single exposure
#
# Output is the matched dome flat exposure number, MJD, and the r.m.s. of the match.
#
###################################################################################################
def findBestFlatExposure(flatTable=None, imtype=None, refpix=None, twodfiles=None, medianrad=100, 
                         minflux=None, highfluxfrac=None, silent=True):

    chips = np.array(['a', 'b', 'c'])
    nchips = len(chips)
    nfibers = 300
    nflats = len(flatTable)

    medFWHM = np.nanmedian(flatTable['GAUSS_SIGMA'], axis=2) * 2.354

    # Loop over the chips
    rms = np.full([nchips, nflats], 50).astype(np.float64)
    for ichip in range(nchips):
        # ap2D file for this chip
        twodfile = twodfiles[ichip]

        # Get exposure number from the filename
        expnum = int(twodfile.split('-')[-1].split('.')[0])

        # Get reference pixels for this chip
        pix0 = np.array(refpix[chips[ichip]])
        # Fit Gaussians to the trace positions
        gpeaks = gaussFitAll(infile=twodfile, medianrad=medianrad, pix0=pix0)

        # Remove failed and discrepant peakfits
        gd, = np.where(gpeaks['success'] == True)
        gpeaks = gpeaks[gd]

        #for k in range(len(gpeaks)):
        #    print(str(gpeaks['num'][k]) + '   ' + str(gpeaks['pars'][k, 1]))

        # Remove discrepant peakfits
        medcenterr = np.nanmedian(gpeaks['perr'][:, 1])
        gd, = np.where(gpeaks['perr'][:, 1] < medcenterr)
        gpeaks = gpeaks[gd]
        ngpeaks = len(gd)

        # Find median gaussian FWHM and restrict lookup table to similar values
        #medFWHM = np.nanmedian(gpeaks['pars'][:, 2]) * 2.354
        #print('Median Science FWHM (chip ' + chips[ichip] + ') = ' + str("%.5f" % round(medFWHM,5)))
        #medFWHM = np.nanmedian(flatTable['GAUSS_SIGMA'][:, ichip, gpeaks['num']], axis=1)*2.354
        #gd, = np.where(np.absolute(medFWHM - medFWHM[:, ichip]) < 0.05)
        flatTable1 = flatTable#[gd]
        nflats1 = len(flatTable1)

        # Option to only use fibers with flux higher than average dome flat flux
        if highfluxfrac is not None:
            if (highfluxfrac < 0) | (highfluxfrac > 1):
                sys.exit("The highfluxfrac value needs to be between 0 and 1. Try again.")
            nkeep = int(np.ceil(ngpeaks * highfluxfrac))
            if silent is False: print("   " + str(ngpeaks) + " successful peakfits. Matching to dome flats based on the " + str(nkeep) + " highest flux fibers")
            # Sort by flux sum and keep the highest flux fibers
            fluxord = np.argsort(gpeaks['sumflux'])[::-1]
            gpeaks = gpeaks[fluxord][:nkeep]
            numord = np.argsort(gpeaks['num'])
            gpeaks = gpeaks[numord]
            ngpeaks = len(gpeaks)
        else:
            if silent is False: print("   " + str(ngpeaks) + " successful peakfits")

        # Option to only keep fibers above a certain flux level
        if minflux is not None:
            gd, = np.where(gpeaks['sumflux'] > minflux)
            gpeaks = gpeaks[gd]
            ngpeaks = len(gpeaks)
            if silent is False: print("   Keeping " + str(ngpeaks) + " fibers with flux > " + str(minflux))

        dcent = flatTable1['GAUSS_CENT'][:, ichip, gpeaks['num']]
        for iflat in range(nflats1):
            diff = np.absolute(dcent[iflat] - gpeaks['pars'][:, 1])
            gd, = np.where(np.isnan(diff) == False)
            if len(gd) < 5: continue
            diff = diff[gd]
            ndiff = len(diff)
            rms[ichip, iflat] = dln.mad(diff,zero=True)  # use robust RMS
            #rms[ichip, iflat] = np.sqrt(np.nansum(diff**2)/ndiff)
        #if silent is False: print("   Final match based on " + str(ndiff) + " fibers:")

    rmsMean = np.nanmean(rms, axis=0)
    gd, = np.where(rmsMean == np.nanmin(rmsMean))
    if silent is False: print("   rms:  " + str(rms[:, gd[0]]))

    gdrms = str("%.5f" % round(rmsMean[gd][0],5))
    if silent is False: print("   Best " + imtype + " for exposure " + str(expnum) + ": " + str(flatTable1['PSFID'][gd][0]) + " (<rms> = " + str(gdrms) + ")")

    #print(medFWHM[gd][0])

    return flatTable1['PSFID'][gd][0], flatTable1['MJD'][gd][0], rmsMean[gd][0]


###################################################################################################
# Program for making the domeflat lookup table.
# Takes about 8 hours to run, unless a later "mjdstart" is specified.
###################################################################################################
def makeLookupTable(apred='daily', telescope='apo25m', imtype='QuartzFlat', medianrad=100, append=True):

    start_time = time.time()

    chips = np.array(['a','b','c'])
    nchips = len(chips)
    nfibers = 300

    instrument = 'apogee-n'
    inst = 'N'
    if telescope == 'lco25m':
        instrument = 'apogee-s'
        inst = 'S'
    fil = os.path.abspath(__file__)
    codedir = os.path.dirname(fil)
    datadir = os.path.dirname(os.path.dirname(os.path.dirname(codedir))) + '/data/domeflat/'
    refpix = ascii.read(datadir + 'refpix' + inst + '.dat')

    apodir = os.environ.get('APOGEE_REDUX') + '/' + apred + '/'
    mdir = apodir + '/monitor/'

    # Output file name
    outfile = mdir + instrument + imtype + 'Trace-all.fits'
    print('Output file = ' + os.path.basename(outfile))

    expdir5 = apodir + 'exposures/' + instrument + '/'
    expdir4 = '/uufs/chpc.utah.edu/common/home/sdss/apogeework/apogee/spectro/redux/dr17/exposures/' + instrument + '/'

    # Read in the exposure summary file and restrict to either dome or quartz
    exp = fits.getdata(mdir + instrument + 'Exp.fits')
    gd, = np.where(exp['IMAGETYP'] == imtype)
    gd, = np.where((exp['IMAGETYP'] == imtype) & (exp['NUM'] == 40110002))
    exp = exp[gd]

    # Default option to append new values rather than remake the entire file
    if append:
        clib = fitsio.read(outfile)
        gd, = np.where(exp['NUM'] > np.max(clib['PSFID']))
        if len(gd) < 1:
            print(os.path.basename(outfile) + ' is already up-to-date.')
            return
        else:
            exp = exp[gd]
            nexp = len(exp)
            nexptr = str(nexp)
            print('Adding ' + str(nexp) + ' exposures to ' + os.path.basename(outfile) + '.')
    else:
        print('Running code on ' + nexptr + ' ' + imtype + ' exposures.')
    print('Estimated runtime: ' + str(int(round(3.86*nexp))) + ' seconds.\n')

    #pdb.set_trace()
    # Lookup table structure.
    dt = np.dtype([('PSFID',           np.int32),
                   ('PLATEID',         np.int32),
                   ('CARTID',          np.int16),
                   ('DATEOBS',         np.str, 23),
                   ('MJD',             np.int32),
                   ('EXPTIME',         np.float64),
                   ('NREAD',           np.int16),
                   ('ROTPOS',          np.float64),
                   ('SEEING',          np.float64),
                   ('AZ',              np.float64),
                   ('ALT',             np.float64),
                   ('IPA',             np.float64),
                   ('FOCUS',           np.float64),
                   ('DITHPIX',         np.float32),
                   ('LN2LEVEL',        np.float32),
                   ('RELHUM',          np.float32),
                   ('MKSVAC',          np.float32),
                   ('TDETTOP',         np.float32),
                   ('TDETBASE',        np.float32),
                   ('TTENTTOP',        np.float32),
                   ('TCLDPMID',        np.float32),
                   ('TGETTER',         np.float32),
                   ('TTLMBRD',         np.float32),
                   ('TLSOUTH',         np.float32),
                   ('TLNORTH',         np.float32),
                   ('TLSCAM2',         np.float32),
                   ('TLSCAM1',         np.float32),
                   ('TLSDETC',         np.float32),
                   ('TLSDETB',         np.float32),
                   ('TPGVAC',          np.float32),
                   ('TCAMAFT',         np.float32),
                   ('TCAMMID',         np.float32),
                   ('TCAMFWD',         np.float32),
                   ('TEMPVPH',         np.float32),
                   ('TRADSHLD',        np.float32),
                   ('TCOLLIM',         np.float32),
                   ('TCPCORN',         np.float32),
                   ('TCLDPHNG',        np.float32),
                   ('PIX0',            np.int16,   (nchips, nfibers)),
                   ('GAUSS_HEIGHT',    np.float64, (nchips, nfibers)),
                   ('E_GAUSS_HEIGHT',  np.float64, (nchips, nfibers)),
                   ('GAUSS_CENT',      np.float64, (nchips, nfibers)),
                   ('E_GAUSS_CENT',    np.float64, (nchips, nfibers)),
                   ('GAUSS_SIGMA',     np.float64, (nchips, nfibers)),
                   ('E_GAUSS_SIGMA',   np.float64, (nchips, nfibers)),
                   ('GAUSS_YOFFSET',   np.float64, (nchips, nfibers)),
                   ('E_GAUSS_YOFFSET', np.float64, (nchips, nfibers)),
                   ('GAUSS_FLUX',      np.float64, (nchips, nfibers)),
                   ('GAUSS_NPEAKS',    np.int16, nchips)])

    outstr = np.zeros(nexp, dtype=dt)

    # Loop over the dome flats
    for i in range(nexp):
        ttxt = '\n(' + str(i+1) + '/' + nexptr + '): '

        # Get values from master exposure table
        outstr['PSFID'][i] =   exp['NUM'][i]
        outstr['PLATEID'][i] = exp['PLATEID'][i]
        if type(exp['CARTID'][i]) != str: outstr['CARTID'][i] =  exp['CARTID'][i]
        outstr['DATEOBS'][i] = exp['DATEOBS'][i]
        outstr['MJD'][i] =     exp['MJD'][i]

        # Make sure there's a valid MJD
        if exp['MJD'][i] < 100: 
            print(ttxt + 'PROBLEM: MJD < 100 for ' + str(exp['NUM'][i]))
            continue

        # Find the ap2D files for all 3 chips
        twodFiles = glob.glob(expdir4 + str(exp['MJD'][i]) + '/ap2D*' + str(exp['NUM'][i]) + '.fits')
        if len(twodFiles) < 1:
            twodFiles = glob.glob(expdir5 + str(exp['MJD'][i]) + '/ap2D*' + str(exp['NUM'][i]) + '.fits')
            if len(twodFiles) < 1:
                print(ttxt + 'PROBLEM: ap2D files not found for exposure ' + str(exp['NUM'][i]) + ', MJD ' + str(exp['MJD'][i]))
                continue
        twodFiles.sort()
        twodFiles = np.array(twodFiles)

        if len(twodFiles) < 3:
            print(ttxt + 'PROBLEM: <3 ap2D files found for exposure ' + str(exp['NUM'][i]) + ', MJD ' + str(exp['MJD'][i]))
            continue
        else:
            print(ttxt + 'ap2D files found for exposure ' + str(exp['NUM'][i]) + ', MJD ' + str(exp['MJD'][i]))

        # Get ap2D header values
        hdr = fits.getheader(twodFiles[0])
        outstr['EXPTIME'][i] =  hdr['EXPTIME']
        outstr['NREAD'][i] =    hdr['NREAD']
        try:
            outstr['ROTPOS'][i] = hdr['ROTPOS']
        except:
            outstr['ROTPOS'][i] = -9999.999
        try:
            doutstr['SEEING'][i] =   hdr['SEEING']
        except:
            outstr['SEEING'][i] = -9999.999
        outstr['AZ'][i] =       hdr['AZ']
        outstr['ALT'][i] =      hdr['ALT']
        outstr['IPA'][i] =      hdr['IPA']
        outstr['FOCUS'][i] =    hdr['FOCUS']
        outstr['DITHPIX'][i] =  hdr['DITHPIX']
        outstr['LN2LEVEL'][i] = hdr['LN2LEVEL']
        outstr['RELHUM'][i] =   hdr['RELHUM']
        outstr['MKSVAC'][i] =   hdr['MKSVAC']
        outstr['TDETTOP'][i] =  hdr['TDETTOP']
        outstr['TDETBASE'][i] = hdr['TDETBASE']
        outstr['TTENTTOP'][i] = hdr['TTENTTOP']
        outstr['TCLDPMID'][i] = hdr['TCLDPMID']
        outstr['TGETTER'][i] =  hdr['TGETTER']
        outstr['TTLMBRD'][i] =  hdr['TTLMBRD']
        outstr['TLSOUTH'][i] =  hdr['TLSOUTH']
        outstr['TLNORTH'][i] =  hdr['TLNORTH']
        outstr['TLSCAM2'][i] =  hdr['TLSCAM2']
        outstr['TLSCAM1'][i] =  hdr['TLSCAM1']
        outstr['TLSDETC'][i] =  hdr['TLSDETC']
        outstr['TLSDETB'][i] =  hdr['TLSDETB']
        outstr['TPGVAC'][i] =   hdr['TPGVAC']
        outstr['TCAMAFT'][i] =  hdr['TCAMAFT']
        outstr['TCAMMID'][i] =  hdr['TCAMMID']
        outstr['TCAMFWD'][i] =  hdr['TCAMFWD']
        outstr['TEMPVPH'][i] =  hdr['TEMPVPH']
        outstr['TRADSHLD'][i] = hdr['TRADSHLD']
        outstr['TCOLLIM'][i] =  hdr['TCOLLIM']
        outstr['TCPCORN'][i] =  hdr['TCPCORN']
        outstr['TCLDPHNG'][i] = hdr['TCLDPHNG']

        try:
            # Loop over the chips
            for ichip in range(nchips):
                pix0 = np.array(refpix[chips[ichip]])

                # Slight corrections to reference pixel values for older apo25m dome flats
                pix0g = pix0
                if telescope == 'apo25m':
                    if exp['MJD'][i] < 56530:
                        if ichip == 0: pix0g = pix0g - 1
                    else:
                        if ichip == 0: pix0g = pix0g - 0.88
                        if ichip == 1: pix0g = pix0g - 0.78
                        if ichip == 2: pix0g = pix0g - 0.74

                # Initial gaussian fit
                gpeaks0 = gaussFitAll(infile=twodFiles[ichip], medianrad=medianrad, pix0=pix0g)

                # Run again to avoid hitting boundaries
                cen0 = gpeaks0['pars'][:, 1]
                bad, = np.where(np.isnan(cen0))
                cen0[bad] = pix0g[bad]
                gpeaks = gaussFitAll(infile=twodFiles[ichip], medianrad=medianrad, pix0=cen0)

                success, = np.where(gpeaks['success'] == True)
                print('  ' + os.path.basename(twodFiles[ichip]) + ': ' + str(len(success)) + ' successful Gaussian fits')

                outstr['PIX0'][i, ichip, :] =            pix0g
                outstr['GAUSS_HEIGHT'][i, ichip, :] =    gpeaks['pars'][:, 0]
                outstr['E_GAUSS_HEIGHT'][i, ichip, :] =  gpeaks['perr'][:, 0]
                outstr['GAUSS_CENT'][i, ichip, :] =      gpeaks['pars'][:, 1]
                outstr['E_GAUSS_CENT'][i, ichip, :] =    gpeaks['perr'][:, 1]
                outstr['GAUSS_SIGMA'][i, ichip, :] =     gpeaks['pars'][:, 2]
                outstr['E_GAUSS_SIGMA'][i, ichip, :] =   gpeaks['perr'][:, 2]
                outstr['GAUSS_YOFFSET'][i, ichip, :] =   gpeaks['pars'][:, 3]
                outstr['E_GAUSS_YOFFSET'][i, ichip, :] = gpeaks['perr'][:, 3]
                outstr['GAUSS_FLUX'][i, ichip, :] =      gpeaks['sumflux']
                outstr['GAUSS_NPEAKS'][i, ichip] =       len(success)
        except:
            print(' problem with exposure ' + str(exp['NUM'][i]))
            continue

    # Either append new results to master file, or create new master file

    pdb.set_trace()

    if append:
        vstack([Table(clib), Table(outstr)]).write(outfile, overwrite=True)
    else:
        Table(outstr).write(outfile, overwrite=True)

    runtime = str("%.2f" % (time.time() - start_time))
    print("\nDone in " + runtime + " seconds.\n")

    return


###################################################################################################
# Function for fitting Gaussians to trace positions of all 300 fibers
###################################################################################################
def gaussFitAll(infile=None, medianrad=None, pix0=None):
    flux = fits.getdata(infile,1)
    error = fits.getdata(infile,2)
    npix = flux.shape[0]

    totflux = np.nanmedian(flux[:, (npix//2) - medianrad:(npix//2) + medianrad], axis=1)
    toterror = np.sqrt(np.nanmedian(error[:, (npix//2) - medianrad:(npix//2) + medianrad]**2, axis=1))
    gpeaks = peakfit.peakfit(totflux, sigma=toterror, pix0=pix0)
    #pars,perr = peakfit.gausspeakfit(totflux, sigma=toterror, pix0=pix0)

    #pdb.set_trace()

    return gpeaks


