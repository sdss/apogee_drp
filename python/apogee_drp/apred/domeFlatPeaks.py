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
def FindAllPeaks(apred='daily', telescope='apo25m', medianrad=100, mjdstart=None):
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

    expdir5 = apodir + 'exposures/' + instrument + '/'
    expdir4 = '/uufs/chpc.utah.edu/common/home/sdss/apogeework/apogee/spectro/redux/dr17/exposures/' + instrument + '/'

    exp = fits.getdata(mdir + instrument + 'Exp.fits')
    gd, = np.where(exp['IMAGETYP'] == 'DomeFlat')
    gd, = np.where((exp['IMAGETYP'] == 'DomeFlat') & (exp['MJD'] > 56880))
    exp = exp[gd]

    # Option to start at a certain MJD
    if mjdstart is not None: 
        gd, = np.where(exp['MJD'] >= mjdstart)
        exp = exp[gd]
    ndomes = len(gd)
    ndomestr = str(ndomes)
    
    print('Running code on ' + ndomestr + ' dome flats')
    print('Estimated runtime: ' + str(int(round(3.86*ndomes))) + ' seconds\n')

    # Output file name
    outfile = mdir + instrument + 'DomeFlatTrace.fits'

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

    outstr = np.zeros(ndomes, dtype=dt)

    # Loop over the dome flats
    for i in range(ndomes):
        ttxt = '\n(' + str(i+1) + '/' + ndomestr + '): '

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

        # Get values from master exposure table
        outstr['PSFID'][i] =   exp['NUM'][i]
        outstr['PLATEID'][i] = exp['PLATEID'][i]
        outstr['CARTID'][i] =  exp['CARTID'][i]
        outstr['DATEOBS'][i] = exp['DATEOBS'][i]
        outstr['MJD'][i] =     exp['MJD'][i]

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

        # Loop over the chips
        for ichip in range(nchips):
            pix0 = np.array(refpix[chips[ichip]])
            gpeaks = gaussFitAll(infile=twodFiles[ichip], medianrad=medianrad, pix0=pix0)

            success, = np.where(gpeaks['success'] == True)
            print('  ' + os.path.basename(twodFiles[ichip]) + ': ' + str(len(success)) + ' successful Gaussian fits')

            outstr['PIX0'][i, ichip, :] =            pix0
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

    Table(outstr).write(outfile, overwrite=True)

    runtime = str("%.2f" % (time.time() - start_time))
    print("\nDone in " + runtime + " seconds.\n")

    return

###################################################################################################
def FindIndPeaks(apred='daily', telescope='apo25m', medianrad=100, mjdstart=None, fibrad=4):
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

    expdir5 = apodir + 'exposures/' + instrument + '/'
    expdir4 = '/uufs/chpc.utah.edu/common/home/sdss/apogeework/apogee/spectro/redux/dr17/exposures/' + instrument + '/'

    exp = fits.getdata(mdir + instrument + 'Exp.fits')
    gd, = np.where(exp['IMAGETYP'] == 'DomeFlat')
    gd, = np.where((exp['IMAGETYP'] == 'DomeFlat') & (exp['MJD'] > 56880))
    exp = exp[gd]

    # Option to start at a certain MJD
    if mjdstart is not None: 
        gd, = np.where(exp['MJD'] >= mjdstart)
        exp = exp[gd]
    ndomes = len(gd)
    ndomestr = str(ndomes)
    
    print('Running code on ' + ndomestr + ' dome flats')
    print('Estimated runtime: ' + str(int(round(3.86*ndomes))) + ' seconds\n')

    # Output file name
    outfile = mdir + instrument + 'DomeFlatTrace.fits'

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

    outstr = np.zeros(ndomes, dtype=dt)

    # Loop over the dome flats
    for i in range(ndomes):
        ttxt = '\n(' + str(i+1) + '/' + ndomestr + '): '

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

        # Get values from master exposure table
        outstr['PSFID'][i] =   exp['NUM'][i]
        outstr['PLATEID'][i] = exp['PLATEID'][i]
        outstr['CARTID'][i] =  exp['CARTID'][i]
        outstr['DATEOBS'][i] = exp['DATEOBS'][i]
        outstr['MJD'][i] =     exp['MJD'][i]

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

        # Loop over the chips
        for ichip in range(nchips):
            flux = fits.open(twodFiles[ichip])[1].data
            error = fits.open(twodFiles[ichip])[2].data
            npix = flux.shape[0]
            pix = np.arange(0, npix, 1)
            totflux = np.nanmedian(flux[:, (npix//2) - medianrad:(npix//2) + medianrad], axis=1)
            toterror = np.sqrt(np.nanmedian(error[:, (npix//2) - medianrad:(npix//2) + medianrad]**2, axis=1))
            pix0 = np.array(refpix[chips[ichip]])
            for ifiber in range(nfibers):
                iflux = totflux[pix0[ifiber] - fibrad : pix0[ifiber] + fibrad]
                ierror = toterror[pix0[ifiber] - fibrad : pix0[ifiber] + fibrad]
                gfit = peakfit.gausspeakfit(iflux, pix0=pix0[ifiber], sigma=ierror)

                import pdb; pdb.set_trace()

                #success, = np.where(gpeaks['success'] == True)
                #print('  ' + os.path.basename(twodFiles[ichip]) + ': ' + str(len(success)) + ' successful Gaussian fits')

                #outstr['PIX0'][i, ichip, :] =            pix0
                #outstr['GAUSS_HEIGHT'][i, ichip, :] =    gpeaks['pars'][:, 0]
                #outstr['E_GAUSS_HEIGHT'][i, ichip, :] =  gpeaks['perr'][:, 0]
                #outstr['GAUSS_CENT'][i, ichip, :] =      gpeaks['pars'][:, 1]
                #outstr['E_GAUSS_CENT'][i, ichip, :] =    gpeaks['perr'][:, 1]
                #outstr['GAUSS_SIGMA'][i, ichip, :] =     gpeaks['pars'][:, 2]
                #outstr['E_GAUSS_SIGMA'][i, ichip, :] =   gpeaks['perr'][:, 2]
                #outstr['GAUSS_YOFFSET'][i, ichip, :] =   gpeaks['pars'][:, 3]
                #outstr['E_GAUSS_YOFFSET'][i, ichip, :] = gpeaks['perr'][:, 3]
                #outstr['GAUSS_FLUX'][i, ichip, :] =      gpeaks['sumflux']
                #outstr['GAUSS_NPEAKS'][i, ichip] =       len(success)

    Table(outstr).write(outfile, overwrite=True)

    runtime = str("%.2f" % (time.time() - start_time))
    print("\nDone in " + runtime + " seconds.\n")

    return


###################################################################################################
def matchTraceAll(apred='daily', telescope='apo25m', medianrad=100, expnum=36760022):
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
    mdir = apodir + 'monitor/'
    expdir = apodir + 'exposures/' + instrument + '/'

    dome = fits.getdata(mdir + instrument + 'DomeFlatTrace.fits')
    ndomes = len(dome)

    twodFiles = glob.glob(expdir + '*/ap2D-*' + str(expnum) + '.fits')
    twodFiles.sort()
    twodFiles = np.array(twodFiles)

    if len(twodFiles) < 3:
        print('PROBLEM: less then 3 ap2D files found for exposure ' + str(expnum))
    else:
        print('ap2D files found for exposure ' + str(expnum))

    # Loop over the chips
    rms = np.full([nchips, ndomes], 50).astype(float)
    for ichip in range(nchips):
        flux = fits.open(twodFiles[ichip])[1].data
        error = fits.open(twodFiles[ichip])[2].data
        npix = flux.shape[0]

        totflux = np.nanmedian(flux[:, (npix//2) - medianrad:(npix//2) + medianrad], axis=1)
        toterror = np.sqrt(np.nanmedian(error[:, (npix//2) - medianrad:(npix//2) + medianrad]**2, axis=1))
        pix0 = np.array(refpix[chips[ichip]])
        gpeaks = peakfit.peakfit(totflux, sigma=toterror, pix0=pix0)

        # Remove failed and discrepant peakfits
        gd, = np.where(gpeaks['success'] == True)
        gpeaks = gpeaks[gd]

        # Remove discrepant peakfits
        medcenterr = np.nanmedian(gpeaks['perr'][:, 1])
        gd, = np.where(gpeaks['perr'][:, 1] < medcenterr)
        gpeaks = gpeaks[gd]
        ngpeaks = len(gd)
        print(str(ngpeaks) + ' good peakfits.')

        dcent = dome['CENT'][:, ichip, gpeaks['num']]
        for idome in range(ndomes):
            diff = np.absolute(dcent[idome] - gpeaks['pars'][:, 1])
            gd, = np.where(np.isnan(diff) == False)
            if len(gd) < 5: continue
            diff = diff[gd]
            ndiff = len(diff)
            rms[ichip, idome] = np.sqrt(np.nansum(diff**2)/ndiff)

    rmsMean = np.nanmean(rms, axis=0)
    gd, = np.where(rmsMean == np.nanmin(rmsMean))
    print(rms[:, gd[0]])

    gdrms = str("%.5f" % round(rmsMean[gd][0],5))
    print(' Best dome flat for exposure ' + str(expnum) + ': ' + str(dome['PSFID'][gd][0]) + ' (<rms> = ' + str(gdrms) + ')')
    runtime = str("%.2f" % (time.time() - start_time))
    print("\nDone in " + runtime + " seconds.\n")
    
    return dome['PSFID'][gd]

###################################################################################################
def plotresid(apred='daily', telescope='apo25m', medianrad=100, expnum=36760022, psfid=None):

    chips = np.array(['a','b','c'])
    nchips = len(chips)
    nfibers = 300
    npix = 2048

    instrument = 'apogee-n'
    refpix = ascii.read('/uufs/chpc.utah.edu/common/home/u0955897/refpixN.dat')
    
    if telescope == 'lco25m':
        instrument = 'apogee-s'
        refpix = ascii.read('/uufs/chpc.utah.edu/common/home/u0955897/refpixS.dat')

    apodir = os.environ.get('APOGEE_REDUX') + '/' + apred + '/'
    mdir = apodir + 'monitor/'
    expdir = apodir + 'exposures/' + instrument + '/'

    twodFiles = glob.glob(expdir + '*/ap2D-*' + str(expnum) + '.fits')
    twodFiles.sort()
    twodFiles = np.array(twodFiles)

    if len(twodFiles) < 3:
        print('PROBLEM: less then 3 ap2D files found for exposure ' + str(expnum))
    else:
        print('ap2D files found for exposure ' + str(expnum))

    dome = fits.getdata(mdir + instrument + 'DomeFlatTrace.fits')
    if psfid == None: psfid = matchtrace(apred=apred, telescope=telescope, medianrad=medianrad, expnum=expnum)
    domeind, = np.where(dome['PSFID'] == psfid)
    dome = dome[domeind]

    # Set up some basic plotting parameters, starting by turning off interactive plotting.
    plt.ioff()
    matplotlib.use('agg')
    fontsize = 24;   fsz = fontsize * 0.75
    matplotlib.rcParams.update({'font.size':fontsize, 'font.family':'serif'})
    alpha = 0.6
    axwidth=1.5
    axmajlen=7
    axminlen=3.5
    colors = np.array(['red','green','blue'])
    markers = np.array(['o','^','s'])

    plotfile = 'dflatTrace_sci' + str(expnum) + '_dome' + str(psfid) + '.png'
    fig=plt.figure(figsize=(20,8))
    ax = plt.subplot2grid((1,1), (0,0))
    ax.tick_params(reset=True)
    ax.minorticks_on()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.set_xlabel(r'Fiber Index')
    ax.set_ylabel(r'Dome - Science Residuals')
    ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
    ax.tick_params(axis='both',which='major',length=axmajlen)
    ax.tick_params(axis='both',which='minor',length=axminlen)
    ax.tick_params(axis='both',which='both',width=axwidth)
    ax.set_xlim(-2, 302)
    ax.set_ylim(-0.5, 0.5)
    ax.axhline(y=0, linestyle='dashed', color='k')

    # Loop over the chips
    for ichip in range(nchips):
        flux = fits.open(twodFiles[ichip])[1].data
        error = fits.open(twodFiles[ichip])[2].data
        npix = flux.shape[0]

        totflux = np.nanmedian(flux[:, (npix//2) - medianrad:(npix//2) + medianrad], axis=1)
        toterror = np.sqrt(np.nanmedian(error[:, (npix//2) - medianrad:(npix//2) + medianrad]**2, axis=1))
        pix0 = np.array(refpix[chips[ichip]])
        gpeaks = peakfit.peakfit(totflux, sigma=toterror, pix0=pix0)

        # Remove failed and discrepant peakfits
        gd, = np.where(gpeaks['success'] == True)
        gpeaks = gpeaks[gd]

        # Remove discrepant peakfits
        medcenterr = np.nanmedian(gpeaks['perr'][:, 1])
        gdd, = np.where(gpeaks['perr'][:, 1] < medcenterr)
        gnum = gpeaks['num'][gdd]
        ngpeaks = len(gdd)

        dcent = dome['CENT'][0, ichip, gpeaks['num']]
        diff = dcent - gpeaks['pars'][:, 1]
        gd, = np.where(np.isnan(diff) == False)
        num = gpeaks['num'][gd]
        gdiff = diff[gd]
        ax.scatter(num, gdiff, marker=markers[ichip], color='white', edgecolors=colors[ichip], linewidth=1.5, alpha=0.5)
        ax.scatter(gnum, diff[gdd], marker=markers[ichip], color=colors[ichip], linewidth=1.5, alpha=1)
        rms = np.sqrt(np.nansum(diff[gdd]**2)/ngpeaks)
        ax.text(0.03, 0.95-0.05*ichip, str("%.4f" % round(rms,3)), transform=ax.transAxes, color=colors[ichip])
        med = np.nanmedian(diff[gdd])
        ax.text(0.97, 0.95-0.05*ichip, str("%.4f" % round(med,3)), transform=ax.transAxes, color=colors[ichip], ha='right')

    fig.subplots_adjust(left=0.08,right=0.99,bottom=0.095,top=0.98,hspace=0.09,wspace=0.04)
    plt.savefig(mdir + plotfile)
    plt.close('all')
    plt.ion()

###################################################################################################
def gaussFitAll(infile=None, medianrad=None, pix0=None):
    flux = fits.open(infile)[1].data
    error = fits.open(infile)[2].data
    npix = flux.shape[0]

    totflux = np.nanmedian(flux[:, (npix//2) - medianrad:(npix//2) + medianrad], axis=1)
    toterror = np.sqrt(np.nanmedian(error[:, (npix//2) - medianrad:(npix//2) + medianrad]**2, axis=1))
    gpeaks = peakfit.peakfit(totflux, sigma=toterror, pix0=pix0)

    return gpeaks

###################################################################################################
def gaussFitFiber(infile=None, medianrad=None, pix0=None):
    flux = fits.open(infile)[1].data
    error = fits.open(infile)[2].data
    npix = flux.shape[0]

    totflux = np.nanmedian(flux[:, (npix//2) - medianrad:(npix//2) + medianrad], axis=1)
    toterror = np.sqrt(np.nanmedian(error[:, (npix//2) - medianrad:(npix//2) + medianrad]**2, axis=1))
    gpeaks = peakfit.peakfit(totflux, sigma=toterror, pix0=pix0)

    return gpeaks





















