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
def FindAllPeaks(apred='daily', telescope='apo25m', nplans=None):
    start_time = time.time()

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
    ap2dir = '/uufs/chpc.utah.edu/common/home/sdss/apogeework/apogee/spectro/redux/dr17/'
    mdir = apodir + apred + '/monitor/'
    exp = fits.getdata(mdir + instrument + 'Exp.fits')

    load = apload.ApLoad(apred=apred, telescope=telescope)

    # Find all the SDSS-III and SDSS-IV plan files
    visitDir4 = ap2dir + '/visit/' + telescope + '/'
    planfiles4 = glob.glob(visitDir4 + '*/*/*/apPlan*par')
    planfiles4.sort()
    planfiles4 = np.array(planfiles4)
    nplans4 = len(planfiles4)
    nplanstr4 = str(nplans4)
    print(str(nplans4) + ' pre-SDSS-V planfiles found')

    # Find all the SDSS-V plan files
    visitDir5 = apodir + apred + '/visit/' + telescope + '/'
    planfiles5 = glob.glob(visitDir5 + '*/*/*/apPlan*yaml')
    planfiles5.sort()
    planfiles5 = np.array(planfiles5)
    nplans5 = len(planfiles5)
    nplanstr5 = str(nplans5)
    print(str(nplans5) + ' SDSS-V planfiles found')

    nplans = len(planfiles)
    print('Running code on ' + str(nplans) + ' planfiles')
    print('Estimated runtime: ' + str(int(round(3.86*nplans))) + ' seconds\n')

    # Output file name
    outfile = apodir + apred + '/monitor/' + instrument + 'DomeFlatTrace.fits'

    # Lookup table structure.
    dt = np.dtype([('PSFID',    np.str, 9),
                   ('PLATEID',  np.int32),
                   ('CARTID',   np.int16),
                   ('NAME',     np.str, 14),
                   ('DATEOBS',  np.str, 23),
                   ('MJD',      np.float64),
                   ('CENT',     np.float64, (nchips, nfiber)),
                   ('HEIGHT',   np.float64, (nchips, nfiber)),
                   ('FLUX',     np.float64, (nchips, nfiber))])

    # Loop over the pre-SDSS-V plan files
    outstr4 = np.zeros(nplans4, dtype=dt)
    for i in range(nplans4):
        print('\n(' + str(i+1) + '/' + nplanstr4 + '):')
        planstr = yanny.yanny(planfiles4[i])
        psfid = planstr['psfid']
        import pdb; pdb.set_trace()
        twod = load.ap2D(int(psfid))
        header = twod['a'][0].header

        outstr4['PSFID'][i] =   psfid
        outstr4['PLATEID'][i] = header['PLATEID']
        outstr4['CARTID'][i] =  header['CARTID']
        outstr4['NAME'][i] =    header['NAME']
        outstr4['DATEOBS'][i] = header['DATE-OBS']

        t = Time(header['DATE-OBS'], format='fits')
        outstr4['MJD'][i] = t.mjd

        # Loop over the chips
        for ichip in range(nchips):
            flux = twod[chips[ichip]][1].data
            error = twod[chips[ichip]][2].data

            totflux = np.nanmedian(flux[:, (npix//2) - 200:(npix//2) + 200], axis=1)
            toterror = np.sqrt(np.nanmedian(error[:, (npix//2) - 200:(npix//2) + 200]**2, axis=1))
            pix0 = np.array(refpix[chips[ichip]])
            gpeaks = peakfit.peakfit(totflux, sigma=toterror, pix0=pix0)

            failed, = np.where(gpeaks['success'] == False)

            outstr4['CENT'][i, ichip, :] =    gpeaks['pars'][:, 1]
            outstr4['HEIGHT'][i, ichip, :] =  gpeaks['pars'][:, 0]
            outstr4['FLUX'][i, ichip, :] =    gpeaks['sumflux']

            print('   ' + str(len(gpeaks)) + ' elements in gpeaks; ' + str(300 - len(failed)) + ' successful Gaussian fits')

    # Loop over the SDSS-V plan files
    outstr5 = np.zeros(nplans5, dtype=dt)
    for i in range(nplans5):
        print('\n(' + str(i+1) + '/' + nplanstr5 + '):')
        planstr = plan.load(planfiles5[i], np=True)
        psfid = planstr['psfid']
        twod = load.ap2D(int(psfid))
        header = twod['a'][0].header

        outstr5['PSFID'][i] =   psfid
        outstr5['PLATEID'][i] = header['PLATEID']
        outstr5['CARTID'][i] =  header['CARTID']
        outstr5['NAME'][i] =    header['NAME']
        outstr5['DATEOBS'][i] = header['DATE-OBS']

        t = Time(header['DATE-OBS'], format='fits')
        outstr5['MJD'][i] = t.mjd

        # Loop over the chips
        for ichip in range(nchips):
            flux = twod[chips[ichip]][1].data
            error = twod[chips[ichip]][2].data

            totflux = np.nanmedian(flux[:, (npix//2) - 200:(npix//2) + 200], axis=1)
            toterror = np.sqrt(np.nanmedian(error[:, (npix//2) - 200:(npix//2) + 200]**2, axis=1))
            pix0 = np.array(refpix[chips[ichip]])
            gpeaks = peakfit.peakfit(totflux, sigma=toterror, pix0=pix0)

            failed, = np.where(gpeaks['success'] == False)

            outstr5['CENT'][i, ichip, :] =    gpeaks['pars'][:, 1]
            outstr5['HEIGHT'][i, ichip, :] =  gpeaks['pars'][:, 0]
            outstr5['FLUX'][i, ichip, :] =    gpeaks['sumflux']

            print('   ' + str(len(gpeaks)) + ' elements in gpeaks; ' + str(300 - len(failed)) + ' successful Gaussian fits')

    import pdb; pdb.set_trace()
    Table(outstr).write(outfile, overwrite=True)

    runtime = str("%.2f" % (time.time() - start_time))
    print("\nDone in " + runtime + " seconds.\n")

    return

###################################################################################################
def FindAllPeaks2(apred='daily', telescope='apo25m', medianrad=100, ndomes=None):
    start_time = time.time()

    chips = np.array(['a','b','c'])
    nchips = len(chips)
    nfiber = 300
    npix = 2048

    instrument = 'apogee-n'
    refpix = ascii.read('/uufs/chpc.utah.edu/common/home/u0955897/refpixN.dat')
    
    if telescope == 'lco25m':
        instrument = 'apogee-s'
        refpix = ascii.read('/uufs/chpc.utah.edu/common/home/u0955897/refpixS.dat')

    apodir = os.environ.get('APOGEE_REDUX') + '/' + apred + '/'
    mdir = apodir + '/monitor/'

    expdir5 = apodir + 'exposures/' + instrument + '/'
    expdir4 = '/uufs/chpc.utah.edu/common/home/sdss/apogeework/apogee/spectro/redux/dr17/exposures/' + instrument + '/'

    exp = fits.getdata(mdir + instrument + 'Exp.fits')
    gd, = np.where(exp['IMAGETYP'] == 'DomeFlat')
    gd, = np.where((exp['IMAGETYP'] == 'DomeFlat') & (exp['MJD'] > 56880))
    exp = exp[gd]
    if ndomes is None: ndomes = len(gd)
    ndomestr = str(ndomes)
    
    print('Running code on ' + ndomestr + ' dome flats')
    print('Estimated runtime: ' + str(int(round(3.86*ndomes))) + ' seconds\n')

    # Output file name
    outfile = mdir + instrument + 'DomeFlatTrace.fits'

    # Lookup table structure.
    dt = np.dtype([('PSFID',    np.int32),
                   ('PLATEID',  np.int32),
                   ('CARTID',   np.int16),
                   ('DATEOBS',  np.str, 23),
                   ('MJD',      np.int32),
                   ('CENT',     np.float64, (nchips, nfiber)),
                   ('HEIGHT',   np.float64, (nchips, nfiber)),
                   ('FLUX',     np.float64, (nchips, nfiber)),
                   ('NPEAKS',   np.int16, nchips)])

    outstr = np.zeros(ndomes, dtype=dt)

    # Loop over the dome flats
    for i in range(ndomes):
        print('\n(' + str(i+1) + '/' + ndomestr + '):')

        twodFiles = glob.glob(expdir4 + str(exp['MJD'][i]) + '/ap2D*' + str(exp['NUM'][i]) + '.fits')
        if len(twodFiles) < 1:
            twodFiles = glob.glob(expdir5 + str(exp['MJD'][i]) + '/ap2D*' + str(exp['NUM'][i]) + '.fits')
            if len(twodFiles) < 1:
                print('PROBLEM: ap2D files not found for exposure ' + str(exp['NUM'][i]) + ', MJD ' + str(exp['MJD'][i]))
                continue

        twodFiles.sort()
        twodFiles = np.array(twodFiles)

        if len(twodFiles) < 3:
            print('PROBLEM: less then 3 ap2D files found for exposure ' + str(exp['NUM'][i]) + ', MJD ' + str(exp['MJD'][i]))
            continue
        else:
            print('ap2D files found for exposure ' + str(exp['NUM'][i]) + ', MJD ' + str(exp['MJD'][i]))

        outstr['PSFID'][i] =   exp['NUM'][i]
        outstr['PLATEID'][i] = exp['PLATEID'][i]
        outstr['CARTID'][i] =  exp['CARTID'][i]
        outstr['DATEOBS'][i] = exp['DATEOBS'][i]
        outstr['MJD'][i] =     exp['MJD'][i]

        # Loop over the chips
        for ichip in range(nchips):
            flux = fits.open(twodFiles[ichip])[1].data
            error = fits.open(twodFiles[ichip])[2].data
            npix = flux.shape[0]

            totflux = np.nanmedian(flux[:, (npix//2) - medianrad:(npix//2) + medianrad], axis=1)
            toterror = np.sqrt(np.nanmedian(error[:, (npix//2) - medianrad:(npix//2) + medianrad]**2, axis=1))
            pix0 = np.array(refpix[chips[ichip]])
            gpeaks = peakfit.peakfit(totflux, sigma=toterror, pix0=pix0)

            failed, = np.where(gpeaks['success'] == False)
            success, = np.where(gpeaks['success'] == True)

            outstr['CENT'][i, ichip, :] =    gpeaks['pars'][:, 1]
            outstr['HEIGHT'][i, ichip, :] =  gpeaks['pars'][:, 0]
            outstr['FLUX'][i, ichip, :] =    gpeaks['sumflux']
            outstr['NPEAKS'][i, ichip] =     len(success)

            print(twodFiles[ichip] + ': ' + str(len(success)) + ' successful Gaussian fits')

    Table(outstr).write(outfile, overwrite=True)

    runtime = str("%.2f" % (time.time() - start_time))
    print("\nDone in " + runtime + " seconds.\n")

    return
    
###################################################################################################
def matchtrace(apred='daily', telescope='apo25m', medianrad=100, expnum=36760022):

    chips = np.array(['a','b','c'])
    nchips = len(chips)
    nfiber = 300
    npix = 2048

    instrument = 'apogee-n'
    refpix = ascii.read('/uufs/chpc.utah.edu/common/home/u0955897/refpixN.dat')
    
    if telescope == 'lco25m':
        instrument = 'apogee-s'
        refpix = ascii.read('/uufs/chpc.utah.edu/common/home/u0955897/refpixS.dat')

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

    rmsSum = np.nansum(rms, axis=0)
    gd, = np.where(rmsSum == np.nanmin(rmsSum))

    print(' Best dome flat for exposure ' + str(expnum) + ': ' + str(dome['PSFID'][gd]))
    return dome['PSFID'][gd]

###################################################################################################
def plotresid(apred='daily', telescope='apo25m', medianrad=100, expnum=36760022, psfid=None):

    chips = np.array(['a','b','c'])
    nchips = len(chips)
    nfiber = 300
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
        ax.text(0.03, 0.95-0.05*ichip, str("%.3f" % round(rms,3)), transform=ax.transAxes, color=colors[ichip])
        med = np.nanmedian(diff[gdd]
        ax.text(0.97, 0.95-0.05*ichip, str("%.3f" % round(med,3)), transform=ax.transAxes, color=colors[ichip], ha='right')

    fig.subplots_adjust(left=0.08,right=0.99,bottom=0.095,top=0.98,hspace=0.09,wspace=0.04)
    plt.savefig(mdir + plotfile)
    plt.close('all')
    plt.ion()

























