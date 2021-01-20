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
from astroplan import moon_illumination
from astropy.coordinates import SkyCoord, get_moon
from astropy import units as astropyUnits
from scipy.signal import medfilt2d as ScipyMedfilt2D
from apogee_drp.utils import plan,apload,yanny,plugmap,platedata,bitmask
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

# import pdb; pdb.set_trace()

''' MONITOR: Instrument monitoring plots and html '''
def monitor(instrument='apogee-n', apred='daily', clobber=True, makesumfiles=False, allplots=True):

    print("----> monitor starting")

    chips = np.array(['blue','green','red'])
    nchips = len(chips)

    fibers = np.array([10,80,150,220,290])
    nfibers = len(fibers)

    nlines = 2

    nquad = 4

    # Establish  directories... hardcode sdss4/apogee2 for now

    specdir5 = os.environ.get('APOGEE_REDUX') + '/' + apred + '/'
    specdir = '/uufs/chpc.utah.edu/common/home/sdss/apogeework/apogee/spectro/redux/current/'
    sdir5 = specdir5 + 'monitor/' + instrument + '/'
    sdir = '/uufs/chpc.utah.edu/common/home/sdss/apogeework/apogee/spectro/redux/current/monitor/apogee-n/'
    mdir = 'monitor/' + instrument + '/'

    if makesumfiles is False:
        # NOTE: we probably don't want the below files in the redux/daily directory.
        allcal =  fits.open(specdir + instrument + 'Cal.fits')[1].data
        alldark = fits.open(specdir + instrument + 'Cal.fits')[2].data
        allexp =  fits.open(specdir + instrument + 'Exp.fits')[1].data
        allsci =  fits.open(specdir + instrument + 'Sci.fits')[1].data
        allepsf = fits.open(specdir + instrument + 'Trace.fits')[1].data
    else:
        ###########################################################################################
        # MAKE MASTER QACAL FILE
        outfile = specdir5 + 'monitor/' + instrument + 'Cal.fits'
        print("----> monitor: Making " + os.path.basename(outfile))

        # Append together the individual QAcal files
        files = glob.glob(specdir + '/cal/' + instrument + '/*/*QAcal*.fits')
        if len(files) < 1:
            print("----> monitor: No QAcal files!")
        else:
            files.sort()
            files = np.array(files)
            nfiles = len(files)
            for i in range(nfiles):
                print("---->    monitor: reading " + files[i])
                a = fits.open(files[i])[1].data

                # Make output structure and write to fits file
                dt = np.dtype([('NAME',    np.str,30),
                               ('MJD',     np.str,30),
                               ('JD',      np.float64),
                               ('NFRAMES', np.int32),
                               ('NREAD',   np.int32),
                               ('EXPTIME', np.float64),
                               ('QRTZ',    np.int32),
                               ('UNE',     np.int32),
                               ('THAR',    np.int32),
                               ('FLUX',    np.float64,(300,nchips)),
                               ('GAUSS',   np.float64,(4,nfibers,nchips,nlines)),
                               ('WAVE',    np.float64,(nfibers,nchips,nlines)),
                               ('FIBERS',  np.float64,(nfibers)),
                               ('LINES',   np.float64,(nchips,nlines))])
                struct = np.zeros(len(a['NAME']), dtype=dt)

                struct['NAME'] = a['NAME']
                struct['MJD'] = a['MJD']
                struct['JD'] = a['JD']
                struct['NFRAMES'] = a['NFRAMES']
                struct['NREAD'] = a['NREAD']
                struct['EXPTIME'] = a['EXPTIME']
                struct['QRTZ'] = a['QRTZ']
                struct['UNE'] = a['UNE']
                struct['THAR'] = a['THAR']
                struct['FLUX'] = a['FLUX']
                struct['GAUSS'] = a['GAUSS']
                struct['WAVE'] = a['WAVE']
                struct['FIBERS'] = a['FIBERS']
                struct['LINES'] = a['LINES']

                if i == 0:
                    allcal = struct
                else:
                    allcal = np.concatenate([allcal, struct])

            Table(allcal).write(outfile, overwrite=True)
            print("----> monitor: Finished adding QAcal info to " + os.path.basename(outfile))

        ###########################################################################################
        # APPEND QADARKFLAT INFO TO MASTER QACAL FILE
        print("----> monitor: Adding QAdarkflat info to " + os.path.basename(outfile))

        # Append together the individual QAdarkflat files
        files = glob.glob(specdir + '/cal/' + instrument + '/*/*QAdarkflat*.fits')
        if len(files) < 1:
            print("----> monitor: No QAdarkflat files!")
        else:
            files.sort()
            files = np.array(files)
            nfiles = len(files)
            for i in range(nfiles):
                print("---->    monitor: reading " + files[i])
                a = fits.open(files[i])[1].data

                # Make output structure.
                dt = np.dtype([('NAME',    np.str, 30),
                               ('MJD',     np.str, 30),
                               ('JD',      np.float64),
                               ('NFRAMES', np.int32),
                               ('NREAD',   np.int32),
                               ('EXPTIME', np.float64),
                               ('QRTZ',    np.int32),
                               ('UNE',     np.int32),
                               ('THAR',    np.int32),
                               ('EXPTYPE', np.str, 30),
                               ('MEAN',    np.float64, (nchips,nquad)),
                               ('SIG',     np.float64, (nchips,nquad))])
                struct = np.zeros(len(a['NAME']), dtype=dt)

                struct['NAME'] = a['NAME']
                struct['MJD'] = a['MJD']
                struct['JD'] = a['JD']
                struct['NFRAMES'] = a['NFRAMES']
                struct['NREAD'] = a['NREAD']
                struct['EXPTIME'] = a['EXPTIME']
                struct['QRTZ'] = a['QRTZ']
                struct['UNE'] = a['UNE']
                struct['THAR'] = a['THAR']
                struct['EXPTYPE'] = a['EXPTYPE']
                struct['MEAN'] = a['MEAN']
                struct['SIG'] = a['SIG']

                if i == 0:
                    alldark = struct
                else:
                    alldark = np.concatenate([alldark, struct])

            hdulist = fits.open(outfile)
            hdu1 = fits.table_to_hdu(Table(struct))
            hdulist.append(hdu1)
            hdulist.writeto(outfile, overwrite=True)
            hdulist.close()
            print("----> monitor: Finished adding QAdarkflat info to " + os.path.basename(outfile))

        ###########################################################################################
        # MAKE MASTER EXP FILE
        outfile = specdir5 + 'monitor/' + instrument + 'Exp.fits'
        print("----> monitor: Making " + os.path.basename(outfile))

        # Get long term trends from dome flats
        # Append together the individual exp files
        files = glob.glob(specdir + '/exposures/' + instrument + '/*/*exp.fits')
        if len(files) < 1:
            print("----> monitor: No exp files!")
        else:
            files.sort()
            files = np.array(files)
            nfiles=len(files)
            for i in range(nfiles):
                print("---->    monitor: reading " + files[i])
                a = fits.open(files[i])[1].data

                # Make output structure.
                dt = np.dtype([('MJD',       np.int32),
                               ('DATEOBS',   np.str, 23),
                               ('JD',        np.float64),
                               ('NUM',       np.int32),
                               ('NFRAMES',   np.int32),
                               ('IMAGETYPE', np.str, 10),
                               ('PLATEID',   np.int32),
                               ('CARTID',    np.int32),
                               ('RA',        np.float64),
                               ('DEC',       np.float64),
                               ('SEEING',    np.float64),
                               ('ALT',       np.float64),
                               ('QRTZ',      np.int32),
                               ('THAR',      np.int32),
                               ('UNE',       np.int32),
                               ('FFS',       np.str, 15),
                               ('LN2LEVEL',  np.float64),
                               ('DITHPIX',   np.float64),
                               ('TRACEDIST', np.float64),
                               ('MED',       np.float64,(300,nchips))])
                struct = np.zeros(len(a['MJD']), dtype=dt)

                struct['MJD'] = a['MJD']
                struct['DATEOBS'] = a['DATEOBS']
                struct['JD'] = a['JD']
                struct['NUM'] = a['NUM']
                struct['NFRAMES'] = a['NFRAMES']
                struct['IMAGETYPE'] = a['IMAGETYPE']
                struct['PLATEID'] = a['PLATEID']
                struct['CARTID'] = a['CARTID']
                struct['RA'] = a['RA']
                struct['DEC'] = a['DEC']
                struct['SEEING'] = a['SEEING']
                struct['ALT'] = a['ALT']
                struct['QRTZ'] = a['QRTZ']
                struct['THAR'] = a['THAR']
                struct['UNE'] = a['UNE']
                struct['FFS'] = a['FFS']
                struct['LN2LEVEL'] = a['LN2LEVEL']
                struct['DITHPIX'] = a['DITHPIX']
                struct['TRACEDIST'] = a['TRACEDIST']
                struct['MED'] = a['MED']

                if i == 0:
                    allexp = struct 
                else:
                    allexp = np.concatenate([allexp, struct])

            Table(allcal).write(outfile, overwrite=True)
            print("----> monitor: Finished making " + os.path.basename(outfile))

        ###########################################################################################
        # MAKE MASTER apPlateSum FILE
        outfile = specdir5 + 'monitor/' + instrument + 'Sci.fits'
        print("----> monitor: Making " + os.path.basename(outfile))

        # Get zeropoint info from apPlateSum files
        files = glob.glob(specdir + '/visit/' + telescope + '/*/*/*/' + 'apPlateSum*.fits')
        if len(files) < 1:
            print("----> monitor: No apPlateSum files!")
        else:
            files.sort()
            files = np.array(files)
            nfiles = len(files)
            for i in range(nfiles):
                print("---->    monitor: reading " + files[i])
                a = fits.open(files[i])[1].data

                # Make output structure.
                dt = np.dtype([('TELESCOPE', np.str, 6),
                               ('PLATE',     np.int32),
                               ('NREADS',    np.int32),
                               ('DATEOBS',   np.str, 30),
                               ('EXPTIME',   np.int32),
                               ('SECZ',      np.float64),
                               ('HA',        np.float64),
                               ('DESIGN_HA', np.float64, 3),
                               ('SEEING',    np.float64),
                               ('FWHM',      np.float64),
                               ('GDRMS',     np.float64),
                               ('CART',      np.int32),
                               ('PLUGID',    np.str, 30),
                               ('DITHER',    np.float64),
                               ('MJD',       np.int32),
                               ('IM',        np.int32),
                               ('ZERO',      np.float64),
                               ('ZERORMS',   np.float64),
                               ('ZERONORM',  np.float64),
                               ('SKY',       np.float64, 3),
                               ('SN',        np.float64, 3),
                               ('SNC',       np.float64, 3),
                               #('SNT',       np.float64, 3),
                               ('ALTSN',     np.float64, 3),
                               ('NSN',       np.int32),
                               ('SNRATIO',   np.float64),
                               ('MOONDIST',  np.float64),
                               ('MOONPHASE', np.float64),
                               ('TELLFIT',   np.float64, (3,6))])
                struct = np.zeros(len(a['PLATE']), dtype=dt)

                struct['TELESCOPE'] = a['TELESCOPE']
                struct['PLATE'] = a['PLATE']
                struct['NREADS'] = a['NREADS']
                struct['DATEOBS'] = a['DATEOBS']
                struct['EXPTIME'] = a['EXPTIME']
                struct['SECZ'] = a['SECZ']
                struct['HA'] = a['HA']
                struct['DESIGN_HA'] = a['DESIGN_HA']
                struct['SEEING'] = a['SEEING']
                struct['FWHM'] = a['FWHM']
                struct['GDRMS'] = a['GDRMS']
                struct['CART'] = a['CART']
                struct['PLUGID'] = a['PLUGID']
                struct['DITHER'] = a['DITHER']
                struct['MJD'] = a['MJD']
                struct['IM'] = a['IM']
                struct['ZERO'] = a['ZERO']
                struct['ZERORMS'] = a['ZERORMS']
                struct['ZERONORM'] = a['ZERONORM']
                struct['SKY'] = a['SKY']
                struct['SN'] = a['SN']
                struct['SNC'] = a['SNC']
                struct['ALTSN'] = a['ALTSN']
                struct['NSN'] = a['NSN']
                struct['SNRATIO'] = a['SNRATIO']
                struct['MOONDIST'] = a['MOONDIST']
                struct['MOONPHASE'] = a['MOONPHASE']
                struct['TELLFIT'] = a['TELLFIT']

                if i == 0:
                    allsci = struct
                else:
                    allsci = np.concatenate([allsci, struct])

            Table(allsci).write(outfile, overwrite=True)
            print("----> monitor: Finished making " + os.path.basename(outfile))

        ###########################################################################################
        # MAKE MASTER TRACE FILE
        outfile = specdir5 + 'monitor/' + instrument + 'Trace.fits'
        print("----> monitor: Making " + os.path.basename(outfile))

        # Append together the individual QAcal files
        files = glob.glob(specdir + '/cal/psf/apEPSF-b-*.fits')
        if len(files) < 1:
            print("----> monitor: No apEPSF-b files!")
        else:
            files.sort()
            files = np.array(files)
            nfiles = len(files)

            dt = np.dtype([('NUM',      np.int32),
                           ('MJD',      np.float64),
                           ('CENT',     np.float64),
                           ('LN2LEVEL', np.int32)])
            struct = np.zeros(nfiles, dtype=dt)

            for i in range(nfiles):
                print("---->    monitor: reading " + files[i])
                a = fits.open(files[i])[1].data
                num = round(int(files[i].split('-b-')[1].split('.')[0]) / 10000)
                if num > 1000:
                    hdr = fits.getheader(files[i])
                    for j in range(147,156):
                        a = fits.open(files[i])[j].data
                        if a['FIBER'] == 150:
                            struct['NUM'][i] = round(int(files[i].split('-b-')[1].split('.')[0]))
                            struct['CENT'][i] = a['CENT'][1000]
                            struct['MJD'][i] = hdr['JD-MJD'] - 2400000.5
                            struct['LN2LEVEL'][i] = hdr['LN2LEVEL']
                            break

            Table(struct).write(outfile, overwrite=True)
            print("----> monitor: Finished making " + os.path.basename(outfile))

    ###############################################################################################
    # Find the different cals
    thar, = np.where(allcal['THAR'] == 1)
    une, = np.where(allcal['UNE'] == 1)
    qrtz, = np.where(allcal['QRTZ'] == 1)
    dome, = np.where(allexp['IMAGETYP'] == 'DomeFlat')
    dark, = np.where(alldark['EXPTYPE'] == 'DARK')

    ###############################################################################################
    # MAKE THE MONITOR HTML
    outfile = specdir5 + 'monitor/' + instrument + '-monitor.html'
    print("----> monitor: Making " + os.path.basename(outfile))

    html = open(outfile, 'w')
    tit = 'APOGEE-N Instrument Monitor'
    if instrument != 'apogee-n': tit = 'APOGEE-S Instrument Monitor'
    html.write('<HTML><HEAD><title>' + tit + '</title></head><BODY>\n')
    html.write('<H1>' + tit + '</H1>\n')
    html.write('<HR>\n')
    html.write('<ul>\n')
    html.write('<li> Throughput / lamp monitors\n')
    html.write('<ul>\n')
    html.write('<li> <a href=#quartz> Cal channel quartz</a>\n')
    html.write('<li> <a href=monitor/fiber/fiber.html>Individual fiber throughputs</a>\n')
    html.write('<li> <a href=#tharflux> Cal channel ThAr</a>\n')
    html.write('<li> <a href=#uneflux> Cal channel UNe</a>\n')
    html.write('<li> <a href=#dome>Dome flats</a>\n')
    html.write('<li> <a href=#zero>Plate zeropoints</a>\n')
    html.write('</ul>\n')
    html.write('<li> Positions\n')
    html.write('<ul>\n')
    html.write('<li> <a href=#tpos>ThAr line position</a>\n')
    html.write('</ul>\n')
    html.write('<li> Line widths\n')
    html.write('<ul>\n')
    html.write('<li> <a href=#tfwhm>ThAr line FWHM</a>\n')
    html.write('</ul>\n')
    html.write('<li> <a href=#trace>Trace locations</a>\n')
    html.write('<li> <a href=#detectors> Detectors\n')
    html.write('<li> <a href=#sky> Sky brightness\n')
    html.write('</ul>\n')
    html.write('<HR>\n')

    html.write('<h3> <a name=qflux></a> Quartz lamp median brightness (per 10 reads) in extracted frame </h3>\n')
    html.write('<A HREF=' + instrument + '/qflux.png target="_blank"><IMG SRC=' + instrument + '/qflux.png WIDTH=1200></A>\n')
    html.write('<HR>\n')

    html.write('<H3> <a href=fiber/fiber.html> Individual fiber throughputs from quartz </H3>\n')
    html.write('<HR>\n')

    html.write('<H3> <a name=tharflux></a>ThAr line brightness (per 10 reads) in extracted frame </H3>\n')
    html.write('<A HREF=' + instrument + '/tharflux.png target="_blank"><IMG SRC=' + instrument + '/tharflux.png WIDTH=1200></A>\n')
    html.write('<HR>\n')

    html.write('<H3> <a name=uneflux></a>UNe line brightness (per 10 reads) in extracted frame </H3>\n')
    html.write('<A HREF=' + instrument + '/uneflux.png target="_blank"><IMG SRC=' + instrument + '/uneflux.png WIDTH=1200></A>\n')
    html.write('<HR>\n')

    html.write('<H3> <a name=dome></a>Dome flat median brightness</H3>\n')
    html.write('<P> (Note: horizontal lines are the medians across all fibers) </P>\n')
    html.write('<A HREF=' + instrument + '/dome.png target="_blank"><IMG SRC=' + instrument + '/dome.png WIDTH=1200></A>\n')
    html.write('<HR>\n')

    html.write('<H3> <a name=zero></a>Science frame zero point</H3>\n')
    html.write('<A HREF=' + instrument + '/zero.png target="_blank"><IMG SRC=' + instrument + '/zero.png WIDTH=1200></A>\n')
    html.write('<HR>\n')

    html.write('<H3> <a name=tpos></a>ThArNe lamp line position</H3>\n')
    html.write('<A HREF=' + instrument + '/tpos.png target="_blank"><IMG SRC=' + instrument + '/tpos.png WIDTH=1200></A>\n')
    html.write('<HR>\n')

    for iline in range(2):
        plotfile='tfwhm' + str(iline) + '.png'
        tmp1 = str("%.1f" % round(allcal['LINES'][thar][0][iline][0],1))
        tmp2 = str("%.1f" % round(allcal['LINES'][thar][0][iline][1],1))
        tmp3 = str("%.1f" % round(allcal['LINES'][thar][0][iline][2],1))
        txt = '<a name=tfwhm></a> ThArNe lamp line FWHM, line position (x pixel): '
        html.write('<H3>' + txt + tmp1 + ' ' + tmp2 + ' ' + tmp3 + '</H3>\n')
        html.write('<P> (Note: horizontal lines are the median value for each fiber.) </P>\n')
        html.write('<A HREF=' + instrument + '/' + plotfile + ' target="_blank"><img src=' + instrument + '/' + plotfile + ' WIDTH=1200></A>\n')
    html.write('<HR>\n')

    html.write('<H3> <a name=trace></a> Trace position, fiber 150, column 1000</H3>\n')
    html.write('<A HREF=' + instrument + '/trace.png target="_blank"><IMG SRC=' + instrument + '/trace.png WIDTH=1200></A>\n')
    html.write('<HR>\n')

    html.write('<H3> <a name=detectors></a>Detectors</H3>\n')
    html.write('<H4> Dark Mean </h4>\n')
    html.write('<A HREF=' + instrument + '/biasmean.png target="_blank"><IMG SRC=' + instrument + '/biasmean.png WIDTH=1200></A>\n')

    html.write('<H4> Dark Sigma </h4>\n')
    html.write('<A HREF=' + instrument + '/biassig.png target="_blank"><IMG SRC=' + instrument + '/biassig.png WIDTH=1200></A>\n')

    html.write('<H3> <a name=sky></a>Sky Brightness</H3>\n')
    html.write('<A HREF=' + instrument + '/moonsky.png target="_blank"><IMG SRC=' + instrument + '/moonsky.png WIDTH=1300></A>\n')

    html.close()

    ###############################################################################################
    # Set up some basic plotting parameters
    matplotlib.use('agg')
    fontsize = 24;   fsz = fontsize * 0.75
    matplotlib.rcParams.update({'font.size':fontsize, 'font.family':'serif'})
    bboxpar = dict(facecolor='white', edgecolor='none', alpha=1.0)
    axwidth = 1.5
    axmajlen = 7
    axminlen = 3.5
    alf = 0.6
    markersz = 7
    colors = np.array(['crimson', 'limegreen', 'orange', 'violet', 'royalblue'])
    colors1 = np.array(['k', 'salmon', 'cornflowerblue'])
    fibers = np.array([10, 80, 150, 220, 290])
    nplotfibs = len(fibers)
    #years = np.array([2011, 2012, 2013, 2014

    tmp = allcal[qrtz]
    caljd = tmp['JD'] - 2.4e6
    t = Time(tmp['JD'], format='jd')
    years = np.unique(np.floor(t.byear)) + 1
    nyears = len(years)
    minjd = np.min(caljd)
    maxjd = np.max(caljd)
    jdspan = maxjd - minjd
    xmin = minjd - jdspan * 0.01
    xmax = maxjd + jdspan * 0.10


    if allplots is True:
        ###############################################################################################
        # qflux.png
        plotfile = specdir5 + 'monitor/' + instrument + '/qflux.png'
        if (os.path.exists(plotfile) == False) | (clobber == True):
            print("----> monitor: Making " + plotfile)

            fig = plt.figure(figsize=(30,14))
            ymax = 44000
            if instrument == 'apogee-s': 
                ymax = 100000
            ymin = 0 - ymax * 0.05
            yspan = ymax - ymin

            gdcal = allcal[qrtz]
            caljd = gdcal['JD'] - 2.4e6

            for ichip in range(nchips):
                chip = chips[ichip]

                ax = plt.subplot2grid((nchips,1), (ichip,0))
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
                ax.minorticks_on()
                ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
                ax.tick_params(axis='both',which='major',length=axmajlen)
                ax.tick_params(axis='both',which='minor',length=axminlen)
                ax.tick_params(axis='both',which='both',width=axwidth)
                if ichip == nchips-1: ax.set_xlabel(r'JD - 2,400,000')
                ax.set_ylabel(r'Median Flux')
                if ichip < nchips-1: ax.axes.xaxis.set_ticklabels([])

                for year in years:
                    t = Time(year, format='byear')
                    ax.axvline(x=t.jd-2.4e6, color='k', linestyle='dashed', alpha=alf)
                    if ichip == 0: ax.text(t.jd-2.4e6, ymax+yspan*0.025, str(int(round(year))), ha='center')

                for ifib in range(nplotfibs):
                    yvals = gdcal['FLUX'][:, ichip, fibers[ifib]]  / gdcal['NREAD']*10.0
                    ax.scatter(caljd, yvals, marker='o', s=markersz, c=colors[ifib], alpha=alf, label='Fiber ' + str(fibers[ifib]))

                ax.text(0.97,0.92,chip.capitalize() + '\n' + 'Chip', transform=ax.transAxes, ha='center', va='top', color=chip, bbox=bboxpar)
                if ichip == 0: ax.legend(loc='lower right', labelspacing=0.5, handletextpad=-0.1, markerscale=4, fontsize=fsz, edgecolor='k', framealpha=1)

            fig.subplots_adjust(left=0.06,right=0.995,bottom=0.06,top=0.96,hspace=0.08,wspace=0.00)
            plt.savefig(plotfile)
            plt.close('all')

        ###############################################################################################
        # tharflux.png
        plotfile = specdir5 + 'monitor/' + instrument + '/tharflux.png'
        if (os.path.exists(plotfile) == False) | (clobber == True):
            print("----> monitor: Making " + plotfile)

            fig = plt.figure(figsize=(30,14))
            ymax = np.array([510000, 58000, 11000]) 
            ymin = 0 - ymax * 0.05
            yspan = ymax - ymin

            gdcal = allcal[thar]
            caljd = gdcal['JD']-2.4e6
            flux = gdcal['GAUSS'][:,:,:,:,0] * gdcal['GAUSS'][:,:,:,:,2]**2

            for ichip in range(nchips):
                chip = chips[ichip]

                ax = plt.subplot2grid((nchips,1), (ichip,0))
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin[ichip], ymax[ichip])
                ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
                ax.minorticks_on()
                ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
                ax.tick_params(axis='both',which='major',length=axmajlen)
                ax.tick_params(axis='both',which='minor',length=axminlen)
                ax.tick_params(axis='both',which='both',width=axwidth)
                if ichip == nchips-1: ax.set_xlabel(r'JD - 2,400,000')
                ax.set_ylabel(r'Line Flux')
                if ichip < nchips-1: ax.axes.xaxis.set_ticklabels([])

                for year in years:
                    t = Time(year, format='byear')
                    ax.axvline(x=t.jd-2.4e6, color='k', linestyle='dashed', alpha=alf)
                    if ichip == 0: ax.text(t.jd-2.4e6, ymax[ichip]+yspan[ichip]*0.025, str(int(round(year))), ha='center')

                for ifib in range(nplotfibs):
                    yvals = flux[:, 0, ichip, ifib] / gdcal['NREAD']*10.0
                    ax.scatter(caljd, yvals, marker='o', s=markersz, c=colors[ifib], alpha=alf, label='Fiber ' + str(fibers[ifib]))

                ax.text(0.97,0.92,chip.capitalize() + '\n' + 'Chip', transform=ax.transAxes, ha='center', va='top', color=chip, bbox=bboxpar)
                if ichip == 0: ax.legend(loc='lower right', labelspacing=0.5, handletextpad=-0.1, markerscale=4, fontsize=fsz, edgecolor='k', framealpha=1)

            fig.subplots_adjust(left=0.06,right=0.995,bottom=0.06,top=0.96,hspace=0.08,wspace=0.00)
            plt.savefig(plotfile)
            plt.close('all')

        ###############################################################################################
        # uneflux.png
        plotfile = specdir5 + 'monitor/' + instrument + '/uneflux.png'
        if (os.path.exists(plotfile) == False) | (clobber == True):
            print("----> monitor: Making " + plotfile)

            fig = plt.figure(figsize=(30,14))
            ymax = np.array([40000, 3000, 7700])
            ymin = 0 - ymax*0.05
            yspan = ymax - ymin

            gdcal = allcal[une]
            caljd = gdcal['JD'] - 2.4e6
            flux = gdcal['GAUSS'][:,:,:,:,0] * gdcal['GAUSS'][:,:,:,:,2]**2

            for ichip in range(nchips):
                chip = chips[ichip]

                ax = plt.subplot2grid((nchips,1), (ichip,0))
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin[ichip], ymax[ichip])
                ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
                ax.minorticks_on()
                ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
                ax.tick_params(axis='both',which='major',length=axmajlen)
                ax.tick_params(axis='both',which='minor',length=axminlen)
                ax.tick_params(axis='both',which='both',width=axwidth)
                if ichip == nchips-1: ax.set_xlabel(r'JD - 2,400,000')
                ax.set_ylabel(r'Line Flux')
                if ichip < nchips-1: ax.axes.xaxis.set_ticklabels([])

                for year in years:
                    t = Time(year, format='byear')
                    ax.axvline(x=t.jd-2.4e6, color='k', linestyle='dashed', alpha=alf)
                    if ichip == 0: ax.text(t.jd-2.4e6, ymax[ichip]+yspan[ichip]*0.025, str(int(round(year))), ha='center')

                for ifib in range(nplotfibs):
                    yvals = flux[:, 0, ichip, ifib] / gdcal['NREAD']*10.0
                    ax.scatter(caljd, yvals, marker='o', s=markersz, c=colors[ifib], alpha=alf, label='Fiber ' + str(fibers[ifib]))

                ax.text(0.97,0.92,chip.capitalize() + '\n' + 'Chip', transform=ax.transAxes, ha='center', va='top', color=chip, bbox=bboxpar)
                if ichip == 0: ax.legend(loc='lower right', labelspacing=0.5, handletextpad=-0.1, markerscale=4, fontsize=fsz, edgecolor='k', framealpha=1)

            fig.subplots_adjust(left=0.06,right=0.995,bottom=0.06,top=0.96,hspace=0.08,wspace=0.00)
            plt.savefig(plotfile)
            plt.close('all')

        ###############################################################################################
        # dome.png
        plotfile = specdir5 + 'monitor/' + instrument + '/dome.png'
        if (os.path.exists(plotfile) == False) | (clobber == True):
            print("----> monitor: Making " + plotfile)

            fig = plt.figure(figsize=(30,14))
            ymax = 13000
            ymin = 0 - ymax*0.05
            yspan = ymax - ymin

            gdcal = allexp[dome]
            caljd = gdcal['JD'] - 2.4e6

            for ichip in range(nchips):
                chip = chips[ichip]

                ax = plt.subplot2grid((nchips,1), (ichip,0))
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
                ax.minorticks_on()
                ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
                ax.tick_params(axis='both',which='major',length=axmajlen)
                ax.tick_params(axis='both',which='minor',length=axminlen)
                ax.tick_params(axis='both',which='both',width=axwidth)
                if ichip == nchips-1: ax.set_xlabel(r'JD - 2,400,000')
                ax.set_ylabel(r'Median Flux')
                if ichip < nchips-1: ax.axes.xaxis.set_ticklabels([])

                for year in years:
                    t = Time(year, format='byear')
                    ax.axvline(x=t.jd-2.4e6, color='k', linestyle='dashed', alpha=alf)
                    if ichip == 0: ax.text(t.jd-2.4e6, ymax+yspan*0.025, str(int(round(year))), ha='center')

                w = np.nanmedian(gdcal['MED'][:, ichip, :])
                ax.axhline(y=w, color='k', linewidth=3, zorder=1)

                for ifib in range(nplotfibs):
                    yvals = gdcal['MED'][:, ichip, fibers[ifib]]
                    ax.scatter(caljd, yvals, marker='o', s=markersz, c=colors[ifib], alpha=alf, label='Fiber ' + str(fibers[ifib]), zorder=3)

                ax.text(0.97,0.92,chip.capitalize() + '\n' + 'Chip', transform=ax.transAxes, ha='center', va='top', color=chip, bbox=bboxpar)
                if ichip == 0: ax.legend(loc='lower right', labelspacing=0.5, handletextpad=-0.1, markerscale=4, fontsize=fsz, edgecolor='k', framealpha=1)

            fig.subplots_adjust(left=0.06,right=0.995,bottom=0.06,top=0.96,hspace=0.08,wspace=0.00)
            plt.savefig(plotfile)
            plt.close('all')

        ###############################################################################################
        # zero.png
        plotfile = specdir5 + 'monitor/' + instrument + '/zero.png'
        if (os.path.exists(plotfile) == False) | (clobber == True):
            print("----> monitor: Making " + plotfile)

            fig = plt.figure(figsize=(30,8))
            ymax = 21
            ymin = 10
            yspan = ymax - ymin

            ax = plt.subplot2grid((1,1), (0,0))
            ax.set_xlim(xmin, xmax)
            ax.set_ylim(ymin, ymax)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
            ax.minorticks_on()
            ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
            ax.tick_params(axis='both',which='major',length=axmajlen)
            ax.tick_params(axis='both',which='minor',length=axminlen)
            ax.tick_params(axis='both',which='both',width=axwidth)
            if ichip == nchips-1: ax.set_xlabel(r'JD - 2,400,000')
            ax.set_ylabel(r'Zeropoint (mag.)')
            if ichip < nchips-1: ax.axes.xaxis.set_ticklabels([])

            for year in years:
                t = Time(year, format='byear')
                ax.axvline(x=t.jd-2.4e6, color='k', linestyle='dashed', alpha=alf)
                ax.text(t.jd-2.4e6, ymax+yspan*0.02, str(int(round(year))), ha='center')

            t = Time(allsci['DATEOBS'], format='fits')
            jd = t.jd - 2.4e6
            ax.scatter(jd, allsci['ZERO'], marker='o', s=markersz, c='teal', alpha=alf)

            fig.subplots_adjust(left=0.04,right=0.99,bottom=0.115,top=0.94,hspace=0.08,wspace=0.00)
            plt.savefig(plotfile)
            plt.close('all')

        ###############################################################################################
        # tpos.png
        plotfile = specdir5 + 'monitor/' + instrument + '/tpos.png'
        if (os.path.exists(plotfile) == False) | (clobber == True):
            print("----> monitor: Making " + plotfile)

            fig = plt.figure(figsize=(30,14))

            gdcal = allcal[thar]
            caljd = gdcal['JD']-2.4e6

            for ichip in range(nchips):
                chip = chips[ichip]

                ax = plt.subplot2grid((nchips,1), (ichip,0))
                ax.set_xlim(xmin, xmax)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
                ax.minorticks_on()
                ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
                ax.tick_params(axis='both',which='major',length=axmajlen)
                ax.tick_params(axis='both',which='minor',length=axminlen)
                ax.tick_params(axis='both',which='both',width=axwidth)
                if ichip == nchips-1: ax.set_xlabel(r'JD - 2,400,000')
                ax.set_ylabel(r'Median Flux')
                if ichip < nchips-1: ax.axes.xaxis.set_ticklabels([])

                w = np.nanmedian(gdcal['GAUSS'][:, 0, ichip, :, 1])
                ymin = w - 10
                ymax = w + 10
                yspan = ymax - ymin
                ax.set_ylim(ymin, ymax)

                for year in years:
                    t = Time(year, format='byear')
                    ax.axvline(x=t.jd-2.4e6, color='k', linestyle='dashed', alpha=alf)
                    if ichip == 0: ax.text(t.jd-2.4e6, ymax+yspan*0.025, str(int(round(year))), ha='center')

                for ifib in range(nplotfibs):
                    yvals = gdcal['GAUSS'][:, 0, ichip, ifib, 1] 
                    ax.scatter(caljd, yvals, marker='o', s=markersz, c=colors[ifib], alpha=alf, label='Fiber ' + str(fibers[ifib]))

                ax.text(0.97,0.92,chip.capitalize() + '\n' + 'Chip', transform=ax.transAxes, ha='center', va='top', color=chip, bbox=bboxpar)
                if ichip == 0: ax.legend(loc='lower right', labelspacing=0.5, handletextpad=-0.1, markerscale=4, fontsize=fsz, edgecolor='k', framealpha=1)

            fig.subplots_adjust(left=0.06,right=0.995,bottom=0.06,top=0.96,hspace=0.08,wspace=0.00)
            plt.savefig(plotfile)
            plt.close('all')

        ###############################################################################################
        # ThArNe lamp line FWHM
        for iline in range(2):
            plotfile = specdir5 + 'monitor/' + instrument + '/tfwhm' + str(iline) + '.png'
            if (os.path.exists(plotfile) == False) | (clobber == True):
                print("----> monitor: Making " + plotfile)

                fig = plt.figure(figsize=(30,14))

                gdcal = allcal[thar]
                caljd = gdcal['JD']-2.4e6

                for ichip in range(nchips):
                    chip = chips[ichip]

                    ax = plt.subplot2grid((nchips,1), (ichip,0))
                    ax.set_xlim(xmin, xmax)
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
                    ax.minorticks_on()
                    ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
                    ax.tick_params(axis='both',which='major',length=axmajlen)
                    ax.tick_params(axis='both',which='minor',length=axminlen)
                    ax.tick_params(axis='both',which='both',width=axwidth)
                    if ichip == nchips-1: ax.set_xlabel(r'JD - 2,400,000')
                    ax.set_ylabel(r'FWHM ($\rm \AA$)')
                    if ichip < nchips-1: ax.axes.xaxis.set_ticklabels([])

                    w = np.nanmedian(2.0 * np.sqrt(2 * np.log(2)) * gdcal['GAUSS'][:, iline, ichip, :, 2])
                    ymin = w * 0.8
                    ymax = w * 1.25
                    yspan = ymax - ymin
                    ax.set_ylim(ymin, ymax)

                    for year in years:
                        t = Time(year, format='byear')
                        ax.axvline(x=t.jd-2.4e6, color='k', linestyle='dashed', alpha=alf)
                        if ichip == 0: ax.text(t.jd-2.4e6, ymax+yspan*0.025, str(int(round(year))), ha='center')

                    for ifib in range(nplotfibs):
                        w = np.nanmedian(2.0 * np.sqrt(2 * np.log(2)) * gdcal['GAUSS'][:, iline, ichip, ifib, 2])
                        ax.axhline(y=w, color=colors[ifib], linewidth=2, zorder=2)
                        ax.axhline(y=w, color='k', linewidth=3, zorder=1)
                        yvals = 2.0 * np.sqrt(2 * np.log(2)) * gdcal['GAUSS'][:, iline, ichip, ifib, 2]
                        ax.scatter(caljd, yvals, marker='o', s=markersz, c=colors[ifib], alpha=alf, label='Fiber ' + str(fibers[ifib]), zorder=3)

                    ax.text(0.97,0.92,chip.capitalize() + '\n' + 'Chip', transform=ax.transAxes, ha='center', va='top', color=chip, bbox=bboxpar)
                    if ichip == 0: ax.legend(loc='lower right', labelspacing=0.5, handletextpad=-0.1, markerscale=4, fontsize=fsz, edgecolor='k', framealpha=1)

                fig.subplots_adjust(left=0.06,right=0.995,bottom=0.06,top=0.96,hspace=0.08,wspace=0.00)
                plt.savefig(plotfile)
                plt.close('all')

        ###############################################################################################
        # trace.png
        plotfile = specdir5 + 'monitor/' + instrument + '/trace.png'
        if (os.path.exists(plotfile) == False) | (clobber == True):
            print("----> monitor: Making " + plotfile)

            fig = plt.figure(figsize=(30,12))
            ymax = np.nanmedian(allepsf['CENT']) + 1
            ymin = np.nanmedian(allepsf['CENT']) - 1
            yspan = ymax - ymin

            caljd = Time(allepsf['MJD'], format='mjd').jd - 2.4e6

            ax1 = plt.subplot2grid((2,1), (0,0))
            ax2 = plt.subplot2grid((2,1), (1,0))
            axes = [ax1, ax2]

            ax1.xaxis.set_major_locator(ticker.MultipleLocator(500))
            ax1.set_xlim(xmin, xmax)
            ax1.set_xlabel(r'JD - 2,400,000')
            ax2.set_xlabel(r'LN2 Level')
            for ax in axes:
                ax.set_ylim(ymin, ymax)
                ax.minorticks_on()
                ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
                ax.tick_params(axis='both',which='major',length=axmajlen)
                ax.tick_params(axis='both',which='minor',length=axminlen)
                ax.tick_params(axis='both',which='both',width=axwidth)
                ax.set_ylabel(r'Trace Center')

            for year in years:
                t = Time(year, format='byear')
                ax1.axvline(x=t.jd-2.4e6, color='k', linestyle='dashed', alpha=alf)
                ax1.text(t.jd-2.4e6, ymax+yspan*0.025, str(int(round(year))), ha='center')

            ax1.scatter(caljd, allepsf['CENT'], marker='o', s=markersz*4, c='cyan', edgecolors='k', alpha=alf)
            ax2.scatter(allepsf['LN2LEVEL'], allepsf['CENT'], marker='o', s=markersz*4, c='cyan', edgecolors='k', alpha=alf)

            fig.subplots_adjust(left=0.06,right=0.995,bottom=0.07,top=0.96,hspace=0.17,wspace=0.00)
            plt.savefig(plotfile)
            plt.close('all')

        ###############################################################################################
        # biasmean.png
        plotfile = specdir5 + 'monitor/' + instrument + '/biasmean.png'
        if (os.path.exists(plotfile) == False) | (clobber == True):
            print("----> monitor: Making " + plotfile)

            fig = plt.figure(figsize=(30,14))
            ymax = 1000.0
            ymin = 0.1
            yspan = ymax - ymin

            gdcal = alldark[dark]
            caljd = gdcal['JD'] - 2.4e6

            for ichip in range(nchips):
                chip = chips[ichip]

                ax = plt.subplot2grid((nchips,1), (ichip,0))
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
                ax.minorticks_on()
                ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
                ax.tick_params(axis='both',which='major',length=axmajlen)
                ax.tick_params(axis='both',which='minor',length=axminlen)
                ax.tick_params(axis='both',which='both',width=axwidth)
                if ichip == nchips-1: ax.set_xlabel(r'JD - 2,400,000')
                if ichip == 1: ax.set_ylabel(r'Mean (column median)')
                if ichip < nchips-1: ax.axes.xaxis.set_ticklabels([])

                for year in years:
                    t = Time(year, format='byear')
                    ax.axvline(x=t.jd-2.4e6, color='k', linestyle='dashed', alpha=alf)
                    if ichip == 0: ax.text(t.jd-2.4e6, ymax+yspan*0.25, str(int(round(year))), ha='center')

                for ibias in range(3):
                    ax.semilogy(caljd, gdcal['MEAN'][:, ibias, ichip], marker='o', ms=3, alpha=alf, mec=colors1[ibias], mfc=colors1[ibias], linestyle='')

                ax.text(0.97,0.92,chip.capitalize() + '\n' + 'Chip', transform=ax.transAxes, ha='center', va='top', color=chip, bbox=bboxpar)
                #if ichip == 0: ax.legend(loc='lower right', labelspacing=0.5, handletextpad=-0.1, markerscale=4, fontsize=fsz, edgecolor='k', framealpha=1)

            fig.subplots_adjust(left=0.06,right=0.995,bottom=0.06,top=0.96,hspace=0.08,wspace=0.00)
            plt.savefig(plotfile)
            plt.close('all')

        ###############################################################################################
        # biassig.png
        plotfile = specdir5 + 'monitor/' + instrument + '/biassig.png'
        if (os.path.exists(plotfile) == False) | (clobber == True):
            print("----> monitor: Making " + plotfile)

            fig = plt.figure(figsize=(30,14))
            ymax = 1000.0
            ymin = 0.1
            yspan = ymax - ymin

            gdcal = alldark[dark]
            caljd = gdcal['JD'] - 2.4e6

            for ichip in range(nchips):
                chip = chips[ichip]

                ax = plt.subplot2grid((nchips,1), (ichip,0))
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
                ax.minorticks_on()
                ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
                ax.tick_params(axis='both',which='major',length=axmajlen)
                ax.tick_params(axis='both',which='minor',length=axminlen)
                ax.tick_params(axis='both',which='both',width=axwidth)
                if ichip == nchips-1: ax.set_xlabel(r'JD - 2,400,000')
                if ichip == 1: ax.set_ylabel(r'$\sigma$ (column median)')
                if ichip < nchips-1: ax.axes.xaxis.set_ticklabels([])

                for year in years:
                    t = Time(year, format='byear')
                    ax.axvline(x=t.jd-2.4e6, color='k', linestyle='dashed', alpha=alf)
                    if ichip == 0: ax.text(t.jd-2.4e6, ymax+yspan*0.25, str(int(round(year))), ha='center')

                for ibias in range(3):
                    ax.semilogy(caljd, gdcal['SIG'][:, ibias, ichip], marker='o', ms=3, alpha=alf, mec=colors1[ibias], mfc=colors1[ibias], linestyle='')

                ax.text(0.97,0.92,chip.capitalize() + '\n' + 'Chip', transform=ax.transAxes, ha='center', va='top', color=chip, bbox=bboxpar)
                #if ichip == 0: ax.legend(loc='lower right', labelspacing=0.5, handletextpad=-0.1, markerscale=4, fontsize=fsz, edgecolor='k', framealpha=1)

            fig.subplots_adjust(left=0.06,right=0.995,bottom=0.06,top=0.96,hspace=0.08,wspace=0.00)
            plt.savefig(plotfile)
            plt.close('all')

    ###############################################################################################
    # moonsky.png
    plotfile = specdir5 + 'monitor/' + instrument + '/moonsky.png'
    if (os.path.exists(plotfile) == False) | (clobber == True):
        print("----> monitor: Making " + plotfile)

        fig = plt.figure(figsize=(31,12))
        ymax = 11
        ymin = 16.8
        yspan = ymax - ymin

        ax1 = plt.subplot2grid((2,1), (0,0))
        ax2 = plt.subplot2grid((2,1), (1,0))
        axes = [ax1, ax2]

        ax1.axes.xaxis.set_ticklabels([])
        ax2.set_xlabel(r'Moon Phase')

        for ax in axes:
            ax.set_xlim(0, 1)
            ax.set_ylim(ymin, ymax)
            ax.minorticks_on()
            ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
            ax.tick_params(axis='both',which='major',length=axmajlen)
            ax.tick_params(axis='both',which='minor',length=axminlen)
            ax.tick_params(axis='both',which='both',width=axwidth)
            ax.text(-0.03, 0.5, 'Sky Brightness', ha='right', va='center', rotation=90, transform=ax.transAxes)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))

        sc1 = ax1.scatter(allsci['MOONPHASE'], allsci['SKY'][:, 1], marker='o', s=markersz, c=allsci['MOONDIST'], cmap='rainbow', alpha=0.8, vmin=0, vmax=90.01)
        gd, = np.where((allsci['ZERO'] != -np.inf) & (allsci['ZERO'] < 20) & (allsci['ZERO'] > 0))
        sc2 = ax2.scatter(allsci['MOONPHASE'][gd], allsci['SKY'][gd][:, 1], marker='o', s=markersz, c=allsci['ZERO'][gd], cmap='rainbow', alpha=0.8, vmin=17, vmax=19)

        ax_divider = make_axes_locatable(ax1)
        cax = ax_divider.append_axes("right", size="2%", pad="1%")
        cb1 = colorbar(sc1, cax=cax, orientation="vertical")
        cax.minorticks_on()
        cax.yaxis.set_major_locator(ticker.MultipleLocator(15))
        ax1.text(1.06, 0.5, r'Moon Distance (deg.)',ha='left', va='center', rotation=-90, transform=ax1.transAxes)

        ax_divider = make_axes_locatable(ax2)
        cax = ax_divider.append_axes("right", size="2%", pad="1%")
        cb1 = colorbar(sc2, cax=cax, orientation="vertical")
        cax.minorticks_on()
        cax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax2.text(1.066, 0.5, r'Zeropoint / Cloudiness',ha='left', va='center', rotation=-90, transform=ax2.transAxes)

        fig.subplots_adjust(left=0.045,right=0.945,bottom=0.07,top=0.96,hspace=0.17,wspace=0.00)
        plt.savefig(plotfile)
        plt.close('all')



    print("----> monitor done")










