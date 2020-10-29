import sys
import glob
import os
import subprocess
import math
import numpy as np
from pathlib import Path
from astropy.io import fits, ascii
from astropy.table import Table
from astropy.time import Time
from numpy.lib.recfunctions import append_fields, merge_arrays
from astroplan import moon_illumination
from astropy.coordinates import SkyCoord, get_moon
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
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar

sdss_path = path.Path()

sort_table_link = 'https://www.kryogenix.org/code/browser/sorttable/sorttable.js'

# put import pdb; pdb.set_trace() wherever you want stop

#sdss_path.full('ap2D',apred=self.apred,telescope=self.telescope,instrument=self.instrument,
#                        plate=self.plate,mjd=self.mjd,prefix=self.prefix,num=0,chip='a')

# Plugmap for plate 8100 mjd 57680
# /uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/mapper/apo/57679/plPlugMapM-8100-57679-01.par
# apPlateSum
# /uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/apogee/spectro/redux/t14/visit/apo25m/200+45/8100/57680/apPlateSum-8100-57680.fits

# Planfile for plate 8100 mjd 57680
# https://data.sdss.org/sas/sdss5/mwm/apogee/spectro/redux/t14/visit/apo25m/200+45/8100/57680/apPlan-8100-57680.par

#------------------------------------------------------------------------------------------------------------------------
# APQA
#
#  call routines to make "QA" plots and web pages for a plate/MJD
#  for calibration frames, measures some features and makes a apQAcal file
#    with some summary information about the calibration data
#--------------------------------------------------------------------------------------------------


'''-----------------------------------------------------------------------------------------'''
'''APQA: Wrapper for running QA subprocedures                                               '''
'''-----------------------------------------------------------------------------------------'''
#def apqa(field='200+45', plate='8100', mjd='57680', telescope='apo25m', apred='t14',

def apqa(field='RM_XMM-LSS', plate='15000', mjd='59146', telescope='apo25m', apred='daily', 
         makeplots=True, overwritePlateSum=True, makeSpectrumPlots=True):

    print("---------------------------------------\nStarting APQA\n")

    # Use telescope, plate, mjd, and apred to load planfile into structure.
    load = apload.ApLoad(apred=apred, telescope=telescope)
    planfile = load.filename('Plan', plate=int(plate), mjd=mjd)
    planstr = plan.load(planfile, np=True)

    # Get values from plan file.
    fixfiberid = planstr['fixfiberid']
    badfiberid = planstr['badfiberid']
    platetype =  planstr['platetype']
    plugmap =    planstr['plugmap']
    fluxid =     planstr['fluxid']
    instrument = planstr['instrument']
    survey =     planstr['survey']
    print("Survey = "+survey)

    # Establish directories.
    datadir = {'apo25m':os.environ['APOGEE_DATA_N'],'apo1m':os.environ['APOGEE_DATA_N'],
               'lco25m':os.environ['APOGEE_DATA_S']}[telescope]

    apodir =     os.environ.get('APOGEE_REDUX')+'/'
    spectrodir = apodir+apred+'/'
    caldir =     spectrodir+'cal/'
    expdir =     spectrodir+'exposures/'+instrument+'/'

    # Get array of object exposures and find out how many are objects.
    flavor = planstr['APEXP']['flavor']
    all_ims = planstr['APEXP']['name']

    gd,= np.where(flavor == 'object')
    n_ims = len(gd)

    if n_ims > 0:
        ims = all_ims[gd]
    else:
        print("No object images. You are hosed. Give up hope.")
        ims = None

    # Get mapper data.
    mapper_data = {'apogee-n':os.environ['MAPPER_DATA_N'],'apogee-s':os.environ['MAPPER_DATA_S']}[instrument]

    # For calibration plates, measure lamp brightesses and/or line widths, etc. and write to FITS file.
    if platetype == 'cal': x = makeCalFits(load=load, ims=all_ims, mjd=mjd, instrument=instrument)

    # For darks and flats, get mean and stdev of column-medianed quadrants.
    if platetype == 'dark': x = makeDarkFits(load=load, planfile=planfile, ims=all_ims, mjd=mjd)

    # Normal plates:.
    if platetype == 'normal': 
        platesum = load.filename('PlateSum', plate=int(plate), mjd=mjd)
        if (os.path.exists(platesum) is False) | (overwritePlateSum is True):
            # Make the apPlateSum file if it doesn't already exist.
            q = makePlateSum(load=load, telescope=telescope, ims=ims, plate=plate, mjd=mjd,
                             field=field, instrument=instrument, clobber=True, plugmap=plugmap,
                             survey=survey, mapper_data=mapper_data, apred=apred, onem=None,
                             starfiber=None, starnames=None, starmag=None,flat=None,
                             fixfiberid=fixfiberid, badfiberid=badfiberid)

            ims = np.array([0,ims[0]])
            q = makePlateSum(load=load, telescope=telescope, ims=ims, plate=plate, mjd=mjd,
                             field=field, instrument=instrument, clobber=True, plugmap=plugmap,
                             survey=survey, mapper_data=mapper_data, apred=apred, onem=None,
                             starfiber=None, starnames=None, starmag=None,flat=None,
                             fixfiberid=fixfiberid, badfiberid=badfiberid)

        # Make the master QA page
        q = masterQAhtml(load=load, plate=plate, mjd=mjd, field=field, fluxid=fluxid, telescope=telescope)

        # Make plots for the master QA page
        if makeplots is True:
            q = masterQAplots(load=load, ims=ims, plate=plate, mjd=mjd, instrument=instrument, 
                              survey=survey, apred=apred, flat=None, fluxid=fluxid)

        # Make the spectrum plots and associated page
        q= makeObjHtml(load=load, plate=plate, mjd=mjd, survey=survey, makeSpectrumPlots=makeSpectrumPlots)

#        platesumfile = load.filename('PlateSum', plate=int(plate), mjd=mjd, chips=True)

        ### NOTE:No python translation for sntab.
#;        sntab,tabs=platefile,outfile=platefile+'.dat'

    # ASDAF and NMSU 1m observations:
#    if platetype == 'single':
#        single = [planstr['APEXP'][i]['single'].astype(int) for i in range(n_ims)]
#        sname = [planstr['APEXP'][i]['singlename'] for i in range(n_ims)]
#        smag = planstr['hmag']
#        x = makePlotsHtml(load=load, telescope=telescope, onem=True, ims=ims, starnames=sname, 
#                          starfiber=single, starmag=smag, fixfiberid=fixfiberid, clobber=True, 
#                          plugmap=plugmap, makeplots=makeplots, badfiberid=badfiberid, survey=survey, apred=apred)

    print("\nDone with APQA")

'''-----------------------------------------------------------------------------------------'''
''' MAKEPLATESUM: Plotmag translation                                                       '''
'''-----------------------------------------------------------------------------------------'''
def makePlateSum(load=None, telescope=None, ims=None, plate=None, mjd=None, field=None, 
                 instrument=None, clobber=True, makeplots=None, plugmap=None, survey=None,
                 mapper_data=None, apred=None, onem=None, starfiber=None, starnames=None, 
                 starmag=None, flat=None, fixfiberid=None, badfiberid=None): 

    platesumfile = load.filename('PlateSum', plate=int(plate), mjd=mjd)
    platesumbase = os.path.basename(platesumfile)

    print("Making "+platesumbase+" ...\n")

    n_exposures = len(ims)
    chips = np.array(['a','b','c'])
    nchips = len(chips)

    # Get the fiber association for this plate. Also get some other values
    onedfile = load.filename('1D',  plate=int(plate), num=ims[1], mjd=mjd, chips=True)
    tothdr = fits.getheader(onedfile.replace('1D-','1D-a-'))
    ra = tothdr['RADEG']
    dec = tothdr['DECDEG']
    DateObs = tothdr['DATE-OBS']

    if ims[0] == 0: 
        tot = load.apPlate(int(plate), mjd)
        n_exposures = 1
    else: 
        tot = load.ap1D(ims[0])

    if type(tot) != dict:
        html.write('<FONT COLOR=red> PROBLEM/FAILURE WITH: '+str(ims[0])+'\n')
        htmlsum.write('<FONT COLOR=red> PROBLEM/FAILURE WITH: '+str(ims[0])+'\n')
        html.close()
        htmlsum.close()
        print("Error in makePlotsHtml!!!")

    plug = platedata.getdata(int(plate), int(mjd), apred, telescope, plugid=plugmap, badfiberid=badfiberid) 

    gd, = np.where(plug['fiberdata']['fiberid'] > 0)
    fiber = plug['fiberdata'][gd]
    nfiber = len(fiber)
    rows = 300 - fiber['fiberid']
    guide = plug['guidedata']

    # Add sn and obsmag columns to fiber structure
    dtype =        np.dtype([('sn', np.float64, (nfiber, n_exposures,3))])
    snColumn =     np.zeros(nfiber, dtype=[('sn', 'float32', (n_exposures, nchips))])
    obsmagColumn = np.zeros(nfiber, dtype=[('obsmag', 'float32', (n_exposures, nchips))])
    fiber =        merge_arrays([fiber, snColumn, obsmagColumn], flatten=True)

    unplugged, = np.where(fiber['fiberid'] < 0)
    nunplugged = len(unplugged)
    if flat is not None:
        fiber['hmag'] = 12
        fiber['object'] = 'FLAT'

    # Find telluric, object, sky, and non-sky fibers.
    fibtype = fiber['objtype']
    fibertelluric, = np.where((fibtype == 'SPECTROPHOTO_STD') | (fibtype == 'HOT_STD'))
    ntelluric = len(fibertelluric)
    telluric = rows[fibertelluric]

    if ntelluric < 1: print("PROBLEM!!! No tellurics found.")

    fiberobj, = np.where((fibtype == 'STAR_BHB') | (fibtype == 'STAR') | (fibtype == 'EXTOBJ') | (fibtype == 'OBJECT'))
    nobj = len(fiberobj)
    obj = rows[fiberobj]

    if nobj < 1: print("PROBLEM!!! No science objects found.")

    fibersky, = np.where(fibtype == 'SKY')
    nsky = len(fibersky)
    sky = rows[fibersky]

    if nsky < 1: print("PROBLEM!!! No skies found.")

    fiberstar = np.concatenate([fiberobj,fibertelluric])
    nstar = len(fiberstar)
    star = rows[fiberstar]

    # Loop through all the images for this plate, and make the plots.
    # Load up and save information for this plate in a FITS table.
    allsky =     np.zeros((n_exposures,3), dtype=np.float64)
    allzero =    np.zeros((n_exposures,3), dtype=np.float64)
    allzerorms = np.zeros((n_exposures,3), dtype=np.float64)

    # Get guider information.
    if onem is None:
        gcamdir = os.environ.get('APOGEE_REDUX')+'/'+apred+'/'+'exposures/'+instrument+'/'+mjd+'/'
        if os.path.exists(gcamdir) is False: subprocess.call(['mkdir',gcamdir])
        gcamfile = gcamdir+'gcam-'+mjd+'.fits'
        if os.path.exists(gcamfile) is False:
            print("  Attempting to make "+os.path.basename(gcamfile)+" ...")
            subprocess.call(['gcam_process', '--mjd', mjd, '--instrument', instrument, '--output', gcamfile], shell=False)
            if os.path.exists(gcamfile):
                print("  Successfully made "+os.path.basename(gcamfile))
            else:
                print("  ... failed to make "+os.path.basename(gcamfile)+"  :(")
        else:
            gcam = fits.getdata(gcamfile)

    mjd0 = 99999
    mjd1 = 0.

    # FITS table structure.
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
                   ('ALTSN',     np.float64, 3),
                   ('NSN',       np.int32),
                   ('SNRATIO',   np.float64),
                   ('MOONDIST',  np.float64),
                   ('MOONPHASE', np.float64),
                   ('TELLFIT',   np.float64, (3,6))])

    platetab = np.zeros(n_exposures,dtype=dt)

    platetab['PLATE'] =     int(plate)
    platetab['TELESCOPE'] = telescope
    platetab['HA'] =        0.0
    platetab['DESIGN_HA'] = -99.0
    platetab['PLUGID'] =    plugmap
    platetab['MJD'] =       mjd
    #platetab['MOONDIST'] =  moondist
    #platetab['MOONPHASE'] = moonphase

    # Loop over the exposures.
    for i in range(n_exposures):
        if ims[0] == 0: 
            pfile = os.path.basename(load.filename('Plate', plate=int(plate), mjd=mjd, chips=True))
            dfile = load.filename('Plate',  plate=int(plate), mjd=mjd, chips=True)
            d = load.apPlate(int(plate), mjd) 
            cframe = load.apPlate(int(plate), mjd)
            if type(d)!=dict: print("Problem with apPlate!!!")
            dhdr = fits.getheader(dfile.replace('apPlate-','apPlate-a-'))

        if ims[0] != 0:
            pfile = os.path.basename(load.filename('1D', plate=int(plate), num=ims[0], mjd=mjd, chips=True))
            dfile = load.filename('1D',  plate=int(plate), num=ims[1], mjd=mjd, chips=True)
            d = load.ap1D(ims[i])
            cframe = load.apCframe(field, int(plate), mjd, ims[i])
            if type(d)!=dict: print("Problem with ap1D!!!")
            dhdr = fits.getheader(dfile.replace('1D-','1D-a-'))

        cframefile = load.filename('Cframe', plate=int(plate), mjd=mjd, num=ims[1], chips='c')
        cframehdr = fits.getheader(cframefile.replace('Cframe-','Cframe-a-'))
        pfile = pfile.replace('.fits','')

        # Get moon distance and phase.
        dateobs = dhdr['DATE-OBS']
        tt = Time(dateobs, format='fits')
        moonpos = get_moon(tt)
        moonra = moonpos.ra.deg
        moondec = moonpos.dec.deg
        c1 = SkyCoord(ra * astropyUnits.deg, dec * astropyUnits.deg)
        c2 = SkyCoord(moonra * astropyUnits.deg, moondec * astropyUnits.deg)
        sep = c1.separation(c2)
        moondist = sep.deg
        moonphase = moon_illumination(tt)

        obs = np.zeros((nfiber,nchips), dtype=np.float64)
        sn  = np.zeros((nfiber,nchips), dtype=np.float64)
        snc = np.zeros((nfiber,nchips), dtype=np.float64)
        snt = np.zeros((nfiber,nchips), dtype=np.float64)

        # For each fiber, get an observed mag from a median value.
        for j in range(nfiber):
            for ichip in range(nchips): 
                obs[j, ichip] = np.median(d[chips[ichip]][1].data[rows[j], :])

        # Get a "magnitude" for each fiber from a median on each chip.
        # Do a crude sky subtraction, calculate S/N.
        for ichip in range(nchips):
            chip = chips[ichip]

            fluxarr = d[chip][1].data
            errarr = d[chip][2].data
            cfluxarr = cframe[chip][1].data
            cerrarr = cframe[chip][2].data

            if ims[0] == 0: medsky = 0.
            if ims[0] != 0: medsky = np.median(obs[fibersky, ichip])

            ### NOTE: using axis=0 caused error, so trying axis=0
            if nobj > 0: obs[fiberobj, ichip] = np.median(fluxarr[obj, :], axis=1) - medsky

            if ntelluric > 0: obs[fibertelluric, ichip] = np.median(fluxarr[telluric, :], axis=1) - medsky

            if nobj > 0:
                sn[fiberobj, ichip] = np.median((fluxarr[obj, :] - medsky) / errarr[obj, :], axis=1)
                if len(cframe) > 1:
                    snc[fiberobj, ichip] = np.median(cfluxarr[obj, :] / cerrarr[obj, :], axis=1)

            if ntelluric > 0:
                sn[fibertelluric, ichip] = np.median((fluxarr[telluric,:] - medsky) / errarr[telluric, :], axis=1)
                if len(cframe) > 1:
                    snc[fibertelluric, ichip] = np.median(cfluxarr[telluric, :] / cerrarr[telluric, :], axis=1)
                    medfilt = ScipyMedfilt2D(cfluxarr[telluric, :], kernel_size=(1,49))
                    sz = cfluxarr.shape
                    i1 = int(np.floor((900 * sz[1]) / 2048))
                    i2 = int(np.floor((1000 * sz[1]) / 2048))
                    for itell in range(ntelluric):
                        p1 = np.mean(cfluxarr[telluric[itell], i1:i2])
                        p2 = np.std(cfluxarr[telluric[itell], i1:i2] - medfilt[itell, i1:i2])
                        snt[fibertelluric[itell], ichip] = p1 / p2

                else:
                    snc[fibertelluric,ichip] = sn[fibertelluric,ichip]
                    medfilt = ScipyMedfilt2D(fluxarr[telluric, :], kernel_size=(1,49))
                    sz = fluxarr.shape
                    i1 = int(np.floor((900 * sz[1]) / 2048))
                    i2 = int(np.floor((1000 * sz[1]) / 2048))
                    for itell in range(ntelluric):
                        p1 = np.mean(fluxarr[telluric[itell], i1:i2 * (int(np.floor(sz[1] / 2048)))])
                        p2 = np.std(fluxarr[telluric[itell], i1:i2] - medfilt[itell, i1:i2])
                        snt[fibertelluric[itell], ichip] = p1 / p2

        # Calculate zeropoints from known H band mags.
        # Use a static zeropoint to calculate sky brightness.
        nreads = 0
        if "NFRAMES" in dhdr: nreads = dhdr['NFRAMES']
        exptime = dhdr['EXPTIME']
        skyzero = 14.75 + (2.5 * np.log10(nreads))
        zero = 0
        zerorms = 0.
        faint = -1
        nfaint = 0
        achievedsn = np.zeros(nchips)
        achievedsnc = np.zeros(nchips)
        altsn = np.zeros(nchips)
        nsn = 0

        tmp = fiber['hmag'][fiberstar] + (2.5 * np.log10(obs[fiberstar,1]))
        zero = np.nanmedian(tmp)
        zerorms = dln.mad(fiber['hmag'][fiberstar] + (2.5 * np.log10(obs[fiberstar,1])))
        faint, = np.where((tmp - zero) < -0.5)
        nfaint = len(faint)
        zeronorm = zero - (2.5 * np.log10(nreads))

        if (flat is None) & (onem is None):
            # Target line that has S/N=100 for 3 hour exposure at H=12.2
            sntarget = 100 * np.sqrt(exptime / (3.0 * 3600))
            sntargetmag = 12.2

            # Get typical S/N for this plate
            snstars, = np.where((fiber['hmag'] > 12) & (fiber['hmag'] < 12.2))
            nsn = len(snstars)
            scale = 1
            if nsn < 3:
                bright, = np.where(fiber['hmag'] < 12)
                hmax = np.nanmax(fiber['hmag'][bright])
                snstars, = np.where((fiber['hmag'] > hmax-0.2) & (fiber['hmag'] <= hmax))
                nsn = len(snstars)
                scale = np.sqrt(10**(0.4 * (hmax - 12.2)))

            achievedsn = np.median(sn[snstars,:], axis=0) * scale

            # Alternative S/N as computed from median of all stars with H<12.2, scaled
            snstars, = np.where(fiber['hmag'] < 12.2)
            scale = np.sqrt(10**(0.4 * (fiber['hmag'][snstars] - 12.2)))
            altsn = achievedsn * 0.
            for ichip in range(nchips): 
                altsn[ichip] = np.nanmedian(sn[snstars,ichip] * scale)
                achievedsnc[ichip] = np.nanmedian(snc[snstars,ichip] * scale)
        else:
            if onem is not None:
                achievedsn = np.nanmedian([sn[obj,:]], axis=0)

        medsky = np.zeros(3, dtype=np.float64)
        for ichip in range(nchips):
            if np.median(obs[fibersky,ichip]) > 0:
                medsky[ichip] = -2.5 * np.log10(np.nanmedian(obs[fibersky,ichip])) + skyzero
            else: 
                medsky[ichip] = 99.999

        # Get guider info.
        if onem is None:
            dateobs = dhdr['DATE-OBS']
            exptime = dhdr['EXPTIME']
            tt = Time(dateobs)
            mjdstart = tt.mjd
            mjdend = mjdstart + (exptime/86400.)
            mjd0 = min([mjd0,mjdstart])
            mjd1 = max([mjd1,mjdend])
            nj = 0
            if os.path.exists(gcamfile):
                jcam, = np.where((gcam['MJD'] > mjdstart) & (gcam['MJD'] < mjdend))
                nj = len(jcam)
            if nj > 1: 
                fwhm = np.median(gcam['FWHM_MEDIAN'][jcam]) 
                gdrms = np.median(gcam['GDRMS'][jcam])
            else:
                fwhm = -1.
                gdrms = -1.
                if i == 0: print("not halted: no matching mjd range in gcam...")
        else:
            fwhm = -1
            gdrms = -1
            exptime=-9.999

        alt = 0
        if "ALT" in dhdr: alt = dhdr['ALT']
        secz = 1. / np.cos((90.-alt) * (math.pi/180.))
        seeing = 0
        if "SEEING" in dhdr: seeing = dhdr['SEEING']
        ### NOTE:'ha' is not in the plugfile, but values are ['-', '-', '-']. Setting design_ha=0 for now
#        design_ha = plug['ha']
        design_ha = [0,0,0]
        dither = -99.
        if len(cframe) > 1: dither = cframehdr['DITHSH']

        allsky[i,:] = medsky
        allzero[i,:] = zero
        allzerorms[i,:] = zerorms


        # Summary information in apPlateSum FITS file.
        if ims[0] != 0:
            tellfile = load.filename('Tellstar', plate=int(plate), mjd=mjd)
            if os.path.exists(tellfile):
                try:
                    telstr = fits.getdata(tellfile)
                except:
                    if i == 0: print("PROBLEM!!! Error reading apTellstar file: "+os.path.basename(tellfile))
                else:
                    telstr = fits.getdata(tellfile)
                    jtell, = np.where(telstr['IM'] == ims[i])
                    ntell = len(jtell)
                    if ntell > 0: platetab['TELLFIT'][i] = telstr['FITPARS'][jtell]
            else:
                print("PROBLEM!!! "+os.path.basename(tellfile)+" does not exist.")

        platetab['IM'][i] =        ims[i]
        platetab['NREADS'][i] =    nreads
        platetab['SECZ'][i] =      secz
        if dhdr.get('HA') is not None: platetab['HA'][i] = dhdr['HA']
        platetab['DESIGN_HA'][i] = design_ha
        platetab['SEEING'][i] =    seeing
        platetab['FWHM'][i] =      fwhm
        platetab['GDRMS'][i] =     gdrms
        cart=0
        if 'CARTID' in dhdr: cart = dhdr['CARTID']
        platetab['CART'][i] =      cart
        platetab['DATEOBS'][i] =   dateobs
        platetab['EXPTIME'][i] =   exptime
        platetab['DITHER'][i] =    dither
        platetab['ZERO'][i] =      zero
        platetab['ZERORMS'][i] =   zerorms
        platetab['ZERONORM'][i] =  zeronorm
        platetab['SKY'][i] =       medsky
        platetab['SN'][i] =        achievedsn
        platetab['ALTSN'][i] =     altsn
        platetab['NSN'][i] =       nsn
        platetab['SNC'][i] =       achievedsnc
        if ntelluric > 0: platetab['SNRATIO'][i] = np.nanmedian(snt[telluric,1] / snc[telluric,1])

        for j in range(len(fiber)):
            fiber['sn'][j][i,:] = sn[j,:]
            fiber['obsmag'][j][i,:] = (-2.5 * np.log10(obs[j,:])) + zero


    # write out the FITS table.
    platesum = load.filename('PlateSum', plate=int(plate), mjd=mjd)
    if ims[0] != 0:
        ### NOTE:the only different between below if statement is that if ims is none, /create is not set in mwrfits
        # ... not sure if we care.
        Table(platetab).write(platesum, overwrite=True)
        hdulist = fits.open(platesum)
        hdu = fits.table_to_hdu(Table(fiber))
        hdulist.append(hdu)
        hdulist.writeto(platesum, overwrite=True)
        hdulist.close()
#;        mwrfits,platetab,platesum,/create
#;        mwrfits,fiber,platesum

    if ims[0] == 0:
        hdulist = fits.open(platesum)
        hdu1 = fits.table_to_hdu(Table(platetab))
        hdu2 = fits.table_to_hdu(Table(fiber))
        hdulist.append(hdu1)
        hdulist.append(hdu2)
        hdulist.writeto(platesum, overwrite=True)
        hdulist.close()
#;        mwrfits,platetab,platesum
#;        mwrfits,fiber,platesum

    print("\nDone making apPlateSum for plate "+plate+", mjd "+mjd)



''' MASTERQAHTML: mkhtmlplate translation                                                   '''
def masterQAhtml(load=None, plate=None, mjd=None, field=None, fluxid=None, telescope=None):

    print("--------------------------------------------------------------------")
    print("Making MASTERQAHTML for plate "+plate+", mjd "+mjd)
    print("--------------------------------------------------------------------\n")

    chips = np.array(['a','b','c'])
    nchips = len(chips)

    prefix = 'ap'
    if telescope == 'lco25m': prefix = 'as'

    # Check for existence of plateSum file
    platesum = load.filename('PlateSum', plate=int(plate), mjd=mjd) 
    platesumfile = os.path.basename(platesum)
    platedir = os.path.dirname(platesum)+'/'

    if os.path.exists(platesum) is False:
        err1 = "PROBLEM!!! "+platesumfile+" does not exist. Halting execution.\n"
        err2 = "You need to run MAKEPLATESUM first to make the file."
        sys.exit(err1 + err2)

    # Read the plateSum file
    tmp = fits.open(platesum)
    tab1 = tmp[1].data
    tab2 = tmp[2].data
#    tab3 = tmp[3].data
#;    tab3 = mrdfits(platesum,3,status=status)

    # Make the html directory if it doesn't already exist
    qafile = load.filename('QA', plate=int(plate), mjd=mjd)
    qafiledir = os.path.dirname(qafile)
    print("Creating "+os.path.basename(qafile)+"\n")
    if os.path.exists(qafiledir) is False: subprocess.call(['mkdir',qafiledir])

    html = open(qafile, 'w')
    html.write('<HTML><HEAD><script src="sorttable.js"></script></head><BODY>\n')
    html.write('<H1>Field: '+field+'   Plate: '+plate+'   MJD: '+mjd+'</H1>\n')
    html.write('<HR>\n')

    ### NOTE:just setting status=1 and hoping for the best.
    status = 1
    if status < 0:
        html.write('<FONT COLOR=red> ERROR READING TAB FILE\n')
        html.write('</BODY></HTML>\n')
        html.close()

    platefile = load.apPlate(int(plate), mjd)
    shiftstr = platefile['a'][13].data
    pairstr = platefile['a'][14].data
#;    shiftstr = mrdfits(platefile,13)
#;    pairstr = mrdfits(platefile,14,status=status)

    if status < 0:
        html.write('<FONT COLOR=red> ERROR READING apPlate FILE\n')
        html.write('</BODY></HTML>\n')
        html.close()
        return

    # Link to combined spectra page.
    html.write('<H3> For plots of apVisit spectra: <A HREF='+prefix+'Plate-'+plate+'-'+mjd+'.html> click here apPlate-'+plate+'-'+mjd+' </a><H3>\n')
    html.write('<H3>apVisit Hmag versus S/N: </H3>\n')
    snrplot = 'apVisitSNR-'+plate+'-'+mjd+'.png'
    html.write('<A HREF=../plots/'+snrplot+' target="_blank"><IMG SRC=../plots/'+snrplot+' WIDTH=900></A>\n')

    html.write('<HR>\n')

    # Table of individual exposures.
    if pairstr is not None:
        html.write('<H3>Individual Exposure Stats:</H3>\n')
    else:
        html.write('<H3>Individual Exposure Stats (undithered):</H3>\n')
    html.write('<p> Note: design HA values are currently missing. </p>\n')
    html.write('<p> Note: Dither and Pixshift values will be zero if exposures not dithered. </p>\n')
    html.write('<TABLE BORDER=2 CLASS="sortable">\n')
    html.write('<TR bgcolor=lightgreen>\n')
    txt1 = '<TH>Frame<TH>Exptime(s)<TH>Cart<TH>sec z<TH>HA<TH>DESIGN HA<TH>seeing<TH>FWHM<TH>GDRMS<TH>Nreads<TH>Dither'
    txt2 = '<TH>Pixshift<TH>Zero<TH>Zero rms<TH>sky continuum<TH>S/N<TH>S/N(cframe)<TH>Moon Phase<TH>Moon Dist.'
    html.write(txt1 + txt2 +'\n')
    for i in range(len(tab1)):
        html.write('<TR>\n')
        html.write('<TD>'+str(int(round(tab1['IM'][i])))+'\n')
        html.write('<TD>'+str(int(round(tab1['EXPTIME'][i])))+'\n')
        html.write('<TD>'+str(int(round(tab1['CART'][i])))+'\n')
        html.write('<TD>'+str("%.2f" % round(tab1['SECZ'][i],2))+'\n')
        html.write('<TD>'+str("%.2f" % round(tab1['HA'][i],2))+'\n')
        html.write('<TD>'+str(np.round(tab1['DESIGN_HA'][i],0)).replace('[',' ')[:-1]+'\n')
        html.write('<TD>'+str("%.2f" % round(tab1['SEEING'][i],2))+'\n')
        html.write('<TD>'+str("%.2f" % round(tab1['FWHM'][i],2))+'\n')
        html.write('<TD>'+str("%.2f" % round(tab1['GDRMS'][i],2))+'\n')
        html.write('<TD>'+str(tab1['NREADS'][i])+'\n')
        j = np.where(shiftstr['FRAMENUM'] == str(tab1['IM'][i]))
        nj = len(j[0])
        if nj > 0:
            html.write('<TD>'+str("%.4f" % round(shiftstr['SHIFT'][j][0],4)).rjust(7)+'\n')
            html.write('<TD>'+str("%.2f" % round(shiftstr['PIXSHIFT'][j][0],2))+'\n')
        else:
            html.write('<TD><TD>\n')
            html.write('<TD><TD>\n')
        html.write('<TD>'+str("%.2f" % round(tab1['ZERO'][i],2))+'\n')
        html.write('<TD>'+str("%.2f" % round(tab1['ZERORMS'][i],2))+'\n')
        q = tab1['SKY'][i]
        txt = str("%.2f" % round(q[0],2))+', '+str("%.2f" % round(q[1],2))+', '+str("%.2f" % round(q[2],2))
        html.write('<TD>'+'['+txt+']\n')
        q = tab1['SN'][i]
        txt = str("%.2f" % round(q[0],2))+', '+str("%.2f" % round(q[1],2))+', '+str("%.2f" % round(q[2],2))
        html.write('<TD>'+'['+txt+']\n')
        q = tab1['SNC'][i]
        txt = str("%.2f" % round(q[0],2))+', '+str("%.2f" % round(q[1],2))+', '+str("%.2f" % round(q[2],2))
        html.write('<TD>'+'['+txt+']\n')
        html.write('<TD>'+str("%.3f" % round(tab1['MOONPHASE'][i],3))+'\n')
        html.write('<TD>'+str("%.3f" % round(tab1['MOONDIST'][i],3))+'\n')
    html.write('</TABLE>\n')

    # Table of exposure pairs.
    if pairstr is not None:
        npairs = len(pairstr)
        if npairs > 0:
            # Pair table.
            html.write('<H3>Dither Pair Stats:</H3>\n')
            html.write('<TABLE BORDER=2 CLASS="sortable">\n')
            html.write('<TR bgcolor=lightgreen><TH>IPAIR<TH>NAME<TH>SHIFT<TH>NEWSHIFT<TH>S/N\n')
            html.write('<TH>NAME<TH>SHIFT<TH>NEWSHIFT<TH>S/N\n')
            for ipair in range(npairs):
                html.write('<TR><TD>'+str(ipair)+'\n')
                for j in range(2):
                    html.write('<TD>'+str(pairstr['FRAMENAME'][ipair][j])+'\n')
                    html.write('<TD>'+str("%.3f" % round(pairstr['OLDSHIFT'][ipair][j],3))+'\n')
                    html.write('<TD>'+str("%.3f" % round(pairstr['SHIFT'][ipair][j],3))+'\n')
                    html.write('<TD>'+str("%.2f" % round(pairstr['SN'][ipair][j],2))+'\n')
            html.write('</TABLE>\n')
#    else:
#        # Table of combination parameters.
#        html.write('<H3>Combination Parameters (undithered):</H3>\n')
#        html.write('<BR><TABLE BORDER=2 CLASS="sortable">\n')
#        for iframe in range(len(shiftstr)):
#            html.write('<TR><TD>'+str(shiftstr['FRAMENUM'][iframe])+'\n')
#            html.write('<TD>'+str("%.3f" % round(shiftstr['SHIFT'][iframe],3))+'\n')
#            html.write('<TD>'+str("%.3f" % round(shiftstr['SN'][iframe],3))+'\n')
#    html.write('</TABLE>\n')

    # Table of exposure plots.
    html.write('<TABLE BORDER=2>\n')
    html.write('<BR>\n')
    html.write('<H3>Individual Exposure QA Plots:</H3>\n')
    html.write('<p> Note: in the Spatial mag deviation plots, color gives deviation of observed mag from expected 2MASS mag using the median zeropoint.</p>\n')
    html.write('<TR bgcolor=lightgreen><TH>Frame<TH>Zeropoints<TH>Mag plots\n')
    html.write('<TH>Spatial mag deviation\n')
    html.write('<TH>Spatial sky telluric CH4\n')
    html.write('<TH>Spatial sky telluric CO2\n')
    html.write('<TH>Spatial sky telluric H2O\n')
    html.write('<TH>Spatial sky 16325A emission deviations (filled: sky, open: star)\n')
    html.write('<TH>Spatial sky continuum emission\n')

    for i in range(len(tab1)):
        im=tab1['IM'][i]
        oneDfile = os.path.basename(load.filename('1D', plate=int(plate), num=im, mjd=mjd, chips=True)).replace('.fits','')
        html.write('<TR><TD bgcolor=lightgreen><A HREF=../html/'+oneDfile+'.html>'+str(im)+'</A>\n')
        html.write('<TD><TABLE BORDER=1><TD><TD bgcolor=lightgreen>Red<TD bgcolor=lightgreen>Green<TD bgcolor=lightgreen>Blue\n')
        html.write('<TR><TD bgcolor=lightgreen>z<TD><TD>'+str("%.2f" % round(tab1['ZERO'][i],2))+'\n')
        html.write('<TR><TD bgcolor=lightgreen>znorm<TD><TD>'+str("%.2f" % round(tab1['ZERONORM'][i],2))+'\n')
        txt='<TD>'+str("%.1f" % round(tab1['SKY'][i][0],1))+'<TD>'+str("%.1f" % round(tab1['SKY'][i][1],1))+'<TD>'+str("%.1f" % round(tab1['SKY'][i][2],1))
        html.write('<TR><TD bgcolor=lightgreen>sky'+txt+'\n')
        txt='<TD>'+str("%.1f" % round(tab1['SN'][i][0],1))+'<TD>'+str("%.1f" % round(tab1['SN'][i][1],1))+'<TD>'+str("%.1f" % round(tab1['SN'][i][2],1))
        html.write('<TR><TD bgcolor=lightgreen>S/N'+txt+'\n')
        txt='<TD>'+str("%.1f" % round(tab1['SNC'][i][0],1))+'<TD>'+str("%.1f" % round(tab1['SNC'][i][1],1))+'<TD>'+str("%.1f" % round(tab1['SNC'][i][2],1))
        html.write('<TR><TD bgcolor=lightgreen>S/N(c)'+txt+'\n')
#        if tag_exist(tab1[i],'snratio'):
        html.write('<TR><TD bgcolor=lightgreen>SN(E/C)<TD>'+str(np.round(tab1['SNRATIO'][i],2))+'\n')
        html.write('</TABLE>\n')

        html.write('<TD><A HREF=../plots/'+oneDfile+'_magplots.png target="_blank"><IMG SRC=../plots/'+oneDfile+'_magplots.png WIDTH=400></A>\n')
        html.write('<TD><A HREF=../plots/'+oneDfile+'_spatialresid.png target="_blank"><IMG SRC=../plots/'+oneDfile+'_spatialresid.png WIDTH=500></A>\n')
        cim=str(im)
        html.write('<TD> <a href=../plots/'+prefix+'telluric_'+cim+'_skyfit_CH4.jpg> <IMG SRC=../plots/'+prefix+'telluric_'+cim+'_skyfit_CH4.jpg height=400></a>\n')
        html.write('<TD> <a href=../plots/'+prefix+'telluric_'+cim+'_skyfit_CO2.jpg> <IMG SRC=../plots/'+prefix+'telluric_'+cim+'_skyfit_CO2.jpg height=400></a>\n')
        html.write('<TD> <a href=../plots/'+prefix+'telluric_'+cim+'_skyfit_H2O.jpg> <IMG SRC=../plots/'+prefix+'telluric_'+cim+'_skyfit_H2O.jpg height=400></a>\n')
        html.write('<TD> <IMG SRC=../plots/'+oneDfile+'sky.jpg>\n')
        html.write('<TD> <IMG SRC=../plots/'+oneDfile+'skycont.jpg>\n')
    html.write('</table>\n')

    html.write('<BR><BR>\n')

    # Flat field plots.
    if fluxid is not None:
        html.write('<TABLE BORDER=2><TR bgcolor=lightgreen>\n')
        txt1 = '<TH> Blue chip relative<BR>flat field flux <TH> Greeen chip relative<BR>flat field flux'
        txt2 = '<TH> Red chip relative<BR>flat field flux<TH> Fiber Blocks <TH> Guider RMS'
        html.write(txt1+txt2+'\n')
        html.write('<TR>\n')
        for chip in chips:
            fluxfile = load.filename('Flux', num=fluxid, chips=True).replace('apFlux-','apFlux-'+chip+'-')
            fluxfile = os.path.basename(fluxfile).replace('.fits','')
            html.write('<TD> <A HREF='+'../plots/'+fluxfile+'.png><IMG SRC=../plots/'+fluxfile+'.png WIDTH=400></A>\n')
        tmp = load.filename('Flux', num=fluxid, chips=True).replace('apFlux-','apFlux-'+chips[0]+'-')
        blockfile = os.path.basename(tmp).replace('.fits','').replace('-a-','-block-')
        html.write('<TD> <A HREF='+'../plots/'+blockfile+'.png><IMG SRC=../plots/'+blockfile+'.png WIDTH=400></A>\n')
        gfile = 'guider-'+plate+'-'+mjd+'.png'
        html.write('<TD> <A HREF='+'../plots/'+gfile+'><IMG SRC=../plots/'+gfile+' WIDTH=400></A>\n')
        html.write('</TABLE>\n')

    html.write('</BODY></HTML>\n')
    html.close()

    print("--------------------------------------------------------------------")
    print("Done with MASTERQAHTML for plate "+plate+", mjd "+mjd)
    print("--------------------------------------------------------------------\n")



''' $$$   MASTERQAPLOTS: plots for the master QA page   $$$ '''
def masterQAplots(load=None, ims=None, plate=None, mjd=None, instrument=None, apred=None,
                  flat=None, fluxid=None, survey=None): 

    print("--------------------------------------------------------------------")
    print("Making MASTERQAPLOTS for plate "+plate+", mjd "+mjd)
    print("--------------------------------------------------------------------\n")

    n_exposures = len(ims)
    chips = np.array(['a','b','c'])
    chiplab = np.array(['blue','green','red'])
    nchips = len(chips)

    # Make plot and html directories if they don't already exist.
    platedir = os.path.dirname(load.filename('Plate', plate=int(plate), mjd=mjd, chips=True))
    plotsdir = platedir+'/plots/'
    if len(glob.glob(plotsdir)) == 0: subprocess.call(['mkdir',plotsdir])

    # Set up some basic plotting parameters, starting by turning off interactive plotting.
    plt.ioff()
    fontsize = 24;   fsz = fontsize * 0.75
    matplotlib.rcParams.update({'font.size':fontsize, 'font.family':'serif'})
    alpha = 0.6
    axwidth=1.5
    axmajlen=7
    axminlen=3.5

    # Check for existence of plateSum file
    platesum = load.filename('PlateSum', plate=int(plate), mjd=mjd) 
    if os.path.exists(platesum) is False:
        err1 = "PROBLEM!!! "+platesumfile+" does not exist. Halting execution.\n"
        err2 = "You need to run MAKEPLATESUM first to make the file."
        sys.exit(err1 + err2)

    # Read the plateSum file
    tmp = fits.open(platesum)
    plSum1 = tmp[1].data
    platesum2 = tmp[2].data
    fibord = np.argsort(platesum2['FIBERID'])
    plSum2 = platesum2[fibord]
    nfiber = len(plSum2['HMAG'])

    #----------------------------------------------------------------------------------------------
    # PLOT 1: HMAG versus S/N for the exposure-combined apVisit
    #----------------------------------------------------------------------------------------------
    Vsum = load.apVisitSum(int(plate), mjd)
    Vsumfile = Vsum.filename()
    Vsum = Vsum[1].data

    plotfile = os.path.basename(Vsumfile).replace('Sum','SNR').replace('.fits','.png')
    print("Making "+plotfile)

    fig=plt.figure(figsize=(18,8))
    ax1 = plt.subplot2grid((1,1), (0,0))
    ax1.tick_params(reset=True)
    ax1.minorticks_on()
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.set_xlabel(r'$H$ mag.');  ax1.set_ylabel(r'apVisit S/N')
    ax1.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
    ax1.tick_params(axis='both',which='major',length=axmajlen)
    ax1.tick_params(axis='both',which='minor',length=axminlen)
    ax1.tick_params(axis='both',which='both',width=axwidth)

    if 'HMAG' in Vsum.columns.names:
        hmagarr = Vsum['HMAG']
    else:
        hmagarr = Vsum['H']
    minH = np.nanmin(hmagarr);       maxH = np.nanmax(hmagarr);        spanH = maxH - minH
    xmin = minH - spanH * 0.05;      xmax = maxH + spanH * 0.05

    minSNR = np.nanmin(Vsum['SNR']); maxSNR = np.nanmax(Vsum['SNR']);  spanSNR = maxSNR - minSNR
    ymin = -5;                       ymax = maxSNR + ((maxSNR - ymin) * 0.05)
    
    ax1.set_xlim(xmin,xmax)#;  ax1.set_ylim(ymin,ymax)

    if 'apogee' in survey.lower():
        telluric, = np.where(bitmask.is_bit_set(Vsum['APOGEE_TARGET2'],9))
        science, = np.where((bitmask.is_bit_set(Vsum['APOGEE_TARGET2'],4) == 0) & 
                            (bitmask.is_bit_set(Vsum['APOGEE_TARGET2'],9) == 0))
    else:
        telluric, = np.where(bitmask.is_bit_set(Vsum['SDSSV_APOGEE_TARGET0'],1))
        science, = np.where((bitmask.is_bit_set(Vsum['SDSSV_APOGEE_TARGET0'],0) == 0) & 
                            (bitmask.is_bit_set(Vsum['SDSSV_APOGEE_TARGET0'],1) == 0))

    x = hmagarr[science];  y = Vsum['SNR'][science]
    psci = ax1.semilogy(x, y, marker='*', ms=15, mec='k', alpha=alpha, mfc='r', linestyle='', label='science')
    x = hmagarr[telluric];  y = Vsum['SNR'][telluric]
    ptel = ax1.semilogy(x, y, marker='o', ms=9, mec='k', alpha=alpha, mfc='cyan', linestyle='', label='Telluric')

    ax1.legend(loc='upper right', labelspacing=0.5, handletextpad=-0.1, facecolor='lightgrey')

    fig.subplots_adjust(left=0.075,right=0.98,bottom=0.11,top=0.98,hspace=0.2,wspace=0.0)
    plt.savefig(plotsdir+plotfile)
    plt.close('all')

    #----------------------------------------------------------------------------------------------
    # PLOTS 2-4: flat field flux and fiber blocks... previously done by plotflux.pro
    #----------------------------------------------------------------------------------------------
    fluxfile = os.path.basename(load.filename('Flux', num=fluxid, chips=True))
    flux = load.apFlux(fluxid)
    ypos = 300 - platesum2['FIBERID']

    for ichip in range(nchips):
        chip = chips[ichip]
        plotfile = fluxfile.replace('Flux-', 'Flux-'+chip+'-').replace('.fits', '.png')
        print("Making "+plotfile)

        fig=plt.figure(figsize=(14,15))
        ax1 = plt.subplot2grid((1,1), (0,0))
        ax1.set_xlim(-1.6,1.6)
        ax1.set_ylim(-1.6,1.6)
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax1.minorticks_on()
        ax1.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
        ax1.tick_params(axis='both',which='major',length=axmajlen)
        ax1.tick_params(axis='both',which='minor',length=axminlen)
        ax1.tick_params(axis='both',which='both',width=axwidth)
        ax1.set_xlabel(r'Zeta');  ax1.set_ylabel(r'Eta')

        med = np.median(flux[chip][1].data, axis=1)
        sc = ax1.scatter(platesum2['Zeta'], platesum2['Eta'], marker='o', s=100, c=med[ypos], edgecolors='k', cmap='jet', alpha=1, vmin=0.5, vmax=1.5)

        ax1.text(0.03,0.97,chiplab[ichip]+'\n'+'chip', transform=ax1.transAxes, ha='left', va='top')

        ax1_divider = make_axes_locatable(ax1)
        cax1 = ax1_divider.append_axes("top", size="4%", pad="1%")
        cb = colorbar(sc, cax=cax1, orientation="horizontal")
        cax1.xaxis.set_ticks_position("top")
        cax1.minorticks_on()
        ax1.text(0.5, 1.10, r'Median Flat Field Flux',ha='center', transform=ax1.transAxes)

        fig.subplots_adjust(left=0.12,right=0.98,bottom=0.08,top=0.93,hspace=0.2,wspace=0.0)
        plt.savefig(plotsdir+plotfile)
        plt.close('all')

    #----------------------------------------------------------------------------------------------
    # PLOT 5: fiber blocks... previously done by plotflux.pro
    #----------------------------------------------------------------------------------------------
    block = np.around((plSum2['FIBERID'] - 1) / 30)
    plotfile = fluxfile.replace('Flux-', 'Flux-block-').replace('.fits', '.png')
    print("Making "+plotfile)

    fig=plt.figure(figsize=(14,15))
    ax1 = plt.subplot2grid((1,1), (0,0))
    ax1.set_xlim(-1.6,1.6)
    ax1.set_ylim(-1.6,1.6)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax1.minorticks_on()
    ax1.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
    ax1.tick_params(axis='both',which='major',length=axmajlen)
    ax1.tick_params(axis='both',which='minor',length=axminlen)
    ax1.tick_params(axis='both',which='both',width=axwidth)
    ax1.set_xlabel(r'Zeta');  ax1.set_ylabel(r'Eta')

    sc = ax1.scatter(plSum2['Zeta'], plSum2['Eta'], marker='o', s=100, c=block, edgecolors='k', cmap='jet', alpha=1, vmin=0, vmax=10)

    ax1_divider = make_axes_locatable(ax1)
    cax1 = ax1_divider.append_axes("top", size="4%", pad="1%")
    cb = colorbar(sc, cax=cax1, orientation="horizontal")
    cax1.xaxis.set_ticks_position("top")
    cax1.minorticks_on()
    ax1.text(0.5, 1.10, r'Fiber Blocks',ha='center', transform=ax1.transAxes)

    fig.subplots_adjust(left=0.12,right=0.98,bottom=0.08,top=0.93,hspace=0.2,wspace=0.0)
    plt.savefig(plotsdir+plotfile)
    plt.close('all')

    #----------------------------------------------------------------------------------------------
    # PLOT 6: guider rms plot
    #----------------------------------------------------------------------------------------------
    expdir = os.environ.get('APOGEE_REDUX')+'/'+apred+'/'+'exposures/'+instrument+'/'
    gcamfile = expdir+mjd+'/gcam-'+mjd+'.fits'
    if os.path.exists(gcamfile):
        gcam = fits.getdata(gcamfile)

        dateobs = plSum1['DATEOBS'][0]
        tt = Time(dateobs)
        mjdstart = tt.mjd
        exptime = np.sum(plSum1['EXPTIME'])
        mjdend = mjdstart + (exptime/86400.)
        jcam, = np.where((gcam['mjd'] > mjdstart) & (gcam['mjd'] < mjdend))

        plotfile = 'guider-'+plate+'-'+mjd+'.png'
        print("Making "+plotfile)

        fig=plt.figure(figsize=(14,14))
        ax1 = plt.subplot2grid((1,1), (0,0))
        ax1.tick_params(reset=True)
        ax1.minorticks_on()
        ax1.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
        ax1.tick_params(axis='both',which='major',length=axmajlen)
        ax1.tick_params(axis='both',which='minor',length=axminlen)
        ax1.tick_params(axis='both',which='both',width=axwidth)
        ax1.set_xlabel(r'Guider MJD');  ax1.set_ylabel(r'Guider RMS')

        ax1.plot(gcam['mjd'][jcam], gcam['gdrms'][jcam], color='k')

        fig.subplots_adjust(left=0.12,right=0.98,bottom=0.08,top=0.98,hspace=0.2,wspace=0.0)
        plt.savefig(plotsdir+plotfile)
        plt.close('all')

    # Loop over the exposures to make other plots.
    for i in range(n_exposures):
        #------------------------------------------------------------------------------------------
        # PLOTS 7: 3 panel mag/SNR plots for each exposure
        #----------------------------------------------------------------------------------------------
        plotfile = 'ap1D-'+str(plSum1['IM'][i])+'_magplots.png'
        print("Making "+plotfile)

        notsky, = np.where(plSum2['HMAG'] < 30)
        hmagarr = plSum2['HMAG'][notsky]
        minH = np.nanmin(hmagarr);       maxH = np.nanmax(hmagarr);        spanH = maxH - minH
        xmin = minH - spanH * 0.05;      xmax = maxH + spanH * 0.05

        fig=plt.figure(figsize=(11,14))
        ax1 = plt.subplot2grid((3,1), (0,0))
        ax2 = plt.subplot2grid((3,1), (1,0))
        ax3 = plt.subplot2grid((3,1), (2,0))
        axes = [ax1, ax2, ax3]#, ax4, ax5]
        ax2.set_ylim(-10,1)

        for ax in axes:
            ax.set_xlim(xmin,xmax)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.minorticks_on()
            ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
            ax.tick_params(axis='both',which='major',length=axmajlen)
            ax.tick_params(axis='both',which='minor',length=axminlen)
            ax.tick_params(axis='both',which='both',width=axwidth)

        ax1.axes.xaxis.set_ticklabels([])
        ax2.axes.xaxis.set_ticklabels([])

        ax3.set_xlabel(r'$H$')
        ax1.text(-0.17,0.50,r'm = -2.5*log(counts)',transform=ax1.transAxes,rotation=90,ha='left',va='center')
        ax2.text(-0.17,0.50,r'$H$ - (m+zero)',transform=ax2.transAxes,rotation=90,ha='left',va='center')
        ax3.text(-0.17,0.50,r'S/N',transform=ax3.transAxes,rotation=90,ha='left',va='center')

        # PLOTS 7a: observed mag vs H mag
        x = plSum2['HMAG'][science];    y = plSum2['obsmag'][science,i,1]-plSum1['ZERO'][i]
        ax1.scatter(x, y, marker='*', s=180, edgecolors='k', alpha=alpha, c='r', label='Science')
        x = plSum2['HMAG'][telluric];   y = plSum2['obsmag'][telluric,i,1]-plSum1['ZERO'][i]
        ax1.scatter(x, y, marker='o', s=60, edgecolors='k', alpha=alpha, c='cyan', label='Telluric')
        ax1.legend(loc='upper left', labelspacing=0.5, handletextpad=-0.1, facecolor='lightgrey')

        # PLOTS 7b: observed mag - fit mag vs H mag
        x = plSum2['HMAG'][science];    y = x - plSum2['obsmag'][science,i,1]
        yminsci = np.nanmin(y);  ymaxsci = np.nanmax(y)
        ax2.scatter(x, y, marker='*', s=180, edgecolors='k', alpha=alpha, c='r')
        x = plSum2['HMAG'][telluric];   y = x - plSum2['obsmag'][telluric,i,1]
        ymintel = np.nanmin(y);  ymaxtel = np.nanmax(y)
        ax2.scatter(x, y, marker='o', s=60, edgecolors='k', alpha=alpha, c='cyan')
        ymin = np.min([yminsci,ymintel])
        ymax = np.max([ymaxsci,ymaxtel])
        yspan = ymax - ymin
        #ax2.set_ylim(-10,1)

        # PLOTS 7c: S/N as calculated from ap1D frame
        c = ['r','g','b']
        for ichip in range(nchips):
            x = plSum2['HMAG'][science];   y = plSum2['SN'][science,i,ichip]
            ax3.semilogy(x, y, marker='*', ms=15, mec='k', alpha=alpha, mfc=c[ichip], linestyle='')
            x = plSum2['HMAG'][telluric];   y = plSum2['SN'][telluric,i,ichip]
            ax3.semilogy(x, y, marker='o', ms=9, mec='k', alpha=alpha, mfc=c[ichip], linestyle='')

        # overplot the target S/N line
        sntarget = 100 * np.sqrt(plSum1['EXPTIME'][i] / (3.0 * 3600))
        sntargetmag = 12.2
        x = [sntargetmag - 10, sntargetmag + 2.5];    y = [sntarget * 100, sntarget / np.sqrt(10)]
        ax3.plot(x, y, color='k',linewidth=1.5)

        fig.subplots_adjust(left=0.155,right=0.98,bottom=0.06,top=0.99,hspace=0.1,wspace=0.0)
        plt.savefig(plotsdir+plotfile)
        plt.close('all')

        #------------------------------------------------------------------------------------------
        # PLOT 3: spatial residuals for each exposure
        #----------------------------------------------------------------------------------------------
        plotfile = 'ap1D-'+str(plSum1['IM'][i])+'_spatialresid.png'
        print("Making "+plotfile)

        fig=plt.figure(figsize=(14,15))
        ax1 = plt.subplot2grid((1,1), (0,0))
        ax1.set_xlim(-1.6,1.6)
        ax1.set_ylim(-1.6,1.6)
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
        ax1.minorticks_on()
        ax1.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
        ax1.tick_params(axis='both',which='major',length=axmajlen)
        ax1.tick_params(axis='both',which='minor',length=axminlen)
        ax1.tick_params(axis='both',which='both',width=axwidth)
        ax1.set_xlabel(r'Zeta (deg.)');  ax1.set_ylabel(r'Eta (deg.)')

        x = plSum2['ZETA'][science];    y = plSum2['ETA'][science]
        c = plSum2['HMAG'][science] - plSum2['obsmag'][science,i,1]
        psci = ax1.scatter(x, y, marker='*', s=240, c=c, edgecolors='k', cmap='jet', alpha=1, vmin=-0.5, vmax=0.5, label='Science')

        x = plSum2['ZETA'][telluric];    y = plSum2['ETA'][telluric]
        c = plSum2['HMAG'][telluric] - plSum2['obsmag'][telluric,i,1]
        ptel = ax1.scatter(x, y, marker='o', s=100, c=c, edgecolors='k', cmap='jet', alpha=1, vmin=-0.5, vmax=0.5, label='Telluric')

        try:
            x = plSum2['ZETA'][sky];    y = plSum2['ETA'][sky]
            c = plSum2['HMAG'][sky] - plSum2['obsmag'][sky,i,1]
            psky = ax1.scatter(x, y, marker='s', s=100, c='white', edgecolors='k', alpha=1, label='Sky')
        except:
            print("PROBLEM!!! Sky fiber subscripting error when trying to make spatial mag. plots.")

        ax1.legend(loc='upper left', labelspacing=0.5, handletextpad=-0.1, facecolor='lightgrey')

        ax1_divider = make_axes_locatable(ax1)
        cax1 = ax1_divider.append_axes("top", size="4%", pad="1%")
        cb = colorbar(psci, cax=cax1, orientation="horizontal")
        cax1.xaxis.set_ticks_position("top")
        cax1.minorticks_on()
        ax1.text(0.5, 1.10, r'$H$ + 2.5*log(m - zero)',ha='center', transform=ax1.transAxes)

        fig.subplots_adjust(left=0.12,right=0.98,bottom=0.08,top=0.93,hspace=0.2,wspace=0.0)
        plt.savefig(plotsdir+plotfile)
        plt.close('all')

        #plotc,xx,yy,cc,min=0.9,max=1.1,xr=lim,yr=lim,ps=8,/xs,/ys
        #if nobj gt 0 then begin
        #    xx=fiber[fiberobj].zeta
        #    yy=fiber[fiberobj].eta
        #    cc=skylines[0].flux[fiberobj]/medsky
        #    plotc,xx,yy,cc,min=0.9,max=1.1,xr=lim,yr=lim,ps=6,overplot=1,/xs,/ys
        #endif
        #if ntelluric gt 0 then begin
        #    xx=fiber[fibertelluric].zeta
        #    yy=fiber[fibertelluric].eta
        #    cc=skylines[0].flux[fibertelluric]/medsky
        #    plotc,xx,yy,cc,min=0.9,max=1.1,xr=lim,yr=lim,ps=4,overplot=1,/xs,/ys
        #endif
        #device,/close
        #ps2jpg,outdir+file+'sky.eps',/eps,chmod='664'o,/delete

        if ims[0] == 0: d = load.apPlate(int(plate), mjd) 
        if ims[0] != 0: d = load.ap1D(ims[i])
        rows = 300-plSum2['FIBERID']

        fibersky, = np.where(plSum2['OBJTYPE'] == 'SKY')
        nsky = len(fibersky)
        sky = rows[fibersky]

        # Define skylines structure which we will use to get crude sky levels in lines.
        dt = np.dtype([('W1',   np.float64),
                       ('W2',   np.float64),
                       ('C1',   np.float64),
                       ('C2',   np.float64),
                       ('C3',   np.float64),
                       ('C4',   np.float64),
                       ('FLUX', np.float64, (nfiber)),
                       ('TYPE', np.int32)])

        skylines = np.zeros(2,dtype=dt);  nskylines=len(skylines)

        skylines['W1']   = 16230.0, 15990.0
        skylines['W2']   = 16240.0, 16028.0
        skylines['C1']   = 16215.0, 15980.0
        skylines['C2']   = 16225.0, 15990.0
        skylines['C3']   = 16245.0, 0.0
        skylines['C4']   = 16255.0, 0.0
        skylines['TYPE'] = 1, 0

        for iline in range(nskylines):
            skylines['FLUX'][iline] = getflux(d=d, skyline=skylines[iline], rows=rows)

        #medsky = np.median(skylines['FLUX'][sky][0])

        #xx = plSum2['ZETA'][science]
        #yy = plSum2['ETA'][science]
        #cc = skylines['FLUX'][sky][0] / medsky

        #------------------------------------------------------------------------------------------
        # PLOT 4: spatial sky line emission
        # https://data.sdss.org/sas/apogeework/apogee/spectro/redux/current/plates/5583/56257/plots/ap1D-06950025sky.jpg
        #------------------------------------------------------------------------------------------
        #print("PLOTS 4: spatial plot of sky line emission will be made here.\n")

        #------------------------------------------------------------------------------------------
        # PLOT 5: spatial continuum emission
        # https://data.sdss.org/sas/apogeework/apogee/spectro/redux/current/plates/5583/56257/plots/ap1D-06950025skycont.jpg
        #------------------------------------------------------------------------------------------
        #print("PLOTS 5: spatial plot of continuum emission will be made here.\n")

        #----------------------------------------------------------------------------------------------
        # PLOT 7: make plot of sky levels for this plate
        # https://data.sdss.org/sas/apogeework/apogee/spectro/redux/current/exposures/apogee-n/56257/plots/56257sky.gif
        #----------------------------------------------------------------------------------------------
        #skyfile = 'sky-'+gfile
        #print("PLOTS 7: Sky level plots will be made here.")

        #----------------------------------------------------------------------------------------------
        # PLOT 8: make plot of zeropoints for this plate
        # https://data.sdss.org/sas/apogeework/apogee/spectro/redux/current/exposures/apogee-n/56257/plots/56257zero.gif
        #----------------------------------------------------------------------------------------------
        #zerofile = 'zero-'+gfile
        #print("PLOTS 8: Zeropoints plots will be made here.")

    plt.ion()

    print("--------------------------------------------------------------------")
    print("Done with MASTERQAPLOTS for plate "+plate+", mjd "+mjd)
    print("--------------------------------------------------------------------\n")



''' $$$   MAKEOBJHTML: make the pages with spectrum plots   $$$ '''
def makeObjHtml(load=None, plate=None, mjd=None, survey=None, makeSpectrumPlots=None): 

    print("--------------------------------------------------------------------")
    print("Running MAKEOBJHTML for plate "+plate+", mjd "+mjd)
    print("--------------------------------------------------------------------\n")

    # Make plot and html directories if they don't already exist.
    platedir = os.path.dirname(load.filename('Plate', plate=int(plate), mjd=mjd, chips=True))
    plotsdir = platedir+'/plots/'
    htmldir = platedir+'/html/'
    if os.path.exists(plotsdir) is False: subprocess.call(['mkdir',plotsdir])
    if os.path.exists(htmldir) is False: subprocess.call(['mkdir',htmldir])

#    if os.path.exists(htmldir+'sorttable.js') is False:
#        print("getting sorttable.js...")
#        subprocess.call(['wget', '-q', sort_table_link])
#        subprocess.call(['mv', 'sorttable.js', htmldir])

    # Set up some basic plotting parameters, starting by turning off interactive plotting.
    plt.ioff()
    fontsize = 24;   fsz = fontsize * 0.75
    matplotlib.rcParams.update({'font.size':fontsize, 'font.family':'serif'})
    axwidth=1.5
    axmajlen=7
    axminlen=3.5

    # Load in the apPlate file
    apPlate = load.apPlate(int(plate), mjd)
    data = apPlate['a'][11].data[::-1]
    objtype = data['OBJTYPE']
    nfiber = len(data)

    # Check for existence of plateSum file
    platesum = load.filename('PlateSum', plate=int(plate), mjd=mjd) 
    if os.path.exists(platesum) is False:
        err1 = "PROBLEM!!! "+platesumfile+" does not exist. Halting execution.\n"
        err2 = "You need to run MAKEPLATESUM first to make the file."
        sys.exit(err1 + err2)

    # Read the plateSum file
    tmp = fits.open(platesum)
    plSum1 = tmp[1].data
    plSum2 = tmp[2].data

    # Read the plateSum file
    telluric, = np.where((objtype == 'SPECTROPHOTO_STD') | (objtype == 'HOT_STD'))
    ntelluric = len(telluric)
    science, = np.where((objtype != 'SPECTROPHOTO_STD') & (objtype != 'HOT_STD') & (objtype != 'SKY'))
    nscience = len(science)
    sky, = np.where(objtype == 'SKY')
    nsky = len(sky)

    # Get the HTML file name... apPlate-plate-mjd
    htmlfile = os.path.basename(load.filename('Plate', plate=int(plate), mjd=mjd, chips=True)).replace('.fits','')

    # For each star, create the exposure entry on the web page and set up the plot of the spectrum.
    objhtml = open(htmldir+htmlfile+'.html','w')
    objhtml.write('<HTML>\n')
    objhtml.write('<HEAD><script src="sorttable.js"></script></head>\n')
    objhtml.write('<BODY>\n')

    objhtml.write('<H1>'+htmlfile+'</H1>\n')
    #objhtml.write('<A HREF=../../../../red/'+mjd+'/html/'+pfile+'.html> 1D frames </A>\n')
    #objhtml.write('<BR><A HREF=../../../../red/'+mjd+'/html/ap2D-'+str(plSum1['IM'][i])+'.html> 2D frames </A>\n')

    objhtml.write('<TABLE BORDER=2 CLASS="sortable">\n')
    objhtml.write('<TR><TH>Fiber<TH>APOGEE ID<TH>H<TH>S/N<TH>Target<BR>Type<TH>Target & Data Flags<TH>Spectrum Plot\n')
#    objhtml.write('<TR><TH>Fiber<TH>APOGEE ID<TH>H<TH>H - obs<TH>S/N<TH>Target<BR>Type<TH>Target & Data Flags<TH>Spectrum Plot\n')

    cfile = open(plotsdir+htmlfile+'.csh','w')
    for j in range(nfiber):
        jdata = data[j]
        fiber = jdata['FIBERID']
        if fiber > 0:
            cfiber = str(fiber).zfill(3)

            objid = jdata['OBJECT']
            objtype = jdata['OBJTYPE']
            hmag = jdata['HMAG']
            chmag = str("%.3f" % round(jdata['HMAG'],2))
    #        magdiff = str("%.2f" % round(plSum2['obsmag'][j][0][1] -hmag,2))
            cra = str("%.5f" % round(jdata['RA'],5))
            cdec = str("%.5f" % round(jdata['DEC'],5))
            txt1 = '<BR><A HREF="http://simbad.u-strasbg.fr/simbad/sim-coo?Coord='+cra+'+'+cdec+'&CooFrame=FK5&CooEpoch=2000'
            txt2 = '&CooEqui=2000&CooDefinedFrames=none&Radius=10&Radius.unit=arcsec&submit=submit+query&CoordList=">SIMBAD link</A>'
            simbadlink = txt1 + txt2

            # Establish html table row background color and spectrum plot color
            color = 'white'
            if (objtype == 'SPECTROPHOTO_STD') | (objtype == 'HOT_STD'): color = 'plum'
            if objtype == 'SKY': color = 'silver'
            pcolor = 'k'
            if objtype == 'SKY': pcolor = 'firebrick'

            # Get target flag strings
            if 'apogee' in survey:
                targflagtxt = bitmask.targflags(jdata['TARGET1'],jdata['TARGET2'],jdata['TARGET3'],
                                                jdata['TARGET4'],survey=survey)
            else:
                if j == 0: print("PROBLEM!!! Need to update bitmask.py to handle SDSSV_APOGEE_TARGET0.")
                targflagtxt = 'to be added ASAP'
                #targflagtxt = bitmask.targflags(jdata['SDSSV_APOGEE_TARGET0'], survey=survey)
            if targflagtxt[-1:] == ',': targflagtxt = targflagtxt[:-1]
            targflagtxt = targflagtxt.replace(' gt ','>').replace(',','<BR>')

            # Find apVisit file
            visitfile = load.filename('Visit', plate=int(plate), mjd=mjd, fiber=fiber)
            visitfilebase = os.path.basename(load.filename('Visit', plate=int(plate), mjd=mjd, fiber=fiber))
            vplotfile = visitfile.replace('.fits','.jpg')

            if os.path.exists(visitfile) is False:
                print("PROBLEM!!! "+visitfilebase+" not found!")
            else:
                visithdr = fits.getheader(visitfile)
                starflagtxt = bitmask.StarBitMask().getname(visithdr['STARFLAG']).replace(',','<BR>')
                snratio = str("%.2f" % round(visithdr['SNR'],2))

                # column 1
                objhtml.write('<TR><TD BGCOLOR='+color+'><A HREF=../'+visitfile+'>'+cfiber+'</A>\n')

                # column 2
                objhtml.write('<TD BGCOLOR='+color+'>'+objid+'\n')
                objhtml.write(simbadlink+'\n')
                objhtml.write('<BR><a href=../plots/'+vplotfile+'>apVisit file</A>\n')
                objhtml.write('<BR>apStar file\n')

                if objtype != 'SKY':
                    objhtml.write('<TD BGCOLOR='+color+' align ="right">'+chmag+'\n')
                    #objhtml.write('<TD BGCOLOR='+color+' align ="right">'+magdiff+'\n')
                    objhtml.write('<TD BGCOLOR='+color+' align ="right">'+snratio+'\n')
                else:
                    objhtml.write('<TD BGCOLOR='+color+'>---\n')
                    #objhtml.write('<TD BGCOLOR='+color+'>---\n')
                    objhtml.write('<TD BGCOLOR='+color+'>---\n')

                if objtype == 'SKY': 
                    objhtml.write('<TD BGCOLOR='+color+'>SKY\n')
                else:
                    if (objtype == 'SPECTROPHOTO_STD') | (objtype == 'HOT_STD'):
                        objhtml.write('<TD BGCOLOR='+color+'>TEL\n')
                    else:
                        objhtml.write('<TD BGCOLOR='+color+'>SCI\n')

                objhtml.write('<TD BGCOLOR='+color+' align="left">'+targflagtxt+'\n')
                objhtml.write('<BR><BR>'+starflagtxt+'\n')

                # Spectrum Plots
                plotfile = 'apPlate-'+plate+'-'+mjd+'-'+cfiber+'.png'
                objhtml.write('<TD BGCOLOR='+color+'><A HREF=../plots/'+plotfile+' target="_blank"><IMG SRC=../plots/'+plotfile+' WIDTH=1000></A>\n')
                if makeSpectrumPlots is True:
#                if (makeSpectrumPlots is True) & (j<60):
                    print("Making "+plotfile)

                    lwidth = 1.5;   axthick = 1.5;   axmajlen = 6;   axminlen = 3.5
                    xmin = 15120;   xmax = 16960;    xspan = xmax - xmin

                    FluxB = apPlate['a'][1].data[300-fiber,:]
                    FluxG = apPlate['b'][1].data[300-fiber,:]
                    FluxR = apPlate['c'][1].data[300-fiber,:]
                    WaveB = apPlate['a'][4].data[300-fiber,:]
                    WaveG = apPlate['b'][4].data[300-fiber,:]
                    WaveR = apPlate['c'][4].data[300-fiber,:]

                    Flux = np.concatenate([FluxB, FluxG, FluxR])
                    Wave = np.concatenate([WaveB, WaveG, WaveR])

                    # Establish Ymax
                    ymxsec1, = np.where((Wave > 15150) & (Wave < 15180))
                    ymxsec2, = np.where((Wave > 15900) & (Wave < 15950))
                    ymxsec3, = np.where((Wave > 16905) & (Wave < 16940))
                    if (len(ymxsec1) == 0) | (len(ymxsec2) == 0) | (len(ymxsec3) == 0): 
                        print("Problem with fiber "+cfib+". Not Plotting.")
                    else:
                        tmpF = convolve(Flux,Box1DKernel(11))
                        ymx1 = np.nanmax(tmpF[ymxsec1])
                        ymx2 = np.nanmax(tmpF[ymxsec2])
                        ymx3 = np.nanmax(tmpF[ymxsec3])
                        ymx = np.nanmax([ymx1,ymx2,ymx3])
                        ymin = 0
                        yspn = ymx-ymin
                        ymax = ymx + (yspn * 0.15)
                        # Establish Ymin
                        ymn = np.nanmin(tmpF)
                        if ymn > 0: 
                            yspn = ymx - ymn
                            ymin = ymn - (yspn * 0.15)
                            ymax = ymx + (yspn * 0.15)

                        fig=plt.figure(figsize=(28,6))
                        ax1 = plt.subplot2grid((1,1), (0,0))
                        ax1.tick_params(reset=True)
                        ax1.set_xlim(xmin,xmax)
                        ax1.set_ylim(ymin,ymax)
                        ax1.xaxis.set_major_locator(ticker.MultipleLocator(200))
                        ax1.minorticks_on()
                        ax1.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
                        ax1.tick_params(axis='both',which='major',length=axmajlen)
                        ax1.tick_params(axis='both',which='minor',length=axminlen)
                        ax1.tick_params(axis='both',which='both',width=axwidth)
                        ax1.set_xlabel(r'Wavelength [$\rm \AA$]')
                        ax1.set_ylabel(r'Flux')

                        ax1.plot(WaveB[np.argsort(WaveB)], FluxB[np.argsort(WaveB)], color=pcolor)
                        ax1.plot(WaveG[np.argsort(WaveG)], FluxG[np.argsort(WaveG)], color=pcolor)
                        ax1.plot(WaveR[np.argsort(WaveR)], FluxR[np.argsort(WaveR)], color=pcolor)

                        fig.subplots_adjust(left=0.06,right=0.995,bottom=0.16,top=0.97,hspace=0.2,wspace=0.0)
                        plt.savefig(plotsdir+plotfile)
                    plt.close('all')

    objhtml.close()
    cfile.close()

    plt.ion()

    print("--------------------------------------------------------------------")
    print("Done with MAKEOBJHTML for plate "+plate+", mjd "+mjd)
    print("--------------------------------------------------------------------\n")

#    if starfiber is None:
#        txt1 = 'Left plots: red are targets, blue are telluric. Observed mags are calculated '
#        txt2 = 'from median value of green chip. Zeropoint gives overall throughput: bigger number is more throughput.'
#        html.write(txt1+txt2+'\n')

#        txt1 = '<br>First spatial plots: circles are objects, squares are tellurics, crosses are sky fibers. '
#        txt2 = 'Colors give deviation of observed mag from expected 2MASS mag using the median zeropoint; red is brighter'
#        html.write(txt1+txt2+'\n')

#        txt1 = '<br>Second spatial plots: circles are sky fibers. '
#        txt2 = 'Colors give sky line brightness relative to plate median sky line brightness'
#        html.write(txt1+txt2+'\n')

#        html.write('<TABLE BORDER=2>\n')
#        html.write('<TR><TD>Frame<TD>Nreads<TD>Zeropoints<TD>Mag plots\n')
#        html.write('<TH>Spatial mag deviation\n')
#        html.write('<TH>Spatial sky 16325A emission deviations (filled: sky, open: star)\n')
#        html.write('<TH>Spatial sky continuum emission \n')
#        html.write('<TH>Spatial sky telluric CO2 absorption deviations (filled: H &lt 10) \n')
#    else:
#        html.write('<TABLE BORDER=2>\n')
#        html.write('<TR><TH>Frame<TH>Fiber<TH>Star\n')

#    unplugged, = np.where(fiber['fiberid'] < 0)
#    nunplugged = len(unplugged)
#    if flat is not None:
#        fiber['hmag'] = 12
#        fiber['object'] = 'FLAT'


'''-----------------------------------------------------------------------------------------'''
''' MAKECALFITS: Make FITS file for cals (lamp brightness, line widths, etc.)               '''
'''-----------------------------------------------------------------------------------------'''
def makeCalFits(load=None, ims=None, mjd=None, instrument=None):

    print("--------------------------------------------------------------------")
    print("Running MAKECALFITS for plate "+plate+", mjd "+mjd)
    print("--------------------------------------------------------------------\n")

    n_exposures = len(ims)

    nlines = 2
    chips=np.array(['a','b','c'])
    nchips = len(chips)

    tharline = np.array([[940.,1128.,1130.],[1724.,623.,1778.]])
    uneline =  np.array([[603.,1213.,1116.],[1763.,605.,1893.]])

    if instrument == 'apogee-s': tharline = np.array([[944.,1112.,1102.],[1726.,608.,1745.]])
    if instrument == 'apogee-s':  uneline = np.array([[607.,1229.,1088.],[1765.,620.,1860.]])

    fibers = np.array([10,80,150,220,290])
    nfibers = len(fibers)

    # Make output structure.
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

    struct = np.zeros(n_exposures, dtype=dt)

    # Loop over exposures and get 1D images to fill structure.
    # /uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/apogee/spectro/redux/t14/exposures/apogee-n/57680/ap1D-21180073.fits
    for i in range(n_exposures):
        oneD = load.ap1D(ims[i])
        oneDhdr = oneD['a'][0].header

        if type(oneD)==dict:
            struct['NAME'][i] =    ims[i]
            struct['MJD'][i] =     mjd
            struct['JD'][i] =      oneDhdr['JD-MID']
            struct['NFRAMES'][i] = oneDhdr['NFRAMES']
            struct['NREAD'][i] =   oneDhdr['NREAD']
            struct['EXPTIME'][i] = oneDhdr['EXPTIME']
            struct['QRTZ'][i] =    oneDhdr['LAMPQRTZ']
            struct['THAR'][i] =    oneDhdr['LAMPTHAR']
            struct['UNE'][i] =     oneDhdr['LAMPUNE']

            # Quartz exposures.
            if struct['QRTZ'][i]==1: struct['FLUX'][i] = np.median(oneD['a'][1].data, axis=0)

            # Arc lamp exposures.
            if (struct['THAR'][i] == 1) | (struct['UNE'][i] == 1):
                if struct['THAR'][i] == 1: line = tharline
                if struct['THAR'][i] != 1: line = uneline

                struct['LINES'][i] = line

                nlines = 1
                if line.shape[0]!=1: nlines = line.shape[1]

                for iline in range(nlines):
                    for ichip in range(nchips):
                        print("Calling appeakfit... no, not really because it's a long IDL code.")
                        ### NOTE:the below does not work yet... maybe use findlines instead?
                        # https://github.com/sdss/apogee/blob/master/pro/apogeereduce/appeakfit.pro
                        # https://github.com/sdss/apogee/blob/master/python/apogee/apred/wave.py

#;                        APPEAKFIT,a[ichip],linestr,fibers=fibers,nsigthresh=10
                        linestr = wave.findlines(oneD, rows=fibers, lines=line)
                        linestr = wave.peakfit(oneD[chips[ichip]][1].data)

                        for ifiber in range(nfibers):
                            fibers = fibers[ifiber]
                            j = np.where(linestr['FIBER'] == fiber)
                            nj = len(j)
                            if nj>0:
                                junk = np.nanmin(np.absolute(linestr['GAUSSX'][j] - line[ichip,iline]))
                                jline = np.argmin(np.absolute(linestr['GAUSSX'][j] - line[ichip,iline]))
                                struct['GAUSS'][:,ifiber,ichip,iline][i] = linestr['GPAR'][j][jline]
                                sz = a['WCOEF'][ichip].shape
                                if sz[0] == 2:
                                    ### NOTE:the below has not been tested
#;                                    struct['WAVE'][i][ifiber,ichip,iline] = pix2wave(linestr['GAUSSX'][j][jline],oneD['WCOEF'][ichip][fiber,:])
                                    pix = linestr['GAUSSX'][j][jline]
                                    wave0 = a['WCOEF'][ichip][fiber,:]
                                    struct['WAVE'][i][ifiber,ichip,iline] = wave.pix2wave(pix, wave0)

                                struct['FLUX'][i][fiber,ichip] = linestr['SUMFLUX'][j][jline]
        else:
            print("type(1D) does not equal dict. This is probably a problem.")

    outfile = load.filename('QAcal', plate=int(plate), mjd=mjd) 
    Table(struct).write(outfile)

    print("--------------------------------------------------------------------")
    print("Done with MAKECALFITS for plate "+plate+", mjd "+mjd)
    print("Made "+outfile)
    print("--------------------------------------------------------------------\n")


'''-----------------------------------------------------------------------------------------'''
''' MAKEDARKFITS: Make FITS file for darks (get mean/stddev of column-medianed quadrants)   '''
'''-----------------------------------------------------------------------------------------'''
def makeDarkFits(load=None, planfile=None, ims=None, mjd=None):

    print("--------------------------------------------------------------------")
    print("Running MAKEDARKFITS for plate "+plate+", mjd "+mjd)
    print("--------------------------------------------------------------------\n")

    n_exposures = len(ims)

    chips=np.array(['a','b','c'])
    nchips = len(chips)
    nquad = 4

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

    struct = np.zeros(n_exposures, dtype=dt)

    # Loop over exposures and get 2D images to fill structure.
    # /uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/apogee/spectro/redux/t14/exposures/apogee-n/57680/ap2D-21180073.fits
    for i in range(n_exposures):
        twoD = load.ap2D(ims[i])
        twoDhdr = twoD['a'][0].header

        if type(twoD) == dict:
            struct['NAME'][i] =    ims[i]
            struct['MJD'][i] =     mjd
            struct['JD'][i] =      twoDhdr['JD-MID']
            struct['NFRAMES'][i] = twoDhdr['NFRAMES']
            struct['NREAD'][i] =   twoDhdr['NREAD']
            struct['EXPTIME'][i] = twoDhdr['EXPTIME']
            struct['QRTZ'][i] =    twoDhdr['LAMPQRTZ']
            struct['THAR'][i] =    twoDhdr['LAMPTHAR']
            struct['UNE'][i] =     twoDhdr['LAMPUNE']

            for ichip in range(nchips):
                i1 = 10
                i2 = 500
                for iquad in range(quad):
                    sm = np.median(twoD[chips[ichip]][1].data[10:2000, i1:i2], axis=0)
                    struct['MEAN'][i,ichip,iquad] = np.mean(sm)
                    struct['SIG'][i,ichip,iquad] = np.std(sm)
                    i1 += 512
                    i2 += 512
        else:
            print("type(2D) does not equal dict. This is probably a problem.")

    outfile = load.filename('QAcal', plate=int(plate), mjd=mjd).replace('apQAcal','apQAdarkflat')
    Table(struct).write(outfile)

    print("--------------------------------------------------------------------")
    print("Done with MAKEDARKFITS for plate "+plate+", mjd "+mjd)
    print("Made "+outfile)
    print("--------------------------------------------------------------------\n")



'''-----------------------------------------------------------------------------------------'''
''' GETFLUX: Translation of getflux.pro                                                     '''
'''-----------------------------------------------------------------------------------------'''
def getflux(d=None, skyline=None, rows=None):

    chips = np.array(['a','b','c'])
    nnrows = len(rows)

    ### NOTE:pretty sure that [2047,150] subscript won't work, but 150,2057 will. Hoping for the best.
    if skyline['W1'] > d['a'][4].data[150,2047]:
        ichip = 0
    else:
        if skyline['W1'] > d['b'][4].data[150,2047]:
            ichip = 1
        else:
            ichip = 2

    cont = np.zeros(nnrows)
    line = np.zeros(nnrows)
    nline = np.zeros(nnrows)

    for i in range(nnrows):
        wave = d[chips[ichip]][4].data[rows[i],:]
        flux = d[chips[ichip]][1].data

        icont, = np.where(((wave > skyline['C1']) & (wave < skyline['C2'])) | 
                          ((wave < skyline['C3']) & (wave < skyline['C4'])))

        #import pdb; pdb.set_trace()
        if len(icont) >= 0: cont[i] = np.median(flux[rows[i],icont])

        iline, = np.where((wave > skyline['W1']) & (wave < skyline['W2']))

        if len(iline) >= 0:
            line[i] = np.nansum(flux[rows[i],iline])
            nline[i] = np.nansum(flux[rows[i],iline] / flux[rows[i],iline])

    skylineFlux = line - (nline * cont)
    if skyline['TYPE'] == 0: skylineFlux /= cont

    return skylineFlux



