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
from apogee_drp.utils import plan,apload,yanny,plugmap,platedata,bitmask,peakfit,colorteff
from apogee_drp.apred import wave,monitor
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
from scipy import interpolate
import datetime

cspeed = 299792.458e0

sdss_path = path.Path()

sort_table_link = 'https://www.kryogenix.org/code/browser/sorttable/sorttable.js'

#matplotlib.use('agg')

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

outdir = '/uufs/chpc.utah.edu/common/home/u0955897/projects/snr/'

###################################################################################################
'''APQAALL: Wrapper for running apqa for ***ALL*** plates '''
def apqaALL(mjdstart='59560', observatory='apo', apred='daily'):

    # Establish telescope
    telescope = observatory + '25m'

    apodir = os.environ.get('APOGEE_REDUX') + '/'
    mjdDirs = np.array(glob.glob(apodir + apred + '/visit/' + telescope + '/*/*/*'))
    ndirs = len(mjdDirs)
    allmjd = np.empty(ndirs).astype(str)
    for i in range(ndirs): allmjd[i] = mjdDirs[i].split('/')[-1]
    gd, = np.where(allmjd != 'plots')
    umjd = np.unique(allmjd[gd])
    gd, = np.where(umjd == mjdstart)
    umjd = umjd[gd[0]:]
    umjd = umjd[::-1]
    nmjd = len(umjd)
    print("Running apqaMJD on " + str(nmjd) + " MJDs")

    out = open(outdir + 'apogeeSNR-FPS.dat')
    out.write('EXPOSURE           SNR_B      SNR_G     SNR_R\n')
    for ii in range(nmjd):
        if umjd[ii][0:1] != 'a':
            x = apqaMJD(mjd=umjd[ii], observatory=observatory, apred=apred, makeplatesum=makeplatesum)
            p1 = str(int(round(x[0])))
            p2 = str("%.3f" % round(x[1],3)).rjust(10)
            p3 = str("%.3f" % round(x[2],3)).rjust(10)
            p4 = str("%.3f" % round(x[3],3)).rjust(10)
            out.write(p1+'  '+p2+p3+p4+'\n')

    out.close()


###################################################################################################
'''APQAMJD: Wrapper for running apqa for all plates on an mjd '''
def apqaMJD(mjd='59146', observatory='apo', apred='daily', makeplatesum=True):

    # Establish telescope and instrument
    telescope = observatory + '25m'
    instrument = 'apogee-n'
    if observatory == 'lco': instrument = 'apogee-s'
    load = apload.ApLoad(apred=apred, telescope=telescope)

    # Find the list of plan files
    apodir = os.environ.get('APOGEE_REDUX')+'/'
    planlist = apodir + apred + '/log/'+observatory+'/' + str(mjd) + '.plans'
    if os.path.exists(planlist) == False:
        print('Uh-oh, not finding ' + planlist + '\n')
        return
    plans = open(planlist, 'r')
    plans = plans.readlines()
    nplans = len(plans)

    # Find the plan files pertaining to science data
    sciplans = []
    calplans = []
    darkplans = []
    for i in range(nplans):
        tmp = plans[i].split('-')
        if tmp[0] == 'apPlan': 
            if 'sky' not in plans[i]: sciplans.append(plans[i].replace('\n',''))
        if tmp[0] == 'apCalPlan': calplans.append(plans[i].replace('\n',''))
        if tmp[0] == 'apDarkPlan': darkplans.append(plans[i].replace('\n',''))
    sciplans = np.array(sciplans)
    nsciplans = len(sciplans)
    calplans = np.array(calplans)
    ncalplans = len(calplans)
    darkplans = np.array(darkplans)
    ndarkplans = len(darkplans)

    # Run apqa on the science data plans
    print("Running APQAMJD for " + str(nsciplans) + " plates observed on MJD " + mjd + "\n")
    for i in range(nsciplans):
        # Get the plate number and mjd
        tmp = sciplans[i].split('-')
        plate = tmp[1]
        mjd = tmp[2].split('.')[0]

        # Load the plan file
        if int(mjd)>59556:
            fps = True
        else:
            fps = False
        planfile = load.filename('Plan', plate=int(plate), mjd=mjd, fps=fps)
        planstr = plan.load(planfile, np=True)

        # Get array of object exposures and find out how many are objects.
        flavor = planstr['APEXP']['flavor']
        all_ims = planstr['APEXP']['name']
        gd,= np.where(flavor == 'object')
        n_ims = len(gd)
        if n_ims > 0:
            ims = all_ims[gd]
            # Make an array indicating which exposures made it to apCframe
            # 0 = not reduced, 1 = reduced
            imsReduced = np.zeros(n_ims)
            for j in range(n_ims):
                cframe = load.filename('Cframe', plate=int(plate), mjd=mjd, num=ims[j], chips=True, fps=fps)
                if os.path.exists(cframe.replace('Cframe-','Cframe-a-')): imsReduced[j] = 1
            good, = np.where(imsReduced == 1)
            if len(good) < 1:
                    continue
                    #sys.exit("PROBLEM!!! 1D files not found for plate " + plate + ", MJD " + mjd + "\n")

        # Only run makemasterqa, makenightqa, and monitor after the last plate on this mjd
        if i < nsciplans-1:
            x = apqa(plate=plate, mjd=mjd, apred=apred, makeplatesum=makeplatesum)
        else:
            x = apqa(plate=plate, mjd=mjd, apred=apred, makeplatesum=makeplatesum)
        
    print("Done with APQAMJD for " + str(nsciplans) + " plates observed on MJD " + mjd + "\n")

###################################################################################################
'''APQA: Wrapper for running QA subprocedures on a plate mjd '''
def apqa(plate='15000', mjd='59146', telescope='apo25m', apred='daily', makeplatesum=True):

    start_time = time.time()

    print("Starting APQA for plate " + plate + ", MJD " + mjd + "\n")

    if int(mjd)>59556:
        fps = True
    else:
        fps = False

    # Use telescope, plate, mjd, and apred to load planfile into structure.
    load = apload.ApLoad(apred=apred, telescope=telescope)
    planfile = load.filename('Plan', plate=int(plate), mjd=mjd, fps=fps)
    planstr = plan.load(planfile, np=True)

    print(os.path.basename(planfile))

    # Get field name
    tmp = planfile.split(telescope+'/')
    field = tmp[1].split('/')[0]

    # Get values from plan file.
    fixfiberid = planstr['fixfiberid']
    badfiberid = planstr['badfiberid']
    platetype =  planstr['platetype']
    plugmap =    planstr['plugmap']
    fluxid =     planstr['fluxid']
    instrument = planstr['instrument']
    survey =     planstr['survey']

    # Establish directories.
    datadir = {'apo25m':os.environ['APOGEE_DATA_N'], 'apo1m':os.environ['APOGEE_DATA_N'],
               'lco25m':os.environ['APOGEE_DATA_S']}[telescope]

    apodir =     os.environ.get('APOGEE_REDUX')+'/'
    spectrodir = apodir + apred + '/'
    caldir =     spectrodir + 'cal/'
    expdir =     spectrodir + 'exposures/' + instrument + '/'

    # Get array of object exposures and find out how many are objects.
    flavor = planstr['APEXP']['flavor']
    all_ims = planstr['APEXP']['name']

    gd,= np.where(flavor == 'object')
    n_ims = len(gd)

    if n_ims > 0:
        ims = all_ims[gd]
        # Make an array indicating which exposures made it to apCframe
        # 0 = not reduced, 1 = reduced
        imsReduced = np.zeros(n_ims)
        for i in range(n_ims):
            cframe = load.filename('Cframe', field=field, plate=int(plate), mjd=mjd, num=ims[i], chips=True)
            if os.path.exists(cframe.replace('Cframe-','Cframe-a-')): imsReduced[i] = 1
        good, = np.where(imsReduced == 1)
        if len(good) > 0:
            ims = ims[good]
        else:
            print("PROBLEM!!! apCframe files not found for plate " + plate + ", MJD " + mjd + ". Skipping.\n")
            return
    else:
        print("PROBLEM!!! no object images found for plate " + plate + ", MJD " + mjd + ". Skipping.\n")
        return

    # Get mapper data.
    mapper_data = {'apogee-n':os.environ['MAPPER_DATA_N'],'apogee-s':os.environ['MAPPER_DATA_S']}[instrument]

    # Normal plates:
    if platetype == 'normal': 

        # Make the apPlateSum file if it doesn't already exist.
        qcheck = 'good'
        platesum = load.filename('PlateSum', plate=int(plate), mjd=mjd, fps=fps)
        if makeplatesum == True:
            qcheck = makePlateSum(load=load, plate=plate, mjd=mjd, telescope=telescope, field=field,
                             instrument=instrument, ims=ims, imsReduced=imsReduced,
                             plugmap=plugmap, survey=survey, mapper_data=mapper_data, 
                             apred=apred, onem=None, starfiber=None, starnames=None, 
                             starmag=None,flat=None, fixfiberid=fixfiberid, badfiberid=badfiberid,
                             clobber=clobber)

            tmpims = np.array([0,ims[0]])
            qcheck = makePlateSum(load=load, plate=plate, mjd=mjd, telescope=telescope, field=field,
                             instrument=instrument, ims=tmpims, imsReduced=imsReduced,
                             plugmap=plugmap, survey=survey, mapper_data=mapper_data, 
                             apred=apred, onem=None, starfiber=None, starnames=None, 
                             starmag=None,flat=None, fixfiberid=fixfiberid, badfiberid=badfiberid,
                             clobber=clobber)

    runtime = str("%.2f" % (time.time() - start_time))
    print("Done with APQA for plate " + plate + ", MJD " + mjd + " in " + runtime + " seconds.\n")

###################################################################################################
''' MAKEPLATESUM: Plotmag translation '''
def makePlateSum(load=None, telescope=None, ims=None, imsReduced=None, plate=None, mjd=None, field=None,
                 instrument=None, clobber=True, makeqaplots=None, plugmap=None, survey=None,
                 mapper_data=None, apred=None, onem=None, starfiber=None, starnames=None, 
                 starmag=None, flat=None, fixfiberid=None, badfiberid=None):

    chips = np.array(['a','b','c'])
    nchips = len(chips)
    
    apodir = os.environ.get('APOGEE_REDUX')+'/'

    if int(mjd)>59556:
        fps = True
    else:
        fps = False

    platesumfile = load.filename('PlateSum', plate=int(plate), mjd=mjd, fps=fps)
    platesumbase = os.path.basename(platesumfile)
    
    print("----> makePlateSum: Making "+platesumbase)

    # Directory where sn*dat and altsn*dat files are stored.
    sntabdir = apodir + apred + '/visit/' + telescope + '/' + field + '/' + plate + '/' + mjd + '/'
    if os.path.exists(sntabdir) == False: os.makedirs(sntabdir)

    # Get the fiber association for this plate. Also get some other values
    if ims[0] == 0:
        n_exposures = 1
        onedfile = load.filename('1D', num=ims[1], mjd=mjd, chips=True)
    else:
        n_exposures = len(ims)
        onedfile = load.filename('1D', num=ims[0], mjd=mjd, chips=True)

    tothdr = fits.getheader(onedfile.replace('1D-','1D-a-'))
    ra = tothdr['RADEG']
    dec = tothdr['DECDEG']
    DateObs = tothdr['DATE-OBS']


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
    if ntelluric < 1: print("----> makePlateSum: PROBLEM!!! No tellurics found.")

    fiberobj, = np.where((fibtype == 'STAR_BHB') | (fibtype == 'STAR') | (fibtype == 'EXTOBJ') | (fibtype == 'OBJECT'))
    nobj = len(fiberobj)
    obj = rows[fiberobj]
    if nobj < 1: print("----> makePlateSum: PROBLEM!!! No science objects found.")

    fibersky, = np.where(fibtype == 'SKY')
    nsky = len(fibersky)
    sky = rows[fibersky]
    if nsky < 1: print("----> makePlateSum: PROBLEM!!! No skies found.")

    if (nobj < 1) & (ntelluric < 1) & (nsky < 1): 
        print("----> makePlateSum: PROBLEM!!! No science objects, tellurics, nor skies found. Skipping this visit.")
        #return 'bad'
    else:
        fiberstar = np.concatenate([fiberobj,fibertelluric])
        nstar = len(fiberstar)
        star = rows[fiberstar]

    # Loop through all the images for this plate, and make the plots.
    # Load up and save information for this plate in a FITS table.
    allsky =     np.zeros((n_exposures,3), dtype=np.float64)
    allzero =    np.zeros((n_exposures,3), dtype=np.float64)
    allzerorms = np.zeros((n_exposures,3), dtype=np.float64)

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
                   ('CART',      np.str, 30),
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

    platetab = np.zeros(n_exposures,dtype=dt)

    platetab['PLATE'] =     int(plate)
    platetab['TELESCOPE'] = telescope
    platetab['HA'] =        0.0
    platetab['DESIGN_HA'] = -99.0
    platetab['PLUGID'] =    plugmap
    platetab['MJD'] =       mjd
    #platetab['MOONDIST'] =  moondist
    #platetab['MOONPHASE'] = moonphase

    #pdb.set_trace()
    # Loop over the exposures.
    for i in range(n_exposures):
        if ims[0] == 0: 
            pfile = os.path.basename(load.filename('Plate', plate=int(plate), mjd=mjd, chips=True, fps=fps))
            dfile = load.filename('Plate',  plate=int(plate), mjd=mjd, chips=True, fps=fps)
            d = load.apPlate(int(plate), mjd) 
            cframe = load.apPlate(int(plate), mjd)
            if type(d)!=dict: print("----> makePlateSum: Problem with apPlate!")
            if os.path.exists(dfile.replace('apPlate-','apPlate-a-')):
                dhdr = fits.getheader(dfile.replace('apPlate-','apPlate-a-'))
            else:
                print("----> makePlateSum: Problem with apPlate!")
                return 'bad'
        else:
            pfile = os.path.basename(load.filename('1D', num=ims[i], mjd=mjd, chips=True))
            dfile = load.filename('1D', num=ims[i], mjd=mjd, chips=True)
            d = load.ap1D(ims[i])
            cframe = load.apCframe(field, int(plate), mjd, ims[i])
            if type(d)!=dict: print("----> makePlateSum: Problem with ap1D!")
            if os.path.exists(dfile.replace('1D-','1D-a-')):
                dhdr = fits.getheader(dfile.replace('1D-','1D-a-'))
            else:
                print("----> makePlateSum: Problem with ap1D!")
                return 'bad'

        ind = 1
        if len(ims) < 2: ind = 0
        cframefile = load.filename('Cframe', plate=int(plate), mjd=mjd, num=ims[ind], chips='c', fps=fps)
        cframehdr = fits.getheader(cframefile.replace('Cframe-','Cframe-a-'))
        pfile = pfile.replace('.fits','')

        # Get moon distance and phase.
        dateobs = dhdr['DATE-OBS']
        tt = Time(dateobs, format='fits')
        moonpos = get_moon(tt)
        moonra = moonpos.ra.deg
        moondec = moonpos.dec.deg
        try:
            c1 = SkyCoord(ra * astropyUnits.deg, dec * astropyUnits.deg)
            c2 = SkyCoord(moonra * astropyUnits.deg, moondec * astropyUnits.deg)
            sep = c1.separation(c2)
            moondist = sep.deg
            moonphase = moon_illumination(tt)
        except:
            moondist = 0
            moonphase = 0

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
        if (nobj > 5) & (ntelluric > 5) & (nsky > 5): 
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

        gdHmag, = np.where((fiber['hmag'] > 5) & (fiber['hmag'] < 20))
        zero = 0
        zerorms = 0.
        zeronorm = 0
        faint = -1
        nfaint = 0
        achievedsn = np.zeros(nchips)
        achievedsnc = np.zeros(nchips)
        #achievedsnt = np.zeros(nchips)
        altsn = np.zeros(nchips)
        nsn = 0

        # Only run this part if there some stars with reasonable H mag
        if len(gdHmag) >= 5:
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
                    try:
                        bright, = np.where(fiber['hmag'] < 12)
                        hmax = np.nanmax(fiber['hmag'][bright])
                        snstars, = np.where((fiber['hmag'] > hmax-0.2) & (fiber['hmag'] <= hmax))
                        nsn = len(snstars)
                        scale = np.sqrt(10**(0.4 * (hmax - 12.2)))
                    except:
                        print("----> makePlateSum: No S/N stars! Skipping.")
                        #return 'bad'

                achievedsn = np.nanmedian(sn[snstars,:], axis=0) * scale
                #gd, = np.where(snt > 0)
                #achievedsnt = np.nanmedian(snt[:], axis=0) * scale

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
            if np.nanmedian(obs[fibersky,ichip]) > 0:
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
                gcam = fits.getdata(gcamfile)
                jcam, = np.where((gcam['MJD'] > mjdstart) & (gcam['MJD'] < mjdend))
                nj = len(jcam)
            if nj > 1: 
                fwhm = np.median(gcam['FWHM_MEDIAN'][jcam]) 
                gdrms = np.median(gcam['GDRMS'][jcam])
            else:
                fwhm = -1.
                gdrms = -1.
                if i == 0: print("----> makePlateSum: Problem! No matching mjd range in gcam.")
        else:
            fwhm = -1
            gdrms = -1
            exptime=-9.999

        secz = 0
        seeing = 0
        if ims[0] != 0: 
            secz = 1. / np.cos((90. - dhdr['ALT']) * (math.pi/180.))
            seeing = dhdr['SEEING']
            if str(seeing).lower().find('nan') != -1: seeing=np.nan
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
            tellfile = load.filename('Tellstar', plate=int(plate), mjd=mjd, fps=fps)
            if os.path.exists(tellfile):
                try:
                    telstr = fits.getdata(tellfile)
                except:
                    if i == 0: print("----> makePlateSum: PROBLEM!!! Error reading apTellstar file: "+os.path.basename(tellfile))
                else:
                    telstr = fits.getdata(tellfile)
                    jtell, = np.where(telstr['IM'] == ims[i])
                    ntell = len(jtell)
                    if ntell > 0: platetab['TELLFIT'][i] = telstr['FITPARS'][jtell]
            else:
                print("----> makePlateSum: PROBLEM!!! "+os.path.basename(tellfile)+" does not exist.")

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
        #platetab['SNT'][i] =       achievedsnt
        if ntelluric > 0: platetab['SNRATIO'][i] = np.nanmedian(snt[fibertelluric,1] / snc[fibertelluric,1])
        platetab['MOONDIST'][i] =  moondist
        platetab['MOONPHASE'][i] = moonphase

        for j in range(len(fiber)):
            fiber['sn'][j][i,:] = sn[j,:]
            fiber['obsmag'][j][i,:] = (-2.5 * np.log10(obs[j,:])) + zero

    pdb.set_trace()
    # Linear fit to log(snr) vs. Hmag for ALL objects
    gdall, = np.where((fiber['hmag'] > 4) & (fiber['hmag'] < 20) & (fiber['sn'] > 0))
    if len(gdall)>2:
        coefall = np.polyfit(fiber[gdall]['hmag'],np.log10(fiber[gdall]['sn']),1)
    else:
        coefall = np.zeros(2,float)+np.nan
    # Linear fit to log(S/N) vs. H for 10<H<11.5
    gd, = np.where((fiber['hmag']>=10.0) & (fiber['hmag']<=11.5) & (fiber['sn'] > 0))
    if len(gd)>2:
        coef = np.polyfit(fiber[gd]['hmag'],np.log10(fiber[gd]['sn']),1)
    else:
        coef = np.zeros(2,float)+np.nan
    if len(gd)>2:
        snr_fid = 10**np.polyval(coef,hfid)
    elif len(gdall)>2:
        snr_fid = 10**np.polyval(coefall,hfid)
    else:
        snr_fid = np.mean(fiber['sn'])




    print("----> makePlateSum: Done with plate "+plate+", MJD "+mjd+"\n")
    return ims[i],achievedsn
