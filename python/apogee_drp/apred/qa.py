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

###################################################################################################
'''DOSTARS: Wrapper for running makeStarHTML and apStar plots on unique fields only '''
def dostars(mjdstart=None, observatory='apo', apred='daily', dohtml=True, doplots=True, clobber=True):

    # Establish telescope and load
    telescope = observatory + '25m'

    # Find unique fields and run star stuff on them
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
    if mjdstart is not None:
        gd, = np.where(allmjd.astype(int) > mjdstart)
        allfield = allfield[gd]
        allplate = allplate[gd]
        allmjd = allmjd[gd]
    ufield, ufieldind = np.unique(allfield, return_index=True)
    umjd = allmjd[ufieldind]
    uplate = allplate[ufieldind]
    nfields = len(ufield)
    print("Running dostars on " + str(nfields) + " unique fields...\n")

    for i in range(nfields):
        q = apqa(plate=uplate[i], mjd=umjd[i], telescope=telescope, apred=apred, makeplatesum=False,
                 makeobshtml=False, makeobsplots=False, makevishtml=False, makevisplots=False,
                 makestarhtml=dohtml, makestarplots=doplots, makenightqa=False, makemasterqa=False,
                 clobber=clobber)

    print("\nDone with dostars for " + str(nfields) + " unique fields...")

###################################################################################################
'''APQAALL: Wrapper for running apqa for ***ALL*** plates '''
def apqaALL(mjdstart='59146', observatory='apo', apred='daily', makeplatesum=True, makeobshtml=True,
            makeobsplots=True, makevishtml=True, makestarhtml=False, makevisplots=True, makestarplots=False,
            makenightqa=True, makemasterqa=True, makeqafits=True, makemonitor=True, clobber=True):

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

    for ii in range(nmjd):
        if umjd[ii][0:1] != 'a':
            x = apqaMJD(mjd=umjd[ii], observatory=observatory, apred=apred, makeplatesum=makeplatesum, 
                        makeobshtml=makeobshtml, makeobsplots=makeobsplots, makevishtml=makevishtml, 
                        makestarhtml=makestarhtml, makevisplots=makevisplots,makestarplots=makestarplots,
                        makenightqa=makenightqa, makemasterqa=makemasterqa, makeqafits=makeqafits, 
                        makemonitor=makemonitor, clobber=clobber)

###################################################################################################
'''APQAMJD: Wrapper for running apqa for all plates on an mjd '''
def apqaMJD(mjd='59146', observatory='apo', apred='daily', makeplatesum=True, makeobshtml=True,
            makeobsplots=True, makevishtml=True, makestarhtml=False, makevisplots=True, 
            makestarplots=False, makemasterqa=True, makenightqa=True, makeqafits=True, 
            makemonitor=True, clobber=True):

    # Establish telescope and instrument
    telescope = observatory + '25m'
    instrument = 'apogee-n'
    prefix = 'ap'
    if observatory == 'lco': 
        instrument = 'apogee-s'
        prefix = 'as'
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
        if tmp[0] == prefix+'Plan': 
            if 'sky' not in plans[i]: sciplans.append(plans[i].replace('\n',''))
        if tmp[0] == prefix+'CalPlan': calplans.append(plans[i].replace('\n',''))
        if tmp[0] == prefix+'DarkPlan': darkplans.append(plans[i].replace('\n',''))
    sciplans = np.array(sciplans)
    nsciplans = len(sciplans)
    calplans = np.array(calplans)
    ncalplans = len(calplans)
    darkplans = np.array(darkplans)
    ndarkplans = len(darkplans)

    if makeqafits is True:
        # Run apqa on the cal  plans
        print("Running APQAMJD for " + str(ncalplans) + " cal plans from MJD " + mjd + "\n")
        for i in range(ncalplans): 
            planfile = load.filename('CalPlan', mjd=mjd)
            planstr = plan.load(planfile, np=True)
            mjd = calplans[i].split('-')[3].split('.')[0]
            all_ims = planstr['APEXP']['name']
            x = makeCalFits(load=load, ims=all_ims, mjd=mjd, instrument=instrument, clobber=clobber)
        print("Done with APQAMJD for " + str(ncalplans) + " cal plans from MJD " + mjd + "\n")

        # Run apqa on the dark  plans
        print("Running APQAMJD for " + str(ndarkplans) + " dark plans from MJD " + mjd + "\n")
        for i in range(ndarkplans): 
            planfile = load.filename('DarkPlan', mjd=mjd)
            print(planfile)
            planstr = plan.load(planfile, np=True)
            mjd = darkplans[i].split('-')[3].split('.')[0]
            all_ims = planstr['APEXP']['name']
            x = makeDarkFits(load=load, ims=all_ims, mjd=mjd, instrument=instrument, clobber=clobber)
        print("Done with APQAMJD for " + str(ndarkplans) + " dark plans from MJD " + mjd + "\n")

        # Make the MJDexp fits file for this MJD
        print("Making " + mjd + "exp.fits\n")
        x = makeExpFits(instrument=instrument, apodir=apodir, apred=apred, load=load, mjd=mjd, clobber=clobber)

        # Update the summary pages if there are not science exposures
        if nsciplans < 1:
            # Make the nightly QA page
            if makenightqa is True:
                q = makeNightQA(load=load, mjd=mjd, telescope=telescope, apred=apred)

            # Make mjd.html and fields.html
            if makemasterqa is True: 
                q = makeMasterQApages(mjdmin=59146, mjdmax=9999999, apred=apred, 
                                      mjdfilebase='mjd.html',fieldfilebase='fields.html',
                                      domjd=True, dofields=True)

            # Make the monitor page
            if makemonitor == True:
                q = monitor.monitor()

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
                # If last plate fails, still make the nightly and master QA pages
                if i == nsciplans-1:
                    # Make the nightly QA page
                    if makenightqa is True:
                        q = makeNightQA(load=load, mjd=mjd, telescope=telescope, apred=apred)

                    # Make mjd.html and fields.html
                    if makemasterqa is True: 
                        q = makeMasterQApages(mjdmin=59146, mjdmax=9999999, apred=apred, 
                                              mjdfilebase='mjd.html',fieldfilebase='fields.html',
                                              domjd=True, dofields=True)
                    if makemonitor is True:
                        q = monitor.monitor(instrument=instrument, apred=apred)
                    continue
                    #sys.exit("PROBLEM!!! 1D files not found for plate " + plate + ", MJD " + mjd + "\n")

        # Only run makemasterqa, makenightqa, and monitor after the last plate on this mjd
        if i < nsciplans-1:
            x = apqa(telescope=telescope, plate=plate, mjd=mjd, apred=apred, makeplatesum=makeplatesum, makeobshtml=makeobshtml, 
                     makeobsplots=makeobsplots, makevishtml=makevishtml, makestarhtml=makestarhtml,
                     makevisplots=makevisplots, makestarplots=makestarplots, makemasterqa=False, 
                     makenightqa=False, makemonitor=False, clobber=clobber)
        else:
            x = apqa(telescope=telescope, plate=plate, mjd=mjd, apred=apred, makeplatesum=makeplatesum, makeobshtml=makeobshtml, 
                     makeobsplots=makeobsplots, makevishtml=makevishtml, makestarhtml=makestarhtml,
                     makevisplots=makevisplots, makestarplots=makestarplots, makemasterqa=makemasterqa, 
                     makenightqa=makenightqa, makemonitor=makemonitor, clobber=clobber)
        
    print("Done with APQAMJD for " + str(nsciplans) + " plates observed on MJD " + mjd + "\n")

###################################################################################################
'''APQA: Wrapper for running QA subprocedures on a plate mjd '''
def apqa(plate='15000', mjd='59146', telescope='apo25m', apred='daily', makeplatesum=True, makeobshtml=True,
         makeobsplots=True, makevishtml=True, makestarhtml=False, makevisplots=True, makestarplots=False, 
         makemasterqa=True, makenightqa=True, makemonitor=True, clobber=True):

    start_time = time.time()

    print("Starting APQA for plate " + plate + ", MJD " + mjd + "\n")

    instrument = 'apogee-n'
    if telescope == 'lco25m': instrument = 'apogee-s'

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

        if qcheck == 'bad': 
            print('plate ' + plate + ', mjd ' + mjd + ' failed to make platesum')
            return
        #pdb.set_trace()

        # Make the observation QA page
        if makeobshtml == True:
            q = makeObsHTML(load=load, ims=ims, imsReduced=imsReduced, plate=plate, mjd=mjd, field=field,
                               fluxid=fluxid, telescope=telescope)

        # Make plots for the observation QA pages
        if makeobsplots == True:
            q = makeObsPlots(load=load, ims=ims, plate=plate, mjd=mjd, instrument=instrument, telescope=telescope,
                             survey=survey, apred=apred, flat=None, fluxid=fluxid, clobber=clobber)

        # Make the visit level pages
        if makevishtml == True:
            q = makeVisHTML(load=load, plate=plate, mjd=mjd, survey=survey, apred=apred, telescope=telescope,
                            fluxid=fluxid)

        # Make the visit plots
        if makevisplots == True:
            q = apVisitPlots(load=load, plate=plate, mjd=mjd, telescope=telescope)

        if os.path.exists(platesum):
            # Make the star level html pages
            if makestarhtml == True:
                q = makeStarHTML(load=load, plate=plate, mjd=mjd, survey=survey, apred=apred, telescope=telescope)

            # Make the star plots
            if makestarplots == True:
                q = apStarPlots(load=load, plate=plate, mjd=mjd, apred=apred, telescope=telescope)

    # Make mjd.html and fields.html
    if makemasterqa == True: 
        q = makeMasterQApages(mjdmin=59146, mjdmax=9999999, apred=apred, domjd=True, dofields=True,
                              mjdfilebase='mjd.html',fieldfilebase='fields.html')

    # Make the nightly QA page
    if makenightqa == True:
        q = makeNightQA(load=load, mjd=mjd, telescope=telescope, apred=apred)

    # Make the monitor page
    if makemonitor == True:
        q = monitor.monitor(instrument=instrument, apred=apred)

    runtime = str("%.2f" % (time.time() - start_time))
    print("Done with APQA for plate " + plate + ", MJD " + mjd + " in " + runtime + " seconds.\n")

###################################################################################################
''' MAKEPLATESUM: Plotmag translation '''
def makePlateSum(load=None, telescope=None, ims=None, imsReduced=None, plate=None, mjd=None, field=None,
                 instrument=None, clobber=True, makeqaplots=None, plugmap=None, survey=None,
                 mapper_data=None, apred=None, onem=None, starfiber=None, starnames=None, 
                 starmag=None, flat=None, fixfiberid=None, badfiberid=None):

    prefix = 'ap'
    if telescope == 'lco25m': prefix = 'as'
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

    #import pdb; pdb.set_trace()

    #if ims[0] == 0: 
    #    tot = load.apPlate(int(plate), mjd)
    #else:
    #    tot = load.ap1D(ims[0])

    #pdb.set_trace()
    #if type(tot) != dict:
    #    html.write('<FONT COLOR=red> PROBLEM/FAILURE WITH: '+str(ims[0])+'\n')
    #    htmlsum.write('<FONT COLOR=red> PROBLEM/FAILURE WITH: '+str(ims[0])+'\n')
    #    html.close()
    #    htmlsum.close()
    #    print("----> makePlateSum: Error!")

    plug = platedata.getdata(int(plate), int(mjd), apred, telescope, plugid=plugmap, badfiberid=badfiberid) 
    #pdb.set_trace()

    #nplug = len(plug['fiberdata']['fiberid'])
    #for k in range(nplug): 
    #    fib = str(plug['fiberdata']['fiberid'][k]).rjust(3)
    #    nm = plug['fiberdata']['twomass_designation'][k].rjust(16)
    #    hm = str(plug['fiberdata']['hmag'][k]).rjust(6)
    #    ot = plug['fiberdata']['objtype'][k]
    #    ra = str("%.6f" % round(plug['fiberdata']['ra'][k],6)).rjust(10)
    #    de = str("%.6f" % round(plug['fiberdata']['dec'][k],6)).rjust(10)
    #    print(fib+'  '+nm+'  '+hm+'  '+ot+'  '+ra+'  '+de)

    #import pdb; pdb.set_trace()

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

    gcamdir = os.environ.get('APOGEE_REDUX')+'/'+apred+'/'+'exposures/'+instrument+'/'+mjd+'/'
    gcamfile = gcamdir+'gcam-'+mjd+'.fits'
    # Get guider information.
    #if onem is None:
    #    gcamdir = os.environ.get('APOGEE_REDUX')+'/'+apred+'/'+'exposures/'+instrument+'/'+mjd+'/'
    #    if os.path.exists(gcamdir) == False: subprocess.call(['mkdir',gcamdir])
    #    gcamfile = gcamdir+'gcam-'+mjd+'.fits'
    #    if os.path.exists(gcamfile) == False:
    #        print("----> makePlateSum: Attempting to make "+os.path.basename(gcamfile)+".")
    #        subprocess.call(['gcam_process', '--mjd', mjd, '--instrument', instrument, '--output', gcamfile], shell=False)
    #        if os.path.exists(gcamfile):
    #            print("----> makePlateSum: Successfully made "+os.path.basename(gcamfile))
    #        else:
    #            print("----> makePlateSum: Failed to make "+os.path.basename(gcamfile))
    #    else:
    #        gcam = fits.getdata(gcamfile)

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
            if os.path.exists(dfile.replace(prefix+'Plate-',prefix+'Plate-a-')):
                dhdr = fits.getheader(dfile.replace(prefix+'Plate-',prefix+'Plate-a-'))
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
            if 'ALT' in dhdr: secz = 1. / np.cos((90. - dhdr['ALT']) * (math.pi/180.))
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
        if 'DESIGN_HA' in dhdr: platetab['SECZ'][i] =      secz
        if dhdr.get('HA') is not None: platetab['HA'][i] = dhdr['HA']
        if 'DESIGN_HA' in dhdr: platetab['DESIGN_HA'][i] = design_ha
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

    # Write out the FITS table.
    platesum = load.filename('PlateSum', plate=int(plate), mjd=mjd, fps=fps)
    if ims[0] != 0:
        Table(platetab).write(platesum, overwrite=True)
        hdulist = fits.open(platesum)
        hdu = fits.table_to_hdu(Table(fiber))
        hdulist.append(hdu)
        hdulist.writeto(platesum, overwrite=True)
        hdulist.close()

        # Make the sn*dat and altsn*dat files
        outfile1 = sntabdir + 'sn-' + plate + '-' + mjd + '.dat'
        outfile2 = sntabdir + 'altsn-' + plate + '-' + mjd + '.dat'
        outfiles = np.array([outfile1, outfile2])
        nout = len(outfiles)
        
        txt = "making " + os.path.basename(outfile1) + " and " + os.path.basename(outfile2)
        print("----> makePlateSum: " + txt)

        tab1 = fits.getdata(platesum,1)
        for j in range(nout):
            out = open(outfiles[j], 'w')
            for i in range(n_exposures):
                gd, = np.where(ims[i] == tab1['IM'])
                if len(gd) == 1:
                    gd = gd[0]
                    # Image number
                    im = str(ims[i])
                    # SN or ALTSN
                    if j == 0:
                        sn = str("%.2f" % round(tab1['SN'][gd][1], 2)).rjust(5)
                    else:
                        sn = str("%.2f" % round(tab1['ALTSN'][gd][1], 2))
                    # APRRED VERSION ?
                    vers = '1'
                    # MJD when plate was plugged
                    plugmjd = plugmap.split('-')[1]
                    # Observation MJD in seconds
                    t = Time(tab1['DATEOBS'][gd], format='fits')
                    tsec = str("%.5f" % round(t.mjd * 86400, 5))
                    # Exposure time
                    exptime=str(tab1['EXPTIME'][gd])
                    # Write to file
                    out.write(im+'  '+sn+'  '+vers+'  '+plugmjd+'  '+plate+'  '+mjd+'  '+tsec+'  '+exptime+'  Object\n')
            out.close()
        print("----> makePlateSum: done " + txt)
    else:
        hdulist = fits.open(platesum)
        hdu1 = fits.table_to_hdu(Table(platetab))
        hdulist.append(hdu1)
        hdulist.writeto(platesum, overwrite=True)
        hdulist.close()

    print("----> makePlateSum: Done with plate "+plate+", MJD "+mjd+"\n")
    return


###################################################################################################
''' MAKEOBSHTML: mkhtmlplate translation '''
def makeObsHTML(load=None, ims=None, imsReduced=None, plate=None, mjd=None, field=None,
                   fluxid=None, telescope=None):

    print("----> makeObsHTML: Running plate "+plate+", MJD "+mjd)

    # HTML header background color
    thcolor = '#DCDCDC'

    if int(mjd)>59556:
        fps = True
    else:
        fps = False
    prefix = 'ap'

    if telescope == 'lco25m':
        if int(mjd)>59808:
            fps = True
        else:
            fps = False
        prefix = 'as'

    chips = np.array(['a','b','c'])
    nchips = len(chips)

    n_exposures = len(ims)

    # Check for existence of plateSum file
    platesum = load.filename('PlateSum', plate=int(plate), mjd=mjd, fps=fps) 
    platedir = os.path.dirname(platesum)+'/'

    if os.path.exists(platesum) == False:
        err1 = "----> makeObsHTML: PROBLEM!!! " + os.path.basename(platesum) + " does not exist. Halting execution.\n"
        err2 = "----> makeObsHTML: You need to run MAKEPLATESUM first to make the file."
        sys.exit(err1 + err2)

    # Read the plateSum file
    tab1 = fits.getdata(platesum,1)
    tab2 = fits.getdata(platesum,2)
    tab3 = fits.getdata(platesum,3)

    # Make the html directory if it doesn't already exist
    qafile = load.filename('QA', plate=int(plate), mjd=mjd)
    qafiledir = os.path.dirname(qafile)
    print("----> makeObsHTML: Creating "+os.path.basename(qafile))
    if os.path.exists(qafiledir) == False: subprocess.call(['mkdir',qafiledir])

    html = open(qafile, 'w')
    tmp = os.path.basename(qafile).replace('.html','')
    html.write('<HTML><HEAD><script src="../../../../../../../sorttable.js"></script><title>'+tmp+'</title></head><BODY>\n')
    html.write('<H1>Field: <FONT COLOR="green">' + field + '</FONT><BR>Plate: <FONT COLOR="green">' + plate)
    html.write('</FONT><BR>MJD: <FONT COLOR="green">' + mjd + '</FONT></H1>\n')
    html.write('<p><a href="../../../../../../qa/mjd.html">back to MJD page</a><BR>\n')
    html.write('<a href="../../../../../../qa/fields.html">back to Fields page</a></p>\n')
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
    html.write('<H3> Plots of apVisit spectra ---> <A HREF='+prefix+'Plate-'+plate+'-'+mjd+'.html>apPlate-'+plate+'-'+mjd+'</a></H3>\n')
    html.write('<HR>\n')

    # SNR plots
    html.write('<H3>apVisit Hmag versus S/N: </H3>\n')
    snrplot1 = 'apVisitSNR-'+plate+'-'+mjd+'.png'
    snrplot2 = 'apVisitSNRblocks-'+plate+'-'+mjd+'.png'
    html.write('<A HREF=../plots/'+snrplot1+' target="_blank"><IMG SRC=../plots/'+snrplot1+' WIDTH=600></A>')
    html.write('<A HREF=../plots/'+snrplot2+' target="_blank"><IMG SRC=../plots/'+snrplot2+' WIDTH=600></A>\n')
    html.write('<HR>\n')

    # Flat field plots.
    if fluxid is not None:
        fluxfile = os.path.basename(load.filename('Flux', num=fluxid, chips=True)).replace('.fits','.png')
        html.write('<H3>Fiber Throughput:</H3>\n')
        html.write('<P><b>Note:</b> Points are color-coded by median dome flat flux divided by the maximum median dome flat flux.</P>\n')
        html.write('<A HREF="'+'../plots/'+fluxfile+'" target="_blank"><IMG SRC=../plots/'+fluxfile+' WIDTH=1200></A>')
        html.write('<HR>\n')

    # Fiber location plots.
    html.write('<H3>Fiber Positions:</H3>\n')
    html.write('<A HREF="'+'../plots/'+fluxfile.replace('Flux','FibLoc')+'" target="_blank"><IMG SRC=../plots/'+fluxfile.replace('Flux','FibLoc')+' WIDTH=900></A>')
    html.write('<HR>\n')

    # Table of individual exposures.
    if pairstr is not None:
        html.write('<H3>Individual Exposures:</H3>\n')
    else:
        html.write('<H3>Individual Exposures (undithered):</H3>\n')
    html.write('<p><b>Note:</b> Design HA values are currently missing.<BR> \n')
    html.write('<b>Note:</b> Dither and Pixshift values will be "---" if exposures not dithered.<BR>\n')
    html.write('<b>Note:</b> S/N columns give S/N for blue, green, and red chips separately. </p>\n')
    html.write('<TABLE BORDER=2 CLASS="sortable">\n')
    html.write('<TR bgcolor="'+thcolor+'">\n')
    txt1 = '<TH>#<TH>FRAME <TH>EXPTIME <TH>CART <TH>SEC Z <TH>HA<TH>DESIGN HA <TH>SEEING <TH>FWHM <TH>GDRMS <TH>NREADS <TH>DITHER'
    txt2 = '<TH>PIXSHIFT <TH>ZERO <TH>ZERO<BR>RMS <TH>SKY<BR>CONTINUUM <TH>S/N <TH>S/N (CFRAME) <TH>MOON<BR>PHASE <TH>MOON<BR>DIST.'
    html.write(txt1 + txt2 +'\n')

    for i in range(n_exposures):
        gd, = np.where(ims[i] == tab1['IM'])
        if len(gd) >= 1:
            html.write('<TR>\n')
            html.write('<TD align="right">'+str(i+1)+'\n')
            html.write('<TD align="right">'+str(int(round(ims[i])))+'\n')
            html.write('<TD align="right">'+str(int(round(tab1['EXPTIME'][gd][0])))+'\n')
            try:
                html.write('<TD align="right">'+str(int(round(tab1['CART'][gd][0])))+'\n')
            except:
                html.write('<TD align="right">'+tab1['CART'][gd][0]+'\n')
            html.write('<TD align="right">'+str("%.3f" % round(tab1['SECZ'][gd][0],3))+'\n')
            html.write('<TD align="right">'+str("%.2f" % round(tab1['HA'][gd][0],2))+'\n')
            html.write('<TD align="right">'+str(np.round(tab1['DESIGN_HA'][gd][0],0)).replace('[',' ')[:-1]+'\n')
            html.write('<TD align="right">'+str("%.3f" % round(tab1['SEEING'][gd][0],3))+'\n')
            html.write('<TD align="right">'+str("%.3f" % round(tab1['FWHM'][gd][0],3))+'\n')
            html.write('<TD align="right">'+str("%.3f" % round(tab1['GDRMS'][gd][0],3))+'\n')
            html.write('<TD align="right">'+str(tab1['NREADS'][gd][0])+'\n')
            j = np.where(shiftstr['FRAMENUM'] == str(tab1['IM'][gd][0]))
            nj = len(j[0])
            nodither, = np.where(shiftstr['SHIFT'] == 0)
            if (nj > 0) & (len(nodither) != len(tab1['IM'])):
                html.write('<TD align="right">'+str("%.4f" % round(shiftstr['SHIFT'][j][0],4)).rjust(7)+'\n')
                html.write('<TD align="right">'+str("%.2f" % round(shiftstr['PIXSHIFT'][j][0],2))+'\n')
            else:
                html.write('<TD align="center">---\n')
                html.write('<TD align="center">---\n')
            html.write('<TD align="right">'+str("%.3f" % round(tab1['ZERO'][gd][0],3))+'\n')
            html.write('<TD align="right">'+str("%.3f" % round(tab1['ZERORMS'][gd][0],3))+'\n')
            q = tab1['SKY'][gd][0]
            txt = str("%.2f" % round(q[2],2))+', '+str("%.2f" % round(q[1],2))+', '+str("%.2f" % round(q[0],2))
            html.write('<TD align="center">'+'['+txt+']\n')
            q = tab1['SN'][gd][0]
            txt = str("%.2f" % round(q[2],2))+', '+str("%.2f" % round(q[1],2))+', '+str("%.2f" % round(q[0],2))
            html.write('<TD align="center">'+'['+txt+']\n')
            q = tab1['SNC'][gd][0]
            txt = str("%.2f" % round(q[2],2))+', '+str("%.2f" % round(q[1],2))+', '+str("%.2f" % round(q[0],2))
            html.write('<TD align="center">'+'['+txt+']\n')
            html.write('<TD align="right">'+str("%.3f" % round(tab1['MOONPHASE'][gd][0],3))+'\n')
            html.write('<TD align="right">'+str("%.3f" % round(tab1['MOONDIST'][gd][0],3))+'\n')
        else:
            html.write('<TR bgcolor=red>\n')
            html.write('<TD align="right">'+str(i+1)+'\n')
            html.write('<TD align="right">'+str(int(round(ims[i])))+'\n')
            html.write('<TD><TD><TD><TD><TD><TD><TD><TD><TD><TD><TD><TD><TD><TD><TD><TD><TD><TD>\n')

    #Msecz = str("%.3f" % round(np.nanmean(tab1['SECZ']),3))
    #Mseeing = str("%.3f" % round(np.nanmean(tab1['SEEING']),3))
    #Mfwhm = str("%.3f" % round(tab3['FWHM'][0],3))
    #Mgdrms = str("%.3f" % round(tab3['GDRMS'][0],3))
    #Mzero = str("%.3f" % round(tab3['ZERO'][0],3))
    #Mzerorms = str("%.3f" % round(tab3['ZERORMS'][0],3))
    #Mmoonphase = str("%.3f" % round(tab3['MOONPHASE'][0],3))
    #Mmoondist = str("%.3f" % round(tab3['MOONDIST'][0],3))
    ##q = tab3['SKY'][0]
    ##sky = str("%.2f" % round(q[0],2))+', '+str("%.2f" % round(q[1],2))+', '+str("%.2f" % round(q[2],2))
    #q = tab3['SN'][0]
    #sn = str("%.2f" % round(q[2],2))+', '+str("%.2f" % round(q[1],2))+', '+str("%.2f" % round(q[0],2))
    #q = tab3['SNC'][0]
    #snc = str("%.2f" % round(q[2],2))+', '+str("%.2f" % round(q[1],2))+', '+str("%.2f" % round(q[0],2))
    #html.write('<TR><TD><B>VISIT<TD><TD><TD><TD align="right"><B>'+Msecz+'<TD><TD><TD align="right"><B>'+Mseeing)
    #html.write('<TD align="right"><B>'+Mfwhm+'<TD align="right"><B>'+Mgdrms+'<TD><TD><TD><TD align="right"><B>'+Mzero)

    ##html.write('<TD align="center">['+sky+']')
    #html.write('<TD align="right"><B>'+Mzerorms+'<TD>')
    #html.write('<TD align="center"><B>['+sn+']')
    #html.write('<TD align="center"><B>['+snc+']')
    #html.write('<TD align="right"><B>'+Mmoonphase+'<TD align="right"><B>'+Mmoondist+'</b>\n')

    html.write('</TABLE>\n')
    html.write('<HR>\n')

    # Table of exposure pairs.
    if pairstr is not None:
        npairs = len(pairstr)
        if npairs > 0:
            # Pair table.
            html.write('<H3>Dither Pair Stats:</H3>\n')
            html.write('<TABLE BORDER=2 CLASS="sortable">\n')
            html.write('<TR bgcolor="'+thcolor+'"><TH>IPAIR<TH>NAME<TH>SHIFT<TH>NEWSHIFT<TH>S/N\n')
            html.write('<TH>NAME<TH>SHIFT<TH>NEWSHIFT<TH>S/N\n')
            for ipair in range(npairs):
                html.write('<TR><TD>'+str(ipair)+'\n')
                for j in range(2):
                    html.write('<TD>'+str(pairstr['FRAMENAME'][ipair][j])+'\n')
                    html.write('<TD>'+str("%.3f" % round(pairstr['OLDSHIFT'][ipair][j],3))+'\n')
                    html.write('<TD>'+str("%.3f" % round(pairstr['SHIFT'][ipair][j],3))+'\n')
                    html.write('<TD>'+str("%.2f" % round(pairstr['SN'][ipair][j],2))+'\n')
            html.write('</TABLE>\n')
        html.write('<HR>\n')

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
    html.write('<H3>Individual Exposure QA Plots:</H3>\n')
    html.write('<TABLE BORDER=2>\n')
    html.write('<p><b>Note:</b> in the Mag plots, the solid line is the target line for getting S/N=100 for an H=12.2 star in 3 hours of exposure time.<BR>\n')
    html.write('<b>Note:</b> in the Spatial mag deviation plots, color gives deviation of observed mag from expected 2MASS mag using the median zeropoint.</p>\n')
    html.write('<TR bgcolor="'+thcolor+'"><TH>FRAME <TH>ZEROPOINTS <TH>MAG PLOTS (GREEN CHIP)\n')
    html.write('<TH>SPATIAL MAG DEVIATION\n')
    html.write('<TH>SPATIAL SKY 16325 &#8491; EMISSION DEVIATION\n')
    html.write('<TH>SPATIAL SKY CONTINUUM EMISSION\n')
    html.write('<TH>SPATIAL SKY TELLURIC CH4\n')
    html.write('<TH>SPATIAL SKY TELLURIC CO2\n')
    html.write('<TH>SPATIAL SKY TELLURIC H2O\n')

    for i in range(n_exposures):
        gd, = np.where(ims[i] == tab1['IM'])
        if len(gd) >= 1:
            oneDfile = os.path.basename(load.filename('1D', num=ims[i], mjd=mjd, chips=True)).replace('.fits','')
            #html.write('<TR><TD bgcolor="'+thcolor+'"><A HREF=../html/'+oneDfile+'.html>'+str(im)+'</A>\n')
            html.write('<TR><TD bgcolor="'+thcolor+'">'+str(int(round(ims[i])))+'\n')
            html.write('<TD><TABLE BORDER=1><TD><TD bgcolor="'+thcolor+'">RED<TD bgcolor="'+thcolor+'">GREEN<TD bgcolor="'+thcolor+'">BLUE\n')
            html.write('<TR><TD bgcolor="'+thcolor+'">Z<TD><TD>'+str("%.2f" % round(tab1['ZERO'][gd][0],2))+'\n')
            html.write('<TR><TD bgcolor="'+thcolor+'">ZNORM<TD><TD>'+str("%.2f" % round(tab1['ZERONORM'][gd][0],2))+'\n')
            txt='<TD>'+str("%.1f" % round(tab1['SKY'][gd][0][0],1))+'<TD>'+str("%.1f" % round(tab1['SKY'][gd][0][1],1))+'<TD>'+str("%.1f" % round(tab1['SKY'][gd][0][2],1))
            html.write('<TR><TD bgcolor="'+thcolor+'">SKY'+txt+'\n')
            txt='<TD>'+str("%.1f" % round(tab1['SN'][gd][0][0],1))+'<TD>'+str("%.1f" % round(tab1['SN'][gd][0][1],1))+'<TD>'+str("%.1f" % round(tab1['SN'][gd][0][2],1))
            html.write('<TR><TD bgcolor="'+thcolor+'">S/N'+txt+'\n')
            txt='<TD>'+str("%.1f" % round(tab1['SNC'][gd][0][0],1))+'<TD>'+str("%.1f" % round(tab1['SNC'][gd][0][1],1))+'<TD>'+str("%.1f" % round(tab1['SNC'][gd][0][2],1))
            html.write('<TR><TD bgcolor="'+thcolor+'">S/N(C)'+txt+'\n')
    #        if tag_exist(tab1[i],'snratio'):
            html.write('<TR><TD bgcolor="'+thcolor+'">SN(E/C)<TD>'+str(np.round(tab1['SNRATIO'][gd][0],2))+'\n')
            html.write('</TABLE>\n')

            html.write('<TD><A HREF=../plots/'+oneDfile+'_magplots.png target="_blank"><IMG SRC=../plots/'+oneDfile+'_magplots.png WIDTH=210></A>\n')
            html.write('<TD><A HREF=../plots/'+oneDfile+'_spatialresid.png target="_blank"><IMG SRC=../plots/'+oneDfile+'_spatialresid.png WIDTH=250></A>\n')
            html.write('<TD><A HREF='+'../plots/'+oneDfile+'_skyemission.png target="_blank"><IMG SRC=../plots/'+oneDfile+'_skyemission.png WIDTH=250>\n')
            html.write('<TD><A HREF='+'../plots/'+oneDfile+'_skycontinuum.png target="_blank"><IMG SRC=../plots/'+oneDfile+'_skycontinuum.png WIDTH=250>\n')
            cim = str(ims[i])
            html.write('<TD> <a href=../plots/'+prefix+'telluric_'+cim+'_skyfit_CH4.jpg target="_blank"> <IMG SRC=../plots/'+prefix+'telluric_'+cim+'_skyfit_CH4.jpg WIDTH=250></a>\n')
            html.write('<TD> <a href=../plots/'+prefix+'telluric_'+cim+'_skyfit_CO2.jpg target="_blank"> <IMG SRC=../plots/'+prefix+'telluric_'+cim+'_skyfit_CO2.jpg WIDTH=250></a>\n')
            html.write('<TD> <a href=../plots/'+prefix+'telluric_'+cim+'_skyfit_H2O.jpg target="_blank"> <IMG SRC=../plots/'+prefix+'telluric_'+cim+'_skyfit_H2O.jpg WIDTH=250></a>\n')
        else:
            html.write('<TR><TD bgcolor="'+thcolor+'">'+str(int(round(ims[i])))+'\n')
            html.write('<TD><TD><TD><TD><TD><TD><TD><TD>\n')
    html.write('</table><HR>\n')
    
    gfile = 'guider-'+plate+'-'+mjd+'.png'
    html.write('<H3>Guider RMS: </H3>\n')
    html.write('<A HREF='+'../plots/'+gfile+'><IMG SRC=../plots/'+gfile+' WIDTH=390 target="_blank"></A>\n')
    
    html.write('<BR><BR>\n')
    html.write('</BODY></HTML>\n')
    html.close()

    print("----> makeObsHTML: Done with plate "+plate+", MJD "+mjd+"\n")

###################################################################################################
''' MAKEOBSPLOTS: plots for the plate QA page '''
def makeObsPlots(load=None, ims=None, imsReduced=None, plate=None, mjd=None, instrument=None, telescope=None,
                   apred=None, flat=None, fluxid=None, survey=None, clobber=None): 

    print("----> makeObsPlots: Running plate "+plate+", MJD "+mjd)

    if int(mjd)>59556:
        fps = True
    else:
        fps = False
    prefix = 'ap'

    if telescope == 'lco25m':
        if int(mjd)>59808:
            fps = True
        else:
            fps = False
        prefix = 'as'

    n_exposures = len(ims)
    chips = np.array(['a','b','c'])
    chiplab = np.array(['blue','green','red'])
    nchips = len(chips)

    # Make plot and html directories if they don't already exist.
    platedir = os.path.dirname(load.filename('Plate', plate=int(plate), mjd=mjd, chips=True, fps=fps))
    plotsdir = platedir+'/plots/'
    if len(glob.glob(plotsdir)) == 0: subprocess.call(['mkdir',plotsdir])

    # Set up some basic plotting parameters, starting by turning off interactive plotting.
    #plt.ioff()
    matplotlib.use('agg')
    fontsize = 24;   fsz = fontsize * 0.75
    matplotlib.rcParams.update({'font.size':fontsize, 'font.family':'serif'})
    matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"
    alpha = 0.6
    axwidth=1.5
    axmajlen=7
    axminlen=3.5
    cmap = 'RdBu'

    # Check for existence of plateSum file
    platesum = load.filename('PlateSum', plate=int(plate), mjd=mjd, fps=fps) 
    if os.path.exists(platesum) == False:
        err1 = "----> makeObsPlots: PROBLEM!!! " + os.path.basename(platesum) + " does not exist. Halting execution.\n"
        err2 = "----> makeObsPlots: You need to run MAKEPLATESUM first to make the file."
        sys.exit(err1 + err2)

    # Read the plateSum file
    plSum1 = fits.getdata(platesum,1)
    platesum2 = fits.getdata(platesum,2)
    fibord = np.argsort(platesum2['FIBERID'])
    plSum2 = platesum2[fibord]
    nfiber = len(plSum2['HMAG'])

    #----------------------------------------------------------------------------------------------
    # PLOTS 1-2: HMAG versus S/N for the exposure-combined apVisit, second version colored by fiber block
    #----------------------------------------------------------------------------------------------
    Vsum = load.apVisitSum(int(plate), mjd)
    Vsumfile = Vsum.filename()
    Vsum = Vsum[1].data
    block = np.floor((Vsum['FIBERID'] - 1) / 30) #[::-1]

    for i in range(2):
        plotfile = os.path.basename(Vsumfile).replace('Sum','SNR').replace('.fits','.png')
        if i == 1: plotfile = plotfile.replace('SNR','SNRblocks')
        if (os.path.exists(plotsdir+plotfile) == False) | (clobber == True):
            print("----> makeObsPlots: Making "+plotfile)

            fig=plt.figure(figsize=(19,10))
            ax = plt.subplot2grid((1,1), (0,0))
            ax.tick_params(reset=True)
            ax.minorticks_on()
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.set_xlabel(r'$H$ mag.')
            ax.set_ylabel(r'apVisit S/N')
            ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
            ax.tick_params(axis='both',which='major',length=axmajlen)
            ax.tick_params(axis='both',which='minor',length=axminlen)
            ax.tick_params(axis='both',which='both',width=axwidth)

            if 'HMAG' in Vsum.columns.names:
                hmagarr = Vsum['HMAG']
            else:
                hmagarr = Vsum['H']
            gd, = np.where((hmagarr > 0) & (hmagarr < 20) & (np.isnan(Vsum['SNR']) == False))
            ngd = len(gd)
            try:
                minH = np.nanmin(hmagarr[gd]);  maxH = np.nanmax(hmagarr[gd])
            except:
                minH = 6;  maxH = 14
            spanH = maxH - minH
            xmin = minH - spanH * 0.05;     xmax = maxH + spanH * 0.05

            try:
                minSNR = np.nanmin(Vsum['SNR']); maxSNR = np.nanmax(Vsum['SNR'])
            except:
                minSNR = 0;  maxSNR = 500
            spanSNR = maxSNR - minSNR
            ymin = -5;                       ymax = maxSNR + ((maxSNR - ymin) * 0.05)

            if fps:
                notsky, = np.where((Vsum['HMAG'] > 5) & (Vsum['HMAG'] < 15) & (np.isnan(Vsum['HMAG']) == False) & 
                                   (np.isnan(Vsum['SNR']) == False) & (Vsum['SNR'] > 0) & (Vsum['ASSIGNED']) & 
                                   (Vsum['ON_TARGET']) & (Vsum['VALID']) & (Vsum['OBJTYPE'] != 'none'))
            else:
                notsky, = np.where((Vsum['HMAG'] > 5) & (Vsum['HMAG'] < 15) & (np.isnan(Vsum['HMAG']) == False) & 
                                   (np.isnan(Vsum['SNR']) == False) & (Vsum['SNR'] > 0) & (Vsum['OBJTYPE'] != 'none'))

            if len(notsky) > 10:
                if i == 0:
                    # First pass at fitting line to S/N as function of Hmag
                    hmag1 = Vsum['HMAG'][notsky]
                    sn1 = Vsum['SNR'][notsky]
                    polynomial1 = np.poly1d(np.polyfit(hmag1, np.log10(sn1), 1))
                    yarrnew1 = polynomial1(hmag1)
                    diff1 = np.log10(sn1) - yarrnew1
                    gd1, = np.where(diff1 > -np.nanstd(diff1))
                    # Second pass at fitting line to S/N as function of Hmag
                    hmag2 = hmag1[gd1]
                    sn2 = sn1[gd1]
                    polynomial2 = np.poly1d(np.polyfit(hmag2, np.log10(sn2), 1))
                    yarrnew2 = polynomial2(hmag2)
                    diff2 = np.log10(sn2) - yarrnew2
                    gd2, = np.where(diff2 > -np.nanstd(diff2))
                    # Final pass at fitting line to S/N as function of Hmag
                    hmag3 = hmag2[gd2]
                    sn3 = sn2[gd2]
                    polynomial3 = np.poly1d(np.polyfit(hmag3, np.log10(sn3), 1))
                    xarrnew3 = np.linspace(np.nanmin(hmag1), np.nanmax(hmag1), 5000)
                    yarrnew3 = polynomial3(xarrnew3)

                ax.plot(xarrnew3, 10**yarrnew3, color='grey', linestyle='dashed')

            ax.set_xlim(xmin,xmax)
            ax.set_ylim(1,1200)
            ax.set_yscale('log')

            if fps:
                science, = np.where((Vsum['HMAG'] > 0) & (Vsum['HMAG'] < 16) & (np.isnan(Vsum['HMAG']) == False) & 
                                    (np.isnan(Vsum['SNR']) == False) & (Vsum['SNR'] > 0) & (Vsum['ASSIGNED']) & 
                                    (Vsum['ON_TARGET']) & (Vsum['VALID']) & 
                                    ((Vsum['OBJTYPE'] == 'OBJECT') | (Vsum['OBJTYPE'] == 'STAR')))

                telluric, = np.where((Vsum['HMAG'] > 0) & (Vsum['HMAG'] < 16) & (np.isnan(Vsum['HMAG']) == False) & 
                                     (np.isnan(Vsum['SNR']) == False) & (Vsum['SNR'] > 0) & (Vsum['ASSIGNED']) & 
                                     (Vsum['ON_TARGET']) & (Vsum['VALID']) & 
                                     ((Vsum['OBJTYPE'] == 'SPECTROPHOTO_STD') | (Vsum['OBJTYPE'] == 'HOT_STD')))
            else:
                science, = np.where((Vsum['HMAG'] > 0) & (Vsum['HMAG'] < 16) & (np.isnan(Vsum['HMAG']) == False) & 
                                    (np.isnan(Vsum['SNR']) == False) & (Vsum['SNR'] > 0) & 
                                    ((Vsum['OBJTYPE'] == 'OBJECT') | (Vsum['OBJTYPE'] == 'STAR')))

                telluric, = np.where((Vsum['HMAG'] > 0) & (Vsum['HMAG'] < 16) & (np.isnan(Vsum['HMAG']) == False) & 
                                     (np.isnan(Vsum['SNR']) == False) & (Vsum['SNR'] > 0) &
                                     ((Vsum['OBJTYPE'] == 'SPECTROPHOTO_STD') | (Vsum['OBJTYPE'] == 'HOT_STD')))

            x = Vsum['HMAG'][science];  y = Vsum['SNR'][science]
            scicol = 'r'
            telcol = 'dodgerblue'
            if i == 1:
                scicol = block[science] + 0.5
                telcol = block[telluric] + 0.5
            psci = ax.scatter(x, y, marker='*', s=400, edgecolors='white', alpha=0.8, c=scicol, cmap='tab10', vmin=0.5, vmax=10.5, label='Science')
            x = Vsum['HMAG'][telluric];  y = Vsum['SNR'][telluric]
            ptel = ax.scatter(x, y, marker='o', s=150, edgecolors='white', alpha=0.8, c=telcol, cmap='tab10', vmin=0.5, vmax=10.5, label='Telluric')

            if i == 1:
                ax_divider = make_axes_locatable(ax)
                cax = ax_divider.append_axes("right", size="2%", pad="1%")
                cb = colorbar(ptel, cax=cax, orientation="vertical")
                #cax.xaxis.set_ticks_position("right")
                cax.yaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.text(1.09, 0.5, r'MTP #', ha='right', va='center', rotation=-90, transform=ax.transAxes)

            ax.legend(loc='upper right', labelspacing=0.5, handletextpad=-0.1, facecolor='lightgrey')

            fig.subplots_adjust(left=0.06,right=0.945,bottom=0.09,top=0.975,hspace=0.2,wspace=0.0)
            plt.savefig(plotsdir+plotfile)
            plt.close('all')

    #----------------------------------------------------------------------------------------------
    # PLOTS 3-7: flat field flux and fiber blocks... previously done by plotflux.pro
    #----------------------------------------------------------------------------------------------
    fluxfile = os.path.basename(load.filename('Flux', num=fluxid, chips=True))
    flux = load.apFlux(fluxid)
    ypos = 300 - platesum2['FIBERID']
    block = np.floor((plSum2['FIBERID'] - 1) / 30) #[::-1]

    plotfile = fluxfile.replace('.fits', '.png')
    if (os.path.exists(plotsdir+plotfile) == False) | (clobber == True):
        print("----> makeObsPlots: Making "+plotfile)

        fig=plt.figure(figsize=(35,8))
        plotrad = 1.6

        for ichip in range(nchips):
            chip = chips[ichip]

            ax = plt.subplot2grid((1,nchips+2), (0,ichip))
            ax.set_xlim(-plotrad, plotrad)
            ax.set_ylim(-plotrad, plotrad)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.minorticks_on()
            ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
            ax.tick_params(axis='both',which='major',length=axmajlen)
            ax.tick_params(axis='both',which='minor',length=axminlen)
            ax.tick_params(axis='both',which='both',width=axwidth)
            ax.set_xlabel(r'Zeta (deg.)')
            if ichip == 0: ax.set_ylabel(r'Eta (deg.)')
            if ichip != 0: ax.axes.yaxis.set_ticklabels([])

            med = np.nanmedian(flux[chip][1].data, axis=1)
            tput = med[ypos] / np.nanmax(med[ypos])
            gd, = np.where((ypos > 60) & (ypos < 90))
            sc = ax.scatter(platesum2['Zeta'], platesum2['Eta'], marker='o', s=100, c=tput, edgecolors='k', cmap='afmhot', alpha=1, vmin=0.01, vmax=0.99)

            ax.text(0.03, 0.97, chiplab[ichip]+'\n'+'chip', transform=ax.transAxes, ha='left', va='top', color=chiplab[ichip])

            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("top", size="4%", pad="1%")
            cb = colorbar(sc, cax=cax, orientation="horizontal")
            cax.xaxis.set_ticks_position("top")
            cax.minorticks_on()
            ax.text(0.5, 1.13, r'Dome Flat Throughput',ha='center', transform=ax.transAxes)

        ax1 = plt.subplot2grid((1,nchips+2), (0,nchips))
        ax1.set_xlim(-1.6,1.6)
        ax1.set_ylim(-1.6,1.6)
        ax1.axes.yaxis.set_ticklabels([])
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax1.minorticks_on()
        ax1.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
        ax1.tick_params(axis='both',which='major',length=axmajlen)
        ax1.tick_params(axis='both',which='minor',length=axminlen)
        ax1.tick_params(axis='both',which='both',width=axwidth)
        ax1.set_xlabel(r'Zeta (deg.)')

        sc = ax1.scatter(plSum2['Zeta'], plSum2['Eta'], marker='o', s=150, c=block+0.5, edgecolors='white', cmap='tab10', alpha=0.9, vmin=0.5, vmax=10.5)

        ax1_divider = make_axes_locatable(ax1)
        cax1 = ax1_divider.append_axes("top", size="4%", pad="1%")
        cb = colorbar(sc, cax=cax1, orientation="horizontal")
        cax1.xaxis.set_ticks_position("top")
        cax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax1.text(0.5, 1.13, r'MTP #', ha='center', transform=ax1.transAxes)

        ax2 = plt.subplot2grid((1,nchips+2), (0,nchips+1))
        ax2.set_xlim(-1.6,1.6)
        ax2.set_ylim(-1.6,1.6)
        ax2.axes.yaxis.set_ticklabels([])
        ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax2.minorticks_on()
        ax2.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
        ax2.tick_params(axis='both',which='major',length=axmajlen)
        ax2.tick_params(axis='both',which='minor',length=axminlen)
        ax2.tick_params(axis='both',which='both',width=axwidth)
        ax2.set_xlabel(r'Zeta (deg.)')

        if fps:
            notsky, = np.where((Vsum['HMAG'] > 5) & (Vsum['HMAG'] < 15) & (np.isnan(Vsum['HMAG']) == False) & 
                               (np.isnan(Vsum['SNR']) == False) & (Vsum['SNR'] > 0) & (Vsum['ASSIGNED']) & 
                               (Vsum['ON_TARGET']) & (Vsum['VALID']) & (Vsum['OBJTYPE'] != 'none'))
        else:
            notsky, = np.where((Vsum['HMAG'] > 5) & (Vsum['HMAG'] < 15) & (np.isnan(Vsum['HMAG']) == False) & 
                               (np.isnan(Vsum['SNR']) == False) & (Vsum['SNR'] > 0) & (Vsum['OBJTYPE'] != 'none'))

        if len(notsky) > 10:
            # First pass at fitting line to S/N as function of Hmag
            hmag1 = Vsum['HMAG'][notsky]
            sn1 = Vsum['SNR'][notsky]
            polynomial1 = np.poly1d(np.polyfit(hmag1, np.log10(sn1), 1))
            yarrnew1 = polynomial1(hmag1)
            diff1 = np.log10(sn1) - yarrnew1
            gd1, = np.where(diff1 > -np.nanstd(diff1))
            # Second pass at fitting line to S/N as function of Hmag
            hmag2 = hmag1[gd1]
            sn2 = sn1[gd1]
            polynomial2 = np.poly1d(np.polyfit(hmag2, np.log10(sn2), 1))
            yarrnew2 = polynomial2(hmag2)
            diff2 = np.log10(sn2) - yarrnew2
            gd2, = np.where(diff2 > -np.nanstd(diff2))
            # Final pass at fitting line to S/N as function of Hmag
            hmag3 = hmag2[gd2]
            sn3 = sn2[gd2]
            polynomial3 = np.poly1d(np.polyfit(hmag3, np.log10(sn3), 1))
            xarrnew3 = np.linspace(np.nanmin(hmag1), np.nanmax(hmag1), 5000)
            yarrnew3 = polynomial3(xarrnew3)
            ratio = np.zeros(len(notsky))
            eta = np.full(len(notsky), -999.9)
            zeta = np.full(len(notsky), -999.9)
            for q in range(len(notsky)):
                hmdif = np.absolute(hmag1[q] - xarrnew3)
                pp, = np.where(hmdif == np.nanmin(hmdif))
                ratio[q] = sn1[q] / 10**yarrnew3[pp][0]
                g, = np.where(Vsum['APOGEE_ID'][notsky][q] == plSum2['TMASS_STYLE'])
                if len(g) > 0:
                    eta[q] = plSum2['ETA'][g][0]
                    zeta[q] = plSum2['ZETA'][g][0]

            telluric, = np.where((eta > -900) & ((Vsum['OBJTYPE'][notsky] == 'SPECTROPHOTO_STD') | (Vsum['OBJTYPE'][notsky] == 'HOT_STD')))
            if len(telluric) > 0:
                x = zeta[telluric]
                y = eta[telluric]
                c = ratio[telluric]
                l = 'telluric'
                sc = ax2.scatter(x, y, marker='o', s=100, c=c, cmap='CMRmap', edgecolors='k', vmin=0, vmax=1, linewidth=0.75, label=l)
            science, = np.where((eta > -900) & ((Vsum['OBJTYPE'][notsky] == 'OBJECT') | (Vsum['OBJTYPE'][notsky] == 'STAR')))
            if len(science) > 0:
                x = zeta[science]
                y = eta[science]
                c = ratio[science]
                l = 'science'
                sc = ax2.scatter(x, y, marker='*', s=250, c=c, cmap='CMRmap', edgecolors='k', vmin=0, vmax=1, linewidth=0.75, label=l)

            ax1_divider = make_axes_locatable(ax2)
            cax1 = ax1_divider.append_axes("top", size="4%", pad="1%")
            cb = colorbar(sc, cax=cax1, orientation="horizontal")
            cax1.xaxis.set_ticks_position("top")
            #cax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax2.text(0.5, 1.13, r'obs SNR $/$ fit SNR', ha='center', transform=ax2.transAxes)
            ax2.legend(loc='upper left', labelspacing=0.5, handletextpad=-0.1, facecolor='lightgrey', fontsize=fontsize*0.75)

        fig.subplots_adjust(left=0.03,right=0.99,bottom=0.098,top=0.90,hspace=0.09,wspace=0.07)
        plt.savefig(plotsdir+plotfile)
        plt.close('all')
        
    oldplotfile = fluxfile.replace('Flux-', 'Flux-block-').replace('.fits', '.png')
    if os.path.exists(plotsdir + oldplotfile): os.remove(plotsdir + oldplotfile)

    #----------------------------------------------------------------------------------------------
    # PLOTS 8: throughput histograms
    #----------------------------------------------------------------------------------------------

    #fluxfile = os.path.basename(load.filename('Flux', num=fluxid, chips=True))
    #oneDfile = os.path.basename(load.filename('1D', num=ims[i], mjd=mjd, chips=True)).replace('.fits','')
    #flux = load.apFlux(fluxid)
    oneD = load.ap1D(fluxid)
    ypos = 300 - platesum2['FIBERID']
    block = np.floor((plSum2['FIBERID'] - 1) / 30) #[::-1]

    plotfile = fluxfile.replace('.fits', '.png').replace('Flux','Tput')
    if (os.path.exists(plotsdir+plotfile) == False) | (clobber == True):
        print("----> makeObsPlots: Making "+plotfile)

        fig=plt.figure(figsize=(20,10))
        xmin = 0.5
        xmax = 300.5
        ymin = 0
        ymax = 1.08
        xspan = xmax-xmin
        yspan = ymax-ymin
        nbins = 300

        mtpLabelPos = np.arange(0,330,30)
        xarr = np.arange(0,300,1)+1
        for ichip in range(nchips):
            c = chiplab[ichip]
            ax = plt.subplot2grid((3,1), (ichip,0))
            ax.minorticks_on()
            ax.grid(True)
            ax.set_xlim(xmin,xmax)
            ax.set_ylim(ymin,ymax)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(30))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))
            ax.tick_params(axis='both',which='both',direction='out',bottom=True,top=True,left=True,right=True,labelsize=fontsize*0.75)
            ax.tick_params(axis='both',which='major',length=axmajlen)
            ax.tick_params(axis='both',which='minor',length=axminlen)
            ax.tick_params(axis='both',which='both',width=axwidth)
            for axis in ['top','bottom','left','right']: ax.spines[axis].set_linewidth(axwidth)
            if ichip == nchips-1: ax.set_xlabel(r'Fiber #')
            if ichip == 1: ax.set_ylabel(r'Flux / Max Flux')
            if ichip < nchips-1: ax.axes.xaxis.set_ticklabels([])
            #ax.hlines([1,0.75,0.50,0.25], xmin=xmin, xmax=xmax, linestyles='dashed', colors='grey', zorder=1)
            ax.vlines([30,60,90,120,150,180,210,240,270], ymin=ymin, ymax=ymax, colors='k', linewidths=2, linestyles='dashed', zorder=11)

            chip = chips[ichip]
            med = np.nanmedian(oneD[chip][1].data, axis=1)
            tput = med / np.nanmax(med)
            ax.bar(xarr, tput[::-1], label=chiplab[ichip]+'\n'+'chip', color=c, width=1, zorder=10)
            for imtp in range(len(mtpLabelPos-1)):
                if ichip == 0: ax.text(mtpLabelPos[imtp]+15, 1.25, 'MTP '+str(imtp+1), ha='center', fontsize=fontsize*0.75)
                g, = np.where((tput > 0.2) & (xarr > mtpLabelPos[imtp]) & (xarr <= mtpLabelPos[imtp]))
                tputPercentage = str(int(round(np.mean(tput[g]*100))))+'%'
                ax.text(mtpLabelPos[imtp]+15, 1.15, tputPercentage, ha='center', color=c, fontsize=fontsize*0.75)


        fig.subplots_adjust(left=0.052,right=0.985,bottom=0.08,top=0.92,hspace=0.2,wspace=0.07)
        plt.savefig(plotsdir+plotfile)
        plt.close('all')
    pdb.set_trace()

    #----------------------------------------------------------------------------------------------
    # PLOTS 7: sky, telluric, science fiber positions, colored by Hmag
    #----------------------------------------------------------------------------------------------
    fluxfile = os.path.basename(load.filename('Flux', num=fluxid, chips=True))
    flux = load.apFlux(fluxid)
    ypos = 300 - platesum2['FIBERID']
    block = np.floor((plSum2['FIBERID'] - 1) / 30) #[::-1]
    fiblabs = np.array(['SKY', 'HOT_STD', 'OBJECT'])
    if fps: fiblabs = np.array(['SKY', 'HOT_STD', 'STAR'])

    plotfile = fluxfile.replace('.fits', '.png').replace('Flux', 'FibLoc')
    if (os.path.exists(plotsdir+plotfile) == False) | (clobber == True):
        print("----> makeObsPlots: Making "+plotfile)

        fig=plt.figure(figsize=(25,10))
        plotrad = 1.6

        for itype in range(3):
            ax = plt.subplot2grid((1, 3), (0,itype))
            ax.set_xlim(-plotrad, plotrad)
            ax.set_ylim(-plotrad, plotrad)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.minorticks_on()
            ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
            ax.tick_params(axis='both',which='major',length=axmajlen)
            ax.tick_params(axis='both',which='minor',length=axminlen)
            ax.tick_params(axis='both',which='both',width=axwidth)
            ax.set_xlabel(r'Zeta (deg.)')
            if itype == 0: ax.set_ylabel(r'Eta (deg.)')
            if itype != 0: ax.axes.yaxis.set_ticklabels([])

            try:
                if fps: 
                    gd, = np.where((platesum2['HMAG'] > 5) & (platesum2['HMAG'] < 15) & (np.isnan(platesum2['HMAG']) == False) & 
                                   (platesum2['ASSIGNED']) & (platesum2['ON_TARGET']) & (platesum2['VALID']) & 
                                   (platesum2['OBJTYPE'] == fiblabs[itype]))
                    if itype == 0:
                        gd, = np.where(platesum2['OBJTYPE'] == fiblabs[itype])

                else: 
                    gd, = np.where(platesum2['objtype'] == fiblabs[itype])
                if len(gd) > 0:
                    x = platesum2['Zeta'][gd]
                    y = platesum2['Eta'][gd]
                    c = platesum2['HMAG'][gd]
                    sc = ax.scatter(x, y, marker='o', s=100, c=c, edgecolors='k', cmap='afmhot', alpha=1)
                    ax_divider = make_axes_locatable(ax)
                    cax = ax_divider.append_axes("top", size="4%", pad="1%")
                    if (len(gd) > 1) & (fiblabs[itype] != 'SKY'): 
                        cb = colorbar(sc, cax=cax, orientation="horizontal")
                        ax.text(0.5, 1.13, r'$H$ (mag)',ha='center', transform=ax.transAxes)
                    cax.xaxis.set_ticks_position("top")
                    cax.minorticks_on()
                else:
                    sc = ax.scatter([-100,-100], [-100,-100], marker='o', s=100, edgecolors='k', cmap='afmhot', alpha=1)
                    ax_divider = make_axes_locatable(ax)
                    cax = ax_divider.append_axes("top", size="4%", pad="1%")
                    cax.xaxis.set_ticks_position("top")
                    cax.axes.yaxis.set_ticklabels([])
                    cax.minorticks_on()
            except:
                nothing = 5

            txt = fiblabs[itype].replace('HOT_STD', 'TELLURIC').replace('STAR', 'SCIENCE').lower() + ' (' + str(len(gd)) + ')'
            ax.text(0.03, 0.97, txt, transform=ax.transAxes, ha='left', va='top', color='k')

        fig.subplots_adjust(left=0.045,right=0.985,bottom=0.09,top=0.90,hspace=0.09,wspace=0.04)
        plt.savefig(plotsdir+plotfile)
        plt.close('all')
        
    #----------------------------------------------------------------------------------------------
    # PLOT 7: guider rms plot
    #----------------------------------------------------------------------------------------------
    expdir = os.environ.get('APOGEE_REDUX')+'/'+apred+'/'+'exposures/'+instrument+'/'
    gcamfile = expdir+mjd+'/gcam-'+mjd+'.fits'
    if os.path.exists(gcamfile):
        gcam = fits.getdata(gcamfile)

        dateobs = plSum1['DATEOBS'][0]
        tt = Time(dateobs)
        mjdstart = tt.mjd
        exptime = np.sum(plSum1['EXPTIME'])
        mjdend = mjdstart + (exptime/(24*60*60))
        jcam, = np.where((gcam['mjd'] > mjdstart) & (gcam['mjd'] < mjdend))

        plotfile = 'guider-'+plate+'-'+mjd+'.png'
        if (os.path.exists(plotsdir+plotfile) == False) | (clobber == True):
            print("----> makeObsPlots: Making "+plotfile)

            fig=plt.figure(figsize=(10,10))
            ax1 = plt.subplot2grid((1,1), (0,0))
            ax1.tick_params(reset=True)
            ax1.minorticks_on()
            ax1.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
            ax1.tick_params(axis='both',which='major',length=axmajlen)
            ax1.tick_params(axis='both',which='minor',length=axminlen)
            ax1.tick_params(axis='both',which='both',width=axwidth)
            ax1.set_xlabel(r'Minutes since visit start')
            ax1.set_ylabel(r'Guider RMS')

            x = (gcam['mjd'][jcam] - np.min(gcam['mjd'][jcam]))*60*24
            ax1.plot(x, gcam['gdrms'][jcam], color='k')

            fig.subplots_adjust(left=0.125,right=0.98,bottom=0.08,top=0.98,hspace=0.2,wspace=0.0)
            plt.savefig(plotsdir+plotfile)
            plt.close('all')

    # Loop over the exposures to make other plots.
    for i in range(n_exposures):
        gd, = np.where(ims[i] == plSum1['IM'])
        if len(gd) >= 1:
            ii = gd[0]
            #------------------------------------------------------------------------------------------
            # PLOTS 8: 3 panel mag/SNR plots for each exposure
            #----------------------------------------------------------------------------------------------
            plotfile = prefix+'1D-'+str(plSum1['IM'][ii])+'_magplots.png'
            if (os.path.exists(plotsdir+plotfile) == False) | (clobber == True):
                print("----> makeObsPlots: Making "+plotfile)

                telluric, = np.where((plSum2['OBJTYPE'] == 'SPECTROPHOTO_STD') | (plSum2['OBJTYPE'] == 'HOT_STD'))
                ntelluric = len(telluric)
                science, = np.where((plSum2['OBJTYPE'] != 'SPECTROPHOTO_STD') & (plSum2['OBJTYPE'] != 'HOT_STD') & (plSum2['OBJTYPE'] != 'SKY'))
                nscience = len(science)
                sky, = np.where(plSum2['OBJTYPE'] == 'SKY')
                nsky = len(sky)

                notsky, = np.where((plSum2['HMAG'] > 0) & (plSum2['HMAG'] < 30))
                hmagarr = plSum2['HMAG'][notsky]
                try:
                    minH = np.nanmin(hmagarr);  maxH = np.nanmax(hmagarr)
                except:
                    minH = 6;  maxH = 14
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
                ax1.text(-0.15,0.50,r'm = -2.5*log(counts)',transform=ax1.transAxes,rotation=90,ha='left',va='center')
                ax2.text(-0.15,0.50,r'$H$ - (m+zero)',transform=ax2.transAxes,rotation=90,ha='left',va='center')
                ax3.text(-0.15,0.50,r'S/N',transform=ax3.transAxes,rotation=90,ha='left',va='center')

                # PLOTS 8a: observed mag vs H mag
                x = plSum2['HMAG'][science];    y = plSum2['obsmag'][science,ii,1]-plSum1['ZERO'][ii]
                ax1.scatter(x, y, marker='*', s=180, edgecolors='k', alpha=alpha, c='r', label='Science')
                if ntelluric>0:
                    x = plSum2['HMAG'][telluric];   y = plSum2['obsmag'][telluric,ii,1]-plSum1['ZERO'][ii]
                    ax1.scatter(x, y, marker='o', s=60, edgecolors='k', alpha=alpha, c='dodgerblue', label='Telluric')
                ax1.legend(loc='upper left', labelspacing=0.5, handletextpad=-0.1, facecolor='lightgrey')

                # PLOTS 8b: observed mag - fit mag vs H mag
                x = plSum2['HMAG'][science];    y = x - plSum2['obsmag'][science,ii,1]
                yminsci = np.nanmin(y); ymaxsci = np.nanmax(y)
                ax2.scatter(x, y, marker='*', s=180, edgecolors='k', alpha=alpha, c='r')
                if ntelluric>0:
                    x = plSum2['HMAG'][telluric];   y = x - plSum2['obsmag'][telluric,ii,1]
                    ymintel = np.nanmin(y); ymaxtel = np.nanmax(y)
                    ax2.scatter(x, y, marker='o', s=60, edgecolors='k', alpha=alpha, c='dodgerblue')
                    ymin = np.min([yminsci,ymintel])
                    ymax = np.max([ymaxsci,ymaxtel])
                else:
                    ymin = yminsci
                    ymax = ymaxsci
                yspan=ymax-ymin
                #ax2.set_ylim(ymin-(yspan*0.05),ymax+(yspan*0.05))
                ax2.set_ylim(-8,2)

                # PLOTS 8c: S/N as calculated from ap1D frame
                #c = ['r','g','b']
                #for ichip in range(nchips):
                #    x = plSum2['HMAG'][science];   y = plSum2['SN'][science,i,ichip]
                #    ax3.semilogy(x, y, marker='*', ms=15, mec='k', alpha=alpha, mfc=c[ichip], linestyle='')
                #    x = plSum2['HMAG'][telluric];   y = plSum2['SN'][telluric,i,ichip]
                #    ax3.semilogy(x, y, marker='o', ms=9, mec='k', alpha=alpha, mfc=c[ichip], linestyle='')
                x = plSum2['HMAG'][science];   y = plSum2['SN'][science,ii,1]
                yminsci = np.nanmin(y); ymaxsci = np.nanmax(y)
                ax3.semilogy(x, y, marker='*', ms=15, mec='k', alpha=alpha, mfc='r', linestyle='')
                if ntelluric>0:
                    x = plSum2['HMAG'][telluric];   y = plSum2['SN'][telluric,ii,1]
                    ymintel = np.nanmin(y); ymaxtel = np.nanmax(y)
                    ax3.semilogy(x, y, marker='o', ms=9, mec='k', alpha=alpha, mfc='dodgerblue', linestyle='')
                    ymin = np.min([yminsci,ymintel])
                    ymax = np.max([ymaxsci,ymaxtel])
                else:
                    ymin = yminsci
                    ymax = ymaxsci
                if np.isfinite(ymin)==False:
                    ymin = 1.0
                if np.isfinite(ymax)==False:
                    ymax = 200.0
                yspan=ymax-ymin
                ax3.set_ylim(ymin-(yspan*0.05),ymax+(yspan*0.05))

                # overplot the target S/N line
                sntarget = 100 * np.sqrt(plSum1['EXPTIME'][ii] / (3.0 * 3600))
                sntargetmag = 12.2
                x = [sntargetmag - 10, sntargetmag + 2.5];    y = [sntarget * 100, sntarget / np.sqrt(10)]
                ax3.plot(x, y, color='k',linewidth=1.5)

                fig.subplots_adjust(left=0.14,right=0.978,bottom=0.08,top=0.99,hspace=0.2,wspace=0.0)
                plt.savefig(plotsdir+plotfile)
                plt.close('all')

            #------------------------------------------------------------------------------------------
            # PLOT 9: spatial residuals for each exposure
            #----------------------------------------------------------------------------------------------
            plotfile = prefix+'1D-'+str(plSum1['IM'][ii])+'_spatialresid.png'
            if (os.path.exists(plotsdir+plotfile) == False) | (clobber == True):
                print("----> makeObsPlots: Making "+plotfile)

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
                #cmap = plt.get_cmap('jet');    minval = 0.05;    maxval = 0.92;    ncol = 100
                #gdcmap = mplcolors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, 
                #           a=minval, b=maxval), cmap(np.linspace(minval, maxval, ncol)))

                try:
                    ass, = np.where(plSum2['ASSIGNED'][science])
                    x = plSum2['ZETA'][science][ass];    y = plSum2['ETA'][science][ass]
                    c = plSum2['HMAG'][science][ass] - plSum2['obsmag'][science[ass],ii,1]
                except:
                    x = plSum2['ZETA'][science];    y = plSum2['ETA'][science]
                    c = plSum2['HMAG'][science] - plSum2['obsmag'][science,ii,1]
                psci = ax1.scatter(x, y, marker='*', s=400, c=c, edgecolors='k', cmap=cmap, alpha=1, vmin=-0.5, vmax=0.5, label='Science')

                if ntelluric>0:
                    try:
                        ass, = np.where(plSum2['ASSIGNED'][telluric])
                        x = plSum2['ZETA'][telluric][ass];    y = plSum2['ETA'][telluric][ass]
                        c = plSum2['HMAG'][telluric][ass] - plSum2['obsmag'][telluric[ass],ii,1]
                    except:
                        x = plSum2['ZETA'][telluric];    y = plSum2['ETA'][telluric]
                        c = plSum2['HMAG'][telluric] - plSum2['obsmag'][telluric,ii,1]
                    ptel = ax1.scatter(x, y, marker='o', s=215, c=c, edgecolors='k', cmap=cmap, alpha=1, vmin=-0.5, vmax=0.5, label='Telluric')

                #try:
                #    x = plSum2['ZETA'][sky];    y = plSum2['ETA'][sky]
                #    c = plSum2['HMAG'][sky] - plSum2['obsmag'][sky,i,1]
                #    psky = ax1.scatter(x, y, marker='s', s=140, c='white', edgecolors='k', alpha=1, label='Sky')
                #except:
                #    print("----> makeObsPlots: Problem!!! Sky fiber subscripting error when trying to make spatial mag. plots.")

                ax1.legend(loc='upper left', labelspacing=0.5, handletextpad=-0.1, facecolor='lightgrey')

                ax1_divider = make_axes_locatable(ax1)
                cax1 = ax1_divider.append_axes("top", size="4%", pad="1%")
                cb = colorbar(psci, cax=cax1, orientation="horizontal")
                cax1.xaxis.set_ticks_position("top")
                cax1.minorticks_on()
                ax1.text(0.5, 1.12, r'$H$ + 2.5*log(m - zero)',ha='center', transform=ax1.transAxes)

                fig.subplots_adjust(left=0.11,right=0.97,bottom=0.07,top=0.91,hspace=0.2,wspace=0.0)
                plt.savefig(plotsdir+plotfile)
                plt.close('all')

            #------------------------------------------------------------------------------------------
            # PLOT 10: spatial sky line emission
            # https://data.sdss.org/sas/apogeework/apogee/spectro/redux/current/plates/5583/56257/plots/ap1D-06950025sky.jpg
            #------------------------------------------------------------------------------------------
            plotfile = prefix+'1D-'+str(plSum1['IM'][gd][0])+'_skyemission.png'
            if (os.path.exists(plotsdir+plotfile) == False) | (clobber == True):
                print("----> makeObsPlots: Making "+plotfile)

                #d = load.apPlate(int(plate), mjd) 
                d = load.ap1D(ims[i])
                rows = 300 - platesum2['FIBERID']

                fibersky, = np.where(platesum2['OBJTYPE'] == 'SKY')
                nsky = len(fibersky)
                if nsky>0:
                    sky = rows[fibersky]
                else:
                    sky = []

                fibertelluric, = np.where((platesum2['OBJTYPE'] == 'SPECTROPHOTO_STD') | (platesum2['OBJTYPE'] == 'HOT_STD'))
                ntelluric = len(fibertelluric)
                if ntelluric>0:
                    telluric = rows[fibertelluric]
                else:
                    telluric = []

                fiberobj, = np.where((platesum2['OBJTYPE'] == 'STAR_BHB') | (platesum2['OBJTYPE'] == 'STAR') |
                                     (platesum2['OBJTYPE'] == 'EXTOBJ') | (platesum2['OBJTYPE'] == 'OBJECT'))
                nobj = len(fiberobj)
                obj = rows[fiberobj]

                # Define skylines structure which we will use to get crude sky levels in lines.
                dt = np.dtype([('W1',   np.float64),
                               ('W2',   np.float64),
                               ('C1',   np.float64),
                               ('C2',   np.float64),
                               ('C3',   np.float64),
                               ('C4',   np.float64),
                               ('FLUX', np.float64, (nfiber)),
                               ('TYPE', np.int32)])

                skylines = np.zeros(2,dtype=dt)
                nskylines=len(skylines)

                skylines['W1']   = 16230.0, 15990.0
                skylines['W2']   = 16240.0, 16028.0
                skylines['C1']   = 16215.0, 15980.0
                skylines['C2']   = 16225.0, 15990.0
                skylines['C3']   = 16245.0, 0.0
                skylines['C4']   = 16255.0, 0.0
                skylines['TYPE'] = 1, 0

                try:
                    for iline in range(nskylines):
                        skylines['FLUX'][iline] = getflux(d=d, skyline=skylines[iline], rows=rows)

                    medsky = np.nanmedian(skylines['FLUX'][0][fibersky])

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

                    xx = platesum2['ZETA'][fiberobj]
                    yy = platesum2['ETA'][fiberobj]
                    cc = skylines['FLUX'][0][fiberobj] / medsky
                    ax1.scatter(xx, yy, marker='*', s=400, c=cc, edgecolors='k', cmap=cmap, alpha=1, vmin=0.9, vmax=1.1, label='Science')

                    if ntelluric>0:
                        xx = platesum2['ZETA'][fibertelluric]
                        yy = platesum2['ETA'][fibertelluric]
                        cc = skylines['FLUX'][0][fibertelluric] / medsky
                        ax1.scatter(xx, yy, marker='o', s=215, c=cc, edgecolors='k', cmap=cmap, alpha=1, vmin=0.9, vmax=1.1, label='Telluric')

                    if nsky>0:
                        xx = platesum2['ZETA'][fibersky]
                        yy = platesum2['ETA'][fibersky]
                        cc = skylines['FLUX'][0][fibersky] / medsky
                        sc = ax1.scatter(xx, yy, marker='s', s=230, c=cc, edgecolors='k', cmap=cmap, alpha=1, vmin=0.9, vmax=1.1, label='Sky')

                    ax1.legend(loc='upper left', labelspacing=0.5, handletextpad=-0.1, facecolor='lightgrey')

                    ax1_divider = make_axes_locatable(ax1)
                    cax1 = ax1_divider.append_axes("top", size="4%", pad="1%")
                    cb = colorbar(sc, cax=cax1, orientation="horizontal")
                    cax1.xaxis.set_ticks_position("top")
                    cax1.minorticks_on()
                    ax1.text(0.5, 1.12, r'Sky emission deviation',ha='center', transform=ax1.transAxes)
                except:
                    nothing = 5

                fig.subplots_adjust(left=0.11,right=0.970,bottom=0.07,top=0.91,hspace=0.2,wspace=0.0)
                plt.savefig(plotsdir+plotfile)
                plt.close('all')

            #------------------------------------------------------------------------------------------
            # PLOT 11: spatial continuum emission
            # https://data.sdss.org/sas/apogeework/apogee/spectro/redux/current/plates/5583/56257/plots/ap1D-06950025skycont.jpg
            #------------------------------------------------------------------------------------------
            plotfile = prefix+'1D-'+str(plSum1['IM'][ii])+'_skycontinuum.png'
            if (os.path.exists(plotsdir+plotfile) == False) | (clobber == True):
                print("----> makeObsPlots: Making "+plotfile)

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

                skyzero=14.75 + 2.5 * np.log10(plSum1['NREADS'][ii])
                xx = platesum2['ZETA'][fibersky]
                yy = platesum2['ETA'][fibersky]
                cc = platesum2['obsmag'][fibersky, ii, 1] + skyzero - plSum1['ZERO'][ii]
                sc = ax1.scatter(xx, yy, marker='s', s=270, c=cc, edgecolors='k', cmap=cmap, alpha=1, vmin=13, vmax=15)

                ax1_divider = make_axes_locatable(ax1)
                cax1 = ax1_divider.append_axes("top", size="4%", pad="1%")
                cb = colorbar(sc, cax=cax1, orientation="horizontal")
                cax1.xaxis.set_ticks_position("top")
                cax1.minorticks_on()
                ax1.text(0.5, 1.12, r'Sky continuum (mag.)',ha='center', transform=ax1.transAxes)

                fig.subplots_adjust(left=0.11,right=0.970,bottom=0.07,top=0.91,hspace=0.2,wspace=0.0)
                plt.savefig(plotsdir+plotfile)
                plt.close('all')


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

    #plt.ion()
    print("----> makeObsPlots: Done with plate "+plate+", MJD "+mjd+"\n")

###################################################################################################
''' makeVisHTML: make the plate/visit level html '''
def makeVisHTML(load=None, plate=None, mjd=None, survey=None, apred=None, telescope=None, fluxid=None): 
    start_time = time.time()

    print("----> makeVisHTML: Running plate " + plate + ", MJD " + mjd)

    # HTML header background color
    thcolor = '#DCDCDC'

    if int(mjd)>59556:
        fps = True
    else:
        fps = False
    prefix = 'ap'

    if telescope == 'lco25m':
        if int(mjd)>59808:
            fps = True
        else:
            fps = False
        prefix = 'as'

    apodir = os.environ.get('APOGEE_REDUX') + '/'

    # Make html directory if it doesn't already exist.
    htmldir = os.path.dirname(load.filename('Plate', plate=int(plate), mjd=mjd, chips=True, fps=fps)) + '/html/'
    if os.path.exists(htmldir) == False: os.makedirs(htmldir)
    plotdir = htmldir.replace('html','plots')

    # Get the HTML file name... apPlate-plate-mjd
    htmlfile = os.path.basename(load.filename('Plate', plate=int(plate), mjd=mjd, chips=True, fps=fps)).replace('.fits','')

    # Base directory where star-level stuff goes
    starHTMLbase = apodir + apred + '/stars/' + telescope +'/'

    # Load in the apPlate file
    apPlate = load.apPlate(int(plate), mjd)
    try:
        data = apPlate['a'][11].data[::-1]
    except:
        print("----> makeVisHTML: PROBLEM! " + prefix + "Plate not found for plate " + plate + ", MJD " + mjd)
        return
    nfiber = len(data)

    # Read in flux file to get an idea of throughput
    fluxfile = os.path.basename(load.filename('Flux', num=fluxid, chips=True))
    flux = load.apFlux(fluxid)
    medflux = np.nanmedian(flux['a'][1].data, axis=1)[::-1]
    throughput = medflux / np.nanmax(medflux)

    # DB query for this visit
    db = apogeedb.DBSession()
    vcat = db.query('visit', where="plate='" + plate + "' and mjd='" + mjd + "' and telescope='" + telescope + "'", fmt='table')
    db.close()
    stars, = np.where((vcat['assigned'] == 1) & (vcat['objtype'] != 'SKY'))
    ustars,uind = np.unique(vcat['apogee_id'][stars], return_index=True)
    nustars = len(ustars)

    # FITS table structure.
    dt = np.dtype([('GMAG',      np.float64),
                   ('BPMAG',     np.float64),
                   ('RPMAG',     np.float64),
                   ('JMAG',      np.float64),
                   ('HMAG',      np.float64),
                   ('KSMAG',     np.float64),
                   ('GMAG_ERR',  np.float64),
                   ('BPMAG_ERR', np.float64),
                   ('RPMAG_ERR', np.float64),
                   ('JMAG_ERR',  np.float64),
                   ('HMAG_ERR',  np.float64),
                   ('KSMAG_ERR', np.float64)])
    colorteffarr = np.zeros(nustars,dtype=dt)
    #colorteffarr['APOGEE_ID'] = ustars
    colorteffarr['GMAG'] = vcat['gaiadr2_gmag'][stars][uind]
    colorteffarr['GMAG_ERR'] = vcat['gaiadr2_gerr'][stars][uind]
    colorteffarr['BPMAG'] = vcat['gaiadr2_bpmag'][stars][uind]
    colorteffarr['BPMAG_ERR'] = vcat['gaiadr2_bperr'][stars][uind]
    colorteffarr['RPMAG'] = vcat['gaiadr2_rpmag'][stars][uind]
    colorteffarr['RPMAG_ERR'] = vcat['gaiadr2_rperr'][stars][uind]
    colorteffarr['JMAG'] = vcat['jmag'][stars][uind]
    colorteffarr['JMAG_ERR'] = vcat['jerr'][stars][uind]
    colorteffarr['HMAG'] = vcat['kmag'][stars][uind]
    colorteffarr['HMAG_ERR'] = vcat['kerr'][stars][uind]
    colorteffarr['KSMAG'] = vcat['hmag'][stars][uind]
    colorteffarr['KSMAG_ERR'] = vcat['herr'][stars][uind]
    tab = Table(colorteffarr)
    teff = np.zeros(nustars)
    av = np.zeros(nustars)
    for i in range(nustars):
        #pdb.set_trace()
        #tab = Table(colorteffarr[i])
        tmp = colorteff.solve(tab[i])
        teff[i] = tmp[0]
        av[i] = tmp[1]

    # For each star, create the exposure entry on the web page and set up the plot of the spectrum.
    vishtml = open(htmldir + htmlfile + '.html', 'w')
    vishtml.write('<HTML>\n')
    vishtml.write('<HEAD><script src="../../../../../../../sorttable.js"></script><title>' + htmlfile + '</title></head>\n')
    vishtml.write('<BODY>\n')

    vishtml.write('<H1>' + htmlfile + '</H1>\n')
    vishtml.write('<H3>' + str(len(stars)) + ' stars observed</H3><HR>\n')
    vishtml.write('<P><B>Note:</B> the "Dflat Tput" column gives the median dome flat flux in each ')
    vishtml.write('fiber divided by the maximum median dome flat flux across all fibers. ')
    vishtml.write('<P><B>Note:</B> the "Rel S/N" column gives the ratio of the observed S/N ')
    vishtml.write('over the linear fit to high S/N fibers. ')
    vishtml.write('<P>Low numbers in the aforementioned columns are generally bad, and the columns are color-coded accordingly.</P>\n')
    vishtml.write('<P>Click the column headers to sort.</p>\n')
    vishtml.write('<TABLE BORDER=2 CLASS="sortable">\n')
    vishtml.write('<TR bgcolor="' + thcolor + '"><TH>Fiber<BR>(MTP) <TH>APOGEE ID <TH>Hmag <TH>Raw<BR>J - K <TH>Targ<BR>Type <TH>Target & Data Flags')
    vishtml.write('<TH>Obs.<BR>S/N <TH>Rel.<BR>S/N <TH>Dflat<BR>Tput  <TH>Vrad<BR>(km/s) <TH>Ncomp <TH>RV<BR>Teff (K) <TH>RV<BR>log(g)')
    vishtml.write('<TH>RV<BR>[Fe/H] <TH>Phot.<BR>Teff <TH>J-K_0 <TH>apVisit Plot\n')

    # Make text file giving ratio of observed S/N over linear fit S/N
    Vsum = load.apVisitSum(int(plate), mjd)
    Vsum = Vsum[1].data
    if fps:
        notsky, = np.where((Vsum['HMAG'] > 5) & (Vsum['HMAG'] < 15) & (np.isnan(Vsum['HMAG']) == False) & 
                           (np.isnan(Vsum['SNR']) == False) & (Vsum['SNR'] > 0) & (Vsum['ASSIGNED']) & 
                           (Vsum['ON_TARGET']) & (Vsum['VALID']) & (Vsum['OBJTYPE'] != 'none'))
    else:
        notsky, = np.where((Vsum['HMAG'] > 5) & (Vsum['HMAG'] < 15) & (np.isnan(Vsum['HMAG']) == False) & 
                           (np.isnan(Vsum['SNR']) == False) & (Vsum['SNR'] > 0) & (Vsum['OBJTYPE'] != 'none'))
    if len(notsky) > 10:
        # First pass at fitting line to S/N as function of Hmag
        hmag1 = Vsum['HMAG'][notsky]
        sn1 = Vsum['SNR'][notsky]
        apID = Vsum['APOGEE_ID'][notsky]
        polynomial1 = np.poly1d(np.polyfit(hmag1, np.log10(sn1), 1))
        yarrnew1 = polynomial1(hmag1)
        diff1 = np.log10(sn1) - yarrnew1
        gd1, = np.where(diff1 > -np.nanstd(diff1))
        # Second pass at fitting line to S/N as function of Hmag
        hmag2 = hmag1[gd1]
        sn2 = sn1[gd1]
        polynomial2 = np.poly1d(np.polyfit(hmag2, np.log10(sn2), 1))
        yarrnew2 = polynomial2(hmag2)
        diff2 = np.log10(sn2) - yarrnew2
        gd2, = np.where(diff2 > -np.nanstd(diff2))
        # Final pass at fitting line to S/N as function of Hmag
        hmag3 = hmag2[gd2]
        sn3 = sn2[gd2]
        polynomial3 = np.poly1d(np.polyfit(hmag3, np.log10(sn3), 1))
        xarrnew3 = np.linspace(np.nanmin(hmag1), np.nanmax(hmag1), 5000)
        yarrnew3 = polynomial3(xarrnew3)
        ratio = np.empty(len(notsky))
        for q in range(len(notsky)):
            hmdif = np.absolute(hmag1[q] - xarrnew3)
            pp, = np.where(hmdif == np.nanmin(hmdif))
            ratio[q] = sn1[q] / 10**yarrnew3[pp][0]
        g, = np.where(ratio > 0)
        if len(g) > 0:
            apID = apID[g]
            ratio = ratio[g]
        sdata = Table()
        sdata['apogee_id'] = apID
        sdata['relSNR'] = ratio
        platesum = load.filename('PlateSum', plate=int(plate), mjd=mjd, fps=fps)
        snroutfile = platesum.replace(prefix + 'PlateSum', 'relSNR').replace('.fits', '.dat')
        ascii.write(sdata, snroutfile, overwrite=True)

    # Loop over the fibers
    for j in range(300):
        jdata = data[j]
        fiber = jdata['FIBERID']
        if fiber > 0:
            cfiber = str(fiber).zfill(3)
            cblock = str(np.ceil(fiber / 30).astype(int))
            objid = jdata['OBJECT']
            objtype = jdata['OBJTYPE']
            visitplotfile = '../plots/' + prefix + 'Plate-' + plate + '-' + mjd + '-' + cfiber + '.png'
            # Establish html table row background color and spectrum plot color
            bgcolor = 'white'
            if (objtype == 'SPECTROPHOTO_STD') | (objtype == 'HOT_STD'): bgcolor = '#D2B4DE'
            vrad = -999.9
            ncomp = -1
            rvteff = -9999.9
            rvlogg = -9.999
            rvfeh = -9.999
            photteff = -9999.9
            jk0 = -9.999
            apStarRelPath = None
            starHTMLrelPath = None
            if objtype == 'SKY': 
                bgcolor = '#D6EAF8'
                firstcarton = 'SKY'
                starflags = 'None'
            else:
                assigned = 1
                vcatind, = np.where(fiber == vcat['fiberid'])
                if len(vcatind) < 1: pdb.set_trace()
                jvcat = vcat[vcatind][0]
                if jvcat['assigned'] == 0: continue
                jmag = jvcat['jmag']
                hmag = jvcat['hmag']
                kmag = jvcat['kmag']
                snr = jvcat['snr']
                if snr < 0: snr = -1
                if (objtype != 'SKY') & (objid != '2MNone') & (objid != '2M') & (objid != ''):
                    gg, = np.where(objid == ustars)
                    if len(gg) > 0:
                        photteff = teff[gg][0]
                        jk0 = jmag - kmag - 0.17*av[gg][0]
                    apstarfile = load.filename('Star', obj=objid)
                    if os.path.exists(apstarfile):
                        apstarheader = fits.getheader(apstarfile)
                        try: vrad = apstarheader['VRAD']
                        except: vrad = apstarheader['VHBARY']
                        ncomp = apstarheader['N_COMP']
                        rvteff = apstarheader['RV_TEFF']
                        rvlogg = apstarheader['RV_LOGG']
                        rvfeh = apstarheader['RV_FEH']
                        if np.isnan(rvteff): rvteff = -9999
                        if np.isnan(rvlogg): rvlogg = -9.999
                        if np.isnan(rvfeh): rvfeh = -9.999
                        tmp = apstarfile.split(apred + '/')
                        apStarRelPath = '../../../../../../' + tmp[1]
                        starHTMLrelPath = '../../../../../../' + os.path.dirname(tmp[1]) + '/html/' + objid + '.html'
                else:
                    objid = 'None'
                    assigned = 0

                starflags = jvcat['starflags'].replace(',','<BR>')
                firstcarton = jvcat['firstcarton']
                visitfile = jvcat['file']

                # Handle case of unassigned or off target fibers 
                if (jvcat['on_target'] == 0) | (jvcat['assigned'] == 0):
                    bgcolor = 'grey'
                    firstcarton = 'OFF TARGET!!!'
                    if jvcat['assigned'] == 0:
                        bgcolor = 'grey'
                        firstcarton = 'UNASSIGNED!!!'

                # Create SIMBAD link
                cra = str("%.5f" % round(jvcat['ra'], 5))
                cdec = str("%.5f" % round(jvcat['dec'], 5))
                txt1 = '<A HREF="http://simbad.u-strasbg.fr/simbad/sim-coo?Coord='+cra+'+'+cdec+'&CooFrame=FK5&CooEpoch=2000&CooEqui=2000'
                txt2 = '&CooDefinedFrames=none&Radius=10&Radius.unit=arcsec&submit=submit+query&CoordList=" target="_blank">SIMBAD Link</A>'
                simbadlink = txt1 + txt2

                #apStarRelPath = None
                #starHTMLrelPath = None
                #if os.path.exists(apstarfile):
                #    tmp = apstarfile.split(apred + '/')
                #    apStarRelPath = '../../../../../' + tmp[1]
                #    starHTMLrelPath = '../../../../../' + os.path.dirname(tmp[1]) + '/html/'
                    #starDir = starHTMLbase + healpixgroup + '/' + healpix + '/'
                    #starRelPath = '../../../../../stars/' + telescope + '/' + healpixgroup + '/' + healpix + '/'
                    #starHTMLrelPath = '../' + starRelPath + 'html/' + objid + '.html'
                    #apStarCheck = glob.glob(starDir + 'apStar-' + apred + '-' + telescope + '-' + objid + '-*.fits')
                    #if len(apStarCheck) > 0:
                    #    # Find the newest apStar file
                    #    apStarCheck.sort()
                    #    apStarCheck = np.array(apStarCheck)
                    #    apStarNewest = os.path.basename(apStarCheck[-1])
                    #    apStarRelPath = '../' + starRelPath + apStarNewest

            # Write data to HTML table
            if objtype != 'SKY':
                vishtml.write('<TR  BGCOLOR=' + bgcolor + '>\n')
                vishtml.write('<TD align="center">' + cfiber + '<BR>(' + cblock + ')')
                vishtml.write('<TD>' + objid + '\n')
                vishtml.write('<BR>' + simbadlink + '\n')
                vishtml.write('<BR><A HREF=../' + visitfile + '>apVisit file</A>\n')
                if apStarRelPath is not None:
                    vishtml.write('<BR><A HREF=' + apStarRelPath + '>apStar file</A>\n')
                    vishtml.write('<BR><A HREF=' + starHTMLrelPath + ' target="_blank">Star Summary Page</A>\n')
                else:
                    vishtml.write('<BR>apStar file??\n')
                    vishtml.write('<BR>Star Summary Page??\n')
                vishtml.write('<TD align ="center">' + str("%.3f" % round(hmag,3)))
                if (jmag > 0) & (kmag > 0) & (jmag < 90) & (kmag < 90):
                    vishtml.write('<TD align ="center">' + str("%.3f" % round(jmag-kmag,3)))
                else:
                    vishtml.write('<TD align ="center">><FONT COLOR="red">99.999</FONT>')
                if (objtype == 'SPECTROPHOTO_STD') | (objtype == 'HOT_STD'):
                    vishtml.write('<TD align="center">TEL')
                else:
                    vishtml.write('<TD align="center">SCI')
                vishtml.write('<TD align="center">' + firstcarton)
                vishtml.write('<BR><BR>' + starflags)
                vcol = 'black'
                if vrad is not None:
                    if np.absolute(vrad) > 400: vcol = 'red'
                vishtml.write('<TD align ="center">' + str("%.1f" % round(snr,1)))
                # Relative S/N (ratio of obs S/N over linear fit S/N)
                if len(notsky) > 10:
                    g, = np.where(objid == apID)
                    if len(g) > 0:
                        iratio = ratio[g][0] 
                        bcolor1 = 'white'
                        if iratio < 0.7: bcolor1 = '#FFFF66'
                        if iratio < 0.6: bcolor1 = '#FF9933'
                        if iratio < 0.5: bcolor1 = '#FF6633'
                        if iratio < 0.4: bcolor1 = '#FF3333'
                        if iratio < 0.3: bcolor1 = '#FF0000'
                        if (firstcarton == 'UNASSIGNED!!!') | (firstcarton == 'OFF TARGET!!!'): bcolor1 = 'grey'
                        vishtml.write('<TD align ="center" BGCOLOR=' + bcolor1 + '>' + str(int(round(ratio[g][0]*100))) + '%')
                    else: 
                        vishtml.write('<TD align ="center" BGCOLOR="Gray">-1%')
                else: 
                    vishtml.write('<TD align ="center" BGCOLOR="Gray">-1%')
                # Throughput column
                tput = throughput[j]
                if np.isnan(tput) == False:
                    bcolor1 = 'white'
                    if tput < 0.7: bcolor1 = '#FFFF66'
                    if tput < 0.6: bcolor1 = '#FF9933'
                    if tput < 0.5: bcolor1 = '#FF6633'
                    if tput < 0.4: bcolor1 = '#FF3333'
                    if tput < 0.3: bcolor1 = '#FF0000'
                    if (firstcarton == 'UNASSIGNED!!!') | (firstcarton == 'OFF TARGET!!!'): bcolor1 = 'grey'
                    tput = str(int(round(tput*100))) + '%'
                    vishtml.write('<TD align ="center" BGCOLOR=' + bcolor1 + '>' + tput + '\n')
                else:
                    vishtml.write('<TD align ="center" BGCOLOR="grey">-1%\n')

                vishtml.write('<TD align ="center">' + str("%.1f" % round(vrad,1)))
                vishtml.write('<TD align ="center">' + str(ncomp))
                vishtml.write('<TD align ="center">' + str(int(round(rvteff))))
                vishtml.write('<TD align ="center">' + str("%.3f" % round(rvlogg,3)))
                vishtml.write('<TD align ="center">' + str("%.3f" % round(rvfeh,3)))
                vishtml.write('<TD align ="center">' + str(int(round(photteff))))
                vishtml.write('<TD align ="center">' + str("%.3f" % round(jk0,3)))
            else:
                snr = '-9.9'
                relsnr = '-1%'
                fcolor = 'red'
                if objtype != 'SKY': 
                    objtype = 'BLANK'
                    snr = '-99.9'
                    relsnr = '-1%'
                    bgcolor = 'grey'
                    fcolor = 'Black'
                vishtml.write('<TR  BGCOLOR=' + bgcolor + '>\n')
                vishtml.write('<TD align="center">' + cfiber + '<BR>(' + cblock + ')')
                vishtml.write('<TD align="center">' + objtype)
                vishtml.write('<TD align="center">99.999')
                vishtml.write('<TD align="center">99.999')
                vishtml.write('<TD align="center">' + objtype)
                vishtml.write('<TD align="center">' + objtype)
                vishtml.write('<TD align="center">' + snr)
                vishtml.write('<TD align="center">' + relsnr)
                # Throughput column
                tput = throughput[j]
                if np.isnan(tput) == False:
                    bcolor = 'white'
                    if tput < 0.7: bcolor = '#FFFF66'
                    if tput < 0.6: bcolor = '#FF9933'
                    if tput < 0.5: bcolor = '#FF6633'
                    if tput < 0.4: bcolor = '#FF3333'
                    if tput < 0.3: bcolor = '#FF0000'
                    if (firstcarton == 'UNASSIGNED!!!') | (firstcarton == 'OFF TARGET!!!'): bcolor1 = 'grey'

                    tput = str(int(round(tput*100))) + '%'
                    vishtml.write('<TD align ="center" BGCOLOR=' + bcolor + '>' + tput + '\n')
                else:
                    vishtml.write('<TD align ="center" BGCOLOR="grey">-1%\n')
                vishtml.write('<TD align="center">-999.9')
                vishtml.write('<TD align="center">-1')
                vishtml.write('<TD align="center">-9999')
                vishtml.write('<TD align="center">-9.999')
                vishtml.write('<TD align="center">-9.999')
                vishtml.write('<TD align="center">-9.999')
                vishtml.write('<TD align="center">-9.999')

            tmp2 = plotdir + os.path.basename(visitplotfile)
            if (firstcarton != 'UNASSIGNED!!!') & (starflags != 'BAD_PIXELS'):# & (os.path.exists(tmp2)):
                vishtml.write('<TD><A HREF=' + visitplotfile + ' target="_blank"><IMG SRC=' + visitplotfile + ' WIDTH=1000></A>\n')
            else:
                vishtml.write('<TD align="center">')
    vishtml.close()

    runtime = str("%.2f" % (time.time() - start_time))
    print("----> makeVisHTML: Done with plate " + plate + ", MJD " + mjd + " in " + runtime + " seconds.\n")

###################################################################################################
''' makeStarHTML: make the visit and star level html '''
def makeStarHTML(objid=None, load=None, plate=None, mjd=None, survey=None, apred=None, telescope=None): 

    load = apload.ApLoad(apred=apred, telescope=telescope)
    prefix = 'ap'
    if telescope == 'lco25m': prefix = 'as'

    if objid is None:
        print("----> makeStarHTML: Running plate "+plate+", MJD "+mjd)
    else:
        print("----> makeStarHTML: Running on single star:" + objid)

    # HTML header background color
    thcolor = '#DCDCDC'

    # Setup doppler cannon models
    models = doppler.cannon.models
    
    apodir = os.environ.get('APOGEE_REDUX') + '/'

    # Base directory where star-level stuff goes
    starHTMLbase = apodir + apred + '/stars/' + telescope +'/'

    nfiber = 300
    if objid is None: 
        # Load in the apPlate file
        apPlate = load.apPlate(int(plate), mjd)
        data = apPlate['a'][11].data[::-1]
        cnfiber = str(nfiber)
    else:
        nfiber = 1
        cnfiber = '1'

    # Start db session for getting all visit info
    db = apogeedb.DBSession()

    # Loop over the fibers
    for j in range(nfiber):
        if objid is None:
            jdata = data[j]
            fiber = jdata['FIBERID']
        else:
            fiber = 100
        if fiber > 0:
            if objid is None:
                objtype = jdata['OBJTYPE']
                objid = jdata['OBJECT']
            else:
                objtype = 'SCI'
            if (objtype != 'SKY') & (objid != '2MNone') & (objid != '2M') & (objid != ''):
                print("----> makeStarHTML:   making html for " + objid + " (" + str(j+1) + "/" + cnfiber + ")")

                # Find which healpix this star is in
                healpix = apload.obj2healpix(objid)
                healpixgroup = str(healpix // 1000)
                healpix = str(healpix)

                # Find the associated healpix html directories and make them if they don't already exist
                starDir = starHTMLbase + healpixgroup + '/' + healpix + '/'
                starHtmlDir = starDir + 'html/'
                if os.path.exists(starHtmlDir) == False: os.makedirs(starHtmlDir)
                starHTMLpath = starHtmlDir + objid + '.html'

                starRelPath = '../../../../../stars/' + telescope + '/' + healpixgroup + '/' + healpix + '/'
                starHTMLrelPath = '../' + starRelPath + 'html/' + objid + '.html'
                apStarCheck = glob.glob(starDir + 'apStar-' + apred + '-' + telescope + '-' + objid + '-*.fits')
                if len(apStarCheck) > 0:
                    # Find the newest apStar file
                    apStarCheck.sort()
                    apStarCheck = np.array(apStarCheck)
                    apStarNewest = os.path.basename(apStarCheck[-1])
                    apStarRelPath = starRelPath + apStarNewest
                    apStarPath = starDir + apStarNewest
                    apStarModelPath = apStarPath.replace('.fits', '_out_doppler.pkl')

                    # Set up plot directories and plot file name
                    starPlotDir = starDir + 'plots/'
                    if os.path.exists(starPlotDir) == False: os.makedirs(starPlotDir)
                    starPlotFile = 'apStar-' + apred + '-' + telescope + '-' + objid + '_spec+model.png'
                    starPlotFilePath = starPlotDir + starPlotFile
                    starPlotFileRelPath = starRelPath + 'plots/' + starPlotFile
                else:
                    apStarRelPath = None

                # DB query to get visit info
                vcat = db.query('visit_latest', where="apogee_id='" + objid + "' and telescope='"+ telescope + "'", fmt='table')

                # Get visit info from DB
                cgl = str("%.5f" % round(vcat['glon'][0],5))
                cgb = str("%.5f" % round(vcat['glat'][0],5))
                cpmra = str("%.2f" % round(vcat['gaiadr2_pmra'][0],2))
                cpmde = str("%.2f" % round(vcat['gaiadr2_pmdec'][0],2))
                cgmag = str("%.3f" % round(vcat['gaiadr2_gmag'][0],3))
                hmag = vcat['hmag'][0]
                cjmag = str("%.3f" % round(vcat['jmag'][0], 3))
                chmag = str("%.3f" % round(vcat['hmag'][0], 3))
                ckmag = str("%.3f" % round(vcat['kmag'][0],3 ))
                jkcolor = vcat['jmag'][0] - vcat['kmag'][0]
                if (vcat['jmag'][0] < 0) | (vcat['kmag'][0] < 0): jkcolor = -9.999
                cjkcolor = str("%.3f" % round(jkcolor, 3))
                cra = str("%.5f" % round(vcat['ra'][0], 5))
                cdec = str("%.5f" % round(vcat['dec'][0], 5))
                txt1 = '<A HREF="http://simbad.u-strasbg.fr/simbad/sim-coo?Coord='+cra+'+'+cdec+'&CooFrame=FK5&CooEpoch=2000&CooEqui=2000'
                txt2 = '&CooDefinedFrames=none&Radius=10&Radius.unit=arcsec&submit=submit+query&CoordList=" target="_blank">SIMBAD Link</A>'
                simbadlink = txt1 + txt2

                nvis = len(vcat)
                cvrad = '----';  cvscatter = '----'
                gd, = np.where(np.absolute(vcat['vrad']) < 400)
                if len(gd) > 0:
                    vels = vcat['vrad'][gd]
                    cvrad = str("%.2f" % round(np.mean(vels),2))
                    cvscatter = str("%.2f" % round(np.max(vels) - np.min(vels),2))

                rvteff = '----'; rvlogg = '----'; rvfeh = '---'
                gd, = np.where((vcat['rv_teff'] > 0) & (np.absolute(vcat['rv_teff']) < 99999))
                if len(gd) > 0:
                    rvteff = str(int(round(vcat['rv_teff'][gd][0])))
                    rvlogg = str("%.3f" % round(vcat['rv_logg'][gd][0],3))
                    rvfeh = str("%.3f" % round(vcat['rv_feh'][gd][0],3))

                starHTML = open(starHTMLpath, 'w')
                starHTML.write('<HTML>\n')
                starHTML.write('<HEAD><script src="../../../../../../sorttable.js"></script><title>' +objid+ '</title></head>\n')
                starHTML.write('<BODY>\n')
                starHTML.write('<H1>' + objid + ', ' + str(nvis) + ' visits</H1> <HR>\n')
                if apStarRelPath is not None:
                    starHTML.write('<P>' + simbadlink + '<BR><A HREF=' + apStarRelPath + '>apStar File</A>\n')
                else:
                    starHTML.write('<P>' + simbadlink + '<BR>apStar File???\n')
                starHTML.write('<HR>\n')
                starHTML.write('<H3>Star info:</H3>')
                starHTML.write('<TABLE BORDER=2>\n')
                starHTML.write('<TR bgcolor="' + thcolor + '">')

                # Star metadata table
                starHTML.write('<TH>RA <TH>DEC <TH>GLON <TH>GLAT')
                starHTML.write('<TH bgcolor="#E6FFE6">2MASS<BR>J<BR>(mag) <TH bgcolor="#E6FFE6">2MASS<BR>H<BR>(mag) <TH bgcolor="#E6FFE6">2MASS<BR>K<BR>(mag) <TH bgcolor="#E6FFE6">Raw J-K')
                starHTML.write('<TH bgcolor="#FFFFE6">Gaia DR2<BR>PMRA<BR>(mas) <TH bgcolor="#FFFFE6">Gaia DR2<BR>PMDEC<BR>(mas) <TH bgcolor="#FFFFE6">Gaia DR2<BR>G<BR>(mag)') 
                starHTML.write('<TH bgcolor="#E6F2FF">Mean<BR>Vrad<BR>(km/s) <TH bgcolor="#E6F2FF">Min-max<BR>Vrad<BR>(km/s) <TH bgcolor="#E6F2FF">RV Teff<BR>(K)')
                starHTML.write('<TH bgcolor="#E6F2FF">RV logg <TH bgcolor="#E6F2FF">RV [Fe/H] \n')
                starHTML.write('<TR> <TD ALIGN=right>' + cra + '<TD ALIGN=right>' + cdec + ' <TD ALIGN=right>' + cgl)
                starHTML.write('<TD ALIGN=right>' + cgb + '<TD ALIGN=right>' + cjmag + ' <TD ALIGN=right>' +chmag)
                starHTML.write('<TD ALIGN=right>' + ckmag + '<TD ALIGN=right>' + cjkcolor + ' <TD ALIGN=right>' +cpmra)
                starHTML.write('<TD ALIGN=right>' + cpmde + '<TD ALIGN=right>' + cgmag + '<TD ALIGN=right>' + cvrad)
                starHTML.write('<TD ALIGN=right>' + cvscatter)
                starHTML.write('<TD ALIGN=right>' + rvteff + ' <TD ALIGN=right>' + rvlogg + ' <TD ALIGN=right>' + rvfeh + '</TR>')
                starHTML.write('</TABLE>\n<BR>\n')
                starHTML.write('<HR>\n')

                # Star + best fitting model plot
                starHTML.write('<H3>apStar (black), best Doppler Cannon model fit (red), and model-apStar residuals (blue):</H3>')
                if apStarRelPath is not None:
                    starHTML.write('<TD><A HREF=' + starPlotFileRelPath + ' target="_blank"><IMG SRC=' + starPlotFileRelPath + ' WIDTH=1000></A></TR>\n')
                else:
                    starHTML.write('<P>No apStar file for this object!</P>\n')
                starHTML.write('<HR>\n')

                # Star visit table
                starHTML.write('<H3>Visit info:</H3>')
                starHTML.write('<P><B>MJD links:</B> QA page for the plate+MJD of the visit.<BR><B>Date-Obs links:</B> apVisit file download.</P>\n')
                starHTML.write('<TABLE BORDER=2 CLASS="sortable">\n')
                starHTML.write('<TR bgcolor="'+thcolor+'">')
                starHTML.write('<TH>MJD <TH>Date-Obs <TH>Field<BR> <TH>Plate <TH>Fiber <TH>MTP <TH>Cart <TH>S/N <TH>Vrad <TH>Spectrum Plot </TR>\n')
                for k in range(nvis):
                    mjd = vcat['mjd'][k]
                    fps = True
                    if mjd < 59556: fps = False
                    cmjd = str(mjd)
                    dateobs = Time(vcat['jd'][k], format='jd').fits.replace('T', '<BR>')
                    cplate = vcat['plate'][k]
                    cfield = vcat['field'][k]
                    cfib = str(int(round(vcat['fiberid'][k]))).zfill(3)
                    cblock = str(11-np.ceil(vcat['fiberid'][k]/30).astype(int))
                    ccart = '?'
                    platefile = load.filename('PlateSum', plate=int(vcat['plate'][k]), mjd=cmjd, fps=fps)
                    if os.path.exists(platefile):
                        platetab = fits.getdata(platefile,1)
                        ccart = str(platetab['CART'][0])
                    
                    csnr = str("%.1f" % round(vcat['snr'][k],1))
                    cvrad = str("%.2f" % round(vcat['vrad'][k],2))
                    visplotname = prefix+'Plate-' + cplate + '-' + cmjd + '-' + cfib + '.png'
                    visplotpath = '../../../../../visit/' + telescope + '/' + cfield + '/' + cplate + '/' + cmjd + '/plots/'
                    visplot = visplotpath + visplotname
                    apvispath = '../../../../../visit/' + telescope + '/' + cfield + '/' + cplate + '/' + cmjd + '/'
                    apqahtml = apvispath + '/html/' + prefix + 'QA-' + cplate + '-' + cmjd + '.html'
                    apvisfile = 'apVisit-' + apred + '-' + telescope + '-' + cplate + '-' + cmjd + '-' + cfib + '.fits'
                    apvis = apvispath + apvisfile

                    starHTML.write('<TR><TD ALIGN=center><A HREF="' + apqahtml + '">' + cmjd + '</A>\n')
                    starHTML.write('<TD ALIGN=center><A HREF="' + apvis + '">' + dateobs + '</A>\n')
                    starHTML.write('<TD ALIGN=center>' + cfield + '\n')
                    starHTML.write('<TD ALIGN=center>' + cplate + '\n')
                    starHTML.write('<TD ALIGN=center>' + cfib + '\n')
                    starHTML.write('<TD ALIGN=center>' + cblock + '\n')
                    starHTML.write('<TD ALIGN=center>' + ccart + '\n')
                    fcol='black'  
                    if float(csnr) < 20: fcol='red'
                    starHTML.write('<TD ALIGN=right><FONT color=' + fcol + '>' + csnr + '</FONT>\n')
                    fcol='black'  
                    if np.absolute(float(cvrad)) > 300: fcol='red'
                    starHTML.write('<TD ALIGN=right><FONT color=' + fcol + '>' + cvrad + '</FONT>\n')
                    starHTML.write('<TD><A HREF=' + visplot + ' target="_blank"><IMG SRC=' + visplot + ' WIDTH=1000></A></TR>\n')
                starHTML.write('</TABLE>\n<BR>\n')
                starHTML.write('<HR>\n')

                # Insert Doppler output plots
                if apStarRelPath is not None:
                    starHTML.write('<H3>Doppler Cross-Correlation Plots:</H3>')
                    ccfplot = '../plots/' + apStarNewest.replace('.fits', '_ccf.png')
                    spec1plot = ccfplot.replace('ccf', 'spec')
                    spec2plot = ccfplot.replace('ccf', 'spec2')
                    starHTML.write('<A HREF=' + ccfplot + ' target="_blank"><IMG SRC=' + ccfplot + ' WIDTH=600></A>\n')
                    starHTML.write('<HR>\n')
                    starHTML.write('<H3>Doppler Visit+Model Plots:</H3>')
                    starHTML.write('<A HREF=' + spec1plot + ' target="_blank"><IMG SRC=' + spec1plot + ' WIDTH=600></A>\n')
                    starHTML.write('<A HREF=' + spec2plot + ' target="_blank"><IMG SRC=' + spec2plot + ' WIDTH=600></A>\n')
                starHTML.write('<BR><BR><BR><BR>\n')
                starHTML.close()

    if objid is None:
        print("----> makeStarHTML: Done with plate " + plate + ", MJD " + mjd + ".\n")
    else:
        print("----> makeStarHTML: Done with " + objid)


###################################################################################################
''' APVISITPLOTS: plots of the apVisit spectra '''
def apVisitPlots(load=None, plate=None, mjd=None, telescope=None):

    print("----> apVisitPlots: Running plate "+plate+", MJD "+mjd)

    prefix = 'ap'
    if telescope == 'lco25m': prefix = 'as'

    if int(mjd)>59556:
        fps = True
    else:
        fps = False

    # Set up some basic plotting parameters
    #plt.ioff()
    #matplotlib.use('agg')
    fontsize = 24;   fsz = fontsize * 0.75
    matplotlib.rcParams.update({'font.size':fontsize, 'font.family':'serif'})
    matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"
    bboxpar = dict(facecolor='white', edgecolor='none', alpha=1.0)
    axwidth=1.5
    axmajlen=7
    axminlen=3.5
    lwidth = 1.5
    hlines = np.array([16811.111,16411.692,16113.732,15884.897,15704.970,15560.718,15443.157,15345.999,15264.725,15196.016,15137.357])
    ce3lines = np.array([15851.880,15961.157,15964.928,16133.170,16292.642])
    mn2lines = np.array([15387.220,15412.667,15586.57,15600.576,15620.314])
    hcolor = 'b'
    ce3color = 'r'
    mn2color = 'b'
    visitxmin = 15120;   visitxmax = 16960;    visitxspan = visitxmax - visitxmin

    # Load in the apPlate file
    apPlate = load.apPlate(int(plate), mjd)
    data = apPlate['a'][11].data[::-1]
    objtype = data['OBJTYPE']
    nfiber = len(data)
    cnfiber = str(nfiber)

    # Make plot and html directories if they don't already exist.
    plotsdir = os.path.dirname(load.filename('Plate', plate=int(plate), mjd=mjd, chips=True, fps=fps)) + '/plots/'
    if os.path.exists(plotsdir) == False: os.makedirs(plotsdir)

    # Loop over the fibers
    for j in range(300):
        jdata = data[j]
        fiber = jdata['FIBERID']
        if fiber > 0:
            cfiber = str(fiber).zfill(3)
            objid = jdata['OBJECT']
            objtype = jdata['OBJTYPE']
            hmag = jdata['HMAG']
            chmag = str("%.3f" % round(jdata['HMAG'], 3))
            jkcolor = jdata['JMAG'] - jdata['KMAG']

            # Red color for sky fibers
            pcolor = 'k'
            if objtype == 'SKY': pcolor = 'firebrick'

            plotfile = prefix + 'Plate-' + plate + '-' + mjd + '-' + cfiber + '.png'

            # Get wavelength and flux arrays
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
            ymxsec3, = np.where((Wave > 16925) & (Wave < 16935))
            ymxsec4, = np.where((Wave > 16800) & (Wave < 16822))
            ymxsec5, = np.where((Wave > 15186) & (Wave < 15206))
            if (len(ymxsec1) == 0) | (len(ymxsec2) == 0) | (len(ymxsec3) == 0) | (len(ymxsec4) == 0) | (len(ymxsec5) == 0): 
                print("----> apVisitPlots: Problem with fiber " + str(fiber).zfill(3) + ". Not Plotting.")
            else:
                print("----> apVisitPlots:    making " + plotfile + " (" + objid + ", " + str(j+1) + "/" + cnfiber + ")")
                tmpF = convolve(Flux, Box1DKernel(11))
                ymx1 = np.nanmax(tmpF[ymxsec1])
                ymx2 = np.nanmax(tmpF[ymxsec2])
                ymx3 = np.nanmax(tmpF[ymxsec3])
                ymx4 = np.nanmax(tmpF[ymxsec4])
                ymx5 = np.nanmax(tmpF[ymxsec5])
                ymx = np.nanmax([ymx1,ymx2,ymx3,ymx4,ymx5])
                med = np.nanmedian(Flux)
                if objtype != 'SKY':
                    ymin = 0
                    yspn = ymx - ymin
                    ymax = ymx + (yspn * 0.15)
                    # Establish Ymin
                    ymn = np.nanmin(tmpF)
                    if ymn > 0: 
                        yspn = ymx - ymn
                        ymin = ymn - (yspn * 0.10)
                        ymax = ymx + (yspn * 0.15)
                else:
                    ymin = med - 50
                    ymax = med + 50
                yspan = ymax-ymin

                fig=plt.figure(figsize=(28,8))
                ax1 = plt.subplot2grid((1,1), (0,0))
                ax1.tick_params(reset=True)
                ax1.set_xlim(visitxmin, visitxmax)
                ax1.set_ylim(ymin, ymax)
                ax1.xaxis.set_major_locator(ticker.MultipleLocator(200))
                ax1.minorticks_on()
                ax1.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
                ax1.tick_params(axis='both', which='major', length=axmajlen)
                ax1.tick_params(axis='both', which='minor', length=axminlen)
                ax1.tick_params(axis='both', which='both', width=axwidth)
                ax1.set_xlabel(r'Wavelength ($\rm \AA$)')
                ax1.set_ylabel(r'Flux')

                if objtype == 'SKY':
                    ax1.plot(WaveB[np.argsort(WaveB)], FluxB[np.argsort(WaveB)], color=pcolor)
                    ax1.plot(WaveG[np.argsort(WaveG)], FluxG[np.argsort(WaveG)], color=pcolor)
                    ax1.plot(WaveR[np.argsort(WaveR)], FluxR[np.argsort(WaveR)], color=pcolor)
                else:
                    for ll in hlines: ax1.axvline(x=ll, color=hcolor, alpha=0.6)
                    ax1.text(16811.111, ymin+yspan*0.03, 'H I', color=hcolor, bbox=bboxpar, fontsize=fsz, ha='center')
                    #if jkcolor < 0.3:
                        #for ll in ce3lines: ax1.axvline(x=ll, color=ce3color, alpha=0.6)
                        #for ll in mn2lines: ax1.axvline(x=ll, color=mn2color, alpha=0.6)
                        #ax1.text(15412.667, ymin+yspan*0.03, 'Mn II', color=mn2color, bbox=bboxpar, fontsize=fsz, ha='center')
                        #ax1.text(15961.157, ymin+yspan*0.03, 'Ce III', color=ce3color, bbox=bboxpar, fontsize=fsz, ha='center')

                    if med > 400:
                        badpixels, = np.where(FluxB < 1)
                        if len(badpixels) > 0:
                            g, = np.where((badpixels > 12) & (badpixels < 2036))
                            if len(g) > 0:
                                badpixels = badpixels[g]
                                for badpixel in badpixels:
                                    badfixrange1 = [badpixel-10, badpixel-1]
                                    badfixrange2 = [badpixel+1, badpixel+10]
                                    g, = np.where((FluxB[badfixrange1[0]:badfixrange1[1]] > med/3) | (FluxB[badfixrange2[0]:badfixrange2[1]] > med/3))
                                    if len(g) > 2: FluxB[badpixel] = np.mean([FluxB[badfixrange1[0]:badfixrange1[1]], FluxB[badfixrange2[0]:badfixrange2[1]]])
                        badpixels, = np.where(FluxG < 1)
                        if len(badpixels) > 0:
                            g, = np.where((badpixels > 12) & (badpixels < 2036))
                            if len(g) > 0:
                                badpixels = badpixels[g]
                                for badpixel in badpixels:
                                    badfixrange1 = [badpixel-10, badpixel-1]
                                    badfixrange2 = [badpixel+1, badpixel+10]
                                    g, = np.where((FluxG[badfixrange1[0]:badfixrange1[1]] > med/3) | (FluxG[badfixrange2[0]:badfixrange2[1]] > med/3))
                                    if len(g) > 2: FluxG[badpixel] = np.mean([FluxG[badfixrange1[0]:badfixrange1[1]], FluxG[badfixrange2[0]:badfixrange2[1]]])
                        badpixels, = np.where(FluxR < 1)
                        if len(badpixels) > 0:
                            g, = np.where((badpixels > 12) & (badpixels < 2036))
                            if len(g) > 0:
                                badpixels = badpixels[g]
                                for badpixel in badpixels:
                                    badfixrange1 = [badpixel-10, badpixel-1]
                                    badfixrange2 = [badpixel+1, badpixel+10]
                                    g, = np.where((FluxR[badfixrange1[0]:badfixrange1[1]] > med/3) | (FluxR[badfixrange2[0]:badfixrange2[1]] > med/3))
                                    if len(g) > 2: FluxR[badpixel] = np.mean([FluxB[badfixrange1[0]:badfixrange1[1]], FluxR[badfixrange2[0]:badfixrange2[1]]])

                    ax1.plot(WaveB[np.argsort(WaveB)], FluxB[np.argsort(WaveB)], color='white', linewidth=10)
                    ax1.plot(WaveG[np.argsort(WaveG)], FluxG[np.argsort(WaveG)], color='white', linewidth=10)
                    ax1.plot(WaveR[np.argsort(WaveR)], FluxR[np.argsort(WaveR)], color='white', linewidth=10)
                    ax1.plot(WaveB[np.argsort(WaveB)], FluxB[np.argsort(WaveB)], color=pcolor)
                    ax1.plot(WaveG[np.argsort(WaveG)], FluxG[np.argsort(WaveG)], color=pcolor)
                    ax1.plot(WaveR[np.argsort(WaveR)], FluxR[np.argsort(WaveR)], color=pcolor)

                    ax1.text(0.02, 0.05, objid + ',  H = ' + chmag, transform=ax1.transAxes, bbox=bboxpar)

                fig.subplots_adjust(left=0.06, right=0.995, bottom=0.12, top=0.98, hspace=0.2, wspace=0.0)
                plt.savefig(plotsdir + plotfile)
                plt.close('all')
                
    print("----> apVisitPlots: Done with plate " + plate + ", MJD " + mjd + ".\n")

###################################################################################################
''' APSTARPLOTS: plots of the apStar spectra + best fitting Cannon model '''
def apStarPlots(objid=None, load=None, plate=None, mjd=None, apred=None, telescope=None):

    if objid is None:
        print("----> apStarPlots: Running plate "+plate+", MJD "+mjd)
    else:
        print("----> apStarPlots: Running on single star:" + objid)

    apodir = os.environ.get('APOGEE_REDUX') + '/'
    load = apload.ApLoad(apred=apred, telescope=telescope)

    # Setup doppler cannon models
    models = doppler.cannon.models

    # Base directory where star-level stuff goes
    starHTMLbase = apodir + apred + '/stars/' + telescope + '/'

    # Basic plotting parameters
    fontsize = 24;   fsz = fontsize * 0.75
    matplotlib.rcParams.update({'font.size':fontsize, 'font.family':'serif'})
    matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"
    bboxpar = dict(facecolor='white', edgecolor='none', alpha=1.0)
    axwidth=2
    axmajlen=7
    axminlen=3.5
    lwidth = 1.5
    xmin = np.array([15130, 15845, 16460])
    xmax = np.array([15825, 16448, 16968])

    nfib = 300
    if objid is None: 
        # Load in the apPlate file
        apPlate = load.apPlate(int(plate), mjd)
        data = apPlate['a'][11].data[::-1]
        objtype = data['OBJTYPE']
        nfiber = len(data)
        cnfiber = str(nfiber)
    else:
        nfib = 1

    # Loop over the fibers
    for j in range(nfib):
        if objid is None:
            jdata = data[j]
            fiber = jdata['FIBERID']
            objtype = jdata['OBJTYPE']
            objid = jdata['OBJECT']
        else:
            objtype = 'SCI'
            fiber = 100

        # Only run it for valid stars
        if (fiber > 0) & (objtype != 'SKY') & (objid != '2MNone') &  (objid != '2M') & (objid != ''):

            # Find which healpix this star is in
            healpix = apload.obj2healpix(objid)
            healpixgroup = str(healpix // 1000)
            healpix = str(healpix)

            # Find the associated healpix html directories and make them if they don't already exist
            starDir = starHTMLbase + healpixgroup + '/' + healpix + '/'
            starRelPath = '../../../../../stars/' + telescope + '/' + healpixgroup + '/' + healpix + '/'

            # Make sure an apStar file exists
            apStarCheck = glob.glob(starDir + 'apStar-' + apred + '-' + telescope + '-' + objid + '-*.fits')
            if len(apStarCheck) < 1: 
                print("----> apStarPlots:    apStar file not found for " + objid)
            else:
                if objid is None:
                    print("----> apStarPlots:    making plot for " + objid + " (" + str(j+1) + "/" + cnfiber + ")")
                else:
                    print("----> apStarPlots:    making plot for " + objid)
                # Find the newest apStar file
                apStarCheck.sort()
                apStarCheck = np.array(apStarCheck)
                apStarNewest = os.path.basename(apStarCheck[-1])
                apStarPath = starDir + apStarNewest
                hdr = fits.getheader(apStarPath)
                chmag = str("%.3f" % round(hdr['HMAG'], 3))
                apStarModelPath = apStarPath.replace('.fits', '_out_doppler.pkl')

                # Set up plot directories and plot file name
                starPlotDir = starDir + 'plots/'
                if os.path.exists(starPlotDir) == False: os.makedirs(starPlotDir)
                starPlotFile = 'apStar-' + apred + '-' + telescope + '-' + objid + '_spec+model.png'
                starPlotFilePath = starPlotDir + starPlotFile
                starPlotFileRelPath = starRelPath + 'plots/' + starPlotFile

                #if objid == '2M14432748+4006125': import pdb; pdb.set_trace()

                # Read the apStar file
                apstar = doppler.read(apStarPath)
                apstar.normalize()
                nvis = apstar.wave.shape[1] - 2
                if nvis < 1: nvis = 1
                if nvis == 1: 
                    wave = apstar.wave[:,0]
                    flux = apstar.flux
                else: 
                    wave = apstar.wave[:, 0]
                    flux = apstar.flux[:, 0]
                if np.nanmax(flux) < 0.1:
                    print('----> apStarPlots:    problem with ' + objid + ' apStar file!!! Skipping.')
                    continue
                gd, = np.where((np.isnan(flux) == False) & (flux > 0))
                wave = wave[gd]
                flux = flux[gd]
                wmin = np.min(wave); wmax = np.max(wave)
                nwave = len(wave)

                # Get model spectrum
                openModel = open(apStarModelPath, 'rb')
                modelVals = pickle.load(openModel)
                try:
                    sumstr, finalstr, bmodel, specmlist, gout = modelVals
                except:
                    print("----> apStarPlots:    BAD! pickle.load returned None for " + objid)
                    return
                pmodels = models.prepare(specmlist[0])
                bestmodel = pmodels(teff=sumstr['teff'], logg=sumstr['logg'], feh=sumstr['feh'], rv=0)
                bestmodel.normalize()
                #swave = bestmodel.wave
                #sflux = bestmodel.flux
                swave = np.concatenate([bestmodel.wave[:, 0], bestmodel.wave[:,1], bestmodel.wave[:,2]])
                sflux = np.concatenate([bestmodel.flux[:, 0], bestmodel.flux[:,1], bestmodel.flux[:,2]])
                Worder = np.argsort(swave)
                swave = swave[Worder]
                sflux = sflux[Worder]
                #f = interpolate.interp1d(swave, sflux, fill_value="extrapolate")
                #swaveg = np.linspace(wmin, wmax, nwave)
                #sfluxg = f(swaveg)
                #resid = sfluxg - flux

                rvteff = str(int(round(sumstr['teff'][0])))
                rvlogg = str("%.3f" % round(sumstr['logg'][0],3))
                rvfeh = str("%.3f" % round(sumstr['feh'][0],3))

                fig=plt.figure(figsize=(28,25))
                ax1 = plt.subplot2grid((23,1), (0,0), rowspan=5)
                ax11 = plt.subplot2grid((23,1), (5,0), rowspan=2)
                ax2 = plt.subplot2grid((23,1), (8,0), rowspan=5)
                ax22 = plt.subplot2grid((23,1), (13,0), rowspan=2)
                ax3 = plt.subplot2grid((23,1), (16,0), rowspan=5)
                ax33 = plt.subplot2grid((23,1), (21,0), rowspan=2)
                axes = [ax1, ax11, ax2, ax22, ax3, ax33]

                ax33.set_xlabel(r'Rest Wavelength ($\rm \AA$)')

                ii = 0
                ichip = 0
                for ax in axes:
                    ax.set_xlim(xmin[ichip], xmax[ichip])
                    if ii % 2 == 0: ax.set_ylim(0.1, 1.3)
                    if ii % 2 == 1: ax.set_ylim(-0.3, 0.3)
                    if ii % 2 == 1: ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
                    ax.tick_params(reset=True)
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
                    ax.minorticks_on()
                    ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
                    ax.tick_params(axis='both',which='major',length=axmajlen)
                    ax.tick_params(axis='both',which='minor',length=axminlen)
                    ax.tick_params(axis='both',which='both',width=axwidth)
                    for axis in ['top','bottom','left','right']: ax.spines[axis].set_linewidth(axwidth)
                    if ii % 2 == 0: ax.text(-0.04, 0.50, r'$F_{\lambda}$ / $F_{\rm cont.}$', transform=ax.transAxes, rotation=90, ha='right', va='center')
                    if ii % 2 == 1: ax.text(-0.04, 0.50, r'Resid.', transform=ax.transAxes, rotation=90, ha='right', va='center')
                    if ii % 2 == 1: ax.axhline(y=0, linestyle='dashed', linewidth=lwidth, color='k')
                    if ii % 2 == 0: ax.axes.xaxis.set_ticklabels([])

                    g, = np.where((wave >= xmin[ichip] - 20) & (wave <= xmax[ichip] + 20))
                    wmin = np.min(wave[g]); wmax = np.max(wave[g]); nwave = len(g)
                    gg, = np.where((swave >= wmin) & (swave <= wmax))
                    f = interpolate.interp1d(swave[gg], sflux[gg], fill_value="extrapolate")
                    swaveg = np.linspace(wmin, wmax, nwave)
                    sfluxg = f(wave[g])
                    
                    if ii % 2 == 0: 
                        ax.plot(wave[g], flux[g], color='k', label='apStar')
                        ax.plot(wave[g], sfluxg, color='r', label='Cannon model', alpha=0.75)
                    else:
                        resid = sfluxg - flux[g]
                        ax.plot(wave[g], resid, color='b', alpha=0.75)

                    if ii % 2 == 1: ichip += 1
                    ii += 1

                txt1 = objid + r'          $H$ = ' + chmag + '          ' + str(nvis) + ' visits          '
                txt2 = r'$T_{\rm eff}$ = ' + rvteff + ' K          log(g) = ' + rvlogg + '          [Fe/H] = '+rvfeh
                ax1.text(0.5, 1.05, txt1 + txt2, transform=ax1.transAxes, ha='center', fontsize=fontsize*1.25, color='k')#, bbox=bboxpar)
                #ax2.legend(loc='upper left', edgecolor='k', ncol=2, fontsize=fontsize*1.25, framealpha=0.8)

                fig.subplots_adjust(left=0.06,right=0.99,bottom=0.04,top=0.96,hspace=0.01,wspace=0.0)
                plt.savefig(starPlotFilePath)
                plt.close('all')

    if objid is None:
        print("----> apStarPlots: Done with plate " + plate + ", MJD " + mjd + ".\n")
    else:
        print("----> apStarPlots: Done with " + objid)

###################################################################################################
'''  MAKENIGHTQA: makes nightly QA pages '''
def makeNightQA(load=None, mjd=None, telescope=None, apred=None): 

    print("----> makeNightQA: Running MJD " + mjd)

    # HTML header background color
    thcolor = '#DCDCDC'

    if int(mjd)>59556:
        fps = True
    else:
        fps = False

    # Set up some basic plotting parameters, starting by turning off interactive plotting.
    #plt.ioff()
    matplotlib.use('agg')
    fontsize = 24;   fsz = fontsize * 0.75
    matplotlib.rcParams.update({'font.size':fontsize, 'font.family':'serif'})
    matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"
    alpha = 0.6
    axwidth=1.5
    axmajlen=7
    axminlen=3.5
    cmap = 'RdBu'

    chips = np.array(['a','b','c'])
    nchips = len(chips)

    # Establish instrument and directories
    instrument = 'apogee-n'
    if telescope == 'lco25m': instrument = 'apogee-s'
    datadir = {'apo25m':os.environ['APOGEE_DATA_N'],'apo1m':os.environ['APOGEE_DATA_N'],
               'lco25m':os.environ['APOGEE_DATA_S']}[telescope] + '/'

    apodir =     os.environ.get('APOGEE_REDUX') + '/'
    spectrodir = apodir + apred + '/'
    platedir =   spectrodir+'/visit/'+telescope+'/*/*/'+mjd+'/'
    caldir =     spectrodir + 'cal/'
    expdir =     spectrodir + 'exposures/' + instrument + '/'
    reddir =     expdir + mjd + '/'
    outdir =     expdir + mjd + '/html/'
    htmlfile =   outdir + mjd + '.html'
    plotsdir =   expdir + mjd + '/plots/'
    print("----> makeNightQA: "+htmlfile)

    # Make the html folder if it doesn't already exist
    if os.path.exists(outdir) == False: subprocess.call(['mkdir',outdir])
    if os.path.exists(plotsdir) == False: subprocess.call(['mkdir',plotsdir])

    # Get all apR file numbers for the night
    rawfiles = glob.glob(datadir + mjd + '/a*R-*.apz')
    rawfiles.sort()
    rawfiles = np.array(rawfiles)
    nrawfiles = len(rawfiles)
    if nrawfiles < 1: sys.exit("----> makeNightQA: PROBLEM! No raw data found.")

#    checksums = np.zeros(nrawfiles)
    exposures = np.zeros(nrawfiles)
    for i in range(nrawfiles):
        exposures[i] = int(rawfiles[i].split('-')[2].split('.')[0])
#        tfile = os.path.basename(rawfiles[i])
#        if not file_test(reddir+file+'.check'):
#            checksum=apg_checksum(files[i],fitsdir=getlocaldir())
#            openw,clun,/get_lun,reddir+file+'.check'
#            printf,clun,checksum
#            free_lun,clun
#        readcol,reddir+file+'.check',format='(i)',check
#        checksums[i]=check
#        comp=strsplit(file,'-',/extract)
#        name=strsplit(comp[2],'.',/extract)
#        chip=comp[1]
#        num=0L
#        reads,name[0],num
#        nums[i]=num

    firstExposure = int(round(np.min(exposures)))
    lastExposure = int(round(np.max(exposures)))
    sortExposures = np.argsort(exposures)
    uExposures = np.unique(exposures)
    nuExposures = len(uExposures)

    html = open(htmlfile, 'w')
    html.write('<HTML><BODY>')
    html.write('<HEAD><script type=text/javascript src=html/sorttable.js></script><TITLE>Nightly QA for MJD '+mjd+'</TITLE></HEAD>\n')
    html.write('<H1>Nightly QA for MJD '+mjd+'</H1>\n')

    # Find the observing log file
    reportsDir = os.environ['SAS_ROOT']+'/data/staging/' + telescope[0:3] + '/reports/'
    if os.path.exists(reportsDir):
        dateobs = Time(int(mjd)-1, format='mjd').fits.split('T')[0]
        if telescope == 'apo25m': reports = glob.glob(reportsDir + dateobs + '*.log')
        if telescope == 'lco25m': reports = glob.glob(reportsDir + dateobs + '*.log.html')
        if len(reports) > 0:
            reports.sort()
            reportfile = reports[0]
            reportLink = 'https://data.sdss.org/sas/sdss5/data/staging/' + telescope[0:3] + '/reports/' + os.path.basename(reportfile)
            #https://data.sdss.org/sas/sdss5/data/staging/apo/reports/2020-10-16.12%3A04%3A20.log
            html.write(' <a href="'+reportLink+'"> <H3>' + telescope.upper().replace('25M',' 2.5m') + ' Observing report </H3></a>\n')
        else:
            print('----> makeNightQA: No observing report found for ' + mjd + '!!!')
            html.write(telescope.upper().replace('25M',' 2.5m') + ' Observing report (missing?)</H3>\n')
    else:
        print('----> makeNightQA: No observing report found for ' + mjd + '!!!')
        html.write(telescope.upper().replace('25M',' 2.5m') + ' Observing report (missing?)</H3>\n')

    # Look for missing raw frames (assuming contiguous sequence)
    html.write('<H2>Raw frames:</H2> ' + str(firstExposure) + ' to ' + str(lastExposure))
#    html.write(' (<a href=../../../../../../'+os.path.basename(dirs.datadir)+'/'+cmjd+'/'+cmjd+'.log.html> image log</a>)\n')
    logFile = 'https://data.sdss.org/sas/sdss5/data/apogee/' + telescope[0:3] + '/' + mjd + '/' + mjd + '.log.html'
    logFileDir = os.path.dirname(logFile)
    html.write(' (<A HREF="'+logFile+'">image log</A>)\n')
    html.write('<BR>\n')

    html.write('<H2>Missing raw data:</H2>\n')
    nmiss = 0
    for i in range(nchips):
        html.write('<FONT color=red>\n')
        for j in range(nuExposures):
            checkfile = datadir + mjd + '/apR-' + chips[i] + '-' + str(int(round(uExposures[j]))) + '.apz'
            if os.path.exists(checkfile) == False:
                if (i != nchips) & (uExposures[j] != lastExposure):
                    html.write('apR-' + chips[i] + '-' + str(int(round(uExposures[j]))) + '.apz, ')
                else:
                    html.write('apR-' + chips[i] + '-' + str(int(round(uExposures[j]))) + '.apz')
                nmiss += 1
        html.write('</font>\n')
    if nmiss == 0: html.write('<font color=green> NONE</font>\n')
    html.write('<BR>\n')

#    if not keyword_set(nocheck):
#        print,'looking for bad checksums...'
#        bad=where(checksums ne 1, nbad)
#        html.write('<h3> Bad CHECKSUMS:</h3>\n')
#        if nbad gt 0:
#            html.write('<font color=red>\n')
#            for i=0,len(bad)-1 do html.write(file_basename(files[bad[i]])
#            html.write('</font><BR>\n')
#        else:
#            html.write('<font color=green> NONE </font>\n')

    # look for missing reduced frames
#    print,'looking for missing reduced data...'
    html.write('<H2>Missing reduced data:</H2><TABLE BORDER=2>\n')
    html.write('<TR bgcolor='+thcolor+'><TH>ID<TH>NFRAMES/NREAD<TH>TYPE<TH>PLATEID<TH>CARTID<TH>1D missing<TH>2D missing\n')
    for i in range(nuExposures):
        n = int(round(uExposures[i]))
        file1d = load.filename('1D', mjd=mjd, num=n, chips='c').replace('1D-', '1D-c-')
        if os.path.exists(file1d) == False:
            file2d = load.filename('2D', mjd=mjd, num=n, chips='c').replace('2D-', '2D-c-')
            if (os.path.exists(file2d) == False) & (os.path.exists(file2d + '.fz') == False):
                miss2d = 1
            else:
                miss2d = 0

            imtype = 'unknown'
            color = 'white'

            rawfile = load.filename('R', num=n, mjd=mjd, chips='a').replace('apR-', 'apR-c-')
            if os.path.exists(rawfile):
                rawdata = load.apR(n)
                head = rawdata['a'][1].header
                imtype = head['IMAGETYP']
                if imtype == 'Object': color = 'red'
                if imtype == 'unknown': color = 'magenta'
                if (imtype == 'Dark') & (miss2d == 1): color = 'yellow'
                if (imtype != 'Dark') | (miss2d == 1):
                    html.write('<TR bgcolor='+color+'><TD> '+str(int(round(n)))+'\n')
                    html.write('<TD><CENTER>'+str(head['NFRAMES'])+'/'+str(head['NREAD'])+'</CENTER>\n')
                    html.write('<TD><CENTER>'+head['IMAGETYP']+'</CENTER>\n')
                    html.write('<TD><CENTER>'+str(head['PLATEID'])+'</CENTER>\n')
                    html.write('<TD><CENTER>'+str(head['CARTID'])+'</CENTER>\n')
                    html.write('<TD> '+os.path.basename(file1d)+'\n')
                    if (os.path.exists(file2d) == False) & (os.path.exists(file2d+'.fz') == False):
                        html.write('<TD> '+os.path.basename(file2d)+'\n')
            else:
                html.write('<TR bgcolor='+color+'><TD> '+str(int(round(n)))+'\n')
                html.write('<TD><CENTER> </CENTER>\n')
                html.write('<TD><CENTER> </CENTER>\n')
                html.write('<TD><CENTER> </CENTER>\n')
                html.write('<TD><CENTER> </CENTER>\n')
                html.write('<TD> '+os.path.basename(file1d)+'\n')
                if (os.path.exists(file2d) == False) & (os.path.exists(file2d+'.fz') == False):
                    html.write('<TD> '+os.path.basename(file2d)+'\n')
    html.write('</TABLE>\n')
    html.write('<BR>\n')

    # Get all observed plates (from planfiles)
    # print,'getting observed plates ....'
    planfiles = glob.glob(platedir + '*Plan*.yaml')
    nplanfiles = len(planfiles)
    if nplanfiles >= 1:
        planfiles = np.array(planfiles)
        html.write('<H2>Observed plates:</H2><TABLE BORDER=2>\n')
        html.write('<TR bgcolor='+thcolor+'><TH>Planfile <TH>Nframes <TH>Median<BR>Zeropoint <TH>Median RMS<BR>Zeropoint <TH>Cart <TH>Unmapped <TH>Missing\n')
        for i in range(nplanfiles):
            planfilebase = os.path.basename(planfiles[i])
            planfilebase_noext = planfilebase.split('.')[0]
            # Planfile name
            html.write('<TR><TD>' + planfilebase_noext + '\n')
            planstr = plan.load(planfiles[i], np=True)
            plate = str(int(round(planstr['plateid'])))
            mjd = str(int(round(planstr['mjd'])))
            platefile = load.filename('PlateSum', plate=int(plate), mjd=mjd, fps=fps)
            platefilebase = os.path.basename(platefile)
            platefiledir = os.path.dirname(planfiles[i])
            if (planstr['platetype'] == 'normal') & (os.path.exists(platefile)): 
                platetab = fits.getdata(platefile,1)
                platefiber = fits.getdata(platefile,2)
                # Nframes
                html.write('<TD align="right">' + str(len(platetab)) + '\n')
                # Zero and zerorms
                if len(platetab['ZERO']) > 1:
                    html.write('<TD align="right">' + str("%.2f" % np.round(np.nanmedian(platetab['ZERO']),2)) + '\n')
                    html.write('<TD align="right">' + str("%.2f" % np.round(np.nanmedian(platetab['ZERORMS']),2)) + '\n')
                else:
                    html.write('<TD align="right">' + str("%.2f" % np.round(platetab['ZERO'],2)) + '\n')
                    html.write('<TD align="right">' + str("%.2f" % np.round(platetab['ZERORMS'],2)) + '\n')
                # Cart
                html.write('<TD align="center">' + str(platetab['CART'][0]) + '\n')
                unplugged, = np.where(platefiber['FIBERID'] < 0)
                html.write('<TD align="center">')
                if len(unplugged) >= 0: html.write(str(300 - unplugged) + '\n')
                html.write('<TD align="center">')
                expfile = load.filename('1D', num=planstr['fluxid'], chips='b')
                if os.path.exists(expfile):
                    domeflat = fits.getdata(expfile)
                    level = np.nanmedian(domeflat, axis=1)
                    bad, = np.where(level == 0)
                    if len(bad) >= 0: html.write(str(300 - bad) + '\n')
    html.write('</TABLE>\n')

#    print,'wavehtml...'
#    wavefile = caldir + 'wave/html/wave' + mjd + '.html'
#    if os.path.exists(wavefile):
#        spawn,'cat '+file,wavehtml
#        for i=1,n_elements(wavehtml)-2 do html.write(wavehtml[i]

    # Get all succesfully reduced plates
    #print,'getting successfully reduced plates...'
    platefiles = glob.glob(platedir + '*PlateSum*.fits')
    # Make master plot of zeropoint and sky levels for the night
    if (len(platefiles) >= 1): 
        platefiles.sort()
        platefiles = np.array(platefiles)
        nplates = len(platefiles)
        for i in range(nplates):
            platefiledir = os.path.dirname(platefiles[i])
            platetab = fits.getdata(platefiles[i],1)
            mjd = str(int(round(platetab['MJD'][0])))
            #sntab, tabs=platefiles[i], outfile=platefiledir + '/sn-' + plate + '-' + mjd + '.dat'
            #sntab, tabs=platefiles[i], outfile=platefiledir + '/altsn-' + plate + '-' + mjd + '.dat', /altsn
            if i == 0:
                zero = platetab['ZERO']
                ims = platetab['IM']
                moondist = platetab['MOONDIST']
                skyr = platetab['SKY'][:,0]
                skyg = platetab['SKY'][:,1]
                skyb = platetab['SKY'][:,2]
            else:
                zero = np.concatenate([zero, platetab['ZERO']])
                ims = np.concatenate([ims, platetab['IM']])
                skyr = np.concatenate([skyr, platetab['SKY'][:,0]])
                skyg = np.concatenate([skyg, platetab['SKY'][:,1]])
                skyb = np.concatenate([skyb, platetab['SKY'][:,2]])
                moondist = np.concatenate([moondist, platetab['MOONDIST']])

        html.write('<H2>Zeropoints and sky levels: </H2>\n')
        html.write('<TABLE BORDER=2><TR bgcolor='+thcolor+'><TH>Zeropoints <TH>Sky level <TH>Sky level vs moon distance\n')

        # Plot of zeropoint versus image number
        plotfile = mjd + 'zero.png'
        print("----> makeNightQA: Making "+plotfile)

        fig=plt.figure(figsize=(14,8))
        ax1 = plt.subplot2grid((1,1), (0,0))
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax1.minorticks_on()
        ax1.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
        ax1.tick_params(axis='both',which='major',length=axmajlen)
        ax1.tick_params(axis='both',which='minor',length=axminlen)
        ax1.tick_params(axis='both',which='both',width=axwidth)
        ax1.set_xlabel(r'Image Number');  ax1.set_ylabel(r'Zeropoint Per Pixel')

        xmin = np.min(ims % 10000)-1
        xmax = np.max(ims % 10000)+1
        ax1.set_xlim(xmin, xmax)
        gd, = np.where(zero > 0)
        if len(gd)>0:
            ymin = np.min(zero[gd])
        else:
            ymin = 15
        ymax = np.max(zero)
        if ymin > 15 or np.isfinite(ymin)==False: ymin = 15
        if ymax < 20 or np.isfinite(ymax)==False: ymax = 20
        ax1.set_ylim(ymin, ymax)

        ax1.scatter(ims % 10000, zero, marker='o', s=150, c='dodgerblue', edgecolors='k', alpha=0.8)

        fig.subplots_adjust(left=0.08,right=0.99,bottom=0.10,top=0.98,hspace=0.2,wspace=0.0)
        plt.savefig(plotsdir + plotfile)
        plt.close('all')

        html.write('<TR><TD><A HREF=../plots/' + mjd + 'zero.png target="_blank"><IMG SRC=../plots/' + mjd + 'zero.png WIDTH=500></A>\n')

        # Plot of zeropoint versus image number
        plotfile = mjd + 'sky.png'
        print("----> makeNightQA: Making "+plotfile)

        fig=plt.figure(figsize=(14,8))
        ax1 = plt.subplot2grid((1,1), (0,0))
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax1.minorticks_on()
        ax1.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
        ax1.tick_params(axis='both',which='major',length=axmajlen)
        ax1.tick_params(axis='both',which='minor',length=axminlen)
        ax1.tick_params(axis='both',which='both',width=axwidth)
        ax1.set_xlabel(r'Image Number');  ax1.set_ylabel(r'Sky Continuum Per Pixel')

        ax1.set_xlim(xmin, xmax)
        if ymin > 11: ymin = 11
        if ymax < 16: ymax = 16
        ax1.set_ylim(ymin, ymax)

        ax1.scatter(ims % 10000, skyr, marker='o', s=150, c='r', edgecolors='k', alpha=0.8)
        ax1.scatter(ims % 10000, skyg, marker='o', s=150, c='g', edgecolors='k', alpha=0.8)
        ax1.scatter(ims % 10000, skyb, marker='o', s=150, c='b', edgecolors='k', alpha=0.8)

        fig.subplots_adjust(left=0.08,right=0.99,bottom=0.10,top=0.98,hspace=0.2,wspace=0.0)
        plt.savefig(plotsdir + plotfile)
        plt.close('all')

        html.write('<TD><A HREF=../plots/' + plotfile + ' target="_blank"><IMG SRC=../plots/' + plotfile + ' WIDTH=500></A>\n')

        # Plot of moon distance versus sky continuum
        plotfile = mjd + 'moonsky.png'
        print("----> makeNightQA: Making "+plotfile)

        fig=plt.figure(figsize=(14,8))
        ax1 = plt.subplot2grid((1,1), (0,0))
        ax1.xaxis.set_major_locator(ticker.MultipleLocator(10))
        ax1.minorticks_on()
        ax1.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
        ax1.tick_params(axis='both',which='major',length=axmajlen)
        ax1.tick_params(axis='both',which='minor',length=axminlen)
        ax1.tick_params(axis='both',which='both',width=axwidth)
        ax1.set_xlabel(r'Moon Distance');  ax1.set_ylabel(r'Sky Continuum Per Pixel')

        if ymin > 11: ymin = 11
        if ymax < 16: ymax = 16
        ax1.set_ylim(ymin, ymax)

        ax1.scatter(moondist, skyr, marker='o', s=150, c='r', edgecolors='k', alpha=0.8)
        ax1.scatter(moondist, skyg, marker='o', s=150, c='g', edgecolors='k', alpha=0.8)
        ax1.scatter(moondist, skyb, marker='o', s=150, c='b', edgecolors='k', alpha=0.8)

        fig.subplots_adjust(left=0.08,right=0.99,bottom=0.10,top=0.98,hspace=0.2,wspace=0.0)
        plt.savefig(plotsdir + plotfile)
        plt.close('all')

        html.write('<TD><A HREF=../plots/' + mjd + 'moonsky.png target="_blank"><IMG SRC=../plots/' + plotfile+' WIDTH=500></A>\n')
        html.write('</TABLE>\n')

        html.write('<BR>Moon phase: ' + str("%.3f" % round(platetab['MOONPHASE'][0],3)) + '<BR>\n')

        html.write('<p><H2>Observed Plate Exposure Data:</H2>\n')
        html.write('<p>Note: Sky continuum, S/N, and S/N(c) columns give values for blue, green, and red detectors</p>\n')
        html.write('<TABLE BORDER=2>\n')
        th0 = '<TR bgcolor='+thcolor+'>'
        th1 = '<TH>Plate <TH>Frame <TH>Cart <TH>sec(z) <TH>HA <TH>Design HA <TH>SEEING <TH>FWHM <TH>GDRMS <TH>Nreads '
        th2 = '<TH>Dither <TH>Zero <TH>Zerorms <TH>Zeronorm <TH>Sky Continuum <TH>S/N <TH>S/N(c) <TH>Unplugged <TH>Faint\n'
        for i in range(nplates):
            platetab = fits.getdata(platefiles[i],1)
            plate = str(int(round(platetab['PLATE'][0])))
            try:
                cart = str(int(round(platetab['CART'][0])))
            except:
                cart = platetab['CART'][0]
            nreads = str(int(round(platetab['NREADS'][0])))
            n_exposures = len(platetab['IM'])
            html.write(th0 + th1 + th2)
            for j in range(n_exposures):
                html.write('<TR>\n')
                html.write('<TD align="left">' + plate + '\n')
                html.write('<TD align="left">' + str(int(round(platetab['IM'][j]))) + '\n')
                html.write('<TD align="center">' + cart + '\n')
                html.write('<TD align="right">' + str("%.3f" % round(platetab['SECZ'][j],3)) + '\n')
                html.write('<TD align="right">' + str(platetab['HA'][j]) + '\n')
                tmp1 = str("%.1f" % round(platetab['DESIGN_HA'][j][0],1)).rjust(5)
                tmp2 = str("%.1f" % round(platetab['DESIGN_HA'][j][1],1)).rjust(5)
                tmp3 = str("%.1f" % round(platetab['DESIGN_HA'][j][2],1)).rjust(5)
                html.write('<TD align="right">' + tmp1 + ', ' + tmp2 + ', ' + tmp3 + '\n')
                html.write('<TD align="right">' + str("%.3f" % round(platetab['SEEING'][j],3)) + '\n')
                html.write('<TD align="right">' + str("%.3f" % round(platetab['FWHM'][j],3)) + '\n')
                html.write('<TD align="right">' + str("%.3f" % round(platetab['GDRMS'][j],3)) + '\n')
                html.write('<TD align="right">' + nreads + '\n')
                html.write('<TD align="right">' + str("%.3f" % round(platetab['DITHER'][j],3)) + '\n')
                html.write('<TD align="right">' + str("%.3f" % round(platetab['ZERO'][j],3)) + '\n')
                html.write('<TD align="right">' + str("%.3f" % round(platetab['ZERORMS'][j],3)) + '\n')
                html.write('<TD align="right">' + str("%.3f" % round(platetab['ZERONORM'][j],3)) + '\n')
                tmp1 = str("%.3f" % round(platetab['SKY'][j][2],3)).rjust(6)
                tmp2 = str("%.3f" % round(platetab['SKY'][j][1],3)).rjust(6)
                tmp3 = str("%.3f" % round(platetab['SKY'][j][0],3)).rjust(6)
                html.write('<TD align="center">' + tmp1 + ', ' + tmp2 + ', ' + tmp3 + '\n')
                tmp1 = str("%.3f" % round(platetab['SN'][j][2],3)).rjust(6)
                tmp2 = str("%.3f" % round(platetab['SN'][j][1],3)).rjust(6)
                tmp3 = str("%.3f" % round(platetab['SN'][j][0],3)).rjust(6)
                html.write('<TD align="center">' + tmp1 + ', ' + tmp2 + ', ' + tmp3 + '\n')
                tmp1 = str("%.3f" % round(platetab['SNC'][j][2],3)).rjust(6)
                tmp2 = str("%.3f" % round(platetab['SNC'][j][1],3)).rjust(6)
                tmp3 = str("%.3f" % round(platetab['SNC'][j][0],3)).rjust(6)
                html.write('<TD align="center">' + tmp1 + ', ' + tmp2 + ', ' + tmp3 + '\n')
                #tmp = fiber['hmag'][fiberstar] + (2.5 * np.log10(obs[fiberstar,1]))
                #zero = np.nanmedian(tmp)
                #zerorms = dln.mad(fiber['hmag'][fiberstar] + (2.5 * np.log10(obs[fiberstar,1])))
                #faint, = np.where((tmp - zero) < -0.5)
                #nfaint = len(faint)
                html.write('<TD> \n')
                html.write('<TD> \n')

    html.write('</TABLE>\n')
    html.write('<BR><BR>\n')
    html.close()
    #plt.ion()
    print("----> makeNightQA: Done with MJD " + mjd + "\n")

###################################################################################################
'''  MAKEMASTERQAPAGES: makes mjd.html and fields.html '''
def makeMasterQApages(mjdmin=None, mjdmax=None, apred=None, mjdfilebase=None, fieldfilebase=None,
                      domjd=True, dofields=True):

    # Establish data directories.
    datadirN = os.environ['APOGEE_DATA_N']
    datadirS = os.environ['APOGEE_DATA_S']
    apodir =   os.environ.get('APOGEE_REDUX')+'/'
    qadir = apodir+apred+'/qa/'

    # Summary file links
    visSumPathN = '../summary/allVisit-daily-apo25m.fits'
    starSumPathN = '../summary/allStar-daily-apo25m.fits'
    visSumPathS = '../summary/allVisit-daily-lco25m.fits'
    starSumPathS = '../summary/allStar-daily-lco25m.fits'

    if domjd is True:
        # Find all .log.html files, get all MJDs with data
        print("----> makeMasterQApages: Finding log files. Please wait.")

        #logsN = np.array(glob.glob(datadirN+'/*/*.log.html'))
        mdirsN = glob.glob(datadirN+'/*')
        logfilesN = [d+'/'+os.path.basename(d)+'.log.html' for d in mdirsN]
        logsN = np.array([f for f in logfilesN if os.path.exists(f)])
        nlogsN = len(logsN)
        hemN = np.full(nlogsN, 'N').astype(str)
        print("----> makeMasterQApages: Found "+str(nlogsN)+" APOGEE-N log files.")
        mjdN = np.empty(nlogsN)
        for i in range(nlogsN): mjdN[i] = int(os.path.basename(logsN[i]).split('.')[0])


        #logsS = np.array(glob.glob(datadirS+'/*/*.log.html'))
        mdirsS = glob.glob(datadirS+'/*')
        logfilesS = [d+'/'+os.path.basename(d)+'.log.html' for d in mdirsS]
        logsS = np.array([f for f in logfilesS if os.path.exists(f)])
        nlogsS = len(logsS)
        hemS = np.full(nlogsS, 'S').astype(str)
        mjdS = np.empty(nlogsS)
        for i in range(nlogsS): mjdS[i] = int(os.path.basename(logsS[i]).split('.')[0])
        g, = np.where(mjdS > 59808)
        mdirsS = np.array(mdirsS)[g]
        logfilesS = np.array(logfilesS)[g]
        logsS = logsS[g]
        hemS = hemS[g]
        mjdS = mjdS[g]
        nlogsS = len(logsS)
        print("----> makeMasterQApages: Found "+str(nlogsS)+" APOGEE-S log files.")

        logs = np.concatenate([logsN,logsS]) 
        hem = np.concatenate([hemN,hemS]) 
        mjd = np.concatenate([mjdN,mjdS])
        nlogs = len(logs)
        print("----> makeMasterQApages: Found "+str(nlogs)+" total log files.")

        # Reverse sort the logs and MJDs so that newest MJD will be at the top
        order = np.argsort(mjd)
        logs = logs[order[::-1]]
        hem = hem[order[::-1]]
        mjd = mjd[order[::-1]]


        # Limit to MJDs within mjdmin-mjdmax range
        gd = np.where((mjd >= mjdmin) & (mjd <= mjdmax))
        logs = logs[gd]
        hem = hem[gd]
        mjd = mjd[gd]
        nmjd = len(mjd)

        # Open the mjd file html
        mjdfile = qadir + mjdfilebase
        print("----> makeMasterQApages: Creating "+mjdfilebase)

        now = datetime.datetime.now()
        today = datetime.date.today()
        current_time = now.strftime("%H:%M:%S")
        current_date = today.strftime("%B %d, %Y")

        html = open(mjdfile,'w')
        html.write('<HTML><BODY>\n')
        html.write('<HEAD><script src="sorttable.js"></script><title>APOGEE MJD Summary</title></head>\n')
        html.write('<H1>APOGEE Observation Summary by MJD</H1>\n')
        html.write('<P><I>last updated ' + current_date + ', ' + current_time + '</I></P>')
        html.write('<HR>\n')
        html.write('<p><A HREF=fields.html>Fields view</A></p>\n')
        html.write('<p><A HREF=../monitor/apogee-n-monitor.html>APOGEE-N Instrument Monitor</A></p>\n')
        html.write('<p><A HREF=../monitor/apogee-s-monitor.html>APOGEE-S Instrument Monitor</A></p>\n')
        html.write('<p> <b>Summary files:</b> <a href="'+visSumPathN+'">allVisit</a>,  <a href="'+starSumPathN+'">allStar</a></p>\n')
        #html.write('<BR>LCO 2.5m Summary Files: <a href="'+visSumPathS+'">allVisit</a>,  <a href="'+starSumPathS+'">allStar</a></p>\n')
        html.write( '<P>Yellow: APO 2.5m, Green: LCO 2.5m <BR>\n')
        html.write( 'Note: numbers in brackets in the "Plots of Spectra" column give the numbers of assigned skies, tellurics, and science targets.</P>\n')
        #html.write('<br>Click on column headings to sort\n')

        # Create web page with entry for each MJD
        html.write('<TABLE BORDER=2 CLASS=sortable>\n')
        html.write('<TR bgcolor="#eaeded"><TH>(1)<BR>Date <TH>(2)<BR>Observer Log <TH>(3)<BR>Exposure Log <TH>(4)<BR>Raw Data <TH>(5)<BR>Night QA')
        html.write('<TH>(6)<BR>Visit QA <TH>(7)<BR>Spectra Plots <TH>(8)Nsky, Ntel, Nsci <TH>(9)<BR>Summary Files <TH>(10)<BR>Moon<BR>Phase\n')
        for i in range(nmjd):
            fps = False
            if mjd[i] > 59556: fps = True

            cmjd = str(int(round(mjd[i])))
            tt = Time(mjd[i], format='mjd')
            date = tt.fits[0:10]
            # Establish telescope and instrument and setup apLoad depending on telescope.
            telescope = 'apo25m'
            instrument = 'apogee-n'
            prefix = 'ap'
            datadir = datadirN
            datadir1 = 'data'
            color = 'FFFFF8A'
            if hem[i] == 'S': 
                telescope = 'lco25m'
                instrument = 'apogee-s'
                prefix = 'as'
                datadir = datadirS
                datadir1 = 'data2s'
                color = 'b3ffb3'
            load = apload.ApLoad(apred=apred, telescope=telescope)

            reportsDir = os.environ['SAS_ROOT']+'/data/staging/' + telescope[0:3] + '/reports/'
            dateobs = Time(int(cmjd) - 1, format='mjd').fits.split('T')[0]
            if telescope == 'apo25m': reports = glob.glob(reportsDir + dateobs + '*.log')
            if telescope == 'lco25m': reports = glob.glob(reportsDir + dateobs + '*.log.html')

            # Column 1: Date
            html.write('<TR bgcolor=' + color + ' align="center"><TD>' + date + '\n')

            # Column 2: Observing log
            if len(reports) != 0:
                reports.sort()
                reportfile = reports[0]
                reportLink = 'https://data.sdss.org/sas/sdss5/data/staging/' + telescope[0:3] + '/reports/' + os.path.basename(reportfile)
                html.write('<TD align="center"><A HREF="' + reportLink + '">' + cmjd + ' obs</A>\n')
                #https://data.sdss.org/sas/sdss5/data/staging/apo/reports/2020-10-16.12%3A04%3A20.log
            else:
                html.write('<TD align="center">' + cmjd + ' obs\n')

            # Column 3-4: Exposure log and raw data link
            logFileDir = '../../' + os.path.basename(datadir) + '/' + cmjd + '/'
            logFilePath = logFileDir + cmjd + '.log.html'

            logFile = 'https://data.sdss.org/sas/sdss5/data/apogee/' + telescope[0:3] + '/' + cmjd + '/' + cmjd + '.log.html'
            logFileDir = os.path.dirname(logFile)

            html.write('<TD align="center"><A HREF="' + logFile + '">' + cmjd + ' exp</A>\n')
            html.write('<TD align="center"><A HREF="' + logFileDir + '">' + cmjd + ' raw</A>\n')

            # Column 5: Night QA
            platePlanPaths = apodir+apred+'/visit/'+telescope+'/*/*/'+cmjd+'/'+prefix+'Plan-*'+cmjd+'.yaml'
            platePlanFiles = np.array(glob.glob(platePlanPaths))
            nplatesall = len(platePlanFiles)

            plateQApaths = apodir+apred+'/visit/'+telescope+'/*/*/'+cmjd+'/html/'+prefix+'QA-*'+cmjd+'.html'
            plateQAfiles = np.array(glob.glob(plateQApaths))
            nplates = len(plateQAfiles)
            if nplates >= 1:
                html.write('<TD align="center"><A HREF="../exposures/'+instrument+'/'+cmjd+'/html/'+cmjd+'.html">'+cmjd+' QA</a>\n')
            else:
                html.write('<TD>\n')

            # Column 6: Visit QA
            html.write('<TD align="left">')
            for j in range(nplatesall):
                field = platePlanFiles[j].split(telescope+'/')[1].split('/')[0]
                plate = platePlanFiles[j].split(telescope+'/')[1].split('/')[1]
                # Check for failed plates
                plateQAfile = apodir+apred+'/visit/'+telescope+'/'+field+'/'+plate+'/'+cmjd+'/html/'+prefix+'QA-'+plate+'-'+cmjd+'.html'
                if os.path.exists(plateQAfile):
                    plateQApathPartial = plateQAfile.split(apred+'/')[1]
                    if j < nplatesall:
                        html.write('('+str(j+1).rjust(2)+') <A HREF="../'+plateQApathPartial+'">'+plate+': '+field+'</A><BR>\n')
                    else:
                        html.write('('+str(j+1).rjust(2)+') <A HREF="../'+plateQApathPartial+'">'+plate+': '+field+'</A><BR>\n')
                else:
                    if j < nplatesall:
                        html.write('<FONT COLOR="black">('+str(j+1).rjust(2)+') '+plate+': '+field+' (failed)<BR>\n')
                    else:
                        html.write('<FONT COLOR="black">('+str(j+1).rjust(2)+') '+plate+': '+field+' (failed)\n')

            # Column 7: Visit spectra plots
            html.write('<TD align="left">')
            for j in range(nplatesall):
                field = platePlanFiles[j].split(telescope+'/')[1].split('/')[0]
                plate = platePlanFiles[j].split(telescope+'/')[1].split('/')[1]
                # Check for failed plates
                plateQAfile = apodir+apred+'/visit/'+telescope+'/'+field+'/'+plate+'/'+cmjd+'/html/'+prefix+'QA-'+plate+'-'+cmjd+'.html'
                if os.path.exists(plateQAfile):
                    plateQApathPartial = plateQAfile.split(apred+'/')[1]
                    if j < nplatesall:
                        html.write('('+str(j+1).rjust(2)+') <A HREF="../'+plateQApathPartial.replace(prefix+'QA',prefix+'Plate')+'">'+plate+': '+field+'</A><BR>\n')
                    else:
                        html.write('('+str(j+1).rjust(2)+') <A HREF="../'+plateQApathPartial.replace(prefix+'QA',prefix+'Plate')+'">'+plate+': '+field+'</A>\n')
                else:
                    if j < nplatesall:
                        html.write('<FONT COLOR="black">('+str(j+1).rjust(2)+') '+plate+': '+field+'</FONT><BR>\n')
                    else:
                        html.write('<FONT COLOR="black">('+str(j+1).rjust(2)+') '+plate+': '+field+'</FONT>\n')

            # Column 8: Number of skies, telluric, and science targets.
            html.write('<TD align="left">')
            for j in range(nplatesall):
                field = platePlanFiles[j].split(telescope+'/')[1].split('/')[0]
                plate = platePlanFiles[j].split(telescope+'/')[1].split('/')[1]
                # Check for failed plates
                plateQAfile = apodir+apred+'/visit/'+telescope+'/'+field+'/'+plate+'/'+cmjd+'/html/'+prefix+'QA-'+plate+'-'+cmjd+'.html'
                if os.path.exists(plateQAfile):
                    note = ''
                    if fps:
                        plsumfile = load.filename('PlateSum', plate=int(plate), mjd=cmjd, fps=fps)
                        if os.path.exists(plsumfile):
                            plsum = fits.getdata(plsumfile,2)
                            try:
                                assignedSky, = np.where((plsum['assigned']) & (plsum['on_target']) & (plsum['objtype'] == 'SKY'))
                                assignedTel, = np.where((plsum['assigned']) & (plsum['on_target']) & (plsum['objtype'] == 'HOT_STD'))
                                assignedSci, = np.where((plsum['assigned']) & (plsum['on_target']) & (plsum['objtype'] == 'STAR'))
                                note = '   [' + str(len(assignedSky)).rjust(3) + ', ' + str(len(assignedTel)).rjust(3) + ', ' + str(len(assignedSci)).rjust(3) + ']'
                            except:
                                note = '   [?]'
                            #if len(assignedFib) < 1: note = ' (ZERO assigned)'
                    plateQApathPartial = plateQAfile.split(apred+'/')[1]
                    if j < nplatesall:
                        html.write('('+str(j+1).rjust(2)+') '+note+'<BR>\n')
                    else:
                        html.write('('+str(j+1).rjust(2)+') '+note+'\n')
                else:
                    if j < nplatesall:
                        html.write('<FONT COLOR="black">('+str(j+1).rjust(2)+') </FONT><BR>\n')
                    else:
                        html.write('<FONT COLOR="black">('+str(j+1).rjust(2)+') </FONT>\n')


            # Column 7: Combined files for this night
            #html.write('<TD>\n')

            # Column 8: Single stars observed for this night
            #html.write('<TD>\n')

            # Column 9: Dome flats observed for this night
            #html.write('<TD>\n')

            # Column 9: Summary files
            visSumPath = '../summary/'+cmjd+'/allVisitMJD-daily-'+telescope+'-'+cmjd+'.fits'
            starSumPath = '../summary/'+cmjd+'/allStarMJD-daily-'+telescope+'-'+cmjd+'.fits'
            if nplates >= 1: 
                html.write('<TD align="center"><a href="'+visSumPath+'">allVisitMJD</a>\n')
                html.write('<BR><a href="'+starSumPath+'">allStarMJD</a>\n')
            else:
                html.write('<TD>\n')

            # Column 10: Mean moon phase
            bgcolor = '#000000'
            txtcolor = '#FFFFFF'
            meanmoonphase = moon_illumination(tt)
            if meanmoonphase > 0.5: txtcolor = '#000000'
            if meanmoonphase > 0.1: bgcolor = '#282828'
            if meanmoonphase > 0.2: bgcolor = '#404040'
            if meanmoonphase > 0.3: bgcolor = '#606060'
            if meanmoonphase > 0.4: bgcolor = '#787878'
            if meanmoonphase > 0.5: bgcolor = '#989898'
            if meanmoonphase > 0.6: bgcolor = '#B0B0B0'
            if meanmoonphase > 0.7: bgcolor = '#C8C8C8'
            if meanmoonphase > 0.8: bgcolor = '#E8E8E8'
            if meanmoonphase > 0.9: bgcolor = '#FFFFFF'
            mphase = str(int(round(meanmoonphase*100)))+'%'
            html.write('<TD bgcolor="'+bgcolor+'" align="right" style = "color:'+txtcolor+';">'+mphase)
        html.write('</table>\n')

        # Summary calibration data
        caldir = 'cal/'
        html.write('<P> Calibration Data:\n')
        html.write('<UL>\n')
        html.write('<LI> <A HREF='+caldir+'/darkcorr/html/darks.html> Darks </A>\n')
        html.write('<LI> <A HREF='+caldir+'/flatcorr/html/flats.html> Flats </A>\n')
        html.write('<LI> <A HREF='+caldir+'/flux/html/flux.html> Fiber fluxes from petal flats </A>\n')
        html.write('<LI> <A HREF='+caldir+'/trace/html/trace.html> Traces </A>\n')
        html.write('<LI> <A HREF='+caldir+'/detector/html/rn.html> Readout noise </A>\n')
        html.write('<LI> <A HREF='+caldir+'/detector/html/gain.html> Gain </A>\n')
        html.write('<LI> <A HREF='+caldir+'/wave/html/wave.html> Wave cals </A>\n')
        html.write('</UL>\n')

        html.write('</body></html>\n')
        html.close()

    #---------------------------------------------------------------------------------------
    # Fields view
    if dofields is True:
        fieldfile = qadir+fieldfilebase
        print("----> makeMasterQApages: Creating "+fieldfilebase)

        now = datetime.datetime.now()
        today = datetime.date.today()
        current_time = now.strftime("%H:%M:%S")
        current_date = today.strftime("%B %d, %Y")

        html = open(fieldfile,'w')
        html.write('<HTML><BODY>\n')
        html.write('<HEAD><script src="sorttable.js"></script><title>APOGEE Field Summary</title></head>\n')
        html.write('<H1>APOGEE Observation Summary by Field</H1>\n')
        html.write('<P><I>last updated ' + current_date + ', ' + current_time + '</I></P>')
        html.write('<HR>\n')
        html.write('<p><A HREF=mjd.html>MJD view</A></p>\n')
        html.write('<p><A HREF=../monitor/apogee-n-monitor.html>APOGEE-N Instrument Monitor</A></p>\n')
        html.write('<p><A HREF=../monitor/apogee-s-monitor.html>APOGEE-S Instrument Monitor</A></p>\n')
        html.write('<p> Summary files: <a href="'+visSumPathN+'">allVisit</a>,  <a href="'+starSumPathN+'">allStar</a></p>\n')

        html.write('<H3>Sky coverage plots: </H3>\n')
        html.write('<A HREF="aitoff_galactic.png" target="_blank"><IMG SRC=aitoff_galactic.png WIDTH=600></A>\n')
        html.write('<A HREF="aitoff_equatorial.png" target="_blank"><IMG SRC=aitoff_equatorial.png WIDTH=600></A>\n')
#        html.write('<img src=aitoff.png width=45%>\n')
#        html.write('<img src=galactic.gif width=45%>\n')

    #    if ~keyword_set(suffix) then suffix='-'+apred_vers+'-'+aspcap_vers+'.fits'
    #    html.write('<a href=../../aspcap/'+apred_vers+'/'+aspcap_vers+'/allStar'+suffix+'> allStar'+suffix+' file </a>\n')
    #    html.write(' and <a href=../../aspcap/'+apred_vers+'/'+aspcap_vers+'/allVisit'+suffix+'> allVisit'+suffix+' file </a>\n')

        html.write('<br><br>Links on field name are to combined spectra plots and info (coming soon)\n')
        html.write('<br>Links on plate name are to visit spectra plots and info.\n')
        html.write('<br>Links on MJD are to QA and summary plots for the visit.\n')
        html.write('<br><br>Click on column headings to sort<br>\n')

        html.write('<TABLE BORDER=2 CLASS=sortable>\n')
        html.write('<TR bgcolor="#DCDCDC"><TH>FIELD <TH>PROGRAM <TH>TELESCOPE <TH>ASSIGNED<BR>FIBERS <TH>ASPCAP <TH>PLATE<BR>OR<BR>CONFIG <TH>MJD')
        html.write('<TH>LOC <TH>RA <TH>DEC <TH>GLON <TH>GLAT <TH>S/N(blue) <TH>S/N(green) <TH>S/N(red)')
        html.write('<TH>N<BR>EXP. <TH>TOTAL<BR>EXPTIME <TH>CART <TH>ZERO <TH>MOON<BR>PHASE\n')
    #    html.write('<TR><TD>FIELD<TD>Program<TD>ASPCAP<br>'+apred_vers+'/'+aspcap_vers+'<TD>PLATE<TD>MJD<TD>LOCATION<TD>RA<TD>DEC<TD>S/N(red)<TD>S/N(green)<TD>S/N(blue)\n')


        plates = np.array(glob.glob(apodir+apred+'/visit/*/*/*/*/'+'*PlateSum*.fits'))
        nplates = len(plates)
        # should really get this next stuff direct from database!
        # We are now!!!
        plans = yanny.yanny(os.environ['PLATELIST_DIR']+'/platePlans.par', np=True)

        # Get arrays of observed data values (plate ID, mjd, telescope, field name, program, location ID, ra, dec)
        iplate = np.zeros(nplates).astype(str)
        imjd = np.zeros(nplates).astype(str)
        iassigned = np.full(nplates, '---')
        itel = np.zeros(nplates).astype(str)
        iname = np.zeros(nplates).astype(str)
        iprogram = np.zeros(nplates).astype(str)
        iloc = np.zeros(nplates).astype(str)
        ira = np.zeros(nplates).astype(str)
        idec = np.zeros(nplates).astype(str)
        ilon = np.zeros(nplates).astype(str)
        ilat = np.zeros(nplates).astype(str)
        inexposures = np.zeros(nplates).astype(str)
        iexptime = np.zeros(nplates).astype(str)
        isnr = np.full((nplates,3), 0).astype(str)
        icart = np.zeros(nplates).astype(str)
        izero = np.zeros(nplates).astype(str)
        imoonphase = np.zeros(nplates)
        inst = np.zeros(nplates).astype(str)
        for i in range(nplates): 
            plate = os.path.basename(plates[i]).split('-')[1]
            iplate[i] = plate
            mjd = os.path.basename(plates[i]).split('-')[2][:-5]
            if int(mjd)>59556:
                fps = True
            else:
                fps = False
            imjd[i] = mjd
            tmp = plates[i].split('visit/')
            tel = tmp[1].split('/')[0]
            itel[i] = tel
            tmp = plates[i].split(tel+'/')
            name = tmp[1].split('/')[0]
            iname[i] = name

            inst[i] = 'apogee-n'
            if itel[i] != 'apo25m': inst[i] = 'apogee-s'

            load = apload.ApLoad(apred=apred, telescope=itel[i])
            platesumfile = load.filename('PlateSum', plate=int(plate), mjd=mjd, fps=fps)

            if fps:
                # Get field center RA and DEC from confSummary file
                plugmapfile = load.filename('confSummary', configid=int(iplate[i]))
                plans = yanny.yanny(plugmapfile, np=True)
                tra = float(plans['raCen'])
                tdec = float(plans['decCen'])
                ira[i] = str("%.6f" % round(tra,6))
                idec[i] = str("%.6f" % round(tdec,6))
                c_icrs = SkyCoord(ra=tra*u.degree, dec=tdec*u.degree, frame='icrs')
                ilon[i] = str("%.6f" % round(c_icrs.galactic.l.deg,6))
                ilat[i] = str("%.6f" % round(c_icrs.galactic.b.deg,6))
                if os.path.exists(platesumfile):
                    try:
                        plsum2 = fits.getdata(platesumfile,2)
                        ass, = np.where(plsum2['assigned'])
                        iassigned[i] = str(len(ass))
                    except:
                        nothing = 5

                ## Read planfile
                #planfile = load.filename('Plan', plate=int(iplate[i]), mjd=imjd[i], fps=fps)
                #planstr = plan.load(planfile, np=True)
                ## Get values from plan file.
                #badfiberid = planstr['badfiberid']
                #plugmap =    planstr['plugmap']
                #plug = platedata.getdata(int(iplate[i]), int(imjd[i]), apred, tel, plugid=plugmap, badfiberid=badfiberid)
                iprogram[i] = 'SDSS-V'
                iloc[i] = name
            else:
                gd, = np.where(int(plate) == plans['PLATEPLANS']['plateid'])
                if len(gd)>0:
                    iprogram[i] = plans['PLATEPLANS']['programname'][gd][0].astype(str)
                    iloc[i] = str(int(round(plans['PLATEPLANS']['locationid'][gd][0])))
                    tra = plans['PLATEPLANS']['raCen'][gd][0]
                    tdec = plans['PLATEPLANS']['decCen'][gd][0]
                    ira[i] = str("%.6f" % round(tra,6))
                    idec[i] = str("%.6f" % round(tdec,6))
                    c_icrs = SkyCoord(ra=tra*u.degree, dec=tdec*u.degree, frame='icrs')
                    ilon[i] = str("%.6f" % round(c_icrs.galactic.l.deg,6))
                    ilat[i] = str("%.6f" % round(c_icrs.galactic.b.deg,6))

            if os.path.exists(platesumfile) is False:
                tmp = glob.glob(platesumfile.replace('None','*'))
                if len(tmp) > 0: 
                    platesumfile = tmp[0]
            if os.path.exists(platesumfile):
                plsum1 = fits.getdata(platesumfile,1)
                inexposures[i] = str(len(plsum1['IM']))
                iexptime[i] = str(np.sum(plsum1['EXPTIME']))
                icart[i] = str(plsum1['CART'][0])
                izero[i] = str("%.2f" % round(np.mean(plsum1['ZERO']),2))
                imoonphase[i] = np.mean(plsum1['MOONPHASE'])
                try:
                    plsum3 = fits.getdata(platesumfile,3)
                    isnr[i,0] = str(int(round(plsum3['SN'][0][0])))
                    isnr[i,1] = str(int(round(plsum3['SN'][0][1])))
                    isnr[i,2] = str(int(round(plsum3['SN'][0][2])))
                except:
                    print("----> makeMasterQApages: no 3rd extension in PlateSum for plate " + iplate[i] + ', MJD ' + imjd[i])
            else:
                print('----> makeMasterQApages: Problem with plate/config ' + iplate[i] + ', MJD ' + imjd[i] + '. PlateSum not found.')

        # Sort by MJD
        order = np.argsort(imjd)
        plates = plates[order]
        iplate = iplate[order]
        imjd = imjd[order]
        iassigned = iassigned[order]
        itel = itel[order]
        iname = iname[order]
        iprogram = iprogram[order]
        iloc = iloc[order]
        ira = ira[order]
        idec = idec[order]
        ilon = ilon[order]
        ilat = ilat[order]
        inexposures = inexposures[order]
        iexptime = iexptime[order]
        isnr = isnr[order]
        icart = icart[order]
        izero = izero[order]
        imoonphase = imoonphase[order]
        inst = inst[order]

        for i in range(nplates):
            color = '#ffb3b3'
            if len(iprogram[i]) < 3:
                if iprogram[i] == 'RM': 
                    color = '#B3E5FC'
            else:
                if iprogram[i][0:2] == 'RM': color = '#D39FE4'
                if iprogram[i][0:5] == 'AQMES': color = '#DCEDC8'
                if iprogram[i] == 'halo_dsph': color = '#D39FE4'
                if iprogram[i][0:3] == 'MWM': color = '#B3E5FC'
                if iprogram[i][0:5] == 'eFEDS': color='#FFF9C4'

            html.write('<TR bgcolor=' + color + '><TD>' + iname[i]) 
            html.write('<TD>' + str(iprogram[i])) 
            html.write('<TD>' + itel[i]) 
            html.write('<TD>' + iassigned[i]) 
            html.write('<TD> --- ')
            qalink = '../visit/' + itel[i] + '/' + iname[i] + '/' + iplate[i] + '/' + imjd[i] + '/html/' + prefix + 'QA-' + iplate[i] + '-' + imjd[i] + '.html'
            html.write('<TD align="center"><A HREF="' + qalink + '" target="_blank">' + iplate[i] + '</A>')
            html.write('<TD align="center"><A HREF="../exposures/' + inst[i] + '/' + imjd[i] + '/html/' + imjd[i] + '.html">' + imjd[i] + '</A>') 
            html.write('<TD align="center">' + iloc[i])
            html.write('<TD align="right">' + ira[i]) 
            html.write('<TD align="right">' + idec[i])
            html.write('<TD align="right">' + ilon[i]) 
            html.write('<TD align="right">' + ilat[i])
            html.write('<TD align="right">' + isnr[i,2]) 
            html.write('<TD align="right">' + isnr[i,1]) 
            html.write('<TD align="right">' + isnr[i,0]) 
            html.write('<TD align="right">' + inexposures[i]) 
            html.write('<TD align="right">' + iexptime[i]) 
            html.write('<TD align="right">' + icart[i]) 
            html.write('<TD align="right">' + izero[i])
            bgcolor = '#000000'
            txtcolor = '#FFFFFF'
            if imoonphase[i] > 0.5: txtcolor = '#000000'

            if imoonphase[i] > 0.1: bgcolor = '#282828'
            if imoonphase[i] > 0.2: bgcolor = '#404040'
            if imoonphase[i] > 0.3: bgcolor = '#606060'
            if imoonphase[i] > 0.4: bgcolor = '#787878'
            if imoonphase[i] > 0.5: bgcolor = '#989898'
            if imoonphase[i] > 0.6: bgcolor = '#B0B0B0'
            if imoonphase[i] > 0.7: bgcolor = '#C8C8C8'
            if imoonphase[i] > 0.8: bgcolor = '#E8E8E8'
            if imoonphase[i] > 0.9: bgcolor = '#FFFFFF'

            mphase = str(int(round(imoonphase[i]*100)))+'%'
            html.write('<TD bgcolor="'+bgcolor+'" align="right" style = "color:'+txtcolor+';">'+mphase+'\n') 

        html.write('</BODY></HTML>\n')
        html.close()

        #---------------------------------------------------------------------------------------
        # Aitoff maps
        # Set up some basic plotting parameters.
        matplotlib.use('agg')
        fontsize = 24;   fsz = fontsize * 0.60
        matplotlib.rcParams.update({'font.size':fontsize, 'font.family':'serif'})
        matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"
        alf = 0.80
        axwidth = 1.5
        axmajlen = 7
        axminlen = 3.5
        msz = 50
        lonlabs = ['210','240','270','300','330','0','30','60','90','120','150']
        nlon = len(lonlabs);  lonstart = 0.085;  lonsep = 0.083
        nplots = 2

        for j in range(nplots):
            if j == 0: ptype = 'galactic'
            if j == 1: ptype = 'equatorial'
            plotfile = 'aitoff_' + ptype + '.png'
            print("----> makeMasterQApages: Making " + plotfile)

            fig=plt.figure(figsize=(13,7))
            ax1 = fig.add_subplot(111, projection = 'aitoff')
            ax1.grid(True)
            ax1.axes.xaxis.set_ticklabels([])

            ax1.text(0.5, 0.013, r'0$^{\circ}$', ha='center', transform=ax1.transAxes)
            #for t in range(nlon): 
            #    ax1.text(lonstart+lonsep*t, 0.5, lonlabs[t], ha='center', va='center', fontsize=fsz, transform=ax1.transAxes)
            #ax1.text(0.165, 0.5, '240', ha='center', va='center', fontsize=fsz, transform=ax1.transAxes)
            #ax1.set_xticks([-150,-120,-90,-60,-30,0,30,60,90,120,150])
            #ax1.set_xticklabels(['210','240','270','300','330','0','30','60','90','120','150'])
            #ax2 = fig.add_subplot(122, projection = 'aitoff')
            #axes = [ax1, ax2]

            ra = ira.astype(float)
            dec = idec.astype(float)
            c = SkyCoord(ra*u.degree, dec*u.degree, frame='icrs')
            if j == 0:
                gl = c.galactic.l.degree
                gb = c.galactic.b.degree
                uhoh, = np.where(gl > 180)
                if len(uhoh) > 0: gl[uhoh] -= 360
                x = gl * (math.pi/180)
                y = gb * (math.pi/180)
            else:
                ra = c.ra.degree
                dec = c.dec.degree
                uhoh, = np.where(ra > 180)
                if len(uhoh) > 0: ra[uhoh] -= 360
                x = ra * (math.pi/180)
                y = dec * (math.pi/180)

            p, = np.where((iprogram == 'RM') | (iprogram == 'RMv2'))
            if len(p) > 0: ax1.scatter(x[p], y[p], marker='P', s=msz, edgecolors='k', alpha=alf, c='#D39FE4', label='RM ('+str(len(p))+')')

            p, = np.where((iprogram == 'AQMES-Wide') | (iprogram == 'AQMES-Medium') | (iprogram == 'AQMES-Bonus'))
            if len(p) > 0: ax1.scatter(x[p], y[p], marker='^', s=msz, edgecolors='k', alpha=alf, c='#DCEDC8', label='AQMES ('+str(len(p))+')')

            p, = np.where((iprogram == 'MWM') | (iprogram == 'MWM_30min') | (iprogram == 'halo_dsph') | (iprogram == 'MWM2') | (iprogram == 'MWM2_sky')
                                              | (iprogram == 'MWM3') | (iprogram == 'MWM_30min2') | (iprogram == 'MWM_30min3'))
            if len(p) > 0: ax1.scatter(x[p], y[p], marker='*', s=msz*2, edgecolors='k', alpha=alf, c='#B3E5FC', label='MWM ('+str(len(p))+')')

            p, = np.where((iprogram == 'eFEDS1') | (iprogram == 'eFEDS2') | (iprogram == 'eFEDS3'))
            if len(p) > 0: ax1.scatter(x[p], y[p], marker='s', s=msz*0.8, edgecolors='k', alpha=alf, c='#FFF9C4', label='eFEDS ('+str(len(p))+')')

            p, = np.where((iprogram != 'RM') & (iprogram != 'RMv2') & (iprogram != 'AQMES-Wide') & (iprogram != 'AQMES-Medium') & (iprogram != 'AQMES-Bonus') & 
                          (iprogram != 'MWM') & (iprogram != 'MWM_30min') & (iprogram != 'halo_dsph') & (iprogram != 'MWM2') & (iprogram != 'MWM2_sky') & 
                          (iprogram != 'MWM3') & (iprogram != 'MWM_30min2') & (iprogram != 'MWM_30min3') & (iprogram != 'eFEDS1') & (iprogram != 'eFEDS2') & 
                          (iprogram != 'eFEDS3'))
            if len(p) > 0: ax1.scatter(x[p], y[p], marker='o', s=msz*0.9, edgecolors='k', alpha=alf, c='#ffb3b3', label='other ('+str(len(p))+')')

            ax1.text(0.5,1.04,ptype.capitalize(),transform=ax1.transAxes,ha='center')
            ax1.legend(loc=[-0.24,-0.06], labelspacing=0.5, handletextpad=-0.1, facecolor='white', fontsize=fsz, borderpad=0.3)

            fig.subplots_adjust(left=0.2,right=0.99,bottom=0.02,top=0.98,hspace=0.09,wspace=0.09)
            plt.savefig(qadir+plotfile)
            plt.close('all')

    #plt.ion()
    print("----> makeMasterQApages: Done.\n")

###################################################################################################
''' MAKECALFITS: Make FITS file for cals (lamp brightness, line widths, etc.) '''
def makeCalFits(load=None, ims=None, mjd=None, instrument=None, clobber=None):

    lineSearchRad = 40

    outfile = load.filename('QAcal', mjd=mjd)
    if (os.path.exists(outfile) is False) | (clobber is True):
        print("--------------------------------------------------------------------")
        print("Running MAKECALFITS for MJD " + mjd)

        # Make directory if it doesn't exist
        if os.path.exists(os.path.dirname(outfile)) is False: os.makedirs(os.path.dirname(outfile))

        n_exposures = len(ims)

        nlines = 2
        chips = np.array(['a','b','c'])
        nchips = len(chips)

        tharline = np.array([[940.3,1129.4,1131.9],[1728.3,623.0,1778.4]])
        uneline =  np.array([[604.5,1214.1,1118.1],[1762.6,605.3,1895.3]])
        if int(mjd) > 59420:
            tharline -= 21.832
            uneline -= 21.855

        if instrument == 'apogee-s': tharline = np.array([[940.,1112.,1102.],[1727.,608.,1745.]])
        if instrument == 'apogee-s':  uneline = np.array([[604.,1229.,1088.],[1763.,620.,1860.]])

        fibers = np.array([10,80,150,220,290])
        nfibers = len(fibers)

        # Make output structure.
        dt = np.dtype([('NAME',    np.str,30),
                       ('MJD',     np.str,30),
                       ('JD',      np.float64),
                       ('NFRAMES', np.int16),
                       ('NREAD',   np.int16),
                       ('EXPTIME', np.float32),
                       ('QRTZ',    np.int16),
                       ('UNE',     np.int16),
                       ('THAR',    np.int16),
                       ('FLUX',    np.float32,(nchips,300)),
                       ('GAUSS',   np.float32,(nlines,nchips,nfibers,4)),
                       ('WAVE',    np.float64,(nlines,nchips,nfibers)),
                       ('FIBERS',  np.int16,(nfibers)),
                       ('LINES',   np.float32,(nlines,nchips))])

        struct = np.zeros(n_exposures, dtype=dt)

        # Loop over exposures and get 1D images to fill structure.
        for i in range(n_exposures):
            try:
                oneD = load.apread('1D', num=ims[i])
            except:
                print('----> makeCalFits: '+prefix+'1D not found for exposure ' + str(ims[i]))
                continue

            oneDflux = np.array([oneD[0].flux, oneD[1].flux, oneD[2].flux])
            oneDerror = np.array([oneD[0].error, oneD[1].error, oneD[2].error])
            oneDhdr = oneD[0].header

            struct['NAME'][i] =    ims[i]
            struct['MJD'][i] =     mjd
            struct['JD'][i] =      oneDhdr['JD-MID']
            struct['NFRAMES'][i] = oneDhdr['NFRAMES']
            struct['NREAD'][i] =   oneDhdr['NREAD']
            struct['EXPTIME'][i] = oneDhdr['EXPTIME']
            try:
                struct['QRTZ'][i] =    oneDhdr['LAMPQRTZ']
                struct['THAR'][i] =    oneDhdr['LAMPTHAR']
                struct['UNE'][i] =     oneDhdr['LAMPUNE']
            except:
                if oneDhdr['IMAGETYP'] == 'QuartzFlat':
                    struct['QRTZ'][i] =    1
                    struct['THAR'][i] =    0
                    struct['UNE'][i] =     0
            struct['FIBERS'][i] =  fibers

            tp = 'quartz'
            if struct['THAR'][i] == 1: tp = 'ThAr'.ljust(6)
            if struct['UNE'][i] == 1: tp = 'UNe'.ljust(6)

            print("----> makeCalFits: running " + tp + " exposure " + str(ims[i]) + " (" + str(i+1) + "/" + str(n_exposures) + ")")

            # Quartz exposures.
            if struct['QRTZ'][i] == 1: struct['FLUX'][i] = np.nanmedian(oneDflux, axis=1)

            # Arc lamp exposures.
            if (struct['THAR'][i] == 1) | (struct['UNE'][i] == 1):
                if struct['THAR'][i] == 1: line = tharline
                if struct['THAR'][i] != 1: line = uneline

                struct['LINES'][i] = line

                nlines = 1
                if line.shape[1] != 1: nlines = line.shape[0]

                print('MEASURED_CENT  EXPECTED_CENT   DIFF')#      SUMFLUX')
                for iline in range(nlines):
                    for ichip in range(nchips):
                        for ifiber in range(nfibers):
                            fiber = fibers[ifiber]
                            intline = int(round(line[iline,ichip]))
                            gflux =   oneDflux[ichip, intline-lineSearchRad:intline+lineSearchRad, fiber]
                            gerror = oneDerror[ichip, intline-lineSearchRad:intline+lineSearchRad, fiber]
                            try:
                                # Try to fit Gaussians to the lamp lines
                                gpeaks = peakfit.peakfit(gflux, sigma=gerror)
                                gd, = np.where(np.isnan(gpeaks['pars'][:, 0]) == False)
                                gpeaks = gpeaks[gd]
                                # Find the desired peak and load struct
                                gpeaks['pars'][:, 1] = gpeaks['pars'][:, 1] + intline - lineSearchRad
                                pixdif = np.abs(gpeaks['pars'][:, 1] - line[iline, ichip])
                                gdline, = np.where(pixdif == np.min(pixdif))
                                tmp = iline+ichip+ifiber
                                diff = gpeaks['pars'][:, 1][gdline][0] - line[iline,ichip]
                                if fiber == 150:
                                    txt1 = str("%.2f" % round(gpeaks['pars'][:, 1][gdline][0],2)).rjust(12)
                                    txt2 = str("%.2f" % round(line[iline,ichip],2)).rjust(15)
                                    txt3 = str("%.2f" % round(diff,2)).rjust(10)
                                    print(txt1 + txt2 + txt3)# + txt4)
                            except:
                                print("----> makeCalFits: ERROR!!! No lines found for " + tp + " exposure " + str(ims[i]))
                                continue

                            if len(gdline) > 0:
                                struct['GAUSS'][i, iline, ichip, ifiber, :] = gpeaks['pars'][gdline, :][0]
                                struct['FLUX'][i, ichip, ifiber] = gpeaks['sumflux'][gdline]
                            else:
                                print("----> makeCalFits: ERROR!!! Desired line not found for " + tp + " exposure " + str(ims[i]))


        Table(struct).write(outfile, overwrite=True)

        print("Done with MAKECALFITS for MJD " + mjd)
        print("Made " + outfile)
        print("--------------------------------------------------------------------\n")

###################################################################################################
''' MAKEDARKFITS: Make FITS file for darks (get mean/stddev of column-medianed quadrants) '''
def makeDarkFits(load=None, ims=None, mjd=None, instrument=None, clobber=None):

    prefix = 'ap'
    if instrument == 'apogee-s': prefix = 'as'

    outfile = load.filename('QAcal', mjd=mjd).replace('QAcal','QAdarkflat')
    if (os.path.exists(outfile) is False) | (clobber is True):
        print("--------------------------------------------------------------------")
        print("Running MAKEDARKFITS for MJD "+mjd)

        if os.path.exists(os.path.dirname(outfile)) is False: os.makedirs(os.path.dirname(outfile))

        n_exposures = len(ims)

        chips=np.array(['a','b','c'])
        nchips = len(chips)
        nquad = 4

        # Make output structure.
        dt = np.dtype([('NAME',    np.str, 30),
                       ('MJD',     np.str, 30),
                       ('JD',      np.float64),
                       ('NFRAMES', np.int16),
                       ('NREAD',   np.int16),
                       ('EXPTIME', np.float32),
                       ('QRTZ',    np.int16),
                       ('UNE',     np.int16),
                       ('THAR',    np.int16),
                       ('EXPTYPE', np.str, 30),
                       ('MEAN',    np.float32, (nquad,nchips)),
                       ('SIG',     np.float32, (nquad,nchips))])

        struct = np.zeros(n_exposures, dtype=dt)

        # Loop over exposures and get 2D images to fill structure.
        for i in range(n_exposures):
            try:
                twoD = load.apread('2D', num=ims[i])
            except:
                print('----> makeDarkFits: '+prefix+'2D not found for exposure ' + str(ims[i]))
                continue

            print("----> makeDarkFits: running exposure " + str(ims[i]) + " (" + str(i+1) + "/" + str(n_exposures) + ")")

            twoD = load.apread('2D', num=ims[i])
            twoDflux = np.array([twoD[0].flux, twoD[1].flux, twoD[2].flux])
            twoDhdr = twoD[0].header

            struct['NAME'][i] =    ims[i]
            struct['MJD'][i] =     mjd
            struct['JD'][i] =      twoDhdr['JD-MID']
            struct['NFRAMES'][i] = twoDhdr['NFRAMES']
            struct['NREAD'][i] =   twoDhdr['NREAD']
            struct['EXPTIME'][i] = twoDhdr['EXPTIME']
            try:
                struct['QRTZ'][i] =    twoDhdr['LAMPQRTZ']
                struct['UNE'][i] =     twoDhdr['LAMPUNE']
                struct['THAR'][i] =    twoDhdr['LAMPTHAR']
            except: 
                pass
            struct['EXPTYPE'][i] = twoDhdr['EXPTYPE']

            # Get the mean and stddev of flux in each quadrant of each detector
            for ichip in range(nchips):
                i1 = 10
                i2 = 500
                for iquad in range(nquad):
                    sm = np.nanmedian(twoDflux[ichip, i1:i2, 10:2000], axis=1)
                    struct['MEAN'][i, iquad, ichip] = np.nanmean(sm)
                    struct['SIG'][i, iquad, ichip] = np.nanstd(sm)
                    i1 += 512
                    i2 += 512

        Table(struct).write(outfile, overwrite=True)

        print("Done with MAKEDARKFITS for MJD " + mjd)
        print("Made " + outfile)
        print("--------------------------------------------------------------------\n")

###################################################################################################
''' MAKEEXPFITS: Make FITS file for darks (get mean/stddev of column-medianed quadrants) '''
def makeExpFits(instrument=None, apodir=None, apred=None, load=None, mjd=None, clobber=None):

    # Establish raw data directory
    rawdir = os.environ['APOGEE_DATA_N'] + '/' + mjd + '/'
    prefix = 'ap'
    if instrument == 'apogee-s': 
        rawdir = os.environ['APOGEE_DATA_S'] + '/' + mjd + '/'
        prefix = 'as'

    expdir = apodir + apred + '/exposures/' + instrument + '/' + mjd + '/'
    outfile = expdir + mjd + 'exp.fits'
    if (os.path.exists(outfile) is False) | (clobber is True):
        print("--------------------------------------------------------------------")
        print("Running MAKEEXPFITS for MJD "+mjd)

        if os.path.exists(os.path.dirname(outfile)) is False: os.makedirs(os.path.dirname(outfile))

        ims = glob.glob(rawdir + prefix+'R-a-*')
        ims.sort()
        ims = np.array(ims)
        n_exposures = len(ims)

        chips=np.array(['a','b','c'])
        nchips = len(chips)

        # Make output structure.
        dt = np.dtype([('MJD',       np.int32),
                       ('DATEOBS',   np.str, 30),
                       ('JD',        np.float64),
                       ('NUM',       np.int32),
                       ('NFRAMES',   np.int16),
                       ('IMAGETYP',  np.str, 30),
                       ('PLATEID',   np.int16),
                       ('CARTID',    np.int16),
                       ('RA',        np.float64),
                       ('DEC',       np.float64),
                       ('SEEING',    np.float32),
                       ('ALT',       np.float32),
                       ('QRTZ',      np.int16),
                       ('THAR',      np.int16),
                       ('UNE',       np.int16),
                       ('FFS',       np.str, 30),
                       ('LN2LEVEL',  np.float32),
                       ('DITHPIX',   np.float32),
                       ('TRACEDIST', np.float32),
                       ('MED',       np.float32, (nchips,300))])

        struct = np.zeros(n_exposures, dtype=dt)

        # Loop over exposures and fill structure.
        for i in range(n_exposures):
            imnum = os.path.basename(ims[i]).split('-')[-1].split('.')[0]
            print("----> makeExpFits: reading exposure " + imnum + " (" + str(i+1) + "/" + str(n_exposures) + ")")

            raw = load.apR(int(imnum))
            hdr = raw['a'][1].header

            dateobs = hdr['DATE-OBS']
            t = Time(dateobs, format='fits')

            struct['MJD'][i] =       int(mjd)
            struct['DATEOBS'][i] =   dateobs
            struct['JD'][i] =        t.jd
            struct['NUM'][i] =       int(imnum)
            struct['NFRAMES'][i] =   hdr['NFRAMES']
            struct['IMAGETYP'][i] =  hdr['IMAGETYP']
            try:
                struct['PLATEID'][i] =   hdr['PLATEID']
                struct['CARTID'][i] =    hdr['CARTID']
                struct['RA'][i] =        hdr['RA']
                struct['DEC'][i] =       hdr['DEC']
                struct['ALT'][i] =       hdr['ALT']
            except: pass
            if hdr['SEEING'] != 'NAN.0':
                struct['SEEING'][i] = hdr['SEEING']
            else:
                print("----> makeExpFits: seeing is 'NAN.0' for exposure " + imnum + ". Setting to -9.999")
                struct['SEEING'][i] = -9.999
            try:
                struct['QRTZ'][i] =      hdr['LAMPQRTZ']
                struct['THAR'][i] =      hdr['LAMPTHAR']
                struct['UNE'][i] =       hdr['LAMPUNE']
            except:
                if hdr['IMAGETYP'] == 'QuartzFlat':
                    struct['QRTZ'][i] =    1
                    struct['THAR'][i] =    0
                    struct['UNE'][i] =     0
            struct['FFS'][i] =       hdr['FFS']
            struct['LN2LEVEL'][i] =  hdr['LN2LEVEL']
            struct['DITHPIX'][i] =   hdr['DITHPIX']
            #struct['TRACEDIST'][i] = ?

            # Get median fluxes from Dome Flats
            if struct['IMAGETYP'][i] == 'DomeFlat':
                fluxfile = load.filename('Flux', num=int(imnum), chips=True).replace('Flux-', 'Flux-c-')
                if os.path.exists(fluxfile):
                    tmp = load.apFlux(int(imnum))
                    for ichip in range(nchips):
                        chip = chips[ichip]
                        flux = tmp[chip][1].data
                        mask = tmp[chip][3].data
                        for ifiber in range(300):
                            flux[ifiber, :] *= mask
                        med = np.nanmedian(flux,axis=1)
                        struct['MED'][i,ichip,:] = med

        Table(struct).write(outfile, overwrite=True)

        print("Done with MAKEEXPFITS for MJD " + mjd)
        print("Made " + outfile)
        print("--------------------------------------------------------------------\n")

###################################################################################################
''' GETFLUX: Translation of getflux.pro '''
def getflux(d=None, skyline=None, rows=None):

    chips = np.array(['a','b','c'])
    nnrows = len(rows)

    try:
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

    except:
        return 0


###################################################################################################
''' old_makeVisHTML: obsolete version of code for making the plate/visit level html '''
def old_makeVisHTML(load=None, plate=None, mjd=None, survey=None, apred=None, telescope=None, fluxid=None): 

    print("----> makeVisHTML: Running plate " + plate + ", MJD " + mjd)

    # HTML header background color
    thcolor = '#DCDCDC'

    if int(mjd)>59556:
        fps = True
    else:
        fps = False

    apodir = os.environ.get('APOGEE_REDUX') + '/'

    # Make html directory if it doesn't already exist.
    htmldir = os.path.dirname(load.filename('Plate', plate=int(plate), mjd=mjd, chips=True, fps=fps)) + '/html/'
    if os.path.exists(htmldir) == False: os.makedirs(htmldir)

    #if os.path.exists(htmldir + 'sorttable.js') == False:
    #    print("----> makeVisHTML: getting sorttable.js...")
    #    subprocess.call(['wget', '-q', sort_table_link])
    #    subprocess.call(['mv', 'sorttable.js', htmldir])

    # Get the HTML file name... apPlate-plate-mjd
    htmlfile = os.path.basename(load.filename('Plate', plate=int(plate), mjd=mjd, chips=True, fps=fps)).replace('.fits','')
    
    # Base directory where star-level stuff goes
    starHTMLbase = apodir + apred + '/stars/' + telescope +'/'

    # Start db session for getting all visit info
    db = apogeedb.DBSession()

    # Load in the allVisitMJD file
    allVpath = apodir + apred + '/summary/' + mjd + '/allVisitMJD-' + apred + '-' + telescope + '-' + mjd + '.fits'
    allV = None
    if os.path.exists(allVpath):
        allV = fits.getdata(allVpath)
        if len(allV)==0: allV=None
    if allV is None:
        # Try the database, copied from runapogee.create_sumfiles()
        vcols = ['apogee_id', 'target_id', 'apred_vers','file', 'uri', 'fiberid', 'plate', 'mjd', 'telescope', 'survey',
                 'field', 'programname', 'ra', 'dec', 'glon', 'glat', 'jmag', 'jerr', 'hmag',
                 'herr', 'kmag', 'kerr', 'src_h', 'pmra', 'pmdec', 'pm_src', 'apogee_target1', 'apogee_target2', 'apogee_target3',
                 'apogee_target4', 'catalogid', 'gaiadr2_plx', 'gaiadr2_plx_error', 'gaiadr2_pmra', 'gaiadr2_pmra_error',
                 'gaiadr2_pmdec', 'gaiadr2_pmdec_error', 'gaiadr2_gmag', 'gaiadr2_gerr', 'gaiadr2_bpmag', 'gaiadr2_bperr',
                 'gaiadr2_rpmag', 'gaiadr2_rperr', 'sdssv_apogee_target0', 'firstcarton', 'targflags', 'snr', 'starflag', 
                 'starflags','dateobs','jd']
        rvcols = ['starver', 'bc', 'vtype', 'vrel', 'vrelerr', 'vrad', 'chisq', 'rv_teff', 'rv_feh',
                  'rv_logg', 'xcorr_vrel', 'xcorr_vrelerr', 'xcorr_vrad', 'n_components', 'rv_components']
        cols = ','.join('v.'+np.char.array(vcols)) +','+ ','.join('rv.'+np.char.array(rvcols))
        allV = db.query(sql="select "+cols+" from apogee_drp.rv_visit as rv join apogee_drp.visit as v on rv.visit_pk=v.pk "+\
                        "where rv.apred_vers='"+apred+"' and rv.telescope='"+telescope+"' and v.mjd="+str(mjd)+" and rv.starver='"+str(mjd)+"'")
        if len(allV)==0: allV=None

    # Load in the apPlate file
    apPlate = load.apPlate(int(plate), mjd)
    try:
        data = apPlate['a'][11].data[::-1]
    except:
        print("----> makeVisHTML: PROBLEM! apPlate not found for plate " + plate + ", MJD " + mjd)
        return
    nfiber = len(data)

    # Read in flux file to get an idea of throughput
    fluxfile = os.path.basename(load.filename('Flux', num=fluxid, chips=True))
    flux = load.apFlux(fluxid)
    medflux = np.nanmedian(flux['a'][1].data, axis=1)[::-1]
    throughput = medflux / np.nanmax(medflux)

    # Read the confSummary file to get first carton values
    plugmapfile = load.filename('confSummary', configid=int(plate))
    plug = yanny.yanny(plugmapfile, np=True)['FIBERMAP']

    # For each star, create the exposure entry on the web page and set up the plot of the spectrum.
    vishtml = open(htmldir + htmlfile + '.html', 'w')
    vishtml.write('<HTML>\n')
    vishtml.write('<HEAD><script src="../../../../../../../sorttable.js"></script><title>' + htmlfile + '</title></head>\n')
    vishtml.write('<BODY>\n')

    vishtml.write('<H1>' + htmlfile + '</H1><HR>\n')
    #vishtml.write('<A HREF=../../../../red/'+mjd+'/html/'+pfile+'.html> 1D frames </A>\n')
    #vishtml.write('<BR><A HREF=../../../../red/'+mjd+'/html/ap2D-'+str(plSum1['IM'][i])+'.html> 2D frames </A>\n')
    vishtml.write('<P><B>Note:</B> the "Dome Flat Throughput" column gives the median dome flat flux in each ')
    vishtml.write('fiber divided by the maximum median dome flat flux across all fibers. ')
    vishtml.write('<BR>Low numbers are generally bad, and that column is color-coded accordingly.</P>\n')
    vishtml.write('<P>Click the column headers to sort.</p>\n')
    vishtml.write('<TABLE BORDER=2 CLASS="sortable">\n')
    vishtml.write('<TR bgcolor="' + thcolor + '"><TH>Fiber<BR>(MTP) <TH>APOGEE ID <TH>H<BR>mag <TH>Raw<BR>J - K <TH>Target<BR>Type <TH>Target & Data Flags')
    vishtml.write('<TH>S/N <TH>Vrad<BR>(km/s) <TH>N<BR>comp <TH>RV<BR>Teff (K) <TH>RV<BR>log(g) <TH>RV<BR>[Fe/H] <TH>Dome Flat<BR>Throughput <TH>apVisit Plot\n')
#    vishtml.write('<TR><TH>Fiber<TH>APOGEE ID<TH>H<TH>H - obs<TH>S/N<TH>Target<BR>Type<TH>Target & Data Flags<TH>Spectrum Plot\n')

    tputfile = load.filename('Plate', plate=int(plate), mjd=mjd, chips=True, fps=fps).replace('apPlate', 'throughput').replace('fits', 'dat')
    tputdat = open(tputfile, 'w')

    db = apogeedb.DBSession()

    # Loop over the fibers
    for j in range(300):
        jdata = data[j]
        fiber = jdata['FIBERID']
        if fiber > 0:
            cfiber = str(fiber).zfill(3)
            cblock = str(np.ceil(fiber / 30).astype(int))

            objid = jdata['OBJECT']
            objtype = jdata['OBJTYPE']
            hmag = jdata['HMAG']
            cjmag = str("%.3f" % round(jdata['JMAG'], 3))
            chmag = str("%.3f" % round(jdata['HMAG'], 3))
            ckmag = str("%.3f" % round(jdata['KMAG'],3 ))
            jkcolor = jdata['JMAG'] - jdata['KMAG']
            if (jdata['JMAG'] < 0) | (jdata['KMAG'] < 0): jkcolor = -9.999
            cjkcolor = str("%.3f" % round(jkcolor, 3))
    #        magdiff = str("%.2f" % round(plSum2['obsmag'][j][0][1] -hmag,2))
            cra = str("%.5f" % round(jdata['RA'], 5))
            cdec = str("%.5f" % round(jdata['DEC'], 5))
            txt1 = '<A HREF="http://simbad.u-strasbg.fr/simbad/sim-coo?Coord='+cra+'+'+cdec+'&CooFrame=FK5&CooEpoch=2000&CooEqui=2000'
            txt2 = '&CooDefinedFrames=none&Radius=10&Radius.unit=arcsec&submit=submit+query&CoordList=" target="_blank">SIMBAD Link</A>'
            simbadlink = txt1 + txt2

            apStarRelPath = None
            if (objtype != 'SKY') & (objid != '2MNone') & (objid != '2M') & (objid != ''):
                # Find which healpix this star is in
                healpix = apload.obj2healpix(objid)
                healpixgroup = str(healpix // 1000)
                healpix = str(healpix)

                # Find the associated healpix html directories
                starDir = starHTMLbase + healpixgroup + '/' + healpix + '/'
                starRelPath = '../../../../../stars/' + telescope + '/' + healpixgroup + '/' + healpix + '/'
                starHTMLrelPath = '../' + starRelPath + 'html/' + objid + '.html'
                apStarCheck = glob.glob(starDir + 'apStar-' + apred + '-' + telescope + '-' + objid + '-*.fits')
                if len(apStarCheck) > 0:
                    # Find the newest apStar file
                    apStarCheck.sort()
                    apStarCheck = np.array(apStarCheck)
                    apStarNewest = os.path.basename(apStarCheck[-1])
                    apStarRelPath = '../' + starRelPath + apStarNewest
            else:
                starHTMLrelPath = 'None'

            # Establish html table row background color and spectrum plot color
            color = 'white'
            if (objtype == 'SPECTROPHOTO_STD') | (objtype == 'HOT_STD'): color = '#D2B4DE'
            if objtype == 'SKY': color = '#D6EAF8'

            # Get target flag strings
            if 'apogee' in survey:
                targflagtxt = bitmask.targflags(jdata['TARGET1'],jdata['TARGET2'],jdata['TARGET3'],
                                                jdata['TARGET4'],survey=survey)
            else:
                targflagtxt = bitmask.targflags(jdata['SDSSV_APOGEE_TARGET0'], 0, 0, 0, survey=survey)
                if targflagtxt == '': targflagtxt = 'OPS_STD_BOSS?'
            if targflagtxt[-1:] == ',': targflagtxt = targflagtxt[:-1]
            targflagtxt = targflagtxt.replace(',','<BR>')

            # Find apVisit file
            visitfile = load.filename('Visit', plate=int(plate), mjd=mjd, fiber=fiber, fps=fps)
            visitfilebase = os.path.basename(visitfile)
            vplotfile = visitfile.replace('.fits','.jpg')

            snratio = ''
            starflagtxt = ''
            if os.path.exists(visitfile) == False:
                visitfile = visitfile.replace('-apo25m-', '-')
            if os.path.exists(visitfile):
                visithdr = fits.getheader(visitfile)
                catid = visithdr['CATID']
                pp, = np.where(catid == plug['catalogid'])
                if len(pp) > 0: targflagtxt = plug['firstcarton'][pp][0].decode('UTF-8')
                starflagtxt = bitmask.StarBitMask().getname(visithdr['STARFLAG']).replace(',','<BR>')
                if type(visithdr['SNR']) != str:
                    snratio = str("%.2f" % round(visithdr['SNR'],2))
                else:
                    print("----> makeVisHTML: Problem with " + visitfilebase + "... SNR = NaN.")

            # column 1
            vishtml.write('<TR BGCOLOR=' + color + '><TD>' + cfiber + '<BR>(' + cblock + ')\n')

            # column 2
            vishtml.write('<TD>' + objid + '\n')
            if objtype != 'SKY':
                vishtml.write('<BR>' + simbadlink + '\n')
                vishtml.write('<BR><A HREF=../' + visitfilebase + '>apVisit file</A>\n')
                if apStarRelPath is not None:
                    vishtml.write('<BR><A HREF=' + apStarRelPath + '>apStar file</A>\n')
                else:
                    vishtml.write('<BR>apStar file??\n')
                vishtml.write('<BR><A HREF=' + starHTMLrelPath + ' target="_blank">Star Summary Page</A>\n')

            if objtype != 'SKY':
                vishtml.write('<TD align ="center">' + chmag)
                vishtml.write('<TD align ="center">' + cjkcolor)
                #vishtml.write('<TD BGCOLOR='+color+' align ="right">'+magdiff+'\n')
            else:
                vishtml.write('<TD align="right"><FONT COLOR="red">99.999</FONT>')
                vishtml.write('<TD align="right"><FONT COLOR="red">99.999</FONT>')
                #vishtml.write('<TD BGCOLOR='+color+'>---\n')

            if objtype == 'SKY': 
                vishtml.write('<TD align="center">SKY')
            else:
                if (objtype == 'SPECTROPHOTO_STD') | (objtype == 'HOT_STD'):
                    vishtml.write('<TD align="center">TEL')
                else:
                    vishtml.write('<TD align="center">SCI')

            if objtype == 'SKY': targflagtxt = 'sky'
            vishtml.write('<TD align="left">' + targflagtxt)
            vishtml.write('<BR><BR>' + starflagtxt)

            # Vrad, N_components, RV_TEFF, RV_LOGG, and RV_FEH from allVisitMJD
            if os.path.exists(allVpath):
                gd, = np.where((objid == allV['APOGEE_ID']) & (allV['PLATE'] == plate))
                if len(gd) == 1:
                    try:
                        vrad = allV['VRAD'][gd][0]
                        if type(vrad) != str: vrad = str("%.3f" % round(vrad,3))
                        ncomp = str(allV['N_COMPONENTS'][gd][0])
                        rvteff = allV['RV_TEFF'][gd][0]
                        if type(rvteff) != str: rvteff = str(int(round(rvteff)))
                        rvlogg = allV['RV_LOGG'][gd][0]
                        if type(rvlogg) != str: rvlogg = str("%.3f" % round(rvlogg,3))
                        rvfeh = allV['RV_FEH'][gd][0]
                        if type(rvfeh) != str: rvfeh = str("%.3f" % round(rvfeh,3))
                        vcol = 'black'
                        if np.absolute(float(vrad)) > 400: vcol = 'red'
                    except:
                        vrad = '???'
                        ncomp = '?'
                        rvteff = '???'
                        rvlogg = '???'
                        rvfeh = '???'
                        vcol = 'red'
                    vishtml.write('<TD align ="center">' + snratio)
                    vishtml.write('<TD align ="center"><FONT COLOR="' + vcol + '">' + vrad + '</FONT>')
                    vishtml.write('<TD align ="center"><FONT COLOR="' + vcol + '">' + ncomp + '</FONT>')
                    vishtml.write('<TD align ="center"><FONT COLOR="' + vcol + '">' + rvteff + '</FONT>')
                    vishtml.write('<TD align ="center"><FONT COLOR="' + vcol + '">' + rvlogg + '</FONT>')
                    vishtml.write('<TD align ="center"><FONT COLOR="' + vcol + '">' + rvfeh + '</FONT>')
                else:
                    vishtml.write('<TD align="center"><FONT COLOR="red">-99.9')
                    vishtml.write('<TD align="center"><FONT COLOR="red">-9999')
                    vishtml.write('<TD align="center"><FONT COLOR="red">-1')
                    vishtml.write('<TD align="center"><FONT COLOR="red">-9999')
                    vishtml.write('<TD align="center"><FONT COLOR="red">-9.999')
                    vishtml.write('<TD align="center"><FONT COLOR="red">-9.999')
            else:
                vishtml.write('<TD align="center"><FONT COLOR="red">-99.9')
                vishtml.write('<TD align="center"><FONT COLOR="red">-9999')
                vishtml.write('<TD align="center"><FONT COLOR="red">-1')
                vishtml.write('<TD align="center"><FONT COLOR="red">-9999')
                vishtml.write('<TD align="center"><FONT COLOR="red">-9.999')
                vishtml.write('<TD align="center"><FONT COLOR="red">-9.999')

            # Throughput column
            tput = throughput[j]
            if np.isnan(tput) == False:
                bcolor = 'white'
                if tput < 0.7: bcolor = '#FFFF66'
                if tput < 0.6: bcolor = '#FF9933'
                if tput < 0.5: bcolor = '#FF6633'
                if tput < 0.4: bcolor = '#FF3333'
                if tput < 0.3: bcolor = '#FF0000'
                tput = str("%.3f" % round(tput,3))
                tputdat.write(plate+'   '+mjd+'   '+cfiber+'   '+objid+'   '+tput+'\n')
                vishtml.write('<TD align ="center" BGCOLOR=' + bcolor + '>' + tput + '\n')
            else:
                vishtml.write('<TD align ="center BGCOLOR="white">----\n')

            visitplotfile = '../plots/apPlate-' + plate + '-' + mjd + '-' + cfiber + '.png'
            vishtml.write('<TD><A HREF=' + visitplotfile + ' target="_blank"><IMG SRC=' + visitplotfile + ' WIDTH=1000></A>\n')
    vishtml.close()
    tputdat.close()

    print("----> makeVisHTML: Done with plate " + plate + ", MJD " + mjd + ".\n")
