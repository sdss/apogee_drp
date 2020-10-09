import copy
import numpy as np
import os
import shutil
from glob import glob
import pdb
import subprocess
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from dlnpyutils import utils as dln
from ..utils import spectra,yanny,apload,platedata
from ..apred import mkcal
from . import mkplan
from . import apogeedb
from sdss_access.path import path
from astropy.io import fits
from collections import OrderedDict
from astropy.time import Time
import logging
from pbs import queue as pbsqueue
import time

def lastnightmjd5():
    """ Compute last night's MJD."""
    tnow = Time.now()
    mjdnow = tnow.mjd
    # The Julian day starts at NOON, while MJD starts at midnight
    # For SDSS MJD we add 0.3 days
    mjdnow += 0.3
    # Truncate for MJD5 number
    mjd5now = int(mjdnow)
    # Subtract one for yesterday
    mjd5 = mjd5now-1
    return mjd5


def nextmjd5(observatory,apred='t14'):
    """ Figure out the next MJD to process."""

    # Check MJD5.done in $APOGEE_REDUX/apred/daily/
    dailydir = os.environ['APOGEE_REDUX']+'/'+apred+'/daily/'+observatory+''/'
    if os.path.exists(dailydir)==True:
        donefiles = glob(dailydir+'?????.done')
        ndonefiles = len(donefiles)
        if ndonefiles>0:
            mjd5list = [os.path.splitext(os.path.basename(df))[0] for df in donefiles]
            maxmjd5 = np.max(np.array(mjd5list))
            return maxmjd5+1
    # Cannot use MJD5.done files, compute last night's MJD5
    return lastnightmjd5()


def getNextMJD(observatory,apred='t14'):
        ''' Returns the next MJD to reduce.  Either a list or one. '''

        # Grab the MJD from the currentmjd file
        nextfile = os.path.join(os.getenv('APOGEE_REDUX'), apred, 'daily', observatory, 'currentmjd')
        f = open(nextfile, 'r')
        mjd = f.read()
        f.close()

        # Increment MJD from file by 1
        nextmjd = str(int(mjd) + 1)

        # Check if multiple new MJDs
        datadir = {'apo':os.getenv['APOGEE_DATA_N'],'lco':os.getenv['APOGEE_DATA_S']}[observatory]
        mjdlist = os.listdir(datadir)
        newmjds = [mjd for mjd in mjdlist if mjd >= nextmjd and mjd.isdigit()]

        # Set next mjd or list of next mjds
        finalmjd = [nextmjd] if newmjds == [] else newmjds

        # Filter out list based on any MJD range
        if self.mjdrange:
            start, end = self.mjdrange.split('-')
            finalmjd = [m for m in finalmjd if m >= start and m <= end]

        return finalmjd

def queue_wait(queue,sleeptime=60):
    """ Wait for the pbs queue to finish."""

    # Wait for jobs to complete
    running = True
    while running:
        time.sleep(sleeptime)
        percent_complete = queue.get_percent_complete()
        if percent_complete == 100:
            running = False


def run_daily(observatory,mjd5=None,apred='t14'):
    """ Perform daily APOGEE data reduction."""

    telescope = observatory+'25m'
    instrument = {'apo':'apogee-n','lco':'apogee-s'}[observatory]

    # Daily reduction directory
    dailydir = os.environ['APOGEE_REDUX']+'/'+apred+'/daily/'+observatory+'/'
    if os.path.exists(dailydir)==False:
        os.makedirs(dailydir)

    # What MJD5 are we doing?
    if mjd5 is None:
        # Could get information on which MJDs were processed from database
        # or from $APOGEE_REDUX/apred/daily/MJD5.done
        mjd5 = nextmjd5()

    # Set up logging to screen and logfile
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger() 
    while rootLogger.hasHandlers(): # some existing loggers, remove them   
        rootLogger.removeHandler(rootLogger.handlers[0]) 
    rootLogger = logging.getLogger()
    logfile = dailydir+str(mjd5)+'.log'
    if os.path.exists(logfile): os.remove(logfile)
    fileHandler = logging.FileHandler(logfile)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.NOTSET)

    rootLogger.info('Running daily APOGEE data reduction for '+str(observatory).upper()+' '+str(mjd5))

    # Check that daily data transfer completed
    datadir = {'apo':os.environ['APOGEE_DATA_N'],'lco':os.environ['APOGEE_DATA_S']}[observatory]
    datadir += '/'+str(mjd5)+'/'
    donefile = datadir+str(mjd5)+'.md5sum'
    if os.path.exists(donefile)==False:
        rootLogger.error('Data has not finished transferring yet')
        return

    # Get exposure information and load into database
    rootLogger.info('Getting exposure information')
    files = glob(datadir+'/??R-c-????????.apz')
    if len(files)==0:
        rootLogger.error('No raw APOGEE files found.')
        return        
    files = np.array(files)
    nfiles = len(files)
    files = files[np.argsort(files)]  # sort
    rootLogger.info(str(nfiles)+' exposures')
    expinfo = mkplan.getexpinfo(files)
    #apogeedb.write('exposure',expinfo)

    # Make MJD5 and plan files
    rootLogger.info('Making plan files')
    plandicts,planfiles = mkplan.make_mjd5_yaml(mjd5,apred,telescope,clobber=True,logger=rootLogger)
    #apogeedb.write('plans',planfiles)  # load plans into db
    dailyplanfile = os.environ['APOGEEREDUCEPLAN_DIR']+'/yaml/'+telescope+'/'+telescope+'_'+str(mjd5)+'auto.yaml'
    planfiles = mkplan.run_mjd5_yaml(dailyplanfile,logger=rootLogger)
    # Write planfiles to MJD5.plans
    dln.writelines(dailydir+str(mjd5)+'.plans',[os.path.basename(pf) for pf in planfiles])

    import pdb;pdb.set_trace()

    # Use "pbs" packages to run "apred" on all visits
    queue = pbsqueue(verbose=True)
    queue.create(label='apred', nodes=2, ppn=16, cpus=15, alloc='sdss-kp', qos=True, umask='002', walltime='240:00:00')
    for pf in planfiles:
        queue.append('apred {0}'.format(pf), outfile=pf.replace('yaml','.log'), errfile=pf.replace('.yaml','.err'))
    import pdb;pdb.set_trace()
    queue.commit(hard=True,submit=True)
    queue_wait(queue)  # wait for jobs to complete
    del queue

    import pdb;pdb.set_trace()

    # Run "rv" on all stars
    queue = pbsqueue(verbose=True)
    vcat = apogeedb.query('SELECT * from apogee_drp.visit where MJD=%d' % MJD5)
    queue.create(label='rv', nodes=2, ppn=16, cpus=15, alloc='sdss-kp', qos=True, umask='002', walltime='240:00:00')
    for obj in vcat['APOGEE_ID']:
        queue.append('rvstar %s %s %s %s' % (obj,apred,instrument,field))
    queue.commit(hard=True,submit=True)
    queue_wait(queue)  # wait for jobs to complete
    del queue

    import pdb;pdb.set_trace()

    # Run QA script
    queue = pbsqueue(verbose=True)
    queue.create(label='qa', nodes=1, ppn=16, cpus=1, alloc='sdss-kp', qos=True, umask='002', walltime='240:00:00')
    queue.append('apqa {0}'.format(mjd5))
    queue.commit(hard=True,submit=True)
    queue_wait(queue)  # wait for jobs to complete
    del queue

    import pdb;pdb.set_trace()
