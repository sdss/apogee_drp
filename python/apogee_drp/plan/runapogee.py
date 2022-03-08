import copy
import numpy as np
import os
import shutil
from glob import glob
import pdb

from dlnpyutils import utils as dln
from ..utils import spectra,yanny,apload,platedata,plan,email,info
from ..apred import mkcal
from ..database import apogeedb
from . import mkplan, check, apogeedrp
from sdss_access.path import path
from astropy.io import fits
from astropy.table import Table
from collections import OrderedDict
#from astropy.time import Time
from datetime import datetime
import logging
import slurm
from slurm import queue as pbsqueue
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
    dailydir = os.environ['APOGEE_REDUX']+'/'+apred+'/log/'+observatory+'/'
    if os.path.exists(dailydir)==True:
        donefiles = glob(dailydir+'?????.done')
        ndonefiles = len(donefiles)
        if ndonefiles>0:
            mjd5list = [os.path.splitext(os.path.basename(df))[0] for df in donefiles]
            maxmjd5 = np.max(np.array(mjd5list))
            return maxmjd5+1
    # Cannot use MJD5.done files, compute last night's MJD5
    return lastnightmjd5()


def getNextMJD(observatory,apred='daily',mjdrange=None):
    ''' Returns the next MJD to reduce.  Either a list or one. '''

    # Grab the MJD from the currentmjd file
    nextfile = os.path.join(os.getenv('APOGEE_REDUX'), apred, 'log', observatory, 'currentmjd')
    f = open(nextfile, 'r')
    mjd = f.read()
    f.close()

    # Increment MJD from file by 1
    nextmjd = str(int(mjd) + 1)

    # Check if multiple new MJDs
    datadir = {'apo':os.environ['APOGEE_DATA_N'],'lco':os.environ['APOGEE_DATA_S']}[observatory]
    mjdlist = os.listdir(datadir)
    newmjds = [mjd for mjd in mjdlist if mjd >= nextmjd and mjd.isdigit()]

    # Set next mjd or list of next mjds
    finalmjd = [nextmjd] if newmjds == [] else newmjds

    # Filter out list based on any MJD range
    if mjdrange is not None:
        start, end = mjdrange.split('-')
        finalmjd = [m for m in finalmjd if m >= start and m <= end]

    return finalmjd


def writeNewMJD(observatory,mjd,apred='daily'):
    ''' Write the new MJD into the currentMJD file. '''

    # write the currentMJD file
    nextfile = os.path.join(os.getenv('APOGEE_REDUX'), apred, 'log', observatory, 'currentmjd')
    f = open(nextfile, 'w')
    f.write(str(mjd))
    f.close()


def summary_email(observatory,mjd5,chkcal,chkexp,chkvisit,chkrv,logfiles=None,debug=False):
    """ Send a summary email."""

    if debug:
        address = 'dnidever@montana.edu'
    else:
        address = 'apogee-pipeline-log@sdss.org'

    subject = 'Daily APOGEE Reduction %s %s' % (observatory,mjd5)
    message = """\
              <html>
                <body>
              """
    message += '<b>Daily APOGEE Reduction %s %s</b><br>\n' % (observatory,mjd5)
    message += '<p>\n'
    message += '<a href="https://data.sdss.org/sas/sdss5/mwm/apogee/spectro/redux/daily/qa/mjd.html">QA Webpage (MJD List)</a><br> \n'

    # Calibration status
    if chkcal is not None:
        indcal, = np.where(chkcal['success']==True)
        message += '%d/%d calibrations successfully processed<br> \n' % (len(indcal),len(chkcal))
    else:
        message += 'No exposures<br> \n'
    # Exposure status
    if chkexp is not None:
        indexp, = np.where(chkexp['success']==True)
        message += '%d/%d exposures successfully processed<br> \n' % (len(indexp),len(chkexp))
    else:
        message += 'No exposures<br> \n'
    # Visit status
    if chkvisit is not None:
        indvisit, = np.where(chkvisit['success']==True)
        message += '%d/%d visits successfully processed<br> \n' % (len(indvisit),len(chkvisit))
        for i in range(len(chkvisit)):
            message += chkvisit['planfile'][i]+'<br> \n'
    else:
        message += 'No visits<br> \n'

    # RV status
    if chkrv is not None:
        indrv, = np.where(chkrv['success']==True)
        message += '%d/%d RV+visit combination successfully processed<br> \n' % (len(indrv),len(chkrv))
    else:
        message += 'No RVs<br> \n'

    message += """\
                 </p>
                 </body>
               </html>
               """

    # Send the message
    email.send(address,subject,message,files=logfiles)


def run_daily(observatory,mjd5=None,apred=None,qos='sdss-fast',clobber=False,debug=False):
    """
    Perform daily APOGEE data reduction.

    Parameters
    ----------
    observatory : str
       Observatory code, 'apo' or 'lco'.
    mjd5 : int, optional
       MJD number of the night of data to reduce.  Default is to process the next night
          of data.
    apred : str, optional
       APOGEE reduction version.  Default is 'daily'.
    qos : str, optional
       The type of slurm queue to use.  Default is "sdss-fast".
    clobber : boolean, optional
       Overwrite any existing files.
    debug : boolean, optional
       For debugging/testing purposes. No email will be sent to the pipeline list.

    Returns
    -------
    Data are processed and output files are created on disk.

    Example
    -------

    run_daily('apo',59640)


    """

    begtime = str(datetime.now())

    telescope = observatory+'25m'
    instrument = {'apo':'apogee-n','lco':'apogee-s'}[observatory]

    nodes = 1
    #alloc = 'sdss-kp'
    alloc = 'sdss-np'
    shared = True
    ppn = 64
    cpus = 32
    walltime = '23:00:00'
    chips = ['a','b','c']

    slurm = {'nodes':nodes, 'alloc':alloc, 'ppn':ppn, 'cpus':cpus, 'qos':qos, 'shared':shared,
             'numpy_num_threads':2,'walltime':walltime,'notification':False}

    # No version input, use 'daily'
    if apred is None:
        apred = 'daily'
    # Get software version (git hash)
    gitvers = plan.getgitvers()

    load = apload.ApLoad(apred=apred,telescope=telescope)

    # Daily reduction logs directory
    logdir = os.environ['APOGEE_REDUX']+'/'+apred+'/log/'+observatory+'/'
    if os.path.exists(logdir)==False:
        os.makedirs(logdir)

    # What MJD5 are we doing?
    updatemjdfile = False
    if mjd5 is None:
        # Could get information on which MJDs were processed from database
        # or from $APOGEE_REDUX/daily/log/apo/MJD5.done
        mjd5 = getNextMJD(observatory)
        if len(mjd5)==0:
            print('No more MJDs to reduce')
            return
        else:
            mjd5 = int(mjd5[0])
        updatemjdfile = True

    # SDSS-V FPS
    fps = False
    if int(mjd5)>=59556:
        fps = True

    # Make sure the data is there
    mjddatadir = {'apo':os.environ['APOGEE_DATA_N'],'lco':os.environ['APOGEE_DATA_S']}[observatory] + '/'+str(mjd5)
    if os.path.exists(mjddatadir):
        #mjdfiles = glob(mjddatadir+'/*.apz')
        mjdfiles = os.listdir(mjddatadir)
    else:
        mjdfiles = []
    if len(mjdfiles)==0:
        print('No data for MJD5='+str(mjd5))
        return

    # Update the currentmjd file
    if updatemjdfile is True:
        writeNewMJD(observatory,mjd5,apred=apred)


    # Set up logging to screen and logfile
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger() 
    while rootLogger.hasHandlers(): # some existing loggers, remove them   
        rootLogger.removeHandler(rootLogger.handlers[0]) 
    rootLogger = logging.getLogger()
    logfile = logdir+str(mjd5)+'.log'
    if os.path.exists(logfile): os.remove(logfile)
    fileHandler = logging.FileHandler(logfile)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.NOTSET)
    logtime = datetime.now().strftime("%Y%m%d%H%M%S") 

    rootLogger.info('Running daily APOGEE data reduction for '+str(observatory).upper()+' '+str(mjd5)+' '+apred)

    # Initialize the DB connection
    db = apogeedb.DBSession()

    # Check that daily data transfer completed
    datadir = {'apo':os.environ['APOGEE_DATA_N'],'lco':os.environ['APOGEE_DATA_S']}[observatory]
    datadir += '/'+str(mjd5)+'/'
    if os.path.exists(datadir)==False:
        rootLogger.error('Data has not finished transferring yet')
        return

    # Get exposure information and load into database
    rootLogger.info('Getting exposure information')
    expinfo = info.expinfo(observatory=observatory,mjd5=mjd5)
    #expinfo = mkplan.getexpinfo(observatory,mjd5)
    nexp = len(expinfo)
    if nexp==0:
        rootLogger.error('No raw APOGEE files found.')
        return        
    rootLogger.info(str(nexp)+' exposures')
    db.ingest('exposure',expinfo)  # insert into database
    expinfo0 = expinfo.copy()
    expinfo = db.query('exposure',where="mjd=%d and observatory='%s'" % (mjd5,observatory))
    si = np.argsort(expinfo['num'])
    expinfo = expinfo[si]


    # Process all exposures through ap3D first
    #-----------------------------------------
    if len(expinfo)>0:
        rootLogger.info('')
        rootLogger.info('--------------------------------')
        rootLogger.info('1) Running AP3D on all exposures')
        rootLogger.info('================================')
        chk3d = apogeedrp.run3d(load,[mjd5],slurm,clobber=clobber,logger=rootLogger)
    else:
        rootLogger.info('No exposures to process with AP3D')

    # Do QA check of the files
    rootLogger.info(' ')
    rootLogger.info('Doing quality checks on all exposures')
    qachk = check.check(expinfo['num'],apred,telescope,verbose=True,logger=rootLogger)
    rootLogger.info(' ')


    # Run daily calibration files
    #----------------------------
    # First we need to run domeflats and quartzflats so there are apPSF files
    # Then the arclamps
    # Then the FPI exposures last (needs apPSF and apWave files)
    # Only use calibration exposures that have passed the quality assurance checks
    calind, = np.where(((expinfo['exptype']=='DOMEFLAT') | (expinfo['exptype']=='QUARTZFLAT') | 
                        (expinfo['exptype']=='ARCLAMP') | (expinfo['exptype']=='FPI')) &
                       (qachk['okay']==True))
    if len(calind)>0:
        rootLogger.info('')
        rootLogger.info('----------------------------------------')
        rootLogger.info('2) Generating daily calibration products')
        rootLogger.info('========================================')
        chkcal = apogeedrp.rundailycals(load,[mjd5],slurm,clobber=clobber,logger=rootLogger)
    else:
        rootLogger.info('No calibration files to run')


    # Make MJD5 and plan files
    #--------------------------
    # Check that the necessary daily calibration files exist
    rootLogger.info(' ')
    rootLogger.info('--------------------')
    rootLogger.info('3) Making plan files')
    rootLogger.info('====================')
    planfiles = apogeedrp.makeplanfiles(load,[mjd5],slurm,clobber=clobber,logger=rootLogger)
    # Start entry in daily_status table
    daycat = np.zeros(1,dtype=np.dtype([('mjd',int),('telescope',(np.str,10)),('nplanfiles',int),
                                        ('nexposures',int),('begtime',(np.str,50)),('success',bool)]))
    daycat['mjd'] = mjd5
    daycat['telescope'] = telescope
    daycat['nplanfiles'] = len(planfiles)
    daycat['nexposures'] = len(expinfo)
    daycat['begtime'] = begtime
    daycat['success'] = False
    db.ingest('daily_status',daycat)


    # Run APRED on all planfiles
    #---------------------------
    if nplanfiles>0:
        rootLogger.info('')
        rootLogger.info('----------------')
        rootLogger.info('4) Running APRED')
        rootLogger.info('================')
        chkexp,chkvisit = apogeedrp.runapred(load,[mjd5],slurm,clobber=clobber,logger=rootLogger)
    else:
        rootLogger.info('No plan files to run')
        chkexp,chkvisit = None,None

    # Run RV + combination on all stars
    #----------------------------------
    rootLogger.info('')
    rootLogger.info('--------------------------------')
    rootLogger.info('5) Running RV+Visit Combination')
    rootLogger.info('================================')
    chkrv = apogeedrp.runrv(load,[mjd5],slurm,daily=True,clobber=clobber,logger=rootLogger)


    # Create daily and full allVisit/allStar files
    # The QA code needs these
    rootLogger.info('')
    rootLogger.info('-----------------------')
    rootLogger.info('6) Create summary files')
    rootLogger.info('=======================')
    apogeedrp.create_sumfiles(apred,telescope,mjd5)


    # 7) Unified directory structure
    #-------------------------------
    #rootLogger.info('')
    #rootLogger.info('---------------------------------------------')
    #rootLogger.info('7) Generating unified MWM directory structure')
    #rootLogger.info('=============================================')
    #apogeedrp.rununified(load[,mjd5],**kws) 

    # 8) Run QA script
    #------------------    
    rootLogger.info('')
    rootLogger.info('--------------')
    rootLogger.info('8) Running QA')
    rootLogger.info('==============')
    apogeedrp.runqa(load,[mjd5],slurm,clobber=clobber,logger=rootLogger)

    # Update daily_status table
    daycat = np.zeros(1,dtype=np.dtype([('pk',int),('mjd',int),('telescope',(np.str,10)),('nplanfiles',int),
                                        ('nexposures',int),('begtime',(np.str,50)),('endtime',(np.str,50)),('success',bool)]))
    daycat['mjd'] = mjd5
    daycat['telescope'] = telescope
    daycat['nplanfiles'] = len(planfiles)
    daycat['nexposures'] = len(expinfo)
    daycat['begtime'] = begtime
    daycat['endtime'] = str(datetime.now())
    daysuccess = True
    if chkvisit is not None:
        daysuccess &= (np.sum(chkvisit['success']==False)==0)
    if chkrv is not None:
        daysuccess &= (np.sum(chkrv['success']==False)==0)
    daycat['success'] = daysuccess
    dayout = db.query('daily_status',where="mjd="+str(mjd5)+" and telescope='"+telescope+"' and begtime='"+begtime+"'")
    daycat['pk'] = dayout['pk'][0]
    db.update('daily_status',daycat)

    rootLogger.info('Daily APOGEE reduction finished for MJD=%d and observatory=%s' % (mjd5,observatory))

    db.close()    # close db session

    # Summary email
    summary_email(observatory,mjd5,chkcal,chkexp,chkvisit,chkrv,logfile,debug=debug)
