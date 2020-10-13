import copy
import numpy as np
import os
import shutil
from glob import glob
import pdb

from dlnpyutils import utils as dln
from ..utils import spectra,yanny,apload,platedata,plan
from ..apred import mkcal
from ..database import apogeedb
from . import mkplan
from sdss_access.path import path
from astropy.io import fits
from collections import OrderedDict
#from astropy.time import Time
from datetime import datetime
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
    dailydir = os.environ['APOGEE_REDUX']+'/'+apred+'/daily/'+observatory+'/'
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

def queue_wait(queue,sleeptime=60,verbose=True,logger=None):
    """ Wait for the pbs queue to finish."""

    if logger is None:
        logger = dln.basiclogger()

    # Wait for jobs to complete
    running = True
    while running:
        time.sleep(sleeptime)
        percent_complete = queue.get_percent_complete()
        if verbose==True:
            logger.info('percent complete = %d' % percent_complete)
        if percent_complete == 100:
            running = False


def check_apred(expinfo,planfiles,pbskey,verbose=False,logger=None):
    """ Check that apred ran okay and load into database."""

    if logger is None:
        logger = dln.basiclogger()

    if verbose==True:
        logger.info('')
        logger.info('--------------------')
        logger.info('Checking APRED runs')
        logger.info('====================')

    # Loop over the planfiles
    nplanfiles = len(planfiles)
    for ip,pfile in enumerate(planfiles):
        planstr = plan.load(pfile,np=True)
        apred_vers = planstr['apred_vers']
        telescope = planstr['telescope']
        instrument = planstr['instrument']
        platetype = planstr['platetype']
        mjd = planstr['mjd']
        plate = planstr['plateid']
        expstr = planstr['APEXP']
        nexp = len(expstr)
        load = apload.ApLoad(apred=apred_vers,telescope=telescope)

        # apred check
        # planfile
        # apred_vers
        # instrument
        # telescope
        # platetype
        # mjd
        # plate
        # nexposures
        # pbs key
        # started timestamp
        # ap3d_nexp_success: number of exposures successfully processed
        # ap3d_success: True or False
        # ap2d_nexp_success: number of exposures successfully processed
        # ap2d_success: True or False
        # ap1dvisit: 
        # ap1dvisit_success: True or False

        # ap3d check
        # planfile
        # apred_vers
        # instrument
        # telescope
        # platetype
        # mjd
        # plate
        # pbs key
        # started timestamp
        # num
        # nreads
        # success: timestamp

        # ap2d check
        # planfile
        # apred_vers
        # instrument
        # telescope
        # platetype
        # mjd
        # plate
        # pbs key
        # started timestamp
        # num
        # nreads
        # success: timestamp

        # Science exposures
        if platetype=='normal':

            # Load the plugmap information
            plugmap = platedata.getdata(plate,mjd,apred_vers,telescope,plugid=planstr['plugmap'])
            fiberdata = plugmap['fiberdata']


            # AP3D
            # ----
            # -ap2D and ap2Dmodel files for all exposures
            dtype3d = np.dtype([('planfile',(np.str,300)),('apred_vers',(np.str,20)),('instrument',(np.str,20)),
                                ('telescope',(np.str,10)),('platetype',(np.str,50)),('mjd',int),('plate',int),
                                ('pbskey',(np.str,50)),('checktime',(np.str,100)),('num',int),('nread',int),('success',bool)])
            chk3d = np.zeros(nexp,dtype=dtype3d)
            chk3d['planfile'] = pfile
            chk3d['apred_vers'] = apred_vers
            chk3d['instrument'] = instrument
            chk3d['telescope'] = telescope
            chk3d['platetype'] = platetype
            chk3d['mjd'] = mjd
            chk3d['plate'] = plate
            chk3d['pbskey'] = pbskey
            chk3d['success'] = False
            for i,num in enumerate(expstr['name']):
                chk3d['num'][i] = num
                ind, = np.where(expinfo['num']==num)
                if len(ind)>0:
                    chk3d['nread'][i] = expinfo['nread'][ind[0]]
                base = load.filename('2D',num=num,mjd=mjd,chips=True)
                chfiles = [base.replace('2D-','2D-'+ch+'-') for ch in ['a','b','c']]
                exists = [os.path.exists(chf) for chf in chfiles]
                chk3d['checktime'][i] = str(datetime.now())
                if np.sum(exists)==3:
                    chk3d['success'][i] = True

            # AP2D
            # ----
            # -ap1D files for all exposures
            dtype2d = np.dtype([('planfile',(np.str,300)),('apred_vers',(np.str,20)),('instrument',(np.str,20)),
                                ('telescope',(np.str,10)),('platetype',(np.str,50)),('mjd',int),('plate',int),
                                ('pbskey',(np.str,50)),('checktime',(np.str,100)),('num',int),('success',bool)])
            chk2d = np.zeros(nexp,dtype=dtype2d)
            chk2d['planfile'] = pfile
            chk2d['apred_vers'] = apred_vers
            chk2d['instrument'] = instrument
            chk2d['telescope'] = telescope
            chk2d['platetype'] = platetype
            chk2d['mjd'] = mjd
            chk2d['plate'] = plate
            chk2d['pbskey'] = pbskey
            chk2d['success'] = False
            for i,num in enumerate(expstr['name']):
                chk2d['num'][i] = num
                base = load.filename('1D',num=num,mjd=mjd,chips=True)
                chfiles = [base.replace('1D-','1D-'+ch+'-') for ch in ['a','b','c']]
                exists = [os.path.exists(chf) for chf in chfiles]
                chk2d['checktime'][i] = str(datetime.now())
                if np.sum(exists)==3:
                    chk2d['success'][i] = True

            # AP1DVISIT
            # ---------
            # -apCframe files for all exposures
            dtypeCf = np.dtype([('planfile',(np.str,300)),('apred_vers',(np.str,20)),('instrument',(np.str,20)),
                                ('telescope',(np.str,10)),('platetype',(np.str,50)),('mjd',int),('plate',int),
                                ('pbskey',(np.str,50)),('checktime',(np.str,100)),('num',int),('success',bool)])
            chkCf = np.zeros(nexp,dtype=dtypeCf)
            chkCf['planfile'] = pfile
            chkCf['apred_vers'] = apred_vers
            chkCf['instrument'] = instrument
            chkCf['telescope'] = telescope
            chkCf['platetype'] = platetype
            chkCf['mjd'] = mjd
            chkCf['plate'] = plate
            chkCf['pbskey'] = pbskey
            chkCf['success'] = False
            for i,num in enumerate(expstr['name']):
                chkCf['num'][i] = num
                base = load.filename('Cframe',num=num,mjd=mjd,chips=True)
                chfiles = [base.replace('Cframe-','Cframe-'+ch+'-') for ch in ['a','b','c']]
                exists = [os.path.exists(chf) for chf in chfiles]
                chkCf['checktime'][i] = str(datetime.now())
                if np.sum(exists)==3:
                    chkCf['success'][i] = True

            # could combine 3d/2d/cframe tables into one

            # -apPlate
            # -apVisit files
            # -apVisitSum file
            dtypeap = np.dtype([('planfile',(np.str,300)),('logfile',(np.str,300)),('errfile',(np.str,300)),
                                ('apred_vers',(np.str,20)),('instrument',(np.str,20)),
                                ('telescope',(np.str,10)),('platetype',(np.str,50)),('mjd',int),('plate',int),
                                ('nobj',int),('pbskey',(np.str,50)),('checktime',(np.str,100)),('ap3d_success',bool),
                                ('ap3d_nexp_success',int),('ap2d_success',bool),('ap2d_nexp_success',int),
                                ('apcframe_success',bool),('apcframe_nexp_success',int),('applate_success',bool),
                                ('apvisit_success',bool),('apvisit_nobj_success',int),
                                ('apvisitsum_success',bool),('success',bool)])
            chkap = np.zeros(1,dtype=dtypeap)
            chkap['planfile'] = pfile
            chkap['logfile'] = pfile.replace('.yaml','_pbs.log')
            chkap['errfile'] = pfile.replace('.yaml','_pbs.err')
            chkap['apred_vers'] = apred_vers
            chkap['instrument'] = instrument
            chkap['telescope'] = telescope
            chkap['platetype'] = platetype
            chkap['mjd'] = mjd
            chkap['plate'] = plate
            chkap['nobj'] = np.sum(fiberdata['objtype']!='SKY')  # stars and tellurics
            chkap['pbskey'] = pbskey
            chkap['checktime'] = str(datetime.now())
            # ap3D, ap2D, apCframe success
            chkap['ap3d_nexp_success'] = np.sum(chk3d['success'])
            chkap['ap3d_success'] = np.sum(chk3d['success'])==nexp
            chkap['ap2d_nexp_success'] = np.sum(chk2d['success'])
            chkap['ap2d_success'] = np.sum(chk2d['success'])==nexp
            chkap['apcframe_nexp_success'] = np.sum(chkCf['success'])
            chkap['apcframe_success'] = np.sum(chkCf['success'])==nexp
            # apPlate
            chkap['applate_success'] = False
            base = load.filename('Plate',plate=plate,mjd=mjd,chips=True)
            chfiles = [base.replace('Plate-','Plate-'+ch+'-') for ch in ['a','b','c']]
            exists = [os.path.exists(chf) for chf in chfiles]
            if np.sum(exists)==3:
                chkap['applate_success'] = True
            # apVisit
            base = load.filename('Visit',plate=plate,mjd=mjd,fiber=1) 
            visitfiles = glob(base.replace('-001.fits','-???.fits'))
            nvisitfiles = len(visitfiles)
            chkap['apvisit_nobj_success']  = nvisitfiles
            chkap['apvisit_success'] = nvisitfiles==chkap['nobj']
            apvisitsumfile = load.filename('VisitSum',plate=plate,mjd=mjd)
            chkap['apvisitsum_success'] = os.path.exists(apvisitsumfile)
            chkap['success'] = chkap['ap3d_success'][0] and chkap['ap2d_success'][0] and chkap['apcframe_success'][0] and \
                               chkap['applate_success'][0] and chkap['apvisit_success'][0] and chkap['apvisitsum_success'][0]

            if verbose:
                logger.info('')
                logger.info('%d/%d' % (ip+1,nplanfiles))
                logger.info('planfile: '+pfile)
                logger.info('log/errfile: '+os.path.basename(chkap['logfile'][0])+', '+os.path.basename(chkap['errfile'][0]))
                logger.info('platetype: %s' % platetype)
                logger.info('mjd: %d' % mjd)
                logger.info('plate: %d' % plate)
                logger.info('nexp: %d' % nexp)
                logger.info('Nobj: %d' % chkap['nobj'][0])
                logger.info('3D/2D/Cframe:')
                logger.info('Num    EXPID   NREAD  3D    2D   Cframe')
                for i,num in enumerate(expstr['name']):
                    logger.info('%2d %10d %4d %6s %6s %6s' % (i+1,chk3d['num'][i],chk3d['nread'][i],
                                                                 chk3d['success'][i],chk2d['success'][i],chkCf['success'][i]))

                logger.info('apPlate files: %s ' % chkap['applate_success'][0])
                logger.info('N apVisit files: %d ' % chkap['apvisit_nobj_success'][0])
                logger.info('apVisitSum file: %s ' % chkap['apvisitsum_success'][0])


            import pdb; pdb.set_trace()
                
            # Load into the database
            db = apogeedb.DBSession()
            db.load('ap3d',chk3d)
            db.load('ap2d',chk2d)
            db.load('apcframe',chkCf)
            db.load('apred',chkap)
            db.close()


            import pdb; pdb.set_trace()

        # Calibration exposures
        else:
            logger.info('calibration exposures')

    import pdb; pdb.set_trace()

        

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

    # Initialize the DB connection
    db = apogeedb.DBSession()

    # Check that daily data transfer completed
    datadir = {'apo':os.environ['APOGEE_DATA_N'],'lco':os.environ['APOGEE_DATA_S']}[observatory]
    datadir += '/'+str(mjd5)+'/'
    donefile = datadir+str(mjd5)+'.md5sum'
    if os.path.exists(donefile)==False:
        rootLogger.error('Data has not finished transferring yet')
        return

    # Get exposure information and load into database
    rootLogger.info('Getting exposure information')
    expinfo = mkplan.getexpinfo(observatory,mjd5)
    nexp = len(expinfo)
    if nexp==0:
        rootLogger.error('No raw APOGEE files found.')
        return        
    rootLogger.info(str(nexp)+' exposures')
    db.load('exposure',expinfo)  # load into database

    # Make MJD5 and plan files
    rootLogger.info('Making plan files')
    plandicts,planfiles = mkplan.make_mjd5_yaml(mjd5,apred,telescope,clobber=True,logger=rootLogger)
    #db.load('plan',planfiles)  # load plans into db
    dailyplanfile = os.environ['APOGEEREDUCEPLAN_DIR']+'/yaml/'+telescope+'/'+telescope+'_'+str(mjd5)+'auto.yaml'
    planfiles = mkplan.run_mjd5_yaml(dailyplanfile,logger=rootLogger)
    # Write planfiles to MJD5.plans
    dln.writelines(dailydir+str(mjd5)+'.plans',[os.path.basename(pf) for pf in planfiles])

    check_apred(expinfo,planfiles,'dummy',verbose=True,logger=rootLogger)

    import pdb;pdb.set_trace()

    # Use "pbs" packages to run "apred" on all visits
    queue = pbsqueue(verbose=True)
    cpus = np.minimum(len(planfiles),30)
    queue.create(label='apred', nodes=2, ppn=16, cpus=cpus, alloc='sdss-kp', qos=True, umask='002', walltime='240:00:00')
    for pf in planfiles:
        queue.append('apred {0}'.format(pf), outfile=pf.replace('.yaml','_pbs.log'), errfile=pf.replace('.yaml','_pbs.err'))
    import pdb;pdb.set_trace()
    queue.commit(hard=True,submit=True)
    queue_wait(queue,sleeptime=120,verbose=True,logger=rootLogger)  # wait for jobs to complete
    check_apred(expinfo,planfiles,queue.key)
    del queue

    import pdb;pdb.set_trace()

    # Run "rv" on all stars
    queue = pbsqueue(verbose=True)
    vcat = db.query('SELECT * from apogee_drp.visit where MJD=%d' % MJD5)
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
