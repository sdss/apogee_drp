import copy
import numpy as np
import os
import shutil
from glob import glob
import pdb

from dlnpyutils import utils as dln
from ..utils import spectra,yanny,apload,platedata,plan,email
from ..apred import mkcal
from ..database import apogeedb
from . import mkplan
from sdss_access.path import path
from astropy.io import fits
from astropy.table import Table
from collections import OrderedDict
#from astropy.time import Time
from datetime import datetime
import logging
from slurm import queue as pbsqueue
import time


def run(observatory,mjdstart,mjdstop,apred,qos='sdss-fast',fresh=False):
    """ Perform APOGEE Data Release Processing """

    begtime = str(datetime.now())

    telescope = observatory+'25m'
    instrument = {'apo':'apogee-n','lco':'apogee-s'}[observatory]

    nodes = 1
    alloc = 'sdss-np'
    shared = True
    ppn = 64
    cpus = 32
    walltime = '23:00:00'

    # Get software version (git hash)
    gitvers = plan.getgitvers()

    load = apload.ApLoad(apred=apred,telescope=telescope)



    # 1) Setup the directory structure
    #---------------------------------
    
    # 2) Master calibration products, make sure to do them in the right order
    #------------------------------------------------------------------------
    
    # 3) Make all daily cals (domeflats, quartzflats, arclamps, FPI)
    #----------------------------------------------------------------
        
    # 4) Run through all of the plan files (ap3d-ap1dvisit), go through each MJD chronologically
    #--------------------------------------------------------------------------------------------
    
    # 5) Run rv for each unique star
    #---------------------------------
    
    # 6) Unified directory structure
    #---------------------------------

    # 7) QA script
    #-------------

    
    # Daily reduction logs directory
    logdir = os.environ['APOGEE_REDUX']+'/'+apred+'/log/'+observatory+'/'
    if os.path.exists(logdir)==False:
        os.makedirs(logdir)

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
    expinfo = mkplan.getexpinfo(observatory,mjd5)
    nexp = len(expinfo)
    if nexp==0:
        rootLogger.error('No raw APOGEE files found.')
        return        
    rootLogger.info(str(nexp)+' exposures')
    db.ingest('exposure',expinfo)  # insert into database
    expinfo0 = expinfo.copy()
    expinfo = db.query('exposure',where="mjd=%d and observatory='%s'" % (mjd5,observatory))

    # Make MJD5 and plan files
    #--------------------------
    rootLogger.info('Making plan files')
    plandicts,planfiles = mkplan.make_mjd5_yaml(mjd5,apred,telescope,clobber=True,logger=rootLogger)
    dailyplanfile = os.environ['APOGEEREDUCEPLAN_DIR']+'/yaml/'+telescope+'/'+telescope+'_'+str(mjd5)+'auto.yaml'
    planfiles = mkplan.run_mjd5_yaml(dailyplanfile,logger=rootLogger)
    nplanfiles = len(planfiles)
    if nplanfiles>0:
        dbload_plans(planfiles)  # load plans into db
        # Write planfiles to MJD5.plans
        dln.writelines(logdir+str(mjd5)+'.plans',[os.path.basename(pf) for pf in planfiles])
    else:
        dln.writelines(logdir+str(mjd5)+'.plans','')   # write blank file

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


    # Run APRED on all planfiles using "pbs" package
    #------------------------------------------------
    if nplanfiles>0:
        rootLogger.info('')
        rootLogger.info('--------------')
        rootLogger.info('Running APRED')
        rootLogger.info('==============')
        rootLogger.info('')
        queue = pbsqueue(verbose=True)
        queue.create(label='apred', nodes=nodes, alloc=alloc, ppn=ppn, cpus=np.minimum(cpus,len(planfiles)),
                     qos=qos, shared=shared, numpy_num_threads=2, walltime=walltime, notification=False)
        for pf in planfiles:
            queue.append('apred {0}'.format(pf), outfile=pf.replace('.yaml','_pbs.'+logtime+'.log'), errfile=pf.replace('.yaml','_pbs.'+logtime+'.err'))
        queue.commit(hard=True,submit=True)
        rootLogger.info('PBS key is '+queue.key)
        queue_wait(queue,sleeptime=120,verbose=True,logger=rootLogger)  # wait for jobs to complete
        chkexp,chkvisit = check_apred(expinfo,planfiles,queue.key,verbose=True,logger=rootLogger)
        del queue
    else:
        rootLogger.info('No plan files to run')
        chkexp,chkvisit = None,None

    # Run "rv" on all stars
    #----------------------
    rootLogger.info('')
    rootLogger.info('------------------------------')
    rootLogger.info('Running RV+Visit Combination')
    rootLogger.info('==============================')
    rootLogger.info('')
    vcat = db.query('visit',cols='*',where="apred_vers='%s' and mjd=%d and telescope='%s'" % (apred,mjd5,telescope))
    if len(vcat)>0:
        queue = pbsqueue(verbose=True)
        queue.create(label='rv', nodes=nodes, alloc=alloc, ppn=ppn, cpus=cpus, qos=qos, shared=shared, numpy_num_threads=2,
                     walltime=walltime, notification=False)
        # Get unique stars
        objects,ui = np.unique(vcat['apogee_id'],return_index=True)
        vcat = vcat[ui]
        for obj in vcat['apogee_id']:
            apstarfile = load.filename('Star',obj=obj)
            outdir = os.path.dirname(apstarfile)  # make sure the output directories exist
            if os.path.exists(outdir)==False:
                os.makedirs(outdir)
            # Run with --verbose and --clobber set
            queue.append('rv %s %s %s -c -v -m %s' % (obj,apred,telescope,mjd5),outfile=apstarfile.replace('.fits','-'+str(mjd5)+'_pbs.'+logtime+'.log'),
                         errfile=apstarfile.replace('.fits','-'+str(mjd5)+'_pbs.'+logtime+'.err'))
        queue.commit(hard=True,submit=True)
        rootLogger.info('PBS key is '+queue.key)        
        queue_wait(queue,sleeptime=120,verbose=True,logger=rootLogger)  # wait for jobs to complete
        chkrv = check_rv(vcat,queue.key)
        del queue
    else:
        rootLogger.info('No visit files for MJD=%d' % mjd5)
        chkrv = None


    # Create daily and full allVisit/allStar files
    # The QA code needs these
    create_sumfiles(mjd5,apred,telescope)


    # Run QA script
    #--------------
    rootLogger.info('')
    rootLogger.info('------------')
    rootLogger.info('Running QA')
    rootLogger.info('============')
    rootLogger.info('')
    queue = pbsqueue(verbose=True)
    queue.create(label='qa', nodes=1, alloc=alloc, ppn=ppn, qos=qos, cpus=1, shared=shared, walltime=walltime, notification=False)
    qaoutfile = os.environ['APOGEE_REDUX']+'/'+apred+'/log/'+observatory+'/'+str(mjd5)+'-qa.'+logtime+'.log'
    qaerrfile = qaoutfile.replace('-qa.log','-qa.'+logtime+'.err')
    if os.path.exists(os.path.dirname(qaoutfile))==False:
        os.makedirs(os.path.dirname(qaoutfile))
    queue.append('apqa {0} {1}'.format(mjd5,observatory),outfile=qaoutfile, errfile=qaerrfile)
    queue.commit(hard=True,submit=True)
    queue_wait(queue)  # wait for jobs to complete
    del queue

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
    # send to apogee_reduction email list
    # include basic information
    # give link to QA page
    # attach full run_daily() log (as attachment)
    # don't see it if using --debug mode
    summary_email(observatory,mjd5,chkexp,chkvisit,chkrv,logfile)
