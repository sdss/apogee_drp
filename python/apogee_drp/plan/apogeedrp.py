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


def run(observatory,mjdstart,mjdstop,apred,qos='sdss',fresh=False,links=None):
    """
    Perform APOGEE Data Release Processing

    Parameters
    ----------
    observatory : str

    mjdstart : int
       Starting MJD date of the reduction.
    mjdstop : int
       Ending MJD date of the reduction.
    apred : str
       Reduction version name.
    qoa : str, optional
       The pbs queue to use.  Default is sdss.
    fresh : boolean, optional
       Start the reduction directory fresh.  The default is continue with what is
         already there.
    links : str, optional
       Name of reduction version to use for symlinks for the calibration files.

    Returns
    -------

    Example
    -------

    run('apo',54566,56666)

    """

    begtime = str(datetime.now())

    telescope = observatory+'25m'
    instrument = {'apo':'apogee-n','lco':'apogee-s'}[observatory]

    nodes = 1
    alloc = 'sdss-np'
    shared = True
    ppn = 64
    cpus = 32
    walltime = '10-00:00:00'

    # Get software version (git hash)
    gitvers = plan.getgitvers()

    load = apload.ApLoad(apred=apred,telescope=telescope)

    # Reduction logs directory
    logdir = os.environ['APOGEE_REDUX']+'/'+apred+'/log/'+observatory+'/'
    if os.path.exists(logdir)==False:
        os.makedirs(logdir)

    # Make sure the data is there
    datadir = {'apo':os.environ['APOGEE_DATA_N'],'lco':os.environ['APOGEE_DATA_S']}[observatory]

    # Set up logging to screen and logfile
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger() 
    while rootLogger.hasHandlers(): # some existing loggers, remove them   
        rootLogger.removeHandler(rootLogger.handlers[0]) 
    rootLogger = logging.getLogger()
    logfile = logdir+str(mjdstart)+'-'+str(mjdstop)+'.log'
    if os.path.exists(logfile): os.remove(logfile)
    fileHandler = logging.FileHandler(logfile)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.NOTSET)
    logtime = datetime.now().strftime("%Y%m%d%H%M%S") 

    rootLogger.info('Running APOGEE DRP for '+str(observatory).upper()+' '+str(mjdstart)+' - '+str(mjdstop)+' apred='+apred)

    # Initialize the DB connection
    db = apogeedb.DBSession()

    # Get information for all of the exposures
    expinfo = db.query('exposure',where="mjd>=%d and mjd<=%d and observatory='%s'" % (mjdstart,mjdstop,observatory))
    nexp = len(expinfo)
    rootLogger.info(str(nexp)+' exposures')

    # 1) Setup the directory structure
    #--------------------------------
    rootLogger.info('')
    rootLogger.info('-------------------------------------')
    rootLogger.info('1) Setting up the directory structure')
    rootLogger.info('=====================================')
    rootLogger.info('')
    queue = pbsqueue(verbose=True)
    queue.create(label='mkvers', nodes=1, alloc=alloc, ppn=ppn, qos=qos, cpus=1, shared=shared, walltime=walltime, notification=False)
    mkvoutfile = os.environ['APOGEE_REDUX']+'/'+apred+'/log/mkvers.'+logtime+'.log'
    mkverrfile = mkvoutfile.replace('-mkvers.log','-mkvers.'+logtime+'.err')
    if os.path.exists(os.path.dirname(mkvoutfile))==False:
        os.makedirs(os.path.dirname(mkvoutfile))
    cmd = 'mkvers {0}'.format(apred)
    if fresh:
        cmd += ' --fresh'
    if links is not None:
        cmd += ' --links '+str(links)
    queue.append(cmd,outfile=mkvoutfile, errfile=mkverrfile)
    queue.commit(hard=True,submit=True)
    queue_wait(queue)  # wait for jobs to complete
    del queue    

    # 2) Master calibration products, make sure to do them in the right order
    #------------------------------------------------------------------------
    rootLogger.info('')
    rootLogger.info('-----------------------------------------')
    rootLogger.info('2) Generating master calibration products')
    rootLogger.info('=========================================')
    rootLogger.info('')
    # Detector
    # Dark
    # Flat
    # BPM
    # Sparse
    # LSF
    # Maybe have an option to copy/symlink them from a previous apred version


    # 3) Make all daily cals (domeflats, quartzflats, arclamps, FPI)
    #----------------------------------------------------------------
    rootLogger.info('')
    rootLogger.info('----------------------------------------')
    rootLogger.info('3) Generating daily calibration products')
    rootLogger.info('========================================')
    rootLogger.info('')
    # First we need to run domeflats and quartzflats so there are apPSF files
    # Then the arclamps
    # Then the FPI exposures last (needs apPSF and apWave files)
    calind, = np.where((expinfo['exptype']=='DOMEFLAT') | (expinfo['exptype']=='QUARTZFLAT') | 
                       (expinfo['exptype']=='ARCLAMP') | (expinfo['exptype']=='FPI'))
    if len(calind)>0:
        calcodedict = {'DOMEFLAT':0, 'QUARTZFLAT':0, 'ARCLAMP':1, 'FPI':2}
        calcode = [calcodedict[etype] for etype in expinfo['exptype'][calind]]
        calnames = ['DOMEFLAT/QUARTZFLAT','ARCLAMP','FPI']
        shcalnames = ['psf','arcs','fpi']
        chkcal = []
        for j,ccode in enumerate([0,1,2]):
            rootLogger.info('')
            rootLogger.info('----------------------------------------')
            rootLogger.info('Running Calibration Files: '+str(calnames[j]))
            rootLogger.info('========================================')
            rootLogger.info('')
            cind, = np.where(np.array(calcode)==ccode)
            if len(cind)>0:
                rootLogger.info(str(len(cind))+' files to run')
                queue = pbsqueue(verbose=True)
                queue.create(label='makecal-'+shcalnames[j], nodes=nodes, alloc=alloc, ppn=ppn, cpus=np.minimum(cpus,len(cind)),
                             qos=qos, shared=shared, numpy_num_threads=2, walltime=walltime, notification=False)
                calplandir = os.path.dirname(load.filename('CalPlan',num=0,mjd=mjd5))
                logfiles = []
                for k in range(len(cind)):
                    num1 = expinfo['num'][calind[cind[k]]]
                    exptype1 = expinfo['exptype'][calind[cind[k]]]
                    rootLogger.info('Calibration file %d : %s %d' % (k+1,exptype1,num1))
                    if exptype1=='DOMEFLAT' or exptype1=='QUARTZFLAT':
                        cmd1 = 'makecal --psf '+str(num1)+' --unlock'
                        if clobber: cmd1 += ' --clobber'
                        logfile1 = calplandir+'/apPSF-'+str(num1)+'_pbs.'+logtime+'.log'
                    if exptype1=='ARCLAMP':
                        cmd1 = 'makecal --wave '+str(num1)+' --unlock'
                        if clobber: cmd1 += ' --clobber'
                        logfile1 = calplandir+'/apWave-'+str(num1)+'_pbs.'+logtime+'.log'
                    if exptype1=='FPI':
                        cmd1 = 'makecal --fpi '+str(num1)+' --unlock'
                        if clobber: cmd1 += ' --clobber'
                        logfile1 = calplandir+'/apFPI-'+str(num1)+'_pbs.'+logtime+'.log'
                    rootLogger.info(logfile1)
                    logfiles.append(logfile1)
                    queue.append(cmd1, outfile=logfile1,errfile=logfile1.replace('.log','.err'))
                queue.commit(hard=True,submit=True)
                rootLogger.info('PBS key is '+queue.key)
                queue_wait(queue,sleeptime=60,verbose=True,logger=rootLogger)  # wait for jobs to complete
                calinfo = expinfo[calind[cind]]
                chkcal1 = check_calib(calinfo,logfiles,queue.key,apred,verbose=True,logger=rootLogger)
                if len(chkcal)==0:
                    chkcal = chkcal1
                else:
                    chkcal = np.hstack((chkcal,chkcal1))
                del queue
            else:
                rootLogger.info('No '+str(calnames[j])+' calibration files to run')
             

    # 4) Run APRED on all of the plan files (ap3d-ap1dvisit), go through each MJD chronologically
    #--------------------------------------------------------------------------------------------
    if nplanfiles>0:
        rootLogger.info('')
        rootLogger.info('----------------')
        rootLogger.info('4) Running APRED')
        rootLogger.info('================')
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

        
    # 5) Run "rv" on all unique stars
    #--------------------------------
    rootLogger.info('')
    rootLogger.info('--------------------------------')
    rootLogger.info('5) Running RV+Visit Combination')
    rootLogger.info('================================')
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


    
    # 6) Unified directory structure
    #---------------------------------
    rootLogger.info('')
    rootLogger.info('---------------------------------------------')
    rootLogger.info('6) Generating unified MWM directory structure')
    rootLogger.info('=============================================')
    rootLogger.info('')
    queue = pbsqueue(verbose=True)
    queue.create(label='unidir', nodes=nodes, alloc=alloc, ppn=ppn, cpus=cpus, qos=qos, shared=shared, numpy_num_threads=2,
                 walltime=walltime, notification=False)
    # Loop over all MJDs
    for m in mjds:
        outfile = os.environ['APOGEE_REDUX']+'/'+apred+'/log/'+observatory+'/'+str(mjd5)+'-unidir.'+logtime+'.log'
        errfile = outfile.replace('.log','.err')
        if os.path.exists(os.path.dirname(outfile))==False:
            os.makedirs(os.path.dirname(outfile))
        queue.append('sas_mwm_healpix --spectro apogee --mjd {0} --telescope {1} --drpver {2} -v'.format(mjd5,telescope,apred),
                     outfile=outfile, errfile=errfile)
    queue.commit(hard=True,submit=True)
    rootLogger.info('PBS key is '+queue.key)        
    queue_wait(queue,sleeptime=60,verbose=True,logg=rootLogger)  # wait for jobs to complete
    del queue    
    #  sas_mwm_healpix --spectro apogee --mjd 59219 --telescope apo25m --drpver daily -v


    # 7) Run QA script
    #------------------
    rootLogger.info('')
    rootLogger.info('--------------')
    rootLogger.info('7) Running QA')
    rootLogger.info('==============')
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
