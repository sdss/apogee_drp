import copy
import numpy as np
import os
import shutil
from glob import glob
import pdb

from dlnpyutils import utils as dln
from ..utils import spectra,yanny,apload,platedata,plan,email,info
from ..apred import mkcal,cal,qa,monitor
from ..database import apogeedb
from . import mkplan,runapogee,check
from sdss_access.path import path
from astropy.io import fits
from astropy.table import Table,hstack,vstack
from collections import OrderedDict
#from astropy.time import Time
from datetime import datetime
import logging
from slurm import queue as pbsqueue
import time
import traceback
import subprocess

def loadmjd(mjd):
    """ Parse and expand input MJD range/list."""
    if mjd is not None:
        mjds = dln.loadinput(mjd)
        # Expand ranges
        newmjds = []
        for m in mjds:
            if str(m).find('-')>-1:
                parts = str(m).split('-')
                part1 = parts[0]
                part2 = parts[1]
                multimjd = list(np.arange(int(part1),int(part2)+1))
                newmjds += multimjd
            else:
                newmjds.append(m)
        newmjds = list(np.unique(newmjds))
        mjds = newmjds
    else:
        mjds = np.arange(59146,runapogee.lastnightmjd()+1)
    return mjds

def loadsteps(steps):
    """ Parse and expand input steps into a list."""
    # Reduction steps
    # The default is to do all
    allsteps = ['setup','master','3d','cal','plan','apred','rv','summary','unified','qa']
    if steps is None:
        steps = allsteps
    else:
        steps = dln.loadinput(steps)
        steps = [str(s).lower() for s in steps]  # lower case
        # expand ranges, i.e. 3d-cals
        newsteps = []
        for s in steps:
            if s.find('-') > -1:
                parts = s.split('-')
                step1 = parts[0]
                ind1, = np.where(np.array(allsteps)==step1)
                if len(ind1)==0:
                    raise ValueError('Step '+step1+' not found in steps: '+','.join(allsteps))
                step2 = parts[1]
                ind2, = np.where(np.array(allsteps)==step2)
                if len(ind2)==0:
                    raise ValueError('Step '+step2+' not found in steps: '+','.join(allsteps))                
                multisteps = list(np.array(allsteps)[ind1[0]:ind2[0]+1])
                newsteps += multisteps
            else:
                newsteps.append(s)
        newsteps = list(np.unique(newsteps))
        steps = newsteps
    # Put them in the correct order
    fsteps = []
    for s in allsteps:
        if s in steps:
            fsteps.append(s)
    steps = fsteps

    return steps

def getexpinfo(load,mjds,logger=None,verbose=True):
    """
    Get all exposure for the MJDs.


    Parameters
    ----------
    load : ApLoad object
       ApLoad object that contains "apred" and "telescope".
    mjds : list
       List of MJDs to process
    logger : logger, optional
       Logging object.  If not is input, then a default one will be created.
    verbose : boolean, optional
       Print output to the screen.  Default is True.

    Returns
    -------
    expinfo : numpy structured array
       Exposure information from the raw data headers for the given MJDs.

    Example
    -------

    expinfo = getexpinfo(load,mjds)

    """

    apred = load.apred
    telescope = load.telescope
    observatory = telescope[0:3]

    if logger is None:
        logger = dln.basiclogger()

    if type(mjds) is str or hasattr(mjds,'__iter__')==False:
        mjds = [mjds]
    nmjds = np.array(mjds).size

    db = apogeedb.DBSession()

    # Get information for all of the exposures
    expinfo = None
    for m in mjds:
        expinfo1 = info.expinfo(observatory=observatory,mjd5=m)
        expinfo1 = Table(expinfo1)
        # Get data from database
        dbexpinfo = db.query('exposure',where="mjd=%d and observatory='%s'" % (m,observatory))
        if len(dbexpinfo)>0:
            vals,ind1,ind2 = np.intersect1d(expinfo1['num'],dbexpinfo['num'],return_indices=True)
        else:
            ind1 = []
        # Load new exposures into database
        if len(ind1) != len(expinfo1):
            db.ingest('exposure',expinfo1)  # insert into database
            expinfo1 = db.query('exposure',where="mjd=%d and observatory='%s'" % (m,observatory))            
            expinfo1 = Table(expinfo1)
        else:
            expinfo1 = Table(dbexpinfo)
        if expinfo is None:
            expinfo = Table(expinfo1)
        else:
            if len(expinfo1)>0:
                # this might cause issues with some string columsns
                #  if the lengths don't agree
                expinfo = vstack((expinfo,expinfo1))

    nexp = len(expinfo)
    logger.info(str(nexp)+' exposures')

    # Sort them
    si = np.argsort(expinfo['num'])
    expinfo = expinfo[si]
    expinfo = np.array(expinfo)

    db.close()

    return expinfo

def getplanfiles(load,mjds,exist=False,logger=None):
    """
    Get all of the plan files for a list of MJDs.

    Parameters
    ----------
    load : ApLoad object
       ApLoad object that contains "apred" and "telescope".
    mjds : list
       List of MJDs to check for plan files.
    exist : bool, optional
       Only return the names of plan files that exist.  Default is False.
    logger : logger, optional
       Logging object.  If not is input, then a default one will be created.

    Returns
    -------
    planfiles : list
       The list of plan files.

    Example
    -------

    planfiles = getplanfiles(load,mjds)

    """

    if logger is None:
        logger = dln.basiclogger()
    
    mjdstart = np.min(mjds)
    mjdstop = np.max(mjds)
    
    # Get plan files from the database
    db = apogeedb.DBSession()
    plans = db.query('plan',where="mjd>='%s' and mjd<='%s'" % (mjdstart,mjdstop))
    db.close()
    # No plan files for these MJDs
    if len(plans)==0:
        logger.info('No plan files to process')
        return []
    # Only get plans for the MJDs that we have
    ind = []
    for m in mjds:
        ind1, = np.where(plans['mjd']==m)
        if len(ind1)>0:
            ind += list(ind1)
    ind = np.array(ind)
    # No plan files for these MJDs
    if len(ind)==0:
        logger.info('No plan files to process')
        return []
    plans = plans[ind]
    planfiles = plans['planfile']

    # Make sure they are unique
    planfiles = list(np.unique(np.array(planfiles)))
    # Check that they exist
    if exist:
        planfiles = [f for f in planfiles if os.path.exists(f)]
    return planfiles


def mkvers(apred,logger=None):
    """
    Setup APOGEE DRP directory structure.
   
    Parameters
    ----------
    apred : str
       Reduction version name.
    logger : logger, optional
       Logging object.  If not is input, then a default one will be created.
    
    Returns
    -------
    It makes the directory structure for a new DRP reduction version.

    Example
    -------

    mkvers('v1.0.0')

    """

    if logger is None:
        logger = dln.basiclogger()
    
    apogee_redux = os.environ['APOGEE_REDUX']+'/'
    apogee_drp_dir = os.environ['APOGEE_DRP_DIR']+'/'

    logger.info('Setting up directory structure for APOGEE DRP version = '+apred)

    # Main directory
    if os.path.exists(apogee_redux+apred)==False:
        logger.info('Creating '+apogee_redux+apred)
        os.makedirs(apogee_redux+apred)
    else:
        logger.info(apogee_redux+apred+' already exists')
    # First level
    for d in ['cal','exposures','stars','fields','visit','qa','plates','monitor','summary','log']:
        if os.path.exists(apogee_redux+apred+'/'+d)==False:
            logger.info('Creating '+apogee_redux+apred+'/'+d)
            os.makedirs(apogee_redux+apred+'/'+d)
        else:
            logger.info(apogee_redux+apred+'/'+d+' already exists')
    # North/south subdirectories
    for d in ['cal','exposures','monitor']:
        for obs in ['apogee-n','apogee-s']:
            if os.path.exists(apogee_redux+apred+'/'+d+'/'+obs)==False:
                logger.info('Creating '+apogee_redux+apred+'/'+d+'/'+obs)
                os.makedirs(apogee_redux+apred+'/'+d+'/'+obs)
            else:
                logger.info(apogee_redux+apred+'/'+d+'/'+obs+' already exists')
    for d in ['visit','stars','fields']:
        for obs in ['apo25m','lco25m']:
            if os.path.exists(apogee_redux+apred+'/'+d+'/'+obs)==False:
                logger.info('Creating '+apogee_redux+apred+'/'+d+'/'+obs)
                os.makedirs(apogee_redux+apred+'/'+d+'/'+obs)
            else:
                logger.info(apogee_redux+apred+'/'+d+'/'+obs+' already exists')
    for d in ['log']:
        for obs in ['apo','lco']:
            if os.path.exists(apogee_redux+apred+'/'+d+'/'+obs)==False:
                logger.info('Creating '+apogee_redux+apred+'/'+d+'/'+obs)
                os.makedirs(apogee_redux+apred+'/'+d+'/'+obs)
            else:
                logger.info(apogee_redux+apred+'/'+d+'/'+obs+' already exists')
    # Cal subdirectories
    for d in ['bpm','darkcorr','detector','flatcorr','flux','fpi','html','littrow','lsf','persist','plans','psf','qa','telluric','trace','wave']:
        for obs in ['apogee-n','apogee-s']:
            if os.path.exists(apogee_redux+apred+'/cal/'+obs+'/'+d)==False:
                logger.info('Creating '+apogee_redux+apred+'/cal/'+obs+'/'+d)
                os.makedirs(apogee_redux+apred+'/cal/'+obs+'/'+d)
            else:
                logger.info(apogee_redux+apred+'/cal/'+obs+'/'+d+' already exists')

    # Webpage files
    #if os.path.exists(apogee_drp_dir+'etc/htaccess'):
    #    os.copy(apogee_drp_dir+'etc/htaccess',apogee_redux+apred+'qa/.htaccess'
    if os.path.exists(apogee_drp_dir+'etc/sorttable.js') and os.path.exists(apogee_redux+apred+'/qa/sorttable.js')==False:
        logger.info('Copying sorttable.js')
        shutil.copyfile(apogee_drp_dir+'etc/sorttable.js',apogee_redux+apred+'/qa/sorttable.js')


def mkmastercals(load,mjds,slurm,clobber=False,linkvers=None,logger=None):
    """
    Make the master calibration products for a reduction version and MJD range.

    Parameters
    ----------
    load : ApLoad object
       ApLoad object that contains "apred" and "telescope".
    mjds : list
       List of MJDs to process
    slurm : dictionary
       Dictionary of slurm settings.
    clobber : boolean, optional
       Overwrite any existing files.  Default is False.
    linkvers : str, optional
       Name of reduction version to use for symlinks for the master calibration files.    
    logger : logger, optional
       Logging object.  If not is input, then a default one will be created.

    Returns
    -------
    All the master calibration products are made for a reduction version.

    Example
    -------

    mkmastercals('v1.0.0','telescope')

    
    apogee/bin/mkcal used to make these master calibration files.
    """

    if logger is None:
        logger = dln.basiclogger()

    apred = load.apred
    telescope = load.telescope
    observatory = telescope[0:3]
    logtime = datetime.now().strftime("%Y%m%d%H%M%S") 

    apogee_redux = os.environ['APOGEE_REDUX']+'/'
    apogee_drp_dir = os.environ['APOGEE_DRP_DIR']+'/'
    
    # Symbolic links to another version
    if linkvers:
        logger.info('Creating calibration product symlinks to version >>'+str(linkvers)+'<<')
        cwd = os.path.abspath(os.curdir)
        for d in ['bpm','darkcorr','detector','flatcorr','littrow','lsf','persist','telluric']:
            for obs in ['apogee-n','apogee-s']:    
                logger.info('Creating symlinks for '+apogee_redux+apred+'/cal/'+obs+'/'+d)
                os.chdir(apogee_redux+apred+'/cal/'+obs+'/'+d)
                subprocess.run(['ln -s '+apogee_redux+linkvers+'/cal/'+obs+'/'+d+'/*fits .'],shell=True)
        return

    # Input MJD range
    mjdstart = np.min(mjds)
    mjdstop = np.max(mjds)

    # Read in the master calibration index
    caldir = os.environ['APOGEE_DRP_DIR']+'/data/cal/'
    calfile = caldir+load.instrument+'.par'
    allcaldict = mkcal.readcal(calfile)

    # Make calibration frames: darks, waves, LSFs
    
    #set host = `hostname -s`
    #if ( $?UUFSCELL ) then
    #   setenv APOGEE_LOCALDIR /scratch/local/$USER/$SLURM_JOB_ID 
    #else
    #   rm $APOGEE_LOCALDIR/*
    #endif


    # -- Master calibration products made every year or so --
    # Detector
    # Dark, sequence of long darks
    # Flat, sequence of internal flats
    # BPM, use dark+flat sequence
    # Sparse, sequence of sparse quartz flats
    # multiwave, set of arclamp exposures
    # LSF, sky flat + multiwave
    # fiber?
    # -- Other master calibration products made only once ---
    # Littrow
    # Persist, PersistModel
    # telluric, need LSF

    # -- Daily calibration products ---
    # PSF/EPSF/Trace, from domeflat or quartzflat
    # Flux, from domeflat or quartzflat
    # Wave, from arclamps

    # Maybe have an option to copy/symlink them from a previous apred version


    # Make Detector and Linearity 
    #----------------------------
    # MKDET.PRO makes both
    detdict = allcaldict['det']
    logger.info('')
    logger.info('--------------------------------------------')
    logger.info('Making master Detector/Linearity in parallel')
    logger.info('============================================')
    logger.info('Slurm settings: '+str(slurm))
    queue = pbsqueue(verbose=True)
    queue.create(label='mkdet', **slurm)
    docal = np.zeros(len(detdict),bool)
    for i in range(len(detdict)):
        name = detdict['name'][i]
        if np.sum((mjds >= detdict['mjd1'][i]) & (mjds <= detdict['mjd2'][i])) > 0:
            logfile1 = os.environ['APOGEE_REDUX']+'/'+apred+'/log/mkdet-'+str(name)+'-'+telescope+'_pbs.'+logtime+'.log'
            errfile1 = logfile1.replace('.log','.err')
            if os.path.exists(os.path.dirname(logfile1))==False:
                os.makedirs(os.path.dirname(logfile1))
            cmd1 = 'makecal --vers {0} --telescope {1}'.format(apred,telescope)
            cmd1 += ' --detector '+str(name)+' --unlock'
            if clobber:
                cmd1 += ' --clobber'
            #logfiles.append(logfile1)
            # Check if files exist already
            docal[i] = True
            if clobber is not True:
                outfile = load.filename('Detector',num=name,chips=True)
                if load.exists('Detector',num=name):
                    logger.info(os.path.basename(outfile)+' already exists and clobber==False')
                    docal[k] = False
            if docal[i]:
                logger.info('Detector file %d : %s' % (i+1,name))
                logger.info('Command : '+cmd1)
                logger.info('Logfile : '+logfile1)
                queue.append(cmd1, outfile=logfile1,errfile=errfile1)
    if np.sum(docal)>0:
        queue.commit(hard=True,submit=True)
        logger.info('PBS key is '+queue.key)
        runapogee.queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
    else:
        logger.info('No master Detector calibration files need to be run')
    del queue    

    import pdb; pdb.set_trace()


    # Make darks in parallel
    #-----------------------
    # they take too much memory to run in parallel
    #idl -e "makecal,dark=1,vers='$vers',telescope='$telescope'" >& log/mkdark-$telescope.$host.log
    #darkplot --apred $vers --telescope $telescope
    darkdict = allcaldict['dark']
    logger.info('')
    logger.info('-------------------------------')
    logger.info('Making master darks in parallel')
    logger.info('===============================')
    # Give each job LOTS of memory
    slurm1 = slurm.copy()
    slurm1['nodes'] = 2
    slurm1['cpus'] = 4
    slurm1['mem_per_cpu'] = 20000  # in MB
    logger.info('Slurm settings: '+str(slurm1))
    queue = pbsqueue(verbose=True)
    queue.create(label='mkdark', **slurm1)
    docal = np.zeros(len(darkdict),bool)
    for i in range(len(darkdict)):
        name = darkdict['name'][i]
        if np.sum((mjds >= darkdict['mjd1'][i]) & (mjds <= darkdict['mjd2'][i])) > 0:
            logfile1 = os.environ['APOGEE_REDUX']+'/'+apred+'/log/mkdark-'+str(name)+'-'+telescope+'_pbs.'+logtime+'.log'
            errfile1 = logfile1.replace('.log','.err')
            if os.path.exists(os.path.dirname(logfile1))==False:
                os.makedirs(os.path.dirname(logfile1))
            cmd1 = 'makecal --vers {0} --telescope {1}'.format(apred,telescope)
            cmd1 += ' --dark '+str(name)+' --unlock'
            if clobber:
                cmd1 += ' --clobber'
            #logfiles.append(logfile1)
            # Check if files exist already
            docal[i] = True
            if clobber is not True:
                outfile = load.filename('Dark',num=name,chips=True)
                if load.exists('Dark',num=name):
                    logger.info(os.path.basename(outfile)+' already exists and clobber==False')
                    docal[k] = False
            if docal[i]:
                logger.info('Dark file %d : %s' % (i+1,name))
                logger.info('Command : '+cmd1)
                logger.info('Logfile : '+logfile1)
                queue.append(cmd1, outfile=logfile1,errfile=errfile1)
    if np.sum(docal)>0:
        queue.commit(hard=True,submit=True)
        logger.info('PBS key is '+queue.key)
        runapogee.queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
    else:
        logger.info('No master Dark calibration files need to be run')
    del queue    
    # Make the dark plots
    if np.sum(docal)>0:
        cal.darkplot(apred=apred,telescope=telescope)


    # Make flats in parallel
    #-------------------------
    #idl -e "makecal,flat=1,vers='$vers',telescope='$telescope'" >& log/mkflat-$telescope.$host.log
    #flatplot --apred $vers --telescope $telescope
    flatdict = allcaldict['flat']
    logger.info('')
    logger.info('-------------------------------')
    logger.info('Making master flats in parallel')
    logger.info('===============================')
    logger.info('Slurm settings: '+str(slurm1))
    queue = pbsqueue(verbose=True)
    queue.create(label='mkflat', **slurm1)
    for i in range(len(flatdict)):
        name = flatdict['name'][i]
        if np.sum((mjds >= flatdict['mjd1'][i]) & (mjds <= flatdict['mjd2'][i])) > 0:
            logfile1 = os.environ['APOGEE_REDUX']+'/'+apred+'/log/mkflat-'+str(name)+'-'+telescope+'_pbs.'+logtime+'.log'
            errfile1 = logfile1.replace('.log','.err')
            if os.path.exists(os.path.dirname(logfile1))==False:
                os.makedirs(os.path.dirname(logfile1))
            cmd1 = 'makecal --vers {0} --telescope {1}'.format(apred,telescope)
            cmd1 += ' --flat '+str(name)+' --unlock'
            if clobber:
                cmd1 += ' --clobber'
            #logfiles.append(logfile1)
            # Check if files exist already
            docal[i] = True
            if clobber is not True:
                outfile = load.filename('Flat',num=name,chips=True)
                if load.exists('Flat',num=name):
                    logger.info(os.path.basename(outfile)+' already exists and clobber==False')
                    docal[i] = False
            if docal[i]:
                logger.info('Flat file %d : %s' % (i+1,name))
                logger.info('Command : '+cmd1)
                logger.info('Logfile : '+logfile1)
                queue.append(cmd1, outfile=logfile1,errfile=errfile1)
    if np.sum(docal)>0:
        queue.commit(hard=True,submit=True)
        logger.info('PBS key is '+queue.key)
        runapogee.queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
    else:
        logger.info('No master Flat calibration files need to be run')
    del queue    
    # Make the flat plots
    if np.sum(docal)>0:
        cal.flatplot(apred=apred,telescope=telescope)


    # Make BPM in parallel
    #----------------------
    #idl -e "makecal,bpm=1,vers='$vers',telescope='$telescope'" >& log/mkbpm-$telescope.$host.log
    bpmdict = allcaldict['bpm']
    logger.info('')
    logger.info('------------------------------')
    logger.info('Making master BPMs in parallel')
    logger.info('==============================')
    logger.info('Slurm settings: '+str(slurm))
    queue = pbsqueue(verbose=True)
    queue.create(label='mkbpm', **slurm)
    for i in range(len(bpmdict)):
        name = bpmdict['name'][i]
        if np.sum((mjds >= bpmdict['mjd1'][i]) & (mjds <= bpmdict['mjd2'][i])) > 0:
            logfile1 = os.environ['APOGEE_REDUX']+'/'+apred+'/log/mkbpm-'+str(name)+'-'+telescope+'_pbs.'+logtime+'.log'
            errfile1 = logfile1.replace('.log','.err')
            if os.path.exists(os.path.dirname(logfile1))==False:
                os.makedirs(os.path.dirname(logfile1))
            cmd1 = 'makecal --vers {0} --telescope {1}'.format(apred,telescope)
            cmd1 += ' --bpm '+str(name)+' --unlock'
            if clobber:
                cmd1 += ' --clobber'
            #logfiles.append(logfile1)
            # Check if files exist already
            docal[i] = True
            if clobber is not True:
                outfile = load.filename('BPM',num=name,chips=True)
                if load.exists('BPM',num=name):
                    logger.info(os.path.basename(outfile)+' already exists and clobber==False')
                    docal[i] = False
            if docal[i]:
                logger.info('BPM file %d : %s' % (i+1,name))
                logger.info('Command : '+cmd1)
                logger.info('Logfile : '+logfile1)
                queue.append(cmd1, outfile=logfile1,errfile=errfile1)
    if np.sum(docal)>0:
        queue.commit(hard=True,submit=True)
        logger.info('PBS key is '+queue.key)
        runapogee.queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
    else:
        logger.info('No master BPM calibration files need to be run')
    del queue    


    # Make Littrow in parallel
    #--------------------------
    #idl -e "makecal,littrow=1,vers='$vers',telescope='$telescope'" >& log/mklittrow-$telescope.$host.log
    littdict = allcaldict['littrow']
    logger.info('')
    logger.info('----------------------------------')
    logger.info('Making master Littrows in parallel')
    logger.info('==================================')
    logger.info('Slurm settings: '+str(slurm))
    queue = pbsqueue(verbose=True)
    queue.create(label='mklittrow', **slurm)
    for i in range(len(littdict)):
        name = littdict['name'][i]
        if np.sum((mjds >= littdict['mjd1'][i]) & (mjds <= littdict['mjd2'][i])) > 0:
            logfile1 = os.environ['APOGEE_REDUX']+'/'+apred+'/log/mklittrow-'+str(name)+'-'+telescope+'_pbs.'+logtime+'.log'
            errfile1 = logfile1.replace('.log','.err')
            if os.path.exists(os.path.dirname(logfile1))==False:
                os.makedirs(os.path.dirname(logfile1))
            cmd1 = 'makecal --vers {0} --telescope {1}'.format(apred,telescope)
            cmd1 += ' --littrow '+str(name)+' --unlock'
            if clobber:
                cmd1 += ' --clobber'
            #logfiles.append(logfile1)
            # Check if files exist already
            docal[i] = True
            if clobber is not True:
                outfile = load.filename('Littrow',num=name,chips=True)
                if load.exists('Littrow',num=name):
                    logger.info(os.path.basename(outfile)+' already exists and clobber==False')
                    docal[i] = False
            if docal[i]:
                logger.info('Littrow file %d : %s' % (i+1,name))
                logger.info('Command : '+cmd1)
                logger.info('Logfile : '+logfile1)
                queue.append(cmd1, outfile=logfile1,errfile=errfile1)
    if np.sum(docal)>0:
        queue.commit(hard=True,submit=True)
        logger.info('PBS key is '+queue.key)
        runapogee.queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
    else:
        logger.info('No master Littrow calibration files need to be run')
    del queue    


    # Make Response in parallel
    #--------------------------
    responsedict = allcaldict['response']
    logger.info('')
    logger.info('-----------------------------------')
    logger.info('Making master responses in parallel')
    logger.info('===================================')
    logger.info('Slurm settings: '+str(slurm))
    queue = pbsqueue(verbose=True)
    queue.create(label='mkresponse', **slurm)
    for i in range(len(responsedict)):
        name = responsedict['name'][i]
        if np.sum((mjds >= responsedict['mjd1'][i]) & (mjds <= responsedict['mjd2'][i])) > 0:
            logfile1 = os.environ['APOGEE_REDUX']+'/'+apred+'/log/mkresponse-'+str(name)+'-'+telescope+'_pbs.'+logtime+'.log'
            errfile1 = logfile1.replace('.log','.err')
            if os.path.exists(os.path.dirname(logfile1))==False:
                os.makedirs(os.path.dirname(logfile1))
            cmd1 = 'makecal --vers {0} --telescope {1}'.format(apred,telescope)
            cmd1 += ' --response '+str(name)+' --unlock'
            if clobber:
                cmd1 += ' --clobber'
            #logfiles.append(logfile1)
            # Check if files exist already
            docal[i] = True
            if clobber is not True:
                outfile = load.filename('Response',num=name,chips=True)
                if load.exists('Response',num=name):
                    logger.info(os.path.basename(outfile)+' already exists and clobber==False')
                    docal[k] = False
            if docal[i]:
                logger.info('Response file %d : %s' % (i+1,name))
                logger.info('Command : '+cmd1)
                logger.info('Logfile : '+logfile1)
                queue.append(cmd1, outfile=logfile1,errfile=errfile1)
    if np.sum(docal)>0:
        queue.commit(hard=True,submit=True)
        logger.info('PBS key is '+queue.key)
        runapogee.queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
    else:
        logger.info('No master Response calibration files need to be run')
    del queue    


    # Make multiwave cals in parallel
    #--------------------------------
    #set n = 0
    #while ( $n < 5 ) 
    #   idl -e "makecal,multiwave=1,vers='$vers',telescope='$telescope'" >& log/mkwave-$telescope"$n".$host.log &
    #   sleep 20
    #   @ n = $n + 1
    #end
    #wait

    #multiwavedict = allcaldict['multiwave']
    #logger.info('')
    #logger.info('-----------------------------------')
    #logger.info('Making master multiwave in parallel')
    #logger.info('===================================')
    #logger.info(str(len(multiwavedict))+' multiwave to make: '+','.join(multiwavedict['name']))
    #logger.info('')
    #slurm1['nodes'] = 1
    #slurm1['cpus'] = 5
    #logger.info('Slurm settings: '+str(slurm1))
    #queue = pbsqueue(verbose=True)
    #queue.create(label='mkmultiwave', **slurm1)
    #for i in range(len(multiwavedict)):
    #    outfile1 = os.environ['APOGEE_REDUX']+'/'+apred+'/log/mkmultiwave-'+str(multiwavedict['name'][i])+telescope+'.'+logtime+'.log'
    #    errfile1 = outfile1.replace('.log','.err')
    #    if os.path.exists(os.path.dirname(outfile1))==False:
    #        os.makedirs(os.path.dirname(outfile1))
    #    cmd = 'makecal --vers {0} --telescope {1}'.format(apred,telescope)
    #    cmd += ' --multiwave '+str(multiwavedict['name'][i])+' --unlock'
    #    queue.append(cmd,outfile=outfile1, errfile=errfile1)
    #queue.commit(hard=True,submit=True)
    #runapogee.queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
    #del queue    


    # Make LSFs in parallel
    #-----------------------
    #set n = 0
    #while ( $n < 5 ) 
    #   idl -e "makecal,lsf=1,/full,/pl,vers='$vers',telescope='$telescope'" >& log/mklsf-$telescope"$n".$host.log &
    #   sleep 20
    #   @ n = $n + 1
    #end
    #wait

    lsfdict = allcaldict['lst']
    logger.info('')
    logger.info('--------------------------------')
    logger.info('Making master LSFs in parallel')
    logger.info('================================')
    logger.info('Slurm settings: '+str(slurm))
    queue = pbsqueue(verbose=True)
    queue.create(label='mklsf', **slurm)
    for i in range(len(littdict)):
        name = lsfdict['name'][i]
        if np.sum((mjds >= lsfdict['mjd1'][i]) & (mjds <= lsfdict['mjd2'][i])) > 0:
            logfile1 = os.environ['APOGEE_REDUX']+'/'+apred+'/log/mklsf-'+str(name)+'-'+telescope+'_pbs.'+logtime+'.log'
            errfile1 = logfile1.replace('.log','.err')
            if os.path.exists(os.path.dirname(logfile1))==False:
                os.makedirs(os.path.dirname(logfile1))
            cmd1 = 'makecal --vers {0} --telescope {1}'.format(apred,telescope)
            cmd1 += ' --lsf '+str(name)+' --unlock'
            if clobber:
                cmd1 += ' --clobber'
            #logfiles.append(logfile1)
            # Check if files exist already
            docal[i] = True
            if clobber is not True:
                outfile = load.filename('LSF',num=name,chips=True)
                if load.exists('LSF',num=name):
                    logger.info(os.path.basename(outfile)+' already exists and clobber==False')
                    docal[i] = False
            if docal[i]:
                logger.info('LSF file %d : %s' % (i+1,name))
                logger.info('Command : '+cmd1)
                logger.info('Logfile : '+logfile1)
                queue.append(cmd1, outfile=logfile1,errfile=errfile1)
    if np.sum(docal)>0:
        queue.commit(hard=True,submit=True)
        logger.info('PBS key is '+queue.key)
        runapogee.queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
    else:
        logger.info('No master LSF calibration files need to be run')
    del queue    


    ## UPDATE THE DATABASE!!!

    # Need to check if the master calibration files actually got made

    
def runap3d(load,mjds,slurm,clobber=False,logger=None):
    """
    Run AP3D on all exposures for a list of MJDs.

    Parameters
    ----------
    load : ApLoad object
       ApLoad object that contains "apred" and "telescope".
    mjds : list
       List of MJDs to process
    slurm : dictionary
       Dictionary of slurm settings.
    clobber : boolean, optional
       Overwrite existing files.  Default is False.
    logger : logger, optional
       Logging object.  If not is input, then a default one will be created.

    Returns
    -------
    chk3d : numpy structured array
       Table of summary and QA information about the exposure processing.

    Example
    -------

    runap3d(load,mjds)

    """

    if logger is None:
        logger = dln.basiclogger()
    
    apred = load.apred
    telescope = load.telescope
    observatory = telescope[0:3]
    logtime = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Get exposures
    expinfo = getexpinfo(load,mjds,logger=logger)

    # Process the files
    if len(expinfo)==0:
        logger.info('No exposures to process with AP3D')
        chk3d = []

    slurm1 = slurm.copy()
    if len(expinfo)<64:
        slurm1['cpus'] = len(expinfo)
    slurm1['numpy_num_threads'] = 2
    logger.info('Slurm settings: '+str(slurm1))
    queue = pbsqueue(verbose=True)
    queue.create(label='ap3d', **slurm1)
    do3d = np.zeros(len(expinfo),bool)
    for i,num in enumerate(expinfo['num']):
        mjd = int(load.cmjd(num))
        logfile1 = load.filename('2D',num=num,mjd=mjd,chips=True).replace('2D','3D')
        logfile1 = os.path.dirname(logfile1)+'/logs/'+os.path.basename(logfile1)
        logfile1 = logfile1.replace('.fits','_pbs.'+logtime+'.log')
        if os.path.dirname(logfile1)==False:
            os.makedirs(os.path.dirname(logfile1))
        # Check if files exist already
        do3d[i] = True
        if clobber is not True:
            outfile = load.filename('2D',num=num,mjd=mjd,chips=True)
            if load.exists('2D',num=num):
                logger.info(os.path.basename(outfile)+' already exists and clobber==False')
                do3d[i] = False
        if do3d[i]:
            cmd1 = 'ap3d --num {0} --vers {1} --telescope {2} --unlock'.format(num,apred,telescope)
            if clobber:
                cmd1 += ' --clobber'
            logger.info('Exposure %d : %d' % (i+1,num))
            logger.info('Command : '+cmd1)
            logger.info('Logfile : '+logfile1)
            queue.append(cmd1,outfile=logfile1,errfile=logfile1.replace('.log','.err'))
    if np.sum(do3d)>0:
        queue.commit(hard=True,submit=True)
        logger.info('PBS key is '+queue.key)
        runapogee.queue_wait(queue,sleeptime=60,verbose=True,logger=logger)  # wait for jobs to complete
        # This should check if the ap3d ran okay and puts the status in the database
        chk3d = runapogee.check_ap3d(expinfo,queue.key,apred,telescope,verbose=True,logger=logger)
    else:
        logger.info('No exposures need AP3D processing')
    del queue
        
    return chk3d

def rundailycals(load,mjds,slurm,clobber=False,logger=None):
    """
    Run daily calibration frames for a list of MJDs.

    Parameters
    ----------
    load : ApLoad object
       ApLoad object that contains "apred" and "telescope".
    mjds : list
       List of MJDs to process
    slurm : dictionary
       Dictionary of slurm settings.
    clobber : boolean, optional
       Overwrite existing files.  Default is False.
    logger : logger, optional
       Logging object.  If not is input, then a default one will be created.

    Returns
    -------
    chkcal : numpy structured array
       Table of summary and QA information about the calibration exposure processing.

    Example
    -------

    chkcal = rundailycals(load,mjds)

    """

    if logger is None:
        logger = dln.basiclogger()
        
    apred = load.apred
    telescope = load.telescope
    observatory = telescope[0:3]
    logtime = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Get exposures
    logger.info('Getting exposure information')
    expinfo = getexpinfo(load,mjds,logger=logger,verbose=False)

    # First we need to run domeflats and quartzflats so there are apPSF files
    # Then the arclamps
    # apFlux files?
    # Then the FPI exposures last (needs apPSF and apWave files)
    calind, = np.where((expinfo['exptype']=='DOMEFLAT') | (expinfo['exptype']=='QUARTZFLAT') | 
                       (expinfo['exptype']=='ARCLAMP') | (expinfo['exptype']=='FPI'))
    if len(calind)>0:
        expinfo = expinfo[calind]
    else:
        logger.info('No calibration files to run')
        return None

    # Run QA check on the files
    logger.info(' ')
    logger.info('Doing quality checks on all calibration exposures')
    qachk = check.check(expinfo['num'],apred,telescope,verbose=True,logger=logger)
    logger.info(' ')
    okay, = np.where(qachk['okay']==True)
    if len(okay)>0:
        expinfo = expinfo[okay]
    else:
        logger.info('No good calibration files to run')
        return None        

    # Only need one FPI per night
    # The FPI processing is done at a NIGHT level
    fpi, = np.where(expinfo['exptype']=='FPI')
    if len(fpi)>0:
        # Take the first for each night
        logger.info('Only keeping ONE FPI exposure per night/MJD')
        vals,ui = np.unique(expinfo['mjd'][fpi],return_index=True)
        todel = np.copy(fpi)
        todel = np.delete(todel,ui)  # remove the ones we want to keep
        expinfo = np.delete(expinfo,todel)

    # 1: psf, 2: flux, 4: arcs, 8: fpi
    calcodedict = {'DOMEFLAT':3, 'QUARTZFLAT':1, 'ARCLAMP':4, 'FPI':8}
    calcode = [calcodedict[etype] for etype in expinfo['exptype']]
    # Do NOT use DOMEFLATS for apPSF with FPS, MJD>=59556 (only quarzflats)
    #  Only use domeflats for apFlux cal files, need to chance calcode
    dome, = np.where((expinfo['exptype']=='DOMEFLAT') & (expinfo['mjd']>=59556))
    if len(dome)>0:
        calcode = np.array(calcode)
        calcode[dome] = 2
        calcode = list(calcode)
    calnames = ['DOMEFLAT/QUARTZFLAT','FLUX','ARCLAMP','FPI']
    shcalnames = ['psf','flux','arcs','fpi']
    filecodes = ['PSF','Flux','Wave','WaveFPI']
    chkcal = []
    for j,ccode in enumerate([1,2,4,8]):
        logger.info('')
        logger.info('----------------------------------------------')
        logger.info('Running Calibration Files: '+str(calnames[j]))
        logger.info('==============================================')
        logger.info('')
        cind, = np.where((np.array(calcode) & ccode) > 0)
        if len(cind)>0:
            logger.info(str(len(cind))+' file(s)')
            slurm1 = slurm.copy()
            if len(cind)<64:
                slurm1['cpus'] = len(cind)
            slurm1['numpy_num_threads'] = 2
            logger.info('Slurm settings: '+str(slurm1))
            queue = pbsqueue(verbose=True)
            queue.create(label='makecal-'+shcalnames[j], **slurm1)
            logfiles = []
            docal = np.zeros(len(cind),bool)
            for k in range(len(cind)):
                num1 = expinfo['num'][cind[k]]
                mjd1 = int(load.cmjd(num1))
                calplandir = os.path.dirname(load.filename('CalPlan',num=0,mjd=mjd1))
                exptype1 = expinfo['exptype'][cind[k]]
                arctype1 = expinfo['arctype'][cind[k]]                    
                if ccode==1:    # psfs
                    cmd1 = 'makecal --psf '+str(num1)+' --unlock'
                    if clobber: cmd1 += ' --clobber'
                    logfile1 = calplandir+'/apPSF-'+str(num1)+'_pbs.'+logtime+'.log'
                elif ccode==2:   # flux
                    cmd1 = 'makecal --flux '+str(num1)+' --unlock'
                    if fps: cmd1 += ' --psflibrary'
                    if clobber: cmd1 += ' --clobber'
                    logfile1 = calplandir+'/apFlux-'+str(num1)+'_pbs.'+logtime+'.log'
                elif ccode==4:  # arcs
                    cmd1 = 'makecal --wave '+str(num1)+' --unlock'
                    if fps: cmd1 += ' --psflibrary'
                    if clobber: cmd1 += ' --clobber'
                    logfile1 = calplandir+'/apWave-'+str(num1)+'_pbs.'+logtime+'.log' 
                elif ccode==8:  # fpi
                    cmd1 = 'makecal --fpi '+str(num1)+' --unlock'
                    if fps: cmd1 += ' --psflibrary'
                    if clobber: cmd1 += ' --clobber'
                    logfile1 = calplandir+'/apFPI-'+str(num1)+'_pbs.'+logtime+'.log'
                logger.info('Logfile : '+logfile1)
                logfiles.append(logfile1)
                # Check if files exist already
                docal[k] = True
                if clobber is not True:
                    outfile = load.filename(filecodes[j],num=num1,mjd=mjd1,chips=True)
                    if load.exists(filecodes[j],num=num1,mjd=mjd1):
                        logger.info(os.path.basename(outfile)+' already exists and clobber==False')
                        docal[k] = False
                if docal[k]:
                    logger.info('Calibration file %d : %s %d' % (k+1,exptype1,num1))
                    logger.info('Command : '+cmd1)
                    logger.info('Logfile : '+logfile1)
                    queue.append(cmd1, outfile=logfile1,errfile=logfile1.replace('.log','.err'))
            if np.sum(docal)>0:
                queue.commit(hard=True,submit=True)
                logger.info('PBS key is '+queue.key)
                runapogee.queue_wait(queue,sleeptime=60,verbose=True,logger=logger)  # wait for jobs to complete
            else:
                logger.info('No '+str(calnames[j])+' calibration files need to be run')
            # Checks the status and updates the database
            calinfo = expinfo[cind]
            chkcal1 = runapogee.check_calib(calinfo,logfiles,queue.key,apred,verbose=True,logger=logger)
            if len(chkcal)==0:
                chkcal = chkcal1
            else:
                chkcal = np.hstack((chkcal,chkcal1))
            del queue
        else:
            logger.info('No '+str(calnames[j])+' calibration files to run')
            
    return chkcal

def makeplanfiles(load,mjds,slurm,clobber=False,logger=None):
    """
    Make plan files for a list of MJDs.

    Parameters
    ----------
    load : ApLoad object
       ApLoad object that contains "apred" and "telescope".
    mjds : list
       List of MJDs to process
    slurm : dictionary
       Dictionary of slurm settings.
    clobber : boolean, optional
       Overwrite existing files.  Default is False.
    logger : logger, optional
       Logging object.  If not is input, then a default one will be created.

    Returns
    -------
    planfiles : list
       List of plan files that were created.

    Example
    -------

    planfiles = makeplanfiles(load,mjds)

    """

    if logger is None:
        logger = dln.basiclogger()
    
    apred = load.apred
    telescope = load.telescope
    observatory = telescope[0:3]
    mjdstart = np.min(mjds)
    mjdstop = np.max(mjds)
    logtime = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Reduction logs directory
    logdir = os.environ['APOGEE_REDUX']+'/'+apred+'/log/'+observatory+'/'
    if os.path.exists(logdir)==False:
        os.makedirs(logdir)

    # Loop over MJDs
    planfiles = []
    for m in mjds:
        logger.info(' ')
        logger.info('Making plan files for MJD='+str(m))
        plandicts,planfiles0 = mkplan.make_mjd5_yaml(m,apred,telescope,clobber=clobber,logger=logger)
        dailyplanfile = os.environ['APOGEEREDUCEPLAN_DIR']+'/yaml/'+telescope+'/'+telescope+'_'+str(m)+'.yaml'
        try:
            planfiles1 = mkplan.run_mjd5_yaml(dailyplanfile,logger=logger)
            nplanfiles1 = len(planfiles1)
        except:
            traceback.print_exc()
            nplanfiles1 = 0

        logger.info('Writing list of plan files to '+logdir+str(m)+'.plans')
        if nplanfiles1>0:
            runapogee.dbload_plans(planfiles1)  # load plans into db
            # Write planfiles to MJD5.plans
            dln.writelines(logdir+str(m)+'.plans',[os.path.basename(pf) for pf in planfiles1])
            planfiles += planfiles1
        else:
            dln.writelines(logdir+str(m)+'.plans','')   # write blank file

        # Start entry in daily_status table
        #daycat = np.zeros(1,dtype=np.dtype([('mjd',int),('telescope',(np.str,10)),('nplanfiles',int),
        #                                    ('nexposures',int),('begtime',(np.str,50)),('success',bool)]))
        #daycat['mjd'] = m
        #daycat['telescope'] = telescope
        #daycat['nplanfiles'] = len(planfiles1)
        #daycat['nexposures'] = len(expinfo1)
        #daycat['begtime'] = begtime
        #daycat['success'] = False
        #db.ingest('daily_status',daycat)

    return planfiles

def runapred(load,mjds,slurm,clobber=False,logger=None):
    """
    Run APRED on all plan files for a list of MJDs.

    Parameters
    ----------
    load : ApLoad object
       ApLoad object that contains "apred" and "telescope".
    mjds : list
       List of MJDs to process
    slurm : dictionary
       Dictionary of slurm settings.
    clobber : boolean, optional
       Overwrite existing files.  Default is False.
    logger : logger, optional
       Logging object.  If not is input, then a default one will be created.

    Returns
    -------
    chkexp : numpy structured array
       Table of summary and QA information about the exposure APRED processing.
    chkvisit : numpy structured array
       Table of summary and QA information about the visit APRED processing.

    Example
    -------

    chkexp,chkvisit = runapred(load,mjds)

    """

    if logger is None:
        logger = dln.basiclogger()
    
    apred = load.apred
    telescope = load.telescope
    observatory = telescope[0:3]
    mjdstart = np.min(mjds)
    mjdstop = np.max(mjds)
    logtime = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Get plan files from the database
    planfiles = getplanfiles(load,mjds,exist=True,logger=logger)
    # No plan files for these MJDs
    if len(planfiles)==0:
        return []
    logger.info(str(len(planfiles))+' plan files')

    # Get exposure information
    expinfo = getexpinfo(load,mjds,logger=logger,verbose=False)
    if len(expinfo)==0:
        logger.info('No exposures')
        return []
        
    slurm1 = slurm.copy()
    if len(planfiles)<64:
        slurm1['cpus'] = len(planfiles)
    slurm1['numpy_num_threads'] = 2

    logger.info('Slurm settings: '+str(slurm1))
    queue = pbsqueue(verbose=True)
    queue.create(label='apred', **slurm1)
    for i,pf in enumerate(planfiles):
        logger.info('planfile %d : %s' % (i+1,pf))
        logfile = pf.replace('.yaml','_pbs.'+logtime+'.log')
        errfile = logfile.replace('.log','.err')
        cmd = 'apred {0}'.format(pf)
        if clobber:
            cmd += ' --clobber'
        logger.info('Command : '+cmd)
        logger.info('Logfile : '+logfile)
        queue.append(cmd, outfile=logfile,errfile=errfile)
    queue.commit(hard=True,submit=True)
    logger.info('PBS key is '+queue.key)
    runapogee.queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
    # This also loads the status into the database using the correct APRED version
    chkexp,chkvisit = runapogee.check_apred(expinfo,planfiles,queue.key,verbose=True,logger=logger)
    del queue

    # -- Summary statistics --
    # Exposure status
    if chkexp is not None:
        indexp, = np.where(chkexp['success']==True)
        logger.info('%d/%d exposures successfully processed' % (len(indexp),len(chkexp)))
    else:
        logger.info('No exposures')
    # Visit status
    if chkvisit is not None:
        indvisit, = np.where(chkvisit['success']==True)
        logger.info('%d/%d visits successfully processed' % (len(indvisit),len(chkvisit)))
    else:
        logger.info('No visits')
    
    return chkexp,chkvisit


def runrv(load,mjds,slurm,clobber=False,logger=None):
    """
    Run RV on all the stars observed from a list of MJDs.

    Parameters
    ----------
    load : ApLoad object
       ApLoad object that contains "apred" and "telescope".
    mjds : list
       List of MJDs to process
    slurm : dictionary
       Dictionary of slurm settings.
    clobber : boolean, optional
       Overwrite existing files.  Default is False.
    logger : logger, optional
       Logging object.  If not is input, then a default one will be created.

    Returns
    -------
    The unified directory structure software is run on the MJDs and relevant symlinks are generated.

    Example
    -------

    runrv(load,mjds)

    """

    if logger is None:
        logger = dln.basiclogger()
    
    apred = load.apred
    telescope = load.telescope
    observatory = telescope[0:3]
    mjdstart = np.min(mjds)
    mjdstop = np.max(mjds)
    logtime = datetime.now().strftime("%Y%m%d%H%M%S")

    # Get the stars that were observed in the MJD range and the MAXIMUM MJD for each star
    sql = "WITH mv as (SELECT apogee_id, apred_vers, telescope, max(mjd) as maxmjd FROM apogee_drp.visit"
    sql += " WHERE apred_vers='%s' and telescope='%s'" % (apred,telescope)
    sql += " GROUP BY apogee_id, apred_vers, telescope)"
    sql += " SELECT v.*,mv.maxmjd from apogee_drp.visit AS v LEFT JOIN mv on mv.apogee_id=v.apogee_id"
    sql += " Where v.apred_vers='%s' and v.mjd>=%d and v.mjd<=%d and v.telescope='%s'" % (apred,mjdstart,mjdstop,telescope)

    db = apogeedb.DBSession()
    vcat = db.query(sql=sql)
    db.close()
    if len(vcat)==0:
        logger.info('No visits found for MJDs')
        return []
    # Pick on the MJDs we want
    ind = []
    for m in mjds:
        gd, = np.where(vcat['mjd']==m)
        if len(gd)>0: ind += list(gd)
    ind = np.array(ind)
    if len(ind)==0:
        logger.info('No visits found for MJDs')
        return []
    vcat = vcat[ind]
    # Get unique stars
    objects,ui = np.unique(vcat['apogee_id'],return_index=True)
    vcat = vcat[ui]
    # Remove rows with missing or blank apogee_ids
    bd, = np.where((vcat['apogee_id']=='') | (vcat['apogee_id']=='None') | (vcat['apogee_id']=='2MNone') | (vcat['apogee_id']=='2M'))
    if len(bd)>0:
        vcat = np.delete(vcat,bd)
    logger.info(str(len(vcat))+' stars to run')

    # Change MJD to MAXMJD because the apStar file will have MAXMJD in the name
    vcat['mjd'] = vcat['maxmjd']    

    slurm1 = slurm.copy()
    if len(vcat)<64:
        slurm1['cpus'] = len(vcat)
    slurm1['numpy_num_threads'] = 2
    logger.info('Slurm settings: '+str(slurm1))
    queue = pbsqueue(verbose=True)
    queue.create(label='rv', **slurm1)
    dorv = np.zeros(len(vcat),bool)
    for i,obj in enumerate(vcat['apogee_id']):
        # We are going to run RV on ALL the visits
        # Use the MAXMJD in the table, now called MJD
        mjd = vcat['mjd'][i]
        apstarfile = load.filename('Star',obj=obj)
        apstarfile = apstarfile.replace('.fits','-'+str(mjd)+'.fits')
        outdir = os.path.dirname(apstarfile)  # make sure the output directories exist
        if os.path.exists(outdir)==False:
            os.makedirs(outdir)
        logfile = apstarfile.replace('.fits','_pbs.'+logtime+'.log')
        errfile = logfile.replace('.log','.err')
        # Run with --verbose
        cmd = 'rv %s %s %s -v' % (obj,apred,telescope)
        if clobber:
            cmd += ' -c'
        # Check if file exists already
        dorv[i] = True
        if clobber==False:
            if os.path.exists(apstarfile):
                logger.info(os.path.basename(apstarfile)+' already exists and clobber==False')
                dorv[i] = False
        if dorv[i]:
            logger.info('rv %d : %s' % (i+1,obj))
            logger.info('Command : '+cmd)
            logger.info('Logfile : '+logfile)
            queue.append(cmd,outfile=logfile,errfile=errfile)
    if np.sum(dorv)>0:
        logger.info('Running RV on '+str(np.sum(dorv))+' stars')
        queue.commit(hard=True,submit=True)
        logger.info('PBS key is '+queue.key)
        runapogee.queue_wait(queue,sleeptime=60,verbose=True,logger=logger)  # wait for jobs to complete
        # This checks the status and puts it into the database
        ind, = np.where(dorv)
        chkrv = runapogee.check_rv(vcat[ind],queue.key,logger=logger,verbose=False)
    else:
        logger.info('No RVs need to be run')
        chkrv = []
    del queue

    # -- Summary statistics --
    # RV status
    if len(chkrv)>0:
        indrv, = np.where(chkrv['success']==True)
        logger.info('%d/%d RV+visit combination successfully processed' % (len(indrv),len(chkrv)))
    
    return chkrv

def runsumfiles(load,mjds,logger=None):
    """
    Create the MJD summary files for the relevant MJDS and final summary file of
    all nights.

    Parameters
    ----------
    load : ApLoad object
       ApLoad object that contains "apred" and "telescope".
    mjds : list
       List of MJDs to process
    logger : logger, optional
       Logging object.  If not is input, then a default one will be created.

    Returns
    -------
    Output the summary files for the relevant MJDs.

    Example
    -------

    runsumfiles(load,mjds)

    """

    if logger is None:
        logger = dln.basiclogger()
    
    apred = load.apred
    telescope = load.telescope

    # Loop over MJDs and create the MJD-level summary files
    #for m in mjds:
    #    runapogee.create_sumfiles_mjd(apred,telescope,m,logger=logger)
    # Create allStar/allVisit file
    runapogee.create_sumfiles(apred,telescope,logger=logger)


def rununified(load,mjds,slurm,clobber=False,logger=None):
    """
    Create the unified MWM directory structure for the relevant MJDs.

    Parameters
    ----------
    load : ApLoad object
       ApLoad object that contains "apred" and "telescope".
    mjds : list
       List of MJDs to process
    slurm : dictionary
       Dictionary of slurm settings.
    clobber : boolean, optional
       Overwrite existing files.  Default is False.
    logger : logger, optional
       Logging object.  If not is input, then a default one will be created.

    Returns
    -------
    The unified directory structure software is run on the MJDs and relevant symlinks are generated.

    Example
    -------

    rununified(load,mjds)

    """

    if logger is None:
        logger = dln.basiclogger()
    
    apred = load.apred
    telescope = load.telescope
    observatory = telescope[0:3]
    mjdstart = np.min(mjds)
    mjdstop = np.max(mjds)
    logtime = datetime.now().strftime("%Y%m%d%H%M%S")
    
    slurm1 = slurm.copy()
    if len(mjds)<64:
        slurm1['cpus'] = len(mjds)
    slurm1['numpy_num_threads'] = 2    
    logger.info('Slurm settings: '+str(slurm1))
    queue = pbsqueue(verbose=True)
    queue.create(label='unidir', **slurm1)
    # Loop over all MJDs
    for m in mjds:
        outfile = os.environ['APOGEE_REDUX']+'/'+apred+'/log/'+observatory+'/'+str(mjd5)+'-unidir.'+logtime+'.log'
        errfile = outfile.replace('.log','.err')
        if os.path.exists(os.path.dirname(outfile))==False:
            os.makedirs(os.path.dirname(outfile))
        queue.append('sas_mwm_healpix --spectro apogee --mjd {0} --telescope {1} --drpver {2} -v'.format(mjd5,telescope,apred),
                     outfile=outfile, errfile=errfile)
    queue.commit(hard=True,submit=True)
    logger.info('PBS key is '+queue.key)        
    runapogee.queue_wait(queue,sleeptime=60,verbose=True,logger=logger)  # wait for jobs to complete
    del queue    
    #  sas_mwm_healpix --spectro apogee --mjd 59219 --telescope apo25m --drpver daily -v


def runqa(load,mjds,slurm,clobber=False,logger=None):
    """
    Run QA on a list of MJDs.

    Parameters
    ----------
    load : ApLoad object
       ApLoad object that contains "apred" and "telescope".
    mjds : list
       List of MJDs to process
    clobber : boolean, optional
       Overwrite existing files.  Default is False.
    logger : logger, optional
       Logging object.  If not is input, then a default one will be created.

    Returns
    -------
    The QA code is run on the MJDs and relevant plots and HTML pages are created.

    Example
    -------

    runqa(load,mjds)

    """

    apred = load.apred
    telescope = load.telescope
    instrument = load.instrument
    observatory = telescope[0:3]
    mjdstart = np.min(mjds)
    mjdstop = np.max(mjds)
    logtime = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Get plan files for these MJDs
    planfiles = getplanfiles(load,mjds,logger=logger)
    # Only want apPlan files
    if len(planfiles)>0:
        planfiles = [p for p in planfiles if os.path.basename(p).startswith('apPlan')]
    if len(planfiles)==0:
        logger.info('No plan files')
        return
    logger.info(str(len(planfiles))+' plan file(s)')

    # Run apqa on each plate visit
    slurm1 = slurm.copy()
    if len(planfiles)<64:
        slurm1['cpus'] = len(planfiles)
    slurm1['numpy_num_threads'] = 2    
    logger.info('Slurm settings: '+str(slurm1))
    queue = pbsqueue(verbose=True)
    queue.create(label='apqa', **slurm1)
    for i,pf in enumerate(planfiles):
        logger.info('planfile %d : %d' % (i+1,pf))
        fdir = os.path.dirname(pf)
        # apPlan-1491-59587.yaml
        base = os.path.basename(pf)
        dum = base.split('-')
        plate = dum[1]
        mjd = dum[2].split('.')[0]
        logfile = fdir+'/apqa-'+plate+'-'+mjd+'_pbs.'+logtime+'.log'
        errfile = logfile.replace('.log','.err')
        cmd = 'apqa {0} {1} --apred {2} --telescope {3} --plate {4}'.format(mjd,observatory,apred,telescope,plate)
        cmd += ' --masterqa False --starhtml False --starplots False --nightqa False --monitor False'
        logger.info('Command : '+cmd)
        logger.info('Logfile : '+logfile)
        queue.append(cmd, outfile=logfile, errfile=errfile)
    queue.commit(hard=True,submit=True)
    logger.info('PBS key is '+queue.key)
    runapogee.queue_wait(queue,sleeptime=60,logger=logger,verbose=True)  # wait for jobs to complete 
    del queue

    # Make nightly QA/summary pages
    # we should parallelize this
    for m in mjds:
        try:
            apodir = os.environ.get('APOGEE_REDUX')+'/'
            qa.makeNightQA(load=load,mjd=str(m),telescope=telescope,apred=apred)
            # Run makeCalFits, makeDarkFits, makeExpFits
            # makeCalFits
            expinfo = getexpinfo(load,m,logger=logger,verbose=False)
            calind, = np.where((expinfo['exptype']=='ARCLAMP') | (expinfo['exptype']=='QUARTZFLAT') |
                               (expinfo['exptype']=='DOMEFLAT'))
            if len(calind)>0:
                all_ims = expinfo['num'][calind]
                qa.makeCalFits(load=load, ims=all_ims, mjd=str(m), instrument=instrument, clobber=clobber)
            # makeDarkFits
            darkind, = np.where(expinfo['exptype']=='DARK')
            if len(darkind)>0:
                all_ims = expinfo['num'][darkind]
                qa.makeDarkFits(load=load, ims=all_ims, mjd=str(m), clobber=clobber)
            qa.makeExpFits(instrument=instrument, apodir=apodir, apred=apred, load=load, mjd=str(m), clobber=clobber)
        except:
            traceback.print_exc()    
    # Make final mjd/fields pages
    try:
        qa.makeMasterQApages(mjdmin=59146, mjdmax=9999999, apred=apred,
                             mjdfilebase='mjd.html',fieldfilebase='fields.html',
                             domjd=True, dofields=True)
    except:
        traceback.print_exc()

    # Run monitor page
    #  always runs on all MJDs
    monitor.monitor()


def summary_email(observatory,apred,mjd,steps,chkmaster=None,chk3d=None,chkcal=None, 
                  planfiles=None,chkapred=None,chkrv=None,logfile=None,slurm=None,
                  clobber=None,debug=False):   
    """ Send a summary email."""

    mjds = loadmjd(mjd)
    nmjd = len(mjds)
    mjdstart = np.min(mjds)
    mjdstop = np.max(mjds)
    address = 'apogee-pipeline-log@sdss.org'
    if debug:
        address = 'dnidever@montana.edu'
    subject = 'APOGEE DRP Reduction %s %s %s-%s' % (observatory,apred,mjdstart,mjdstop)
    message = """\
              <html>
                <body>
              """

    message += '<b>APOGEE DRP Reduction %s %s %s</b><br>\n' % (observatory,apred,str(mjd))
    message += str(nmjd)+' MJDs: '+','.join(np.char.array(mjds).astype(str))+'<br>\n'
    message += 'Steps: '+','.join(steps)+'<br>\n'
    if clobber:
        message += 'clobber: '+str(clobber)+'<br>\n'
    if slurm:
        message += 'Slurm settings: '+str(slurm)+'<br>\n'
    message += '<p>\n'
    message += '<a href="https://data.sdss.org/sas/sdss5/mwm/apogee/spectro/redux/'+str(apred)+'/qa/mjd.html">QA Webpage (MJD List)</a><br> \n'

    # Master Cals step
    if 'master' in steps and chkmaster:
        ind, = np.where(chkmaster['success']==True)
        message += '%d/%d Master calibrations successfully processed<br> \n' % (len(ind),len(chkmaster))
        
    # AP3D step
    if '3d' in steps and chk3d:
        ind, = np.where(chk3d['success']==True)
        message += 'AP3D: %d/%d exposures successfully processed through AP3D<br> \n' % (len(ind),len(chk3d))

    # Daily Cals step
    if 'cal' in steps and chkcal:
        ind, = np.where(chkcal['success']==True)
        message += 'Cal: %d/%d daily calibrations successfully processed<br> \n' % (len(ind),len(chkcal))
 
    # Plan files
    if 'plan' in steps and planfiles:
        message += 'Plan: %d plan files successfully made<br> \n' % len(planfiles)

    # APRED step
    if 'apred' in steps and chkapred:
        ind, = np.where(chkapred['success']==True)
        message += 'APRED: %d/%d visits successfully processed<br> \n' % (len(ind),len(chkapred))

    # RV step
    if 'rv' in steps and chkrv:
        ind, = np.where(chkrv['success']==True)
        message += 'RV: %d/%d RV+visit combination successfully processed<br> \n' % (len(ind),len(chkrv))

    message += """\
                 </p>
                 </body>
               </html>
               """

    # Send the message
    email.send(address,subject,message,files=logfile,send_from='noreply.apogeedrp')
    

def run(observatory,apred,mjd=None,steps=None,clobber=False,fresh=False,
        linkvers=None,nodes=5,debug=False):
    """
    Perform APOGEE Data Release Processing

    Parameters
    ----------
    observatory : str
       The observatory: "apo" or "lco".
    apred : str
       Reduction version name.
    mjd : str, list, optional
       Set of MJDs to run.  This can be (a) a string with a date range (e.g., '59610-59650'), or
         (b) a string with a comma-separated list (e.g. '59610,59610,59650'), (c) a python list of
         MJDs, or (d) a combination of all of the above.
         By default, all SDSS-V MJDs are run.
    steps : list, optional
       Processing steps to perform.  The full list is:
         ['setup','master','3d','cal','plan','apred','rv','summary','unified','qa']
         By default, all steps are run.
    clobber : boolean, optional
       Overwrite any existing data.  Default is False.
    fresh : boolean, optional
       Start the reduction directory fresh.  The default is continue with what is
         already there.
    linkvers : str, optional
       Name of reduction version to use for symlinks for the calibration files.
    nodes : int, optional
       Number of nodes to use on the CHPC.  Default is 5.
    debug : boolean, optional
       For testing purposes.  Default is False.

    Returns
    -------
    Nothing is returned.  The APOGEE data are reduced on disk.

    Example
    -------

    run('apo','v1.1',mjd=[54566,56666])

    """

    begtime = str(datetime.now())

    telescope = observatory+'25m'
    instrument = {'apo':'apogee-n','lco':'apogee-s'}[observatory]

    # MJDs to process
    mjds = loadmjd(mjd)
    mjds = np.sort(np.array(mjds).astype(int))
    nmjd = len(mjds)
    mjdstart = np.min(mjds)
    mjdstop = np.max(mjds)

    # Reduction steps
    # The default is to do all
    steps = loadsteps(steps)
    nsteps = len(steps)

    # Slurm settings
    alloc = 'sdss-np'
    shared = True
    ppn = 64
    walltime = '336:00:00'
    # Only set cpus if you want to use less than 64 cpus
    slurm = {'nodes':nodes, 'alloc':alloc, 'shared':shared, 'ppn':ppn,
             'walltime':walltime, 'notification':False}
    
    # Get software version (git hash)
    gitvers = plan.getgitvers()

    load = apload.ApLoad(apred=apred,telescope=telescope)

    # Reduction logs directory
    logdir = os.environ['APOGEE_REDUX']+'/'+apred+'/log/'+observatory+'/'
    if os.path.exists(logdir)==False:
        os.makedirs(logdir)

    # Data directory 
    datadir = {'apo':os.environ['APOGEE_DATA_N'],'lco':os.environ['APOGEE_DATA_S']}[observatory]

    # Starting fresh
    #  need to do this before we start the log file
    if 'setup' in steps and fresh:
        print('Starting '+str(apred)+' fresh')
        apogee_redux = os.environ['APOGEE_REDUX']+'/'
        if os.path.exists(apogee_redux+apred):
            shutil.rmtree(apogee_redux+apred)
        os.makedirs(logdir)        

    # Set up logging to screen and logfile
    logFormatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s]  %(message)s")
    rootLogger = logging.getLogger() 
    while rootLogger.hasHandlers(): # some existing loggers, remove them   
        rootLogger.removeHandler(rootLogger.handlers[0]) 
    rootLogger = logging.getLogger()
    logtime = datetime.now().strftime("%Y%m%d%H%M%S") 
    if mjdstart==mjdstop:
        logfile = logdir+'apogeedrp-'+str(mjdstart)+'.'+logtime+'.log'
    else:
        logfile = logdir+'apogeedrp-'+str(mjdstart)+'-'+str(mjdstop)+'.'+logtime+'.log'
    if os.path.exists(logfile): os.remove(logfile)
    fileHandler = logging.FileHandler(logfile)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)
    rootLogger.setLevel(logging.NOTSET)

    rootLogger.info('Running APOGEE DRP for '+str(observatory).upper()+' APRED='+apred)
    rootLogger.info('MJD: '+str(mjd))
    rootLogger.info(str(nmjd)+' MJDs: '+','.join(np.char.array(mjds).astype(str)))
    rootLogger.info(str(nsteps)+' steps: '+','.join(steps))
    rootLogger.info('Clobber: '+str(clobber))
    if fresh:
        rootLogger.info('Starting '+str(apred)+' fresh')
    rootLogger.info('Slurm settings: '+str(slurm))

    # Common keyword arguments
    kws = {'slurm':slurm, 'clobber':clobber, 'logger':rootLogger}

    # Defaults for check tables
    chkmaster,chk3d,chkcal,planfiles,chkapred,chkrv = None,None,None,None,None,None

    # 1) Setup the directory structure
    #----------------------------------
    if 'setup' in steps:
        rootLogger.info('')
        rootLogger.info('-------------------------------------')
        rootLogger.info('1) Setting up the directory structure')
        rootLogger.info('=====================================')
        rootLogger.info('')
        queue = pbsqueue(verbose=True)
        queue.create(label='mkvers', nodes=1, alloc=alloc, ppn=ppn, cpus=1, shared=shared, walltime=walltime, notification=False)
        mkvoutfile = os.environ['APOGEE_REDUX']+'/'+apred+'/log/mkvers.'+logtime+'.log'
        mkverrfile = mkvoutfile.replace('-mkvers.log','-mkvers.'+logtime+'.err')
        if os.path.exists(os.path.dirname(mkvoutfile))==False:
            os.makedirs(os.path.dirname(mkvoutfile))
        cmd = 'mkvers {0}'.format(apred)
        rootLogger.info('Command : '+cmd)
        rootLogger.info('Logfile : '+mkvoutfile)
        queue.append(cmd,outfile=mkvoutfile, errfile=mkverrfile)
        queue.commit(hard=True,submit=True)
        runapogee.queue_wait(queue,sleeptime=30,logger=rootLogger,verbose=True)  # wait for jobs to complete 
        del queue    

    # 2) Master calibration products, make sure to do them in the right order
    #------------------------------------------------------------------------
    if 'master' in steps:
        rootLogger.info('')
        rootLogger.info('-----------------------------------------')
        rootLogger.info('2) Generating master calibration products')
        rootLogger.info('=========================================')
        rootLogger.info('')
        chkmaster = mkmastercals(load,mjds,linkvers=linkvers,**kws)

    # 3) Process all exposures through ap3d
    #---------------------------------------
    if '3d' in steps:
        rootLogger.info('')
        rootLogger.info('--------------------------------')
        rootLogger.info('3) Running AP3D on all exposures')
        rootLogger.info('================================')
        rootLogger.info('')
        chk3d = runap3d(load,mjds,**kws)

    # 4) Make all daily cals (domeflats, quartzflats, arclamps, FPI)
    #----------------------------------------------------------------
    if 'cal' in steps:
        rootLogger.info('')
        rootLogger.info('----------------------------------------')
        rootLogger.info('5) Generating daily calibration products')
        rootLogger.info('========================================')
        rootLogger.info('')
        chkcal = rundailycals(load,mjds,**kws)

    # 5) Make plan files
    #-------------------
    if 'plan' in steps:
        rootLogger.info('')
        rootLogger.info('--------------------')
        rootLogger.info('6) Making plan files')
        rootLogger.info('====================')
        rootLogger.info('')
        planfiles = makeplanfiles(load,mjds,**kws)

    # 6) Run APRED on all of the plan files (ap3d-ap1dvisit), go through each MJD chronologically
    #--------------------------------------------------------------------------------------------
    if 'apred' in steps:
        rootLogger.info('')
        rootLogger.info('----------------')
        rootLogger.info('7) Running APRED')
        rootLogger.info('================')
        rootLogger.info('')
        chkapred = runapred(load,mjds,**kws)
        
    # 7) Run "rv" on all unique stars
    #--------------------------------
    if 'rv' in steps:
        rootLogger.info('')
        rootLogger.info('--------------------------------')
        rootLogger.info('8) Running RV+Visit Combination')
        rootLogger.info('================================')
        rootLogger.info('')
        chkrv = runrv(load,mjds,**kws)

    # 8) Create full allVisit/allStar files
    #--------------------------------------
    if 'summary' in steps:
        rootLogger.info('')
        rootLogger.info('-----------------------')
        rootLogger.info('9) Create summary files')
        rootLogger.info('=======================')
        rootLogger.info('')
        runsumfiles(load,mjds,logger=rootLogger)
    
    # 9) Unified directory structure
    #-------------------------------
    if 'unified' in steps:
        rootLogger.info('')
        rootLogger.info('---------------------------------------------')
        rootLogger.info('10) Generating unified MWM directory structure')
        rootLogger.info('=============================================')
        rootLogger.info('')
        #rununified(load,mjds,**kws)

    # 10) Run QA script
    #------------------
    if 'qa' in steps:
        rootLogger.info('')
        rootLogger.info('--------------')
        rootLogger.info('11) Running QA')
        rootLogger.info('==============')
        rootLogger.info('')
        runqa(load,mjds,**kws)


    # Update daily_status table
    #daycat = np.zeros(1,dtype=np.dtype([('pk',int),('mjd',int),('telescope',(np.str,10)),('nplanfiles',int),
    #                                    ('nexposures',int),('begtime',(np.str,50)),('endtime',(np.str,50)),('success',bool)]))
    #daycat['mjd'] = mjd5
    #daycat['telescope'] = telescope
    #daycat['nplanfiles'] = len(planfiles)
    #daycat['nexposures'] = len(expinfo)
    #daycat['begtime'] = begtime
    #daycat['endtime'] = str(datetime.now())
    #daysuccess = True
    #if chkvisit is not None:
    #    daysuccess &= (np.sum(chkvisit['success']==False)==0)
    #if chkrv is not None:
    #    daysuccess &= (np.sum(chkrv['success']==False)==0)
    #daycat['success'] = daysuccess
    #dayout = db.query('daily_status',where="mjd="+str(mjd5)+" and telescope='"+telescope+"' and begtime='"+begtime+"'")
    #daycat['pk'] = dayout['pk'][0]
    #db.update('daily_status',daycat)


    ## UPDATE THE DATABASE!!!
    
    rootLogger.info('APOGEE DRP reduction finished for %s APRED=%s MJD=%d to MJD=%d' % (observatory,apred,mjdstart,mjdstop))


    # Summary email
    summary_email(observatory,apred,mjd,steps,chkmaster=chkmaster,chk3d=chk3d,chkcal=chkcal,
                  planfiles=planfiles,chkapred=chkapred,chkrv=chkrv,logfile=logfile,slurm=slurm,
                  clobber=clobber,debug=debug)

