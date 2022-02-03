import copy
import numpy as np
import os
import shutil
from glob import glob
import pdb

from dlnpyutils import utils as dln
from ..utils import spectra,yanny,apload,platedata,plan,email
from ..apred import mkcal, cal
from ..database import apogeedb
from . import mkplan,runapogee,check
from sdss_access.path import path
from astropy.io import fits
from astropy.table import Table
from collections import OrderedDict
#from astropy.time import Time
from datetime import datetime
import logging
from slurm import queue as pbsqueue
import time

def mkvers(apred,fresh=False):
    """
    Setup APOGEE DRP directory structure.
   
    Parameters
    ----------
    apred : str
       Reduction version name.
    fresh : boolean, optional
       Start the reduction directory fresh.  The default is continue with what is
         already there.
    
    Returns
    -------
    It makes the directory structure for a new DRP reduction version.

    Example
    -------

    mkvers('v1.0.0')

    """

    apogee_redux = os.environ['APOGEE_REDUX']+'/'
    apogee_drp_dir = os.environ['APOGEE_DRP_DIR']+'/'

    print('Setting up directory structure for APOGEE DRP version = ',vers)

    # Start fresh
    if args.fresh:
        print('Starting fresh')
        if os.path.exists(apogee_redux+vers):
            shutil.rmtree(apogee_redux+vers)

    # Main directory
    if os.path.exists(apogee_redux+vers)==False:
        print('Creating ',apogee_redux+vers)
        os.makedirs(apogee_redux+vers)
    else:
        print(apogee_redux+vers,' already exists')
    # First level
    for d in ['cal','exposures','stars','fields','visit','qa','plates','monitor','summary','log']:
        if os.path.exists(apogee_redux+vers+'/'+d)==False:
            print('Creating ',apogee_redux+vers+'/'+d)
            os.makedirs(apogee_redux+vers+'/'+d)
        else:
            print(apogee_redux+vers+'/'+d,' already exists')
    # North/south subdirectories
    for d in ['cal','exposures','monitor']:
        for obs in ['apogee-n','apogee-s']:
            if os.path.exists(apogee_redux+vers+'/'+d+'/'+obs)==False:
                print('Creating ',apogee_redux+vers+'/'+d+'/'+obs)
                os.makedirs(apogee_redux+vers+'/'+d+'/'+obs)
            else:
                print(apogee_redux+vers+'/'+d+'/'+obs,' already exists')
    for d in ['visit','stars','fields']:
        for obs in ['apo25m','lco25m']:
            if os.path.exists(apogee_redux+vers+'/'+d+'/'+obs)==False:
                print('Creating ',apogee_redux+vers+'/'+d+'/'+obs)
                os.makedirs(apogee_redux+vers+'/'+d+'/'+obs)
            else:
                print(apogee_redux+vers+'/'+d+'/'+obs,' already exists')
    for d in ['log']:
        for obs in ['apo','lco']:
            if os.path.exists(apogee_redux+vers+'/'+d+'/'+obs)==False:
                print('Creating ',apogee_redux+vers+'/'+d+'/'+obs)
                os.makedirs(apogee_redux+vers+'/'+d+'/'+obs)
            else:
                print(apogee_redux+vers+'/'+d+'/'+obs,' already exists')
    # Cal subdirectories
    for d in ['bpm','darkcorr','detector','flatcorr','flux','fpi','html','littrow','lsf','persist','plans','psf','qa','telluric','trace','wave']:
        for obs in ['apogee-n','apogee-s']:
            if os.path.exists(apogee_redux+vers+'/cal/'+obs+'/'+d)==False:
                print('Creating ',apogee_redux+vers+'/cal/'+obs+'/'+d)
                os.makedirs(apogee_redux+vers+'/cal/'+obs+'/'+d)
            else:
                print(apogee_redux+vers+'/cal/'+obs+'/'+d,' already exists')

    # Webpage files
    #if os.path.exists(apogee_drp_dir+'etc/htaccess'):
    #    os.copy(apogee_drp_dir+'etc/htaccess',apogee_redux+vers+'qa/.htaccess'
    if os.path.exists(apogee_drp_dir+'etc/sorttable.js') and os.path.exists(apogee_redux+vers+'/qa/sorttable.js')==False:
        print('Copying sorttable.js')
        shutil.copyfile(apogee_drp_dir+'etc/sorttable.js',apogee_redux+vers+'/qa/sorttable.js')



def mkmastercals(apred,telescope,clobber=False,links=None,logger=None):
    """
    Make the master calibration products for a reduction version.

    Parameters
    ----------
    apred : str
       Reduction version name.
    telescope : str
       The telescope name: apo25m or lco25m.
    clobber : boolean, optional
       Overwrite any existing files.  Default is False.
    links : str, optional
       Name of reduction version to use for symlinks for the calibration files.    

    Returns
    -------
    All the master calibration products are made for a reduction version.

    Example
    -------

    mkmastercals('v1.0.0','telescope')

    """

    if logger is None:
        logger = dln.basiclogger()

    nodes = 1
    alloc = 'sdss-np'
    shared = True
    ppn = 64
    cpus = 32
    walltime = '10-00:00:00'

    load = apload.ApLoad(apred=apred,telescope=telescope)
    apogee_redux = os.environ['APOGEE_REDUX']+'/'
    apogee_drp_dir = os.environ['APOGEE_DRP_DIR']+'/'
    

    # Symbolic links to another version
    if links is not None:
        linkvers = links
        logger.info('Creating calibration product symlinks to version >>'+linkvers+'<<')
        cwd = os.path.abspath(os.curdir)
        for d in ['bpm','darkcorr','detector','flatcorr','fpi','littrow','lsf','persist','telluric','wave']:
            for obs in ['apogee-n','apogee-s']:    
                logger.info('Creating symlinks for '+apogee_redux+vers+'/cal/'+obs+'/'+d)
                os.chdir(apogee_redux+vers+'/cal/'+obs+'/'+d)
                subprocess.run(['ln -s '+apogee_redux+linkvers+'/cal/'+obs+'/'+d+'/*fits .'],shell=True)
        return


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


    # Make darks sequentially
    #-------------------------
    # they take too much memory to run in parallel
    #idl -e "makecal,dark=1,vers='$vers',telescope='$telescope'" >& log/mkdark-$telescope.$host.log
    #darkplot --apred $vers --telescope $telescope
    darkdict = allcaldict['dark']
    logger.info('')
    logger.info('--------------------------------')
    logger.info('Making master darks sequentially')
    logger.info('================================')
    logger.info(str(len(darkdict))+' Darks to make: '+','.join(darkdict['name']))
    logger.info('')
    queue = pbsqueue(verbose=True)
    queue.create(label='mkdark', nodes=1, alloc=alloc, ppn=ppn, qos=qos, cpus=1, shared=shared, walltime=walltime, notification=False)
    for i in range(len(darkdict)):
        outfile1 = os.environ['APOGEE_REDUX']+'/'+apred+'/log/mkdark-'+str(darkdict['name'][i])+telescope+'.'+logtime+'.log'
        errfile1 = mkvoutfile.replace('.log','.err')
        if os.path.exists(os.path.dirname(outfile1))==False:
            os.makedirs(os.path.dirname(outfile1))
        cmd = 'makecal --vers {0} --telescope {1}'.format(apred,telescope)
        cmd += ' --dark '+str(darkdict['name'][i])+' --unlock'
        queue.append(cmd,outfile=outfile1, errfile=errfile1)
    queue.commit(hard=True,submit=True)
    queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
    del queue    
    # Make the dark plots
    cal.darkplot(apred=apred,telescope=telescope)

    # Make flats sequentially
    #-------------------------
    #idl -e "makecal,flat=1,vers='$vers',telescope='$telescope'" >& log/mkflat-$telescope.$host.log
    #flatplot --apred $vers --telescope $telescope
    flatdict = allcaldict['flat']
    logger.info('')
    logger.info('--------------------------------')
    logger.info('Making master flats sequentially')
    logger.info('================================')
    logger.info(str(len(flatdict))+' Flats to make: '+','.join(flatdict['name']))
    logger.info('')
    queue = pbsqueue(verbose=True)
    queue.create(label='mkflat', nodes=1, alloc=alloc, ppn=ppn, qos=qos, cpus=1, shared=shared, walltime=walltime, notification=False)
    for i in range(len(flatdict)):
        outfile1 = os.environ['APOGEE_REDUX']+'/'+apred+'/log/mkflat-'+str(flatdict['name'][i])+telescope+'.'+logtime+'.log'
        errfile1 = mkvoutfile.replace('.log','.err')
        if os.path.exists(os.path.dirname(outfile1))==False:
            os.makedirs(os.path.dirname(outfile1))
        cmd = 'makecal --vers {0} --telescope {1}'.format(apred,telescope)
        cmd += ' --flat '+str(flatdict['name'][i])+' --unlock'
        queue.append(cmd,outfile=outfile1, errfile=errfile1)
    queue.commit(hard=True,submit=True)
    queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
    del queue    
    # Make the flat plots
    cal.flatplot(apred=apred,telescope=telescope)

    # Make BPM sequentially
    #----------------------
    #idl -e "makecal,bpm=1,vers='$vers',telescope='$telescope'" >& log/mkbpm-$telescope.$host.log
    bpmdict = allcaldict['bpm']
    logger.info('')
    logger.info('--------------------------------')
    logger.info('Making master BPMs sequentially')
    logger.info('================================')
    logger.info(str(len(bpmdict))+' BPMs to make: '+','.join(bpmdict['name']))
    logger.info('')
    queue = pbsqueue(verbose=True)
    queue.create(label='mkbpm', nodes=1, alloc=alloc, ppn=ppn, qos=qos, cpus=1, shared=shared, walltime=walltime, notification=False)
    for i in range(len(bpmdict)):
        outfile1 = os.environ['APOGEE_REDUX']+'/'+apred+'/log/mkbpm-'+str(bpmdict['name'][i])+telescope+'.'+logtime+'.log'
        errfile1 = mkvoutfile.replace('.log','.err')
        if os.path.exists(os.path.dirname(outfile1))==False:
            os.makedirs(os.path.dirname(outfile1))
        cmd = 'makecal --vers {0} --telescope {1}'.format(apred,telescope)
        cmd += ' --bpm '+str(bpmdict['name'][i])+' --unlock'
        queue.append(cmd,outfile=outfile1, errfile=errfile1)
    queue.commit(hard=True,submit=True)
    queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
    del queue    

    # Make Littrow sequentially
    #--------------------------
    #idl -e "makecal,littrow=1,vers='$vers',telescope='$telescope'" >& log/mklittrow-$telescope.$host.log
    littdict = allcaldict['littrow']
    logger.info('')
    logger.info('-----------------------------------')
    logger.info('Making master Littrows sequentially')
    logger.info('===================================')
    logger.info(str(len(littdict))+' Littrows to make: '+','.join(littdict['name']))
    logger.info('')
    queue = pbsqueue(verbose=True)
    queue.create(label='mklittrow', nodes=1, alloc=alloc, ppn=ppn, qos=qos, cpus=1, shared=shared, walltime=walltime, notification=False)
    for i in range(len(littdict)):
        outfile1 = os.environ['APOGEE_REDUX']+'/'+apred+'/log/mklittrow-'+str(littdict['name'][i])+telescope+'.'+logtime+'.log'
        errfile1 = mkvoutfile.replace('.log','.err')
        if os.path.exists(os.path.dirname(outfile1))==False:
            os.makedirs(os.path.dirname(outfile1))
        cmd = 'makecal --vers {0} --telescope {1}'.format(apred,telescope)
        cmd += ' --littrow '+str(littdict['name'][i])+' --unlock'
        queue.append(cmd,outfile=outfile1, errfile=errfile1)
    queue.commit(hard=True,submit=True)
    queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
    del queue    

    # Make Response sequentially
    #--------------------------
    responsedict = allcaldict['response']
    logger.info('')
    logger.info('------------------------------------')
    logger.info('Making master responses sequentially')
    logger.info('====================================')
    logger.info(str(len(responsedict))+' Responses to make: '+','.join(responsedict['name']))
    logger.info('')
    queue = pbsqueue(verbose=True)
    queue.create(label='mkresponse', nodes=1, alloc=alloc, ppn=ppn, qos=qos, cpus=1, shared=shared, walltime=walltime, notification=False)
    for i in range(len(responsedict)):
        outfile1 = os.environ['APOGEE_REDUX']+'/'+apred+'/log/mkresponse-'+str(responsedict['name'][i])+telescope+'.'+logtime+'.log'
        errfile1 = mkvoutfile.replace('.log','.err')
        if os.path.exists(os.path.dirname(outfile1))==False:
            os.makedirs(os.path.dirname(outfile1))
        cmd = 'makecal --vers {0} --telescope {1}'.format(apred,telescope)
        cmd += ' --response '+str(responsedict['name'][i])+' --unlock'
        queue.append(cmd,outfile=outfile1, errfile=errfile1)
    queue.commit(hard=True,submit=True)
    queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
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

    multiwavedict = allcaldict['multiwave']
    logger.info('')
    logger.info('-----------------------------------')
    logger.info('Making master multiwave in parallel')
    logger.info('===================================')
    logger.info(str(len(multiwavedict))+' multiwave to make: '+','.join(multiwavedict['name']))
    logger.info('')
    queue = pbsqueue(verbose=True)
    queue.create(label='mkmultiwave', nodes=1, alloc=alloc, ppn=ppn, qos=qos, cpus=5, shared=shared, walltime=walltime, notification=False)
    for i in range(len(multiwavedict)):
        outfile1 = os.environ['APOGEE_REDUX']+'/'+apred+'/log/mkmultiwave-'+str(multiwavedict['name'][i])+telescope+'.'+logtime+'.log'
        errfile1 = mkvoutfile.replace('.log','.err')
        if os.path.exists(os.path.dirname(outfile1))==False:
            os.makedirs(os.path.dirname(outfile1))
        cmd = 'makecal --vers {0} --telescope {1}'.format(apred,telescope)
        cmd += ' --multiwave '+str(multiwavedict['name'][i])+' --unlock'
        queue.append(cmd,outfile=outfile1, errfile=errfile1)
    queue.commit(hard=True,submit=True)
    queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
    del queue    


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
    logger.info(str(len(lsfdict))+' LSFs to make: '+','.join(lsfdict['name']))
    logger.info('')
    queue = pbsqueue(verbose=True)
    queue.create(label='mklsf', nodes=1, alloc=alloc, ppn=ppn, qos=qos, cpus=5, shared=shared, walltime=walltime, notification=False)
    for i in range(len(littdict)):
        outfile1 = os.environ['APOGEE_REDUX']+'/'+apred+'/log/mklsf-'+str(lsfdict['name'][i])+telescope+'.'+logtime+'.log'
        errfile1 = mkvoutfile.replace('.log','.err')
        if os.path.exists(os.path.dirname(outfile1))==False:
            os.makedirs(os.path.dirname(outfile1))
        cmd = 'makecal --vers {0} --telescope {1}'.format(apred,telescope)
        cmd += ' --lsf '+str(lsfdict['name'][i])+' --full --pl --unlock'
        queue.append(cmd,outfile=outfile1, errfile=errfile1)
    queue.commit(hard=True,submit=True)
    queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
    del queue    


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
    allsteps = ['setup','master','3d','check','cals','plans','apred','rv','summary','unified','qa']
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
    return steps

def run(observatory,apred,mjd=None,steps=None,qos='sdss',fresh=False,links=None):
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
    qos : str, optional
       The pbs queue to use.  Default is "sdss".
    fresh : boolean, optional
       Start the reduction directory fresh.  The default is continue with what is
         already there.
    links : str, optional
       Name of reduction version to use for symlinks for the calibration files.

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
    allsteps = ['setup','master','3d','check','cals','plans','apred','rv','summary','unified','qa']
    steps = loadsteps(steps)
    nsteps = len(steps)

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

    rootLogger.info('Running APOGEE DRP for '+str(observatory).upper()+' apred='+apred)
    rootLogger.info(str(nmjd)+' MJDs: '+','.join(np.char.array(mjds).astype(str)))
    rootLogger.info(str(nsteps)+' steps: '+','.join(steps))

    # Initialize the DB connection
    db = apogeedb.DBSession()

    # Get information for all of the exposures
    expinfo = db.query('exposure',where="mjd>=%d and mjd<=%d and observatory='%s'" % (mjdstart,mjdstop,observatory))
    ind = []
    for m in mjds:
        gd, = np.where(expinfo['mjd']==m)
        if len(gd)>0:
            ind += list(gd)
    ind = np.array(ind)
    if len(ind)>0:
        expinfo = expinfo[ind]
    nexp = len(expinfo)
    rootLogger.info(str(nexp)+' exposures')

    # 1) Setup the directory structure
    #----------------------------------
    if 'setup' in steps:
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
    if 'master' in steps:
        rootLogger.info('')
        rootLogger.info('-----------------------------------------')
        rootLogger.info('2) Generating master calibration products')
        rootLogger.info('=========================================')
        rootLogger.info('')
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
        mkmastercals(apred,telescope,clobber=clobber,links=links,logger=rootLogger)


        # -- Daily calibration products ---
        # PSF/EPSF/Trace, from domeflat or quartzflat
        # Flux, from domeflat or quartzflat
        # Wave, from arclamps
        

        # Maybe have an option to copy/symlink them from a previous apred version


    # 3) Process all exposures through ap3d
    #---------------------------------------
    if '3d' in steps:
        rootLogger.info('')
        rootLogger.info('--------------------------------')
        rootLogger.info('3) Running AP3D on all exposures')
        rootLogger.info('================================')
        rootLogger.info('')
        if len(expinfo)>0:
            queue = pbsqueue(verbose=True)
            queue.create(label='ap3d', nodes=nodes, alloc=alloc, ppn=ppn, cpus=np.minimum(cpus,len(expinfo)),
                         qos=qos, shared=shared, numpy_num_threads=2, walltime=walltime, notification=False)
            for num in expinfo['num']:
                outfile = load.filename('2D',num=num,mjd=mjd5,chips=True).replace('2D','3D')
                outfile = os.path.dirname(outfile)+'/logs/'+os.path.basename(outfile)
                if os.path.dirname(outfile)==False:
                    os.makedirs(os.path.dirname(outfile))
                queue.append('ap3d --num {0} --vers {1} --telescope {2} --unlock'.format(num,apred,telescope),
                             outfile=outfile.replace('.fits','_pbs.'+logtime+'.log'),
                             errfile=outfile.replace('.fits','_pbs.'+logtime+'.err'))
            queue.commit(hard=True,submit=True)
            rootLogger.info('PBS key is '+queue.key)
            queue_wait(queue,sleeptime=60,verbose=True,logger=rootLogger)  # wait for jobs to complete
            chk3d = check_ap3d(expinfo,queue.key,apred,telescope,verbose=True,logger=rootLogger)
            del queue
        else:
            rootLogger.info('No exposures to process with ap3D')


    # 4) Perform initial quality check on all exposures
    #--------------------------------------------------
    # Do QA check of the files
    if 'check' in steps:
        rootLogger.info(' ')
        rootLogger.info('Doing quality checks on all exposures')
        qachk = check.check(expinfo['num'],apred,telescope,verbose=True,logger=rootLogger)
        rootLogger.info(' ')


    # 5) Make all daily cals (domeflats, quartzflats, arclamps, FPI)
    #----------------------------------------------------------------
    if 'cals' in steps:
        rootLogger.info('')
        rootLogger.info('----------------------------------------')
        rootLogger.info('5) Generating daily calibration products')
        rootLogger.info('========================================')
        rootLogger.info('')
        # First we need to run domeflats and quartzflats so there are apPSF files
        # Then the arclamps
        # apFlux files?
        # Then the FPI exposures last (needs apPSF and apWave files)
        calind, = np.where((expinfo['exptype']=='DOMEFLAT') | (expinfo['exptype']=='QUARTZFLAT') | 
                           (expinfo['exptype']=='ARCLAMP') | (expinfo['exptype']=='FPI'))
        if len(calind)>0:
            # 1: psf, 2: flux, 4: arcs, 8: fpi
            calcodedict = {'DOMEFLAT':3, 'QUARTZFLAT':1, 'ARCLAMP':4, 'FPI':8}
            calcode = [calcodedict[etype] for etype in expinfo['exptype'][calind]]
            calnames = ['DOMEFLAT/QUARTZFLAT','FLUX','ARCLAMP','FPI']
            shcalnames = ['psf','flux','arcs','fpi']
            chkcal = []
            for j,ccode in enumerate([1,2,4,8]):
                rootLogger.info('')
                rootLogger.info('----------------------------------------------')
                rootLogger.info('Running Calibration Files: '+str(calnames[j]))
                rootLogger.info('==============================================')
                rootLogger.info('')
                cind, = np.where((np.array(calcode) & ccode) > 0)
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
                        arctype1 = expinfo['arctype'][calind[cind[k]]]                    
                        rootLogger.info('Calibration file %d : %s %d' % (k+1,exptype1,num1))
                        if ccode==1:   # psfs
                            cmd1 = 'makecal --psf '+str(num1)+' --unlock'
                            if clobber: cmd1 += ' --clobber'
                            logfile1 = calplandir+'/apPSF-'+str(num1)+'_pbs.'+logtime+'.log'
                        elif ccode==2:   # flux
                            cmd1 = 'makecal --psf '+str(num1)+' --flux '+str(num1)+' --unlock'
                            if clobber: cmd1 += ' --clobber'
                            logfile1 = calplandir+'/apFlux-'+str(num1)+'_pbs.'+logtime+'.log'
                        elif ccode==4:  # arcs 
                        #elif exptype1=='ARCLAMP' and (arctype1=='UNE' or arctype1=='THARNE'):  # arcs                       
                            cmd1 = 'makecal --wave '+str(num1)+' --unlock'
                            if clobber: cmd1 += ' --clobber'
                            logfile1 = calplandir+'/apWave-'+str(num1)+'_pbs.'+logtime+'.log' 
                        elif ccode==8:  # fpi
                        #elif exptype1=='ARCLAMP' and arctype1=='None':    # fpi                       
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
             

    # Make plan files for all nights!
    # 6) Make plan files
    #-------------------
    if 'plans' in steps:
        rootLogger.info('')
        rootLogger.info('--------------------')
        rootLogger.info('6) Making plan files')
        rootLogger.info('====================')
        rootLogger.info('')
        pass

    # 7) Run APRED on all of the plan files (ap3d-ap1dvisit), go through each MJD chronologically
    #--------------------------------------------------------------------------------------------
    if 'apred' in steps:
        rootLogger.info('')
        rootLogger.info('----------------')
        rootLogger.info('7) Running APRED')
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
        chkexp,chkvisit = runapogee.check_apred(expinfo,planfiles,queue.key,verbose=True,logger=rootLogger)
        del queue

        
    # 8) Run "rv" on all unique stars
    #--------------------------------
    if 'rv' in steps:
        rootLogger.info('')
        rootLogger.info('--------------------------------')
        rootLogger.info('8) Running RV+Visit Combination')
        rootLogger.info('================================')
        rootLogger.info('')
        vcat = db.query('visit',cols='*',where="apred_vers='%s' and mjd>=%d and mjd<=%d and telescope='%s'" % (apred,mjdstart,mjdstop,telescope))
        # Pick on the MJDs we want
        ind = []
        for m in mjds:
            gd, = np.where(vcat['mjd']==m)
            if len(gd)>0: ind += list(gd)
        ind = np.array(ind)
        if len(ind)>0:
            vcat = vcat[ind]
        # Get unique stars
        objects,ui = np.unique(vcat['apogee_id'],return_index=True)
        vcat = vcat[ui]
        # remove ones with missing or blank apogee_ids
        bd, = np.where((vcat['apogee_id']=='') | (vcat['apogee_id']=='None') | (vcat['apogee_id']=='2MNone'))
        if len(bd)>0:
            vcat = np.delete(vcat,bd)
        rootLogger.info(len(vcat)+' stars to run')

        import pdb; pdb.set_trace()

        queue = pbsqueue(verbose=True)
        queue.create(label='rv', nodes=nodes, alloc=alloc, ppn=ppn, cpus=cpus, qos=qos, shared=shared, numpy_num_threads=2,
                     walltime=walltime, notification=False)
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
        chkrv = runapogee.check_rv(vcat,queue.key)
        del queue


    # 9) Create full allVisit/allStar files
    # The QA code needs these
    if 'summary' in steps:
        rootLogger.info('')
        rootLogger.info('-----------------------')
        rootLogger.info('9) Create summary files')
        rootLogger.info('=======================')
        rootLogger.info('')

        runapogee.create_sumfiles(apred,telescope)

    
    # 10) Unified directory structure
    #---------------------------------
    if 'unified' in steps:
        rootLogger.info('')
        rootLogger.info('---------------------------------------------')
        rootLogger.info('10) Generating unified MWM directory structure')
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


    # 11) Run QA script
    #------------------
    if 'qa' in steps:
        rootLogger.info('')
        rootLogger.info('--------------')
        rootLogger.info('11) Running QA')
        rootLogger.info('==============')
        rootLogger.info('')
        queue = pbsqueue(verbose=True)
        queue.create(label='qa', nodes=1, alloc=alloc, ppn=ppn, qos=qos, cpus=1, shared=shared, walltime=walltime, notification=False)
        qaoutfile = os.environ['APOGEE_REDUX']+'/'+apred+'/log/'+observatory+'/'+str(mjd5)+'-qa.'+logtime+'.log'
        qaerrfile = qaoutfile.replace('-qa.log','-qa.'+logtime+'.err')
        if os.path.exists(os.path.dirname(qaoutfile))==False:
            os.makedirs(os.path.dirname(qaoutfile))

        # apqa on each plate/config
        # nightly QA
        # monitor ppate

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
    runapogee.summary_email(observatory,mjdstart,chkexp,chkvisit,chkrv,logfile)
