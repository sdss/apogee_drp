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
from . import mkplan,check
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
                newmjds.append(int(m))
        newmjds = list(np.unique(newmjds))
        mjds = newmjds
    else:
        mjds = np.arange(59146,lastnightmjd()+1)
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


def dbload_plans(planfiles):
    """  Load plan files into the database."""
    db = apogeedb.DBSession()   # open db session
    nplans = len(planfiles)
    
    gitvers = plan.getgitvers()

    # Loop over the planfiles
    dtype = np.dtype([('planfile',(np.str,300)),('apred_vers',(np.str,20)),('v_apred',(np.str,50)),('telescope',(np.str,10)),
                      ('instrument',(np.str,20)),('mjd',int),('plate',int),('configid',(np.str,20)),('designid',(np.str,20)),
                      ('fieldid',(np.str,20)),('fps',bool),('platetype',(np.str,20))])
    plantab = np.zeros(nplans,dtype=dtype)
    for i,planfile in enumerate(planfiles):
        planstr = plan.load(planfile)
        plantab['planfile'][i] = planfile
        plantab['apred_vers'][i] = planstr['apred_vers']
        plantab['v_apred'][i] = gitvers
        plantab['telescope'][i] = planstr['telescope']
        plantab['instrument'][i] = planstr['instrument']
        plantab['mjd'][i] = planstr['mjd']
        plantab['plate'][i] = planstr['plateid']
        if planstr['fps']:
            plantab['configid'] = planstr['configid']
            plantab['designid'] = planstr['designid']
            plantab['fieldid'] = planstr['fieldid']
            plantab['fps'] = True
        else:
            plantab['fps'] = False
        plantab['platetype'][i] = planstr['platetype']

    # Insert into the database
    db.ingest('plan',plantab)
    db.close()   # close db session


def check_queue_status(queue):
    """
    Check the status of the slurm jobs.

    This performs a rigorous check of the status of the tasks and
    of the individual tasks running on each cpu.  Sometimes the signal
    that a task completed does *not* go through.  This function should
    be able to deal with that.
    """

    # Get the tasks
    tasks = queue.client.job.tasks

    # Gather the information on all the tasks
    dt = np.dtype([('task_number',int),('node_number',int),('proc_number',int),('status',int),('complete',bool)])
    data = np.zeros(len(tasks),dtype=dt)
    nodeproc = []
    for i,t in enumerate(tasks):
        # Make sure we have the most up-to-date information
        #  redo the query to update the task
        slurm.db.session.refresh(t)
        data['task_number'][i] = t.task_number
        data['node_number'][i] = t.node_number
        data['proc_number'][i] = t.proc_number
        data['status'][i] = t.status
        if t.status==5:
            data['complete'][i] = True
        nodeproc.append(str(t.node_number)+'-'+str(t.proc_number))

    index = dln.create_index(nodeproc)
    for i,unp in enumerate(index['value']):
        ind = index['index'][index['lo'][i]:index['hi'][i]+1]
        data1 = data[ind]
        node,proc = unp.split('-')
        # Order by task number
        si = np.argsort(data1['task_number'])
        data1 = data1[si]
        # If last task in this group is complete,
        #   then they should all be done!
        if data1['status'][-1]==5:
            data['complete'][ind] = True

    # Calculate the completeness percentage
    ncomplete = np.sum(data['complete']==True)
    ntasks = len(data)
    percent_complete = ncomplete/ntasks*100
    return percent_complete


def queue_wait(queue,sleeptime=60,verbose=True,logger=None):
    """ Wait for the pbs queue to finish."""

    if logger is None:
        logger = dln.basiclogger()

    # Wait for jobs to complete
    running = True
    while running:
        time.sleep(sleeptime)
        percent_complete = queue.get_percent_complete()
        # Do a more detailed check once some have finished
        ntasks_complete = queue.client.job.task_count_with_status(5)
        if ntasks_complete>0:
            percent_complete2 = check_queue_status(queue)
            percent_complete = np.maximum(percent_complete,percent_complete2)
        if verbose==True:
            logger.info('percent complete = %d' % percent_complete)
        if percent_complete == 100:
            running = False


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


def check_ap3d(expinfo,pbskey,apred=None,telescope=None,verbose=False,logger=None):
    """ Check that ap3d ran okay and load into database."""

    if logger is None:
        logger = dln.basiclogger()

    db = apogeedb.DBSession()  # open db session

    if verbose==True:
        logger.info('')
        logger.info('-----------------')
        logger.info('Checking ap3D run')
        logger.info('=================')

    load = apload.ApLoad(apred=apred,telescope=telescope)

    if verbose:
        logger.info('')
        logger.info('%d exposures' % len(expinfo))
        logger.info(' NUM         SUCCESS')

    # Loop over the stars
    nexp = len(expinfo)

    dtype = np.dtype([('exposure_pk',int),('planfile',(np.str,300)),('apred_vers',(np.str,20)),('v_apred',(np.str,50)),
                      ('instrument',(np.str,20)),('telescope',(np.str,10)),('platetype',(np.str,50)),('mjd',int),
                      ('plate',int),('configid',(np.str,20)),('designid',(np.str,20)),('fieldid',(np.str,20)),
                      ('proctype',(np.str,30)),('pbskey',(np.str,50)),('checktime',(np.str,100)),
                      ('num',int),('success',bool)])
    chk3d = np.zeros(nexp,dtype=dtype)
    chk3d['apred_vers'] = apred
    chk3d['telescope'] = telescope
    chk3d['pbskey'] = pbskey
    chk3d['proctype'] = 'AP3D'
    chk3d['success'] = False
    chips = ['a','b','c']
    for i,num in enumerate(expinfo['num']):
        chk3d['exposure_pk'] = expinfo['pk'][i]
        chk3d['num'][i] = num
        mjd = int(load.cmjd(num))
        outfile = load.filename('2D',num=num,mjd=mjd,chips=True)
        outfiles = [outfile.replace('2D-','2D-'+ch+'-') for ch in chips]
        planfile = os.path.dirname(outfile)+'/logs/'+os.path.basename(outfile)
        planfile = outfile.replace('2D','3DPlan').replace('.fits','.yaml')
        chk3d['planfile'][i] = planfile
        exist = [os.path.exists(o) for o in outfiles]
        if exist[0]:
            head = fits.getheader(outfiles[0])
            chk3d['v_apred'][i] = head.get('V_APRED')
            head = fits.getheader(outfiles[0])
            chk3d['v_apred'][i] = head.get('V_APRED')
            head = fits.getheader(outfiles[0])
            chk3d['v_apred'][i] = head.get('V_APRED')
            head = fits.getheader(outfiles[0])
            chk3d['v_apred'][i] = head.get('V_APRED')
        chk3d['checktime'][i] = str(datetime.now())
        chk3d['success'][i] = np.sum(exist)==3

        if verbose:
            logger.info('%5d %20s %9s' % (i+1,num,chk3d['success'][i]))
    success, = np.where(chk3d['success']==True)
    logger.info('%d/%d succeeded' % (len(success),nexp))
    
    # Inset into the database
    db.ingest('exposure_status',chk3d)
    db.close()        

    return chk3d


def check_calib(expinfo,logfiles,pbskey,apred,verbose=False,logger=None):
    """ Check that the calibration files ran okay and load into database."""

    if logger is None:
        logger = dln.basiclogger()

    if verbose==True:
        logger.info('')
        logger.info('-------------------------')
        logger.info('Checking Calibration runs')
        logger.info('=========================')

    expinfo = np.atleast_1d(expinfo)

    chkcal = None

    # Exposure-level processing: ap3d, ap2d, calibration file
    ncal = np.array(expinfo).size
    dtype = np.dtype([('logfile',(np.str,300)),('apred_vers',(np.str,20)),('v_apred',(np.str,50)),
                      ('instrument',(np.str,20)),('telescope',(np.str,10)),('mjd',int),('caltype',(np.str,30)),
                      ('plate',int),('configid',(np.str,20)),('designid',(np.str,20)),('fieldid',(np.str,20)),
                      ('pbskey',(np.str,50)),('checktime',(np.str,100)),
                      ('num',int),('calfile',(np.str,300)),('success3d',bool),('success2d',bool),('success',bool)])
    chkcal = np.zeros(ncal,dtype=dtype)

    # Loop over the planfiles
    for i in range(ncal):
        # domeflat, quartzflat
        # arclamp
        # fpi
        lgfile = logfiles[i]

        # apWave-49920071_pbs.123232121.log
        caltype = os.path.basename(lgfile)
        caltype = caltype.split('_pbs')[0]
        caltype = caltype.split('-')[0]
        caltype = caltype[2:]  # remove 
        filecode = caltype
        if caltype=='DailyWave':
            filecode = 'Wave'
        if caltype=='FPI':
            filecode = 'WaveFPI'

        num = expinfo['num'][i]
        mjd = int(expinfo['mjd'][i])
        chkcal['logfile'][i] = lgfile
        chkcal['num'][i] = num
        chkcal['apred_vers'][i] = apred
        if expinfo['observatory'][i]=='apo':
            chkcal['instrument'][i] = 'apogee-n'
            chkcal['telescope'][i] = 'apo25m'
        else:
            chkcal['instrument'][i] = 'apogee-s'
            chkcal['telescope'] = 'lco25m'
        chkcal['mjd'][i] = mjd
        chkcal['caltype'][i] = caltype
        try:
            chkcal['plate'][i] = expinfo['plateid'][i]
        except:
            chkcal['plate'][i] = -1
        chkcal['configid'][i] = expinfo['configid'][i]
        chkcal['designid'][i] = expinfo['designid'][i]
        chkcal['fieldid'][i] = expinfo['fieldid'][i]
        chkcal['pbskey'][i] = pbskey
        chkcal['checktime'][i] = str(datetime.now())
        chkcal['success'][i] = False
        load = apload.ApLoad(apred=apred,telescope=chkcal['telescope'][i])
        # AP3D
        #-----
        if caltype != 'DailyWave':
            base = load.filename('2D',num=num,mjd=mjd,chips=True)
            chfiles = [base.replace('2D-','2D-'+ch+'-') for ch in ['a','b','c']]
            exists = [os.path.exists(chf) for chf in chfiles]
            if exists[0]==True:  # get V_APRED (git version) from file
                chead = fits.getheader(chfiles[0])
                chkcal['v_apred'][i] = chead.get('V_APRED')
            if np.sum(exists)==3:
                chkcal['success3d'][i] = True
        else:
            chkcal['success3d'][i] = True
        # AP2D
        #-----
        if caltype != 'DailyWave':
            base = load.filename('1D',num=num,mjd=mjd,chips=True)
            chfiles = [base.replace('1D-','1D-'+ch+'-') for ch in ['a','b','c']]
            exists = [os.path.exists(chf) for chf in chfiles]
            if np.sum(exists)==3:
                chkcal['success2d'][i] = True
        else:
            chkcal['success2d'][i] = True
        # Final calibration file
        #-----------------------
        if caltype.lower()=='fpi':
            # Should really check fpi/apFPILines-EXPNUM8.fits
            base = load.filename('Wave',num=num,chips=True).replace('Wave-','WaveFPI-'+str(mjd)+'-')
        elif caltype.lower()=='dailywave':
            # Should really check fpi/apFPILines-EXPNUM8.fits
            base = load.filename('Wave',num=num,chips=True)[0:-13]+str(num)+'.fits'
        else:
            base = load.filename(caltype,num=num,chips=True)
        chkcal['calfile'][i] = base
        chfiles = [base.replace(filecode+'-',filecode+'-'+ch+'-') for ch in ['a','b','c']]
        exists = [os.path.exists(chf) for chf in chfiles]
        if exists[0]==True:  # get V_APRED (git version) from file
            chead = fits.getheader(chfiles[0])
            chkcal['v_apred'][i] = chead.get('V_APRED')
        # Overall success
        if np.sum(exists)==3:
            chkcal['success'][i] = True

        if verbose:
            logger.info('')
            logger.info('%d/%d' % (i+1,ncal))
            logger.info('Calibration type: %s' % chkcal['caltype'][i])
            logger.info('Calibration file: %s' % chkcal['calfile'][i])
            logger.info('log/errfile: '+os.path.basename(lgfile)+', '+os.path.basename(lgfile).replace('.log','.err'))
            logger.info('Calibration success: %s ' % chkcal['success'][i])

    # Load everything into the database
    db = apogeedb.DBSession()
    db.ingest('calib_status',chkcal)
    db.close()

    return chkcal


def check_apred(expinfo,planfiles,pbskey,verbose=False,logger=None):
    """ Check that apred ran okay and load into database."""

    if logger is None:
        logger = dln.basiclogger()

    if verbose==True:
        logger.info('')
        logger.info('--------------------')
        logger.info('Checking APRED runs')
        logger.info('====================')

    chkexp = None
    chkap = None

    # Loop over the planfiles
    nplanfiles = len(planfiles)
    for ip,pfile in enumerate(planfiles):
        if os.path.exists(pfile)==False:
            logger.info(pfile+' NOT FOUND')
            continue
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

        # normal: ap3d, ap2d, apCframe and ap1dvisit
        # dark: ap3d  (apDarkPlan)
        # cal: ap3d and ap2d (apCalPlan)

        # Load the plugmap information
        if platetype=='normal' and str(plate) != '0':
            plugmap = platedata.getdata(plate,mjd,apred_vers,telescope,plugid=planstr['plugmap'])
            fiberdata = plugmap['fiberdata']
        else:
            fiberdata = None

        # Exposure-level processing: ap3d, ap2d, apcframe
        dtype = np.dtype([('exposure_pk',int),('planfile',(np.str,300)),('apred_vers',(np.str,20)),('v_apred',(np.str,50)),
                          ('instrument',(np.str,20)),('telescope',(np.str,10)),('platetype',(np.str,50)),('mjd',int),
                          ('plate',int),('configid',(np.str,20)),('designid',(np.str,20)),('fieldid',(np.str,20)),
                          ('proctype',(np.str,30)),('pbskey',(np.str,50)),('checktime',(np.str,100)),
                          ('num',int),('success',bool)])
        chkexp1 = np.zeros(nexp*3,dtype=dtype)
        chkexp1['planfile'] = pfile
        chkexp1['apred_vers'] = apred_vers
        chkexp1['instrument'] = instrument
        chkexp1['telescope'] = telescope
        chkexp1['platetype'] = platetype
        chkexp1['mjd'] = mjd
        chkexp1['plate'] = plate
        if planstr['fps']:
            chkexp1['configid'] = planstr['configid']
            chkexp1['designid'] = planstr['designid']
            chkexp1['fieldid'] = planstr['fieldid']
            field = planstr['fieldid']
        else:
            field,survey,program = apload.apfield(planstr['plate'])
        chkexp1['proctype'] = 'AP3D'
        chkexp1['pbskey'] = pbskey
        chkexp1['checktime'] = str(datetime.now())
        chkexp1['success'] = False
        cnt = 0
        for num in expstr['name']:
            ind, = np.where(expinfo['num']==num)
            exposure_pk = expinfo['pk'][ind[0]]
            # AP3D
            #-----
            chkexp1['exposure_pk'][cnt] = exposure_pk
            chkexp1['num'][cnt] = num
            chkexp1['proctype'][cnt] = 'AP3D'
            base = load.filename('2D',num=num,mjd=mjd,chips=True)
            chfiles = [base.replace('2D-','2D-'+ch+'-') for ch in ['a','b','c']]
            exists = [os.path.exists(chf) for chf in chfiles]
            if exists[0]==True:  # get V_APRED (git version) from file
                chead = fits.getheader(chfiles[0])
                chkexp1['v_apred'][cnt] = chead.get('V_APRED')
            chkexp1['checktime'][cnt] = str(datetime.now())
            if np.sum(exists)==3:
                chkexp1['success'][cnt] = True
            cnt += 1  # increment counter
            # AP2D
            #-----
            if (platetype=='normal') | (platetype=='cal'):
                chkexp1['exposure_pk'][cnt] = exposure_pk
                chkexp1['num'][cnt] = num
                chkexp1['proctype'][cnt] = 'AP2D'
                base = load.filename('1D',num=num,mjd=mjd,chips=True)
                chfiles = [base.replace('1D-','1D-'+ch+'-') for ch in ['a','b','c']]
                exists = [os.path.exists(chf) for chf in chfiles]
                if exists[0]==True:  # get V_APRED (git version) from file
                    chead = fits.getheader(chfiles[0])
                    chkexp1['v_apred'][cnt] = chead.get('V_APRED')
                if np.sum(exists)==3:
                    chkexp1['success'][cnt] = True
                cnt += 1  # increment counter
            # APCframe
            #---------
            if platetype=='normal':
                chkexp1['exposure_pk'][cnt] = exposure_pk
                chkexp1['num'][cnt] = num
                chkexp1['proctype'][cnt] = 'APCFRAME'
                base = load.filename('Cframe',num=num,mjd=mjd,plate=plate,chips=True,field=field)
                chfiles = [base.replace('Cframe-','Cframe-'+ch+'-') for ch in ['a','b','c']]
                exists = [os.path.exists(chf) for chf in chfiles]
                if exists[0]==True:  # get V_APRED (git version) from file
                    chead = fits.getheader(chfiles[0])
                    chkexp1['v_apred'][cnt] = chead.get('V_APRED')
                if np.sum(exists)==3:
                    chkexp1['success'][cnt] = True
                cnt += 1  # increment counter
        # Trim extra elements
        chkexp1 = chkexp1[0:cnt]

        # Plan summary and ap1dvisit
        #---------------------------
        dtypeap = np.dtype([('planfile',(np.str,300)),('logfile',(np.str,300)),('errfile',(np.str,300)),
                            ('apred_vers',(np.str,20)),('v_apred',(np.str,50)),('instrument',(np.str,20)),
                            ('telescope',(np.str,10)),('platetype',(np.str,50)),('mjd',int),('plate',int),
                            ('configid',(np.str,20)),('designid',(np.str,20)),('fieldid',(np.str,20)),
                            ('nobj',int),('pbskey',(np.str,50)),('checktime',(np.str,100)),('ap3d_success',bool),
                            ('ap3d_nexp_success',int),('ap2d_success',bool),('ap2d_nexp_success',int),
                            ('apcframe_success',bool),('apcframe_nexp_success',int),('applate_success',bool),
                            ('apvisit_success',bool),('apvisit_nobj_success',int),
                            ('apvisitsum_success',bool),('success',bool)])
        chkap1 = np.zeros(1,dtype=dtypeap)
        chkap1['planfile'] = pfile
        chkap1['logfile'] = pfile.replace('.yaml','_pbs.log')
        chkap1['errfile'] = pfile.replace('.yaml','_pbs.err')
        chkap1['apred_vers'] = apred_vers
        chkap1['instrument'] = instrument
        chkap1['telescope'] = telescope
        chkap1['platetype'] = platetype
        chkap1['mjd'] = mjd
        chkap1['plate'] = plate
        if planstr['fps']:
            chkap1['configid'] = planstr['configid']
            chkap1['designid'] = planstr['designid']
            chkap1['fieldid'] = planstr['fieldid']
        if platetype=='normal' and fiberdata is not None:
            chkap1['nobj'] = np.sum(fiberdata['objtype']!='SKY')  # stars and tellurics
        chkap1['pbskey'] = pbskey
        chkap1['checktime'] = str(datetime.now())
        # ap3D, ap2D, apCframe success
        ind3d, = np.where(chkexp1['proctype']=='AP3D')
        chkap1['ap3d_nexp_success'] = np.sum(chkexp1['success'][ind3d])
        chkap1['ap3d_success'] = np.sum(chkexp1['success'][ind3d])==nexp
        ind2d, = np.where(chkexp1['proctype']=='AP2D')
        if len(ind2d)>0:
            chkap1['ap2d_nexp_success'] = np.sum(chkexp1['success'][ind2d])
            chkap1['ap2d_success'] = np.sum(chkexp1['success'][ind2d])==nexp
        indcf, = np.where(chkexp1['proctype']=='APCFRAME')
        if len(indcf)>0:
            chkap1['apcframe_nexp_success'] = np.sum(chkexp1['success'][indcf])
            chkap1['apcframe_success'] = np.sum(chkexp1['success'][indcf])==nexp
        if platetype=='normal':
            # apPlate
            chkap1['applate_success'] = False
            base = load.filename('Plate',plate=plate,mjd=mjd,chips=True,field=field)
            chfiles = [base.replace('Plate-','Plate-'+ch+'-') for ch in ['a','b','c']]
            exists = [os.path.exists(chf) for chf in chfiles]
            if exists[0]==True:  # get V_APRED (git version) from file
                chead = fits.getheader(chfiles[0])
                chkap1['v_apred'] = chead.get('V_APRED')
            if np.sum(exists)==3:
                chkap1['applate_success'] = True
            # apVisit
            base = load.filename('Visit',plate=plate,mjd=mjd,fiber=1,field=field) 
            visitfiles = glob(base.replace('-001.fits','-???.fits'))
            nvisitfiles = len(visitfiles)
            chkap1['apvisit_nobj_success']  = nvisitfiles
            # take broken fibers into account for visit success!!!
            nbadfiber = dln.size(planstr['badfiberid'])
            if planstr['fps']:
                # take two FPI fibers into account
                chkap1['apvisit_success'] = (nvisitfiles>=(chkap1['nobj']-nbadfiber-2))
            else:
                chkap1['apvisit_success'] = (nvisitfiles>=(chkap1['nobj']-nbadfiber))
            apvisitsumfile = load.filename('VisitSum',plate=plate,mjd=mjd,field=field)
            chkap1['apvisitsum_success'] = os.path.exists(apvisitsumfile)
        # Success of plan file
        if platetype=='normal':
            chkap1['success'] = chkap1['ap3d_success'][0] and chkap1['ap2d_success'][0] and chkap1['apcframe_success'][0] and \
                                chkap1['applate_success'][0] and chkap1['apvisit_success'][0] and chkap1['apvisitsum_success'][0]
        elif platetype=='cal':
            chkap1['success'] = chkap1['ap3d_success'][0] and chkap1['ap2d_success'][0]
        elif platetype=='dark':
            chkap1['success'] = chkap1['ap3d_success'][0]
        else:
            chkap1['success'] = chkap1['ap3d_success'][0]

        if verbose:
            logger.info('')
            logger.info('%d/%d' % (ip+1,nplanfiles))
            logger.info('planfile: '+pfile)
            logger.info('log/errfile: '+os.path.basename(chkap1['logfile'][0])+', '+os.path.basename(chkap1['errfile'][0]))
            logger.info('platetype: %s' % platetype)
            logger.info('mjd: %d' % mjd)
            if platetype=='normal': logger.info('plate: %d' % plate)
            logger.info('nexp: %d' % nexp)
            if platetype=='normal': logger.info('Nobj: %d' % chkap1['nobj'][0])
            logger.info('3D/2D/Cframe:')
            logger.info('Num    EXPID   NREAD  3D     2D  Cframe')
            for k,num in enumerate(expstr['name']):
                ind, = np.where(expinfo['num']==num)
                success3d,success2d,successcf = False,False,False
                ind3d, = np.where((chkexp1['num']==num) & (chkexp1['proctype']=='AP3D'))
                if len(ind3d)>0: success3d=chkexp1['success'][ind3d[0]]
                ind2d, = np.where((chkexp1['num']==num) & (chkexp1['proctype']=='AP2D'))
                if len(ind2d)>0: success2d=chkexp1['success'][ind2d[0]]
                indcf, = np.where((chkexp1['num']==num) & (chkexp1['proctype']=='APCFRAME'))
                if len(indcf)>0: successcf=chkexp1['success'][indcf[0]]
                logger.info('%2d %10d %4d %6s %6s %6s' % (k+1,chkexp1['num'][ind3d],expinfo['nread'][ind],
                                                          success3d,success2d,successcf))
            if platetype=='normal':
                logger.info('apPlate files: %s ' % chkap1['applate_success'][0])
                logger.info('N apVisit files: %d ' % chkap1['apvisit_nobj_success'][0])
                logger.info('apVisitSum file: %s ' % chkap1['apvisitsum_success'][0])
            logger.info('Plan success: %s ' % chkap1['success'][0])
            

        # Add to the global catalog
        if chkexp is not None:
            chkexp = np.hstack((chkexp,chkexp1))
        else:
            chkexp = chkexp1.copy()
        if chkap is not None:
            chkap = np.hstack((chkap,chkap1))
        else:
            chkap = chkap1.copy()


    # Load everything into the database
    db = apogeedb.DBSession()
    db.ingest('exposure_status',chkexp)
    db.ingest('visit_status',chkap)
    db.close()

    return chkexp,chkap


def check_rv(visits,pbskey,verbose=False,logger=None):
    """ Check that rv ran okay and load into database."""

    if logger is None:
        logger = dln.basiclogger()

    db = apogeedb.DBSession()  # open db session

    if verbose==True:
        logger.info('')
        logger.info('-------------------------------------')
        logger.info('Checking RV + Visit Combination runs')
        logger.info('=====================================')

    apred_vers = visits['apred_vers'][0]
    telescope = visits['telescope'][0]
    load = apload.ApLoad(apred=apred_vers,telescope=telescope)

    if verbose:
        logger.info('')
        logger.info('%d stars' % len(visits))
        logger.info('apred_vers: %s' % apred_vers)
        logger.info('telescope: %s' % telescope)
        logger.info(' NUM         APOGEE_ID       HEALPIX NVISITS SUCCESS')

    # Loop over the stars
    nstars = len(visits)
    dtype = np.dtype([('apogee_id',(np.str,50)),('apred_vers',(np.str,20)),('v_apred',(np.str,50)),
                      ('telescope',(np.str,10)),('healpix',int),('nvisits',int),('pbskey',(np.str,50)),
                      ('file',(np.str,300)),('checktime',(np.str,100)),('success',bool)])
    chkrv = np.zeros(nstars,dtype=dtype)
    chkrv['apred_vers'] = apred_vers
    chkrv['telescope'] = telescope
    chkrv['pbskey'] = pbskey
    for i,visit in enumerate(visits):
        starfilenover = load.filename('Star',obj=visit['apogee_id'])
        # add version number, should be MJD of the latest visit
        stardir = os.path.dirname(starfilenover)
        starbase = os.path.splitext(os.path.basename(starfilenover))[0]
        starbase += '-'+str(visit['mjd'])   # add star version 
        starfile = stardir+'/'+starbase+'.fits'
        # Get nvisits for this star
        starvisits = db.query('visit',cols='*',where="apogee_id='"+visit['apogee_id']+"' and "+\
                              "telescope='"+telescope+"' and apred_vers='"+apred_vers+"'")
        chkrv['apogee_id'][i] = visit['apogee_id']
        chkrv['healpix'][i] = apload.obj2healpix(visit['apogee_id'])
        chkrv['nvisits'][i] = len(starvisits)
        chkrv['file'][i] = starfile
        if os.path.exists(starfile):
            head = fits.getheader(starfile)
            chkrv['v_apred'][i] = head.get('V_APRED')
        chkrv['checktime'][i] = str(datetime.now())
        chkrv['success'][i] = os.path.exists(starfile)

        if verbose:
            logger.info('%5d %20s %8d %5d %9s' % (i+1,chkrv['apogee_id'][i],chkrv['healpix'][i],
                                                  chkrv['nvisits'][i],chkrv['success'][i]))
    success, = np.where(chkrv['success']==True)
    if verbose:
        logger.info('%d/%d succeeded' % (len(success),nstars))
    
    # Inset into the database
    db.ingest('rv_status',chkrv)
    db.close()        

    return chkrv


def create_sumfiles_mjd(apred,telescope,mjd5,logger=None):
    """ Create allVisit/allStar files and summary of objects for this night."""

    if logger is None:
        logger = dln.basiclogger()

    load = apload.ApLoad(apred=apred,telescope=telescope)

    # Start db session
    db = apogeedb.DBSession()

    # Nightly summary files

    # Nightly allStar, allStarMJD
    allstarmjd = db.query('star',cols='*',where="apred_vers='%s' and telescope='%s' and starver='%s'" % (apred,telescope,mjd5))

    # for visit except that we'll get the multiple visit rows returned for each unique star row
    #   Get more info by joining with the visit table.
    vcols = ['apogee_id', 'target_id', 'apred_vers','file', 'uri', 'fiberid', 'plate', 'mjd', 'telescope', 'survey',
             'field', 'programname', 'ra', 'dec', 'glon', 'glat', 'jmag', 'jerr', 'hmag',
             'herr', 'kmag', 'kerr', 'src_h', 'pmra', 'pmdec', 'pm_src', 'apogee_target1', 'apogee_target2', 'apogee_target3',
             'apogee_target4', 'catalogid', 'gaiadr2_plx', 'gaiadr2_plx_error', 'gaiadr2_pmra', 'gaiadr2_pmra_error',
             'gaiadr2_pmdec', 'gaiadr2_pmdec_error', 'gaiadr2_gmag', 'gaiadr2_gerr', 'gaiadr2_bpmag', 'gaiadr2_bperr',
             'gaiadr2_rpmag', 'gaiadr2_rperr', 'sdssv_apogee_target0', 'firstcarton', 'targflags', 'snr', 'starflag', 
             'starflags','dateobs','jd']
    rvcols = ['starver', 'bc', 'vtype', 'vrel', 'vrelerr', 'vheliobary', 'chisq', 'rv_teff', 'rv_feh',
              'rv_logg', 'xcorr_vrel', 'xcorr_vrelerr', 'xcorr_vheliobary', 'n_components', 'rv_components']

    # Nightly allVisit, allVisitMJD
    cols = ','.join('v.'+np.char.array(vcols)) +','+ ','.join('rv.'+np.char.array(rvcols))
    allvisitmjd = db.query(sql="select "+cols+" from apogee_drp.rv_visit as rv join apogee_drp.visit as v on rv.visit_pk=v.pk "+\
                           "where rv.apred_vers='"+apred+"' and rv.telescope='"+telescope+"' and v.mjd="+str(mjd5)+" and rv.starver='"+str(mjd5)+"'")

    # maybe in summary/MJD/ or qa/MJD/ ?
    #allstarmjdfile = load.filename('allStarMJD')
    allstarfile = load.filename('allStar').replace('.fits','-'+telescope+'.fits')
    allstarmjdfile = allstarfile.replace('allStar','allStarMJD').replace('.fits','-'+str(mjd5)+'.fits')
    mjdsumdir = os.path.dirname(allstarmjdfile)+'/'+str(mjd5)
    allstarmjdfile = mjdsumdir+'/'+os.path.basename(allstarmjdfile)
    if os.path.exists(mjdsumdir)==False:
        os.makedirs(mjdsumdir)
    logger.info('Writing Nightly allStarMJD file to '+allstarmjdfile)
    logger.info(str(len(allstarmjd))+' stars for '+str(mjd5))
    Table(allstarmjd).write(allstarmjdfile,overwrite=True)

    allvisitfile = load.filename('allVisit').replace('.fits','-'+telescope+'.fits')
    allvisitmjdfile = allvisitfile.replace('allVisit','allVisitMJD').replace('.fits','-'+str(mjd5)+'.fits')
    allvisitmjdfile = mjdsumdir+'/'+os.path.basename(allvisitmjdfile)
    logger.info('Writing Nightly allVisitMJD file to '+allvisitmjdfile)
    logger.info(str(len(allvisitmjd))+' visits for '+str(mjd5))
    Table(allvisitmjd).write(allvisitmjdfile,overwrite=True)

    db.close()


def create_sumfiles(apred,telescope,mjd5=None,logger=None):
    """ Create allVisit/allStar files and summary of objects for this night."""

    if logger is None:
        logger = dln.basiclogger()

    load = apload.ApLoad(apred=apred,telescope=telescope)

    # Start db session
    db = apogeedb.DBSession()

    # USE STAR_LATEST AND VISIT_LATEST "VIEWS" IN THE FUTURE!

    # Full allVisit and allStar files
    #  apogee_id+apred_vers+telescope+starver uniquely identifies a particular star row
    #  For each apogee_id+apred_vers+telescope we want the maximum starver
    #  The subquery does this for us by grouping by apogee_id+apred_vers+telescope and
    #    calculating the aggregate value MAX(starver).
    #  We then select the particular row (with all columns) using apogee_id+apred_vers+telescope+starver
    #    from this subquery.
    #allstar = db.query(sql="select * from apogee_drp.star where (apogee_id, apred_vers, telescope, starver) in "+\
    #                   "(select apogee_id, apred_vers, telescope, max(starver) from apogee_drp.star where "+\
    #                   "apred_vers='"+apred+"' and telescope='"+telescope+"' group by apogee_id, apred_vers, telescope)")
    # Using STAR_LATEST seems much faster
    allstar = db.query('star_latest',cols='*',where="apred_vers='"+apred+"' and telescope='"+telescope+"'")
    allstarfile = load.filename('allStar').replace('.fits','-'+telescope+'.fits')
    logger.info('Writing allStar file to '+allstarfile)
    logger.info(str(len(allstar))+' stars')
    if os.path.exists(os.path.dirname(allstarfile))==False:
        os.makedirs(os.path.dirname(allstarfile))
    allstar = Table(allstar)
    del allstar['nres']    # temporary kludge, nres is causing write problems
    allstar.write(allstarfile,overwrite=True)

    # allVisit
    # Same thing for visit except that we'll get the multiple visit rows returned for each unique star row
    #   Get more info by joining with the visit table.
    vcols = ['apogee_id', 'target_id', 'apred_vers','file', 'uri', 'fiberid', 'plate', 'mjd', 'telescope', 'survey',
             'field', 'programname', 'ra', 'dec', 'glon', 'glat', 'jmag', 'jerr', 'hmag',
             'herr', 'kmag', 'kerr', 'src_h', 'pmra', 'pmdec', 'pm_src', 'apogee_target1', 'apogee_target2', 'apogee_target3',
             'apogee_target4', 'catalogid', 'gaiadr2_plx', 'gaiadr2_plx_error', 'gaiadr2_pmra', 'gaiadr2_pmra_error',
             'gaiadr2_pmdec', 'gaiadr2_pmdec_error', 'gaiadr2_gmag', 'gaiadr2_gerr', 'gaiadr2_bpmag', 'gaiadr2_bperr',
             'gaiadr2_rpmag', 'gaiadr2_rperr', 'sdssv_apogee_target0', 'firstcarton', 'targflags', 'snr', 'starflag', 
             'starflags','dateobs','jd']
    rvcols = ['starver', 'bc', 'vtype', 'vrel', 'vrelerr', 'vheliobary', 'chisq', 'rv_teff', 'rv_feh',
              'rv_logg', 'xcorr_vrel', 'xcorr_vrelerr', 'xcorr_vheliobary', 'n_components', 'rv_components']
    cols = ','.join(vcols+rvcols)
    allvisit = db.query('visit_latest',cols=cols,where="apred_vers='"+apred+"' and telescope='"+telescope+"'")
    allvisitfile = load.filename('allVisit').replace('.fits','-'+telescope+'.fits')
    logger.info('Writing allVisit file to '+allvisitfile)
    logger.info(str(len(allvisit))+' visits')
    if os.path.exists(os.path.dirname(allvisitfile))==False:
        os.makedirs(os.path.dirname(allvisitfile))
    Table(allvisit).write(allvisitfile,overwrite=True)

    db.close()

    # Nightly summary files
    if mjd5 is not None:
        create_sumfiles_mjd(apred,telescope,mjd5,logger=logger)

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

    # Read in the master calibration index
    caldir = os.environ['APOGEE_DRP_DIR']+'/data/cal/'
    caldictn = mkcal.readcal(caldir+'apogee-n.par')
    caldicts = mkcal.readcal(caldir+'apogee-s.par')
    
    # Symbolic links to another version
    if linkvers:
        logger.info('Creating calibration product symlinks to version >>'+str(linkvers)+'<<')
        cwd = os.path.abspath(os.curdir)
        for d in ['bpm','darkcorr','detector','flatcorr','littrow','lsf','persist','telluric','sparse','fiber']:
            for obs in ['apogee-n','apogee-s']:
                if obs=='apogee-n':
                    prefix = 'ap'
                else:
                    prefix = 'as'
                srcdir = apogee_redux+linkvers+'/cal/'+obs+'/'+d
                destdir = apogee_redux+apred+'/cal/'+obs+'/'+d
                if d=='sparse' or d=='fiber':
                    srcdir = apogee_redux+linkvers+'/cal/'+obs+'/psf'
                    destdir = apogee_redux+apred+'/cal/'+obs+'/psf'
                logger.info('Creating symlinks for '+d+' '+obs)
                os.chdir(destdir)
                if d=='sparse':
                    subprocess.run(['ln -s '+srcdir+'/'+prefix+'Sparse*.fits .'],shell=True)
                    # Need to link apEPSF files as well
                    sfiles = glob(srcdir+'/'+prefix+'Sparse*.fits')
                    if len(sfiles)>0:
                        snum = [os.path.basename(s)[9:-5] for s in sfiles]
                        for num in snum:
                            subprocess.run(['ln -s '+srcdir+'/'+prefix+'EPSF-?-'+num+'.fits .'],shell=True)
                elif d=='darkcorr' or d=='flatcorr':
                    subprocess.run(['ln -s '+srcdir+'/*.fits .'],shell=True)
                    subprocess.run(['ln -s '+srcdir+'/*.tab .'],shell=True)
                elif d=='fiber':
                    # Create symlinks for all the fiber cal files, PSF, EPSF, ETrace
                    caldict = mkcal.readcal(caldir+obs+'.par')
                    fiberdict = caldict['fiber']
                    for f in fiberdict['name']:
                        subprocess.run(['ln -s '+srcdir+'/'+prefix+'EPSF-?-'+f+'.fits .'],shell=True)
                        subprocess.run(['ln -s '+srcdir+'/'+prefix+'PSF-?-'+f+'.fits .'],shell=True)
                        tsrcdir = apogee_redux+linkvers+'/cal/'+obs+'/trace'
                        tdestdir = apogee_redux+apred+'/cal/'+obs+'/trace'
                        os.chdir(tdestdir)
                        subprocess.run(['ln -s '+tsrcdir+'/'+prefix+'ETrace-?-'+f+'.fits .'],shell=True)
                        os.chdir(destdir)
                else:
                    subprocess.run(['ln -s '+srcdir+'/*.fits .'],shell=True)
                    
        # Link all of the PSF files in the PSF library

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
            outfile = load.filename('Detector',num=name,chips=True)
            logfile1 = os.path.dirname(outfile)+'/mkdet-'+str(name)+'-'+telescope+'_pbs.'+logtime+'.log'
            errfile1 = logfile1.replace('.log','.err')
            if os.path.exists(os.path.dirname(logfile1))==False:
                os.makedirs(os.path.dirname(logfile1))
            cmd1 = 'makecal --vers {0} --telescope {1}'.format(apred,telescope)
            cmd1 += ' --det '+str(name)+' --unlock'
            if clobber:
                cmd1 += ' --clobber'
            #logfiles.append(logfile1)
            # Check if files exist already
            docal[i] = True
            if clobber is not True:
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
        queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
    else:
        logger.info('No master Detector calibration files need to be run')
    del queue    


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
            outfile = load.filename('Dark',num=name,chips=True)
            logfile1 = os.path.dirname(outfile)+'/mkdark-'+str(name)+'-'+telescope+'_pbs.'+logtime+'.log'
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
        queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
    else:
        logger.info('No master Dark calibration files need to be run')
    del queue    
    # Make the dark plots
    if np.sum(docal)>0:
        cal.darkplot(apred=apred,telescope=telescope)


    # I could process the individual flat exposures in parallel first
    # that would dramatically speed things up

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
            outfile = load.filename('Flat',num=name,chips=True)
            logfile1 = os.path.dirname(outfile)+'/mkflat-'+str(name)+'-'+telescope+'_pbs.'+logtime+'.log'
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
        queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
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
            outfile = load.filename('BPM',num=name,chips=True)
            logfile1 = os.path.dirname(outfile)+'/mkbpm-'+str(name)+'-'+telescope+'_pbs.'+logtime+'.log'
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
        queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
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
            outfile = load.filename('Littrow',num=name,chips=True)
            logfile1 = os.path.dirname(outfile)+'/mklittrow-'+str(name)+'-'+telescope+'_pbs.'+logtime+'.log'
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
        queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
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
            outfile = load.filename('Response',num=name,chips=True)
            logfile1 = os.path.dirname(outfile)+'/mkresponse-'+str(name)+'-'+telescope+'_pbs.'+logtime+'.log'
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
        queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
    else:
        logger.info('No master Response calibration files need to be run')
    del queue    


    # Make Sparse in parallel
    #--------------------------
    sparsedict = allcaldict['sparse']
    logger.info('')
    logger.info('---------------------------------')
    logger.info('Making master Sparses in parallel')
    logger.info('=================================')
    logger.info('Slurm settings: '+str(slurm))
    queue = pbsqueue(verbose=True)
    queue.create(label='mksparse', **slurm)
    for i in range(len(littdict)):
        name = sparsedict['name'][i]
        if np.sum((mjds >= sparsedict['mjd1'][i]) & (mjds <= sparsedict['mjd2'][i])) > 0:
            outfile = load.filename('Sparse',num=name,chips=True)
            logfile1 = os.path.dirname(outfile)+'/mksparse-'+str(name)+'-'+telescope+'_pbs.'+logtime+'.log'
            errfile1 = logfile1.replace('.log','.err')
            if os.path.exists(os.path.dirname(logfile1))==False:
                os.makedirs(os.path.dirname(logfile1))
            cmd1 = 'makecal --vers {0} --telescope {1}'.format(apred,telescope)
            cmd1 += ' --sparse '+str(name)+' --unlock'
            if clobber:
                cmd1 += ' --clobber'
            #logfiles.append(logfile1)
            # Check if files exist already
            docal[i] = True
            if clobber is not True:
                if load.exists('Sparse',num=name):
                    logger.info(os.path.basename(outfile)+' already exists and clobber==False')
                    docal[i] = False
            if docal[i]:
                logger.info('Sparse file %d : %s' % (i+1,name))
                logger.info('Command : '+cmd1)
                logger.info('Logfile : '+logfile1)
                queue.append(cmd1, outfile=logfile1,errfile=errfile1)
    if np.sum(docal)>0:
        queue.commit(hard=True,submit=True)
        logger.info('PBS key is '+queue.key)
        queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
    else:
        logger.info('No master Sparse calibration files need to be run')
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
    #queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
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
            outfile = load.filename('LSF',num=name,chips=True)
            logfile1 = os.path.dirname(outfile)+'/mklsf-'+str(name)+'-'+telescope+'_pbs.'+logtime+'.log'
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
        queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
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
        queue_wait(queue,sleeptime=60,verbose=True,logger=logger)  # wait for jobs to complete
        # This should check if the ap3d ran okay and puts the status in the database
        chk3d = check_ap3d(expinfo,queue.key,apred,telescope,verbose=True,logger=logger)
    else:
        chk3d = None
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
    chips = ['a','b','c']
    
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

    # Create cal plan directories for each night
    for m in mjds:
        calplandir = os.path.dirname(load.filename('CalPlan',num=0,mjd=m))
        if os.path.exists(calplandir)==False:
            os.makedirs(calplandir)

    # Loop over calibration types
    calnames = ['psf','flux','arcs','dailywave','fpi']
    filecodes = ['PSF','Flux','Wave','Wave','WaveFPI']
    chkcal = []
    for i,caltype in enumerate(calnames):
        logger.info('')
        logger.info('----------------------------------------------')
        logger.info('Running Calibration Files: '+caltype.upper())
        logger.info('==============================================')
        logger.info('')

        # Get data for this calibration type
        if caltype=='psf':
            # Domeflats and quartzflats for plates
            # Quartzflats only for FPS
            ind, = np.where( (((expinfo['exptype']=='DOMEFLAT') | (expinfo['exptype']=='QUARTZFLAT')) & (expinfo['mjd']<59556)) | 
                             ((expinfo['exptype']=='QUARTZFLAT') & (expinfo['mjd']>=59556)) )
            ncal = len(ind)
            if len(ind)>0:
                calinfo = expinfo[ind]
        elif caltype=='flux':
            ind, = np.where(expinfo['exptype']=='DOMEFLAT')
            ncal = len(ind)
            if len(ind)>=0:
                calinfo = expinfo[ind]
        elif caltype=='arcs':
            ind, = np.where(expinfo['exptype']=='ARCLAMP')
            ncal = len(ind)
            if len(ind)>0:
                calinfo = expinfo[ind]
        elif caltype=='dailywave':
            ncal = len(mjds)
            calinfo = np.zeros(ncal,dtype=np.dtype([('num',int),('mjd',int),('exptype',np.str,20),('observatory',np.str,3),
                                                    ('configid',int),('designid',int),('fieldid',int)]))
            calinfo['num'] = mjds
            calinfo['mjd'] = mjds
            calinfo['exptype'] = 'dailywave'
            calinfo['observatory'] = load.observatory
        elif caltype=='fpi':
            # Only FPI exposure number per MJD
            fpi, = np.where(expinfo['exptype']=='FPI')
            if len(fpi)>0:
                # Take the first for each night
                vals,ui = np.unique(expinfo['mjd'][fpi],return_index=True)
                ncal = len(ui)
                calinfo = expinfo[fpi][ui]
                si = np.argsort(calinfo['num'])
                calinfo = calinfo[si]

        logger.info(str(ncal)+' file(s)')
        slurm1 = slurm.copy()
        if ncal<64:
            slurm1['cpus'] = ncal
        slurm1['numpy_num_threads'] = 2
        logger.info('Slurm settings: '+str(slurm1))
        queue = pbsqueue(verbose=True)
        queue.create(label='makecal-'+calnames[i], **slurm1)

        # Loop over calibration and check if we need to run them
        docal = np.zeros(ncal,bool)
        for j in range(ncal):
            num1 = calinfo['num'][j]
            mjd1 = calinfo['mjd'][j]
            # Check if files exist already
            docal[j] = True
            if clobber is not True:
                if caltype=='dailywave':
                    outfile = load.filename(filecodes[i],num=num1,mjd=mjd1,chips=True)
                    outfile = outfile[0:-13]+str(mjd1)+'.fits'
                    allfiles = [outfile.replace('Wave-','Wave-'+ch+'-') for ch in chips]
                    allexist = [os.path.exists(f) for f in allfiles]
                    exists = np.sum(allexist)==3
                else:
                    outfile = load.filename(filecodes[i],num=num1,mjd=mjd1,chips=True)
                    exists = load.exists(filecodes[i],num=num1,mjd=mjd1)
                if exists:
                    logger.info(str(j+1)+'  '+os.path.basename(outfile)+' already exists and clobber==False')
                    docal[j] = False
        logger.info(str(np.sum(docal))+' '+caltype.upper()+' to run')
        # Loop over the calibrations and make the commands for the ones that we will run
        logfiles = []
        torun, = np.where(docal==True)
        ntorun = len(torun)
        for j in range(ntorun):
            num1 = calinfo['num'][torun[j]]
            mjd1 = calinfo['mjd'][torun[j]]
            if mjd1>=59556:
                fps = True
            else:
                fps = False
            cmd1 = 'makecal --vers {0} --telescope {1} --unlock'.format(apred,telescope)
            if clobber: cmd1 += ' --clobber'                
            if caltype=='psf':    # psfs
                cmd1 += ' --psf '+str(num1)
                logfile1 = calplandir+'/apPSF-'+str(num1)+'_pbs.'+logtime+'.log'
            elif caltype=='flux':   # flux
                cmd1 += ' --flux '+str(num1)
                if fps: cmd1 += ' --librarypsf'
                logfile1 = calplandir+'/apFlux-'+str(num1)+'_pbs.'+logtime+'.log'
            elif caltype=='arcs':  # arcs
                cmd1 += ' --wave '+str(num1)
                if fps: cmd1 += ' --librarypsf'
                logfile1 = calplandir+'/apWave-'+str(num1)+'_pbs.'+logtime+'.log' 
            elif caltype=='dailywave':  # dailywave
                cmd1 += ' --dailywave '+str(num1)
                if fps: cmd1 += ' --librarypsf'
                logfile1 = calplandir+'/apDailyWave-'+str(num1)+'_pbs.'+logtime+'.log' 
            elif caltype=='fpi':  # fpi
                cmd1 += ' --fpi '+str(num1)
                if fps: cmd1 += ' --librarypsf'
                logfile1 = calplandir+'/apFPI-'+str(num1)+'_pbs.'+logtime+'.log'
            if os.path.exists(os.path.dirname(logfile1))==False:
                os.makedirs(os.path.dirname(logfile1))
            logfiles.append(logfile1)
            logger.info('Calibration file %d : %s %d' % (j+1,caltype,num1))
            logger.info('Command : '+cmd1)
            logger.info('Logfile : '+logfile1)
            queue.append(cmd1, outfile=logfile1,errfile=logfile1.replace('.log','.err'))
        if ntorun>0:
            queue.commit(hard=True,submit=True)
            logger.info('PBS key is '+queue.key)
            queue_wait(queue,sleeptime=60,verbose=True,logger=logger)  # wait for jobs to complete
        else:
            logger.info('No '+str(calnames[i])+' calibration files need to be run')
        # Checks the status and updates the database
        if ntorun>0:
            chkcal1 = check_calib(calinfo[torun],logfiles,queue.key,apred,verbose=True,logger=logger)
            if len(chkcal)==0:
                chkcal = chkcal1
            else:
                chkcal = np.hstack((chkcal,chkcal1))
        del queue


    # make sure to run mkwave on all arclamps needed for daily cals


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
            dbload_plans(planfiles1)  # load plans into db
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
    dorun = np.zeros(len(planfiles),bool)
    for i,pf in enumerate(planfiles):
        pfbase = os.path.basename(pf)
        logger.info('planfile %d : %s' % (i+1,pf))
        logfile1 = pf.replace('.yaml','_pbs.'+logtime+'.log')
        errfile1 = logfile1.replace('.log','.err')
        cmd1 = 'apred {0}'.format(pf)
        if clobber:
            cmd1 += ' --clobber'
        # Check if files exist already
        dorun[i] = True
        if clobber is not True:
            # apPlan
            if pfbase.startswith('apPlan'):
                # apPlan-3370-59623.yaml
                config1,mjd1 = pfbase.split('.')[0].split('-')[1:3]
                # check for apVisitSum file
                outfile = load.filename('VisitSum',plate=config1,mjd=mjd1,chips=True)
                outexists = os.path.exists(outfile)
            # apCalPlan
            elif pfbase.startswith('apCalPlan'):
                # apCalPlan-apogee-n-59640.yaml
                # It will take too long to load all of the plan files and check all of
                #  the output files.  Default is to redo them.
                outexists = False
                outfile = pf+' output files '
            # apDarkPlan
            elif pfbase.startswith('apDarkPlan'):
                # apDarkPlan-apogee-n-59640.yaml
                # It will take too long to load all of the plan files and check all of
                #  the output files.  Default is to redo them.
                outexists = False
                outfile = pf+' output files '
            # apExtraPlan
            elif pfbase.startswith('apExtraPlan'):
                # apExtraPlan-apogee-n-59629.yaml
                # It will take too long to load all of the plan files and check all of
                #  the output files.  Default is to redo them.
                outexists = False
                outfile = pf+' output files '
            if outexists:
                logger.info(os.path.basename(outfile)+' already exists and clobber==False')
                dorun[i] = False
        if dorun[i]:
            logger.info('planfile %d : %s' % (i+1,pf))
            logger.info('Command : '+cmd1)
            logger.info('Logfile : '+logfile1)
            queue.append(cmd1, outfile=logfile1,errfile=errfile1)
    if np.sum(dorun)>0:
        queue.commit(hard=True,submit=True)
        logger.info('PBS key is '+queue.key)
        queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
    else:
        logger.info('No planfiles need to be run')

    # This also loads the status into the database using the correct APRED version
    chkexp,chkvisit = check_apred(expinfo,planfiles,queue.key,verbose=True,logger=logger)
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
        queue_wait(queue,sleeptime=60,verbose=True,logger=logger)  # wait for jobs to complete
        # This checks the status and puts it into the database
        ind, = np.where(dorv)
        chkrv = check_rv(vcat[ind],queue.key,logger=logger,verbose=False)
    else:
        logger.info('No RVs need to be run')
        chkrv = None
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
    create_sumfiles(apred,telescope,logger=logger)


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
    queue_wait(queue,sleeptime=60,verbose=True,logger=logger)  # wait for jobs to complete
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
    queue_wait(queue,sleeptime=60,logger=logger,verbose=True)  # wait for jobs to complete 
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
    if nmjd>1:
        subject = 'APOGEE DRP Reduction %s %s %s-%s' % (observatory,apred,mjdstart,mjdstop)
    else:
        subject = 'APOGEE DRP Reduction %s %s %s' % (observatory,apred,mjdstart)
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
    if 'master' in steps and chkmaster is not None:
        ind, = np.where(chkmaster['success']==True)
        message += '%d/%d Master calibrations successfully processed<br> \n' % (len(ind),len(chkmaster))
        
    # AP3D step
    if '3d' in steps and chk3d is not None:
        ind, = np.where(chk3d['success']==True)
        message += 'AP3D: %d/%d exposures successfully processed through AP3D<br> \n' % (len(ind),len(chk3d))

    # Daily Cals step
    if 'cal' in steps and chkcal is not None:
        ind, = np.where(chkcal['success']==True)
        message += 'Cal: %d/%d daily calibrations successfully processed<br> \n' % (len(ind),len(chkcal))
 
    # Plan files
    if 'plan' in steps and planfiles is not None:
        message += 'Plan: %d plan files successfully made<br> \n' % len(planfiles)

    # APRED step
    if 'apred' in steps and chkapred is not None:
        ind, = np.where(chkapred['success']==True)
        message += 'APRED: %d/%d visits successfully processed<br> \n' % (len(ind),len(chkapred))

    # RV step
    if 'rv' in steps and chkrv is not None:
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
        queue_wait(queue,sleeptime=10,logger=rootLogger,verbose=True)  # wait for jobs to complete 
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
    
    if nmjd>1:
        rootLogger.info('APOGEE DRP reduction finished for %s APRED=%s MJD=%d to MJD=%d' % (observatory,apred,mjdstart,mjdstop))
    else:
        rootLogger.info('APOGEE DRP reduction finished for %s APRED=%s MJD=%d' % (observatory,apred,mjdstart))


    # Summary email
    summary_email(observatory,apred,mjd,steps,chkmaster=chkmaster,chk3d=chk3d,chkcal=chkcal,
                  planfiles=planfiles,chkapred=chkapred,chkrv=chkrv,logfile=logfile,slurm=slurm,
                  clobber=clobber,debug=debug)

