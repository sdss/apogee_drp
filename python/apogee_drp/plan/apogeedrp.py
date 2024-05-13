import copy
import numpy as np
import os
import shutil
from glob import glob
import pdb

from dlnpyutils import utils as dln
from ..utils import spectra,yanny,apload,platedata,plan,email,info,slurm as slrm
from ..apred import mkcal,cal,qa,monitor
from ..database import apogeedb
from . import mkplan,check
from sdss_access.path import path
from astropy.time import Time
from astropy.io import fits
from astropy.table import Table,hstack,vstack
from collections import OrderedDict
#from astropy.time import Time
from datetime import datetime
import logging
#import slurm
#from slurm import queue as pbsqueue
import time
import traceback
import subprocess

chips = ['a','b','c']

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
        mjds = np.arange(59146,lastnightmjd5()+1)
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
    dtype = np.dtype([('planfile',(str,300)),('apred_vers',(str,20)),('v_apred',(str,50)),('telescope',(str,10)),
                      ('instrument',(str,20)),('mjd',int),('plate',int),('configid',(str,20)),('designid',(str,20)),
                      ('fieldid',(str,20)),('fps',bool),('platetype',(str,20))])
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
    expinfo = []
    for m in mjds:
        # Get exposure information from the flat files
        expinfo1 = info.expinfo(observatory=observatory,mjd5=m)
        expinfo1 = Table(expinfo1)
        # Get exposure iformation from the database
        dbexpinfo = db.query('exposure',where="mjd=%d and observatory='%s'" % (m,observatory))
        # No exposures for this night
        if len(expinfo1)==0 and len(dbexpinfo)==0:
            logger.info('MJD={:}  {:3d} exposures'.format(m,0))
            continue
        if len(expinfo1)>0 and verbose:
            logger.info('MJD={:}  {:3d} exposures'.format(m,len(expinfo1)))
        # Crossmatch the two catalogs
        if len(expinfo1)>0 and len(dbexpinfo)>0:
            vals,ind1,ind2 = np.intersect1d(expinfo1['num'],dbexpinfo['num'],return_indices=True)
        else:
            ind1 = []
        # If there are new exposures then update the database
        doupdate = False   # default is no db update        
        if len(ind1) != len(expinfo1) and len(expinfo1)>0:
            doupdate = True
        # Early on the FPI exposures were labeled 'ARCLAMP', fix those in the database
        if len(dbexpinfo)>0 and (np.sum(expinfo1['exptype']=='FPI') > np.sum(dbexpinfo['exptype']=='FPI')):
            doupdate = True
        # Update the database
        #  for exposures that already exist in the db, it will update them
        if doupdate:
            db.ingest('exposure',expinfo1)  # insert into database
            # redo the query
            expinfo1 = db.query('exposure',where="mjd=%d and observatory='%s'" % (m,observatory))            
            expinfo1 = Table(expinfo1)
        else:
            expinfo1 = Table(dbexpinfo)
        # Stack the MJD exposure catalogs
        if len(expinfo)==0:
            expinfo = Table(expinfo1)
        else:
            if len(expinfo1)>0:
                # this might cause issues with some string columns
                #  if the lengths don't agree
                expinfo = vstack((expinfo,expinfo1))

    nexp = len(expinfo)
    logger.info(str(nexp)+' exposures')
    if len(expinfo)==0:
        return expinfo
    
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
    plans = db.query('plan',where="mjd>='%s' and mjd<='%s' and apred_vers='%s' and telescope='%s'" % (mjdstart,mjdstop,load.apred,load.telescope))
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


def check_mastercals(names,caltype,logfiles,pbskey,apred,telescope,verbose=False,logger=None):
    """ Check that the master calibration files ran okay and load into database."""

    if logger is None:
        logger = dln.basiclogger()

    chkmaster = None

    if type(names) is str:
        names = [names]
    nnames = len(names)
    if type(caltype) is str:
        caltype = nnames*[caltype]
    
    if verbose==True:
        logger.info('')
        logger.info('--- Checking Master Calibration runs ---')

    # Exposure-level processing: ap3d, ap2d, calibration file
    dtype = np.dtype([('logfile',(str,300)),('apred_vers',(str,20)),('v_apred',(str,50)),
                      ('instrument',(str,20)),('telescope',(str,10)),('caltype',(str,30)),
                      ('pbskey',(str,50)),('checktime',(str,100)),
                      ('name',str,100),('calfile',(str,300)),('success',bool)])
    chkmaster = np.zeros(nnames,dtype=dtype)
    chkmaster['telescope'] = telescope
    if telescope[0:3]=='apo':
        chkmaster['instrument'] = 'apogee-n'
    else:
        chkmaster['instrument'] = 'apogee-s'        
    
    # Loop over the files
    for i in range(nnames):
        lgfile = logfiles[i]
        name = names[i]
        chkmaster['logfile'][i] = lgfile
        chkmaster['name'][i] = name
        chkmaster['apred_vers'][i] = apred
        chkmaster['caltype'][i] = caltype[i]
        chkmaster['pbskey'][i] = pbskey
        chkmaster['checktime'][i] = str(datetime.now())
        chkmaster['success'][i] = False
        load = apload.ApLoad(apred=apred,telescope=chkmaster['telescope'][i])
        # Final calibration file
        #-----------------------
        base = load.filename(caltype[i],num=name,chips=True)
        chkmaster['calfile'][i] = base
        # Sparse, only one file, not chip tag
        if caltype[i]=='Sparse':
            chfiles = [base]
        # Littrow, only detector b
        elif caltype[i]=='Littrow':
            chfiles = [base.replace(caltype[i]+'-',caltype[i]+'-b-')]
        else:
            chfiles = [base.replace(caltype[i]+'-',caltype[i]+'-'+ch+'-') for ch in ['a','b','c']]
        chinfo = info.file_status(chfiles)
        if chinfo['okay'][0]:  # get V_APRED (git version) from file
            chead = fits.getheader(chfiles[0])
            chkmaster['v_apred'][i] = chead.get('V_APRED')
        # Overall success
        if load.exists(caltype[i],num=name):
            chkmaster['success'][i] = True

        if verbose:
            logger.info('')
            logger.info('%d/%d' % (i+1,nnames))
            logger.info('Master calibration type: %s' % chkmaster['caltype'][i])
            logger.info('Master calibration file: %s' % chkmaster['calfile'][i])
            logger.info('log/errfile: '+lgfile+'/.err')
            logger.info('Master calibration success: %s ' % chkmaster['success'][i])

    # Load everything into the database
    db = apogeedb.DBSession()
    db.ingest('mastercal_status',chkmaster)
    db.close()

    return chkmaster


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

    dtype = np.dtype([('exposure_pk',int),('planfile',(str,300)),('apred_vers',(str,20)),('v_apred',(str,50)),
                      ('instrument',(str,20)),('telescope',(str,10)),('platetype',(str,50)),('mjd',int),
                      ('plate',int),('configid',(str,20)),('designid',(str,20)),('fieldid',(str,20)),
                      ('proctype',(str,30)),('pbskey',(str,50)),('checktime',(str,100)),
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
        outinfo = info.file_status(outfiles)
        if outinfo['okay'][0]:
            head = fits.getheader(outfiles[0])
            chk3d['v_apred'][i] = head.get('V_APRED')
            head = fits.getheader(outfiles[0])
            chk3d['v_apred'][i] = head.get('V_APRED')
            head = fits.getheader(outfiles[0])
            chk3d['v_apred'][i] = head.get('V_APRED')
            head = fits.getheader(outfiles[0])
            chk3d['v_apred'][i] = head.get('V_APRED')
        chk3d['checktime'][i] = str(datetime.now())
        chk3d['success'][i] = np.sum(outinfo['exists'])==3

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
    dtype = np.dtype([('logfile',(str,300)),('apred_vers',(str,20)),('v_apred',(str,50)),
                      ('instrument',(str,20)),('telescope',(str,10)),('mjd',int),('caltype',(str,30)),
                      ('plate',int),('configid',(str,20)),('designid',(str,20)),('fieldid',(str,20)),
                      ('pbskey',(str,50)),('checktime',(str,100)),
                      ('num',(str,100)),('calfile',(str,300)),('success3d',bool),('success2d',bool),('success',bool)])
    chkcal = np.zeros(ncal,dtype=dtype)

    # Loop over the files
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
            chinfo3 = info.file_status(chfiles)
            if chinfo3['okay'][0] and chinfo3['size'][0]>0: # get V_APRED (git version) from file 
                chead = fits.getheader(chfiles[0])
                chkcal['v_apred'][i] = chead.get('V_APRED')
            if np.sum(chinfo3['okay'])==3:
                chkcal['success3d'][i] = True
        else:
            chkcal['success3d'][i] = True
        # AP2D
        #-----
        if caltype != 'DailyWave':
            base = load.filename('1D',num=num,mjd=mjd,chips=True)
            chfiles = [base.replace('1D-','1D-'+ch+'-') for ch in ['a','b','c']]
            chinfo2 = info.file_status(chfiles)
            if np.sum(chinfo2['okay'])==3:
                chkcal['success2d'][i] = True
        else:
            chkcal['success2d'][i] = True
        # Final calibration file
        #-----------------------
        if caltype.lower()=='fpi':
            # Should really check fpi/apFPILines-EXPNUM8.fits
            base = load.filename('Wave',num=num,chips=True).replace('Wave-','WaveFPI-'+str(mjd)+'-')
        elif caltype.lower()=='dailywave':
            base = load.filename('Wave',num=num,chips=True)[0:-13]+str(num)+'.fits'
        else:
            base = load.filename(caltype,num=num,chips=True)
        chkcal['calfile'][i] = base
        chfiles = [base.replace(filecode+'-',filecode+'-'+ch+'-') for ch in ['a','b','c']]
        chinfo = info.file_status(chfiles)
        if chinfo['okay'][0] and chinfo['size'][0]>0:   # get V_APRED (git version) from file
            chead = fits.getheader(chfiles[0])
            chkcal['v_apred'][i] = chead.get('V_APRED')
        # Overall success
        if np.sum(chinfo['okay'])==3:
            chkcal['success'][i] = True

        if verbose:
            logger.info('')
            logger.info('%d/%d' % (i+1,ncal))
            logger.info('Calibration type: %s' % chkcal['caltype'][i])
            logger.info('Calibration file: %s' % chkcal['calfile'][i])
            logger.info('log/errfile: '+lgfile+'/.err')
            logger.info('Calibration success: %s ' % chkcal['success'][i])

    # Load everything into the database
    db = apogeedb.DBSession()
    db.ingest('calib_status',chkcal)
    db.close()

    return chkcal


def check_apred(expinfo,planfiles,pbskey,verbose=False,dbload=True,logger=None):
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
        if planstr['fps']:
            plate = planstr['configid']
            design = planstr['designid']
            field = planstr['fieldid']
        else:
            field,survey,program = apload.apfield(planstr['plateid'])
        
        # normal: ap3d, ap2d, apCframe and ap1dvisit
        # dark: ap3d  (apDarkPlan)
        # cal: ap3d and ap2d (apCalPlan)

        # Load the plugmap information
        if platetype=='normal' and str(plate) != '0':
            #plugmap = platedata.getdata(plate,mjd,apred_vers,telescope,plugid=planstr['plugmap'])
            #fiberdata = plugmap['fiberdata']
            # This is very slow, get it directly from apPlate file
            base = load.filename('Plate',plate=plate,mjd=mjd,chips=True,field=field)
            chfiles = [base.replace('Plate-','Plate-'+ch+'-') for ch in ['a','b','c']]
            chinfo = info.file_status(chfiles)
            if chinfo['okay'][0]:
                fiberdata = Table.read(chfiles[0],11)
            else:
                logger.info('Cannot get fiber data from '+chfiles[0])
                logger.info('info: okay ',chinfo['okay'][0])
                logger.info('info: exists ',chinfo['exists'][0])
                logger.info('info: size ',chinfo['size'][0])                
                fiberdata = None
        else:
            fiberdata = None
        if fiberdata is None:
            logger.info('No fiber data!')
            
        # Exposure-level processing: ap3d, ap2d, apcframe
        dtype = np.dtype([('exposure_pk',int),('planfile',(str,300)),('apred_vers',(str,20)),('v_apred',(str,50)),
                          ('instrument',(str,20)),('telescope',(str,10)),('platetype',(str,50)),('mjd',int),
                          ('plate',int),('configid',(str,20)),('designid',(str,20)),('fieldid',(str,20)),
                          ('proctype',(str,30)),('pbskey',(str,50)),('checktime',(str,100)),
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
            field,survey,program = apload.apfield(planstr['plateid'])
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
            chinfo3 = info.file_status(chfiles)
            if chinfo3['okay'][0]:  # get V_APRED (git version) from file
                chead = fits.getheader(chfiles[0])
                chkexp1['v_apred'][cnt] = chead.get('V_APRED')
            chkexp1['checktime'][cnt] = str(datetime.now())
            if np.sum(chinfo3['okay'])==3:
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
                chinfo2 = info.file_status(chfiles)
                if chinfo2['okay'][0]:  # get V_APRED (git version) from file
                    chead = fits.getheader(chfiles[0])
                    chkexp1['v_apred'][cnt] = chead.get('V_APRED')
                if np.sum(chinfo2['okay'])==3:
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
                chinfo = info.file_status(chfiles)
                if chinfo['okay'][0]:  # get V_APRED (git version) from file
                    chead = fits.getheader(chfiles[0])
                    chkexp1['v_apred'][cnt] = chead.get('V_APRED')
                if np.sum(chinfo['okay'])==3:
                    chkexp1['success'][cnt] = True
                cnt += 1  # increment counter
        # Trim extra elements
        chkexp1 = chkexp1[0:cnt]

        # Plan summary and ap1dvisit
        #---------------------------
        dtypeap = np.dtype([('planfile',(str,300)),('logfile',(str,300)),('errfile',(str,300)),
                            ('apred_vers',(str,20)),('v_apred',(str,50)),('instrument',(str,20)),
                            ('telescope',(str,10)),('platetype',(str,50)),('mjd',int),('plate',int),
                            ('configid',(str,20)),('designid',(str,20)),('fieldid',(str,20)),
                            ('nobj',int),('pbskey',(str,50)),('checktime',(str,100)),('ap3d_success',bool),
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
            objtype = np.char.array(fiberdata['OBJTYPE']).astype(str)
            chkap1['nobj'] = np.sum((fiberdata['FIBERID']>-1) & ((objtype=='STAR') | (objtype=='HOT_STD'))) # stars and tellurics 
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
            chinfo = info.file_status(chfiles)
            if chinfo['okay'][0]:  # get V_APRED (git version) from file
                chead = fits.getheader(chfiles[0])
                chkap1['v_apred'] = chead.get('V_APRED')
            if np.sum(chinfo['okay'])==3:
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
            logger.info('log/errfile: '+chkap1['logfile'][0]+'/.err')
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
    if dbload:
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
    dtype = np.dtype([('apogee_id',(str,50)),('apred_vers',(str,20)),('v_apred',(str,50)),
                      ('telescope',(str,10)),('healpix',int),('nvisits',int),('pbskey',(str,50)),
                      ('file',(str,300)),('checktime',(str,100)),('success',bool)])
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
             'apogee_target4', 'catalogid', 'sdss_id', 'gaia_release', 'gaia_plx', 'gaia_plx_error', 'gaia_pmra', 'gaia_pmra_error',
             'gaia_pmdec', 'gaia_pmdec_error', 'gaia_gmag', 'gaia_gerr', 'gaia_bpmag', 'gaia_bperr',
             'gaia_rpmag', 'gaia_rperr', 'sdssv_apogee_target0', 'firstcarton', 'targflags', 'snr', 'starflag',
             'starflags','dateobs','jd']
    rvcols = ['starver', 'bc', 'vtype', 'vrel', 'vrelerr', 'vrad', 'chisq', 'rv_teff', 'rv_feh',
              'rv_logg', 'xcorr_vrel', 'xcorr_vrelerr', 'xcorr_vrad', 'n_components', 'rv_components']

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
    #allstar = db.query('star_latest',cols='*',where="apred_vers='"+apred+"' and telescope='"+telescope+"'")
    vstar = db.query('star',cols='*',where="apred_vers='"+apred+"' and telescope='"+telescope+"'")
    # Deal with multiple STARVER versions per star
    star_index = dln.create_index(vstar['apogee_id'])
    ndups = np.sum(star_index['num']>1)
    if ndups>0:
        allstar = np.zeros(len(star_index['value']),dtype=vstar.dtype)
        for i,obj in enumerate(star_index['value']):
            if star_index['num'][i]>1:
                ind = star_index['index'][star_index['lo'][i]]
                allstar[i] = vstar[ind]
            else:
                ind = star_index['index'][star_index['lo'][i]:star_index['hi'][i]+1]
                starver = vstar['starver'][ind]
                si = np.argsort(starver)
                useind = ind[si[-1]]   # use last/largest STARVER
                allstar[i] = vstar[useind]
    else:
        allstar = vstar
    
    allstarfile = load.filename('allStar')#.replace('.fits','-'+telescope+'.fits')
    logger.info('Writing allStar file to '+allstarfile)
    logger.info(str(len(allstar))+' stars')
    if os.path.exists(os.path.dirname(allstarfile))==False:
        os.makedirs(os.path.dirname(allstarfile))
    allstar = Table(allstar)
    if 'nres' in allstar.colnames:
        del allstar['nres']    # temporary kludge, nres is causing write problems
    allstar.write(allstarfile,overwrite=True)

    # allVisit
    # Same thing for visit except that we'll get the multiple visit rows returned for each unique star row
    #   Get more info by joining with the visit table.
    vcols = ['apogee_id', 'target_id', 'apred_vers','file', 'uri', 'fiberid', 'plate', 'mjd', 'telescope', 'survey',
             'field', 'programname', 'ra', 'dec', 'glon', 'glat', 'jmag', 'jerr', 'hmag',
             'herr', 'kmag', 'kerr', 'src_h', 'pmra', 'pmdec', 'pm_src', 'apogee_target1', 'apogee_target2', 'apogee_target3',
             'apogee_target4', 'catalogid', 'sdss_id', 'gaia_release', 'gaia_plx', 'gaia_plx_error', 'gaia_pmra', 'gaia_pmra_error',
             'gaia_pmdec', 'gaia_pmdec_error', 'gaia_gmag', 'gaia_gerr', 'gaia_bpmag', 'gaia_bperr',
             'gaia_rpmag', 'gaia_rperr', 'sdssv_apogee_target0', 'firstcarton', 'targflags', 'snr', 'starflag',
             'starflags','dateobs','jd']
    rvcols = ['starver', 'bc', 'vtype', 'vrel', 'vrelerr', 'vrad', 'chisq', 'rv_teff', 'rv_feh',
              'rv_logg', 'xcorr_vrel', 'xcorr_vrelerr', 'xcorr_vrad', 'n_components', 'rv_components']

    # Straight join query of visit and rv_visit
    cols = np.hstack(('v.'+np.char.array(vcols),'rv.'+np.char.array(rvcols)))
    sql = 'select '+','.join(cols)+' from apogee_drp.visit as v LEFT JOIN apogee_drp.rv_visit as rv ON rv.visit_pk=v.pk'
    sql += " where v.apred_vers='"+apred+"' and v.telescope='"+telescope+"'"
    allvisit = db.query(sql=sql)

    # Fix bad STARVER values
    bdstarver, = np.where(np.char.array(allvisit['starver']) == '')
    if len(bdstarver)>0:
        allvisit['starver'][bdstarver] = allvisit['mjd'][bdstarver]
    # Check for duplicate STARVER for each star
    idindex = dln.create_index(allvisit['apogee_id'])
    duplicate = np.zeros(len(allvisit),bool)
    for i in range(len(idindex['value'])):
        ind = idindex['index'][idindex['lo'][i]:idindex['hi'][i]+1]
        allv = allvisit[ind]
        if np.min(allv['starver'].astype(int)) != np.max(allv['starver'].astype(int)):
            # Only keep rows for the maximum STARVER per star
            maxstarver = np.max(allv['starver'].astype(int))
            bd1, = np.where(allv['starver'].astype(int) != maxstarver)
            duplicate[ind[bd1]] = True
    torem, = np.where(duplicate==True)
    if len(torem)>0:
        allvisit = np.delete(allvisit,torem)
                
    # Use visit_latest, this can sometimes take forever
    #cols = ','.join(vcols+rvcols)        
    #allvisit = db.query('visit_latest',cols=cols,where="apred_vers='"+apred+"' and telescope='"+telescope+"'")
    # rv_components can sometimes be an object type
    if allvisit.dtype['rv_components'] == np.object:
        allvisit = Table(allvisit)
        allvisit['rv_components'] = np.zeros(len(allvisit),dtype=np.dtype((np.float32,3)))
    allvisitfile = load.filename('allVisit')#.replace('.fits','-'+telescope+'.fits')
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
            # Make cron directory
            if os.path.exists(apogee_redux+apred+'/'+d+'/'+obs+'/cron')==False:
                logger.info('Creating '+apogee_redux+apred+'/'+d+'/'+obs+'/cron')
                os.makedirs(apogee_redux+apred+'/'+d+'/'+obs+'/cron')
            else:
                logger.info(apogee_redux+apred+'/'+d+'/'+obs+'/cron already exists')            
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

    # Make local scratch directory for this version
    localdir = os.environ['APOGEE_LOCALDIR']
    localdir += '/'+apred
    if os.path.exists(localdir)==False:
        os.makedirs(localdir)
        

def mkmastercals(load,mjds,slurmpars,caltypes=None,clobber=False,linkvers=None,logger=None):
    """
    Make the master calibration products for a reduction version and MJD range.

    Parameters
    ----------
    load : ApLoad object
       ApLoad object that contains "apred" and "telescope".
    mjds : list
       List of MJDs to process
    slurmpars : dictionary
       Dictionary of slurmpars settings.
    caltypes : list, optional
       List of master calibration types to run.  The default is all 9 of them.
       ['detector','dark','flat','bpm','sparse','littrow','response','modelpsf','lsf']
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

    if caltypes is None:
        caltypes = ['detector','dark','flat','bpm','sparse','littrow','response','modelpsf','multiwave','lsf']
    else:
        caltypes = [c.lower() for c in caltypes]
        
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
        for d in ['bpm','darkcorr','detector','flatcorr','littrow','lsf','persist','sparse','fiber','modelpsf','fpi']:
            for obs in ['apogee-n','apogee-s']:
                prefix = {'apogee-n':'ap','apogee-s':'as'}[obs]
                srcdir = apogee_redux+linkvers+'/cal/'+obs+'/'+d
                destdir = apogee_redux+apred+'/cal/'+obs+'/'+d
                if d=='sparse' or d=='modelpsf' or d=='fiber':
                    srcdir = apogee_redux+linkvers+'/cal/'+obs+'/psf'
                    destdir = apogee_redux+apred+'/cal/'+obs+'/psf'
                logger.info('Creating symlinks for '+d+' '+obs)
                os.chdir(destdir)
                # PSFModel
                if d=='modelpsf':
                    psfmfiles = glob(srcdir+'/'+prefix+'PSFModel-*.fits')
                    if len(psfmfiles)>0:
                        subprocess.run(['ln -s '+srcdir+'/'+prefix+'PSFModel-*.fits .'],shell=True)
                    else:
                        logger.info('No PSFModel files')
                # Sparse
                elif d=='sparse':
                    sparsefiles = glob(srcdir+'/'+prefix+'Sparse*.fits')
                    if len(sparsefiles)>0:
                        subprocess.run(['ln -s '+srcdir+'/'+prefix+'Sparse*.fits .'],shell=True)
                    else:
                        logger.info('No Sparse files')
                    # Need to link apEPSF, apPSF and apETrace files as well
                    #   otherwise the daily cal PSF stage will try to remake them
                    sfiles = glob(srcdir+'/'+prefix+'Sparse*.fits')
                    if len(sfiles)>0:
                        snum = [os.path.basename(s)[9:-5] for s in sfiles]
                        for num in snum:
                            subprocess.run(['ln -s '+srcdir+'/'+prefix+'EPSF-?-'+num+'.fits .'],shell=True)
                            subprocess.run(['ln -s '+srcdir+'/'+prefix+'PSF-?-'+num+'.fits .'],shell=True)                            
                            trcdir = apogee_redux+linkvers+'/cal/'+obs+'/trace'
                            dtrcdir = apogee_redux+apred+'/cal/'+obs+'/trace'                            
                            subprocess.run(['ln -s '+trcdir+'/'+prefix+'ETrace-?-'+num+'.fits '+dtrcdir],shell=True)
                # Darks and Flats
                elif d=='darkcorr' or d=='flatcorr':
                    darkfiles = glob(srcdir+'/*.fits')
                    if len(darkfiles)>0:
                        subprocess.run(['ln -s '+srcdir+'/*.fits .'],shell=True)
                    else:
                        logger.info('No '+os.path.basename(srcdir)+' files ')
                    tabfiles = glob(srcdir+'/*.tab')
                    if len(tabfiles)>0:
                        subprocess.run(['ln -s '+srcdir+'/*.tab .'],shell=True)
                    else:
                        logger.info('No '+os,path.basename(srcdir)+' .tab files')
                # Detector
                elif d=='detector':
                    detfiles = glob(srcdir+'/*.fits')
                    if len(detfiles)>0:
                        subprocess.run(['ln -s '+srcdir+'/*.fits .'],shell=True)
                    else:
                        logger.info('No Detector files')
                    detdatfiles = glob(srcdir+'/*.dat')
                    if len(detdatfiles)>0:
                        subprocess.run(['ln -s '+srcdir+'/*.dat .'],shell=True)
                    else:
                        logger.info('No Detector .dat files')
                # LSF
                elif d=='lsf':
                    lsffiles = glob(srcdir+'/*.fits')
                    if len(lsffiles)>0:
                        subprocess.run(['ln -s '+srcdir+'/*.fits .'],shell=True)
                    else:
                        logger.info('No LSF files')
                    lsfsavfiles = glob(srcdir+'/*.sav')
                    if len(lsfsavfiles)>0:
                        subprocess.run(['ln -s '+srcdir+'/*.sav .'],shell=True)
                    else:
                        logger.info('No LSF .sav files')
                ## FPI
                # fpi_peaks.fits is now in the repo (apogee_drp/data/arclines/fpi_peaks.fits)
                #elif d=='fpi':
                #    fpifiles = glob(srcdir+'/fpi_peaks.fits')
                #    if len(fpifiles)>0:
                #        subprocess.run(['ln -s '+srcdir+'/fpi_peaks.fits .'],shell=True)
                #    else:
                #        logger.info('No fpi_peaks.fits files')
                # Fiber
                elif d=='fiber':
                    # Create symlinks for all the fiber cal files, PSF, EPSF, ETrace
                    caldict = mkcal.readcal(caldir+obs+'.par')
                    fiberdict = caldict['fiber']
                    for f in fiberdict['name']:
                        epsffiles = glob(srcdir+'/'+prefix+'EPSF-?-'+f+'.fits')
                        if len(epsffiles)>0:
                            subprocess.run(['ln -s '+srcdir+'/'+prefix+'EPSF-?-'+f+'.fits .'],shell=True)
                        else:
                            logger.info('No EPSF files found for fiber '+prefix+'EPSF-?-'+f+'.fits')
                        psffiles = glob(srcdir+'/'+prefix+'PSF-?-'+f+'.fits')
                        if len(psffiles)>0:
                            subprocess.run(['ln -s '+srcdir+'/'+prefix+'PSF-?-'+f+'.fits .'],shell=True)
                        else:
                            logger.info('No PSF files found for fiber '+prefix+'PSF-?-'+f+'.fits')                            
                        tsrcdir = apogee_redux+linkvers+'/cal/'+obs+'/trace'
                        tdestdir = apogee_redux+apred+'/cal/'+obs+'/trace'
                        os.chdir(tdestdir)
                        tracefiles = glob(tsrcdir+'/'+prefix+'ETrace-?-'+f+'.fits')
                        if len(tracefiles)>0:
                            subprocess.run(['ln -s '+tsrcdir+'/'+prefix+'ETrace-?-'+f+'.fits .'],shell=True)
                        else:
                            logger.info('No ETrace files found for fiber '+prefix+'ETrace-?-'+f+'.fits')                            
                        os.chdir(destdir)
                else:
                    cfiles = glob(srcdir+'/*.fits')
                    if len(cfiles)>0:
                        subprocess.run(['ln -s '+srcdir+'/*.fits .'],shell=True)
                    else:
                        logger.info('No '+os.path.basename(srcdir)+' .fits files')
                        
        # Link all of the PSF files in the PSF library
        psflibrary = False
        if psflibrary:
            for obs in ['apogee-n','apogee-s']:
                logger.info('Creating symlinks for PSF library files '+obs)
                tscope = {'apogee-n':'apo25m','apogee-s':'lco25m'}
                sload = apload.ApLoad(apred=linkvers,telescope=tscope)
                dload = apload.ApLoad(apred=apred,telescope=tscope)
                dpsflibraryfile = os.environ['APOGEE_REDUX']+'/'+linkvers+'/monitor/'+obs+'DomeFlatTrace-all.fits'
                qpsflibraryfile = os.environ['APOGEE_REDUX']+'/'+linkvers+'/monitor/'+obs+'QuartzFlatTrace-all.fits'
                psfid = []
                if os.path.exists(dpsflibraryfile):
                    dpsf = Table.read(dpsflibraryfile)
                    psfid = np.array(dpsf['PSFID'])
                else:
                    logger.info(dpsflibraryfile+' not found')
                if os.path.exists(qpsflibraryfile):
                    qpsf = Table.read(qpsflibraryfile)
                    if len(psfid)==0:
                        psfid = np.array(qpsf['PSFID'])
                    else:
                        psfid = np.hstack((psfid,np.array(qpsf['PSFID'])))
                else:
                    logger.info(qpsflibraryfile+' not found')
                if len(psfid)==0:
                    continue
                # Loop over the files and link them
                psfid = np.unique(psfid)
                for i in range(len(psfid)):
                    srcfile = sload.filename('PSF',num=psfid[i],chips=True)
                    destfile = dload.filename('PSF',num=psfid[i],chips=True)
                    for ch in chips:
                        srcfile1 = srcfile.replace('PSF-','PSF-'+ch+'-')
                        destfile1 = destfile.replace('PSF-','PSF-'+ch+'-')
                        if os.path.exists(srcfile1):
                            subprocess.run(['ln -s '+srcfile1+' '+destfile1],shell=True)
                            srcfile1 = srcfile.replace('PSF-','EPSF-'+ch+'-')
                            destfile1 = destfile.replace('PSF-','EPSF-'+ch+'-')
                            subprocess.run(['ln -s '+srcfile1+' '+destfile1],shell=True)
                            srcfile1 = sload.filename('ETrace',num=psfid[i],chips=True).replace('ETrace-','ETrace-'+ch+'-')
                            destfile1 = load.filename('ETrace',num=psfid[i],chips=True).replace('ETrace-','ETrace-'+ch+'-')
                            subprocess.run(['ln -s '+srcfile1+' '+destfile1],shell=True)

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

    
    chkmaster = None

    # Make Detector and Linearity 
    #----------------------------
    if 'detector' in caltypes:
        # MKDET.PRO makes both
        detdict = allcaldict['det']
        logger.info('--------------------------------------------')
        logger.info('Making master Detector/Linearity in parallel')
        logger.info('============================================')
        if detdict is None or len(detdict)==0:
            detdict = []
            logger.info('No master Detector calibration files to make')
        slurmpars1 = slurmpars.copy()
        slurmpars1['nodes'] = 1
        slurmpars1['memory'] = 55000  # in MB
        logger.info('Slurm settings: '+str(slurmpars1))
        dt = [('cmd',str,1000),('name',str,1000),('outfile',str,1000),('errfile',str,1000),('dir',str,1000)] 
        tasks = np.zeros(len(detdict),dtype=np.dtype(dt))
        tasks = Table(tasks)
        docal = np.zeros(len(detdict),bool)
        donames = []
        logfiles = []
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
                # Check if files exist already
                docal[i] = True
                if clobber is not True:
                    if load.exists('Detector',num=name):
                        logger.info(os.path.basename(outfile)+' already exists and clobber==False')
                        docal[i] = False
                if docal[i]:
                    donames.append(name)
                    logfiles.append(logfile1)
                    logger.info('Detector file %d : %s' % (i+1,name))
                    logger.info('Command : '+cmd1)
                    logger.info('Logfile : '+logfile1)
                    tasks['cmd'][i] = cmd1
                    tasks['name'][i] = name
                    tasks['outfile'][i] = logfile1
                    tasks['errfile'][i] = errfile1
                    tasks['dir'][i] = os.path.dirname(logfile1)                    
        if np.sum(docal)>0:
            gd, = np.where(tasks['cmd'] != '')
            tasks = tasks[gd]
            logger.info(str(len(tasks))+' Detector files to run')        
            key,jobid = slrm.submit(tasks,label='mkdet',verbose=True,logger=logger,**slurmpars1)
            slrm.queue_wait('mkdet',key,jobid,sleeptime=60,verbose=True,logger=logger) # wait for jobs to complete
            # This should check if the ran okay and puts the status in the database            
            chkmaster1 = check_mastercals(tasks['name'],'Detector',logfiles,key,apred,telescope,verbose=True,logger=logger)
            if chkmaster is None:
                chkmaster = chkmaster1
            else:
                chkmaster = np.hstack((chkmaster,chkmaster1))
        else:
            logger.info('No master Detector calibration files need to be run')
            

    # Make darks in parallel
    #-----------------------
    if 'dark' in caltypes:
        # they take too much memory to run in parallel
        #idl -e "makecal,dark=1,vers='$vers',telescope='$telescope'" >& log/mkdark-$telescope.$host.log
        #darkplot --apred $vers --telescope $telescope
        darkdict = allcaldict['dark']
        logger.info('')
        logger.info('-------------------------------')
        logger.info('Making master Darks in parallel')
        logger.info('===============================')
        if darkdict is None or len(darkdict)==0:
            darkdict = []
            logger.info('No master Dark calibration files to make')
        # Give each job LOTS of memory
        slurmpars1 = slurmpars.copy()
        slurmpars1['nodes'] = np.minimum(slurmpars1['nodes'],2)
        slurmpars1['cpus'] = 4
        slurmpars1['memory'] = 55000  # in MB
        logger.info('Slurm settings: '+str(slurmpars1))
        dt = [('cmd',str,1000),('name',str,1000),('outfile',str,1000),('errfile',str,1000),('dir',str,1000)] 
        tasks = np.zeros(len(darkdict),dtype=np.dtype(dt))
        tasks = Table(tasks)
        docal = np.zeros(len(darkdict),bool)
        donames = []
        logfiles = []
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
                # Check if files exist already
                docal[i] = True
                if clobber is not True:
                    if load.exists('Dark',num=name):
                        logger.info(os.path.basename(outfile)+' already exists and clobber==False')
                        docal[i] = False
                if docal[i]:
                    donames.append(name)
                    logfiles.append(logfile1)                
                    logger.info('Dark file %d : %s' % (i+1,name))
                    logger.info('Command : '+cmd1)
                    logger.info('Logfile : '+logfile1)
                    tasks['cmd'][i] = cmd1
                    tasks['name'][i] = name
                    tasks['outfile'][i] = logfile1
                    tasks['errfile'][i] = errfile1
                    tasks['dir'][i] = os.path.dirname(logfile1)                    
        if np.sum(docal)>0:
            gd, = np.where(tasks['cmd'] != '')
            tasks = tasks[gd]
            logger.info(str(len(tasks))+' Dark files to run')        
            key,jobid = slrm.submit(tasks,label='mkdark',verbose=True,logger=logger,**slurmpars1)
            slrm.queue_wait('mkdark',key,jobid,sleeptime=120,verbose=True,logger=logger) # wait for jobs to complete
            # This should check if the ran okay and puts the status in the database            
            chkmaster1 = check_mastercals(tasks['name'],'Dark',logfiles,key,apred,telescope,verbose=True,logger=logger)
            if chkmaster is None:
                chkmaster = chkmaster1
            else:
                chkmaster = np.hstack((chkmaster,chkmaster1))
        else:
            logger.info('No master Dark calibration files need to be run')
        # Make the dark plots
        if np.sum(docal)>0:
            cal.darkplot(apred=apred,telescope=telescope)


    # I could process the individual flat exposures in parallel first
    # that would dramatically speed things up

    # Make flats in parallel
    #-------------------------
    if 'flat' in caltypes:
        #idl -e "makecal,flat=1,vers='$vers',telescope='$telescope'" >& log/mkflat-$telescope.$host.log
        #flatplot --apred $vers --telescope $telescope
        flatdict = allcaldict['flat']
        logger.info('')
        logger.info('-------------------------------')
        logger.info('Making master Flats in parallel')
        logger.info('===============================')
        logger.info('Slurm settings: '+str(slurmpars))
        if flatdict is None or len(flatdict)==0:
            flatdict = []
            logger.info('No master Flat calibration files to make')
        dt = [('cmd',str,1000),('name',str,1000),('outfile',str,1000),('errfile',str,1000),('dir',str,1000)] 
        tasks = np.zeros(len(flatdict),dtype=np.dtype(dt))
        tasks = Table(tasks)
        docal = np.zeros(len(flatdict),bool)
        donames = []
        logfiles = []
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
                # Check if files exist already
                docal[i] = True
                if clobber is not True:
                    if load.exists('Flat',num=name):
                        logger.info(os.path.basename(outfile)+' already exists and clobber==False')
                        docal[i] = False
                if docal[i]:
                    donames.append(name)
                    logfiles.append(logfile1)                
                    logger.info('Flat file %d : %s' % (i+1,name))
                    logger.info('Command : '+cmd1)
                    logger.info('Logfile : '+logfile1)
                    tasks['cmd'][i] = cmd1
                    tasks['name'][i] = name
                    tasks['outfile'][i] = logfile1
                    tasks['errfile'][i] = errfile1
                    tasks['dir'][i] = os.path.dirname(logfile1)                    
        if np.sum(docal)>0:
            gd, = np.where(tasks['cmd'] != '')
            tasks = tasks[gd]
            logger.info(str(len(tasks))+' Flat files to run')        
            key,jobid = slrm.submit(tasks,label='mkflat',verbose=True,logger=logger,**slurmpars)
            slrm.queue_wait('mkflat',key,jobid,sleeptime=120,verbose=True,logger=logger) # wait for jobs to complete
            # This should check if the ran okay and puts the status in the database            
            chkmaster1 = check_mastercals(tasks['name'],'Flat',logfiles,key,apred,telescope,verbose=True,logger=logger)
            if chkmaster is None:
                chkmaster = chkmaster1
            else:
                chkmaster = np.hstack((chkmaster,chkmaster1))
        else:
            logger.info('No master Flat calibration files need to be run')
        # Make the flat plots
        if np.sum(docal)>0:
            cal.flatplot(apred=apred,telescope=telescope)

    
    # Make BPM in parallel
    #----------------------
    if 'bpm' in caltypes:
        #idl -e "makecal,bpm=1,vers='$vers',telescope='$telescope'" >& log/mkbpm-$telescope.$host.log
        bpmdict = allcaldict['bpm']
        logger.info('')
        logger.info('------------------------------')
        logger.info('Making master BPMs in parallel')
        logger.info('==============================')
        logger.info('Slurm settings: '+str(slurmpars))
        if bpmdict is None or len(bpmdict)==0:
            bpmdict = []
            logger.info('No master BPM calibration files to make')
        dt = [('cmd',str,1000),('name',str,1000),('outfile',str,1000),('errfile',str,1000),('dir',str,1000)] 
        tasks = np.zeros(len(bpmdict),dtype=np.dtype(dt))
        tasks = Table(tasks)
        docal = np.zeros(len(bpmdict),bool)
        donames = []
        logfiles = []
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
                # Check if files exist already
                docal[i] = True
                if clobber is not True:
                    if load.exists('BPM',num=name):
                        logger.info(os.path.basename(outfile)+' already exists and clobber==False')
                        docal[i] = False
                if docal[i]:
                    donames.append(name)
                    logfiles.append(logfile1)                
                    logger.info('BPM file %d : %s' % (i+1,name))
                    logger.info('Command : '+cmd1)
                    logger.info('Logfile : '+logfile1)
                    tasks['cmd'][i] = cmd1
                    tasks['name'][i] = name
                    tasks['outfile'][i] = logfile1
                    tasks['errfile'][i] = errfile1
                    tasks['dir'][i] = os.path.dirname(logfile1)                    
        if np.sum(docal)>0:
            gd, = np.where(tasks['cmd'] != '')
            tasks = tasks[gd]
            logger.info(str(len(tasks))+' BPM files to run')        
            key,jobid = slrm.submit(tasks,label='mkbpm',verbose=True,logger=logger,**slurmpars)
            slrm.queue_wait('mkbpm',key,jobid,sleeptime=120,verbose=True,logger=logger) # wait for jobs to complete
            # This should check if the ran okay and puts the status in the database            
            chkmaster1 = check_mastercals(tasks['name'],'BPM',logfiles,key,apred,telescope,verbose=True,logger=logger)
            if chkmaster is None:
                chkmaster = chkmaster1
            else:
                chkmaster = np.hstack((chkmaster,chkmaster1))
        else:
            logger.info('No master BPM calibration files need to be run')


    # Make Sparse in parallel
    #--------------------------
    if 'sparse' in caltypes:
        sparsedict = allcaldict['sparse']
        logger.info('')
        logger.info('---------------------------------')
        logger.info('Making master Sparses in parallel')
        logger.info('=================================')
        logger.info('Slurm settings: '+str(slurmpars))
        if sparsedict is None or len(sparsedict)==0:
            sparsedict = []
            logger.info('No master Sparse calibration files to make')
        dt = [('cmd',str,1000),('name',str,1000),('outfile',str,1000),('errfile',str,1000),('dir',str,1000)] 
        tasks = np.zeros(len(sparsedict),dtype=np.dtype(dt))
        tasks = Table(tasks)
        docal = np.zeros(len(sparsedict),bool)
        donames = []
        logfiles = []
        for i in range(len(sparsedict)):
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
                # Check if files exist already
                docal[i] = True
                if clobber is not True:
                    if load.exists('Sparse',num=name):
                        logger.info(os.path.basename(outfile)+' already exists and clobber==False')
                        docal[i] = False
                if docal[i]:
                    donames.append(name)
                    logfiles.append(logfile1)                
                    logger.info('Sparse file %d : %s' % (i+1,name))
                    logger.info('Command : '+cmd1)
                    logger.info('Logfile : '+logfile1)
                    tasks['cmd'][i] = cmd1
                    tasks['name'][i] = name
                    tasks['outfile'][i] = logfile1
                    tasks['errfile'][i] = errfile1
                    tasks['dir'][i] = os.path.dirname(logfile1)                    
        if np.sum(docal)>0:
            gd, = np.where(tasks['cmd'] != '')
            tasks = tasks[gd]
            logger.info(str(len(tasks))+' Sparse files to run')        
            key,jobid = slrm.submit(tasks,label='mksparse',verbose=True,logger=logger,**slurmpars)
            slrm.queue_wait('mksparse',key,jobid,sleeptime=120,verbose=True,logger=logger) # wait for jobs to complete
            # This should check if the ran okay and puts the status in the database            
            chkmaster1 = check_mastercals(tasks['name'],'Sparse',logfiles,key,apred,telescope,verbose=True,logger=logger)
            if chkmaster is None:
                chkmaster = chkmaster1
            else:
                chkmaster = np.hstack((chkmaster,chkmaster1))
        else:
            logger.info('No master Sparse calibration files need to be run')
            

    # Make Littrow in parallel
    #--------------------------
    if 'littrow' in caltypes:
        #idl -e "makecal,littrow=1,vers='$vers',telescope='$telescope'" >& log/mklittrow-$telescope.$host.log
        littdict = allcaldict['littrow']
        logger.info('')
        logger.info('----------------------------------')
        logger.info('Making master Littrows in parallel')
        logger.info('==================================')
        logger.info('Slurm settings: '+str(slurmpars))
        if littdict is None or len(littdict)==0:
            littdict = []
            logger.info('No master Littrow calibration files to make')
        dt = [('cmd',str,1000),('name',str,1000),('outfile',str,1000),('errfile',str,1000),('dir',str,1000)] 
        tasks = np.zeros(len(littdict),dtype=np.dtype(dt))
        tasks = Table(tasks)        
        docal = np.zeros(len(littdict),bool)
        donames = []
        logfiles = []
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
                # Check if files exist already
                docal[i] = True
                if clobber is not True:
                    if load.exists('Littrow',num=name):
                        logger.info(os.path.basename(outfile)+' already exists and clobber==False')
                        docal[i] = False
                if docal[i]:
                    donames.append(name)
                    logfiles.append(logfile1)                
                    logger.info('Littrow file %d : %s' % (i+1,name))
                    logger.info('Command : '+cmd1)
                    logger.info('Logfile : '+logfile1)
                    tasks['cmd'][i] = cmd1
                    tasks['name'][i] = name
                    tasks['outfile'][i] = logfile1
                    tasks['errfile'][i] = errfile1
                    tasks['dir'][i] = os.path.dirname(logfile1)                    
        if np.sum(docal)>0:
            gd, = np.where(tasks['cmd'] != '')
            tasks = tasks[gd]
            logger.info(str(len(tasks))+' Littrow files to run')        
            key,jobid = slrm.submit(tasks,label='mklittrow',verbose=True,logger=logger,**slurmpars)
            slrm.queue_wait('mklittrow',key,jobid,sleeptime=120,verbose=True,logger=logger) # wait for jobs to complete
            # This should check if the ran okay and puts the status in the database            
            chkmaster1 = check_mastercals(tasks['name'],'Littrow',logfiles,key,apred,telescope,verbose=True,logger=logger)
            if chkmaster is None:
                chkmaster = chkmaster1
            else:
                chkmaster = np.hstack((chkmaster,chkmaster1))
        else:
            logger.info('No master Littrow calibration files need to be run')

    
    # Make Response in parallel
    #--------------------------
    if 'response' in caltypes:
        responsedict = allcaldict['response']
        logger.info('')
        logger.info('-----------------------------------')
        logger.info('Making master Responses in parallel')
        logger.info('===================================')
        logger.info('Slurm settings: '+str(slurmpars))
        if responsedict is None or len(responsedict)==0:
            responsedict = []
            logger.info('No master Response calibration files to make')
        dt = [('cmd',str,1000),('name',str,1000),('outfile',str,1000),('errfile',str,1000),('dir',str,1000)] 
        tasks = np.zeros(len(responsedict),dtype=np.dtype(dt))
        tasks = Table(tasks)        
        docal = np.zeros(len(responsedict),bool)
        donames = []
        logfiles = []
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
                # Check if files exist already
                docal[i] = True
                if clobber is not True:
                    if load.exists('Response',num=name):
                        logger.info(os.path.basename(outfile)+' already exists and clobber==False')
                        docal[i] = False
                if docal[i]:
                    donames.append(name)
                    logfiles.append(logfile1)
                    logger.info('Response file %d : %s' % (i+1,name))
                    logger.info('Command : '+cmd1)
                    logger.info('Logfile : '+logfile1)
                    tasks['cmd'][i] = cmd1
                    tasks['name'][i] = name
                    tasks['outfile'][i] = logfile1
                    tasks['errfile'][i] = errfile1
                    tasks['dir'][i] = os.path.dirname(logfile1)                    
        if np.sum(docal)>0:
            gd, = np.where(tasks['cmd'] != '')
            tasks = tasks[gd]
            logger.info(str(len(tasks))+' Response files to run')        
            key,jobid = slrm.submit(tasks,label='mkresponse',verbose=True,logger=logger,**slurmpars)
            slrm.queue_wait('mkresponse',key,jobid,sleeptime=120,verbose=True,logger=logger) # wait for jobs to complete
            # This should check if the ran okay and puts the status in the database            
            chkmaster1 = check_mastercals(tasks['name'],'Response',logfiles,key,apred,telescope,verbose=True,logger=logger)
            if chkmaster is None:
                chkmaster = chkmaster1
            else:
                chkmaster = np.hstack((chkmaster,chkmaster1))
        else:
            logger.info('No master Response calibration files need to be run')

            
    # Make Model PSFs in parallel
    #--------------------------
    if 'modelpsf' in caltypes:
        modelpsfdict = allcaldict['modelpsf']
        logger.info('')
        logger.info('------------------------------------')
        logger.info('Making master PSF Models in parallel')
        logger.info('====================================')
        logger.info('Slurm settings: '+str(slurmpars))
        if modelpsfdict is None or len(modelpsfdict)==0:
            modelpsfdict = []
            logger.info('No master PSF Model calibration files to make')
        dt = [('cmd',str,1000),('name',str,1000),('outfile',str,1000),('errfile',str,1000),('dir',str,1000)] 
        tasks = np.zeros(len(modelpsfdict),dtype=np.dtype(dt))
        tasks = Table(tasks)
        docal = np.zeros(len(modelpsfdict),bool)    
        donames = []
        logfiles = []
        for i in range(len(modelpsfdict)):
            name = modelpsfdict['name'][i]
            if np.sum((mjds >= modelpsfdict['mjd1'][i]) & (mjds <= modelpsfdict['mjd2'][i])) > 0:
                outfile = load.filename('PSFModel',num=name,chips=True)
                logfile1 = os.path.dirname(outfile)+'/mkpsfmodel-'+str(name)+'-'+telescope+'_pbs.'+logtime+'.log'
                errfile1 = logfile1.replace('.log','.err')
                if os.path.exists(os.path.dirname(logfile1))==False:
                    os.makedirs(os.path.dirname(logfile1))
                cmd1 = 'makecal --vers {0} --telescope {1}'.format(apred,telescope)
                cmd1 += ' --modelpsf '+str(name)+' --unlock'
                if clobber:
                    cmd1 += ' --clobber'
                # Check if files exist already
                docal[i] = True
                if clobber is not True:
                    if load.exists('PSFModel',num=name):
                        logger.info(os.path.basename(outfile)+' already exists and clobber==False')
                        docal[i] = False
                if docal[i]:
                    donames.append(name)
                    logfiles.append(logfile1)                
                    logger.info('PSFModel file %d : %s' % (i+1,name))
                    logger.info('Command : '+cmd1)
                    logger.info('Logfile : '+logfile1)
                    tasks['cmd'][i] = cmd1
                    tasks['name'][i] = name
                    tasks['outfile'][i] = logfile1
                    tasks['errfile'][i] = errfile1
                    tasks['dir'][i] = os.path.dirname(logfile1)
        if np.sum(docal)>0:
            gd, = np.where(tasks['cmd'] != '')
            tasks = tasks[gd]
            logger.info(str(len(tasks))+' PSFModel files to run')        
            key,jobid = slrm.submit(tasks,label='mkpsfmodel',verbose=True,logger=logger,**slurmpars)
            slrm.queue_wait('mkpsfmodel',key,jobid,sleeptime=120,verbose=True,logger=logger) # wait for jobs to complete
            # This should check if the ran okay and puts the status in the database            
            chkmaster1 = check_mastercals(tasks['name'],'PSFModel',logfiles,key,apred,telescope,verbose=True,logger=logger)
            if chkmaster is None:
                chkmaster = chkmaster1
            else:
                chkmaster = np.hstack((chkmaster,chkmaster1))
        else:
            logger.info('No master PSF Model calibration files need to be run')
    

    # Make multiwave cals in parallel
    #--------------------------------
    # we need the multiwave for the LSFs
    #set n = 0
    #while ( $n < 5 ) 
    #   idl -e "makecal,multiwave=1,vers='$vers',telescope='$telescope'" >& log/mkwave-$telescope"$n".$host.log &
    #   sleep 20
    #   @ n = $n + 1
    #end
    #wait

    if 'multiwave' in caltypes:
        multiwavedict = allcaldict['multiwave']
        logger.info('')
        logger.info('-----------------------------------')
        logger.info('Making master multiwave in parallel')
        logger.info('===================================')
        logger.info('Slurm settings: '+str(slurmpars))
        if multiwavedict is None or len(multiwavedict)==0:
            multiwavedict = []
            logger.info('No master multiwave calibration files to make')

        # Which multiwave and individual wave still need to be made
        multiwave_names = []
        wave_names = []
        for i in range(len(multiwavedict)):
            name = multiwavedict['name'][i]
            if np.sum((mjds >= multiwavedict['mjd1'][i]) & (mjds <= multiwavedict['mjd2'][i])) > 0:
                if load.exists('Wave',num=multiwavedict['name'][i])==False:
                    multiwave_names.append(name)
                    wnames = multiwavedict['frames'][i].split(',')
                    wave_names += wnames
                else:
                    logger.info(load.filename('Wave',num=multiwavedict['name'][i],chips=True)+' exists already')
        logger.info(str(len(multiwave_names))+' multiwave files need to be made')
        wave_names = list(np.unique(wave_names))
        wave_names = [n for n in wave_names if load.exists('Wave',num=n)==False]
        logger.info(str(len(wave_names))+' apWave files need to be made')
                
        # Which PSFs are we going to use for the apWave files
        psf_names = []
        for i in range(len(wave_names)):
            name = wave_names[i]
            mjd = int(load.cmjd(int(name)))            
            # Use a quartzflat for the PSF, the PSF cal file will automatically be created
            expinfo1 = info.expinfo(observatory=load.observatory,mjd5=mjd)
            if len(expinfo1)==0:
                print('no quartz for this '+str(mjd))
                continue
            qtzind, = np.where(expinfo1['exptype']=='QUARTZFLAT')
            psfid = None
            if len(qtzind)>0:
                qtzinfo = expinfo1[qtzind]
                bestind = np.argsort(np.abs(qtzinfo['num']-int(name)))[0]
                psfid = qtzinfo['num'][bestind]
                psf_names.append(psfid)
        psf_names = list(np.unique(psf_names))
        psf_names = [n for n in psf_names if load.exists('PSF',num=n)==False]
        logger.info(str(len(psf_names))+' apPSF files need to be made')

        # Create the apPSF files so we can extract the individual wave files
        dt = [('cmd',str,1000),('name',str,1000),('outfile',str,1000),('errfile',str,1000),
              ('dir',str,1000),('num',int),('mjd',int),('observatory',str,10),
              ('configid',str,50),('designid',str,50),('fieldid',str,50)]
        # num, mjd, observatory, configid, designid, fieldid        
        tasks = np.zeros(len(psf_names),dtype=np.dtype(dt))
        tasks = Table(tasks)
        docal = np.zeros(len(psf_names),bool)
        donames = []
        logfiles = []
        for i in range(len(psf_names)):
            name = psf_names[i]
            mjd = int(load.cmjd(int(name)))
            outfile = load.filename('PSF',num=name,chips=True)
            logfile1 = os.path.dirname(outfile)+'/'+load.prefix+'PSF-'+str(name)+'_pbs.'+logtime+'.log'
            errfile1 = logfile1.replace('.log','.err')
            if os.path.exists(os.path.dirname(logfile1))==False:
                os.makedirs(os.path.dirname(logfile1))
            cmd1 = 'makecal --vers {0} --telescope {1}'.format(apred,telescope)
            cmd1 += ' --psf '+str(name)+' --unlock'
            if clobber:
                cmd1 += ' --clobber'
            # Check if files exist already
            docal[i] = True
            if clobber is not True:
                if load.exists('PSF',num=name):
                    logger.info(os.path.basename(outfile)+' already exists and clobber==False')
                    docal[i] = False
            if docal[i]:
                donames.append(name)
                logfiles.append(logfile1)                
                logger.info('PSF file %d : %s' % (i+1,name))
                logger.info('Command : '+cmd1)
                logger.info('Logfile : '+logfile1)
                tasks['cmd'][i] = cmd1
                tasks['name'][i] = name
                tasks['outfile'][i] = logfile1
                tasks['errfile'][i] = errfile1
                tasks['dir'][i] = os.path.dirname(logfile1)
                tasks['num'][i] = int(name)
                tasks['mjd'][i] = load.cmjd(int(name))
                tasks['observatory'][i] = load.observatory
                tasks['configid'][i] = ''
                tasks['designid'][i] = ''
                tasks['fieldid'][i] =  ''               
        if np.sum(docal)>0:
            gd, = np.where(tasks['cmd'] != '')
            tasks = tasks[gd]
            logger.info(str(len(tasks))+' PSF files to run')
            key,jobid = slrm.submit(tasks,label='makecal-psf',verbose=True,logger=logger,**slurmpars)
            slrm.queue_wait('makecal-psf',key,jobid,sleeptime=120,verbose=True,logger=logger) # wait for jobs to complete
            # This should check if it ran okay and puts the status in the database
            #  'tasks' doubles as 'expinfo' for check_calib()
            chkcal = check_calib(tasks,logfiles,key,apred,verbose=True,logger=logger)
            # Summary
            indcal, = np.where(chkcal['success']==True)
            logger.info('%d/%d apPSF successfully processed' % (len(indcal),len(chkcal)))
        else:
            logger.info('No individual PSF calibration files need to be run')
            
        # Creating individual wave files to the multiwave calibration files
        dt = [('cmd',str,1000),('name',str,1000),('outfile',str,1000),('errfile',str,1000),
              ('dir',str,1000),('num',int),('mjd',int),('observatory',str,10),
              ('configid',str,50),('designid',str,50),('fieldid',str,50)]
        # num, mjd, observatory, configid, designid, fieldid        
        tasks = np.zeros(len(wave_names),dtype=np.dtype(dt))
        tasks = Table(tasks)
        docal = np.zeros(len(wave_names),bool)
        donames = []
        logfiles = []
        for i in range(len(wave_names)):
            name = wave_names[i]
            mjd = int(load.cmjd(int(name)))
            outfile = load.filename('Wave',num=name,chips=True)
            logfile1 = os.path.dirname(outfile)+'/'+load.prefix+'Wave-'+str(name)+'_pbs.'+logtime+'.log'
            errfile1 = logfile1.replace('.log','.err')
            if os.path.exists(os.path.dirname(logfile1))==False:
                os.makedirs(os.path.dirname(logfile1))
            cmd1 = 'makecal --vers {0} --telescope {1}'.format(apred,telescope)            
            # Use a quartzflat for the PSF, the PSF cal file will automatically be created
            expinfo1 = info.expinfo(observatory=load.observatory,mjd5=mjd)
            qtzind, = np.where(expinfo1['exptype']=='QUARTZFLAT')
            psfid = None
            if len(qtzind)>0:
                qtzinfo = expinfo1[qtzind]
                bestind = np.argsort(np.abs(qtzinfo['num']-int(name)))[0]
                psfid = qtzinfo['num'][bestind]
                cmd1 += ' --psf '+str(psfid)
            # Use modelpsf, if possible
            caldata1 = mkcal.getcal(calfile,mjd,verbose=False)
            if caldata1.get('modelpsf') is not None:
                cmd1 += ' --modelpsf '+str(caldata1['modelpsf'])
            cmd1 += ' --wave '+str(name)+' --unlock'
            if clobber:
                cmd1 += ' --clobber'
            # Check if files exist already
            docal[i] = True
            if clobber is not True:
                if load.exists('Wave',num=name):
                    logger.info(os.path.basename(outfile)+' already exists and clobber==False')
                    docal[i] = False
            if docal[i]:
                donames.append(name)
                logfiles.append(logfile1)                
                logger.info('wave file %d : %s' % (i+1,name))
                logger.info('Command : '+cmd1)
                logger.info('Logfile : '+logfile1)
                tasks['cmd'][i] = cmd1
                tasks['name'][i] = name
                tasks['outfile'][i] = logfile1
                tasks['errfile'][i] = errfile1
                tasks['dir'][i] = os.path.dirname(logfile1)
                tasks['num'][i] = int(name)
                tasks['mjd'][i] = load.cmjd(int(name))
                tasks['observatory'][i] = load.observatory
                tasks['configid'][i] = ''
                tasks['designid'][i] = ''
                tasks['fieldid'][i] =  ''               
        if np.sum(docal)>0:
            gd, = np.where(tasks['cmd'] != '')
            tasks = tasks[gd]
            logger.info(str(len(tasks))+' wave files to run')
            key,jobid = slrm.submit(tasks,label='makecal-wave',verbose=True,logger=logger,**slurmpars)
            slrm.queue_wait('makecal-wave',key,jobid,sleeptime=120,verbose=True,logger=logger) # wait for jobs to complete
            # This should check if it ran okay and puts the status in the database
            #  'tasks' doubles as 'expinfo' for check_calib()
            chkcal = check_calib(tasks,logfiles,key,apred,verbose=True,logger=logger)
            # Summary
            indcal, = np.where(chkcal['success']==True)
            logger.info('%d/%d apWave successfully processed' % (len(indcal),len(chkcal)))
        else:
            logger.info('No individual wave calibration files need to be run')        

        # Creating the multiwave calibration files
        dt = [('cmd',str,1000),('name',str,1000),('outfile',str,1000),('errfile',str,1000),('dir',str,1000)]
        tasks = np.zeros(len(multiwave_names),dtype=np.dtype(dt))
        tasks = Table(tasks)
        docal = np.zeros(len(multiwave_names),bool)    
        donames = []
        logfiles = []
        for i in range(len(multiwave_names)):
            name = multiwave_names[i]
            outfile = load.filename('Wave',num=name,chips=True)
            logfile1 = os.path.dirname(outfile)+'/mkmultiwave-'+str(name)+'-'+telescope+'_pbs.'+logtime+'.log'
            errfile1 = logfile1.replace('.log','.err')
            if os.path.exists(os.path.dirname(logfile1))==False:
                os.makedirs(os.path.dirname(logfile1))
            cmd1 = 'makecal --vers {0} --telescope {1}'.format(apred,telescope)
            cmd1 += ' --multiwave '+str(name)+' --unlock'
            if clobber:
                cmd1 += ' --clobber'
            # Check if files exist already
            docal[i] = True
            if clobber is not True:
                if load.exists('Wave',num=name):
                    logger.info(os.path.basename(outfile)+' already exists and clobber==False')
                    docal[i] = False
            if docal[i]:
                donames.append(name)
                logfiles.append(logfile1)                
                logger.info('multiwave file %d : %s' % (i+1,name))
                logger.info('Command : '+cmd1)
                logger.info('Logfile : '+logfile1)
                tasks['cmd'][i] = cmd1
                tasks['name'][i] = name
                tasks['outfile'][i] = logfile1
                tasks['errfile'][i] = errfile1
                tasks['dir'][i] = os.path.dirname(logfile1)
        if np.sum(docal)>0:
            gd, = np.where(tasks['cmd'] != '')
            tasks = tasks[gd]
            logger.info(str(len(tasks))+' multiwave files to run')        
            key,jobid = slrm.submit(tasks,label='mkmultiwave',verbose=True,logger=logger,**slurmpars)
            slrm.queue_wait('mkmultiwave',key,jobid,sleeptime=120,verbose=True,logger=logger) # wait for jobs to complete
            # This should check if the ran okay and puts the status in the database            
            chkmaster1 = check_mastercals(tasks['name'],'Wave',logfiles,key,apred,telescope,verbose=True,logger=logger)
            if chkmaster is None:
                chkmaster = chkmaster1
            else:
                chkmaster = np.hstack((chkmaster,chkmaster1))
        else:
            logger.info('No master multiwave calibration files need to be run')
    

    # Make LSFs in parallel
    #-----------------------
    if 'lsf' in caltypes:
        lsfdict = allcaldict['lsf']
        logger.info('')
        logger.info('--------------------------------')
        logger.info('Making master LSFs in parallel')
        logger.info('================================')
        logger.info('Slurm settings: '+str(slurmpars))
        if lsfdict is None or len(lsfdict)==0:
            lsfdict = []
            logger.info('No master LSF calibration files to make')
        dt = [('cmd',str,1000),('name',str,1000),('outfile',str,1000),('errfile',str,1000),('dir',str,1000)] 
        tasks = np.zeros(len(lsfdict),dtype=np.dtype(dt))
        tasks = Table(tasks)
        docal = np.zeros(len(lsfdict),bool)
        donames = []
        logfiles = []
        for i in range(len(lsfdict)):
            name = lsfdict['name'][i]
            if np.sum((mjds >= lsfdict['mjd1'][i]) & (mjds <= lsfdict['mjd2'][i])) > 0:
                outfile = load.filename('LSF',num=name,chips=True)
                logfile1 = os.path.dirname(outfile)+'/mklsf-'+str(name)+'-'+telescope+'_pbs.'+logtime+'.log'
                errfile1 = logfile1.replace('.log','.err')
                if os.path.exists(os.path.dirname(logfile1))==False:
                    os.makedirs(os.path.dirname(logfile1))
                cmd1 = 'makecal --vers {0} --telescope {1} --full'.format(apred,telescope)
                cmd1 += ' --lsf '+str(name)+' --unlock'
                if clobber:
                    cmd1 += ' --clobber'
                # Check if files exist already
                docal[i] = True
                if clobber is not True:
                    if load.exists('LSF',num=name):
                        logger.info(os.path.basename(outfile)+' already exists and clobber==False')
                        docal[i] = False
                if docal[i]:
                    donames.append(name)
                    logfiles.append(logfile1)
                    logger.info('LSF file %d : %s' % (i+1,name))
                    logger.info('Command : '+cmd1)
                    logger.info('Logfile : '+logfile1)
                    tasks['cmd'][i] = cmd1
                    tasks['name'][i] = name
                    tasks['outfile'][i] = logfile1
                    tasks['errfile'][i] = errfile1
                    tasks['dir'][i] = os.path.dirname(logfile1)
        if np.sum(docal)>0:
            gd, = np.where(tasks['cmd'] != '')
            tasks = tasks[gd]
            logger.info(str(len(tasks))+' LSF files to run')        
            key,jobid = slrm.submit(tasks,label='mklsf',verbose=True,logger=logger,**slurmpars)
            slrm.queue_wait('mklsf',key,jobid,sleeptime=60,verbose=True,logger=logger) # wait for jobs to complete
            # This should check if the ran okay and puts the status in the database            
            chkmaster1 = check_mastercals(tasks['name'],'LSF',logfiles,key,apred,telescope,verbose=True,logger=logger)
            if chkmaster is None:
                chkmaster = chkmaster1
            else:
                chkmaster = np.hstack((chkmaster,chkmaster1))        
        else:
            logger.info('No master LSF calibration files need to be run')
        
    return chkmaster
    
    
def runap3d(load,mjds,slurmpars,clobber=False,logger=None):
    """
    Run AP3D on all exposures for a list of MJDs.

    Parameters
    ----------
    load : ApLoad object
       ApLoad object that contains "apred" and "telescope".
    mjds : list
       List of MJDs to process
    slurmpars : dictionary
       Dictionary of slurmpars settings.
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

    # Loop over exposures and see if the outputs exist already
    do3d = np.zeros(len(expinfo),bool)
    for i,num in enumerate(expinfo['num']):    
        mjd = int(load.cmjd(num))
        # Check if files exist already
        do3d[i] = True
        if clobber is not True:
            outfile = load.filename('2D',num=num,mjd=mjd,chips=True)
            if load.exists('2D',num=num):
                logger.info(str(i+1)+' '+os.path.basename(outfile)+' already exists and clobber==False')
                do3d[i] = False
    logger.info(str(np.sum(do3d))+' exposures to run')

    # Loop over the exposures and make the commands for the ones that we will run
    torun, = np.where(do3d==True)
    ntorun = len(torun)
    if ntorun>0:
        slurmpars1 = slurmpars.copy()
        if ntorun<slurmpars1['ppn']:
            slurmpars1['cpus'] = ntorun
        slurmpars1['numpy_num_threads'] = 2
        logger.info('Slurm settings: '+str(slurmpars1))
        tasks = np.zeros(ntorun,dtype=np.dtype([('cmd',str,1000),('outfile',str,1000),('errfile',str,1000),('dir',str,1000)]))
        tasks = Table(tasks)
        #queue = pbsqueue(verbose=True)
        #queue.create(label='ap3d', **slurmpars1)
        for i in range(ntorun):
            num = expinfo['num'][torun[i]]
            mjd = int(load.cmjd(num))
            logfile1 = load.filename('2D',num=num,mjd=mjd,chips=True).replace('2D','3D')
            logfile1 = os.path.dirname(logfile1)+'/logs/'+os.path.basename(logfile1)
            logfile1 = logfile1.replace('.fits','_pbs.'+logtime+'.log')
            if os.path.exists(os.path.dirname(logfile1))==False:
                os.makedirs(os.path.dirname(logfile1))
            cmd1 = 'ap3d --num {0} --vers {1} --telescope {2} --unlock'.format(num,apred,telescope)
            if clobber:
                cmd1 += ' --clobber'
            logger.info('Exposure %d : %d' % (i+1,num))
            logger.info('Command : '+cmd1)
            logger.info('Logfile : '+logfile1)
            tasks['cmd'][i] = cmd1
            tasks['outfile'][i] = logfile1
            tasks['errfile'][i] = logfile1.replace('.log','.err')
            tasks['dir'][i] = os.path.dirname(logfile1)
            #queue.append(cmd1,outfile=logfile1,errfile=logfile1.replace('.log','.err'))
        logger.info('Running AP3D on '+str(ntorun)+' exposures')
        key,jobid = slrm.submit(tasks,label='ap3d',verbose=True,logger=logger,**slurmpars1)
        slrm.queue_wait('ap3d',key,jobid,sleeptime=60,verbose=True,logger=logger) # wait for jobs to complete  
        #queue.commit(hard=True,submit=True)
        #logger.info('PBS key is '+queue.key)
        #queue_wait(queue,sleeptime=60,verbose=True,logger=logger)  # wait for jobs to complete
        # This should check if the ap3d ran okay and puts the status in the database
        chk3d = check_ap3d(expinfo,key,apred,telescope,verbose=True,logger=logger)
    else:
        chk3d = None
        logger.info('No exposures need AP3D processing')
        
    return chk3d


def rundailycals(load,mjds,slurmpars,caltypes=None,clobber=False,logger=None):
    """
    Run daily calibration frames for a list of MJDs.

    Parameters
    ----------
    load : ApLoad object
       ApLoad object that contains "apred" and "telescope".
    mjds : list
       List of MJDs to process
    slurmpars : dictionary
       Dictionary of slurmpars settings.
    caltypes : list, optional
       List of calibration types to run.  The default is all of them.
       ['psf','flux','arcs','dailywave','fpi','telluric']
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

    if caltypes is None:
        caltypes = ['psf','flux','arcs','dailywave','fpi','telluric']
    else:
        caltypes = [c.lower() for c in caltypes]
        if 'wave' in caltypes:
            caltypes.append('arcs')
        
    apred = load.apred
    telescope = load.telescope
    observatory = telescope[0:3]
    logtime = datetime.now().strftime("%Y%m%d%H%M%S")
    chips = ['a','b','c']
    caldir = os.environ['APOGEE_DRP_DIR']+'/data/cal/'
    calfile = caldir+load.instrument+'.par' 
    reduxdir = os.environ['APOGEE_REDUX']+'/'+load.apred+'/'
    
    # Get exposures
    logger.info('Getting exposure information')
    allexpinfo = getexpinfo(load,mjds,logger=logger,verbose=False)
    # Calculate dither groups
    allexpinfo = info.getdithergroups(allexpinfo)
    expinfo = allexpinfo.copy()
    
    # First we need to run domeflats and quartzflats so there are apPSF files
    # Then the arclamps
    # apFlux files?
    # Then the FPI exposures last (needs apPSF and apWave files)
    # only select the cal types that we requested
    exptypes = []
    for c in caltypes:
        if c=='psf' or c=='flux':
            exptype = ['DOMEFLAT','QUARTZFLAT']
        elif c=='arcs' or c=='dailywave' or c=='telluric':
            exptype = ['ARCLAMP']
        elif c=='fpi':
            exptype = ['FPI','ARCLAMP']
        elif c=='telluric':
            exptype = ''
        exptypes += exptype
    exptypes = np.unique(exptypes)
    calind = np.array([],int)
    for e in exptypes:
        calind1, = np.where(expinfo['exptype']==e)
        if len(calind1)>0:
            calind = np.append(calind,calind1)    
    if len(calind)>0:
        expinfo = expinfo[calind]
    else:
        if 'dailywave' not in caltypes and 'telluric' not in caltypes:
            logger.info('No calibration files to run')
            return None
        else:
            # dailywave and telluric do not require any calibration exposures            
            expinfo = []

    # Run QA check on the files
    logger.info(' ')
    logger.info('Doing quality checks on all calibration exposures')
    if len(expinfo)>0:
        qachk = check.check(expinfo['num'],apred,telescope,verbose=True,logger=logger)
        logger.info(' ')
        okay, = np.where(qachk['okay']==True)
        if len(okay)>0:
            expinfo = expinfo[okay]
        else:
            if 'dailywave' not in caltypes and 'telluric' not in caltypes:
                logger.info('No calibration files to run')
                return None
            else:
                # dailywave and telluric do not require any calibration exposures            
                expinfo = []
                
    # Create cal plan directories for each night
    for m in mjds:
        calplandir = os.path.dirname(load.filename('CalPlan',num=0,mjd=m))
        if os.path.exists(calplandir)==False:
            os.makedirs(calplandir)

    # Loop over calibration types
    calnames = ['psf','flux','arcs','dailywave','fpi','telluric']
    filecodes = ['PSF','Flux','Wave','Wave','WaveFPI','Telluric']
    chkcal = None
    for i,ctype in enumerate(calnames):
        # Only run for caltypes that we asked for
        if ctype in caltypes:
            logger.info('')
            logger.info('----------------------------------------------')
            logger.info('Running Calibration Files: '+ctype.upper())
            logger.info('==============================================')
            logger.info('')

            # No calibration exposures for psf/flux/arcs/fpi
            if len(expinfo)==0 and ctype in ['psf','flux','arcs','fpi']:
                logger.info('Cannot make '+ctype.upper()+'. No calibration exposures')
                continue
        
            
            # Step 1) Get data for this calibration type
            calinfo = []
            
            # --- PSF ---
            if ctype=='psf':
                # Domeflats and quartzflats for plates
                # Quartzflats only for FPS
                ind = np.array([],int)
                logger.info('Daily PSF Calibration Products')
                logger.info('------------------------------')                
                for m in mjds:
                    # plates, can use domeflats or quartzflats
                    if m < 59556:
                        ind1, = np.where((expinfo['mjd']==m) & ((expinfo['exptype']=='DOMEFLAT') |
                                                               (expinfo['exptype']=='QUARTZFLAT')))
                    # FPS, use quartzflats because 2 FPI fibers missing in domeflats
                    else:            
                        ind1, = np.where((expinfo['mjd']==m) & (expinfo['exptype']=='QUARTZFLAT'))
                    if len(ind1)==0:
                        logger.info(str(m)+' No PSF calibration products')
                        continue
                    ind = np.append(ind,ind1)
                    logger.info(str(m)+' '+','.join(expinfo['num'][ind1].astype(str)))                        
                ncal = len(ind)
                if len(ind)>0:
                    calinfo = expinfo[ind]
            # --- FLUX ---
            elif ctype=='flux':
                # Make sure there is at least ONE flux calibration file per MJD
                #  if no dome flat was taken, then use quartz flat
                ind = np.array([],int)
                logger.info('Daily Flux Calibration Products')
                logger.info('-------------------------------')                
                for m in mjds:
                    # ALWAYS need to use domeflats so we get the fiber-to-fiber throughput
                    #    corrections right
                    ind1, = np.where((expinfo['mjd']==m) & (expinfo['exptype']=='DOMEFLAT'))
                    if len(ind1)==0:
                        logger.info(str(m)+' No Flux calibration products')
                        continue
                    ind = np.append(ind,ind1)
                    logger.info(str(m)+' '+','.join(expinfo['num'][ind1].astype(str)))
                ncal = len(ind)
                if len(ind)>=0:
                    calinfo = expinfo[ind]
            # ---  ARCS ---
            elif ctype=='arcs':
                ind = np.array([],int)                
                logger.info('Daily Arclamp Calibration Products')
                logger.info('----------------------------------')
                for m in mjds:
                    ind1, = np.where((expinfo['mjd']==m) & (expinfo['exptype']=='ARCLAMP'))
                    if len(ind1)==0:
                        logger.info(str(m)+' No arclamp exposures')
                        continue
                    ind = np.append(ind,ind1)
                    logger.info(str(m)+' '+str(len(ind1))+' arclamp exposures')
                ncal = len(ind)
                if len(ind)>0:
                    calinfo = expinfo[ind]
            # ---  DAILY WAVE ---
            elif ctype=='dailywave':
                ncal = len(mjds)
                calinfo = np.zeros(ncal,dtype=np.dtype([('num',int),('mjd',int),('exptype',(str,20)),('observatory',(str,3)),
                                                        ('configid',int),('designid',int),('fieldid',int)]))
                calinfo['num'] = mjds
                calinfo['mjd'] = mjds
                calinfo['exptype'] = 'dailywave'
                calinfo['observatory'] = load.observatory
                # Make sure there are exposures for each night
                #  some APO summer shutdown nights have no data and we don't need to make dailywave
                logger.info('DailyWave calibration products')
                logger.info('------------------------------')                
                for j,m in enumerate(mjds):                
                    mind, = np.where(allexpinfo['mjd']==m)                    
                    if len(mind)==0:
                        logger.info(str(m)+' No exposures for MJD')
                        calinfo['num'][j] = 0
                        continue
                    arcind, = np.where((expinfo['mjd']==m) & (expinfo['exptype']=='ARCLAMP'))
                    if len(arcind)==0:
                        logger.info(str(m)+' No ARCLAMP exposures')
                        calinfo['num'][j] = 0
                        continue                    
                    logger.info(str(m)+' '+str(len(arcind))+' arclamp exposures')
                # Keeping only nights with data
                gdmjd, = np.where(calinfo['num'] != 0)
                if len(gdmjd)>0:
                    calinfo = calinfo[gdmjd]
                    ncal = len(calinfo)
                else:
                    calinfo = []
                    ncal = 0
            # --- FPI ---
            elif ctype=='fpi':
                # Only FPI exposure number per MJD
                calfpiind = []
                logger.info('Daily FPI calibration products')
                logger.info('------------------------------')                
                for m in mjds:
                    fpiind, = np.where((expinfo['mjd']==m) & (expinfo['exptype']=='FPI'))
                    if len(fpiind)==0:
                        logger.info(str(m)+' No FPI exposures')
                        continue
                    # Make sure there is DailyWave calibration product
                    wfile = reduxdir+'cal/'+load.instrument+'/wave/'+load.prefix+'Wave-b-{:5d}.fits'.format(int(m))
                    if os.path.exists(wfile)==False:
                        logger.info(str(m)+' DailyWave does not exist')
                        continue
                    # Make sure there is an associated arclamp dither group for the FPI
                    ftable = Table.read(wfile,7)
                    ind3,ind4 = dln.match(ftable['frame'],allexpinfo['num'])
                    ftable['dithergroup'] = -1
                    if len(ind3)>0:
                        ftable['dithergroup'][ind3] = allexpinfo['dithergroup'][ind4]
                    # Loop over the FPIs and make sure there is an associated arclamp dither group
                    goodfpiind = []
                    for f in fpiind:
                        gdpair, = np.where(ftable['dithergroup'] == expinfo['dithergroup'][f])
                        if len(gdpair)>0:
                            goodfpiind.append(f)
                    # No good FPI exposure for this mjd
                    if len(goodfpiind)==0:
                        logger.info(str(m)+' No FPI exposures with associated arc dither pairs')
                        continue
                    logger.info(str(m)+' '+str(expinfo['num'][goodfpiind[0]]))
                    # Pick the first good FPI exposure
                    calfpiind.append(goodfpiind[0])
                # Make calinfo tale for all nights
                if len(calfpiind)>0:
                    calinfo = expinfo[calfpiind]
                    ncal = len(calinfo)
                else:
                    calinfo = []
                    ncal = 0
            # --- TELLURIC ---
            elif ctype=='telluric':
                ncal = len(mjds)
                calinfo = np.zeros(ncal,dtype=np.dtype([('num',(str,100)),('mjd',int),('exptype',(str,20)),('observatory',(str,3)),
                                                        ('configid',int),('designid',int),('fieldid',int)]))
                calinfo['mjd'] = mjds
                calinfo['exptype'] = 'telluric'
                calinfo['observatory'] = load.observatory
                logger.info('Daily Telluric calibration products')
                logger.info('-----------------------------------')
                for j,m in enumerate(mjds):
                    mind, = np.where(allexpinfo['mjd']==m)
                    if len(mind)==0:
                        logger.info(str(m)+' No exposures')
                        calinfo['num'][j] = ''
                        continue
                    caldata = mkcal.getcal(calfile,m,verbose=False)
                    lsfid = caldata['lsf']
                    if lsfid is None:
                        logger.info(str(m)+' No LSF calibration file')
                        continue
                    # Make sure there is DailyWave calibration product
                    wfile = reduxdir+'cal/'+load.instrument+'/wave/'+load.prefix+'Wave-b-{:5d}.fits'.format(int(m))
                    if os.path.exists(wfile)==False:
                        logger.info(str(m)+' DailyWave does not exist')
                        continue                    
                    # Need some object exposures for this night
                    objind, = np.where(allexpinfo['mjd']==m)
                    if len(objind)==0:
                        logger.info(str(m)+' No object exposures')
                        continue
                    calinfo['num'][j] = str(m)+'-'+str(lsfid)
                    logger.info(str(m)+' '+str(m)+'-'+str(lsfid))
                # Remove nights with issues
                goodtell, = np.where(calinfo['num'] != '')
                if len(goodtell)>0:
                    calinfo = calinfo[goodtell]
                    ncal = len(calinfo)
                else:
                    calinfo = []
                    ncal = 0
                    
            logger.info(str(ncal)+' file(s)')
            
            # Step 2) Loop over calibration files and check if we need to run them
            docal = np.zeros(ncal,bool)
            for j in range(ncal):
                num1 = calinfo['num'][j]
                mjd1 = calinfo['mjd'][j]
                # Check if files exist already
                docal[j] = True
                if clobber is not True:
                    if ctype=='dailywave':
                        outfile = load.filename(filecodes[i],num=num1,mjd=mjd1,chips=True)
                        outfile = outfile[0:-13]+str(mjd1)+'.fits'
                        allfiles = [outfile.replace('Wave-','Wave-'+ch+'-') for ch in chips]
                        allexist = [os.path.exists(f) for f in allfiles]
                        exists = np.sum(allexist)==3
                    else:
                        outfile = load.filename(filecodes[i],num=num1,mjd=mjd1,chips=True)
                        exists = load.exists(filecodes[i],num=num1,mjd=mjd1)
                    # WaveFPI files are not getting checked properly!!!!
                    if exists:
                        logger.info(str(j+1)+'  '+os.path.basename(outfile)+' already exists and clobber==False')
                        docal[j] = False
            logger.info(str(np.sum(docal))+' '+ctype.upper()+' to run')
            
            # Step 3) Loop over the calibrations, make the commands and submit to SLURM
            logfiles = []
            torun, = np.where(docal==True)
            ntorun = len(torun)
            if ntorun>0:
                slurmpars1 = slurmpars.copy()
                if ntorun<slurmpars1['ppn']:
                    slurmpars1['cpus'] = ntorun
                slurmpars1['numpy_num_threads'] = 2
                logger.info('Slurm settings: '+str(slurmpars1))
                tasks = np.zeros(ntorun,dtype=np.dtype([('cmd',str,1000),('outfile',str,1000),('errfile',str,1000),('dir',str,1000)]))
                tasks = Table(tasks)
                for j in range(ntorun):
                    num1 = calinfo['num'][torun[j]]
                    mjd1 = calinfo['mjd'][torun[j]]
                    if mjd1>=59556:
                        fps = True
                    else:
                        fps = False
                    cmd1 = 'makecal --vers {0} --telescope {1} --unlock'.format(apred,telescope)
                    if clobber: cmd1 += ' --clobber'
                    calplandir = os.path.dirname(load.filename('CalPlan',num=0,mjd=mjd1))                
                    if ctype=='psf':    # psfs
                        cmd1 += ' --psf '+str(num1)
                        logfile1 = calplandir+'/'+load.prefix+'PSF-'+str(num1)+'_pbs.'+logtime+'.log'
                    elif ctype=='flux':   # flux
                        cmd1 += ' --flux '+str(num1)
                        logfile1 = calplandir+'/'+load.prefix+'Flux-'+str(num1)+'_pbs.'+logtime+'.log'
                    elif ctype=='arcs':  # arcs
                        cmd1 += ' --wave '+str(num1)
                        logfile1 = calplandir+'/'+load.prefix+'Wave-'+str(num1)+'_pbs.'+logtime+'.log' 
                    elif ctype=='dailywave':  # dailywave
                        cmd1 += ' --dailywave '+str(num1)
                        logfile1 = calplandir+'/'+load.prefix+'DailyWave-'+str(num1)+'_pbs.'+logtime+'.log' 
                    elif ctype=='fpi':  # fpi
                        cmd1 += ' --fpi '+str(num1)
                        logfile1 = calplandir+'/'+load.prefix+'FPI-'+str(num1)+'_pbs.'+logtime+'.log'
                        if os.path.exists(os.path.dirname(logfile1))==False:
                            os.makedirs(os.path.dirname(logfile1))
                    elif ctype=='telluric':  # dailywave
                        cmd1 += ' --telluric '+str(num1)
                        logfile1 = calplandir+'/'+load.prefix+'Telluric-'+str(num1)+'_pbs.'+logtime+'.log' 
                    logfiles.append(logfile1)
                    logger.info('Calibration file %d : %s %s' % (j+1,ctype,str(num1)))
                    logger.info('Command : '+cmd1)
                    logger.info('Logfile : '+logfile1)
                    tasks['cmd'][j] = cmd1
                    tasks['outfile'][j] = logfile1
                    tasks['errfile'][j] = logfile1.replace('.log','.err')
                    tasks['dir'][j] = os.path.dirname(logfile1)
                logger.info('Creating '+str(ntorun)+' '+calnames[i]+' files')
                label = 'makecal-'+calnames[i]
                key,jobid = slrm.submit(tasks,label=label,verbose=True,logger=logger,**slurmpars1)
                slrm.queue_wait(label,key,jobid,sleeptime=60,verbose=True,logger=logger) # wait for jobs to complete   
            else:
                logger.info('No '+str(calnames[i])+' calibration files need to be run')

            # Step 4) Checks the status and update the database
            if ntorun>0:
                chkcal1 = check_calib(calinfo[torun],logfiles,key,apred,verbose=True,logger=logger)
                # Summary
                indcal, = np.where(chkcal1['success']==True)
                logger.info('%d/%d calibrations successfully processed' % (len(indcal),len(chkcal1)))
                if chkcal is None:
                    chkcal = chkcal1
                else:
                    chkcal = np.hstack((chkcal,chkcal1))

    return chkcal


def makeplanfiles(load,mjds,slurmpars,clobber=False,logger=None):
    """
    Make plan files for a list of MJDs.

    Parameters
    ----------
    load : ApLoad object
       ApLoad object that contains "apred" and "telescope".
    mjds : list
       List of MJDs to process
    slurmpars : dictionary
       Dictionary of slurmpars settings.
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

    # Should we parallelise this?  it can take a while to run for many nights
    
    # Loop over MJDs
    planfiles = []
    for m in mjds:
        logger.info(' ')
        logger.info('Making plan files for MJD='+str(m))
        plandicts,planfiles0 = mkplan.make_mjd5_yaml(m,apred,telescope,clobber=clobber,logger=logger)
        mjd5planfile = os.environ['APOGEEREDUCEPLAN_DIR']+'/yaml/'+apred+'/'+telescope+'/'+telescope+'_'+str(m)+'.yaml'
        if os.path.exists(mjd5planfile)==False:
            logger.info(mjd5planfile+' NOT FOUND')
            continue
        try:
            planfiles1 = mkplan.run_mjd5_yaml(mjd5planfile,clobber=clobber,logger=logger)
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
        #daycat = np.zeros(1,dtype=np.dtype([('mjd',int),('telescope',(str,10)),('nplanfiles',int),
        #                                    ('nexposures',int),('begtime',(str,50)),('success',bool)]))
        #daycat['mjd'] = m
        #daycat['telescope'] = telescope
        #daycat['nplanfiles'] = len(planfiles1)
        #daycat['nexposures'] = len(expinfo1)
        #daycat['begtime'] = begtime
        #daycat['success'] = False
        #db.ingest('daily_status',daycat)

    return planfiles


def runapred(load,mjds,slurmpars,clobber=False,logger=None):
    """
    Run APRED on all plan files for a list of MJDs.

    Parameters
    ----------
    load : ApLoad object
       ApLoad object that contains "apred" and "telescope".
    mjds : list
       List of MJDs to process
    slurmpars : dictionary
       Dictionary of slurmpars settings.
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
        return None,None
    logger.info(str(len(planfiles))+' plan files')

    # Get exposure information
    expinfo = getexpinfo(load,mjds,logger=logger,verbose=False)
    if len(expinfo)==0:
        logger.info('No exposures')
        return None,None
    
    # Loop over planfiles and see if the outputs exist already
    dorun = np.zeros(len(planfiles),bool)
    for i,pf in enumerate(planfiles):    
        pfbase = os.path.basename(pf)
        # Check if files exist already
        dorun[i] = True
        if clobber is not True:
            # Sky plan
            if pfbase.startswith(load.prefix+'Plan') and pfbase.find('sky') > -1:
                # apPlan-15404-59381sky.yaml
                config1,mjd1 = pfbase.split('sky')[0].split('-')[1:3]
                outlog = glob(pf.replace('.yaml','_pbs.??????????????.log'))
                outexists = False
                outfile = pf+' output files '
                if len(outlog)>0:                
                    outlog = np.flip(np.sort(outlog))
                    loglines = dln.readlines(outlog[0])
                    if loglines[-1]=='apred completed successfully':
                        outexists = True                
            # apPlan
            elif pfbase.startswith(load.prefix+'Plan'):
                # apPlan-3370-59623.yaml
                config1,mjd1 = pfbase.split('.')[0].split('-')[1:3]
                # check for apVisitSum file
                outfile = load.filename('VisitSum',plate=config1,mjd=mjd1,chips=True)
                outexists = os.path.exists(outfile)
            # apCalPlan
            elif pfbase.startswith(load.prefix+'CalPlan'):
                # apCalPlan-apogee-n-59640.yaml
                # It will take too long to load all of the plan files and check all of
                #  the output files.  Default is to redo them.
                outlog = glob(pf.replace('.yaml','_pbs.??????????????.log'))
                outexists = False
                outfile = pf+' output files '
                if len(outlog)>0:                
                    outlog = np.flip(np.sort(outlog))
                    loglines = dln.readlines(outlog[0])
                    if loglines[-1]=='apred completed successfully':
                        outexists = True
            # apDarkPlan
            elif pfbase.startswith(load.prefix+'DarkPlan'):
                # apDarkPlan-apogee-n-59640.yaml
                # It will take too long to load all of the plan files and check all of
                #  the output files.  Default is to redo them.
                outlog = glob(pf.replace('.yaml','_pbs.??????????????.log'))
                outexists = False
                outfile = pf+' output files '
                if len(outlog)>0:                
                    outlog = np.flip(np.sort(outlog))
                    loglines = dln.readlines(outlog[0])
                    if loglines[-1]=='apred completed successfully':
                        outexists = True
            # apExtraPlan
            elif pfbase.startswith(load.prefix+'ExtraPlan'):
                # apExtraPlan-apogee-n-59629.yaml
                # It will take too long to load all of the plan files and check all of
                #  the output files.  Default is to redo them.
                outlog = glob(pf.replace('.yaml','_pbs.??????????????.log'))
                outexists = False
                outfile = pf+' output files '
                if len(outlog)>0:                
                    outlog = np.flip(np.sort(outlog))
                    loglines = dln.readlines(outlog[0])
                    if loglines[-1]=='apred completed successfully':
                        outexists = True
            else:
                outexists = False
            if outexists:
                logger.info(str(i+1)+' '+os.path.basename(outfile)+' already exists and clobber==False')
                dorun[i] = False
    logger.info(str(np.sum(dorun))+' planfiles to run')
    
    # Loop over the planfiles and make the commands for the ones that we will run
    torun, = np.where(dorun==True)
    ntorun = len(torun)
    if ntorun>0:
        slurmpars1 = slurmpars.copy()
        if ntorun<slurmpars['ppn']:
            slurmpars1['cpus'] = ntorun
        slurmpars1['numpy_num_threads'] = 2
        logger.info('Slurm settings: '+str(slurmpars1))
        tasks = np.zeros(ntorun,dtype=np.dtype([('cmd',str,1000),('outfile',str,1000),('errfile',str,1000),('dir',str,1000)]))
        tasks = Table(tasks)
        #queue = pbsqueue(verbose=True)
        #queue.create(label='apred', **slurmpars1)
        for i in range(ntorun):
            pf = planfiles[torun[i]]
            pfbase = os.path.basename(pf)
            logfile1 = pf.replace('.yaml','_pbs.'+logtime+'.log')
            errfile1 = logfile1.replace('.log','.err')
            outdir = os.path.dirname(logfile1)
            if os.path.exists(outdir)==False:   # make sure the output directory exists
                os.makedirs(outdir)
            cmd1 = 'apred {0}'.format(pf)
            if clobber:
                cmd1 += ' --clobber'
            logger.info('planfile %d : %s' % (i+1,pf))
            logger.info('Command : '+cmd1)
            logger.info('Logfile : '+logfile1)
            tasks['cmd'][i] = cmd1
            tasks['outfile'][i] = logfile1
            tasks['errfile'][i] = errfile1
            tasks['dir'][i] = os.path.dirname(logfile1)
            #queue.append(cmd1, outfile=logfile1,errfile=errfile1)
        logger.info('Running APRED on '+str(ntorun)+' planfiles')
        key,jobid = slrm.submit(tasks,label='apred',verbose=True,logger=logger,**slurmpars1)
        #queue.commit(hard=True,submit=True)
        slrm.queue_wait('apred',key,jobid,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete
        #queue_wait(queue,sleeptime=120,verbose=True,logger=logger)  # wait for jobs to complete        
        # This also loads the status into the database using the correct APRED version
        chkexp,chkvisit = check_apred(expinfo,planfiles,key,verbose=True,logger=logger)
    else:
        logger.info('No planfiles need to be run')
        chkexp,chkvisit = None,None


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


def runrv(load,mjds,slurmpars,daily=False,clobber=False,logger=None):
    """
    Run RV on all the stars observed from a list of MJDs.

    Parameters
    ----------
    load : ApLoad object
       ApLoad object that contains "apred" and "telescope".
    mjds : list
       List of MJDs to process
    slurmpars : dictionary
       Dictionary of slurmpars settings.
    daily : boolean, optional
       Run for the daily processing.  Only include visits up to and including this night.
    clobber : boolean, optional
       Overwrite existing files.  Default is False.
    logger : logger, optional
       Logging object.  If not is input, then a default one will be created.

    Returns
    -------
    The RV + combination software is run on the MJDs and relevant output files are created.

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

    # Get the visit information from the database
    logger.info('Getting visit information from the database')
    if daily:
        sql = "SELECT apogee_id,mjd from apogee_drp.visit WHERE apred_vers='%s' and mjd<=%d and telescope='%s'" % (apred,mjdstop,telescope)        
    else:
        sql = "SELECT apogee_id,mjd from apogee_drp.visit WHERE apred_vers='%s' and telescope='%s'" % (apred,telescope)
    db = apogeedb.DBSession()
    allvisit = db.query(sql=sql)
    db.close()

    if len(allvisit)==0:
        logger.info('No visits found for MJDs')
        return None

    # Remove rows with missing or blank apogee_ids
    apidlen = np.array([len(v['apogee_id']) for v in allvisit])   # 2M name should have 18 characters
    bd, = np.where((allvisit['apogee_id']=='') | (allvisit['apogee_id']=='None') | (allvisit['apogee_id']=='none') |
                   (allvisit['apogee_id'] is None) | (allvisit['apogee_id']=='2MNone') | (allvisit['apogee_id']=='2M') | (apidlen != 18))
    if len(bd)>0:
        allvisit = np.delete(allvisit,bd)

    # Find the visits for the input MJDs
    ind = []
    for m in mjds:
        gd, = np.where(allvisit['mjd']==m)
        if len(gd)>0: ind += list(gd)
    ind = np.array(ind)
    if len(ind)==0:
        logger.info('No visits found for MJDs')
        return None
    visits = allvisit[ind]

    # Make table for all the stars we are interested in
    apogee_id = np.unique(visits['apogee_id'])             # get the IDs of the stars for the input MJDs
    star_index = dln.create_index(allvisit['apogee_id'])    # visit index for all stars
    vind1,vind2 = dln.match(apogee_id,star_index['value'])  # match up our star IDs with the visit index
    dtype = [('apogee_id',(str,50)),('mjd',int),('maxmjd',int),('nvisits',int),('apred_vers',(str,50)),('telescope',(str,50))]
    vcat = np.zeros(len(apogee_id),dtype=np.dtype(dtype))
    vcat['apogee_id'] = apogee_id
    vcat['nvisits'][vind1] = star_index['num'][vind2]
    
    # Get MAXMJD for each unique star
    for i in range(len(vind1)):
        v1 = vind1[i]
        v2 = vind2[i]
        sind = star_index['index'][star_index['lo'][v2]:star_index['hi'][v2]+1]
        maxmjd = np.max(allvisit['mjd'][sind])
        vcat['mjd'][v1] = maxmjd
        vcat['maxmjd'][v1] = maxmjd
        vcat['apred_vers'][v1] = apred
        vcat['telescope'][v1] = telescope
            
    logger.info(str(len(vcat))+' stars to run')
    
    # Change MJD to MAXMJD because the apStar file will have MAXMJD in the name
    if daily==False and len(mjds)>1:
        vcat['mjd'] = vcat['maxmjd']    

    # Loop over the stars and figure out the ones that need to be run
    if clobber==False:
        logger.info('Checking which stars need to be run')
        dorv = np.zeros(len(vcat),bool)
        for i,obj in enumerate(vcat['apogee_id']):
            # We are going to run RV on ALL the visits
            # Use the MAXMJD in the table, now called MJD
            mjd = vcat['mjd'][i]
            apstarfile = load.filename('Star',obj=obj)
            if daily:
                # Want all visits up to this day
                apstarfile = apstarfile.replace('.fits','-'+str(mjds[0])+'.fits')
            else:
                apstarfile = apstarfile.replace('.fits','-'+str(mjd)+'.fits')
            # Check if file exists already
            dorv[i] = True
            if os.path.exists(apstarfile):
                logger.info(str(i+1)+' '+os.path.basename(apstarfile)+' already exists and clobber==False')
                dorv[i] = False
    else:
        dorv = np.ones(len(vcat),bool)
    logger.info(str(np.sum(dorv))+' objects to run')
    
    # Loop over the objects and make the commands for the ones that we will run
    torun, = np.where(dorv==True)
    ntorun = len(torun)
    if ntorun>0:
        slurmpars1 = slurmpars.copy()
        if ntorun<slurmpars1['ppn']:
            slurmpars1['cpus'] = ntorun
        slurmpars1['numpy_num_threads'] = 2
        logger.info('Slurm settings: '+str(slurmpars1))
        tasks = np.zeros(ntorun,dtype=np.dtype([('cmd',str,1000),('outfile',str,1000),('errfile',str,1000),('dir',str,1000)]))
        tasks = Table(tasks)
        for i in range(ntorun):
            obj = vcat['apogee_id'][torun[i]]
            # We are going to run RV on ALL the visits
            # Use the MAXMJD in the table, now called MJD
            mjd = vcat['mjd'][torun[i]]
            apstarfile = load.filename('Star',obj=obj)
            if daily:
                # Want all visits up to this day
                apstarfile = apstarfile.replace('.fits','-'+str(mjdstop)+'.fits')
            else:
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
            if daily:
                cmd += '  --m '+str(mjds[0])
            logger.info('rv %d : %s' % (i+1,obj))
            logger.info('Command : '+cmd)
            logger.info('Logfile : '+logfile)
            tasks['cmd'][i] = cmd
            tasks['outfile'][i] = logfile
            tasks['errfile'][i] = errfile
            tasks['dir'][i] = os.path.dirname(logfile) 
        logger.info('Running RV on '+str(ntorun)+' stars')
        key,jobid = slrm.submit(tasks,label='rv',verbose=True,logger=logger,**slurmpars1)
        slrm.queue_wait('rv',key,jobid,sleeptime=60,verbose=True,logger=logger) # wait for jobs to complete  
        # This checks the status and puts it into the database
        chkrv = check_rv(vcat[torun],key,logger=logger,verbose=False)
    else:
        logger.info('No RVs need to be run')
        chkrv = None

    # -- Summary statistics --
    # RV status
    if chkrv is not None:
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


def rununified(load,mjds,slurmpars,clobber=False,logger=None):
    """
    Create the unified MWM directory structure for the relevant MJDs.

    Parameters
    ----------
    load : ApLoad object
       ApLoad object that contains "apred" and "telescope".
    mjds : list
       List of MJDs to process
    slurmpars : dictionary
       Dictionary of slurmpars settings.
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
    
    slurmpars1 = slurmpars.copy()
    if len(mjds)<slurmpars1['ppn']:
        slurmpars1['cpus'] = len(mjds)
    slurmpars1['numpy_num_threads'] = 2    
    logger.info('Slurm settings: '+str(slurmpars1))
    tasks = np.zeros(len(mjds),dtype=np.dtype([('cmd',str,1000),('outfile',str,1000),('errfile',str,1000),('dir',str,1000)]))
    tasks = Table(tasks)
    #queue = pbsqueue(verbose=True)
    #queue.create(label='unidir', **slurmpars1)
    # Loop over all MJDs
    for m in mjds:
        logfile = os.environ['APOGEE_REDUX']+'/'+apred+'/log/'+observatory+'/'+str(mjd5)+'-unidir.'+logtime+'.log'
        errfile = logfile.replace('.log','.err')
        if os.path.exists(os.path.dirname(logfile))==False:
            os.makedirs(os.path.dirname(logfile))
        cmd = 'sas_mwm_healpix --spectro apogee --mjd {0} --telescope {1} --drpver {2} -v'.format(mjd5,telescope,apred)
        tasks['cmd'][i] = cmd
        tasks['outfile'][i] = logfile
        tasks['errfile'][i] = errfile
        tasks['dir'][i] = os.path.dirname(logfile)
        #queue.append(cmd,outfile=logfile, errfile=errfile)
    logger.info('Making unified directory structure for '+str(len(MJDS)))
    key,jobid = slrm.submit(tasks,label='unidir',verbose=True,logger=logger,**slurmpars1)
    slrm.queue_wait('unidir',key,jobid,sleeptime=60,verbose=True,logger=logger) # wait for jobs to complete   
    #queue.commit(hard=True,submit=True)
    #logger.info('PBS key is '+queue.key)        
    #queue_wait(queue,sleeptime=60,verbose=True,logger=logger)  # wait for jobs to complete
    #del queue    
    #  sas_mwm_healpix --spectro apogee --mjd 59219 --telescope apo25m --drpver daily -v


def runqa(load,mjds,slurmpars,clobber=False,logger=None):
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

    if logger is None:
        logger = dln.basiclogger()
        
    # Get plan files for these MJDs
    logger.info('Getting plan files')
    planfiles = getplanfiles(load,mjds,exist=True,logger=logger)
    nplans = len(planfiles)
    # Only want apPlan files
    if nplans>0:
        planfiles = [p for p in planfiles if os.path.basename(p).startswith(load.prefix+'Plan')]
        nplans = len(planfiles)
        #if load.instrument=='apogee-n':
        #    planfiles = [p for p in planfiles if os.path.basename(p).startswith('apPlan')]
        #else:
        #    planfiles = [p for p in planfiles if os.path.basename(p).startswith('asPlan')]
    logger.info(str(nplans)+' plan file(s)')

    # Run apqa on each plate visit
    if nplans>0:
        slurmpars1 = slurmpars.copy()
        if nplans<slurmpars['ppn']:
            slurmpars1['cpus'] = nplans
        slurmpars1['numpy_num_threads'] = 2    
        logger.info('Slurm settings: '+str(slurmpars1))
        tasks = np.zeros(len(planfiles),dtype=np.dtype([('cmd',str,1000),('outfile',str,1000),('errfile',str,1000),('dir',str,1000)]))
        tasks = Table(tasks)
        for i,pf in enumerate(planfiles):
            logger.info('planfile %d : %s' % (i+1,pf))
            fdir = os.path.dirname(pf)
            # apPlan-1491-59587.yaml
            base = os.path.basename(pf)
            dum = base.split('-')
            plate = dum[1]
            mjd = dum[2].split('.')[0]
            field = os.path.dirname(os.path.dirname(fdir)).split('/')[-1]
            logfile = fdir+'/apqa-'+plate+'-'+mjd+'_pbs.'+logtime+'.log'
            errfile = logfile.replace('.log','.err')
            cmd = 'apqa {0} {1} --apred {2} --telescope {3} --plate {4}'.format(mjd,observatory,apred,telescope,plate)
            #cmd += ' --masterqa False --starhtml False --starplots False --nightqa False --monitor False'
            cmd += ' --masterqa False --starhtml True --starplots True --nightqa False --monitor False'
            logger.info('Command : '+cmd)
            logger.info('Logfile : '+logfile)
            tasks['cmd'][i] = cmd
            tasks['outfile'][i] = logfile
            tasks['errfile'][i] = errfile
            tasks['dir'][i] = os.path.dirname(logfile)
        logger.info('Running APQA on '+str(len(planfiles))+' planfiles')
        key,jobid = slrm.submit(tasks,label='apqa',verbose=True,logger=logger,**slurmpars1)
        slrm.queue_wait('apqa',key,jobid,sleeptime=60,verbose=True,logger=logger) # wait for jobs to complete 

    
    # Make nightly QA/summary pages
    # we should parallelize this
    for m in mjds:
        try:
            apodir = os.environ.get('APOGEE_REDUX')+'/'
            qa.makeNightQA(load=load,mjd=str(m),telescope=telescope,apred=apred)
            # Run makeCalFits, makeDarkFits, makeExpFits
            # makeCalFits
            expinfo = getexpinfo(load,int(m),logger=logger,verbose=False)
            if len(expinfo)>0:
                calind, = np.where((expinfo['exptype']=='ARCLAMP') | (expinfo['exptype']=='QUARTZFLAT') |
                                   (expinfo['exptype']=='DOMEFLAT') | (expinfo['exptype']=='FPI'))
                if len(calind)>0:
                    all_ims = expinfo['num'][calind]
                    all_types = np.full(len(all_ims), 'cal')
                    all_types[expinfo['exptype'][calind]=='FPI'] = 'fpi'
                    qa.makeCalFits(load=load, ims=all_ims, types=all_types, mjd=str(m), instrument=instrument, clobber=clobber)
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
        mjdmin = np.minimum(np.min(np.array(mjds).astype(int)),59146)
        qa.makeMasterQApages(mjdmin=mjdmin, mjdmax=9999999, apred=apred,
                             mjdfilebase='mjd.html',fieldfilebase='fields.html',
                             domjd=True, dofields=True)
    except:
        traceback.print_exc()

    # Make apStar HTML and plots
    # Loop over the apVisitSum files and figure out the stars that need to be run
    #qastars = []
    #for i in range(nplans):
    #    visSumFile = load.filename('VisitSum',plate=int(plates[i]),mjd=int(mjds[i]),field=fields[i])
    #    visSum = fits.getdata(visSumFile)
    #    obj = visSum['APOGEE_ID']
    #    ns = len(ids)
    #    for j in range(ns):
    #        apstarfile = load.filename('Star',fields[i],obj[j])
    #        if os.path.exists(apstarfile): qastars.append(obj[j])
    #qastars = np.array(qastars)
    #nstars = len(qastars)
    #logger.info(str(nstars)+' objects to run apStar QA on')

    #if nstars > 0: 

    # Run monitor page
    #  always runs on all MJDs
    monitor.monitor(instrument=instrument, apred=apred)

    
def summary_email(observatory,apred,mjd,steps,chkmaster=None,chk3d=None,chkcal=None, 
                  planfiles=None,chkexp=None,chkvisit=None,chkrv=None,logfile=None,slurmpars=None,
                  clobber=None,debug=False):   
    """ Send a summary email."""

    urlbase = 'https://data.sdss5.org/sas/sdsswork/mwm/apogee/spectro/redux/'
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
    if slurmpars:
        message += 'Slurm settings: '+str(slurmpars)+'<br>\n'
    message += '<p>\n'
    message += '<a href="'+urlbase+str(apred)+'/qa/mjd.html">QA Webpage (MJD List)</a><br> \n'

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
    if 'apred' in steps and chkexp is not None and chkvisit is not None:
        inde, = np.where(chkexp['success']==True)
        message += 'APRED: %d/%d exposures successfully processed<br> \n' % (len(inde),len(chkexp))
        indv, = np.where(chkvisit['success']==True)
        message += 'APRED: %d/%d visits successfully processed<br> \n' % (len(indv),len(chkvisit))

    # RV step
    if 'rv' in steps and chkrv is not None:
        ind, = np.where(chkrv['success']==True)
        message += 'RV: %d/%d RV+visit combination successfully processed<br> \n' % (len(ind),len(chkrv))

    # Link to logfile
    url = urlbase+logfile[logfile.find('/redux/')+7:]
    message += '\n\n Logfile: <a href="'+url+'">'+os.path.basename(logfile)+'</a><br>\n'

    #   If logfile is too large (>1MB), then do not attach the file    
    if os.path.getsize(logfile)>1e6:
        message += 'Log file is too large to attach\n'
        
    message += """\
                 </p>
                 </body>
               </html>
               """

    # Send the message
    #   If logfile is too large (>1MB), then do not attach the file
    if os.path.getsize(logfile)>1e6 or len(mjds)>1:
        email.send(address,subject,message,send_from='noreply.apogeedrp')
    else:
        email.send(address,subject,message,files=logfile,send_from='noreply.apogeedrp')    
    

def run(observatory,apred,mjd=None,steps=None,caltypes=None,clobber=False,
        fresh=False,linkvers=None,nodes=5,alloc='sdss-np',qos=None,
        walltime='336:00:00',debug=False):
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
    caltypes : list, optional
       Calibration types to run.  This is used to select a subset of the master cals or daily cals
         to run.  Default is to run all of them.
    clobber : boolean, optional
       Overwrite any existing data.  Default is False.
    fresh : boolean, optional
       Start the reduction directory fresh.  The default is continue with what is
         already there.
    linkvers : str, optional
       Name of reduction version to use for symlinks for the calibration files.
    nodes : int, optional
       Number of nodes to use on the CHPC.  Default is 5.
    alloc : str, optional
       The slurm partition to use.  Default is 'sdss-np'.
    qos : str, optional
       The type of slurm queue to use.  Default is 'sdss'.
    walltime : str, optional
       Maximum runtime for the slurm jobs.  Default is '336:00:00' or 14 days.
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
    #alloc = 'sdss-np'
    if alloc == 'sdss-kp':
        shared = False
        ppn = 16
    else:
        shared = True
        ppn = 64
    #ppn = 64
    #walltime = '336:00:00'
    # Only set cpus if you want to use less than 64 cpus
    slurmpars = {'nodes':nodes, 'alloc':alloc, 'qos':qos, 'ppn':ppn, 'cpus':ppn,
                 'shared':shared, 'walltime':walltime, 'notification':False}
    
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
            if os.path.islink(apogee_redux+apred)==False:
                shutil.rmtree(apogee_redux+apred)
            else:
                raise Exception('Cannot remove directory that is a symbolic link')
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
    rootLogger.info('Slurm settings: '+str(slurmpars))

    # Common keyword arguments
    kws = {'slurmpars':slurmpars, 'clobber':clobber, 'logger':rootLogger}

    # Defaults for check tables
    chkmaster,chk3d,chkcal,planfiles,chkexp,chkvisit,chkrv = None,None,None,None,None,None,None
    
    # 1) Setup the directory structure
    #----------------------------------
    if 'setup' in steps:
        rootLogger.info('')
        rootLogger.info('-------------------------------------')
        rootLogger.info('1) Setting up the directory structure')
        rootLogger.info('=====================================')
        rootLogger.info('')

        tasks = np.zeros(1,dtype=np.dtype([('cmd',str,1000),('outfile',str,1000),('errfile',str,1000),('dir',str,1000)]))
        tasks = Table(tasks)
        cmd = 'mkvers {0}'.format(apred)        
        mkvoutfile = os.environ['APOGEE_REDUX']+'/'+apred+'/log/mkvers.'+logtime+'.log'
        mkverrfile = mkvoutfile.replace('-mkvers.log','-mkvers.'+logtime+'.err')
        if os.path.exists(os.path.dirname(mkvoutfile))==False:
            os.makedirs(os.path.dirname(mkvoutfile))        
        rootLogger.info('Command : '+cmd)
        rootLogger.info('Logfile : '+mkvoutfile)        
        tasks['cmd'][0] = cmd
        tasks['outfile'][0] = mkvoutfile
        tasks['errfile'][0] = mkverrfile
        tasks['dir'][0] = os.path.dirname(mkvoutfile)
        slurmpars1 = {'nodes':1, 'alloc':alloc, 'cpus':1, 'shared':shared, 'walltime':walltime, 'notification':False}
        key,jobid = slrm.submit(tasks,label='mkvers',verbose=True,logger=rootLogger,**slurmpars1)
        slrm.queue_wait('mkvers',key,jobid,sleeptime=60,verbose=True,logger=rootLogger) # wait for jobs to complete 

    # 2) Master calibration products, make sure to do them in the right order
    #------------------------------------------------------------------------
    if 'master' in steps:
        rootLogger.info('')
        rootLogger.info('-----------------------------------------')
        rootLogger.info('2) Generating master calibration products')
        rootLogger.info('=========================================')
        rootLogger.info('')
        chkmaster = mkmastercals(load,mjds,caltypes=caltypes,linkvers=linkvers,**kws)

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
        chkcal = rundailycals(load,mjds,caltypes=caltypes,**kws)

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
        chkexp,chkvisit = runapred(load,mjds,**kws)
        
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
    #daycat = np.zeros(1,dtype=np.dtype([('pk',int),('mjd',int),('telescope',(str,10)),('nplanfiles',int),
    #                                    ('nexposures',int),('begtime',(str,50)),('endtime',(str,50)),('success',bool)]))
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
                  planfiles=planfiles,chkexp=chkexp,chkvisit=chkvisit,chkrv=chkrv,logfile=logfile,
                  slurmpars=slurmpars,clobber=clobber,debug=debug)

