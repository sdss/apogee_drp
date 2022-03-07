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


def summary_email(observatory,mjd5,chkcal,chkexp,chkvisit,chkrv,logfiles=None):
    """ Send a summary email."""

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


def run_daily(observatory,mjd5=None,apred=None,qos='sdss-fast',clobber=False):
    """ Perform daily APOGEE data reduction."""

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
        rootLogger.info('-----------------------------')
        rootLogger.info('Running AP3D on all exposures')
        rootLogger.info('=============================')
        rootLogger.info('')
        queue = pbsqueue(verbose=True)
        queue.create(label='ap3d', nodes=nodes, alloc=alloc, ppn=ppn, cpus=np.minimum(cpus,len(expinfo)),
                     qos=qos, shared=shared, numpy_num_threads=2, walltime=walltime, notification=False)
        do3d = np.zeros(len(expinfo),bool)
        for i,num in enumerate(expinfo['num']):
            logfile1 = load.filename('2D',num=num,mjd=mjd5,chips=True).replace('2D','3D')
            logfile1 = os.path.dirname(logfile1)+'/logs/'+os.path.basename(logfile1)
            logfile1 = logfile1.replace('.fits','_pbs.'+logtime+'.log')
            if os.path.dirname(logfile1)==False:
                os.makedirs(os.path.dirname(logfile1))
            # Check if files exist already
            do3d[i] = True
            if clobber is not True:
                outfile = load.filename('2D',num=num,mjd=mjd5,chips=True)
                if load.exists('2D',num=num):
                    rootLogger.info(os.path.basename(outfile)+' already exists and clobber==False')
                    do3d[i] = False
            if do3d[i]:
                cmd1 = 'ap3d --num {0} --vers {1} --telescope {2} --unlock'.format(num,apred,telescope)
                rootLogger.info('Exposure %d : %d' % (i+1,num))
                rootLogger.info('Command : '+cmd1)
                rootLogger.info('Logfile : '+logfile1)
                queue.append(cmd1,outfile=logfile1,errfile=logfile1.replace('.log','.err'))
        if np.sum(do3d)>0:
            queue.commit(hard=True,submit=True)
            rootLogger.info('PBS key is '+queue.key)
            apogeedrp.queue_wait(queue,sleeptime=60,verbose=True,logger=rootLogger)  # wait for jobs to complete
            chk3d = apogeedrp.check_ap3d(expinfo,queue.key,apred,telescope,verbose=True,logger=rootLogger)
        else:
            rootLogger.info('No exposures need AP3D processing')
        del queue
    else:
        rootLogger.info('No exposures to process with AP3D')

    # Do QA check of the files
    rootLogger.info(' ')
    rootLogger.info('Doing quality checks on all exposures')
    qachk = check.check(expinfo['num'],apred,telescope,verbose=True,logger=rootLogger)
    rootLogger.info(' ')

    # Run calibration files using "pbs" packages
    #-------------------------------------------
    # First we need to run domeflats and quartzflats so there are apPSF files
    # Then the arclamps
    # Then the FPI exposures last (needs apPSF and apWave files)
    # Only use calibration exposures that have passed the quality assurance checks
    calind, = np.where(((expinfo['exptype']=='DOMEFLAT') | (expinfo['exptype']=='QUARTZFLAT') | 
                        (expinfo['exptype']=='ARCLAMP') | (expinfo['exptype']=='FPI')) &
                       (qachk['okay']==True))

    if len(calind)>0:
        # Only need one FPI per night
        # The FPI processing is done at a NIGHT level
        fpi, = np.where(expinfo['exptype'][calind]=='FPI')
        if len(fpi)>1:
            # Take the first FPI exposure
            rootLogger.info('Only keeping ONE FPI exposure per night/MJD')
            calind = np.delete(calind,fpi[1:])  # remove all except the first one


        # 1: psf, 2: flux, 4: arcs, 8: fpi
        if fps:
            # Only use QUARTZFLATs in the FPS era because they have all 300 fibers
            #  domeflats are missing the 2 dedicated FPI fibers
            calcodedict = {'DOMEFLAT':2, 'QUARTZFLAT':1, 'ARCLAMP':4, 'FPI':8}
        else:
            calcodedict = {'DOMEFLAT':3, 'QUARTZFLAT':1, 'ARCLAMP':4, 'FPI':8}
        calcode = [calcodedict[etype] for etype in expinfo['exptype'][calind]]
        calnames = ['DOMEFLAT/QUARTZFLAT','FLUX','ARCLAMP','FPI']
        shcalnames = ['psf','flux','arcs','fpi']
        filecodes = ['PSF','Flux','Wave','WaveFPI']
        chkcal = []
        for j,ccode in enumerate([1,2,4,8]):
            rootLogger.info('')
            rootLogger.info('----------------------------------------------')
            rootLogger.info('Running Calibration Files: '+str(calnames[j]))
            rootLogger.info('==============================================')
            rootLogger.info('')
            cind, = np.where((np.array(calcode) & ccode) > 0)
            if len(cind)>0:
                if ccode==8: cind = cind[[0]] # Only run first FPI exposure
                rootLogger.info(str(len(cind))+' file(s)')
                queue = pbsqueue(verbose=True)
                queue.create(label='makecal-'+shcalnames[j], nodes=nodes, alloc=alloc, ppn=ppn, cpus=np.minimum(cpus,len(cind)),
                             qos=qos, shared=shared, numpy_num_threads=2, walltime=walltime, notification=False)
                calplandir = os.path.dirname(load.filename('CalPlan',num=0,mjd=mjd5))
                logfiles = []
                docal = np.zeros(len(cind),bool)
                for k in range(len(cind)):
                    num1 = expinfo['num'][calind[cind[k]]]
                    exptype1 = expinfo['exptype'][calind[cind[k]]]
                    arctype1 = expinfo['arctype'][calind[cind[k]]]
                    if ccode==1:   # psfs                    
                        cmd1 = 'makecal --psf '+str(num1)+' --unlock'
                        if clobber: cmd1 += ' --clobber'
                        logfile1 = calplandir+'/apPSF-'+str(num1)+'_pbs.'+logtime+'.log'
                    elif ccode==2:  # flux
                        cmd1 = 'makecal --flux '+str(num1)+' --unlock'
                        if fps: cmd1 += ' --psflibrary'
                        if clobber: cmd1 += ' --clobber'
                        logfile1 = calplandir+'/apFlux-'+str(num1)+'_pbs.'+logtime+'.log'
                    elif ccode==4: # and exptype1=='ARCLAMP' and (arctype1=='UNE' or arctype1=='THARNE'):  # arcs
                        cmd1 = 'makecal --wave '+str(num1)+' --unlock'
                        if fps: cmd1 += ' --psflibrary'
                        if clobber: cmd1 += ' --clobber'
                        logfile1 = calplandir+'/apWave-'+str(num1)+'_pbs.'+logtime+'.log'
                    elif ccode==8: # and exptype1=='ARCLAMP' and arctype1=='FPI':    # fpi
                        cmd1 = 'makecal --fpi '+str(num1)+' --unlock'
                        if fps: cmd1 += ' --psflibrary'
                        if clobber: cmd1 += ' --clobber'
                        logfile1 = calplandir+'/apFPI-'+str(num1)+'_pbs.'+logtime+'.log'
                    logfiles.append(logfile1)
                    # Check if files exist already
                    docal[k] = True
                    if clobber is not True:
                        outfile = load.filename(filecodes[j],num=num1,mjd=mjd5,chips=True)
                        if load.exists(filecodes[j],num=num1,mjd=mjd5):
                            rootLogger.info(os.path.basename(outfile)+' already exists and clobber==False')
                            docal[k] = False
                    if docal[k]:
                        rootLogger.info('Calibration file %d : %s %d' % (k+1,exptype1,num1))
                        rootLogger.info('Command : '+cmd1)
                        rootLogger.info('Logfile : '+logfile1)
                        queue.append(cmd1, outfile=logfile1,errfile=logfile1.replace('.log','.err'))
                if np.sum(docal)>0:
                    queue.commit(hard=True,submit=True)
                    rootLogger.info('PBS key is '+queue.key)
                    apogeedrp.queue_wait(queue,sleeptime=60,verbose=True,logger=rootLogger)  # wait for jobs to complete
                else:
                    rootLogger.info('No '+str(calnames[j])+' calibration files need to be run') 
                calinfo = expinfo[calind[cind]]
                chkcal1 = apogeedrp.check_calib(calinfo,logfiles,queue.key,apred,verbose=True,logger=rootLogger)
                if len(chkcal)==0:
                    chkcal = chkcal1
                else:
                    chkcal = np.hstack((chkcal,chkcal1))
                del queue
            else:
                rootLogger.info('No '+str(calnames[j])+' calibration files to run')


    # Make MJD5 and plan files
    #--------------------------
    # Check that the necessary daily calibration files exist
    rootLogger.info(' ')
    rootLogger.info('Making plan files')
    plandicts,planfiles = mkplan.make_mjd5_yaml(mjd5,apred,telescope,clobber=True,logger=rootLogger)
    dailyplanfile = os.environ['APOGEEREDUCEPLAN_DIR']+'/yaml/'+telescope+'/'+telescope+'_'+str(mjd5)+'.yaml'
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
        for i,pf in enumerate(planfiles):
            rootLogger.info('planfile %d : %s' % (i+1,pf))
            logfile = pf.replace('.yaml','_pbs.'+logtime+'.log')
            errfile = logfile.replace('.log','.err')
            cmd1 = 'apred {0}'.format(pf)
            rootLogger.info('Command : '+cmd1)
            rootLogger.info('Logfile : '+logfile)
            queue.append(cmd1, outfile=logfile, errfile=errfile)
        queue.commit(hard=True,submit=True)
        rootLogger.info('PBS key is '+queue.key)
        apogeedrp.queue_wait(queue,sleeptime=120,verbose=True,logger=rootLogger)  # wait for jobs to complete
        chkexp,chkvisit = apogeedrp.check_apred(expinfo,planfiles,queue.key,verbose=True,logger=rootLogger)
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
        # Get unique stars
        objects,ui = np.unique(vcat['apogee_id'],return_index=True)
        vcat = vcat[ui]
        # Remove ones with missing or blank apogee_ids
        bd, = np.where((vcat['apogee_id']=='') | (vcat['apogee_id']=='None') | (vcat['apogee_id']=='2MNone') | (vcat['apogee_id']=='2M'))
        if len(bd)>0:
            vcat = np.delete(vcat,bd)
        else:
            vcat = []
    if len(vcat)>0:
        queue = pbsqueue(verbose=True)
        queue.create(label='rv', nodes=nodes, alloc=alloc, ppn=ppn, cpus=cpus, qos=qos, shared=shared, numpy_num_threads=2,
                     walltime=walltime, notification=False)
        for i,obj in enumerate(vcat['apogee_id']):
            apstarfile = load.filename('Star',obj=obj)
            outdir = os.path.dirname(apstarfile)  # make sure the output directories exist
            if os.path.exists(outdir)==False:
                os.makedirs(outdir)
            # Run with --verbose and --clobber set
            rootLogger.info('rv %d : %s' % (i+1,obj))
            logfile = apstarfile.replace('.fits','-'+str(mjd5)+'_pbs.'+logtime+'.log')
            errfile = logfile.replace('.log','.err')
            cmd1 = 'rv %s %s %s -c -v -m %s' % (obj,apred,telescope,mjd5)
            rootLogger.info('Command : '+cmd1)
            rootLogger.info('Logfile : '+logfile)
            queue.append(cmd1,outfile=logfile,errfile=errfile)
        queue.commit(hard=True,submit=True)
        rootLogger.info('PBS key is '+queue.key)        
        apogeedrp.queue_wait(queue,sleeptime=120,verbose=True,logger=rootLogger)  # wait for jobs to complete
        chkrv = apogeedrp.check_rv(vcat,queue.key)
        del queue
    else:
        rootLogger.info('No visit files for MJD=%d' % mjd5)
        chkrv = None


    # Create daily and full allVisit/allStar files
    # The QA code needs these
    apogeedrp.create_sumfiles(apred,telescope,mjd5)


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
    apogeedrp.queue_wait(queue)  # wait for jobs to complete
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
    summary_email(observatory,mjd5,chkcal,chkexp,chkvisit,chkrv,logfile)
