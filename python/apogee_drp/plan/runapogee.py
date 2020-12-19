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


def getNextMJD(observatory,apred='t14'):
        ''' Returns the next MJD to reduce.  Either a list or one. '''

        # Grab the MJD from the currentmjd file
        nextfile = os.path.join(os.getenv('APOGEE_REDUX'), 'daily', 'log', observatory, 'currentmjd')
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

def dbload_plans(planfiles):
    """  Load plan files into the database."""
    db = apogeedb.DBSession()   # open db session
    nplans = len(planfiles)
    
    # Loop over the planfiles
    dtype = np.dtype([('planfile',(np.str,300)),('apred_vers',(np.str,20)),('telescope',(np.str,10)),
                      ('instrument',(np.str,20)),('mjd',int),('plate',int),('platetype',(np.str,20))])
    plantab = np.zeros(nplans,dtype=dtype)
    for i,planfile in enumerate(planfiles):
        planstr = plan.load(planfile)
        plantab['planfile'][i] = planfile
        plantab['apred_vers'][i] = planstr['apred_vers']
        plantab['telescope'][i] = planstr['telescope']
        plantab['instrument'][i] = planstr['instrument']
        plantab['mjd'][i] = planstr['mjd']
        plantab['plate'][i] = planstr['plateid']
        plantab['platetype'][i] = planstr['platetype']

    # Insert into the database
    db.ingest('plan',plantab)
    db.close()   # close db session


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
        if platetype=='normal':
            plugmap = platedata.getdata(plate,mjd,apred_vers,telescope,plugid=planstr['plugmap'])
            fiberdata = plugmap['fiberdata']
        else:
            fiberdata = None

        # Exposure-level processing: ap3d, ap2d, apcframe
        dtype = np.dtype([('exposure_pk',int),('planfile',(np.str,300)),('apred_vers',(np.str,20)),('instrument',(np.str,20)),
                          ('telescope',(np.str,10)),('platetype',(np.str,50)),('mjd',int),('plate',int),
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
            chkexp1['checktime'][cnt] = str(datetime.now())
            if np.sum(exists)==3:
                chkexp1['success'][cnt] = True
            cnt += 1
            # AP2D
            #-----
            if (platetype=='normal') | (platetype=='cal'):
                chkexp1['exposure_pk'][cnt] = exposure_pk
                chkexp1['num'][cnt] = num
                chkexp1['proctype'][cnt] = 'AP2D'
                base = load.filename('1D',num=num,mjd=mjd,chips=True)
                chfiles = [base.replace('1D-','1D-'+ch+'-') for ch in ['a','b','c']]
                exists = [os.path.exists(chf) for chf in chfiles]
                if np.sum(exists)==3:
                    chkexp1['success'][cnt] = True
                cnt += 1
            # APCframe
            #---------
            if platetype=='normal':
                chkexp1['exposure_pk'][cnt] = exposure_pk
                chkexp1['num'][cnt] = num
                chkexp1['proctype'][cnt] = 'APCFRAME'
                base = load.filename('Cframe',num=num,mjd=mjd,plate=plate,chips=True)
                chfiles = [base.replace('Cframe-','Cframe-'+ch+'-') for ch in ['a','b','c']]
                exists = [os.path.exists(chf) for chf in chfiles]
                if np.sum(exists)==3:
                    chkexp1['success'][cnt] = True
                cnt += 1
        # Trim extra elements
        chkexp1 = chkexp1[0:cnt]

        # Plan summary and ap1dvisit
        #---------------------------
        dtypeap = np.dtype([('planfile',(np.str,300)),('logfile',(np.str,300)),('errfile',(np.str,300)),
                            ('apred_vers',(np.str,20)),('instrument',(np.str,20)),
                            ('telescope',(np.str,10)),('platetype',(np.str,50)),('mjd',int),('plate',int),
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
        if platetype=='normal':
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
            base = load.filename('Plate',plate=plate,mjd=mjd,chips=True)
            chfiles = [base.replace('Plate-','Plate-'+ch+'-') for ch in ['a','b','c']]
            exists = [os.path.exists(chf) for chf in chfiles]
            if np.sum(exists)==3:
                chkap1['applate_success'] = True
            # apVisit
            base = load.filename('Visit',plate=plate,mjd=mjd,fiber=1) 
            visitfiles = glob(base.replace('-001.fits','-???.fits'))
            nvisitfiles = len(visitfiles)
            chkap1['apvisit_nobj_success']  = nvisitfiles
            chkap1['apvisit_success'] = nvisitfiles==chkap1['nobj']
            apvisitsumfile = load.filename('VisitSum',plate=plate,mjd=mjd)
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
    dtype = np.dtype([('apogee_id',(np.str,50)),('apred_vers',(np.str,20)),('telescope',(np.str,10)),
                      ('healpix',int),('nvisits',int),('pbskey',(np.str,50)),
                      ('file',(np.str,300)),('checktime',(np.str,100)),('success',bool)])
    chkrv = np.zeros(nstars,dtype=dtype)
    chkrv['apred_vers'] = apred_vers
    chkrv['telescope'] = telescope
    chkrv['pbskey'] = pbskey
    for i,visit in enumerate(visits):
        starfilenover = load.filename('Star',obj=visit['apogee_id'])
        # add version number, should be MJD of of the latest visit
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
        chkrv['checktime'][i] = str(datetime.now())
        chkrv['success'][i] = os.path.exists(starfile)

        if verbose:
            logger.info('%5d %20s %8d %5d %9s' % (i+1,chkrv['apogee_id'][i],chkrv['healpix'][i],
                                                  chkrv['nvisits'][i],chkrv['success'][i]))
    success, = np.where(chkrv['success']==True)
    logger.info('%d/%d succeeded' % (len(success),nstars))
    
    # Inset into the database
    db.ingest('rv_status',chkrv)
    db.close()        

    return chkrv


def create_sumfiles(mjd5,apred,telescope,logger=None):
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
    allstar = db.query(sql="select * from apogee_drp.star where (apogee_id, apred_vers, telescope, starver) in "+\
                       "(select apogee_id, apred_vers, telescope, max(starver) from apogee_drp.star where "+\
                       "apred_vers='"+apred+"' and telescope='"+telescope+"' group by apogee_id, apred_vers, telescope)")
    allstarfile = load.filename('allStar').replace('.fits','-'+telescope+'.fits')
    logger.info('Writing allStar file to '+allstarfile)
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
    cols = ','.join('v.'+np.char.array(vcols)) +','+ ','.join('rv.'+np.char.array(rvcols))
    allvisit = db.query(sql="select "+cols+" from apogee_drp.rv_visit as rv join apogee_drp.visit as v on rv.visit_pk=v.pk "+\
                        "where (rv.apogee_id, rv.apred_vers, rv.telescope, rv.starver) in "+\
                        "(select apogee_id, apred_vers, telescope, max(starver) from apogee_drp.rv_visit where "+\
                        "rv.apred_vers='"+apred+"' and rv.telescope='"+telescope+"' group by apogee_id, apred_vers, telescope)")
    allvisitfile = load.filename('allVisit').replace('.fits','-'+telescope+'.fits')
    logger.info('Writing allVisit file to '+allvisitfile)
    if os.path.exists(os.path.dirname(allvisitfile))==False:
        os.makedirs(os.path.dirname(allvisitfile))
    Table(allvisit).write(allvisitfile,overwrite=True)

    # Nightly allVisit and allStar, allVisitMJD/allStarMJD
    gdstar, = np.where(allstar['starver']==str(mjd5))
    allstarmjd = allstar[gdstar]
    gdvisit, = np.where(allvisit['mjd']==int(mjd5))
    allvisitmjd = allvisit[gdvisit]

    # maybe in summary/MJD/ or qa/MJD/ ?
    #allstarmjdfile = load.filename('allStarMJD')
    allstarmjdfile = allstarfile.replace('allStar','allStarMJD').replace('.fits','-'+str(mjd5)+'.fits')
    mjdsumdir = os.path.dirname(allstarmjdfile)+'/'+str(mjd5)
    allstarmjdfile = mjdsumdir+'/'+os.path.basename(allstarmjdfile)
    if os.path.exists(mjdsumdir)==False:
        os.makedirs(mjdsumdir)
    logger.info('Writing Nightly allStarMJD file to '+allstarmjdfile)

    Table(allstarmjd).write(allstarmjdfile,overwrite=True)
    allvisitmjdfile = allvisitfile.replace('allVisit','allVisitMJD').replace('.fits','-'+str(mjd5)+'.fits')
    allvisitmjdfile = mjdsumdir+'/'+os.path.basename(allvisitmjdfile)
    logger.info('Writing Nightly allVisitMJD file to '+allvisitmjdfile)
    Table(allvisitmjd).write(allvisitmjdfile,overwrite=True)

    db.close()

def summary_email(observatory,mjd5,chkexp,chkvisit,chkrv,logfiles=None):
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


def run_daily(observatory,mjd5=None,apred=None,qos='sdss-fast'):
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

    # No version input, use 'daily'
    if apred is None:
        apred = 'daily'

    load = apload.ApLoad(apred=apred,telescope=telescope)

    # Daily reduction logs directory
    logdir = os.environ['APOGEE_REDUX']+'/'+apred+'/log/'+observatory+'/'
    if os.path.exists(logdir)==False:
        os.makedirs(logdir)

    # What MJD5 are we doing?
    if mjd5 is None:
        # Could get information on which MJDs were processed from database
        # or from $APOGEE_REDUX/daily/log/apo/MJD5.done
        mjd5 = nextmjd5()

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
                     qos=qos, shared=shared, walltime=walltime, mem_per_cpu=4000, notification=False)
        for pf in planfiles:
            queue.append('apred {0}'.format(pf), outfile=pf.replace('.yaml','_pbs.log'), errfile=pf.replace('.yaml','_pbs.err'))
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
        queue.create(label='rv', nodes=nodes, alloc=alloc, ppn=ppn, cpus=cpus, qos=qos, shared=shared, walltime=walltime, notification=False)
        # Get unique stars
        objects,ui = np.unique(vcat['apogee_id'],return_index=True)
        vcat = vcat[ui]
        for obj in vcat['apogee_id']:
            apstarfile = load.filename('Star',obj=obj)
            outdir = os.path.dirname(apstarfile)  # make sure the output directories exist
            if os.path.exists(outdir)==False:
                os.makedirs(outdir)
            # Run with --verbose and --clobber set
            queue.append('rv %s %s %s -c -v' % (obj,apred,telescope),outfile=apstarfile.replace('.fits','_pbs.log'),
                         errfile=apstarfile.replace('.fits','_pbs.err'))
        queue.commit(hard=True,submit=True)
        rootLogger.info('PBS key is '+queue.key)        
        queue_wait(queue,sleeptime=120,verbose=True,logger=rootLogger)  # wait for jobs to complete
        chkrv = check_rv(vcat,queue.key)
        del queue
    else:
        rootLogger.info('No visit files for MJD=%d' % mjd5)
        chkrv = None

    # Run QA script
    #--------------
    rootLogger.info('')
    rootLogger.info('------------')
    rootLogger.info('Running QA')
    rootLogger.info('============')
    rootLogger.info('')
    queue = pbsqueue(verbose=True)
    queue.create(label='qa', nodes=1, alloc=alloc, ppn=ppn, qos=qos, cpus=1, shared=shared, walltime=walltime, notification=False)
    qaoutfile = os.environ['APOGEE_REDUX']+'/'+apred+'/log/'+observatory+'/'+str(mjd5)+'-qa.log'
    qaerrfile = qaoutfile.replace('-qa.log','-qa.err')
    if os.path.exists(os.path.dirname(qaoutfile))==False:
        os.makedirs(os.path.dirname(qaoutfile))
    queue.append('apqa {0} {1}'.format(mjd5,observatory),outfile=qaoutfile, errfile=qaerrfile)
    queue.commit(hard=True,submit=True)
    queue_wait(queue)  # wait for jobs to complete
    del queue

    # Create daily and full allVisit/allStar files
    create_sumfiles(mjd5,apred,telescope)

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
