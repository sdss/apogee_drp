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

        # Science exposures
        if platetype=='normal':

            # Load the plugmap information
            plugmap = platedata.getdata(plate,mjd,apred_vers,telescope,plugid=planstr['plugmap'])
            fiberdata = plugmap['fiberdata']

            # Exposure-level processing: ap3d, ap2d, apcframe
            dtype = np.dtype([('exposure_pk',int),('planfile',(np.str,300)),('apred_vers',(np.str,20)),('instrument',(np.str,20)),
                                ('telescope',(np.str,10)),('platetype',(np.str,50)),('mjd',int),('plate',int),
                                ('proctype',(np.str,30)),('pbskey',(np.str,50)),('checktime',(np.str,100)),
                                ('num',int),('success',bool)])
            chkexp = np.zeros(nexp*3,dtype=dtype3d)
            chkexp['planfile'] = pfile
            chkexp['apred_vers'] = apred_vers
            chkexp['instrument'] = instrument
            chkexp['telescope'] = telescope
            chkexp['platetype'] = platetype
            chkexp['mjd'] = mjd
            chkexp['plate'] = plate
            chkexp['proctype'] = 'AP3D'
            chkexp['pbskey'] = pbskey
            chkexp['success'] = False
            cnt = 0
            for num in expstr['name']:
                ind, = np.where(expinfo['num']==num)
                exposure_pk = expinfo['pk'][ind[0]]
                # AP3D
                #-----
                chkexp['exposure_pk'][cnt] = exposure_pk
                chkexp['num'][cnt] = num
                chkexp['proctype'][cnt] = 'AP3D'
                base = load.filename('2D',num=num,mjd=mjd,chips=True)
                chfiles = [base.replace('2D-','2D-'+ch+'-') for ch in ['a','b','c']]
                exists = [os.path.exists(chf) for chf in chfiles]
                chkexp['checktime'][cnt] = str(datetime.now())
                if np.sum(exists)==3:
                    chkexp['success'][cnt] = True
                cnt += 1
                # AP2D
                #-----
                chkexp['exposure_pk'][cnt] = exposure_pk
                chkexp['num'][cnt] = num
                chkexp['proctype'][cnt] = 'AP2D'
                base = load.filename('1D',num=num,mjd=mjd,chips=True)
                chfiles = [base.replace('1D-','1D-'+ch+'-') for ch in ['a','b','c']]
                exists = [os.path.exists(chf) for chf in chfiles]
                chkexp['checktime'][cnt] = str(datetime.now())
                if np.sum(exists)==3:
                    chkexp['success'][cnt] = True
                cnt += 1
                # APCframe
                #---------
                chkexp['exposure_pk'][cnt] = exposure_pk
                chkexp['num'][cnt] = num
                chkexp['proctype'][cnt] = 'APCFRAME'
                base = load.filename('Cframe',num=num,mjd=mjd,chips=True)
                chfiles = [base.replace('Cframe-','Cframe-'+ch+'-') for ch in ['a','b','c']]
                exists = [os.path.exists(chf) for chf in chfiles]
                chkexp['checktime'][cnt] = str(datetime.now())
                if np.sum(exists)==3:
                    chkexp['success'][cnt] = True
                cnt += 1


            # AP1DVISIT
            # ---------
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
                for num in expstr['name']:
                    ind, = np.where(expinfo['num']==num)
                    ind3d, = np.where((chkexp['num']==num) & (chkexp['proctype']=='AP3D'))
                    ind2d, = np.where((chkexp['num']==num) & (chkexp['proctype']=='AP2D'))
                    indcf, = np.where((chkexp['num']==num) & (chkexp['proctype']=='APCFRAME'))
                    logger.info('%2d %10d %4d %6s %6s %6s' % (i+1,chkexp['num'][ind3d],expinfo['nread'][ind],
                                                              chkexp['success'][ind3d],chkexp['success'][ind2d],
                                                              chkexp['success'][indcf]))
                logger.info('apPlate files: %s ' % chkap['applate_success'][0])
                logger.info('N apVisit files: %d ' % chkap['apvisit_nobj_success'][0])
                logger.info('apVisitSum file: %s ' % chkap['apvisitsum_success'][0])


            import pdb; pdb.set_trace()
                
            # Load into the database
            db = apogeedb.DBSession()
            db.ingest('exposure_status',chkexp)
            db.ingest('apred_status',chkap)
            db.close()


            import pdb; pdb.set_trace()

        # Calibration exposures
        else:
            logger.info('calibration exposures')

    import pdb; pdb.set_trace()

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

    load = apload.ApLoad(apred,telescope)

    # Start db session
    db = apogeedb.DBSession()

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
    allstarfile = load.filename('allStar')
    logger.info('Writing allStar file to '+allstarfile)
    if os.path.exists(os.path.dirname(allstarfile))==False:
        os.makedirs(os.path.dirname(allstarfile))
    Table(allstar).write(allstarfile,overwrite=True)

    # allVisit
    # Same thing for visit except that we'll get the multiple visit rows returned for each unique star row
    #   Get more info by joining with the visit table.
    vcols = ['apogee_id', 'target_id', 'apred_vers','file', 'uri', 'fiberid', 'plate', 'mjd', 'telescope', 'survey',
             'field', 'programname', 'alt_id', 'location_id', 'ra', 'dec', 'glon', 'glat', 'j', 'j_err', 'h',
             'h_err', 'k', 'k_err', 'src_h', 'wash_m', 'wash_m_err', 'wash_t2', 'wash_t2_err', 'ddo51',
             'ddo51_err', 'irac_3_6', 'irac_3_6_err', 'irac_4_5', 'irac_4_5_err', 'irac_5_8', 'irac_5_8_err',
             'irac_8_0', 'irac_8_0_err', 'wise_4_5', 'wise_4_5_err', 'targ_4_5', 'targ_4_5_err',
             'wash_ddo51_giant_flag', 'wash_ddo51_star_flag', 'pmra', 'pmdec', 'pm_src', 'ak_targ',
             'ak_targ_method', 'ak_wise', 'sfd_ebv', 'apogee_target1', 'apogee_target2', 'apogee_target3',
             'apogee_target4', 'targflags', 'snr', 'starflag', 'starflags','dateobs','jd']
    rvcols = ['starver', 'bc', 'vtype', 'vrel', 'vrelerr', 'vhelio', 'chisq', 'rv_teff', 'rv_feh',
              'rv_logg', 'xcorr_vrel', 'xcorr_vrelerr', 'xcorr_vhelio', 'n_components', 'rv_components']
    cols = ','.join('v.'+np.char.array(vcols)) +','+ ','.join('rv.'+np.char.array(rvcols))
    allvisit = db.query(sql="select "+cols+" from apogee_drp.rv_visit as rv join apogee_drp.visit as v on rv.visit_pk=v.pk "+\
                        "where (rv.apogee_id, rv.apred_vers, rv.telescope, rv.starver) in "+\
                        "(select apogee_id, apred_vers, telescope, max(starver) from apogee_drp.rv_visit where "+\
                        "rv.apred_vers='"+apred+"' and rv.telescope='"+telescope+"' group by apogee_id, apred_vers, telescope)")
    allvisitfile = load.filename('allVisit')
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
    allstarmjdfile = load.filename('allStarMJD')
    logger.info('Writing Nightly allStarMJD file to '+allstarmjdfile)
    if os.path.exists(os.path.dirname(allstarmjdfile))==False:
        os.makedirs(os.path.dirname(allstarmjdfile))
    Table(allstarmjd).write(allstarmjdfile,overwrite=True)
    allvisitmjdfile = load.filename('allVisitMJD')
    logger.info('Writing Nightly allVisitMJD file to '+allvisitmjdfile)
    if os.path.exists(os.path.dirname(allvisitmjdfile))==False:
        os.makedirs(os.path.dirname(allvisitmjdfile))
    Table(allvisitmjd).write(allvisitmjdfile,overwrite=True)

    import pdb; pdb.set_trace()

    db.close()

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
    db.ingest('exposure',expinfo)  # insert into database
    expinfo0 = expinfo.copy()
    expinfo = db.query('exposure',where="mjd=%d and observatory='%s'" % (mjd5,observatory))

    # Make MJD5 and plan files
    rootLogger.info('Making plan files')
    plandicts,planfiles = mkplan.make_mjd5_yaml(mjd5,apred,telescope,clobber=True,logger=rootLogger)
    dbload_plans(planfiles)  # load plans into db
    #db.ingest('plan',planfiles)  # load plans into db
    dailyplanfile = os.environ['APOGEEREDUCEPLAN_DIR']+'/yaml/'+telescope+'/'+telescope+'_'+str(mjd5)+'auto.yaml'
    planfiles = mkplan.run_mjd5_yaml(dailyplanfile,logger=rootLogger)
    # Write planfiles to MJD5.plans
    dln.writelines(dailydir+str(mjd5)+'.plans',[os.path.basename(pf) for pf in planfiles])

    check_apred(expinfo,planfiles,'dummy',verbose=True,logger=rootLogger)

    import pdb;pdb.set_trace()

    # Run APRED on all planfiles using "pbs" package
    rootLogger.info('')
    rootLogger.info('--------------')
    rootLogger.info('Running APRED')
    rootLogger.info('==============')
    rootLogger.info('')
    queue = pbsqueue(verbose=True)
    cpus = np.minimum(len(planfiles),30)
    queue.create(label='apred', nodes=2, ppn=16, cpus=cpus, alloc='sdss-kp', qos=True, umask='002', walltime='240:00:00')
    for pf in planfiles:
        queue.append('apred {0}'.format(pf), outfile=pf.replace('.yaml','_pbs.log'), errfile=pf.replace('.yaml','_pbs.err'))
    import pdb;pdb.set_trace()
    queue.commit(hard=True,submit=True)
    queue_wait(queue,sleeptime=120,verbose=True,logger=rootLogger)  # wait for jobs to complete
    chkexp,chkvisit = check_apred(expinfo,planfiles,queue.key)
    del queue

    import pdb;pdb.set_trace()

    # Run "rv" on all stars
    rootLogger.info('')
    rootLogger.info('------------------------------')
    rootLogger.info('Running RV+Visit Combination')
    rootLogger.info('==============================')
    rootLogger.info('')
    vcat = db.query('visit',cols='*',where='MJD=%d'%mjd5)
    if len(vcat)>0:
        queue = pbsqueue(verbose=True)
        cpus = np.minimum(len(vcat),30)
        queue.create(label='rv', nodes=2, ppn=16, cpus=cpus, alloc='sdss-kp', qos=True, umask='002', walltime='240:00:00')
        for obj in vcat['apogee_id']:
            apstarfile = load.filename('Star',obj=obj)
            outdir = os.path.dirname(apstarfile)  # make sure the output directories exist
            if os.path.exists(outdir)==False:
                os.makedirs(outdir)
            queue.append('rv %s %s %s' % (obj,apred,telescope),outfile=apstarfile.replace('.fits','_pbs.log'),
                         errfile=apstarfile.replace('.fits','_pbs.err'))
        queue.commit(hard=True,submit=True)
        queue_wait(queue,sleeptime=120,verbose=True,logger=rootLogger)  # wait for jobs to complete
        chkrv = check_rv(vcat,queue.key)
        del queue
    else:
        rootLogger.info('No visit files for MJD=%d' % mjd5)

    import pdb;pdb.set_trace()

    # Run QA script
    queue = pbsqueue(verbose=True)
    queue.create(label='qa', nodes=1, ppn=16, cpus=1, alloc='sdss-kp', qos=True, umask='002', walltime='240:00:00')
    queue.append('apqa {0}'.format(mjd5))
    queue.commit(hard=True,submit=True)
    queue_wait(queue)  # wait for jobs to complete
    del queue

    # Create daily and full allVisit/allStar files
    create_sumfiles(mjd5,apred,telescope)

    import pdb;pdb.set_trace()

    rootLogger.info('Daily APOGEE reduction finished for MJD=%d and observatory=%s' % (mjd5,observatory))

    db.close()    # close db session
