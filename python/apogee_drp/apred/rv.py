# Radial velocity and visit combination code

import os
import copy
import glob
import pdb
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import esutil
import pickle
import yaml
from astropy.io import fits
from ..utils import apload, applot, bitmask, spectra, norm, yanny
from ..utils.apspec import ApSpec
from ..database import apogeedb
from holtztools import plots, html, match, struct
from dlnpyutils import utils as dln
from scipy import interpolate
from scipy.signal import correlate
from scipy.ndimage.filters import median_filter, gaussian_filter
import doppler 
import multiprocessing as mp
from astropy.table import Table, Column
from apogee_drp.apred import bc


colors = ['r','g','b','c','m','y','k']
chips = ['a','b','c']


def doppler_rv(star,apred,telescope,nres=[5,4.25,3.5],windows=None,tweak=False,
               clobber=False,verbose=False,plot=False,logger=None):
    """
    Run Doppler on one star and perform visit combination.

    Parameters
    ----------
    star : str
       The '2M' star name.
    apred : str
       APOGEE reduction version.
    telescope : str
       APOGEE telescope (apo25m, loc25m, apo1m).
    nres : array, optional
       Array of sinc widths.  Default is nres=[5,4.25,3.5].
    windows : array, optional
       Array of spectral windows to use.
    tweak : bool, optional
       Have Doppler tweak the continuum with the best-fit template.  Default is False.
    clobber : bool, optional
       Overwrite any existing files (both RV and visit combined).
    verbose : bool, optional
       Verbose output to the screen.
    plot : bool, optional
       Make a plot of spectrum and best-fitting template.
    logger : logger, optional
       A logger for printed output.

    Returns
    -------
    The program outputs Doppler files an apStar combined file to the appropriate
    place specified in the SDSS/APOGEE tree product.

    """

    if logger is None:
        logger = dln.basiclogger()

    snmin = 3
    apstar_vers = 'stars'

    logger.info('Running Doppler and performing visit combination for %s and telescope=%s' % (star,telescope))

    # Get the visit files for this star and telescope
    db = apogeedb.DBSession()
    allvisits = db.query('visit',cols='*',where="apogee_id='"+star+"' and telescope='"+telescope+"' and apred_vers='"+apred+"'")
    db.close()
    nallvisits = len(allvisits)
    if nallvisits==0:
        logger.info('No visit files found')
        return
    logger.info('%d visit file(s) found' % nallvisits)
    allvisits = Table(allvisits)
    # Change datatype of STARFLAG to 64-bit
    allvisits['starflag'] = allvisits['starflag'].astype(np.uint64)

    # Get the star version number
    #  this is the largest MJD5 in the FULL list of visits
    starver = str(np.max(allvisits['mjd'].astype(int)))
    logger.info('Version='+starver)


    # Output directory
    load = apload.ApLoad(apred=apred,telescope=telescope)
    starfile = load.filename('Star',obj=star)
    stardir = os.path.dirname(starfile)
    try : os.makedirs(os.path.dirname(stardir))
    except FileExistsError: pass

    # Initalize star-level summary table
    startype = np.dtype([('apogee_id','U30'),('file','U100'),('uri','U300'),('starver','U50'),('mjdbeg',int),('mjdend',int),
                         ('telescope','U6'),('apred_vers','U50'),('healpix',int),('snr',float),
                         ('ra',float),('dec',float),('glon',float),('glat',float),
                         ('jmag',float),('jerr',float),('hmag',float),('herr',float),('kmag',float),('kerr',float),
                         ('src_h','U16'),('targ_pmra',float),('targ_pmdec',float),('targ_pm_src','U16'),
                         ('apogee_target1',int),('apogee_target2',int),
                         ('apogee2_target1',int),('apogee2_target2',int),('apogee2_target3',int),('apogee2_target4',int),
                         ('catalogid',int),('gaiadr2_sourceid',int),('gaiadr2_plx',float),('gaiadr2_plx_error',float),
                         ('gaiadr2_pmra',float),('gaiadr2_pmra_error',float),('gaiadr2_pmdec',float),('gaiadr2_pmdec_error',float),
                         ('gaiadr2_gmag',float),('gaiadr2_gerr',float),('gaiadr2_bpmag',float),('gaiadr2_bperr',float),
                         ('gaiadr2_rpmag',float),('gaiadr2_rperr',float),('sdssv_apogee_target0',int),('firstcarton','U50'),
                         ('targflags','U132'),('nvisits',int),('ngoodvisits',int),('ngoodrvs',int),
                         ('starflag',np.uint64),('starflags','U132'),('andflag',np.uint64),('andflags','U132'),
                         ('vheliobary',float),('vscatter',float),('verr',float),('vmederr',float),('chisq',float),
                         ('rv_teff',float),('rv_tefferr',float),('rv_logg',float),('rv_loggerr',float),('rv_feh',float),
                         ('rv_feherr',float),('rv_ccpfwhm',float),('rv_autofwhm',float),
                         ('n_components',int),('meanfib',float),('sigfib',float)
                     ])
    startab = np.zeros(1,dtype=startype)
    startab['apogee_id'] = star
    startab['telescope'] = telescope
    startab['starver'] = starver
    startab['apred_vers'] = apred
    startab['mjdbeg'] = np.min(allvisits['mjd'].astype(int))
    startab['mjdend'] = np.max(allvisits['mjd'].astype(int))    
    startab['healpix'] = apload.obj2healpix(star)
    startab['nvisits'] = nallvisits
    # Copy data from visit
    tocopy = ['ra','dec','glon','glat','jmag','jerr','hmag','herr','kmag','kerr','src_h','catalogid',
              'gaiadr2_sourceid','gaiadr2_plx','gaiadr2_plx_error','gaiadr2_pmra','gaiadr2_pmra_error',
              'gaiadr2_pmdec','gaiadr2_pmdec_error','gaiadr2_gmag','gaiadr2_gerr','gaiadr2_bpmag',
              'gaiadr2_bperr','gaiadr2_rpmag','gaiadr2_rperr','sdssv_apogee_target0','firstcarton',
              'targflags']
    for c in tocopy:
        startab[c] = allvisits[c][0]
    startab['targ_pmra'] = allvisits['pmra'][0]
    startab['targ_pmdec'] = allvisits['pmdec'][0]
    startab['targ_pm_src'] = allvisits['pm_src'][0]
    # Initialize some parameters in case RV fails
    startab['ngoodvisits'] = 0
    startab['ngoodrvs'] = 0
    startab['vheliobary'] = np.nan
    startab['vscatter'] = np.nan
    startab['verr'] = np.nan
    startab['vmederr'] = np.nan
    startab['rv_teff'] = np.nan
    startab['rv_logg'] = np.nan
    startab['rv_feh'] = np.nan
    startab['rv_ccpfwhm'] = np.nan
    startab['rv_autofwhm'] = np.nan
    startab['n_components'] = -1
    if len(allvisits) > 0: meanfib=(allvisits['fiberid']*allvisits['snr']).sum()/allvisits['snr'].sum()
    else: meanfib = 999999.
    if len(allvisits) > 1: sigfib=allvisits['fiberid'].std(ddof=1)
    else: sigfib = 0.
    startab['meanfib'] = meanfib
    startab['sigfib'] = sigfib
    starmask = bitmask.StarBitMask()
    
    # Select good visit spectra
    gd, = np.where(((allvisits['starflag'] & starmask.badval()) == 0) &
                   (allvisits['snr'] > snmin) )
    # No good visits, but still write to star table
    if len(gd)==0:
        logger.info('No visits passed QA cuts')
        # Add starflag and andflag
        starflag,andflag = np.uint64(0),np.uint64(0)
        for v in len(allvisits):
            starflag |= v['starflag'] # bitwise OR
            andflag &= v['starflag']  # bitwise AND
        starflag |= starmask.getval('RV_FAIL')
        andflag |= starmask.getval('RV_FAIL')
        startab['starflag'] = starflag
        startab['andflag'] = andflag
        # Load star summary information into database
        dbingest(startab,None)
        return
    logger.info('%d visit(s) passed QA cuts' % len(gd))
    
    # Initialize STARVISITS which will hold all visit-level information
    #   for visits that passed the QA cuts
    starvisits = allvisits[gd].copy()
    nvisits = len(gd)
    del starvisits['created']
    startab['ngoodvisits'] = nvisits   # visits that pass QA cuts
    # Add STARVER                                                                                                                                             
    starvisits['starver'] = starver
    # Flag all visits as RV_FAIL to start with, will remove if they worked okay
    starvisits['starflag'] |= starmask.getval('RV_FAIL')
    # Initialize visit RV tags
    for col in ['vtype','vrel','vrelerr','vheliobary','bc','chisq','rv_teff','rv_tefferr','rv_logg','rv_loggerr','rv_feh','rv_feherr']:
        if col == 'vtype':
            starvisits[col] = 0
        else:
            starvisits[col] = np.nan
    for col in ['xcorr_vrel','xcorr_vrelerr','xcorr_vheliobary','bc']:
        starvisits[col] = np.nan

    # Add columns for RV components
    starvisits['n_components'] = -1
    rv_components = Column(name='rv_components',dtype=float,shape=(3),length=len(starvisits))
    starvisits.add_column(rv_components)
    rvtab = Column(name='rvtab',dtype=Table,length=len(starvisits))
    starvisits.add_column(rvtab)


    # Run Doppler with dorv() on the good visits
    try:
        dopsumstr,dopvisitstr,gaussout = dorv(starvisits,starver,clobber=clobber,verbose=verbose,tweak=tweak,
                                              plot=plot,windows=windows,apstar_vers=apstar_vers,logger=logger)
        logger.info('Doppler completed successfully for {:s}'.format(star))
    except:
        logger.info('Doppler failed for {:s}'.format(star))
        raise


    # Now load the the Doppler the results
    visits = []
    ncomponents = 0
    for i,(v,g) in enumerate(zip(dopvisitstr,gaussout)) :
        # Match by filename components in case there was an error reading in doppler
        name = os.path.basename(v['filename'])
        if telescope == 'apo1m':
            vind, = np.where( np.char.strip(starvisits['file']).astype(str) == os.path.basename(v['filename'].strip()) )
            if len(vind) == 0:
                # special case for incremental release...yuck
                vind, = np.where( np.char.strip(starvisits['file']).astype(str) == 
                                   os.path.basename(v['filename'].strip()).replace('-r13-','-r12-') )
        else:
            vind, = np.where( starvisits['file']==name )
        if len(vind) > 0:
            vind = vind[0]
        else:
            continue
        visits.append(vind)
        # Remove RV_FAIL that we added above
        starvisits['starflag'][vind] &= ~starmask.getval('RV_FAIL')
        # Add Doppler outputs
        starvisits['vrel'][vind] = v['vrel']
        starvisits['vrelerr'][vind] = v['vrelerr']
        starvisits['vheliobary'][vind] = v['vhelio']
        starvisits['xcorr_vrel'][vind] = v['xcorr_vrel']
        starvisits['xcorr_vrelerr'][vind] = v['xcorr_vrelerr']
        starvisits['xcorr_vheliobary'][vind] = v['xcorr_vhelio']
        starvisits['bc'][vind] = v['bc']
        starvisits['chisq'][vind] = v['chisq']
        starvisits['rv_teff'][vind] = v['teff']
        starvisits['rv_tefferr'][vind] = v['tefferr']
        starvisits['rv_logg'][vind] = v['logg']
        starvisits['rv_loggerr'][vind] = v['loggerr']
        starvisits['rv_feh'][vind] = v['feh']
        starvisits['rv_feherr'][vind] = v['feherr']
        if g is None:
            starvisits['n_components'][vind] = 0
        else:
            starvisits['n_components'][vind] = g['N_components']
        if starvisits['n_components'][vind] > 1 :
            starvisits['starflag'][vind] |= starmask.getval('MULTIPLE_SUSPECT')
            n = len(g['best_fit_parameters'])//3
            gd, = np.where(np.array(g['best_fit_parameters'])[0:n] > 0)
            rv_comp = np.array(g['best_fit_parameters'])[2*n+gd]
            n_rv_comp = np.min([3,len(rv_comp)])
            starvisits[vind]['rv_components'][0:n_rv_comp] = rv_comp[0:n_rv_comp]
        starvisits['rvtab'][vind] = v
        # Flag visits with suspect RVs
        if starvisits['rv_teff'][vind] < 6000:
            bd_diff = 10
        else:
            bd_diff = 50.
        if (np.abs(starvisits['vheliobary'][vind]-starvisits['xcorr_vheliobary'][vind]) > bd_diff) :
            starvisits['starflag'][vind] |= starmask.getval('RV_REJECT')
        elif (np.abs(starvisits['vheliobary'][vind]-starvisits['xcorr_vheliobary'][vind]) > 0) :
            starvisits['starflag'][vind] |= starmask.getval('RV_SUSPECT')

    # Set STARFLAGS for the visits (successful and failed ones)
    for i in range(len(starvisits)):
        starvisits['starflags'][i] = starmask.getname(starvisits['starflag'][i])


    # Compute final star-level values
    #--------------------------------

    # Targeting flags
    #  don't have apogee-1/2 targeting flags implemented yet
    #startab['targflags'] = (bitmask.targflags(apogee_target1,apogee_target2,0,0,survey='apogee')+
    #                        bitmask.targflags(apogee2_target1,apogee2_target2,apogee2_target3,apogee2_target4,survey='apogee2')+
    #                        bitmask.targflags(sdssv_apogee_target0,survey='sdss5'))
    startab['targflags'] = bitmask.targflags(starvisits['sdssv_apogee_target0'][0],0,0,0,survey='sdss5')

    # Make final STARFLAG and ANDFLAG
    starflag = startab['starflag']
    andflag = startab['andflag']
    for v in starvisits:
        starflag |= v['starflag']
        andflag &= v['starflag']
    startab['starflags'] = starmask.getname(startab['starflag'])
    startab['andflags'] = starmask.getname(startab['andflag'])

    # Initialize meanfib/sigfib using all good visits
    if len(starvisits) > 1:
        meanfib = (starvisits['fiberid']*starvisits['snr']).sum()/starvisits['snr'].sum()
        sigfib = starvisits['fiberid'].std(ddof=1)
    else:
        meanfib = starvisits['fiberid'][0]
        sigfib = 0.0
    startab['meanfib'] = meanfib
    startab['sigfib'] = sigfib

    # Average Doppler values for this star
    if len(visits)>0:
        visits = np.array(visits)
        gdrv, = np.where((starvisits['starflag'][visits] & starmask.getval('RV_REJECT')) == 0)
        ngdrv = len(gdrv)
        if ngdrv>0:
            startab['ngoodrvs'] = ngdrv
            try: startab['n_components'] = starvisits['n_components'][gdrv].max()
            except: pass
            startab['vheliobary'] = (starvisits['vheliobary'][gdrv]*starvisits['snr'][gdrv]).sum() / starvisits['snr'][gdrv].sum()
            if ngdrv>1:
                startab['vscatter'] = starvisits['vheliobary'][gdrv].std(ddof=1)
                startab['verr'] = startab['vscatter'][0]/np.sqrt(ngdrv)
                startab['vmederr'] = np.median(starvisits['vrelerr'])
            else:
                startab['vscatter'] = 0.0
                startab['verr'] = starvisits['vrelerr'][gdrv][0]
                startab['vmederr'] = starvisits['vrelerr'][0]
            startab['chisq'] = dopsumstr['chisq']
            startab['rv_teff'] = dopsumstr['teff']
            startab['rv_tefferr'] = dopsumstr['tefferr']
            startab['rv_logg'] = dopsumstr['logg']
            startab['rv_loggerr'] = dopsumstr['loggerr']
            startab['rv_feh'] = dopsumstr['feh']
            startab['rv_feherr'] = dopsumstr['feherr']
            # Update meanfib/sigfig only using visits with good RVs
            if ngdrv > 1:
                meanfib = (starvisits['fiberid'][gdrv]*starvisits['snr'][gdrv]).sum()/starvisits['snr'][gdrv].sum()
                sigfib = starvisits['fiberid'][gdrv].std(ddof=1)
            else:
                meanfib = starvisits['fiberid'][gdrv][0]
                sigfib = 0.0
            startab['meanfib'] = meanfib
            startab['sigfib'] = sigfib
            # Get filename and URI
            outfilenover = load.filename('Star',obj=star)
            outbase = os.path.splitext(os.path.basename(outfilenover))[0]
            outbase += '-'+starver   # add star version
            outdir = os.path.dirname(outfilenover)
            outfile = outdir+'/'+outbase+'.fits'
            if apstar_vers != 'stars' :
                outfile = outfile.replace('/stars/','/'+apstar_vers+'/')
            startab['file'] = os.path.basename(outfile)
            mwm_root = os.environ['MWM_ROOT']
            startab['uri'] = outfile[len(mwm_root)+1:]

    else:
        gdrv = []

    # Load information into the database
    dbingest(startab,starvisits)

    # Do the visit combination and write out apStar file
    if len(gdrv)>0:
        apstar = visitcomb(starvisits[visits[gdrv]],starver,load=load,
                           apstar_vers=apstar_vers,apred=apred,nres=nres,logger=logger)
    else:
        logger.info('No good visits for '+star)

    return


def dorv(allvisit,starver,obj=None,telescope=None,apred=None,clobber=False,verbose=False,tweak=False,
         plot=False,windows=None,apstar_vers='stars',logger=None):
    """ Do the Doppler rv jointfit from list of files
    """

    if logger is None:
        logger = dln.basiclogger()

    if tweak==True:
        suffix = '_tweak'
    else:
        suffix = '_out'
    if obj is None:
        obj = str(allvisit['apogee_id'][0])
    if type(obj) is not str:
        obj = obj.decode('UTF-8')
    if apred is None:
        apred = str(allvisit['apred_vers'][0])
    if telescope is None:
        telescope = str(allvisit['telescope'][0])
    load = apload.ApLoad(apred=apred,telescope=telescope)
    outfile = load.filename('Star',obj=obj)
    outdir = os.path.dirname(outfile)
    outbase = os.path.splitext(os.path.basename(outfile))[0]
    outbase += '-'+starver  # add star version
    if os.path.exists(outdir)==False:
        os.makedirs(outdir)
    if apstar_vers != 'stars':
        outdir = outdir.replace('/stars/','/'+apstar_vers+'/')

    if os.path.exists(outdir+'/'+outbase+suffix+'_doppler.pkl') and not clobber:
        logger.info(obj+' already done')
        fp = open(outdir+'/'+outbase+suffix+'_doppler.pkl','rb')
        try: 
            out = pickle.load(fp)
            sumstr,finalstr,bmodel,specmlist,gout = out
            fp.close()
            return sumstr,finalstr,gout
        except: 
            logger.warning('error loading: '+outbase+suffix+'_doppler.pkl')
            #pass

    speclist = []
    pixelmask = bitmask.PixelBitMask()
    badval = pixelmask.badval()|pixelmask.getval('SIG_SKYLINE')|pixelmask.getval('LITTROW_GHOST')
   
    # If we have a significant number of low S/N visits, combine first using
    #    barycentric correction only, use that to get an estimate of systemic
    #    velocity, then do RV determination restricting RVs to within 50 km/s
    #    of estimate. This seems to help significant for faint visits
    lowsnr_visits, = np.where(allvisit['snr']<10)
    if (len(lowsnr_visits) > 1) & (len(lowsnr_visits)/len(allvisit) > 0.1) :
        try :
            apstar_bc = visitcomb(allvisit,starver,bconly=True,load=load,write=False,dorvfit=False,apstar_vers=apstar_vers) 
            apstar_bc.setmask(badval)
            spec = doppler.Spec1D(apstar_bc.flux[0,:],err=apstar_bc.err[0,:],bitmask=apstar_bc.bitmask[0,:],
                                  mask=apstar_bc.mask[0,:],wave=apstar_bc.wave,lsfpars=np.array([0]),
                                  lsfsigma=apstar_bc.wave/22500/2.354,instrument='APOGEE',
                                  filename=apstar_bc.filename)
            logger.info('Lots of low-S/N visits. Running BC jointfit for :',obj)
            out = doppler.rv.jointfit([spec],verbose=verbose,plot=plot,tweak=tweak,maxvel=[-500,500])
            rvrange = [out[1][0]['vrel']-50, out[1][0]['vrel']+50]
        except :
            logger.info('  BC jointfit failed')
            rvrange = [-500,500]
    elif allvisit['hmag'].max() > 13.5 : 
        # If it's faint, restrict to +/- 500 km/s
        rvrange = [-500,500]
    else:
        # Otherwise, restrict to +/ 1000 km/s
        rvrange = [-1000,1000]

    # Loop over visits
    for i in range(len(allvisit)):

        # Load all of the visits into doppler Spec1D objects
        if load.telescope == 'apo1m' :
            visitfile = load.allfile('Visit',plate=allvisit['plate'][i],mjd=allvisit['mjd'][i],
                                     reduction=allvisit['apogee_id'][i],field=allvisit['field'][i])
        else :
            visitfile = load.allfile('Visit',plate=int(allvisit['plate'][i]),mjd=allvisit['mjd'][i],
                                     fiber=allvisit['fiberid'][i],field=allvisit['field'][i])
        spec = doppler.read(visitfile,badval=badval)

        if windows is not None :
            # If we have spectral windows to mask, do so here
            for ichip in range(3) :
                mask = np.full_like(spec.mask[:,ichip],True)
                gd = []
                for window in windows :
                    gd.extend(np.where((spec.wave[:,ichip] > window[0]) & (spec.wave[:,ichip] < window[1]))[0])
                mask[gd] = False
                spec.mask[:,ichip] |= mask
                 
        if spec is not None : speclist.append(spec)

    if len(speclist)==0:
        raise Exception('No visit spectra loaded')

    # Now do the Doppler jointfit to get RVs
    # Dump empty pickle to stand in case of failure (to prevent redo if not clobber)
    try:
        # Dump empty pickle to stand in case of failure (to prevent redo if not clobber)
        fp = open(outdir+'/'+outbase+suffix+'_doppler.pkl','wb')
        pickle.dump(None,fp)
        fp.close()
        logger.info('Running Doppler jointfit for: {:s}  rvrange:[{:.1f},{:.1f}]  nvisits: {:d}'.format(obj,*rvrange,len(speclist)))
        sumstr,finalstr,bmodel,specmlist,dt = doppler.rv.jointfit(speclist,maxvel=rvrange,verbose=verbose,
                                                                  plot=plot,saveplot=plot,outdir=outdir+'/',tweak=tweak)
        logger.info('Running CCF decomposition for: '+obj)
        gout = gauss_decomp(finalstr,phase='two',filt=True)
        fp = open(outdir+'/'+outbase+suffix+'_doppler.pkl','wb')
        pickle.dump([sumstr,finalstr,bmodel,specmlist,gout],fp)
        fp.close()
        # Making plots
        logger.info('Making plots for :'+obj+' '+outdir)
        try: os.makedirs(outdir+'/plots/')
        except: pass
        dop_plot(outdir+'/plots/',outbase,[sumstr,finalstr,bmodel,specmlist],decomp=gout)
    except KeyboardInterrupt: 
        raise
    except ValueError as err:
        logger.error('Exception raised in dorv for: '+obj)
        logger.error("ValueError: {0}".format(err))
        return
    except RuntimeError as err:
        logger.error('Exception raised in dorv for: '+obj)
        logger.error("Runtime error: {0}".format(err))
        return
    except :
        raise
        logger.error('Exception raised in dorv for: ', field, obj)
        return

    # Return summary RV info, visit RV info, decomp info 
    return sumstr,finalstr,gout



def gaussian(amp, fwhm, mean):
    """ Gaussian as defined by gausspy
    """
    return lambda x: amp * np.exp(-4. * np.log(2) * (x-mean)**2 / fwhm**2)


import gausspy.gp as gp

def gauss_decomp(out,phase='one',alpha1=0.5,alpha2=1.5,thresh=[4,4],plot=None,filt=False) :
    """ Do Gaussian decomposition of CCF using gausspy

        Parameters:
        out : list of dictionaries for each frame, giving x_ccf, ccf, and ccferr
        phase : gausspy paramater
        alpha1 : gausspy parameter
        alpha2 : gausspy parameter for second set of gaussians if phase=='two'
        thresh : gausspy parameter
        plot (str) : if not None, do plot and use as root file name for plot
        filt (bool) : if true, apply filtering to remove components judged to be insignificant
    """
    g = gp.GaussianDecomposer()
    g.set('phase',phase)
    g.set('SNR_thresh',thresh)
    g.set('alpha1',alpha1)
    g.set('alpha2',alpha2)
    gout=[]
    if plot is not None : fig,ax=plots.multi(1,len(out),hspace=0.001,figsize=(6,2+n))
    for i,final in enumerate(out) :
        gd, = np.where(np.isfinite(final['x_ccf']))
        x = final['x_ccf'][gd]
        y = final['ccf'][gd] 
        # high pass filter for better performance
        if filt : final['ccf'][gd]-= gaussian_filter(final['ccf'][gd],50,mode='nearest')
        try : 
            decomp=g.decompose(x,final['ccf'][gd],final['ccferr'][gd])
            n=decomp['N_components']
        except :
            print('Exception in Gaussian decomposition, setting to 0 components')
            n=0
            decomp=None
        if filt and n>0 :
            # remove components if they are within width of brighter component, or <0.25 peak ,
            #   or more than twice as wide, or if primary component is wide
            for j in range(1,n) :
                pars_j = decomp['best_fit_parameters'][j::n]
                for k in range(j) :
                    pars_k = decomp['best_fit_parameters'][k::n]
                    if (pars_j[0]>pars_k[0] and pars_k[0]>0 and 
                                (abs(pars_j[2]-pars_k[2])<abs(pars_j[1])  or 
                                 pars_k[0]<0.25*pars_j[0] or 
                                 abs(pars_j[1])>100 or
                                 np.abs(pars_k[1])>2*np.abs(pars_j[1]) ) ) :
                        decomp['best_fit_parameters'][k] = 0
                        decomp['N_components'] -= 1
                    elif (pars_k[0]>pars_j[0] and pars_j[0]>0 and
                                (abs(pars_j[2]-pars_k[2])<abs(pars_k[1]) or 
                                 pars_j[0]<0.25*pars_k[0] or 
                                 abs(pars_k[1])>100 or
                                 np.abs(pars_j[1])>2*np.abs(pars_k[1]) ) )  :
                        decomp['best_fit_parameters'][j] = 0
                        pars_j = decomp['best_fit_parameters'][j::n]
                        decomp['N_components'] -= 1
                  
        gout.append(decomp)
        if plot is not None:
            plots.plotl(ax[i],final['x_ccf'],final['ccf'])
            ax[i].plot(final['x_ccf'],final['ccferr'],color='r')
            for j in range(n) :
                pars=gout[i]['best_fit_parameters'][j::n]
                ax[i].plot(x,gaussian(*pars)(x))
                if pars[0] > 0 : color='k'
                else : color='r'
                ax[i].text(0.1,0.8-j*0.1,'{:8.1f}{:8.1f}{:8.1f}'.format(*pars),transform=ax[i].transAxes,color=color)
            fig.savefig(plot+'_ccf.png')
    del g
    return gout


def dop_plot(outdir,obj,dopout,decomp=None) :
    """ RV diagnostic plots
    """
    sumstr,finalstr,bmodel,specmlist = dopout

    matplotlib.use('Agg')
    n = len(bmodel)
    # Plot final spectra and final models
    # full spectrum
    fig,ax = plots.multi(1,n,hspace=0.001,figsize=(8,2+n))
    ax = np.atleast_1d(ax)
    # continuum
    figc,axc = plots.multi(1,n,hspace=0.001,figsize=(8,2+n))
    axc = np.atleast_1d(axc)
    # windows
    windows = [[15700,15780],[15850,16000],[16700,16930]]
    fig2,ax2 = plots.multi(len(windows),n,hspace=0.001,wspace=0.001,figsize=(12,2+n))
    ax2 = np.atleast_2d(ax2)

    # Loop over visits
    for i,(mod,spec) in enumerate(zip(bmodel,specmlist)) :
        ax[i].plot(spec.wave,spec.flux,color='k')
        for iorder in range(3) :
            gd, = np.where(~spec.mask[:,iorder])
            ax[i].plot(spec.wave[gd,iorder],spec.flux[gd,iorder],color='g')
        ax[i].plot(mod.wave,mod.flux,color='r')
        ax[i].text(0.1,0.1,'{:d}'.format(spec.head['MJD5']),transform=ax[i].transAxes)
        for iwind,wind in enumerate(windows) :
            ax2[i,iwind].plot(spec.wave,spec.flux,color='k')
            for iorder in range(3) :
                gd, = np.where(~spec.mask[:,iorder])
                ax2[i,iwind].plot(spec.wave[gd,iorder],spec.flux[gd,iorder],color='g')
            ax2[i,iwind].plot(mod.wave,mod.flux,color='r')
            ax2[i,iwind].set_xlim(wind[0],wind[1])
            ax2[i,iwind].set_ylim(0.5,1.3)
            if iwind == 0 : ax2[i,iwind].text(0.1,0.1,'{:d}'.format(spec.head['MJD5']),transform=ax2[i,0].transAxes)
        axc[i].plot(spec.wave,spec.flux*spec.cont,color='k')
        axc[i].plot(spec.wave,spec.cont,color='g')
        axc[i].text(0.1,0.1,'{:d}'.format(spec.head['MJD5']),transform=axc[i].transAxes)
    fig.savefig(outdir+'/'+obj+'_spec.png')
    plt.close()
    fig2.savefig(outdir+'/'+obj+'_spec2.png')
    plt.close()
    figc.savefig(outdir+'/'+obj+'_cont.png')
    plt.close()

    # Plot cross correlation functions with final model
    fig,ax = plots.multi(1,n,hspace=0.001,figsize=(6,2+n))
    ax = np.atleast_1d(ax)
    vmed = np.median(finalstr['vrel'])
    for i,(final,spec) in enumerate(zip(finalstr,specmlist)):
        ax[i].plot(final['x_ccf'],final['ccf'],color='k')
        ax[i].plot(final['x_ccf'],final['ccferr'],color='r')
        ax[i].plot([final['vrel'],final['vrel']],ax[i].get_ylim(),color='g',label='fit RV')
        ax[i].plot([final['xcorr_vrel'],final['xcorr_vrel']],ax[i].get_ylim(),color='r',label='xcorr RV')
        ax[i].text(0.1,0.9,'{:d}'.format(spec.head['MJD5']),transform=ax[i].transAxes)
        ax[i].set_xlim(vmed-200,vmed+200)
        ax[i].legend()
        if decomp is not None :
            try: n=decomp[i]['N_components']
            except: n=0
            if n>0 : n=len(decomp[i]['best_fit_parameters'])//3
            x = final['x_ccf']
            for j in range(n) :
                pars = decomp[i]['best_fit_parameters'][j::n]
                ax[i].plot(x,gaussian(*pars)(x))
                if pars[0] > 0 : color='k'
                else : color='r'
                ax[i].text(0.1,0.8-j*0.1,'{:8.1f}{:8.1f}{:8.1f}'.format(*pars),transform=ax[i].transAxes,color=color)
    fig.savefig(outdir+'/'+obj+'_ccf.png')
    plt.close()


from apogee_drp.apred import wave
from apogee_drp.apred import sincint

def visitcomb(allvisit,starver,load=None, apred='r13',telescope='apo25m',nres=[5,4.25,3.5],bconly=False,
              plot=False,write=True,dorvfit=True,apstar_vers='stars',logger=None):
    """ Combine multiple visits with individual RVs to rest frame sum
    """

    if logger is None:
        logger = dln.basiclogger()

    if load is None: load = apload.ApLoad(apred=apred,telescope=telescope)
    cspeed = 2.99792458e5  # speed of light in km/s

    logger.info('Doing visitcomb for {:s} '.format(allvisit['apogee_id'][0]))

    wnew = norm.apStarWave()  
    nwave = len(wnew)
    nvisit = len(allvisit)

    # initialize array for stack of interpolated spectra
    zeros = np.zeros([nvisit,nwave])
    izeros = np.zeros([nvisit,nwave],dtype=int)
    stack = apload.ApSpec(zeros,err=zeros.copy(),bitmask=izeros,cont=zeros.copy(),
                          sky=zeros.copy(),skyerr=zeros.copy(),telluric=zeros.copy(),telerr=zeros.copy())

    apogee_target1, apogee_target2, apogee_target3 = 0, 0, 0
    apogee2_target1, apogee2_target2, apogee2_target3, apogee2_target4 = 0, 0, 0, 0
    starflag,andflag = np.uint64(0),np.uint64(0)
    starmask = bitmask.StarBitMask()

    # Loop over each visit and interpolate to final wavelength grid
    if plot : fig,ax=plots.multi(1,2,hspace=0.001)
    for i,visit in enumerate(allvisit) :

        if bconly: vrel = -visit['bc']
        else: vrel = visit['vrel']

        # Skip if we don't have an RV
        if np.isfinite(vrel) is False : continue

        # Load the visit
        if load.telescope == 'apo1m':
            apvisit = load.apVisit1m(visit['plate'],visit['mjd'],visit['apogee_id'],load=True)
        else:
            apvisit = load.apVisit(int(visit['plate']),visit['mjd'],visit['fiberid'],load=True)
        pixelmask = bitmask.PixelBitMask()

        # Rest-frame wavelengths transformed to this visit spectra
        w = norm.apStarWave()*(1.0+vrel/cspeed)

        # Loop over the chips
        for chip in range(3) :
            # Get the pixel values to interpolate to
            pix = wave.wave2pix(w,apvisit.wave[chip,:])
            gd, = np.where(np.isfinite(pix))

            # Get a smoothed, filtered spectrum to use as replacement for bad values
            cont = gaussian_filter(median_filter(apvisit.flux[chip,:],[501],mode='reflect'),100)
            errcont = gaussian_filter(median_filter(apvisit.flux[chip,:],[501],mode='reflect'),100)
            bd, = np.where(apvisit.bitmask[chip,:]&pixelmask.badval())
            if len(bd) > 0: 
                apvisit.flux[chip,bd] = cont[bd] 
                apvisit.err[chip,bd] = errcont[bd] 

            # Load up quantity/error pairs for interpolation
            raw = [[apvisit.flux[chip,:],apvisit.err[chip,:]**2],
                   [apvisit.sky[chip,:],apvisit.skyerr[chip,:]**2],
                   [apvisit.telluric[chip,:],apvisit.telerr[chip,:]**2]]

            # Load up individual mask bits
            for ibit,name in enumerate(pixelmask.name):
                if name is not '' and len(np.where(apvisit.bitmask[chip,:]&2**ibit)[0]) > 0:
                    raw.append([np.clip(apvisit.bitmask[chip,:]&2**ibit,None,1),None])

            # Do the sinc interpolation
            out = sincint.sincint(pix[gd],nres[chip],raw)

            # From output flux, get continuum to remove, so that all spectra are
            #   on same scale. We'll later multiply in the median continuum
            flux = out[0][0]
            stack.cont[i,gd] = gaussian_filter(median_filter(flux,[501],mode='reflect'),100)

            # Load interpolated spectra into output stack
            stack.flux[i,gd] = out[0][0] / stack.cont[i,gd]
            stack.err[i,gd] = out[0][1] / stack.cont[i,gd]
            stack.sky[i,gd] = out[1][0]
            stack.skyerr[i,gd] = out[1][1]
            stack.telluric[i,gd] = out[2][0]
            stack.telerr[i,gd] = out[2][1]
            # For mask, set bits where interpolated value is above some threshold
            #   defined for each mask bit
            iout = 3
            for ibit,name in enumerate(pixelmask.name):
                if name is not '' and len(np.where(apvisit.bitmask[chip,:]&2**ibit)[0]) > 0:
                    j = np.where(np.abs(out[iout][0]) > pixelmask.maskcontrib[ibit])[0]
                    stack.bitmask[i,gd[j]] |= 2**ibit
                    iout += 1

        # Increase uncertainties for persistence pixels
        bd, = np.where((stack.bitmask[i,:]&pixelmask.getval('PERSIST_HIGH')) > 0)
        if len(bd) > 0: stack.err[i,bd] *= np.sqrt(5)
        bd, = np.where(((stack.bitmask[i,:]&pixelmask.getval('PERSIST_HIGH')) == 0) &
                       ((stack.bitmask[i,:]&pixelmask.getval('PERSIST_MED')) > 0) )
        if len(bd) > 0: stack.err[i,bd] *= np.sqrt(4)
        bd, = np.where(((stack.bitmask[i,:]&pixelmask.getval('PERSIST_HIGH')) == 0) &
                       ((stack.bitmask[i,:]&pixelmask.getval('PERSIST_MED')) == 0) &
                       ((stack.bitmask[i,:]&pixelmask.getval('PERSIST_LOW')) > 0) )
        if len(bd) > 0: stack.err[i,bd] *= np.sqrt(3)
        bd, = np.where((stack.bitmask[i,:]&pixelmask.getval('SIG_SKYLINE')) > 0)
        if len(bd) > 0: stack.err[i,bd] *= np.sqrt(100)

        if plot:
            ax[0].plot(norm.apStarWave(),stack.flux[i,:])
            ax[1].plot(norm.apStarWave(),stack.flux[i,:]/stack.err[i,:])
            plt.draw()
            pdb.set_trace()

        # Accumulate for header of combined frame. Turn off visit specific RV flags first
        visitflag = visit['starflag'] & ~starmask.getval('RV_REJECT') & ~starmask.getval('RV_SUSPECT')
        starflag |= visitflag
        andflag &= visitflag
        if visit['survey'] == 'apogee' :
            apogee_target1 |= visit['apogee_target1']
            apogee_target2 |= visit['apogee_target2']
            apogee_target3 |= visit['apogee_target3']
        elif visit['survey'].find('apogee2') >=0  :
            apogee2_target1 |= visit['apogee_target1']
            apogee2_target2 |= visit['apogee_target2'] 
            apogee2_target3 |= visit['apogee_target3']
            try: apogee2_target4 |= visit['apogee_target4']
            except: pass
        elif visit['survey'] == 'apo1m' :
            apogee_target2 |= visit['APOGEE_TARGET2'] 
            apogee2_target2 |= visit['APOGEE_TARGET2'] 
        # MWM target flags?
            

    # Create final spectrum
    if nvisit>1:
        zeros = np.zeros([nvisit+2,nwave])
        izeros = np.zeros([nvisit+2,nwave],dtype=int)
    else:
        zeros = np.zeros([1,nwave])
        izeros = np.zeros([1,nwave],dtype=int)
    if len(allvisit)==1:
        rvtab = Table(np.vstack(allvisit['rvtab']))
    else:
        rvtab = Table(np.squeeze(np.vstack(allvisit['rvtab'])))
    apstar = apload.ApSpec(zeros,err=zeros.copy(),bitmask=izeros,wave=norm.apStarWave(),
                           sky=zeros.copy(),skyerr=zeros.copy(),telluric=zeros.copy(),telerr=zeros.copy(),
                           cont=zeros.copy(),template=zeros.copy(),rvtab=rvtab)
    apstar.header['CRVAL1'] = norm.logw0
    apstar.header['CDELT1'] = norm.dlogw
    apstar.header['CRPIX1'] = 1
    apstar.header['CTYPE1'] = ('LOG-LINEAR','Logarithmic wavelength scale in subsequent HDU')
    apstar.header['DC-FLAG'] = 1

    # Pixel-by-pixel weighted average
    cont = np.median(stack.cont,axis=0)
    apstar.flux[0,:] = np.sum(stack.flux/stack.err**2,axis=0)/np.sum(1./stack.err**2,axis=0) * cont
    apstar.err[0,:] =  np.sqrt(1./np.sum(1./stack.err**2,axis=0)) * cont
    apstar.bitmask[0,:] = np.bitwise_and.reduce(stack.bitmask,0)
    apstar.cont[0,:] = cont

    # global weighting and individual visits
    if nvisit > 1 :
        # "global" weighted average
        newerr = median_filter(stack.err,[1,100],mode='reflect')
        bd = np.where((stack.bitmask&pixelmask.getval('SIG_SKYLINE')) > 0)[0]
        if len(bd) > 0 : newerr[bd[0],bd[1]] *= np.sqrt(100)
        apstar.flux[1,:] = np.sum(stack.flux/newerr**2,axis=0)/np.sum(1./newerr**2,axis=0) * cont
        apstar.err[1,:] =  np.sqrt(1./np.sum(1./newerr**2,axis=0)) * cont

        # Individual visits
        apstar.flux[2:,:] = stack.flux * stack.cont
        apstar.err[2:,:] = stack.err * stack.cont
        apstar.bitmask[2:,:] = stack.bitmask
        apstar.sky[2:,:] = stack.sky
        apstar.skyerr[2:,:] = stack.skyerr
        apstar.telluric[2:,:] = stack.telluric
        apstar.telerr[2:,:] = stack.telerr

    # Populate header
    apstar.header['OBJID'] = (allvisit['apogee_id'][0], 'APOGEE object name')
    apstar.header['APRED'] = (apred, 'APOGEE reduction version')
    apstar.header['STARVER'] = (starver, 'apStar version')
    apstar.header['HEALPIX'] = ( apload.obj2healpix(allvisit['apogee_id'][0]), 'HEALPix location')
    try :apstar.header['SNR'] = (np.nanmedian(apstar.flux[0,:]/apstar.err[0,:]), 'Median S/N per apStar pixel')
    except :apstar.header['SNR'] = (0., 'Median S/N per apStar pixel')
    apstar.header['RA'] = (allvisit['ra'].max(), 'right ascension, deg, J2000')
    apstar.header['DEC'] = (allvisit['dec'].max(), 'declination, deg, J2000')
    apstar.header['GLON'] = (allvisit['glon'].max(), 'Galactic longitude')
    apstar.header['GLAT'] = (allvisit['glat'].max(), 'Galactic latitude')
    apstar.header['JMAG'] = (allvisit['jmag'].max(), '2MASS J magnitude')
    apstar.header['JERR'] = (allvisit['jerr'].max(), '2MASS J magnitude uncertainty')
    apstar.header['HMAG'] = (allvisit['hmag'].max(), '2MASS H magnitude')
    apstar.header['HERR'] = (allvisit['herr'].max(), '2MASS H magnitude uncertainty')
    apstar.header['KMAG'] = (allvisit['kmag'].max(), '2MASS K magnitude')
    apstar.header['KERR'] = (allvisit['kerr'].max(), '2MASS K magnitude uncertainty')
    try: apstar.header['SRC_H'] = (allvisit['src_h'][0], 'source of H magnitude')
    except KeyError: pass

    # SDSS-V info
    apstar.header['CATID'] = (allvisit['catalogid'][0], 'SDSS-V catalog ID')
    apstar.header['PLX'] = (allvisit['gaiadr2_plx'].max(), 'GaiaDR2 parallax')
    apstar.header['EPLX'] = (allvisit['gaiadr2_plx_error'].max(), 'GaiaDR2 parallax uncertainty')
    apstar.header['PMRA'] = (allvisit['gaiadr2_pmra'].max(), 'GaiaDR2 proper motion in RA')
    apstar.header['EPMRA'] = (allvisit['gaiadr2_pmra_error'].max(), 'GaiaDR2 proper motion in RA uncertainty')
    apstar.header['PMDEC'] = (allvisit['gaiadr2_pmdec'].max(), 'GaiaDR2 proper motion in DEC')
    apstar.header['EPMDEC'] = (allvisit['gaiadr2_pmdec_error'].max(), 'GaiaDR2 proper motion in DEC uncertainty')
    apstar.header['GMAG'] = (allvisit['gaiadr2_gmag'].max(), 'GaiaDR2 G magnitude')
    apstar.header['GERR'] = (allvisit['gaiadr2_gerr'].max(), 'GaiaDR2 G magnitude uncertainty')
    apstar.header['BPMAG'] = (allvisit['gaiadr2_bpmag'].max(), 'GaiaDR2 Bp magnitude')
    apstar.header['BPERR'] = (allvisit['gaiadr2_bperr'].max(), 'GaiaDR2 Bp magnitude uncertainty')
    apstar.header['RPMAG'] = (allvisit['gaiadr2_rpmag'].max(), 'GaiaDR2 Rp magnitude')
    apstar.header['RPERR'] = (allvisit['gaiadr2_rperr'].max(), 'GaiaDR2 Rp magnitude uncertainty')
    apstar.header['SVAPTRG0'] = (allvisit['sdssv_apogee_target0'].max(),'SDSS-V APOGEE TARGET0 targeting flag')
    apstar.header['FRSTCRTN'] = (allvisit['sdssv_apogee_target0'][0],'SDSS-V MWM priorrity carton')

    apstar.header['APTARG1'] = (apogee_target1, 'APOGEE_TARGET1 targeting flag')
    apstar.header['APTARG2'] = (apogee_target2, 'APOGEE_TARGET2 targeting flag')
    apstar.header['APTARG3'] = (apogee_target3, 'APOGEE_TARGET3 targeting flag')
    apstar.header['AP2TARG1'] = (apogee2_target1, 'APOGEE2_TARGET1 targeting flag')
    apstar.header['AP2TARG2'] = (apogee2_target2, 'APOGEE2_TARGET2 targeting flag')
    apstar.header['AP2TARG3'] = (apogee2_target3, 'APOGEE2_TARGET3 targeting flag')
    apstar.header['AP2TARG4'] = (apogee2_target4, 'APOGEE2_TARGET4 targeting flag')
    apstar.header['NVISITS'] = (len(allvisit), 'Number of visit spectra combined flag')
    apstar.header['STARFLAG'] = (starflag,'bitwise OR of individual visit starflags')
    apstar.header['ANDFLAG'] = (andflag,'bitwise AND of individual visit starflags')

    try: apstar.header['N_COMP'] = (allvisit['n_components'].max(),'Maximum number of components in RV CCFs')
    except: pass
    apstar.header['VHBARY'] = ((allvisit['vheliobary']*allvisit['snr']).sum() / allvisit['snr'].sum(),'S/N weighted mean barycentric RV')
    if len(allvisit) > 1 : apstar.header['vscatter'] = (allvisit['vheliobary'].std(ddof=1), 'standard deviation of visit RVs')
    else: apstar.header['VSCATTER'] = (0., 'standard deviation of visit RVs')
    apstar.header['VERR'] = (0.,'unused')
    apstar.header['RV_TEFF'] = (allvisit['rv_teff'].max(),'Effective temperature from RV fit')
    apstar.header['RV_LOGG'] = (allvisit['rv_logg'].max(),'Surface gravity from RV fit')
    apstar.header['RV_FEH'] = (allvisit['rv_feh'].max(),'Metallicity from RV fit')

    if len(allvisit) > 0: meanfib=(allvisit['fiberid']*allvisit['snr']).sum()/allvisit['snr'].sum()
    else: meanfib = 999999.
    if len(allvisit) > 1: sigfib=allvisit['fiberid'].std(ddof=1)
    else: sigfib = 0.
    apstar.header['MEANFIB'] = (meanfib,'S/N weighted mean fiber number')
    apstar.header['SIGFIB'] = (sigfib,'standard deviation (unweighted) of fiber number')
    apstar.header['NRES'] = ('{:5.2f}{:5.2f}{:5.2f}'.format(*nres),'number of pixels/resolution used for sinc')

    # individual visit information in header
    for i0,visit in enumerate(allvisit) :
        i = i0+1
        apstar.header['SFILE{:d}'.format(i)] = (visit['file'],' Visit #{:d} spectrum file'.format(i))
        apstar.header['DATE{:d}'.format(i)] = (visit['dateobs'], 'DATE-OBS of visit {:d}'.format(i))
        apstar.header['JD{:d}'.format(i)] = (visit['jd'], 'Julian date of visit {:d}'.format(i))
        # hjd = helio_jd(visitstr[i].jd-2400000.0,visitstr[i].ra,visitstr[i].dec)
        #apstar.header['HJD{:d}'.format(i)] = 
        apstar.header['FIBER{:d}'.format(i)] = (visit['fiberid'],' Fiber, visit {:d}'.format(i))
        apstar.header['BC{:d}'.format(i)] = (visit['bc'],' Barycentric correction (km/s), visit {:d}'.format(i))
        apstar.header['CHISQ{:d}'.format(i)] = (visit['chisq'],' Chi-squared fit of Cannon model, visit {:d}'.format(i))
        apstar.header['VRAD{:d}'.format(i)] = (visit['vrel'],' Doppler shift (km/s) of visit {:d}'.format(i))
        #apstar.header['VERR%d'.format(i)] = 
        apstar.header['VHBARY{:d}'.format(i)] = (visit['vheliobary'],' Barycentric velocity (km/s), visit {:d}'.format(i))
        apstar.header['SNRVIS{:d}'.format(i)] = (visit['snr'],' Signal/Noise ratio, visit {:d}'.format(i))
        apstar.header['FLAG{:d}'.format(i)] = (visit['starflag'],' STARFLAG for visit {:d}'.format(i))
        apstar.header.insert('SFILE{:d}'.format(i),('COMMENT','VISIT {:d} INFORMATION'.format(i)))

    # Fix any NaNs in the header, astropy doesn't allow them
    for k in apstar.header.keys():
        if (k!='HISTORY') and (k!='COMMENT') and (k!='SIMPLE'):
            val = apstar.header.get(k)
            if np.issubdtype(type(val),np.number):
                if np.isnan(val)==True:
                    apstar.header[k] = 'NaN'   # change to string

    # Do a RV fit just to get a template and normalized spectrum, for plotting
    if dorvfit:
        try:
            apstar.setmask(pixelmask.badval())
            spec = doppler.Spec1D(apstar.flux[0,:],err=apstar.err[0,:],bitmask=apstar.bitmask[0,:],
                                  mask=apstar.mask[0,:],wave=apstar.wave,lsfpars=np.array([0]),
                                  lsfsigma=apstar.wave/22500/2.354,instrument='APOGEE',
                                  filename=apstar.filename)
            out = doppler.rv.jointfit([spec],verbose=False,plot=False,tweak=False,maxvel=[-50,50])
            apstar.cont = out[3][0].flux
            apstar.template = out[2][0].flux
        except ValueError as err:
            logger.error('Exception raised in visitcomb RV for: ', apstar.header['FIELD'],apstar.header['OBJID'])
            logger.error("ValueError: {0}".format(err))
        except RuntimeError as err:
            logger.error('Exception raised in visitcomb RV for: ', apstar.header['FIELD'],apstar.header['OBJID'])
            logger.error("Runtime error: {0}".format(err))
        except: 
            logger.error('Exception raised in visitcomb RV fit for: ',apstar.header['FIELD'],apstar.header['OBJID'])

    # Write the spectrum to file
    if write:
        outfilenover = load.filename('Star',obj=apstar.header['OBJID'])
        outdir = os.path.dirname(outfilenover)
        outbase = os.path.splitext(os.path.basename(outfilenover))[0]
        outbase += '-'+starver   # add star version
        outfile = outdir+'/'+outbase+'.fits'
        if apstar_vers != 'stars' :
            outfile = outfile.replace('/stars/','/'+apstar_vers+'/')
        outdir = os.path.dirname(outfile)
        try: os.makedirs(os.path.dirname(outfile))
        except: pass
        logger.info('Writing apStar file to '+outfile)
        apstar.write(outfile)
        apstar.filename = outfile
        mwm_root = os.environ['MWM_ROOT']
        apstar.uri = outfile[len(mwm_root)+1:]
        # Create symlink no file with no version
        if os.path.exists(outfilenover) or os.path.islink(outfilenover): os.remove(outfilenover)
        os.symlink(os.path.basename(outfile),outfilenover)  # relative path
        

        # Plot
        gd, = np.where((apstar.bitmask[0,:] & (pixelmask.badval()|pixelmask.getval('SIG_SKYLINE'))) == 0)
        fig,ax = plots.multi(1,3,hspace=0.001,figsize=(48,6))
        med = np.nanmedian(apstar.flux[0,:])
        plots.plotl(ax[0],norm.apStarWave(),apstar.flux[0,:],color='k',yr=[0,2*med])
        ax[0].plot(norm.apStarWave()[gd],apstar.flux[0,gd],color='g')
        ax[0].set_ylabel('Flux')
        try:
            ax[1].plot(norm.apStarWave()[gd],apstar.cont[gd],color='g')
            ax[1].set_ylabel('Normalized')
            ax[1].plot(norm.apStarWave(),apstar.template,color='r')
        except: pass
        plots.plotl(ax[2],norm.apStarWave(),apstar.flux[0,:]/apstar.err[0,:],yt='S/N')
        for i in range(3) : ax[i].set_xlim(15100,17000)
        ax[0].set_xlabel('Wavelength')
        fig.savefig(outdir+'/plots/'+outbase+'.png')

    # Plot
    if plot: 
        ax[0].plot(norm.apStarWave(),apstar.flux,color='k')
        ax[1].plot(norm.apStarWave(),apstar.flux/apstar.err,color='k')
        plt.draw()
        pdb.set_trace()

    return apstar

def dbingest(startab,starvisits):
    """ Insert the star and visit information into the database."""

    # Open db session
    db = apogeedb.DBSession()
    
    # Load star table
    db.ingest('star',startab)   # load summary information into "star" table
    

    # Load visit RV information into "rv_visit" table
    #  get star_pk from "star" table
    if starvisits is not None:
        starout = db.query('star',where="apogee_id='"+startab['apogee_id'][0]+"' and apred_vers='"+startab['apred_vers'][0]+"' "+\
                           "and telescope='"+startab['telescope'][0]+"' and starver='"+startab['starver'][0]+"'")
        starvisits['star_pk'] = starout['pk'][0]
        # Remove some unnecessary columns (duplicates what's in visit table)
        delcols = ['target_id','objtype','survey', 'field', 'programname', 'alt_id', 'location_id', 'glon','glat',
                   'jmag','jerr', 'herr', 'kmag', 'kerr', 'src_h','pmra', 'pmdec', 'pm_src','apogee_target1',
                   'apogee_target2', 'apogee_target3', 'apogee_target4','gaiadr2_sourceid',
                   'gaiadr2_plx','gaiadr2_plx_error','gaiadr2_pmra','gaiadr2_pmra_error','gaiadr2_pmdec',
                   'gaiadr2_pmdec_error','gaiadr2_gmag',
                   'gaiadr2_gerr','gaiadr2_bpmag','gaiadr2_bperr','gaiadr2_rpmag','gaiadr2_rperr',
                   'sdssv_apogee_target0','firstcarton',
                   'targflags', 'starflag', 'starflags','created','rvtab']
        visits = starvisits.copy()  # make a local copy
        for c in delcols:
            if c in visits.dtype.names:
                del visits[c]
        # Rename columns
        visits['pk'].name = 'visit_pk'
        visits['starver'] = starout['starver'][0]

        db.ingest('rv_visit',np.array(visits))   # Load the visit information into the table  

    # Close db session
    db.close()

