
# Software to use the arclamp and FPI data to get improved
# wavelength solution and accurate wavelengths for the FPI lines

# D. Nidever, July 2021

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import glob
import pdb
import time
#from functools import wraps
from astropy.io import ascii, fits
#from scipy import signal, interpolate
from scipy.optimize import curve_fit
#from scipy.special import erf, erfc
#from scipy.signal import medfilt, convolve, boxcar
from ..utils import apload, yanny, plan, peakfit, info
from ..plan import mkplan
from ..database import apogeedb
from . import wave
#from holtztools import plots, html
from astropy.table import Table,hstack,vstack
from dlnpyutils import utils as dln, robust, coords
#matplotlib.use('Qt5Agg')
#matplotlib.use('Agg')

chips = ['a','b','c']

def dailyfpiwave(mjd5,observatory='apo',apred='daily',num=None,clobber=False,verbose=True):
    """
    Function to run daily that generates a wavelength solution using a week's worth of
    arclamp data simultaneously fit with "apmultiwavecal" then we use the FPI full-frame exposure
    to get the FPI wavelengths and redo the wavelength solution.
    """

    t0 = time.time()

    db = apogeedb.DBSession()

    print('Getting daily FPI wavelengths for MJD='+str(mjd5))
    
    reduxdir = os.environ['APOGEE_REDUX']+'/'+apred+'/'
    datadir = {'apo':os.environ['APOGEE_DATA_N'],'lco':os.environ['APOGEE_DATA_S']}[observatory]
    instrument = {'apo':'apogee-n','lco':'apogee-s'}[observatory]
    load = apload.ApLoad(apred=apred,instrument=instrument)

    # Check if multiple new MJDs
    mjdlist = os.listdir(datadir)
    mjds = [mjd for mjd in mjdlist if int(mjd) >= (int(mjd5)-7) and int(mjd)<=int(mjd5) and mjd.isdigit()]
    if verbose:
        print(len(mjds), ' nights: ',','.join(mjds))

    # Get exposure information
    if verbose:
        print('Getting exposure information')
    for i,m in enumerate(mjds):
        expinfo1 = info.expinfo(observatory=observatory,mjd5=m)
        nexp = len(expinfo1)
        if verbose:
            print(m,' ',nexp,' exposures')
        if i==0:
            expinfo = expinfo1
        else:
            expinfo = np.hstack((expinfo,expinfo1))
    # Sort them
    si = np.argsort(expinfo['num'])
    expinfo = expinfo[si]

    # Step 1: Find the arclamp frames for the last week
    #--------------------------------------------------
    print(' ')
    print('Step 1: Find the arclamp frames for the last week')
    print('--------------------------------------------------')

    # Get arclamp exposures
    arc, = np.where((expinfo['exptype']=='ARCLAMP') & ((expinfo['arctype']=='UNE') | (expinfo['arctype']=='THAR')) &
                    (expinfo['mjd']>=mjd5-7) & (expinfo['mjd']<=mjd5))
    narc = len(arc)
    if narc==0:
        print('No arclamp exposures for these nights')
        return
    arcframes = expinfo['num'][arc]
    print(len(arcframes),' arclamp exposures for these nights')

    # Get full frame FPI exposure to use
    fpi, = np.where((expinfo['exptype']=='FPI') & (expinfo['mjd']==mjd5))
    if num is None:
        if len(fpi)==0:
            raise ValueError('No FPI exposures for MJD='+str(mjd5))
        # Take the first one this night
        fpinum = expinfo['num'][fpi][0]
        # Make sure that the FPI exposure has arclamp exposures at the same dither position?
    else:
        # Check that the input NUM is an FPI exposure
        g, = np.where(expinfo['num']==num)
        if len(g)==0:
            raise ValueError(str(num)+' not found')
        if expinfo['exptype'][g][0] != 'FPI':
            raise ValueError(str(num)+' is not a FPI exposure')
        fpi = num
    print('FPI full-frame exposure ',fpinum)
    fpiframe = load.ap1D(fpinum)


    # Step 2: Fit wavelength solutions simultaneously
    #------------------------------------------------
    print(' ')
    print('Step 2: Fit wavelength solutions simultaneously')
    print('------------------------------------------------')
    # This is what apmultiwavecal does
    if verbose:
        print('Solving wavelength solutions simultaneously using all arclamp exposures')
    #wfile = reduxdir+'cal/'+instrument+'/wave/apWave-%08d.fits' % mjd5
    wfile = reduxdir+'cal/'+instrument+'/wave/apWave-%s.fits' % str(mjd5)
    if os.path.exists(wfile.replace('apWave-','apWave-b-')) is False or clobber is True:
        # The previously measured lines in the apLines files will be reused if they exist
        npoly = 4 # 5
        #print('KLUDGE!! only using existing frames!!')
        #arcframes = arcframes[0:4]
        #afiles = ['/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/apogee/spectro/redux/daily/cal/apogee-n/wave/apWave-'+str(e)+'_lines.fits' for e in arcframes]
        #exists = dln.exists(afiles)
        #arcframes = arcframes[exists]
        #import pdb; pdb.set_trace()
        #arcframes = arcframes[-12:]
        pars,arclinestr = wave.wavecal(arcframes,rows=np.arange(300),name=str(mjd5),npoly=npoly,inst=instrument,verbose=verbose,vers=apred)
        # npoly=4 gives lower RMS values
        # Check that it's there

        if os.path.exists(wfile.replace('apWave-','apWave-b-')) is False:
            raise Exception(wfile+' not found')
    else:
        print(wfile,' wavelength calibration file already exists')


    # Load the wavelength solution from today
    daynum = mjd2day(mjd5)
    print('Using ',wfile,' wavelength calibration file')
    wavecal = load._readchip(wfile,'Wave')

    # The wavelength solutions are fit in groups
    #  find the group associated with the FPI
    ftable = wavecal['a'][7].data
    wheader = wavecal['a'][0].header
    nframes = wheader['nframes']
    ngroups = wheader['ngroup']
    mjdframeind, = np.where(ftable['frame'].find(str(daynum)) > -1)
    if len(mjdframeind)==0:
        raise Exception('No frames for MJD='+str(mjd5))

    # Get dither position for all the arclamp exposures and fpi 
    # to make sure there hasn't been a dither shift between them
    # but don't include darks
    
    # Get dithering information for this night
    expinfo = db.query(sql='select * from apogee_drp.exposure where mjd='+str(mjd5))
    si = np.argsort(expinfo['num'])
    expinfo = expinfo[si]
    # Loop over the exposures and mark time periods of constant dither position
    expinfo = Table(expinfo)
    expinfo['dithergroup'] = -1
    currentditherpix = expinfo['dithpix'][0]
    dithergroup = 1
    for e in range(len(expinfo)):
        if np.abs(expinfo['dithpix'][e]-currentditherpix)<0.01:
            expinfo['dithergroup'][e] = dithergroup
        else:
            dithergroup += 1
            currentditherpix = expinfo['dithpix'][e]
            expinfo['dithergroup'][e] = dithergroup

    # Pick the group closest to the FPI exposure at the same dither period    
    arcdithergroup = np.zeros(len(ftable),int)-1
    ind1,ind2 = dln.match(expinfo['num'],ftable['frame'])    
    if len(ind1)>0:
        arcdithergroup[ind2] = expinfo['dithergroup'][ind1]
    fpiind, = np.where(expinfo['num']==fpinum)
    fpidithergroup = expinfo['dithergroup'][fpiind][0]
    bestind, = np.where(arcdithergroup==fpidithergroup)
    if len(bestind)==0:
        raise ValueError('No arclamp frames in the same dither group as the FPI exposure')
    wgroup = ftable['group'][bestind][0]  # the wavelength group associated with the FPI
    print('Wavelength group ',wgroup,' is associated with the FPI')

    # Get the wavelength solution parameters for this group
    #  allpars has shape [npoly+3*ngroup,300]
    #  the first parameters are the polynomial coefficients, and then
    #  there are three offset values per group [chip1 offset ~ chipgap1, chip2 offset ~ 0, chip3 offset ~ chipgap2]
    npoly = 4
    allpars = wavecal['a'][6].data
    wpars = np.zeros((npoly+3,300),float)
    wpars[0:npoly,:] = allpars[0:npoly,:]
    wpars[npoly:,:] = allpars[npoly+wgroup*3:npoly+(wgroup+1)*3,:]
    # Now calculate the 2D wavelength arrays
    waves = np.zeros((3,300,2048),float)
    for row in range(300):
        x = np.zeros([3,2048])
        for ichip,chip in enumerate(chips):
            x[0,:] = np.arange(2048)
            x[1,:] = ichip+1
            x[2,:] = 0
            waves[ichip,row,:] = wave.func_multi_poly(x,*wpars[:,row],npoly=npoly)


    # Step 3: Fit peaks to the full-frame FPI data
    # --------------------------------------------
    print(' ')
    print('Step 3: Fit peaks to the full-frame FPI data')
    print('--------------------------------------------')
    fpilinesfile = reduxdir+'cal/'+instrument+'/fpi/apFPILines-%8d.fits' % fpinum
    if os.path.exists(fpilinesfile) and clobber is False:
        print('Loading previously measured FPI lines for ',fpinum)
        fpilines = Table.read(fpilinesfile)
    else:
        fpilines = fitlines(fpiframe,verbose=verbose)
        # Save the catalog
        print('Writing FPI lines to ',fpilinesfile)
        fpilines.write(fpilinesfile,overwrite=True)
    # write out median numbes of lines per chip


    # Step 4: Determine median wavelength per FPI lines
    # -------------------------------------------------
    print(' ')
    print('Step 4: Determine median wavelength per FPI lines')
    print(' -------------------------------------------------')
    # Load initial guesses
    fpipeaksfile = reduxdir+'cal/'+instrument+'/fpi/fpi_peaks.fits'
    if os.path.exists(fpipeaksfile):
        fpipeaks = Table.read(fpipeaksfile)
    else:
        print('No initial FPI peaks file found')
        fpipeaks = None
    fpilinestr, fpilines = getfpiwave(fpilines,wpars,fpipeaks)


    # Step 5: Refit wavelength solutions using FPI lines
    # --------------------------------------------------
    print(' ')
    print('Step 5: Refit wavelength solutions using FPI lines')
    print('--------------------------------------------------')
    fpiwcoef,fpiwaves = fpiwavesol(fpilinestr,fpilines,wpars)


    # Save the results
    #-----------------
    fpiwavefile = reduxdir+'cal/'+instrument+'/wave/apWaveFPI-%5d-%8d.fits' % (mjd5,fpinum)
    print('Writing new FPI wavelength information to '+fpiwavefile)
    save_fpiwave(fpiwavefile,mjd5,fpinum,fpiwcoef,fpiwaves,fpilinestr,fpilines)
    # table of FPI lines data: chip, gauss center, Gaussian parameters, wavelength, flux
    # wavelength coefficients
    # wavelength array??

    print("elapsed: %0.1f sec." % (time.time()-t0))
    db.close()   # close the database connection


def fitlines(frame,rows=np.arange(300),chips=['a','b','c'],verbose=False):
    """
    Fit the FPI lines with binned Gaussians.
    frame: FPI full-frame data loaded with load.ap1D().
    """

    # chip loop
    linestr = None
    for ichip,chip in enumerate(chips):
        flux = frame[chip][1].data
        err = frame[chip][2].data
        nfibers,npix = flux.shape
        for f in rows:
            linestr1 = peakfit.peakfit(flux[f,:],err[f,:])
            if linestr1 is not None:
                nlines = len(linestr1)
                linestr1 = Table(linestr1)
                linestr1['row'] = f 
                linestr1['chip'] = chip
                linestr1['expnum'] = 0
                if verbose:
                    print(chip,f,nlines,' lines')
                if linestr is None:
                    linestr = linestr1
                else:
                    linestr = vstack((linestr,linestr1))
            else:
                if verbose:
                    print(chip,f,0,' lines')
    return linestr


def getfpiwave(fpilines,wcoef,fpipeaks,verbose=True):
    """
    Determine median wavelength for the FPI lines
    fpilines: catalog of detected FPI peaks in full-frame image
    wcoef: wavelength solution parameters [7,300]
    fpipeaks: fiducial wavelengths for the unique FPI peaks
    """

    # Prune out lines that had unsuccessful fits
    bd, = np.where(fpilines['success']==False)
    if len(bd)>0:
        if verbose:
            print('Pruning ',len(bd),' FPI lines with unsuccessful Gaussian fits')
        fpilines.remove_rows(bd)

    # Use fpilines
    # Use wavecal to get wavelengths for fpilines
    if verbose:
        print('Calculating initial wavelengths using the wavelength solution')
    fpilines['wave'] = 999999.        # wavelength from the arclamp wavelength solution
    fpilines['linewave'] = 999999.    # median wavelength of the FPI line
    fpilines['lineid'] = -1           # FPI line ID
    #wcoef = wavecal['a'][3].data  # [7,300], same for all three chips
    chipnum = np.zeros(len(fpilines),int)
    for ichip,chip in enumerate(['a','b','c']):
        ind, = np.where(fpilines['chip']==chip)
        chipnum[ind] = ichip
    rowindex = dln.create_index(fpilines['row'])
    for i in range(len(rowindex['num'])):
        row = rowindex['value'][i]
        ind = rowindex['index'][rowindex['lo'][i]:rowindex['hi'][i]+1]
        nind = len(ind)
        x = np.zeros((3,nind),float)
        x[0,:] = fpilines['pars'][ind,1].data
        x[1,:] = chipnum[ind]+1
        x[2,:] = 0
        wpars = wcoef[:,row]
        fpilines['wave'][ind] = wave.func_multi_poly(x,*wpars,npoly=4)

    # Final FPI line table, one row per unique FPI line
    fpilinestr = np.zeros(len(fpipeaks),dtype=np.dtype([('id',int),('chip',np.str,10),('x',float),('height',float),
                                                        ('flux',float),('wave',float),('wsig',float),('nfiber',int)]))
    fpilinestr['id'] = np.arange(len(fpipeaks))+1
    # chip loop
    #  not entirely necessary, but speeds up the where statements a bit
    for ichip,chip in enumerate(['a','b','c']):
        ind, = np.where(fpipeaks['CHIP']==chip)
        fpipeaks1 = fpipeaks[ind]
        fpilinestr1 = fpilinestr[ind]
        #wavecal1 = wavecal[chip]
        lineind, = np.where(fpilines['chip']==chip)
        fpilines1 = fpilines[lineind]
        # FPI lines loop
        if verbose:
            print('CHIP   NUM         X       HEIGHT       FLUX        WAVE       WSIG   NFIBER')
        for i in range(len(ind)):
            ind1, = np.where(np.abs(fpipeaks1['WAVE'][i]-fpilines1['wave']) < 1.0)
            wave1 = np.median(fpilines1['wave'][ind1])
            wsig1 = dln.mad(fpilines1['wave'][ind1].data)
            # outlier rejection
            ind2, = np.where(np.abs(fpilines1['wave']-wave1) < 4*wsig1)
            wave2 = np.median(fpilines1['wave'][ind2])
            wsig2 = dln.mad(fpilines1['wave'][ind2].data)
            fpilinestr1['chip'][i] = chip
            fpilinestr1['x'][i] = np.median(fpilines1['pars'][ind2,1])
            fpilinestr1['height'][i] = np.median(fpilines1['pars'][ind2,0])
            fpilinestr1['flux'][i] = np.median(fpilines1['sumflux'][ind2])
            fpilinestr1['wave'][i] = wave2
            fpilinestr1['wsig'][i] = wsig2
            fpilinestr1['nfiber'][i] = len(ind2)
            # Update the fpilines table
            fpilines1['linewave'][ind2] = wave2
            fpilines1['lineid'][ind2] = fpilinestr1['id'][i]

            if verbose:
                print('%3s %8s %10.4f %11.4f %11.4f %11.4f %8.4f %5d' % (chip,str(i+1)+'/'+str(len(ind)),
                      fpilinestr1['x'][i],fpilinestr1['height'][i],fpilinestr1['flux'][i],
                      fpilinestr1['wave'][i],fpilinestr1['wsig'][i],fpilinestr1['nfiber'][i]))

        # stuff back into large structure
        fpilinestr[ind] = fpilinestr1
        fpilines[lineind] = fpilines1

    return fpilinestr, fpilines


def fpiwavesol(fpilinestr,fpilines,wcoef,verbose=True):
    """ 
    Refit wavelength solution using FPI wavelengths
    fpilinestr: information on each unique FPI line
    fpilines: catalog of FPI full-frame line measurements (all fibers)
    wcoef: original arclamp wavelength solution coefficients
    """

    # Prune out lines that had unsuccessful fits or no mean wavelength
    bd, = np.where((fpilines['success']==False) | (fpilines['lineid']==-1))
    if len(bd)>0:
        if verbose:
            print('Pruning ',len(bd),' FPI lines with unsuccessful Gaussian fits or no mean line wavelength')
        fpilines.remove_rows(bd)

    npoly = 4
    # increasing npoly doesn't improve the solution
    # probably because the original wavelength solution had npoly=4

    chipnum = np.zeros(len(fpilines),int)
    for ichip,chip in enumerate(['a','b','c']):
        ind, = np.where(fpilines['chip']==chip)
        chipnum[ind] = ichip

    allpars = np.zeros((npoly+3,300),float)
    allrms = np.zeros(300,float)

    norder = 4
    newwaves = np.zeros((3,300,2048),float)
    newwcoef = np.zeros((3,norder+1,300),float)

    for row in np.arange(300):
        # set up independent variable array with pixel, chip, groupid, and dependent variable (wavelength)
        thisrow, = np.where(fpilines['row'] == row)
        if len(thisrow)==0:
            print('No lines for row ',row)
            continue

        # Global 3-chip fitting
        x = np.zeros([3,len(thisrow)])
        x[0,:] = fpilines['pars'][thisrow,1]
        x[1,:] = chipnum[thisrow]+1
        x[2,:] = 0
        y = fpilines['linewave'][thisrow].data

        ngroup = 1
        npars = npoly+3*ngroup
        pars = np.zeros(npars,float)

        # Get initial guess from arclamp wavelength solution
        pars0 = np.zeros(npoly+3,float)
        #pars0[len(pars0)-7:] = wavecal['a'][3].data[:,row]
        pars0[len(pars0)-7:] = wcoef[:,row]

        # initialize bounds (to no bounds)
        bounds = ( np.zeros(len(pars))-np.inf, np.zeros(len(pars))+np.inf)
        # lock the middle chip position if we have one group, else the central wavelength
        bounds[0][npoly+1] = pars0[npoly+1]-1.e-7
        bounds[1][npoly+1] = pars0[npoly+1]+1.e-7
        
        # use curve_fit to optimize parameters
        try :
            if verbose:
                print('row: ', row, 'nlines: ', len(thisrow))
                print(pars0)
            popt,pcov = curve_fit(wave.func_multi_poly,x,y,p0=pars0,bounds=bounds)
            pars = copy.copy(popt)
            yfit = wave.func_multi_poly(x,*pars)
            res = y-yfit
            rms = np.sqrt(np.mean(res**2))
            if verbose:
                print('res: ',len(thisrow),np.median(res),np.median(np.abs(res)),dln.mad(np.array(res)))
                print(pars)
        except :
            print('Solution failed for row: ', row)
            import pdb; pdb.set_trace()
            popt = pars*0.
            rms = 999999.

        # Save the parameters
        allpars[:,row] = popt
        allrms[row] = rms


        # Do cubic fit to each chip separately
        #  This produces better results
        chres = np.zeros(len(thisrow),float)
        for ichip,chip in enumerate(chips):
            ind1, = np.where(fpilines['chip'][thisrow]==chip)
            chind = thisrow[ind1]
            xx = fpilines['pars'][chind,1]
            yy = fpilines['linewave'][chind]
            coef = robust.polyfit(xx,yy,4)  #3
            yfit = robust.npp_polyval(xx,np.flip(coef))
            chres1 = yy-yfit
            chrms = np.sqrt(np.mean(chres1**2))
            chres[ind1] = chres1
            newwcoef[ichip,:,row] = coef
            newwaves[ichip,row,:] = robust.npp_polyval(np.arange(2048),np.flip(coef))
        sig = dln.mad(np.array(chres))
        # this gets rid of the extra wiggle structure in the residuals
        print('chip res: ',len(thisrow),np.median(chres),np.median(np.abs(chres)),sig)
        

        # there is obvious structure in the residuals
        # fitting all lines from just ONE chip at a time gives much lower rms
        # -should I be fitting with local X values and higher order?
        # -should the original arclamp wavelength solutions use a higher order
        #   cubic seems very low
        if plt==True:
            xglobal = np.zeros(len(thisrow),float)
            g1,=np.where(x[1,:]==1)
            g2,=np.where(x[1,:]==2)
            g3,=np.where(x[1,:]==3)
            xglobal[g1] = x[0,g1]-1023.5-2048+pars[npoly]
            xglobal[g2] = x[0,g2]-1023.5+pars[npoly+1]
            xglobal[g3] = x[0,g3]-1023.5+2048+pars[npoly+2]

            figfile = 'wave_resid_fiber'+str(row)+'.png'
            matplotlib.use('Agg')
            fig,ax = plt.subplots()
            plt.scatter(xglobal,res)
            plt.scatter(xglobal,chres)
            plt.ylim([-0.02,0.02])
            plt.xlabel('Xglobal (pix)')
            plt.ylabel('Residuals (A)')
            plt.title('Residual to global fit - fiber='+str(row)+' rms='+str(rms)+' npoly='+str(npoly))
            plt.savefig(figfile,bbox_inches='tight')
            print('Saving figure to ',figfile)

    #import pdb; pdb.set_trace()

    return newwcoef,newwaves
    
def save_fpiwave(outfile,mjd5,fpinum,fpiwcoef,fpiwaves,fpilinestr,fpilines):
    """
    Save the FPI wavelength information
    outfile: output file name (with no chip tag in name)
    mjd5: day MJD value
    fpinum: FPI full-frame exposure number
    fpiwcoef: new wavelength coefficients [3,5,300]
    fpiwaves: new wavelength array [3,300,2048]
    fpilinestr: table of unique FPI lines and wavelengths
    fpilines: table of all FPI full-frame line measurements (all fibers)
    """

    nchips,npoly,nfibers = fpiwcoef.shape

    # Save the new wavelength solution
    for ichip,chip in enumerate(chips):
        hdu = fits.HDUList()
        hdu.append(fits.PrimaryHDU())
        hdu[0].header['FRAME'] = fpinum
        hdu[0].header['NPOLY'] = npoly
        hdu[0].header['COMMENT'] = 'HDU#1 : wavelength calibration parameters [5,300]'
        hdu[0].header['COMMENT'] = 'HDU#2 : wavelength calibration array [300,2048]'
        hdu[0].header['COMMENT'] = 'HDU#3 : table of unique FPI lines and wavelengths'
        hdu[0].header['COMMENT'] = 'HDU#4 : table of full-frame FPI lines measurements'
        hdu.append(fits.ImageHDU(fpiwcoef[ichip,:,:]))
        hdu.append(fits.ImageHDU(fpiwaves[ichip,:,:]))
        ind, = np.where(fpilinestr['chip']==chip)
        hdu.append(fits.table_to_hdu(Table(fpilinestr[ind])))
        ind, = np.where(fpilines['chip']==chip)
        hdu.append(fits.table_to_hdu(Table(fpilines[ind])))    
        hdu.writeto(outfile.replace('WaveFPI','WaveFPI-'+chip),overwrite=True)


def fpi1dwavecal(planfile=None,frameid=None,out=None,instrument=None,fpiid=None,
                 vers='daily',telescope='apo25m',plugmap=None,verbose=False):
    """ 
    Determine positions of FPI lines and figure out shifts for all fibers
    """

    # Deal with the two cases: 1) planfile, or 2) individual exposure
    if planfile is not None:
        # read planfile
        if type(planfile) is dict:
            p = planfile
            dirname = '.'
        else :
            p = plan.load(planfile,np=True)
            dirname = os.path.dirname(planfile)
        telescope = p['telescope']
        instrument = p['instrument']
    # single exposure, make fake plan file dictionary
    else:
        p={}
        p['APEXP']={}
        p['APEXP']['name']=[str(frameid)]
        p['mjd'] = int(frameid) // 10000  + 55562
        p['waveid'] = str(fpiid)
        p['fpiid'] = str(fpiid)        
        if plugmap is None :
            p['platetype'] = 'sky'
        else :
            p['platetype'] = 'object'
            p['plugmap'] = plugmap
        p['telescope'] = telescope
        if telescope == 'lco25m' : instrument = 'apogee-s'
        else : instrument = 'apogee-n'
        p['apred_vers'] = vers
        p['instrument'] = instrument

    
    reduxdir = os.environ['APOGEE_REDUX']+'/'+vers+'/'
    observatory = {'apo25m':'apo', 'apo1m':'apo', 'lco25m':'lco'}[telescope]
    datadir = {'apo':os.environ['APOGEE_DATA_N'],'lco':os.environ['APOGEE_DATA_S']}[observatory]
    load = apload.ApLoad(apred=vers,instrument=instrument)
    mjd = int(load.cmjd(fpiid))
        
    # Load the wavelength array/solution
    print('loading fpiid wavelengtth solution: ', fpiid)
    fpiwavefile = reduxdir+'cal/'+instrument+'/wave/apWaveFPI-%5d-%8d.fits' % (mjd,fpiid)
    waveframe = load._readchip(fpiwavefile,'apWaveFPI')
    npoly = waveframe['a'][0].header['NPOLY']
    norder = npoly-1
    wcoef = waveframe['a'][1].data
    waves = np.zeros((3,300,2048),float)    
    for ichip,chip in enumerate(chips):
        waves[ichip,:,:] = waveframe[chip][2].data

    # FPI fibers
    fpirows = [75,225]

    # Loop over all frames in the planfile and assess FPI lines in each
    grid = []
    ytit = []
    x = np.arange(2048).astype(float)
    for iframe,name in enumerate(p['APEXP']['name']) :
        name = str(name)  # make sure it's a string
        print('frame: ', name)
        frame = load.ap1D(int(name))
        plot = dirname+'/plots/fpipixshift-'+name+'-'+str(fpiid)

        newwaves = np.zeros((3,300,2048),float)
        newcoef = np.zeros((3,norder+1,300),float)    
        newchippars = np.zeros([3,14,300],float)
        newlincoef = np.zeros([3,4],float)

        # Do each chip separately
        for ichip,chip in enumerate(chips):
            print('chip: ',chip)

            # Find the FPI lines
            linestr = fitlines(frame,fpirows,chips=[chip],verbose=verbose)
            linestr['x'] = linestr['pars'][:,1]

            # Match up the lines with the reference frame
            fpilinestr = Table(waveframe[chip][3].data)   # unique FPI lines
            fpilines = Table(waveframe[chip][4].data)     # all lines, these already have lineid's and linewave 
            fpilines['x'] = fpilines['pars'][:,1]
            mlinestr = []     # matched lines
            mfpilines = []    # matched lines
            for f in fpirows:
                indfpi, = np.where(fpilines['row']==f)
                indline, = np.where(linestr['row']==f)
                if len(indfpi)>0 and len(indline)>0:
                    fpilines1 = fpilines[indfpi]
                    linestr1 = linestr[indline]
                    # crossmatch the two lists of positions, find matches within 1 pixel
                    x1 = np.vstack((np.array(linestr1['pars'][:,1]),np.zeros(len(linestr1),float))).T
                    x2 = np.vstack((np.array(fpilines1['pars'][:,1]),np.zeros(len(fpilines1),float))).T
                    dist,ind = coords.crossmatch(x1,x2,max_distance=1.0)
                    gd, = np.where(dist<1.0)
                    print('row: ',f,' ',len(gd),' matches')
                    ind1 = gd
                    ind2 = ind[gd]
                    if len(mlinestr)==0:
                        mlinestr = linestr1[ind1]
                        mfpilines = fpilines1[ind2]
                    else:
                        mlinestr = np.append(mlinestr,linestr1[ind1])
                        mfpilines = np.append(mfpilines,fpilines1[ind2])
                else:
                    print('row: ',f,' No measured FPI lines')
            mlinestr = dln.addcatcols(mlinestr,np.dtype([('dx',float)]))
            mlinestr['dx'] = mlinestr['x']-mfpilines['x']

            # Perform the linear surface fit
            fpifitmethod = 'xy'
            lincoef,xoffset = fpisurfit(mlinestr,mfpilines,method=fpifitmethod)
            print('Xshift 2D fit coef: ',lincoef)
            newlincoef[ichip,:] = lincoef
        
            # Get new wavelength solutions for each fiber using the
            #  shifts in X position
            # loop over rows
            # [14,300]
            #oldpars = frame[chip][5].data
            x = np.arange(2048)
            chipoffsets = [-143.8, 0.0, 154.4]  # mean chip offsets
            for irow in np.arange(300):
                # get wavelengths for the FPI reference frame
                w1 = waves[ichip][irow,:]
                x1 = x+xoffset[irow,:]  # new x positions
                # refit polynomial
                newcoef1 = robust.polyfit(x1,w1,norder)
                newwave1 = robust.npp_polyval(x,np.flip(newcoef1))
                newcoef[ichip,:,irow] = newcoef1
                newwaves[ichip,irow,:] = newwave1
                # 14 parameter chip coefficients
                pw = npoly-1-np.arange(npoly)
                #xoffset1 = oldpars[0,irow]
                xoffset1 = -1023.5 + (ichip-1)*2048 + chipoffsets[ichip]
                polypars = np.polyfit((x+xoffset1)/3000.,newwave1,norder)  # refit for (x+xoffset)/3000 values
                newchippars[ichip,:,irow] = np.append([xoffset1, 0., 0., 1., 0., 0.], 
                                                      np.flip( np.append(np.zeros(8-npoly),polypars)))

            # Fix wavelengths for broken fibers
            #  just use neighboring good fibers
            totwave = np.sum(newwaves[ichip,:,:],axis=1)
            goodwave = (totwave > 1)
            bdrows, = np.where(totwave < 1)
            for brow in bdrows:
                if brow==0:
                    newcoef[ichip,:,brow] = newcoef[ichip,:,brow+1]
                    newwaves[ichip,brow,:] = newwaves[ichip,brow+1,:]
                elif brow==299:
                    newcoef[ichip,:,brow] = newcoef[ichip,:,brow-1]
                    newwaves[ichip,brow,:] = newwaves[ichip,brow-1,:]
                else:
                    if goodwave[brow+1]:
                        newcoef[ichip,:,brow] = newcoef[ichip,:,brow+1]
                        newwaves[ichip,brow,:] = newwaves[ichip,brow+1,:]
                    else:
                        newcoef[ichip,:,brow] = newcoef[ichip,:,brow-1]
                        newwaves[ichip,brow,:] = newwaves[ichip,brow-1,:]

            # Update header for this chip
            frame[chip][0].header['HISTORY'] = 'Added wavelengths from FPI cal, fpiid: '+str(fpiid)
            frame[chip][0].header['FPIMETHD'] = fpifitmethod
            frame[chip][0].header['FPINPARS'] = len(lincoef)
            for ip,p in enumerate(lincoef):
                frame[chip][0].header['FPIPAR'+str(ip)] = p
            frame[chip][0].header['FPIFILE'] = fpiwavefile
            frame[chip][0].header['WAVEFILE'] = fpiwavefile
            frame[chip][0].header['WAVEHDU'] = 5

        # Rewrite out 1D file with adjusted wavelength information
        outname = load.filename('1D',num=int(name),mjd=load.cmjd(int(name)),chips=True)
        print('Writing to ',outname)
        for ichip,chip in enumerate(chips) :
            hdu = fits.HDUList()
            hdu.append(frame[chip][0])           # header
            hdu.append(frame[chip][1])           # flux
            hdu.append(frame[chip][2])           # err
            hdu.append(frame[chip][3])           # mask
            hdu.append(fits.ImageHDU(newwaves[ichip,:,:]))      # wavelength array
            hdu.append(fits.ImageHDU(newchippars[ichip,:,:]))   # wave coefficients
            hdu.writeto(outname.replace('1D-','1D-'+chip+'-'),overwrite=True)

        # Plots
        plot = None
        if plot is not None:
            try: os.mkdir(dirname+'/plots')
            except: pass

            # this is the plotting code from wave.skycal()
            # needs to be updated

            import pdb; pdb.set_trace()
            
            # plot the pixel shift for each chip derived from the FPI lines
            fig,ax = plots.multi(1,1)
            wfig,wax = plots.multi(1,3)
            for ichip,chip in enumerate(chips):
                gd = np.where(mlinestr['chip'] == chip)[0]
                med = np.median(mlinestr['dx'][gd])
                x = mlinestr['x'][gd]
                y = mlinestr['row'][gd]
                z = mlinestr['dx'][gd]
                zfit = func_poly2d(x,y,*newlincoef[ichip,:])
                plots.plotp(ax,x,y,color=colors[ichip],xr=[0,2048],yr=[med-0.2,med+0.2],
                            size=12,xt='X',yt='Pixel shift')
                plots.plotc(wax[ichip],x,y,z,zr=[-1,1],yr=[med-0.5,med+0.5],
                            xr=xlim[ichip],size=12,xt='Wavelength',yt='Pixel shift')
                gdfit = np.where(np.abs(y-med) < 0.5)[0]
                xx = np.arange(300)
                yy = w[0]*xx
                yy += w[ichip+1]
                plots.plotl(ax,xx,yy,color=colors[ichip])
                if waveid > 0 : label = 'Frame: {:s}  Waveid: {:8d}'.format(name,waveid)
                else : label = 'Frame: {:s}  Delta from ap1dwavecal'.format(name)
                ax.text(0.1,0.9,label,transform=ax.transAxes)
            if type(plot) is str or type(plot) is unicode: 
                wfig.tight_layout()
                wfig.savefig(plot+'_wave.png')
                fig.savefig(plot+'.png')
                grid.append(['../plots/'+os.path.basename(plot)+'.png','../plots/'+os.path.basename(plot)+'_wave.png'])
                ytit.append(name)
            else: 
                plt.show()
                plt.draw()
                pdb.set_trace()
            plt.close('all')

    
def fpisurfit(mlinestr,mfpilines,method='y'):
    """
    Fit linear surface to FPI line offsets
    mlinestr: (matched) measured lines in this exposure
    mfpilines: (matched)  measured lines from the full-frame FPI reference exposure
    method: "y" means only a surface in Y, while "xy" means linear surface
            in X and Y (row/fiber).
    """

    # Get unique rows
    rows = np.unique(mlinestr['row'])
    nrows = len(rows)
    npix = 2048
    x = np.arange(npix).astype(float)

    # Linear surface in X
    if method=='y':

        # translated from fpi_peaks_ylinsurf.pro

        # median dx values for reference fibers
        refmeddx = np.zeros(len(rows),float)        
        for irow,row in enumerate(rows):
            indline, = np.where(mlinestr['row']==row)
            indfpi, = np.where(mfpilines['row']==row)
            mlinestr1 = mlinestr[indline]
            mfpilines1 = mfpilines[indfpi]
            refmeddx[irow] = np.median(mlinestr1['x']-mfpilines1['x'])
        # linear fit of dx in Y
        pars = np.poly_fit(rows,refmeddx,1)
        yyall = (np.arange(300).reshape(300,1)+np.zeros(npix,float)).T
        dxoffset = np.polyval(pars,yyall)
        dxoffset = dxoffset.reshape(npix,300).T  # [300,2048]

        return pars, dxoffset

    # Linear surface in X and Y
    elif method=='xy':

        # translated from fpi_peaks_xylinsurf.pro

        ncoef = 4   # 3 or 4 for linear surface, need 4 to get a good fit
        # Loop over the rows and get robust linear fits
        refcoef = np.zeros((2,len(rows)),float)
        xx = np.zeros(nrows*npix,float)
        yy = np.zeros(nrows*npix,float)
        zz = np.zeros(nrows*npix,float)
        for irow,row in enumerate(rows):
            indline, = np.where(mlinestr['row']==row)
            indfpi, = np.where(mfpilines['row']==row)
            mlinestr1 = mlinestr[indline]
            mfpilines1 = mfpilines[indfpi]
            coef1 = robust.polyfit(mlinestr1['x'],mlinestr1['x']-mfpilines1['x'],1)
            refcoef[:,irow] = coef1
            xx[irow*npix:(irow+1)*npix] = x
            yy[irow*npix:(irow+1)*npix] = row
            zz[irow*npix:(irow+1)*npix] = robust.npp_polyval(x,np.flip(coef1))

        # Now perform linear surface fit
        err = zz*0+1
        initpar = np.zeros(ncoef,float)
        xinp = np.zeros((nrows*npix,2),float)
        xinp[:,0] = xx
        xinp[:,1] = yy
        pars,pcov = curve_fit(func_poly2d_wrap,xinp,zz,p0=initpar,sigma=err)
        dx_fit = func_poly2d_wrap(xinp,*pars)
        perror = np.diag(np.sqrt(pcov))
        xxall = x.reshape(npix,1)+np.zeros(300,float)
        yyall = (np.arange(300).reshape(300,1)+np.zeros(npix,float)).T
        dxoffset = func_poly2d(xxall.flatten(),yyall.flatten(),*pars)
        dxoffset = dxoffset.reshape(npix,300).T  # [300,2048]
        mnx = np.median(mlinestr['x'])

        # Compare 2D fit to original line offsets
        dx_lines = mlinestr['x']-mfpilines['x']
        dx_lines_fit = func_poly2d(mlinestr['x'],mlinestr['row'],*pars)
        sig = dln.mad(dx_lines-dx_lines_fit,zero=True)
        gdlines, = np.where(np.abs(dx_lines-dx_lines_fit) < 4*sig)
        rms = np.sqrt(np.mean((dx_lines[gdlines]-dx_lines_fit[gdlines])**2))
        print('rms = %.4f pixels' % rms)
        print('sig = %.4f pixels' % sig)

        return pars, dxoffset

    else:
        raise Exception('method '+str(method)+' not supported')

    return lincoef, xoffset

def func_poly2d_wrap(x,*args):
    """ thin wrapper for curve_fit"""
    xx = x[:,0]
    yy = x[:,1]
    return func_poly2d(xx,yy,*args)

def func_poly2d(x,y,*args):
    """ 2D polynomial surface"""

    p = args
    np = len(p)
    if np==0:
        a = p[0] 
    elif np==3:
        a = p[0] + p[1]*x + p[2]*y 
    elif np==4:
        a = p[0] + p[1]*x + p[2]*x*y + p[3]*y 
    elif np==6:
        a = p[0] + p[1]*x + p[2]*x**2 + p[3]*x*y + p[4]*y + p[5]*y**2
    elif np==8:
        a = p[0] + p[1]*x + p[2]*x**2 + p[3]*x*y + p[4]*(x**2)*y + p[5]*x*y**2 + p[6]*y + p[7]*y**2
    elif np==11:
        a = p[0] + p[1]*x + p[2]*x**2.0 + p[3]*x**3.0 + p[4]*x*y + p[5]*(x**2.0)*y + \
            p[6]*x*y**2.0 + p[7]*(x**2.0)*(y**2.0) + p[8]*y + p[9]*y**2.0 + p[10]*y**3.0
    elif np==15:
        a = p[0] + p[1]*x + p[2]*x**2 + p[3]*x**3 + p[4]*x**4 + p[5]*y + p[6]*x*y + \
            p[7]*(x**2)*y + p[8]*(x**3)*y + p[9]*y**2 + p[10]*x*y**2 + p[11]*(x**2)*y**2 + \
            p[12]*y**3 + p[13]*x*y**3 + p[14]*y**4
    elif np==21:
        a = p[0] + p[1]*x + p[2]*x**2 + p[3]*x**3 + p[4]*x**4 + p[5]*x**5 + p[6]*y + p[7]*x*y + \
            p[8]*(x**2)*y + p[9]*(x**3)*y + p[10]*(x**4)*y + p[11]*y**2 + p[12]*x*y**2 + \
            p[13]*(x**2)*y**2 + p[14]*(x**3)*y**2 + p[15]*y**3 + p[16]*x*y**3 + p[17]*(x**2)*y**3 + \
            p[18]*y**4 + p[19]*x*y**4 + p[20]*y**5
    else:
        raise Exception('Only 3, 4, 6, 8, 11 amd 15 parameters supported')

    return a

    
def frame2mjd(frame) :
    """ Get MJD from frame number """
    mjd = 55562+int(frame//10000)
    return mjd

def mjd2day(mjd) :
    """ Get frame day number from MJD """
    daynum = int(mjd)-55562
    return daynum
