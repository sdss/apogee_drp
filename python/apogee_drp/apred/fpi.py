
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
#from functools import wraps
from astropy.io import ascii, fits
#from scipy import signal, interpolate
from scipy.optimize import curve_fit
#from scipy.special import erf, erfc
#from scipy.signal import medfilt, convolve, boxcar
from ..utils import apload, yanny, plan, peakfit
from ..plan import mkplan
from . import wave
#from holtztools import plots, html
from astropy.table import Table,hstack,vstack
from dlnpyutils import utils as dln
matplotlib.use('Qt5Agg')

def dailyfpiwave(mjd5,observatory='apo',apred='daily',clobber=False,verbose=True):
    """
    Function to run daily that generates a wavelength solution using a week's worth of
    arclamp data simultaneously fit with "apmultiwavecal" then we use the FPI full-frame exposure
    to get the FPI wavelengths and redo the wavelength solution.
    """
    
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
        expinfo1 = mkplan.getexpinfo(observatory,m)
        nexp = len(expinfo1)
        if verbose:
            print(m,' ',nexp,' exposures')
        if i==0:
            expinfo = expinfo1
        else:
            expinfo = np.hstack((expinfo,expinfo1))


    # Step 1: Find the arclamp frames for the last week
    #--------------------------------------------------

    # Get arclamp exposures
    arc, = np.where((expinfo['exptype']=='ARCLAMP') & ((expinfo['arctype']=='UNE') | (expinfo['arctype']=='THAR')))
    narc = len(arc)
    if narc==0:
        print('No arclamp exposures for these nights')
        return
    arcframes = expinfo['num'][arc]
    print(len(arcframes),' arclamp exposures for these nights')

    # Get full frame FPI exposure to use
    fpi, = np.where(expinfo['exptype']=='FPI')
    fpinum = 38310023
    print('KLUDGE!!!  Hardcoding FPI full-frame exposure number')
    print('FPI full-frame exposure ',fpinum)


    # Step 2: Fit wavelength solutions simultaneously
    #------------------------------------------------
    # This is what apmultiwavecal does
    if verbose:
        print('Solving wavelength solutions simultaneously using all arclamp exposures')
    wfile = reduxdir+'cal/'+instrument+'/wave/apWave-%8d.fits' % mjd5
    if os.path.exists(wfile.replace('apWave-','apWave-b-')) is False or clobber is True:
        # The previously measured lines in the apLines files will be reused if they exist
        npoly = 4 # 5
        #print('KLUDGE!! only using first four frames!!')
        #arcframes = arcframes[0:4]
        #import pdb; pdb.set_trace()
        pars,arclinestr = wave.wavecal(arcframes,rows=np.arange(300),name=str(mjd5),npoly=npoly,inst=instrument,verbose=verbose,vers=apred)
        # npoly=4 gives lower RMS values
        # Check that it's there
        if os.path.exists(wavefile) is False:
            raise Exception(wfile+' not found')
    else:
        print(wfile,' wavelength calibration file already exists')

    # Load the wavelength solution from today
    #daynum = mjd2day(mjd5)
    #import pdb; pdb.set_trace()
    #wavefiles = glob.glob(reduxdir+'cal/'+instrument+'/wave/apWave-b-%4d????.fits' % daynum)
    #print('KLUDGE!  hard coding apWave-31690004')
    #wavefiles = 'apWave-b-31690004.fits'
    #waveid = os.path.basename(dln.first_el(wavefiles))[9:17]
    #print('Using wavelength cal file ',waveid)
    #wavecal = load.apWave(waveid)  # why doesn't this work??
    #wfile = load.allfile('Wave',num=str(mjd5),chips=True)
    print('Using ',wfile,' wavelength calibration file')
    wavecal = load._readchip(wfile,'Wave')
    
    # Step 3: Fit peaks to the full-frame FPI data
    # --------------------------------------------
    print('Finding FPI lines')
    fpilinesfile = reduxdir+'cal/'+instrument+'/fpi/apFPILines-%8d.fits' % fpinum
    if os.path.exists(fpilinesfile) and clobber is False:
        print('Loading previously measured FPI lines for ',fpinum)
        fpilines = Table.read(fpilinesfile)
    else:
        fpiframe = load.ap1D(fpinum)
        fpilines = fitlines(fpiframe)
        # Save the catalog
        print('Writing FPI lines to ',fpilinesfile)
        fpicat.write(fpilinesfile,overwrite=True)

    # Step 4: Determine median wavelength per FPI lines
    # -------------------------------------------------
    # Load initial guesses
    fpipeaksfile = reduxdir+'cal/'+instrument+'/fpi/fpi_peaks.fits'
    if os.path.exists(fpipeaksfile):
        fpipeaks = Table.read(fpipeaksfile)
    else:
        print('No initial FPI peaks file found')
        fpipeaks = None
    fpilinestr, fpilines = getfpiwave(fpilines,wavecal,fpipeaks)

    # Step 5: Refit wavelength solutions using FPI lines
    # --------------------------------------------------
    fpiwavecal = fpiwavesol(fpilinestr,fpilines,wavecal)

    import pdb; pdb.set_trace()


    # Save the results
    #-----------------
    save_fpiwave(mjd5,fpiwavecal,fpilinestr,wavecal)
    # table of FPI lines data: chip, gauss center, Gaussian parameters, wavelength, flux
    # wavelength coefficients
    # wavelength array??




    # make a little python function that generates a wavelength solution using a week's worth of
    # arclamp data simultaneously fit with "apmultiwavecal" and then we use the FPI full-frame exposure
    # to get the FPI wavelengths and redo the wavelength solution
    # -find the arclamp frames for the last week
    # -run apmultiwavecal on them
    # -fit peaks in FPI data
    # -define median wavelengths per FPI line
    # -refit wavelength solution with FPI lines, maybe holding higher-order coefficients fixed



def fitlines(frame,verbose=False):
    """
    Fit the FPI lines with binned Gaussians.
    frame: FPI full-frame data loaded with load.ap1D().
    """

    # chip loop
    linestr = None
    for ichip,chip in enumerate(['a','b','c']):
        flux = frame[chip][1].data
        err = frame[chip][2].data
        nfibers,npix = flux.shape
        fibers = np.arange(nfibers)
        for f in fibers:
            linestr1 = peakfit.peakfit(flux[f,:],err[f,:])
            if linestr1 is not None:
                nlines = len(linestr1)
                linestr1 = Table(linestr1)
                linestr1['fiber'] = f 
                linestr1['chip'] = chip
                linestr1['expnum'] = 0
                if verbose:
                    print(chip,f,nlines,' lines')
                if linestr is None:
                    linestr = linestr1
                else:
                    linestr = vstack((linestr,linestr1))

    return linestr


def getfpiwave(fpilines,wavecal,fpipeaks,verbose=True):
    """
    Determine median wavelength for the FPI lines
    fpilines: catalog of detected FPI peaks in full-frame image
    wavecal: wavelength solution
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
    wcoef = wavecal['a'][3].data  # [7,300], same for all three chips
    chipnum = np.zeros(len(fpilines),int)
    for ichip,chip in enumerate(['a','b','c']):
        ind, = np.where(fpilines['chip']==chip)
        chipnum[ind] = ichip
    findex = dln.create_index(fpilines['fiber'])
    for i in range(len(findex['num'])):
        fiber = findex['value'][i]
        ind = findex['index'][findex['lo'][i]:findex['hi'][i]+1]
        nind = len(ind)
        x = np.zeros((3,nind),float)
        x[0,:] = fpilines['pars'][ind,1].data
        x[1,:] = chipnum[ind]+1
        x[2,:] = 0
        wpars = wcoef[:,fiber]
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
        wavecal1 = wavecal[chip]
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
            fpilinestr1['chip'][i] = ichip
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

def fpiwavesol(fpilinestr,fpilines,wavecal,verbose=True):
    """ 
    Refit wavelength solution using FPI wavelengths
    fpilinestr: information on each unique FPI line
    fpilines: information on FPI full-frame
    wavecal: original wavelength solution used
    """

    npoly = 4
    # increasing npoly doesn't improve the solution
    # probably because the original wavelength solution had npoly=4

    chipnum = np.zeros(len(fpilines),int)
    for ichip,chip in enumerate(['a','b','c']):
        ind, = np.where(fpilines['chip']==chip)
        chipnum[ind] = ichip

    allpars = np.zeros((npoly,300),float)
    allrms = np.zeros(300,float)

    for row in np.arange(300):
        # set up independent variable array with pixel, chip, groupid, and dependent variable (wavelength)
        thisrow, = np.where(fpilines['fiber'] == row)
        if len(thisrow)==0:
            print('No lines for row ',row)
            continue
        x = np.zeros([3,len(thisrow)])
        x[0,:] = fpilines['pars'][thisrow,1]
        x[1,:] = chipnum[thisrow]+1
        x[2,:] = 0
        y = fpilines['linewave'][thisrow].data

        ngroup = 1
        npars = npoly+3*ngroup
        pars = np.zeros(npars,float)

        # initialize bounds (to no bounds)
        bounds = ( np.zeros(len(pars))-np.inf, np.zeros(len(pars))+np.inf)
        # lock the middle chip position if we have one group, else the central wavelength
        bounds[0][npoly+1] = -1.e-7
        bounds[1][npoly+1] = 1.e-7

        # reject lines with bad wavelengths
        gd, = np.where(y < 2e5)

        # Get initial guess from arclamp wavelength solution
        pars0 = np.zeros(npoly+3,float)
        pars0[len(pars0)-7:] = wavecal['a'][3].data[:,row]
        

        # use curve_fit to optimize parameters
        try :
            if verbose:
                print('row: ', row, 'nlines: ', len(thisrow), 'gd: ', len(gd))
                print(pars0)
            popt,pcov = curve_fit(wave.func_multi_poly,x[:,gd],y[gd],p0=pars0,bounds=bounds)
            pars = copy.copy(popt)
            res = y-wave.func_multi_poly(x,*pars)
            rms = np.sqrt(np.mean(res[gd]**2))
            if verbose:
                print('res: ',len(gd),np.median(res),np.median(np.abs(res)),res[gd].std())
                print(pars)
        except :
            print('Solution failed for row: ', row)
            import pdb; pdb.set_trace()
            popt = pars*0.
            rms = 999999.

        # Save the parameters
        allpars[0:npoly,row] = popt[0:npoly]
        allrms[row] = rms

        xglobal = np.zeros(len(thisrow),float)
        g1,=np.where(x[1,:]==1)
        g2,=np.where(x[1,:]==2)
        g3,=np.where(x[1,:]==3)
        xglobal[g1] = x[0,g1]-1023.5+pars[npoly]
        xglobal[g2] = x[0,g2]-1023.5+pars[npoly+1]
        xglobal[g3] = x[0,g3]-1023.5+pars[npoly+2]

        # there is obvious structure in the residuals
        # fitting all lines from just ONE chip at a time gives much lower rms
        # -should I be fitting with local X values and higher order?
        # -should the original arclamp wavelength solutions use a higher order
        #   cubic seems very low
        
        figfile = 'wave_resid_fiber'+str(row)+'.png'
        matplotlib.use('Agg')
        fig,ax = plt.subplots()
        plt.scatter(xglobal[gd],y[gd]-wave.func_multi_poly(x[:,gd],*pars))
        plt.ylim([-0.02,0.02])
        plt.xlabel('Xglobal (pix)')
        plt.ylabel('Residuals (A)')
        plt.title('Residual to global fit - fiber='+str(row))
        plt.savefig(figfile,bbox_inches='tight')
        print('Saving figure to ',figfile)

        import pdb; pdb.set_trace()

        #waves[chip][row,:] = func_multi_poly(x,*allpars[:,row],npoly=4)

    import pdb; pdb.set_trace()

def frame2mjd(frame) :
    """ Get MJD from frame number """
    mjd = 55562+int(frame//10000)
    return mjd

def mjd2day(mjd) :
    """ Get frame day number from MJD """
    daynum = int(mjd)-55562
    return daynum
