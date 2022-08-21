#!/usr/bin/env python

import os
import sys
import time
import numpy as np
from ..utils import plan,apload,platedata,utils
from . import psf,wave
from dlnpyutils import utils as dln
from astropy.io import fits
import subprocess
import getpass

global savedepsf, savedepsffiles, epsfchip 
try:
    dum = len(savedepsfiles)
except: # initialize if needed
    savedepsffiles = [None,None,None]  # initialize if needed 
    epsfchip = [None,None,None]

BADERR = 1.0000000e+10

def errout(data):
    """ Errout sets the value to output for the error for bad pixels """
    bad = ((data==BADERR) | (data <= 0) | (np.isfinite(data)==False))
    if np.sum(bad)>0:
        data[bad] = BADERR
    return data
    
def ap2dproc(inpfile,psffile,extract_type=1,apred=None,telescope=None,load=None,
             modelpsffile=None,outdir=None,clobber=False,fixbadpix=False,
             fluxcalfile=None,responsefile=None,wavefile=None,skywave=False,
             plugmap=0,highrej=7,lowrej=10,recenterfit=False,recenterln2=False,fitsigma=False,
             refpixzero=False,outlong=False,nowrite=False,npolyback=0,
             chips=[0,1,2],fibers=None,compress=False,verbose=False,
             silent=False,unlock=False):
    """
    This program extracts a 2D APOGEE image.
    This is called from AP2D

    Parameters
    ----------
    inpfile : str
        The name of the 2D APOGEE file.  This
          should be the directory and the ID8
          concatenated.
    psffile : str
        The name of the calibration PSF file to
          use. This should also be the directory
          and "base" name or ID concatenated.
    extract_type : int, optional
        The extraction method to use:
          1-Boxcar extraction (the default)
          2-PSF image extraction
          3-Gaussian PSF fitting extraction
          4-Jon's Empirical PSF extraction
          5-Model PSF extraction
    apred : str, optional
        The APOGEE reduction version, e.g. 'daily'.
    telescope : str, optional
        The telescope, e.g. 'apo25m', 'lco25m'.
    load : ApLoad
        The ApLoad object.  Either this must be input or
          apred and telescope
    modelpsffile : str
        The name of the Model PSF calibration file to
          use. This should also be the directory
          and "base" name or ID concatenated.
    outdir : str, optional
        The output directory.  By default the 1D extracted
          files are written to the same directory that
          the input 2D files are in.
    fluxcalfile : str, optional
        The name of the relative flux calibration file to use.
          This should also be the directory and "base" name
          or ID concatenated.
    wavefile : str, optonal
        The name of the wavelength calibration to use to
          add wavelengths to the output file.  This should
          also be the directory and "base" name or ID concatenated.
    skywave : boolean,, optional
          To enable a pixel-shift to wavelength solution based on sky lines
    plugmap : str, optional
          To specify a plugmap for the sky-line wavelength solution.
            if plugmap is given, only use SKY fibers for the correction.
    highrej : int, optional
        High rejection threshold for Gaussian PSF fitting
          The default is 7.
    lowrej : int, optional
        Low rejection threshold for Gaussian PSF fitting
          The default is 10.
    npolyback: int, optional
        The number of polynomial coeffiecients to use for
          the background.  Only for extract_type=3 for now.
          The default is npolyback=0.
    recenterfit : boolean, optional
           Recenter the traces/PSF with one constant offset for
            each chip.  The shift is found empirically from
            the image itself.  Default is False.
    recenterln2 : boolean, optional
           Recenter the traces/PSF with one constant offset for
             all chip.  The shift is determined by the LN2LEVEL
             header value in the image compared to the PSF file
             and an empirically derived relation between the
             LN2LEVEL and fiber trace shifts.  The LN2 level
             (due to gravity) slightly warps the instrument and
             shifts the traces (by fractions of a pixel).
             Default is False.
    refpixzero : boolean, optional
           Set the image zeropoint using the reference pixels.
             Default is False.
    fibers : list, optional
        Array of fibers to extract (0 for first fiber).  Default
          is all fibers.
    chips : list, optional
        Array of chips to use (0 for first chip).  Default is [0,1,2].
    fitsigma : boolean, optional
        Allow the sigma to vary for extract_type=3.  Default is False.
    fixbadpix : boolean, optional
        Fix bad pixels using 2D interpolation of neighboring
            pixels.  Default is False.
    outlong : boolean, optional
        The output files should use LONG type intead of FLOAT.
          This actually takes up the same amount of space, but
          this can be losslessly compressed with FPACK.  Default is False.
    nowrite : boolean, optional
        Don't write the output to files.  Default is False.
    clobber : boolean, optional
        Overwrite existing files.  Default is False.
    verbose : boolean, optional
        Print a lot of information to the screen.  Default is False.
    silent : boolean, optional
        Don't print anything to the screen. Default is False.
    unlock : boolean, optional
        Delete lock file and start fresh.  Default is False.
    
    Returns
    -------
    1D extracted spectra are output.  One file for each frame.
    output : table
        Structure with the extracted spectra for all three chips.
    outmodel : table
        Structure with the model of the 2D image for all three chips.

    Example
    -------
    output,outmodel = ap2dproc(inpfile,tracefile,outdir,1)

    Written by D.Nidever  July 2010
    Translated to python by D.Nidever  Feb 2022  
    """
     
    # Output directory
    if outdir is None:
        outdir = os.path.dirname(inpfile)+'/' 
    if outdir.endswith('/')==False:
        outdir += '/'
     
    chiptag = ['a','b','c'] 
    localdir = os.environ['APOGEE_LOCALDIR']

    # Need load or apred+telescope
    if apred is None and telescope is None and load is None:
        raise ValueError(' Must input load or apred+telescope')
    if load is None:
        load = apload.ApLoad(apred=apred,telescope=telescope)

    # outdir must be a string outdir must be a string 
    #if size(outdir,/type) != 7:if size(outdir,/type) != 7: 
    #  print('outdir must be a string'  print('outdir must be a string' 
    #  return  return 

    # Does the output directory exist?
    if os.path.exists(outdir)==False: 
        if not silent:
            print('')
            print('creating ',outdir)
        os.makedirs(outdir)
     
    # Chips to extract
    if len(chips) > 3 or np.min(chips)<0 or np.max(chips)>2: 
        raise ValueError( 'chips must have <=3 elements with values [0-2].')
     
    # Fibers to extract
    if fibers is not None:
        if len(fibers) > 300 or np.min(fibers) < 0 or np.max(fibers) > 299 : 
            raise ValueError('fibers must have <=300 elements with values [0-299].')
     
    # make the filenames and check the files

    fdir = os.path.dirname(inpfile) 
    base = os.path.basename(inpfile) 
    if os.path.exists(fdir)==False:
        print('directory '+fdir+' not found')
        return [],[]
     
    baseframeid = '%8d' % int(base)
    files = [fdir+'/'+load.prefix+'2D-'+ch+'-'+baseframeid+'.fits' for ch in chiptag]
    exists = [os.path.exists(f) for f in files]
    framenum = int(base)
    #info = apfileinfo(files)
    #okay = (info.exists and info.sp2dfmt and info.allchips and ((info.naxis == 3) or (info.exten == 1))) 
    okay = np.sum(exists) == 3
    if not okay:
        if not silent: 
            print('halt: there is a problem with files: '+' '.join(files))
        import pdb; pdb.set_trace() 
        return [],[]
         
    # Get PSF info
    psf_dir = os.path.dirname(psffile) 
    psf_base = os.path.basename(psffile) 
    if os.path.exists(psf_dir) == False: 
        if not silent: 
            print('halt: psf directory '+psf_dir+' not found')
        import pdb; pdb.set_trace() 
        return [],[]
     
    psfframeid = '%8d' % int(psf_base)
    psffiles = load.filename('PSF',num=psfframeid,chips=True)
    psffiles = [psffiles.replace('PSF-','PSF-'+ch+'-') for ch in chiptag]
    epsffiles = load.filename('EPSF',num=psfframeid,chips=True)
    epsffiles = [epsffiles.replace('EPSF-','EPSF-'+ch+'-') for ch in chiptag]
    if load.exists('PSF',int(psfframeid))==False:
        if not silent: 
            print('halt: there is a problem with psf files: '+' '.join(psffiles))
        import pdb; pdb.set_trace() 
        return [],[]
     
    if not silent: 
        print('')
        print('Extracting file ',inpfile)
        print('--------------------------------------------------')
     

    # Parameters
    frame = load.ap2D(int(base))
    mjd5 = int(load.cmjd(int(base)))
    nreads = frame['a'][0].header['nread']
    if not silent: 
        print('mjd5 = ',str(mjd5))
         
    # Check header
    head = fits.getheader(files[0],0)
         
    # determine file type determine file type 
    #-------------------------------------------- 
    # dark - should be processed with  dark - should be processed with 
    # flat flat 
    # lamps lamps 
    # object frame object frame 
    #obstype = sxpar(head,'obstype',count=nobs)obstype = sxpar(head,'obstype',count=nobs) 
    imagetyp = head.get('imagetyp')
    if imagetyp is None:
        if not silent: 
            print('no imagetyp keyword found for '+baseframeid)
     
    #obstype = strlowcase(str(obstype,2))obstype = strlowcase(str(obstype,2)) 
    imagetyp = str(imagetyp).lower()
     
    # Double-check the flux calibration file
    if fluxcalfile is not None: 
        fluxcalfiles = [os.path.dirname(fluxcalfile)+'/'+load.prefix+'Flux-'+ch+'-'+os.path.basename(fluxcalfile)+'.fits' for ch in chiptag]
        ftest = [os.path.exists(f) for f in fluxcalfiles]
        if np.sum(ftest) < 3: 
            if not silent: 
                print('halt: problems with flux calibration file '+fluxcalfile)
            import pdb; pdb.set_trace() 
            return [],[]
         
    # Double-check the response calibration file
    if responsefile is not None: 
        responsefiles = [os.path.dirname(responsefile)+'/'+load.prefix+'Response-'+ch+'-'+os.path.basename(responsefile)+'.fits' for ch in chiptag]
        ftest = [os.path.exists(f) for f in responsefiles]
        if np.sum(ftest) < 3: 
            if not silent: 
                print('halt: problems with response calibration file '+responsefile)
            import pdb; pdb.set_trace() 
            return [],[]
         
    # Double-check the wave calibration file
    if wavefile is not None: 
        wavefiles = [os.path.dirname(wavefile)+'/'+load.prefix+'Wave-'+ch+'-'+os.path.basename(wavefile)+'.fits' for ch in chiptag]
        wtest = [os.path.exists(f) for f in wavefiles]
        if np.sum(wtest) < 3: 
            if not silent: 
                print('halt: problems with wavelength file '+wavefile)
            import pdb; pdb.set_trace() 
            return [],[]
         
    # Wait if another process is working on this
    lockfile = outdir+load.prefix+'1D-'+str(framenum) # lock file
    if localdir: 
        lockfile = localdir+'/'+load.prefix+'1D-'+str(framenum)+'.lock' 
    else: 
        lockfile = outdir+load.prefix+'1D-'+str(framenum)+'.lock' 
    
    if not unlock and not clobber:
        while os.path.exists(lockfile):
            print('Waiting for lockfile '+lockfile)
            time.sleep(10)
    else: 
        if os.path.exists(lockfile): 
            os.remove(lockfile)

    if os.path.exists(os.path.dirname(lockfile))==False:
        os.mkdirs(os.path.dirname(lockfile))
    open(lockfile,'w').close()
                         
    # Since final ap1dwavecal requires simultaneous fit of all three chips, and
    #  this required final output to be put off until after all chips are done,
    #  all 3 need to be done here if any at all, so that data from all chips is loaded
    # Output files
    outfiles = [outdir+load.prefix+'1D-'+ch+'-'+str(framenum)+'.fits' for ch in chiptag]  # output file
    outtest = [os.path.exists(f) for f in outfiles]
    if np.sum(outtest)==3 and not clobber:
        print(outdir+load.prefix+'1D-'+str(framenum)+'.fits already exists and clobber not set')
        if os.path.exists(lockfile): os.remove(lockfile)
        return [],[]
         
    #--------------------------------
    # Looping through the three chips
    #--------------------------------
    output = None
    outmodel = None
    outstr = None
    ifirst = 0 
    for i in range(len(chips)): 
        t1 = time.time()
        ichip = chips[i]   # chip index, 0-first chip
        ifile = files[ichip] 
             
        # The chip structure
        frame1 = frame[chiptag[ichip]]
        chstr = {'header':frame1[0].header,'flux':frame1[1].data,'err':frame1[2].data,
                 'mask':frame1[3].data}
             
        # Chip trace filename
        ipsffile = psffiles[ichip] 
        iepsffile = epsffiles[ichip] 
             
        # Cutput file
        outfile = outdir+load.prefix+'1D-'+chiptag[ichip]+'-'+str(framenum)+'.fits'  # output file output file 
             
        if not silent: 
            if i > 0 : 
                print('')
            print(' processing chip '+chiptag[ichip]+' - '+os.path.basename(ifile))
            print('  psf file = '+ipsffile)
         
             
        # Fix the bad pixels and "unfixable" pixels
        #------------------------------------------
        if fixbadpix:
            chstr = ap2dproc_fixpix(chstr)
             
        ###################################################################
        # Need to remove the littrow ghost and secondary ghost here!!!!!!!!
        ###################################################################
             
        # Restore the trace structure
        tracestr = fits.getdata(ipsffile,1)
             
        # Fibers to extract
        if fibers is not None: 
            if max(fibers) > len(tracestr)-1: 
                error = 'max(fibers) is larger than the number of fibers in psf file.' 
                if not silent: 
                    print('halt: '+error)
                import pdb; pdb.set_trace() 
                return 
             
        # Measuring the trace shift
        if recenterfit:
            im = chstr['flux']
            sz = im.shape
            npix = sz[1] 
            nfibers = len(tracestr) 
            # The red chip has problems on the left side,
            #  so use columns farther to the right
            if ichip == 0: 
                xmid = npix*0.75 
            else: 
                xmid = npix*0.5 
                 
            medspec = np.median(im[xmid-50:xmid+50,:],dim=1) 
            gdpix , = np.where(medspec > 0.5*max(medspec),ngdpix) 
            if ngdpix <= 20: 
                # We're probably trying to process a dark as object or flat
                # I'm not sure if 20 is the right number but seems to work with darks
                if not silent: 
                    print('no signal was seen on any fibers for chip ',ichip)
                xshift = 0.0
            else: 
                medht = np.median(medspec[gdpix]) > 0.5*max(medspec) 
                         
                tpar = fltarr(nfibers*3) 
                yfib = fltarr(nfibers) 
                for l in range(nfibers): 
                    yfib[l]=poly(xmid,tracestr[l].coef) 
                     
                tpar[0:3*nfibers-3:3] = medht 
                #tpar[1:3*nfibers-2:3] = xsol[xmid,:]tpar[1:3*nfibers-2:3] = xsol[xmid,:] 
                tpar[1:3*nfibers-2:3] = yfib 
                #tpar[2:3*nfibers-1:3] = np.median(sigma2[xmid,:])tpar[2:3*nfibers-1:3] = np.median(sigma2[xmid,:]) 
                tpar[2:3*nfibers-1:3] = 1.0 #1.51.5 
                x = findgen(npix) 
                temp = gfunc(x,tpar) 
                mask1d = int(medspec > 0.5*max(medspec)) 
                #xcorlb,temp,medspec,20,xsh,mask=mask1dxcorlb,temp,medspec,20,xsh,mask=mask1d 
                         
                lag = findgen(9)-4 
                xc = c_correlate(temp,medspec*mask1d,lag) 
                bestind = first_el(maxloc(xc)) 
                fitlo = (bestind-2) > 0 
                fithi = (bestind+2) < 20 
                estimates = [xc[bestind],lag[bestind],1,np.median(xc)] 
                yfit = mpfitpeak(lag[fitlo:fithi],xc[fitlo:fithi],par,nterms=4,gaussian=True,positive=True,estimates=estimates) 
                xshift = par[1] 
                 
                if not silent: 
                    print('recentering shift = %.3f ' % xshift)
                    
                # this is an ADDITIVE offset!
                 
        # Calculate the trace shift ln2level header values
        if recenterln2: 
            head_psf = headfits(ipsffile,exten=0) 
            ln2level_psf = head_psf['ln2level']
            ln2level_im = chstr['header']['ln2level']
                    
            if nln2level_psf > 0 and nln2level_im > 0:
                # The slope of trace shift vs. ln2level is (from green chip):  0.0117597
                # Fits from check_traceshift.def 
                # linear: coef=[ -1.02611, 0.0117597]
                # quadratic:  coef=[-3.33460, 0.0613117, -0.000265449]
                # A higher LN2LEVEL shifts the fibers DOWNWARDS
                xshift = (ln2level_im - ln2level_psf) * (-0.0117597) 
                if not silent : 
                    print('Recentering shift = %.3f' % xshift)

                # this is an ADDITIVE offset!
                         
            # Don't have ln2levels
            else: 
                if nln2level_psf == 0 and not silent: 
                    print('Do NOT have header LN2LEVEL for PSF exposure')
                if nln2level_im == 0 and not silent: 
                    print('Do NOT have header LN2LEVEL for this exposure')
                if not silent: 
                    print('CANNOT calculate fiber shift from LN2LEVEL in headers')
                 
        # Reset the zeropoint threshold using the reference pixels
        if refpixzero: 
            medref = np.median( [ chstr.flux[:,0:3], transpose(chstr.flux[0:3,:]), chstr.flux[:,2044:2047], transpose(chstr.flux[2044:2047,:]) ]) 
            if not silent: 
                print('setting image zeropoint using reference pixels.  subtracting ',str(medref))
                chstr['flux'] -= medref 
                 
        # Initialize the output header
        #-----------------------------
        head = chstr['header']
        head['LONGSTRN'] = 'OGIP 1.0'    # allows us to use long/continued strings     
        head['PSFFILE'] = ipsffile,' PSF file used' 
        leadstr = 'AP2D: ' 
        pyvers = sys.version.split()[0]
        head['V_APRED'] = plan.getgitvers(),'APOGEE software version' 
        head['APRED'] = load.apred,'APOGEE Reduction version' 
        head['HISTORY'] = leadstr+time.asctime()
        import socket
        #head['HISTORY'] = leadstr+os.getlogin()+' on '+socket.gethostname()
        head['HISTORY'] = leadstr+getpass.getuser()+' on '+socket.gethostname()
        import platform
        head['HISTORY'] = leadstr+'Python '+pyvers+' '+platform.system()+' '+platform.release()+' '+platform.architecture()[0]
        # add reduction pipeline version to the header
        head['HISTORY'] = leadstr+' APOGEE Reduction Pipeline Version: '+load.apred
        head['HISTORY'] = leadstr+'Output file:'
        head['HISTORY'] = leadstr+' HDU1 - Image (ADU)'
        head['HISTORY'] = leadstr+' HDU2 - Error (ADU)'
        if (extract_type == 1): 
            head['HISTORY'] = leadstr+' HDU3 - Flag mask (bitwise OR combined)'
            head['HISTORY'] = leadstr+'        1 - bad pixels'
            head['HISTORY'] = leadstr+'        2 - cosmic ray'
            head['HISTORY'] = leadstr+'        4 - saturated'
            head['HISTORY'] = leadstr+'        8 - unfixable'
        else: 
            head['HISTORY'] = leadstr+' HDU3 - Flag mask'
            head['HISTORY'] = leadstr+'        0 - good pixels'
            head['HISTORY'] = leadstr+'        1 - bad pixels'
        if wavefile is not None:
            head['HISTORY'] = leadstr+' HDU4 - Wavelengths (Ang)'
            head['HISTORY'] = leadstr+' HDU5 - Wavelength coefficients'
             
        outstr = None
        ymodel = None
                 
        # Extraction type
        #----------------
                     
        # Boxcar extraction
        #------------------
        if extract_type==1:
            if not silent: 
                print('Using Boxcar Extraction')

            raise ValueError('Not Translated yet')
                 
            # Update header
            head['HISTORY'] = leadstr+'Extract_type=1 - Using Boxcar Extraction'
            head['EXTRTYPE'] = 1,'Extraction Type' 
                     
            # Recenter, shift the traces recenter, shift the traces 
            if recenterfit or recenterln2: 
                tracestr['coef'][0] += xshift 
                tracestr['gaussy'] += xshift 
                if recenterfit and not recenterln2: 
                    head['HISTORY'] = leadstr+' /RECENTERFIT set, shifting traces by %0.3f' % xshift
                if keyword_set(recenterln2) : 
                    head['HISTORY'] = leadstr+' /RECENTERLN2 set, shifting traces by %0.3f' % xshift
                     
            # Extract the fibers
            outstr = apextract(chstr,tracestr,fibers=fibers)
            
        # PSF image extraction
        #---------------------
        elif extract_type==2:
            if not silent: 
                print('Using PSF Image Extraction')
             
            raise ValueError('Not Translated yet')

            # Load the PSF image
            psfim,head_psfim = fits.getdata(ipsffile,2,header=True)
                 
            # Update header
            head['HISTORY'] = leadstr+'Extract_type=2 - Using PSF Image extraction'
            head['EXTRTYPE'] = 2,'Extraction Type' 
                 
            # Recenter, shift the traces and the psf image
            if keyword_set(recenterfit) or keyword_set(recenterln2): 
                tracestr.coef[0] += xshift 
                tracestr.gaussy += xshift 
                psfim0 = psfim 
                psfim = imdrizzle(psfim0,0.0,xshift)  # shift the image with imdrizzle
                if keyword_set(recenterfit) and not keyword_set(recenterln2) : 
                    head['HISTORY'] = leadstr+' /RECENTERFIT set, shifting traces by %0.3f' % xshift
                if keyword_set(recenterln2) : 
                    head['HISTORY'] = leadstr+' /RECENTERLN2 set, shifting traces by %0.3f' % xshift
                 
            # Extract the fibers
            outstr,ymodel = apextractpsf(chstr,tracestr,psfim,model=ymodel,fibers=fibers)
             
        # Gaussian psf fitting
        #---------------------
        #   Maybe use the idlspec2d extraction code for this
        elif extract_type==3:
            if not silent: 
                print('Esing Gaussian PSF fitting Extraction')

            raise ValueError('Not Translated yet')
            
            # Update header
            head['HISTORY'] = leadstr+'Extract_type=3 - Using Gaussian PSF fitting Extraction'
            head['EXTRTYPE'] = 3,'Extraction Type' 
             
            # The idlspec2d programs expect the fibers to run along the y
            # transposing the arrays for now
             
            # Get the idlspec2d-style trace and widthset information
            tset_coeff,tset_head = fits.getdata(ipsffile,3,header=True)
            tset = {'func':str(tset_head['func']),'xmin':tset_head['xmin'],
                    'xmax':tset_head['xmax'],'coeff':tset_coeff} 
            wset_coeff,wset_head = fits.getdata(ipsffile,4,header=True)
            widthset = {'func':str(wset_head['func']),'xmin':wset_head['xmin'],
                        'xmax':wset_head['xmax'],'coeff':wset_coeff} 
            proftype = wset_head.get('proftype') 
             
            # Get the trace and sigma arrays
            ycen,xsol,xx,sigma2 = None,None,None,None
            traceset2xy, tset, ycen, xsol 
            traceset2xy, widthset, xx, sigma2 
             
            # Get the images ready
            img = frame[ichip]['flux'].astype(float).T
            ivar = ( 1.0/frame[ichip]['err'].astype(float)**2 ).T
            mask = frame[ichip]['mask'].T
            mask = 1-( ((mask and 1) == 1) or ((mask and 4) == 4) or ((mask and 8) == 8) ) 
             
            # Recenter the traces
            if recenterfit or recenterln2: 
                # Need to add this to the traces
                xsol += xshift 
                if recenterfit and not recenterln2: 
                    head['HISTORY'] = leadstr+' /recenterfit set, shifting traces by %0.3f' % xshift
                if recenterln2: 
                    head['HISTORY'] = leadstr+' /recenterln2 set, shifting traces by %0.3f' % xshift
             
            #-------------------------------------------------------------
            # Extract the spectra
            #-------------------------------------------------------------
            # since the gaussian is not a good fit use a lower
            #  order background
            npoly = npolyback 
            wfixed = [1]   # keep the sigmas fixed
            if fitsigma:
                wfixed = [1,1]  # fit sigma 
         
            # Only extract fibers
            if len(fibers) > 0: 
                xsol = xsol[:,fibers] 
                sigma2 = sigma2[:,fibers] 
         
            #splog, 'extracting arc'splog, 'extracting arc' 
            ymodel = ap_extract_image(img, ivar, xsol, sigma2,
                                   flux, fluxivar, proftype=proftype,
                                   wfixed=wfixed, highrej=highrej,
                                   lowrej=lowrej, npoly=npoly, relative=1,
                                   reject=[0.1, 0.6, 0.6],
                                   mask=mask,chisq=chisq)
             
            # transpose the model
            ymodel = ymodel.T
             
            # Create outstr
            #  bad pixels have fluxivar=0, they are given high err
            #  mask make it: 0-good, 1-bad
            outstr = {'flux':flux, 'err':1/(np.sqrt(np.maximum(fluxivar,1e-12))), 'mask':fluxivar.astype(int)*0} 
            outstr['mask'] = (fluxivar == 0)  # pixels with fluxivar=0 are bad
            # negative pixels
            #bd , = np.where(outstr.flux < 0,nbd)
            #if nbd > 0:
            #  outstr.flux[bd] = 0
            #  outstr.err[bd] = 1e6
            #  outstr.mask[bd] = 1  # maybe give this a different value
            # 
            # Fix reference pixels
            outstr['flux'][0:4,:] = 0 
            outstr['flux'][2040:2048,:] = 0 
            outstr['err'][0:4,:] = BADERR 
            outstr['err'][2040:2048,:] = BADERR 
            outstr['mask'][0:4,:] = 1 
            outstr['mask'][2040:2048,:] = 1 
         
        # Empirical PSF Extraction
        #-------------------------
        if extract_type==4:
            if not silent: 
                print('Using Empirical PSF extraction')
            # Copied from holtz/approcess.pro
            if epsffiles[ichip] != savedepsffiles[ichip]:
                # Load Empirical PSF data
                if not silent:
                    print('Loading empirical PSF data from '+iepsffile)
                epsf = psf.loadepsf(iepsffile)
                # Save for later
                epsfchip[ichip] = epsf
                savedepsffiles[ichip] = epsffiles[ichip] 
            else:
                epsf = epsfchip[ichip]
            if fibers is None:
                #fibers = [e['fiber'] for e in epsf]
                fibers = np.arange(len(epsf))
         
            # Update header
            head['HISTORY'] = leadstr+'Extract_type=4 - Using Empirical PSF Extraction'
            head['EXTRTYPE'] = 4,'Extraction Type' 
         
            # Recenter, shift the traces and the psf image
            if recenterfit or recenterln2: 
                # Shift the image with imdrizzle
                for l in range(len(epsf)): 
                    epsf[l]['img'] = imdrizzle(epsf[l]['img'],0.0,xshift) 
         
                if recenterfit and not recenterln2: 
                    head['HISTORY'] = leadstr+' /recenterfit set, shifting traces by %.3f' % xshift
                if recenterln2: 
                    head['HISTORY'] = leadstr+' /recenterln2 set, shifting traces by %.3f' % xshift

            outstr,back,ymodel = psf.extract(chstr,epsf,outstr,scat=True)

     
        # Model PSF extraction
        #---------------------
        elif extract_type==5:
            if not silent: 
                print('Using Model PSF extraction')
         
            # Update header
            head['HISTORY'] = leadstr+'Extract_type=5 - Model PSF Extraction'
            head['EXTRTYPE'] = 5,'Extraction Type' 

            if modelpsffile is None:
                raise ValueError('Need Model PSF file for Model PSF Extraction')
            modelpsfid = os.path.basename(modelpsffile)
            modelpsffile1 = load.filename('PSFModel',num=modelpsfid,chips=True).replace('PSFModel-','PSFModel-'+chiptag[i]+'-')
            head['HISTORY'] = 'Model PSF file: '+modelpsffile1
            tracefile = load.filename('ETrace',num=psfframeid,chips=True).replace('ETrace-','ETrace-'+chiptag[i]+'-')
            chstr['header'] = head
            epsffile1 = epsffiles[i]
            outstr,back,ymodel = psf.extractwing(chstr,modelpsffile1,epsffile1,tracefile)
            head = outstr['header']
 
        t2 = time.time()
        #import pdb; pdb.set_trace()
 
 
        # Do the fiber-to-fiber throughput corrections and relative
        #   flux calibration
        #----------------------------------------------------------
        if fluxcalfile is not None:
            # restore the relative flux calibration correction file
            if not silent: 
                print('Flux calibrating with ',os.path.dirname(fluxcalfile)+'/'+load.prefix+'Flux-'+os.path.basename(fluxcalfile))
            fluxcalfiles = [os.path.dirname(fluxcalfile)+'/'+load.prefix+'Flux-'+ch+'-'+os.path.basename(fluxcalfile)+'.fits' for ch in chiptag]
            fluxcal,fluxcal_head = fits.getdata(fluxcalfiles[ichip],header=True)
            outstr['flux'] /= fluxcal.T          # correct flux
            bderr = (outstr['err'] == BADERR) 
            nbd = np.sum(bderr)
            outstr['err'] /= fluxcal.T           # correct error
            if np.sum(bderr) > 0: 
                outstr['err'][bderr] = BADERR 
            bd = (np.isfinite(outstr['flux']) == False)
            nbd = np.sum(bd)
            if nbd > 0: 
                outstr['flux'][bd] = 0. 
                outstr['err'][bd] = BADERR 
                outstr['mask'][bd] = 1 
 
            # Update header
            head['HISTORY'] = leadstr+'Flux Calibrating the spectra with:'
            head['HISTORY'] = leadstr+fluxcalfiles[ichip]
            head['FLUXFILE'] = fluxcalfile,' Flux Calibration file used' 
 
        # Response curve calibration
        #---------------------------
        if responsefile is not None: 
            # Restore the relative flux calibration correction file
            if not silent: 
                print('response calibrating with ',os.path.dirname(responsefile)+'/'+load.prefix+'flux-'+os.path.basename(responsefile))
            responsefiles = [os.path.dirname(responsefile)+'/'+load.prefix+'response-'+ch+'-'+os.path.basename(responsefile)+'.fits' for ch in chiptag]
            response,response_head = fits.getdata(responsefiles[ichip],header=True)
 
            sz = outstr['flux'].shape
            outstr['flux'] *= response.reshape(-1,1) + np.zeros(sz[1])      # correct flux
            bderr = (outstr['err'] == BADERR)
            outstr['err'] *= response.reshape(-1,1) + np.zeros(sz[1])       # correct error
            if np.sum(bderr) > 0: 
                outstr['err'][bderr] = BADERR 
 
            # Update header
            head['HISTORY'] = leadstr+'Applying response function:'
            head['HISTORY'] = leadstr+responsefiles[ichip]
            head['RESPFILE'] = responsefile,' Response file used' 
 
        # Adding wavelengths
        #-------------------
        if wavefile is not None:
            wavefiles = os.path.dirname(wavefile)+'/'+load.prefix+'wave-'+chiptag+'-'+os.path.basename(wavefile)+'.fits' 
            if not silent: 
                print('Adding wavelengths from ',os.path.dirname(wavefile)+'/'+load.prefix+'wave-'+os.path.basename(wavefile))
            # Get the wavelength calibration data
            wcoef,whead = fits.getdata(wavefiles[ichip],1,header=True)
            wim,whead2 = fits.getdata(wavefiles[ichip],2,header=True)
            # this is now fixed in the apwave files this is now fixed in the apwave files 
            #wim = transpose(wim)  # want it [npix, nfibers]wim = transpose(wim)   want it [npix, nfibers] 
 
            head['HISTORY'] = leadstr+'Adding wavelengths from'
            head['HISTORY'] = leadstr+wavefiles[ichip]
            head['WAVEFILE'] = wavefile,' Wavelength Calibration file' 
            head['WAVEHDU'] = 5,' Wavelength coef HDU' 
    
        # Add header to structure
        outstr['header'] = head
 
        # Add fibers to structure
        if fibers is not None:
            outstr['fibers'] = fibers
 
        # Output the 2D model spectrum
        if ymodel is not None:
            modelfile = outdir+load.prefix+'2Dmodel-'+chiptag[ichip]+'-'+str(framenum)+'.fits'  # model output file
            if not silent: 
                print('Writing 2D model to: ',modelfile)
            hdu = fits.HDUList()
            hdu.append(fits.PrimaryHDU(ymodel.astype(np.float32)))
            hdu.writeto(modelfile,overwrite=True)
            hdu.close()
            #    # compress model and 2D image done in ap2d
            #    if keyword_set(compress):
            #      os.remove(modelfile+'.fz',/allow_nonexistent
            #      spawn,'fpack -d -y '+modelfile 
            #      origfile = outdir+load.prefix+'2D-'+chiptag[ichip]+'-'+framenum+'.fits'
            #      if os.path.exists(origfile):
            #        os.remove(origfile+'.fz',/allow_nonexistent
            #        spawn,'fpack -d -y '+origfile

        # Add to output structure
        if ifirst == 0: 
            output = {ichip:outstr}
            if ymodel is not None: 
                outmodel = {ichip:ymodel}
            ifirst = 1 
        else: 
            output[ichip] = outstr
            if ymodel is not None:
                outmodel[ichip] = ymodel

  
    # Now we have output structure with three chips, each with tags header, flux, err, mask
 
    # Add wavelength information to the frame structure
    #--------------------------------------------------
    # Loop through the chips
    if wavefile is not None:
        for k in range(2+1): 
            # Get the wavelength calibration data
            wcoef,whead = fits.getdata(wavefiles[k],1,header=True)
            # Add to the chip structure
            chstr = output[k]
            chstr = create_struct(temporary(chstr),'filename',files[k],'wave_dir',outdir,'wavefile',wavefiles[k],'wcoef',wcoef) 
            # Now add this to the final frame structure
            if k == 0: 
                frame = create_struct('chip'+chiptag[k],chstr) 
            else: 
                frame = create_struct(frame,'chip'+chiptag[k],chstr) 
        del output  # free up memory
        if os.path.exists(outdir)==False:
            os.makedirs(outdir)
        plotfile = outdir+'/plots/pixshift-'+framenum 
        if skywave:
            #wave.getskywave(args.frameid,args.waveid,vers=args.apred,telescope=args.telescope)            
            frame_wave = wave.ap1dwavecal(frame,plugmap=plugmap,verbose=True,plot=True,pfile=plotfile)
        else: 
            #wave.getskywave(args.frameid,args.waveid,vers=args.apred,telescope=args.telescope)
            frame_wave = wave.ap1dwavecal(frame,verbose=True,noshift=True)
        del frame  # free up memory
    else:
        frame_wave = output 

 
    # Write output file
    #------------------
    if not nowrite:
 
        for i in range(len(chips)): 
            ichip = chips[i]   # chip index, 0-first chip
            # output file
            outfile = outdir+load.prefix+'1D-'+chiptag[ichip]+'-'+str(framenum)+'.fits'  # output file
            if not silent: 
                print('writing output to: ',outfile)
 
            if outlong and not silent:
                print('Saving flux/err as long instead of float')
 
            # HDU0 - header only
            hdu = fits.HDUList()
            hdu.append(fits.PrimaryHDU(header=head))
 
            # HDU1 - Flux
            flux = frame_wave[i]['flux']
            if outlong:
                flux = np.round(flux).astype(int)
            else:
                flux = flux.astype(np.float32)
            hdu.append(fits.ImageHDU(flux.T))
            hdu[1].header['CTYPE1'] = 'Pixel'
            hdu[1].header['CTYPE2'] = 'Fiber'
            hdu[1].header['BUNIT'] = 'Flux (ADU)'
 
            # HDU2 - error
            err = errout(frame_wave[i]['err']) 
            if outlong:
                err = np.round(err).astype(np.int32) 
            else:
                err = err.astype(np.float32)
            hdu.append(fits.ImageHDU(err.T))
            hdu[2].header['CTYPE1'] = 'Pixel'
            hdu[2].header['CTYPE2'] = 'Fiber'
            hdu[2].header['BUNIT'] = 'Error (ADU)'
 
            # HDU3 - mask
            mask = frame_wave[i]['mask']
            mask = mask.astype(np.int16)
            hdu.append(fits.ImageHDU(mask.T))
            hdu[3].header['CTYPE1'] = 'Pixel'
            hdu[3].header['CTYPE2'] = 'Fiber'
            if (extract_type == 1): 
                hdu[3].header['BUNIT'] = 'Flag Mask (bitwise)' 
                hdu[3].header['HISTORY'] = 'Explanation of BITWISE flag mask (OR combined)'
                hdu[3].header['HISTORY'] = ' 1 - bad pixels'
                hdu[3].header['HISTORY'] = ' 2 - cosmic ray'
                hdu[3].header['HISTORY'] = ' 4 - saturated'
                hdu[3].header['HISTORY'] = ' 8 - unfixable'
            else: 
                hdu[3].header['BUNIT'] = 'Flag Mask' 
                hdu[3].header['HISTORY'] = 'Explanation of flag mask'
                hdu[3].header['HISTORY'] = ' 0 - good pixels'
                hdu[3].header['HISTORY'] = ' 1 - bad pixels'
 
            if wavefile is not None:
                # HDU4 - Wavelengths
                wave = frame_wave[i]['wavelength']
                hdu.append(fits.ImageHDU(wave.T))
                hdu[4].header['CTYPE1'] = 'Pixel'
                hdu[4].header['CTYPE2'] = 'Fiber'
                hdu[4].header['BUNIT'] = 'Wavelength (Angstroms)' 
 
                # HDU5 - Wavelength solution coefficients [DOUBLE]
                #-------------------------------------------------
                wcoef = frame_wave[i]['wcoef'].astype(float)
                hdu.append(fits.ImageHDU(wcoef.T))
                hdu[5].header['CTYPE1'] = 'Pixel'
                hdu[5].header['CTYPE2'] = 'Parameters'
                hdu[5].header['BUNIT'] = 'Wavelength Coefficients'
                hdu[5].header['HISTORY'] = 'Wavelength Coefficients to be used with PIX2WAVE.PRO:'
                hdu[5].header['HISTORY'] = ' 1 Global additive pixel offset'
                hdu[5].header['HISTORY'] = ' 4 Sine Parameters'
                hdu[5].header['HISTORY'] = ' 7 Polynomial parameters (first is a zero-point offset'
                hdu[5].header['HISTORY'] = '                     in addition to the pixel offset)'

            # Write the data to disk
            hdu.writeto(outfile,overwrite=True)
            hdu.close()

    # Remove the lock file
    if os.path.exists(lockfile):
        os.remove(lockfile)
 
    if not silent:
        print('AP2DPROC finished')

    return frame_wave,outmodel


def ap2d(planfiles,verbose=False,clobber=False,exttype=4,mapper_data=None,
         calclobber=False,psflibrary=False,unlock=False):  
    """
    This program processes 2D APOGEE spectra.  It extracts the
    spectra.

    Parameters
    ----------
    planfiles : str or list
        Input list of plate plan files
    exttype : int
        Extraction type.  Default is 4.           
    mapper_data : str, optional
        Directory for mapper data.
    verbose : boolean, optional
        Print a lot of information to the screen.  Default is False.
    clobber : boolean, optional
        Overwrite existing files (ap1D).  Default is False.
    calclobber : boolean, optional
        Overwrite existing daily calibration files (apPSF, apFlux).
           Default is False.
    psflibrary : boolean, optional
        Use the PSF library.  Default is False.
    unlock : boolean, optional
        Delete lock file and start fresh.  Default is False.

    Returns
    -------
    1D extracted spectra are output.  Three files for each exposure.

    Example
    -------

    out = ap2d(planfile)

    Written by D.Nidever  Mar. 2010
    Modifications: J. Holtzman 2011+
    Translated to python by D. Nidever 2022
    """
          
    savedepsffiles = [' ',' ',' '] 
    epsfchip = 0 
 
    t0 = time.time() 
 
    nplanfiles = np.array(planfiles).size
    if nplanfiles==1 and type(planfiles) is not list:
        planfiles = [planfiles]

    print('' )
    print('Running AP2D')
    print('')
    print(str(nplanfiles)+' plan files')
 
    chiptag = ['a','b','c'] 
    wavefile,responsefile = None,None
 
    #-------------------------------------------
    # loop through the unique plate observations
    #-------------------------------------------
    for i in range(nplanfiles): 
        t1 = time.time() 
        planfile = planfiles[i] 
        print('' )
        print('=====================================================================')
        print(str(i+1)+'/'+str(nplanfiles)+'  processing plan file '+planfile)
        print('=====================================================================')
 
        # Load the plan file
        #-------------------
        print('')
        print('Plan file information:')
        planstr = plan.load(planfile,np=True)
        if planstr is None:
            continue
 
        load = apload.ApLoad(apred=planstr['apred_vers'],telescope=planstr['telescope'])
        logfile = load.filename('Diag',plate=planstr['plateid'],mjd=planstr['mjd'])


        # Make sure PSFID exists
        if 'psfid' not in planstr.keys():
            planstr['psfid'] = 0
        # Add PSFID tag to planstr APEXP structure
        if 'psfid' not in planstr['APEXP'].dtype.names:
            apexp = planstr['APEXP']
            apexp = dln.addcatcols(apexp,np.dtype([('psfid',int)]))
            planstr['APEXP'] = apexp

        if 'modelpsf' in planstr.keys():
            modelpsf = planstr['modelpsf']
        else:
            modelpsf = 0
 
        # Use domeflat/quartzflat PSF library
        #------------------------------------
        # if (1) no PSFID set in planfile or PSFID=0, or (2) psflibrary parameter
        # set in planfile, or (3) /psflibrary keyword is set.  
        if 'psflibrary' in planstr.keys():
            planpsflibrary = planstr['psflibrary']
        else: 
            planpsflibrary = None
        if (psflibrary or planstr['psfid']==0 or planpsflibrary) and modelpsf==0:
            usepsflibrary = True
        else:
            usepsflibrary = False
        if usepsflibrary:
            print('Using PSF library')
            # You can do "domeflattrace --mjdplate" where mjdplate could be
            # e.g. 59223-9244, or "domeflattrace --planfile", with absolute
            # path of planfile
            # force single domeflat if a short visit or domelibrary=='single'
            observatory = planstr['telescope'][0:3]
            if str(psflibrary) == 'single' or str(planpsflibrary) == 'single' or len(planstr['APEXP']) <= 3: 
                out = subprocess.check_output(['psflibrary',observatory,'--planfile',planfile,'--s'],shell=False)
            else: 
                out = subprocess.check_output(['psflibrary',observatory,'--planfile',planfile],shell=False)
            out = out.decode().split('\n')
            nout = len(out) 
            # parse the output parse the output 
            lo = dln.grep(out,'^PSF FLAT RESULTS:',index=True)[0]
            hi = np.where((np.char.array(out)=='') & (np.arange(nout) > lo))[0][0]
            if lo == -1 or hi == -1: 
                print('Problem running psflibrary for '+planfile+'. Skipping this planfile.')
                continue 
            outarr = [o.split() for o in out[lo+1:hi]]
            ims = np.array([int(o[0]) for o in outarr])
            psfflatims = np.array([int(o[1]) for o in outarr])
            # update planstr
            vals,ind1,ind2 = np.intersect1d(apexp['name'],ims,return_indices=True)
            planstr['APEXP']['psfid'][ind1] = psfflatims[ind2] 
            planstr['psfid'] = psfflatims[0]
        else: 
            planstr['APEXP']['psfid'] = planstr['psfid']
 
        # Don't extract dark frames
        if 'platetype' in planstr.keys():
            if planstr['platetype'] == 'dark' or planstr['platetype'] == 'intflat' :
                print('Not extracting Dark or internalflat frames')
                continue
 
        # Try to make the required calibration files (if not already made)
        # Then check if the calibration files exist
        #-------------------------------------------------
 
        # apPSF files
        if planstr['sparseid'] != 0:
            if load.exists('Sparse',num=planstr['sparseid']):
                print(load.filename('Sparse',num=planstr['sparseid'])+' already made')
            else:
                sout = subprocess.run(['makecal','--sparse',str(planstr['sparseid']),
                                       '--telescope',str(planstr['telescope'])],shell=False)
        if planstr['fiberid'] != 0: 
            if load.exists('ETrace',num=planstr['fiberid']):
                print(load.filename('ETrace',num=planstr['fiberid'],chips=True)+' already made')
            else:
                sout = subprocess.run(['makecal','--fiber',str(planstr['fiberid']),
                                       '--telescope',str(planstr['telescope'])],shell=False)
        if 'psfid' in planstr.keys():
            if load.exists('PSF',num=planstr['psfid']):
                print(load.filename('PSF',num=planstr['psfid'],chips=True)+' already made')
            else:
                cmd = ['makecal','--psf',str(planstr['psfid']),'--telescope',str(planstr['telescope'])]
                if calclobber:
                    cmd += ['--clobber']
                sout = subprocess.run(cmd,shell=False)
            tracefiles = load.filename('PSF',num=planstr['psfid'],chips=True)
            tracefiles = [tracefiles.replace('PSF-','PSF-'+ch+'-') for ch in chiptag]
            tracefile = os.path.dirname(tracefiles[0])+'/%8d' % int(planstr['psfid'])
            tracetest = [os.path.exists(t) for t in tracefiles]
            if np.sum(tracetest) != 3:
                bd1, = np.where(np.array(tracetest)==False)
                nbd1 = len(bd1)
                if nbd1>0:
                    print('halt: '+','.join(np.array(tracefiles)[bd1])+' not found')
                    import pdb; pdb.set_trace()
                for ichip in range(2+1): 
                    p = fits.getdata(tracefiles[ichip],1)
                    if len(p) != 300: 
                        print( 'halt: tracefile '+','.join(np.array(tracefiles)[ichip])+' does not have 300 traces')
 
        # apWave files : wavelength calibration
        waveid = int(planstr['waveid'])
        if 'platetype' in planstr.keys():
            if planstr['platetype'] == 'cal' or planstr['platetype'] == 'extra': 
                waveid = 0 
        if waveid > 0: 
            # This is now normally a dailywave with the MJD name  
            if int(waveid) < 1e7:
                wfile = load.filename('Wave',num=planstr['waveid'],chips=True)[0:-13]+str(planstr['waveid'])+'.fits'
                wavefiles = [wfile.replace('Wave-','Wave-'+ch+'-') for ch in chiptag]
                exists = [os.path.exists(f) for f in wavefiles]
                if np.sum(exists)==3:
                    print(wfile+' already made')
                else:
                    sout = subprocess.run(['makecal','--dailywave',str(waveid),
                                           '--telescope',str(planstr['telescope'])],shell=False)
            else:
                if load.exists('Wave',num=planstr['waveid']):
                    print(load.filename('Wave',num=planstr['waveid'],chips=True)+' already made')
                else:
                    sout = subprocess.run(['makecal','--multiwave',str(waveid),
                                           '--telescope',str(planstr['telescope'])],shell=False)

        # FPI calibration file
        if 'fpi' in planstr.keys():
            fpiid = int(planstr['fpi'])
        else: 
            fpiid = 0 
 
        # apFlux files : since individual frames are usually made per plate
        if int(planstr['fluxid']) != 0: 
            if load.exists('Flux',num=planstr['fluxid']):
                print(load.filename('Flux',num=planstr['fluxid'],chips=True)+' already made')
            else:
                #makecal,flux=planstr.fluxid,psf=planstr.psfid,clobber=calclobber 
                cmd = ['makecal','--flux',str(planstr['fluxid']),'--psf',str(planstr['psfid']),
                       '--telescope',str(planstr['telescope'])]
                if calclobber:
                    cmd += ['--clobber']
                sout = subprocess.run(cmd,shell=False)
            fluxfiles = load.filename('Flux',num=planstr['fluxid'],chips=True)
            fluxfiles = [fluxfiles.replace('Flux-','Flux-'+ch+'-') for ch in chiptag]
            fluxfile = os.path.dirname(fluxfiles[0])+'/%8d' % int(planstr['fluxid'])
            fluxtest = [os.path.exists(f) for f in fluxfiles]
            fludtest = True if np.sum(fluxtest)==3 else False
            if np.sum(fluxtest) != 3: 
                bd1, = np.where(np.array(fluxtest)==False)
                nbd1 = len(bd1)
                if nbd1 > 0: 
                    print('halt: '+str(np.array(fluxfiles)[bd1])+' not found')
                    import pdb; pdb.set_trace()
        else:
            fluxtest = False 
 
        # apResponse files
        #  these aren't used anymore
        if 'responseid' not in planstr.keys():
            planstr['responseid'] = 0
        if int(planstr['responseid']) != 0: 
            if load.exists('Response',num=planstr['responseid']):
                print(load.filename('Response',num=planstr['responseid'])+' exists already')
            else:
                sout = subprocess.run(['makecal','--response',str(planstr['responseid']),
                                       '--telescope',str(planstr['telescope'])],shell=False)
            responsefiles = load.filename('Response',num=planstr['responseid'],chips=True) 
            responsefiles = [tracefiles.replace('PSF-','PSF-'+ch+'-') for ch in chiptag]
            responsefile = os.path.dirname(responsefiles[0])+'/%8d' % planstr['responseid']
            responsetest = [os.path.exists(f) for f in responsefiles]
            if np.sum(responsetest) != 3:
                bd1 , = np.where(np.array(responsetest)==False)
                nbd1 = len(bd1)
                if nbd1 > 0 : 
                    print('halt: ',responsefiles[bd1],' not found')
                    import pdb; pdb.set_trace()
         
        # Model PSF files
        if 'modelpsf' in planstr.keys():
            if load.exists('PSFModel',num=planstr['modelpsf']):
                print(load.filename('PSFModel',num=planstr['modelpsf'],chips=True)+' exists already')
            else:
                sout = subprocess.run(['makecal','--modelpsf',str(planstr['responseid']),
                                       '--telescope',str(planstr['telescope'])],shell=False)
            print('Using Model PSF: '+str(modelpsf))
            modelpsffiles = load.filename('PSFModel',num=planstr['modelpsf'],chips=True)
            modelpsffiles = [modelpsffiles.replace('PSFModel-','PSFModel-'+ch+'-') for ch in chiptag]
            modelpsffile = os.path.dirname(modelpsffiles[0])+'/'+str(planstr['modelpsf'])
            modelpsftest = [os.path.exists(t) for t in modelpsffiles]
            exttype = 5
            if np.sum(modelpsftest) != 3:
                bd1, = np.where(np.array(modelpsftest)==False)
                nbd1 = len(bd1)
                if nbd1 > 0: 
                    print('halt: ',modelpsffiles[bd1],' not found')
                    import pdb; pdb.set_modelpsf()
        else:
            modelpsffile = None
 
        # Load the Plug Plate Map file
        #-----------------------------
        if 'platetype' in planstr.keys():
            if planstr['platetype'] == 'cal' or planstr['platetype'] == 'extra' or planstr['platetype'] == 'single': 
                plugmap = 0 
            else: 
                print('')
                print('plug map file information:')
                plugfile = planstr['plugmap']
                if 'fixfiberid' in planstr.keys():
                    fixfiberid = planstr['fixfiberid']
                if type(fixfiberid) is str and np.array(fixfiberid).size == 1: # null/none 
                    if (str(fixfiberid) == 'null' or str(fixfiberid).lower() == 'none'): 
                        fixfiberid = None
                if 'badfiberid' in planstr.keys():
                    badfiberid = planstr['badfiberid']
                if type(badfiberid) is str and np.array(badfiberid).size == 1: # null/none 
                    if (str(badfiberid) == 'null' or str(badfiberid).lower() == 'none'): 
                        badfiberid = None  # null/none 
                # we only need the information on sky fibers
                plugmap = platedata.getdata(planstr['plateid'],planstr['mjd'],load.apred,load.telescope,
                                            plugid=planstr['plugmap'],fixfiberid=fixfiberid,badfiberid=badfiberid,
                                            mapper_data=mapper_data,noobject=True)
                if plugmap is None:
                    print('halt: error with plugmap: ',plugfile)
                    import pdb; pdb.set_trace()
                plugmap['mjd'] = planstr['mjd']   # enter mjd from the plan file
 
        # Are there enough files
        nframes = len(planstr['APEXP']) 
        if nframes < 1: 
            print('no frames to process')
            continue
                                            
        # Process each frame
        #-------------------
        for j in range(nframes):
            # Get trace files
            tracefiles = load.filename('PSF',num=planstr['APEXP']['psfid'][i],chips=True)
            tracefiles = [tracefiles.replace('PSF-','PSF-'+ch+'-') for ch in chiptag]
            tracefile = os.path.dirname(tracefiles[0])+'/%8d' % planstr['APEXP']['psfid'][i]
            tracetest = [os.path.exists(t) for t in tracefiles]
            if np.sum(tracetest) != 3:
                bd1, = np.where(np.array(tracetest)==False)
                nbd1 = len(bd1)
                if nbd1 > 0: 
                    print('halt: ',tracefiles[bd1],' not found')
                    import pdb; pdb.set_trace()
                for ichip in range(3): 
                    p = fits.getdata(tracefiles[ichip],1)
                    if len(p) != 300: 
                        print( 'halt: tracefile ', tracefiles[ichip],' does not have 300 traces')
         
            # Make the filenames and check the files
            rawfiles = load.filename('R',num=planstr['APEXP']['name'][j],chips=True)
            rawfiles = [rawfiles.replace('R-','R-'+ch+'-') for ch in chiptag]
            framenum = planstr['APEXP']['name'][j]
            #rawinfo = apfileinfo(rawfiles)        # this returns useful info even if the files don't exist
            #framenum = rawinfo[0].fid8       # the frame number the frame number 
            files = load.filename('2D',num=framenum,chips=True)
            files = [files.replace('2D-','2D-'+ch+'-') for ch in chiptag]
            inpfile = os.path.dirname(files[0])+'/'+str(framenum)
            #info = apfileinfo(files) 
            okay = load.exists('R',num=planstr['APEXP']['name'][j]) and load.exists('2D',num=planstr['APEXP']['name'][j])
            #okay = (info.exists and info.sp2dfmt and info.allchips and (info.mjd5 == planstr.mjd) and 
            #        ((info.naxis == 3) or (info.exten == 1))) 
            if okay==False:
                print('halt: there is a problem with files: '+' '.join(files))
                import pdb; pdb.set_trace()
 
            print('') 
            print('-------------------------------------------')
            print(str(j+1)+'/'+str(nframes)+'  processing frame number >>'+str(framenum)+'<<')
            print('-------------------------------------------')
 
            # Run AP2DPROC
            if 'platetype' in planstr.keys():
                if planstr['platetype'] == 'cal': 
                    skywave = False
                else: 
                    skywave = True 
            if 'platetype' in planstr.keys():
                if planstr['platetype'] == 'sky': 
                    plugmap = 0
            outdir = os.path.dirname(load.filename('1D',num=framenum,chips=True))
            if os.path.exists(outdir)==False:
                file_mkdir,outdir 
            if fluxtest==False or planstr['APEXP']['flavor'][j]=='flux': 
                ap2dproc(inpfile,tracefile,exttype,load=load,outdir=outdir,unlock=unlock,modelpsffile=modelpsffile,
                         wavefile=wavefile,skywave=skywave,plugmap=plugmap,clobber=clobber,compress=True)
            elif waveid > 0: 
                ap2dproc(inpfile,tracefile,exttype,load=load,outdir=outdir,unlock=unlock,modelpsffile=modelpsffile,
                         fluxcalfile=fluxfile,responsefile=responsefile,
                         wavefile=wavefile,skywave=skywave,plugmap=plugmap,clobber=clobber,compress=True)
            else:
                ap2dproc(inpfile,tracefile,exttype,load=load,outdir=outdir,unlock=unlock,modelpsffile=modelpsffile,
                         fluxcalfile=fluxfile,responsefile=responsefile,
                         clobber=clobber,compress=True)
 
        # Now add in wavelength calibration information, with shift from
        #  FPI or sky lines
        # this used to call "apskywavecal", "ap1dwavecal" now handles
        # both cases (sky lines and fpi lines)
        if int(waveid) > 0 or int(fpiid) > 0: 
            cmd = ['ap1dwavecal',planfile] 

            # Check if there is FPI flux in the 2 fibers
            if fpiid > 0: 
                outfile1 = load.filename('1D',num=framenum,chips=True).replace('1D-','1D-b-')
                if os.path.exists(outfile1)==False: 
                    print(outfile1+' not found')
                    return      
                head0 = fits.getheader(outfile1,0)
                nread = head0['nread']
                exptime = head0['exptime']
                flux, head = fits.getdata(outfile1,1,header=True)
                flux1 = flux[[75,225],:]
                # average on the level of the lsf, ~13 pixels
                bflux1 = flux1[:,0:157*13].reshape(2,157,13).mean(axis=2) 
                medbflux = np.nanmedian(bflux1) 
                medbfluxnread = medbflux/nread
                # flux/nreads ~65 for FPI (chip b)  
                if medbfluxnread < 20: 
                    print('FPIID is set but not enough flux in the 2-FPI fibers.  Using sky lines instead!')
                    fpiid = 0 

            if fpiid > 0:  # use FPI lines
                cmd += ['--fpiid',str(fpiid)] 
            else:  # use sky lines
                if skywave==False:
                    cmd += ['--nosky']
            sout = subprocess.run(cmd,shell=False)
     
        # Compress 2D files
        nframes = len(planstr['APEXP']) 
        for j in range(nframes): 
            files = load.filename('2D',num=planstr['APEXP']['name'][j],chips=True)
            files = [files.replace('2D-','2D-'+ch+'-') for ch in chiptag]
            modfiles = load.filename('2Dmodel',num=planstr['APEXP']['name'][j],mjd=load.cmjd(planstr['APEXP']['name'][j]),chips=True)
            modfiles = [modfiles.replace('2Dmodel-','2Dmodel-'+ch+'-') for ch in chiptag]
            for jj in range(len(files)): 
                if os.path.exists(files[jj]): 
                    if os.path.exists(files[jj]+'.fz'): os.remove(files[jj]+'.fz')
                #sout = subprocess.run(['fpack','-D','-Y',files[jj]],shell=False)
                if os.path.exists(modfiles[jj]): 
                    if os.path.exists(modfiles[jj]+'.fz'): os.remove(modfiles[jj]+'.fz')
                    sout = subprocess.run(['fpack','-D','-Y',modfiles[jj]],shell=False)
                              
        utils.writelog(logfile,'AP2D: '+os.path.basename(planfile)+('%.1f' % (time.time()-t0)))
 
    del epsfchip 

    print('AP2D finished')
    dt = time.time()-t0 
    print('dt = %.1f sec' % dt)
  
 
 
