#!/usr/bin/env python

import os
import time
import numpy as np

def ap2dproc(inpfile,psffile,extract_type=1,outdir=None,clobber=False,fixbadpix=False,
             fluxcalfile=None,responsefile=None,wavefile=None,skywave=False,
             plugmap=0,highrej=7,lowrej=10,recenterfit=False,recenterln2=False,fitsigma=False,
             refpixzero=False,outlong=False,nowrite=False,npolyback=0,
             chips=[0,1,2],fibers=np.arange(300),compress=False,verbose=False,
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
          5-Full Gaussian-Hermite PSF fitting extraction
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
 
    #common savedepsf, savedepsffiles, epsfchip 
    #if len(savedepsfiles) == 0 : initialize if needed 
    #    savedepsffiles=[' ',' ',' ']  # initialize if needed 
 
    if len(epsfchip) == 0: 
        epsfchip=0 

    if np.array(inpfile).size==1:
        inpfile = [inpfile]
    ninpfile = len(inpfile) 
 
    # More than one file input more than one file input 
    if ninpfile > 1: 
        raise ValueError('Only one file can be input at a time')
     
    # Output directory output directory 
    if outdir is None:
        outdir = os.path.dirname(inpfile)+'/' 
 
    #dirs = getdir() 
     
    chiptag = ['a','b','c'] 
     
    # outdir must be a string outdir must be a string 
    #if size(outdir,/type) != 7:if size(outdir,/type) != 7: 
    #  print('outdir must be a string'  print('outdir must be a string' 
    #  return  return 
    # 
    #:es the output directory exist?:es the output directory exist? 
    if os.path.exists(outdir)==False: 
        if not silent:
            print('')
            print('creating ',outdir)
        os.makedirs(outdir)
     
    # Chips to extract chips to extract 
    if len(chips) > 3 or np.min(chips)<0 or np.max(chips)>2: 
        raise ValueError( 'chips must have <=3 elements with values [0-2].')
     
    # Fibers to extract fibers to extract 
    if len(fibers) > 0: 
        if len(fibers) > 300 or np.min(fibers) < 0 or np.max(fibers) > 299 : 
            raise ValueError('fibers must have <=300 elements with values [0-299].')
     
    # make the filenames and check the files make the filenames and check the files 
    fdir = os.path.dirname(inpfile) 
    base = os.path.basename(inpfile) 
    if os.path.exists(fdir)==False:
        print('directory '+fdir+' not found')
        return [],[]
     
    baseframeid = string(int(base),format='(i08)') 
    files = dir+'/'+dirs.prefix+'2d-'+chiptag+'-'+baseframeid+'.fits' 
    info = apfileinfo(files,/silent) 
    framenum = info[0].fid8   # the frame number the frame number 
    okay = (info.exists and info.sp2dfmt and info.allchips and ((info.naxis == 3) or (info.exten == 1))) 
    if min(okay) < 1: 
        bd , = np.where(okay == 0,nbd) 
        error = 'there is a problem with files: '+strjoin((files)(bd),' ') 
        if not silent: 
            print('halt: '+error)
        import pdb; pdb.set_trace() 
        return
         
    # get psf info get psf info 
    psf_dir = os.path.dirname(psffile) 
    psf_base = os.path.basename(psffile) 
    if os.path.exists(psf_dir) == 0: 
        error = 'psf directory '+psf_dir+' not found' 
        if not silent: 
            print('halt: '+error)
        import pdb; pdb.set_trace() 
        return 
     
    psfframeid = string(int(psf_base),format='(i08)') 
    psffiles = apogee_filename('psf',num=psfframeid,chip=chiptag) 
    epsffiles = apogee_filename('epsf',num=psfframeid,chip=chiptag) 
    pinfo = apfileinfo(psffiles,/silent) 
    pokay = (pinfo.exists and pinfo.allchips) 
    if min(pokay) < 1: 
        pbd , = np.where(pokay == 0,nbd) 
        error = 'there is a problem with psf files: '+strjoin((psffiles)(pbd),' ') 
        if not silent: 
            print('halt: '+error)
        import pdb; pdb.set_trace() 
        return 
     
    if not silent: 
        print('')
        print('extracting file ',inpfile)
        print('--------------------------------------------------')
     
         
    # parameters parameters 
    mjd5 = info[0].mjd5 
    nreads = info[0].nreads 
    if not silent: 
        print('mjd5 = ',str(mjd5))
         
    # check header check header
    head = fits.getheader(files[0],0)
    if errmsg != '': 
        error = 'there was an error loading the header for '+file 
        if not silent: 
            print('halt: '+error)
        import pdb; pdb.set_trace() 
        return 
     
         
    # determine file type determine file type 
    #-------------------------------------------- 
    # dark - should be processed with  dark - should be processed with 
    # flat flat 
    # lamps lamps 
    # object frame object frame 
    #obstype = sxpar(head,'obstype',count=nobs)obstype = sxpar(head,'obstype',count=nobs) 
    imagetyp = head.get'imagetyp')
    if imagetyp is None:
        error = 'no imagetyp keyword found for '+baseframeid 
        if not silent: 
            print(error) 
     
    #obstype = strlowcase(str(obstype,2))obstype = strlowcase(str(obstype,2)) 
    imagetyp = str(imagetyp).lower()
         
    # load the frame load the frame 
    frame = load.ap2D(files)
     
    # Double-check the flux calibration file
    if fluxcalfile is not None: 
        fluxcalfiles = os.path.dirname(fluxcalfile)+'/'+dirs.prefix+'flux-'+chiptag+'-'+os.path.basename(fluxcalfile)+'.fits' 
        ftest = os.path.exists(fluxcalfiles) 
        if np.sum(ftest) < 3: 
            error = 'problems with flux calibration file '+fluxcalfile 
            if not silent: 
                print('halt: '+error)
            import pdb; pdb.set_trace() 
            return
         
    # Double-check the response calibration file
    if responsefile is not None: 
        responsefiles = os.path.dirname(responsefile)+'/'+dirs.prefix+'response-'+chiptag+'-'+os.path.basename(responsefile)+'.fits' 
        ftest = os.path.exists(responsefiles) 
        if np.sum(ftest) < 3: 
            error = 'problems with response calibration file '+responsefile 
            if not silent: 
                print('halt: '+error)
            import pdb; pdb.set_trace() 
            return 
         
    # Double-check the wave calibration file
    if wavefile is not None: 
        wavefiles = os.path.dirname(wavefile)+'/'+dirs.prefix+'wave-'+chiptag+'-'+os.path.basename(wavefile)+'.fits' 
        wtest = os.path.exists(wavefiles) 
        if np.sum(wtest) < 3: 
            error = 'problems with wavelength file '+wavefile 
            if not silent: 
                print('halt: '+error)
            import pdb; pdb.set_trace() 
            return 

         
    # wait if another process is working on this wait if another process is working on this 
    lockfile = outdir+dirs.prefix+'1d-'+framenum # lock file lock file 
    if getlocaldir(): 
        lockfile = getlocaldir()+'/'+dirs.prefix+'1d-'+framenum+'.lock' 
    else: 
        lockfile = outdir+dirs.prefix+'1d-'+framenum+'.lock' 
    
    if not unlock:
        while os.path.exists(lockfile):
            apwait(lockfile,10)
        else: 
            if os.path.exists(lockfile): 
                os.remove(lockfile)
             
    err = 1 
    while err != 0: 
        if os.path.exists(os.path.dirname(lockfile))==False:
            os.mkdirs(os.path.dirname(lockfile))
        openw,lock,/get_lun,lockfile, error=err 
        if (err != 0): 
            printf, -2, !error_state.msg 
            print(file_search('/scratch/local','*'))
                         
    # Since final ap1dwavecal requires simultaneous fit of all three chips, and
    #  this required final output to be put off until after all chips are done,
    #  all 3 need to be done here if any at all, so that data from all chips is loaded
    # Output files
    outfiles = outdir+dirs.prefix+'1d-'+chiptag+'-'+framenum+'.fits'  # output file output file 
    outtest = os.path.exists(outfiles) 
    if min(outtest) == 0 : 
        clobber=1 
         
    if not keyword_set(clobber) : 
        goto,ap 
         
    #-------------------------------------------------------------------- 
    # looping through the three chips looping through the three chips 
    #-------------------------------------------------------------------- 
    head_chip=strarr(len(chips),5000) #initialise an array to store the headers
    output = None
    outmodel = None
    outstr = None
    ifirst = 0 
    for i in range(len(chips)): 
        t1 = time.time()
        ichip = chips[i]   # chip index, 0-first chip chip index, 0-first chip 
             
        ifile = files[ichip] 
             
        # the chip structure the chip structure 
        chstr = frame.(ichip) 
             
        # chip trace filename chip trace filename 
        ipsffile = psffiles[ichip] 
        iepsffile = epsffiles[ichip] 
             
        # output file output file 
        outfile = outdir+dirs.prefix+'1d-'+chiptag[ichip]+'-'+framenum+'.fits'  # output file output file 
             
        if not silent: 
            if i > 0 : 
                print('')
            print(' processing chip '+chiptag[ichip]+' - '+os.path.basename(ifile))
            print('  psf file = ',ipsffile)
         
             
        # fix the bad pixels and "unfixable" pixels fix the bad pixels and "unfixable" pixels 
        #------------------------------------------------------------------------------------------ 
        if fixbadpix:
            chstr = ap2dproc_fixpix(chstr)
             
        ###################################################################
        # need to remove the littrow ghost and secondary ghost here!!!!!!!!
        ###################################################################
             
        # restore the trace structure restore the trace structure 
        tracestr = fits.getdata(ipsffile,1)
             
        # fibers to extract fibers to extract 
        #if len(fibers) == 0:fibers=indgen(len(tracestr))if len(fibers) == 0:fibers=indgen(len(tracestr)) 
        if len(fibers) > 0: 
            if max(fibers) > len(tracestr)-1: 
                error = 'max(fibers) is larger than the number of fibers in psf file.' 
                if not silent: 
                    print('halt: '+error)
                import pdb; pdb.set_trace() 
                return 
             
        # measuring the trace shift measuring the trace shift 
        if recenterfit:
            im = frame.(ichip).flux 
            sz = size(im) 
            npix = sz[1] 
            nfibers = len(tracestr) 
            # the red chip has problems on the left side, the red chip has problems on the left side, 
            #  so use columns farther to the right  so use columns farther to the right 
            if ichip == 0: 
                xmid=npix*0.75 
            else: 
                xmid=npix*0.5 
                 
            medspec = np.median(im[xmid-50:xmid+50,:],dim=1) 
            gdpix , = np.where(medspec > 0.5*max(medspec),ngdpix) 
            if ngdpix <= 20: 
                # we're probably trying to process a dark as object or flat we're probably trying to process a dark as object or flat 
                # i'm not sure if 20 is the right number but seems to work with darks i'm not sure if 20 is the right number but seems to work with darks 
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
                yfit = mpfitpeak(lag[fitlo:fithi],xc[fitlo:fithi],par,nterms=4,/gaussian,/positive,estimates=estimates) 
                xshift = par[1] 
                 
                if not silent: 
                    print('recentering shift = %.3f ' % xshift)
                    
                # this is an ADDITIVE offset!
                 
        # calculate the trace shift ln2level header values calculate the trace shift ln2level header values 
        if recenterln2: 
            head_psf = headfits(ipsffile,exten=0) 
            ln2level_psf = sxpar(head_psf,'ln2level',count=nln2level_psf) 
            ln2level_im = sxpar(chstr.header,'ln2level',count=nln2level_im) 
                    
            if nln2level_psf > 0 and nln2level_im > 0:
                # the slope of trace shift vs. ln2level is (from green chip):  0.0117597 the slope of trace shift vs. ln2level is (from green chip):  0.0117597 
                # fits from check_traceshift.def fits from check_traceshift.def 
                # linear: coef=[ -1.02611, 0.0117597] linear: coef=[ -1.02611, 0.0117597] 
                # quadratic:  coef=[-3.33460, 0.0613117, -0.000265449] quadratic:  coef=[-3.33460, 0.0613117, -0.000265449] 
                # a higher ln2level shifts the fiber:wnwards a higher ln2level shifts the fiber:wnwards 
                xshift = (ln2level_im - ln2level_psf) * (-0.0117597) 
                if not silent : 
                    print('recentering shift = %.3f' % xshift)

                # this is an ADDITIVE offset!
                         
            # Don't have ln2levels
            else: 
                if nln2level_psf == 0 and not silent: 
                    print('do not have header ln2level for psf exposure')
                if nln2level_im == 0 and not silent: 
                    print('do not have header ln2level for this exposure')
                if not silent: 
                    print('cannot calculate fiber shift from ln2level in headers')
                 
        # reset the zeropoint threshold using the reference pixels reset the zeropoint threshold using the reference pixels 
        if refpixzero: 
            medref = np.median( [ chstr.flux[:,0:3], transpose(chstr.flux[0:3,:]), chstr.flux[:,2044:2047], transpose(chstr.flux[2044:2047,:]) ]) 
            if not silent: 
                print('setting image zeropoint using reference pixels.  subtracting ',str(medref))
                chstr.flux -= medref 
                 
        # initialize the output header initialize the output header 
        #------------------------------------------------------------ 
        head = chstr.header 
        sxaddpar,head,'psffile',ipsffile,' psf file used' 
        leadstr = 'ap2d: ' 
        sxaddpar,head,'v_apred',getgitvers(),'apogee software version' 
        sxaddpar,head,'apred',getvers(),'apogee reduction version' 
        sxaddhist,leadstr+systime(0),head 
        info = get_login_info() 
        sxaddhist,leadstr+info.user_name+' on '+info.machine_name,head 
        sxaddhist,leadstr+'idl '+!version.release+' '+!version.os+' '+!version.arch,head 
        # add reduction pipeline version to the header add reduction pipeline version to the header 
        sxaddhist,leadstr+' apogee reduction pipeline version: '+getvers(),head 
        sxaddhist,leadstr+'output file:',head 
        sxaddhist,leadstr+' hdu1 - image (adu)',head 
        sxaddhist,leadstr+' hdu2 - error (adu)',head 
        if (extract_type == 1): 
            sxaddhist,leadstr+' hdu3 - flag mask (bitwise or combined)',head 
            sxaddhist,leadstr+'        1 - bad pixels',head 
            sxaddhist,leadstr+'        2 - cosmic ray',head 
            sxaddhist,leadstr+'        4 - saturated',head 
            sxaddhist,leadstr+'        8 - unfixable',head 
        else: 
            sxaddhist,leadstr+' hdu3 - flag mask',head 
            sxaddhist,leadstr+'        0 - good pixels',head 
            sxaddhist,leadstr+'        1 - bad pixels',head 
             
        if len(wavefile) > 0: 
            sxaddhist,leadstr+' hdu4 - wavelengths (ang)',head 
            sxaddhist,leadstr+' hdu5 - wavelength coefficients',head 
             
        outstr = None
        ymodel = None
                 
        # Extraction type
        #----------------
                     
        # boxcar extraction boxcar extraction 
        #-------------------------------------- 
        if extract_type==1:
            if not silent: 
                print('using boxcar extraction')
                 
            # Update header
            sxaddhist,leadstr+'extract_type=1 - using boxcar extraction',head 
            sxaddpar,head,'extrtype',1,'extraction type' 
                     
            # recenter, shift the traces recenter, shift the traces 
            if keyword_set(recenterfit) or keyword_set(recenterln2): 
                tracestr.coef[0] += xshift 
                tracestr.gaussy += xshift 
                if keyword_set(recenterfit) and not keyword_set(recenterln2) : 
                    sxaddhist,leadstr+' /recenterfit set, shifting traces by '+stringize(xshift,ndec=3),head
                if keyword_set(recenterln2) : 
                    sxaddhist,leadstr+' /recenterln2 set, shifting traces by '+stringize(xshift,ndec=3),head 
                     
            # extract the fibers
            outstr = apextract(chstr,tracestr,fibers=fibers)
            
        # psf image extraction
        #---------------------
        elif extract_type==2:
            if not silent: 
                print('using psf image extraction')
             
            # load the psf image load the psf image 
            psfim,head_psfim = fits.getdata(ipsffile,2,header=True)
                 
            # update header
            sxaddhist,leadstr+'extract_type=2 - using psf image extraction',head 
            sxaddpar,head,'extrtype',2,'extraction type' 
                 
            # recenter, shift the traces and the psf image recenter, shift the traces and the psf image 
            if keyword_set(recenterfit) or keyword_set(recenterln2): 
                tracestr.coef[0] += xshift 
                tracestr.gaussy += xshift 
                psfim0 = psfim 
                psfim = imdrizzle(psfim0,0.0,xshift)  # shift the image with imdrizzle shift the image with imdrizzle 
                if keyword_set(recenterfit) and not keyword_set(recenterln2) : 
                    sxaddhist,leadstr+' /recenterfit set, shifting traces by '+stringize(xshift,ndec=3),head 
                if keyword_set(recenterln2) : 
                    sxaddhist,leadstr+' /recenterln2 set, shifting traces by '+stringize(xshift,ndec=3),head 
                 
            # extract the fibers extract the fibers 
            outstr,ymodel = apextractpsf(chstr,tracestr,psfim,model=ymodel,fibers=fibers)
             
        # gaussian psf fitting
        #---------------------
        #   maybe use the idlspec2d extraction code for this
        elif extract_type==3:
            if not silent: 
                print('using gaussian psf fitting extraction')
            
            # update header update header 
            sxaddhist,leadstr+'extract_type=3 - using gaussian psf fitting extraction',head 
            sxaddpar,head,'extrtype',3,'extraction type' 
             
            # the idlspec2d programs expect the fibers to run along the y
            # transposing the arrays for now
             
            # Get the idlspec2d-style trace and widthset information get the idlspec2d-style trace and widthset information 
            tset_coeff,tset_head = fits.getdata(ipsffile,3,header=True)
            tset = {func:str(sxpar(tset_head,'func'),2),xmin:sxpar(tset_head,'xmin'),
                    xmax:sxpar(tset_head,'xmax'),coeff:tset_coeff} 
            wset_coeff,wset_head = fits.getdata(ipsffile,4,header=True)
            widthset = {func:str(sxpar(wset_head,'func'),2),xmin:sxpar(wset_head,'xmin'),
                        xmax:sxpar(wset_head,'xmax'),coeff:wset_coeff} 
            proftype = wset_head.get('proftype') 
             
            # Get the trace and sigma arrays get the trace and sigma arrays 
            ycen,xsol,xx,sigma2 = None,None,None,None
            traceset2xy, tset, ycen, xsol 
            traceset2xy, widthset, xx, sigma2 
             
            # Get the images ready get the images ready 
            img = transpose( float(frame.(ichip).flux) ) 
            ivar = transpose( 1/float(frame.(ichip).err)**2 ) 
            mask = transpose( frame.(ichip).mask ) 
            mask = 1-( ((mask and 1) == 1) or ((mask and 4) == 4) or ((mask and 8) == 8) ) 
             
            # Recenter the traces recenter the traces 
            if recenterfit or recenterln2: 
                # Need to add this to the traces
                xsol += xshift 
                if recenterfit and not recenterln2: 
                    sxaddhist,leadstr+' /recenterfit set, shifting traces by '+stringize(xshift,ndec=3),head 
                if recenterln2: 
                    sxaddhist,leadstr+' /recenterln2 set, shifting traces by '+stringize(xshift,ndec=3),head 
             
            #------------------------------------------------------------------------------------------------------------------------------------------ 
            # extract the spectra extract the spectra 
            #------------------------------------------------------------------------------------------------------------------------------------------ 
            # since the gaussian is not a good fit use a lower since the gaussian is not a good fit use a lower 
            #  order background  order background 
            npoly = npolyback 
            wfixed = [1]   # keep the sigmas fixed keep the sigmas fixed 
            if fitsigma:
                wfixed = [1,1]  # fit sigma 
         
            # only extract fibers only extract fibers 
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
             
            # transpose the model transpose the model 
            ymodel = ymodel.T
             
            # create outstr create outstr 
            #  bad pixels have fluxivar=0, they are given high err  bad pixels have fluxivar=0, they are given high err 
            #  mask make it: 0-good, 1-bad  mask make it: 0-good, 1-bad 
            outstr = {flux:flux, err:1/(sqrt(fluxivar>1d-12)), mask:fix(fluxivar*0)} 
            outstr.mask = (fluxivar == 0)  # pixels with fluxivar=0 are bad pixels with fluxivar=0 are bad 
            # negative pixels negative pixels 
            #bd , = np.where(outstr.flux < 0,nbd)bd , = np.where(outstr.flux < 0,nbd) 
            #if nbd > 0:if nbd > 0: 
            #  outstr.flux[bd] = 0  outstr.flux[bd] = 0 
            #  outstr.err[bd] = 1e6  outstr.err[bd] = 1e6 
            #  outstr.mask[bd] = 1  # maybe give this a different value  outstr.mask[bd] = 1   maybe give this a different value 
            # 
            # fix reference pixels fix reference pixels 
            outstr.flux[0:3,:] = 0 
            outstr.flux[2040:2047,:] = 0 
            outstr.err[0:3,:] = baderr() 
            outstr.err[2040:2047,:] = baderr() 
            outstr.mask[0:3,:] = 1 
            outstr.mask[2040:2047,:] = 1 
         
        # Empirical psf extraction
        #-------------------------
        if extract_type==4:
            if not silent: 
                print('using empirical psf extraction')
              
            # copied from holtz/approcess.def copied from holtz/approcess.def 
            if epsffiles[ichip] != savedepsffiles[ichip]: 
             
                # load empirical psf data
                tmp = mrdfits(iepsffile,0,phead,status=status_epsf,/silent) 
                ntrace = sxpar(phead,'ntrace') 
                img = ptrarr(ntrace,/allocate_heap) 
                for itrace in range(ntrace): 
                    ptmp = fits.getdata(iepsffile,itrace+1) 
                    *img[itrace] = ptmp.img 
                    p ={fiber: ptmp.fiber, lo: ptmp.lo, hi: ptmp.hi, img: img[itrace]} 
                    if itrace == 0: 
                        epsf=replicate(p,ntrace) 
                    epsf[itrace] = p 
     
                if ichip == 0: 
                    epsfchip = epsf 
                    sz0 = size(epsf,/dim) 
                    sz = sz0 
                else: 
                    sz = size(epsf,/dim) 
                    if sz == sz0: 
                        epsfchip = [[epsfchip],[epsf]] 
    
                if status_epsf != 0: 
                    if not silent: 
                        print('psf file ',iepsffile,':es not contain an empirical psf image')
                    continue

                if sz == sz0: 
                    savedepsffiles[ichip] = epsffiles[ichip] 
     
            else:
                epsf = epsfchip[:,ichip]
         
            # update header update header 
            sxaddhist,leadstr+'extract_type=4 - using empirical psf extraction',head 
            sxaddpar,head,'extrtype',4,'extraction type' 
         
            # recenter, shift the traces and the psf image recenter, shift the traces and the psf image 
            if recenterfit or recenterln2: 
                # shift the image with imdrizzle shift the image with imdrizzle 
                for l in range(len(epsf)): 
                    epsf[l].img = imdrizzle(epsf[l].img,0.0,xshift) 
         
                if keyword_set(recenterfit) and not keyword_set(recenterln2) : 
                    sxaddhist,leadstr+' /recenterfit set, shifting traces by '+stringize(xshift,ndec=3),head 
         
                if keyword_set(recenterln2) : 
                    sxaddhist,leadstr+' /recenterln2 set, shifting traces by '+stringize(xshift,ndec=3),head 
         
            ymodel = apextract_epsf(frame.(ichip),epsf,outstr,scat=True)
            #import pdb; pdb.set_trace()import pdb; pdb.set_trace() 
     
        # full gaussian-hermite psf fitting full gaussian-hermite psf fitting 
        #---------------------------------------------------------------------- 
        elif extract_type==5:
            error = 'full gaussian-hermite psf fitting is not supported yet' 
            if not silent: 
                print(error)
            return 
 
        t2 = time.time()
        #import pdb; pdb.set_trace()
 
 
        # Do the fiber-to-fiber throughput corrections and relative: the fiber-to-fiber throughput corrections and relative 
        # flux calibration flux calibration 
        #------------------------------------------------------------------------------------------------------------------------ 
        if len(fluxcalfile) > 0:
            # restore the relative flux calibration correction file
            if not silent: 
                print('flux calibrating with ',os.path.dirname(fluxcalfile)+'/'+dirs.prefix+'flux-'+os.path.basename(fluxcalfile))
 
            fluxcalfiles = os.path.dirname(fluxcalfile)+'/'+dirs.prefix+'flux-'+chiptag+'-'+os.path.basename(fluxcalfile)+'.fits' 
            fluxcal,fluxcal_head = fits.getdata(fluxcalfiles[ichip],header=True)
            outstr.flux /= fluxcal          # correct flux correct flux 
            bderr, = np.where(outstr.err == baderr()) 
            nbd = len(bderr)
            outstr.err /= fluxcal           # correct error correct error 
            if nbd > 0: 
                outstr.err[bderr] = baderr() 
            bd, = np.where(np.isfinite(outstr.flux) == 0) 
            nbd = len(bd)
            if nbd > 0: 
                outstr.flux[bd] = 0. 
                outstr.err[bd] = baderr() 
                outstr.mask[bd] = 1 
 
            # update header update header 
            sxaddhist,leadstr+'flux calibrating the spectra with:',head 
            sxaddhist,leadstr+fluxcalfiles[ichip],head 
            sxaddpar,head,'fluxfile',fluxcalfile,' flux calibration file used' 
 
        # Response curve calibration
        #---------------------------
        if responsefile is not None: 
            # restore the relative flux calibration correction file
            if not silent: 
                print('response calibrating with ',os.path.dirname(responsefile)+'/'+dirs.prefix+'flux-'+os.path.basename(responsefile))
 
            responsefiles = os.path.dirname(responsefile)+'/'+dirs.prefix+'response-'+chiptag+'-'+os.path.basename(responsefile)+'.fits' 
            response,response_head = fits.getdata(responsefiles[ichip],header=True)
 
            sz = size(outstr.flux,/dim) 
            outstr.flux *= response#replicate(1.,sz[1])         # correct flux correct flux 
            bderr, = np.where(outstr.err == baderr(),nbd) 
            outstr.err *= response#replicate(1.,sz[1])           # correct error correct error 
            if nbd > 0 : 
                outstr.err[bderr] = baderr() 
 
            # Update header
            sxaddhist,leadstr+'applying response function:',head 
            sxaddhist,leadstr+responsefiles[ichip],head 
            sxaddpar,head,'respfile',responsefile,' response file used' 
 
        # Adding wavelengths
        #-------------------
        if wavefile is not None:
            wavefiles = os.path.dirname(wavefile)+'/'+dirs.prefix+'wave-'+chiptag+'-'+os.path.basename(wavefile)+'.fits' 
            if not silent: 
                print('adding wavelengths from ',os.path.dirname(wavefile)+'/'+dirs.prefix+'wave-'+os.path.basename(wavefile))

            # Get the wavelength calibration data get the wavelength calibration data 
            wcoef,whead = fits.getdata(wavefiles[ichip],1,header=True)
            wim,whead2 = fits.getdata(wavefiles[ichip],2,header=True)
            # this is now fixed in the apwave files this is now fixed in the apwave files 
            #wim = transpose(wim)  # want it [npix, nfibers]wim = transpose(wim)   want it [npix, nfibers] 
 
            sxaddhist,leadstr+'adding wavelengths from',head 
            sxaddhist,leadstr+wavefiles[ichip],head 
            sxaddpar,head,'wavefile',wavefile,' wavelength calibration file' 
            sxaddpar,head,'wavehdu',5,' wavelength coef hdu' 
    
        # Add header to structure
        head0 = head 
        head = strarr(5000) 
        nhead = len(head0) 
        head[0:nhead-1] = head0 
        outstr = create_struct(outstr,'header',head) 
 
        head_chip[ichip,:]=head 
 
        # Add fibers to structure
        if len(fibers) > 0: 
            outstr = create_struct(outstr,'fibers',fibers) 
 
        # Output the 2d model spectrum
        if len(ymodel) > 0: 
            modelfile = outdir+dirs.prefix+'2dmodel-'+chiptag[ichip]+'-'+framenum+'.fits'  # model output file model output file 
            if not silent: 
                print('writing 2d model to: ',modelfile)
            mwrfits,ymodel,modelfile,/create 
            #    # compress model and 2d image:ne in ap2d     compress model and 2d image:ne in ap2d 
            #    if keyword_set(compress):    if keyword_set(compress): 
            #      os.remove(modelfile+'.fz',/allow_nonexistent      os.remove(modelfile+'.fz',/allow_nonexistent 
            #      spawn,'fpack -d -y '+modelfile      spawn,'fpack -d -y '+modelfile 
            #      origfile = outdir+dirs.prefix+'2d-'+chiptag[ichip]+'-'+framenum+'.fits'      origfile = outdir+dirs.prefix+'2d-'+chiptag[ichip]+'-'+framenum+'.fits' 
            #      if os.path.exists(origfile):      if os.path.exists(origfile): 
            #        os.remove(origfile+'.fz',/allow_nonexistent        os.remove(origfile+'.fz',/allow_nonexistent 
            #        spawn,'fpack -d -y '+origfile        spawn,'fpack -d -y '+origfile 
            #             
            #         

        # Add to output structure
        if ifirst == 0: 
            output = create_struct('chip'+chiptag[ichip],outstr) 
        if ymodel is not None: 
            outmodel=create_struct('chip'+chiptag[ichip],{model:ymodel})
            ifirst = 1 
        else: 
            output = create_struct(output,'chip'+chiptag[ichip],outstr) 
            if len(ymodel) > 0: 
                outmodel=create_struct(outmodel,'chip'+chiptag[ichip],{model:ymodel}) 
 
        if logfile is not None:
            writeline(logfile,os.path.basename(outfile),string(format='(f8.3)',systime(/seconds)-t1))
 
        #import pdb; pdb.set_trace()import pdb; pdb.set_trace() 
  
    # Aow we have output structure with three chips, each with tags header, flux, err, mask
 
    # Add wavelength information to the frame structure
    #--------------------------------------------------
    # loop through the chips loop through the chips 
    if len(wavefiles) > 0: 
        for k in range(2+1): 
            # Get the wavelength calibration data
            wcoef,whead = fits.getdata(wavefiles[k],1,header=True)
            # Add to the chip structure
            chstr = output.(k) 
            chstr = create_struct(temporary(chstr),'filename',files[k],'wave_dir',outdir,'wavefile',wavefiles[k],'wcoef',wcoef) 
            # now add this to the final frame structure now add this to the final frame structure 
            if k == 0: 
                frame = create_struct('chip'+chiptag[k],chstr) 
            else: 
                frame = create_struct(frame,'chip'+chiptag[k],chstr) 
        del output  # free up memory
        if os.path.exists(outdir)==False:
            os.makedirs(outdir)
        plotfile = outdir+'/plots/pixshift-'+framenum 
        if skywave:
            frame_wave = ap1dwavecal(frame,plugmap=plugmap,verbose=True,plot=True,pfile=plotfile)
        else: 
            frame_wave = ap1dwavecal(frame,verbose=True,noshift=True)
        del,frame  # free up memory
    else:
        frame_wave = output 
 
    # Write output file
    #------------------
    if not nowrite:
 
        for i in range(len(chips)): 
            ichip = chips[i]   # chip index, 0-first chip chip index, 0-first chip 
            # output file output file 
            outfile = outdir+dirs.prefix+'1d-'+chiptag[ichip]+'-'+framenum+'.fits'  # output file output file 
            if not silent: 
                print('writing output to: ',outfile)
 
            if outlong and not silent:
                print('saving flux/err as long instead of float')
 
            # hdu0 - header only hdu0 - header only 
            fits_write,outfile,0,reform(head_chip[ichip,:]),/no_abort,message=write_error 
 
            # hdu1 - flux hdu1 - flux 
            flux = frame_wave.(i).flux 
            if outlong:
                flux=int(np.round(flux))
 
            mkhdr,head1,flux,/image 
            sxaddpar,head1,'ctype1','pixel' 
            sxaddpar,head1,'ctype2','fiber' 
            sxaddpar,head1,'bunit','flux (adu)' 
            mwrfits,flux,outfile,head1,/silent 
 
            # hdu2 - error hdu2 - error 
            err = errout(frame_wave.(i).err) 
            if outlong:
                err = int(np.round(err))
 
            mkhdr,head2,err #,/image 
            sxaddpar,head2,'ctype1','pixel' 
            sxaddpar,head2,'ctype2','fiber' 
            sxaddpar,head2,'bunit','error (adu)' 
            mwrfits,err,outfile,head2,/silent 
 
            # hdu3 - mask hdu3 - mask 
            mask = frame_wave.(i).mask 
            mkhdr,head3,mask,/image 
            sxaddpar,head3,'ctype1','pixel' 
            sxaddpar,head3,'ctype2','fiber' 
            if (extract_type == 1): 
                sxaddpar,head3,'bunit','flag mask (bitwise)' 
                sxaddhist,'explanation of bitwise flag mask (or combined)',head3 
                sxaddhist,' 1 - bad pixels',head3 
                sxaddhist,' 2 - cosmic ray',head3 
                sxaddhist,' 4 - saturated',head3 
                sxaddhist,' 8 - unfixable',head3 
            else: 
                sxaddpar,head3,'bunit','flag mask' 
                sxaddhist,'explanation of flag mask',head3 
                sxaddhist,' 0 - good pixels',head3 
                sxaddhist,' 1 - bad pixels',head3 
 
            mwrfits,mask,outfile,head3,/silent 
 
            if len(wavefiles) > 0: 
                # hdu4 - wavelengths hdu4 - wavelengths 
                mkhdr,head4,frame_wave.(i).wavelength,/image 
                sxaddpar,head4,'ctype1','pixel' 
                sxaddpar,head4,'ctype2','fiber' 
                sxaddpar,head4,'bunit','wavelength (angstroms)' 
                mwrfits,frame_wave.(i).wavelength,outfile,head4,/silent 
 
            # hdu5 = wavelength solution coefficients [double] hdu5 = wavelength solution coefficients [double] 
            #---------------------------------------------------------------------------------------------------------- 
            wcoef = double(frame_wave.(i).wcoef) 
            mkhdr,head5,wcoef,/image 
            sxaddpar,head5,'ctype1','fiber' 
            sxaddpar,head5,'ctype2','parameters' 
            sxaddpar,head5,'bunit','wavelength coefficients' 
            sxaddhist,'wavelength coefficients to be used with pix2wave.pro:',head5 
            sxaddhist,' 1 global additive pixel offset',head5 
            sxaddhist,' 4 sine parameters',head5 
            sxaddhist,' 7 polynomial parameters (first is a zero-point offset',head5 
            sxaddhist,'                     in addition to the pixel offset)',head5 
            mwrfits,wcoef,outfile,head5,/silent 
 
    if os.path.exists(lockfile):
        os.remove(lockfile)
 
    if not silent:
        print('ap2proc finished')
 

def ap2d(planfiles,verbose=verbose,stp=stp,clobber=clobber,exttype=exttype,mapper_data=mapper_data,
         calclobber=calclobber,domelibrary=domelibrary,unlock=unlock):  
    """
    This program processes 2D APOGEE spectra.  It extracts the
    spectra.

    Parameters
    ----------
    planfiles     Input list of plate plan files
    =exttype      
    =mapper_data  Directory for mapper data.
    /verbose      Print a lot of information to the screen
    /clobber      Overwrite existing files (ap1D).
    /calclobber   Overwrite existing daily calibration files (apPSF, apFlux).
    /domelibrary  Use the domeflat library.
    /stp          Stop at the end of the prrogram
    /unlock      Delete lock file and start fresh

    Returns
    -------
    1D extracted spectra are output.  One file for each frame.

    Example
    -------
    out = ap2d(planfile)

    Written by D.Nidever  Mar. 2010
    Modifications: J. Holtzman 2011+
    Translated to python by D. Nidever 2022
    """
 
    #common savedepsf, savedepsffiles, epsfchip 
          
    savedepsffiles = [' ',' ',' '] 
    epsfchip = 0 
 
    # default parameters default parameters 
    if len(verbose) == 0 : not verbose by default 
    verbose=0  # not verbose by default 
 
# calclobber will redo psf, flux and 1d frames (but not other fundamental calibration frames) calclobber will redo psf, flux and 1d frames (but not other fundamental calibration frames) 
if not keyword_set(calclobber) : not calclobber by default, 
calclobber=0  # not calclobber by default, 
 
# clobber will redo 1d frames clobber will redo 1d frames 
if not keyword_set(clobber) : not clobber by default, 
clobber=0  # not clobber by default, 
 
if not keyword_set(exttype) : 
exttype=4 
 
if len(domelibrary) == 0 : 
domelibrary=0 
 
 
t0 = time.time() 
 
nplanfiles = len(planfiles) 
# not enough inputs not enough inputs 
if nplanfiles == 0: 
print('syntax - ap2d,planfiles' 
return 
 
 
print('' 
print('running ap2d' 
print('' 
print(str(nplanfiles,2),' plan files' 
 
chiptag = ['a','b','c'] 
apgundef,wavefile,responsefile 
 
#---------------------------------------------------------------------------------------- 
# loop through the unique plate observations loop through the unique plate observations 
#---------------------------------------------------------------------------------------- 
for i in np.arange(0l,nplanfiles): 
begin 
 
 
t1 = time.time() 
planfile = planfiles[i] 
 
print('' 
print('=========================================================================' 
print(str(i+1,2),'/',str(nplanfiles,2),'  processing plan file ',planfile 
print('=========================================================================' 
 
# load the plan file load the plan file 
#---------------------------------------- 
print('' & print('plan file information:' 
aploadplan,planfile,planstr,/verbose,error=planerror 
if len(planerror) > 0 : 
goto,bomb 
 
logfile = apogee_filename('diag',plate=planstr.plateid,mjd=planstr.mjd) 
 
# add psfid tag to planstr apexp structure add psfid tag to planstr apexp structure 
if tag_exist(planstr.apexp,'psfid') == 0: 
apexp = planstr.apexp 
add_tag,apexp,'psfid',0l,apexp 
old = planstr 
oldtags = tag_names(old) 
planstr = create_struct(oldtags[0],old.(0)) 
for j in np.arange(1,len(oldtags)): 
begin 
 
if oldtags[j] == 'apexp': 
planstr = create_struct(planstr,'apexp',apexp) 
else: 
planstr = create_struct(planstr,oldtags[j],old.(j)) 
 
 
undefine,old,oldtags 
 
 
# use:meflat library use:meflat library 
#------------------------------------------ 
# if (1) no:meflat id set in planfile, or (2):melibrary parameter if (1) no:meflat id set in planfile, or (2):melibrary parameter 
# set in planfile, or (3) /domelibrary keyword is set. set in planfile, or (3) /domelibrary keyword is set. 
if tag_exist(planstr,'domelibrary') == 1: 
plandomelibrary=planstr.domelibrary 
else: 
plandomelibrary=0 
 
if keyword_set(domelibrary) or tag_exist(planstr,'psfid') == 0 or keyword_set(plandomelibrary): 
print('using:meflat library' 
# you can: "domeflattrace --mjdplate" where mjdplate could be you can: "domeflattrace --mjdplate" where mjdplate could be 
# e.g. 59223-9244, or "domeflattrace --planfile", with absolute e.g. 59223-9244, or "domeflattrace --planfile", with absolute 
# path of planfile path of planfile 
# force single:meflat if a short visit or:melibrary=='single' force single:meflat if a short visit or:melibrary=='single' 
if planstr.telescope == 'apo25m' or planstr.telescope == 'apo1m': 
observatory='apo' 
else: 
observatory='lco' 
 
if str(domelibrary,2) == 'single' or str(plandomelibrary,2) == 'single' or len(planstr.apexp) <= 3: 
spawn,['domeflattrace',observatory,'--planfile',planfile,'--s'],out,errout,/noshell 
else: 
spawn,['domeflattrace',observatory,'--planfile',planfile],out,errout,/noshell 
 
nout = len(out) 
for f in range(nout): 
print(out[f] 
 
# parse the output parse the output 
lo , = np.where(stregex(out,'**dome flat results:',/boolean) == 1,nlo) 
hi = first_el(where(str(out,2) == '' and lindgen(nout) > lo[0])) 
if lo == -1 or hi == -1: 
print('problem running:meflattrace for ',planfile,'.  skipping this planfile.' 
continue 
 
outarr = strsplitter(out[lo+1:hi-1],' ',/extract) 
ims = reform(outarr[0,:]) 
domeflatims = reform(outarr[1,:]) 
# update planstr update planstr 
match,apexp.name,ims,ind1,ind2,/sort 
planstr.apexp[ind1].psfid =:meflatims[ind2] 
else: 
planstr.apexp.psfid = planstr.psfid 
 
 
#:n't extract dark frames:n't extract dark frames 
if tag_exist(planstr,'platetype') : 
if planstr.platetype == 'dark' or planstr.platetype == 'intflat' : 
goto,bomb 
 
 
 
# try to make the required calibration files (if not already made) try to make the required calibration files (if not already made) 
#:check if the calibration files exist:check if the calibration files exist 
#---------------------------------------------------------------------------- 
 
# appsf files  appsf files 
if planstr.sparseid != 0 : 
makecal,sparse=planstr.sparseid 
 
if planstr.fiberid != 0 : 
makecal,fiber=planstr.fiberid 
 
if tag_exist(planstr,'psfid'): 
makecal,psf=planstr.psfid,clobber=calclobber 
tracefiles = apogee_filename('psf',num=planstr.psfid,chip=chiptag) 
tracefile = os.path.dirname(tracefiles[0])+'/'+string(format='(i8.8)',planstr.psfid) 
tracetest = os.path.exists(tracefiles) 
if min(tracetest) == 0: 
bd1 , = np.where(tracetest == 0,nbd1) 
if nbd1 > 0 : 
import pdb; pdb.set_trace(),'halt: ',tracefiles[bd1],' not found' 
 
for ichip in range(2+1): 
begin 
 
p = mrdfits(tracefiles[ichip],1,/silent) 
if len(p) != 300: 
print( 'halt: tracefile ', tracefiles[ichip],':es not have 300 traces' 
 
 
 
 
 
# apwave files : wavelength calibration apwave files : wavelength calibration 
waveid = planstr.waveid 
if tag_exist(planstr,'platetype') : 
if planstr.platetype == 'cal' or planstr.platetype == 'extra' : 
waveid=0 
 
 
if waveid > 0 : 
makecal,multiwave=waveid 
 
 
# fpi calibration file fpi calibration file 
if tag_exist(planstr,'fpi'): 
fpiid = planstr.fpi 
else: 
fpiid=0 
 
 
# apflux files : since individual frames are usually made per plate apflux files : since individual frames are usually made per plate 
if planstr.fluxid != 0: 
makecal,flux=planstr.fluxid,psf=planstr.psfid,clobber=calclobber 
fluxfiles = apogee_filename('flux',chip=chiptag,num=planstr.fluxid) 
fluxfile = os.path.dirname(fluxfiles[0])+'/'+string(format='(i8.8)',planstr.fluxid) 
fluxtest = os.path.exists(fluxfiles) 
if min(fluxtest) == 0: 
bd1 , = np.where(fluxtest == 0,nbd1) 
if nbd1 > 0 : 
import pdb; pdb.set_trace(),'halt: ',fluxfiles[bd1],' not found' 
 
 
 else fluxtest=0 
 
# apresponse files  apresponse files 
#  these aren't used anymore  these aren't used anymore 
if tag_exist(planstr,'responseid') == 0 : 
add_tag,planstr,'responseid',0,planstr 
 
if planstr.responseid != 0: 
makecal,response=planstr.responseid 
responsefiles = apogee_filename('response',chip=chiptag,num=planstr.responseid) 
responsefile = os.path.dirname(responsefiles[0])+'/'+string(format='(i8.8)',planstr.responseid) 
responsetest = os.path.exists(responsefiles) 
if min(responsetest) == 0: 
bd1 , = np.where(responsetest == 0,nbd1) 
if nbd1 > 0 : 
import pdb; pdb.set_trace(),'halt: ',responsefiles[bd1],' not found' 
 
 
 
 
# load the plug plate map file load the plug plate map file 
#------------------------------------------------------------ 
if tag_exist(planstr,'platetype') : 
if planstr.platetype == 'cal' or planstr.platetype == 'extra' or      planstr.platetype == 'single' : 
plugmap=0 
else: 
if tag_exist(planstr,'platetype') : 
if planstr.platetype == 'cal' or planstr.platetype == 'extra' or      planstr.platetype == 'single' : 
plugmap=0 
else: 
print('' & print('plug map file information:' 
plugfile = planstr.plugmap 
if tag_exist(planstr,'fixfiberid') : 
fixfiberid=planstr.fixfiberid 
 
if size(fixfiberid,/type) == 7 and len(fixfiberid) == 1 : null/none 
if (str(fixfiberid,2) == 'null' or str(strlowcase(fixfiberid),2) == 'none') : 
    undefine,fixfiberid  # null/none 
 
 
if tag_exist(planstr,'badfiberid') : 
badfiberid=planstr.badfiberid 
 
if size(badfiberid,/type) == 7 and len(badfiberid) == 1 : null/none 
if (str(badfiberid,2) == 'null' or str(strlowcase(badfiberid),2) == 'none') : 
    undefine,badfiberid  # null/none 
 
 
# we only need the information on sky fibers we only need the information on sky fibers 
plugmap = getplatedata(planstr.plateid,string(planstr.mjd,format='(i5.5)'),plugid=planstr.plugmap,fixfiberid=fixfiberid,                           badfiberid=badfiberid,mapper_data=mapper_data,/noobject) 
if len(plugerror) > 0 : 
import pdb; pdb.set_trace(),'halt: error with plugmap: ',plugfile 
 
plugmap.mjd = planstr.mjd   # enter mjd from the plan file enter mjd from the plan file 
 
 
# are there enough files are there enough files 
nframes = len(planstr.apexp) 
if nframes < 1: 
print('no frames to process' 
goto,bomb 
 
 
# process each frame process each frame 
#-------------------------------------- 
for j in np.arange(0l,nframes): 
begin 
 
 
# get trace files get trace files 
tracefiles = apogee_filename('psf',num=planstr.apexp[i].psfid,chip=chiptag) 
tracefile = os.path.dirname(tracefiles[0])+'/'+string(format='(i8.8)',planstr.apexp[i].psfid) 
tracetest = os.path.exists(tracefiles) 
if min(tracetest) == 0: 
bd1 , = np.where(tracetest == 0,nbd1) 
if nbd1 > 0 : 
import pdb; pdb.set_trace(),'halt: ',tracefiles[bd1],' not found' 
 
for ichip in range(2+1): 
begin 
 
p = mrdfits(tracefiles[ichip],1,/silent) 
if len(p) != 300: 
print( 'halt: tracefile ', tracefiles[ichip],':es not have 300 traces' 
 
 
 
 
# make the filenames and check the files make the filenames and check the files 
rawfiles = apogee_filename('r',chip=chiptag,num=planstr.apexp[j].name) 
rawinfo = apfileinfo(rawfiles,/silent)        # this returns useful info even if the files:n't exist this returns useful info even if the files:n't exist 
framenum = rawinfo[0].fid8   # the frame number the frame number 
files = apogee_filename('2d',chip=chiptag,num=framenum) 
inpfile = os.path.dirname(files[0])+'/'+framenum 
info = apfileinfo(files,/silent) 
okay = (info.exists and info.sp2dfmt and info.allchips and (info.mjd5 == planstr.mjd) and             ((info.naxis == 3) or (info.exten == 1))) 
if min(okay) < 1: 
bd , = np.where(okay == 0,nbd) 
import pdb; pdb.set_trace(),'halt: there is a problem with files: ',strjoin((files)(bd),' ') 
 
 
print('' 
print('-----------------------------------------' 
print(str(j+1,2),'/',str(nframes,2),'  processing frame number >>',str(framenum,2),'<<' 
print('-----------------------------------------' 
 
# run ap2dproc run ap2dproc 
if tag_exist(planstr,'platetype') : 
if planstr.platetype == 'cal' : 
skywave=0 
else: 
skywave=1 
 
 
if tag_exist(planstr,'platetype') : 
if planstr.platetype == 'sky' : 
plugmap=0 
 
 
outdir=apogee_filename('1d',num=framenum,chip='a',/dir) 
if os.path.exists(outdir,/directory) == 0 : 
file_mkdir,outdir 
 
if min(fluxtest) == 0 or planstr.apexp[j].flavor == 'flux' : 
ap2dproc,inpfile,tracefile,exttype,outdir=outdir,unlock=unlock,               wavefile=wavefile,skywave=skywave,plugmap=plugmap,clobber=clobber,/compress     else if waveid > 0 : 
begin 
else: 
if min(fluxtest) == 0 or planstr.apexp[j].flavor == 'flux' : 
ap2dproc,inpfile,tracefile,exttype,outdir=outdir,unlock=unlock,               wavefile=wavefile,skywave=skywave,plugmap=plugmap,clobber=clobber,/compress     else if waveid > 0 : 
    begin 
else: 
    ap2dproc,inpfile,tracefile,exttype,outdir=outdir,unlock=unlock,               fluxcalfile=fluxfile,responsefile=responsefile,               wavefile=wavefile,skywave=skywave,plugmap=plugmap,clobber=clobber,/compress 
 else       ap2dproc,inpfile,tracefile,exttype,outdir=outdir,unlock=unlock,               fluxcalfile=fluxfile,responsefile=responsefile,               clobber=clobber,/compress 
     
    bomb1: 
     
 # frame loop frame loop 
 
# now add in wavelength calibration information, with shift from now add in wavelength calibration information, with shift from 
#  fpi or sky lines  fpi or sky lines 
# this used to call "apskywavecal", "ap1dwavecal" now handles this used to call "apskywavecal", "ap1dwavecal" now handles 
# both cases (sky lines and fpi lines) both cases (sky lines and fpi lines) 
if waveid > 0 or fpiid > 0: 
    cmd = ['ap1dwavecal',planfile] 
     
    # check if there is fpi flux in the 2 fibers check if there is fpi flux in the 2 fibers 
    if fpiid > 0: 
        outfile1 = apogee_filename('1d',num=framenum,chip='b') 
        if os.path.exists(outfile1) == 0: 
            print(outfile1,' not found' 
            return 
     
        fits.getdata(outfile1,flux,head,exten=1 
        flux1 = flux[:,[75,225]] 
        # average on the level of the lsf, ~13 pixels average on the level of the lsf, ~13 pixels 
        bflux1 = rebin(flux1[0:157*13-1],157,2) 
        medbflux = np.median(bflux1) 
        # ~3800 for fpi (chip b) ~3800 for fpi (chip b) 
        if medbflux < 2000: 
            print('fpiid is set but not enough flux in the 2-fpi fibers.  using sky lines instead!' 
            fpiid = 0 
     
 
     
    if fpiid > 0:  # use fpi lines use fpi lines 
        cmd = [cmd,'--fpiid',str(fpiid,2)] 
    else:  # use sky lines use sky lines 
        if not keyword_set(skywave) : 
            cmd=[cmd,'--nosky'] 
     
 
    spawn,cmd,/noshell 
    # if skywave:spawn,['apskywavecal',planfile],/noshell     # else  spawn,['apskywavecal',planfile,'--nosky'],/noshell if skywave:spawn,['apskywavecal',planfile],/noshell      else  spawn,['apskywavecal',planfile,'--nosky'],/noshell 
 
     
    bomb: 
     
    # compress 2d files compress 2d files 
    nframes = len(planstr.apexp) 
    for j in np.arange(0l,nframes): 
        begin 
 
    files = apogee_filename('2d',num=planstr.apexp[j].name,chip=chiptag) 
    modfiles = apogee_filename('2dmodel',num=planstr.apexp[j].name,chip=chiptag) 
    for jj in range(len(files)): 
        begin 
 
    if os.path.exists(files[jj]): 
        os.remove(files[jj]+'.fz',/allow_nonexistent 
        #       spawn,['fpack','-d','-y',files[jj]],/noshell       spawn,['fpack','-d','-y',files[jj]],/noshell 
 
    if os.path.exists(modfiles[jj]): 
        os.remove(modfiles[jj]+'.fz',/allow_nonexistent 
        spawn,['fpack','-d','-y',modfiles[jj]],/noshell 
 
 
 
 
writelog,logfile,'ap2d: '+os.path.basename(planfile)+string(format='(f8.2)',time.time()-t1) 
 
  # plan file loop plan file loop 
 
apgundef,epsfchip 
 
print('ap2d finished' 
dt = time.time()-t0 
print('dt = ',str(string(dt,format='(f10.1)'),2),' sec' 
 
if keyword_set(stp) : 
import pdb; pdb.set_trace() 
 
 
 
