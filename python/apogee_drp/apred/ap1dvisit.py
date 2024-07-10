import os
import time
import numpy as np
from astropy.io import fits
from astropy.table import Table
from ...utils import apload,plugmap,platedata,info,plan

chiptag = ['a','b','c']
BADERR = 1.0000000e+10

# dithershift()
# outcframe()
# ditherpairs()
# dithercombine()
# fluxing
# visitoutput()
# sky.skysub()
# sky.telluric()

def dithershift(frame1,frame2,xcorr=False,lines=True,objspec=None,plot=False,
                pfile=None,plugmap=None,nofit=False,mjd=None):
    """
    This program measured the SHIFT in two dithered images

    Parameters
    ----------
    frame1 : list
       A structure giving 1D extracted data and headers for
          all three chips for the FIRST dithered frame.
    frame2 : list
       The same as "frame1" but for the SECOND dithered frame.
    xcorr : bool, optional
       Use cross-correlation to measure the shift.
    lines : bool, optional
       Use emission lines to measure the shift.  This is the default.
    objspec : bool
       This is an object spectrum (stellar spectra).  The
         spectra will be normalized by the stellar continuum.
    pl : bool, optional
       Show plots of the fits.

    Returns
    -------
    shifttab : table
       Table with results.
         type       'xcorr' or 'lines'
         shiftfit    The shift in pixels between frame2 and frame1.  A positive
                       shift means that frame2 is to the RIGHT of frame1.
         shifterr   The uncertainty of the shift measurement.
         chipshift  The measured shifts for each chip.
         chipsfit   The fitted shift parameters for each chip.

    Examples
    --------

    shifttab = dithershift(frame1,frame2)

    By D. Nidever  March 2010
    Translated to Python by D. Nidever 2024
    """

    if mjd is None:
        mjd = 999999

    # Checking the tags of the input structure in FRAME1
    needtags1 = ['CHIPA','CHIPB','CHIPC']
    needtags2 = ['HEADER','FLUX','ERR','MASK']    
    for i in range(len(needtags1)):
        if needtags1[i] not in frame1.keys():
            print('TAG ',needtags1[i],' NOT FOUND in input structure')
            return []
    for i in range(3):
        for j in range(len(needtags2)):
            if needtags2[j] not in frame1[i].keys():
                print('TAG ',needtags2[j],' NOT FOUND in input structure')
                return []
    # Checking the tags of the input structure in FRAME2
    for i in range(len(needtags1)):
        if needtags1[i] not in frame2.keys():
            print('TAG ',needtags1[i],' NOT FOUND in input structure')
            return []
    for i in range(3):
        for j in range(len(needtags2)):
            if needtags2[j] not in frame2[j].keys():
                print('TAG ',needtags2[j],' NOT FOUND in input structure')
                return []

    # Temporary versions of the data
    f1 = frame1.copy()
    f2 = frame2.copy()

    sz = f1['chipa']['flux'][:,:].shape
    npix = sz[0]
    nfibers = sz[1]

    chipshift = np.zeros((3,2),float)
    pars = np.zeros(4,float)

    #-------------------------
    # Using CROSS-CORRELATION
    #-------------------------
    if xcorr:
        shtype = 'xcorr'
        print('Using Cross-correlation to measure the dither shift')

        if plugmap is not None:
            iplugind, = np.where(plugmap['fiberdata']['spectrographid'] == 2)
            iplugind = 300-plugmap['fiberdata'][iplugind]['fiberid']
        else:
            iplugind = np.arange(nfibers)
        f1chipaflux = f1.chipa.flux[:,iplugind]
        f1chipbflux = f1.chipb.flux[:,iplugind]
        f1chipcflux = f1.chipc.flux[:,iplugind]
        f2chipaflux = f2.chipa.flux[:,iplugind]
        f2chipbflux = f2.chipb.flux[:,iplugind]
        f2chipcflux = f2.chipc.flux[:,iplugind]
        sz = size(f1chipaflux[:,:])
        nfibers = sz[1]

        # Should use the PlugMap to only pick object spectra??

        # The input arrays are [2048,300,8]
        #  the planes are: [spec, wave, error, flag, sky, errsky,
        #      telluric, error_telluric]
        
        # Object spectra, normalize
        #  DON'T want to do this for SKY FIBERS!!!!
        if objspec:
            print('Median filtering the spectra')
            f1chipaflux = f1chipaflux / ( MEDFILT2D(f1chipaflux,100,dim=1,/edge_copy) > 1)
            f1chipbflux = f1chipbflux / ( MEDFILT2D(f1chipbflux,100,dim=1,/edge_copy) > 1)
            f1chipcflux = f1chipcflux / ( MEDFILT2D(f1chipcflux,100,dim=1,/edge_copy) > 1)
            f2chipaflux = f2chipaflux / ( MEDFILT2D(f2chipaflux,100,dim=1,/edge_copy) > 1)
            f2chipbflux = f2chipbflux / ( MEDFILT2D(f2chipbflux,100,dim=1,/edge_copy) > 1)
            f2chipcflux = f2chipcflux / ( MEDFILT2D(f2chipcflux,100,dim=1,/edge_copy) > 1)

        # Do the cross-correlation
        nlags = 21
        lags = np.arange(nlags)-nlags//2
        lo = nlags
        hi = npix-nlags-1

        fiber = np.zeros((nfibers,3),float)
        chip = np.zeros((nfibers,3),float)
        xshiftarr = np.zeros((nfibers,3),float)
        for ichip in range(3):
            xcorr = np.zeros((nlags,nfibers),float)
            for i in range(nlags):
                if ichip == 0:
                    xcorr[i,:] = np.sum(f1chipaflux[lo:hi,*]*f2chipaflux[lo+lags[i]:hi+lags[i],:],axis=0)
                if ichip == 1:
                    xcorr[i,:] = np.sum(f1chipbflux[lo:hi,*]*f2chipbflux[lo+lags[i]:hi+lags[i],:],axis=0)
                if ichip == 2:
                    xcorr[i,:] = np.sum(f1chipcflux[lo:hi,*]*f2chipcflux[lo+lags[i]:hi+lags[i],:],axis=0)

        for i in range(nfibers):
            fiber[i,ichip] = iplugind[i]
            chip[i,ichip] = ichip
            xshiftarr[i,ichip] = -100.

            xcorr0 = xcorr[:,i]
            if np.sum(xcorr0) == 0:
                continue
            # Now fit a Gaussian to it
            coef1 = ap_robust_poly_fit(lags,xcorr0,1)
            estimates = [max(xcorr0), 0.0, 2.0, coef1[0], coef1[1]]
            yfit1 = MPFITPEAK(lags,xcorr0,par1,nterms=5,estimates=estimates,
                              /gaussian,/positive,$
                              perror=perror1,chisq=chisq1,dof=dof1,
                              yerror=yerror1,status=status1)

            estimates = [par1[0], 0.0, par1[2], coef1[0], coef1[1], 0.0, 0.0]
            par2 = MPFITFUN('gausspoly',lags,xcorr0,xcorr0*0+1,estimates,
                            perror=perror2,chisq=chisq2,$
                            dof=dof2,yerror=yerror2,status=status2,yfit=yfit2,/quiet)
  
            gd, = np.where(abs(lags-par2[1]) lt 3.0*par2[2])
            if len(gd)==0:
                continue
            par3 = MPFITFUN('gausspoly',lags[gd],xcorr0[gd],xcorr0[gd]*0+1,par2,
                            perror=perror3,chisq=chisq3,
                            dof=dof3,yerror=yerror3,status=status3,yfit=yfit3,/quiet)
            
            xshiftarr[i,ichip] = par3[1]

        if shiftarr is None:
            shiftarr = xshiftarr
        # Measure mean shift
        gd, = np.where(xshiftarr > -99)
        ROBUST_MEAN,xshiftarr[gd],shift,shiftsig
        shifterr = shiftsig/np.sqrt(nfibers*3)
        # Printing the results
        print('Shift = {:.5f} +/- {:.5f} pixels'.format(shift,shifterr))

        #if nofit, we only have one row, so return chip offsets
        if nofit:
            pars=[0.]
            for ichip in range(3):
                gd, = np.where(xshiftarr[:,ichip] > -99)
                robust_mean,xshiftarr[gd,ichip],tmp,tmpsig
                pars = [pars,tmp]
            print(pars)
            shift = [shift,0.]
            #goto, fitend

        # do linear fits to the shifts
        # don't use blue fibers in superpersistence region
        if mjd < 56860:
            bd, = np.where((chip == 2) & (fiber > 200))
            if len(bd)>0:
                xshiftarr[bd] = -100

        # single slope and offset
        bad = ((xshiftarr < -99) | ((chip==2) & (fiber>200)))
        gd, = np.where(~bad)
        shift = AP_ROBUST_POLY_FIT(fiber[gd],xshiftarr[gd],1)
        # slope and offset for each chip
        print('Fit coefficients: ', shift)
        for ichip in range(3):
            bad = (xshiftarr[:,ichip] < -99)
            gd, = np.where(~bad)
            chipshift[ichip,:] = AP_ROBUST_POLY_FIT(fiber[gd,ichip],xshiftarr[gd,ichip],1)
        # Global fit for slope with row and 3 independent chip offsets
        for iter in range(3):
            gd, = np.where(xshiftarr > -99)
            print('n: ',len(gd))
            design = np.zeros((len(gd),4),float)
            design[:,0] = fiber[gd]
            for ichip in range(3):
                j, = np.where(chip[gd] == ichip)
                design[j,ichip+1] = 1.
            y = xshiftarr[gd]
            a = matrix_multiply(design,design,/atranspose)
            b = matrix_multiply(design,y,/atranspose)
            pars = invert(a)#b
            res = y-matrix_multiply(pars,design,/btrans)
            bd, = np.where(np.abs(res) > 5*dln.mad(res))
            print(pars)
            print('mad: ', dln.mad(res))
            xshiftarr[bd] = -100
        
        # Plot all the xcorr shifts
        if plot:
            #if plfile:
            #    set_plot,'PS'
            #    file_mkdir,file_dirname(pfile)
            #    device,file=pfile+'.eps',/encap,/color,xsize=16,ysize=16
            #    smcolor,/ps
            xr = [0,nfibers*3]
            yr = [-3,3]*dln.mad(xshiftarr)+np.median(xshiftarr)
            xr = [0,nfibers]
            yr = [-0.1,0.1]+np.median(xshiftarr)
            plt.plot(np.arange(nfibers),np.polyval(shift,np.arange(nfibers)),
                     linewidth=3)
            plt.xlabel('Spectrum #')
            plt.ylabel('Pixel Shift')
            plt.xlim(xr)
            plt.ylim(yr)
            plt.plot(xshiftarr[:nfibers])
            plt.plot(xshiftarr[nfibers:2*nfibers])
            plt.plot(xshiftarr[2*nfibers:3*nfibers])
            for ichip in range(3):
                if ichip=0: color=2
                if ichip=1: color=3
                if ichip=2: color=4
                plt.plot(np.arange(nfibers),np.polyval(chipshift[ichip,:],
                                                       np.arange(nfibers)),color=color)
            #f1 = strsplit(file_basename(frame1.chipa.filename,'.fits'),'-',/ext)
            #f2 = strsplit(file_basename(frame2.chipa.filename,'.fits'),'-',/ext)
            #al_legend,['Zero '+string(format='(f8.3)',shift[0]),
            #'Slope '+string(format='(e10.2)',shift[1])],textcolor=[1,1],/top,/left
            plt.legend()
            plt.savefig(pfile,bbox_inches='tight')
            #if keyword_set(pfile) then begin
            #device,/close
            #ps2gif,pfile+'.eps',/delete,/eps,chmod='664'o
            #endif

    #----------------------
    # Using EMISSION LINES
    #----------------------
    else:
        shtype = 'lines'
        print('Using Emission Lines to measure the dither shift')
        allmatchtab = []
        # Loop through the chips
        for i in range(3):
            print('Fitting lines for Chip ',chiptag[i])
            spec1 = f1[i]
            spec2 = f2[i]

            # Fit the peaks
            #   this can find peaks in ThAr or object frames
            print(' Frame 1')
            linetab1 = peakfit.peakfit(spec1)
            print(' Frame 2')
            linetab2 = peakfit.peakfit(spec2)            
            # Add chip numbers
            linetab1['chip'] = i+1
            linetab2['chip'] = i+1
            
            matchtab = np.zeros(50000,dtype=linetab1.dtype)
            
            # Now match the lines
            cntlines = 0
            for j in range(nfibers):
                gd1, = np.where(linetab1['fiber']==j)
                gd2, = np.where(linetab2['fiber']==j)
                if len(gd1)>0 and len(gd2)>0:
                    ifiber1 = linetab1[gd1]
                    ifiber2 = linetab2[gd2]
                    thresh = 1.0
                    ind1,ind2 = srcor2(ifiber1['fiber'],ifiber1['gaussx'],
                                       ifiber2['fiber'],ifiber2['gaussx'],
                                       thresh,opt=1)
                    nmatch = len(ind1)
                    if nmatch>0:
                        matchstr[cntlines:cntlines+nmatch]['f1'] = ifiber1[ind1]
                        matchstr[cntlines:cntlines+nmatch]['f2'] = ifiber2[ind2]
                        cntlines += nmatch

            # Trim trailing elements
            matchtab = matchtab[:cntlines]
            # Add to the total structure
            allmatchtab.append(matchtab)

        # Calculate the shift
        diffy = allmatchtab['f2']['gaussx']-allmatchtab['f1']['gaussx']
        nlines = len(allmatchtab)
        # don't want lines in superpersistence region, give different shifts
        gdlines, = np.where( ((allmatchtab['f2']['chip']<3) |
                              ((allmatchtab['f2']['chip']==3) &
                              allmatchtab['f2']['fiber'] < 200)) &
                             (allmatchtab['f1']['gerror'][1]<2) &
                             (allmatchtab['f2']['gerror'][1]<2))
        ROBUST_MEAN,diffy[gdlines],shift,shiftsig
        shifterr = shiftsig/np.sqrt(ngdlines)

        # use height and error to weight the values
        # don't use persistence region for blue chip
        err = np.maximum(np.sqrt(allmatchtab['f1']['gerror'][2]**2 +
                                 allmatchtab['f2']['gerror'][2]**2), 0.001)
        # use height > 300 and err lt 0.15 (or some percentile)
        # for blue chip use fiber < 200

        # Plot all of the shifts
        if pl:
            xr = [0,len(allmatchtab)]
            yr = [np.min(diffy),np.max(diffy)]
            plt.scatter(diffy[gdlines])
            plt.xlabel('Emission Line #')
            plt.ylabel('Pixel Shift')
            plt.xlim(xr)
            plt.ylim(yr)
            plt.axvline(shift,c='r')
            plt.annotate('Shift = {:.5f}+/-{:.5f} pixels'.format(shift,shifterr),
                         xy=[np.mean(xr),yr[1]-0.1*(yr[1]-yr[0])])
            #xyouts,mean(xr),yr[1]-0.1*(yr[1]-yr[0]),
            #'Shift = '+strtrim(shift,2)+'+/-'+strtrim(shifterr,2)+' pixels',$
            #align=0.5,charsize=1.5

        # Printing the results
        print('Shift = {:.5f} +/- {:.5f} pixels'.format(shift,shifterr))
        # This should be accurate to ~0.001 if there are ~18 lines per fiber
        shift = [shift,0.]

    shiftstr = {'type':shtype, 'shiftfit':shift, 'shifterr':shifterr,
                'chipshift':chipshift, 'chipfit:'pars}

    return shiftstr

def outcframe(frame,plugmap,outfiles):
    """
    This outputs a sky-corrected frame, apCframe
    This is called from ap1dvisit.pro

    Parameters
    ----------
    frame     The structure that contains the sky-corrected
              frame with all three chips.
    plugmap   The Plug Map structure for this plate
    /silent   Don't print anything to the screen

    Returns
    -------
    The frame is written to the "cframe" directory.

    Examples
    --------

    outcframe(frame)

    By D.Nidever  May 2010
    Modifications J. Holtzman 2011+
    Translated to Python by D. Nidever  July 2024
    """

    nframe = n_elements(frame)
    nplugmap = n_elements(plugmap)

    # Checking the tags of the input structure
    needtags1 = ['CHIPA','CHIPB','CHIPC','SHIFT','TELLSTAR']
    for i in range(len(needtags1)):
        if needtags1[i] not in frame.keys():
            print('TAG ',needtags1[i],' NOT FOUND in input structure')
            return
    needtags2 = ['HEADER','FLUX','ERR','MASK','WAVELENGTH','SKY',
                 'SKYERR','TELLURIC','TELLURICERR','LSFCOEF','WCOEF']
    for i in range(3):
        for j in range(len(needtags2)):
            if needtags2[j] not in frame[i].keys():
                print('TAG ',needtags2[j],' NOT FOUND in input structure')
                return

    # Get the frame information
    rawname = frame['chipa']['filename']
    info = apfileinfo(rawname,/silent)
    id8 = info['fid8']
    if id8 == '':
        id8 = info['suffix']

    # apCframe files contain the following:
    #    * HDU #0 = Header only
    #    * HDU #1 = Flux in units of ADU [FLOAT]
    #    * HDU #2 = Error in units of ADU [FLOAT]
    #    * HDU #3 = Pixel mask [32-bit INT]
    #    * HDU #4 = Wavelength in units of A [DOUBLE]
    #    * HDU #5 = Sky flux in units of ADU [FLOAT]
    #    * HDU #6 = Sky error in units of ADU [FLOAT]
    #    * HDU #7 = Telluric absorption flux in units of [FLOAT]
    #    * HDU #8 = Telluric error [FLOAT]
    #    * HDU #9 = Wavelength solution coefficients [BINARY FITS TABLE or DOUBLE]
    #    * HDU #10 = LSF coefficients [BINARY FITS TABLE or DOUBLE]
    #    * HDU #11 = Plug-map structure from plPlugMapM file [BINARY FITS TABLE]
    #    * HDU #12 = Plugmap header
    #    * HDU #13 = Telluric scaling table
    #    * HDU #14 = Shift information table
    #
    # There is a separate file for each chip - [abc]

    # Loop through the three chips
    for i in range(3):
        hdu = fits.HDUList()
        
        # Update the header:
        #-------------------
        header = frame[i]['header']
        
        # Remove the trailing blank lines
        indend = where(stregex(header,'^END',/boolean) eq 1,nindend)
        if indend[0] eq -1 then indend=n_elements(header)-1
        header = header[0:indend[0]]

        # Add extension explanations
        #----------------------------
        leadstr = 'AP1DVISIT: '
        header['HISTORY'] = leadstr+systime(0)
        info = GET_LOGIN_INFO()
        header['HISTORY'] = leadstr+info.user_name+' on '+info.machine_name
        header['HISTORY'] = leadstr+'IDL '+!version.release+' '+!version.os+' '+!version.arch
        header['HISTORY'] = leadstr+' APOGEE Reduction Pipeline Version: '+getvers()
        header['HISTORY'] = leadstr+'Output File:'
        header['HISTORY'] = leadstr+' HDU0 - Header only'
        header['HISTORY'] = leadstr+' HDU1 - Flux (ADU)'
        header['HISTORY'] = leadstr+' HDU2 - Error (ADU)'
        header['HISTORY'] = leadstr+' HDU3 - flag mask (bitwise OR combined)'
        header['HISTORY'] = leadstr+'        1 - bad pixels'
        header['HISTORY'] = leadstr+'        2 - cosmic ray'
        header['HISTORY'] = leadstr+'        4 - saturated'
        header['HISTORY'] = leadstr+'        8 - unfixable'
        header['HISTORY'] = leadstr+' HDU4 - Wavelength (Ang)'
        header['HISTORY'] = leadstr+' HDU5 - Sky (ADU)'
        header['HISTORY'] = leadstr+' HDU6 - Sky Error (ADU)'
        header['HISTORY'] = leadstr+' HDU7 - Telluric'
        header['HISTORY'] = leadstr+' HDU8 - Telluric Error'
        header['HISTORY'] = leadstr+' HDU9 - Wavelength coefficients'
        header['HISTORY'] = leadstr+' HDU10 - LSF coefficients'
        header['HISTORY'] = leadstr+' HDU11 - Plugmap structure'
        header['HISTORY'] = leadstr+' HDU12 - Plugmap header'
        header['HISTORY'] = leadstr+' HDU13 - Telluric structure'
        header['HISTORY'] = leadstr+' HDU14 - Shift structure'

        # Create filename
        #   apCframe-[abc]-ID8.fits 
        outfile = outfiles[i]
        print('Writing Cframe to ',outfile)

        # HDU #0 = Header only
        #----------------------
        hdu.append(fits.PrimaryHDU(header=header))
        
        # HDU #1 = Flux in units of ADU [FLOAT]
        #---------------------------------------
        flux = frame[i]['flux'].astype(np.float32)
        hdu.append(fits.ImageHDU(flux))
        hdu[1].header['CTYPE1'] = 'Pixel'
        hdu[1].header['CTYPE2'] = 'Fiber'
        hdu[1].header['BUNIT'] = 'Flux (ADU)'
        hdu[1].header['EXTNAME'] = 'FLUX'

        # HDU #2 = Flux Error in ADU [FLOAT]
        #------------------------------------
        bderr, = np.where(frame[i]['err'] >= BADERR)
        err = frame[i]['err'].astype(np.float32)
        if len(bderr)>0:
            err[bderr] = BADERR
        hdu.append(fits.ImageHDU(bderr))
        hdu[2].header['CTYPE1'] = 'Pixel'
        hdu[2].header['CTYPE2'] = 'Fiber'
        hdu[2].header['BUNIT'] = 'Flux Error (ADU)'
        hdu[2].header['EXTNAME'] = 'ERROR'  

        # HDU #3 = Pixel mask [32-bit INT]
        #---------------------------------
        mask = frame[i]['mask'].astype(np.int16)
        hdu.append(fits.ImageHDU(mask))
        hdu[3].header['CTYPE1'] = 'Pixel'
        hdu[3].header['CTYPE2'] = 'Fiber'
        hdu[3].header['BUNIT'] = 'Flag Mask (bitwise)'
        hdu[3].header['HISTORY'] = 'Explanation of BITWISE flag mask (OR combined)'
        hdu[3].header['HISTORY'] = ' 1 - bad pixels'
        hdu[3].header['HISTORY'] = ' 2 - cosmic ray'
        hdu[3].header['HISTORY'] = ' 4 - saturated'
        hdu[3].header['HISTORY'] = ' 8 - unfixable'
        hdu[3].header['EXTNAME'] = 'MASK'

        # HDU #4 = Wavelength in units of Ang [DOUBLE]
        #----------------------------------------------
        wave = frame[i]['wavelength'].astype(np.float64)
        hdu.append(fits.ImageHDU(wave))
        hdu[4].header['CTYPE1'] = 'Pixel'
        hdu[4].header['CTYPE2'] = 'Fiber'
        hdu[4].header['BUNIT'] = 'Wavelength (Ang)'
        hdu[4].header['EXTNAME'] = 'WAVELENGTH'
        
        # HDU #5 = Sky flux in units of ADU [FLOAT]
        #-------------------------------------------
        sky = frame[i]['sky'].astype(np.float32)
        hdu.append(fits.ImageHDU(sky))
        hdu[5].header['CTYPE1'] = 'Pixel'
        hdu[5].header['CTYPE2'] = 'Fiber'
        hdu[5].header['BUNIT'] = 'Sky (ADU)'
        hdu[5].header['EXTNAME'] = 'SKY FLUX'

        # HDU #6 = Sky error in units of ADU [FLOAT]
        #--------------------------------------------
        skyerr = frame[i]['skyerr'].astype(np.float32)
        hdu.append(fits.ImageHDU(skyerr))
        hdu[6].header['CTYPE1'] = 'Pixel'
        hdu[6].header['CTYPE2'] = 'Fiber'
        hdu[6].header['BUNIT'] = 'Sky Error (ADU)'
        hdu[6].header['EXTNAME'] = 'SKY ERROR'

        # HDU #7 = Telluric absorption flux in units of [FLOAT]
        #--------------------------------------------------------
        telluric = frame[i]['telluric'].astype(np.float32)
        hdu.append(fits.ImageHDU(telluric))
        hdu[7].header['CTYPE1'] = 'Pixel'
        hdu[7].header['CTYPE2'] = 'Fiber'
        hdu[7].header['BUNIT'] = 'Telluric'
        hdu[7].header['EXTNAME'] = 'TELLURIC'

        # HDU #8 = Telluric error [FLOAT]
        #-------------------------------------
        telerr = frame[i]['telluricerr'].astype(np.float32)
        hdu.append(fits.ImageHDU(telerr))
        hdu[8].header['CTYPE1'] = 'Pixel'
        hdu[8].header['CTYPE2'] = 'Fiber'
        hdu[8].header['BUNIT'] = 'Telluric Error'
        hdu[8].header['EXTNAME'] = 'TELLURIC ERROR'

        # HDU #9 = Wavelength solution coefficients [DOUBLE]
        #-----------------------------------------------------
        wcoef = frame[i]['wcoef'].astype(np.float64)
        hdu.append(fits.ImageHDU(wcoef))
        hdu[9].header['CTYPE1'] = 'Pixel'
        hdu[9].header['CTYPE2'] = 'Fiber'
        hdu[9].header['BUNIT'] = 'Wavelength Coefficients'
        hdu[9].header['HISTORY'] = 'Wavelength Coefficients to be used with PIX2WAVE.PRO:'
        hdu[9].header['HISTORY'] = ' 1 Global additive pixel offset'
        hdu[9].header['HISTORY'] = ' 4 Sine Parameters'
        hdu[9].header['HISTORY'] = ' 7 Polynomial parameters (first is a zero-point offset'
        hdu[9].header['HISTORY'] = '                     in addition to the pixel offset)'
        hdu[9].header['EXTNAME'] = 'WAVE COEFFICIENTS'

        # HDU #10 = LSF coefficients [DOUBLE]
        #-------------------------------------
        lsfcoef = frame[i]['lsfcoef'].astype(np.float64)
        hdu.append(fits.ImageHDU(lsfcoef))
        hdu[10].header['CTYPE1'] = 'Fiber'
        hdu[10].header['CTYPE2'] = 'Parameters'
        hdu[10].header['BUNIT'] = 'LSF Coefficients'
        hdu[10].header['HISTORY'] = 'LSF Coefficients to be used with LSF_GH.PRO:'
        hdu[10].header['HISTORY'] = '  binsize  The width of a pixel in X-units.  If this is non-zero'
        hdu[10].header['HISTORY'] = '             then a "binned" Gauss-Hermite function is used.  If'
        hdu[10].header['HISTORY'] = '             binsize=0 then a "normal, unbinned" Gauss-Hermite'
        hdu[10].header['HISTORY'] = '             function is used.'
        hdu[10].header['HISTORY'] = '  X0       An additive x-offset.  This is only used to'
        hdu[10].header['HISTORY'] = '             evaluate the GH parameters that vary globally'
        hdu[10].header['HISTORY'] = '             with X.'
        hdu[10].header['HISTORY'] = '  Horder   The highest Hermite order, Horder=0 means'
        hdu[10].header['HISTORY'] = '             only a constant term (i.e. only Gaussian).'
        hdu[10].header['HISTORY'] = '             There are Horder Hermite coefficients (since we fix H0=1).'
        hdu[10].header['HISTORY'] = '  Porder   This array gives the polynomial order for the'
        hdu[10].header['HISTORY'] = '             global variation (in X) of each LSF parameter.'
        hdu[10].header['HISTORY'] = '             That includes sigma and the Horder Hermite'
        hdu[10].header['HISTORY'] = '             coefficients (starting with H1 because we fix H0=1)'
        hdu[10].header['HISTORY'] = '             There will be Porder[i]+1 coefficients for'
        hdu[10].header['HISTORY'] = '             parameter i.'
        hdu[10].header['HISTORY'] = '  GHcoefs  The polynomial coefficients for sigma and the'
        hdu[10].header['HISTORY'] = '             Horder Hermite parameters.  There are Porder[i]+1'
        hdu[10].header['HISTORY'] = '             coefficients for parameter i.  The Hermite parameters'
        hdu[10].header['HISTORY'] = '             start with H1 since we fix H0=1.'
        hdu[10].header['EXTNAME'] = 'LSF COEFFICIENTS'
        
        # HDU #11 = Plug-map structure from plPlugMapM file [BINARY FITS TABLE]
        #----------------------------------------------------------------------
        plugdata = plugmap['fiberdata']
        hdu.append(fits.table_to_hdu(plugdata))

        # HDU # 12 = Plug-map header values
        #-------------------------------------
        # remove FIBERDATA and GUIDEDATA
        pltags = tag_names(plugmap)
        plind = indgen(n_elements(pltags))
        bd, = np.where(pltags == 'FIBERDATA',nbd)
        tmp = where(pltags == 'GUIDEDATA',nbd)
        if nbd gt 0 then bd=[bd,tmp]
        REMOVE,bd,plind
        newplug = CREATE_STRUCT(pltags[plind[0]],plugmap.(plind[0]))
        for k=1,n_elements(plind)-1 do newplug = CREATE_STRUCT(newplug,pltags[plind[k]],plugmap.(plind[k]))
        hdu.append(fits.table_to_hdu(newplug))
        
        # HDU # 13 = Telluric table
        hdu.append(fits.table_to_hdu(frame['tellstar']))
        
        # HDU # 14 = Telluric table
        hdu.append(fits.table_to_hdu(frame['shift']))

        hdu.writeto(outfile,overwrite=True)

def ditherpairs(shifttab,pairtab,verbose=False,snsort=False):
    """
    This program pairs up a number of dithered APOGEE frames

    Parameters
    ----------
    shifttab : table
        A structure that gives the shifts for all of the frames
         relative to the first. A positive shift means that
         frame2 is to the RIGHT of frame1 (the reference frame).
         This is normally measured by APDITHERSHIFT.PRO
         The SHIFTSTR structure should have the following tag:
             INDEX - A running index
             FRAMENUM - The ID8 frame number (string)
             SHIFT - The shift relative to the first frame
             SHIFTERR - Error in SHIFT
    verbose : bool, optional
       Print a lot to the screen.

    Returns
    -------
    pairtab : table
       A structure that pairs up the dithered frames,
         gives their relative shifts and other information
         that is needed to for APDITHERCOMB.PRO do combine
         the frames.  The 1st frame of the 1st pair is the
         new reference frame and has the most negative shift
         (original).
         The PAIRTAB structure has tags:
              FRAMENAME - 2-element string array of frame numbers
              FRAMENUM - Same as FRAMENAME but long type
              OLDSHIFT - 2-element array of the original SHIFT
              SHIFT - 2-element array of the shifts relative to
                        the NEW reference image (1st frame of 1st pair)
              RELSHIFT - The shift of 2nd frame wrt 1st frame
              NUSED - 2-element array giving the number of times
                        that the frame is used in the PAIRSTR
              INDEX - 2-element array giving the index in the
                        original SHIFTSTR.

    Examples
    --------

    apditherpairs,shiftstr,pairstr

    By D.Nidever  May 2010
    J.Holtzman, various mods
    Translated to Python by D. Nidever  July 2024
    """

    pairtab = []
    nframes = nshifttab
    if nframes < 2:
        error = 'Only ONE frame input. Need at least TWO'
        print(error)
        return


    # This is how the frames are paired:
    # 1.) need a frame that has a relative dither shift of at least 0.2 pixels
    # 2.) Closer in time is preferred
    # 3.) Not already taken as a pair is also preferred.

    # Frame structure for internal use to keep track of things
    framestr = REPLICATE({index:0L,framename:'',framenum:0L,shift:0.0,sn:0.0,$
                          pairframename:'',pairframenum:0L,$
                          pairshift:0.0,pairsn:0.0,pairindex:-1L,nused:0L},nframes)
    framestr.index = shiftstr.index
    framestr.framename = shiftstr.framenum
    framestr.framenum = long(shiftstr.framenum)
    framestr.shift = shiftstr.shift
    framestr.sn = shiftstr.sn

    # Print out information
    if verbose:
        print,strtrim(nframes,2),' frames input'
        print,' NUM   FRAME      SHIFT    S/N'
        for i=0,nframes-1 do print,format='(I4,A10,2F10.5)',i+1,framestr[i].framename,framestr[i].shift,framestr[i].sn

    minshift = 0.2  ; minimum shift for a dither pair
    maxshift = 0.8  ; maximum shift for a dither pair
    minsn = 3  ; minimum required S/N

    # Loop through the frames
    if snsort:
        isort=reverse(sort(framestr.sn))
    else:
        isort = np.arange(nframes)
    verbose = True
    for j in range(nframes):
        i = isort[j]
        # Not paired up yet
        if framestr[i]['nused']==0:
            relshift = framestr[isort].shift-framestr[i].shift # shift relative to this one
#    relshift[isort[i]] = 999999.0  ; don't want to pair with self

    fracrelshift = relshift-fix(relshift)  ; only the fraction of the shift

    gdframes = where(abs(fracrelshift) ge minshift,ngdframes)

    # No observation to pair this one with
    if ngdframes eq 0 then begin
      print,'No frame shifted enough for ',framestr[i].framename
      goto,BOMB

    if snsort:
      passind = where(abs(fracrelshift) ge minshift AND $
                    abs(fracrelshift) le maxshift AND $
                    framestr[isort].sn gt minsn AND $
                    framestr[isort].framenum ne framestr[i].framenum AND $
                    framestr[isort].nused eq 0,npassind)
    else:
      # Try following shifted frames that aren't paired yet
      passind = where(abs(fracrelshift) ge minshift AND $
                    framestr[isort].framenum gt framestr[i].framenum AND $
                    framestr[isort].nused eq 0,npassind)

      # Try preceding shifted frames that aren't paired yet
      if npassind eq 0 then $
      passind = where(abs(fracrelshift) ge minshift AND $
                    framestr[isort].framenum lt framestr[i].framenum AND $
                    framestr[isort].nused eq 0,npassind)

    # Try following shifted frames that ARE paired
    # No observation to pair with
    if npassind eq 0:
      print,'No frame to pair with ',framestr[i].framename
      goto,BOMB

    # Do the pairing
    ipair = isort[passind[0]]
    framestr[i]['pairframename'] = framestr[ipair]['framename']
    framestr[i]['pairframenum'] = framestr[ipair]['framenum']
    framestr[i]['pairshift'] = framestr[ipair]['shift']
    framestr[i]['pairsn'] = framestr[ipair]['sn']
    framestr[i]['pairindex'] = ipair
    framestr[i]['nused'] += 1
    framestr[ipair]['nused'] += 1
    if verbose:
        print('Pairing ',framestr[i]['framename'],' with ',
              framestr[i]['pairframename'])

    # Add to the pair structure
    dt = [('framename',str,(100,2)),('framenum',str,(100,2)),
          ('oldshift',float,2),('shift',float,2),('sn',float,2),
          ('refshift',float),('relshift',float),('nused',int,2),('index',int,2)]
    newpairtab = np.zeros(1,dtype=np.dtype(dt))
    #newpairtab = {framename:strarr(2),framenum:lonarr(2),
    #              oldshift:fltarr(2),shift:fltarr(2),sn:fltarr(2),
    #              refshift:0.0,relshift:0.0,nused:lonarr(2),index:lonarr(2)}
    newpairtab['framename'] = [framestr[i]['framename'], framestr[i]['pairframename']]
    newpairtab['framenum'] = [framestr[i]['framenum'], framestr[i]['pairframenum']]
    newpairtab['oldshift'] = [framestr[i]['shift'],framestr[i]['pairshift']]
    newpairtab['shift'] = [framestr[i]['shift'],framestr[i]['pairshift']]
    newpairtab['sn'] = [framestr[i]['sn'],framestr[i]['pairsn']]
    newpairtab['relshift'] = newpairtab['shift'][1]-newpairtab['shift'][0] # relative to first frame
    newpairtab['index'] = [i,framestr[i]['pairindex']]
    # we'll update the Nused values later
    pairtab.append(newpairtab)

    npairs = len(pairtab)
    if npairs == 0:
        error = 'NO PAIRS'
        print(error)
        return []
    
    # Put the pair with the most POSITIVE shift first
    #  it will have the lowest wavelength on the left
    #pairshifts = MIN(pairtab.shift,dim=1)
    #refind = first_el(minloc(pairshifts))
    pairshifts = np.ax(pairtab['shift'],axis=0)
    refind = first_el(maxloc(pairshifts))
    if refind != 0:
        refpair = pairtab[refind]
        REMOVE,refind,pairtab
        pairtab = [refpair,pairtab]
    # Change SHIFT so it is relative to the reference image
    #refshift = pairtab[0].shift[0]
    #pairtab.shift -= refshift
    refshift = np.max(pairtab[0]['shift'])
    pairtab['shift'] = refshift - pairtab['shift']
    # THIS SHIFT now indicates where this frame BEGINS (e.g. wavelength)
    #  relative to the reference frame.

    if verbose:
        print(npairs,' PAIRS')
        print(' NUM   FRAME1    FRAME2    SHIFT1    SHIFT2 NUSED1 NUSED2 RELSHIFT')

    # Loop through the pair structure
    for i in range(npairs):
        # Update NUSED in the pair index
        frame1 = pairtab[i]['framename'][0]
        frame2 = pairtab[i]['framename'][1]

        # How often was the first frame used
        used1, = np.where(pairtab['framename'][0] == frame1 or 
                          pairtab['framename'][1] == frame1)

        # How often was the second frame used
        used2, = np.where(pairtab['framename'][0] == frame2 or 
                          pairtab['framename'][1] == frame2)
        
        pairtab[i]['nused'] = [nused1,nused2]

        # They are in the wrong order, FLIP
        #  want the first spectrum to have the lowest wavelength
        if pairtab[i]['relshift'] > 0.0:
            thistab = pairtab[i]
            pairtab[i]['framename'] = reverse(thispair['framename'])
            pairtab[i]['framenum'] = reverse(thispair['framenum'])
            pairtab[i]['oldshift'] = reverse(thispair['oldshift'])
            pairtab[i]['shift'] = reverse(thispair['shift'])
            pairtab[i]['sn'] = reverse(thispair['sn'])
            pairtab[i]['nused'] = reverse(thispair['nused'])
            pairtab[i]['index'] = reverse(thispair['index'])
            #pairtab[i]['relshift'] = -thispair.relshift
        # right order
        else:
            # want RELSHIFT to be POSITIVE
            pairtab[i]['relshift'] = -pairtab[i]['relshift']

        # REFSHIFT
        #  This is the shift of the dither pair relative to the
        #  "reference" frame.  Should just be equal to the
        #  "shift" value of the first frame of the pair.
        pairtab[i]['refshift'] = pairtab[i]['shift'][0]

        # Printing out the pairs
        if verbose:
            fmt = '{:4d}{:10s}{:10s}{:10.5f}{:10.5f}{:5d}{:5d}{:10.5f}'
            print(fmt.format(i+1,pairtab[i]['framename'][0],pairtab[i]['framename'][1],
                  pairtab[i]['shift'][0],pairtab[i]['shift'][1],pairtab[i]['nused'][0],
                  pairtab[i]['nused'][1],pairtab[i]['relshift']))

        # The "0th" frame might not be first.

        # Use the indices of the shiftstr to correctly identify the
        # right frame in the "allframes" structure

    return pairtab


def dithercomb(allframes,shiftstr,pairstr,plugmap,outframe,noscale=noscale,
                 globalwt=globalwt,nodither=nodither,verbose=verbose,
                 newerr=newerr,npad=npad,median=median,avgwave=avgwave):
    """
    This combines a number of dithered APOGEE frames.

    Parameters
    ----------
    allframes    An array of "frame" structures with the three chip headers and data for
                 all of the frames.
    shiftstr     A structure that gives the shifts for all of the frames
                 relative to the first. A positive shift means that
                 frame2 is to the RIGHT of frame1 (the reference frame).
                 This is normally measured by APDITHERSHIFT.PRO
    plugmap      The Plug Map structure for this plate
    /noscale     Do NOT scale the two frames.  This would be used for
                 ThAr or other frames where there are no throughput
                 variations.
    /globalwt    When combining the dither-combined spectra use a
                 single weight for each spectrum (NOT pixel-by-pixel).
    /nodither    No dithers were performed. Just combine the spectra.
    /newerr      New and improved method for combining error arrays.
    /avgwave     Average the wavelength solutions of the frames.
    /verbose     Print a lot to the screen.

    Returns
    -------
    outframe      A structure that contains the combined images
                 and headers of the three chips for the dither pair.

    Examples
    --------

    dithercomb(allframes,shiftstr,plugmap,outframe)

    By D.Nidever  March 2010
    Significant revisions: Jon Holtzman Feb 2012
    Translated to Python by D. Nidever  July 2024
    """

    outframe = None
    nallframes = n_elements(allframes)
    nshiftstr = n_elements(shiftstr)
    nplugmap = n_elements(plugmap)

    # Not enough inputs
    if nallframes eq 0 or nshiftstr eq 0 or nplugmap eq 0:
        return

    # Checking the tags of the input structure
    for f in range(nallframes):
        tags = tag_names(allframes[f])
        needtags1 = ['CHIPA','CHIPB','CHIPC']
        for i in range(len(needtags1)):
            if (where(tags eq needtags1[i]))[0] eq -1:
                print('TAG ',needtags1[i],' NOT FOUND in input structure')
                return []
        needtags2 = ['HEADER','FLUX','ERR','MASK']
        for i in range(3):
            tags2 = tag_names(allframes[f].(i))
            for j in range(len(needtags2)):
                if (where(tags2 eq needtags2[j]))[0] eq -1:
                    print('TAG ',needtags2[j],' NOT FOUND in input structure')
                    return []

    # Only one exposure
    if nallframes == 1:
        print('Only one exposure.  Nothing to combine.')
        outframe = allframes
        return outframe

    # No dither
    if nodither:
        print('NO Dithers, just combining')
        allcombframes = allframes
        npix = n_elements(allframes[0].(0).flux[*,0])
        nfibers = n_elements(allframes[0].(0).flux[0,*])
        npairs = nallframes  # Npairs=Nexposures
        y2 = findgen(npix)
        goto,combine

    #-----------------------------
    # PART I - PAIR UP THE FRAMES
    #-----------------------------
    pairtab = ditherpairs(shifttab,verbose=verbose,/snsort)
    npairs = len(pairtab)
    if npairs == 0 and nodither==False:
        print('Error: no dither pairs.')  
        return []

    if npairs == 0:
        print,'No dither pairs.  Assuming /nodither and just combining'
        nodither = 1
        allcombframes = allframes
        npix = n_elements(allframes[0].(0).flux[*,0])
        nfibers = n_elements(allframes[0].(0).flux[0,*])
        npairs = nallframes  # Npairs=Nexposures
        y2 = findgen(npix)
        goto,combine
        return []
    if n_elements(error) > 0:
        error = 'There was a problem with the pairing'
        print(error)
        return

    # Use the first frame of the first pair as the reference frame

    refind = pairtab[0]['index'][0]
    refframe = allframes[refind]
    sz = refframe['chipa']['flux'].shape
    npix = sz[1]
    nfibers = sz[2]
    ncol = sz[3]
    
    ## Size not the same
    #if total(abs(sz1-sz2)) ne 0 then begin
    #  print,'Sizes are not the same'
    #  return
    #endif


    # THINGS TO ADD:
    # -NEED TO SINC INTERPOLATE THE DITHER PAIRS ONTO THE SAME FINAL PIXEL
    #   ARRAY
    #  --> I believe this is being done
    # -NEED TO DEAL WITH "MISSING" PIXELS AND "EXTRA" PIXELS AT THE ENDS
    # -NEED TO DEAL WITH NOT COUNTING FRAMES MULTIPLE TIMES IN THE
    #   VARIANCE ARRAY
    #  --> don't use frames multiple times!
    # -WHEN COMBINING THE WELL-SAMPLED SPECTRA I NEED TO RESCALE THEM
    #   AGAIN.
    #  --> I don't think so, they should be weighted by errors
    # -PUT ALL THE INFORMATION IN THE HEADER
    
    # I think not counting frames multiple times needs to be done in
    # part II just before doing sincinterlace for the variance.
    # need to inflate the variance for the spectrum/frame that is being
    # used multiple times.

    # The first spectrum of the first pair has the largest shift to the
    # left.  Use this as the reference frame.

    #---------------------------------------------
    # PART II - SINC INTERLACE THE DITHER PAIRS
    #---------------------------------------------
    # Loop through the pairs
    for p in range(npairs):
        ipairstr = pairstr[p]
        frame1 = allframes[ipairstr.index[0]]
        frame2 = allframes[ipairstr.index[1]]
        shift = ipairstr.relshift
        ## Reference frame is first frame of first pair
        i0 = pairstr[0].index[0]
        ## indices of the frames for this pair
        i1 = ipairstr.index[0]
        i2 = ipairstr.index[1]
        print,'Combining Pair ',strtrim(p+1,2),' - ',ipairstr.framename[0],' + ',ipairstr.framename[1]

        ## Initialize combframe structure. Need 2x as many pixels, if dithered
        for i in range(3):
            chstr0 = frame1.(i)
            tags = tag_names(chstr0)
            apgundef,chstr

            ## Make the chip structure which will go into the combframe structure
            for c in tags:
                arr = chstr0.(j)
                type = size(arr,/type)
                # Data arrays. These are [NPix,Nfiber]
                dum = where(stregex(['FLUX','ERR','MASK','WAVELENGTH','SKY','SKYERR','TELLURIC','TELLURICERR'],tags[j],/boolean) eq 1,ndata)
                if ndata gt 0 then arr=make_array(2*npix,nfibers,type=type)
                
                ## Add normal tags/data
                if tags[j] ne 'FILENAME':
                if n_elements(chstr) eq 0:
                    chstr = CREATE_STRUCT(tags[j],arr)
                else:
                    chstr = CREATE_STRUCT(chstr,tags[j],arr)

      ## Add FILENAME1 and FILENAME2
      endif else begin
        if n_elements(chstr) eq 0 then begin
          chstr = CREATE_STRUCT('FILENAME1',frame1.(0).filename,'FILENAME2',frame2.(0).filename)
        endif else begin
          chstr = CREATE_STRUCT(chstr,'FILENAME1',frame1.(0).filename,'FILENAME2',frame2.(0).filename)
        endelse
      endelse
    endfor # tag loop

    ## Add to the final COMBFRAME
    if i == 0:
      combframe = CREATE_STRUCT('chip'+chiptag[i],chstr)
    else:
      combframe = CREATE_STRUCT(combframe,'chip'+chiptag[i],chstr)

  Endfor # chip loop
  combtags = tag_names(combframe.(0))


    #----------------------
    # COMBINE THE FRAMES
    #----------------------

#  shift_rnd = round(shift*1000.0)/1000.0  # round to nearest 1/1000th of a pixel
#  shift=shift_rnd

    # Combine the data with SINC interlacing
    #------------------------------------------
    y = np.arange(npix)

  #print,'Combining the dither images with SINC interlacing'

  ## Make dummy chip structure with quantities we need to combine
  #apgundef,usetags,usetagsnum,f0ch
  tags = tag_names(frame1.(0))
  postags = ['FLUX','ERR','MASK','WAVELENGTH','SKY','SKYERR','TELLURIC','TELLURICERR']  # use these if they exist
  for k=0,n_elements(tags)-1 do begin
    gd, = np.where(postags == tags[k],ngd)
    if ngd > 0:
      PUSH,usetags,tags[k]
      PUSH,usetagsnum,k
      if n_elements(f0ch) == 0:
        f0ch = CREATE_STRUCT(tags[k],(frame1.(0).(k))[*,0])
      else:
        f0ch = CREATE_STRUCT(f0ch,tags[k],(frame1.(0).(k))[*,0])
      endelse
    endif
  endfor
  PUSH,usetags,'SCALE'
  f0ch = CREATE_STRUCT(f0ch,'SCALE',fltarr(npix))
  nusetags = len(usetags)

  ## We will "pad" the spectra at either end with masked pixels
  ##   to determine which of the edge pixels should be declared
  ##   bad
  if not keyword_set(npad) then npad=0
  if npad > 0:
    lftpad = fltarr(npad)
    rtpad = fltarr(npad)
    maskpad = fltarr(npad)
    maskpad[*] = 1

  ## Loop through the fibers
  ##-------------------------
  for jfiber in range(nfibers):
    if (jfiber+1) % 50 == 0:
        print(strtrim(jfiber+1,2),'/',strtrim(nfibers,2))

    ## The plugmap index for this fiber
    ## fiberid=1 is at the top of the detector or index=299
    ## index = 300-fiberid
    iplugind, = np.where((plugmap['fiberdata']['spectrographId']==2) &
                         (plugmap['fiberdata']['fiberid']==300-jfiber))
    ## No information for this fiber
    if len(iplugind)==0:
      print('No information for Fiber=',strtrim(300-jfiber,2),' in the plugmap file')
      goto,BOMB

    ## Fiber type from the plugmap structure
    fiberobjtype = plugmap['fiberdata'][iplugind]['objtype']
    fiberobjid = plugmap['fiberdata'][iplugind]['tmass_style']

    ## Loop through the chips
    for ichip in range(3):
        ## Insert the quantities that we want to combine
        data1 = f0ch
        for k=0,nusetags-2 do data1.(k) = (frame1.(ichip).(usetagsnum[k]))[*,jfiber]
        data2 = f0ch
        for k=0,nusetags-2 do data2.(k) = (frame2.(ichip).(usetagsnum[k]))[*,jfiber]

        ## Replace bad pixels and errors with smoothed version to minimize their impact on adjacent pixels
        ## We will extend masks if these pixels are important
        if median:
            bd, = np.where(data1.mask AND badmask(),nbd)
            if len(bd)>0:
                data1.flux[bd]=!values.f_nan
	    scale1 = smooth(medfilt1d(data1.flux,501,edge=2),100,/nan)
	    tmperr = smooth(medfilt1d(data1.err,501,edge=2),100,/nan)
            if len(bd)>0:
                data1['flux'][bd] = scale1[bd]
                data1['err'][bd] = tmperr[bd]
            bd, = np.where(data2['mask'] and BADMASK)
            if len(bd)>0:
                data2['flux'][bd] = np.nan
	    scale2 = smooth(medfilt1d(data2['flux'],501,edge=2),100,/nan)
	    temperr = smooth(medfilt1d(data2['err'],501,edge=2),100,/nan)
            if len(bd)>0:
                data2['flux'][bd] = scale2[bd]
                data2['err'][bd] = tmperr[bd]
        else:
            ## Fit a low-order polynomial to the data
            scalecoef1 = ROBUST_POLY_FIT(y,data1.flux,5)
            scale1 = np.polyval(y,scalecoef1,y)
            scalecoef2 = ROBUST_POLY_FIT(y,data2.flux,5)
            scale2 = np.polyval(y,scalecoef2,y)

      ## Need to normalize them if they are object spectra
      ## Don't do this for sky spectra, since they may have real variations
      ##   of course, this means you can't really dither-combine the sky spectra, either...
      ##   but we will just to put something in the dither-combined output
      ## Use the plugmap information to see which fibers are sky
      ## Only normalize the flux and errors, not sky or telluric!
      if (fiberobjtype != 'SKY') and noscale=False:
          # "Normalize" the spectra in case there's been some variation in response
          data1['flux'] = data1['flux']/scale1        # normalize spectrum
          data1['err'] = data1['err']/scale1          # normalize error
          data1['scale'] = scale1
          data2['flux'] = data2['flux']/scale2        # normalize spectrum
          data2['err'] = data2['err']/scale2          # normalize error
          data2['scale'] = scale2

      # We want to interpolate onto a 1/2 pixel scale.  So we only
      # really need to interpolate half of the pixels, the other half
      # just stay the way they are.
      #-------------------
      # FRAME TWO is FIRST
      #-------------------
      # Frame2 is a fraction of a pixel to the RIGHT of Frame1
      #  and so for Frame2 the detector moved to the LEFT
      #  and Frame2 should be interleaved FIRST, i.e.
      #  [Frame2, Frame1]

      ##  Now the planes are: [spec, wave, error, flag, sky, errsky,
      ##    telluric, error_telluric]

      ## Relative shift between this pair and the ABSOLUTE frame
      abs_shift = ipairstr['refshift']

      # Dec 2018: use chip and fiber dependent shifts
      # shifts are all measured in the reverse direction from the maximum shift (which is put in first pair, see apditherpairs)
      #new_abs_shift = allframes[i0].shift.shiftfit[0] - allframes[i1].shift.shiftfit[0]
      #new_shift = allframes[i0].shift.shiftfit[0] - allframes[i2].shift.shiftfit[0] - new_abs_shift
      #shift_i0 = allframes[i0].shift.shiftfit[0]
      #shift_i1 = allframes[i1].shift.shiftfit[0]
      #shift_i2 = allframes[i2].shift.shiftfit[0]
      ## CHIPFIT, chip-dependent shifts, [Fiber term, chipa offset, chipb offset, chipc offset]
      shift_i0 = (allframes[i0]['shift']['chipfit'][0]*jfiber +
                  allframes[i0]['shift']['chipfit'][ichip+1])
      shift_i1 = (allframes[i1]['shift']['chipfit'][0]*jfiber +
                  allframes[i1]['shift']['chipfit'][ichip+1])
      shift_i2 = (allframes[i2]['shift']['chipfit'][0]*jfiber +
                  allframes[i2]['shift']['chipfit'][ichip+1])
      new_abs_shift = shift_i0 - shift_i1
      new_shift = shift_i1 - shift_i2
      #print,new_shift,new_abs_shift
      shift = new_shift
      abs_shift = new_abs_shift

      ## if abs_shift is positive then we are to the RIGHT of the
      ## absolute frame and we want pixels to the LEFT

      ## If the shift is large, then SINCINTERLACED.PRO will return
      ## garbage for the "overhanging/missing" pixels.  We just need
      ## to modify these to show they are bad.
      ## This is done at the end.

      ## Combine the SPECTRA
      ##---------------------
      ## pad the data and error arrays
      if npad == 0:
          spec1 = data1['flux']
          spec2 = data2['flux']
          err1 = data1['err']
          err2 = data2['err']
      else:
          lftpad[:] = data1['flux'][0]
          rtpad[:] = data1['flux'][npix-1]
          spec1 = [lftpad,data1.flux,rtpad]   # the leftmost (in wavelength)
          lftpad[:] = data1['err'][0]
          rtpad[:] = data1['err'][npix-1]
          err1 = [lftpad,data1['err'],rtpad]
          lftpad[:] = data2['flux'][0]
          rtpad[:] = data2['flux'][npix-1]
          spec2 = [lftpad,data2['flux'],rtpad]   # starting to the right
          lftpad[:] = data2['err'][0]
          rtpad[:] = data2['err'][npix-1]
          err2 = [lftpad,data2['err'],rtpad]

      ## Do the interlaced sinc interpolation
      ##  If shift = 0.5 then it will output the appropriate spectrum
      ##  and NOT do the interpolation

      ## Need to get the interpolation on the right absolute scale
      ##  spec1 is the leftmost
      ##  spec2 starts at a higher value, but goes first
      spec_lfthalf = SINCINTERLACED(spec2,spec1,shift, 0.0-abs_shift,
                        err1=err2,err2=err1,errout=err_lfthalf)
      spec_rthalf = SINCINTERLACED(spec2,spec1,shift, 0.5-abs_shift,
                        err1=err2,err2=err1,errout=err_rthalf)
      ## Now combine the spectra, without the padded pixels
      combspec = np.zeros(npix*2,float)
      combspec[:npix*2-1:2] = spec_lfthalf[npad:npad+npix]
      combspec[1:npix*2:2] = spec_rthalf[npad:npad+npix]
      comberr = np.zeros(npix*2,float)
      comberr[:npix*2-1:2] = np.sqrt(err_lfthalf[npad:npad+npix])
      comberr[1:npix*2:2] = np.sqrt(err_rthalf[npad:npad+npix])
      ## RENORMALIZE THE DATA back up
      ##  use the average of scale1 + scale2
      if fiberobjtype != 'SKY' and noscale==False:
          #y2 = findgen(npix*2)/2
          ## Using continuum poly fit for Frame1
          ##if data1[npix/2,3] gt data2[npix/2,3] then begin
          #if scale1[npix/2] gt scale2[npix/2] then begin
          #  rescale = POLY(y2,scalecoef1) / scale1[npix/2] * MEAN([ scale1[npix/2], scale2[npix/2] ])
          ## Using continuum poly fit for Frame2
          #endif else begin
          #  rescale = POLY(y2,scalecoef2) / scale2[npix/2] * MEAN([ scale1[npix/2], scale2[npix/2] ])
          #endelse
          samp = np.arange(npix*2)/2.
          rescale = interpolate((scale1+scale2)/2.,samp)
          combspec = combspec * rescale                       # rescale spectrum
          comberr = comberr * rescale                         # rescale error

      ## old way of doing the error is to interlace the scaled error arrays,
      ##  but since there may be a mismatch in S/N, this doesn't seem
      ##  like a good idea
      if newerr==False:
          # Combine the ERROR
          #----------------------
          err1 = data1['err']
          err2 = data2['err']
          # Do the interlaced sinc interpolation
          ##  If shift = 0.5 then it will output the appropriate fiber
          ##  and NOT do the interpolation
          ## Need to get the interpolation on the right absolute scale
          ##  err1 is the leftmost
          ##  err2 starts at a higher value, but goes first
          err_lfthalf = SINCINTERLACED(err2,err1,shift, 0.0-abs_shift)
          err_rthalf = SINCINTERLACED(err2,err1,shift, 0.5-abs_shift)

          ## Now combine the error
          comberr = np.zeros(npix*2,float)
          comberr[:npix*2-1:2] = err_lfthalf
          comberr[1:npix*2:2] = err_rthalf
          if fiberobjtype != 'SKY' and noscale==False:
              comberr = comberr * rescale                         # rescale error
          # Make sure we don't get negative or zero values
          bderr, = np.where(comberr <= 0.0)
          if len(bderr) > 0:
              gderr = where(comberr gt 0.0,ngderr)
              comberr[bderr] = min(comberr[gderr])  # use the minimum "good" error
              # should these pixels be considered "bad" and put
              # to a high value???

      ## Put results in output structure
      combframe[ichip]['flux'][:,jfiber] = combspec
      combframe[ichip]['err'][:,jfiber] = comberr

      ## Get the contribution of masked pixels to the output. Do this separately
      ##   for each bit to keep track of the flags
      combframe.(ichip).mask[*,jfiber]=0
      getmaskvals,flag,badflag,maskcontrib
      for ibit in range(len(flag)):
          vmask = 2**ibit
          mask1 = np.minimum((data1['mask'] & vmask),1)
          mask2 = np.minimum((data2['mask'] & vmask),1)
          nmask1 = np.sum(mask1>0)
          nmask2 = np.sum(mask2>0)
          ## Only continue if we have any of these bits set!
          if nmask1 > 0 or nmask2 > 0:
            if npad == 0:
               mask1 = float(mask1)
               mask2 = float(mask2)
             else:
               mask1 = [maskpad,float(mask1),maskpad]   # the leftmost (in wavelength)
               mask2 = [maskpad,float(mask2),maskpad]   # starting to the right
          spec_lfthalf = SINCINTERLACED(mask2,mask1,shift, 0.0-abs_shift)
          spec_rthalf = SINCINTERLACED(mask2,mask1,shift, 0.5-abs_shift)
          combmask = np.zeros(npix*2,float)
          combmask[:npix*2-1:2] = spec_lfthalf[npad:npad+npix]
          combmask[1:npix*2:2] = spec_rthalf[npad:npad+npix]
          ## Any pixel that has more than allowed contribution from a bad pixel gets this mask set
          bd, = np.where(abs(combmask) > maskcontrib[ibit])
          if len(bd)>0:
              combframe[ichip]['mask'][bd,jfiber] = combframe[ichip]['mask'][bd,jfiber] | vmask
      ## If this maskval corresponds to a bad pixel, inflate the error
      bd, = np.where(combframe[ichip]['mask'][:,jfiber] & BADMASK)
      if len(bd)>0:
          combframe[ichip]['err'][bd,jfiber] *= 10.

      ## Flag "bad/missing" pixels at the ends
      ##--------------------------------------
      if npad == 0 and abs(abs_shift) > 0.6:
          # how many pixels are bad,  we can interpolate on pixel at ends
          #  the shift is in original pixels
          #  we are flagging dither combined pixels
          nbadpix = floor(abs(abs_shift*2)) - 1

          # pixels at beginning are bad
          if abs_shift gt 0.0:
              # just set the flag value to NAN
              #  this might already happen above when the non-interpolated
              #  values are shifted
              #combframe.(ichip).data[0:nbadpix-1,jfiber,*] = !values.f_nan
              #combframe.(ichip).mask[0:nbadpix-1,jfiber] = !values.f_nan
              combframe[ichip]['mask'][:nbadpix,jfiber] = maskval('BADPIX')

        # pixels at the end are bad
        else:
            # just set the flag value to NAN
            #combframe.(ichip).data[npix*2-nbadpix:npix*2-1,jfiber,*] = !values.f_nan
            #combframe.(ichip).mask[npix*2-nbadpix:npix*2-1,jfiber] = !values.f_nan
            combframe[ichip]['mask'][npix*2-nbadpix:npix*2,jfiber] = maskval('BADPIX')

      ## Make the combined wavelength array
      ##-------------------------------------
      ##  Just use the wavelength coefficients of Frame 1
      ##    since it is on the left (lowest wavelength)
      dum = where(usetags == 'WAVELENGTH',Nwavelength)
      if Nwavelength>0:
        w1 = data1['wavelength']
        w2 = data2['wavelength']
        wcoef1 = frame1[ichip]['wcoef'][jfiber,:]
        wcoef2 = frame2[ichip]['wcoef'][jfiber,:]

        ## We need wavelengths on the absolute scale
        ##  same as for the sinc interpolation
        wave1_lfthalf = pix2wave(y+0.0-abs_shift,wcoef1)
        wave1_rthalf = pix2wave(y+0.5-abs_shift,wcoef1)
        wave1 = np.zeros(npix*2,float)
        wave1[:npix*2-1:2] = wave1_lfthalf
        wave1[1:npix*2:2] = wave1_rthalf

        wave2_lfthalf = pix2wave(y+0.0-abs_shift-shift,wcoef2)
        wave2_rthalf = pix2wave(y+0.5-abs_shift-shift,wcoef2)
        wave2 = np.zeros(npix*2,float)
        wave2[:npix*2-1:2] = wave2_lfthalf
        wave2[1:npix*2:2] = wave2_rthalf
        
        ## Now average the wavelengths
        if avgwave:
            combwave = 0.5*(wave1+wave2)
            ## Fit polynomial model to the wavelengths
            newy = np.zeros(npix*2,float)
            newy[:npix*2-1:2] = y+0.0-abs_shift
            newy[1:npix*2:2] = y+0.5-abs_shift
            ## poly2wave.pro scales the x values using (x+xoffset)/3000.0
            newcoef = poly_fit((newy+wcoef1[0])/3000.,combwave,3)
            combwcoef = wcoef1
            combwcoef[6:9] = newcoef

        ## Use the frame1 wavelengths (old way)
        else:
            combwave = np.zeros(npix*2,float)
            combwave[:npix*2-1:2] = wave1_lfthalf
            combwave[1:npix*2:2] = wave1_rthalf
            combwcoef = wcoef1
            ##  modify wcoef1 for the absolute shift
            combwcoef1[0] = wcoef1[0]-abs_shift

        ## Copy the Wavelength coefficients to the output frame
        combframe[ichip]['wcoef'][jfiber,:] = combwcoef

        ## Stuff it in the output structure
        combframe[ichip]['wavelength'][:,jfiber] = combwave

      ## Combine flags, sky, errsky, telluric, error_telluric
      ##-------------------------------------------------------
      ##   the sky and telluric come from different dithered
      ##   exposures and will be on different levels, so we can't sinc interlace them
      ##   just do a simple interlace (really, we should shift these!)
      extratags = ['SKY','SKYERR','TELLURIC','TELLURICERR']  # all possible extras
      nextra = len(extratags)

      ## Loop through the extra columns
      for k in range(nextra):
        xtraind = where(usetags eq extratags[k],nxtraind)
        combind = where(combtags eq extratags[k],ncombind)

        ## We have this extra column
        if nxtraind>0:
          ## Combine the data
          combextra = np.zeros(npix*2,float)
          combextra[:npix*2-1:2] = data2.(xtraind)
          combextra[1:npix*2:2] = data1.(xtraind)

          ## SHIFT NON-INTERPOLATED VALUES
          ##  if abs_shift is greater than 1/2 dither combined pixel
          ##  then we need to shift these values
          if abs(abs_shift*2) > 0.5:
            combextra0 = combextra
            xtra_shift = ceil(abs(abs_shift*2))

            ## if abs_shift is positive then extra pixels will be "added"
            ## on the left side and our arrays need to shifted to the right

            ## shift to the right
            if abs_shift>0:
              combextra = shift(combextra,xtra_shift)
              combextra[0:xtra_shift-1] = !values.f_nan  # these values are garbage

            ## shift to the left
            else:
               combextra = shift(combextra,-xtra_shift)
               combextra[npix*2-xtra_shift:npix*2-1] = !values.f_nan   # these values are garbage

          # shifting

          ## Stuff in the output structure
          combframe[ichip]['combind'][:,jfiber] = combextra

        # we have this extra

      #endfor    # extras loop

    #Endfor # chip loop

    # Plotting
    #----------
    pl = False
    if pl:
      xx = [np.arange(npix*2), np.arange(npix*2)+npix*2+300,
            np.arange(npix*2)+npix*4+2*300]
      yy = [combframe[0]['flux'][:,jfiber], combframe[1]['flux'][:,jfiber],
            combframe[2]['flux'][:,jfiber] ]

      #xr = [480,490]
      #yr = minmax(combspec[xr[0]*2:xr[1]*2])
      #xr = [0,npix*2]
      #yr = [min(combspec),max(combspec)]
      xr = minmax(xx)
      yr = minmax(yy)
 
      plt.plot([0],[0])
      #,/nodata,xr=xr,yr=yr,xs=1,ys=1,xtit='Pixel',ytit='Counts',$
      #     tit=strtrim(fiberobjid,2)+' (OBJTYPE='+fiberobjtype+')'
      for k in range(3):
          plt.plot(np.arange(npix*2)+npix*2*k+k*300,
                   combframe[k]['flux'][:,jfiber])
      #oplot,combspec
      #oplot,sqrt(combvar),co=250
      #oplot,xx,yy

      #wait,1
      #stop

    #endif

    #BOMB:

  #Endfor # fiber loop


  #-------------------------------------------------------------------------
  # Convert the WAVELENGTH solution coefficients to dither combined pixels
  #-------------------------------------------------------------------------
  # The equation is:
  # wave = P[1]*( SIN( (Y+P[0]+P[2])/P[3]/radeg ) + P[4]) + POLY(Y+P[0]+P[5],P[6:*])

  # Since we changing Y->2*Y we need to do:
  # P[0] -> P[0]*2
  # P[2] -> P[1]*2
  # P[3] -> P[3]*2
  # P[5] -> P[5]*2
  # P[6] -> P[6]  constant term
  # P[7] -> P[7]/2
  # P[8] -> P[8]/2^2
  # P[9] -> P[9]/2^3  and so on
  nwcoef = np.sum(combtags == 'WCOEF')
  if nwcoef > 0:

    # Loop through the chips
    for i in range(3):
      wcoef_old = combframe[i]['wcoef']
      wcoef_new = wcoef_old
      nwcoef = len(wcoef_new[0,:])
      npolycoef = nwcoef-6
      wcoef_new[:,0] *= 2
      wcoef_new[:,2] *= 2
      wcoef_new[:,3] *= 2
      wcoef_new[:,5] *= 2
      for j in range(npolycoef):
          wcoef_new[:,j+6] /= 2**j

      combframe[i]['wcoef'] = wcoef_new
    #endfor

  #endif # WCOEF is there

  #-------------------------------------------------------
  # Convert the LSF parameters to dither combined pixels
  #-------------------------------------------------------

  # Since we're changing Y->2*Y we need to do:
  #  polynomial coefficients change depending on the power
  # P[0] -> P[0]      constant term
  # P[1] -> P[1]/2    linear term
  # P[2] -> P[2]/2^2  quadratic term
  # P[3] -> P[3]/2^3  and so on

  # Only convert if the LSF parameters were created from non-dithered
  #  exposures and have binsize=1
  nlsfcoef = np.sum(combtags == 'LSFCOEF')
  if nlsfcoef > 0 and combframe[0]['lsfcoef'][0,0] == 1:

    # Loop through the chips
    for ichip in range(3):
        # Loop through the fibers        
        for jfiber in range(nfibers):
          lsfpar = combframe[ichip]['lsfcoef'][jfiber,:]
          npar = len(lsfpar)

          # Breaking up the parameters
          binsize = lsfpar[0]
          Xoffset = lsfpar[1]   # Additive Xoffset
          Horder = lsfpar[2]
          Porder = lsfpar[3:Horder+3]   # Horder+1 array
          nGHcoefs = total(Porder+1)

          # Getting the GH parameters that vary globally
          cpar = lsfpar[Horder+4:Horder+4+nGHcoefs-1]
          coefarr = np.zeros((Horder+1,np.max(Porder)+1),float)
          cstart = [0,np.cumsum(Porder+1)]  # extra one at the end
          # Coefarr might have extra zeros at the end, but it shouldn't
          #  make a difference.
          for k in range(Horder):
            coefarr[k,0:Porder[k]] = cpar[cstart[k]:cstart[k]+Porder[k]]

          lsfpar_new = lsfpar # initialize the new lsf parameter array
          lsfpar_new[0] *= 2  # update binsize
          lsfpar_new[1] *= 2  # update Xoffset

          # Update polynomial coefficients for factor of 2x change
          cpar_new = cpar
          coefarr_new = coefarr
          for k in np.arange(1,np.max(Porder)):
              coefarr_new[:,k] /= 2**k  # correct for 2 starting w linear term

          # Now we need to update the GH parameters themselves to
          #  account for the change in X.
          #  Need to multiply all polynomial coefficients by the
          #   appropriate factor
          # Just need to scale SIGMA
          coefarr_new[0,:] *= 2  # sigma -> sigma*2

          # Stuff back in
          for k in range(Horder):
              cpar_new[cstart[k]:cstart[k]+Porder[k]] = coefarr_new[k,0:Porder[k]]
          lsfpar_new[Horder+4:Horder+4+nGHcoefs-1] = cpar_new  # stuff it back in

          # Wing parameters
          if npar > (3+Horder+1+nGHcoefs):
            wpar = lsfpar[3+Horder+1+nGHcoefs:*]

            # Nwpar     number of W parameters
            # WPorder   the polynomial order for each
            # Wing coefficients
            wproftype = wpar[0]
            nWpar = wpar[1]
            wPorder = wpar[2:2+nWpar-1]
            nWcoefs = np.sum(wPorder+1)

            # Getting the Wing parameters that vary globally
            wcoef = wpar[nWpar+2:*]
            wcoefarr = np.zeros((nWpar,np.max(wPorder)+1),float)
            wcstart = [0,np.cumsum(wPorder+1)]  # extra one at the end
            # wcoefarr might have extra zeros at the end, but it shouldn't
            #  make a difference.
            for k in range(nWpar):
                wcoefarr[k,0:wPorder[k]] = wcoef[wcstart[k]:wcstart[k]+wPorder[k]]

            # Wing input parameters
            # 1st par - area under curve
            # 2nd par - center

            # Update polynomial coefficients
            wcoef_new = wcoef
            wcoefarr_new = wcoefarr
            for k in np.arange(1,np.max(wPorder)):
                wcoefarr_new[:,k] /= 2**k  # correct for 2 starting w linear term

            # Now we need to update the GH parameters themselves to
            #  account for the change in X.
            #  Need to multiply all polynomial coefficients by the
            #   appropriate factor
            # Just need to scale SIGMA
            wcoefarr_new[1,:] *= 2  # sigma -> sigma*2

            # Stuff it back in
            for k in range(nWpar):
              wcoef_new[wcstart[k]:wcstart[k]+wPorder[k]] = wcoefarr_new[k,0:wPorder[k]]
            wpar_new = wpar
            wpar_new[nWpar+2:] = wcoef_new
            lsfpar_new[3+Horder+1+nGHcoefs:] = wpar_new  # stuff it back in

          #endif # wing parameters

          # Stuff the new LSF parameters into the combframe structure
          combframe[ichip]['lsfcoef'][jfiber,:] = lsfpar_new

      #endfor # fiber loop
    #endfor  # chip loop

  #endif # LSFCOEF is there

  # Update the headers
  #----------------------
  leadstr = 'APDITHERCOMB: '
  maxlen = 72-strlen(leadstr)
  line = 'Dither combined frames '+combframe.chipa.filename1+' and '+combframe.chipa.filename2
  ncuts = ceil(strlen(line)/float(maxlen))
  for l=0,ncuts-1 do apaddpar,combframe,leadstr+strmid(line,l*maxlen,maxlen),/history


  # Add to structure of all dither combined frames
  #------------------------------------------------
  if p==0:
      allcombframes = combframe
  else:
      allcombframes = [allcombframes,combframe]

#ENDFOR  # pair loop


#---------------------------------------------------------------
# PART III - COMBINE all fully-sampled (dither combined) frames
#---------------------------------------------------------------
#COMBINE:

npix2 = len(allcombframes[0][0]['flux'][:,0])

## Initialize OUTFRAME structure. Need 2x as many pixels
for i in range(3):
  #chstr0 = allframes[0].(i)
  chstr0 = allcombframes[0].(i)
  tags = tag_names(chstr0)
  apgundef,chstr

  ## Make the new chip structure
  for j in range(len(tags)):
    arr = chstr0.(j)
    sz = size(arr)
    type = size(arr,/type)
    # Data arrays. These are [NPix,Nfibers]
    dum = where(stregex(['FLUX','ERR','MASK','WAVELENGTH','SKY','SKYERR','TELLURIC','TELLURICERR'],tags[j],/boolean) eq 1,ndata)
    if ndata gt 0 then arr=make_array(npix2,nfibers,type=type)
    # I think this is redundant now that we are starting with ALLCOMBFRAME
    # instead of ALLFRAMES

    # Skip FILENAMEs
    if strmid(tags[j],0,8) ne 'FILENAME' then begin
      if n_elements(chstr) eq 0 then begin
        chstr = CREATE_STRUCT(tags[j],arr)
      else:
        chstr = CREATE_STRUCT(chstr,tags[j],arr)
  #endfor # tag loop

  # Reset to start with blank header
  mkhdr,header,0
  chstr['header'] = header

  ## Add to the final OUTFRAME
  if i==0:
    outframe = CREATE_STRUCT('chip'+chiptag[i],chstr)
  else:
    outframe = CREATE_STRUCT(outframe,'chip'+chiptag[i],chstr)

#Endfor # chip loop

# Put information in the header
leadstr = 'APDITHERCOMB: '
maxlen = 72-strlen(leadstr)
apaddpar,outframe,leadstr+'Combining '+strtrim(npairs,2)+' dither pairs',/history
if keyword_set(globalwt) then wtmethod = 'Using global spectrum weighting' else $ # weight method
  wtmethod = 'Using pixel-by-pixel weighting'
apaddpar,outframe,leadstr+wtmethod,/history
if nodither==False:

  ## Add dither information to headers
  for i in range(npairs):
    apaddpar,outframe,leadstr+'Pair '+strtrim(i+1,2),/history
    apaddpar,outframe,leadstr+file_basename(allcombframes[i].(0).filename1),/history
    apaddpar,outframe,leadstr+'Shift='+strtrim(pairstr[i].shift[0],2)+' at row 0',/history
    apaddpar,outframe,leadstr+file_basename(allcombframes[i].(0).filename2),/history
    apaddpar,outframe,leadstr+'Shift='+strtrim(pairstr[i].shift[1],2)+' at row 0',/history
  ## Add frame names to header
  apgundef,framenames
  apaddpar,outframe,'NPAIRS',strtrim(npairs,2)
  apaddpar,outframe,'NCOMBINE',strtrim(npairs,2)*2
  for i=0,npairs-1 do PUSH,framenames,pairstr[i].framename
  nframes = len(framenames)
  #for i=0,nframes-1 do apaddpar,outframe,'FRAME'+strtrim(i+1,2),framenames[i]
  ii = 0
  for i in range(npairs):
      apaddpar,outframe,'SHIFT'+strtrim(ii+1,2),pairstr[i].shift[0]
      apaddpar,outframe,'FRAME'+strtrim(ii+1,2),pairstr[i].framenum[0]
      ii += 1
      apaddpar,outframe,'SHIFT'+strtrim(ii+1,2),pairstr[i].shift[1]
      apaddpar,outframe,'FRAME'+strtrim(ii+1,2),pairstr[i].framenum[1]
      ii += 1
else:
  ## No dither combining
  ## Add frame names to header
  apgundef,framenames
  for i=0,npairs-1 do PUSH,framenames,shiftstr[i].framenum
  nframes = n_elements(framenames)
  apaddpar,outframe,'NCOMBINE',nframes
  for i=0,nframes-1 do apaddpar,outframe,'FRAME'+strtrim(i+1,2),framenames[i]

## Update the exposure time
##  We sum the exposures so we should sum the exposure times
totexptime = 0.0
for i=0,npairs-1 do totexptime+=sxpar(allcombframes[i].(0).header,'EXPTIME')
apaddpar,outframe,'EXPTIME',totexptime,' Total visit exposure time per dither pos'

## Only one pair
combtags = tag_names(allcombframes[0].(0))
outtags = tag_names(outframe.(0))
if npairs==1:
  ## Loop through the tags
  for i in range(len(outtag)):
      ind = where(combtags eq outtags[i],nind)
      for k=0,2 do outframe.(k).(i) = combframe.(k).(ind)
  return


## Add UT-MID, JD-MID and EXPTIME to the header
##---------------------------------------------
## get JD-MID and EXPTIME for all exposures
jdmid = np.zeros(nallframes,float)
exptime = np.zeros(nallframes,float)
for i in range(nallframes):
    jdmid1 = allframes[i][0]['header']['JD-MID']
    exptime1 = allframes[i][0]['header']['EXPTIME']
    if njdmid==0:
        dateobs = allframes[i][0]['header']['DATE-OBS']
        jd = date2jd(dateobs)
        jdmid1 = jd + (0.5*exptime1)/24./3600.d0
    jdmid[i] = jdmid1-2400000.
    exptime[i] = exptime1
## calculate exptime-weighted mean JD-MID
comb_jdmid = np.sum(exptime*jdmid)/np.sum(float(exptime))+2400000.
comb_utmid = jd2date(comb_jdmid)
apaddpar,outframe,'UT-MID',comb_utmid,' Date at midpoint of visit'
apaddpar,outframe,'JD-MID',comb_jdmid,' JD at midpoint of visit'

## Use DATE-OBS of the first exposure for the combined frame
minind = first_el(minloc(jdmid))
apaddpar,outframe,'DATE-OBS',sxpar(allframes[minind].(0).header,'DATE-OBS')


print('Combining ',npairs,' dither pairs')

## Loop through the fibers
##------------------------
for jfiber in range(nfibers):
  if (jfiber+1) % 50 == 0:
      print(str(jfiber+1),'/',str(nfibers))

  ## Get object type
  ## The plugmap index for this fiber
  iplugind = where((plugmap['fiberdata']['spectrographId']==2) & 
                   (plugmap['fiberdata']['holetype'] == 'OBJECT') &
                   (plugmap['fiberdata']['fiberid']==300-jfiber))
  ## Getting information on the object
  if niplugind > 0:
    ## Fiber type from the plugmap structure
    fiberobjtype = plugmap['fiberdata'][iplugind]['objtype']
    fiberobjid = plugmap['fiberdata'][iplugind]['tmass_style']
  else:
    ## No information for this fiber
    print('No information for FiberID=',strtrim(300-jfiber,2),' in the plugmap file')
    fiberobjtype = ''
    fiberobjid = -1

  ## Measure the average wavelength zeropoint for all pairs
  ##--------------------------------------------------------
  ##  The accuracy of each wavelength solution is about the same,
  ##  and so it's probably best to give them all the same weight
  wpix0 = allcombframes[0]['wcoef'][jfiber,0]
  wpix0 = wpix0-wpix0[0]   # relative to the first one
  wpix0_offset = np.mean(wpix0)

  ## Loop through the chips
  ##-----------------------
  for ichip in range(3):

    ## Initialize the "data" structure for all spectra
      dt = [('flux',float,npix2),('err',float,npix2),('mask',int,npix2),
            ('wavelength',float,npix2),('sky',float,npix2),
            ('skyerr',float,npix2),('telluric',float,npix2),
            ('telluricerr',float,npix2),('scale',float,npix2)]
      #dumstr = {flux:fltarr(npix2),err:fltarr(npix2),mask:intarr(npix2),$
      #       wavelength:dblarr(npix2),sky:fltarr(npix2),skyerr:fltarr(npix2),$
      #       telluric:fltarr(npix2),telluricerr:fltarr(npix2),scale:fltarr(npix2)}
      #data = REPLICATE(dumstr,npairs)
      data = np.zeros(npairs,dtype=np.dtype(dt))
      
    ## Loop through the spectra/pairs
    ##-------------------------------
    for k in range(npairs):
        ## Load the data
        data[k]['flux'] = allcombframes[k][ichip]['flux'][:,jfiber]
        data[k]['err'] = allcombframes[k][ichip]['err'][:,jfiber]
        data[k]['mask'] = allcombframes[k](ichip]['mask'][:,jfiber]
        data[k]['wavelength'] = allcombframes[k][ichip]['wavelength'][:,jfiber]
        data[k]['sky'] = allcombframes[k][ichip]['sky'][:,jfiber]
        data[k]['skyerr'] = allcombframes[k][ichip]['skyerr'][:,jfiber]
        data[k]['telluric'] = allcombframes[k][ichip]['telluric'][:,jfiber]
        data[k]['telluricerr'] = allcombframes[k][ichip]['telluricerr'][:,jfiber]

       ## Need to normalize them if they are not sky spectra
       if (fiberobjtype != 'SKY') and noscale==False:

         ## Fit a low-order polynomial to the data
         ##  bad/missing values at the ends have flag=NAN
         if median:
            bd, = np.where(data[k]['mask'] & BADMASK)
            if len(bd)>0:
                data[k]['flux'][bd] = np.nan
	    scale = smooth(medfilt1d(data[k]['flux'],501,edge=2),100,/nan)
            if len(bd)>0:
                data[k]['flux'][bd]=scale[bd]
        else:
          gd, = np.where(np.isfinite(data[k]['flux']) &
                         (data[k]['sky'] < np.maximum(2*np.median(data[k]['flux']),2*np.median(data[k]['sky']))) & 
                         ((data[k]['mask'] & BADMASK) == 0))   # only want good values
          if ngd > 100:
              if keyword_set(nointerp) then y2=findgen(npix) else y2 = findgen(npix2)/2
              scalecoef = AP_ROBUST_POLY_FIT(y2[gd],data[k].flux[gd],5,status=status)
              scale = POLY(y2,scalecoef)
          else:
              scale = np.ones(npix2,float)
        #endelse
        #if status eq 0 then stop
      else:
          scale = np.ones(npix2,float)


      data[k]['flux'] = data[k]['flux']/scale      # normalize spectrum
      data[k]['err'] = data[k]['err']/scale        # normalize error
      data[k]['scale'] = scale

    #Endfor # pairs/spectra loop
    scales = data['scale']
    #scales = reform(data[*,*,ncol])
    sumscales = np.sum(scales,axis=1)

    ## Now combine the spectra and errors
    ##-------------------------------------
    dataspec = data['flux']
    dataerr = data['err']
    datamask = data['mask']

    ## Combine the errors for the mean
    ##  this will be redone below
    comberr = np.sqrt( np.sum(dataerr**2,axis=1)/npairs**2 )    # prop. of errors for mean

    nbdpix = 0
    bdpix = mp.where(datamask and badmask(),nbdpix)

    ## Do outlier rejection: skip, can fail in case of inhomogeneous S/N, etc.
#    medspec = MEDIAN(dataspec,dim=2)                        # median spectrum
#    diffspec = dataspec - medspec#replicate(1.0,npairs)     # difference spectrum
#    bdpix = where(abs(diffspec/dataerr) gt 5 OR $
#                  data.mask and badmask(),nbdpix)  # 5 sigma outliers
    maskspec = dataspec
    maskerr = dataerr
    #if nbdpix gt 0 then maskspec[bdpix] = !values.f_nan     # set bad pix to NAN
    #if nbdpix gt 0 then maskerr[bdpix] = !values.f_nan      # set bad pix to NAN
    #if nbdpix gt 0 then maskerr[bdpix] *= 10. 
    # Number of "good" points for each pixel
    #masknpairs = replicate(1L,npix2,npairs)
    #if nbdpix gt 0 then masknpairs[bdpix] = 0
    #npairs2d = TOTAL(masknpairs,2)               # number of spectra/pairs for each pixel

    ## Take a weighted mean of the spectrum
    ##  wt = 1/error^2
    ##  weighted mean = Sum(wt*X)/Sum(wt)

    ## Global weight per spectrum
    ##---------------------------
    if globalwt:
      ## Create the error array using same values for all
      ##   pixels of a spectrum
      mederr = np.median(dataerr,axis=1)              # median err per spectrum
      meddataerr = replicate(1.0,npix2)#mederr   # make 2d array
      ## mask bad pixels
      if nbdpix>0:
          meddataerr[bdpix] = !values.f_nan

      # Now do a weighted mean (global) while ignoring the outliers (NANs)
      combspec = np.nansum(maskspec/meddataerr**2,axis=1) / np.nansum(1.0/meddataerr**2,axis=1)
      #comberr = sqrt( TOTAL(maskerr^2,2,/NAN)/npairs2d^2 )  # ignore NANs
      #comberr = sqrt( 1./TOTAL(1./maskerr^2,1,/NAN) )   # ignore NANs
      comberr = np.sqrt( 1./np.nansum(1./meddataerr**2,axis=1) )   # ignore NANs

    ## Pixel-by-pixel weighting
    ##-------------------------
    else:

      ## Now do a weighted mean (pixel-by-pixel) while ignoring the outliers (NANs)
      combspec = np.nansum(maskspec/maskerr**2,axis=1) / np.nansum(1.0/maskerr**2,axis=1)
      ## Combine the errors for mean and ignore outlier points
      #comberr = sqrt( TOTAL(maskerr^2,2,/NAN)/npairs2d^2 )  # ignore NANs
      comberr = np.sqrt( 1./np.nansum(1./maskerr**2,axis=1) )   # ignore NANs

    ## Initialize combined mask
    combmask = fix(combspec*0)

    ## Set bad pixels to zero
    bdpix, = np.where(np.isfinite(combspec)==False))
    if len(bdpix)>0:
      combspec[bdpix] = 0.0
      comberr[bdpix] = BADERR
      combmask[bdpix] = combmask[bdpix] or maskval('BADPIX')

    ## Rescale spec/error using the SUM of the continua
    combspec = combspec * sumscales
    comberr = comberr * sumscales

    ## Wavelength array
    ##  They are going to be slightly different because the measured dither
    ##  shift and wavelength zeropoint offset are done independently.
    ##  Use a mean of the wavelength zeropoint offset all other wavelength
    ##  coefficients should be the same (since they used the same apWave file).
    ## The offset is relative to the wavelength zeropoint of the first pair
    wcoef = reform(allcombframes[0][ichip]['wcoef'][jfiber,:])
    wcoef[0] = wcoef[0] + wpix0_offset   # relative to the first
    combwave = pix2wave(np.arange(npix2),wcoef)

    ## Average the wavelengths across all of the pairs
    if avgwave:
        combwave = np.zeros(npix2,float)
        for p in range(npairs):
            combwave += allcombframes[p][ichip]['wavelength'][:,jfiber]
        combwave /= npairs
        ## Fit polynomial model to the wavelengths
        newy = np.arange(npix2)
        ## poly2wave.pro scales the x values using (x+xoffset)/3000.0
        wcoef = allcombframes[0][ichip]['wcoef'][jfiber,:]
        newcoef = poly_fit((newy+wcoef[0])/3000.,combwave,3)
        wcoef[6:9] = newcoef

    ## flags, sky, skyerr, telluric, telluric_err
    ##combflags = TOTAL(data.mask,2)            # flags????
    for k in range(npairs):
        combmask = combmask | data[k]['mask']
    combsky = np.sum(data.sky,axis=1)               # sum sky
    combskyerr = np.sum(data.skyerr,axis=1)         # sum sky error
    combtel = np.sum(data.telluric,axis=1)/npairs   # average telluric
    combtelerr = np.sum(data.telluricerr,axis=1)    # sum telluric error

    ## Now stuff it in the output structure
    ##-------------------------------------
    outframe[ichip]['flux'][:,jfiber] = combspec
    outframe[ichip]['wavelength'][:,jfiber] = combwave
    outframe[ichip]['err'][:,jfiber] = comberr
    outframe[ichip]['mask'][:,jfiber] = combmask  # combflags
    outframe[ichip]['sky'][:,jfiber] = combsky
    outframe[ichip]['skyerr'][:,jfiber] = combskyerr
    outframe[ichip]['telluric'][:,jfiber] = combtel
    outframe[ichip]['telluricerr'][:,jfiber] = combtelerr

    ## Update the wavelength coefficients
    outframe[ichip]['wcoef'][jfiber,:] = wcoef

    ## DO I NEED TO MAKE A NEW LSF ARRAY???????
    ## They are all using the SAME LSF calibration frame and it's
    ## only going to be slightly shifted.  So I think we can just use
    ## the LSF coefficients of the first frame

  #Endfor # chip loop

  # Plotting
  #----------
  #pl = 1 # 0 #1
  if pl:
      xx = [np.arange(npix2), np.arange(npix2)+npix2+300,
            np.arange(npix2)+2*npix2+2*300]
      yy = [outframe[0]['flux'][:,jfiber],
            outframe[1]['flux'][:,jfiber],
            outframe[2]['flux'][:,jfiber]]

      #xr = [480,490]
      #yr = minmax(combspec[xr[0]*2:xr[1]*2])
      #xr = [0,npix*2]
      #yr = [min(combspec),max(combspec)]
      xr = minmax(xx)
      #yr = minmax(yy)
      yr = [0.0, np.median(yy)*2] 

      plt.plot([0],[0],/nodata,xr=xr,yr=yr,xs=1,ys=1,xtit='Pixel',ytit='Counts',
               tit=strtrim(fiberobjid,2)+' (OBJTYPE='+fiberobjtype+')')
      for k in range(3):
          plt.plot(np.arange((npix2)+npix2*k+k*300,
                             outframe[k]['flux'][:,jfiber])
      #oplot,combspec
      #oplot,sqrt(combvar),co=250
      #oplot,xx,yy

  #endif

 # BOMB2:

#Endfor # fiber loop

    return outframe


def dithercombine():
    pass

def fluxing():
    pass

def visitoutput():
    pass


def calibrate_exposure(framenum,plantab,plugmap,logger=None,
                       clobber=False,verbose=False,nowrite=False):
    """
    This corrects a single APOGEE exposure/frame.
    It measures the dither shift, subtracts the sky and
    corrects for the telluric absorption.
    Finally, it writes the apCframe files.

    Parameters
    ----------
    framenum : int
       Expsore number.
    plantab : table
       Plan table with information about this visit.
    plugmap : table
       Plugmap information on the observed objects.
    logger : logging object
       Logging object.
    clobber : bool, optional
       Overwrite any existing files.  Default is False.
    verbose : bool, optional
       Verbose output to the screen.  Default is False.
    nowrite : bool, optional
       Don't outpout any apCframe files to disk.  Default is False.

    Returns
    -------
    frame_telluric : 
       Calibrated frame
    Unless nowrite=True is set, apCframe files are written to disk.

    Examples
    --------

    frame = calibrate_exposure(framenum,plantab,plugmap,logger)

    """

    t1 = time.time()

    load = apload.ApLoad(apred=plantab['redux'],telescope=plantab['telescope'])
    
    #------------------------------------------
    # Correcting and Calibrating the ap1D files
    #------------------------------------------
    cfiles = load.filename('Cframe',chip=chiptag,num=framenum,plate=plantab['plateid'],
                           mjd=plantab['mjd'],field=plantab['field'])
    ctest = np.array([os.path.exists(f) for f in cfiles])
    if clobber==False and np.sum(ctest)==3:
        print('Already done')
        return None

    logger.info(' 1d processing '+os.path.basename(files[0])+'{:8.2f}'.format(time.time()-t1))
            
    # Load the 1D files
    #--------------------
    frame0 = load.frame(cfiles)
    #APLOADFRAME,files,frame0,/exthead      # loading frame 1
    apaddpar,frame0,'LONGSTRN','OGIP 1.0'  # allows us to use long/continued strings
    
    # Fix INF and NAN
    for k in range(3):
        bdnan, = np.where(~np.isfinite(frame0[k]['flux']) |
                          ~np.isfinite(frame0[k]['err']))
        if len(bdnan)>0:
            frame0[k]['flux'][bdnan] = 0.0
            frame0[k]['err'][bdnan] = BADERR
            frame0[k]['mask'][bdnan] = 1   # bad
        # Fix ERR=0
        bdzero, = np.where(frame0[k]['err'] < 0)
        if len(bdzero)>0:
            frame0[k]['flux'][bdzero] = 0.0
            frame0[k]['err'][bdzero] = BADERR
            frame0[k]['mask'][bdzero] = 1

    # Add Wavelength and LSF information to the frame structure
    #---------------------------------------------------------
    # Loop through the chips
    for k in range(3):
        chtab = frame0[k]
        # Get the LSF calibration data
        lsfcoef,lhead = fits.getdata(lsffiles[k],header=True)
        if newwave is not None:
            del chtab['wcoef']
            del chtab['wavelength']
        # Add to the chip structure
        # Wavelength calibration data already added by ap2dproc with ap1dwavecal
        #if tag_exist(frame0.(0),'WCOEF') and not keyword_set(newwave) then begin
        if 'wcoef' in chtab.colnames: and newwave is None:
            if verbose:
                print('using WCOEF from 1D...')
            chtab.update({'lsffile':lsffiles[k],'lsfcoef':lsfcoef,'wave_dir':plate_dir,
                          'wavefile':wavefiles[k]})
        # Need wavelength information
        else:
            wcoef,whead = fits.getdata(wavefiles[k],header=True)
            chtab.update({'lsffile':lsffiles[k],'lsfcoef':lsfcoef,'wave_dir':plate_dir,
                          'wavefile':wavefiles[k],'wcoef':wcoef})
            # Now add this to the final FRAME structure
            frame['chip'+chiptag[k]] = chtab
    
    del frame0   # free up memory

    #----------------------------------
    # STEP 1:  Measure dither Shift
    #----------------------------------
    if verbose:
        print('STEP 1: Measuring the DITHER SHIFT with APDITHERSHIFT')
    # Not first frame, measure shift relative to 1st frame
    dither_commanded = frame[0]['header']['DITHPIX']
    if verbose:
        print('dither_commanded: ',dither_commanded)
    if j == 0 and verbose:
        print('ref_dither_commanded: ',ref_dither_commanded)
    print('nodither: ', nodither)
    if j > 0:
        if dither_commanded != 0 and np.abs(dither_commanded-ref_dither_commanded) > 0.002:
            nodither = False
    # Measure dither shift
    if (j > 0) and not nodither:
        ashift = [0.0,0.0]
        ashifterr = 0.0
        #APDITHERSHIFT,ref_frame,frame,ashift,ashifterr
        if 'platetype' in plantab.colnames:
            if plantab['platetype'] == 'sky' or plantab['platetype'] == 'cal':
                plot = True
                pfile = os.path.join(plate_dir,'plots','dithershift-'+framenum)
            else:
                #pfile=0 & plot=0
                plot = True
                pfile = os.path.join(plate_dir,'plots','dithershift-'+framenum)
        if plantab['platetype']=='single':
            nofit = True
        else:
            nofit = False
        shiftout = dithershift(ref_frame,frame,xcorr=True,pfile=pfile,plot=plot,
                               plugmap=plugmap,nofit=nofit,mjd=plantab['mjd'])
        shift = shiftout['shiftfit']
        shifterr = shiftout['shifterr']
        if verbose:
            print('Measured dither shift: ',ashift,shift)
    # First frame, reference frame
    else:
        # measure shift anyway
        if j > 0:
            shiftout = dithershift(ref_frame,frame,xcorr=True,pfile=pfile,plot=plot,
                                   plugmap=plugmap,nofit=nofit,mjd=plantab['mjd'])
            if verbose:
                print('Measured dither shift: ',shiftout['shiftfit'])
        # note reference frame wants to include sky and telluric!
        ref_frame = frame
        shift = [0.0,0.0]
        ashift = [0.0,0.0]
        shifterr = 0.0
        ashifterr = 0.0
        if dither_commanded != 0:
            ref_dither_commanded = dither_commanded
        print('Shift = 0.0')
        shiftout = {'type':'xcorr','shiftfit':np.zeros(2,float),'shfiterr':shifterr,
                    'chipshift':np.zeros((3,2),float),'chipfit':np.zeros(4,float)}
    apaddpar,frame,'APDITHERSHIFT: Measuring the dither shift',/history
    if shift[0] == 0.0: apaddpar,frame,'APDITHERSHIFT: This is the REFERENCE FRAME',/history
    apaddpar,frame,'DITHSH',shift[0],' Measured dither shift (pixels)'
    apaddpar,frame,'DITHSLOP',shift[1],' Measured dither shift slope (pixels/fiber)'
    apaddpar,frame,'EDITHSH',shifterr,' Dither shift error (pixels)'
    #apaddpar,frame,'ADITHSH',ashift,' Measured dither shift (pixels)'
    #apaddpar,frame,'AEDITHSH',ashifterr,' Dither shift error (pixels)'
    ADD_TAG,frame,'SHIFT',shiftout,frame_shift
        
    logger.info('  dithershift '+'{:8.2f}{:8.2f}'.format(time.time()-t1,time.time()-t0))
    if 'platetype'] in plantab.colnames:
        if plantab['platetype'] not in ['normal','single','twilight']:
            continue

    #----------------------------------
    # STEP 2:  Wavelength Calibrate
    #----------------------------------
    # THIS IS NOW DONE AS PART OF AP2DPROC, USING PYTHON ROUTINES
    if ap1dwavecal:
        if verbose:
            print('STEP 2: Wavelength Calibrating with AP1DWAVECAL')
        plotfile = plate_dir+'/plots/pixshift_chip-'+framenum
        if dithonly:
            ap1dwavecal_refit(frame,frame_wave,plugmap=plugmap,verbose=True,plot=True,pfile=plotfile)
        plotfile = plate_dir+'/plots/pixshift-'+framenum
        if plantab['platetype'] == 'twilight':
            ap1dwavecal(frame_shift,frame_wave,verbose=True,plot=True,pfile=plotfile)
        else:
            ap1dwavecal(frame_shift,frame_wave,plugmap=plugmap,verbose=True,plot=True,pfile=plotfile)
        del frame  # free up memory
        logger.info('  wavecal '+'{:8.2f}{:8.2f}'.format(time.time()-t1,time.time()-t0))
    else:
        frame_wave = frame_shift

    #----------------------------------
    # STEP 3:  Airglow Subtraction
    #----------------------------------
    if verbose:
        print('STEP 3: Airglow Subtraction with APSKYSUB')
    sky.skysub(frame_wave,plugmap,frame_skysub,subopt=1,error=skyerror,force=force)
    #if n_elements(skyerror) gt 0 and planstr.platetype ne 'twilight' then begin
    #  stop,'halt: APSKYSUB Error: ',skyerror
    #  apgundef,frame_wave,frame_skysub,skyerror
    #endif
    del frame_wave  # free up memory
    logger.info('  airglow '+'{:8.2f}{:8.2f}'.format(time.time()-t1,time.time()-t0))
    if 'platetype' in plantab.colnames:
        if plantab['platetype'] not in ['normal','single','twilight']:
            continue

    #----------------------------------
    # STEP 4:  Telluric Correction
    #----------------------------------
    if verbose:
        print('STEP 4: Telluric Correction with APTELLURIC')
    if plantab['platetype'] == 'single':
        starfit = 2
        single = 1
        pltelstarfit = 1
    elif plantab['platetype'] == 'twilight':
        starfit = 0
    else:
        starfit = 1
        single = 0
        pltelstarfit = 0
        visittab = None
    if 'pltelstarfit'] in plantab.colnames:
        pltelstarfit = plantab['pltelstarfit']
    if 'usetelstarfit'] in plantab.colnames:
        usetelstarfit = 1
    else:
        usetelstarfit = 0
    if 'maxtellstars' in plantab.colnames:
        maxtellstars = plantab['maxtellstars']
    else:
        maxtellstars = 0
    if 'tellzones' in plantab.colnames:
        tellzones = plantab['tellzones']
    else:
        tellzones = 0
    sky.telluric(frame_skysub,plugmap,frame_telluric,tellstar,starfit=starfit,
                 single=single,pltelstarfit=pltelstarfit,usetelstarfit=usetelstarfit,
                 maxtellstars=maxtellstars,tellzones=tellzones,specfitopt=1,
                 plots_dir=plots_dir,error=telerror,save=True,preconv=True,visittab=visittab,
                 test=test,force=force)
    tellstar['im'] = plantab['apexp'][j]['name']
    ADD_TAG,frame_telluric,'TELLSTAR',tellstar,frame_telluric
    if len(alltellstar)==0:
        alltellstar = tellstar
    else:
        alltellstar.append(tellstar)
    #if n_elements(telerror) gt 0 and planstr.platetype ne 'single' and not keyword_set(force) then begin
    #  print('not halted: APTELLURIC Error: ',telerror)
    #  ntellerror+=1
    #  apgundef,frame_skysub,frame_telluric,telerror
    #  goto, BOMB1
    del frame_skysub  # free up memory
    logger.info(logfile,'  telluric '+'{:8.2f}{:8.2f}'.format(time.time()-t1,time.time()-t0))

    #-----------------------
    # Output apCframe files
    #-----------------------
    if nowrite==False:
        if verbose:
            print('Writing output apCframe files')
        outfiles = load.filename('Cframe',chip=chiptag,num=framenum,
                                 plate=plantab['plateid'],mjd=plantab['mjd'],
                                 field=plantab['field'])
        outcframe(frame_telluric,plugmap,outfiles,verbose=False)

    return frame_telluric


def ap1dvisit(planfiles,clobber=False,verbose=False,newwave=None,
              test=None,mapper_data=None,halt=None,dithonly=None,
              ap1dwavecal=None,force=False):
    """
    This program processes 1D APOGEE spectra.  It does dither
    combination, wavelength calibration and sky correction.

    Parameters
    ----------
    planfiles : str or list
       Input list of plate plan files
    clobber : bool, optional
       Don't use the apCframe files previously created (if it exists)
    verbose : bool, optional
       Print a lot of information to the screen

    Returns
    -------
    1D dither combined, wavelength calibrated, and sky corrected
    spectra.  4Kx300x5 with planes:
         1. flux (sky subtracted, absorption/flux corrected)
         2. (wavelength, if needed)
         3. flux error
         4. sky estimate shifted to wavelength sampling of object
         5. telluric/flux response shifted to wavelength sampling of object
         6. flags
    The names are apSpec-[abc]-PLATE4-MJD5.fits

    Examples
    --------

    ap1dvisit(planfiles)

    Written by D.Nidever  Mar. 2010
    Modifications J. Holtzman 2011+
    Translated to Python by D. Nidever  2024
    """

    #common telluric,convolved_telluric

    if ap1dwavecal is not None:
        newwave = True

    t0 = time.time()
    nplanfiles = len(np.atleast_1d(planfiles))

    print('')
    print('RUNNING AP1DVISIT')
    print('')
    print(len(nplanfiles),' PLAN files')
    
    chiptag = ['a','b','c']

    #--------------------------------------------
    # Loop through the unique PLATE Observations
    #--------------------------------------------
    for i in range(nplanfiles):
        planfile = planfiles[i]
        print('')
        print('=========================================================================')
        print('{:d}/{:d}  Processing Plan file {:}'.format(i+1,nplanfiles,planfile))
        print('=========================================================================')

        # Load the plan file
        #--------------------
        print('')
        print('Plan file information:')
        plantab = aploadplan(planfile,verbose=False)
        if plantab['mjd'] >= 59556:
            fps = True
        else:
            fps = False
        if 'field' not in plantab.colnames:
            plantab['field'] = '     '
            plantab['field'] = apogee_field(0,plantab['plateid'])

	# Get APOGEE directories
        load = apload.ApLoad(apred=plantab['redux'],telescope=plantab['telescope'])
        #dirs = getdir(apogee_dir,cal_dir,spectro_dir,apred_vers=apred_vers,datadir=datadir)
        #logfile = apogee_filename('Diag',plate=plantab['plateid'],mjd=plantab['mjd'])
        logfile = load.filename('Diag',plate=plantab['plateid'],mjd=plantab['mjd'])

        # Only process "normal" plates
        if 'platetype' in plantab.colnames:
            normal_plate_types = ['normal','twilight','sky','single','cal']
            if plantabe['platetype'] not in normal_plate_types:
                continue

        if 'survey' in plantab.colnames:
            survey = plantab['survey']
        else:
            if plantab['plateid'] >= 15000:
                survey = 'mwm'
            else:
                survey = 'apogee'
            if plantab['plateid'] == 0:  # cals
                survey = 'mwm'

        # Load the Plug Plate Map file
        #------------------------------
        print('')
        print('Plug Map file information:')
        if 'force' in plantab.colnames and force is None:
            force = plantab['force']
        if 'fixfiberid'] in plantab.colnames:
            fixfiberid = plantab['fixfiberid']
        if type(fixfiberid) is str and np.atleast_1d(fixfiberid).size==1:
            if fixfiberid.strip()=='null' or fixfiberid.strip()=='none':
                fixfiberid = None
        if 'badfiberid' in plantab.colnames:
            badfiberid = plantab['badfiberid']
        if type(badfiberid) is str and np.atleast_1d(badfiberid).size==1:
            if badfiberid.strip()=='null' or badfiberid.strip()=='none':
                badfiberid = None

        # Check for existing plate data file
        if plantab['platetype']=='single':
            plugfile = getenv('APOGEEREDUCE_DIR')+'/data/plPlugMapA-0001.par'
            plugmap = platedata.getdata(plantab['plateid'],plantab['mjd'],
                                        obj1m=plantab['apexp'][0]['singlename'],
                                        starfiber=plantab['apexp'][0]['single'],
                                        fixfiberid=fixfiberid)
        elif plantab['platetype']=='twilight':
            plugmap = platedata.getdata(plantab['plateid'],plantab['mjd'],twilight=True)
        elif plantab['platetype']=='cal':
            print('no plugmap for cal frames')
        else:
            plugfile = plantab['plugmap']
            plugmap = platedata.getdata(plantab['plateid'],plantab['mjd'],
                                        plugid=plantab['plugmap'],
                                        fixfiberid=fixfiberid,badfiberid=badfiberid,
                                        mapper_data=mapper_data)

        #if n_elements(plugerror) gt 0 then goto,BOMB

        if plantab['platetype']=='cal':
            plugmap['mjd'] = plantab['mjd']   # enter MJD from the plan file

        # Get objects
        if plantab['platetype']=='single':
            obj, = np.where((plugmap['fiberdata']['objtype'] != 'SKY') &
                            (plugmap['fiberdata']['spectrographId']==2))
        else:
            obj, = np.where((plugmap['fiberdata']['objtype'] != 'SKY') &
                            (plugmap['fiberdata']['spectrographId']==2) &
                            (plugmap['fiberdata']['mag'][1] > 7.5))

        # Check if the calibration files exist
        #-------------------------------------
        makecal.makecal(lsf=plantab['lsfid'],full=True)
        wavefiles = load.filename('Wave',chip=chiptag,num=plantab['waveid'])
        # We are now using dailywave files with MJD names
        if plantab['waveid'] < 1e7:
            wavefiles = os.path.dirname(wavefiles[0])
            wavefiles += load.prefix+'Wave-'+chiptag+'-'+str(plantab['waveid'])+'.fits'
            wavetest = np.array([os.path.exists(f) for f in wavefiles])
            lsffiles = load.filename('LSF',chip=chiptag,num=plantab['lsfid'])
            lsftest = np.array([os.path.exists(f) for f in lsffiles])
            if np.sum(wavetest) == 0 or np.sum(lsftest) == 0:
                bd1, = np.where(wavetest==0)
                if len(bd1) > 0:
                    print(wavefiles[bd1],' NOT FOUND')
                bd2 = where(lsftest==0)
                if len(bd2) > 0:
                    print(lsffiles[bd2],' NOT FOUND')
                continue

        # Do the output directories exist?
        plate_dir = load.filename('Plate',mjd=plantab['mjd'],
                                  plate=plantab['plateid'],chip='a',
                                  field=plantab['field'],dir=True)
        if os.path.exists(plate_dir)==False:
            os.makedirs(plate_dir)
        sdir = plate_dir.split('/')
        if load.telescope != 'apo1m' and plantab['platetype'] != 'cal':
            if os.path.exists(spectro_dir+'/plates/'+sdir[-2]):
                srcfile = '../'+sdir[-5]+'/'+sdir[-4]+'/'+sdir[-3]+'/'+sdir[-2]
                destfile = spectro_dir+'/plates/'+sdir[-2]
                os.link(srcfile,destfile)
        #cloc=strtrim(string(format='(i)',plugmap.locationid),2)
        #file_mkdir,spectro_dir+'/location_id/'+dirs.telescope
        #if ~file_test(spectro_dir+'/location_id/'+dirs.telescope+'/'+cloc) then file_link,'../../'+s[-5]+'/'\
        #    +s[-4]+'/'+s[-3],spectro_dir+'/location_id/'+dirs.telescope+'/'+cloc


        # Are there enough files
        nframes = len(plantab['apexp'])
        #if nframes lt 2 then begin
        #  print,'Need 2 OBSERVATIONS/DITHERS to Process'
        #  goto,BOMB
        #endif

        # Start the plots directory
        plots_dir = plate_dir+'/plots/'
        if os.path.exists(plots_dir)==False:
            os.makedirs(plots_dir)
        
        visittab = []
        alltellstar = []
        allframes = []

        # Do we already have apPlate file?
        filename = load.filename('Plate',chip='c',mjd=plantab['mjd'],
                                 plate=plantab['plateid'],field=plantab['field'])
        if os.path.exists(filename) and clobber==False:
            print('File already exists: ', filename)
            #goto,dorv
        else:
            print('cannot find file: ', filename)

        # Process each frame
        #-------------------
        dt = [('index',int),('framenum',int),('shift',float),('shifterr',float),('shiftfit',float,2),
              ('chipshift',float,(3,2)),('chipfit',float,4),('pixshift',float),('sn',float)]
        shifttab = np.zeros(nframes,dtype=np.dtype(dt))
        shifttab['index'] = -1
        shifttab['shift'] = np.nan
        shifttab['shifterr'] = np.nan
        shifttab['sn'] = -1.0

        # Assume no dithering until we see that a dither has been commanded from the
        #   header cards
        nodither = 1
        ntellerror = 0

        # Loop over the exposures/frames
        #-------------------------------
        for j in range(frames):
            t1 = time.time()

            # for ASDAF plate, fix up the plugmap structure
            if plantab['platetype'] == 'asdaf':
                allind, = np.where(plugmap['fiberdata']['spectrographId']==2)
                plugmap['fiberdata'][allind]['objtype'] = 'SKY'
                star, = np.where((plugmap['fiberdata']['spectrographId']==2) &
                                 (plugmap['fiberdata']['fiberid']==plantab['apexp'][j]['single']))
                plugmap['fiberdata'][star]['objtype'] = 'STAR'
                plugmap['fiberdata'][star]['tmass_style'] = plantab['apexp'][j]['singlename']
                plugmap['fiberdata'][star]['mag'] = np.zeros(5,float)+99.
                if 'hmag' in plantab.colnames:
                    plugmap['fiberdata'][star]['mag'][1] = plantab['hmag']
                else:
                    plugmap['fiberdata'][star]['mag[']1] = 5

            # Make the filenames and check the files
            rawfiles = load.filename('R',chip=chiptag,num=plantab['apexp'][j]['name'])
            rawinfo = info.info(rawfiles,verbose=False)    # this returns useful info even if the files don't exist
            framenum = rawinfo[0]['fid8']   # the frame number
            files = load.filename('1D',chip=chiptag,num=framenum)
            einfo = info.info(files,verbose=False)
            okay = (einfo['exists'] & einfo['sp1dfmt'] & einfo['allchips'] &
                    (einfo['mjd5'] == plantab['mjd']) & ((einfo['naxis']==3) | (einfo['exten']==True)))
            if np.sum(okay) == 0:
                bd, = np.where(np.array(okay)==False)
                raise Exception('halt: There is a problem with files: '+','.join(files)[bd])

            print('')
            print('-----------------------------------------')
            print('{:d}/{:d}  Processing Frame Number >>{:}<<'.format(j+1,nframes,framenum))
            print('-----------------------------------------')

            dum = calibrate_exposure(framenum,plantab,plugmap,logger,
                                     clobber=clobber,verbose=verbose)

            #---------------------------------------------
            # Using the apCframe files previously created
            #---------------------------------------------
            
            # Make the filenames and check the files
            # Cframe files
            cfiles = load.filename('Cframe',chip=chiptag,num=framenum,plate=plantab['plateid'],
                                   mjd=plantab['mjd'],field=plantab['field'])
            cinfo = info.info(cfiles,verbose=False)
            okay = (cinfo['exists'] & cinfo['allchips'] & (cinfo['mjd5']==plantab['mjd')) &
                    ((cinfo['naxis']==3) | (cinfo['exten']==1)))
            if np.sum(okay) < 1:
                bd, = np.where(np.array(okay)==False)
                raise Exceptioon('halt: There is a problem with files: '+','.join(cfiles[bd]))

            print('Using apCframe files: '+cfiles)

            # Load the apCframe file
            frame_telluric = load.frame(cfiles)
            #APLOADCFRAME,cfiles,frame_telluric,/exthead

            # Get the dither shift information from the header
            if j == 0:
                ref_dither_commanded = frame_telluric[0]['header']['DITHPIX']
                if ref_frame is None:
                    ref_frame = frame_telluric
            else:
                dither_commanded = frame_telluric[0]['header']['DITHPIX']
                if dither_commanded != 0 and abs(dither_commanded-ref_dither_commanded) > 0.002:
                    nodither = False
            shift = frame_telluric[0]['header']['DITHSH']
            shifterr = frame_telluric[0]['header']['EDITHSH']
            pixshift = frame_telluric[0]['header']['MEDWSH']

            # Add to the ALLFRAMES structure
            #--------------------------------
            if len(allframes) == 0:
                allframes = frame_telluric
            else:
                allframes.append(frame_telluric)

            # Update SHIFTTAB
            shifttab[j]['index'] = j
            shifttab[j]['framenum'] = framenum
            shifttab[j]['shift'] = shift
            shifttab[j]['shifterr'] = shifterr
            shifttab[j]['pixshift'] = pixshift
            shifttab[j]['shiftfit'] = frame_telluric['shift']['shiftfit']
            shifttab[j]['chipshift'] = frame_telluric['shift']['chipshift']
            shifttab[j]['chipfit'] = frame_telluric['shift']['chipfit']
            # Get S/N of brightest non-saturated object, just for sorting by S/N
            if plantab['platetype']=='single':
                obj, = np.where((plugmap['fiberdata']['objtype'] != 'SKY') &
                                (plugmap['fiberdata']['spectrographId'] == 2))
            else:
                obj, = np.where((plugmap['fiberdata']['objtype'] != 'SKY') &
                                (plugmap['fiberdata']['spectrographid'] == 2) &
                                (plugmap['fiberdata']['mag'][1] > 7.5) &
                                (plugmap['fiberdata']['fiberid'] != 195))
            hmag = plugmap['fiberdata'][obj]['mag'][1]
            isort = np.argsort(hmag)
            ibright = obj[isort[0]]
            fbright = 300-plugmap['fiberdata'][ibright]['fiberid']
            shifttab[j]['sn'] = np.median(frame_telluric[1]['flux'][:,fbright]/frame_telluric[1].['err'][:,fbright])

            # Instead of using S/N of brightest object, which is subject to any issue with that object,
            #  use median frame zeropoint instead
            if plantab['platetype'] != 'single':
                fiberloc = 300-plugmap['fiberdata'][obj]['fiberid']
                zero = np.median([hmag+2.5*np.log10(np.median(frame_telluric[1]['flux'][:,fiberloc],axis=0))])
                shifttab[j]['sn'] = zero
            del frame_telluric  # free up memory

        if dithonly:
            return

    # Write summary telluric file
    if plantab['platetype'] == 'single':
        tellstarfile = plantab['plate_dir']+'/apTellstar-'+str(plantab['mjd'])+'-'+str(plantab['name'])+'.fits'
        fits.writeto(tellstatfile,alltellstar,overwrite=True)
    elif plantab['platetype'] == 'normal':
        tellstarfile = load.filename('Tellstar',plate=plantab['plateid'],
                                     mjd=plantab['mjd'],field=plantab['field'])
        fits.writeto(tellstarfile,alltellstar,overwrite=True)
    t1 = time.time()

    if 'platetype' in plantab.colnames:
        if plantab['platetype'] in ['normal','single','twilight']:
            continue

    # Remove frames that had problems from SHIFTTAB
    if nodither:
        minframes = 1
    else:
        minframes = 2
    if nodither == 0 or nframes > 1:
        bdframe, = np.where(shifttab['index'] == -1)
        if (nframes-len(bdframe)) < minframes:
            raise Exception('halt: Error: dont have two good frames to proceed')
        if len(bdframe)>0:
            shiftab = np.delete(bdframe,shifttab)

    # stop with telluric errors
    if ntellerror>0:
        print(ntellerror,' frames had APTELLURIC errors')
        raise Exception('halt: '+str(ntellerror)+' frames had APTELLURIC errors')

    #----------------------------------
    # STEP 5:  Dither Combining
    #----------------------------------
    print('STEP 5: Combining DITHER FRAMES with APDITHERCOMB')
    dithercombine(allframes,shifttab,pairtab,plugmap,combframe,median=True,
                  newerr=True,npad=50,nodither=nodither,avgwave=True,verbose=True)
    logger.info(' dithercomb '+os.path.dirname(planfile)+'{:8.2f}{:8.2f}'.format(time.time()-t1,time.time()-t0))
    if len(pairtab)==0 and nodither==0:
        raise Exception('halt: Error: no dither pairs')


    #----------------------------------
    # STEP 6:  Flux Calibration
    #----------------------------------
    print('STEP 6: Flux Calibration with AP1DFLUXING')
    if plantab['platetype'] != 'single':
        fiberloc = 300-plugmap['fiberdata'][obj]['fiberid']
        zero = np.median([hmag+2.5*np.log10(np.median(combframe[1]['flux'][:,fiberloc],axis=0))])
    else:
        zero = 0.
    finalframe = fluxing(combframe,plugmap)
    
    #----------------------------------------------------
    # Output apPlate frames and individual visit spectra, and load apVisit headers with individual star info
    #----------------------------------------------------
    print('Writing output apPlate and apVisit files')
    if plantab['platetype']=='single':
        single = True
    else:
        single = False
    mjdfrac = None
    if 'mjdfrac' in plantab.colnames and plantab['mjdfrac']==1:
        mjdfrac = finalframe[0]['header']['JD-MID']-2400000.5
    visitoutput(finalframe,plugmap,shifttab,pairtab,single=single,
                verbose=False,mjdfrac=mjdfrac,survey=survey)
    logger.info(' output '+os.pathbasename(planfile)+'{:8.2f}{:8.2f}'.format(time.time()-t1,time.time()-t0))

    #---------------
    # Radial velocity measurements for this visit
    #--------------
    #dorv:
    if 'platetype' in plantab.colnames:
        if plantab['platetype'] in ['normal','single']:
            continue
    print('Radial velocity measurements')
    locid = plugmap['locationid']
    visittabfile = load.filename('VisitSum',plate=plantab['plateid'],mjd=plantab['mjd'],
                                 reduction=plugmap['fiberdata'][obj]['tmass_style'],
                                 field=plantab['field'])
    if 'mjdfrac' in plantab.colnames and plantab['mjdfrac']==1:
        cmjd = str(plantab['mjd']).strip()
        s = visittabfile.split(cmjd)
        visittabfile = s[0] + '{:8.2f}'.format(mjdfrac) + s[1]
        #s = strsplit(visittabfile,cmjd,/extract,/regex)
        #visittabfile = s[0]+string(format='(f8.2)',mjdfrac)+s[1]
    if os.path.exists(visittabfile) and clobber==False:
        print('File already exists: ', visittabfile)
        return
    outdir = os.path.dirname(visittabfile)
    if os.path.exists(outdir)==False:
        os.makedirs(outdir)

    objtype = np.char.array(plugmap['fiberdata']['objtype']).strip()
    if fps:
        objind, = np.where((plugmap['fiberdata']['spectrographid'] == 2) &
                           (plugmap['fiberdata']['holetype'] == 'OBJECT') &
                           (plugmap['fiberdata']['assigned'] == 1) &
                           (plugmap['fiberdata']['on_target'] == 1) &
                           (plugmap['fiberdata']['valid'] == 1) &
                           (objtype == 'STAR' | objtype == 'HOT_STD') &
                           objtype != 'SKY' & objtype != 'none' &
                           np.char.array(plugmap['fiberdata']['tmass_style']).strip() !- '2MNone')
    else:
        objind, = np.where((plugmap['fiberdata']['spectrographid'] == 2) &
                           (plugmap['fiberdata']['holetype'] == 'OBJECT') &
                           (objtype == 'STAR' | objtype == 'HOT_STD') &
                           objtype != 'SKY' & objtype != 'none' &
                           np.char.array(plugmap['fiberdata']['tmass_style']).strip() !- '2MNone')
    objdata = plugmap['fiberdata'][objind]
    obj = plugmap['fiberdata'][objind]['tmass_style']

    if single:
        if 'mjdfrac' in plantab.colnames and plantab['mjdfrac']==1:
            mjd = finalframe[0]['header']['JD-MID']]-2400000.5
        else:
            mjd = plantab['mjd']
        visitfile = apread('Visit',plate=plantab['plateid'],mjd=mjd,fiber=objdata[0]['fiberid'],
                           reduction=obj,field=plantab['field'])
        header0 = visitfile[0]['hdr']
    else:
        finalframe = apread('Plate',mjd=plantab['mjd'],plate=plantab['plateid'],field=plantab['field'])
        header0 = finalframe[0]['hdr']

    if 'plateid' in plugmap.colnames:
        plate = plugmap['plateid']
    else:
        plate = plugmap['plate']
    mjd = plugmap['mjd']
    platemjd5 = str(plate)+'-'+str(mjd)

    # Loop over the objects
    allvisittab = []
    for istar in range(nobjind):
        visitfile = apogee_filename('Visit',plate=plantab['plateid'],mjd=plantab['mjd'],
                                    fiber=objdata[istar]['fiberid'],reduction=obj,field=plantab['field'])
        if 'mjdfrac' in plantab.colnames and plantab['mjdfrac']==1:
            cmjd = str(mjd)
            s = visitfile.split(cmjd)
            visitfile = s[0]+cmjd+s[1] + '{:8.2f}'.format(mjdfrac) + s[2]
            #s = strsplit(visitfile,cmjd,/extract,/regex)
            #visitfile = s[0]+cmjd+s[1]+string(format='(f8.2)',mjdfrac)+s[2]

        vtb = [('apogee_id',str,25),('target_id',str,50),('file',str,300),('uri',str,300),
               ('apred_vers',str,50),('fiberid',int),('plate',str,50),('exptime',float),('nframes',int),
               ('mjd',int),('telescope',str,10),('survey',str,20),('field',str,50),('design',str,50),
               ('programname',str,20),('objtype',str,20),('assigned',int),('on_target',int),('valid',int),
               ('ra',float),('dec',float),('glon',float),('glat',float),('healpix',int),('jmag',float),
               ('jerr',float),('hmag',float),('herr',float),('kmag',float),('kerr',float),('src_h',str,20),
               ('pmra',float),('pmdec',float),('pm_src',str,20),('apogee_target1',int),('apogee_target2',int),
               ('apogee_target3',int),('apogee_target4',int),('sdss5_target_pks',str,100),
               ('sdss5_target_catalogids',str,300),('sdss5_target_carton_pks',str,300),('sdss5_target_cartons',str,1000),
               ('sdss5_target_flagshex',str,1000),('brightneicount',int),('brightneiflag',int),('brightneiflux',float),
               ('catalogid',int),('sdss_id',int),('ra_sdss_id',float),('dec_sdss_id',float),('gaia_release',str,10),
               ('gaia_sourceid',int),('gaia_plx',float),('gaia_plx_error',float),('gaia_pmra',float),
               ('gaia_pmra_error',float),('gaia_pmdec',float),('gaia_pmdec_error',float),('gaia_gmag',float),
               ('gaia_gerr',float),('gaia_bpmag',float),('gaia_bperr',float),('gaia_rpmag',float),('gaia_rperr',float),
               ('sdssv_apogee_target',int),('firstcarton',str,50),('cadence',str,20),('program',str,50),
               ('category',str,50),('targflags',str,100),('snr',float),('starflag',int),('starflags',str,200),
               ('dateobs',str,50),('jd',float)]
        visittab = Table(np.zeros(1,dtype=np.dtype(vtb)))
        visittab['apogee_id'] = obj[istar]
        visittab['target_id'] = objdata[istar]['object']
        visittab['file'] = os.path.basename(visitfile)
        # URI is what you need to get the file, either on the web or at Utah
        mwm_root = os.environ['MWM_ROOT']
        visittab['uri'] = visitfile[len(mwm_root)+1:]
        visittab['apred_vers'] = apred_vers
        visittab['fiberid'] = objdata[istar]['fiberid']
        visittab['plate'] = str(plantab['plateid']).strip()
        visittab['field'] = str(plantab['field']).strip()
        if 'designid' in plantab.colnames:
            visittab['design'] = str(plantab['designid']).strip()
        else:
            visittab['design'] = -999
        visittab['exptime'] = finalframe[0]['hdr']['exptime']
        visittab['nframes'] = nframes
        visittab['mjd'] = plantab['mjd']
        visittab['telescope'] = load.telescope
        # Copy over all relevant columns from plugmap/plateHoles/catalogdb
        if 'gaia_sourceid' in objdata.colnames:
            if objdata[istar]['gaia_sourceid']=='':
                objdata[istar]['gaia_sourceid'] = '-1'
        for c in visittab.colnames:
            visittab[c] = objdata[istar][c]
        #STRUCT_ASSIGN,objdata[istar],visittab,/nozero
        #GLACTC,visittab.ra,visittab.dec,2000.0,glon,glat,1,/deg
        coo = SkyCoord(visittab['ra'],visittab['dec'],unit='deg',frame='icrs')
        visittab['glon'] = coo.galactic.l.deg
        visittab['glat'] = coo.galactic.b.deg
        visittab['apogee_target1'] = objdata[istar]['target1']
        visittab['apogee_target2'] = objdata[istar]['target2']
        visittab['apogee_target3'] = objdata[istar]['target3']
        visittab['apogee_target4'] = objdata[istar]['target4']

        # SDSS-V flags
        if plantab['plateid'] >= 15000:
            visittab['targflags'] = targflag(visittab['sdssv_apogee_target0'],survey=survey)
        # APOGEE-1/2 flags
        else:
            visittab['targflags'] = targflag(visittab['apogee_target1'],visittab['apogee_target2'],
                                             visittab['apogee_target3'],visittab['apogee_target4'],survey=survey)
        visittab['survey'] = survey
        visittab['field'] = str(plugmap['field']).strip()
        visittab['programname'] = plugmap['programname']

        # Get a few things from apVisit file (done in aprv also, but not
        #   if that is skipped....)
        #apgundef,str
        vtab = load.visit(visitfile)
        #APLOADVISIT,visitfile,vstr
        visittab['dateobs'] = vtab['dateobs']
        if 'JDMID' in vtab.colnames:
            visittab['jd'] = vtab['jdmid']
            aprvjd = vtab['jdmid']
        else:
            visittab['jd'] = vtab['jd']
            aprvjd = vtab['jd']            
        visittab['snr'] = vtab['snr']
        visittab['starflag'] = vtab['starflag']
        visittab['starflags'] = starflag(vtab['starflag'])
        visittab.write(visitfile,overwrite=True)
        #MWRFITS,visittab,visitfile,/silent
        allvisittab.append(visittab)
    # object loop
    logger.info(' aprv '+file_basename(planfile)+string(format='(f8.2)',systime(1)-t1)+string(format='(f8.2)',systime(1)-t0))


    # Save all RV info for all stars to apVisitSum file 
    #---------------------------
    hdu = fits.HDUList()
    # HDU0 - header only
    hdu.append(fits.ImageHDU())
    hdu[0].header['PLATEID'] = plantab['plateid']
    hdu[0].header['MJD'] = plantab['mjd']
    hdu[0].header['EXPTIME'] = header0['exptime'],'Total visit exptime per dither pos'
    hdu[0].header['JD-MID'] = header0['JD-MID'],' JD at midpoint of visit'
    hdu[0].header['UT-MID'] = header0['UT-MID'],' Date at midpoint of visit'
    ncombine = header0.get('NCOMBINE')
    if ncombine is None:
        ncombine = 1
    hdu[0].header['NCOMBINE'] = ncombine
    for j in range(ncombine):
        hdu[0].header['FRAME'+str(j+1)] = header0['FRAME'+str(j+1)],'Constituent frame'
    hdu[0].header['NPAIRS'] = header0['NPAIRS'],' Number of dither pairs combined'
    leadstr = 'AP1DVISIT: '
    hdu[0].header['V_APRED'] = plan.getgitvers(),'APOGEE software version'
    hdu[0].header['APRED'] = load.apred,'APOGEE Reduction version'
    hdu[0].header['HISTORY'] = leadstr+time.asctime()
    import socket
    hdu[0].header['HISTORY'] = leadstr+getpass.getuser()+' on '+socket.gethostname()
    import platform
    hdu[0].header['HISTORY'] = leadstr+'Python '+pyvers+' '+platform.system()+' '+platform.release()+' '+platform.architecture()[0]
    hdu[0].header['HISTORY'] = leadstr+load.prefix+'Visit information for '+str(len(allvisittab))+' Spectra'
    # HDU1 - table
    hdu.append(fits.table_to_hdu(allvisittab))
    hdu.writeto(visittabfile,overwrite=True)

    # Insert the apVisitSum information into the apogee_drp database
    if nobjind gt 0:
        print,'Loading visit data into the database'
        db = apogeedb.DBSession()
        db.ingest('visit',allvisittab)
        db.close()
        #DBINGEST_VISIT,allvisittab
    else:
        print('No data to load into the database')

    # plan files loop

    print('AP1DVISIT finished')
    logger.info('AP1DVISIT '+file_basename(planfile)+string(format='(f8.2)',systime(1)-t1)+string(format='(f8.2)',systime(1)-t0))
    dt = time.time()-t0
    print('dt = {:10.f} sec'.format(dt))

