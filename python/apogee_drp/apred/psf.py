#!/usr/bin/env python

import copy
import numpy as np
import os
import shutil
import time
from dlnpyutils import utils as dln, bindata
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema
import statsmodels.api as sm
from apogee_drp.utils import peakfit

def getprofdata(fibs,cols,hdulist,fiber2hdu):
    """ Get the apEPSF profile data for a range of fibers and columns."""

    # Fiber range
    if len(fibs)==2:
        nfibers = fibs[1]-fibs[0]+1
        fibers = np.arange(nfibers)+fibs[0]
    else:
        fibers = fibs
    nfibers = len(fibers)
    ncols = cols[1]-cols[0]
    mncol = (cols[0]+cols[1])*0.5
        
    # Initialize final arrays
    # dy, flux, X and Y
    data = np.zeros((nfibers*ncols*30,4),float)
    cnt = 0
        
    # Fiber loop    
    for i,f in enumerate(fibers):
        hind = fiber2hdu.get(f)
        if hind is not None:
            psfcat = hdulist[hind].data
            psfim = psfcat['IMG'][0,:,:]
            mny = np.mean(psfcat['CENT'])
            subim = psfim[:,cols[0]:cols[1]]
            ny,nx = subim.shape
            y1 = np.arange(ny)
            ymn = np.sum(y1.reshape(-1,1)*subim,axis=0)/np.sum(subim,axis=0)
            dy = y1.reshape(-1,1)-ymn.reshape(1,-1)
            col2d = (np.arange(ncols)+cols[0]).reshape(-1,1) + np.zeros(ny).reshape(1,-1)
            y2d = psfcat['CENT'][0,cols[0]:cols[1]].reshape(-1,1) + np.zeros(ny).reshape(1,-1)
            data[cnt:cnt+ncols*ny,0] = dy.ravel()      # dy        
            data[cnt:cnt+ncols*ny,1] = subim.ravel()   # flux
            data[cnt:cnt+ncols*ny,2] = col2d.ravel()   # X
            data[cnt:cnt+ncols*ny,3] = y2d.ravel()     # Y
            cnt += ncols*ny
    # Trim data
    data = data[0:cnt,:]
    # Trim out zero flux values
    gd, = np.where(data[:,1]>0)
    data = data[gd,:]

    return data


def avgprofile(fibs,cols,hdulist,fiber2hdu):
    """ Get the average profile for a range of fibers and columns."""
    # composite profile
    
    # Get profile data
    data = getprofdata(fibs,cols,hdulist,fiber2hdu)

    # Do binning first
    xr = [-7.0,7.0]
    binsize = 0.1
    nbins = int(np.ceil((xr[1]-xr[0])/binsize)+1)
    bins = np.linspace(xr[0],xr[1],nbins)
    ybin, bin_edges, binnumber = bindata.binned_statistic(data[:,0],data[:,1],statistic='percentile',
                                                          percentile=50,bins=bins)
    xbin = bin_edges[0:-1]+0.5*binsize

    # Use Gaussian smoothing
    gd, = np.where(np.isfinite(ybin) & (ybin>0))
    temp = ybin.copy()
    temp[~np.isfinite(ybin) | (ybin<=0)] = np.nan
    ybinsm = dln.gsmooth(temp,5)
    bad = ~np.isfinite(ybinsm)
    if np.sum(bad)>0:
        ybinsm[bad] = interp1d(xbin[~bad],ybinsm[~bad])(bad)

    # Make sure it's normalized
    ybinsm /= np.sum(ybinsm)*binsize
        
    # Use LOWESS to generate empirical template
    # it will use closest frac*N data points to a given point to estimate the smooth version
    # want at least 5 points
    #gd, = np.where(np.isfinite(ybin) & (ybin>0))
    ##lowess = sm.nonparametric.lowess(ybin[gd],xbin[gd], frac=0.05)    
    ## interpolate onto fine grid, leave some overhang
    #gdl, = np.where(np.isfinite(lowess[:,1]) & (lowess[:,1]>0))
    #lowint = interp1d(lowess[gdl,0],lowess[gdl,1],kind='quadratic',bounds_error=None,
    #                  fill_value="extrapolate")(xbin)
    #return data, xbin, ybin, lowess, lowint
    
    return data, xbin, ybin, ybinsm


def getprofilegrid(psffile,sparsefile,nfbin=5,ncbin=200):
    """ Get all of the profiles across the detector."""

    allim,head = fits.getdata(sparsefile,0,header=True)
    sim = allim[1,:,:]
    psfhdu = fits.open(psffile)

    # Get fiber numbers for each hdu
    fiber2hdu = {}
    fibernum = np.zeros(len(psfhdu)-1,int)
    for i in range(len(psfhdu)-1):
        fibernum[i] = psfhdu[i+1].data['FIBER']
        fiber2hdu[psfhdu[i+1].data[0]['FIBER']] = i+1
    
    fibers = np.arange(0,300,nfbin)
    columns = np.arange(10,2000,ncbin)

    # Get sparse data
    
    #data = np.zeros((len(fibers),len(columns),700),float)
    data = []
    mnx = np.zeros((len(columns),len(fibers)),float)
    mny = np.zeros((len(columns),len(fibers)),float)
    profiles = np.zeros((len(columns),len(fibers),300),float)
    binsize = 0.1
    xx = np.arange(300)*binsize-14.95
    
    # Column loop
    for i,c in enumerate(columns):

        # Get sparse profile
        sflux = np.zeros(2048,float)
        sflux[4:2044] = np.nanmedian(sim[4:2044,c:c+ncbin],axis=1)
        # Find the peaks
        maxind, = argrelextrema(sflux, np.greater)
        gd, = np.where(sflux[maxind] > 0.1*np.max(sflux))
        peaks = maxind[gd]
        linestr0 = peakfit.peakfit(sflux,pix0=peaks)
        # Distances to neighbors
        ldiff = linestr0['pars'][:,1]-np.hstack((0,linestr0['pars'][0:-1,1]))
        rdiff = np.hstack((linestr0['pars'][1:,1],2048))-linestr0['pars'][:,1]
        gd, = np.where((ldiff >= 22) & (rdiff >= 22))
        ngd = len(gd)
        # 15
        linestr = linestr0[gd]
        
        # Fiber loop
        for j,f in enumerate(fibers):

            print(f,c)
            #data1, xbin,ybin,lowess,ylowess = avgprofile([f,f+nfbin],[c,c+ncbin],psfhdu,fiber2hdu)
            data1, xbin,ybin,ybinsm = avgprofile([f,f+nfbin],[c,c+ncbin],psfhdu,fiber2hdu)            
            
            # Get closest sparse fiber
            ytracearr = []
            for k in np.arange(f,f+nfbin):
                if fiber2hdu.get(k) is not None:
                    psfcat = psfhdu[fiber2hdu[k]].data
                    ytracearr.append(np.median(psfcat['CENT']))
            ytrace = np.median(np.array(ytracearr))
            diff = linestr['pars'][:,1]-ytrace
            bestind = np.argmin(np.abs(diff))
            linestr1 = linestr[bestind]
            ycensparse = linestr1['pars'][1]
            dysparse = np.arange(31).astype(float)-15
            fluxsparse = sflux[int(round(ycensparse))-15:int(round(ycensparse))+16]
            fluxsparse /= np.sum(fluxsparse)   # normalize
            # replace very low values with point on opposite side
            bad, = np.where(fluxsparse<1e-5)
            if len(bad)>0:
                good = len(fluxsparse)-bad-1
                fluxsparse[bad] = fluxsparse[good]
                print('fixing fluxsparse edge bad value')
                #import matplotlib.pyplot as plt
                #import pdb; pdb.set_trace()
            fluxsparse /= np.sum(fluxsparse)   # normalize again          
            ymnsparse = np.sum(dysparse*fluxsparse)/np.sum(fluxsparse)
            dysparse -= ymnsparse


            #import matplotlib.pyplot as plt
            #plt.clf()
            #plt.scatter(dysparse,fluxsparse,c='blue',s=100,marker='+')
            #plt.plot(dysparse,fluxsparse,c='blue')
            #plt.yscale('log')
            #plt.plot(xbin,ybinsm,c='r')
            #plt.show()

            # Use points +/-3 for scaling
            gdpt, = np.where((np.abs(dysparse) <= 3) & (fluxsparse > 0.4*np.max(fluxsparse)))
            ybinsm2 = interp1d(xbin,ybinsm)(dysparse[gdpt])
            ratio = np.median(ybinsm2/fluxsparse[gdpt])
            ybinsm /= ratio   # scale thin curve to sparse one
            
            # Interpolate sparse onto finer scale
            nxfine = 30/0.1
            xfine = np.arange(nxfine)*0.1-14.95
            fluxsparsefine = 10**interp1d(dysparse,np.log10(fluxsparse),kind='quadratic',bounds_error=False,fill_value=np.nan)(xfine)
            #fluxsparsefine = 10**interp1d(dysparse,np.log10(fluxsparse),kind='quadratic',bounds_error=False,fill_value=np.nan)(xbin)            

            # switch to the sparse courve around x~3, around 3sigma
            sigma = np.sqrt(np.sum(ybinsm*xbin**2)/np.sum(ybinsm))
            # use logistic curve
            wt = 1/(1+np.exp(-2*(np.abs(xbin)-2.5*sigma)))
            gdsparse, = np.where((xfine>=np.min(xbin)-0.001) & (xfine<=np.max(xbin)+0.001))            
            combflux = fluxsparsefine[gdsparse]*wt + (1-wt)*ybinsm
            #plt.plot(xbin,combflux,c='orange')

            # Stuff the central combined portion into the final profile
            yprofile = fluxsparsefine.copy()
            yprofile[gdsparse] = combflux

            # Fix any NaNs
            ind = np.arange(len(yprofile))
            good1 = np.where(np.isfinite(yprofile))[0][0]
            bad1, = np.where(~np.isfinite(yprofile) & (ind<10))
            if len(bad1)>0:
                yprofile[bad1] = yprofile[good1]
            good2 = np.where(np.isfinite(yprofile))[0][-1]
            bad2, = np.where(~np.isfinite(yprofile) & (ind>len(yprofile)-10))
            if len(bad2)>0:
                yprofile[bad2] = yprofile[good2]
            bad = ~np.isfinite(yprofile)
            if np.sum(bad)>0:
                print('some nans')
                import pdb; pdb.set_trace()
            
            # Normalize
            yprofile /= np.sum(yprofile)*binsize

            if np.min(yprofile)<1e-5:
                print('problem')
                import matplotlib.pyplot as plt
                import pdb; pdb.set_trace()
            
            #plt.clf()
            #plt.scatter(data[:,0],data[:,1],s=5)
            #plt.plot(xbin,ybin,c='r')
            ##plt.plot(lowess[:,0],lowess[:,1],c='g')
            #plt.plot(xbin,ybinsm,c='b')  
            #plt.yscale('log')
            #plt.xlim(-8,8)
            #plt.ylim(1e-5,1)
            #plt.title('fiber='+str(f)+' column='+str(c))

            data.append( [xbin,ybin,ybinsm,f,c] )
            mnx[i,j] = np.median(data1[:,2])
            mny[i,j] = np.median(data1[:,3])
            profiles[i,j,:] = yprofile 
            
            #import pdb; pdb.set_trace()


            
    import pdb; pdb.set_trace()

    return data,mnx,mny,profiles


def extract_pmul(p1lo,p1hi,img,p2):

    lo = np.max([p1lo,p2['lo'][0]])
    k1 = lo-p1lo
    l1 = lo-p2['lo'][0]
    hi = np.min([p1hi,p2['hi'][0]])
    k2 = hi-p1lo
    l2 = hi-p2['lo'][0]
    if lo>hi:
        return np.zeros(2048,float)
    img2 = p2['img'][0].T  # transpose
    if lo==hi:
        return img[:,k1:k2+1]*img2[:,l1:l2+1]
    else:
        return np.nansum(img[:,k1:k2+1]*img2[:,l1:l2+1],axis=1)

def extract(frame,epsf,doback=False,skip=False,scat=None,subonly=False):
    """
    This extracts spectra using an empirical PSF.

    Extract spectrum under the assumption that a given pixel only contributes
    to two neighboring traces, leading to a tridiagonal matrix inversion.

    Parameters
    ----------
    frame : dict
       The 2D input structure with FLUX, VAR and MASK.
    epsf : list
       A list with the empirical PSF.
    doback : boolean, optional
       Subtract the background.  False by default.

    Returns
    -------
    outstr : dict
        The 1D output structure with FLUX, VAR and MASK.
    back : numpy array
        The background
    model : numpy array
        The model 2D image

    Example
    -------
    outstr,back,model = extract(frame,epsf)

    By J. Holtzman  2011
      Incorporated into ap2dproc.pro  D.Nidever May 2011  

    """

    WARNMASK = -16640
    BADMASK = 16639
    BADERR = 1.00000e+10
    
    nframe = len(frame)
    ntrace = len(epsf)

    fibers = [e['fiber'] for e in epsf]
    red = frame['flux'].T
    var = frame['err'].T**2
    inmask = frame['mask'].T
    # use the transposes
    
    #if keyword_set(scat) then scat_remove,red,scat=scat,mask=inmask

    # Initialize output arrays
    spec = np.zeros((2048,300),float)
    err = np.zeros((2048,300),float)+999999.09 #+baderr()
    outmask = np.ones((2048,300),int)

    # calculate extraction matrix
    if doback:
        nback = 1 
        back = np.zeros(2048,float)
    else:
        nback = 0
    beta = np.zeros((ntrace+nback,2048),float)
    betavar = np.zeros((ntrace+nback,2048),float)
    psftot = np.zeros((ntrace+nback,2048),float)
    tridiag = np.zeros((3,ntrace+nback,2048),float)
    warnmasked = np.zeros((ntrace+nback,2048),int)
    badmasked = np.zeros((ntrace+nback,2048),int)
    inmask_warn = (inmask & WARNMASK) > 0
    inmask_bad = (inmask & BADMASK) > 0
    for k in np.arange(0,ntrace+nback):
        # Background
        if k > ntrace-1:
            beta[k,:] = np.nansum(red[:,lo:hi+1],axis=1)
            betavar[k,:] = np.nansum(var[:,lo:hi+1],axis=1)
            psftot[k,:] = 1.

        # Fibers
        else:
            # get EPSF and set bad pixels to NaN
            p1 = epsf[k]
            lo = epsf[k]['lo'][0]
            hi = epsf[k]['hi'][0]
            bad = (~np.isfinite(red[:,lo:hi+1]) | (red[:,lo:hi+1] == 0) |
                   ((inmask[:,lo:hi+1] & BADMASK) > 0) )
            nbad = np.sum(bad)
            img = p1['img'][0].T   # transpose
            if nbad > 0:
                img[bad] = np.nan

            # are there any warning flags for this trace? If so, flag the output
            for i in range(2048):
                warnmasked[k,i] = inmask_warn[i,lo]
                badmasked[k,i] = inmask_bad[i,lo]
                for j in np.arange(lo+1,hi+1):
                    warnmasked[k,i] = warnmasked[k,i] | inmask_warn[i,j]
                    badmasked[k,i] = badmasked[k,i] | inmask_bad[i,j]
            psftot[k,:] = np.nansum(img,axis=1)
            beta[k,:] = np.nansum(red[:,lo:hi+1]*img,axis=1)
            betavar[k,:] = np.nansum(var[:,lo:hi+1]*img**2,axis=1)

        # First fiber (on the bottom edge)
        if k==0:
            ll = 1
            for l in np.arange(k,k+2):
                tridiag[ll,k,:] = extract_pmul(p1['lo'][0],p1['hi'][0],img,epsf[l])
                ll += 1

        # Last fiber (on top edge)
        elif k == ntrace-1:
            ll = 0
            for k in np.arange(k-1,k+1):
                tridiag[ll,k,:] = extract_pmul(p1['lo'][0],p1['hi'][0],img,epsf[l])
                ll += 1

        # Background terms
        elif k > ntrace-1:
            tridiag[1,k,:] = hi-lo+1

        # Middle fibers (not first or last)
        else:
            ll = 0
            for l in np.arange(k-1,k+2):
                tridiag[ll,k,:] = extract_pmul(p1['lo'][0],p1['hi'][0],img,epsf[l])
                ll += 1

    for i in np.arange(4,2044):
        # Good fibers
        good, = np.where(psftot[:,i] > 0.5)
        ngood = len(good)
        bad, = np.where(psftot[:,i] <= 0.5)
        nbad = len(bad)
        if nbad > 0:
            bad0, = np.where(bad>0)
            nbad0 = len(bad0)
            if nbad0 > 0:
                tridiag[2,bad[bad0]-1,i]=0 
            bad1, = np.where(bad < ntrace-1)
            nbad1 = len(bad1)
            if nbad1 > 0:
                tridiag[0,bad[bad1]+1,i] = 0 
    if ngood>0:
        a = tridiag[0,good,i]
        b = tridiag[1,good,i]
        c = tridiag[2,good,i]
        v = beta[good,i]
        vvar = betavar[good,i]
        for j in np.arange(1,ngood):
            m = a[j]/b[j-1]
            b[j] = b[j]-m*c[j-1]
            v[j] = v[j]-m*v[j-1]
            vvar[j] = vvar[j]+m**2*vvar[j-1]
        x = np.zeros(ngood,float)
        xvar = np.zeros(ngood,float)
        x[ngood-1] = v[ngood-1]/b[ngood-1]
        xvar[ngood-1] = vvar[ngood-1]/b[ngood-1]**2
        for j in np.flip(np.arange(0,ngood-1)):
            x[j] = (v[j]-c[j]*x[j+1])/b[j]
            xvar[j] = (vvar[j]+c[j]**2*xvar[j+1])/b[j]**2
        spec[i,fibers[good]] = x
        err[i,fibers[good]] = np.sqrt(xvar)
        # mask the bad pixels
        outmask[i,fibers[good]] = 0
        if nbad > 0:
            outmask[i,fibers[bad]] = maskval('NOT_ENOUGH_PSF') | badmasked[bad,i]
        # put the warning bits into the mask
        outmask[i,fibers] = outmask[i,fibers] | warnmasked[:,i]

    # No good fibers for this column
    else:
        spec[i,:] = 0
        err[i,:] = BADERR
        outmask[i,fibers] = maskval('NOT_ENOUGH_PSF') | badmasked[:,i]

    if doback:
        back[i] = x[ngood-1]


    # Catch any NaNs (shouldn't be there, but ....)
    bad, = np.where(~np.isfinite(spec))
    nbad = len(bad)
    if nbad > 0:
        spec[bad] = 0.
        err[bad] = BADERR
        outmask[bad] = 1


    # Put together the output dictionary
    outstr = {'flux':spec, 'err':err, 'mask':outmask}

    # Create the Model 2D image
    model = np.zeros(red.shape,float)
    t = spec
    bad, = np.where(t<=0)
    if len(bad)>0:
        t[bad] = 0
    for k in range(ntrace):
        nf = 1
        ns = 0
        if subonly:
            junk, = np.where(subonly==k)
            nf = len(junk)
        if skip:
            junk, = np.where(skip==k)
            ns = len(junk)
        if nf > 0 and ns==0:
            p1 = epsf[k]
            lo = epsf[k]['lo'][0]
            hi = epsf[k]['hi'][0]
            img = p1['img'][0].T
            rows = np.ones(hi-lo+1,int)
            fiber = epsf[k]['fiber']
            #model[:,lo:hi] += img[:,:]*(rows##t[:,fiber])
            model[:,lo:hi] += img[:,:]*(rows.reshape(-1,1)*t[:,fiber])                                        

    return outstr,back,model

      
if __name__ == '__main__' :

    psfdir = '/Users/nidever/sdss5/mwm/apogee/spectro/redux/daily/cal/apogee-n/psf/'
    psffile = 'apEPSF-b-39880014.fits'
    sparsefile = 'apSparse-39870034.fits'
    allim,head = fits.getdata(psfdir+sparsefile,0,header=True)
    im = allim[1,:,:]
    psfhdu = fits.open(psfdir+psffile)

    data = getprofilegrid(psfdir+psffile,psfdir+sparsefile)


    import pdb; pdb.set_trace()

    # Check domeflat vs. quartzflat profile to see if the profile changes
    # if the light goes through the FPS octagonal fibers or not

    # To combine the 300-fiber quartzflat ("narrow") and sparse data
    # for each "narrow" profile, find the closest good sparse profile
    # then scale the narrow profile to the sparse profile for the inner
    # points (r<4 or so).  Then splice them together, or maybe use
    # an average in the crossover region

    # Interpolating the profiles
    # 1) grid interpolation
    #  Use the whole 3D grid (Nprofile,Nx,Ny) to do interpolation
    #  For each profile point, we can use RectBivariateSpline
    #  to get the value for any detector X/Y position.

    # For each fiber, you could interpolate the profile for all columns
    # at once.  Then use

    # When we do extraction, we do it one column at a time.
    # We want profiles for all 300 fibers and a single column
    # take the two closest Y positions in the 3D profile grid
    # then linearly interpolate (Nprofile,Ny) for those two X points,
    # Then we have (Nprofile,Ny) for the column of interest.
    # Use RectBivariateSpline on the (Nprofile,Ny) data for this column
    # and the trace positions for the 300 fibers for this column to
    # get full profiles for each fiber (Nprofile,300).
    # Then get the actual profiles for each fiber using the trace
    # positions and pixel values (could do the last two steps together).
    
    # 2) ANN
    # train an ANN on the profiles as an emulator
    # you give it the detector X/Y position and it returns the profile
    # for that position
  

    # When we do extraction, we want to solve the three neighbors simultanously
    # but the wings affect +/-15 pixels or two neighbors on each side.
    # The pixels nearest the peak have the most weight for solving the fiber's
    # flux, but wings will affect the solution (especially for bright fibers).
    
