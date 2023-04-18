#!/usr/bin/env python

"""APQUICK_UTAH.PY - APOGEE Quick reduction of Utah data.

This performs a very quick reduction of an APOGEE exposure to
give feedback about the S/N.  This program can be run on an exposure
in progress or on that has already finished because it uses the apRaw
files (individual reads). 

Here are the main steps:
1) Use Fowler/CDS to collapse cube to a 2D image using very few reads.
    This is only done for the green chip.
2) Construct the noise model assuming we used up-the-ramp as is done
     by the full pipeline.
3) Boxcar extract all fibers but only for ~50 columns at the center
     of the image.
4) Fit a line to log(S/N) vs. Hmag and use it to find the S/N at the
     fiducial H magnitude.
5) Write out the results to a single multi-extension FITS file.

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
#__version__ = '20180922'  # yyyymmdd                                                                                                                           

import os
import pdb
import numpy as np
import warnings
from astropy.io import fits,ascii
from astropy.table import Table, Column
from glob import glob
from scipy.ndimage import median_filter,generic_filter
from apogee_drp.utils import apzip,plan,apload,yanny,plugmap,platedata,bitmask,info
#from apogee_drp.utils import yanny, apload
#from sdss_access.path import path
#from . import config  # get loaded config values 
#from . import __version__
#from . import bitmask as bmask
#from . import yanny
import subprocess

# In the future, use yanny tools from pyDL
#from pydl.pydlutils.yanny import (is_yanny, read_table_yanny,
#                                  write_table_yanny)
#from pydl.pydlutils import yanny


# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings('ignore', category=UserWarning, append=True)

#observatory = os.getenv("OBSERVATORY")
#if observatory == "APO":
#    prefix = 'ap'
#else:
    #prefix = 'as'
    # Fix plugmap_dir for lco
    # plugmap_dir: /home/sdss5/software/sdsscore/main/apo/summary_files/
    #config['apogee_mountain']['plugmap_dir'] = config['apogee_mountain']['plugmap_dir'].replace('apo','lco')

"""
Wrappper for running apquick on data stored at Utah
"""

detNumPlateN = 13390003
detNumPlateS = 22810006
bpmNumPlateN = 33770001
bpmNumPlateS = 35800002

def getPsfList(load=None, update=False):
    # Find the psf list directory
    codedir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.dirname(os.path.dirname(os.path.dirname(codedir))) + '/data/psflists/'
    pfile = datadir+load.observatory+'PSFall.dat'

    if update:
        pfilesPlate = ascii.read(datadir+load.observatory+'PSFplate.dat')
        numPlate = np.array(d['COL1'])
        expPlate = np.char.zfill(np.array(d['COL2']).astype(str),8)
        redux_dir = os.environ.get('APOGEE_REDUX')+'/'+load.apred+'/'
        pdir = redux_dir+'cal/'+load.instrument+'/psf/'
        pfiles = glob.glob(pdir+load.prefix+'PSF-b-*fits')
        pfiles.sort()
        pfiles = np.array(pfiles)
        npfiles = len(pfiles)
        expFPS = np.empty(npfiles).astype(str)
        for i in range(npfiles): exp[i]=os.path.basename(f[i]).split('-')[2].split('.')[0]
        expAll = np.concatenate([expPlate,expFPS])
        numAll = np.char.zfill(np.arange(len(expAll)).astype(str),8)
        ascii.write([numAll,expAll], pfile, format='no_header', overwrite=True)

    d = ascii.read(pfile)
    return np.array(d['col2'])

def utah(telescope='apo25m', apred='daily', updatePSF=False):
    # Set up directory paths
    load = apload.ApLoad(apred=apred, telescope=telescope)
    apodir = os.environ.get('APOGEE_REDUX')+'/'+apred+'/'
    outdir = apodir+'quickred/'+telescope+'/'
    
    # Raw data will be extracted temporarily to current working directory (then removed)
    cwd = os.getcwd()+'/'

    # Get science exposure numbers from summary file
    edata0 = fits.getdata(apodir+'monitor/'+load.instrument+'Sci.fits')
    nexp = len(edata0)

    # Get PSF exposure numbers from getPsfList subroutine
    expPSF = getPsfList(load=load, update=updatePSF)

    pdb.set_trace()
    # Loop over exposures
    for iexp in range(500,501):
        edata = edata0[iexp]
        framenum = edata['IM']
        rawfilepath = load.filename('R', num=framenum, chips='b').replace('R-','R-b-')
        if os.path.exists(rawfilepath) == False:
            print(rawfilepath+' not found!')
            continue
        rawfile = os.path.basename(rawfilepath)
        rawfilefits = rawfile.replace('.apz','.fits')
        mjd = os.path.basename(os.path.dirname(rawfilepath))
        print(mjd)
        infile = cwd+rawfilefits
        #pdb.set_trace()
        # Unzip the file
        if os.path.exists(infile) == False: apzip.unzip(rawfilepath, fitsdir=cwd)
        hdulist = fits.open(rawfilefits)
        nreads = len(hdulist)-1
        #for iread in range(nreads):
        #    d = hdulist[iread+1].data
        #    dname = rawfilefits.replace('R-','Raw-').replace('.fits','-'+str(iread+1).zfill(3)+'.fits')
        #    print(dname)
        #    Table(d).write(dname,overwrite=True)
        output = runquick(infile, hdulist=hdulist, framenum=framenum, mjd=mjd, load=load)

        pdb.set_trace()


def nanmedfilt(x,size,mode='reflect'):
    return generic_filter(x, np.nanmedian, size=size)

def getmjd(frame) :
    """ Get chracter MJD from frame number """
    num = (int(frame) - int(frame)%10000 ) / 10000
    return int(num)+55562

def mad(data):
    """ Calculate the median absolute deviation."""
    data_median = np.nanmedian(data)
    result =  np.nanmedian(np.abs(data - data_median))
    return result * 1.482602218505602

class Frame:
    """Object class for a single APOGEE read image.

    Parameters
    ----------
    imfile: str
          The filename of the FITS file.
    head : astropy header
         The header of the FITS file.
    im : numpy array
       The 2D raw image.
    framenum : int
       APOGEE 8-digit frame number.
    num : int
       The read number of the image.

    Attributes
    ----------
    Same as the parameters.

    """

    def __init__(self,imfile,head,im,framenum,num):
        self.file = imfile
        self.head = head
        self.im = im
        self.framenum = framenum
        self.num = num


class SpecFrame:
    """Object class for an extracted APOGEE spectrum.

    Parameters
    ----------
    flux : numpy array
        The 2D flux array [Nfibers,Ncolumns].
    err : numpy array
        The 2D error array [Nfibers,Ncolumns].
    head : astropy header
         The header of the FITS file (from the last read used).

    """

    def __init__(self,flux,err,head,framenum):
        self.flux = flux
        self.err = err
        self.head = head
        self.framenum = framenum

def refcorrsub(image,ref):
    # refsub subtracts the reference array from each quadrant with proper flipping 
    revref = np.flip(ref,axis=1)
    image[:,0:512] -= ref
    image[:,512:1024] -= revref
    image[:,1024:1536] -= ref
    image[:,1536:2048] -= revref
    return image

def refcorr(image):
    """ Apply reference pixel correction to an apRaw APOGEE image."""

    # The RAW images:
    # Basic orientation is long wavelengths on the left, shorter to the
    # right (higher column number).  readout direction is:
    #         Red                       Green                 Blue           WG RefB RefG RefR
    #|---->|<----|--->|<----||---->|<----|--->|<----||---->|<----|--->|<----||---->|<----|--->|<----|
    image = image.astype(int)

    # Chip loop
    im = image[:,:2048]
    ref = image[:,2048:]
    # Reference subtraction
    im = refcorrsub(im,ref)
    # Stick it back in the array
    image[:,:2048] = im

    return image

# Load the frames
def loadframes(filename,hdulist,framenum,load=None,nfowler=2,chip=2,lastread=None):
    """Loads apRaw reads at beginning and end of an exposure.

    This function loads apRaw reads for an exposure at the beginning
    and end to be used for Fowler/CDS sampling.
    
    Parameters
    ----------
    rawdir : str
           Directory for the apRaw files.
    framenum : str
          The 8 digit APOGEE frame number.
    nfowler : int
            Number of reads to load.  The default is 2.
    chip : int
            Which detector to use (1, 2, or 3).  The default is 2. 
    lastread : int
            The number of the last read to use.

    Returns
    -------
    bframes : list of Frame objects
            List of Frame objects with images and header information for the
            reads at the beginning of the exposure.  Note, the first read is
            always skipped.
    eframes : list of Frame objects
            List of Frame objects with images and header information for the
            reads at the end of the exposure.
    nreads : int
           The number of reads (files) used.

    Example
    -------

    .. code-block:: python

        bframes,eframes,nreads = loadframes('/data/apogee/spectro/raw/58382/',28200054,nfowler=2,chip=2)

    """

    # Get the file list
    #files = glob(rawdir+load.prefix+"Raw-"+str(framenum)+"-???.fits")
    nfiles = len(hdulist)-1
    if nfiles==0:
        print("No files for "+str(framenum))
        return None,None,None
    # Sort the files
    #files = np.sort(files)
    # Get the read numbers
    readnum = np.arange(0,nfiles)
    #readnum = np.zeros(nfiles,dtype=int)
    #for i in range(nfiles):
    #    base = os.path.basename(files[i])
    #    # apRaw-28190009-059.fits
    #    readnum[i] = np.int(base[15:18])
    # If readnum input then only use reads up to that number
    if lastread is not None:
        gdf, = np.where(readnum <= int(lastread))
        ngdf = len(gdf)
        if ngdf < 2:
            raise Exception("Not enough reads")
        # Only keep the files that we want to use
        files = files[gdf]
        readnum = readnum[gdf]
        nfiles = ngdf
    # What nfowler are we using
    # Use nfowler=1 for DOMEFLAT
    #head = fits.getheader(files[0].replace('Raw','R').replace('-001.fits','.fits'))
    head = fits.getheader(filename)
    #exptype = head1.get('exptype')
    #if exptype=='DOMEFLAT':
    #    print('Using nfowler=1 for DOMEFLAT')
    #    nfowler = 1
    nfowler_use = nfowler
    # Raise an error if we don't have enough reads for nfowler
    if nfiles<3:
        raise Exception("Not enough reads ("+str(nfiles)+")")
    if nfiles<(2*nfowler+1):
        nfowler_use = np.int(np.floor((nfiles-1)/2.))
        print("Not enough reads ("+str(nfiles)+") for Nfowler="+str(nfowler)+".  Using "+str(nfowler_use)+" instead")
    #if nfiles<nfowler:
    #    raise Exception("Not enough reads ("+str(nfiles)+") for Nfowler="+str(nfowler))
    # Load the begging set of frames
    #  skip the first one, bad
    bframes = []
    for i in range(nfowler_use):
        imfile = filename#files[i+1]
        num = readnum[i+1]
        im = hdulist[i+1].data
        #im = fits.getdata(imfile)#,header=True)
        im = refcorr(im)  # apply reference correction   
        if chip is not None:
            im = im[:,:2048]
            #im = im[:,(chip-1)*2048:chip*2048]
        frame = Frame(imfile,head,im,framenum,num)
        bframes.append(frame)
    # Load the ending set of frames
    eframes= []
    for i in range(nfowler_use):
        imfile = filename#files[nfiles-nfowler_use+i]
        num = readnum[nfiles-nfowler_use+i]
        im = hdulist[nfiles-nfowler_use+i].data
        #im,head = fits.getdata(imfile,header=True)
        im = refcorr(im)  # apply reference correction   
        if chip is not None:
            im = im[:,:2048]
            #im = im[:,(chip-1)*2048:chip*2048]
        frame = Frame(imfile,head,im,framenum,num)
        eframes.append(frame)
    # Return the lists
    return bframes,eframes,nfiles



# Perform Fowler sampling
def fowler(bframes,eframes):
    """Collapses exposure to 2D using Fowler/CDS sampling.

    This function performs Fowler/CDS sampling given a list of
    raw frames from the beginning and end of the exposure.
    
    Parameters
    ----------
    bframes : list of Frame objects
            The read images at the beginning of the exposure.
    eframes : list of Frame objects
            The read images at the end of the exposure.

    Returns
    -------
    im : numpy array
       The collapsed 2D image.


    Example
    -------

    .. code-block:: python

        im = fowler(bframes,eframes)

    """

    # Beginning sample
    nbeg = len(bframes)
    nx,ny = bframes[0].im.shape
    if (nbeg==1):
        im_beg = np.array(bframes[0].im.copy(),dtype=float)
    else:
        im_beg = np.array(bframes[0].im.copy(),dtype=float)*0
        for i in range(nbeg): im_beg += np.array(bframes[i].im.copy(),dtype=float)
        im_beg /= np.float(nbeg)
    # End sample
    nend = len(eframes)
    if (nend==1):
        im_end = np.array(eframes[0].im.copy(),dtype=float)
    else:
        im_end = np.array(eframes[0].im.copy(),dtype=float)*0
        for i in range(nend): im_end += np.array(eframes[i].im.copy(),dtype=float)
        im_end /= np.float(nend)
    # The middle read will be used twice for 3 reads
    # Subtract beginning from end
    im = im_end - im_beg

    # Fix the background level by using the upper and lower edges
    # first and last 10 rows
    # Do a linear ramp for each quadrant
    exptype = eframes[0].head['exptype']
    if exptype != 'INTERNALFLAT' and exptype != 'DARK':
        for i in range(4):
            med1 = np.median(im[0:10,i*512:(i+1)*512])
            med2 = np.median(im[-10:,i*512:(i+1)*512])
            im[:,i*512:(i+1)*512] -= (np.arange(2048).astype(float)*(med2-med1)/2047+med1).reshape(-1,1)

    return im


# Construct the noise model images
def noisemodel(im,nreads,noise,gain):
    """Computes the noise model for an image.

    This function computes the noise image for the case where
    the data cube had been collapsed using UTR as the full
    reduction code does.
    
    Parameters
    ----------
    im : numpy array
       The 2D collapsed image.
    nreads : int
       The number of reads in the image.
    noise : float
       The read noise for a single read in ADU.
    gain : float
       The gain in electrons/ADU.

    Returns
    -------
    err : numpy array
        The noise model image in ADU.

    Example
    -------

    .. code-block:: python

        err = noisemodel(im,nreads,noise,gain)

    """

    # need nread, ngdreads, noise, gain
    ngdreads = np.float(nreads-1)

    ## READ NOISE
    #if n_elements(rdnoiseim) gt 0 then begin
    #  noise = median(rdnoiseim)
    #endif else begin
    #  noise = 12.0  ; default value
    #endelse

    # See Equation 1 in Rauscher et al.(2007), SPIE
    #  with m=1
    #  noise and image/flux should be in electrons, sample_noise is in electrons
    impos = im.copy()
    bad = (impos < 0)
    impos[bad] = 0
    sample_noise = np.sqrt( 12*(ngdreads-1)/(np.float(nreads)*(ngdreads+1))*noise**2 + 6.*(ngdreads**2+1)/(5*ngdreads*(ngdreads+1))*impos*gain )
    sample_noise /= gain  # convert to ADU

    # Start the variance array
    varim = im.copy()*0
    # Sample/read noise
    varim += sample_noise**2 

    #Now convert to ELECTRONS
    #if n_elements(detcorr) gt 0 and keyword_set(outelectrons) then begin
    #  varim *= gain^2
    #  im *= gain

    # Convert variance to error
    err = im.copy()*0+1
    good = (varim > 0)
    err[good] = np.sqrt(varim[good])

    return err

def bpmfix(frame,bpm):
    """ Fix bad pixels."""

    # Replace them with averages of the neighboring pixels
    bad = (bpm>0)
    nbpm = np.sum(bad)
    if nbpm>0:
        nei = np.zeros((nbpm,8),float)
        neimask = np.zeros((nbpm,8),bool)
        ind1,ind2 = np.where(bad)
        ind1p = ind1+1
        ind1p[ind1p>2047] = 0
        ind2p = ind2+1
        ind2p[ind2p>2047] = 0
        nei[:,0] = frame.im[ind1-1,ind2-1]
        nei[:,1] = frame.im[ind1,ind2-1]
        nei[:,2] = frame.im[ind1p,ind2-1]
        nei[:,3] = frame.im[ind1-1,ind2]
        nei[:,4] = frame.im[ind1p,ind2]
        nei[:,5] = frame.im[ind1-1,ind2p]
        nei[:,6] = frame.im[ind1,ind2p]
        nei[:,7] = frame.im[ind1p,ind2p]
        new = np.median(nei,axis=1)
        frame.im[ind1,ind2] = new

    frame.err[bad] = 1e30
    frame.head['HISTORY'] = str(nbpm)+' bad pixels masked out'

    return frame

# Boxcar extract fibers
def boxextract(frame,tracestr,fibers=None,xlo=0,xhi=None):
    """This function performs boxcar extraction on a 2D image.
    
    Parameters
    ----------
    frame : Frame object
          The Frame object that includes the flux and error images.
    tracestr : numpy structured array
          Numpy structured array that gives information on the traces.
    fibers : array or list, optional
           List/array of fibers to extract.  By default all fibers
           in tracestr are extracted.
    xlo : int, optional
        The starting column to extract.  By default this is 0.
    xhi : int, optional
        The ending column to extract.  This is set to the final column
        of the 2D iamge.

    Returns
    -------
    spec : SpecFrame object
         The SpecFrame object that contains extracted flux and error arrays.

    Example
    -------

    .. code-block:: python

        spec = boxextract(frame,tracestr)

    """

    ntrace = len(tracestr)
    if fibers is None:
        fibers = np.arange(ntrace)
    if xhi is None:
        xhi = frame.im.shape[1]-1
    # transpose b/c x/y are flipped in python
    flux = frame.im.copy().T
    err = frame.err.copy().T
    nx,ny = flux.shape
    # Loop through the Fibers
    ncol = np.int(xhi-xlo+1)
    x = np.arange(xlo,xhi+1)
    nfibers = len(fibers)
    oflux = np.zeros((nfibers,ncol))
    oerr = np.zeros((nfibers,ncol))
    for i in range(nfibers):
        fwhm = tracestr['FWHM'][fibers[i]]
        coef = tracestr['COEF'][fibers[i]]
        ymid = np.polyval(coef[::-1],x)   # reverse order
        ylo = np.int(np.min(np.floor(ymid-fwhm)))
        if ylo<0: ylo=0
        yhi = np.int(np.max(np.ceil(ymid+fwhm)))
        if yhi>(ny-1): yhi=ny-1
        num = yhi-ylo+1

        # Make a MASK based on the trace and FWHM
        yy = np.zeros(ncol,int).reshape(-1,1) + np.arange(ylo,yhi+1).reshape(1,-1)
        ymid2d = np.resize(np.repeat(ymid,num),(ncol,num))
        mask = np.array((yy >= np.floor(ymid2d-fwhm)) & (yy <= np.ceil(ymid2d+fwhm)),dtype=int)
        # Flux
        oflux[i,:] = np.sum( flux[xlo:xhi+1,ylo:yhi+1]*mask, axis=1)
        # Error
        #  add in quadrature
        oerr[i,:] = np.sqrt( np.sum( (err[xlo:xhi+1,ylo:yhi+1]**2)*mask, axis=1) )

    return SpecFrame(oflux,oerr,frame.head,frame.framenum)


# Calculate mean sky spectrum
def skysub(spec,plugmap):
    """This subtracts the median sky spectrum from all of the fiber spectra.
    
    Parameters
    ----------
    spec : SpecFrame object
         The SpecFrame object that constrains the 1D extracted spectra.
    plugmap : numpy structured array
            The plugmap information for each fiber including which fiber contains
            sky or stars.

    Returns
    -------
    spec : SpecFrame object
         The same SpecFrame object but now with the sky spectrum subtracted.

    Example
    -------

    .. code-block:: python

        spec2 = skysub(spec,plugmap)

    """

    # Find the object and sky fibers
    if 'STRUCT1' in plugmap.keys():
        fibermap = plugmap['STRUCT1']
        objtype = np.char.array(fibermap['targettype'].astype(str)).upper()
        sfibs, = np.where( (fibermap['fiberid']>=0) & (fibermap['assigned']==1) & (objtype=='SKY'))
        sfibid = fibermap['fiberid'][sfibs]
    else:
        fibermap = plugmap['FIBERMAP']
        # get objtype from the targeting information in sdssv_apogee_target0
        category = np.char.array(fibermap['category'].astype(str)).upper()
        sfibs, = np.where( (fibermap['fiberId']>=0) & (fibermap['spectrographId']==2) &
                           ((category=='SKY') | (category=='SKY_APOGEE') | (category=='SKY_BOSS')))
        sfibid = fibermap['fiberId'][sfibs]

    # We have sky fibers
    if len(sfibs)>0:
        skyindex = 300-sfibid
        # Calculate the median sky spectrum
        skyspec = np.median(spec.flux[skyindex,:],axis=0)
        # Median smooth to keep the sky lines (observers use these to check dither position)
        smskyspec = nanmedfilt(skyspec,200)
        # Subtract from all fibers
        nspec,ncol = spec.flux.shape
        # Make new object
        subspec = SpecFrame(spec.flux,spec.err,spec.head,spec.framenum)
        for i in range(nspec): subspec.flux[i,:] -= smskyspec
    # No sky fibers
    else:
        subspec = SpecFrame(spec.flux,spec.err,spec.head,spec.framenum)

    return subspec


# Calculate median S/N per fiber
def snrcat(spec,plugmap):
    """This function calculates the S/N for each fiber.
    
    Parameters
    ----------
    spec : SpecFrame object
         The SpecFrame object that constrains the 1D extracted spectra.
    plugmap : numpy structured array
            The plugmap information for each fiber including which fiber contains
            sky or stars.

    Returns
    -------
    cat : numpy structured array
         A catalog containing information on each object in the fibers and the
         median S/N.

    Example
    -------

    .. code-block:: python

        cat = snrcat(spec,plugmap)

    """

    nfibers,npix = spec.flux.shape
    dtype = np.dtype([('apogee_id',np.str,30),('catalogid',np.int),('ra',np.float64),('dec',np.float64),('hmag',np.float),
                      ('objtype',np.str,30),('fiberid',np.int),('fiberindex',np.int),('flux',np.float),('err',np.float),
                      ('snr',np.float)])
    cat = np.zeros(nfibers,dtype=dtype)

    # Load the spectral data
    cat['fiberindex'] = np.arange(nfibers)
    cat['flux'] = np.median(spec.flux,axis=1)
    cat['err'] = np.median(spec.err,axis=1)
    err = cat['err']
    bad = (err <= 0.0)
    err[bad] = 1.0
    cat['snr'] = cat['flux']/err

    # Load the plugging data
    if plugmap is not None:
        if 'STRUCT1' in plugmap.keys():
            fibermap = plugmap['STRUCT1']   # SDSS plates
            fibs, = np.where( (fibermap['fiberid']>=0) & (fibermap['holetype'].astype(str)=='APOGEE') & (fibermap['assigned']==1) )      
            pdb.set_trace()             
            fiberindex = 300-fibermap[fibs]['fiberid']
            cat['hmag'][fiberindex] = fibermap[fibs]['tmass_h']
            cat['objtype'][fiberindex] = fibermap[fibs]['targettype'].astype(str)
            cat['apogee_id'][fiberindex] = fibermap[fibs]['targetids']
            skyind = np.where(cat['objtype'] == 'SKY')
            if np.sum(skyind)>0:
                cat['objtype'][fiberindex[skyind]] = 'SKY'
            tellind = np.where(cat['objtype'] == 'STANDARD')
            if np.sum(skyind)>0:
                cat['objtype'][fiberindex[skyind]] = 'HOT_STD'
            cat['ra'][fiberindex] = fibermap[fibs]['ra']
            cat['dec'][fiberindex] = fibermap[fibs]['dec']
            cat['fiberid'][fiberindex] = fibermap[fibs]['fiberid']
        else:
            fibermap = plugmap['FIBERMAP']     # SDSS-V FPS
            fibs, = np.where( (fibermap['fiberId']>=0) & (fibermap['spectrographId']==2) )            
            fiberindex = 300-fibermap[fibs]['fiberId']
            cat['hmag'][fiberindex] = fibermap[fibs]['h_mag']
            cat['catalogid'][fiberindex] = fibermap[fibs]['catalogid']
            cat['objtype'][fiberindex] = 'OBJECT'   # default
            category = np.char.array(fibermap['category'].astype(str)).upper()
            skyind = (category[fibs]=='SKY') | (category[fibs]=='SKY_APOGEE') | (category[fibs]=='SKY_BOSS')
            #skyind = bmask.is_bit_set(fibermap['sdssv_apogee_target0'][fibs],0)==1    # SKY
            if np.sum(skyind)>0:
                cat['objtype'][fiberindex[skyind]] = 'SKY'
            tellind = (category[fibs]=='HOT_STD') | (category[fibs]=='STANDARD_APOGEE') 
            #tellind = bmask.is_bit_set(fibermap['sdssv_apogee_target0'][fibs],1)==1   # HOT_STD/telluric
            if np.sum(tellind)>0:
                cat['objtype'][fiberindex[tellind]] = 'HOT_STD'
            cat['ra'][fiberindex] = fibermap[fibs]['ra']
            cat['dec'][fiberindex] = fibermap[fibs]['dec']
            cat['fiberid'][fiberindex] = fibermap[fibs]['fiberId']
    cat = Table(cat)
    return cat


# Linear fit to log(S/N) vs. H
def snrhmag(cat,nreads,nframes,hfid=11.0):
    """Returns the S/N for the fiducial H magnitude.
apPlateSum-5815-56396.fits
    This fits a line to log(S/N) vs. Hmag for the stars and
    predicts what the S/N should be at the end of the exposure.
    
    Parameters
    ----------
    cat : numpy structured array
         A catalog containing information on each object in the fibers and the
         median S/N.
    nreads: int
         The number of reads used in the current processing.
    nframes: int
         The total number of reads in the entire exposure.
    hfid : float, optional
         The fiducial Hmag.  The default is H=11.0 for SDSS-V/MWM in the FPS era.

    Returns
    -------
    coefstr : numpy structured array
            An structure that contains all of the information computed:
            hmag_fid : The fiducial Hmag
            logsnr_hmag_coef : The linear coefficients of the fit of log(S/N) vs. Hmag
            snr_fid : The S/N at the fiducial Hmag using the linear fit.
            snr_predict : The predicted S/N at the fiducial Hmag at the end of the expsure.

    Example
    -------

    .. code-block:: python

        coefstr = snrhmag(cat,30,47)

    """

    # Linear fit to log(snr) vs. Hmag for ALL objects
    gdall, = np.where( (cat['objtype'] != 'SKY') & (cat['hmag'] > 4) & (cat['hmag'] < 20) & (cat['snr'] > 0))
    if len(gdall)>2:
        coefall = np.polyfit(cat[gdall]['hmag'],np.log10(cat[gdall]['snr']),1)
    else:
        coefall = np.zeros(2,float)+np.nan
    # Linear fit to log(S/N) vs. H for 10<H<11.5
    gd, = np.where( (cat['objtype'] != 'SKY') & (cat['hmag']>=10.0) & (cat['hmag']<=11.5) & (cat['snr'] > 0))
    if len(gd)>2:
        coef = np.polyfit(cat[gd]['hmag'],np.log10(cat[gd]['snr']),1)
    else:
        coef = np.zeros(2,float)+np.nan
    if len(gd)>2:
        snr_fid = 10**np.polyval(coef,hfid)
    elif len(gdall)>2:
        snr_fid = 10**np.polyval(coefall,hfid)
    else:
        snr_fid = np.mean(cat['snr'])

    # This is the full pipeline exposure-level S/N (altsn)
    #   copied from apogeereduce/qa/plotmag.pro
    #   alternative S/N as computed from median of all stars with H<12.2, scaled
    snstars, = np.where( (cat['objtype'] != 'SKY') & (cat['hmag'] > 4) & (cat['hmag'] < hfid) & (cat['snr'] > 0))
    if len(snstars)>0:
        scale = np.sqrt(10**(0.4*(cat['hmag'][snstars]-hfid)))
        snr_fid_scale = np.median(cat['snr'][snstars]*scale)
    else:
        snr_fid_scale = 0.0

    # Predicted S/N at end of exposure
    #  (S/N)^2 should scale with time
    snr_predict = np.sqrt( snr_fid**2*np.float(nframes)/np.float(nreads) )

    # Calculate zeropoints from known H band mags.
    # Use a static zeropoint to calculate sky brightness.
    fiberobj, = np.where((cat['objtype'] != 'SKY') & (cat['hmag'] > 4) & (cat['flux']>0))
    skyzero = 14.75 + (2.5 * np.log10(nreads))
    tmp = cat['hmag'][fiberobj] + (2.5 * np.log10(cat['flux'][fiberobj]))
    zero = np.nanmedian(tmp)
    zerorms = mad(cat['hmag'][fiberobj] + (2.5 * np.log10(cat['flux'][fiberobj])))
    zeronorm = zero - (2.5 * np.log10(nreads))

    dtype = np.dtype([('framenum',int),('read',int),('hmag_fid',np.float),('snr_fid',np.float),
                      ('snr_fid_scale',float),('logsnr_hmag_coef_all',(np.float,2)),('logsnr_hmag_coef',(np.float,2)),
                      ('snr_predict',np.float),('zero',float),('zeronorm',float)])
    coefstr = np.zeros(1,dtype=dtype)
    coefstr['hmag_fid'] = hfid
    coefstr['snr_fid'] = snr_fid
    coefstr['snr_fid_scale'] = snr_fid_scale
    coefstr['logsnr_hmag_coef_all'] = coefall
    coefstr['logsnr_hmag_coef'] = coef
    coefstr['snr_predict'] = snr_predict
    coefstr['zero'] = zero
    coefstr['zeronorm'] = zeronorm
    coefstr = Table(coefstr)

    return coefstr


# Run everything
def runquick(filename,hdulist=None,framenum=None,mjd=None,load=None,apred='daily',lastread=None,hfid=11.0,plugfile=None,ncol=51):
    """This runs all of the main steps of the quick reduction.
    
    Parameters
    ----------
    filename : string
             The full path of the apR file to process.
    framenum : string
             The APOGEE frame number of the exposure to process.
    mjd : string
             The modified Julian date of the apR file.
    lastread : int or string
             The number for the last read to use.
    hfid : float, optional
         The fiducial Hmag.  The default is 11.0.
    plugfile : string, optional
         The absolute path to the plugmap filename.
    ncol : int, optional
         Number of columns to extract.  Default is 51.

    Returns
    -------
    frame : Frame object
          The extracted 2D image and noise model.
    spec : SpecFrame object
         The extracted spectra and errors with the sky subtracted.
    cat : numpy structured array
        The catalog of information on each fiber including the S/N.
    coefstr : numpy structured array
            Information on the fit of log(S/N) vs. Hmag and the
            S/N at the fiducial Hmag.

    Example
    -------

    .. code-block:: python

        frame, spec, cat, coefstr = runquick('28200054','apo',10)

    """


    print('Running APQUICK on '+os.path.basename(filename)+' MJD='+mjd+' all reads ')

    fps = False
    if load.telescope == 'apo25m' and int(mjd) >= 59146: fps = True
    if load.telescope == 'lco25m' and int(mjd) >= 59809: fps = True

    head = fits.getheader(filename)#eframes[0].head.copy()   # header of first read


    # Setup directories and load the plugmap (FPS) or plateHolesSorted (plate) 
    plugmap = None
    rawdir = os.path.dirname(filename)+'/'
    if fps:
        redux_dir = os.environ.get('APOGEE_REDUX')+'/'+load.apred+'/'
        caldir = redux_dir+'cal/'+load.instrument+'/'
        plugdir = os.environ['SDSSCORE_DIR']+'/'+load.observatory.lower()+'/summary_files/'
        configid = head['configid']
        if configid is not None and str(configid) != '':
            configgrp = '{:0>4d}XX'.format(int(configid) // 100)
            plugfile = plugdir+'/'+configgrp+'/confSummary-'+str(configid)+'.par'
            print('Using configid from first read header: '+str(configid))
        else:
            print('No configID in header.')
            if plugfile is not None:
                print('Using input plugfile '+str(plugfile))
    else:
        redux_dir = '/uufs/chpc.utah.edu/common/home/sdss/apogeework/apogee/spectro/redux/current/'
        caldir = redux_dir+'cal/'
        plugdir = '/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/mapper/'+load.observatory+'/'+mjd+'/'
        phsdir = '/uufs/chpc.utah.edu/common/home/sdss/software/svn.sdss.org/data/sdss/platelist/trunk/plates/'
        plfolder = '{:0>4d}XX'.format(int(plateid) // 100)
        plstr = str(head['PLATEID']).zfill(6)
        plugfile = phsdir+plfolder+'/'+plstr+'/plateHolesSorted-'+plstr+'.par'
    psfdir = caldir+'psf/'
    detdir = caldir+'detector/'
    bpmdir = caldir+'bpm/'

    # Load plugmap/fibermap file
    if plugfile is not None:
        if os.path.exists(plugfile) is False:
            print(plugfile+' NOT FOUND')
            plugmap = None
            pdb.set_trace()
        else:
            print('Loading '+plugfile)
            plugmap = yanny.yanny(plugfile,np=True)

    pdb.set_trace()


    # Load the reads
    nfowler = 2
    #rawdir = os.path.dirname(filename)+'/'
    #framenum = int(os.path.basename(filename).split('-')[2].split('.')[0])
    bframes,eframes,nreads = loadframes(filename,hdulist,framenum,load=load,nfowler=nfowler,lastread=lastread)
    if bframes is None:
        print('Cannot run quicklook')
        return None,None,None,None

    # plugmap/configuration directory
    #plugmapfile = load.filename('confSummary', configid=int(iplate[i]))
    #if config['apogee_mountain'].get('plugmap_dir') is not None:
    #    plugdir = config['apogee_mountain'].get('plugmap_dir')
    #else:
    #    plugdir = os.environ['SDSSCORE_DIR']+observatory.lower()+'/summary_files/'
        #plugdir = '/data/apogee/plugmaps/'

    nframes = np.int(head['NFRAMES'])
    exptype = head['EXPTYPE']
    plateid = str(head['PLATEID'])
    if exptype is None: exptype=''
    # Do Fowler/CDS collapse
    im = fowler(bframes,eframes)
    #im = fits.getdata(filename)
    try: nreads = head['NREADS']
    except: nreads = head['NFRAMES']
    # Get rdnoise/gain from apDetector file
    detfiles = glob(detdir+'/'+load.prefix+'Detector-b-????????.fits')
    detfiles = np.sort(detfiles)
    print('Using '+detfiles[-1])
    rdnoiseim = fits.getdata(detfiles[-1],1)
    rdnoise = np.median(rdnoiseim)
    gainim = fits.getdata(detfiles[-1],2)
    gain = np.median(gainim)
    # Generate the noise image
    err = noisemodel(im,nreads,rdnoise,gain)
    frame = Frame("",head,im,framenum,0)
    frame.err = err
    # Add some important values to the header
    frame.head['FRAMENUM'] = framenum
    # Use bad pixel mask
    bpmfiles = glob(bpmdir+'/'+load.prefix+'BPM-b-????????.fits')
    bpmfiles = list(np.sort(bpmfiles))
    print('Using '+bpmfiles[-1])
    bpm = fits.getdata(bpmfiles[-1],0)
    frame = bpmfix(frame,bpm)
    # Load the trace information
    psffiles = np.sort(glob(psfdir+'/'+load.prefix+'PSF-b-*.fits'))
    #psffiles = np.sort(glob(psfdir+'/apPSF-b-????????.fits'))
    print('Using '+psffiles[-1])
    tracestr = Table.read(psffiles[-1],1)
    # Boxcar extract the fibers
    frame.head['DETFILE'] = detfiles[-1]
    frame.head['PSFFILE'] = psffiles[-1]
    midcol = 1024
    if ncol==2048:
        xlo = 0
        xhi = 2047
    else:
        half_ncol = ncol//2
        xlo = np.maximum(midcol-half_ncol,0)
        xhi = np.minimum(xlo+ncol-1,2047)
    pdb.set_trace()
    spec = boxextract(frame,tracestr,xlo=xlo,xhi=xhi)

    # Add some important configuration values to the header
    if plugmap is not None:
        try: frame.head['CONFIGID'] = plugmap.get('configuration_id')
        except: frame.head['CONFIGID'] = ''
        frame.head['DESIGNID'] = plugmap.get('design_id')
        frame.head['FIELDID'] = plugmap.get('field_id')
        frame.head['CONFIGFL'] = plugfile
    else:
        frame.head['CONFIGID'] = ''
        frame.head['DESIGNID'] = ''
        frame.head['FIELDID'] = ''
        frame.head['CONFIGFL'] = ''
    # Subtract the sky
    if exptype == 'OBJECT' and plugmap is not None:
        subspec = skysub(spec,plugmap)
    else:
        subspec = spec
    # Create the S/N catalog
    cat = snrcat(subspec,plugmap)
    print('Mean S/N = %5.2f' % np.mean(cat['snr']))
    if exptype != 'OBJECT':
        cat['apogee_id'] = 'N/A'
        cat['objtype'] = 'N/A'
        blank = ['ra','dec','hmag']
        for b in blank: cat[b] = 999999.
    # Fit line and get fiducial S/N
    coefstr = snrhmag(cat,nreads,nframes,hfid=hfid)
    # add some relevant columns
    coefstr['framenum'] = framenum
    coefstr['read'] = nreads
    coefstr['exptype'] = exptype
    coefstr['ditherpos'] = head.get('DITHPIX')
    if exptype=='OBJECT':
        # Print out the S/N values
        print('S/N (H=%5.2f) = %5.2f (%3d reads)' % (hfid,coefstr['snr_fid'],nreads))
        print('S/N (H=%5.2f) = %5.2f (prediction for %3d reads)' % (hfid,coefstr['snr_predict'],nframes))
    else:
        # set values to nan that don't make sense for non-object exposures
        coefstr['hmag_fid'] = np.nan
        coefstr['snr_fid'] = np.mean(cat['snr'])
        coefstr['snr_fid_scale'] = np.nan
        coefstr['logsnr_hmag_coef_all'] = np.nan
        coefstr['logsnr_hmag_coef'] = np.nan
        coefstr['zero'] = np.nan
        coefstr['zeronorm'] = np.nan
        print('Mean S/N = %5.2f' % coefstr['snr_fid'])

    return frame, subspec, cat, coefstr


# Write out results
def writeresults(outfile, frame, sframe, cat, coefstr, compress=False):
    """This writes out the results of the quick reduction to an output file.
    
    Parameters
    ----------
    outfile : str
          The output filename to write the results to.
    frame : Frame object
          The extracted 2D image and noise model.
    sframe : SpecFrame object
         The extracted spectra and errors with the sky subtracted.
    cat : numpy structured array
        The catalog of information on each fiber including the S/N.
    coefstr : numpy structured array
            Information on the fit of log(S/N) vs. Hmag and the
            S/N at the fiducial Hmag.
    compress : boolean, optional
         Gzip compress the final FITS file.  This reduces the file size
         by a factor of ~3x.  Default is False.
             
    Returns
    -------
    The function doesn't return anything but writes the results to an
    output file.

    Example
    -------

    .. code-block:: python

        writeresults('apq-28200054.fits', frame, spec, cat, coefstr)

    """

    print('Final catalog = '+outfile)

    if os.path.exists(outfile): os.remove(outfile)

    # HDU0: header only
    head = frame.head
    #head.add_history('apogee_mountain version: '+str(__version__))
    head.add_history("APQUICK results for "+str(head['FRAMENUM']))
    head.add_history("HDU0: Header only")
    head.add_history("HDU1: coefstr, S/N for fiducial Hmag")
    head.add_history("HDU2: cat, catalog of S/N values")
    head.add_history("HDU3: spec/err, numpy structured array")
    head.add_history("HDU4: image/err, numpy structured array")
    fits.writeto(outfile,None,head,overwrite=True)
    hdulist = fits.open(outfile)
    
    # HDU1: Coef structure
    if coefstr is not None:
          hdu = fits.table_to_hdu(Table(coefstr))
          hdu.header.add_history("APQUICK results for "+str(head['FRAMENUM']))
          hdu.header.add_history("Final fiducial S/N values")
    else:
          hdu = fits.PrimaryHDU(0)
          hdu.header.add_history("No S/N results for non-OBJECT frame "+str(head['FRAMENUM']))
    hdulist.append(hdu)
    # HDU2: Catalog
    hdu = fits.table_to_hdu(cat)
    hdu.header.add_history("APQUICK results for "+str(head['FRAMENUM']))
    hdu.header.add_history("Catalog of S/N values and other fiber information")
    hdulist.append(hdu)
    # HDU3: Spectra
    dtype = np.dtype([('flux',(np.float32,sframe.flux.shape)),('err',(np.float32,sframe.err.shape))])
    spec = np.zeros(1,dtype=dtype)
    spec['flux'] = sframe.flux
    spec['err'] = sframe.err
    spec = Table(spec)
    hdu = fits.table_to_hdu(spec)
    hdu.header.add_history("APQUICK results for "+str(head['FRAMENUM']))
    hdu.header.add_history("Extracted spectra and errors (ADU)")
    hdulist.append(hdu)
    # HDU4: Image
    dtype = np.dtype([('im',(np.float32,frame.im.shape)),('err',(np.float32,frame.err.shape))])
    image = np.zeros(1,dtype=dtype)
    image['im'] = frame.im
    image['err'] = frame.err
    image = Table(image)
    hdu = fits.table_to_hdu(image)
    hdulist.append(hdu)
    hdu.header.add_history("APQUICK results for "+str(head['FRAMENUM']))
    hdu.header.add_history("Collapsed 2D image and noise model (ADU)")
    hdulist.writeto(outfile,overwrite=True)
    hdulist.close() 
    
    # gzip compress?
    if compress:
        ret = subprocess.call(['gzip','-f',outfile])    # compress final catalog
