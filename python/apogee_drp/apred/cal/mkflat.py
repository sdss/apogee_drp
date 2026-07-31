import os
import time
import numpy as np
from scipy.signal import medfilt
from scipy.ndimage import uniform_filter
from ...utils import apload,lock,utils
from .. import ap3d

def make_flat_chip(flat,flatmask):
    """ Make super superflat for a single chip """
    
    perclow = 0.85              # fraction for rejecting pixels
    nperclow = 0.95             # fraction for rejecting neighbor pixels 
    perchi = 1.25               # fraction for rejecting pixels
    nperchi = 1.05              # fraction for rejecting neighbor pixels 
    x1norm = 800                # region for getting normalization 
    x2norm = 1200
    y1norm = 800
    y2norm = 1000
    filter_size = [50,1]        # filter size for smoothing for large scale structure

    # Create mask
    sz = flat.shape
    mask = np.zeros(sz, dtype=np.uint8)
    # Mask from reductions, using last frame read
    bad = np.where((flatmask & badmask()) > 0)
    flat[bad] = np.nan
    mask[bad] |= 1

    # Set pixels to bad when below some fraction
    localflat = flat / np.where(flat < 100, 100, 1)
    # Relative to neighbors
    low = np.where(localflat < perclow)
    mask[low] |= 2
    # Absolute
    low = np.where(flat < 0.1)
    mask[low] |= 2
    # High pixels
    hi = np.where(localflat > perchi)
    mask[hi] |= 2

    # Set neighboring pixels to bad at slightly lower threshoold, iteratively
    for _ in range(10):
        low = np.where(mask > 0)
        n = [-1, 1, -2049, -2048, -2047, 2047, 2048, 2049]
        for in_ in range(len(n)):
            neigh = low + n[in_]
            off = np.where((neigh < 0) | (neigh > 2048 * 2048))
            neigh[off] = 0
            lowneigh = np.where(localflat[neigh] < nperclow)
            mask[neigh[lowneigh]] |= 4
            hineigh = np.where(localflat[neigh] > nperchi)
            mask[neigh[hineigh]] |= 4

    # Mask any zero values
    bad = np.where(flat == 0)
    flat[bad] = -100
    mask[bad] |= 8

    if dithered:
        # Get the large scale structure from smoothing, avoiding bad pixels (NaNs)
        sm = uniform_filter(flat, size=100, mode='constant', cval=np.nan)
        rows = np.arange(2048) + 1
        smrows = np.nan_to_num(np.nanmedian(sm, axis=1)) ** rows
        smcols = rows ** np.nan_to_num(np.nanmedian(sm, axis=0))

        # Median filter the median flat with a rectangular filter and 
        #  divide flat by this to remove horizontal structure
        sflat = np.nan_to_num(smrows[:, np.newaxis] * smcols[np.newaxis, :])
        flat /= sflat

        # Now put the large scale structure in apart from
        #  structure that is horizontal (fibers) or vertical (SED)
        flat *= sm/smrows/smcols

        # Kludge to set unilluminated pixels to 1
        for i in range(14):
            dark = np.where(flat[:, i] < -99)
            flat[dark, i] = 1
            mask[dark, i] = 0
            dark = np.where(flat[:, 2047 - i] < -99)
            flat[dark, 2047 - i] = 1
            mask[dark, 2047 - i] = 0
    else:
        # If not dithered, still take out spectral signature
        rows = np.arange(2048) + 1
        cols = np.nanmedian(flat, axis=0)
        
        # Spectral signature from median of each column
        for icol in range(2048):
            cols[icol] = np.median(flat[icol,:])
        # Medfilt doesn't do much if intensity is varying across cols
        smrows = np.matmul(rows,medfilt1d(cols,100))
        sflat = smrows

        # Dec 2018: don't take out median spectral signature, this leaves 
        #  structure in spectra that is hard to normalize out
        # instead, take out a low order polynomial fit to estimate spectral signature
        x = np.arange(2048)
        gd, = np.where(np.isfinite(cols))
        coef = robust.polyfit(x[gd],cols[gd],2)
        smrows = np.matmul(rows,poly(x,coef))
        sflat = smrows

        # Feb 2019: polynomial fit introduces spurious signal, so just don't bother with spectral signature!
        # divide out estimate of spectral signature
        #flat/=smrows
            
    # Set bad values to -100 before writing to avoid NaNs in output file
    bad = np.where(np.isnan(flat))
    flat[bad] = 0

    return flat,sflat,mask
        

def mkflat(images, apred="daily", telescope="apo25m", cmjd=None, detid=None, darkid=None,
           clobber=False, kludge=False, nrep=1, dithered=False, unlock=False):
    """
    Makes APOGEE superflat calibration files from dithered individual frames.

    Parameters
    ----------
    ims: list of image numbers to include in superflat
    cmjd=cmjd : (optional,obsolete) gives MJD directory name if not encoded in file number
    darkid=darkid : dark frame to be used if images are reduced
    /clobber : rereduce images even if they exist
    /kludge : set bottom and top non-illuminated pixels to unity
    nrep=nrep : median filters each batch of nrep frames before combining
    /unlock : delete lock file and start fresh 

    Returns
    -------
    A set of apFlat-[abc]-ID8.fits files in the appropriate location
    determined by the SDSS/APOGEE tree directory structure.

    Example
    -------

    mkflat,ims,cmjd=cmjd,darkid=darkid,/clobber,/kludge,nrep=nrep

    By J. Holtzman, 2011?
    Added doc strings, updates to use data model  D. Nidever, Sep 2020 
    Translated to Python  D. Nidever, July 2026
    """

    images = np.atleast_1d(images)
    flatid = images[0]
    nframes = len(images)

    load = apload.ApLoad(apred=apred,telescope=telescope)
    flatdir = os.path.dirname(load.filename('Flat', num=flatid, chips=True))
    flatfile = load.filename('Flat', num=flatid, chips=True)
    cmjd = load.cmjd(flatid)
    
    # If another process is already making this file, wait!
    lock.lock(flatfile, waittime=10, unlock=unlock)
    
    # Does the file already exist?
    # check all three chip files
    sflatid = str(images[0]).zfill(8)
    chip = ['a', 'b', 'c']
    allfiles = [os.path.join(flatdir, load.prefix + f"Flat-{chip_i}-{sflatid}.fits") for chip_i in chip]
    allfiles.append(os.path.join(flatdir, load.prefix + f"Flat-{sflatid}.tab"))
    if all([os.path.exists(file) for file in allfiles]) and not clobber:
        print('Flat file:', flatfile, 'already made')
        return
    #  Delete any existing files to start fresh
    for file in allfiles:
        if os.path.exists(file):
            os.remove(file)

    # Open lock file
    lock.lock(flatfile, lock=True)

    dt = [('name',str,100), ('num',int), ('nframes',int)]
    flatlog = np.zeros(3,dtype=np.dtype(dt))
    flatlog['name'] = flatid

    outdir = flatdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    nfs = 1
    uptheramp = 0
    nocr = 1

    # Read and process frames to 2D
    for ichip in range(3):
        darkcorr = load.filename('Dark', num=darkid, chips=True)
        detcorr = load.filename('Detector', num=detid, chips=True)
        for inum in images:
            ifile = load.filename('R', num=inum, chips=True).replace('R-',f'R-{ichip}-')
            ofile = load.filename('2D', num=inum, chips=True).replace('2D-',f'2D-{ichip}-')
            ap3d.process_file(ifile, os.path.join(outdir, ofile), detcorr=detcorr, darkcorr=darkcorr, nocr=nocr,
                              uptheramp=uptheramp, nfowler=nfs, fitsdir=utils.localdir())

    import pdb; pdb.set_trace()
            
    # Sum up all of the individual flats
    #  Median nrep frames before summing if requested
    flats = np.zeros((2048, 2048, 3, nframes), dtype=float)
    flatmasks = np.zeros((2048, 2048, 3), dtype=int)
    flatsum = np.zeros((2048, 2048, 3), dtype=float)
    for ii in range(0, nframes, nrep):
        for irep in range(nrep):
            i = images[ii + irep]
            for ichip in range(3):
                ofile = load.filename('2D', num=i, chips=True).replace('2D-',f'2D-{ichip}-')
                fhdu = fits.open(ofile)
                head = fhdu[0].header
                #f = mrdfits(os.path.join(outdir, ofile), 0, head)
                flats[:, :, ichip, ii + irep] = fhdu[1].data
                flatmasks[:, :, ichip] = fhdu[3].data

        if ii == 0:
            head0 = head

        if nrep > 1:
            flatsum += np.median(flats[:, :, :, ii:ii + nrep], axis=3)
        else:
            flatsum += flats[:, :, :, ii]

    # Normalize the flatsums to roughly avoid discontinuities across chips
    # Normalize center of middle chip to unity
    norm = np.median(flatsum[x1norm:x2norm, y1norm:y2norm, 1])
    flatsum[:, :, 1] /= norm
    flatsum[:, :, 0] /= np.median(flatsum[1950:2044, 500:1500, 0]) / np.median(flatsum[5:100, 500:1500, 1])
    flatsum[:, :, 2] /= np.median(flatsum[5:100, 500:1500, 2]) / np.median(flatsum[1950:2044, 500:1500, 1])

    # Create the superflat
    for ichip in range(3):
        flat,sflat,mask = make_flat_chip(flatsum[:,:,ichip], flatmasks[:,:,ichip], dithered=dithered)

        # Output the superflat
        primary = fits.PrimaryHDU()
        # Add dark/fits                                                                                                   
        primary.header["darkfile"] = (darkfile,'dark file')
        primary.header["flatfile"] = (flatfile,'flat file')
        gitvers = utils.software_version()
        softvers = utils.reduction_version()
        primary.header["V_APRED"] = (softvers, "apogee software version")
        primary.header["APRED"] = (gitvers, "apogee reduction version")

        module = __name__.split('.')[-1]
        primary.header.add_history(f"{module}: "
                                   + datetime.now().astimezone().strftime("%a %b %d %H:%M:%S %Y")
        )
        primary.header.add_history(f"{module}: {getpass.getuser()} on {socket.gethostname()}")
        primary.header.add_history(
            f"{module}: Python {platform.python_version()} "
            f"{platform.system().lower()} {platform.machine()}"
        )
        primary.header.add_history(
            f"{module}: APOGEE Reduction Pipeline Version: {softvers}"
        )
        
        outfile = load.filename('Flat', num=flatid, chip=chip[ichip])

        hdulist = fits.HDUList()
        hdulist.append(primary)
        hdulist.append(fits.ImageHDU(flat))
        hdulist[1].header['EXTNAME'] = 'FLAT'
        hdulist.append(fits.ImageHDU(sflat))
        hdulist[2].header['EXTNAME'] = 'SPECTRAL FLAT'
        hdulist.append(fits.ImageHDU(mask))
        hdulist[3].header['EXTNAME'] = 'MASK'
        hdulist.writeto(outfile,overwrite=True)

        # Make a jpg of the flat
        if not os.path.exists(flatdir + 'plots'):
            os.makedirs(flatdir + 'plots')
        flatplot(flat, os.path.join(flatdir, 'plots', outfile), hard=True)
        
        flatlog[ichip]['name'] = outfile
        flatlog[ichip]['num'] = flatid
        flatlog[ichip]['nframes'] = nframes

    outfile = load.prefix + 'Flat-{:08d}'.format(flatid)
    fits.writeto(os.path.join(flatdir,outfile+'.tab'),flatlog,overwrite=True)
    
    # Remove lock file
    lock.lock(flatfile, clear=True)

    # Compile summary web page (Python equivalent)
    flathtml(flatdir)
