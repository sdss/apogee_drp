import os
import time
import numpy as np
from astropy.io import fits
from ..utils import apload,lock
from . import process

def mkdark(ims, cmjd=None, step=None, psfid=None, clobber=False, unlock=False):
    """
    Makes APOGEE superdark calibration product.

    Parameters
    ----------
    ims : list
       List of image numbers to include in superdark
    cmjd : str, optional, obsolete
       Gives MJD directory name if not encoded in file number
    step : int, optional, obsolete
       Process every step image in UTR.
    psfid : int, optional
       EPSF id to use to try to subtract off thermal traces.
    clobber : bool, optional
       Overwrite any existing files.  Default is False.
    unlock : bool, optional
       Delete lock file and start fresh.  Default is False.

    Returns
    -------
    A set of apDark-[abc]-ID8.fits files.

    Example
    -------

    mkdark,ims,cmjd=cmjd,step=step,psfid=psfid

    By J. Holtzman, 2011??
    Updates, added doc strings, and cleanup by D. Nidever, Sep 2020
    Translated to Python, D. Nidever 2024
    """

    images = np.atleast_1d(ims)
    i1 = images[0]
    nframes = len(images)
    chips = ['a', 'b', 'c']
    
    #dirs = getdir()
    #caldir = dirs.caldir

    load = apload.ApLoad(apred=apred,telescope=telescope)
    adarkfile = load.filename('Dark',num=i1, chips=True)
    adarkfile = adarkfile.replace('Dark-','Dark-a-')
    darkdir = os.path.dirname(filename)
    darkfile = darkdir + load.prefix + 'Dark-{:08d}.tab'.format(i1)

    # Does the file already exist?
    # Check all three chip files and the .tab file
    sdarkid = '{:08d}'.format(ims[0])

    allfiles = [darkdir+load.prefix + 'Dark-{:s}-{:s}.fits'.format(chip,sdarkid) for chip in chips]
    allfiles.append(darkdir+load.prefix+'Dark-'+sdarkid+'.tab')
    if all([os.path.exists(f) for f in allfiles]) and not clobber:
        print('Dark file:', darkfile, 'already made')
        return
    # Delete any existing files to start fresh
    for f in allfiles:
        if os.path.exists(f): os.remove(f)

    # Is another process already creating the file?
    lock.lock(darkfile, waittime=10, unlock=unlock)

    # Initialize summary structure
    dt = [('num',int), ('nframes',int), ('nreads',int), ('nsat',int), ('nhot',int),
          ('nhotneigh',int), ('nbad',int), ('medrate',float), ('psfid',int), ('nneg',int)]
    darklog = np.zeros(3,dtype=np.dtype(dt))
    darklog['num'] = i1

    if step is None:
        step = 0

    # Loop over the chips
    for ichip in range(3):
        chip = chips[ichip]
        time0 = time.time()
        # Image loop
        for j,im in enumerate(ims):
            if cmjd is None:
                cm = load.cmjd(im)
            else:
                cm = cmjd
            print('{:d}/{:d} {:s} {:s}'.format(j,len(ims),chip,im))

            # Process (bias-only) each individual frame
            d = ap3d.ap3dproc(im)
            #d = process.process(cm, im, chip, head, r, step=step, nofs=True, nofix=True, nocr=True)
            print('Done process')
            if d.shape[0] != 2048:
                raise Exception('Not 2048')

            mask = np.zeros(d.shape, dtype=np.byte)

            # Construct cube of reads minus second read
            if j == 0:
                head0 = head.copy()

            sz = np.shape(r)

            if j == 0:
                if ichip == 0:
                    red = np.zeros((2048, 2048, sz[2], nframes), dtype=float)
                else:
                    red *= 0.0

            red[:,:,:,j] = r
            del r
            for iread in range(sz[2] - 1, 0, -1):
                red[:, :, iread, ii] -= red[:, :, 1, ii]

        # Median them all
        print('Median...')
        dark = np.median(red, axis=3)

        # Option to remove any trace of spectral traces
        if psfid:
            darklog[ichip]['psfid'] = psfid
            print('Reading epsf')
            epsffile = load.filename('EPSF', num=psfid, chip=chip)
            thdu = fits.open(epsffile)
            head = thdu[0].header
            ntrace = head['NTRACE']
            img = [None] * ntrace

            for i in range(ntrace):
                ptmp = thdu[i+1].data
                #ptmp = mrdfits(epsffile, i + 1, silent=True)
                img[i] = ptmp['img']
                p = {'lo': ptmp.lo, 'hi': ptmp.hi, 'img': img[i]}
                if i == 0:
                    psf = [p] * ntrace
                psf[i] = p

            nread = sz[2]

            for iread in np.arange(1, nread):
                var = dark[:, :, iread]
                # Want to subtract off mean background dark level before fitting traces
                # Iterate once for this
                back = np.median(dark[:, :, iread], 10)

                niter = 2

                for iter in range(niter):
                    print(iread, iter)
                    d = dark[:, :, iread] - back
                    spec = extract(d, ntrace, psf, var)
                    sspec = zap(spec, [200, 1])
                    d *= 0

                    for k in range(ntrace):
                        p1 = psf[k]
                        lo, hi = p1['lo'], p1['hi']
                        img = p1['img']
                        r = np.arange(lo, hi + 1) + 1
                        sub = sspec[:, k] * r
                        bad = np.where(sub < 0)
                        sub[bad] = 0
                        d[:, lo:hi] += sub * img

                    if iter < niter - 1:
                        back = np.median(dark[:, :, iread] - d, 10)

                dark[:, :, iread] -= d

        # Flag "hot" pixels in mask image
        nread = sz[2]
        rate = (dark[:, :, nread - 1] - dark[:, :, 1]) / (nread - 2)

        # Create mask array
        # NaN is bad!
        bad = (np.isnan(rate))
        nsat = np.sum(bad)
        mask[bad] = mask[bad] | 1

        # Flux accumulating very fast is bad!
        maxrate = 10.0
        hot = (rate > maxrate)
        nhot = np.sum(hot)
        mask[hot] = mask[hot] | 2

        # Flag adjacent pixels to hot pixels as bad at 1/4 the maximum rate
        n = [-1, 1, -2048, 2048]
        nhotneigh = 0

        for in_ in range(4):
            # Only consider neighbors on the chip!
            neigh = hot + n[in_]
            on = np.where((neigh >= 0) & (neigh < 2048 * 2048))
            nlow = np.where(rate[neigh[on]] > maxrate / 4.0)
            if len(nlow) > 0:
                mask[neigh[on[nlow]]] = mask[neigh[on[nlow]]] | 4
            nhotneigh += len(hot)
            # Same for bad
            neigh = bad + n[in_]
            on = np.where((neigh >= 0) & (neigh < 2048 * 2048))
            nlow = np.where(rate[neigh[on]] > maxrate / 4.0)
            if len(nlow) > 0:
                mask[neigh[on[nlow]]] = mask[neigh[on[nlow]]] | 4
            nhotneigh += len(hot)
            
        print('Creating chi2 array ....')
        chi2 = np.zeros(2048 * 2048 * nread, dtype=float)
        n = np.zeros(2048 * 2048 * nread, dtype=int)
        dark = dark.reshape((2048 * 2048 * nread,))
        for ii in range(nframes):
            tmp = red[:, :, :, ii].reshape((2048 * 2048 * nread,))
            good = np.where(np.isfinite(tmp))
            chi2[good] += ((tmp[good] - dark[good]) ** 2) / apvariance(dark[good], 1)
            n[good] += 1

        chi2 /= n
        dark = dark.reshape((2048, 2048, nread))
        chi2 = chi2.reshape((2048, 2048, nread))

        # Set nans to 0 before writing
        bad = (np.isnan(dark))
        dark[bad] = 0
        medrate = np.median(rate)

        # Median filter along reads dimension
        for i in range(2048):
            slc = dark[i, :, :]
            dark[i, :, :] = medfilt2d(slc, 7, dim=2)

        # Set negative pixels to zero
        neg = (dark < -10)
        dark[neg] = 0

        # Write them out
        if step:
            outfile = load.prefix + 'Dark{:d}-{:s}-{:08d}'.format(step,chip,i1)
        else:
            outfile = load.prefix + 'Dark-{:s}-{:08d}'.format(chip,i1)

        leadstr = 'APMKDARK: '
        head['HISTORY'] = leadstr+time.asctime()
        import socket
        head['HISTORY'] = leadstr+getpass.getuser()+' on '+socket.gethostname()
        import platform
        head['HISTORY'] = leadstr+'Python '+pyvers+' '+platform.system()+' '+platform.release()+' '+platform.architecture()[0]
        # add reduction pipeline version to the header
        head['HISTORY'] = leadstr+' APOGEE Reduction Pipeline Version: '+load.apred

        hdulist = fits.HDUList()
        hdulist.append(fits.PrimaryHDU(header=head0))
        hdulist.append(fits.PrimaryHDU(dark))
        hdulist[1].header['EXTNAME'] = 'DARK'
        hdulist.append(fits.ImageHDU(chi2))
        hdulist[2].header['EXTNAME'] = 'CHI-SQUARED'
        hdulist.append(fits.ImageHDU(mask))
        hdulist[3].header['EXTNAME'] = 'MASK'
        outfile = os.path.join(darkdir, file + '.fits')
        hdulist.writeto(outfile,overwrite=True)
        
        #MWRFITS(0, os.path.join(darkdir, file + '.fits'), head0, create=True)
        #MKHDR(head1, dark)
        #sxaddpar(head1, 'EXTNAME', 'DARK')
        #MWRFITS(dark, os.path.join(darkdir, file + '.fits'), head1)
        #MKHDR(head2, chi2)
        #sxaddpar(head2, 'EXTNAME', 'CHI-SQUARED')
        #MWRFITS(chi2, os.path.join(darkdir, file + '.fits'), head2)
        #MKHDR(head3, mask)
        #sxaddpar(head3, 'EXTNAME', 'MASK')
        #MWRFITS(mask, os.path.join(darkdir, file + '.fits'), head3)

        # Make some plots/images
        if not os.path.exists(os.path.join(darkdir, 'plots')):
            os.makedirs(os.path.join(darkdir, 'plots'))

        darkplot(dark, mask, os.path.join(darkdir, 'plots', file), hard=True)

        # Summary data table
        darklog[ichip]['num'] = i1
        darklog[ichip]['nframes'] = nframes
        darklog[ichip]['nreads'] = nread
        darklog[ichip]['nsat'] = nsat
        darklog[ichip]['nhot'] = nhot
        darklog[ichip]['nhotneigh'] = nhotneigh
        darklog[ichip]['nbad'] = nbad
        darklog[ichip]['medrate'] = medrate
        darklog[ichip]['nneg'] = nneg

        # Save the rate file
        outfile = load.prefix + 'DarkRate-{:s}-{:08d}'.format(chip,i1)
        fits.writeto(darkdir+outfile+'.fits',rate,overwrite=True)
        #MWRFITS(rate, darkdir+outfile+'.fits', create=True)

        dark = 0
        time = time.time()
        print('Done', chip, time - time0)

    del red

    # Write the summary log information
    outfile = prefix + 'Dark-{:08d}'.format(i1)
    fits.writeto(darkdir+outfile + '.tab',darklog,overwriteTrue)
    #MWRFITS(darklog, darkdir+outfile + '.tab', create=True)

    # Remove lock file
    lock.lock(darkfile, clear=True)

    # Compile summary web page
    darkhtml(darkdir)
