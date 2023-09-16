import os
import time
import numpy as np
#from ..utils import apload

def mkdark(ims, cmjd=None, step=0, psfid=None, clobber=False, unlock=False):
    """
    Makes APOGEE superdark calibration product.

    Parameters
    ----------
    ims: list of image numbers to include in superdark
    cmjd=cmjd : (optional,obsolete) gives MJD directory name if not encoded in file number
    step=step : (optional,obsolete) process every step image in UTR
    psfid=psfid : (optional) EPSF id to use to try to subtract off thermal traces
    /unlock : delete lock file and start fresh 

    Returns
    -------
    A set of apDark-[abc]-ID8.fits files.

    Example
    -------

    mkdark,ims,cmjd=cmjd,step=step,psfid=psfid

    By J. Holtzman, 2011??
    Updates, added doc strings, and cleanup by D. Nidever, Sep 2020
    """
    
    i1 = ims[0]
    nframes = len(ims)

    dirs = getdir()
    caldir = dirs.caldir

    adarkfile = apogee_filename('Dark', num=i1, chip='a')
    darkdir = file_dirname(adarkfile)
    prefix = strmid(file_basename(adarkfile), 0, 2)
    darkfile = os.path.join(darkdir, load.prefix + 'Dark-{:08d}.tab'.format(i1))
    lockfile = darkfile + '.lock'
    chips = ['a', 'b', 'c']

    # Does the file already exist?
    # Check all three chip files and the .tab file
    sdarkid = str(ims[0]).zfill(8)
    allfiles = [os.path.join(darkdir, dirs.prefix + f'Dark-{chip}-{sdarkid}.fits') for chip in chips]
    allfiles.append(os.path.join(darkdir, dirs.prefix + f'Dark-{sdarkid}.tab'))
    if all([os.path.exists(file) for file in allfiles]) and not clobber:
        print('Dark file:', darkfile, 'already made')
        return
    # Delete any existing files to start fresh
    for file in allfiles:
        if os.path.exists(file):
            os.remove(file)

    # Is another process already creating the file?
    aplock(darkfile, waittime=10, unlock=unlock)

    # Initialize summary structure
    dt = [('num',int), ('nframes',int), ('nreads',int), ('nsat',int), ('nhot',int),
          ('nhotneigh',int), ('nbad',int), ('medrate',float), ('psfid',int), ('nneg',int)]
    darklog = np.zeros(3,dtype=np.dtype(dt))
    darklog['num'] = i1

    if not step:
        step = 0

    # Loop over the chips
    for ichip in range(3):
        chip = chips[ichip]

        time0 = systime(seconds=True)
        ii = 0

        for jj in range(nframes):
            i = ims[jj]

            if not cmjd:
                cm = getcmjd(i)
            else:
                cm = cmjd

            print(f'{jj}/{nframes} {chip} {i}')

            # Process (bias-only) each individual frame
            d = process(cm, i, chip, head, r, step=step, nofs=True, nofix=True, nocr=True)
            print('Done process')
            sz = np.shape(d)

            if sz[0] != 2048:
                raise Exception('Not 2048')

            mask = np.zeros(sz[0], sz[1], dtype=np.byte)

            # Construct cube of reads minus second read
            if jj == 0:
                head0 = head

            sz = np.shape(r)

            if jj == 0:
                if ichip == 0:
                    red = np.zeros((2048, 2048, sz[2], nframes), dtype=float)
                else:
                    red *= 0.0

            red[:, :, :, ii] = r
            del r
            for iread in range(sz[2] - 1, 0, -1):
                red[:, :, iread, ii] -= red[:, :, 1, ii]

            ii += 1


        # Median them all
        print('Median...')
        dark = np.median(red, axis=3)

        # Option to remove any trace of spectral traces
        if psfid:
            darklog[ichip]['psfid'] = psfid
            print('Reading epsf')
            epsffile = apogee_filename('EPSF', psfid, chip=chip)
            tmp = mrdfits(epsffile, 0, head)
            ntrace = sxpar(head, 'NTRACE')
            img = [None] * ntrace

            for i in range(ntrace):
                ptmp = mrdfits(epsffile, i + 1, silent=True)
                img[i] = ptmp.img
                p = {'lo': ptmp.lo, 'hi': ptmp.hi, 'img': img[i]}

                if i == 0:
                    psf = [p] * ntrace

                psf[i] = p

            nread = sz[2]

            for iread in range(1, nread):
                var = dark[:, :, iread]
                # Want to subtract off mean background dark level before fitting traces
                # Iterate once for this
                back = median(dark[:, :, iread], 10)

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
                        back = median(dark[:, :, iread] - d, 10)

                dark[:, :, iread] -= d

        # Flag "hot" pixels in mask image
        nread = sz[2]
        rate = (dark[:, :, nread - 1] - dark[:, :, 1]) / (nread - 2)

        # Create mask array
        # NaN is bad!
        bad = np.where(np.isnan(rate))
        nsat = len(bad)
        mask[bad] = mask[bad] | 1

        # Flux accumulating very fast is bad!
        maxrate = 10.0
        hot = np.where(rate > maxrate)
        nhot = len(hot)
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
        bad = np.where(np.isnan(dark))
        dark[bad] = 0
        medrate = median(rate)

        # Median filter along reads dimension
        for i in range(2048):
            slice = dark[i, :, :]
            dark[i, :, :] = medfilt2d(slice, 7, dim=2)

        # Set negative pixels to zero
        neg = np.where(dark < -10)
        dark[neg] = 0

        # Write them out
        if step:
            file = prefix + 'Dark{:d}-{:s}-{:08d}'.format(step,chip,i1)
        else:
            file = prefix + 'Dark-{:s}-{:08d}'.format(chip,i1)

        leadstr = 'APMKDARK: '
        head['HISTORY'] = leadstr+time.asctime()
        import socket
        head['HISTORY'] = leadstr+getpass.getuser()+' on '+socket.gethostname()
        import platform
        head['HISTORY'] = leadstr+'Python '+pyvers+' '+platform.system()+' '+platform.release()+' '+platform.architecture()[0]
        # add reduction pipeline version to the header
        head['HISTORY'] = leadstr+' APOGEE Reduction Pipeline Version: '+load.apred
        MWRFITS(0, os.path.join(darkdir, file + '.fits'), head0, create=True)
        MKHDR(head1, dark)
        sxaddpar(head1, 'EXTNAME', 'DARK')
        MWRFITS(dark, os.path.join(darkdir, file + '.fits'), head1)
        MKHDR(head2, chi2)
        sxaddpar(head2, 'EXTNAME', 'CHI-SQUARED')
        MWRFITS(chi2, os.path.join(darkdir, file + '.fits'), head2)
        MKHDR(head3, mask)
        sxaddpar(head3, 'EXTNAME', 'MASK')
        MWRFITS(mask, os.path.join(darkdir, file + '.fits'), head3)

        # Make some plots/images
        if not os.path.exists(os.path.join(darkdir, 'plots')):
            os.makedirs(os.path.join(darkdir, 'plots'))

        DARKPLOT(dark, mask, os.path.join(darkdir, 'plots', file), hard=True)

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
        file = prefix + 'DarkRate-{:s}-{:08d}'.format(chip,i1)
        MWRFITS(rate, os.path.join(darkdir, file + '.fits'), create=True)

        dark = 0
        time = systime(seconds=True)
        print('Done', chip, time - time0)

    del red

    # Write the summary log information
    file = prefix + 'Dark-{:08d}'.format(i1)
    MWRFITS(darklog, os.path.join(darkdir, file + '.tab'), create=True)

    # Remove lock file
    aplock(darkfile, clear=True)

    # Compile summary web page
    DARKHTML(darkdir)
