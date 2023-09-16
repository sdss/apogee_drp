import os
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def mklinearity(frameid, clobber=False, chip=None, stp=False, lindata=None,
                nread=None, minread=2, norder=2, nskip=4, inter=False, unlock=False):
    """
    Procedure to derive linearity correction from an internal flat field frame.

    Parameters
    ----------
    frameid   The ID8 number of the internal LED exposure. 
    =chip     Do a single chip, the default is to do all three.
    =lindata  The linearity data.
    =nread    Only use this number of reads.  Default is to use all reads.
    =minread  The lowest read to start with.  Default minread=2.
    =norder   Polynomial order to use in linearity fit.  Default norder=2.
    =nskip    Default nskip=4.
    /inter    Interactive plots.
    /clobber  Rereduce images even if they exist
    /stp      Stop at the end of the program.
    /unlock   Delete the lock file and start fresh.

    Returns
    -------
    A set of apLinearity-[abc]-ID8.fits files in the appropriate location
    determined by the SDSS/APOGEE tree directory structure.

    Example
    -------

    mklinearity,frameid

    By J. Holtzman, 2011?
      Added doc strings, updates to use data model  D. Nidever, Sep 2020 
    """

    cref = 3000.0

    # Get directories
    apodir = None  # Provide the actual directory path
    caldir = None  # Provide the actual directory path
    specdir = None  # Provide the actual directory path
    apovers = None  # Provide the actual version
    libdir = None  # Provide the actual directory path
    datadir = None  # Provide the actual directory path
    dirs = getdir(apodir, caldir, specdir, apovers, libdir, datadir=datadir)

    # Character frameid
    cframeid = str(frameid).zfill(8)

    # Get calibration file names for this MJD
    mjd = None  # Set the MJD value
    cmjd = getcmjd(frameid, mjd=mjd)
    darkid, bpmid, detid = getcal(mjd, libdir + '/cal/' + dirs.instrument + '.par')

    # chip= keyword specifies single chip, else use all 3 chips
    chips = ['a', 'b', 'c']
    if chip is not None:
        ichip1 = chip
        ichip2 = chip
    else:
        ichip1 = 0
        ichip2 = 2

    # Get the name of the file for output linearity data
    lindir = apogee_filename('Detector', num=1, chip='a', dir=True)
    if not os.path.exists(lindir + 'plots'):
        os.makedirs(lindir + 'plots')
    
    if lindata is not None:
        linfile = lindir + dirs.prefix + 'LinearityTest-' + cframeid + '.dat'
    else:
        linfile = lindir + dirs.prefix + 'Linearity-' + cframeid + '.dat'

    # Make sure file construction isn't already in process
    aplock(linfile, waittime=10, unlock=unlock)

    # Does the file already exist?
    if os.path.exists(linfile) and not clobber:
        return

    print('Making Linearity:', frameid)
    # Open .lock file
    aplock(linfile, lock=True)

    # Loop over the chips
    for ichip in range(ichip1, ichip2 + 1):
        chip_name = chips[ichip]

        # Uncompress data cube
        datadir = apogee_filename('R', num=cframeid, chip=chip_name, dir=True)
        file = apogee_filename('R', num=cframeid, chip=chip_name, base=True)
        base = os.path.splitext(file)[0]
        info = apfileinfo(datadir + file)

        if nread is not None:
            info.nreads = nread
        
        if os.path.isdir(getlocaldir()):
            outdir = getlocaldir()
        else:
            outdir = './'

        if not os.path.exists(os.path.join(outdir, file + '.fits')):
            apunzip(datadir + file, fitsdir=outdir)

        # Read the cube
        cube = np.empty((2048, 2048, info.nreads), dtype=np.uint)
        for i in range(1, info.nreads + 1):
            im, head = fits.getdata(os.path.join(outdir, base + f'_r{i:02d}.fits'), header=True)
            cube[:, :, i - 1] = im.astype(np.uint)

        # Do reference correction (assuming aprefcorr is defined)
        tmp = aprefcorr(cube, head, mask, indiv=0, cds=True)
        cube = tmp

        # If we have input linearity data, we will use it to test that things are working!
        if lindata is not None:
            oldcube = cube.copy()
            for iy in range(2048):
                if iy % 10 == 0:
                    print('Linearity...', iy)
                slice = cube[:, iy, :]
                slice_out = APLINCORR(slice, lindata)
                cube[:, iy, :] = slice_out.reshape((2048, 1, info.nreads))

        # Loop over different sections on chip
        for ix in range(0, 40, 5):
            ix1 = 24 + ix * 50
            ix2 = ix1 + 10
            for iy in range(0, 40, 5):
                # Counts in section                
                iy1 = 24 + iy * 50
                iy2 = iy1 + 10

                # Get median in region
                cts = np.zeros(info.nreads - 2, dtype=float)
                rate = np.zeros(info.nreads - 2, dtype=float)
                instrate = np.zeros(info.nreads - 2, dtype=float)
                for i in range(2, info.nreads, nskip):
                    cts[i - 2] = np.median(cube[ix1:ix2, iy1:iy2, i] - cube[ix1:ix2, iy1:iy2, 0])
                    rate[i - 2] = cts[i - 2] / (i - 1)
                    # Correct to "zero" read
                    cts[i - 2] *= (i + 1) / (i - 1)
                    instrate[i - 2] = np.median(cube[ix1:ix2, iy1:iy2, i] - cube[ix1:ix2, iy1:iy2, i - 1])

                # Normalize to rate at cref DN
                j = np.where((cts > cref - 2000) & (cts < cref + 2000))[0]
                if len(j) > 2:
                    par = np.polyfit(cts[j], rate[j], 2)
                    ref = par[0] + par[1] * cref + par[2] * cref ** 2
                    for i in range(2, info.nreads, nskip):
                        print(i, ichip, ix, iy, cts[i - 2], rate[i - 2] / ref, instrate[i - 2] / ref)

    aplock(linfile, clear=True)

    # Read the linearity data
    data = np.genfromtxt(linfile, dtype=[('chip', int), ('ix', int), ('iy', int), ('cts', float), ('rate', float), ('rate2', float)])

    if not lindata:
        set_plot('PS')
    else:
        set_plot('X')
    # !p.multi = [0,1,2]
    smcolor()

    ichip1, ichip2 = 0, 2  # Set your ichip1 and ichip2 values

    for ichip in range(ichip1, ichip2 + 1):
        gd = np.where(data['chip'] == ichip)[0]
        ymax = 50
        if ichip == 2 and dirs['instrument'] == 'apogee-n':
            ymax = 18
        if ichip == 2 and dirs['instrument'] == 'apogee-n':
            ymax = 0
        print(ichip, ymax)

        # Make some plots
        if not lindata:
            if lindata:
                file = lindir + 'plots/' + dirs['prefix'] + 'LinearityTest-' + cframeid + '_' + str(ichip) + '.eps'
            else:
                file = lindir + 'plots/' + dirs['prefix'] + 'Linearity-' + cframeid + '_' + str(ichip) + '.eps'
            device(file=file, encap=True, color=True, xsize=12, ysize=12, _in=True)

        # Plot of instantaneous rate vs counts
        plot(data['cts'][gd], data['rate2'][gd], psym=6, yr=[0.8, 1.2], xr=[0, max(data['cts'])], thick=3, charthick=3,
             charsize=1.5, xtit='DN', ytit='Relative count rate')
        ii = 1
        for i in range(0, 36, 5):
            j = np.where((data['ix'][gd] == i) & (data['iy'][gd] < ymax))[0]
            if len(j) > 1:
                oplot(data['cts'][gd][j], data['rate2'][gd][j], psym=6, color=(ii % 6) + 1)
            ii += 1

        # Plot of accumulated rate, normalized by final rate, vs counts
        # Since illumination isn't uniform, this normalized rate is not the same for
        # all regions. Do a fit to get rate at cref DN
        plot(data['cts'][gd], data['rate'][gd], psym=6, yr=[0.8, 1.2], xr=[0, max(data['cts'])], thick=3,
             charthick=3, charsize=1.5, xtit='DN', ytit='Relative count rate')
        ii = 1
        for i in range(0, 36, 5):
            j = np.where((data['ix'][gd] == i) & (data['iy'][gd] < ymax))[0]
            if len(j) > 1:
                oplot(data['cts'][gd][j], data['rate'][gd][j], psym=6, color=(ii % 6) + 1)
            ii += 1

        if not lindata:
            device(close=True)
            ps2jpg(file, eps=True)
        else:
            stop()

    # Now do the final linearity fit using all regions in chip a and non-persistence; region of chip c
    x = np.array([])  # Initialize x and y arrays to collect data for the final fit
    y = np.array([])

    ii = 0
    for ichip in range(3):  # Loop through all chips
        ymax = 50
        if ichip == 1 and dirs['instrument'] == 'apogee-n':
            ymax = 0
        if ichip == 2 and dirs['instrument'] == 'apogee-n':
            ymax = 18
        gd = np.where((data['chip'] == ichip) & (data['iy'] < ymax) & (data['cts'] >= minread) & (data['cts'] < 50000))[0]
        if len(gd) > 0:
            if ii == 0:
                x = data['cts'][gd]
                y = data['rate'][gd]
            else:
                x = np.concatenate((x, data['cts'][gd]))
                y = np.concatenate((y, data['rate'][gd]))
            ii += 1

    # Final fit and plot
    if lindata:
        file = lindir + 'plots/' + dirs['prefix'] + 'LinearityTest-' + cframeid + '.eps'
    else:
        file = lindir + 'plots/' + dirs['prefix'] + 'Linearity-' + cframeid + '.eps'
    set_plot('PS')
    device(file=file, encap=True, color=True, xsize=12, ysize=8, _in=True)
    p.multi = [0, 0, 0]

    plot(x, y, psym=6, yr=[0.9, 1.1], xtit='DN', ytit='Relative count rate', thick=2, charthick=2)

    # Perform polynomial fit
    par = np.polyfit(x, y, norder)
    xx = np.arange(5000) * 10.
    yy = par[0]
    tmp = xx
    for iorder in range(1, norder + 1):
        yy += par[iorder] * tmp
        tmp *= xx
    oplot(xx, yy, color=3, thick=3)
    device(close=True)
    ps2jpg(file, eps=True)
    set_plot('X')

    # Commented out in the original code
    # file_delete(lockfile, allow=True)
    aplock(linfile, clear=True)

    return par
