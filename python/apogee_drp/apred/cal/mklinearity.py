import os
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from ...utils import apzip,apload
from ...apred import ap3d,mkcal

def mklinearity(frameid, apred='daily', telescope='apo25m', darkid=None, bpmid=None,
                detid=None, chip=None, lindata=None, nread=None, minread=2,
                norder=2, nskip=4, inter=False, clobber=False, unlock=False):
    """
    Procedure to derive linearity correction from an internal flat field frame.

    Parameters
    ----------
    frameid : int
       The ID8 number of the internal LED exposure. 
    apred : str, optional
       APOGEE reduction version.  Default is 'daily'.
    telescope : str, optional
       Telescope name, 'apo25m' or 'lco25m'.  Default is 'apo25m'.
    darkid : int, optional
       The name of the dark calibration file to use.  By default, this information
         is obtained from the master calibration library file.
    bpmid : int, optional
       The name of the BPM calibration file to use.  By default, this information
         is obtained from the master calibration library file.
    detid : int, optional
       The name of the detector calibration file to use.  By default, this information
         is obtained from the master calibration library file.
    chip : int, optional
       Do a single chip, the default is to do all three. 0, 1 or 2.
    lindata : numpy array, optional
       Input linearity data.
    nread : int, optional
       Only use this number of reads.  Default is to use all reads.
    minread : int, optional
       The lowest read to start with.  Default minread=2.
    norder : int, optional
       Polynomial order to use in linearity fit.  Default norder=2.
    nskip : int, optional
       Default nskip=4.
    inter : bool, optional
       Make interactive plots.  Default is False.
    clobber : bool, optional
       Rereduce images even if they exist
    unlock : bool, optional
       Delete the lock file and start fresh.  Default is False.

    Returns
    -------
    par : numpy
       The linearity coefficients.

    A set of apLinearity-[abc]-ID8.fits files in the appropriate location
    determined by the SDSS/APOGEE tree directory structure.

    Example
    -------

    par = mklinearity(frameid)

    By J. Holtzman, 2011?
    Added doc strings, updates to use data model  D. Nidever, Sep 2020 
    Translated to Python  D. Nidever, Nov 2023
    """

    cref = 3000.0
    load = apload.ApLoad(apred=apred,telescope=telescope)
    
    # Character frameid
    cframeid = '{:08d}'.format(frameid)

    # Get calibration file names for this MJD
    cmjd = load.cmjd(frameid)
    mjd = float(cmjd)
    libdir = os.environ['APOGEE_DRP_DIR']+'/data'
    caldata = mkcal.getcal(libdir + '/cal/' + load.instrument + '.par', mjd)
    if darkid is None: darkid = caldata['dark']
    if bpmid is None: bpmid = caldata['bpm']
    if detid is None: detid = caldata['det']

    # chip= keyword specifies single chip, else use all 3 chips
    chips = ['a', 'b', 'c']
    if chip is not None:
        ichip1 = chip
        ichip2 = chip
    else:
        ichip1 = 0
        ichip2 = 2

    # Get the name of the file for output linearity data
    lindir = os.path.dirname(load.filename('Detector', num=0, chips=True)) + '/'
    if not os.path.exists(lindir + 'plots'):
        os.makedirs(lindir + 'plots')
    
    if lindata is not None:
        linfile = lindir + load.prefix + 'LinearityTest-' + cframeid + '.dat'
    else:
        linfile = lindir + load.prefix + 'Linearity-' + cframeid + '.dat'

    # Make sure file construction isn't already in process
    lock.lock(linfile, waittime=10, unlock=unlock)

    # Does the file already exist?
    if os.path.exists(linfile) and not clobber:
        return None

    print('Making Linearity:', frameid)
    # Open .lock file
    lock.lock(linfile, lock=True)

    # Loop over the chips
    for ichip in np.arange(ichip1, ichip2 + 1):
        chip_name = chips[ichip]

        # Uncompress data cube
        datafile = load.filename('R', num=frameid, chips=True).replace('R-','R-'+chip_name+'-')
        datadir = os.path.dirname(datafile) + '/'
        base = os.path.splitext(os.path.basename(datafile))[0]
        einfo = info.expinfo(files=[datafile])[0]

        if nread is not None:
            nreads = nread
        else:
            nreads = einfo['nread']

        localdir = os.environ['APOGEE_LOCALDIR']
        #if os.path.isdir(getlocaldir()):
        #    outdir = getlocaldir()
        #else:
        #    outdir = './'

        if not os.path.exists(outdir + base + '.fits')):
            unzip.unzip(datafile, fitsdir=outdir)

        # Read the cube
        cube = np.zeros((2048, 2048, nreads), dtype=np.uint)
        for i in range(1, nreads + 1):
            im, head = fits.getdata(outdir + base + f'_r{i:02d}.fits'), header=True)
            cube[:,:,i-1] = im.astype(np.uint)

        # Do reference correction (assuming aprefcorr is defined)
        tmp = ap3d.refcorr(cube, head, mask, indiv=False, cds=True)
        cube = tmp

        # If we have input linearity data, we will use it to test that things are working!
        if lindata is not None:
            oldcube = cube.copy()
            for iy in range(2048):
                if iy % 10 == 0:
                    print('Linearity...', iy)
                slcim = cube[:, iy, :]
                slcim_out = ap3d.lincorr(slcim, lindata)
                cube[:, iy, :] = slcim_out.reshape((2048, 1, nreads))

        # Loop over different sections on chip
        for ix in range(0, 40, 5):
            ix1 = 24 + ix * 50
            ix2 = ix1 + 10
            for iy in range(0, 40, 5):
                # Counts in section
                iy1 = 24 + iy * 50
                iy2 = iy1 + 10

                # Get median in region
                cts = np.zeros(nreads-2, dtype=float)
                rate = np.zeros(nreads-2, dtype=float)
                instrate = np.zeros(nreads-2, dtype=float)
                for i in np.arange(2, nreads, nskip):
                    cts[i-2] = np.median(cube[ix1:ix2, iy1:iy2, i] - cube[ix1:ix2, iy1:iy2, 0])
                    rate[i-2] = cts[i - 2] / (i - 1)
                    # Correct to "zero" read
                    cts[i-2] *= (i + 1) / (i - 1)
                    instrate[i-2] = np.median(cube[ix1:ix2, iy1:iy2, i] - cube[ix1:ix2, iy1:iy2, i - 1])

                # Normalize to rate at cref DN
                j = np.where((cts > cref - 2000) & (cts < cref + 2000))[0]
                if len(j) > 2:
                    par = np.polyfit(cts[j], rate[j], 2)
                    ref = par[0] + par[1] * cref + par[2] * cref ** 2
                    for i in np.range(2, nreads, nskip):
                        print(i, ichip, ix, iy, cts[i - 2], rate[i - 2] / ref, instrate[i - 2] / ref)

    lock.lock(linfile, clear=True)

    # Read the linearity data
    data = np.genfromtxt(linfile, dtype=[('chip', int), ('ix', int), ('iy', int),
                                         ('cts', float), ('rate', float), ('rate2', float)])

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
        if ichip == 2 and telescope[0:3]=='apo':
            ymax = 18
        if ichip == 2 and telescope[0:3]=='apo':
            ymax = 0
        print(ichip, ymax)

        # Make some plots
        if not lindata:
            if lindata:
                figfile = lindir + 'plots/' + load.prefix + 'LinearityTest-' + cframeid + '_' + str(ichip) + '.png'
            else:
                figfile = lindir + 'plots/' + load.prefix + 'Linearity-' + cframeid + '_' + str(ichip) + '.png'
            device(file=file, encap=True, color=True, xsize=12, ysize=12, _in=True)

        # Plot of instantaneous rate vs counts
        plt.scatter(data['cts'][gd], data['rate2'][gd], marker='s')
        plt.xlim(0, max(data['cts']))
        plt.ylim(0.8,1.2)
        plt.xlabel('DN',fontsize=18)
        plt.ylabel('Relative count rate',fontsize=19)
        ii = 1
        for i in range(0, 36, 5):
            j = np.where((data['ix'][gd] == i) & (data['iy'][gd] < ymax))[0]
            if len(j) > 1:
                oplot(data['cts'][gd][j], data['rate2'][gd][j], psym=6, color=(ii % 6) + 1)
            ii += 1

        # Plot of accumulated rate, normalized by final rate, vs counts
        # Since illumination isn't uniform, this normalized rate is not the same for
        # all regions. Do a fit to get rate at cref DN
        plt.scatter(data['cts'][gd], data['rate'][gd], marker='s')
        plt.ylim(0.8, 1.2)
        plt.xim(0, max(data['cts'])])
        plt.xlabel('DN',fontsize=18)
        plt.ylabel('Relative count rate',fontsize=18)

        ii = 1
        for i in range(0, 36, 5):
            ind, = np.where((data['ix'][gd] == i) & (data['iy'][gd] < ymax))
            if len(ind) > 1:
                oplot(data['cts'][gd][ind], data['rate'][gd][ind], marker='s', color=(ii % 6) + 1)
            ii += 1

        if not lindata:
            device(close=True)
            ps2jpg(file, eps=True)


    # Now do the final linearity fit using all regions in chip a and non-persistence; region of chip c
    x = np.array([])  # Initialize x and y arrays to collect data for the final fit
    y = np.array([])

    ii = 0
    for ichip in range(3):  # Loop through all chips
        ymax = 50
        if ichip == 1 and telescope[0:3]=='apo':
            ymax = 0
        if ichip == 2 and telescope[0:3]=='apo':
            ymax = 18
        gd, = np.where((data['chip'] == ichip) & (data['iy'] < ymax) &
                       (data['cts'] >= minread) & (data['cts'] < 50000))
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
        figfile = lindir + 'plots/' + load.prefix + 'LinearityTest-' + cframeid + '.png'
    else:
        figfile = lindir + 'plots/' + load.prefix + 'Linearity-' + cframeid + '.png'
    set_plot('PS')
    device(file=file, encap=True, color=True, xsize=12, ysize=8, _in=True)
    p.multi = [0, 0, 0]

    plt.scatter(x, y, marker='s')
    plt.ylim(0.9,1.1)
    plt.xlabel('DN',fontsize=18)
    plt.ylabel('Relative count rate',fontsize=18)

    # Perform polynomial fit
    par = np.polyfit(x, y, norder)
    xx = np.arange(5000) * 10.
    yy = par[0]
    tmp = xx
    for iorder in range(1, norder + 1):
        yy += par[iorder] * tmp
        tmp *= xx
    plt.plot(xx,yy,c='r',linewidth=10)
    plt.savefig(figfile,bbox_inches='tght')

    #set_plot('X')

    # Commented out in the original code
    # file_delete(lockfile, allow=True)
    lock.lock(linfile, clear=True)

    return par
