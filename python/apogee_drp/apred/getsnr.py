import sys
import glob
import os
import subprocess
import math
import time
import pickle
import doppler
import numpy as np
from pathlib import Path
from astropy.io import fits, ascii
from astropy.table import Table
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.lib.recfunctions import append_fields, merge_arrays
from astroplan import moon_illumination
from astropy.coordinates import SkyCoord, get_moon
from astropy import units as astropyUnits
from apogee_drp.utils import plan,apload,yanny,plugmap,platedata,bitmask,peakfit,colorteff
from apogee_drp.apred import wave,monitor
from apogee_drp.database import apogeedb
from dlnpyutils import utils as dln
from sdss_access.path import path
import pdb
import matplotlib.pyplot as plt
import matplotlib
from astropy.convolution import convolve, Box1DKernel
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator
import matplotlib.ticker as ticker
import matplotlib.colors as mplcolors
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from scipy.signal import medfilt2d as ScipyMedfilt2D
from scipy.signal import medfilt, convolve, boxcar, argrelextrema, find_peaks
from scipy.optimize import curve_fit
from scipy import interpolate
import datetime

cspeed = 299792.458e0

sdss_path = path.Path()

sort_table_link = 'https://www.kryogenix.org/code/browser/sorttable/sorttable.js'

#matplotlib.use('agg')

# put import pdb; pdb.set_trace() wherever you want stop

#sdss_path.full('ap2D',apred=self.apred,telescope=self.telescope,instrument=self.instrument,
#                        plate=self.plate,mjd=self.mjd,prefix=self.prefix,num=0,chip='a')

# Plugmap for plate 8100 mjd 57680
# /uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/mapper/apo/57679/plPlugMapM-8100-57679-01.par
# apPlateSum
# /uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/apogee/spectro/redux/t14/visit/apo25m/200+45/8100/57680/apPlateSum-8100-57680.fits

# Planfile for plate 8100 mjd 57680
# https://data.sdss.org/sas/sdss5/mwm/apogee/spectro/redux/t14/visit/apo25m/200+45/8100/57680/apPlan-8100-57680.par

#------------------------------------------------------------------------------------------------------------------------
# APQA
#
#  call routines to make "QA" plots and web pages for a plate/MJD
#  for calibration frames, measures some features and makes a apQAcal file
#    with some summary information about the calibration data
#--------------------------------------------------------------------------------------------------

outdir = '/uufs/chpc.utah.edu/common/home/u0955897/projects/snr/'
observatory='apo'
apred='daily'
fps=True

###################################################################################################
'''APQAALL: Wrapper for running apqa for ***ALL*** plates '''
def doit(mjdstart=59560, observatory='apo', apred='daily'):

    # Establish telescope
    telescope = observatory + '25m'
    load = apload.ApLoad(apred=apred, telescope=telescope)

    out = open(outdir + 'apogeeSNR-FPS.dat', 'w')
    out.write('EXPOSURE      SNR_G\n')

    mdir = os.environ.get('APOGEE_REDUX') + '/' + apred + '/monitor/'
    expdata = fits.getdata(mdir + 'apogee-nSci.fits')
    g, = np.where(expdata['MJD'] >= mjdstart)
    expdata = expdata[g]
    order = np.argsort(expdata['MJD'])
    expdata = expdata[order][::-1]
    exp = expdata['IM']
    plate = expdata['PLATE']
    mjd = expdata['MJD']
    nexp = len(expdata)
    for i in range(nexp):
        platesumfile = load.filename('PlateSum', plate=plate[i], mjd=str(mjd[i]), fps=fps)
        if os.path.exists(platesumfile) is False:
            return
        else:
            data = fits.getdata(platesumfile)
            g, = np.where(exp[i] == data['IM'])
            if len(g) < 1:
                return
            else:
                fiber = fits.getdata(platesumfile,2)
                hmag = fiber['hmag']
                snr = np.squeeze(fiber['sn'][:, g[0], 1])

                # Linear fit to log(snr) vs. Hmag for ALL objects
                gdall, = np.where((hmag > 4) & (hmag < 20) & (snr > 0))
                if len(gdall) > 2:
                    coefall = np.polyfit(hmag[gdall],np.log10(snr[gdall]),1)
                else:
                    coefall = np.zeros(2,float) + np.nan
                # Linear fit to log(S/N) vs. H for 10<H<11.5
                gd, = np.where((hmag >= 10.0) & (hmag <= 11.5) & (snr > 0))
                if len(gd) > 2:
                    coef = np.polyfit(hmag[gd],np.log10(snr[gd]),1)
                else:
                    coef = np.zeros(2,float) + np.nan
                if len(gd) > 2:
                    snr_fid = 10**np.polyval(coef,11)
                elif len(gdall) > 2:
                    snr_fid = 10**np.polyval(coefall,11)
                else:
                    snr_fid = np.mean(snr)

                ssnr = str("%.3f" % round(snr_fid,3)).rjust(8)
                print('('+str(i+1).zfill(4)+'/'+str(nexp)+'): ' + os.path.basename(platesumfile) + '  ' + ssnr)
                
                out.write(str(int(round(exp[i]))) + '    ' + ssnr + '\n')
                #p1 = str(int(round(x[0])))
                #p2 = str("%.3f" % round(x[1],3)).rjust(10)
                #p3 = str("%.3f" % round(x[2],3)).rjust(10)
                #p4 = str("%.3f" % round(x[3],3)).rjust(10)
                #out.write(p1+'  '+p2+p3+p4+'\n')

    out.close()







