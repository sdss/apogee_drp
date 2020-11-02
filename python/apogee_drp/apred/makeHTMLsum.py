import sys
import glob
import os
import subprocess
import math
import numpy as np
from pathlib import Path
from astropy.io import fits, ascii
from astropy.table import Table
from astropy.time import Time
from numpy.lib.recfunctions import append_fields, merge_arrays
from astroplan import moon_illumination
from astropy.coordinates import SkyCoord, get_moon
from astropy import units as astropyUnits
from scipy.signal import medfilt2d as ScipyMedfilt2D
from apogee_drp.utils import plan,apload,yanny,plugmap,platedata,bitmask
from apogee_drp.apred import wave
from dlnpyutils import utils as dln
from sdss_access.path import path
import pdb
import matplotlib.pyplot as plt
import matplotlib
from astropy.convolution import convolve, Box1DKernel
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, MaxNLocator
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar


'''-----------------------------------------------------------------------------------------'''
'''makeHTMLsum: makes mjd.html and fields.html                                              '''
'''-----------------------------------------------------------------------------------------'''
def makeHTMLsum(mjdmin=59146, mjdmax=9999999, apred='daily', mjdfilebase='mjdNew'):
    # Establish data directories.
    datadirN = os.environ['APOGEE_DATA_N']
    datadirS = os.environ['APOGEE_DATA_S']
    apodir =   os.environ.get('APOGEE_REDUX')+'/'
    qadir = apodir+apred+'/qa/'

    # Find all .log.html files, get all MJDs with data
    print("----> makeHTMLsum: finding log files. Please wait.")
    logsN = np.array(glob.glob(datadirN+'/*/*.log.html'))
    logsS = np.array(glob.glob(datadirS+'/*/*.log.html'))
    logs = np.concatenate([logsN,logsS]) 
    nlogs = len(logs)
    print("----> makeHTMLsum: found "+str(nlogs)+" log files.")

    # Get array of MJDs and run mkhtml if MJD[i] within mjdmin-mjdmax range
    mjd = np.empty(nlogs)
    for i in range(nlogs): 
        mjd[i] = int(os.path.basename(logs[i]).split('.')[0])
#        if doall is not None:
#            if (mjd[i] >= mjdmin) & (mjd[i] <= mjdmax):
#                q = mkhtml(mjd[i], apred)

    # Reverse sort the logs and MJDs so that newest MJD will be at the top
    order = np.argsort(mjd)
    logs = logs[order[::-1]]
    mjd = mjd[order[::-1]]

    # Limit to MJDs within mjdmin-mjdmax range
    gd = np.where((mjd >= mjdmin) & (mjd <= mjdmax))
    logs = logs[gd]
    mjd = mjd[gd]
    nmjd = len(mjd)

    # Open the mjd file html
    if mjdfilebase is None: mjdfilebase = 'mjd'
    mjdfile = qadir+mjdfilebase+'.html'
    print("----> makeHTMLsum: creating "+mjdfile)
    html = open(mjdfile,'w')
    html.write('<HTML><BODY>\n')
    html.write('<HEAD><script type=text/javascript src=html/sorttable.js></script><title>APOGEE MJD Summary</title></head>\n')
    html.write('<H1>APOGEE Observation Summary by MJD</H1>\n')
    html.write('<p> Fields View (link coming soon) <p>\n')
    visSumPathN = 'summary/allVisit-daily-apo25m.fits'
    starSumPathN = 'summary/allStar-daily-apo25m.fits'
    visSumPathS = 'summary/allVisit-daily-lco25m.fits'
    starSumPathS = 'summary/allStar-daily-lco25m.fits'
    html.write('<p> Summary files: <a href="'+visSumPathN+'">allVisit</a>,  <a href="'+starSumPathN+'">allStar</a></p>\n')
    #html.write('<BR>LCO 2.5m Summary Files: <a href="'+visSumPathS+'">allVisit</a>,  <a href="'+starSumPathS+'">allStar</a></p>\n')
    html.write( 'Yellow: APO 2.5m, Green: LCO 2.5m\n')
    #html.write('<br>Click on column headings to sort\n')

    # Create web page with entry for each MJD
    html.write('<TABLE BORDER=2 CLASS=sortable>\n')
    html.write('<TR bgcolor=eaeded><TH>Logs (data)<TH>Night QA<TH>Observed Plate QA<TH>Summary Files\n')
    for i in range(nmjd):
        cmjd = str(int(round(mjd[i])))
        # Establish telescope and instrument and setup apLoad depending on telescope.
        telescope = 'apo25m'
        instrument = 'apogee-n'
        datadir = datadirN
        datadir1 = 'data'
        color = 'FFFFF33'
        if 'lco' in logs[i]: 
            telescope = 'lco25m'
            instrument = 'apogee-s'
            datadir = datadirS
            datadir1 = 'data2s'
            color = 'b3ffb3'
        load = apload.ApLoad(apred=apred, telescope=telescope)

        # Column 1: Logs(data)
        logFileDir = '../../' + os.path.basename(datadir) + '/' + cmjd + '/'
        logFilePath = logFileDir + cmjd + '.log.html'

        logFile = 'https://data.sdss.org/sas/apogeework/apogee/spectro/' + datadir1 + '/' + cmjd + '/' + cmjd + '.log.html'
        logFileDir = 'https://data.sdss.org/sas/apogeework/apogee/spectro/' + datadir1 + '/' + cmjd + '/'
        html.write('<TR bgcolor='+color+'><TD align="center"><A HREF='+logFile+'>'+cmjd+'</A>\n')
        html.write('<A HREF='+logFileDir+'><BR>(raw)</A>\n')

        # Column 2: Exposures (removed)
#        exposureLogPath = '../exposures/' + instrument + '/' + cmjd + '/html/' + cmjd + 'exp.html'
#        exposureLog = 'https://data.sdss.org/sas/apogeework/apogee/spectro/redux/current/exposures/'+instrument+'/'+cmjd+'/html/'+cmjd+'exp.html'
#        html.write('<TD><center><A HREF='+exposureLog+'>'+cmjd+'</A></center>\n')

        # Column 2: Night QA
        # NOTE: This directory does not exist yet.
        qaPage = apodir + apred + '/exposures/' + instrument + '/' + cmjd + '/html/' + cmjd + '.html'
        if os.path.exists(qaPage):
            qaPagePath = '../exposures/' + instrument + '/' + cmjd + '/html/' + cmjd + '.html'
            html.write('<TD><center><A HREF='+qaPagePath+'>'+cmjd+' QA </a></center>\n')
        else:
            html.write('<TD><center><FONT COLOR=red> '+cmjd+' QA </font></center>\n')

        # Column 3: Plates reduced for this night
        plateQApaths = apodir+apred+'/visit/'+telescope+'/*/*/'+cmjd+'/html/apQA-*'+cmjd+'.html'
        plateQAfiles = np.array(glob.glob(plateQApaths))
        nplates = len(plateQAfiles)
        html.write('<TD align="left">\n')
        for j in range(nplates):
            if plateQAfiles[j] != '':
                plateQApathPartial = plateQAfiles[j].split(apred+'/')[1]
                tmp = plateQApathPartial.split('/')
                field = tmp[2]
                plate = tmp[3]
                if j < nplates:
                    html.write('('+str(j+1)+') <A HREF=../'+plateQApathPartial+'>'+plate+': '+field+'<BR></A>\n')
                else:
                    html.write('('+str(j+1)+') <A HREF=../'+plateQApathPartial+'>'+plate+': '+field+'</A>\n')

        # Column 5: Combined files for this night
        #html.write('<TD>\n')

        # Column 6: Single stars observed for this night
        #html.write('<TD>\n')

        # Column 7: Dome flats observed for this night
        #html.write('<TD>\n')

        # Column 4: Summary files
        visSumPath = 'summary/'+cmjd+'/allVisitMJD-daily-'+telescope+'-'+cmjd+'.fits'
        starSumPath = 'summary/'+cmjd+'/allStarMJD-daily-'+telescope+'-'+cmjd+'.fits'
        if len(plateQAfiles) != 0: 
            html.write('<TD align="center"><a href="'+visSumPath+'">allVisitMJD</a>\n')
            html.write('<BR><a href="'+starSumPath+'">allStarMJD</a>\n')
        else:
            html.write('<TD>\n')

    html.write('</table>\n')

    # Summary calibration data
    caldir = 'cal/'
    html.write('<P> Calibration Data:\n')
    html.write('<UL>\n')
    html.write('<LI> <A HREF='+caldir+'/darkcorr/html/darks.html> Darks </A>\n')
    html.write('<LI> <A HREF='+caldir+'/flatcorr/html/flats.html> Flats </A>\n')
    html.write('<LI> <A HREF='+caldir+'/flux/html/flux.html> Fiber fluxes from petal flats </A>\n')
    html.write('<LI> <A HREF='+caldir+'/trace/html/trace.html> Traces </A>\n')
    html.write('<LI> <A HREF='+caldir+'/detector/html/rn.html> Readout noise </A>\n')
    html.write('<LI> <A HREF='+caldir+'/detector/html/gain.html> Gain </A>\n')
    html.write('<LI> <A HREF='+caldir+'/wave/html/wave.html> Wave cals </A>\n')
    html.write('</UL>\n')

    html.write('</body></html>\n')
    html.close()

    print("----> makeHTMLsum: Done.")










