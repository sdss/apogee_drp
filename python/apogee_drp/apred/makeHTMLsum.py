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
def makeHTMLsum(mjdmin=59146, mjdmax=9999999, apred='daily', mjdfilebase='mjd',fieldfilebase='fields',
                domjd = True, dofields = True):
    # Establish data directories.
    datadirN = os.environ['APOGEE_DATA_N']
    datadirS = os.environ['APOGEE_DATA_S']
    apodir =   os.environ.get('APOGEE_REDUX')+'/'
    qadir = apodir+apred+'/qa/'

    if domjd is True:
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
        html.write('<p><A HREF=fields.html>Fields view</A></p>\n')
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
            color = 'FFFFF8A'
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

    #---------------------------------------------------------------------------------------
    # Fields view
    if dofields is True:
        if fieldfilebase is None: fieldfilebase = 'fields'
        fieldfile = qadir+fieldfilebase+'.html'
        print("----> makeHTMLsum: creating "+fieldfile)
        html = open(fieldfile,'w')
        html.write('<HTML><BODY>\n')
        html.write('<HEAD><script type=text/javascript src=html/sorttable.js></script><title>APOGEE Field Summary</title></head>\n')
        html.write('<H1>APOGEE Observation Summary by Field</H1>\n')
        html.write('<p><A HREF=mjd.html>MJD view</A></p>\n')

        html.write('<p>APOGEE sky coverage: red=APOGEE1 (yellow: commissioning), green=APOGEE2, magenta=APOGEE2S, cyan=MaNGA-APOGEE2<p>\n')
        html.write('<img src=sky.gif width=45%>\n')
        html.write('<img src=galactic.gif width=45%>\n')
        html.write('<p>Summary files:\n')

    #    if ~keyword_set(suffix) then suffix='-'+apred_vers+'-'+aspcap_vers+'.fits'
    #    html.write('<a href=../../aspcap/'+apred_vers+'/'+aspcap_vers+'/allStar'+suffix+'> allStar'+suffix+' file </a>\n')
    #    html.write(' and <a href=../../aspcap/'+apred_vers+'/'+aspcap_vers+'/allVisit'+suffix+'> allVisit'+suffix+' file </a>\n')

        html.write('<br>Links on field name are to combined spectra plots and info\n')
        html.write('<br>Links on plate name are to visit spectra plots and info\n')
        html.write('<br>Links on MJD are to QA and summary plots for the visit\n')
        html.write('<br>Click on column headings to sort\n')

        html.write('<TABLE BORDER=2 CLASS=sortable>\n')
        html.write('<TR><TH>FIELD<TH>Program<TH>ASPCAP<TH>PLATE<TH>MJD<TH>LOCATION<TH>RA<TH>DEC<TH>S/N(red)<TH>S/N(green)<TH>S/N(blue)\n')
    #    html.write('<TR><TD>FIELD<TD>Program<TD>ASPCAP<br>'+apred_vers+'/'+aspcap_vers+'<TD>PLATE<TD>MJD<TD>LOCATION<TD>RA<TD>DEC<TD>S/N(red)<TD>S/N(green)<TD>S/N(blue)\n')

        plates = np.array(glob.glob(apodir+apred+'/visit/*/*/*/*/'+'*PlateSum*.fits'))
        nplates = len(plates)

        # should really get this next stuff direct from database!
        plans = yanny.yanny(os.environ['PLATELIST_DIR']+'/platePlans.par', np=True)

        # Get arrays of observed data values (plate ID, mjd, telescope, field name, program, location ID, ra, dec)
        iplate = [];  imjd = [];  itel = [];   iname = [];  iprogram = [];   iloc = [];   ira=[];   idec=[]
        for i in range(nplates): 
            plate = os.path.basename(plates[i]).split('-')[1]
            iplate.append(plate)
            mjd = os.path.basename(plates[i]).split('-')[2][:-5]
            imjd.append(mjd)
            tmp = plates[i].split('visit/')
            tel = tmp[1].split('/')[0]
            itel.append(tel)
            tmp = plates[i].split(tel+'/')
            name = tmp[1].split('/')[0]
            iname.append(name)
            gd = np.where(int(plate) == plans['PLATEPLANS']['plateid'])
            iprogram.append(plans['PLATEPLANS']['programname'][gd][0])
            ira.append(str("%.6f" % round(plans['PLATEPLANS']['raCen'][gd][0],6)))
            idec.append(str("%.6f" % round(plans['PLATEPLANS']['decCen'][gd][0],6)))
        iplate = np.array(iplate)
        imjd = np.array(imjd)
        itel = np.array(imjd)
        iname = np.array(iname)
        iprogram = np.array(iprogram)
        ira = np.array(ira)
        idec = np.array(idec)

        # Sort by MJD
        order = np.argsort(imjd)
        iplate = iplate[order]
        imjd = imjd[order]
        itel = imjd[order]
        iname = iname[order]
        iprogram = iprogram[order]
        ira = ira[order]
        idec = idec[order]

        for i in range(nplates):
            color='#ffb3b3'
            if iprogram[i] == 'RM': color = '#FCF793' 
            if iprogram[i] == 'AQMES-Wide': color='#B9FC93'

            html.write('<TR bgcolor='+color+'><TD>'+iname[i]+'\n') 
            html.write('<TD>'+iprogram[i]+'\n') 
            html.write('<TD> --- \n')
            qalink = '../visit/'+itel[i]+'/'+iname[i]+'/'+iplate[i]+'/'+imjd[i]+'/html/apQA-'+iplate[i]+'-'+imjd[i]+'.html'
            html.write('<TD><A href="'+qalink+'">'+iplate[i]+'</a>\n')
            html.write('<TD><center>'+imjd[i]+'</center>\n') 
    #        html.write('<TD><center><A HREF=exposures/'+dirs.instrument+'/'+cmjd+'/html/'+cmjd+'.html> '+cmjd+' </a></center>\n')
            html.write('<TD><center>'+iloc[i]+'</center>\n')
            html.write('<TD><center>'+ira[i]+'</center>\n') 
            html.write('<TD><center>'+idec[i]+'</center>\n') 
            html.write('<TD><center>---</center>\n') 
            html.write('<TD><center>---</center>\n') 
            html.write('<TD><center>---</center>\n') 

        html.write('</TABLE>\n')
        html.write('</BODY></HTML>\n')
        html.close()







