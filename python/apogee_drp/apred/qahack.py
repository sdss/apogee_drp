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
from apogee_drp.utils import plan,apload,yanny,plugmap,platedata,bitmask,peakfit
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

vispath = '/uufs/chpc.utah.edu/common/home/sdss40/apogeework/apogee/spectro/redux/dr17/visit/apo25m/'
fluxpath = '/uufs/chpc.utah.edu/common/home/sdss40/apogeework/apogee/spectro/redux/dr17/cal/flux/'
exppath = '/uufs/chpc.utah.edu/common/home/sdss40/apogeework/apogee/spectro/redux/dr17/exposures/apogee-n'
allvpathUtah = '/uufs/chpc.utah.edu/common/home/sdss40/apogeework/apogee/spectro/aspcap/dr17/synspec/allVisit-dr17-synspec.fits'
htmldir = '/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u0955897/qa/'
plotdir = htmldir + 'plots/'
chiplab = np.array(['blue','green','red'])
#------------------------------------------------------------------------------------------------------------------------
# APQA
#
#  call routines to make "QA" plots and web pages for a plate/MJD
#  for calibration frames, measures some features and makes a apQAcal file
#    with some summary information about the calibration data
#--------------------------------------------------------------------------------------------------

###################################################################################################
'''APQAALL: Wrapper for running apqa for ***ALL*** plates '''
def apqaALL(allv=None, mjdstart=57824, mjdstop=57878, observatory='apo', apred='daily', makeobshtml=True, makeobsplots=True):

    if allv is None: allv = fits.getdata(allvpathUtah)
    g, = np.where((allv['MJD'] >= mjdstart) & (allv['MJD'] <= mjdstop) & (allv['TELESCOPE'] == 'apo25m'))
    #pdb.set_trace()
    gallv = allv[g]
    ujd,uind = np.unique(gallv['JD'], return_index=True)
    uallv = gallv[uind]
    nplate = len(ujd)
    nplatestr = str(nplate)
    print('Running on ' + nplatestr + ' visits')
    for i in range(nplate):
        plate = uallv['PLATE'][i].strip()
        mjd = str(uallv['MJD'][i])
        field = uallv['FIELD'][i]
        plsumpath = vispath + field + '/' + plate + '/' + mjd + '/apPlateSum-' + plate + '-' + mjd + '.fits'
        if os.path.exists(plsumpath) == False: pdb.set_trace()
        x = makeObsHTML(plate=plate, mjd=mjd, field=field)

#def makeObsHTML(load=None, ims=None, imsReduced=None, plate=None, mjd=None, field=None,
#                   fluxid=None, telescope=None):
#        x = apqaMJD(mjd=umjd[ii], observatory=observatory, apred=apred, makeplatesum=makeplatesum, 
#                    makeobshtml=makeobshtml, makeobsplots=makeobsplots, makevishtml=makevishtml, 
#                    makestarhtml=makestarhtml, makevisplots=makevisplots,makestarplots=makestarplots,
#                    makenightqa=makenightqa, makemasterqa=makemasterqa, makeqafits=makeqafits, 
#                    makemonitor=makemonitor, clobber=clobber)

###################################################################################################
''' MAKEOBSHTML: mkhtmlplate translation '''
def makeObsHTML(plate=None, mjd=None, field=None, fluxid=None, telescope='apo25m'):

    print("----> makeObsHTML: Running plate "+plate+", MJD "+mjd)

    # HTML header background color
    thcolor = '#DCDCDC'

    fps = False

    chips = np.array(['a','b','c'])
    nchips = len(chips)

    #n_exposures = len(ims)

    prefix = 'ap'

    # Check for existence of plateSum file
    gvispath = vispath + field + '/' + plate + '/' + mjd + '/'
    plsumpath = gvispath + 'apPlateSum-' + plate + '-' + mjd + '.fits'
    platepath = gvispath + 'apPlate-a-' + plate + '-' + mjd + '.fits'
    planpath = gvispath + 'apPlan-' + plate + '-' + mjd + '.par'
    plans = yanny.yanny(planpath)
    fluxid = plans['fluxid'].zfill(8)
    gd, = np.where(np.array(plans['APEXP']['flavor']) == 'object')
    ims = np.array(plans['APEXP']['name'])[gd]
    n_exposures = len(ims)

    # Read the plateSum file
    tab1 = fits.getdata(plsumpath,1)
    tab2 = fits.getdata(plsumpath,2)
    tab3 = fits.getdata(plsumpath,3)

    shiftstr = fits.getdata(platepath, 13)
    pairstr = fits.getdata(platepath, 14)

    # Make the html directory if it doesn't already exist
    qafile = htmldir + 'apQA-' + plate + '-' + mjd + '.html'
    print("----> makeObsHTML: Creating "+os.path.basename(qafile))

    html = open(qafile, 'w')
    tmp = os.path.basename(qafile).replace('.html','')
    html.write('<HTML><HEAD><title>'+tmp+'</title></head><BODY>\n')
    html.write('<H1>Field: <FONT COLOR="green">' + field + '</FONT><BR>Plate: <FONT COLOR="green">' + plate)
    html.write('</FONT><BR>MJD: <FONT COLOR="green">' + mjd + '</FONT></H1>\n')
    html.write('<HR>\n')

    # SNR plots
    html.write('<H3>apVisit H versus S/N: </H3>\n')
    snrplot1 = 'apVisitSNR-'+plate+'-'+mjd+'.png'
    snrplot2 = 'apVisitSNRblocks-'+plate+'-'+mjd+'.png'
    html.write('<A HREF=plots/'+snrplot1+' target="_blank"><IMG SRC=plots/'+snrplot1+' WIDTH=600></A>')
    html.write('<A HREF=plots/'+snrplot2+' target="_blank"><IMG SRC=plots/'+snrplot2+' WIDTH=600></A>\n')
    html.write('<HR>\n')

    #fluxfile = os.path.basename(load.filename('Flux', num=fluxid, chips=True)).replace('.fits','.png')
    fluxfile = 'apFlux-' + fluxid + '.png'
    html.write('<H3>Fiber Throughput:</H3>\n')
    html.write('<P><b>Note:</b> Points are color-coded by median dome flat flux divided by the maximum median dome flat flux.</P>\n')
    html.write('<A HREF="'+'plots/'+fluxfile+'" target="_blank"><IMG SRC=plots/'+fluxfile+' WIDTH=1200></A>')
    html.write('<HR>\n')

    # Table of individual exposures.
    if pairstr is not None:
        html.write('<H3>Individual Exposures:</H3>\n')
    else:
        html.write('<H3>Individual Exposures (undithered):</H3>\n')
    html.write('<p><b>Note:</b> Design HA values are currently missing.<BR> \n')
    html.write('<b>Note:</b> Dither and Pixshift values will be "---" if exposures not dithered.<BR>\n')
    html.write('<b>Note:</b> S/N columns give S/N for blue, green, and red chips separately. </p>\n')
    html.write('<TABLE BORDER=2 CLASS="sortable">\n')
    html.write('<TR bgcolor="'+thcolor+'">\n')
    txt1 = '<TH>#<TH>FRAME <TH>EXPTIME <TH>CART <TH>SEC Z <TH>HA<TH>DESIGN HA <TH>SEEING <TH>FWHM <TH>GDRMS <TH>NREADS <TH>DITHER'
    txt2 = '<TH>PIXSHIFT <TH>ZERO <TH>ZERO<BR>RMS <TH>SKY<BR>CONTINUUM <TH>S/N <TH>S/N (CFRAME) <TH>MOON<BR>PHASE <TH>MOON<BR>DIST.'
    html.write(txt1 + txt2 +'\n')

    for i in range(n_exposures):
        gd, = np.where(int(ims[i]) == tab1['IM'])
        cframepath = gvispath + 'apCframe-a-' + ims[i] + '.fits'
        if (len(gd) >= 1) & (os.path.exists(cframepath)):
            html.write('<TR>\n')
            html.write('<TD align="right">'+str(i+1)+'\n')
            html.write('<TD align="right">'+ims[i]+'\n')
            hdr = fits.getheader(cframepath)
            html.write('<TD align="right">'+str(int(round(hdr['EXPTIME'])))+'\n')
            try:
                html.write('<TD align="right">'+str(int(round(tab1['CART'][gd][0])))+'\n')
            except:
                html.write('<TD align="right">'+tab1['CART'][gd][0]+'\n')
            html.write('<TD align="right">'+str("%.3f" % round(tab1['SECZ'][gd][0],3))+'\n')
            html.write('<TD align="right">'+str("%.2f" % round(tab1['HA'][gd][0],2))+'\n')
            html.write('<TD align="right">'+str(np.round(tab1['DESIGN_HA'][gd][0],0)).replace('[',' ')[:-1]+'\n')
            html.write('<TD align="right">'+str("%.3f" % round(tab1['SEEING'][gd][0],3))+'\n')
            html.write('<TD align="right">'+str("%.3f" % round(tab1['FWHM'][gd][0],3))+'\n')
            html.write('<TD align="right">'+str("%.3f" % round(tab1['GDRMS'][gd][0],3))+'\n')
            html.write('<TD align="right">'+str(tab1['NREADS'][gd][0])+'\n')
            j = np.where(shiftstr['FRAMENUM'] == str(tab1['IM'][gd][0]))
            nj = len(j[0])
            nodither, = np.where(shiftstr['SHIFT'] == 0)
            if (nj > 0) & (len(nodither) != len(tab1['IM'])):
                html.write('<TD align="right">'+str("%.4f" % round(shiftstr['SHIFT'][j][0],4)).rjust(7)+'\n')
                html.write('<TD align="right">'+str("%.2f" % round(shiftstr['PIXSHIFT'][j][0],2))+'\n')
            else:
                html.write('<TD align="center">---\n')
                html.write('<TD align="center">---\n')
            html.write('<TD align="right">'+str("%.3f" % round(tab1['ZERO'][gd][0],3))+'\n')
            html.write('<TD align="right">'+str("%.3f" % round(tab1['ZERORMS'][gd][0],3))+'\n')
            q = tab1['SKY'][gd][0]
            txt = str("%.2f" % round(q[2],2))+', '+str("%.2f" % round(q[1],2))+', '+str("%.2f" % round(q[0],2))
            html.write('<TD align="center">'+'['+txt+']\n')
            q = tab1['SN'][gd][0]
            txt = str("%.2f" % round(q[2],2))+', '+str("%.2f" % round(q[1],2))+', '+str("%.2f" % round(q[0],2))
            html.write('<TD align="center">'+'['+txt+']\n')
            q = tab1['SNC'][gd][0]
            txt = str("%.2f" % round(q[2],2))+', '+str("%.2f" % round(q[1],2))+', '+str("%.2f" % round(q[0],2))
            html.write('<TD align="center">'+'['+txt+']\n')
            html.write('<TD align="right">'+str("%.3f" % round(tab1['MOONPHASE'][gd][0],3))+'\n')
            html.write('<TD align="right">'+str("%.3f" % round(tab1['MOONDIST'][gd][0],3))+'\n')
        else:
            html.write('<TR bgcolor=red>\n')
            html.write('<TD align="right">'+str(i+1)+'\n')
            html.write('<TD align="right">'+ims[i]+'\n')
            html.write('<TD><TD><TD><TD><TD><TD><TD><TD><TD><TD><TD><TD><TD><TD><TD><TD><TD><TD>\n')

    html.write('</TABLE>\n')
    html.write('<HR>\n')

    # Table of exposure pairs.
    if pairstr is not None:
        npairs = len(pairstr)
        if npairs > 0:
            # Pair table.
            html.write('<H3>Dither Pair Stats:</H3>\n')
            html.write('<TABLE BORDER=2 CLASS="sortable">\n')
            html.write('<TR bgcolor="'+thcolor+'"><TH>IPAIR<TH>NAME<TH>SHIFT<TH>NEWSHIFT<TH>S/N\n')
            html.write('<TH>NAME<TH>SHIFT<TH>NEWSHIFT<TH>S/N\n')
            for ipair in range(npairs):
                html.write('<TR><TD>'+str(ipair)+'\n')
                for j in range(2):
                    html.write('<TD>'+str(pairstr['FRAMENAME'][ipair][j])+'\n')
                    html.write('<TD>'+str("%.3f" % round(pairstr['OLDSHIFT'][ipair][j],3))+'\n')
                    html.write('<TD>'+str("%.3f" % round(pairstr['SHIFT'][ipair][j],3))+'\n')
                    html.write('<TD>'+str("%.2f" % round(pairstr['SN'][ipair][j],2))+'\n')
            html.write('</TABLE>\n')
        html.write('<HR>\n')

    # Table of exposure plots.
    html.write('<H3>Individual Exposure QA Plots:</H3>\n')
    html.write('<TABLE BORDER=2>\n')
    html.write('<p><b>Note:</b> in the Mag plots, the solid line is the target line for getting S/N=100 for an H=12.2 star in 3 hours of exposure time.<BR>\n')
    html.write('<b>Note:</b> in the Spatial mag deviation plots, color gives deviation of observed mag from expected 2MASS mag using the median zeropoint.</p>\n')
    html.write('<TR bgcolor="'+thcolor+'"><TH>FRAME <TH>ZEROPOINTS <TH>MAG PLOTS (GREEN CHIP)\n')
    html.write('<TH>SPATIAL MAG DEVIATION\n')
    #html.write('<TH>SPATIAL SKY TELLURIC CH4\n')
    #html.write('<TH>SPATIAL SKY TELLURIC CO2\n')
    #html.write('<TH>SPATIAL SKY TELLURIC H2O\n')

    for i in range(n_exposures):
        gd, = np.where(int(ims[i]) == tab1['IM'])
        if len(gd) >= 1:
            oneDfile = 'ap1D-' + ims[i]
            #oneDfile = os.path.basename(load.filename('1D', num=ims[i], mjd=mjd, chips=True)).replace('.fits','')
            #html.write('<TR><TD bgcolor="'+thcolor+'"><A HREF=../html/'+oneDfile+'.html>'+str(im)+'</A>\n')
            html.write('<TR><TD bgcolor="'+thcolor+'">'+ims[i]+'\n')
            html.write('<TD><TABLE BORDER=1><TD><TD bgcolor="'+thcolor+'">RED<TD bgcolor="'+thcolor+'">GREEN<TD bgcolor="'+thcolor+'">BLUE\n')
            html.write('<TR><TD bgcolor="'+thcolor+'">Z<TD><TD>'+str("%.2f" % round(tab1['ZERO'][gd][0],2))+'\n')
            html.write('<TR><TD bgcolor="'+thcolor+'">ZNORM<TD><TD>'+str("%.2f" % round(tab1['ZERONORM'][gd][0],2))+'\n')
            txt='<TD>'+str("%.1f" % round(tab1['SKY'][gd][0][0],1))+'<TD>'+str("%.1f" % round(tab1['SKY'][gd][0][1],1))+'<TD>'+str("%.1f" % round(tab1['SKY'][gd][0][2],1))
            html.write('<TR><TD bgcolor="'+thcolor+'">SKY'+txt+'\n')
            txt='<TD>'+str("%.1f" % round(tab1['SN'][gd][0][0],1))+'<TD>'+str("%.1f" % round(tab1['SN'][gd][0][1],1))+'<TD>'+str("%.1f" % round(tab1['SN'][gd][0][2],1))
            html.write('<TR><TD bgcolor="'+thcolor+'">S/N'+txt+'\n')
            txt='<TD>'+str("%.1f" % round(tab1['SNC'][gd][0][0],1))+'<TD>'+str("%.1f" % round(tab1['SNC'][gd][0][1],1))+'<TD>'+str("%.1f" % round(tab1['SNC'][gd][0][2],1))
            html.write('<TR><TD bgcolor="'+thcolor+'">S/N(C)'+txt+'\n')
    #        if tag_exist(tab1[i],'snratio'):
            html.write('<TR><TD bgcolor="'+thcolor+'">SN(E/C)<TD>'+str(np.round(tab1['SNRATIO'][gd][0],2))+'\n')
            html.write('</TABLE>\n')

            html.write('<TD><A HREF=plots/'+oneDfile+'_magplots.png target="_blank"><IMG SRC=plots/'+oneDfile+'_magplots.png WIDTH=210></A>\n')
            html.write('<TD><A HREF=plots/'+oneDfile+'_spatialresid.png target="_blank"><IMG SRC=plots/'+oneDfile+'_spatialresid.png WIDTH=250></A>\n')
            
            #html.write('<TD> <a href=plots/'+prefix+'telluric_'+cim+'_skyfit_CH4.jpg target="_blank"> <IMG SRC=plots/'+prefix+'telluric_'+cim+'_skyfit_CH4.jpg WIDTH=250></a>\n')
            #html.write('<TD> <a href=plots/'+prefix+'telluric_'+cim+'_skyfit_CO2.jpg target="_blank"> <IMG SRC=plots/'+prefix+'telluric_'+cim+'_skyfit_CO2.jpg WIDTH=250></a>\n')
            #html.write('<TD> <a href=plots/'+prefix+'telluric_'+cim+'_skyfit_H2O.jpg target="_blank"> <IMG SRC=plots/'+prefix+'telluric_'+cim+'_skyfit_H2O.jpg WIDTH=250></a>\n')
        else:
            html.write('<TR><TD bgcolor="'+thcolor+'">'+ims[i]+'\n')
            html.write('<TD><TD><TD><TD><TD><TD><TD><TD>\n')
    html.write('</table><HR>\n')
    
    html.write('<BR><BR>\n')
    html.write('</BODY></HTML>\n')
    html.close()


    if int(mjd)>59556:
        fps = True
    else:
        fps = False

    # Set up some basic plotting parameters, starting by turning off interactive plotting.
    matplotlib.use('agg')
    fontsize = 24;   fsz = fontsize * 0.75
    matplotlib.rcParams.update({'font.size':fontsize, 'font.family':'serif'})
    matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"
    alpha = 0.6
    axwidth=1.5
    axmajlen=7
    axminlen=3.5
    cmap = 'RdBu'

    # Read the plateSum file
    plSum1 = fits.getdata(plsumpath,1)
    platesum2 = fits.getdata(plsumpath,2)
    fibord = np.argsort(platesum2['FIBERID'])
    plSum2 = platesum2[fibord]
    nfiber = len(plSum2['H'])

    #----------------------------------------------------------------------------------------------
    # PLOTS 1-2: H versus S/N for the exposure-combined apVisit, second version colored by fiber block
    #----------------------------------------------------------------------------------------------
    Vsumfile = vispath + field + '/apVisitSum-' + plate + '-' + mjd + '.fits'
    Vsum = fits.getdata(Vsumfile,1) 
    block = np.floor((Vsum['FIBERID'] - 1) / 30) #[::-1]

    for i in range(2):
        plotfile = os.path.basename(Vsumfile).replace('Sum','SNR').replace('.fits','.png')
        if i == 1: plotfile = plotfile.replace('SNR','SNRblocks')
        print("----> makeObsPlots: Making "+plotfile)

        fig=plt.figure(figsize=(19,10))
        ax = plt.subplot2grid((1,1), (0,0))
        ax.tick_params(reset=True)
        ax.minorticks_on()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.set_xlabel(r'$H$ mag.')
        ax.set_ylabel(r'apVisit S/N')
        ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
        ax.tick_params(axis='both',which='major',length=axmajlen)
        ax.tick_params(axis='both',which='minor',length=axminlen)
        ax.tick_params(axis='both',which='both',width=axwidth)

        if 'H' in Vsum.columns.names:
            Harr = Vsum['H']
        else:
            Harr = Vsum['H']
        gd, = np.where((Harr > 0) & (Harr < 20) & (np.isnan(Vsum['SNR']) == False))
        ngd = len(gd)
        try:
            minH = np.nanmin(Harr[gd]);  maxH = np.nanmax(Harr[gd])
        except:
            minH = 6;  maxH = 14
        spanH = maxH - minH
        xmin = minH - spanH * 0.05;     xmax = maxH + spanH * 0.05

        try:
            minSNR = np.nanmin(Vsum['SNR']); maxSNR = np.nanmax(Vsum['SNR'])
        except:
            minSNR = 0;  maxSNR = 500
        spanSNR = maxSNR - minSNR
        ymin = -5;                       ymax = maxSNR + ((maxSNR - ymin) * 0.05)

        if fps:
            notsky, = np.where((Vsum['H'] > 5) & (Vsum['H'] < 15) & (np.isnan(Vsum['H']) == False) & 
                               (np.isnan(Vsum['SNR']) == False) & (Vsum['SNR'] > 0) & (Vsum['ASSIGNED']) & 
                               (Vsum['ON_TARGET']) & (Vsum['VALID']) & (Vsum['OBJTYPE'] != 'none'))
        else:
            notsky, = np.where((Vsum['H'] > 5) & (Vsum['H'] < 15) & (np.isnan(Vsum['H']) == False) & 
                               (np.isnan(Vsum['SNR']) == False) & (Vsum['SNR'] > 0))# & (Vsum['OBJTYPE'] != 'none'))

        if len(notsky) > 10:
            if i == 0:
                # First pass at fitting line to S/N as function of H
                H1 = Vsum['H'][notsky]
                sn1 = Vsum['SNR'][notsky]
                polynomial1 = np.poly1d(np.polyfit(H1, np.log10(sn1), 1))
                yarrnew1 = polynomial1(H1)
                diff1 = np.log10(sn1) - yarrnew1
                gd1, = np.where(diff1 > -np.nanstd(diff1))
                # Second pass at fitting line to S/N as function of H
                H2 = H1[gd1]
                sn2 = sn1[gd1]
                polynomial2 = np.poly1d(np.polyfit(H2, np.log10(sn2), 1))
                yarrnew2 = polynomial2(H2)
                diff2 = np.log10(sn2) - yarrnew2
                gd2, = np.where(diff2 > -np.nanstd(diff2))
                # Final pass at fitting line to S/N as function of H
                H3 = H2[gd2]
                sn3 = sn2[gd2]
                polynomial3 = np.poly1d(np.polyfit(H3, np.log10(sn3), 1))
                xarrnew3 = np.linspace(np.nanmin(H1), np.nanmax(H1), 5000)
                yarrnew3 = polynomial3(xarrnew3)

            ax.plot(xarrnew3, 10**yarrnew3, color='grey', linestyle='dashed')

        ax.set_xlim(xmin,xmax)
        ax.set_ylim(1,1200)
        ax.set_yscale('log')

        if fps:
            science, = np.where((Vsum['H'] > 0) & (Vsum['H'] < 16) & (np.isnan(Vsum['H']) == False) & 
                                (np.isnan(Vsum['SNR']) == False) & (Vsum['SNR'] > 0) & (Vsum['ASSIGNED']) & 
                                (Vsum['ON_TARGET']) & (Vsum['VALID']) & 
                                ((Vsum['OBJTYPE'] == 'OBJECT') | (Vsum['OBJTYPE'] == 'STAR')))

            telluric, = np.where((Vsum['H'] > 0) & (Vsum['H'] < 16) & (np.isnan(Vsum['H']) == False) & 
                                 (np.isnan(Vsum['SNR']) == False) & (Vsum['SNR'] > 0) & (Vsum['ASSIGNED']) & 
                                 (Vsum['ON_TARGET']) & (Vsum['VALID']) & 
                                 ((Vsum['OBJTYPE'] == 'SPECTROPHOTO_STD') | (Vsum['OBJTYPE'] == 'HOT_STD')))
        else:
            science, = np.where((Vsum['H'] > 0) & (Vsum['H'] < 16) & (np.isnan(Vsum['H']) == False) & 
                                (np.isnan(Vsum['SNR']) == False) & (Vsum['SNR'] > 0))

            #telluric, = np.where((Vsum['H'] > 0) & (Vsum['H'] < 16) & (np.isnan(Vsum['H']) == False) & 
            #                     (np.isnan(Vsum['SNR']) == False) & (Vsum['SNR'] > 0) &
            #                     ((Vsum['OBJTYPE'] == 'SPECTROPHOTO_STD') | (Vsum['OBJTYPE'] == 'HOT_STD')))

        x = Vsum['H'][science];  y = Vsum['SNR'][science]
        scicol = 'r'
        telcol = 'dodgerblue'
        if i == 1:
            scicol = block[science] + 0.5
            #telcol = block[telluric] + 0.5
        psci = ax.scatter(x, y, marker='*', s=400, edgecolors='white', alpha=0.8, c=scicol, cmap='tab10', vmin=0.5, vmax=10.5, label='Science')
        #x = Vsum['H'][telluric];  y = Vsum['SNR'][telluric]
        #ptel = ax.scatter(x, y, marker='o', s=150, edgecolors='white', alpha=0.8, c=telcol, cmap='tab10', vmin=0.5, vmax=10.5, label='Telluric')

        if i == 1:
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("right", size="2%", pad="1%")
            cb = colorbar(psci, cax=cax, orientation="vertical")
            #cax.xaxis.set_ticks_position("right")
            cax.yaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.text(1.09, 0.5, r'MTP #', ha='right', va='center', rotation=-90, transform=ax.transAxes)

        #ax.legend(loc='upper right', labelspacing=0.5, handletextpad=-0.1, facecolor='lightgrey')

        fig.subplots_adjust(left=0.06,right=0.945,bottom=0.09,top=0.975,hspace=0.2,wspace=0.0)
        plt.savefig(plotdir+plotfile)
        plt.close('all')

    #----------------------------------------------------------------------------------------------
    # PLOTS 3-6: flat field flux and fiber blocks... previously done by plotflux.pro
    #----------------------------------------------------------------------------------------------
    fluxfile = 'apFlux-a-' + fluxid + '.fits'
    gfluxpath = fluxpath + fluxfile
    ypos = 300 - platesum2['FIBERID']
    block = np.floor((plSum2['FIBERID'] - 1) / 30) #[::-1]

    plotfile = fluxfile.replace('.fits', '.png').replace('-a-','-')
    print("----> makeObsPlots: Making "+plotfile)

    fig=plt.figure(figsize=(35,8))
    plotrad = 1.6

    for ichip in range(nchips):
        chip = chips[ichip]

        ax = plt.subplot2grid((1,nchips+2), (0,ichip))
        ax.set_xlim(-plotrad, plotrad)
        ax.set_ylim(-plotrad, plotrad)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.minorticks_on()
        ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
        ax.tick_params(axis='both',which='major',length=axmajlen)
        ax.tick_params(axis='both',which='minor',length=axminlen)
        ax.tick_params(axis='both',which='both',width=axwidth)
        ax.set_xlabel(r'Zeta (deg.)')
        if ichip == 0: ax.set_ylabel(r'Eta (deg.)')
        if ichip != 0: ax.axes.yaxis.set_ticklabels([])

        if ichip == 0: flux = fits.getdata(gfluxpath)
        if ichip == 1: flux = fits.getdata(gfluxpath.replace('-a-','-b-'))
        if ichip == 2: flux = fits.getdata(gfluxpath.replace('-a-','-c-'))
        med = np.nanmedian(flux, axis=1)
        tput = med[ypos] / np.nanmax(med[ypos])
        gd, = np.where((ypos > 60) & (ypos < 90))
        sc = ax.scatter(platesum2['Zeta'], platesum2['Eta'], marker='o', s=100, c=tput, edgecolors='k', cmap='afmhot', alpha=1, vmin=0.01, vmax=0.99)

        ax.text(0.03, 0.97, chiplab[ichip]+'\n'+'chip', transform=ax.transAxes, ha='left', va='top', color=chiplab[ichip])

        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("top", size="4%", pad="1%")
        cb = colorbar(sc, cax=cax, orientation="horizontal")
        cax.xaxis.set_ticks_position("top")
        cax.minorticks_on()
        ax.text(0.5, 1.13, r'Dome Flat Throughput',ha='center', transform=ax.transAxes)

    ax1 = plt.subplot2grid((1,nchips+2), (0,nchips))
    ax1.set_xlim(-1.6,1.6)
    ax1.set_ylim(-1.6,1.6)
    ax1.axes.yaxis.set_ticklabels([])
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.minorticks_on()
    ax1.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
    ax1.tick_params(axis='both',which='major',length=axmajlen)
    ax1.tick_params(axis='both',which='minor',length=axminlen)
    ax1.tick_params(axis='both',which='both',width=axwidth)
    ax1.set_xlabel(r'Zeta (deg.)')

    sc = ax1.scatter(plSum2['Zeta'], plSum2['Eta'], marker='o', s=150, c=block+0.5, edgecolors='white', cmap='tab10', alpha=0.9, vmin=0.5, vmax=10.5)

    ax1_divider = make_axes_locatable(ax1)
    cax1 = ax1_divider.append_axes("top", size="4%", pad="1%")
    cb = colorbar(sc, cax=cax1, orientation="horizontal")
    cax1.xaxis.set_ticks_position("top")
    cax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax1.text(0.5, 1.13, r'MTP #', ha='center', transform=ax1.transAxes)

    ax2 = plt.subplot2grid((1,nchips+2), (0,nchips+1))
    ax2.set_xlim(-1.6,1.6)
    ax2.set_ylim(-1.6,1.6)
    ax2.axes.yaxis.set_ticklabels([])
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax2.minorticks_on()
    ax2.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
    ax2.tick_params(axis='both',which='major',length=axmajlen)
    ax2.tick_params(axis='both',which='minor',length=axminlen)
    ax2.tick_params(axis='both',which='both',width=axwidth)
    ax2.set_xlabel(r'Zeta (deg.)')

    if fps:
        notsky, = np.where((Vsum['H'] > 5) & (Vsum['H'] < 15) & (np.isnan(Vsum['H']) == False) & 
                           (np.isnan(Vsum['SNR']) == False) & (Vsum['SNR'] > 0) & (Vsum['ASSIGNED']) & 
                           (Vsum['ON_TARGET']) & (Vsum['VALID']) & (Vsum['OBJTYPE'] != 'none'))
    else:
        notsky, = np.where((Vsum['H'] > 5) & (Vsum['H'] < 15) & (np.isnan(Vsum['H']) == False) & 
                           (np.isnan(Vsum['SNR']) == False) & (Vsum['SNR'] > 0))# & (Vsum['OBJTYPE'] != 'none'))

    if len(notsky) > 10:
        # First pass at fitting line to S/N as function of H
        H1 = Vsum['H'][notsky]
        sn1 = Vsum['SNR'][notsky]
        polynomial1 = np.poly1d(np.polyfit(H1, np.log10(sn1), 1))
        yarrnew1 = polynomial1(H1)
        diff1 = np.log10(sn1) - yarrnew1
        gd1, = np.where(diff1 > -np.nanstd(diff1))
        # Second pass at fitting line to S/N as function of H
        H2 = H1[gd1]
        sn2 = sn1[gd1]
        polynomial2 = np.poly1d(np.polyfit(H2, np.log10(sn2), 1))
        yarrnew2 = polynomial2(H2)
        diff2 = np.log10(sn2) - yarrnew2
        gd2, = np.where(diff2 > -np.nanstd(diff2))
        # Final pass at fitting line to S/N as function of H
        H3 = H2[gd2]
        sn3 = sn2[gd2]
        polynomial3 = np.poly1d(np.polyfit(H3, np.log10(sn3), 1))
        xarrnew3 = np.linspace(np.nanmin(H1), np.nanmax(H1), 5000)
        yarrnew3 = polynomial3(xarrnew3)
        ratio = np.zeros(len(notsky))
        eta = np.full(len(notsky), -999.9)
        zeta = np.full(len(notsky), -999.9)
        for q in range(len(notsky)):
            hmdif = np.absolute(H1[q] - xarrnew3)
            pp, = np.where(hmdif == np.nanmin(hmdif))
            ratio[q] = sn1[q] / 10**yarrnew3[pp][0]
            g, = np.where(Vsum['APOGEE_ID'][notsky][q] == plSum2['TMASS_STYLE'])
            if len(g) > 0:
                eta[q] = plSum2['ETA'][g][0]
                zeta[q] = plSum2['ZETA'][g][0]

        #telluric, = np.where((eta > -900) & ((Vsum['OBJTYPE'][notsky] == 'SPECTROPHOTO_STD') | (Vsum['OBJTYPE'][notsky] == 'HOT_STD')))
        #if len(telluric) > 0:
        #    x = zeta[telluric]
        #    y = eta[telluric]
        #    c = ratio[telluric]
        #    l = 'telluric'
        #    sc = ax2.scatter(x, y, marker='o', s=100, c=c, cmap='CMRmap', edgecolors='k', vmin=0, vmax=1, linewidth=0.75, label=l)
        science, = np.where((eta > -900))# & ((Vsum['OBJTYPE'][notsky] == 'OBJECT') | (Vsum['OBJTYPE'][notsky] == 'STAR')))
        if len(science) > 0:
            x = zeta[science]
            y = eta[science]
            c = ratio[science]
            l = 'science'
            sc = ax2.scatter(x, y, marker='*', s=250, c=c, cmap='CMRmap', edgecolors='k', vmin=0, vmax=1, linewidth=0.75, label=l)

        ax1_divider = make_axes_locatable(ax2)
        cax1 = ax1_divider.append_axes("top", size="4%", pad="1%")
        cb = colorbar(sc, cax=cax1, orientation="horizontal")
        cax1.xaxis.set_ticks_position("top")
        #cax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax2.text(0.5, 1.13, r'obs SNR $/$ fit SNR', ha='center', transform=ax2.transAxes)
        #ax2.legend(loc='upper left', labelspacing=0.5, handletextpad=-0.1, facecolor='lightgrey', fontsize=fontsize*0.75)

    fig.subplots_adjust(left=0.03,right=0.99,bottom=0.098,top=0.90,hspace=0.09,wspace=0.07)
    plt.savefig(plotdir+plotfile)
    plt.close('all')

    # Loop over the exposures to make other plots.
    for i in range(n_exposures):
        gd, = np.where(int(ims[i]) == plSum1['IM'])
        if len(gd) >= 1:
            ii = gd[0]
            #------------------------------------------------------------------------------------------
            # PLOTS 8: 3 panel mag/SNR plots for each exposure
            #----------------------------------------------------------------------------------------------
            plotfile = 'ap1D-'+str(plSum1['IM'][ii])+'_magplots.png'
            print("----> makeObsPlots: Making "+plotfile)

            telluric, = np.where((plSum2['OBJTYPE'] == 'SPECTROPHOTO_STD') | (plSum2['OBJTYPE'] == 'HOT_STD'))
            ntelluric = len(telluric)
            science, = np.where((plSum2['OBJTYPE'] != 'SPECTROPHOTO_STD') & (plSum2['OBJTYPE'] != 'HOT_STD') & (plSum2['OBJTYPE'] != 'SKY'))
            nscience = len(science)
            sky, = np.where(plSum2['OBJTYPE'] == 'SKY')
            nsky = len(sky)

            notsky, = np.where((plSum2['H'] > 0) & (plSum2['H'] < 30))
            Harr = plSum2['H'][notsky]
            try:
                minH = np.nanmin(Harr);  maxH = np.nanmax(Harr)
            except:
                minH = 6;  maxH = 14
            xmin = minH - spanH * 0.05;      xmax = maxH + spanH * 0.05

            fig=plt.figure(figsize=(11,14))
            ax1 = plt.subplot2grid((3,1), (0,0))
            ax2 = plt.subplot2grid((3,1), (1,0))
            ax3 = plt.subplot2grid((3,1), (2,0))
            axes = [ax1, ax2, ax3]#, ax4, ax5]
            ax2.set_ylim(-10,1)

            for ax in axes:
                ax.set_xlim(xmin,xmax)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.minorticks_on()
                ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
                ax.tick_params(axis='both',which='major',length=axmajlen)
                ax.tick_params(axis='both',which='minor',length=axminlen)
                ax.tick_params(axis='both',which='both',width=axwidth)

            ax1.axes.xaxis.set_ticklabels([])
            ax2.axes.xaxis.set_ticklabels([])

            ax3.set_xlabel(r'$H$')
            ax1.text(-0.15,0.50,r'm = -2.5*log(counts)',transform=ax1.transAxes,rotation=90,ha='left',va='center')
            ax2.text(-0.15,0.50,r'$H$ - (m+zero)',transform=ax2.transAxes,rotation=90,ha='left',va='center')
            ax3.text(-0.15,0.50,r'S/N',transform=ax3.transAxes,rotation=90,ha='left',va='center')

            # PLOTS 8a: observed mag vs H mag
            x = plSum2['H'][science];    y = plSum2['obsmag'][science,1,ii]-plSum1['ZERO'][ii]
            ax1.scatter(x, y, marker='*', s=180, edgecolors='k', alpha=alpha, c='r', label='Science')
            if ntelluric>0:
                x = plSum2['H'][telluric];   y = plSum2['obsmag'][telluric,1,ii]-plSum1['ZERO'][ii]
                ax1.scatter(x, y, marker='o', s=60, edgecolors='k', alpha=alpha, c='dodgerblue', label='Telluric')
            ax1.legend(loc='upper left', labelspacing=0.5, handletextpad=-0.1, facecolor='lightgrey')

            # PLOTS 8b: observed mag - fit mag vs H mag
            x = plSum2['H'][science];    y = x - plSum2['obsmag'][science,1,ii]
            yminsci = np.nanmin(y); ymaxsci = np.nanmax(y)
            ax2.scatter(x, y, marker='*', s=180, edgecolors='k', alpha=alpha, c='r')
            if ntelluric>0:
                x = plSum2['H'][telluric];   y = x - plSum2['obsmag'][telluric,1,ii]
                ymintel = np.nanmin(y); ymaxtel = np.nanmax(y)
                ax2.scatter(x, y, marker='o', s=60, edgecolors='k', alpha=alpha, c='dodgerblue')
                ymin = np.min([yminsci,ymintel])
                ymax = np.max([ymaxsci,ymaxtel])
            else:
                ymin = yminsci
                ymax = ymaxsci
            yspan=ymax-ymin
            #ax2.set_ylim(ymin-(yspan*0.05),ymax+(yspan*0.05))
            ax2.set_ylim(-8,2)

            # PLOTS 8c: S/N as calculated from ap1D frame
            #c = ['r','g','b']
            #for ichip in range(nchips):
            #    x = plSum2['H'][science];   y = plSum2['SN'][science,i,ichip]
            #    ax3.semilogy(x, y, marker='*', ms=15, mec='k', alpha=alpha, mfc=c[ichip], linestyle='')
            #    x = plSum2['H'][telluric];   y = plSum2['SN'][telluric,i,ichip]
            #    ax3.semilogy(x, y, marker='o', ms=9, mec='k', alpha=alpha, mfc=c[ichip], linestyle='')
            x = plSum2['H'][science];   y = plSum2['SN'][science,1,ii]
            yminsci = np.nanmin(y); ymaxsci = np.nanmax(y)
            ax3.semilogy(x, y, marker='*', ms=15, mec='k', alpha=alpha, mfc='r', linestyle='')
            if ntelluric>0:
                x = plSum2['H'][telluric];   y = plSum2['SN'][telluric,1,ii]
                ymintel = np.nanmin(y); ymaxtel = np.nanmax(y)
                ax3.semilogy(x, y, marker='o', ms=9, mec='k', alpha=alpha, mfc='dodgerblue', linestyle='')
                ymin = np.min([yminsci,ymintel])
                ymax = np.max([ymaxsci,ymaxtel])
            else:
                ymin = yminsci
                ymax = ymaxsci
            if np.isfinite(ymin)==False:
                ymin = 1.0
            if np.isfinite(ymax)==False:
                ymax = 200.0
            yspan=ymax-ymin
            ax3.set_ylim(ymin-(yspan*0.05),ymax+(yspan*0.05))

            # overplot the target S/N line
            cframepath = gvispath + 'apCframe-a-' + ims[ii] + '.fits'
            try:
                hdr = fits.getheader(cframepath)
                sntarget = 100 * np.sqrt(hdr['EXPTIME'] / (3.0 * 3600))
                sntargetmag = 12.2
                x = [sntargetmag - 10, sntargetmag + 2.5];    y = [sntarget * 100, sntarget / np.sqrt(10)]
                ax3.plot(x, y, color='k',linewidth=1.5)
            except: pass

            fig.subplots_adjust(left=0.14,right=0.978,bottom=0.08,top=0.99,hspace=0.2,wspace=0.0)
            plt.savefig(plotdir+plotfile)
            plt.close('all')

            #------------------------------------------------------------------------------------------
            # PLOT 9: spatial residuals for each exposure
            #----------------------------------------------------------------------------------------------
            plotfile = 'ap1D-'+str(plSum1['IM'][ii])+'_spatialresid.png'
            print("----> makeObsPlots: Making "+plotfile)

            fig=plt.figure(figsize=(14,15))
            ax1 = plt.subplot2grid((1,1), (0,0))
            ax1.set_xlim(-1.6,1.6)
            ax1.set_ylim(-1.6,1.6)
            ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            ax1.minorticks_on()
            ax1.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
            ax1.tick_params(axis='both',which='major',length=axmajlen)
            ax1.tick_params(axis='both',which='minor',length=axminlen)
            ax1.tick_params(axis='both',which='both',width=axwidth)
            ax1.set_xlabel(r'Zeta (deg.)');  ax1.set_ylabel(r'Eta (deg.)')
            #cmap = plt.get_cmap('jet');    minval = 0.05;    maxval = 0.92;    ncol = 100
            #gdcmap = mplcolors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, 
            #           a=minval, b=maxval), cmap(np.linspace(minval, maxval, ncol)))

            try:
                ass, = np.where(plSum2['ASSIGNED'][science])
                x = plSum2['ZETA'][science][ass];    y = plSum2['ETA'][science][ass]
                c = plSum2['H'][science][ass] - plSum2['obsmag'][science[ass],1,ii]
            except:
                x = plSum2['ZETA'][science];    y = plSum2['ETA'][science]
                c = plSum2['H'][science] - plSum2['obsmag'][science,1,ii]
            psci = ax1.scatter(x, y, marker='*', s=400, c=c, edgecolors='k', cmap=cmap, alpha=1, vmin=-0.5, vmax=0.5, label='Science')

            if ntelluric>0:
                try:
                    ass, = np.where(plSum2['ASSIGNED'][telluric])
                    x = plSum2['ZETA'][telluric][ass];    y = plSum2['ETA'][telluric][ass]
                    c = plSum2['H'][telluric][ass] - plSum2['obsmag'][telluric[ass],1,ii]
                except:
                    x = plSum2['ZETA'][telluric];    y = plSum2['ETA'][telluric]
                    c = plSum2['H'][telluric] - plSum2['obsmag'][telluric,1,ii]
                ptel = ax1.scatter(x, y, marker='o', s=215, c=c, edgecolors='k', cmap=cmap, alpha=1, vmin=-0.5, vmax=0.5, label='Telluric')

            #try:
            #    x = plSum2['ZETA'][sky];    y = plSum2['ETA'][sky]
            #    c = plSum2['H'][sky] - plSum2['obsmag'][sky,i,1]
            #    psky = ax1.scatter(x, y, marker='s', s=140, c='white', edgecolors='k', alpha=1, label='Sky')
            #except:
            #    print("----> makeObsPlots: Problem!!! Sky fiber subscripting error when trying to make spatial mag. plots.")

            ax1.legend(loc='upper left', labelspacing=0.5, handletextpad=-0.1, facecolor='lightgrey')

            ax1_divider = make_axes_locatable(ax1)
            cax1 = ax1_divider.append_axes("top", size="4%", pad="1%")
            cb = colorbar(psci, cax=cax1, orientation="horizontal")
            cax1.xaxis.set_ticks_position("top")
            cax1.minorticks_on()
            ax1.text(0.5, 1.12, r'$H$ + 2.5*log(m - zero)',ha='center', transform=ax1.transAxes)

            fig.subplots_adjust(left=0.11,right=0.97,bottom=0.07,top=0.91,hspace=0.2,wspace=0.0)
            plt.savefig(plotdir+plotfile)
            plt.close('all')


            #----------------------------------------------------------------------------------------------
            # PLOT 7: make plot of sky levels for this plate
            # https://data.sdss.org/sas/apogeework/apogee/spectro/redux/current/exposures/apogee-n/56257/plots/56257sky.gif
            #----------------------------------------------------------------------------------------------
            #skyfile = 'sky-'+gfile
            #print("PLOTS 7: Sky level plots will be made here.")

            #----------------------------------------------------------------------------------------------
            # PLOT 8: make plot of zeropoints for this plate
            # https://data.sdss.org/sas/apogeework/apogee/spectro/redux/current/exposures/apogee-n/56257/plots/56257zero.gif
            #----------------------------------------------------------------------------------------------
            #zerofile = 'zero-'+gfile
            #print("PLOTS 8: Zeropoints plots will be made here.")

    #plt.ion()
    print("----> makeObsPlots: Done with plate "+plate+", MJD "+mjd+"\n")

###################################################################################################
''' GETFLUX: Translation of getflux.pro '''
def getflux(d=None, skyline=None, rows=None):

    chips = np.array(['a','b','c'])
    nnrows = len(rows)

    try:
        ### NOTE:pretty sure that f[2047,150] subscript won't work, but 150,2057 will. Hoping for the best.
        if skyline['W1'] > d['a'][4].data[150,2047]:
            ichip = 0
        else:
            if skyline['W1'] > d['b'][4].data[150,2047]:
                ichip = 1
            else:
                ichip = 2

        cont = np.zeros(nnrows)
        line = np.zeros(nnrows)
        nline = np.zeros(nnrows)

        for i in range(nnrows):
            wave = d[chips[ichip]][4].data[rows[i],:]
            flux = d[chips[ichip]][1].data

            icont, = np.where(((wave > skyline['C1']) & (wave < skyline['C2'])) | 
                              ((wave < skyline['C3']) & (wave < skyline['C4'])))

            #import pdb; pdb.set_trace()
            if len(icont) >= 0: cont[i] = np.median(flux[rows[i],icont])

            iline, = np.where((wave > skyline['W1']) & (wave < skyline['W2']))

            if len(iline) >= 0:
                line[i] = np.nansum(flux[rows[i],iline])
                nline[i] = np.nansum(flux[rows[i],iline] / flux[rows[i],iline])

        skylineFlux = line - (nline * cont)
        if skyline['TYPE'] == 0: skylineFlux /= cont

        return skylineFlux

    except:
        return 0


