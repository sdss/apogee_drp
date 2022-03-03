import sys
import glob
import os
import subprocess
import math
import time
import numpy as np
from pathlib import Path
from astropy.io import fits, ascii
from astropy.table import Table, vstack
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.lib.recfunctions import append_fields, merge_arrays
from astroplan import moon_illumination
from astropy.coordinates import SkyCoord, get_moon
from astropy import units as astropyUnits
from scipy.signal import medfilt2d as ScipyMedfilt2D
from apogee_drp.utils import plan,apload,yanny,plugmap,platedata,bitmask
from apogee_drp.apred import wave
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
from matplotlib import cm as cmaps
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.axes_grid1.colorbar import colorbar
from datetime import date,datetime

# import pdb; pdb.set_trace()

apred = 'daily'
instrument = 'apogee-n'
telescope = 'apo25m'
fiberdaysbin=20

chips = np.array(['blue','green','red'])
nchips = len(chips)
fibers = np.array([10,80,150,220,290])[::-1]
nfibers = len(fibers)
nlines = 2
nquad = 4

load = apload.ApLoad(apred=apred, telescope=telescope)

# Establish  directories... hardcode sdss4/apogee2 for now
specdir4 = '/uufs/chpc.utah.edu/common/home/sdss/apogeework/apogee/spectro/redux/current/'
sdir4 = '/uufs/chpc.utah.edu/common/home/sdss/apogeework/apogee/spectro/redux/current/monitor/' + instrument + '/'

specdir5 = os.environ.get('APOGEE_REDUX') + '/' + apred + '/'
sdir5 = specdir5 + 'monitor/' + instrument + '/'

allexp4 =  fits.open(specdir4 + instrument + 'Exp.fits')[1].data
allsci4 =  fits.open(specdir4 + instrument + 'Sci.fits')[1].data
allsnr4 = fits.getdata(specdir5 + 'monitor/' + instrument + 'SNR_ap1-2.fits')

# Read in the master summary files
allcal =  fits.getdata(specdir5 + 'monitor/' + instrument + 'Cal.fits', 1)
alldark = fits.getdata(specdir5 + 'monitor/' + instrument + 'Cal.fits', 2)
allexp =  fits.getdata(specdir5 + 'monitor/' + instrument + 'Exp.fits', 1)
allsci =  fits.getdata(specdir5 + 'monitor/' + instrument + 'Sci.fits', 1)
allsnr = fits.getdata(specdir5 + 'monitor/' + instrument + 'SNR.fits')
dometrace = fits.getdata(specdir5 + 'monitor/' + instrument + 'DomeFlatTrace-all.fits')
quartztrace = fits.getdata(specdir5 + 'monitor/' + instrument + 'QuartzFlatTrace-all.fits')

badComObs = ascii.read(sdir5 + 'commisData2ignore.dat')

###############################################################################################
# Find the different cals
thar, = np.where(allcal['THAR'] == 1)
une, =  np.where(allcal['UNE'] == 1)
qrtz, = np.where(allcal['QRTZ'] == 1)
dome, = np.where(allexp['IMAGETYP'] == 'DomeFlat')
qrtzexp, = np.where(allexp['IMAGETYP'] == 'QuartzFlat')
dark, = np.where(alldark['EXPTYPE'] == 'DARK')

###############################################################################################
# Set up some basic plotting parameters
matplotlib.use('agg')
fontsize = 24;   fsz = fontsize * 0.75
matplotlib.rcParams.update({'font.size':fontsize, 'font.family':'serif'})
matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"
bboxpar = dict(facecolor='white', edgecolor='none', alpha=0.9)
axwidth = 1.5
axmajlen = 7
axminlen = 3.5
alf = 0.6
markersz = 1
colors = np.array(['midnightblue', 'deepskyblue', 'mediumorchid', 'red', 'orange'])[::-1]
colors1 = np.array(['k', 'b', 'r', 'gold'])
colors2 = np.array(['dodgerblue', 'seagreen', 'orange'])
fibers = np.array([10, 80, 150, 220, 290])[::-1]
nplotfibs = len(fibers)
#years = np.array([2011, 2012, 2013, 2014

tmp = allcal[qrtz]
caljd = tmp['JD'] - 2.4e6
t = Time(tmp['JD'], format='jd')
years = np.unique(np.floor(t.byear)) + 1
cyears = years.astype(int).astype(str)
nyears = len(years)
t = Time(years, format='byear')
yearjd = t.jd - 2.4e6
minjd = np.min(caljd)
maxjd = np.max(caljd)
jdspan = maxjd - minjd
xmin = minjd - jdspan * 0.01
xmax = maxjd + jdspan * 0.08
xspan = xmax-xmin

###########################################################################################
def snhistory3():
    # snhistory3.png
    plotfile = specdir5 + 'monitor/' + instrument + '/snhistory3.png'
    print("----> commissNplots: Making " + os.path.basename(plotfile))

    snbin = 10
    magmin = '10.8'
    magmax = '11.2'

    medesnrG = np.nanmedian(allsnr['ESNBINS'][:,10,1])
    medesnrG = np.nanstd(allsnr['ESNBINS'][:,10,1])
    limesnrG = medesnrG + 2*medesnrG
    gd, = np.where((allsnr['NSNBINS'][:, snbin] > 5) & (allsnr['SNBINS'][:, snbin, 1] > 0) & (allsnr['ESNBINS'][:, snbin, 1] < limesnrG))
    allsnrg = allsnr[gd]
    ngd = len(allsnrg)

    ymin = -0.01
    ymax = 0.18
    yspan = ymax-ymin

    fig = plt.figure(figsize=(30,14))

    for ichip in range(nchips):
        chip = chips[ichip]
        ax = plt.subplot2grid((nchips,1), (ichip,0))
        ax.set_xlim(xmin, xmax)
        #ax.set_ylim(ymin, ymax)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
        ax.minorticks_on()
        ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
        ax.tick_params(axis='both',which='major',length=axmajlen)
        ax.tick_params(axis='both',which='minor',length=axminlen)
        ax.tick_params(axis='both',which='both',width=axwidth)
        if ichip == nchips-1: ax.set_xlabel(r'JD - 2,400,000')
        #if ichip == 1: ax.set_ylabel(r'S/N$^{2}$ per minute ($' + magmin + r'>=H>' + magmax + r'$)')
        if ichip == 1: ax.text(-0.035, 0.5, r'S/N per minute ($' + magmin + r'>=H>' + magmax + r'$)', transform=ax.transAxes, rotation=90, ha='right', va='center')
        if ichip < nchips-1: ax.axes.xaxis.set_ticklabels([])
        ax.axvline(x=59146, color='r', linewidth=2)

        xvals = allsnrg['JD']
        yvals = (allsnrg['MEDSNBINS'][:, snbin, 2-ichip]) / (allsnrg['EXPTIME'] / 60)
        #pdb.set_trace()
        #if ichip == 0: pdb.set_trace()
        scolors = allsnrg['MOONPHASE']
        sc1 = ax.scatter(xvals, yvals, marker='o', s=markersz, c=scolors, cmap='copper')#, c=colors[ifib], alpha=alf)#, label='Fiber ' + str(fibers[ifib]))

        ax.text(0.97,0.92,chip.capitalize() + '\n' + 'Chip', transform=ax.transAxes, 
                ha='center', va='top', color=chip, bbox=bboxpar)

        if ichip == 0: ylims = ax.get_ylim()
        for iyear in range(nyears):
            ax.axvline(x=yearjd[iyear], color='k', linestyle='dashed', alpha=alf)
            if ichip == 0: ax.text(yearjd[iyear], ylims[1]+((ylims[1]-ylims[0])*0.025), cyears[iyear], ha='center')

        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="2%", pad="1%")
        cb1 = colorbar(sc1, cax=cax, orientation="vertical")
        cax.minorticks_on()
        cax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        if ichip == 1: ax.text(1.06, 0.5, r'Moon Phase',ha='left', va='center', rotation=-90, transform=ax.transAxes)

    fig.subplots_adjust(left=0.05,right=0.95,bottom=0.06,top=0.96,hspace=0.08,wspace=0.00)
    plt.savefig(plotfile)
    plt.close('all')

    return

###########################################################################################
def dillum(mjdstart=59604):
    # HTML header background color
    thcolor = '#DCDCDC'

    htmlfile = specdir5 + 'monitor/' + instrument + '/fiber2fiber/fiber2fiber.html'
    html = open(htmlfile, 'w')
    html.write('<HTML><HEAD><title>'+os.path.basename(htmlfile).replace('.html','')+'</title></head><BODY>\n')
    html.write('<H1> Fiber-to-fiber Throughput Investigation (using dome flats)</H1>\n')
    html.write('<HR>\n')

    ################################3
    html.write('<H3>Night-to-night variation:</H3>\n')
    html.write('<P>The following plots show dome flat fluxes since MJD 59590. The leftmost columns of plots, ')
    html.write('shows the median flux in spatial direction over all fibers, taken in pixel slices as indicated ')
    html.write('in the first column of the table. Due to the bimodal fluxes (i.e. green versus blue+red fluxes), ')
    html.write('the fluxes are normalized by the maximum flux in the "Median Flux / Max Flux" column. The rightmost ')
    html.write('column of plots then has the mean flux across all dome flats for each fiber divided out to highlight ')
    html.write('the fiber-to-fiber throughput variations. The Mean Absolute Deviation is given, indicating that ')
    html.write('night-to-night variations are typically on the 1.5% level.</P>\n')
    html.write('<BR>\n')

    html.write('<TABLE BORDER=2>\n')
    html.write('<TR bgcolor="'+thcolor+'"><TH>Pixels <TH>Median Flux <TH>Median Flux / Max Flux <TH>Median Flux / Max Flux / Overall Mean Flux \n')

    html.write('<TR><TD align="center" bgcolor="'+thcolor+'">all\n')
    pfile = '../dillum_FPSonly_0-2047.png'
    pfile1 = pfile.replace('.png','_norm.png')
    pfile2 = pfile1.replace('.png','_resid.png')
    html.write('<TD><A HREF=' + pfile + ' target="_blank"><IMG SRC=' + pfile + ' HEIGHT=400></A>\n')
    html.write('<TD><A HREF=' + pfile1 + ' target="_blank"><IMG SRC=' + pfile1 + ' HEIGHT=400></A>\n')
    html.write('<TD><A HREF=' + pfile2 + ' target="_blank"><IMG SRC=' + pfile2 + ' HEIGHT=400></A>\n')
    #html.write('<TR><TD align="center" bgcolor="'+thcolor+'">224:624\n')
    #pfile = '../dillum_FPSonly_224-624.png'
    #pfile1 = pfile.replace('.png','_norm.png')
    #pfile2 = pfile1.replace('.png','_resid.png')
    #html.write('<TD><A HREF=' + pfile + ' target="_blank"><IMG SRC=' + pfile + ' HEIGHT=400></A>\n')
    #html.write('<TD><A HREF=' + pfile1 + ' target="_blank"><IMG SRC=' + pfile1 + ' HEIGHT=400></A>\n')
    #html.write('<TD><A HREF=' + pfile2 + ' target="_blank"><IMG SRC=' + pfile2 + ' HEIGHT=400></A>\n')
    html.write('<TR><TD align="center" bgcolor="'+thcolor+'">824:1224\n')
    pfile = '../dillum_FPSonly_824-1224.png'
    pfile1 = pfile.replace('.png','_norm.png')
    pfile2 = pfile1.replace('.png','_resid.png')
    html.write('<TD><A HREF=' + pfile + ' target="_blank"><IMG SRC=' + pfile + ' HEIGHT=400></A>\n')
    html.write('<TD><A HREF=' + pfile1 + ' target="_blank"><IMG SRC=' + pfile1 + ' HEIGHT=400></A>\n')
    html.write('<TD><A HREF=' + pfile2 + ' target="_blank"><IMG SRC=' + pfile2 + ' HEIGHT=400></A>\n')
    #html.write('<TR><TD align="center" bgcolor="'+thcolor+'">1424:1824\n')
    #pfile = '../dillum_FPSonly_1424-1824.png'
    #pfile1 = pfile.replace('.png','_norm.png')
    #pfile2 = pfile1.replace('.png','_resid.png')
    #html.write('<TD><A HREF=' + pfile + ' target="_blank"><IMG SRC=' + pfile + ' HEIGHT=400></A>\n')
    #html.write('<TD><A HREF=' + pfile1 + ' target="_blank"><IMG SRC=' + pfile1 + ' HEIGHT=400></A>\n')
    #html.write('<TD><A HREF=' + pfile2 + ' target="_blank"><IMG SRC=' + pfile2 + ' HEIGHT=400></A>\n')
    html.write('</table><BR><BR><HR>\n')

    ################################3
    html.write('<H3>Night-to-night variation (no normalization):</H3>\n')
    html.write('<P>On the other hand, if we skip the normalization step, variations are on the 3.2% level.</P>\n')

    html.write('<TABLE BORDER=2>\n')
    html.write('<TR bgcolor="'+thcolor+'"><TH>Pixels <TH>Median Flux <TH>Median Fiber Flux / Overall Mean Flux \n')

    html.write('<TR><TD align="center" bgcolor="'+thcolor+'">all\n')
    pfile = '../dillum_FPSonly_0-2047.png'
    pfile1 = pfile.replace('.png','_resid.png')
    html.write('<TD><A HREF=' + pfile + ' target="_blank"><IMG SRC=' + pfile + ' HEIGHT=400></A>\n')
    html.write('<TD><A HREF=' + pfile1 + ' target="_blank"><IMG SRC=' + pfile1 + ' HEIGHT=400></A>\n')
    #html.write('<TR><TD align="center" bgcolor="'+thcolor+'">224:624\n')
    #pfile = '../dillum_FPSonly_224-624.png'
    #pfile1 = pfile.replace('.png','_resid.png')
    #html.write('<TD><A HREF=' + pfile + ' target="_blank"><IMG SRC=' + pfile + ' HEIGHT=400></A>\n')
    #html.write('<TD><A HREF=' + pfile1 + ' target="_blank"><IMG SRC=' + pfile1 + ' HEIGHT=400></A>\n')
    html.write('<TR><TD align="center" bgcolor="'+thcolor+'">824:1224\n')
    pfile = '../dillum_FPSonly_824-1224.png'
    pfile1 = pfile.replace('.png','_resid.png')
    html.write('<TD><A HREF=' + pfile + ' target="_blank"><IMG SRC=' + pfile + ' HEIGHT=400></A>\n')
    html.write('<TD><A HREF=' + pfile1 + ' target="_blank"><IMG SRC=' + pfile1 + ' HEIGHT=400></A>\n')
    #html.write('<TR><TD align="center" bgcolor="'+thcolor+'">1424:1824\n')
    #pfile = '../dillum_FPSonly_1424-1824.png'
    #pfile1 = pfile.replace('.png','_resid.png')
    #html.write('<TD><A HREF=' + pfile + ' target="_blank"><IMG SRC=' + pfile + ' HEIGHT=400></A>\n')
    #html.write('<TD><A HREF=' + pfile1 + ' target="_blank"><IMG SRC=' + pfile1 + ' HEIGHT=400></A>\n')
    html.write('</table><BR><BR><HR>\n')


    ################################3
    html.write('<H3>Nightly variation using dome flat sequence from 59557:</H3>\n')
    html.write('<P>These plots are similar to the above, but looking only at the series of ')
    html.write('36 dome flats taken on MJD 59557. If we normalize by maximum flux, fiber-to-fiber ')
    html.write('variations over the course of a night are on the ~1.5% level.</P>\n')

    html.write('<TABLE BORDER=2>\n')
    html.write('<TR bgcolor="'+thcolor+'"><TH>Pixels <TH>Median Flux <TH>Median Flux / Max Flux <TH>Median Flux / Max Flux / Overall Mean Flux \n')

    html.write('<TR><TD align="center" bgcolor="'+thcolor+'">all\n')
    pfile = '../dillum59557_0-2047.png'
    pfile1 = pfile.replace('.png','_norm.png')
    pfile2 = pfile1.replace('.png','_resid.png')
    html.write('<TD><A HREF=' + pfile + ' target="_blank"><IMG SRC=' + pfile + ' HEIGHT=400></A>\n')
    html.write('<TD><A HREF=' + pfile1 + ' target="_blank"><IMG SRC=' + pfile1 + ' HEIGHT=400></A>\n')
    html.write('<TD><A HREF=' + pfile2 + ' target="_blank"><IMG SRC=' + pfile2 + ' HEIGHT=400></A>\n')
    #html.write('<TR><TD align="center" bgcolor="'+thcolor+'">224:624\n')
    #pfile = '../dillum59557_224-624.png'
    #pfile1 = pfile.replace('.png','_norm.png')
    #pfile2 = pfile1.replace('.png','_resid.png')
    #html.write('<TD><A HREF=' + pfile + ' target="_blank"><IMG SRC=' + pfile + ' HEIGHT=400></A>\n')
    #html.write('<TD><A HREF=' + pfile1 + ' target="_blank"><IMG SRC=' + pfile1 + ' HEIGHT=400></A>\n')
    #html.write('<TD><A HREF=' + pfile2 + ' target="_blank"><IMG SRC=' + pfile2 + ' HEIGHT=400></A>\n')
    html.write('<TR><TD align="center" bgcolor="'+thcolor+'">824:1224\n')
    pfile = '../dillum59557_824-1224.png'
    pfile1 = pfile.replace('.png','_norm.png')
    pfile2 = pfile1.replace('.png','_resid.png')
    html.write('<TD><A HREF=' + pfile + ' target="_blank"><IMG SRC=' + pfile + ' HEIGHT=400></A>\n')
    html.write('<TD><A HREF=' + pfile1 + ' target="_blank"><IMG SRC=' + pfile1 + ' HEIGHT=400></A>\n')
    html.write('<TD><A HREF=' + pfile2 + ' target="_blank"><IMG SRC=' + pfile2 + ' HEIGHT=400></A>\n')
    #html.write('<TR><TD align="center" bgcolor="'+thcolor+'">1424:1824\n')
    #pfile = '../dillum59557_1424-1824.png'
    #pfile1 = pfile.replace('.png','_norm.png')
    #pfile2 = pfile1.replace('.png','_resid.png')
    #html.write('<TD><A HREF=' + pfile + ' target="_blank"><IMG SRC=' + pfile + ' HEIGHT=400></A>\n')
    #html.write('<TD><A HREF=' + pfile1 + ' target="_blank"><IMG SRC=' + pfile1 + ' HEIGHT=400></A>\n')
    #html.write('<TD><A HREF=' + pfile2 + ' target="_blank"><IMG SRC=' + pfile2 + ' HEIGHT=400></A>\n')
    html.write('</table><BR><BR><HR>\n')

    ################################3
    html.write('<H3>Nightly variation using dome flat sequence from 59557 (no normalization):</H3>\n')
    html.write('<P>If we apply no normalization, nightly varations are on the ~0.5% level.</P>\n')

    html.write('<TABLE BORDER=2>\n')
    html.write('<TR bgcolor="'+thcolor+'"><TH>Pixels <TH>Median Flux <TH>Median Fiber Flux / Overall Mean Flux \n')

    html.write('<TR><TD align="center" bgcolor="'+thcolor+'">all\n')
    pfile = '../dillum59557_0-2047.png'
    pfile1 = pfile.replace('.png','_resid.png')
    html.write('<TD><A HREF=' + pfile + ' target="_blank"><IMG SRC=' + pfile + ' HEIGHT=400></A>\n')
    html.write('<TD><A HREF=' + pfile1 + ' target="_blank"><IMG SRC=' + pfile1 + ' HEIGHT=400></A>\n')
    #html.write('<TR><TD align="center" bgcolor="'+thcolor+'">224:624\n')
    #pfile = '../dillum59557_224-624.png'
    #pfile1 = pfile.replace('.png','_resid.png')
    #html.write('<TD><A HREF=' + pfile + ' target="_blank"><IMG SRC=' + pfile + ' HEIGHT=400></A>\n')
    #html.write('<TD><A HREF=' + pfile1 + ' target="_blank"><IMG SRC=' + pfile1 + ' HEIGHT=400></A>\n')
    html.write('<TR><TD align="center" bgcolor="'+thcolor+'">824:1224\n')
    pfile = '../dillum59557_824-1224.png'
    pfile1 = pfile.replace('.png','_resid.png')
    html.write('<TD><A HREF=' + pfile + ' target="_blank"><IMG SRC=' + pfile + ' HEIGHT=400></A>\n')
    html.write('<TD><A HREF=' + pfile1 + ' target="_blank"><IMG SRC=' + pfile1 + ' HEIGHT=400></A>\n')
    #html.write('<TR><TD align="center" bgcolor="'+thcolor+'">1424:1824\n')
    #pfile = '../dillum59557_1424-1824.png'
    #pfile1 = pfile.replace('.png','_resid.png')
    #html.write('<TD><A HREF=' + pfile + ' target="_blank"><IMG SRC=' + pfile + ' HEIGHT=400></A>\n')
    #html.write('<TD><A HREF=' + pfile1 + ' target="_blank"><IMG SRC=' + pfile1 + ' HEIGHT=400></A>\n')
    html.write('</table><BR><BR><HR>\n')

    html.write('<BR><BR>\n')
    html.write('</BODY></HTML>\n')
    html.close()

###########################################################################################
def dillum_FPSonly(mjdstart=59604, pix=[0, 2047], norm=True, resid=True):
    # dillum_FPSonly.png
    # Time series plot of median dome flat flux from cross sections across fibers

    plotfile = specdir5 + 'monitor/' + instrument + '/dillum_FPSonly_' + str(pix[0]) + '-' + str(pix[1]) + '.png'
    ylabel = r'Median Flux'
    if norm:
        plotfile = plotfile.replace('.png', '_norm.png')
        ylabel = r'Median Flux  /  Max Flux'
    if resid:
        plotfile = plotfile.replace('.png', '_resid.png')
        ylabel = r'Median Fiber Flux  /  Max Fiber Flux  /  Overall Mean Flux'

    print("----> commissNplots: Making " + os.path.basename(plotfile))

    fig = plt.figure(figsize=(30,22))
    xarr = np.arange(0, 300, 1) + 1

    coltickval = 5
    if mjdstart > 59590: coltickval = 2
    gd, = np.where((allexp[dome]['MJD'] >= mjdstart) & (allexp[dome]['MJD'] != 59557) & (allexp[dome]['MJD'] != 59566) & (allexp[dome]['NUM'] != 40580049))
    gdcal = allexp[dome][gd]
    umjd = gdcal['MJD']
    ndome = len(gdcal)

    mycmap = 'inferno_r'
    mycmap = 'brg_r'
    cmap = cmaps.get_cmap(mycmap, ndome)
    sm = cmaps.ScalarMappable(cmap=mycmap, norm=plt.Normalize(vmin=np.min(umjd), vmax=np.max(umjd)))

    txt = 'median over pixel range ' + str(pix[0]) + ':' + str(pix[1])

    for ichip in range(nchips):
        chip = chips[ichip]
        ax = plt.subplot2grid((nchips, 1), (ichip, 0))
        ax.set_xlim(0, 301)
        if resid: ax.set_ylim(0.75, 1.25)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.minorticks_on()
        ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
        ax.tick_params(axis='both',which='major',length=axmajlen)
        ax.tick_params(axis='both',which='minor',length=axminlen)
        ax.tick_params(axis='both',which='both',width=axwidth)
        if ichip == nchips-1: ax.set_xlabel(r'Fiber Index')
        if ichip == 1: ax.set_ylabel(ylabel)
        if ichip < nchips-1: ax.axes.xaxis.set_ticklabels([])
        if ichip == 0:
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("top", size="7%", pad="2%")
            cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
            cax.xaxis.set_ticks_position("top")
            #cax.minorticks_on()
            cax.xaxis.set_major_locator(ticker.MultipleLocator(coltickval))
            #cax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
            cax.xaxis.set_label_position('top') 
            cax.set_xlabel('MJD')
        ax.text(0.2, 0.19, txt, transform=ax.transAxes, ha='left', bbox=bboxpar)

        flux = np.zeros((ndome, len(xarr)))
        for idome in range(ndome):
            chp = 'c'
            if ichip == 1: chp = 'b'
            if ichip == 2: chp = 'a'
            file1d = load.filename('1D', mjd=str(umjd[idome]), num=gdcal['NUM'][idome], chips='c')
            file1d = file1d.replace('1D-', '1D-' + chp + '-')
            if os.path.exists(file1d):
                hdr = fits.getheader(file1d)
                oned = fits.getdata(file1d)
                flux[idome] = np.nanmedian(oned[:, pix[0]:pix[1]], axis=1)[::-1]
                flux[idome][74] = np.nanmean([flux[idome][72],flux[idome][73],flux[idome][75],flux[idome][76]])
                flux[idome][224] = np.nanmean([flux[idome][222],flux[idome][223],flux[idome][225],flux[idome][226]])
                if ichip == 0: 
                    print(str(umjd[idome])+'   '+str(gdcal['NUM'][idome])+'   '+str(int(round(np.max(flux[idome]))))+'  expt='+str(int(round(hdr['exptime'])))+'  nread='+str(hdr['nread']))
                mnf = np.nanmin(flux[idome][135:145])
                if (ichip == 0) & (mnf < 7500): print("BAD FLAT")
                mycolor = cmap(idome)
                if norm: flux[idome] = flux[idome] / np.nanmax(flux[idome])
                if resid is False: ax.plot(xarr, flux[idome], color=mycolor)

        gd, = np.where(np.nanmean(flux,axis=1) > 0)
        ndome = len(gd)
        if ndome > 0: flux=flux[gd]

        if resid:
            meanflux = np.nanmean(flux,axis=0)
            medflux = np.nanmedian(flux,axis=0)
            div = flux / meanflux
            divmed = flux / medflux
            for idome in range(ndome):
                mycolor = cmap(idome)
                ax.plot(xarr, divmed[idome], color=mycolor)
                #bd, = np.where(divmed[idome] < 0.85)
                #if len(bd) > 0: print(bd)

            medresid = np.nanmedian(np.absolute(divmed))
            medresidpercent = (medresid / np.nanmedian(meanflux))*100
            madresid = dln.mad(div)
            madresidpercent = (madresid / np.nanmean(div))*100
            txt1 = ''#med = ' + str("%.1f" % round(medresid, 1)) + ' (' + str("%.1f" % round(medresidpercent, 1)) + '%)'
            txt2 = 'MAD = ' + str("%.3f" % round(madresid, 3)) + ' (' + str("%.3f" % round(madresidpercent, 3)) + '%)'
            #ax.text(0.1, 0.15, txt1+',   '+txt2, transform=ax.transAxes, ha='left')
            ax.text(0.2, 0.10, txt2, transform=ax.transAxes, ha='left', bbox=bboxpar)

        ax.text(0.97,0.06,chip.capitalize() + '\n' + 'Chip', transform=ax.transAxes, 
                ha='center', va='bottom', color=chip, bbox=bboxpar)

    fig.subplots_adjust(left=0.06,right=0.985,bottom=0.045,top=0.955,hspace=0.08,wspace=0.1)
    plt.savefig(plotfile)
    plt.close('all')

###########################################################################################
def dillum2(mjdstart=59557, mjdmean=False, chip=2, do59557=False):
    # dillum_FPSonly.png
    # Time series plot of median dome flat flux from cross sections across fibers

    fsize = 32
    matplotlib.rcParams.update({'font.size':fsize, 'font.family':'serif'})


    if chip == 0: schip = 'a'
    if chip == 1: schip = 'b'
    if chip == 2: schip = 'c'

    plotfile = specdir5 + 'monitor/' + instrument + '/fiber2fiber/tputVar-' + schip + '.png'
    if do59557: plotfile = plotfile.replace('tputVar', 'tputVar59557')
    ylabels = np.array(['Flux', 'Norm Flux', 'Norm Flux / Med Flux', '% Variation'])

    print("----> commissNplots: Making " + os.path.basename(plotfile))

    fig = plt.figure(figsize=(28,28))
    xarr = np.arange(0, 300, 1) + 1

    coltickval = 5
    if mjdstart > 59590: coltickval = 2
    gd, = np.where((allexp[dome]['MJD'] >= mjdstart) & (allexp[dome]['MJD'] != 59566) & (allexp[dome]['NUM'] != 40580049))# & (allexp[dome]['MJD'] != 59557))
    if do59557: gd, = np.where(allexp['MJD'][dome] == 59557)
    gdcal = allexp[dome][gd]
    umjd = gdcal['MJD']
    if mjdmean: umjd = np.unique(umjd)
    nmjd = len(umjd)
    ndome = len(gdcal)

    mycmap = 'brg_r'
    cmap = cmaps.get_cmap(mycmap, ndome)
    sm = cmaps.ScalarMappable(cmap=mycmap, norm=plt.Normalize(vmin=np.min(umjd), vmax=np.max(umjd)))
    if do59557: sm = cmaps.ScalarMappable(cmap=mycmap, norm=plt.Normalize(vmin=1, vmax=ndome))

    ax1 = plt.subplot2grid((4, 1), (0, 0))
    ax2 = plt.subplot2grid((4, 1), (1, 0))
    ax3 = plt.subplot2grid((4, 1), (2, 0))
    ax4 = plt.subplot2grid((4, 1), (3, 0))
    axes = [ax1, ax2, ax3, ax4]
    i = 0
    for ax in axes:
        ax.set_xlim(0, 301)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        if i != 3: 
            ax.minorticks_on()
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
        ax.tick_params(axis='both',which='major',length=axmajlen)
        ax.tick_params(axis='both',which='minor',length=axminlen)
        ax.tick_params(axis='both',which='both',width=axwidth)
        ax.text(-0.082, 0.5, ylabels[i], transform=ax.transAxes, rotation=90, ha='left', va='center')
        if i < 3: ax.axes.xaxis.set_ticklabels([])
        i += 1
    ax4.set_xlabel(r'Fiber ID')
    ax3.set_ylim(0.92, 1.08)

    ax_divider = make_axes_locatable(ax1)
    cax = ax_divider.append_axes("top", size="7%", pad="2%")
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position('top')
    cax.minorticks_on()
    if do59557:
        cax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        cax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax1.text(0.5, 1.23, 'Exposure', transform=ax1.transAxes, ha='center')
    else:
        cax.xaxis.set_major_locator(ticker.MultipleLocator(coltickval))
        cax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax1.text(0.5, 1.23, 'MJD', transform=ax1.transAxes, ha='center')

    flux = np.zeros((ndome, len(xarr)))
    for idome in range(ndome):
        file1d = load.filename('1D', mjd=str(umjd[idome]), num=gdcal['NUM'][idome], chips='c')
        file1d = file1d.replace('1D-', '1D-' + schip + '-')
        if os.path.exists(file1d):
            hdr = fits.getheader(file1d)
            oned = fits.getdata(file1d)
            flux[idome] = np.nanmedian(oned, axis=1)[::-1]
            flux[idome][74] = np.nanmean([flux[idome][72],flux[idome][73],flux[idome][75],flux[idome][76]])
            flux[idome][224] = np.nanmean([flux[idome][222],flux[idome][223],flux[idome][225],flux[idome][226]])
            print(str(umjd[idome])+'   '+str(gdcal['NUM'][idome])+'   '+str(int(round(np.max(flux[idome]))))+'  expt='+str(int(round(hdr['exptime'])))+'  nread='+str(hdr['nread']))
            mycolor = cmap(idome)
            ax1.plot(xarr, flux[idome], color=mycolor)
            flux[idome] = flux[idome] / np.nanmedian(flux[idome])
            ax2.plot(xarr, flux[idome], color=mycolor)

    gd, = np.where(np.nanmean(flux,axis=1) > 0)
    ndome = len(gd)
    flux=flux[gd]
    medflux = np.nanmedian(flux,axis=0)
    divmed = flux / medflux
    for idome in range(ndome):
        mycolor = cmap(idome)
        ax3.plot(xarr, divmed[idome], color=mycolor)

    percentDif = np.nanmax(np.absolute(divmed-1),axis=0)*100
    ax4.bar(xarr, percentDif, color='aquamarine', edgecolor='k')

    ymin,ymax = ax4.get_ylim()
    ax4.set_ylim(0, ymax)
    ax4.grid(True)

    mad = dln.mad(percentDif)
    med = np.nanmedian(percentDif)
    txt = 'median deviation = ' + str("%.3f" % round(med, 3)) + '%'
    ax4.text(0.5, 0.95, txt, transform=ax4.transAxes, ha='center', va='top', bbox=bboxpar, c='r')
    ax4.axhline(med, color='r', linestyle='dashed')

    fig.subplots_adjust(left=0.084,right=0.985,bottom=0.043,top=0.95,hspace=0.08,wspace=0.1)
    plt.savefig(plotfile)
    plt.close('all')

###########################################################################################
def dillum59557(pix=[824,1224], norm=True, resid=True):
    ###########################################################################################
    # dillum59557.png
    # Time series plot of median dome flat flux from cross sections across fibers from series of 59557 flats
    plotfile = specdir5 + 'monitor/' + instrument + '/dillum59557_' + str(pix[0]) + '-' + str(pix[1]) + '.png'
    ylabel = r'Median Flux'
    if norm:
        plotfile = plotfile.replace('.png', '_norm.png')
        ylabel = r'Median Flux  /  Max Flux'
    if resid:
        plotfile = plotfile.replace('.png', '_resid.png')
        ylabel = r'Median Fiber Flux  /  Max Fiber Flux  /  Overall Median Flux'

    print("----> commissNplots: Making " + os.path.basename(plotfile))

    fig = plt.figure(figsize=(30,22))
    xarr = np.arange(0, 300, 1) + 1

    gd, = np.where(allexp['MJD'][dome] == 59557)
    gdcal = allexp[dome][gd]
    ndome = len(gdcal)

    mycmap = 'brg_r'
    cmap = cmaps.get_cmap(mycmap, ndome)
    sm = cmaps.ScalarMappable(cmap=mycmap, norm=plt.Normalize(vmin=1, vmax=ndome))

    txt = 'median over pixel range ' + str(pix[0]) + ':' + str(pix[1])

    for ichip in range(nchips):
        chip = chips[ichip]
        ax = plt.subplot2grid((nchips, 1), (ichip, 0))
        ax.set_xlim(0, 301)
        if resid: ax.set_ylim(0.75, 1.25)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        ax.minorticks_on()
        ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
        ax.tick_params(axis='both',which='major',length=axmajlen)
        ax.tick_params(axis='both',which='minor',length=axminlen)
        ax.tick_params(axis='both',which='both',width=axwidth)
        if ichip == nchips-1: ax.set_xlabel(r'Fiber Index')
        if ichip == 1: ax.set_ylabel(ylabel)
        if ichip < nchips-1: ax.axes.xaxis.set_ticklabels([])
        if ichip == 0:
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("top", size="7%", pad="2%")
            cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
            cax.xaxis.set_ticks_position("top")
            #cax.minorticks_on()
            cax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            #cax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
            cax.xaxis.set_label_position('top') 
            cax.set_xlabel('Exposure')
        ax.text(0.2, 0.19, txt, transform=ax.transAxes, ha='left', bbox=bboxpar)

        flux = np.zeros((ndome, len(xarr)))
        for idome in range(ndome):
            chp = 'c'
            if ichip == 1: chp = 'b'
            if ichip == 2: chp = 'a'
            file1d = load.filename('1D', mjd='59557', num=gdcal['NUM'][idome], chips='c')
            file1d = file1d.replace('1D-', '1D-' + chp + '-')
            if os.path.exists(file1d):
                hdr = fits.getheader(file1d)
                oned = fits.getdata(file1d)
                flux[idome] = np.nanmedian(oned[:, pix[0]:pix[1]], axis=1)[::-1]
                flux[idome][74] = np.nanmean([flux[idome][72],flux[idome][73],flux[idome][75],flux[idome][76]])
                flux[idome][224] = np.nanmean([flux[idome][222],flux[idome][223],flux[idome][225],flux[idome][226]])
                mnf = np.nanmin(flux[idome][135:145])
                if (ichip == 0) & (mnf < 7500): print("BAD FLAT")
                mycolor = cmap(idome)
                if norm: flux[idome] = flux[idome] / np.nanmax(flux[idome])
                if resid is False: ax.plot(xarr, flux[idome], color=mycolor)

        gd, = np.where(np.nanmean(flux,axis=1) > 0)
        ndome = len(gd)
        if ndome > 0: flux=flux[gd]

        if resid:
            meanflux = np.nanmean(flux,axis=0)
            medflux = np.nanmedian(flux,axis=0)
            div = flux / meanflux
            divmed = flux / medflux
            for idome in range(ndome):
                mycolor = cmap(idome)
                ax.plot(xarr, divmed[idome], color=mycolor)
                #bd, = np.where(divmed[idome] < 0.85)
                #if len(bd) > 0: print(bd)

            medresid = np.nanmedian(np.absolute(divmed))
            medresidpercent = (medresid / np.nanmedian(meanflux))*100
            madresid = dln.mad(div)
            madresidpercent = (madresid / np.nanmean(div))*100
            txt1 = ''#med = ' + str("%.1f" % round(medresid, 1)) + ' (' + str("%.1f" % round(medresidpercent, 1)) + '%)'
            txt2 = 'MAD = ' + str("%.3f" % round(madresid, 3)) + ' (' + str("%.3f" % round(madresidpercent, 3)) + '%)'
            #ax.text(0.1, 0.15, txt1+',   '+txt2, transform=ax.transAxes, ha='left')
            ax.text(0.2, 0.10, txt2, transform=ax.transAxes, ha='left', bbox=bboxpar)

        ax.text(0.97,0.06,chip.capitalize() + '\n' + 'Chip', transform=ax.transAxes, 
                ha='center', va='bottom', color=chip, bbox=bboxpar)

    fig.subplots_adjust(left=0.06,right=0.985,bottom=0.045,top=0.955,hspace=0.08,wspace=0.1)
    plt.savefig(plotfile)
    plt.close('all')

###########################################################################################
def tellmagcolor(alls4=None, alls5=None, latlims=[10,12]):
    # tellmagcolor.png
    # Check of the magnitude and color distribution of plate tellurics vs. FPS tellurics

    plotfile = specdir5 + 'monitor/' + instrument + '/tellmagcolor_lat_'+str(latlims[0])+'-'+str(latlims[1])+'.png'
    print("----> commissNplots: Making " + os.path.basename(plotfile))

    fpsdata = fits.getdata('/uufs/chpc.utah.edu/common/home/u0955897/projects/com/ops_std_apogee-0.5.0.fits', 1)
    c_icrs = SkyCoord(ra=fpsdata['ra']*u.degree, dec=fpsdata['dec']*u.degree, frame='icrs')
    lon = c_icrs.galactic.l.deg
    lat = c_icrs.galactic.b.deg
    g, = np.where((lat > latlims[0]) & (lat < latlims[1]) & (fpsdata['selected'] == True))
    fpsdata = fpsdata[g]

    g, = np.where(((bitmask.is_bit_set(alls4['APOGEE_TARGET2'],9)) | (bitmask.is_bit_set(alls4['APOGEE2_TARGET2'],9))) & (alls4['GLAT'] >= latlims[0]) & (alls4['GLAT'] <= latlims[1]))
    alls4g = alls4[g]

    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.02
    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + spacing, width, 0.2]
    rect_histy = [left + width + spacing, bottom, 0.2, height]

    fontsize = 36;   fsz = fontsize * 0.75
    fig = plt.figure(figsize=(22,18))
    ax1 = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax1)
    ax_histy = fig.add_axes(rect_histy, sharey=ax1)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    axes = [ax1, ax_histx, ax_histy]
    for ax in axes:
        ax.minorticks_on()
        ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
        ax.tick_params(axis='both',which='major',length=axmajlen)
        ax.tick_params(axis='both',which='minor',length=axminlen)
        ax.tick_params(axis='both',which='both',width=axwidth)
    ax_histx.set_xlim(-0.2, 0.55)
    ax_histy.set_ylim(11.2, 6.5)
    ax1.set_xlim(-0.2, 0.65)
    ax1.set_ylim(11.1, 5.8)
    ax1.set_xlabel(r'J $-$ K')
    ax1.set_ylabel(r'H')
    ax1.minorticks_on()

    #ax1.axhline(y=0, linestyle='dashed', color='k', zorder=1)
    #ax.plot([-100,100000], [-100,100000], linestyle='dashed', color='k')

    ax_histx.text(0.5, 1.025, str(int(round(latlims[0]))) + r' < lat < ' + str(int(round(latlims[1]))), transform=ax_histx.transAxes, ha='center')
    ax_histx.set_ylabel(r'N (FPS)')
    ax_histy.set_xlabel(r'N (FPS)')

    symbol = 'o'
    symsz = 40
    cmap = 'rainbow'
    vmin = 0.2
    vmax = 1.0

    # FPS
    x = fpsdata['j_m'] - fpsdata['k_m']
    y = fpsdata['h_m']
    ax1.scatter(x, y, marker=symbol, c='r', s=3, alpha=0.75, zorder=1, label='FPS')

    binwidth = 0.02
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth
    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins, color='r', zorder=1, histtype='step')

    binwidth = 0.1
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth
    bins = np.arange(-lim, lim + binwidth, binwidth)
    #ax_histx.hist(x, bins=bins)
    ax_histy.hist(y, bins=bins, orientation='horizontal', color='r', zorder=1, histtype='step')

    # Plate
    x = alls4g['J'] - alls4g['K']
    y = alls4g['H']
    ax1.scatter(x, y, marker=symbol, c='k', s=10, alpha=0.75, zorder=2, label='Plate')

    ax_histx1 = ax_histx.twinx()
    ax_histy1 = ax_histy.twiny()
    ax_histx1.yaxis.tick_right()
    ax_histy1.xaxis.tick_top()
    ax_histx1.minorticks_on()
    ax_histy1.minorticks_on()
    ax_histx1.yaxis.set_label_position("right")
    ax_histx1.set_ylabel(r'N (Plate)')
    ax_histy1.xaxis.set_label_position("top")
    ax_histy1.set_xlabel(r'N (Plate)')

    binwidth = 0.02
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth
    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx1.hist(x, bins=bins, color='k', zorder=2, histtype='step')

    binwidth = 0.1
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth
    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histy1.hist(y, bins=bins, orientation='horizontal', color='k', zorder=2, histtype='step')

    ax1.legend(loc='upper right', labelspacing=0.5, handletextpad=-0.1, markerscale=3, edgecolor='k', framealpha=1)

    fig.subplots_adjust(left=0.05, right=0.92, bottom=0.045, top=0.92, hspace=0.1, wspace=0.17)
    plt.savefig(plotfile)
    plt.close('all')

    return

###########################################################################################
def apsky(mjdstart='59146'):

    apodir = os.environ.get('APOGEE_REDUX')+'/'


    exp = np.concatenate([exp59569a,exp59569b,exp59570a,exp59570b,exp59592a])#,exp59592b])
    mjd = np.concatenate([mjd59569a,mjd59569b,mjd59570a,mjd59570b,mjd59592a])#,mjd59592b])
    expord = np.argsort(exp)
    exp = exp[expord]
    mjd = mjd[expord]
    nexp = len(exp)

    print('EXPOSURE  RA         DEC         MOONPHASE MOONDIST   MEDB      MEDG      MEDR')
    for iexp in range(nexp):
        onedfile = load.filename('1D', num=exp[iexp], mjd=mjd[iexp], chips=True)
        if os.path.exists(onedfile.replace('1D-','1D-c-')) == False:
            print('ap1D not found for ' + str(exp[iexp]))
            continue
        fileb = onedfile.replace('1D-','1D-c-')
        fileg = onedfile.replace('1D-','1D-b-')
        filer = onedfile.replace('1D-','1D-a-')
        hdr = fits.getheader(fileb)
        ra = hdr['RADEG']
        dec = hdr['DECDEG']
        dateobs = hdr['DATE-OBS']
        tt = Time(dateobs, format='fits')

        # Get moon distance and phase.
        moonpos = get_moon(tt)
        moonra = moonpos.ra.deg
        moondec = moonpos.dec.deg
        c1 = SkyCoord(ra * astropyUnits.deg, dec * astropyUnits.deg)
        c2 = SkyCoord(moonra * astropyUnits.deg, moondec * astropyUnits.deg)
        sep = c1.separation(c2)
        moondist = sep.deg
        moonphase = moon_illumination(tt)

        fluxb = fits.getdata(fileb,1)
        fluxg = fits.getdata(fileg,1)
        fluxr = fits.getdata(filer,1)
        medb = np.nanmedian(fluxb[:, 424:1624])
        medg = np.nanmedian(fluxg[:, 424:1624])
        medr = np.nanmedian(fluxr[:, 424:1624])

        p1 = str(exp[iexp])
        p2 = str("%.5f" % round(ra,5))
        p3 = str("%.5f" % round(dec,5))
        p4 = str("%.3f" % round(moonphase,3))
        p5 = str("%.3f" % round(moondist,3)).rjust(6)
        p6 = str("%.3f" % round(medb,3)).rjust(7)
        p7 = str("%.3f" % round(medg,3)).rjust(7)
        p8 = str("%.3f" % round(medr,3)).rjust(7)
        print(p1+'   '+p2+'   '+p3+'   '+p4+'     '+p5+'     '+p6+'   '+p7+'   '+p8)


    return

###########################################################################################
def skytests(mjd='59592'):

    apodir = os.environ.get('APOGEE_REDUX')+'/'

    exp59569a = np.array([40070027,40070031,40070032,40070033,40070034])
    exp59569b = np.array([40070028,40070035])
    mjd59569a = np.array(['59569','59569','59569','59569','59569'])
    mjd59569b = np.array(['59569','59569'])
    
    exp59570a = np.array([40080011,40080014,40080015,40080018,40080019,40080022,40080023])
    exp59570b = np.array([40080012,40080013,40080016,40080017,40080020,40080021,40080024])
    mjd59570a = np.array(['59570','59570','59570','59570','59570','59570','59570'])
    mjd59570b = np.array(['59570','59570','59570','59570','59570','59570','59570'])

    exp59592a = np.array([40300046,40300050,40300052,40300056,40300058,40300062])
    exp59592b = np.array([40300047,40300049,40300053,40300055,40300059,40300061])
    mjd59592a = np.array(['59592','59592','59592','59592','59592','59592'])
    mjd59592b = np.array(['59592','59592','59592','59592','59592','59592'])
    nexp59592 = len(exp59592a)

    exp = np.concatenate([exp59569a,exp59569b,exp59570a,exp59570b,exp59592a])#,exp59592b])
    mjd = np.concatenate([mjd59569a,mjd59569b,mjd59570a,mjd59570b,mjd59592a])#,mjd59592b])
    expord = np.argsort(exp)
    exp = exp[expord]
    mjd = mjd[expord]
    nexp = len(exp)

    print('EXPOSURE  RA         DEC         MOONPHASE MOONDIST   MEDB      MEDG      MEDR')
    for iexp in range(nexp):
        onedfile = load.filename('1D', num=exp[iexp], mjd=mjd[iexp], chips=True)
        if os.path.exists(onedfile.replace('1D-','1D-c-')) == False:
            print('ap1D not found for ' + str(exp[iexp]))
            continue
        fileb = onedfile.replace('1D-','1D-c-')
        fileg = onedfile.replace('1D-','1D-b-')
        filer = onedfile.replace('1D-','1D-a-')
        hdr = fits.getheader(fileb)
        ra = hdr['RADEG']
        dec = hdr['DECDEG']
        dateobs = hdr['DATE-OBS']
        tt = Time(dateobs, format='fits')

        # Get moon distance and phase.
        moonpos = get_moon(tt)
        moonra = moonpos.ra.deg
        moondec = moonpos.dec.deg
        c1 = SkyCoord(ra * astropyUnits.deg, dec * astropyUnits.deg)
        c2 = SkyCoord(moonra * astropyUnits.deg, moondec * astropyUnits.deg)
        sep = c1.separation(c2)
        moondist = sep.deg
        moonphase = moon_illumination(tt)

        fluxb = fits.getdata(fileb,1)
        fluxg = fits.getdata(fileg,1)
        fluxr = fits.getdata(filer,1)
        medb = np.nanmedian(fluxb[:, 424:1624])
        medg = np.nanmedian(fluxg[:, 424:1624])
        medr = np.nanmedian(fluxr[:, 424:1624])

        p1 = str(exp[iexp])
        p2 = str("%.5f" % round(ra,5))
        p3 = str("%.5f" % round(dec,5))
        p4 = str("%.3f" % round(moonphase,3))
        p5 = str("%.3f" % round(moondist,3)).rjust(6)
        p6 = str("%.3f" % round(medb,3)).rjust(7)
        p7 = str("%.3f" % round(medg,3)).rjust(7)
        p8 = str("%.3f" % round(medr,3)).rjust(7)
        print(p1+'   '+p2+'   '+p3+'   '+p4+'     '+p5+'     '+p6+'   '+p7+'   '+p8)


    return


###########################################################################################
def rvparams(allv4=None, allv5=None, remake=False, restrict=False):
    # rvparams.png
    # Plot of stellar parameters, plate vs. FPS

    plotfile = specdir5 + 'monitor/' + instrument + '/rvparams1.png'
    if restrict: plotfile = plotfile.replace('rvparams1', 'rvparams2')
    print("----> monitor: Making " + os.path.basename(plotfile))

    datafile = 'rvparams_plateVSfps.fits'

    if remake:
        if allv4 is None:
            allv4path = '/uufs/chpc.utah.edu/common/home/sdss40/apogeework/apogee/spectro/aspcap/dr17/synspec/allVisit-dr17-synspec.fits'
            allv4 = fits.getdata(allv4path)
        if allv5 is None:
            allv5path = specdir5 + 'summary/allVisit-daily-apo25m.fits'
            allv5 = fits.getdata(allv5path)
        gd, = np.where((allv5['MJD'] > 59580) & 
                       (np.isnan(allv5['vheliobary']) == False) & 
                       (np.absolute(allv5['vheliobary']) < 400) & 
                       (np.isnan(allv5['SNR']) == False) &  
                       (allv5['n_components'] == 1) & 
                       (allv5['SNR'] > 10))
        allv5fps = allv5[gd]

        gd, = np.where((np.isnan(allv4['VHELIO']) == False) &
                       (np.absolute(allv4['VHELIO']) < 400) &
                       (np.isnan(allv4['SNR']) == False) &  
                       (allv4['n_components'] == 1) & 
                       (allv4['SNR'] > 10))
        allv4g = allv4[gd]

        uplateIDs = np.unique(allv4g['APOGEE_ID'])
        ufpsIDs = np.unique(allv5fps['APOGEE_ID'])

        gdIDs, plate_ind, fps_ind = np.intersect1d(uplateIDs, ufpsIDs, return_indices=True)
        ngd = len(gdIDs)
        print(ngd)

        dt = np.dtype([('APOGEE_ID', np.str, 18),
                       ('JMAG',      np.float64),
                       ('HMAG',      np.float64),
                       ('KMAG',      np.float64),
                       ('NVIS',      np.int32, 2),
                       ('NCOMP',     np.float64, 2),
                       ('SNRTOT',    np.float64, 2),
                       ('SNRAVG',    np.float64, 2),
                       ('VHELIO',    np.float64, 2),
                       ('EVHELIO',   np.float64, 2),
                       ('TEFF',      np.float64, 2),
                       ('ETEFF',     np.float64, 2),
                       ('LOGG',      np.float64, 2),
                       ('ELOGG',     np.float64, 2),
                       ('FEH',       np.float64, 2),
                       ('EFEH',      np.float64, 2)])
        gdata = np.zeros(ngd, dtype=dt)
        gdata['APOGEE_ID'] = gdIDs

        for i in range(ngd):
            print(i)
            p4, = np.where(gdIDs[i] == allv4g['APOGEE_ID'])
            p5, = np.where(gdIDs[i] == allv5fps['APOGEE_ID'])
            gdata['APOGEE_ID'][i] = gdIDs[i]
            gdata['JMAG'][i] = allv4['J'][p4][0]
            gdata['HMAG'][i] = allv4['H'][p4][0]
            gdata['KMAG'][i] = allv4['K'][p4][0]
            gdata['NVIS'][i,0] = len(p4)
            gdata['NVIS'][i,1] = len(p5)
            gdata['SNRTOT'][i,0] = np.nansum(allv4g['SNR'][p4])
            gdata['SNRTOT'][i,1] = np.nansum(allv5fps['SNR'][p5])
            gdata['SNRAVG'][i,0] = np.nanmean(allv4g['SNR'][p4])
            gdata['SNRAVG'][i,1] = np.nanmean(allv5fps['SNR'][p5])
            gdata['VHELIO'][i,0] = np.nanmean(allv4g['VHELIO'][p4])
            gdata['VHELIO'][i,1] = np.nanmean(allv5fps['vheliobary'][p5])
            gdata['EVHELIO'][i,0] = np.nanmean(allv4g['VRELERR'][p4])
            gdata['EVHELIO'][i,1] = np.nanmean(allv5fps['vrelerr'][p5])
            gdata['TEFF'][i,0] = np.nanmean(allv4g['RV_TEFF'][p4])
            gdata['TEFF'][i,1] = np.nanmean(allv5fps['rv_teff'][p5])
            gdata['ETEFF'][i,0] = np.nanstd(allv4g['RV_TEFF'][p4])
            gdata['ETEFF'][i,1] = np.nanstd(allv5fps['rv_teff'][p5])
            gdata['LOGG'][i,0] = np.nanmean(allv4g['RV_LOGG'][p4])
            gdata['LOGG'][i,1] = np.nanmean(allv5fps['rv_logg'][p5])
            gdata['ELOGG'][i,0] = np.nanstd(allv4g['RV_LOGG'][p4])
            gdata['ELOGG'][i,1] = np.nanstd(allv5fps['rv_logg'][p5])
            gdata['FEH'][i,0] = np.nanmean(allv4g['RV_FEH'][p4])
            gdata['FEH'][i,1] = np.nanmean(allv5fps['rv_feh'][p5])
            gdata['EFEH'][i,0] = np.nanstd(allv4g['RV_FEH'][p4])
            gdata['EFEH'][i,1] = np.nanstd(allv5fps['rv_feh'][p5])

        Table(gdata).write('rvparams_plateVSfps.fits', overwrite=True)

    gdata = fits.getdata(datafile)

    fig = plt.figure(figsize=(22,18))
    ax1 = plt.subplot2grid((2,3), (0,0), colspan=3)
    ax2 = plt.subplot2grid((2,3), (1,0))
    ax3 = plt.subplot2grid((2,3), (1,1))
    ax4 = plt.subplot2grid((2,3), (1,2))
    axes = [ax1,ax2,ax3,ax4]
    if restrict:
        ax1.set_xlim(-150, 150)
        ax1.set_ylim(-1.8, 1.8)
        ax2.set_xlim(3300, 6800)
        ax2.set_ylim(-650, 650)
        ax3.set_xlim(-0.1, 5.1)
        ax3.set_ylim(-1.4, 1.4)
        ax4.set_xlim(-1.5, 0.4)
        ax4.set_ylim(-0.5, 0.5)
    ax1.text(0.05, 0.95, r'$V_{\rm helio}$ (km$\,s^{-1}$)', transform=ax1.transAxes, va='top', bbox=bboxpar, zorder=20)
    ax2.text(0.05, 0.95, r'RV $T_{\rm eff}$ (K)', transform=ax2.transAxes, va='top', bbox=bboxpar, zorder=20)
    ax3.text(0.05, 0.95, r'RV log$\,g$ (dex)', transform=ax3.transAxes, va='top', bbox=bboxpar, zorder=20)
    ax4.text(0.05, 0.95, r'RV [Fe/H] (dex)', transform=ax4.transAxes, va='top', bbox=bboxpar, zorder=20)
    #ax1.set_xlabel(r'DR17 ')
    ax1.set_ylabel(r'DR17 $-$ FPS')
    ax2.set_xlabel(r'DR17')
    ax3.set_xlabel(r'DR17')
    ax4.set_xlabel(r'DR17')
    ax2.set_ylabel(r'DR17 $-$ FPS')
    #ax1.xaxis.set_major_locator(ticker.MultipleLocator(50))
    #ax1.yaxis.set_major_locator(ticker.MultipleLocator(50))
    ax4.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    #ax4.yaxis.set_major_locator(ticker.MultipleLocator(1.0))
    #ax1.text(1.05, 1.03, tmp, transform=ax1.transAxes, ha='center')
    for ax in axes:
        ax.minorticks_on()
        ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
        ax.tick_params(axis='both',which='major',length=axmajlen)
        ax.tick_params(axis='both',which='minor',length=axminlen)
        ax.tick_params(axis='both',which='both',width=axwidth)
        ax.axhline(y=0, linestyle='dashed', color='k', zorder=1)
        #ax.plot([-100,100000], [-100,100000], linestyle='dashed', color='k')

    symbol = 'o'
    symsz = 40
    cmap = 'rainbow_r'
    vmin = 3500
    vmax = 8000

    g, = np.where((np.isnan(gdata['TEFF'][:,0]) == False) & (np.isnan(gdata['TEFF'][:,1]) == False) & (gdata['TEFF'][:,0] < 7000))
    x = gdata['VHELIO'][:,0][g]
    y = gdata['VHELIO'][:,0][g] - gdata['VHELIO'][:,1][g]
    c = gdata['TEFF'][:,0][g]
    ax1.text(0.05, 0.88, 'med: ' + str("%.3f" % round(np.median(np.absolute(y)), 3)), transform=ax1.transAxes, va='top', fontsize=fsz, bbox=bboxpar)
    ax1.text(0.05, 0.82, 'MAD: ' + str("%.3f" % round(dln.mad(np.absolute(y)), 3)), transform=ax1.transAxes, va='top', fontsize=fsz, bbox=bboxpar)
    sc1 = ax1.scatter(x, y, marker=symbol, c=c, cmap=cmap, s=symsz, edgecolors='k', alpha=0.75, zorder=10, vmin=vmin, vmax=vmax)

    g, = np.where((np.isnan(gdata['TEFF'][:,0]) == False) & (np.isnan(gdata['TEFF'][:,1]) == False) & (gdata['TEFF'][:,0] < 7000))
    x = gdata['TEFF'][:,0][g]# / 1000
    y = (gdata['TEFF'][:,0][g] - gdata['TEFF'][:,1][g])# / 1000
    c = gdata['TEFF'][:,0][g]
    gg, = np.where(np.absolute(y) < 2000)
    x = x[gg]
    y = y[gg]
    c = c[gg]
    ax2.text(0.05, 0.88, 'med: ' + str("%.3f" % round(np.median(np.absolute(y)), 3)), transform=ax2.transAxes, va='top', fontsize=fsz, bbox=bboxpar, zorder=20)
    ax2.text(0.05, 0.82, 'MAD: ' + str("%.3f" % round(dln.mad(np.absolute(y)), 3)), transform=ax2.transAxes, va='top', fontsize=fsz, bbox=bboxpar, zorder=20)
    sc2 = ax2.scatter(x, y, marker=symbol, c=c, cmap=cmap, s=symsz, edgecolors='k', alpha=0.75, zorder=10, vmin=vmin, vmax=vmax)

    g, = np.where((np.isnan(gdata['LOGG'][:,0]) == False) & (np.isnan(gdata['LOGG'][:,1]) == False) & (gdata['TEFF'][:,0] < 7000))
    x = gdata['LOGG'][:,0][g]
    y = gdata['LOGG'][:,0][g] - gdata['LOGG'][:,1][g]
    c = gdata['TEFF'][:,0][g]
    ax3.text(0.05, 0.88, 'med: ' + str("%.3f" % round(np.median(np.absolute(y)), 3)), transform=ax3.transAxes, va='top', fontsize=fsz, bbox=bboxpar, zorder=20)
    ax3.text(0.05, 0.82, 'MAD: ' + str("%.3f" % round(dln.mad(np.absolute(y)), 3)), transform=ax3.transAxes, va='top', fontsize=fsz, bbox=bboxpar, zorder=20)
    ax3.scatter(x, y, marker=symbol, c=c, cmap=cmap, s=symsz, edgecolors='k', alpha=0.75, zorder=10, vmin=vmin, vmax=vmax)

    g, = np.where((np.isnan(gdata['FEH'][:,0]) == False) & (np.isnan(gdata['FEH'][:,1]) == False) & (gdata['TEFF'][:,0] < 7000))
    x = gdata['FEH'][:,0][g]
    y = gdata['FEH'][:,0][g] - gdata['FEH'][:,1][g]
    c = gdata['TEFF'][:,0][g]
    ax4.text(0.05, 0.88, 'med: ' + str("%.3f" % round(np.median(np.absolute(y)), 3)), transform=ax4.transAxes, va='top', fontsize=fsz, bbox=bboxpar, zorder=20)
    ax4.text(0.05, 0.82, 'MAD: ' + str("%.3f" % round(dln.mad(np.absolute(y)), 3)), transform=ax4.transAxes, va='top', fontsize=fsz, bbox=bboxpar, zorder=20)
    sc4 = ax4.scatter(x, y, marker=symbol, c=c, cmap=cmap, s=symsz, edgecolors='k', alpha=0.75, zorder=10, vmin=vmin, vmax=vmax)

    ax1_divider = make_axes_locatable(ax1)
    cax1 = ax1_divider.append_axes("right", size="2%", pad="1%")
    cb1 = colorbar(sc1, cax=cax1, orientation="vertical")
    cax1.minorticks_on()
    #cax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax1.text(1.1, 0.5, r'DR17 RV $T_{\rm eff}$ (K)',ha='left', va='center', rotation=-90, transform=ax1.transAxes)

    #ax4_divider = make_axes_locatable(ax4)
    #cax4 = ax4_divider.append_axes("right", size="5%", pad="1%")
    #cb4 = colorbar(sc4, cax=cax4, orientation="vertical")
    #cax4.minorticks_on()
    ##cax4.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    #ax4.text(1.16, 0.5, r'DR17 RV $T_{\rm eff}$ (K)',ha='left', va='center', rotation=-90, transform=ax4.transAxes)

    fig.subplots_adjust(left=0.07, right=0.92, bottom=0.05, top=0.98, hspace=0.1, wspace=0.17)
    plt.savefig(plotfile)
    plt.close('all')

    return

    if p2 is True:
        ###########################################################################################
        # telluricJK.png
        plotfile = specdir5 + 'monitor/' + instrument + '/telluricJK.png'
        print("----> monitor: Making " + os.path.basename(plotfile))

        gd,=np.where((allv5['mjd'] > 59145) & ((allv5['targflags'] == 'MWM_TELLURIC') | (allv5['firstcarton'] == 'ops_std_apogee')) &
                     (allv5['hmag'] > 0) & (allv5['hmag'] < 20))
        allv5g = allv5[gd]
        uplate = np.unique(allv5g['plate'])
        nplate = len(uplate)
        #pl, = np.where(uplate > 14999)
        pl = np.zeros(nplate)
        meanh = np.zeros(nplate)
        sigh = np.zeros(nplate)
        minh = np.zeros(nplate)
        maxh = np.zeros(nplate)
        meanjk = np.zeros(nplate)
        minjk = np.zeros(nplate)
        maxjk = np.zeros(nplate)
        

        xarr = np.arange(0,nplate+20,1)

        fig = plt.figure(figsize=(30,14))

        ax = plt.subplot2grid((1,1), (0,0))
        #ax.set_xlim(0, 1000)
        ax.set_ylim(-0.3, 0.65)
        #ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
        ax.minorticks_on()
        ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
        ax.tick_params(axis='both',which='major',length=axmajlen)
        ax.tick_params(axis='both',which='minor',length=axminlen)
        ax.tick_params(axis='both',which='both',width=axwidth)
        #if ichip == nchips-1: ax.set_xlabel(r'MJD')
        ax.set_xlabel(r'Plate/config ID (minus 10000 if > 10000)')
        ax.set_ylabel(r'Mean Telluric STD $J-K$')
        #if ichip < nchips-1: ax.axes.xaxis.set_ticklabels([])
        #ax.axvline(x=59146, color='r', linewidth=2)

        for i in range(nplate):
            p, = np.where(uplate[i] == allv5g['plate'])
            #if len(p) > 10:
            pl[i] = int(uplate[i])
            meanh[i] = np.nanmean(allv5g['hmag'][p])
            if meanh[i] < 0: pdb.set_trace()
            sigh[i] = np.nanstd(allv5g['hmag'][p])
            minh[i] = np.nanmin(allv5g['hmag'][p])
            maxh[i] = np.nanmax(allv5g['hmag'][p])
            meanjk[i] = np.nanmean(allv5g['jmag'][p] - allv5g['kmag'][p])
            minjk[i] = np.nanmin(allv5g['jmag'][p] - allv5g['kmag'][p])
            maxjk[i] = np.nanmax(allv5g['jmag'][p] - allv5g['kmag'][p])
            xx = xarr[i]
            color = 'red'
            if pl[i] > 10000: 
                xx = xx+20
                color = 'cyan'
            xx = pl[i]
            if xx > 10000: 
                xx = xx-10000
                ax.plot([xx,xx], [minjk[i],maxjk[i]], color='k', zorder=1)

        g, = np.where(pl < 10000)
        uh,uind = np.unique(meanjk[g], return_index=True)
        for i in range(len(uh)):
            ax.plot([pl[g][uind][i],pl[g][uind][i]], [minjk[g][uind][i],maxjk[g][uind][i]], color='k', zorder=1)
        ax.scatter(pl[g][uind], uh, marker='o', s=100, c='cyan', edgecolors='k', zorder=10)#, c=colors[ifib], alpha=alf)#, label='Fiber ' + str(fibers[ifib]))
        ax.axhline(np.mean(uh), linestyle='dashed', color='cyan', zorder=1)
        g, = np.where(pl > 10000)
        ax.scatter(pl[g]-10000, meanjk[g], marker='o', s=100, c='red', edgecolors='k', zorder=10)#, c=colors[ifib], alpha=alf)#, label='Fiber ' + str(fibers[ifib]))
        ax.axhline(np.mean(meanjk[g]), linestyle='dashed', color='red', zorder=1)

        fig.subplots_adjust(left=0.06,right=0.995,bottom=0.06,top=0.96,hspace=0.08,wspace=0.00)
        plt.savefig(plotfile)
        plt.close('all')

        return

        ###########################################################################################
        # telluricH.png
        plotfile = specdir5 + 'monitor/' + instrument + '/telluricH.png'
        print("----> monitor: Making " + os.path.basename(plotfile))

        gd,=np.where((allv5['mjd'] > 59145) & ((allv5['targflags'] == 'MWM_TELLURIC') | (allv5['firstcarton'] == 'ops_std_apogee')) &
                     (allv5['hmag'] > 0) & (allv5['hmag'] < 20))
        allv5g = allv5[gd]
        uplate = np.unique(allv5g['plate'])
        nplate = len(uplate)
        #pl, = np.where(uplate > 14999)
        pl = np.zeros(nplate)
        meanh = np.zeros(nplate)
        sigh = np.zeros(nplate)
        minh = np.zeros(nplate)
        maxh = np.zeros(nplate)

        xarr = np.arange(0,nplate+20,1)

        fig = plt.figure(figsize=(30,14))

        ax = plt.subplot2grid((1,1), (0,0))
        #ax.set_xlim(0, 1000)
        ax.set_ylim(6.7, 11.3)
        #ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
        ax.minorticks_on()
        ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
        ax.tick_params(axis='both',which='major',length=axmajlen)
        ax.tick_params(axis='both',which='minor',length=axminlen)
        ax.tick_params(axis='both',which='both',width=axwidth)
        #if ichip == nchips-1: ax.set_xlabel(r'MJD')
        ax.set_xlabel(r'Plate/config index ')
        ax.set_ylabel(r'Mean Telluric STD $H$ mag')
        #if ichip < nchips-1: ax.axes.xaxis.set_ticklabels([])
        #ax.axvline(x=59146, color='r', linewidth=2)

        for i in range(nplate):
            p, = np.where(uplate[i] == allv5g['plate'])
            #if len(p) > 10:
            pl[i] = int(uplate[i])
            meanh[i] = np.nanmean(allv5g['hmag'][p])
            if meanh[i] < 0: pdb.set_trace()
            sigh[i] = np.nanstd(allv5g['hmag'][p])
            minh[i] = np.nanmin(allv5g['hmag'][p])
            maxh[i] = np.nanmax(allv5g['hmag'][p])
            xx = xarr[i]
            color = 'red'
            if pl[i] > 10000: 
                xx = xx+20
                color = 'cyan'
            xx = pl[i]
            if xx > 10000: 
                xx = xx-10000
                ax.plot([xx,xx], [minh[i],maxh[i]], color='k', zorder=1)

        g, = np.where(pl < 10000)
        uh,uind = np.unique(meanh[g], return_index=True)
        for i in range(len(uh)):
            ax.plot([pl[g][uind][i],pl[g][uind][i]], [minh[g][uind][i],maxh[g][uind][i]], color='k', zorder=1)
        ax.scatter(pl[g][uind], uh, marker='o', s=100, c='cyan', edgecolors='k', zorder=10)#, c=colors[ifib], alpha=alf)#, label='Fiber ' + str(fibers[ifib]))
        ax.axhline(np.mean(uh), linestyle='dashed', color='cyan', zorder=1)
        g, = np.where(pl > 10000)
        ax.scatter(pl[g]-10000, meanh[g], marker='o', s=100, c='red', edgecolors='k', zorder=10)#, c=colors[ifib], alpha=alf)#, label='Fiber ' + str(fibers[ifib]))
        ax.axhline(np.mean(meanh[g]), linestyle='dashed', color='red', zorder=1)


        fig.subplots_adjust(left=0.06,right=0.995,bottom=0.06,top=0.96,hspace=0.08,wspace=0.00)
        plt.savefig(plotfile)
        plt.close('all')


        return

        ###########################################################################################
        # seeingSNR.png
        plotfile = specdir5 + 'monitor/' + instrument + '/seeingSNR.png'
        print("----> monitor: Making " + os.path.basename(plotfile))


        #xarr = np.arange(0,nplate+20,1)

        fig = plt.figure(figsize=(18,14))

        ax = plt.subplot2grid((1,1), (0,0))
        ax.set_xlim(-1, 50)
        ax.set_ylim(-0.2, 6.5)
        #ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
        ax.minorticks_on()
        ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
        ax.tick_params(axis='both',which='major',length=axmajlen)
        ax.tick_params(axis='both',which='minor',length=axminlen)
        ax.tick_params(axis='both',which='both',width=axwidth)
        #if ichip == nchips-1: ax.set_xlabel(r'MJD')
        ax.set_xlabel(r'exposure S/N')
        ax.set_ylabel(r'exposure seeing')
        #if ichip < nchips-1: ax.axes.xaxis.set_ticklabels([])
        #ax.axvline(x=59146, color='r', linewidth=2)

        x = np.mean(allsnr['SN'], axis=1)
        y = allsnr['SEEING']
        c = allsnr['JD']
        cmap = 'gnuplot_r'
        sc1 = ax.scatter(x, y, marker='o', s=10, c=c, cmap=cmap, alpha=0.5)#, c=colors[ifib], alpha=alf)#, label='Fiber ' + str(fibers[ifib]))

        ax_divider = make_axes_locatable(ax)
        cax = ax_divider.append_axes("right", size="2%", pad="1%")
        cb1 = colorbar(sc1, cax=cax, orientation="vertical")
        cax.minorticks_on()
        #cax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
        ax.text(1.12, 0.5, r'MJD',ha='left', va='center', rotation=-90, transform=ax.transAxes)

        fig.subplots_adjust(left=0.055,right=0.90,bottom=0.06,top=0.96,hspace=0.08,wspace=0.00)
        plt.savefig(plotfile)
        plt.close('all')

        return






