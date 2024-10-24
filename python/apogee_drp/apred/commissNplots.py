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
from numpy.lib.recfunctions import append_fields, merge_arrays
from astroplan import moon_illumination
from astropy.coordinates import SkyCoord, get_moon, EarthLocation, AltAz
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
from scipy import stats

# import pdb; pdb.set_trace()

""" 
Location data for APO+LCO
"""
#            LON          LAT         ALT
LCOcoords = [-29.015970,  -70.692080, 2380]
APOcoords = [ 32.780278, -105.820278, 2788]

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

molecules = np.array(['CH4', 'CO2', 'H2O'])
nmolecules = len(molecules)

###########################################################################################
def getTputScatter(mjd1=59650, mjd2=59709, niter=3, sigclip=-1):
    Z = '  '

    g1, = np.where((allsnr['MJD'] > mjd1) & (allsnr['MJD'] < mjd2))
    if mjd1 > 59550: g1, = np.where((allsnr['MJD'] > mjd1) & (allsnr['MJD'] < mjd2) & (allsnr['EXPTIME'] == 457))
    ng = len(g1)
    print(ng)
    a = allsnr[g1]
    order = np.argsort(a['MJD'])
    a = a[order]
    hmagAll = a['HMAG']
    plateAll = a['PLATE']
    snAll = np.nanmean(a['SNFIBER'], axis=2)
    seczAll = a['SECZ']
    seeingAll = a['SEEING']
    mphaseAll = a['MOONPHASE']
    mdistAll = a['MOONDIST']
    exptAll = a['EXPTIME']
    skyAll = a['SKY']
    zeroAll = a['ZERO']

    tputSigma = np.zeros(ng)
    tputMad = np.zeros(ng)

    num = ng
    #num = 500
    out = np.empty(num).astype(str)
    zero = np.empty(num)
    seeing = np.full(num, -999.999)
    secz = np.empty(num)
    sigR = np.empty(num)
    sigD = np.empty(num)
    madR = np.empty(num)
    nbad = np.empty(num)
    sn9 = np.empty(num)
    for iexp in range(num):
        g, = np.where((snAll[iexp] > 0) & (hmagAll[iexp] > 5) & (hmagAll[iexp] < 15))
        if len(g) > 150:
            # First pass at fitting line to S/N as function of Hmag
            hm1 = hmagAll[iexp][g]
            sn1 = snAll[iexp][g]
            polynomial1 = np.poly1d(np.polyfit(hm1, np.log10(sn1), 1))
            yarrnew1 = polynomial1(hm1)
            diff1 = np.log10(sn1) - yarrnew1
            gd1, = np.where(diff1 > sigclip*np.nanstd(diff1))
            # Second pass at fitting line to S/N as function of Hmag
            hm2 = hm1[gd1]
            sn2 = sn1[gd1]
            polynomial2 = np.poly1d(np.polyfit(hm2, np.log10(sn2), 1))
            yarrnew2 = polynomial2(hm2)
            diff2 = np.log10(sn2) - yarrnew2
            gd2, = np.where(diff2 > sigclip*np.nanstd(diff2))
            # Final pass at fitting line to S/N as function of Hmag
            hm3 = hm2[gd2]
            sn3 = sn2[gd2]
            polynomial3 = np.poly1d(np.polyfit(hm3, np.log10(sn3), 1))
            xarrnew3 = np.linspace(np.nanmin(hm1), np.nanmax(hm1), 5000)
            yarrnew3 = polynomial3(xarrnew3)
            ratio = np.zeros(len(g))
            diff = np.zeros(len(g))
            for q in range(len(g)):
                hmdif = np.absolute(hm1[q] - xarrnew3)
                pp, = np.where(hmdif == np.nanmin(hmdif))
                ratio[q] = sn1[q] / 10**yarrnew3[pp][0]
                diff[q] = sn1[q] - 10**yarrnew3[pp][0]

            diff9 = np.absolute(9 - xarrnew3)
            pp9, = np.where(diff9 == np.nanmin(diff9))
            sn9[iexp] = 10**yarrnew3[pp9][0]

            sigratio = np.nanstd(ratio)
            bad, = np.where(ratio < 1-sigratio)
            nbad[iexp] = len(bad)
            madratio = dln.mad(ratio)
            sigdiff = dln.mad(diff)
            maddiff = dln.mad(diff)
            p000 = str(plateAll[iexp])
            p00 = str(a['MJD'][iexp])
            p0 = a['sumfile'][iexp]
            p1 = str("%.3f" % round(sigratio,3))
            #p2 = str("%.3f" % round(madratio,3))
            p3 = str("%.3f" % round(sigdiff,3))
            p4 = str("%.3f" % round(maddiff,3))
            p5 = str("%.3f" % round(seeingAll[iexp],3))
            p6 = str("%.3f" % round(seczAll[iexp],3))
            #p7 = str("%.3f" % round(np.nanmean(skyAll[iexp]),3))
            p7 = str("%.3f" % round(np.nanmean(zeroAll[iexp]),3))
            p8 = str(int(round(nbad[iexp]))).rjust(3)
            print(p000+Z+p00+Z+p0+Z+p1+Z+p3+Z+p4+Z+p5+Z+p6+Z+p7+Z+p8)

            zero[iexp] = np.nanmean(zeroAll[iexp])
            seeing[iexp] = seeingAll[iexp]
            secz[iexp] = seczAll[iexp]
            sigR[iexp] = sigratio
            sigD[iexp] = sigdiff
            madR[iexp] = madratio

    g, = np.where((seeing > 0.2) & (seeing < 5))

    return sigR[g],sigD[g],madR[g],seeing[g],secz[g],zero[g],nbad[g],sn9[g]
    pdb.set_trace()

    t1 = Column(name='TPUT_SIGMA', data=tputSigma)
    table = Table(allsnrg)
    table.add_column(t1)
    table.write('tmp.fits', format='fits', overwrite='True')

###########################################################################################
def persist1(cmap='brg', vrad=400):
    mjd = '59686'
    #exp1 = np.array([41240010,41240015,41240017,41240019,41240021,41240023,41240025,
    #                 41240047,41240049,41240051,41240053,41240055])
    exp1 = np.array([41240015,41240017,41240019,41240021,
                     41240047,41240049,41240051,41240053,41240055])
    nexp1 = len(exp1)
    chps = np.array(['c','b','a'])

    expdir = specdir5 + 'exposures/apogee-n/' + mjd + '/'
    plotdir = specdir5 + 'monitor/apogee-n/persist/'
    plotfile = plotdir + 'persist' + mjd + '.png'
    print('making ' + os.path.basename(plotfile))

    nrows = 2
    molcols = ['mediumseagreen', 'purple', 'darkorange']

    matplotlib.rcParams.update({'font.size':18, 'font.family':'serif'})
    fig = plt.figure(figsize=(33,10))
    for iexp in range(nexp1):
        for ichip in range(nchips):
            ax = plt.subplot2grid((nchips,nexp1), (ichip,iexp))
            ax.minorticks_on()
            ax.xaxis.set_major_locator(ticker.MultipleLocator(500))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(500))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(100))
            ax.tick_params(axis='both',which='both',direction='out',bottom=True,top=True,left=True,right=True)
            ax.tick_params(axis='both',which='major',length=axmajlen)
            ax.tick_params(axis='both',which='minor',length=axminlen)
            ax.tick_params(axis='both',which='both',width=axwidth)
            ax.axes.xaxis.set_ticklabels([])
            if iexp > 0: ax.axes.yaxis.set_ticklabels([])
            #if irow == nrows-1: ax.set_xlabel(r'$N$ tellurics')
            #if imol == 0: ax.set_ylabel(r'RMS (fit $-$ poly)')
            if ichip == 0: ax.text(0.5, 1.04, str(exp1[iexp])+'$-$'+str(exp1[iexp]+1), fontsize=14, transform=ax.transAxes, ha='center', va='bottom')

            if (iexp == 0) & (ichip == 0): ax.set_ylabel('Blue Pix')
            if (iexp == 0) & (ichip == 1): ax.set_ylabel('Green Pix')
            if (iexp == 0) & (ichip == 2): ax.set_ylabel('Red Pix')

            twod1 = fits.getdata(expdir + 'ap2D-' + chps[ichip] + '-' + str(exp1[iexp]) + '.fits')
            twod2 = fits.getdata(expdir + 'ap2D-' + chps[ichip] + '-' + str(exp1[iexp]+1) + '.fits')

            im = ax.imshow(twod1-twod2, cmap=cmap, vmin=-vrad, vmax=vrad)
            if iexp == nexp1 - 1: 
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad="10%")
                cax.minorticks_on()
                #cax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
                cb1 = colorbar(im, cax=cax)
                #ax.text(1.235, 0.5, r'$N$ tellurics',ha='left', va='center', rotation=-90, transform=ax.transAxes)

    fig.subplots_adjust(left=0.045,right=0.97,bottom=0.04,top=0.94,hspace=0.08,wspace=0.05)
    plt.savefig(plotfile)
    plt.close('all')

###########################################################################################
def tellredtests1(field='21200', conf='3922', mjd='59648', fiber='273'):
    outdir = os.environ.get('APOGEE_REDUX')+'/caltests1.0/visit/apo25m/plots/'
    plotfile = outdir+'specplot-'+field+'-'+conf+'-'+mjd+'-'+fiber+'.png'
    print('making ' + os.path.basename(plotfile))
    origvis0 = 'apVisit-daily-apo25m-'+conf+'-'+mjd+'-'+fiber+'.fits'
    origvis = os.environ.get('APOGEE_REDUX')+'/daily/visit/apo25m/'+field+'/'+conf+'/'+mjd+'/'+origvis0

    colors = ['r','b','orange','violet','seagreen']
    labels = np.array(['red 15', 'blue 15', 'bright 15', 'faint 15', '"best" 15'])
    labnums = np.array(['01','02','03','04','05'])
    nruns = len(labels)

    visitxmin = 15120;   visitxmax = 16960;    visitxspan = visitxmax - visitxmin
    fig = plt.figure(figsize=(28,28))
    ax1 = plt.subplot2grid((9,1), (0,0), rowspan=2)
    ax2 = plt.subplot2grid((9,1), (2,0), rowspan=2)
    ax3 = plt.subplot2grid((9,1), (4,0), rowspan=2)
    ax4 = plt.subplot2grid((9,1), (6,0), rowspan=2)
    ax5 = plt.subplot2grid((9,1), (8,0))
    axes = [ax1,ax2,ax3,ax4,ax5]
    for ax in axes:
        ax.tick_params(reset=True)
        ax.set_xlim(visitxmin, visitxmax)
        #ax.set_ylim(ymin, ymax)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(200))
        ax.minorticks_on()
        ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
        ax.tick_params(axis='both', which='major', length=axmajlen)
        ax.tick_params(axis='both', which='minor', length=axminlen)
        ax.tick_params(axis='both', which='both', width=axwidth)
    ax1.axes.xaxis.set_ticklabels([])
    ax2.axes.xaxis.set_ticklabels([])
    ax3.axes.xaxis.set_ticklabels([])
    ax4.axes.xaxis.set_ticklabels([])
    ax4.set_ylim(-3,3)

    ax5.set_xlabel(r'Wavelength ($\rm \AA$)')

    ax1.text(-0.05, 0.5, r'Flux + offset', transform=ax1.transAxes, ha='right', va='center', rotation=90)
    ax2.text(-0.05, 0.5, r'Flux', transform=ax2.transAxes, ha='right', va='center', rotation=90)
    ax3.text(-0.05, 0.5, r'% Residual (Flux / Flux90) + offset', transform=ax3.transAxes, ha='right', va='center', rotation=90)
    ax4.text(-0.05, 0.5, r'% Residual (Flux / Flux90)', transform=ax4.transAxes, ha='right', va='center', rotation=90)
    ax5.text(-0.05, 0.5, r'Telluric', transform=ax5.transAxes, ha='right', va='center', rotation=90)
    #ax1.set_ylabel(r'Flux + offset')
    #ax2.set_ylabel(r'Flux')
    #ax3.set_ylabel(r'% Residual (Flux / Flux90) + offset')
    #ax4.set_ylabel(r'% Residual (Flux / Flux90)')
    #ax5.set_ylabel(r'Telluric')

    #pdb.set_trace()
    flux = fits.getdata(origvis,1)
    flux0 = np.concatenate([flux[0,:],flux[1,:],flux[2,:]])
    wave = fits.getdata(origvis,4)
    wave0 = np.concatenate([wave[0,:],wave[1,:],wave[2,:]])
    tell = fits.getdata(origvis,7)
    tell0 = np.concatenate([tell[0,:],tell[1,:],tell[2,:]])
    med0 = np.nanmedian(flux0)
    ax1.plot(wave0, flux0, 'k', linewidth=0.5, label='all 90')
    ax2.plot(wave0, flux0, 'k', linewidth=0.5)
    med = np.empty(nruns)
    ax5.plot(wave0, tell0, 'k', linewidth=0.75)
    for i in range(nruns):
        vis = os.environ.get('APOGEE_REDUX')+'/caltests1.0/visit/apo25m/'+field+'_'+labnums[i]+'/'+conf+'/'+mjd+'/'+origvis0.replace('daily','caltests1.0')
        flux = fits.getdata(vis,1)
        flux = np.concatenate([flux[0,:],flux[1,:],flux[2,:]])
        wave = fits.getdata(vis,4)
        wave = np.concatenate([wave[0,:],wave[1,:],wave[2,:]])
        ax1.plot(wave, flux+med0*0.075*(i+1), color=colors[i], linewidth=0.5, label=labels[i])
        ax2.plot(wave, flux, linewidth=0.5, color=colors[i])
        ax3.plot(wave, (100*((flux/flux0)-1))+(i*2), color=colors[i], linewidth=0.5)
        ax4.plot(wave, 100*((flux/flux0)-1), color=colors[i], linewidth=0.5)
        med = np.nanmedian(100*((flux/flux0)-1))
        print(np.absolute(np.nansum(100*((flux/flux0)-1))))

    ax1.legend(loc='upper right', ncol=2, labelspacing=0.5, handletextpad=0.1, fontsize=fsz, edgecolor='k', framealpha=1)
    ax1.text(0.5, 1.02, field+'-'+conf+'-'+mjd+'-'+fiber, ha='center', va='bottom', transform=ax1.transAxes)

    fig.subplots_adjust(left=0.065,right=0.99,bottom=0.035,top=0.975,hspace=0.12,wspace=0.05)
    plt.savefig(plotfile)
    plt.close('all')

###########################################################################################
def tellredtests2(field='21200', conf='3922', mjd='59648', fiber='273'):
    outdir = os.environ.get('APOGEE_REDUX')+'/caltests1.0/visit/apo25m/plots/'
    plotfile = outdir+'specplot1-'+field+'-'+conf+'-'+mjd+'-'+fiber+'.png'
    print('making ' + os.path.basename(plotfile))
    origvis0 = 'apVisit-daily-apo25m-'+conf+'-'+mjd+'-'+fiber+'.fits'
    origvis = os.environ.get('APOGEE_REDUX')+'/daily/visit/apo25m/'+field+'/'+conf+'/'+mjd+'/'+origvis0

    colors = ['r','b','orange','violet','seagreen']
    labels = np.array(['red 15', 'blue 15', 'bright 15', 'faint 15', '"best" 15'])
    labnums = np.array(['01','02','03','04','05'])
    nruns = len(labels)

    xxmin = [15135, 15665, 16605]
    xxmax = [15270, 15810, 16740]

    labels = ['Flux + offset', 'Flux', '% Residual (Flux / Flux90) + offset', '% Residual (Flux / Flux90)', 'Telluric']

    flux = fits.getdata(origvis,1)
    flux0 = np.concatenate([flux[0,:],flux[1,:],flux[2,:]])
    wave = fits.getdata(origvis,4)
    wave0 = np.concatenate([wave[0,:],wave[1,:],wave[2,:]])
    tell = fits.getdata(origvis,7)
    tell0 = np.concatenate([tell[0,:],tell[1,:],tell[2,:]])
    med0 = np.nanmedian(flux0)

    fig = plt.figure(figsize=(28,28))

    nrows = 5
    ncols = 3
    npanels = int(nrows*ncols)
    axes = []
    rownum = 0
    for i in range(nrows):
        for j in range(ncols):
            if j == 0:
                rowspan = 2
                if i == nrows: rowspan = 1
                if i > 0: rownum = rownum+rowspan
            print(str(i) + '  ' + str(rowspan) + '  ' + str(rownum) + '  ' + str(j))
            #pdb.set_trace()
            ax = plt.subplot2grid((9,3), (rownum,j), rowspan=rowspan)
            ax.set_xlim(xxmin[j], xxmax[j])
            if i < nrows-1: ax.axes.xaxis.set_ticklabels([])
            if j > 0: ax.axes.yaxis.set_ticklabels([])
            ax.minorticks_on()
            ax.tick_params(axis='both', which='both', direction='in', bottom=True, top=True, left=True, right=True)
            ax.tick_params(axis='both', which='major', length=axmajlen)
            ax.tick_params(axis='both', which='minor', length=axminlen)
            ax.tick_params(axis='both', which='both', width=axwidth)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(50))
            ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
            if i == 2: ax.set_ylim(-1.3,9.3)
            if i == 3: ax.set_ylim(-3,3)
            if i == 4: ax.set_ylim(0.7, 1.02)
            if i == nrows-1: ax.set_xlabel(r'Wavelength ($\rm \AA$)')
            if j == 0: ax.text(-0.155, 0.5, labels[i], transform=ax.transAxes, ha='right', va='center', rotation=90)
            if i < 2:
                yminsec, = np.where((wave0 > xxmin[j]) & (wave0 < xxmax[j]))
                ymin = np.min(flux0[yminsec]) - np.min(flux0[yminsec])*0.05
                if i == 0:
                    ymax = np.min(flux0[yminsec]) + (med0*0.1*(nruns+1.5)) + (med0*0.1*(nruns+1.5)*0.05)
                else:
                    ymax = np.max(flux0[yminsec]) + np.max(flux0[yminsec])*0.05
                ax.set_ylim(ymin,ymax)
            axes.append(ax)

    axes[1].text(0.5, 1.02, field+'-'+conf+'-'+mjd+'-'+fiber, ha='center', va='bottom', transform=axes[1].transAxes)
    axes[0].plot(wave0, flux0, 'k', linewidth=0.75, label='all 90')
    axes[1].plot(wave0, flux0, 'k', linewidth=0.75, label='all 90')
    axes[2].plot(wave0, flux0, 'k', linewidth=0.75, label='all 90')
    axes[3].plot(wave0, flux0, 'k', linewidth=0.75, label='all 90')
    axes[4].plot(wave0, flux0, 'k', linewidth=0.75, label='all 90')
    axes[5].plot(wave0, flux0, 'k', linewidth=0.75, label='all 90')

    axes[npanels-3].plot(wave0, tell0, 'k', linewidth=0.75)
    axes[npanels-2].plot(wave0, tell0, 'k', linewidth=0.75)
    axes[npanels-1].plot(wave0, tell0, 'k', linewidth=0.75)

    axes[npanels-3].text(0.03, 0.08, 'H2O', color='r', transform=axes[npanels-3].transAxes)
    axes[npanels-2].text(0.03, 0.08, 'CO2', color='r', transform=axes[npanels-2].transAxes)
    axes[npanels-1].text(0.03, 0.08, 'CH4', color='r', transform=axes[npanels-1].transAxes)

    yminsec1, = np.where((wave0 > xxmin[0]) & (wave0 < xxmax[0]))
    ymin1 = np.min(flux0[yminsec1]) - np.min(flux0[yminsec1])*0.05

    for i in range(nruns):
        vis = os.environ.get('APOGEE_REDUX')+'/caltests1.0/visit/apo25m/'+field+'_'+labnums[i]+'/'+conf+'/'+mjd+'/'+origvis0.replace('daily','caltests1.0')
        flux = fits.getdata(vis,1)
        flux = np.concatenate([flux[0,:],flux[1,:],flux[2,:]])
        wave = fits.getdata(vis,4)
        wave = np.concatenate([wave[0,:],wave[1,:],wave[2,:]])
        for j in range(npanels):
            if j < 3: 
                axes[j].plot(wave, flux+med0*0.075*(i+1), color=colors[i], linewidth=0.75, label=labels[i])
            else:
                if j < 6: 
                    axes[j].plot(wave, flux, linewidth=0.75, color=colors[i])
                else:
                    if j < 9:
                        axes[j].plot(wave, (100*((flux/flux0)-1))+(i*2), color=colors[i], linewidth=0.75)
                    else:
                        if j < 12:
                            axes[j].plot(wave, 100*((flux/flux0)-1), color=colors[i], linewidth=0.75)
        #med = np.nanmedian(100*((flux/flux0)-1))
        #print(np.absolute(np.nansum(100*((flux/flux0)-1))))


    fig.subplots_adjust(left=0.065,right=0.99,bottom=0.035,top=0.975,hspace=0.14,wspace=0.0)
    plt.savefig(plotfile)
    plt.close('all')

###########################################################################################
def tellspatialnew1(zoom=False, cmap='brg'):
    data = fits.getdata('/uufs/chpc.utah.edu/common/home/u0955897/projects/com/tellfit.fits')
    expdata = fits.getdata('/uufs/chpc.utah.edu/common/home/u0955897/projects/com/tellfitstats.fits')

    expnum = expdata['EXPNUM']
    mask = np.in1d(data['expnum'], expnum)
    gd, = np.where(mask == True)
    data = data[gd]

    xy,x_ind,y_ind = np.intersect1d(expnum, data['expnum'], return_indices=True)
    expdata = expdata[x_ind]

    nrows = 2
    molcols = ['mediumseagreen', 'purple', 'darkorange']

    plotfile = sdir5 + 'tellspatial1.png'
    if zoom: plotfile = plotfile.replace('.png', '_zoom.png')
    print('making ' + os.path.basename(plotfile))
    fig = plt.figure(figsize=(32,17))
    for irow in range(nrows):
        for imol in range(nmolecules):
            ax = plt.subplot2grid((nrows,nmolecules), (irow,imol))
            ax.minorticks_on()
            #ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            #ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
            ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
            ax.tick_params(axis='both',which='major',length=axmajlen)
            ax.tick_params(axis='both',which='minor',length=axminlen)
            ax.tick_params(axis='both',which='both',width=axwidth)
            ax.set_ylim(-0.001,0.1)
            ax.set_xlim(0,75)
            if irow == 0: ax.axes.xaxis.set_ticklabels([])
            if imol > 0: ax.axes.yaxis.set_ticklabels([])
            if irow == nrows-1: ax.set_xlabel(r'$N$ tellurics')
            if irow == 0:
                if imol == 0: ax.set_ylabel(r'RMS (fit $-$ poly)')
            if irow == 1:
                if imol == 0: ax.set_ylabel(r'RMS (fit $-$ poly)')
            if irow == 0: ax.text(0.5, 1.02, molecules[imol], transform=ax.transAxes, ha='center', va='bottom')
            #if imol == 0:
            #    if ipar == 0: ax.text(0.5, 1.02, r'Constant$/$no variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)
            #    if ipar == 1: ax.text(0.5, 1.02, r'Linear variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)
            #    if ipar == 2: ax.text(0.5, 1.02, r'Quadratic variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)

            xvals = data['ntell'][:,imol]#-np.nanmin(data['JD'])
            if irow == 0: yvals = data['RMS'+str(imol+1)][:,0] - data['RMS'+str(imol+1)][:,1]
            if irow == 1: yvals = data['RMS'+str(imol+1)][:,0] - data['RMS'+str(imol+1)][:,2]
            c = expdata['sigeta'][:,imol] + expdata['sigzeta'][:,imol]

            #med = np.nanmedian(yvals)
            #ax.axhline(med, color='grey', linestyle='dashed')
            #ax.text(0.75, 0.85, 'med RMS = ' + str("%.3f" % round(med,3)), transform=ax.transAxes, ha='center', va='center', bbox=bboxpar)
            sc1 = ax.scatter(xvals, yvals, marker='o', s=3, c=molcols[imol])#, cmap=cmap)#, alpha=0.8)#, vmin=10, vmax=50)#, edgecolors='k'

            if imol == 2:
                if irow == 0: ax.text(1.03, 0.5, 'constant $-$ linear', transform=ax.transAxes, rotation=-90, ha='left', va='center')
                if irow == 1: ax.text(1.03, 0.5, 'constant $-$ quadratic', transform=ax.transAxes, rotation=-90, ha='left', va='center')

            #    divider = make_axes_locatable(ax)
            #    cax = divider.append_axes("right", size="5%", pad="10%")
            #    cax.minorticks_on()
            #    #cax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
            #    cb1 = colorbar(sc1, cax=cax)
                #ax.text(1.235, 0.5, r'$N$ tellurics',ha='left', va='center', rotation=-90, transform=ax.transAxes)

    fig.subplots_adjust(left=0.045,right=0.97,bottom=0.057,top=0.96,hspace=0.08,wspace=0.05)
    plt.savefig(plotfile)
    plt.close('all')

###########################################################################################
def tellspatialnew2(zoom=False, cmap='brg'):
    data = fits.getdata('/uufs/chpc.utah.edu/common/home/u0955897/projects/com/tellfit.fits')

    nrows = 2
    molcols = ['mediumseagreen', 'purple', 'darkorange']

    plotfile = sdir5 + 'tellspatial2.png'
    if zoom: plotfile = plotfile.replace('.png', '_zoom.png')
    print('making ' + os.path.basename(plotfile))
    fig = plt.figure(figsize=(32,17))
    for irow in range(nrows):
        for imol in range(nmolecules):
            ax = plt.subplot2grid((nrows,nmolecules), (irow,imol))
            ax.minorticks_on()
            #ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            #ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
            ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
            ax.tick_params(axis='both',which='major',length=axmajlen)
            ax.tick_params(axis='both',which='minor',length=axminlen)
            ax.tick_params(axis='both',which='both',width=axwidth)
            ax.set_ylim(0.0,1.8)
            ax.set_xlim(0,75)
            if irow == 0: ax.axes.xaxis.set_ticklabels([])
            if imol > 0: ax.axes.yaxis.set_ticklabels([])
            if irow == nrows-1: ax.set_xlabel(r'$N$ tellurics')
            if irow == 0:
                if imol == 0: ax.set_ylabel(r'Poly RMS (constant $-$ linear)')
            if irow == 1:
                if imol == 0: ax.set_ylabel(r'Poly RMS (constant $-$ quadratic)')
            if irow == 0: ax.text(0.5, 1.02, molecules[imol], transform=ax.transAxes, ha='center', va='bottom')
            #if imol == 0:
            #    if ipar == 0: ax.text(0.5, 1.02, r'Constant$/$no variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)
            #    if ipar == 1: ax.text(0.5, 1.02, r'Linear variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)
            #    if ipar == 2: ax.text(0.5, 1.02, r'Quadratic variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)

            xvals = data['ntell'][:,imol]#-np.nanmin(data['JD'])
            if irow == 0: yvals = data['MODELRMS'+str(imol+1)][:,1] - data['MODELRMS'+str(imol+1)][:,0]
            if irow == 1: yvals = data['MODELRMS'+str(imol+1)][:,2] - data['MODELRMS'+str(imol+1)][:,0]
            c = data['NTELL'][:,imol]

            #med = np.nanmedian(yvals)
            #ax.axhline(med, color='grey', linestyle='dashed')
            #ax.text(0.75, 0.85, 'med RMS = ' + str("%.3f" % round(med,3)), transform=ax.transAxes, ha='center', va='center', bbox=bboxpar)
            sc1 = ax.scatter(xvals, yvals, marker='o', s=3)#, c=c, cmap=cmap, alpha=0.8, vmin=10, vmax=50)#, edgecolors='k'

            #if ipar == 2:
            #    divider = make_axes_locatable(ax)
            #    cax = divider.append_axes("right", size="5%", pad="10%")
            #    cax.minorticks_on()
            #    #cax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
            #    cb1 = colorbar(sc1, cax=cax)
            #    ax.text(1.235, 0.5, r'$N$ tellurics',ha='left', va='center', rotation=-90, transform=ax.transAxes)

    fig.subplots_adjust(left=0.055,right=0.945,bottom=0.057,top=0.96,hspace=0.08,wspace=0.05)
    plt.savefig(plotfile)
    plt.close('all')

###########################################################################################
def tellspatialnew3(zoom=False, cmap='brg'):
    data = fits.getdata('/uufs/chpc.utah.edu/common/home/u0955897/projects/com/tellfit.fits')
    expdata = fits.getdata('/uufs/chpc.utah.edu/common/home/u0955897/projects/com/tellfitstats.fits')

    expnum = expdata['EXPNUM']
    mask = np.in1d(data['expnum'], expnum)
    gd, = np.where(mask == True)
    data = data[gd]

    xy,x_ind,y_ind = np.intersect1d(expnum, data['expnum'], return_indices=True)
    expdata = expdata[x_ind]

    nrows = 2
    molcols = ['mediumseagreen', 'purple', 'darkorange']

    plotfile = sdir5 + 'tellspatial3.png'
    if zoom: plotfile = plotfile.replace('.png', '_zoom.png')
    print('making ' + os.path.basename(plotfile))
    fig = plt.figure(figsize=(32,17))
    for irow in range(nrows):
        for imol in range(nmolecules):
            ax = plt.subplot2grid((nrows,nmolecules), (irow,imol))
            ax.minorticks_on()
            #ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            #ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
            ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
            ax.tick_params(axis='both',which='major',length=axmajlen)
            ax.tick_params(axis='both',which='minor',length=axminlen)
            ax.tick_params(axis='both',which='both',width=axwidth)
            ax.set_ylim(-0.2,0.2)
            ax.set_xlim(0,75)
            if irow == 0: ax.axes.xaxis.set_ticklabels([])
            if imol > 0: ax.axes.yaxis.set_ticklabels([])
            if irow == nrows-1: ax.set_xlabel(r'$N$ tellurics')
            if irow == 0:
                if imol == 0: ax.set_ylabel(r'constant $-$ median linear')
            if irow == 1:
                if imol == 0: ax.set_ylabel(r'constant $-$ median quadratic')
            if irow == 0: ax.text(0.5, 1.02, molecules[imol], transform=ax.transAxes, ha='center', va='bottom')
            ax.axhline(y=0, linestyle='dashed', color='grey')
            #if imol == 0:
            #    if ipar == 0: ax.text(0.5, 1.02, r'Constant$/$no variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)
            #    if ipar == 1: ax.text(0.5, 1.02, r'Linear variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)
            #    if ipar == 2: ax.text(0.5, 1.02, r'Quadratic variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)

            xvals = data['ntell'][:,imol]#-np.nanmin(data['JD'])
            if irow == 0: yvals = data['MEDOBS'+str(imol+1)] - data['MEDFIT'+str(imol+1)][:,1]
            if irow == 1: yvals = data['MEDOBS'+str(imol+1)] - data['MEDFIT'+str(imol+1)][:,2]
            c = expdata['sigeta'][:,imol] + expdata['sigzeta'][:,imol]

            #med = np.nanmedian(yvals)
            #ax.axhline(med, color='grey', linestyle='dashed')
            #ax.text(0.75, 0.85, 'med RMS = ' + str("%.3f" % round(med,3)), transform=ax.transAxes, ha='center', va='center', bbox=bboxpar)
            sc1 = ax.scatter(xvals, yvals, marker='o', s=3, c=molcols[imol])#, cmap=cmap)#, alpha=0.8)#, vmin=10, vmax=50)#, edgecolors='k'

            #if imol == 2:
            #    if irow == 0: ax.text(1.03, 0.5, 'constant $-$ linear', transform=ax.transAxes, rotation=-90, ha='left', va='center')
            #    if irow == 1: ax.text(1.03, 0.5, 'constant $-$ quadratic', transform=ax.transAxes, rotation=-90, ha='left', va='center')

            #    divider = make_axes_locatable(ax)
            #    cax = divider.append_axes("right", size="5%", pad="10%")
            #    cax.minorticks_on()
            #    #cax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
            #    cb1 = colorbar(sc1, cax=cax)
                #ax.text(1.235, 0.5, r'$N$ tellurics',ha='left', va='center', rotation=-90, transform=ax.transAxes)

    fig.subplots_adjust(left=0.05,right=0.97,bottom=0.057,top=0.96,hspace=0.08,wspace=0.05)
    plt.savefig(plotfile)
    plt.close('all')

###########################################################################################
def tellspatial(zoom=False):
    data = fits.getdata('/uufs/chpc.utah.edu/common/home/u0955897/projects/com/tellfit.fits')

    npars = 3
    molcols = ['mediumseagreen', 'purple', 'darkorange']

    plotfile = sdir5 + 'tellspatialRMS.png'
    if zoom: plotfile = plotfile.replace('.png', '_zoom.png')
    print('making ' + os.path.basename(plotfile))
    fig = plt.figure(figsize=(32,16))
    for imol in range(nmolecules):
        for ipar in range(npars):
            ax = plt.subplot2grid((nmolecules, npars), (imol,ipar%3))
            ax.minorticks_on()
            #ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            #ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
            ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
            ax.tick_params(axis='both',which='major',length=axmajlen)
            ax.tick_params(axis='both',which='minor',length=axminlen)
            ax.tick_params(axis='both',which='both',width=axwidth)
            ax.set_ylim(-0.05,1.1)
            if zoom: ax.set_ylim(-0.01, 0.1)
            if (ipar == 0) & (imol == 1): ax.set_ylabel('RMS (fit $-$ poly)')
            if imol == 2: ax.set_xlabel('Days since time[0]')
            if imol < 2: ax.axes.xaxis.set_ticklabels([])
            if ipar > 0: ax.axes.yaxis.set_ticklabels([])
            if ipar == 2: ax.text(1.02, 0.50, molecules[imol], transform=ax.transAxes, ha='left', va='center', rotation=-90, bbox=bboxpar)
            if imol == 0:
                if ipar == 0: ax.text(0.5, 1.02, r'Constant$/$no variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)
                if ipar == 1: ax.text(0.5, 1.02, r'Linear variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)
                if ipar == 2: ax.text(0.5, 1.02, r'Quadratic variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)

            xvals = data['JD']-np.nanmin(data['JD'])
            yvals = data['RMS'+str(imol+1)][:,ipar]

            med = np.nanmedian(yvals)
            ax.axhline(med, color='grey', linestyle='dashed')
            ax.text(0.75, 0.85, 'med RMS = ' + str("%.3f" % round(med,3)), transform=ax.transAxes, ha='center', va='center', bbox=bboxpar)
            ax.scatter(xvals, yvals, marker='o', s=3, c=molcols[imol], alpha=0.8)#, vmin=vmin, vmax=vmax)#, edgecolors='k'

    fig.subplots_adjust(left=0.055,right=0.97,bottom=0.057,top=0.96,hspace=0.08,wspace=0.05)
    plt.savefig(plotfile)
    plt.close('all')

###########################################################################################
def tellspatial1(zoom=False, cmap='brg'):
    data = fits.getdata('/uufs/chpc.utah.edu/common/home/u0955897/projects/com/tellfit.fits')

    npars = 3
    molcols = ['mediumseagreen', 'purple', 'darkorange']

    plotfile = sdir5 + 'tellspatialRMS1.png'
    if zoom: plotfile = plotfile.replace('.png', '_zoom.png')
    print('making ' + os.path.basename(plotfile))
    fig = plt.figure(figsize=(32,17))
    for imol in range(nmolecules):
        for ipar in range(npars):
            ax = plt.subplot2grid((nmolecules, npars), (imol,ipar%3))
            ax.minorticks_on()
            #ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            #ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
            ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
            ax.tick_params(axis='both',which='major',length=axmajlen)
            ax.tick_params(axis='both',which='minor',length=axminlen)
            ax.tick_params(axis='both',which='both',width=axwidth)
            ax.set_ylim(-0.05,1.1)
            if zoom: ax.set_ylim(-0.01, 0.1)
            if (ipar == 0) & (imol == 1): ax.set_ylabel('RMS (measured $-$ fit)')
            if imol == 2: ax.set_xlabel('Days since time[0]')
            if imol < 2: ax.axes.xaxis.set_ticklabels([])
            if ipar > 0: ax.axes.yaxis.set_ticklabels([])
            if ipar == 2: ax.text(1.02, 0.50, molecules[imol], transform=ax.transAxes, ha='left', va='center', rotation=-90, bbox=bboxpar)
            if imol == 0:
                if ipar == 0: ax.text(0.5, 1.02, r'Constant$/$no variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)
                if ipar == 1: ax.text(0.5, 1.02, r'Linear variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)
                if ipar == 2: ax.text(0.5, 1.02, r'Quadratic variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)

            xvals = data['JD']-np.nanmin(data['JD'])
            yvals = data['RMS'+str(imol+1)][:,ipar]
            c = data['NTELL'][:,imol]

            med = np.nanmedian(yvals)
            ax.axhline(med, color='grey', linestyle='dashed')
            ax.text(0.75, 0.85, 'med RMS = ' + str("%.3f" % round(med,3)), transform=ax.transAxes, ha='center', va='center', bbox=bboxpar)
            sc1 = ax.scatter(xvals, yvals, marker='o', s=3, c=c, cmap=cmap, alpha=0.8, vmin=10, vmax=50)#, edgecolors='k'

            if ipar == 2:
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad="10%")
                cax.minorticks_on()
                #cax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
                cb1 = colorbar(sc1, cax=cax)
                ax.text(1.235, 0.5, r'$N$ tellurics',ha='left', va='center', rotation=-90, transform=ax.transAxes)

    fig.subplots_adjust(left=0.055,right=0.945,bottom=0.057,top=0.96,hspace=0.08,wspace=0.05)
    plt.savefig(plotfile)
    plt.close('all')

###########################################################################################
def tellspatial2(zoom=False):
    data = fits.getdata('/uufs/chpc.utah.edu/common/home/u0955897/projects/com/tellfit.fits')

    npars = 3
    molcols = ['mediumseagreen', 'purple', 'darkorange']

    plotfile = sdir5 + 'tellspatialRMS_diff.png'
    if zoom: plotfile = plotfile.replace('.png', '_zoom.png')
    print('making ' + os.path.basename(plotfile))
    fig = plt.figure(figsize=(32,16))
    for imol in range(nmolecules):
        for ipar in range(npars):
            ax = plt.subplot2grid((nmolecules, npars), (imol,ipar%3))
            ax.minorticks_on()
            #ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            #ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
            ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
            ax.tick_params(axis='both',which='major',length=axmajlen)
            ax.tick_params(axis='both',which='minor',length=axminlen)
            ax.tick_params(axis='both',which='both',width=axwidth)
            ax.set_ylim(-0.05, 0.75)
            if zoom: ax.set_ylim(-0.01, 0.1)
            if (ipar == 0) & (imol == 1): ax.set_ylabel('RMS (modeled $-$ modeled)')
            if imol == 2: ax.set_xlabel('Days since time[0]')
            if imol < 2: ax.axes.xaxis.set_ticklabels([])
            if ipar > 0: ax.axes.yaxis.set_ticklabels([])
            if ipar == 2: ax.text(1.02, 0.50, molecules[imol], transform=ax.transAxes, ha='left', va='center', rotation=-90, bbox=bboxpar)
            if imol == 0:
                if ipar == 0: ax.text(0.5, 1.02, r'Constant $-$ Linear', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)
                if ipar == 1: ax.text(0.5, 1.02, r'Constant $-$ Quadratic', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)
                if ipar == 2: ax.text(0.5, 1.02, r'Linear $-$ Quadratic', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)

            xvals = data['JD']-np.nanmin(data['JD'])
            yvals = data['RMSDIFF'+str(imol+1)][:,ipar]

            med = np.nanmedian(yvals)
            ax.axhline(med, color='grey', linestyle='dashed')
            ax.text(0.75, 0.85, 'med RMS = ' + str("%.3f" % round(med,3)), transform=ax.transAxes, ha='center', va='center', bbox=bboxpar)
            ax.scatter(xvals, yvals, marker='o', s=3, c=molcols[imol], alpha=0.8)#, vmin=vmin, vmax=vmax)#, edgecolors='k'

    fig.subplots_adjust(left=0.055,right=0.97,bottom=0.057,top=0.96,hspace=0.08,wspace=0.05)
    plt.savefig(plotfile)
    plt.close('all')

###########################################################################################
def tellspatial3(zoom=False):
    data = fits.getdata('/uufs/chpc.utah.edu/common/home/u0955897/projects/com/tellfit.fits')

    npars = 3
    molcols = ['mediumseagreen', 'purple', 'darkorange']

    plotfile = sdir5 + 'tellspatialRMS_model.png'
    if zoom: plotfile = plotfile.replace('.png', '_zoom.png')
    print('making ' + os.path.basename(plotfile))
    fig = plt.figure(figsize=(32,16))
    for imol in range(nmolecules):
        for ipar in range(npars):
            ax = plt.subplot2grid((nmolecules, npars), (imol,ipar%3))
            ax.minorticks_on()
            #ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            #ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
            ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
            ax.tick_params(axis='both',which='major',length=axmajlen)
            ax.tick_params(axis='both',which='minor',length=axminlen)
            ax.tick_params(axis='both',which='both',width=axwidth)
            ax.set_ylim(-0.1,1.8)
            if zoom: ax.set_ylim(-0.05, 0.05)
            if ipar == 0: ax.set_ylabel('RMS (modeled)')
            if imol == 2: ax.set_xlabel('Days since time[0]')
            if imol < 2: ax.axes.xaxis.set_ticklabels([])
            if ipar > 0: ax.axes.yaxis.set_ticklabels([])
            if ipar == 2: ax.text(1.02, 0.50, molecules[imol], transform=ax.transAxes, ha='left', va='center', rotation=-90, bbox=bboxpar)
            if imol == 0:
                if ipar == 0: ax.text(0.5, 1.02, r'Constant$/$no variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)
                if ipar == 1: ax.text(0.5, 1.02, r'Linear variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)
                if ipar == 2: ax.text(0.5, 1.02, r'Quadratic variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)

            xvals = data['JD']-np.nanmin(data['JD'])
            yvals = data['MODELRMS'+str(imol+1)][:,ipar]

            med = np.nanmedian(yvals)
            ax.axhline(med, color='grey', linestyle='dashed')
            ax.text(0.75, 0.85, 'med RMS = ' + str("%.3f" % round(med,3)), transform=ax.transAxes, ha='center', va='center', bbox=bboxpar)
            ax.scatter(xvals, yvals, marker='o', s=3, c=molcols[imol], alpha=0.8)#, vmin=vmin, vmax=vmax)#, edgecolors='k'

    fig.subplots_adjust(left=0.055,right=0.97,bottom=0.057,top=0.96,hspace=0.08,wspace=0.05)
    plt.savefig(plotfile)
    plt.close('all')

###########################################################################################
def tellspatial4(zoom=False):
    data = fits.getdata('/uufs/chpc.utah.edu/common/home/u0955897/projects/com/tellfit.fits')

    npars = 3
    molcols = ['mediumseagreen', 'purple', 'darkorange']

    plotfile = sdir5 + 'tellspatialRMS_synth.png'
    if zoom: plotfile = plotfile.replace('.png', '_zoom.png')
    print('making ' + os.path.basename(plotfile))
    fig = plt.figure(figsize=(32,16))
    for imol in range(nmolecules):
        for ipar in range(npars):
            ax = plt.subplot2grid((nmolecules, npars), (imol,ipar%3))
            ax.minorticks_on()
            #ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            #ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
            ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
            ax.tick_params(axis='both',which='major',length=axmajlen)
            ax.tick_params(axis='both',which='minor',length=axminlen)
            ax.tick_params(axis='both',which='both',width=axwidth)
            ax.set_ylim(-0.05,0.3)
            #if zoom: ax.set_ylim(-0.05, 0.05)
            if ipar == 0: ax.set_ylabel('Factional RMS (fine grid)')
            if imol == 2: ax.set_xlabel('Days since time[0]')
            if imol < 2: ax.axes.xaxis.set_ticklabels([])
            if ipar > 0: ax.axes.yaxis.set_ticklabels([])
            if ipar == 2: ax.text(1.02, 0.50, molecules[imol], transform=ax.transAxes, ha='left', va='center', rotation=-90, bbox=bboxpar)
            if imol == 0:
                if ipar == 0: ax.text(0.5, 1.02, r'Constant$/$no variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)
                if ipar == 1: ax.text(0.5, 1.02, r'Linear variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)
                if ipar == 2: ax.text(0.5, 1.02, r'Quadratic variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)

            xvals = data['JD']-np.nanmin(data['JD'])
            yvals = data['synthdifrms'+str(imol+1)][:,ipar]# / data['MEDFIT'+str(imol+1)][:,ipar]

            med = np.nanmedian(yvals)
            #ax.axhline(med, color='grey', linestyle='dashed')
            #ax.text(0.75, 0.85, 'med RMS = ' + str("%.3f" % round(med,3)), transform=ax.transAxes, ha='center', va='center', bbox=bboxpar)
            ax.scatter(xvals, yvals, marker='o', s=3, c=molcols[imol], alpha=0.8)#, vmin=vmin, vmax=vmax)#, edgecolors='k'

    fig.subplots_adjust(left=0.055,right=0.97,bottom=0.057,top=0.96,hspace=0.08,wspace=0.05)
    plt.savefig(plotfile)
    plt.close('all')

###########################################################################################
def tellspatial44(zoom=False):
    data = fits.getdata('/uufs/chpc.utah.edu/common/home/u0955897/projects/com/tellfit.fits')

    npars = 3
    molcols = ['mediumseagreen', 'purple', 'darkorange']

    plotfile = sdir5 + 'tellspatial_synth.png'
    if zoom: plotfile = plotfile.replace('.png', '_zoom.png')
    print('making ' + os.path.basename(plotfile))
    fig = plt.figure(figsize=(32,16))
    for imol in range(nmolecules):
        for ipar in range(npars):
            ax = plt.subplot2grid((nmolecules, npars), (imol,ipar%3))
            ax.minorticks_on()
            #ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            #ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
            ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
            ax.tick_params(axis='both',which='major',length=axmajlen)
            ax.tick_params(axis='both',which='minor',length=axminlen)
            ax.tick_params(axis='both',which='both',width=axwidth)
            ax.set_ylim(-0.05,0.7)
            #if zoom: ax.set_ylim(-0.05, 0.05)
            if (ipar == 0) & (imol == 1): ax.set_ylabel('(max fitscale $-$ min fitscale) $/$ med fitscale')
            if imol == 2: ax.set_xlabel('Days since time[0]')
            if imol < 2: ax.axes.xaxis.set_ticklabels([])
            if ipar > 0: ax.axes.yaxis.set_ticklabels([])
            if ipar == 2: ax.text(1.02, 0.50, molecules[imol], transform=ax.transAxes, ha='left', va='center', rotation=-90, bbox=bboxpar)
            if imol == 0:
                if ipar == 0: ax.text(0.5, 1.02, r'Constant$/$no variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)
                if ipar == 1: ax.text(0.5, 1.02, r'Linear variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)
                if ipar == 2: ax.text(0.5, 1.02, r'Quadratic variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)

            xvals = data['JD']-np.nanmin(data['JD'])
            yvals = data['synthfrac'+str(imol+1)][:,ipar]# / data['MEDFIT'+str(imol+1)][:,ipar]

            med = np.nanmedian(yvals)
            #ax.axhline(med, color='grey', linestyle='dashed')
            #ax.text(0.75, 0.85, 'med RMS = ' + str("%.3f" % round(med,3)), transform=ax.transAxes, ha='center', va='center', bbox=bboxpar)
            ax.scatter(xvals, yvals, marker='o', s=3, c=molcols[imol], alpha=0.8)#, vmin=vmin, vmax=vmax)#, edgecolors='k'

    fig.subplots_adjust(left=0.055,right=0.97,bottom=0.057,top=0.96,hspace=0.08,wspace=0.05)
    plt.savefig(plotfile)
    plt.close('all')

###########################################################################################
def tellspatial444(zoom=False, cmap='rainbow'):
    data = fits.getdata('/uufs/chpc.utah.edu/common/home/u0955897/projects/com/tellfit.fits')
    expdata = fits.getdata('/uufs/chpc.utah.edu/common/home/u0955897/projects/com/tellfitstats.fits')

    expdata = fits.getdata('tellfitstats_all.fits')

    expnum = expdata['EXPNUM']
    mask = np.in1d(data['expnum'], expnum)
    gd, = np.where(mask == True)
    data = data[gd]

    xy,x_ind,y_ind = np.intersect1d(expnum, data['expnum'], return_indices=True)
    expdata = expdata[x_ind]

    npars = 3
    molcols = ['mediumseagreen', 'purple', 'darkorange']

    plotfile = sdir5 + 'tellspatial_synth2.png'
    if zoom: plotfile = plotfile.replace('.png', '_zoom.png')
    print('making ' + os.path.basename(plotfile))
    fig = plt.figure(figsize=(20,16))
    for imol in range(nmolecules):
        ax = plt.subplot2grid((nmolecules, 1), (imol, 0))
        ax.minorticks_on()
        #ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
        #ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
        ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
        ax.tick_params(axis='both',which='major',length=axmajlen)
        ax.tick_params(axis='both',which='minor',length=axminlen)
        ax.tick_params(axis='both',which='both',width=axwidth)
        ax.set_ylim(-0.05,0.7)
        #if zoom: ax.set_ylim(-0.05, 0.05)
        if imol == 1: ax.set_ylabel('(max fitscale $-$ min fitscale) $/$ med fitscale')
        if imol == 2: ax.set_xlabel('Days since time[0]')
        if imol < 2: ax.axes.xaxis.set_ticklabels([])
        ax.text(0.97, 0.94, molecules[imol], transform=ax.transAxes, ha='right', va='top', bbox=bboxpar)
        if imol == 0:
            ax.text(0.5, 1.02, r'Linear variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)

        xvals = data['JD']-np.nanmin(data['JD'])
        yvals = data['synthfrac'+str(imol+1)][:,1]# / data['MEDFIT'+str(imol+1)][:,ipar]
        c = expdata['NTELL']#[:,imol]

        med = np.nanmedian(yvals)
        #ax.axhline(med, color='grey', linestyle='dashed')
        #ax.text(0.75, 0.85, 'med RMS = ' + str("%.3f" % round(med,3)), transform=ax.transAxes, ha='center', va='center', bbox=bboxpar)
        #ax.scatter(xvals, yvals, marker='o', s=3, c=molcols[imol], alpha=0.8)#, vmin=vmin, vmax=vmax)#, edgecolors='k'
        sc1 = ax.scatter(xvals, yvals, marker='o', s=3, c=c, cmap=cmap, alpha=0.8, vmin=10, vmax=35)#, vmin=vmin, vmax=vmax)#, edgecolors='k'

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad="1%")
        cax.minorticks_on()
        #cax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
        cb1 = colorbar(sc1, cax=cax)
        ax.text(1.072, 0.5, r'$N$ tellurics',ha='left', va='center', rotation=-90, transform=ax.transAxes)


    fig.subplots_adjust(left=0.055,right=0.94,bottom=0.057,top=0.96,hspace=0.08,wspace=0.05)
    plt.savefig(plotfile)
    plt.close('all')

###########################################################################################
def tellspatial5(zoom=False):
    data = fits.getdata('/uufs/chpc.utah.edu/common/home/u0955897/projects/com/tellfit.fits')

    npars = 3
    molcols = ['mediumseagreen', 'purple', 'darkorange']

    plotfile = sdir5 + 'tellspatialRMS_med.png'
    if zoom: plotfile = plotfile.replace('.png', '_zoom.png')
    print('making ' + os.path.basename(plotfile))
    fig = plt.figure(figsize=(32,16))
    for imol in range(nmolecules):
        for ipar in range(npars):
            ax = plt.subplot2grid((nmolecules, npars), (imol,ipar%3))
            ax.minorticks_on()
            #ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            #ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
            ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
            ax.tick_params(axis='both',which='major',length=axmajlen)
            ax.tick_params(axis='both',which='minor',length=axminlen)
            ax.tick_params(axis='both',which='both',width=axwidth)
            ax.set_ylim(-0.05,1.1)
            if zoom: ax.set_ylim(-0.01, 0.1)
            if (ipar == 0) & (imol == 1): ax.set_ylabel('RMS (measured $-$ modeled) / Median (measured)')
            if imol == 2: ax.set_xlabel('Days since time[0]')
            if imol < 2: ax.axes.xaxis.set_ticklabels([])
            if ipar > 0: ax.axes.yaxis.set_ticklabels([])
            if ipar == 2: ax.text(1.02, 0.50, molecules[imol], transform=ax.transAxes, ha='left', va='center', rotation=-90, bbox=bboxpar)
            if imol == 0:
                if ipar == 0: ax.text(0.5, 1.02, r'Constant$/$no variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)
                if ipar == 1: ax.text(0.5, 1.02, r'Linear variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)
                if ipar == 2: ax.text(0.5, 1.02, r'Quadratic variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)

            xvals = data['JD']-np.nanmin(data['JD'])
            yvals = data['RMS'+str(imol+1)][:,ipar] / data['MEDOBS'+str(imol+1)]

            med = np.nanmedian(yvals)
            ax.axhline(med, color='grey', linestyle='dashed')
            ax.text(0.75, 0.85, 'med y = ' + str("%.3f" % round(med,3)), transform=ax.transAxes, ha='center', va='center', bbox=bboxpar)
            ax.scatter(xvals, yvals, marker='o', s=3, c=molcols[imol], alpha=0.8)#, vmin=vmin, vmax=vmax)#, edgecolors='k'

    fig.subplots_adjust(left=0.055,right=0.97,bottom=0.057,top=0.96,hspace=0.08,wspace=0.05)
    plt.savefig(plotfile)
    plt.close('all')

###########################################################################################
def tellspatial6(zoom=False):
    data = fits.getdata('/uufs/chpc.utah.edu/common/home/u0955897/projects/com/tellfit.fits')

    npars = 3
    molcols = ['mediumseagreen', 'purple', 'darkorange']

    plotfile = sdir5 + 'tellspatialRMS_sig.png'
    if zoom: plotfile = plotfile.replace('.png', '_zoom.png')
    print('making ' + os.path.basename(plotfile))
    fig = plt.figure(figsize=(32,16))
    for imol in range(nmolecules):
        for ipar in range(npars):
            ax = plt.subplot2grid((nmolecules, npars), (imol,ipar%3))
            ax.minorticks_on()
            #ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            #ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
            ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
            ax.tick_params(axis='both',which='major',length=axmajlen)
            ax.tick_params(axis='both',which='minor',length=axminlen)
            ax.tick_params(axis='both',which='both',width=axwidth)
            ax.set_ylim(-0.05,1.1)
            if zoom: ax.set_ylim(-0.01, 0.1)
            if (ipar == 0) & (imol == 1): ax.set_ylabel('$\sigma$ (synth)')
            if imol == 2: ax.set_xlabel('Days since time[0]')
            if imol < 2: ax.axes.xaxis.set_ticklabels([])
            if ipar > 0: ax.axes.yaxis.set_ticklabels([])
            if ipar == 2: ax.text(1.02, 0.50, molecules[imol], transform=ax.transAxes, ha='left', va='center', rotation=-90, bbox=bboxpar)
            if imol == 0:
                if ipar == 0: ax.text(0.5, 1.02, r'Constant$/$no variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)
                if ipar == 1: ax.text(0.5, 1.02, r'Linear variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)
                if ipar == 2: ax.text(0.5, 1.02, r'Quadratic variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)

            xvals = data['JD']-np.nanmin(data['JD'])
            yvals = data['SYNTHSIG'+str(imol+1)][:,ipar]# / data['MEDOBS'+str(imol+1)]

            med = np.nanmedian(yvals)
            ax.axhline(med, color='grey', linestyle='dashed')
            ax.text(0.75, 0.85, 'med y = ' + str("%.3f" % round(med,3)), transform=ax.transAxes, ha='center', va='center', bbox=bboxpar)
            ax.scatter(xvals, yvals, marker='o', s=3, c=molcols[imol], alpha=0.8)#, vmin=vmin, vmax=vmax)#, edgecolors='k'

    fig.subplots_adjust(left=0.055,right=0.97,bottom=0.057,top=0.96,hspace=0.08,wspace=0.05)
    plt.savefig(plotfile)
    plt.close('all')

###########################################################################################
def tellspatial7(zoom=False):
    data = fits.getdata('/uufs/chpc.utah.edu/common/home/u0955897/projects/com/tellfit.fits')

    npars = 3
    molcols = ['mediumseagreen', 'purple', 'darkorange']

    plotfile = sdir5 + 'tellspatialRMS_sigmean.png'
    if zoom: plotfile = plotfile.replace('.png', '_zoom.png')
    print('making ' + os.path.basename(plotfile))
    fig = plt.figure(figsize=(32,16))
    for imol in range(nmolecules):
        for ipar in range(npars):
            ax = plt.subplot2grid((nmolecules, npars), (imol,ipar%3))
            ax.minorticks_on()
            #ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            #ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
            ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
            ax.tick_params(axis='both',which='major',length=axmajlen)
            ax.tick_params(axis='both',which='minor',length=axminlen)
            ax.tick_params(axis='both',which='both',width=axwidth)
            ax.set_ylim(-0.05,1.1)
            if zoom: ax.set_ylim(-0.01, 0.2)
            if (ipar == 0) & (imol == 1): ax.set_ylabel('$\sigma$ / Median (synth)')
            if imol == 2: ax.set_xlabel('Days since time[0]')
            if imol < 2: ax.axes.xaxis.set_ticklabels([])
            if ipar > 0: ax.axes.yaxis.set_ticklabels([])
            if ipar == 2: ax.text(1.02, 0.50, molecules[imol], transform=ax.transAxes, ha='left', va='center', rotation=-90, bbox=bboxpar)
            if imol == 0:
                if ipar == 0: ax.text(0.5, 1.02, r'Constant$/$no variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)
                if ipar == 1: ax.text(0.5, 1.02, r'Linear variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)
                if ipar == 2: ax.text(0.5, 1.02, r'Quadratic variation', transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)

            xvals = data['JD']-np.nanmin(data['JD'])
            yvals = data['SYNTHSIG'+str(imol+1)][:,ipar] / data['MEDFIT'+str(imol+1)][:,ipar]

            med = np.nanmedian(yvals)
            ax.axhline(med, color='grey', linestyle='dashed')
            ax.text(0.75, 0.85, 'med y = ' + str("%.3f" % round(med,3)), transform=ax.transAxes, ha='center', va='center', bbox=bboxpar)
            ax.scatter(xvals, yvals, marker='o', s=3, c=molcols[imol], alpha=0.8)#, vmin=vmin, vmax=vmax)#, edgecolors='k'

    fig.subplots_adjust(left=0.055,right=0.97,bottom=0.057,top=0.96,hspace=0.08,wspace=0.05)
    plt.savefig(plotfile)
    plt.close('all')

###########################################################################################
def tellfitstats1(outfile='tellfitstats3.fits', mjdstart=59146, mjdstop=59647, 
                  remake=False, plot=True, plotx='MEANH', cmap='rainbow',
                  color=None):

    dir4 = specdir4 + 'visit/' + telescope + '/'

    if remake:
        print('remaking ' + outfile)
        gd, = np.where((allsnr['TELESCOPE'] == telescope) & (allsnr['MJD'] > mjdstart) & (allsnr['MJD'] < mjdstop))
        #allsnrg = allsnr[gd]
        #medsn = np.nanmedian(allsnrg['SN'][:,1])
        #gd, = np.where((allsnrg['MJD'] > mjdstart) & (allsnrg['MJD'] < mjdstop))# & (allsnrg['SN'][:,1] > medsn))
        mjdord = np.argsort(allsnr['MJD'][gd])
        allsnrg = allsnr[gd][mjdord]
        num = allsnrg['IM']
        field = allsnrg['FIELD']
        plate = allsnrg['PLATE']
        mjd = allsnrg['MJD']
        nexp = len(num)

        # Structure for exposure level info
        dt = np.dtype([('EXPNUM',       np.int32),
                       ('FIELD',        np.str, 30),
                       ('PLATE',        np.int32),
                       ('MJD',          np.int32),
                       ('JD',           np.float64),
                       ('DATEOBS',      np.str, 30),
                       ('SEEING',       np.float64),
                       ('ZERO',         np.float64),
                       ('MOONDIST',     np.float64),
                       ('MOONPHASE',    np.float64),
                       ('SECZ',         np.float64),
                       ('SKY',          np.float64),
                       ('SN',           np.float64),
                       ('NTELL',        np.int32),
                       ('MEANH',        np.float64),
                       ('SIGH',         np.float64),
                       ('MEANJK',       np.float64),
                       ('SIGJK',        np.float64),
                       ('NFIT',         np.int32, nmolecules),
                       ('FITMEANH',     np.float64, nmolecules),
                       ('FITSIGH',      np.float64, nmolecules),
                       ('FITMEANJK',    np.float64, nmolecules),
                       ('FITSIGJK',     np.float64, nmolecules),
                       ('MEANFITSCALE', np.float64, nmolecules),
                       ('MEDFITSCALE',  np.float64, nmolecules),
                       ('MADFITSCALE',  np.float64, nmolecules),
                       ('SIGFITSCALE',  np.float64, nmolecules),
                       ('MINFITSCALE',  np.float64, nmolecules),
                       ('MAXFITSCALE',  np.float64, nmolecules),
                       ('MEANSCALE',    np.float64, nmolecules),
                       ('MEDSCALE',     np.float64, nmolecules),
                       ('MADSCALE',     np.float64, nmolecules),
                       ('SIGSCALE',     np.float64, nmolecules),
                       ('MINSCALE',     np.float64, nmolecules),
                       ('MAXSCALE',     np.float64, nmolecules),
                       ('MEANCHISQ',    np.float64, nmolecules),
                       ('MEDCHISQ',     np.float64, nmolecules),
                       ('MADCHISQ',     np.float64, nmolecules),
                       ('SIGCHISQ',     np.float64, nmolecules),
                       ('SIGETA',       np.float64, nmolecules),
                       ('SIGZETA',      np.float64, nmolecules),
                       ('NREJ',         np.int32, nmolecules)])
        out = np.zeros(nexp, dtype=dt)

        # Structure for individual star level info
        dtstar = np.dtype([('APOGEE_ID', np.str, 18),
                           ('RA',        np.float64),
                           ('DEC',       np.float64),
                           ('ETA',       np.float64),
                           ('ZETA',      np.float64),
                           ('JMAG',      np.float64),
                           ('HMAG',      np.float64),
                           ('KMAG',      np.float64),
                           ('EXPNUM',    np.int32),
                           ('FIELD',     np.str, 30),
                           ('PLATE',     np.int32),
                           ('MJD',       np.int32),
                           ('JD',        np.float64),
                           ('DATEOBS',   np.str, 30),
                           ('SEEING',    np.float64),
                           ('ZERO',      np.float64),
                           ('MOONDIST',  np.float64),
                           ('MOONPHASE', np.float64),
                           ('SECZ',      np.float64),
                           ('SKY',       np.float64),
                           ('SN',        np.float64),
                           ('NTELL',     np.int32),
                           ('NFIT',      np.float64, nmolecules),
                           ('BESTMOD',   np.int32, nmolecules),
                           ('SCALE',     np.float64, nmolecules),
                           ('FITSCALE',  np.float64, nmolecules),
                           ('RCHISQ',    np.float64, nmolecules),
                           ('MAG',       np.float64, nmolecules),
                           ('REJ',       np.int32, nmolecules)])

        for i in range(nexp):
        #for i in range(5):
            sloan4 = False
            print('(' + str(i+1) + '/' + str(nexp) + '): field = ' + field[i] + ', plate = ' + str(plate[i]) + ', mjd = ' + str(mjd[i]) + ', exp = ' + str(num[i]))
            cframe = load.filename('Cframe', field=field[i], plate=plate[i], mjd=str(mjd[i]), num=num[i], chips=True)
            cframe = cframe.replace('apCframe-', 'apCframe-a-')
            if os.path.exists(cframe) == False:
                tmp = glob.glob(dir4 + field[i] + '/' + str(plate[i]) + '/' + str(mjd[i]) + '/' + os.path.basename(cframe))
                if len(tmp) > 0:
                    cframe = tmp[0]
                    sloan4 = True
                else:
                    continue
            if os.path.exists(cframe):
                magnames = ['JMAG', 'HMAG', 'KMAG']
                if sloan4: magnames = ['J', 'H', 'K']

                tellfit = fits.getdata(cframe,13)
                plugmap = fits.getdata(cframe,11)
                bestmod = np.squeeze(tellfit['BESTMOD'])
                nfit = np.squeeze(tellfit['NFIT'])
                scale = np.squeeze(tellfit['SCALE'])
                fitscale = np.squeeze(tellfit['FITSCALE'])
                rchisq = np.squeeze(tellfit['RCHISQ'])
                mag = np.squeeze(tellfit['MAG'])
                eta = np.squeeze(tellfit['ETA'])
                zeta = np.squeeze(tellfit['ZETA'])

                hdr = np.array(fits.getheader(cframe)['HISTORY'])
                tellind = np.flatnonzero(np.core.defchararray.find(hdr,'APTELLURIC: Fiber=')!=-1)
                if len(tellind) > 0:
                    hdr = hdr[tellind]
                    ntell = len(hdr) // 2

                    tfitscale = np.zeros((ntell,3))
                    tscale = np.zeros((ntell,3))
                    tellfib = np.zeros(ntell)
                    for itel in range(ntell):
                        tellfib[itel] = int(hdr[itel+ntell].split('=')[1].split(' Norm')[0])
                        tfitscale[itel, 0] = float(hdr[itel].split('Norm=')[1].split(', ')[0])
                        tfitscale[itel, 1] = float(hdr[itel].split('Norm=')[1].split(', ')[1])
                        tfitscale[itel, 2] = float(hdr[itel].split('Norm=')[1].split(', ')[2])
                        tscale[itel, 0] = float(hdr[itel+ntell].split('Norm=')[1].split(', ')[0])
                        tscale[itel, 1] = float(hdr[itel+ntell].split('Norm=')[1].split(', ')[1])
                        tscale[itel, 2] = float(hdr[itel+ntell].split('Norm=')[1].split(', ')[2])
                    g1, = np.where(tfitscale[:,0] > 0)
                    g2, = np.where(tfitscale[:,1] > 0)
                    g3, = np.where(tfitscale[:,2] > 0)
                    tellfibindex1 = (300 - tellfib[g1]).astype(int)
                    tellfibindex2 = (300 - tellfib[g2]).astype(int)
                    tellfibindex3 = (300 - tellfib[g3]).astype(int)
                    tellfibindex = (300 - tellfib).astype(int)
                    pmap = plugmap[tellfibindex]

                    #tell, = np.where((plugmap['objtype'] == 'HOT_STD') & (np.isnan(plugmap['HMAG']) == False) & (plugmap['HMAG'] < 15) & (plugmap['HMAG'] > 5))
                    #ntell = len(tell)

                    out['EXPNUM'][i] = num[i]
                    out['FIELD'][i] = field[i]
                    out['PLATE'][i] = plate[i]
                    out['MJD'][i] = mjd[i]
                    out['JD'][i] = allsnrg['JD'][i]
                    out['DATEOBS'][i] = allsnrg['DATEOBS'][i]
                    out['SEEING'][i] = allsnrg['SEEING'][i]
                    out['ZERO'][i] = allsnrg['ZERO'][i]
                    out['MOONDIST'][i] = allsnrg['MOONDIST'][i]
                    out['MOONPHASE'][i] = allsnrg['MOONPHASE'][i]
                    out['SECZ'][i] = allsnrg['SECZ'][i]
                    out['SKY'][i] = np.nanmean(allsnrg['SKY'][i])
                    out['SN'][i] = np.nanmean(allsnrg['SN'][i])
                    out['NTELL'][i] = ntell
                    out['MEANH'][i] = np.nanmean(plugmap[magnames[1]][tellfibindex])
                    out['SIGH'][i] = np.nanstd(plugmap[magnames[1]][tellfibindex])
                    out['MEANJK'][i] = np.nanmean(plugmap[magnames[0]][tellfibindex] - plugmap[magnames[2]][tellfibindex])
                    out['SIGJK'][i] = np.nanstd(plugmap[magnames[0]][tellfibindex] - plugmap[magnames[2]][tellfibindex])

                    out['NFIT'][i,0] = len(g1); out['NFIT'][i,1] = len(g2); out['NFIT'][i,2] = len(g3)
                    out['FITMEANH'][i,0] = np.nanmean(plugmap[magnames[1]][tellfibindex1])
                    out['FITSIGH'][i,0] = np.nanstd(plugmap[magnames[1]][tellfibindex1])
                    out['FITMEANJK'][i,0] = np.nanmean(plugmap[magnames[0]][tellfibindex1] - plugmap[magnames[2]][tellfibindex1])
                    out['FITSIGJK'][i,0] = np.nanstd(plugmap[magnames[0]][tellfibindex1] - plugmap[magnames[2]][tellfibindex1])

                    out['FITMEANH'][i,1] = np.nanmean(plugmap[magnames[1]][tellfibindex2])
                    out['FITSIGH'][i,1] = np.nanstd(plugmap[magnames[1]][tellfibindex2])
                    out['FITMEANJK'][i,1] = np.nanmean(plugmap[magnames[0]][tellfibindex2] - plugmap[magnames[2]][tellfibindex2])
                    out['FITSIGJK'][i,1] = np.nanstd(plugmap[magnames[0]][tellfibindex2] - plugmap[magnames[2]][tellfibindex2])

                    out['FITMEANH'][i,2] = np.nanmean(plugmap[magnames[1]][tellfibindex3])
                    out['FITSIGH'][i,2] = np.nanstd(plugmap[magnames[1]][tellfibindex3])
                    out['FITMEANJK'][i,2] = np.nanmean(plugmap[magnames[0]][tellfibindex3] - plugmap[magnames[2]][tellfibindex3])
                    out['FITSIGJK'][i,2] = np.nanstd(plugmap[magnames[0]][tellfibindex3] - plugmap[magnames[2]][tellfibindex3])

                    out['MEANFITSCALE'][i,0] = np.nanmean(tfitscale[g1,0])
                    out['MEANFITSCALE'][i,1] = np.nanmean(tfitscale[g2,1])
                    out['MEANFITSCALE'][i,2] = np.nanmean(tfitscale[g3,2])

                    out['MEDFITSCALE'][i,0] = np.nanmedian(tfitscale[g1,0])
                    out['MEDFITSCALE'][i,1] = np.nanmedian(tfitscale[g2,1])
                    out['MEDFITSCALE'][i,2] = np.nanmedian(tfitscale[g3,2])

                    out['MADFITSCALE'][i,0] = dln.mad(tfitscale[g1,0])
                    out['MADFITSCALE'][i,1] = dln.mad(tfitscale[g2,1])
                    out['MADFITSCALE'][i,2] = dln.mad(tfitscale[g3,2])

                    out['SIGFITSCALE'][i,0] = np.std(tfitscale[g1,0])
                    out['SIGFITSCALE'][i,1] = np.std(tfitscale[g2,1])
                    out['SIGFITSCALE'][i,2] = np.std(tfitscale[g3,2])

                    try:
                        out['MINFITSCALE'][i,0] = np.nanmin(tfitscale[g1,0])
                        out['MINFITSCALE'][i,1] = np.nanmin(tfitscale[g2,1])
                        out['MINFITSCALE'][i,2] = np.nanmin(tfitscale[g3,2])

                        out['MAXFITSCALE'][i,0] = np.nanmax(tfitscale[g1,0])
                        out['MAXFITSCALE'][i,1] = np.nanmax(tfitscale[g2,1])
                        out['MAXFITSCALE'][i,2] = np.nanmax(tfitscale[g3,2])
                    except:
                        nothing = 1

                    gg1, = np.where(scale[0,:] > 0)
                    gg2, = np.where(scale[1,:] > 0)
                    gg3, = np.where(scale[2,:] > 0)

                    out['MEANSCALE'][i,0] = np.nanmean(scale[0,gg1])
                    out['MEANSCALE'][i,1] = np.nanmean(scale[1,gg2])
                    out['MEANSCALE'][i,2] = np.nanmean(scale[2,gg3])

                    out['MEDSCALE'][i,0] = np.nanmedian(scale[0,gg1])
                    out['MEDSCALE'][i,1] = np.nanmedian(scale[1,gg2])
                    out['MEDSCALE'][i,2] = np.nanmedian(scale[2,gg3])

                    out['MADSCALE'][i,0] = dln.mad(scale[0,gg1])
                    out['MADSCALE'][i,1] = dln.mad(scale[1,gg2])
                    out['MADSCALE'][i,2] = dln.mad(scale[2,gg3])

                    out['SIGSCALE'][i,0] = np.std(scale[0,gg1])
                    out['SIGSCALE'][i,1] = np.std(scale[1,gg2])
                    out['SIGSCALE'][i,2] = np.std(scale[2,gg3])

                    try:
                        out['MINSCALE'][i,0] = np.nanmin(scale[0,gg1])
                        out['MINSCALE'][i,1] = np.nanmin(scale[1,gg2])
                        out['MINSCALE'][i,2] = np.nanmin(scale[2,gg3])

                        out['MAXSCALE'][i,0] = np.nanmax(scale[0,gg1])
                        out['MAXSCALE'][i,1] = np.nanmax(scale[1,gg2])
                        out['MAXSCALE'][i,2] = np.nanmax(scale[2,gg3])
                    except:
                        nothing = 1

                    out['NREJ'][i,0] = int(ntell - len(g1))
                    out['NREJ'][i,1] = int(ntell - len(g2))
                    out['NREJ'][i,2] = int(ntell - len(g3))

                    #out['MAD1'][i] = dln.mad(fitscale[0, tell[gd]])
                    #out['MADRESID1'][i] = dln.mad(fitscale[0, tell[gd]] - scale[0, tell[gd]])
                    #gd, = np.where(fitscale[1, tell] > 0)
                    #out['MAD2'][i] = dln.mad(fitscale[1, tell])
                    #out['MADRESID2'][i] = dln.mad(fitscale[1, tell[gd]] - scale[1, tell[gd]])
                    #gd, = np.where(fitscale[2, tell] > 0)
                    #out['MAD3'][i] = dln.mad(fitscale[2, tell])
                    #out['MADRESID3'][i] = dln.mad(fitscale[2, tell[gd]] - scale[2, tell[gd]])


                    outstar = np.empty(ntell, dtype=dtstar)
                    outstar['APOGEE_ID'] = plugmap['TMASS_STYLE'][tellfibindex]
                    outstar['RA'] =        plugmap['RA'][tellfibindex]
                    outstar['DEC'] =       plugmap['DEC'][tellfibindex]
                    outstar['ETA'] =       plugmap['ETA'][tellfibindex]
                    outstar['ZETA'] =      plugmap['ZETA'][tellfibindex]
                    outstar['JMAG'] =      plugmap[magnames[0]][tellfibindex]
                    outstar['HMAG'] =      plugmap[magnames[1]][tellfibindex]
                    outstar['KMAG'] =      plugmap[magnames[2]][tellfibindex]
                    outstar['EXPNUM'] =    np.full(ntell, num[i])
                    outstar['FIELD'] =     np.full(ntell, field[i])
                    outstar['PLATE'] =     np.full(ntell, plate[i])
                    outstar['MJD'] =       np.full(ntell, mjd[i])
                    outstar['JD'] =        np.full(ntell, allsnrg['JD'][i])
                    outstar['DATEOBS'] =   np.full(ntell, allsnrg['DATEOBS'][i])
                    outstar['SEEING'] =    np.full(ntell, allsnrg['SEEING'][i])
                    outstar['ZERO'] =      np.full(ntell, allsnrg['ZERO'][i])
                    outstar['MOONDIST'] =  np.full(ntell, allsnrg['MOONDIST'][i])
                    outstar['MOONPHASE'] = np.full(ntell, allsnrg['MOONPHASE'][i])
                    outstar['SECZ'] =      np.full(ntell, allsnrg['SECZ'][i])
                    outstar['NTELL'] =     np.full(ntell, ntell)
                    outstar['NFIT'][:,0] = np.full(ntell, len(g1))
                    outstar['NFIT'][:,1] = np.full(ntell, len(g2))
                    outstar['NFIT'][:,2] = np.full(ntell, len(g3))
                    outstar['BESTMOD'][:,0] = np.full(ntell, bestmod[0])
                    outstar['BESTMOD'][:,1] = np.full(ntell, bestmod[1])
                    outstar['BESTMOD'][:,2] = np.full(ntell, bestmod[2])
                    outstar['SCALE'][g1,0] =  tscale[g1, 0]
                    outstar['SCALE'][g2,1] =  tscale[g1, 1]
                    outstar['SCALE'][g3,2] =  tscale[g1, 2]
                    outstar['FITSCALE'][g1,0] =  tfitscale[g1, 0]
                    outstar['FITSCALE'][g2,1] =  tfitscale[g2, 0]
                    outstar['FITSCALE'][g3,2] =  tfitscale[g3, 0]

                    rej1, = np.where(outstar['FITSCALE'][:,0] == 0)
                    rej2, = np.where(outstar['FITSCALE'][:,1] == 0)
                    rej3, = np.where(outstar['FITSCALE'][:,2] == 0)
                    if len(rej1) > 0: outstar['REJ'][rej1,0] = 1
                    if len(rej2) > 0: outstar['REJ'][rej2,1] = 1
                    if len(rej3) > 0: outstar['REJ'][rej3,2] = 1

                    #pdb.set_trace()

                    #outstar['FITSCALE1'] = fitscale[0, tell]
                    #outstar['SCALE2'] =    scale[1, tell]
                    #outstar['FITSCALE2'] = fitscale[1, tell]
                    #outstar['SCALE3'] =    scale[2, tell]
                    #outstar['FITSCALE3'] = fitscale[2, tell]

                    if i == 0:
                        outS = outstar
                    else:
                        outS = np.concatenate([outS, outstar])

        gd, = np.where(out['EXPNUM'] > 0)
        print('writing ' + str(len(gd)) + ' results to ' + outfile)
        Table(out[gd]).write(outfile, overwrite=True)

        starfile = outfile.replace('.fits', '_stardata.fits')
        print('making ' + starfile)
        Table(outS).write(starfile, overwrite=True)

    out = fits.getdata(outfile)

    if color == 'seeing':
        gd, = np.where((np.isnan(out[color]) == False) & (out[color] > 0))
        out = out[gd]

    if plot:
        plotfile = sdir5 + 'tellfitstats1_' + plotx + '.png'
        if color is not None: plotfile = plotfile.replace('.png', '_'+color+'.png')
        print('making ' + os.path.basename(plotfile))
        fig = plt.figure(figsize=(32,16))
        for imol in range(nmolecules):
            ax1 = plt.subplot2grid((2,nmolecules), (0,imol))
            ax2 = plt.subplot2grid((2,nmolecules), (1,imol))
            axes = [ax1,ax2]
            for ax in axes:
                ax.minorticks_on()
                ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
                ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
                ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
                ax.tick_params(axis='both',which='major',length=axmajlen)
                ax.tick_params(axis='both',which='minor',length=axminlen)
                ax.tick_params(axis='both',which='both',width=axwidth)
                if plotx == 'MEANH': ax.set_xlim(7.3, 10.7)
                if plotx == 'MEANJK': ax.set_xlim(-0.1, 0.43)
            if imol == 0:
                ax1.set_ylabel('MAD (fitscale)')
                ax2.set_ylabel(r'MAD (fitscale$-$scale)')
            ax1.text(0.5, 1.02, molecules[imol], transform=ax1.transAxes, ha='center', va='bottom', bbox=bboxpar)
            ax1.axes.xaxis.set_ticklabels([])
            if plotx == 'MEANH': ax2.set_xlabel(r'Mean Telluric $H$')
            if plotx == 'MEANJK': ax2.set_xlabel(r'Mean Telluric $J-K$')

            xvals = out[plotx]
            yvals1 = out['MAD'+str(imol+1)]
            yvals2 = out['MADRESID'+str(imol+1)]
            #print(np.min(out['MEANJK']))
            #print(np.max(out['MEANJK']))
            #print(np.min(out['MEANH']))
            #print(np.max(out['MEANH']))
            if plotx == 'MEANH': 
                vmin = -0.078
                vmax = 0.41
                c = out['MEANJK']
            if plotx == 'MEANJK':
                vmin = 7.489
                vmax = 10.544
                c = out['MEANH']
            if color is not None: c = out[color]
            if color == 'seeing':
                vmin = 0.85
                vmax = 2.5
            if color == 'secz':
                vmin = 1
                vmax = 1.5

            sc1 = ax1.scatter(xvals, yvals1, marker='o', s=10, cmap=cmap, c=c, alpha=0.8, vmin=vmin, vmax=vmax)#, edgecolors='k'
            sc2 = ax2.scatter(xvals, yvals2, marker='o', s=10, cmap=cmap, c=c, alpha=0.8, vmin=vmin, vmax=vmax)#, edgecolors='k'

            if imol == 2:
                ii = 0
                for ax in axes:
                    ax_divider = make_axes_locatable(ax)
                    cax = ax_divider.append_axes("right", size="4%", pad="3%")
                    cb1 = colorbar(sc1, cax=cax, orientation="vertical")
                    cax.minorticks_on()
                    if color is not None:
                        if color == 'seeing':
                            ax.text(1.18, 0.5, r'Seeing',ha='left', va='center', rotation=-90, transform=ax.transAxes)
                        if color == 'secz':
                            ax.text(1.18, 0.5, r'sec $z$',ha='left', va='center', rotation=-90, transform=ax.transAxes)
                    else:
                        if plotx == 'MEANH': 
                            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
                            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                            cax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
                            cax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
                            ax.text(1.18, 0.5, r'$J-K$',ha='left', va='center', rotation=-90, transform=ax.transAxes)
                        if plotx == 'MEANJK': 
                            ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
                            ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.05))
                            cax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
                            cax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))
                            ax.text(1.18, 0.5, r'$H$',ha='left', va='center', rotation=-90, transform=ax.transAxes)
                    ii += 1

        fig.subplots_adjust(left=0.04,right=0.95,bottom=0.057,top=0.96,hspace=0.08,wspace=0.12)
        plt.savefig(plotfile)
        plt.close('all')

    return out

###########################################################################################
def tellfitstats2(infile='tellfitstats2.fits', plotx='seeing', color=None):
    out = fits.getdata(infile)

    plotfile = sdir5 + 'tellfitstats_' + plotx + '.png'
    if color is not None: plotfile = plotfile.replace('.png', '_'+color+'.png')
    print('making ' + os.path.basename(plotfile))
    fig = plt.figure(figsize=(32,16))
    for imol in range(nmolecules):
        ax1 = plt.subplot2grid((2,nmolecules), (0,imol))
        ax2 = plt.subplot2grid((2,nmolecules), (1,imol))
        axes = [ax1,ax2]
        for ax in axes:
            ax.minorticks_on()
            ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
            ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
            ax.tick_params(axis='both',which='major',length=axmajlen)
            ax.tick_params(axis='both',which='minor',length=axminlen)
            ax.tick_params(axis='both',which='both',width=axwidth)
            if plotx == 'MEANH': ax.set_xlim(7.3, 10.7)
            if plotx == 'MEANJK': ax.set_xlim(-0.1, 0.43)
        if imol == 0:
            ax1.set_ylabel('MAD (fitscale)')
            ax2.set_ylabel(r'MAD (fitscale$-$scale)')

        ax1.text(0.5, 1.02, molecules[imol], transform=ax1.transAxes, ha='center', va='bottom', bbox=bboxpar)
        ax1.axes.xaxis.set_ticklabels([])
        if plotx == 'seeing': ax2.set_xlabel(r'seeing')
        if plotx == 'secz': ax2.set_xlabel(r'sec $z$')
        if plotx == 'MEANH': ax2.set_xlabel(r'Mean Telluric $H$')
        if plotx == 'MEANJK': ax2.set_xlabel(r'Mean Telluric $J-K$')

        xvals = out[plotx]
        yvals1 = out['MAD'+str(imol+1)]
        yvals2 = out['MADRESID'+str(imol+1)]
        #print(np.min(out['MEANJK']))
        #print(np.max(out['MEANJK']))
        #print(np.min(out['MEANH']))
        #print(np.max(out['MEANH']))
        if plotx == 'MEANH': 
            vmin = -0.078
            vmax = 0.41
            #c = out['MEANJK']
        if plotx == 'MEANJK':
            vmin = 7.489
            vmax = 10.544
            #c = out['MEANH']
        #if color is not None: c = out[color]
        if color == 'seeing':
            vmin = 0.85
            vmax = 2.5
        if color == 'secz':
            vmin = 1
            vmax = 1.5

        sc1 = ax1.scatter(xvals, yvals1, marker='o', s=10, c='cyan', edgecolor='k', alpha=0.8)#, vmin=vmin, vmax=vmax)#, edgecolors='k'
        sc2 = ax2.scatter(xvals, yvals2, marker='o', s=10, c='cyan', edgecolor='k', alpha=0.8)#, vmin=vmin, vmax=vmax)#, edgecolors='k'

    fig.subplots_adjust(left=0.04,right=0.95,bottom=0.057,top=0.96,hspace=0.08,wspace=0.12)
    plt.savefig(plotfile)
    plt.close('all')

###########################################################################################
def tellfitstats3(infile='tellfitstats2.fits', plotx='seeing', color=None):
    out = fits.getdata(infile)

    plotfile = sdir5 + 'tellfitstats_MAD2.png'
    if color is not None: plotfile = plotfile.replace('.png', '_'+color+'.png')
    print('making ' + os.path.basename(plotfile))
    fig = plt.figure(figsize=(32,10))
    for imol in range(nmolecules):
        ax = plt.subplot2grid((1,nmolecules), (0,imol))
        ax.minorticks_on()
        ax.set_xlim(0.0, 0.08)
        ax.set_ylim(0.0, 0.08)
        ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
        ax.tick_params(axis='both',which='major',length=axmajlen)
        ax.tick_params(axis='both',which='minor',length=axminlen)
        ax.tick_params(axis='both',which='both',width=axwidth)
        ax.set_xlabel('MAD (fitscale)')
        if imol == 0: ax.set_ylabel(r'MAD (fitscale$-$scale)')
        if imol > 0: ax.axes.yaxis.set_ticklabels([])
        ax.text(0.5, 1.02, molecules[imol], transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)
        ax.plot([-100,100], [-100,100], linestyle='dashed', color='grey')

        xvals = out['MAD'+str(imol+1)]
        yvals = out['MADRESID'+str(imol+1)]

        sc1 = ax.scatter(xvals, yvals, marker='o', s=10, c='cyan', edgecolor='k', alpha=0.8)#, vmin=vmin, vmax=vmax)#, edgecolors='k'

    fig.subplots_adjust(left=0.045,right=0.985,bottom=0.085,top=0.94,hspace=0.08,wspace=0.08)
    plt.savefig(plotfile)
    plt.close('all')

###########################################################################################
def tellfitstats4(infile='tellfitstats2_stardata.fits', cmap='rainbow', nbins=40,
                  vmin=[0, 0, 0], vmax=[0.1, 0.035, 0.4], doall=True):

    plotfile = sdir5 + 'tellfitstats_indstars.png'
    if doall: 
        infile='tellfitstats_all_stardata.fits'
        plotfile = plotfile.replace('.png', '_all.png')

    print('making ' + os.path.basename(plotfile))

    data = fits.getdata(infile)

    fig = plt.figure(figsize=(32,10))
    for imol in range(nmolecules):
        g, = np.where((data['FITSCALE1'] > 0) & (data['HMAG'] <=11) & (data['JMAG']-data['KMAG'] < 0.58))
        gdata = data[g]

        ax = plt.subplot2grid((1,nmolecules), (0,imol))
        ax.minorticks_on()
        ax.set_ylim(11.2, 6.8)
        ax.set_xlim(-0.2, 0.53)
        ax.set_xlabel(r'$J-K$')
        if imol == 0: ax.set_ylabel(r'$H$')
        ax.tick_params(axis='both',which='both',direction='out',bottom=True,top=True,left=True,right=True)
        ax.tick_params(axis='both',which='major',length=axmajlen)
        ax.tick_params(axis='both',which='minor',length=axminlen)
        ax.tick_params(axis='both',which='both',width=axwidth)
        if imol > 0: ax.axes.yaxis.set_ticklabels([])
        ax.text(0.5, 1.02, molecules[imol], transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)

        x = gdata['JMAG'] - gdata['KMAG']
        y = gdata['HMAG']
        values = gdata['FITSCALE'+str(imol+1)]# - data['SCALE1']
        ret = stats.binned_statistic_2d(x, y, values, statistic=dln.mad, bins=(nbins,nbins))
        ext = [ret.x_edge[0], ret.x_edge[-1:][0], ret.y_edge[-1:][0], ret.y_edge[0]]
        im = ax.imshow(ret.statistic, cmap=cmap, aspect='auto', origin='upper', extent=ext, vmin=vmin[imol], vmax=vmax[imol])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad="2%")
        cax.minorticks_on()
        cax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
        cb1 = colorbar(im, cax=cax)
        if imol == 2:
            ax.text(1.19, 0.5, r'MAD (measured)',ha='left', va='center', rotation=-90, transform=ax.transAxes)

    fig.subplots_adjust(left=0.045,right=0.945,bottom=0.093,top=0.94,hspace=0.08,wspace=0.18)
    plt.savefig(plotfile)
    plt.close('all')

###########################################################################################
def tellfitstats5(infile='tellfitstats2_stardata.fits', cmap='rainbow', nbins=40,
                  vmin=[0.01, 0.01, 0.01], vmax=[0.03, 0.03, 0.03], doall=True,
                  statistic=dln.mad):

    if statistic == dln.mad:
        vmin=[0.01, 0.01, 0.01]
        vmax=[0.03, 0.03, 0.03]
    if statistic == 'count':
        vmin=[0, 0, 0]
        vmax=[1000, 1000, 1000]

    plotfile = sdir5 + 'tellfitstats_indstars_fitscale-scale.png'
    if doall: 
        infile='tellfitstats_all_stardata.fits'
        plotfile = plotfile.replace('.png', '_all.png')

    print('making ' + os.path.basename(plotfile))

    data = fits.getdata(infile)
    if statistic == 'count':
        uname,uind = np.unique(data['APOGEE_ID'], return_index=True)
        data = data[uind]

    fig = plt.figure(figsize=(32,10))
    for imol in range(nmolecules):
        g, = np.where((data['FITSCALE1'] > 0) & (data['HMAG'] <= 11) & (data['HMAG'] >= 6) & (data['JMAG']-data['KMAG'] < 0.5))
        gdata = data[g]

        ax = plt.subplot2grid((1,nmolecules), (0,imol))
        ax.minorticks_on()
        ax.set_ylim(11, 6)
        ax.set_xlim(-0.2, 0.5)
        ax.set_xlabel(r'$J-K$')
        if imol == 0: ax.set_ylabel(r'$H$')
        ax.tick_params(axis='both',which='both',direction='out',bottom=True,top=True,left=True,right=True)
        ax.tick_params(axis='both',which='major',length=axmajlen)
        ax.tick_params(axis='both',which='minor',length=axminlen)
        ax.tick_params(axis='both',which='both',width=axwidth)
        if imol > 0: ax.axes.yaxis.set_ticklabels([])
        ax.text(0.5, 1.02, molecules[imol], transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)

        x = gdata['JMAG'] - gdata['KMAG']
        y = gdata['HMAG']
        values = gdata['FITSCALE'+str(imol+1)] - gdata['SCALE'+str(imol+1)]
        if statistic != 'count':
            ret = stats.binned_statistic_2d(x, y, values, statistic=statistic, bins=(nbins,nbins))
            ext = [ret.x_edge[0], ret.x_edge[-1:][0], ret.y_edge[-1:][0], ret.y_edge[0]]
            im = ax.imshow(ret.statistic, cmap=cmap, aspect='auto', origin='upper', extent=ext, vmin=vmin[imol], vmax=vmax[imol])
            #print(ext)
        else:
            H, yedges, xedges = np.histogram2d(y, x, bins=nbins)
            im = ax.pcolormesh(xedges, yedges, H, cmap=cmap, vmin=0, vmax=150)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad="2%")
        cax.yaxis.set_major_locator(ticker.MultipleLocator(0.005))
        cax.yaxis.set_minor_locator(ticker.MultipleLocator(0.001))
        cax.minorticks_on()
        cb1 = colorbar(im, cax=cax)
        if imol == 2:
            if statistic == dln.mad: ax.text(1.195, 0.5, r'MAD (measured$-$fit)',ha='left', va='center', rotation=-90, transform=ax.transAxes)
            if statistic == 'count': ax.text(1.195, 0.5, r'$N$ stars',ha='left', va='center', rotation=-90, transform=ax.transAxes)
            if statistic == 'median': ax.text(1.195, 0.5, r'Median (measured$-$fit)',ha='left', va='center', rotation=-90, transform=ax.transAxes)

    fig.subplots_adjust(left=0.04,right=0.945,bottom=0.093,top=0.94,hspace=0.08,wspace=0.18)
    plt.savefig(plotfile)
    plt.close('all')

###########################################################################################
def tellfitstatsgrid(infile='tellfitstats_all_stardata.fits', cmap='rainbow', nbins=40, doall=True,
                     do4=False, do5=False, ntell=15):

    #infile = '/uufs/chpc.utah.edu/common/home/u0955897/projects/com/tellfitstats_stardata.fits'
    #expdata = fits.getdata('/uufs/chpc.utah.edu/common/home/u0955897/projects/com/tellfitstats.fits')

    infile = 'tellfitstats_all_stardata.fits'
    expdata = fits.getdata('tellfitstats_all.fits')

    xmin = -0.2
    xmax = 0.5
    ymin = 6
    ymax = 11

    plotfile = sdir5 + 'tellfitstats_indstars_grid.png'
    if ntell == 15: plotfile = sdir5 + 'tellfitstats_indstars_grid15.png'
    if ntell == 'lt15': plotfile = sdir5 + 'tellfitstats_indstars_gridlt15.png'
    if ntell == 'gt15': plotfile = sdir5 + 'tellfitstats_indstars_gridgt15.png'
    #if doall: 
        #infile='tellfitstats_all_stardata.fits'
        #plotfile = plotfile.replace('.png', '_all.png')

    print('making ' + os.path.basename(plotfile))

    data = fits.getdata(infile)
    #ntellmean = np.nanmean(expdata['NFIT'], axis=1) 
    if ntell is not None:
        #gd, = np.where((ntellmean >= 10) & (ntellmean <= 15))
        #gd, = np.where((ntellmean > 15))# & (ntellmean <= 15))
        if ntell == 15: gd, = np.where(expdata['NTELL'] == 15)
        if ntell == 'lt15': gd, = np.where(expdata['NTELL'] < 15)
        if ntell == 'gt15': gd, = np.where(expdata['NTELL'] > 15)
        expnum = expdata['EXPNUM'][gd]
        mask = np.in1d(data['expnum'], expnum)
        gd, = np.where(mask == True)
        data = data[gd]

    #xy,x_ind,y_ind = np.intersect1d(expnum, data['expnum'], return_indices=True)
    #pdb.set_trace()

    g, = np.where((data['HMAG'] <= ymax) & (data['HMAG'] >= ymin) & (data['JMAG']-data['KMAG'] < xmax))
    data = data[g]

    statistics = ['count', 'median fitscale', 'median scale', 'mad diff']
    cmaps = ['gnuplot2_r', 'rainbow', 'rainbow', 'rainbow']
    nrows = len(statistics)

    matplotlib.rcParams.update({'font.size':32, 'font.family':'serif'})
    fig = plt.figure(figsize=(32,28))
    for irow in range(nrows):
        print(statistics[irow])
        if statistics[irow] == 'count':
            vmin = [0, 0, 0]
            vmax = [150, 150, 150]
            if doall is False: vmax = [25, 25, 25]
            if ntell == 15: vmax = [50, 50, 50]
            if ntell == 'lt15': vmax = [12, 12, 12]
        if statistics[irow][0:6] == 'median':
            vmin = [0.98, 1.085, 0.44]
            vmax = [1.10, 1.155, 0.97]
            #if doall is False: 
            #    vmin = [1.03, 1.1085, 0.35]
            #    vmax = [1.11, 1.130, 0.98]
            #if ntell == 15: 
            #    vmin = [0.98, 1.085, 0.50]
            #    vmax = [1.08, 1.13, 1.00]
            #if ntell == 'lt15': 
            #    vmin = [0.98, 1.085, 0.50]
            #    vmax = [1.08, 1.13, 1.00]
            #if ntell == 'gt15': 
            #    vmin = [0.98, 1.085, 0.50]
            #    vmax = [1.08, 1.13, 1.00]
        if statistics[irow] == 'mad diff':
            vmin=[0.008, 0.008, 0.008]
            vmax=[0.030, 0.030, 0.030]
            #if ntell == 15: 
            #    vmin = [0.008, 0.008, 0.008]
            #    vmax = [0.028, 0.028, 0.028]

        for imol in range(nmolecules):
            print(imol)
            #g, = np.where(data['FITSCALE'][:,imol] > 0)
            g, = np.where(data['FITSCALE'+str(imol+1)] > 0)
            gdata = data[g]

            if statistics[irow] == 'count':
                uname,uind = np.unique(gdata['APOGEE_ID'], return_index=True)
                gdata = gdata[uind]

            ax = plt.subplot2grid((nrows,nmolecules), (irow,imol))
            ax.minorticks_on()
            ax.set_ylim(ymax, ymin)
            ax.set_xlim(xmin, xmax)
            if irow == nrows-1: ax.set_xlabel(r'$J-K$')
            if imol == 0: ax.set_ylabel(r'$H$')
            ax.tick_params(axis='both',which='both',direction='out',bottom=True,top=True,left=True,right=True)
            ax.tick_params(axis='both',which='major',length=axmajlen)
            ax.tick_params(axis='both',which='minor',length=axminlen)
            ax.tick_params(axis='both',which='both',width=axwidth, labelsize=24)
            if imol > 0: ax.axes.yaxis.set_ticklabels([])
            if irow < nrows-1: ax.axes.xaxis.set_ticklabels([])
            if irow == 0: ax.text(0.5, 1.02, molecules[imol], transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)

            x = gdata['JMAG'] - gdata['KMAG']
            y = gdata['HMAG']

            if statistics[irow] == 'count':
                H, yedges, xedges = np.histogram2d(y, x, bins=nbins)
                im = ax.pcolormesh(xedges, yedges, H, cmap=cmaps[irow], vmin=vmin[imol], vmax=vmax[imol])
                if imol == 2: ax.text(1.235, 0.5, r'$N$ stars',ha='left', va='center', rotation=-90, transform=ax.transAxes)
            if statistics[irow] == 'median fitscale':
                #values = gdata['FITSCALE'][:,imol]
                values = gdata['FITSCALE'+str(imol+1)]
                ret = stats.binned_statistic_2d(x, y, values, statistic='median', bins=(nbins,nbins))
                ext = [ret.x_edge[0], ret.x_edge[-1:][0], ret.y_edge[-1:][0], ret.y_edge[0]]
                im = ax.imshow(ret.statistic, cmap=cmaps[irow], aspect='auto', origin='upper', extent=ext, vmin=vmin[imol], vmax=vmax[imol])
                if imol == 2: ax.text(1.235, 0.5, r'Median Fit Scale',ha='left', va='center', rotation=-90, transform=ax.transAxes)
            if statistics[irow] == 'median scale':
                #values = gdata['SCALE'][:,imol]
                values = gdata['SCALE'+str(imol+1)]
                ret = stats.binned_statistic_2d(x, y, values, statistic='median', bins=(nbins,nbins))
                ext = [ret.x_edge[0], ret.x_edge[-1:][0], ret.y_edge[-1:][0], ret.y_edge[0]]
                im = ax.imshow(ret.statistic, cmap=cmaps[irow], aspect='auto', origin='upper', extent=ext, vmin=vmin[imol], vmax=vmax[imol])
                if imol == 2: ax.text(1.235, 0.5, r'Median Poly Scale',ha='left', va='center', rotation=-90, transform=ax.transAxes)
            if statistics[irow] == 'mad diff':
                #values = gdata['FITSCALE'][:,imol] - gdata['SCALE'][:,imol]
                values = gdata['FITSCALE'+str(imol+1)] - gdata['SCALE'+str(imol+1)]
                ret = stats.binned_statistic_2d(x, y, values, statistic=dln.mad, bins=(nbins,nbins))
                ext = [ret.x_edge[0], ret.x_edge[-1:][0], ret.y_edge[-1:][0], ret.y_edge[0]]
                im = ax.imshow(ret.statistic, cmap=cmap, aspect='auto', origin='upper', extent=ext, vmin=vmin[imol], vmax=vmax[imol])
                if imol == 2: ax.text(1.235, 0.5, r'MAD Fit $-$ Poly',ha='left', va='center', rotation=-90, transform=ax.transAxes)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="3%", pad="2%")
            cax.tick_params(axis='both',which='both',direction='out',bottom=True,top=True,left=True,right=True)
            cax.tick_params(axis='both',which='major',length=axmajlen)
            cax.tick_params(axis='both',which='minor',length=axminlen)
            cax.tick_params(axis='both',which='both',width=axwidth, labelsize=24)
            cax.minorticks_on()
            cb1 = colorbar(im, cax=cax)

    fig.subplots_adjust(left=0.044,right=0.925,bottom=0.047,top=0.967,hspace=0.1,wspace=0.2)
    plt.savefig(plotfile)
    plt.close('all')



###########################################################################################
def tellfitstats6(infile='tellfitstats2_stardata.fits', cmap='rainbow', nbins=40,
                  vmin=[-0.02, -0.02, -0.02], vmax=[0.02, 0.02, 0.02]):
    data = fits.getdata(infile)

    plotfile = sdir5 + 'tellfitstats_indstars_fitscale-scale2.png'
    print('making ' + os.path.basename(plotfile))

    fig = plt.figure(figsize=(32,10))
    for imol in range(nmolecules):
        g, = np.where((data['FITSCALE1'] > 0) & (data['HMAG'] <=11) & (data['JMAG']-data['KMAG'] < 0.58))
        gdata = data[g]

        ax = plt.subplot2grid((1,nmolecules), (0,imol))
        ax.minorticks_on()
        ax.set_ylim(11.2, 6.8)
        ax.set_xlim(-0.2, 0.53)
        ax.set_xlabel(r'$J-K$')
        if imol == 0: ax.set_ylabel(r'$H$')
        ax.tick_params(axis='both',which='both',direction='out',bottom=True,top=True,left=True,right=True)
        ax.tick_params(axis='both',which='major',length=axmajlen)
        ax.tick_params(axis='both',which='minor',length=axminlen)
        ax.tick_params(axis='both',which='both',width=axwidth)
        if imol > 0: ax.axes.yaxis.set_ticklabels([])
        ax.text(0.5, 1.02, molecules[imol], transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)

        x = gdata['JMAG'] - gdata['KMAG']
        y = gdata['HMAG']
        values = gdata['FITSCALE'+str(imol+1)] - gdata['SCALE'+str(imol+1)]
        ret = stats.binned_statistic_2d(x, y, values, statistic='mean', bins=(nbins,nbins))
        ext = [ret.x_edge[0], ret.x_edge[-1:][0], ret.y_edge[-1:][0], ret.y_edge[0]]
        im = ax.imshow(ret.statistic, cmap=cmap, aspect='auto', origin='upper', extent=ext, vmin=vmin[imol], vmax=vmax[imol])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad="2%")
        cax.minorticks_on()
        cax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
        cb1 = colorbar(im, cax=cax)
        if imol == 2:
            ax.text(1.22, 0.5, r'Mean (measured$-$fit)',ha='left', va='center', rotation=-90, transform=ax.transAxes)

    fig.subplots_adjust(left=0.045,right=0.94,bottom=0.093,top=0.94,hspace=0.08,wspace=0.18)
    plt.savefig(plotfile)
    plt.close('all')

###########################################################################################
def tellfitstats7(infile='tellfitstats2_stardata.fits', cmap='rainbow', nbins=40,
                  vmin=[0.9, 1.1, 0.35], vmax=[1.1, 1.17, 1]):
    data = fits.getdata(infile)

    plotfile = sdir5 + 'tellfitstats_indstars_meanFitscale.png'
    print('making ' + os.path.basename(plotfile))

    fig = plt.figure(figsize=(32,10))
    for imol in range(nmolecules):
        g, = np.where((data['FITSCALE1'] > 0) & (data['HMAG'] <=11) & (data['JMAG']-data['KMAG'] < 0.58))
        gdata = data[g]

        ax = plt.subplot2grid((1,nmolecules), (0,imol))
        ax.minorticks_on()
        ax.set_ylim(11.2, 6.8)
        ax.set_xlim(-0.2, 0.53)
        ax.set_xlabel(r'$J-K$')
        if imol == 0: ax.set_ylabel(r'$H$')
        ax.tick_params(axis='both',which='both',direction='out',bottom=True,top=True,left=True,right=True)
        ax.tick_params(axis='both',which='major',length=axmajlen)
        ax.tick_params(axis='both',which='minor',length=axminlen)
        ax.tick_params(axis='both',which='both',width=axwidth)
        if imol > 0: ax.axes.yaxis.set_ticklabels([])
        ax.text(0.5, 1.02, molecules[imol], transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)

        x = gdata['JMAG'] - gdata['KMAG']
        y = gdata['HMAG']
        values = gdata['FITSCALE'+str(imol+1)]# - data['SCALE1']
        ret = stats.binned_statistic_2d(x, y, values, statistic='mean', bins=(nbins,nbins))
        ext = [ret.x_edge[0], ret.x_edge[-1:][0], ret.y_edge[-1:][0], ret.y_edge[0]]
        im = ax.imshow(ret.statistic, cmap=cmap, aspect='auto', origin='upper', extent=ext, vmin=vmin[imol], vmax=vmax[imol])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad="2%")
        cax.minorticks_on()
        cax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
        cb1 = colorbar(im, cax=cax)
        if imol == 2:
            ax.text(1.19, 0.5, r'Mean fitscale',ha='left', va='center', rotation=-90, transform=ax.transAxes)

    fig.subplots_adjust(left=0.045,right=0.945,bottom=0.093,top=0.94,hspace=0.08,wspace=0.18)
    plt.savefig(plotfile)
    plt.close('all')

###########################################################################################
def tellfitstats44(infile='tellfitstats2_stardata.fits', cmap='rainbow', nbins=50):
    data = fits.getdata(infile)

    g, = np.where((data['FITSCALE1'] > 0) & (data['HMAG'] <=11) & (data['JMAG']-data['KMAG'] < 0.58))
    data = data[g]

    plotfile = sdir5 + 'tellfitstats_indstars2.png'
    print('making ' + os.path.basename(plotfile))

    fig = plt.figure(figsize=(19,15))
    ax = plt.subplot2grid((1,1), (0,0))
    ax.minorticks_on()
    #ax.set_ylim(11, 7)
    #ax.set_xlim(-0.1, 0.43)
    ax.set_xlabel(r'$J-K$')
    ax.set_ylabel(r'$H$')
    ax.tick_params(axis='both',which='both',direction='out',bottom=True,top=True,left=True,right=True)
    ax.tick_params(axis='both',which='major',length=axmajlen)
    ax.tick_params(axis='both',which='minor',length=axminlen)
    ax.tick_params(axis='both',which='both',width=axwidth)
    ax.text(0.5, 1.02, molecules[0], transform=ax.transAxes, ha='center', va='bottom', bbox=bboxpar)

    x = data['JMAG'] - data['KMAG']
    y = data['HMAG']
    values = data['FITSCALE1']# - data['SCALE1']
    ret = stats.binned_statistic_2d(x, y, values, statistic=dln.mad, bins=(nbins,nbins))
    ext = [ret.x_edge[0], ret.x_edge[-1:][0], ret.y_edge[-1:][0], ret.y_edge[0]]
    im = ax.imshow(ret.statistic, cmap=cmap, aspect='auto', origin='upper', extent=ext, vmin=0, vmax=0.1)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad="2%")
    cax.minorticks_on()
    cax.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    ax.text(1.12, 0.5, r'MAD (fitscale)',ha='left', va='center', rotation=-90, transform=ax.transAxes)
    cb1 = colorbar(im, cax=cax)

    fig.subplots_adjust(left=0.075,right=0.91,bottom=0.075,top=0.94,hspace=0.08,wspace=0.08)
    plt.savefig(plotfile)
    plt.close('all')

    pdb.set_trace()

    return ret

###########################################################################################
def tellfitstats98(infile='tellfitstats1.fits', plotx='MAD'):
    molecules = np.array(['CH4', 'CO2', 'H2O'])
    nmolecules = len(molecules)

    out = fits.getdata(infile)
    colors = ['r', 'g', 'b']
    plotfile = sdir5 + 'tellfitstats2_' + plotx + '.png'
    print('making ' + os.path.basename(plotfile))
    fig = plt.figure(figsize=(30,18))
    for ichip in range(nchips):
        for imol in range(nmolecules):
            ax = plt.subplot2grid((nchips,nmolecules), (ichip,imol))
            ax.minorticks_on()
            ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
            ax.tick_params(axis='both',which='major',length=axmajlen)
            ax.tick_params(axis='both',which='minor',length=axminlen)
            ax.tick_params(axis='both',which='both',width=axwidth)
            if ichip < 2: ax.axes.xaxis.set_ticklabels([])
            ax.text(0.97, 0.97, molecules[imol], transform=ax.transAxes, ha='right', va='top', bbox=bboxpar)
            if ichip == 2: 
                if plotx == 'MAD': ax.set_xlabel(r'MAD (fitscale)')
                if plotx == 'MADRESID': ax.set_xlabel(r'MAD (fitscale$-$scale)')
            if imol == 0: ax.set_ylabel('N')
            n, bins, patches = ax.hist(out[plotx][:,ichip,imol], 50, density=True, facecolor=colors[ichip], alpha=0.75)

    fig.subplots_adjust(left=0.05,right=0.99,bottom=0.055,top=0.985,hspace=0.1,wspace=0.15)
    plt.savefig(plotfile)
    plt.close('all')

    return out

###########################################################################################
def tellfitstats99(infile='tellfitstats1.fits', plotx='MAD'):
    molecules = np.array(['CH4', 'CO2', 'H2O'])
    nmolecules = len(molecules)

    out = fits.getdata(infile)
    colors = ['r', 'g', 'b']
    plotfile = sdir5 + 'tellfitstats3_' + plotx + '.png'
    print('making ' + os.path.basename(plotfile))
    fig = plt.figure(figsize=(30,18))
    for ichip in range(nchips):
        for imol in range(nmolecules):
            ax = plt.subplot2grid((nchips,nmolecules), (ichip,imol))
            ax.minorticks_on()
            ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
            ax.tick_params(axis='both',which='major',length=axmajlen)
            ax.tick_params(axis='both',which='minor',length=axminlen)
            ax.tick_params(axis='both',which='both',width=axwidth)
            if ichip < 2: ax.axes.xaxis.set_ticklabels([])
            ax.text(0.97, 0.97, molecules[imol], transform=ax.transAxes, ha='right', va='top', bbox=bboxpar)
            if ichip == 2: 
                if plotx == 'MAD': ax.set_xlabel(r'MAD (fitscale)')
                if plotx == 'MADRESID': ax.set_xlabel(r'MAD (fitscale$-$scale)')
            if imol == 0: ax.set_ylabel('N')
            n, bins, patches = ax.hist(out[plotx][:,ichip,imol], 50, density=True, facecolor=colors[ichip], alpha=0.75)

    fig.subplots_adjust(left=0.05,right=0.99,bottom=0.055,top=0.985,hspace=0.1,wspace=0.15)
    plt.savefig(plotfile)
    plt.close('all')

    return out


###########################################################################################
def tellstats(allv4=None):
    if allv4 is None:
        allv4path = '/uufs/chpc.utah.edu/common/home/sdss40/apogeework/apogee/spectro/aspcap/dr17/synspec/allVisit-dr17-synspec.fits'
        allv4 = fits.getdata(allv4path)

    #gd, = np.where((allv4['MJD'] > 58000) & ((bitmask.is_bit_set(allv4['APOGEE_TARGET2'],9)) | (bitmask.is_bit_set(allv4['APOGEE2_TARGET2'],9))))
    gd, = np.where((allv4['TELESCOPE'] == 'apo25m') & (bitmask.is_bit_set(allv4['APOGEE2_TARGET2'],9)))
    allv4g = allv4[gd]
    ufield,uind = np.unique(allv4g['FIELD'], return_index=True)
    nfields = len(ufield)

    for ifield in range(nfields):
        field = ufield[ifield]
        gd, = np.where(allv4g['FIELD'] == field)
        ustars,uind = np.unique(allv4g['APOGEE_ID'][gd], return_index=True)
        jmag = allv4g['J'][gd][uind]
        hmag = allv4g['H'][gd][uind]
        kmag = allv4g['K'][gd][uind]
        jk = jmag-kmag
        nstars = str(len(ustars))
        meanh = str("%.3f" % round(np.nanmean(hmag),3)).rjust(6)
        medh = str("%.3f" % round(np.nanmedian(hmag),3)).rjust(6)
        sigh = str("%.3f" % round(np.nanstd(hmag),3)).rjust(6)
        meanjk = str("%.3f" % round(np.nanmean(jk),3)).rjust(6)
        medjk = str("%.3f" % round(np.nanmedian(jk),3)).rjust(6)
        sigjk = str("%.3f" % round(np.nanstd(jk),3)).rjust(6)
        print(field.ljust(24)+'  '+nstars+'  '+meanh+'  '+medh+'  '+sigh+'  '+meanjk+'  '+medjk+'  '+sigjk)

    return allv4g

###########################################################################################
def telescopePos(field='17049', star='2M07303923+3111106', cmap='gnuplot_r'):
    # telescopePosPerform.png
    plotfile = specdir5 + 'monitor/' + instrument + '/telescopePos/telescopePos_' + field + '_' + star + '.png'
    print("----> commissNplots: Making " + os.path.basename(plotfile))

    num = allexp['NUM']
    #p, = np.where((num == 40630031) | (num == 40630039) | (num == 40630040) | 
    #              (num == 40630048) | (num == 40630049) | (num == 40630057) |
    #              (num == 40630058))
    p, = np.where((num == 40630039) | (num == 40630040) | (num == 40630048) |
                  (num == 40630049) | (num == 40630057) | (num == 40630058))
    altord = np.argsort(allexp['alt'][p])[::-1]
    num = allexp['num'][p][altord]
    alt = allexp['alt'][p][altord]
    upl = allexp['plateid'][p][altord]
    dateobs = allexp['dateobs'][p][altord]
    fra = 113.495888
    fdec = 32.171619
    apo = EarthLocation.of_site('Apache Point Observatory')
    #            LON          LAT         ALT
    #APOcoords = [ 32.780278, -105.820278, 2788]
    #num = np.array([40630031, 40630039, 40630040, 40630048, 40630049, 40630057, 40630058])
    num = np.array([40630039, 40630040, 40630048, 40630049, 40630057, 40630058])
    upl = np.array([3471, 3471, 3477, 3477, 3483, 3483])
    umjd = allexp['mjd'][p][altord]
    allexpg = allexp[p][altord]
    nexp = len(allexpg)

    cmap = cmaps.get_cmap(cmap, 100)
    cmapConst = 0.5
    cmapShift = 0.1

    fig = plt.figure(figsize=(28,14))
    ax1 = plt.subplot2grid((2,8), (0,0), colspan=7)
    ax2 = plt.subplot2grid((2,8), (1,0), colspan=7)
    ax11 = plt.subplot2grid((2,8), (0,7), colspan=1)
    ax22 = plt.subplot2grid((2,8), (1,7), colspan=1)
    #ax3 = plt.subplot2grid((2,8), (1,6), colspan=1)
    axes = [ax1, ax2, ax11, ax22]
    for ax in axes:
        ax.minorticks_on()
        ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
        ax.tick_params(axis='both',which='major',length=axmajlen)
        ax.tick_params(axis='both',which='minor',length=axminlen)
        ax.tick_params(axis='both',which='both',width=axwidth)
    ax1.set_xlim(16475, 16945)
    ax2.set_xlim(16475, 16945)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax11.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax22.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax11.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax22.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax11.set_xlim(16747, 16770)
    ax22.set_xlim(16747, 16770)
    ax1.axes.xaxis.set_ticklabels([])
    ax11.axes.xaxis.set_ticklabels([])
    ax11.axes.yaxis.set_ticklabels([])
    ax22.axes.yaxis.set_ticklabels([])
    ax1.set_ylabel(r'Flux')
    ax2.set_ylabel(r'Norm Flux')
    ax2.set_xlabel(r'Wavelength ($\rm \AA$)')
    ax2.set_ylim(0.25, 1.35)
    ax22.set_ylim(0.25, 1.35)
    visdir = specdir5 + 'visit/apo25m/' + field + '/'

    ax11.text(1.1, 1.00, r'EXPNUM    SECZ   S/N', transform=ax11.transAxes, fontsize=fsz)

    snsecz = open(specdir5 + 'monitor/' + instrument + '/telescopePos/snsecz_'+star+'.dat', 'w')
    snsecz.write('NUM        SECZ      SN\n')
    ymx = np.zeros(nexp)
    for iexp in range(nexp):
        visdir1 = visdir + str(upl[iexp]) + '/' + str(umjd[iexp]) + '/'
        cfile = visdir1 + 'apCframe-a-' + str(num[iexp]) + '.fits'
        plsumfile = visdir1 + 'apPlateSum-' + str(upl[iexp]) + '-' + str(umjd[iexp]) + '.fits'
        flux = fits.getdata(cfile)
        wave = fits.getdata(cfile,4)
        obj = fits.getdata(cfile,11)
        g, = np.where(obj['TMASS_STYLE'] == star)
        if len(g) > 0:
            txt = star + r'  ($H=$' + str("%.3f" % round(obj['hmag'][g][0],3)) + ', field = ' + field + ')'
            if iexp == 0: ax1.text(0.5, 1.02, txt, transform=ax1.transAxes, ha='center')

            sra = obj['ra'][g][0]
            sdec = obj['dec'][g][0]
            obstime = Time(dateobs[iexp], format='fits')
            aa = AltAz(location=apo, obstime=obstime)
            coord = SkyCoord(sra, sdec, unit='deg')
            staralt = coord.transform_to(aa).alt.degree
            secz = 1. / np.cos((90-staralt)*(np.pi/180))

            pl1 = fits.getdata(plsumfile,1)
            pl2 = fits.getdata(plsumfile,2)
            gg1, = np.where(num[iexp] == pl1['IM'])
            gg2, = np.where(star == pl2['TMASS_STYLE'])
            #secz = pl1['SECZ'][gg1][0]
            snr = pl2['sn'][gg2[0], gg1[0], 0]

            c = cmap(((iexp+1)/nexp)+cmapShift)
            w = wave[g][0]; f = flux[g][0]
            p = ax1.plot(w, f, color=c)
            ax11.plot(w, f, color=c)
            #c = p[0].get_color()
            txt = str(num[iexp]) + '   ' + str("%.3f" % round(secz,3)) + '   ' + str(int(round(snr)))
            ax11.text(1.1, 0.97-.04*iexp, txt, color=c, fontsize=fsz, transform=ax11.transAxes, va='top')
            ax2.plot(w, f/np.nanmedian(f), color=c)
            ax22.plot(w, f/np.nanmedian(f), color=c)

            ymxsec, = np.where((w > 16780) & (w < 16820))
            ymx[iexp] = np.nanmax(f[ymxsec])

            snsecz.write(str(num[iexp])+'   '+str("%.5f" % round(secz,5))+'   '+str("%.3f" % round(snr,3)) + '\n')

    ax1.set_ylim(0, np.nanmax(ymx)*1.15)

    snsecz.close()

    fig.subplots_adjust(left=0.073,right=0.875,bottom=0.06,top=0.96,hspace=0.08,wspace=0.1)
    plt.savefig(plotfile)
    plt.close('all')

    return

###########################################################################################
def telescopePos2(field='17049', cmap='gnuplot_r'):
    # telescopePosPerform.png
    plotfile = specdir5 + 'monitor/' + instrument + '/telescopePos/telescopePos_' + field + '_all.png'
    print("----> commissNplots: Making " + os.path.basename(plotfile))

    gstars = np.array(['2M07355107+3113096','2M07320091+3110341','2M07342631+3151001','2M07295449+3146083',
                       '2M07303923+3111106','2M07363035+3239591','2M07293021+3227021','2M07330674+3117112'])

    p, = np.where((allsnr['FIELD'] == field) & ((allsnr['exptime'] == 457) | (allsnr['exptime'] == 489)) & (allsnr['mjd'] != 59609))
    upl,uind = np.unique(allsnr['plate'][p], return_index=True)
    upl = allsnr['plate'][p]#[uind]
    umjd = allsnr['mjd'][p]#[uind]
    allsnrg = allsnr[p]#[uind]
    nexp = len(allsnrg)

    cmap = cmaps.get_cmap(cmap, 100)
    cmapConst = 0.5
    cmapShift = 0.1

    fig = plt.figure(figsize=(28,14))
    ax1 = plt.subplot2grid((2,8), (0,0), colspan=7)
    ax2 = plt.subplot2grid((2,8), (1,0), colspan=7)
    ax11 = plt.subplot2grid((2,8), (0,7), colspan=1)
    ax22 = plt.subplot2grid((2,8), (1,7), colspan=1)
    #ax3 = plt.subplot2grid((2,8), (1,6), colspan=1)
    axes = [ax1, ax2, ax11, ax22]
    for ax in axes:
        ax.minorticks_on()
        ax.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
        ax.tick_params(axis='both',which='major',length=axmajlen)
        ax.tick_params(axis='both',which='minor',length=axminlen)
        ax.tick_params(axis='both',which='both',width=axwidth)
    ax1.set_xlim(16475, 16945)
    ax2.set_xlim(16475, 16945)
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(50))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax2.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax11.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax22.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax11.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax22.xaxis.set_minor_locator(ticker.MultipleLocator(5))
    ax11.set_xlim(16747, 16770)
    ax22.set_xlim(16747, 16770)
    ax1.axes.xaxis.set_ticklabels([])
    ax11.axes.xaxis.set_ticklabels([])
    ax11.axes.yaxis.set_ticklabels([])
    ax22.axes.yaxis.set_ticklabels([])
    ax1.set_ylabel(r'Flux')
    ax2.set_ylabel(r'Norm Flux')
    ax2.set_xlabel(r'Wavelength ($\rm \AA$)')
    ax2.set_ylim(0.25, 1.35)
    ax22.set_ylim(0.25, 1.35)
    visdir = specdir5 + 'visit/apo25m/' + field + '/'

    ax11.text(1.1, 1.00, r'EXPNUM    SECZ   S/N', transform=ax11.transAxes, fontsize=fsz)
    ax1.text(0.5, 1.02, 'field = ' + field, transform=ax1.transAxes, ha='center')

    ymx = np.zeros(nexp)
    secz = np.zeros(nexp)
    snr = np.zeros(nexp)
    for iexp in range(nexp):
        visdir1 = visdir + str(allsnrg['plate'][iexp]) + '/' + str(allsnrg['mjd'][iexp]) + '/'
        cfile = visdir1 + 'apCframe-a-' + str(allsnrg['IM'][iexp]) + '.fits'
        plsumfile = visdir1 + 'apPlateSum-' + str(upl[iexp]) + '-' + str(umjd[iexp]) + '.fits'
        flux = fits.getdata(cfile)
        wave = fits.getdata(cfile,4)
        print(os.path.basename(cfile))
        pl1 = fits.getdata(plsumfile,1)
        gg1, = np.where(allsnrg['IM'][iexp] == pl1['IM'])
        secz[iexp] = pl1['SECZ'][gg1][0]

    #print(ymx/np.nanmax(ymx))
    #ax1.set_ylim(0, np.nanmax(ymx)*1.15)
    #gd, = np.where((snr > 0) & (ymx > 0) & (ymx/np.nanmax(ymx) > 0.2))
    sord = np.argsort(secz)
    secz = secz[sord]
    upl = upl[sord]
    umjd = umjd[sord]
    allsnrg = allsnrg[sord]

    for iexp in range(nexp):
        visdir1 = visdir + str(allsnrg['plate'][iexp]) + '/' + str(allsnrg['mjd'][iexp]) + '/'
        cfile = visdir1 + 'apCframe-a-' + str(allsnrg['IM'][iexp]) + '.fits'
        plsumfile = visdir1 + 'apPlateSum-' + str(upl[iexp]) + '-' + str(umjd[iexp]) + '.fits'
        pl1 = fits.getdata(plsumfile,1)
        pl2 = fits.getdata(plsumfile,2)
        pdb.set_trace()
        flux = fits.getdata(cfile)
        wave = fits.getdata(cfile,4)
        obj = fits.getdata(cfile,11)
        gd, = np.where(obj['fiberid'] > 0)
        obj = obj[gd]
        #starind = np.where(
        gg1, = np.where(allsnrg['IM'][iexp] == pl1['IM'])
        #gdstars, = np.where((obj['objtype'] == 'STAR') & (obj['hmag'] < 9) & (pl2['sn'][:, gg1[0], 0] > 100))
        gdstars, = np.where((pl2['objtype'] == 'STAR') & (pl2['hmag'] < 8) & (pl2['sn'][:, gg1[0], 0] > 100))
        #pdb.set_trace()
        print(len(gdstars))
        w = np.nanmean(wave[gdstars], axis=0)
        f = np.nanmean(flux[gdstars], axis=0)
        snr = np.nanmean(pl2['sn'][gdstars, gg1[0], 0])
        c = cmap(((iexp+1)/nexp)+cmapShift)
        p = ax1.plot(w, f, color=c)
        ax11.plot(w, f, color=c)
        txt = str(allsnrg['IM'][iexp]) + '   ' + str("%.3f" % round(secz[iexp],3)) + '   ' + str(int(round(snr)))
        ax11.text(1.1, 0.97-.04*iexp, txt, color=c, fontsize=fsz, transform=ax11.transAxes, va='top')
        ax2.plot(w, f/np.nanmedian(f), color=c)
        ax22.plot(w, f/np.nanmedian(f), color=c)



    fig.subplots_adjust(left=0.073,right=0.875,bottom=0.06,top=0.96,hspace=0.08,wspace=0.1)
    plt.savefig(plotfile)
    plt.close('all')

    return

###########################################################################################
def telescopePos3(field='17049', cmap='nipy_spectral', cut=True):
    # telescopePosPerform.png
    plotfile = specdir5 + 'monitor/' + instrument + '/telescopePos/telescopePos_' + field + '_seczXsnr.png'
    if cut: plotfile = plotfile.replace('.png', '_cut.png')
    print("----> commissNplots: Making " + os.path.basename(plotfile))

    cmap = cmaps.get_cmap(cmap, 100)
    cmapConst = 0.7
    cmapShift = 0.05

    fig = plt.figure(figsize=(21,14))
    ax1 = plt.subplot2grid((1,1), (0,0))
    ax1.minorticks_on()
    ax1.tick_params(axis='both',which='both',direction='in',bottom=True,top=True,left=True,right=True)
    ax1.tick_params(axis='both',which='major',length=axmajlen)
    ax1.tick_params(axis='both',which='minor',length=axminlen)
    ax1.tick_params(axis='both',which='both',width=axwidth)
    ax1.set_xlabel(r'sec(z)')
    ax1.set_ylabel(r'Normalized S/N (red chip)')
    visdir = specdir5 + 'visit/apo25m/' + field + '/'

    infiles = glob.glob(specdir5 + 'monitor/' + instrument + '/telescopePos/*dat')
    infiles.sort()
    infiles = np.array(infiles)
    nfiles = len(infiles)

    for i in range(nfiles):
        d = ascii.read(infiles[i])
        star = os.path.basename(infiles[i]).split('_')[1].split('.')[0]
        x = d['SECZ']
        y = d['SN'] / np.max(d['SN'])
        if cut:
            if (y[0] < 0.9) | (y[3] < 0.8): continue
        c = cmap((i+1)/nfiles)
        ax1.plot(x, y, marker='o', color=c)
        ax1.scatter(x, d['SN']/np.max(d['SN']), marker='o', color=c, edgecolors='k', s=80, label=star)

    ax1.legend(loc=[1.01, 0.0], labelspacing=0.5, handletextpad=-0.1, fontsize=fsz, edgecolor='k', framealpha=1)

    fig.subplots_adjust(left=0.06,right=0.805,bottom=0.063,top=0.985,hspace=0.08,wspace=0.1)
    plt.savefig(plotfile)
    plt.close('all')

    return

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
def skysub0(dosky=True, field='20833', plate='3801', mjd='59638'):
    pixrad = 9
    skylinesa = np.array([ 210.5,  497.0, 1139.6, 1908.0])
    skylinesb = np.array([1100.0, 1270.8, 1439.8, 1638.4])
    skylinesc = np.array([ 656.2, 1467.0, 1599.7, 1738.0])
    skylines = np.array([skylinesa, skylinesb, skylinesc])
    nskylines = len(skylinesa)

    specdir = specdir5 + 'visit/' + telescope + '/' + field + '/' + plate + '/' + mjd + '/'
    cframes = glob.glob(specdir + '*Cframe-a*')
    cframes.sort()
    cframes = np.array(cframes)
    ncframes = len(cframes)

    apPlate = load.apPlate(int(plate), mjd)
    data = apPlate['a'][11].data[::-1]
    sky, = np.where((data['objtype'] == 'SKY') & (data['fiberid'] != 75) & (data['fiberid'] != 225) & 
                    (data['fiberid'] != 25) & (data['fiberid'] != 18) & (data['fiberid'] != 21) & 
                    (data['fiberid'] != 109) & (data['fiberid'] != 289))


    notsky, = np.where((data['objtype'] == 'none') & (data['fiberid'] != 75) & (data['fiberid'] != 225) & 
                       (data['fiberid'] != 25) & (data['fiberid'] != 18) & (data['fiberid'] != 21) & 
                       (data['fiberid'] != 109) & (data['fiberid'] != 289))

    goodind = notsky
    if dosky: goodind = sky
    diff = np.zeros((ncframes, nchips, nskylines))
    for iframe in range(ncframes):
        num = int(os.path.basename(cframes[iframe]).split('-')[2].split('.')[0])
        for ichip in range(nchips):
            gfile = cframes[iframe]
            if ichip == 1: gfile = gfile.replace('-a-', '-b-')
            if ichip == 2: gfile = gfile.replace('-a-', '-c-')
            cflux = fits.getdata(gfile)
            msky = np.nanmedian(cflux[goodind], axis=0)
            oneDflux = load.apread('1D', num=num)[ichip].flux
            msky0 = np.nanmedian(oneDflux[:,299-goodind], axis=1)
            for iline in range(nskylines):
                lstart = int(round(skylines[ichip, iline] - pixrad))
                lstop  = int(round(skylines[ichip, iline] + pixrad))
                diff[iframe, ichip, iline] = (np.nansum(msky[lstart:lstop]) / np.nansum(msky0[lstart:lstop])) * 100.0

    print(diff)

    return

###########################################################################################
def skysub(dosky=True, xmin=59597, ajd=None, resid=None, cont=False):
    # skysub.png

    skysyms = np.array(['o', '^', 'v', 'P'])

    pixrad = 9
    skylinesa = np.array([ 210.5,  497.0, 1139.6, 1908.0])
    skylinesb = np.array([1100.0, 1270.8, 1439.8, 1638.4])
    skylinesc = np.array([ 656.2, 1467.0, 1599.7, 1738.0])

    plotfile = specdir5 + 'monitor/' + instrument + '/skysub.png'

    if cont:
        pixrad = 30
        skylinesa = np.array([764., 1340.])
        skylinesb = np.array([915., 1545.])
        skylinesc = np.array([423., 1825.])
        plotfile = plotfile.replace('.png', '_cont.png')

    skylines = np.array([skylinesa, skylinesb, skylinesc])
    nskylines = len(skylinesa)

    if dosky: plotfile = plotfile.replace('.png', '_sky.png')
    print("----> monitor: Making " + os.path.basename(plotfile))

    fig = plt.figure(figsize=(26,14))

    g, = np.where(allsnr['MJD'] >= xmin)
    allsnrg = allsnr[g]
    snrord = np.argsort(allsnrg['JD'])
    allsnrg = allsnrg[snrord]
    jd = allsnrg['JD']
    nexp = len(g)

    mxjd = np.nanmax(jd)
    xspan = mxjd - xmin
    xmax = mxjd + xspan*0.15

    ax1 = plt.subplot2grid((nchips,1), (2,0))
    ax2 = plt.subplot2grid((nchips,1), (1,0))
    ax3 = plt.subplot2grid((nchips,1), (0,0))
    axes = [ax1,ax2,ax3]
    for ax in axes:
        ax.minorticks_on()
        ax.tick_params(axis='both',which='both',direction='out',bottom=True,top=True,left=True,right=True)
        ax.tick_params(axis='both',which='major',length=axmajlen)
        ax.tick_params(axis='both',which='minor',length=axminlen)
        ax.tick_params(axis='both',which='both',width=axwidth)
        ax.set_xlim(xmin, xmax)
        if cont is False: ax.set_ylim(0.0, 2.6)
        ax.axhline(y=1, zorder=1, color='grey', linewidth=3)#, linestyle='dashed')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))
        if cont is False: ax.yaxis.set_major_locator(ticker.MultipleLocator(0.5))
        if cont is False: ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    ax1.set_xlabel(r'JD - 2,400,000')
    if cont is False: ax2.set_ylabel(r'Airglow Line Residual (%)')
    if cont: ax2.set_ylabel(r'Airglow Continuum Residual (%)')
    ax2.axes.xaxis.set_ticklabels([])
    ax3.axes.xaxis.set_ticklabels([])

    if resid is None:
        resid = np.zeros((nexp, nchips, nskylines))
        for iexp in range(nexp):
            stel = allsnrg['telescope'][iexp]
            sfield = allsnrg['field'][iexp]
            splate = str(allsnrg['plate'][iexp])
            smjd = str(allsnrg['mjd'][iexp])
            snum = str(allsnrg['im'][iexp])
            specdir = specdir5 + 'visit/' + stel + '/' + sfield + '/' + splate + '/' + smjd + '/'

            cframe = glob.glob(specdir + 'apCframe-a-' + snum + '.fits')
            if os.path.exists(cframe[0]) == False: continue

            apPlate = load.apPlate(int(splate), smjd)
            objdata = apPlate['a'][11].data[::-1]

            asstot = np.sum(objdata['assigned'])
            if asstot < 20: continue

            if dosky is False:
                gdind, = np.where((objdata['objtype'] == 'none') & (objdata['fiberid'] != 75) & (objdata['fiberid'] != 225) & 
                                  (objdata['fiberid'] != 25) & (objdata['fiberid'] != 18) & (objdata['fiberid'] != 21) & 
                                  (objdata['fiberid'] != 109) & (objdata['fiberid'] != 289))
            else:
                gdind, = np.where((objdata['objtype'] == 'SKY') & (objdata['fiberid'] != 75) & (objdata['fiberid'] != 225) & 
                                  (objdata['fiberid'] != 25) & (objdata['fiberid'] != 18) & (objdata['fiberid'] != 21) & 
                                  (objdata['fiberid'] != 109) & (objdata['fiberid'] != 289))

            if len(gdind) < 20: continue

            try:
                print('(' + str(iexp) + '/' + str(nexp) + '):  field=' + sfield + ', plate=' + splate + ', mjd=' + smjd + ', num=' + snum)
                for ichip in range(nchips):
                    chip = chips[ichip]
                    gfile = cframe[0]
                    if ichip == 1: gfile = gfile.replace('-a-', '-b-')
                    if ichip == 2: gfile = gfile.replace('-a-', '-c-')
                    cflux = fits.getdata(gfile)
                    oneDflux = load.apread('1D', num=int(snum))[ichip].flux
                    contcheck = np.nanmedian(oneDflux[:,299-gdind], axis=0)
                    medall = np.nanmedian(contcheck)
                    stdall = np.nanstd(contcheck)
                    gdsky, = np.where((contcheck < medall+stdall) & (contcheck > 0))
                    gdind = gdind[gdsky]
                    msky = np.nanmedian(cflux[299-gdind], axis=0)
                    msky0 = np.nanmedian(oneDflux[:,299-gdind], axis=1)
                    for iline in range(nskylines):
                        lstart = int(round(skylines[ichip, iline] - pixrad))
                        lstop  = int(round(skylines[ichip, iline] + pixrad))
                        resid[iexp, ichip, iline] = (np.nansum(np.absolute(msky[lstart:lstop])) / np.nansum(np.absolute(msky0[lstart:lstop]))) * 100.0
                        if cont: (np.nansum(msky[lstart:lstop]) / np.nansum(msky0[lstart:lstop])) * 100.0
            except:
                print('problem')

    ichip = 0
    for ax in axes:
        chip = chips[2-ichip]
        for iline in range(nskylines):
            gd, = np.where(resid[:, ichip, iline] < 50)
            med = np.nanmedian(resid[gd, ichip, iline])
            ax.axhline(y=med, color=colors[iline], linestyle='dashed')
            c = colors[iline]
            x = [jd[gd], jd[gd]]
            y = [resid[gd, ichip, iline], resid[gd, ichip, iline]]
            lab = str(int(round(skylines[ichip, iline]))).rjust(4) + '  (' + str("%.2f" % round(med, 2)) + '%)'
            ax.scatter(x, y, marker=skysyms[iline], s=25, c=c, alpha=0.7, zorder=50, label=lab)

        ichip += 1

        ax.text(0.03,0.94,chip.capitalize() + '\n' + 'Chip', transform=ax.transAxes, ha='center', va='top', color=chip, bbox=bboxpar)
        ax.text(0.93, 0.85, 'airglow' + '\n' + 'pixel', transform=ax.transAxes, ha='center', va='bottom', color='k', bbox=bboxpar, fontsize=fsz*0.9)
        ax.text(0.97, 0.85, 'median' + '\n' + 'resid', transform=ax.transAxes, ha='center', va='bottom', color='k', bbox=bboxpar, fontsize=fsz*0.9)
        if cont:
            ax.legend(loc=[0.9,0.63], labelspacing=0.5, handletextpad=-0.1, markerscale=2, fontsize=fsz*0.9, edgecolor='k', framealpha=1)
        else:
            ax.legend(loc=[0.9,0.45], labelspacing=0.5, handletextpad=-0.1, markerscale=2, fontsize=fsz*0.9, edgecolor='k', framealpha=1)

    fig.subplots_adjust(left=0.05,right=0.985,bottom=0.065,top=0.98,hspace=0.2,wspace=0.00)
    plt.savefig(plotfile)
    plt.close('all')

    return jd, resid

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

    fsize = 33
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
    #ax4.grid(True)

    mad = dln.mad(percentDif)
    med = np.nanmedian(percentDif)
    vmax = np.nanmax(percentDif)
    vmin = np.nanmin(percentDif)
    ax4.text(0.47, 0.95, 'median = ' + str("%.3f" % round(med, 3)) + '%', transform=ax4.transAxes, ha='right', va='top', bbox=bboxpar, c='r')
    ax4.text(0.47, 0.85, 'max = ' + str("%.3f" % round(vmax, 3)) + '%', transform=ax4.transAxes, ha='right', va='top', bbox=bboxpar, c='k')
    ax4.text(0.47, 0.75, 'min = ' + str("%.3f" % round(vmin, 3)) + '%', transform=ax4.transAxes, ha='right', va='top', bbox=bboxpar, c='k')
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






