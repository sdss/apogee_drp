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
bboxpar = dict(facecolor='white', edgecolor='none', alpha=1.0)
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
                       (allv5['SNR'] > 10))
        allv5fps = allv5[gd]

        gd, = np.where((np.isnan(allv4['VHELIO']) == False) &
                       (np.absolute(allv4['VHELIO']) < 400) &
                       (np.isnan(allv4['SNR']) == False) & 
                       (allv4['SNR'] > 10))
        allv4g = allv4[gd]

        uplateIDs = np.unique(allv4g['APOGEE_ID'])
        ufpsIDs = np.unique(allv5fps['APOGEE_ID'])

        gdIDs, plate_ind, fps_ind = np.intersect1d(uplateIDs, ufpsIDs, return_indices=True)
        ngd = len(gdIDs)
        print(ngd)

        dt = np.dtype([('APOGEE_ID', np.str, 18)
                       ('JMAG',      np.float64),
                       ('HMAG',      np.float64),
                       ('KMAG',      np.float64),
                       ('NVIS',      np.int32, 2),
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
    ax1 = plt.subplot2grid((2,2), (0,0))
    ax2 = plt.subplot2grid((2,2), (0,1))
    ax3 = plt.subplot2grid((2,2), (1,0))
    ax4 = plt.subplot2grid((2,2), (1,1))
    axes = [ax1,ax2,ax3,ax4]
    if restrict:
        ax1.set_xlim(-150, 150)
        ax1.set_ylim(-5, 5)
        ax2.set_xlim(3300, 6800)
        ax2.set_ylim(-600, 600)
        ax3.set_xlim(-0.1, 5.1)
        ax3.set_ylim(-1.4, 1.4)
        ax4.set_xlim(-1.5, 0.4)
        ax4.set_ylim(-0.5, 0.5)
    ax1.text(0.05, 0.95, r'$V_{\rm helio}$ (km$\,s^{-1}$)', transform=ax1.transAxes, va='top')
    ax2.text(0.05, 0.95, r'RV $T_{\rm eff}$ (K)', transform=ax2.transAxes, va='top')
    ax3.text(0.05, 0.95, r'RV log$\,g$ (dex)', transform=ax3.transAxes, va='top')
    ax4.text(0.05, 0.95, r'RV [Fe/H] (dex)', transform=ax4.transAxes, va='top')
    #ax1.set_xlabel(r'DR17 ')
    ax1.set_ylabel(r'DR17 $-$ FPS')
    ax3.set_xlabel(r'DR17')
    ax3.set_ylabel(r'DR17 $-$ FPS')
    ax4.set_xlabel(r'DR17')
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
    cmap = 'rainbow'
    vmin = 0.2
    vmax = 1.0

    g, = np.where((np.isnan(gdata['TEFF'][:,0]) == False) & (np.isnan(gdata['TEFF'][:,1]) == False) & (gdata['TEFF'][:,0] < 7000))
    x = gdata['VHELIO'][:,0][g]
    y = gdata['VHELIO'][:,0][g] - gdata['VHELIO'][:,1][g]
    ax1.text(0.05, 0.88, 'med: ' + str("%.3f" % round(np.median(np.absolute(y)), 3)), transform=ax1.transAxes, va='top', fontsize=fsz, bbox=bboxpar)
    ax1.text(0.05, 0.82, 'dlnmad: ' + str("%.3f" % round(dln.mad(np.absolute(y)), 3)), transform=ax1.transAxes, va='top', fontsize=fsz, bbox=bboxpar)
    ax1.scatter(x, y, marker=symbol, c=gdata['JMAG'][g]-gdata['KMAG'][g], cmap=cmap, s=symsz, edgecolors='k', alpha=0.75, zorder=10, vmin=vmin, vmax=vmax)

    g, = np.where((np.isnan(gdata['TEFF'][:,0]) == False) & (np.isnan(gdata['TEFF'][:,1]) == False) & (gdata['TEFF'][:,0] < 7000))
    x = gdata['TEFF'][:,0][g]# / 1000
    y = (gdata['TEFF'][:,0][g] - gdata['TEFF'][:,1][g])# / 1000
    gg, = np.where(np.absolute(y) < 2000)
    x = x[gg]
    y = y[gg]
    ax2.text(0.05, 0.88, 'med: ' + str("%.3f" % round(np.median(np.absolute(y)), 3)), transform=ax2.transAxes, va='top', fontsize=fsz, bbox=bboxpar, zorder=20)
    ax2.text(0.05, 0.82, 'dlnmad: ' + str("%.3f" % round(dln.mad(np.absolute(y)), 3)), transform=ax2.transAxes, va='top', fontsize=fsz, bbox=bboxpar, zorder=20)
    sc2 = ax2.scatter(x, y, marker=symbol, c=gdata['JMAG'][g][gg]-gdata['KMAG'][g][gg], cmap=cmap, s=symsz, edgecolors='k', alpha=0.75, zorder=10, vmin=vmin, vmax=vmax)

    g, = np.where((np.isnan(gdata['LOGG'][:,0]) == False) & (np.isnan(gdata['LOGG'][:,1]) == False) & (gdata['TEFF'][:,0] < 7000))
    x = gdata['LOGG'][:,0][g]
    y = gdata['LOGG'][:,0][g] - gdata['LOGG'][:,1][g]
    ax3.text(0.05, 0.88, 'med: ' + str("%.3f" % round(np.median(np.absolute(y)), 3)), transform=ax3.transAxes, va='top', fontsize=fsz, bbox=bboxpar, zorder=20)
    ax3.text(0.05, 0.82, 'dlnmad: ' + str("%.3f" % round(dln.mad(np.absolute(y)), 3)), transform=ax3.transAxes, va='top', fontsize=fsz, bbox=bboxpar, zorder=20)
    ax3.scatter(x, y, marker=symbol, c=gdata['JMAG'][g]-gdata['KMAG'][g], cmap=cmap, s=symsz, edgecolors='k', alpha=0.75, zorder=10, vmin=vmin, vmax=vmax)

    g, = np.where((np.isnan(gdata['FEH'][:,0]) == False) & (np.isnan(gdata['FEH'][:,1]) == False) & (gdata['TEFF'][:,0] < 7000))
    x = gdata['FEH'][:,0][g]
    y = gdata['FEH'][:,0][g] - gdata['FEH'][:,1][g]
    ax4.text(0.05, 0.88, 'med: ' + str("%.3f" % round(np.median(np.absolute(y)), 3)), transform=ax4.transAxes, va='top', fontsize=fsz, bbox=bboxpar, zorder=20)
    ax4.text(0.05, 0.82, 'dlnmad: ' + str("%.3f" % round(dln.mad(np.absolute(y)), 3)), transform=ax4.transAxes, va='top', fontsize=fsz, bbox=bboxpar, zorder=20)
    sc4 = ax4.scatter(x, y, marker=symbol, c=gdata['JMAG'][g]-gdata['KMAG'][g], cmap=cmap, s=symsz, edgecolors='k', alpha=0.75, zorder=10, vmin=vmin, vmax=vmax)

    ax2_divider = make_axes_locatable(ax2)
    cax2 = ax2_divider.append_axes("right", size="5%", pad="1%")
    cb2 = colorbar(sc2, cax=cax2, orientation="vertical")
    cax2.minorticks_on()
    #cax2.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax2.text(1.16, 0.5, r'J$-$K',ha='left', va='center', rotation=-90, transform=ax2.transAxes)

    ax4_divider = make_axes_locatable(ax4)
    cax4 = ax4_divider.append_axes("right", size="5%", pad="1%")
    cb4 = colorbar(sc4, cax=cax4, orientation="vertical")
    cax4.minorticks_on()
    #cax4.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax4.text(1.16, 0.5, r'J$-$K',ha='left', va='center', rotation=-90, transform=ax4.transAxes)

    fig.subplots_adjust(left=0.07, right=0.935, bottom=0.05, top=0.98, hspace=0.1, wspace=0.2)
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

        ###########################################################################################
        # snhistory3.png
        plotfile = specdir5 + 'monitor/' + instrument + '/snhistory3.png'
        print("----> monitor: Making " + os.path.basename(plotfile))

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


