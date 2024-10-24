#!/usr/bin/env python

"""APQUICK_UTAH.PY - APOGEE Quick reduction of Utah data.

This performs a very quick reduction of an APOGEE exposure to
give feedback about the S/N.  This program can be run on an exposure
in progress or on that has already finished because it uses the apRaw
files (individual reads). 

Here are the main steps:
1) Use Fowler/CDS to collapse cube to a 2D image using very few reads.
    This is only done for the green chip.
2) Construct the noise model assuming we used up-the-ramp as is done
     by the full pipeline.
3) Boxcar extract all fibers but only for ~50 columns at the center
     of the image.
4) Fit a line to log(S/N) vs. Hmag and use it to find the S/N at the
     fiducial H magnitude.
5) Write out the results to a single multi-extension FITS file.

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
#__version__ = '20180922'  # yyyymmdd                                                                                                                           

import os
import pdb
import numpy as np
import warnings
import subprocess
from astropy.io import fits,ascii
from astropy.table import Table, Column, vstack
from astropy.time import Time
from glob import glob
from scipy.ndimage import median_filter,generic_filter
from apogee_drp.utils import apzip,plan,apload,yanny,plugmap,platedata,bitmask,info,slurm as slrm
import slurm
from slurm import queue as pbsqueue
import pandas as pd
#from apogee_drp.utils import yanny, apload
#from sdss_access.path import path
#from . import config  # get loaded config values 
#from . import __version__
#from . import bitmask as bmask
#from . import yanny

# In the future, use yanny tools from pyDL
#from pydl.pydlutils.yanny import (is_yanny, read_table_yanny,
#                                  write_table_yanny)
#from pydl.pydlutils import yanny


# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings('ignore', category=UserWarning, append=True)

#observatory = os.getenv("OBSERVATORY")
#if observatory == "APO":
#    prefix = 'ap'
#else:
    #prefix = 'as'
    # Fix plugmap_dir for lco
    # plugmap_dir: /home/sdss5/software/sdsscore/main/apo/summary_files/
    #config['apogee_mountain']['plugmap_dir'] = config['apogee_mountain']['plugmap_dir'].replace('apo','lco')

"""
RUNUTAH is a parallelized wrapper for running apquick on data stored at Utah.
It calls the procedure UTAH.
The GETPSFLIST procedure is no longer need.
MAKESUMFILES concatenates the individual apQ files into master apQ files
"""

expdir4 = '/uufs/chpc.utah.edu/common/home/sdss/apogeework/apogee/spectro/redux/current/exposures/'
detPlateN = '/uufs/chpc.utah.edu/common/home/sdss/apogeework/apogee/spectro/redux/current/cal/detector/apDetector-b-13390003.fits'
detPlateS = '/uufs/chpc.utah.edu/common/home/sdss/apogeework/apogee/spectro/redux/current/cal/detector/asDetector-b-22810006.fits'
bpmPlateN = '/uufs/chpc.utah.edu/common/home/sdss/apogeework/apogee/spectro/redux/current/cal/bpm/apBPM-b-33770001.fits'
bpmPlateS = '/uufs/chpc.utah.edu/common/home/sdss/apogeework/apogee/spectro/redux/current/cal/bpm/asBPM-b-35800002.fits'

"""
This part isn't used anymore. We just one PSF file for quickred extraction, regardless of MJD.
"""
def getPsfList(load=None, update=False):
    # Find the psf list directory
    codedir = os.path.dirname(os.path.abspath(__file__))
    datadir = os.path.dirname(os.path.dirname(os.path.dirname(codedir))) + '/data/psflists/'
    pfile = datadir+load.observatory+'PSFall.dat'

    if update:
        data = ascii.read(datadir+load.observatory+'PSFplate.dat')
        numPlate = np.array(data['col1'])
        expPlate = np.char.zfill(np.array(data['col2']).astype(str),8)
        redux_dir = os.environ.get('APOGEE_REDUX')+'/'+load.apred+'/'
        pdir = redux_dir+'cal/'+load.instrument+'/psf/'
        pfiles = glob(pdir+load.prefix+'PSF-b-*fits')
        pfiles.sort()
        pfiles = np.array(pfiles)
        npfiles = len(pfiles)
        expFPS = np.empty(npfiles).astype(str)
        for i in range(npfiles): expFPS[i]=os.path.basename(pfiles[i]).split('-')[2].split('.')[0]
        expAll = np.concatenate([expPlate,expFPS])
        numAll = np.char.zfill(np.arange(len(expAll)).astype(str),8)
        ascii.write([numAll,expAll], pfile, format='no_header', overwrite=True)

    data = ascii.read(pfile)
    return np.array(data['col2'])

def runutah(telescope='lco25m', apred='daily',nodes=2, updatePSF=False, startnum=78520):
    # Slurm settings
    alloc = 'sdss-np'
    shared = True
    ppn = 64
    walltime = '336:00:00'
    # Only set cpus if you want to use less than 64 cpus
    slurmpars = {'nodes':nodes, 'alloc':alloc, 'shared':shared, #'ppn':ppn,
                 'walltime':walltime, 'notification':False}

    # Set up directory paths
    load = apload.ApLoad(apred=apred, telescope=telescope)
    apodir = os.environ.get('APOGEE_REDUX')+'/'+apred+'/'
    
    # Raw data will be extracted temporarily to current working directory (then removed)
    cwd = os.getcwd()+'/'

    # Get science exposure numbers from summary file
    edata0 = fits.getdata(apodir+'monitor/'+load.instrument+'Sci.fits')
    nexp = len(edata0)
    print(str(nexp)+' exposures to run\n')

    # Get PSF exposure numbers from getPsfList subroutine
    #psfnums = getPsfList(load=load, update=updatePSF)

    tasks = np.zeros(nexp,dtype=np.dtype([('cmd',str,1000),('outfile',str,1000),('errfile',str,1000),('dir',str,1000)]))
    tasks = Table(tasks)
    # Loop over exposures
    for i in range(nexp):
        exp = str(edata0['IM'][i])
        rawfilepath = load.filename('R', num=int(exp), chips='b').replace('R-','R-b-')
        mjd = os.path.basename(os.path.dirname(rawfilepath))
        outdir = apodir+'quickred/'+telescope+'/'+mjd+'/'
        if os.path.exists(outdir) == False: os.makedirs(outdir)
        outfile = outdir+'apQ-'+str(exp).zfill(8)+'.log'
        errfile = outfile.replace('.log','.err')
        cmd = 'python -c "from apogee_drp.apred import apquick_utah; apquick_utah.utah('+exp+',telescope='
        cmd += "'"+telescope+"',apred='"+apred+"')"
        cmd += '"'
        #utah(edata0['NUM'][iexp],telscope=telescope,apred=apred)
        tasks['cmd'][i] = cmd
        tasks['outfile'][i] = outfile
        tasks['errfile'][i] = errfile
        tasks['dir'][i] = os.environ.get('APOGEE_REDUX')+'/'
        #tasks['dir'][i] = errfile

    key,jobid = slrm.submit(tasks,label='apq',verbose=True,logger=None,**slurmpars)
    slrm.queue_wait('apq',key,jobid,sleeptime=60,verbose=True,logger=None) # wait for jobs to complete  

def utah(framenum,telescope='lco25m',apred='daily'):
    # Raw data will be extracted temporarily to current working directory (then removed)
    cwd = os.getcwd()+'/'
    load = apload.ApLoad(apred=apred, telescope=telescope)
    apodir = os.environ.get('APOGEE_REDUX')+'/'+apred+'/'
    rawfilepath = load.filename('R', num=framenum, chips='b').replace('R-','R-b-')
    mjd = os.path.basename(os.path.dirname(rawfilepath))
    outdir = apodir+'quickred/'+telescope+'/'+mjd+'/'
    outfile = outdir+'apQ-'+str(framenum).zfill(8)+'.fits'
    if os.path.exists(rawfilepath) == False:
        print(rawfilepath+' not found!')
        return
    rawfile = os.path.basename(rawfilepath)
    rawfilefits = rawfile.replace('.apz','.fits')
    mjd = os.path.basename(os.path.dirname(rawfilepath))
    #infile = os.environ.get('APOGEE_REDUX')+'/'+apred+'/'+rawfilefits
    infile = '/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u0955897/projects/'+rawfilefits
    # Unzip the file
    if os.path.exists(infile) == False: apzip.unzip(rawfilepath, fitsdir='/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/users/u0955897/projects/')
    hdulist = fits.open(infile)
    nreads = len(hdulist)-1
    try:
        frame, subspec, cat, coefstr = runquick(infile, hdulist=hdulist, framenum=framenum, mjd=mjd, load=load)
        print('writing '+outfile)
        writeresults(outfile, frame, subspec, cat, coefstr, compress=False)
    except: 
        print(str(framenum)+' failed.')
        pass

    print('made '+outfile)

    os.remove(infile)

def makesumfile(telescope='lco25m',apred='daily'):
    load = apload.ApLoad(apred=apred, telescope=telescope)
    apodir = os.environ.get('APOGEE_REDUX')+'/'+apred+'/'
    qdir = apodir+'quickred/'+telescope+'/'
    print('Finding apQ files...')
    files = glob(qdir+'*/*fits')
    files.sort()
    files = np.array(files)
    nfiles = len(files)
    nfilesS = str(nfiles)
    print('Found '+str(nfiles)+' files')

    outfile = qdir+'apQ-'+telescope+'.fits'
    outstr = Table(fits.getdata(files[0]))

    for i in range(1,nfiles):
        print('('+str(i+1).zfill(5)+'/'+nfilesS+'): '+os.path.basename(files[i]))
        d1 = Table(fits.getdata(files[i]))
        outstr = vstack([outstr,d1])
        del d1
    outstr.write(outfile, overwrite=True)

def makesumfile2(telescope='lco25m',apred='daily', ndo=None):

    load = apload.ApLoad(apred=apred, telescope=telescope)
    apodir = os.environ.get('APOGEE_REDUX')+'/'+apred+'/'
    qdir = apodir+'quickred/'+telescope+'/'
    outfile = qdir+'apQ-'+telescope+'_new.fits'

    print('Finding apQ files...')
    files = glob(qdir+'*/*fits')
    files.sort()
    files = np.array(files)
    nfiles = len(files)
    if ndo != None: nfiles = ndo
    nfilesS = str(nfiles)
    print('Found '+str(nfiles)+' files')

    # Find the psf list directory
    codedir = os.path.dirname(os.path.abspath(__file__))

    # Get Magellan telescope seeing data
    print('Reading Magellan telescope seeing data')
    magfile = os.path.dirname(os.path.dirname(os.path.dirname(codedir))) + '/data/seeing/magellan_2014.csv'
    magdata = pd.read_csv(magfile)
    magT = Time(np.array(magdata['tm']).astype(str), format='fits')
    magMJD = magT.mjd
    magSeeing = np.array(magdata['fw'])
    g, = np.where(np.array(magdata['un']) == 0)
    magMJD1 = magMJD[g]
    magSeeing1 = magSeeing[g]
    magAz1 = np.array(magdata['az'])[g]
    magAlt1 = np.array(magdata['el'])[g]
    magSecz1 = 1/(np.cos((90-magAlt1)*(np.pi/180)))
    g, = np.where(np.array(magdata['un']) == 1)
    magMJD2 = magMJD[g]
    magSeeing2 = magSeeing[g]
    magAz2 = np.array(magdata['az'])[g]
    magAlt2 = np.array(magdata['el'])[g]
    magSecz2 = 1/(np.cos((90-magAlt2)*(np.pi/180)))

    dimfile = os.path.dirname(os.path.dirname(os.path.dirname(codedir))) + '/data/seeing/dimm_2014.csv'
    dimdata = pd.read_csv(dimfile)
    dimT = Time(np.array(dimdata['tm']).astype(str), format='fits')
    dimMJD = dimT.mjd
    dimSeeing = np.array(dimdata['se'])

    # Get DIMM seeing data
    print('Reading DIMM seeing data')

    exp = fits.getdata(apodir+'monitor/'+load.instrument+'Sci.fits')

    dt = np.dtype([('FRAMENUM',               np.int32),
                   ('MJD',                    np.int32),
                   ('PLATE',                  np.int32),
                   ('EXPTIME',                np.int32),
                   ('NREAD',                  np.int16),
                   ('HMAG_FID',               np.float64),
                   ('SNR_FID',                np.float64),
                   ('SNR_FID_SCALE',          np.float64),
                   ('LOGSNR_HMAG_COEF_ALL',   np.float64,2),
                   ('LOGSNR_HMAG_COEF',       np.float64,2),
                   ('SNR_PREDICT',            np.float64),
                   ('SNR_FID_1',              np.float64),
                   ('SNR_FID_SCALE_1',        np.float64),
                   ('LOGSNR_HMAG_COEF_1',     np.float64,2),
                   ('ZERO',                   np.float64),
                   ('ZERONORM',               np.float64),
                   ('EXPTYPE',                np.str),
                   ('DITHERPOS',              np.float64),
                   ('N_10pt0_11pt5',          np.int16),
                   ('FIBID',                  np.int16,300),
                   ('FIBINDEX',               np.int16,300),
                   ('FIBHMAG',                np.float64,300),
                   ('FIBRA',                  np.float64,300),
                   ('FIBDEC',                 np.float64,300),
                   ('FIBTYPE',                np.int16,300), # 1 if science, 0 if sky
                   ('FIBFLUX',                np.float64,300),
                   ('FIBERR',                 np.float64,300),
                   ('FIBSNR',                 np.float64,300),
                   ('SEEING',                 np.float64),
                   ('SNRATIO',                np.float64),
                   ('MOONDIST',               np.float64),
                   ('MOONPHASE',              np.float64),
                   ('SECZ',                   np.float64),
                   ('ZERO1',                  np.float64),
                   ('ZERORMS1',               np.float64),
                   ('ZERONORM1',              np.float64),
                   ('SKY',                    np.float64),
                   ('SEEING_BAADE',           np.float64),
                   ('SECZ_BAADE',             np.float64),
                   ('AZ_BAADE',               np.float64),
                   ('SEEING_CLAY',            np.float64),
                   ('SECZ_CLAY',              np.float64),
                   ('AZ_CLAY',                np.float64),
                   ('SEEING_DIMM',            np.float64),
                   ('SECZ_DIMM',              np.float64),
                   ('AZ_DIMM',                np.float64)])

    outstr = np.zeros(nfiles, dtype=dt)

    for i in range(nfiles):
        #if i % 10 != 0: continue
        print('('+str(i+1).zfill(5)+'/'+nfilesS+'): '+os.path.basename(files[i]))
        d1 = fits.getdata(files[i])
        d2 = fits.getdata(files[i],2)
        mjdS = str(getmjd(d1['framenum'][0]))
        outstr['FRAMENUM'][i] = d1['framenum'][0]
        outstr['NREAD'][i] = d1['read'][0]
        outstr['HMAG_FID'][i] = d1['hmag_fid'][0]
        outstr['SNR_FID'][i] = d1['snr_fid'][0]
        outstr['SNR_FID_SCALE'][i] = d1['snr_fid_scale'][0]
        outstr['LOGSNR_HMAG_COEF_ALL'][i] = d1['logsnr_hmag_coef_all'][0]
        outstr['LOGSNR_HMAG_COEF'][i] = d1['logsnr_hmag_coef'][0]
        outstr['SNR_PREDICT'][i] = d1['snr_predict'][0]
        outstr['ZERO'][i] = d1['zero'][0]
        outstr['ZERONORM'][i] = d1['zeronorm'][0]
        outstr['EXPTYPE'][i] = d1['exptype'][0]
        outstr['DITHERPOS'][i] = d1['ditherpos'][0]
        g, = np.where((d2['hmag'] >= 10.0) & (d2['hmag'] <= 11.5))
        outstr['N_10pt0_11pt5'][i] = len(g)
        if outstr['N_10pt0_11pt5'][i] == 0: print('  zero stars with 10.0 < H < 11.5')
        g1, = np.where(d1['framenum'] == exp['im'])
        if len(g1) > 0:
            apT = Time(exp['DATEOBS'][g1][0], format='fits')
            outstr['MJD'][i] = apT.mjd
            outstr['PLATE'][i] = exp['PLATE'][g1][0]
            outstr['EXPTIME'][i] = exp['EXPTIME'][g1][0]
            outstr['SEEING'][i] = exp['SEEING'][g1][0]
            outstr['SNRATIO'][i] = exp['SNRATIO'][g1][0]
            outstr['MOONDIST'][i] = exp['MOONDIST'][g1][0]
            outstr['MOONPHASE'][i] = exp['MOONPHASE'][g1][0]
            outstr['SECZ'][i] = exp['SECZ'][g1][0]
            outstr['ZERO1'][i] = exp['ZERO'][g1][0]
            outstr['ZERORMS1'][i] = exp['ZERORMS'][g1][0]
            outstr['ZERONORM1'][i] = exp['ZERONORM'][g1][0]
            outstr['SKY'][i] = exp['SKY'][g1][0][1]
            
        outstr['FIBID'][i] = d2['fiberid']
        outstr['FIBINDEX'][i] = d2['fiberid']
        outstr['FIBHMAG'][i] = d2['hmag']
        outstr['FIBRA'][i] = d2['fiberid']
        outstr['FIBDEC'][i] = d2['fiberid']
        g, = np.where(d2['objtype'] != 'SKY')
        if len(g) > 0: outstr['FIBTYPE'][i][g] = np.full(len(g),1)
        outstr['FIBFLUX'][i] = d2['flux']
        outstr['FIBERR'][i] = d2['err']
        outstr['FIBSNR'][i] = d2['snr']
        g, = np.where((d2['objtype'] != 'SKY') & (d2['hmag']>8.0) & (d2['HMAG'] < 11.5) & (d2['snr'] > 0))
        if len(g) > 20:
            print('  '+str(len(g))+' stars with H between 8 and 11.5')
            coefall = np.polyfit(d2['hmag'][g], np.log10(d2['snr'][g]), 1)
            snr_fid = 10**np.polyval(coefall,11.0)
            scale = np.sqrt(10**(0.4*(d2['hmag'][g] - 11.0)))
            snr_fid_scale = np.median(d2['snr'][g] * scale)
            outstr['SNR_FID_1'][i] = snr_fid
            outstr['SNR_FID_SCALE_1'][i] = snr_fid_scale
            outstr['LOGSNR_HMAG_COEF_1'][i] = coefall
        if len(g1) > 0:
            # Magellen Baade seeing data
            tdif = np.abs(apT.mjd - magMJD1)
            g, = np.where(tdif == np.nanmin(tdif))
            tdifmin = tdif[g][0]
            if tdifmin*24*60 < 15:
                print('  adding Magellan-Baade seeing data ('+str("%.5f" % round(magMJD1[g][0],5))+'-'+str("%.5f" % round(apT.mjd,5))+' = '+str("%.3f" % round(tdifmin*24*60,3))+' minutes)')
                outstr['SEEING_BAADE'][i] = magSeeing1[g][0]
                outstr['AZ_BAADE'][i] = magAz1[g][0]
                outstr['SECZ_BAADE'][i] = magSecz1[g][0]
            # Magellen Clay seeing data
            tdif = np.abs(apT.mjd - magMJD2)
            g, = np.where(tdif == np.nanmin(tdif))
            tdifmin = tdif[g][0]
            if tdifmin*24*60 < 15:
                print('  adding Magellan-Clay seeing data ('+str("%.5f" % round(magMJD2[g][0],5))+'-'+str("%.5f" % round(apT.mjd,5))+' = '+str("%.3f" % round(tdifmin*24*60,3))+' minutes)')
                outstr['SEEING_CLAY'][i] = magSeeing2[g][0]
                outstr['AZ_CLAY'][i] = magAz2[g][0]
                outstr['SECZ_CLAY'][i] = magSecz2[g][0]
            # DIMMy seeing data
            tdif = np.abs(apT.mjd - dimMJD)
            g, = np.where(tdif == np.nanmin(tdif))
            tdifmin = tdif[g][0]
            if tdifmin*24*60 < 15:
                print('  adding DIMM seeing data ('+str("%.5f" % round(dimMJD[g][0],5))+'-'+str("%.5f" % round(apT.mjd,5))+' = '+str("%.3f" % round(tdifmin*24*60,3))+' minutes)')
                outstr['SEEING_DIMM'][i] = dimSeeing[g][0]
        # Add in secz if missing
        if outstr['SECZ'][i] == 0:
            onedfile = load.filename('1D', num=outstr['FRAMENUM'][i], chips=True).replace('1D-','1D-b-')
            if os.path.exists(onedfile) == False: onedfile = expdir4+load.instrument+'/'+mjdS+'/as1D-b-'+str(outstr['FRAMENUM'][i])+'.fits'
            if os.path.exists(onedfile) == False: continue
            hdr=fits.getheader(onedfile)
            try: 
                outstr['SECZ'][i] = hdr['ARMASS']
                print('  added in SECZ from as1D header')
                del onedfile
                del hdr
            except: pass
        del d1
        del d2
    print('made '+outfile)
    Table(outstr).write(outfile, overwrite=True)

def nanmedfilt(x,size,mode='reflect'):
    return generic_filter(x, np.nanmedian, size=size)

def getmjd(frame) :
    """ Get chracter MJD from frame number """
    num = (int(frame) - int(frame)%10000 ) / 10000
    return int(num)+55562

def mad(data):
    """ Calculate the median absolute deviation."""
    data_median = np.nanmedian(data)
    result =  np.nanmedian(np.abs(data - data_median))
    return result * 1.482602218505602

class Frame:
    """Object class for a single APOGEE read image.

    Parameters
    ----------
    imfile: str
          The filename of the FITS file.
    head : astropy header
         The header of the FITS file.
    im : numpy array
       The 2D raw image.
    framenum : int
       APOGEE 8-digit frame number.
    num : int
       The read number of the image.

    Attributes
    ----------
    Same as the parameters.

    """

    def __init__(self,imfile,head,im,framenum,num):
        self.file = imfile
        self.head = head
        self.im = im
        self.framenum = framenum
        self.num = num


class SpecFrame:
    """Object class for an extracted APOGEE spectrum.

    Parameters
    ----------
    flux : numpy array
        The 2D flux array [Nfibers,Ncolumns].
    err : numpy array
        The 2D error array [Nfibers,Ncolumns].
    head : astropy header
         The header of the FITS file (from the last read used).

    """

    def __init__(self,flux,err,head,framenum):
        self.flux = flux
        self.err = err
        self.head = head
        self.framenum = framenum

def refcorrsub(image,ref):
    # refsub subtracts the reference array from each quadrant with proper flipping 
    revref = np.flip(ref,axis=1)
    image[:,0:512] -= ref
    image[:,512:1024] -= revref
    image[:,1024:1536] -= ref
    image[:,1536:2048] -= revref
    return image

def refcorr(image):
    """ Apply reference pixel correction to an apRaw APOGEE image."""

    # The RAW images:
    # Basic orientation is long wavelengths on the left, shorter to the
    # right (higher column number).  readout direction is:
    #         Red                       Green                 Blue           WG RefB RefG RefR
    #|---->|<----|--->|<----||---->|<----|--->|<----||---->|<----|--->|<----||---->|<----|--->|<----|
    image = image.astype(int)

    # Chip loop
    im = image[:,:2048]
    ref = image[:,2048:]
    # Reference subtraction
    im = refcorrsub(im,ref)
    # Stick it back in the array
    image[:,:2048] = im

    return image

# Load the frames
def loadframes(filename,hdulist,framenum,load=None,nfowler=2,chip=2,lastread=None):
    """Loads apRaw reads at beginning and end of an exposure.

    This function loads apRaw reads for an exposure at the beginning
    and end to be used for Fowler/CDS sampling.
    
    Parameters
    ----------
    rawdir : str
           Directory for the apRaw files.
    framenum : str
          The 8 digit APOGEE frame number.
    nfowler : int
            Number of reads to load.  The default is 2.
    chip : int
            Which detector to use (1, 2, or 3).  The default is 2. 
    lastread : int
            The number of the last read to use.

    Returns
    -------
    bframes : list of Frame objects
            List of Frame objects with images and header information for the
            reads at the beginning of the exposure.  Note, the first read is
            always skipped.
    eframes : list of Frame objects
            List of Frame objects with images and header information for the
            reads at the end of the exposure.
    nreads : int
           The number of reads (files) used.

    Example
    -------

    .. code-block:: python

        bframes,eframes,nreads = loadframes('/data/apogee/spectro/raw/58382/',28200054,nfowler=2,chip=2)

    """

    # Get the file list
    #files = glob(rawdir+load.prefix+"Raw-"+str(framenum)+"-???.fits")
    nfiles = len(hdulist)-1
    if nfiles==0:
        print("No files for "+str(framenum))
        return None,None,None
    # Sort the files
    #files = np.sort(files)
    # Get the read numbers
    readnum = np.arange(0,nfiles)
    #readnum = np.zeros(nfiles,dtype=int)
    #for i in range(nfiles):
    #    base = os.path.basename(files[i])
    #    # apRaw-28190009-059.fits
    #    readnum[i] = np.int(base[15:18])
    # If readnum input then only use reads up to that number
    if lastread is not None:
        gdf, = np.where(readnum <= int(lastread))
        ngdf = len(gdf)
        if ngdf < 2:
            raise Exception("Not enough reads")
        # Only keep the files that we want to use
        files = files[gdf]
        readnum = readnum[gdf]
        nfiles = ngdf
    # What nfowler are we using
    # Use nfowler=1 for DOMEFLAT
    #head = fits.getheader(files[0].replace('Raw','R').replace('-001.fits','.fits'))
    head = fits.getheader(filename)
    #exptype = head1.get('exptype')
    #if exptype=='DOMEFLAT':
    #    print('Using nfowler=1 for DOMEFLAT')
    #    nfowler = 1
    nfowler_use = nfowler
    # Raise an error if we don't have enough reads for nfowler
    if nfiles<3:
        raise Exception("Not enough reads ("+str(nfiles)+")")
    if nfiles<(2*nfowler+1):
        nfowler_use = np.int(np.floor((nfiles-1)/2.))
        print("Not enough reads ("+str(nfiles)+") for Nfowler="+str(nfowler)+".  Using "+str(nfowler_use)+" instead")
    #if nfiles<nfowler:
    #    raise Exception("Not enough reads ("+str(nfiles)+") for Nfowler="+str(nfowler))
    # Load the begging set of frames
    #  skip the first one, bad
    bframes = []
    for i in range(nfowler_use):
        imfile = filename#files[i+1]
        num = readnum[i+1]
        im = hdulist[i+1].data
        #im = fits.getdata(imfile)#,header=True)
        im = refcorr(im)  # apply reference correction   
        if chip is not None:
            im = im[:,:2048]
            #im = im[:,(chip-1)*2048:chip*2048]
        frame = Frame(imfile,head,im,framenum,num)
        bframes.append(frame)
    # Load the ending set of frames
    eframes= []
    for i in range(nfowler_use):
        imfile = filename#files[nfiles-nfowler_use+i]
        num = readnum[nfiles-nfowler_use+i]
        im = hdulist[nfiles-nfowler_use+i].data
        #im,head = fits.getdata(imfile,header=True)
        im = refcorr(im)  # apply reference correction   
        if chip is not None:
            im = im[:,:2048]
            #im = im[:,(chip-1)*2048:chip*2048]
        frame = Frame(imfile,head,im,framenum,num)
        eframes.append(frame)
    # Return the lists
    return bframes,eframes,nfiles



# Perform Fowler sampling
def fowler(bframes,eframes):
    """Collapses exposure to 2D using Fowler/CDS sampling.

    This function performs Fowler/CDS sampling given a list of
    raw frames from the beginning and end of the exposure.
    
    Parameters
    ----------
    bframes : list of Frame objects
            The read images at the beginning of the exposure.
    eframes : list of Frame objects
            The read images at the end of the exposure.

    Returns
    -------
    im : numpy array
       The collapsed 2D image.


    Example
    -------

    .. code-block:: python

        im = fowler(bframes,eframes)

    """

    # Beginning sample
    nbeg = len(bframes)
    nx,ny = bframes[0].im.shape
    if (nbeg==1):
        im_beg = np.array(bframes[0].im.copy(),dtype=float)
    else:
        im_beg = np.array(bframes[0].im.copy(),dtype=float)*0
        for i in range(nbeg): im_beg += np.array(bframes[i].im.copy(),dtype=float)
        im_beg /= np.float(nbeg)
    # End sample
    nend = len(eframes)
    if (nend==1):
        im_end = np.array(eframes[0].im.copy(),dtype=float)
    else:
        im_end = np.array(eframes[0].im.copy(),dtype=float)*0
        for i in range(nend): im_end += np.array(eframes[i].im.copy(),dtype=float)
        im_end /= np.float(nend)
    # The middle read will be used twice for 3 reads
    # Subtract beginning from end
    im = im_end - im_beg

    # Fix the background level by using the upper and lower edges
    # first and last 10 rows
    # Do a linear ramp for each quadrant
    exptype = eframes[0].head['exptype']
    if exptype != 'INTERNALFLAT' and exptype != 'DARK':
        for i in range(4):
            med1 = np.median(im[0:10,i*512:(i+1)*512])
            med2 = np.median(im[-10:,i*512:(i+1)*512])
            im[:,i*512:(i+1)*512] -= (np.arange(2048).astype(float)*(med2-med1)/2047+med1).reshape(-1,1)

    return im


# Construct the noise model images
def noisemodel(im,nreads,noise,gain):
    """Computes the noise model for an image.

    This function computes the noise image for the case where
    the data cube had been collapsed using UTR as the full
    reduction code does.
    
    Parameters
    ----------
    im : numpy array
       The 2D collapsed image.
    nreads : int
       The number of reads in the image.
    noise : float
       The read noise for a single read in ADU.
    gain : float
       The gain in electrons/ADU.

    Returns
    -------
    err : numpy array
        The noise model image in ADU.

    Example
    -------

    .. code-block:: python

        err = noisemodel(im,nreads,noise,gain)

    """

    # need nread, ngdreads, noise, gain
    ngdreads = np.float(nreads-1)

    ## READ NOISE
    #if n_elements(rdnoiseim) gt 0 then begin
    #  noise = median(rdnoiseim)
    #endif else begin
    #  noise = 12.0  ; default value
    #endelse

    # See Equation 1 in Rauscher et al.(2007), SPIE
    #  with m=1
    #  noise and image/flux should be in electrons, sample_noise is in electrons
    impos = im.copy()
    bad = (impos < 0)
    impos[bad] = 0
    sample_noise = np.sqrt( 12*(ngdreads-1)/(np.float(nreads)*(ngdreads+1))*noise**2 + 6.*(ngdreads**2+1)/(5*ngdreads*(ngdreads+1))*impos*gain )
    sample_noise /= gain  # convert to ADU

    # Start the variance array
    varim = im.copy()*0
    # Sample/read noise
    varim += sample_noise**2 

    #Now convert to ELECTRONS
    #if n_elements(detcorr) gt 0 and keyword_set(outelectrons) then begin
    #  varim *= gain^2
    #  im *= gain

    # Convert variance to error
    err = im.copy()*0+1
    good = (varim > 0)
    err[good] = np.sqrt(varim[good])

    return err

def bpmfix(frame,bpm):
    """ Fix bad pixels."""

    # Replace them with averages of the neighboring pixels
    bad = (bpm>0)
    nbpm = np.sum(bad)
    if nbpm>0:
        nei = np.zeros((nbpm,8),float)
        neimask = np.zeros((nbpm,8),bool)
        ind1,ind2 = np.where(bad)
        ind1p = ind1+1
        ind1p[ind1p>2047] = 0
        ind2p = ind2+1
        ind2p[ind2p>2047] = 0
        nei[:,0] = frame.im[ind1-1,ind2-1]
        nei[:,1] = frame.im[ind1,ind2-1]
        nei[:,2] = frame.im[ind1p,ind2-1]
        nei[:,3] = frame.im[ind1-1,ind2]
        nei[:,4] = frame.im[ind1p,ind2]
        nei[:,5] = frame.im[ind1-1,ind2p]
        nei[:,6] = frame.im[ind1,ind2p]
        nei[:,7] = frame.im[ind1p,ind2p]
        new = np.median(nei,axis=1)
        frame.im[ind1,ind2] = new

    frame.err[bad] = 1e30
    frame.head['HISTORY'] = str(nbpm)+' bad pixels masked out'

    return frame

# Boxcar extract fibers
def boxextract(frame,tracestr,fibers=None,xlo=0,xhi=None):
    """This function performs boxcar extraction on a 2D image.
    
    Parameters
    ----------
    frame : Frame object
          The Frame object that includes the flux and error images.
    tracestr : numpy structured array
          Numpy structured array that gives information on the traces.
    fibers : array or list, optional
           List/array of fibers to extract.  By default all fibers
           in tracestr are extracted.
    xlo : int, optional
        The starting column to extract.  By default this is 0.
    xhi : int, optional
        The ending column to extract.  This is set to the final column
        of the 2D iamge.

    Returns
    -------
    spec : SpecFrame object
         The SpecFrame object that contains extracted flux and error arrays.

    Example
    -------

    .. code-block:: python

        spec = boxextract(frame,tracestr)

    """

    ntrace = len(tracestr)
    if fibers is None:
        fibers = np.arange(ntrace)
    if xhi is None:
        xhi = frame.im.shape[1]-1
    # transpose b/c x/y are flipped in python
    flux = frame.im.copy().T
    err = frame.err.copy().T
    nx,ny = flux.shape
    # Loop through the Fibers
    ncol = np.int(xhi-xlo+1)
    x = np.arange(xlo,xhi+1)
    nfibers = len(fibers)
    oflux = np.zeros((nfibers,ncol))
    oerr = np.zeros((nfibers,ncol))
    for i in range(nfibers):
        fwhm = tracestr['FWHM'][fibers[i]]
        coef = tracestr['COEF'][fibers[i]]
        ymid = np.polyval(coef[::-1],x)   # reverse order
        ylo = np.int(np.min(np.floor(ymid-fwhm)))
        if ylo<0: ylo=0
        yhi = np.int(np.max(np.ceil(ymid+fwhm)))
        if yhi>(ny-1): yhi=ny-1
        num = yhi-ylo+1

        # Make a MASK based on the trace and FWHM
        yy = np.zeros(ncol,int).reshape(-1,1) + np.arange(ylo,yhi+1).reshape(1,-1)
        ymid2d = np.resize(np.repeat(ymid,num),(ncol,num))
        mask = np.array((yy >= np.floor(ymid2d-fwhm)) & (yy <= np.ceil(ymid2d+fwhm)),dtype=int)
        # Flux
        oflux[i,:] = np.sum( flux[xlo:xhi+1,ylo:yhi+1]*mask, axis=1)
        # Error
        #  add in quadrature
        oerr[i,:] = np.sqrt( np.sum( (err[xlo:xhi+1,ylo:yhi+1]**2)*mask, axis=1) )

    return SpecFrame(oflux,oerr,frame.head,frame.framenum)


# Calculate mean sky spectrum
def skysub(spec,plugmap=None,fps=False):
    """This subtracts the median sky spectrum from all of the fiber spectra.
    
    Parameters
    ----------
    spec : SpecFrame object
         The SpecFrame object that constrains the 1D extracted spectra.
    plugmap : numpy structured array
            The plugmap information for each fiber including which fiber contains
            sky or stars.

    Returns
    -------
    spec : SpecFrame object
         The same SpecFrame object but now with the sky spectrum subtracted.

    Example
    -------

    .. code-block:: python

        spec2 = skysub(spec,plugmap)

    """

    # Find the object and sky fibers
    if fps:
        fibermap = plugmap['FIBERMAP']
        category = np.char.array(fibermap['category']).astype(str).upper()
        sfibs, = np.where( (np.array(fibermap['fiberId'])>=0) & (np.array(fibermap['spectrographId'])==2) &
                           ((category=='SKY') | (category=='SKY_APOGEE') | (category=='SKY_BOSS')))
    else:
        fibermap = plugmap['PLUGMAPOBJ']
        #fibermap2 = plugmap2['STRUCT1']
        holetype = np.char.array(fibermap['holeType']).astype(str).upper()
        objtype = np.char.array(fibermap['objType']).astype(str).upper()
        sfibs, = np.where((np.array(fibermap['fiberId'])>=0) & (holetype=='OBJECT') & (np.array(fibermap['spectrographId'])==2) & (objtype=='SKY'))
        #sfibs2, = np.where( (fibermap2['fiberid']>=0) & (fibermap2['assigned']==1) & (objtype2=='SKY'))
        #sfibid2 = fibermap2['fiberid'][sfibs2]
    sfibid = np.array(fibermap['fiberId'])[sfibs]

    # We have sky fibers
    if len(sfibs)>0:
        skyindex = 300-sfibid
        # Calculate the median sky spectrum
        skyspec = np.median(spec.flux[skyindex,:],axis=0)
        # Median smooth to keep the sky lines (observers use these to check dither position)
        smskyspec = nanmedfilt(skyspec,200)
        # Subtract from all fibers
        nspec,ncol = spec.flux.shape
        # Make new object
        subspec = SpecFrame(spec.flux,spec.err,spec.head,spec.framenum)
        for i in range(nspec): subspec.flux[i,:] -= smskyspec
    # No sky fibers
    else:
        subspec = SpecFrame(spec.flux,spec.err,spec.head,spec.framenum)

    return subspec


# Calculate median S/N per fiber
def snrcat(spec,plugmap1=None,plugmap2=None,fps=False):
    """This function calculates the S/N for each fiber.
    
    Parameters
    ----------
    spec : SpecFrame object
         The SpecFrame object that constrains the 1D extracted spectra.
    plugmap : numpy structured array
            The plugmap information for each fiber including which fiber contains
            sky or stars.

    Returns
    -------
    cat : numpy structured array
         A catalog containing information on each object in the fibers and the
         median S/N.

    Example
    -------

    .. code-block:: python

        cat = snrcat(spec,plugmap)

    """

    nfibers,npix = spec.flux.shape
    dtype = np.dtype([('apogee_id',np.str,30),('catalogid',np.int),('ra',np.float64),('dec',np.float64),('hmag',np.float),
                      ('objtype',np.str,30),('fiberid',np.int),('fiberindex',np.int),('flux',np.float),('err',np.float),
                      ('snr',np.float)])
    cat = np.zeros(nfibers,dtype=dtype)

    # Load the spectral data
    cat['fiberindex'] = np.arange(nfibers)
    cat['flux'] = np.median(spec.flux,axis=1)
    cat['err'] = np.median(spec.err,axis=1)
    err = cat['err']
    bad = (err <= 0.0)
    err[bad] = 1.0
    cat['snr'] = cat['flux']/err
            
    # Load the plugging data
    if fps:
        plugmap = plugmap1
        fibermap = plugmap['FIBERMAP']     # SDSS-V FPS
        fibs, = np.where( (fibermap['fiberId']>=0) & (fibermap['spectrographId']==2) )            
        fiberindex = 300-fibermap['fiberId'][fibs]
        cat['hmag'][fiberindex] = fibermap[fibs]['h_mag']
        cat['catalogid'][fiberindex] = fibermap[fibs]['catalogid']
        cat['objtype'][fiberindex] = 'OBJECT'   # default
        category = np.char.array(fibermap['category']).astype(str).upper()
        skyind = (category[fibs]=='SKY') | (category[fibs]=='SKY_APOGEE') | (category[fibs]=='SKY_BOSS')
        #skyind = bmask.is_bit_set(fibermap['sdssv_apogee_target0'][fibs],0)==1    # SKY
        if np.sum(skyind)>0:
            cat['objtype'][fiberindex[skyind]] = 'SKY'
        tellind = (category[fibs]=='HOT_STD') | (category[fibs]=='STANDARD_APOGEE') 
        #tellind = bmask.is_bit_set(fibermap['sdssv_apogee_target0'][fibs],1)==1   # HOT_STD/telluric
        if np.sum(tellind)>0:
            cat['objtype'][fiberindex[tellind]] = 'HOT_STD'
        cat['ra'][fiberindex] = fibermap[fibs]['ra']
        cat['dec'][fiberindex] = fibermap[fibs]['dec']
        cat['fiberid'][fiberindex] = fibermap[fibs]['fiberId']
    else:
        fibermap1 = plugmap1['PLUGMAPOBJ']     # SDSS-III-IV
        fibermap2 = plugmap2['STRUCT1']
        holetype1 = np.char.array(fibermap1['holeType']).astype(str).upper()
        objtype1 = np.char.array(fibermap1['objType']).astype(str).upper()
        holetype2 = np.char.array(fibermap2['holetype']).astype(str).upper()
        objtype2 = np.char.array(fibermap2['targettype']).astype(str).upper()
        #fibs1, = np.where((fibermap1['fiberId']>=0) & (holetype1=='OBJECT') & (fibermap1['spectrographId']==2))
        fibs2, = np.where((fibermap2['fiberid']>=0) & (fibermap2['assigned']==1) & ((holetype2=='APOGEE') | (holetype2=='APOGEE_SOUTH')) & (objtype2!='NA'))          
        fiberindex = 300-fibermap2['fiberid'][fibs2]
        cat['objtype'][fiberindex] = 'OBJECT'   # default
        skyind, = np.where(objtype2[fibs2] == 'SKY')
        if len(skyind) > 0: cat['objtype'][fiberindex[skyind]] = 'SKY'
        tellind, = np.where(objtype2[fibs2] == 'STANDARD')
        if len(tellind) > 0: cat['objtype'][fiberindex[tellind]] = 'HOT_STD'
        cat['fiberid'][fiberindex] = fibermap2[fibs2]['fiberid']
        cat['ra'][fiberindex] = fibermap2[fibs2]['target_ra']
        cat['dec'][fiberindex] = fibermap2[fibs2]['target_dec']
        cat['hmag'][fiberindex] = fibermap2[fibs2]['tmass_h']

    cat = Table(cat)
    return cat


# Linear fit to log(S/N) vs. H
def snrhmag(cat,nreads,nframes,hfid=11.0):
    """Returns the S/N for the fiducial H magnitude.
apPlateSum-5815-56396.fits
    This fits a line to log(S/N) vs. Hmag for the stars and
    predicts what the S/N should be at the end of the exposure.
    
    Parameters
    ----------
    cat : numpy structured array
         A catalog containing information on each object in the fibers and the
         median S/N.
    nreads: int
         The number of reads used in the current processing.
    nframes: int
         The total number of reads in the entire exposure.
    hfid : float, optional
         The fiducial Hmag.  The default is H=11.0 for SDSS-V/MWM in the FPS era.

    Returns
    -------
    coefstr : numpy structured array
            An structure that contains all of the information computed:
            hmag_fid : The fiducial Hmag
            logsnr_hmag_coef : The linear coefficients of the fit of log(S/N) vs. Hmag
            snr_fid : The S/N at the fiducial Hmag using the linear fit.
            snr_predict : The predicted S/N at the fiducial Hmag at the end of the expsure.

    Example
    -------

    .. code-block:: python

        coefstr = snrhmag(cat,30,47)

    """

    # Linear fit to log(snr) vs. Hmag for ALL objects
    gdall, = np.where( (cat['objtype'] != 'SKY') & (cat['hmag'] > 4) & (cat['hmag'] < 20) & (cat['snr'] > 0))
    if len(gdall)>2:
        coefall = np.polyfit(cat[gdall]['hmag'],np.log10(cat[gdall]['snr']),1)
    else:
        coefall = np.zeros(2,float)+np.nan
    # Linear fit to log(S/N) vs. H for 10<H<11.5
    gd, = np.where( (cat['objtype'] != 'SKY') & (cat['hmag']>=10.0) & (cat['hmag']<=11.5) & (cat['snr'] > 0))
    if len(gd)>2:
        coef = np.polyfit(cat[gd]['hmag'],np.log10(cat[gd]['snr']),1)
    else:
        coef = np.zeros(2,float)+np.nan
    if len(gd)>2:
        snr_fid = 10**np.polyval(coef,hfid)
    elif len(gdall)>2:
        snr_fid = 10**np.polyval(coefall,hfid)
    else:
        snr_fid = np.mean(cat['snr'])

    # This is the full pipeline exposure-level S/N (altsn)
    #   copied from apogeereduce/qa/plotmag.pro
    #   alternative S/N as computed from median of all stars with H<11.0, scaled
    snstars, = np.where( (cat['objtype'] != 'SKY') & (cat['hmag'] > 4) & (cat['hmag'] < hfid) & (cat['snr'] > 0))
    if len(snstars)>0:
        scale = np.sqrt(10**(0.4*(cat['hmag'][snstars]-hfid)))
        snr_fid_scale = np.median(cat['snr'][snstars]*scale)
    else:
        snr_fid_scale = 0.0

    # Predicted S/N at end of exposure
    #  (S/N)^2 should scale with time
    snr_predict = np.sqrt( snr_fid**2*np.float(nframes)/np.float(nreads) )

    # Calculate zeropoints from known H band mags.
    # Use a static zeropoint to calculate sky brightness.
    fiberobj, = np.where((cat['objtype'] != 'SKY') & (cat['hmag'] > 4) & (cat['flux']>0))
    skyzero = 14.75 + (2.5 * np.log10(nreads))
    tmp = cat['hmag'][fiberobj] + (2.5 * np.log10(cat['flux'][fiberobj]))
    zero = np.nanmedian(tmp)
    zerorms = mad(cat['hmag'][fiberobj] + (2.5 * np.log10(cat['flux'][fiberobj])))
    zeronorm = zero - (2.5 * np.log10(nreads))

    dtype = np.dtype([('framenum',int),('read',int),('hmag_fid',np.float),('snr_fid',np.float),
                      ('snr_fid_scale',float),('logsnr_hmag_coef_all',(np.float,2)),('logsnr_hmag_coef',(np.float,2)),
                      ('snr_predict',np.float),('zero',float),('zeronorm',float)])
    coefstr = np.zeros(1,dtype=dtype)
    coefstr['hmag_fid'] = hfid
    coefstr['snr_fid'] = snr_fid
    coefstr['snr_fid_scale'] = snr_fid_scale
    coefstr['logsnr_hmag_coef_all'] = coefall
    coefstr['logsnr_hmag_coef'] = coef
    coefstr['snr_predict'] = snr_predict
    coefstr['zero'] = zero
    coefstr['zeronorm'] = zeronorm
    coefstr = Table(coefstr)

    return coefstr


# Run everything
def runquick(filename,hdulist=None,framenum=None,mjd=None,load=None,psfnums=None,apred='daily',lastread=None,hfid=11.0,plugfile=None,ncol=2048):
    """This runs all of the main steps of the quick reduction.
    
    Parameters
    ----------
    filename : string
             The full path of the apR file to process.
    framenum : string
             The APOGEE frame number of the exposure to process.
    mjd : string
             The modified Julian date of the apR file.
    lastread : int or string
             The number for the last read to use.
    hfid : float, optional
         The fiducial Hmag.  The default is 11.0.
    plugfile : string, optional
         The absolute path to the plugmap filename.
    ncol : int, optional
         Number of columns to extract.  Default is 51.

    Returns
    -------
    frame : Frame object
          The extracted 2D image and noise model.
    spec : SpecFrame object
         The extracted spectra and errors with the sky subtracted.
    cat : numpy structured array
        The catalog of information on each fiber including the S/N.
    coefstr : numpy structured array
            Information on the fit of log(S/N) vs. Hmag and the
            S/N at the fiducial Hmag.

    Example
    -------

    .. code-block:: python

        frame, spec, cat, coefstr = runquick('28200054','apo',10)

    """

    print('Running APQUICK on '+os.path.basename(filename)+' MJD='+mjd+' all reads ')

    fps = False
    imjd = int(mjd)
    if load.telescope == 'apo25m' and imjd >= 59146: fps = True
    if load.telescope == 'lco25m' and imjd >= 59809: fps = True

    head = fits.getheader(filename)#eframes[0].head.copy()   # header of first read
    nframes = np.int(head['NFRAMES'])
    exptype = head['EXPTYPE']
    plateid = head['PLATEID']
    if exptype is None: exptype=''
    #try: nreads = head['NREADS']
    #except: nreads = head['NFRAMES']

    # Setup directories and load the plugmap (FPS) or plateHolesSorted (plate) 
    plugmap = None
    rawdir = os.path.dirname(filename)+'/'
    if fps == False:
        redux_dir = '/uufs/chpc.utah.edu/common/home/sdss/apogeework/apogee/spectro/redux/current/'
        caldir = redux_dir+'cal/'
        plugdir = '/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/mapper/'+load.observatory+'/'
        phsdir = '/uufs/chpc.utah.edu/common/home/sdss/software/svn.sdss.org/data/sdss/platelist/trunk/plates/'
        planfile = load.filename('Plan', num=framenum, plate=plateid, mjd=mjd).replace('.yaml','.par')
        tmp = planfile.split('/')
        planfile = redux_dir+'visit/'+load.telescope+'/'+tmp[-4]+'/'+tmp[-3]+'/'+tmp[-2]+'/'+tmp[-1]
        if os.path.exists(planfile) == False:
            print(planfile+' NOT FOUND. Stopping.')
            return
        plans = yanny.yanny(planfile,np=True)
        plugid = plans['plugmap'].replace("'",'')
        pludmjd = plugid.split('-')[1]
        plugfile1 = plugdir+pludmjd+'/''plPlugMapM-'+plugid+'.par'
        plfolder = '{:0>4d}XX'.format(int(head['PLATEID']) // 100)
        plstr = str(plateid).zfill(6)
        plugfile2 = phsdir+plfolder+'/'+plstr+'/plateHolesSorted-'+plstr+'.par'
        detfile = caldir+'detector/'+load.prefix+'Detector-b-'+str(plans['detid']).zfill(8)+'.fits'
        bpmfile = caldir+'bpm/'+load.prefix+'BPM-b-'+str(plans['bpmid']).zfill(8)+'.fits'
        if load.telescope == 'lco25m': psffile = caldir+'psf/'+load.prefix+'PSF-b-'+str(plans['psfid']).zfill(8)+'.fits'
    else:
        redux_dir = os.environ.get('APOGEE_REDUX')+'/'+load.apred+'/'
        caldir = redux_dir+'cal/'+load.instrument+'/'
        plugdir = os.environ['SDSSCORE_DIR']+'/'+load.observatory.lower()+'/summary_files/'
        configid = head['configid']
        if configid is not None and str(configid) != '':
            configgrp = '{:0>4d}XX'.format(int(configid) // 100)
            plugfile = plugdir+configgrp+'/confSummary-'+str(configid)+'.par'
            print('Using configid from first read header: '+str(configid))
        else:
            print('No configID in header.')
            if plugfile is not None:
                print('Using input plugfile '+str(plugfile))
        plugfile1 = plugfile
        plugfile2 = plugfile
        planfile = load.filename('Plan', num=framenum, plate=configid, mjd=mjd)
        if os.path.exists(planfile) == False:
            print(planfile+' NOT FOUND. Stopping.')
            pdb.set_trace()
        plans = yanny.yanny(planfile,np=True)
        detfile = load.filename('Detector', num=plans['detid:'], chips=True).replace('Detector-','Detector-b-')
        bpmfile = load.filename('BPM', num=plans['bpmid:'], chips=True).replace('BPM-','BPM-b-')
        if load.telescope == 'lco25m': psffile = load.filename('PSF', num=plans['psfid:'], chips=True).replace('PSF-','PSF-b-')
    redux_dir = os.environ.get('APOGEE_REDUX')+'/'+load.apred+'/'
    caldir = redux_dir+'cal/'+load.instrument+'/'
    if load.telescope == 'apo25m': psffile = caldir+'psf/apPSF-b-39540016_300fibers.fits'
    if load.telescope == 'apo25m': bpmfile = caldir+'bpm/apBPM-b-33770001.fits'
    if load.telescope == 'lco25m':
        if imjd < 58000: psffile = caldir+'psf/asPSF-b-23500008.fits'
        if imjd > 58000 and imjd < 58125: psffile = '/uufs/chpc.utah.edu/common/home/sdss/apogeework/apogee/spectro/redux/current/cal/psf/asPSF-b-25280053.fits'
        if imjd > 58125 and imjd < 58520: psffile = '/uufs/chpc.utah.edu/common/home/sdss/apogeework/apogee/spectro/redux/current/cal/psf/asPSF-b-27960083.fits'
        if imjd > 58520 and imjd < 58725: psffile = '/uufs/chpc.utah.edu/common/home/sdss/apogeework/apogee/spectro/redux/current/cal/psf/asPSF-b-30660018.fits'
        if imjd > 58725 and imjd < 59000: psffile = '/uufs/chpc.utah.edu/common/home/sdss/apogeework/apogee/spectro/redux/current/cal/psf/asPSF-b-32930007.fits'
        if imjd > 59000 and imjd < 59500: psffile = '/uufs/chpc.utah.edu/common/home/sdss/apogeework/apogee/spectro/redux/current/cal/psf/asPSF-b-36380009.fits'
        if imjd > 59500: psffile = '/uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/apogee/spectro/redux/daily/cal/apogee-s/psf/asPSF-b-43880003.fits'
        #psffile = caldir+'psf/asPSF-b-36680072.fits'

    if os.path.exists(plugfile1) == False:
        print(plugfile1+' NOT FOUND. Stopping.')
        return
    if os.path.exists(plugfile2) == False:
        print(plugfile2+' NOT FOUND. Stopping.')
        return
    if os.path.exists(detfile) == False:
        print(detfile+' NOT FOUND. Stopping.')
        return
    if os.path.exists(bpmfile) == False: 
        print(bpmfile+' NOT FOUND. Stopping.')
        return
    if os.path.exists(psffile) == False: 
        print(psffile+' NOT FOUND. Stopping.')
        return

    print('Using '+plugfile1)
    print('      '+plugfile2)
    print('      '+planfile)
    print('      '+detfile)
    print('      '+bpmfile)
    print('      '+psffile)

    # Load files
    rdnoiseim = fits.getdata(detfile,1)
    rdnoise = np.median(rdnoiseim)
    gainim = fits.getdata(detfile,2)
    gain = np.median(gainim)
    bpm = fits.getdata(bpmfile,0)
    tracestr = fits.getdata(psffile)
    if fps:
        plugmap1 = yanny.yanny(plugfile1,np=True)
        plugmap2 = plugmap1#,np=True)
        fibermap1 = plugmap1['FIBERMAP']
        fibermap2 = fibermap1
        g, = np.where((np.array(fibermap1['fiberType'])=='APOGEE') &
                      (np.array(fibermap1['assigned'])==1) &
                      (np.array(fibermap1['on_target'])==1) &
                      (np.array(fibermap1['valid'])==1) &
                      (np.array(fibermap1['spectrographId'])==2) &
                      (np.array(fibermap1['fiberId'])>=0))
        fiberid = np.array(fibermap1['fiberId'])[g]
    else: 
        plugmap1 = yanny.yanny(plugfile1,np=True)
        plugmap2 = yanny.yanny(plugfile2,np=True)
        fibermap1 = plugmap1['PLUGMAPOBJ']
        fibermap2 = plugmap2['STRUCT1']        
        holetype = np.char.array(fibermap1['holeType'].astype(str)).upper()
        objtype = np.char.array(fibermap1['objType'].astype(str)).upper()
        g, = np.where((np.array(fibermap1['fiberId'])>=0) &
                      (holetype=='OBJECT') &
                      (np.array(fibermap1['spectrographId'])==2))
        fiberid = np.array(fibermap1['fiberId'])[g]

    # Load the reads
    nfowler = 2
    bframes,eframes,nreads = loadframes(filename,hdulist,framenum,load=load,nfowler=nfowler,lastread=lastread)
    if bframes is None:
        print('Cannot run quicklook')
        return None,None,None,None

    # Do Fowler/CDS collapse
    im = fowler(bframes,eframes)

    # Generate the noise image
    err = noisemodel(im,nreads,rdnoise,gain)
    frame = Frame("",head,im,framenum,0)
    frame.err = err

    # Add some important values to the header
    frame.head['FRAMENUM'] = framenum
    frame.head['PSFFILE'] = psffile
    frame.head['DETFILE'] = detfile
    frame.head['BPMFILE'] = bpmfile

    # Fix bad pixels
    frame = bpmfix(frame,bpm)

    # Load the trace information
    #gd, = np.where(psfnums - int(framenum) > 0)
    #gd1, = np.where(psfnums[gd] - int(framenum) == np.min(psfnums[gd] - int(framenum)))
    #psffile = psfdir+load.prefix+'PSF-b-'+str(psfnums[gd][gd1][0]).zfill(8)+'.fits'
    #gd, = np.where(np.abs(int(framenum)-psfnums) == np.nanmin(np.abs(int(framenum)-psfnums)))
    #psffile = psfdir+load.prefix+'PSF-b-'+str(psfnums[gd][0]).zfill(8)+'.fits'
    #if os.path.exists(psffile) == False:
    #    print(psffile+' NOT FOUND. Stopping.')
    #    pdb.set_trace()
    #psffiles = np.sort(glob(psfdir+'/'+load.prefix+'PSF-b-*.fits'))
    #psffiles = np.sort(glob(psfdir+'/apPSF-b-????????.fits'))

    # Boxcar extract the fibers
    midcol = 1024
    if ncol==2048:
        xlo = 0
        xhi = 2047
    else:
        half_ncol = ncol//2
        xlo = np.maximum(midcol-half_ncol,0)
        xhi = np.minimum(xlo+ncol-1,2047)
    spec = boxextract(frame,tracestr,xlo=xlo,xhi=xhi)

    # Add some important configuration values to the header
    if plugmap is not None:
        try: frame.head['CONFIGID'] = plugmap.get('configuration_id')
        except: frame.head['CONFIGID'] = ''
        frame.head['DESIGNID'] = plugmap.get('design_id')
        frame.head['FIELDID'] = plugmap.get('field_id')
        frame.head['CONFIGFL'] = plugfile
    else:
        frame.head['CONFIGID'] = ''
        frame.head['DESIGNID'] = ''
        frame.head['FIELDID'] = ''
        frame.head['CONFIGFL'] = ''
    # Subtract the sky
    if exptype == 'OBJECT' and plugmap1 is not None:
        subspec = skysub(spec,plugmap=plugmap1,fps=fps)
    else:
        subspec = spec

    # Create the S/N catalog
    cat = snrcat(subspec,plugmap1=plugmap1,plugmap2=plugmap2,fps=fps)
    print('Mean S/N = %5.2f' % np.mean(cat['snr']))
    if exptype != 'OBJECT':
        cat['apogee_id'] = 'N/A'
        cat['objtype'] = 'N/A'
        blank = ['ra','dec','hmag']
        for b in blank: cat[b] = 999999.
    # Fit line and get fiducial S/N
    coefstr = snrhmag(cat,nreads,nframes,hfid=hfid)
    # add some relevant columns
    coefstr['framenum'] = framenum
    coefstr['read'] = nreads
    coefstr['exptype'] = exptype
    coefstr['ditherpos'] = head.get('DITHPIX')
    if exptype=='OBJECT':
        # Print out the S/N values
        print('S/N (H=%5.2f) = %5.2f (%3d reads)' % (hfid,coefstr['snr_fid'],nreads))
        print('S/N (H=%5.2f) = %5.2f (prediction for %3d reads)' % (hfid,coefstr['snr_predict'],nframes))
    else:
        # set values to nan that don't make sense for non-object exposures
        coefstr['hmag_fid'] = np.nan
        coefstr['snr_fid'] = np.mean(cat['snr'])
        coefstr['snr_fid_scale'] = np.nan
        coefstr['logsnr_hmag_coef_all'] = np.nan
        coefstr['logsnr_hmag_coef'] = np.nan
        coefstr['zero'] = np.nan
        coefstr['zeronorm'] = np.nan
        print('Mean S/N = %5.2f' % coefstr['snr_fid'])

    return frame, subspec, cat, coefstr


# Write out results
def writeresults(outfile, frame, sframe, cat, coefstr, compress=False):
    """This writes out the results of the quick reduction to an output file.
    
    Parameters
    ----------
    outfile : str
          The output filename to write the results to.
    frame : Frame object
          The extracted 2D image and noise model.
    sframe : SpecFrame object
         The extracted spectra and errors with the sky subtracted.
    cat : numpy structured array
        The catalog of information on each fiber including the S/N.
    coefstr : numpy structured array
            Information on the fit of log(S/N) vs. Hmag and the
            S/N at the fiducial Hmag.
    compress : boolean, optional
         Gzip compress the final FITS file.  This reduces the file size
         by a factor of ~3x.  Default is False.
             
    Returns
    -------
    The function doesn't return anything but writes the results to an
    output file.

    Example
    -------

    .. code-block:: python

        writeresults('apq-28200054.fits', frame, spec, cat, coefstr)

    """

    print('Final catalog = '+outfile)

    if os.path.exists(outfile): os.remove(outfile)

    # HDU0: header only
    head = frame.head
    #head.add_history('apogee_mountain version: '+str(__version__))
    head.add_history("APQUICK results for "+str(head['FRAMENUM']))
    head.add_history("HDU0: Header only")
    head.add_history("HDU1: coefstr, S/N for fiducial Hmag")
    head.add_history("HDU2: cat, catalog of S/N values")
    head.add_history("HDU3: spec/err, numpy structured array")
    head.add_history("HDU4: image/err, numpy structured array")
    fits.writeto(outfile,None,head,overwrite=True)
    hdulist = fits.open(outfile)
    
    # HDU1: Coef structure
    if coefstr is not None:
          hdu = fits.table_to_hdu(Table(coefstr))
          hdu.header.add_history("APQUICK results for "+str(head['FRAMENUM']))
          hdu.header.add_history("Final fiducial S/N values")
    else:
          hdu = fits.PrimaryHDU(0)
          hdu.header.add_history("No S/N results for non-OBJECT frame "+str(head['FRAMENUM']))
    hdulist.append(hdu)
    # HDU2: Catalog
    hdu = fits.table_to_hdu(cat)
    hdu.header.add_history("APQUICK results for "+str(head['FRAMENUM']))
    hdu.header.add_history("Catalog of S/N values and other fiber information")
    hdulist.append(hdu)
    # HDU3: Spectra
    dtype = np.dtype([('flux',(np.float32,sframe.flux.shape)),('err',(np.float32,sframe.err.shape))])
    spec = np.zeros(1,dtype=dtype)
    spec['flux'] = sframe.flux
    spec['err'] = sframe.err
    spec = Table(spec)
    hdu = fits.table_to_hdu(spec)
    hdu.header.add_history("APQUICK results for "+str(head['FRAMENUM']))
    hdu.header.add_history("Extracted spectra and errors (ADU)")
    hdulist.append(hdu)
    # HDU4: Image
    dtype = np.dtype([('im',(np.float32,frame.im.shape)),('err',(np.float32,frame.err.shape))])
    image = np.zeros(1,dtype=dtype)
    image['im'] = frame.im
    image['err'] = frame.err
    image = Table(image)
    hdu = fits.table_to_hdu(image)
    hdulist.append(hdu)
    hdu.header.add_history("APQUICK results for "+str(head['FRAMENUM']))
    hdu.header.add_history("Collapsed 2D image and noise model (ADU)")
    hdulist.writeto(outfile,overwrite=True)
    hdulist.close() 
    
    # gzip compress?
    if compress:
        ret = subprocess.call(['gzip','-f',outfile])    # compress final catalog
