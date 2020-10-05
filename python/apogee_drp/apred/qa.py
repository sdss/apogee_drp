import glob
import os
import sys
import subprocess
import math
import numpy as np
from astropy.io import fits, ascii

import yanny as yanny
from apogee_drp.plan import plan
from apogee.utils import apload
from sdss_access.path import path
import pdb

# put pdb.set_trace() wherever you want stop

sdss_path = path.Path()

#sdss_path.full('ap2D',apred=self.apred,telescope=self.telescope,instrument=self.instrument,
#                        plate=self.plate,mjd=self.mjd,prefix=self.prefix,num=0,chip='a')

#--------------------------------------------------------------------------------------------------
# APQA
#
#  call routines to make "QA" plots and web pages for a plate/MJD
#  for calibration frames, measures some features and makes a apQAcal file
#    with some summary information about the calibration data
#--------------------------------------------------------------------------------------------------


# Plugmap for plate 8100 mjd 57680
# /uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/mapper/apo/57679/plPlugMapM-8100-57679-01.par
# Planfile for plate 8100 mjd 57680
# https://data.sdss.org/sas/sdss5/mwm/apogee/spectro/redux/t14/visit/apo25m/200+45/8100/57680/apPlan-8100-57680.par

'''-----------------------------------------------------------------------------------------'''
''' Wrapper for running QA subprocedures '''
'''-----------------------------------------------------------------------------------------'''

def apqa(tel='apo25m',field='200+45',plate='8100',mjd='57680',apred='t14',noplot=False,verbose=True):

    ''' Use telescope, plate, mjd, and apred to load planfile into structure '''

#    load=apload.Apload(apred=apred,telescope=tel)
    planfile=load.filename('Plan',reduction=apred,plate=plate,mjd=mjd,field=field)
#    planstr=plan.loadplan(planfile)
    planstr=yanny.yanny(planfile,np=True)

#    self.sdss_path=path.Path()
#    if tel=='lco25m': self.instrument='apogee-s'
#    self.prefix='ap'
#    if tel=='lco25m': self.prefix='as'


    ''' Establish directories '''
    # Note: Python version of getdir does not exist yet!

#;        dirs=getdir(apodir,caldir,spectrodir,vers)
#;        apodir=getenv('APOGEE_REDUX')
#;        datadir=getenv('APOGEE_DATA')+'/' 
#;        spectrodir=apodir+'/'+apred+'/'
#;        caldir=spectrodir+'cal/'
#;        expdir=spectrodir+'/exposures/'+instrume+'/'

    ''' Find where flavor = object '''
    objs=np.where(planstr['APEXP']['flavor']=='object')
    nobjs=len(objs[0])
    if nobjs<1: print("You're hosed. Give up hope.")

    ''' Get array of object exposures '''
    ims=planstr['APEXP']['NAME'][objs]
    n_exposures=len(planstr['APEXP'])

    ''' Check for tags for fixing fiberid in plugmap files and remove quotes '''
    if planstr.has_key('fixfiberid') is True: fixfiberid=planstr['fixfiberid'].replace("'",'')
    if planstr.has_key('badfiberid') is True: badfiberid=planstr['badfiberid'].replace("'",'')
    if planstr.has_key('survey') is True: survey=planstr['survey'].replace("'",'')

    ''' Get platetype, plugmap, character MJD '''
    platetype=planstr['platetype'].replace("'",'')
    plugmap=planstr['plugmap'].replace("'",'')
    cmjd=load.cmjd(ims[0])

    ''' For calibration plates, measure lamp brightesses and/or line widths, etc. and write to FITS file '''
    if platetype=='cal': x=makeCalStruct(planstr)

    ''' For normal plates, make plots and html '''
    if platetype=='normal': x=makePlotsHtml()

    ''' For single plates, do nothing '''
    if platetype=='single': print("You are shit out of luck.")



'''-----------------------------------------------------------------------------------------'''
''' Make FITS structure for calibration frames'''
'''-----------------------------------------------------------------------------------------'''

def makeCalStruct(planstr=None):
    nlines=2

    tharline=np.array([[940.,1128.,1130.],[1724.,623.,1778.]])
    uneline=np.array([[603.,1213.,1116.],[1763.,605.,1893.]])

    if planstr['instrument']=='apogee-s':
        tharline=np.array([[944.,1112.,1102.],[1726.,608.,1745.]])
        uneline=np.array([[607.,1229.,1088.],[1765.,620.,1860.]])

    fibers=np.array([10,80,150,220,290])
    nfibers=len(fibers)
    nchips=3

    struct_name=np.full(n_exposures,' ')
    struct_mjd=np.full(n_exposures,' ')
    struct_jd=np.zeroes(n_exposures,dtype=float)
    struct_nframes=np.zeros(n_exposures,dtype=int)
    struct_nread=np.zeros(n_exposures,dtype=int)
    struct_exptime=np.zeros(n_exposures,dtype=float)
    struct_qrtz=np.zeros(n_exposures,dtype=int)
    struct_une=np.zeros(n_exposures,dtype=int)
    struct_thar=np.zeros(n_exposures,dtype=int)
    struct_flux=np.zeros((n_exposures,300,nchips),dtype=float)
    struct_gauss=np.zeros((n_exposures,4,nfibers,nchips,nlines),dtype=float)
    struct_wave=np.zeros((n_exposures,nfibers,nchips,nlines),dtype=float)
    struct_fibers=fibers
    struct_lines=np.zeros((n_exposures,nchips,nlines),dtype=float)

#    dt=np.dtype([('name',np.int32),
#                 ('mjd',np.float64),
#                 ('jd',np.float64),
#                 ('nframes',np.int32),
#                 ('nread',np.int32),
#                 ('exptime',np.int32),
#                 ('qrtz',np.int32),
#                 ('une',np.int32),
#                 ('thar',np.int32),
#                 ('flux',np.empty([300,nchips]),np.float64),
#                 ('gauss',np.empty([4,nfibers,nchips,nlines]),np.float64),
#                 ('wave',np.empty([nfibers,nchips,nlines]),np.float64),
#                 ('fibers',np.empty(nfibers),np.int32),
#                 ('lines',np.empty([nchips,nlines]),np.float64)])

    for i in range(n_exposures):
        a=load.filename('1D',num=planstr['APEXP']['name'][i])
#;        if size(a,/type) eq 8 then begin
#;            if ~tag_exist(a,'flux',index=fluxid) then junk =tag_exist(a,'data',index=fluxid)
        struct_name[i]=planstr['APEXP']['name'][i]
        struct_mjd[i]=planstr['mjd']
        struct_jd[i]=sxpar(a[0].hdr,'JD-MID')
        struct_nframes[i]=sxpar(a[0].hdr,'NFRAMES')
        struct_nread[i]=sxpar(a[0].hdr,'NREAD')
        struct_exptime[i]=sxpar(a[0].hdr,'EXPTIME')
        struct_qrtz[i]=sxpar(a[0].hdr,'LAMPQRTZ')
        struct_thar[i]=sxpar(a[0].hdr,'LAMPTHAR')
        struct_une[i]=sxpar(a[0].hdr,'LAMPUNE')

        # quartz exposures
        if struct_qrtz[i]==1: struct_flux[i]=np.median(a['fluxid'],axis=1)

        # arc lamp exposures
        if (struct_thar[i]==1) | (struct_une[i]==1):
            line=tharline
            if struct_thar[i]!=1: line=uneline

            struct_lines[i]=line

            sz=type(line)
            nlines=1
            if line.shape[0]!=1: nlines=line.shape[1]

            for iline in range(nlines):
                for ichip in range(nchips):
                    print('calling appeakfit')
#;                    APPEAKFIT,a[ichip],linestr,fibers=fibers,nsigthresh=10
                    for ifiber in range(nfibers):
                        fibers=fibers[ifiber]
                        j=np.where(linestr['fiber']==fiber)
                        nj=len(j)
                        if nj>0:
                            junk=np.min(np.absolute(linestr['gaussx'][j]-line[ichip,iline]),jline)
                            # NOTE: where does jline come from???
                            struct_gauss[i][*,ifiber,ichip,iline] = linestr['gpar'][j][jline]
#;                            sz=size(a[ichip].wcoef,/dim)
#;                            if sz[0] eq 2 then str[i].wave[ifiber,ichip,iline] = pix2wave(linestr[j[jline]].gaussx,a[ichip].wcoef[fiber,*])
#;                            str[i].flux[fiber,ichip] = linestr[j[jline]].sumflux

#;        outfile=APOGEE_FILENAME('QAcal',mjd=planstr.mjd)
        col1 =  fits.Column(name='name',array=struct_name)
        col2 =  fits.Column(name='mjd',array=struct_mjd)
        col3 =  fits.Column(name='jd',array=struct_jd)
        col4 =  fits.Column(name='nframes',array=struct_nframes)
        col5 =  fits.Column(name='nread',array=struct_nread)
        col6 =  fits.Column(name='exptime',array=struct_exptime)
        col7 =  fits.Column(name='qrtz',array=struct_qrtz)
        col8 =  fits.Column(name='une',array=struct_une)
        col9 =  fits.Column(name='thar',array=struct_thar)
        col10 = fits.Column(name='flux',array=struct_flux)
        col11 = fits.Column(name='gauss',array=struct_gauss)
        col12 = fits.Column(name='wave',array=struct_wave)
        col13 = fits.Column(name='fibers',array=struct_fibers)
        col14 = fits.Column(name='lines',array=struct_lines)
        coldefs = fits.ColDefs([col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14])
        hdu = fits.BinTableHDU.from_columns(coldefs)
        hdul = fits.HDUList([hdu])
        hdul.writeto(outfile)



'''-----------------------------------------------------------------------------------------'''
''' Make plots and html for normal plates '''
'''-----------------------------------------------------------------------------------------'''

def makePlotsHtml(self):
    # Note: Python versions of plotmag, plotflux, mkhtmlplate, and apogee_filename do not exist yet!
;        x=plotmag(ims,self.plateid,clobber=True,mapname=plugmap,noplot=True,fixfiberid=fixfiberid,
;                  badfiberid=badfiberid,survey=survey,plugmap=plugmap)

;        x=plotmag(0,self.plateid,cmjd=cmjd,clobber=True,mapname=plugmap,noplot=self.noplot,fixfiberid=fixfiberid,
;                  badfiberid=badfiberid,survey=survey)

;        x=plotflux(planfile)

    fluxid=planstr['fluxid'].replace("'",'')
;        x=mkhtmlplate(plateid=self.plateid,mjd=cmjd,fluxid=fluxid)

    platefile=load.filename('PlateSum',reduction=self.apred,plate=self.plate,mjd=self.mjd,field=self.field)
;        sntab,tabs=platefile,outfile=platefile+'.dat'


