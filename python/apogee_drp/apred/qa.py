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


class ApQA :

    '''-----------------------------------------------------------------------------------------'''
    ''' Initialize the object '''
    '''-----------------------------------------------------------------------------------------'''

    def __init__(self,tel='apo25m',field='200+45',plate='8100',mjd='57680',apred='t14',noplot=False,verbose=True):

        load=apload.Apload(apred=apred,telescope=tel)
        self.sdss_path=path.Path()
        self.telescope=tel
        self.field=field
        self.plate=plate
        self.mjd=mjd
        self.apred=apred
        self.noplot=noplot
        self.verbose=verbose
        self.instrument='apogee-n'
        if tel=='lco25m': self.instrument='apogee-s'
        self.prefix='ap'
        if tel=='lco25m': self.prefix='as'


        ''' Use telescope, plate, mjd, and apred to load planfile into structure '''
        #!!!! NOT SURE IF THIS WILL WORK

        planfile=load.filename('Plan',reduction=self.apred,plate=self.plate,mjd=self.mjd,field=self.field)
        planstr=plan.loadplan(planfile)


        ''' Establish directories '''
        # Note: Python version of getdir does not exist yet!

;        dirs=getdir(apodir,caldir,spectrodir,vers)
;        apodir=getenv('APOGEE_REDUX')
;        datadir=getenv('APOGEE_DATA')+'/' 
;        spectrodir=apodir+'/'+apred+'/'
;        caldir=spectrodir+'cal/'
;        expdir=spectrodir+'/exposures/'+instrume+'/'


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
        if platetype=='cal': x=self.makeCalStruct()


        ''' For normal plates, make plots and html '''
        if platetype=='normal': x=self.makePlotsHtml()



        ''' For single plates, do nothing '''
        if platetype=='single': print("You are shit out of luck.")




    '''-----------------------------------------------------------------------------------------'''
    ''' Make FITS structure for calibration frames'''
    '''-----------------------------------------------------------------------------------------'''

    def makeCalStruct(self):
        nlines=2

        tharline=np.array([[940.,1128.,1130.],[1724.,623.,1778.]])
        uneline=np.array([[603.,1213.,1116.],[1763.,605.,1893.]])

        if self.instrument=='apogee-s':
            tharline=np.array([[944.,1112.,1102.],[1726.,608.,1745.]])
            uneline=np.array([[607.,1229.,1088.],[1765.,620.,1860.]])

        fibers=np.array([10,80,150,220,290])
        nfibers=len(fibers)
        nchips=3

        dt=np.dtype([('name',np.str,100),
                     ('mjd',np.float64),
                     ('jd',np.float64),
                     ('nframes',int),
                     ('nread',int),
                     ('exptime',int),
                     ('qrtz',int),
                     ('une',int),
                     ('thar',int),
                     ('flux',np.empty([300,nchips],dtype=np.float64)),
                     ('gauss',np.empty([4,nfibers,nchips,nlines],dtype=np.float64)),
                     ('wave',np.empty([nfibers,nchips,nlines],dtype=np.float64)),
                     ('fibers',np.empty(nfibers,dtype=int)),
                     ('lines',np.empty([nchips,nlines],dtype=np.float64)) ])

        struct=np.zeros(n_exposures,dtype=dt)

        for i in range(n_exposures):
            a=load.filename('1D',num=planstr['APEXP']['name'][i])

        etc, etc, etc


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


