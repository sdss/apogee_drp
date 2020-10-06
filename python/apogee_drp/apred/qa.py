import glob
import os
import sys
import subprocess
import math
import numpy as np
from pathlib import Path
from astropy.io import fits, ascii

from apogee_drp.plan import plan
from apogee_drp.utils import apload,yanny,plugmap,getdata
from apogee_drp.utils import load as aploadplugmap
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
''' Wrapper for running QA subprocedures                                                    '''
'''-----------------------------------------------------------------------------------------'''

def apqa(tel='apo25m',field='200+45',plate='8100',mjd='57680',apred='t14',noplot=False,verbose=True):

    #------------------------------------------------------------------
    # Use telescope, plate, mjd, and apred to load planfile into structure
    load=apload.ApLoad(apred=apred,telescope=tel)
    planfile=load.filename('Plan',plate=int(plate),mjd=mjd,field=field)
    planstr=yanny.yanny(planfile,np=True)

    ''' Establish directories '''
    # Note: Python version of getdir does not exist yet!

#;        dirs=getdir(apodir,caldir,spectrodir,vers)
#;        apodir=getenv('APOGEE_REDUX')
#;        datadir=getenv('APOGEE_DATA')+'/' 
#;        spectrodir=apodir+'/'+apred+'/'
#;        caldir=spectrodir+'cal/'
#;        expdir=spectrodir+'/exposures/'+instrume+'/'

    #------------------------------------------------------------------
    # Find where flavor = object 
    objs=np.where(planstr['APEXP']['flavor'].astype(str)=='object')
    nobjs=len(objs[0])
    if nobjs<1: print("You're hosed. Give up hope.")

    #------------------------------------------------------------------
    # Get array of object exposures 
    ims=planstr['APEXP']['NAME'][objs].astype(str)
    n_exposures=len(planstr['APEXP'])

    #------------------------------------------------------------------
    # Check for tags for fixing fiberid in plugmap files and remove quotes 
    if planstr.get('fixfiberid') is not None: fixfiberid=planstr['fixfiberid'].replace("'",'')
    if planstr.get('badfiberid') is not None: badfiberid=planstr['badfiberid'].replace("'",'')
    if planstr.get('survey') is not None: survey=planstr['survey'].replace("'",'')

    #------------------------------------------------------------------
    # Get platetype, plugmap, etc
    platetype=planstr['platetype'].replace("'",'')
    plugmap=planstr['plugmap'].replace("'",'')
    fluxid=planstr['fluxid'].replace("'",'')

    #------------------------------------------------------------------
    # For calibration plates, measure lamp brightesses and/or line widths, etc. and write to FITS file 
    if platetype=='cal': x=makeCalFits(planstr,n_exposures)

    #------------------------------------------------------------------
    # For darks and flats, get mean and stdev of column-medianed quadrants
    if platetype=='dark': x=makeDarkFits(planstr,n_exposures)

    #------------------------------------------------------------------
    # For normal plates, make plots and html (calling the re-write of plotmag.pro
    if platetype=='normal': 
        x=makePlotsHtml(telescope=tel,ims=ims,plateid=plate,clobber=True,mapname=plugmap,
                        noplot=True,fixfiberid=fixfiberid,badfiberid=badfiberid,
                        survey=survey,mapper_data=mapper_data,field=field)

        x=makePlotsHtml(telescope=tel,ims=None,plateid=plate,mjd=mjd,clobber=True,
                        mapname=plugmap,noplot=noplot,fixfiberid=fixfiberid,
                        badfiberid=badfiberid,survey=survey,mapper_data=mapper_data,
                        field=field)

        x=plotFlux(planfile)

        x=makeHTMLplate(plateid=plate,mjd=mjd,fluxid=fluxid)

#;        platefile=APOGEE_FILENAME('PlateSum',plate=plateid,mjd=cmjd)

#;        sntab,tabs=platefile,outfile=platefile+'.dat'

    #------------------------------------------------------------------
    # For single (ASDAF and NMSU 1m) plates, do nothing 
    if platetype=='single': print("You are shit out of luck.")



'''-----------------------------------------------------------------------------------------'''
''' Make FITS structure for calibration frames (lamp brightness, line widths, etc.)         '''
'''-----------------------------------------------------------------------------------------'''

def makeCalFits(planstr=None,n_exposures=None):
    nlines=2

    tharline=np.array([[940.,1128.,1130.],[1724.,623.,1778.]])
    uneline=np.array([[603.,1213.,1116.],[1763.,605.,1893.]])

    if planstr['instrument']=='apogee-s':
        tharline=np.array([[944.,1112.,1102.],[1726.,608.,1745.]])
        uneline=np.array([[607.,1229.,1088.],[1765.,620.,1860.]])

    fibers=np.array([10,80,150,220,290])
    nfibers=len(fibers)
    nchips=3

    #------------------------------------------------------------------
    # Make output structure
    dt=np.dtype([('name',np.str,30),
                 ('mjd',np.str,30),
                 ('jd',np.float64),
                 ('nframes',np.int32),
                 ('nread',np.int32),
                 ('exptime',np.float64),
                 ('qrtz',np.int32),
                 ('une',np.int32),
                 ('thar',np.int32),
                 ('flux',np.float64,(300,nchips)),
                 ('gauss',np.float64,(4,nfibers,nchips,nlines)),
                 ('wave',np.float64,(nfibers,nchips,nlines)),
                 ('fibers',np.float64,(nfibers)),
                 ('lines',np.float64,(nchips,nlines))])

    struct=np.zeros(n_exposures,dtype=dt)

    #------------------------------------------------------------------
    # Loop over exposures to fill structure
    for i in range(n_exposures):
        oneD=load.ap1d(num=planstr['APEXP']['name'][i],hdu=hdr)
        if type(oneD)==dict:
            keylist=list(oneD.keys())
            if oneD.get('flux') is None:
                fluxid=-1
            else: 
                fluxid=np.where(keylist=='data')

            struct['name'][i]=planstr['APEXP']['name'][i].replace("'",'')
            struct['mjd'][i]=planstr['mjd'].replace("'",'')
            struct['jd'][i]==hdr['JD-MID']
            struct['nframes'][i]==hdr['NFRAMES']
            struct['nread'][i]==hdr['NREAD']
            struct['exptime'][i]==hdr['EXPTIME']
            struct['qrtz'][i]==hdr['LAMPQRTZ']
            struct['thar'][i]==hdr['LAMPTHAR']
            struct['une'][i]==hdr['LAMPUNE']

        #------------------------------------------------------------------
        # quartz exposures
        if struct['qrtz'][i]==1: struct['flux'][i]=np.median(a['data'],axis=1)

        #------------------------------------------------------------------
        # arc lamp exposures
        if struct['thar'][i]==1 or struct['une'][i]==1:
            line=tharline
            if struct['thar'][i]!=1: line=uneline

            struct['lines'][i]=line

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
                            struct['gauss'][*,ifiber,ichip,iline][i] = linestr['gpar'][j][jline]
#;                            sz=size(a[ichip].wcoef,/dim)
#;                            if sz[0] eq 2 then str[i].wave[ifiber,ichip,iline] = pix2wave(linestr[j[jline]].gaussx,a[ichip].wcoef[fiber,*])
#;                            str[i].flux[fiber,ichip] = linestr[j[jline]].sumflux

#;        outfile=APOGEE_FILENAME('QAcal',mjd=planstr.mjd)
#         write fits file here


'''-----------------------------------------------------------------------------------------'''
''' Make FITS structure for dark frames (get mean and stddev of column-medianed quadrants)  '''
'''-----------------------------------------------------------------------------------------'''

def makeDarkFits(planstr=None,n_exposures=None):
    nchips=3
    nquad=4

    #------------------------------------------------------------------
    # Make output structure
    dt=np.dtype([('name',np.str,30),
                 ('mjd',np.str,30),
                 ('jd',np.float64),
                 ('nframes',np.int32),
                 ('nread',np.int32),
                 ('exptime',np.float64),
                 ('qrtz',np.int32),
                 ('une',np.int32),
                 ('thar',np.int32),
                 ('exptype',np.str,(300,nchips)),
                 ('mean',np.float64,(nchips,nquad)),
                 ('sig',np.float64,(nchips,nquad))])

    struct=np.zeros(n_exposures,dtype=dt)

    #------------------------------------------------------------------
    # Loop over exposures to fill structure
    for i in range(n_exposures):
        twoD=load.ap2d(num=planstr['APEXP']['name'][i],hdu=hdr)
        if type(oneD)==dict:
            keylist=list(oneD.keys())
            if oneD.__contains__('flux') is False:
                fluxid=-1
            else:
                fluxid=np.where(keylist=='data')

            struct['name'][i]=planstr['APEXP']['name'][i].replace("'",'')
            struct['mjd'][i]=planstr['mjd'].replace("'",'')
            struct['jd'][i]==hdr['JD-MID']
            struct['nframes'][i]==hdr['NFRAMES']
            struct['nread'][i]==hdr['NREAD']
            struct['exptime'][i]==hdr['EXPTIME']
            struct['qrtz'][i]==hdr['LAMPQRTZ']
            struct['thar'][i]==hdr['LAMPTHAR']
            struct['une'][i]==hdr['LAMPUNE']

            for ichip in range(nchips):
                i1=10
                i2=500
                for iquad in range(quad):
                    sm=np.median(a['flux'][ichip][i1:i2,10:2000],axis=1)
                    struct['mean'][i,ichip,iquad]=np.mean(sm)
                    struct['sig'][i,ichip,iquad]=np.std(sm)
                    i1=i1+512
                    i2=i2+512

#;    outfile=FILE_DIRNAME(planfile)+'/apQAdarkflat-'+string(planstr.mjd,format='(i5.5)')+'.fits'
#         write fits file here



'''-----------------------------------------------------------------------------------------'''
''' Plotmag translation '''
'''-----------------------------------------------------------------------------------------'''

def makePlotsHtml(telescope=None,ims=None,plate=None,cmjd=None,flat=None,clobber=True,starfiber=None,
                  starnames=None,noplot=None,mapname=None,starmag=None,onem=None,fixfiberid=None,
                  badfiberid=None,survey=None,mapper_data=None,field=None):

    # I'm not sure this is needed
    if cmjd is None: cmjd=load.cmjd(ims[0])

    if type(plate)==int: plate=str(plate)

    # Set up directory names
#;    dirs=GETDIR(apodir,caldir,spectrodir,vers,apred_vers=apred_vers)
#;    reddir=spectrodir+'red/'+cmjd
#;    telescope=dirs.telescope
#;    platedir=APOGEE_FILENAME('Plate',plate=plate,mjd=cmjd,chip='a',/dir)
#;    outdir=platedir+'/plots/'
#;    if file_test(outdir,/directory) eq 0 then file_mkdir,outdir
#;    htmldir=platedir+'/html/'
#;    if file_test(htmldir,/directory) eq 0 then file_mkdir,htmldir

    #------------------------------------------------------------------
    # Open the output HTML file for this plate
    if flat is not None: gfile=plate+'-'+mjd+'flat'
    if onem is not None: gfile=mjd+'-'+starnames[0] 
    if flat is None and onem is None: gfile=plate+'-'+mjd
    platefile=gfile

    if ims is None: 
        gfile='sum'+gfile
    else:
        n_exposures=len(ims)

    html=open(htmldir+gfile+'.html','w')
    htmlsum=open(htmldir+gfile+'sum.html','w')

    html.write('<HTML><BODY>\n')
    htmlsum.write('<HTML><BODY>\n')
    if starfiber is None:
        txt1='Left plots: red are targets, blue are telluric. Observed mags are calculated '
        txt2='from median value of green chip. Zeropoint gives overall throughput: bigger number is more throughput.'
        html.write(txt1+txt2+'\n')

        txt1='<br>First spatial plots: circles are objects, squares are tellurics, crosses are sky fibers. '
        txt2='Colors give deviation of observed mag from expected 2MASS mag using the median zeropoint; red is brighter'
        html.write(txt1+txt2+'\n')

        txt1='<br>Second spatial plots: circles are sky fibers. '
        txt2='Colors give sky line brightness relative to plate median sky line brightness'
        html.write(txt1+txt2+'\n')

    if starfiber is None:
        html.write('<TABLE BORDER=2>\n')
        html.write('<TR><TD>Frame<TD>Nreads<TD>Zeropoints<TD>Mag plots\n')
        html.write('<TD>Spatial mag deviation\n')
        html.write('<TD>Spatial sky 16325A emission deviations (filled: sky, open: star)\n')
        html.write('<TD>Spatial sky continuum emission \n')
        html.write('<TD>Spatial sky telluric CO2 absorption deviations (filled: H &lt 10) \n')
    else:
        html.write('<TABLE BORDER=2>\n')
        html.write('<TR><TD>Frame<TD>Fiber<TD>Star\n')

    htmlsum.write('<TABLE BORDER=2>\n')

    txt1='<TR bgcolor=lightgreen><TD>Frame<TD>Plate<TD>Cart<TD>sec z<TD>HA<TD>DESIGN HA<TD>seeing<TD>FWHM<TD>GDRMS'
    txt2='<TD>Nreads<TD>Dither<TD>Zero<TD>Zerorms<TD>Zeronorm<TD>sky continuum<TD>S/N<TD>S/N(c)<TD>unplugged<TD>faint'
    htmlsum.write(txt1+txt2+'\n')

    #------------------------------------------------------------------
    # Get the fiber association for this plate
    if ims is None: tot=load.apPlate(plate=int(plate),mjd=mjd) 
    if ims is not None: tot=load.ap1D('1D',mjd=mjd,num=ims[0])

    if type(tot)!=dict:
        html.write('<FONT COLOR=red> PROBLEM/FAILURE WITH: '+str(ims[0])+'\n')
        htmlsum.write('<FONT COLOR=red> PROBLEM/FAILURE WITH: '+str(ims[0])+'\n')
        html.close()
        htmlsum.close()
        print('Error in makePlotsHtml!!!')

    if mapname is not None:
        if mapname[0]=='header':
#;            plugid=sxpar(tot[0].hdr,'NAME') 
        else:
            plugid=mapname[0]
    else:
#;        plugid=sxpar(tot[0].hdr,'NAME')

    if onem is True:
        telescope='apo1m'
        reduction_id=starnames[0]
        platedata=getdata(cplate,mjd,plugid=plugid,obj1m=starnames[0],starfiber=starfiber,fixfiberid=fixfiberid) 
    endif else begin
        platedata=getdata(cplate,mjd,plugid=plugid,fixfiberid=fixfiberid,badfiberid=badfiberid,mapper_data=mapper_data) 
    endelse

    gd=np.where(platedata['fiberdata']['fiberid']>0)
    fiber=platedata['fiberdata'][gd]
    nfiber=len(fiber)
    rows=300-fiber['fiberid']
    guide=platedata['guidedata']
#;    ADD_TAG,fiber,'sn',fltarr(n_elements(ims),3),fiber
#;    ADD_TAG,fiber,'obsmag',fltarr(n_elements(ims),3),fiber

    unplugged=np.where(fiber['fiberid']<0)
    nunplugged=len(unplugged[0])
    if flat is not None:
        fiber['hmag']=12
        fiber['object']='FLAT'

    fibertelluric=np.where((fiber['objtype']=='SPECTROPHOTO_STD') | (fiber['objtype']=='HOT_STD'))
    ntelluric=len(fibertelluric[0])
    telluric=rows[fibertelluric]

    fiberobj=np.where((fiber['objtype']=='STAR_BHB') | (fiber['objtype']=='STAR') | (fiber['objtype']=='EXTOBJ'))
    nobj=len(fiberobj[0])
    obj=rows[fiberobj]

    fibersky=np.where(fiber['objtype']=='SKY')
    nsky=len(fibersky[0])
    sky=rows[fibersky]

    #------------------------------------------------------------------
    # Define skylines structure which we will use to get crude sky levels in lines
    dt=np.dtype([('w1',np.float64),
                 ('w2',np.float64),
                 ('c1',np.float64),
                 ('c2',np.float64),
                 ('c3',np.float64),
                 ('c4',np.float64),
                 ('flux',np.float64,(nfiber)),
                 ('type',np.int32)])

    skylines=np.zeros(2,dtype=dt)

    skylines['w1']=16230.0,15990.0
    skylines['w2']=16240.0,16028.0
    skylines['c1']=16215.0,15980.0
    skylines['c2']=16225.0,15990.0
    skylines['c3']=16245.0,0.0
    skylines['c4']=16255.0,0.0
    skylines['type']=1,0

    #------------------------------------------------------------------
    # Loop through all the images for this plate, and make the plots.
    # Load up and save information for this plate in a FITS table.

    allsky=np.zeros(n_exposures,3),dtype=np.float64)
    allzero=np.zeros(n_exposures,3),dtype=np.float64)
    allzerorms=np.zeros(n_exposures,3),dtype=np.float64)
#;    ra=sxpar(tot[0].hdr,'RADEG')
#;    dec=sxpar(tot[0].hdr,'DECDEG')
#;    mjd=0L
#;    READS,cmjd,mjd

    #------------------------------------------------------------------
    # Get moon information for this observation
#:    MOONPOS,2400000+mjd,ramoon,decmoon
#;    GCIRC,2,ra,dec,ramoon,decmoon,moondist
#;    moondist/=3600.
#;    MPHASE,2400000+mjd,moonphase

    #------------------------------------------------------------------
    # Get guider information
    if onem is None:
#;        gcam=get_gcam(cmjd)
    mjd0=99999
    mjd1=0.

    #------------------------------------------------------------------ 
    # FITS table structure
    dt=np.dtype([('telescope',np.str,6),
                 ('plate',np.str,6),
                 ('nreads',np.int32),
                 ('dateobs',np.str,30),
                 ('secz',np.float64),
                 ('ha',np.float64),
                 ('design_ha',np.float64,3),
                 ('seeing',np.float64),
                 ('fwhm',np.float64),
                 ('gdrms',np.float64),
                 ('cart',np.int32),
                 ('plugid',np.int32),
                 ('dither',np.float64),
                 ('mjd',np.int32),
                 ('im',np.int32),
                 ('zero',np.float64),
                 ('zerorms',np.float64),
                 ('zeronorm',np.float64),
                 ('sky',np.float64,3),
                 ('sn',np.float64,3),
                 ('snc',np.float64,3),
                 ('altsn',np.float64,3),
                 ('nsn',np.int32),
                 ('snratio',np.float64),
                 ('moondist',np.float64),
                 ('moonphase',np.float64),
                 ('tellfit',np.float64,(6,3))])
    
    platetab=np.zeros(n_exposures,dtype=dt)

    platetab['plate']=plate
    platetab['telescope']=-99.0
    platetab['ha']=-99.0
    platetab['design_ha']=-99.0
    platetab['plugid']=plugid
    platetab['mjd']=mjd
    platetab['moondist']=moondist
    platetab['moonphase']=moonphase

    for i in range(n_exposures):
        #------------------------------------------------------------------
        # Read image

        if ims is None:
#;            file=FILE_BASENAME(APOGEE_FILENAME('Plate',plate=plate,mjd=mjd,/nochip,/base),'.fits')
            tmp=load.filename('Plate',plate=int(plate),mjd=mjd,field=field)
            path = Path(__tmp__).parent.absolute()

        else:
#;            file=FILE_BASENAME(APOGEE_FILENAME('1D',num=ims[i],/nochip,/base),'.fits')

        if (clobber is True) or (len(glob.glob(outdir+file+'.tab'))!=0):
            if ims is not None:
#;                d=APREAD('Plate',mjd=cmjd,plate=plate)
            else:
#;                d=APREAD('1D',mjd=cmjd,num=ims[i])

            if type(d)!=dict:
#;                GOTO,badim

            if ims if None:
#;                cframe=APREAD('Plate',mjd=cmjd,plate=plate)
            else:
#;                cframefile=APOGEE_FILENAME('Cframe',plate=plate,mjd=cmjd,num=ims[i],chip='c')

                if len(glob.glob(cframefile[0]))!=0 then begin
 #;                   cframe=APREAD('Cframe',plate=plate,mjd=cmjd,num=ims[i])
                else: 
                    cframe=0

            obs=np.zeros((nfiber,3),dtype=np.float64)
            sn=np.zeros((nfiber,3),dtype=np.float64)
            snc=np.zeros((nfiber,3),dtype=np.float64)
            snt=np.zeros((nfiber,3),dtype=np.float64)

            objhtml=open(htmldir+file+'.html','w')
            objhtml.write('<HTML>\n')
            objhtml.write('<HEAD><script type=text/javascript src=../../../../html/sorttable.js></script></head>\n')
            objhtml.write('<BODY>\n')

            if ims is not None:
                objhtml.write('<H2>'+file+'</H2>\n')
#;                platefile=APOGEE_FILENAME('Plate',plate=cplate,mjd=cmjd,chip=['a','b','c'],/file)
                for ichip in range(3):
                    objhtml.write('<A HREF=../'+platefile[ichip]+'>'+platefile[ichip]+'</A>\n')
            else:
                objhtml.write('<H2>'+str(ims[i])+'</H2>\n')
                if noplot is not None:
                    objhtml.write('<A HREF=../../../../red/'+mjd+'/html/'+file+'.html> 1D frames </A>\n')
                    objhtml.write('<BR><A HREF=../../../../red/'+mjd+'/html/ap2D-'+str(ims[i])+'.html> 2D frames </A>\n')

            objhtml.write('<TABLE BORDER=2 CLASS=sortable>\n')
            objhtml.write('<TR><TD>Fiber<TD>Star<TD>H mag<TD>Diff<TD>S/N<TD>S/N (cframe)<TD>Target flags\n')

            stars=np.arange(0,300,1)
            stars=stars[::-1]

            #------------------------------------------------------------------
            # For each fiber, get an observed mag from a median value
            for j in range(nfiber):
                for ichip in range(3):
#;                    obs[j,ichip]=np.median(d[ichip].flux[*,rows[j]])
            endfor

            if flat is None:
                for iline in range(len(skylines)):
                    skyline=skylines[iline]
#;                    GETFLUX,d,skyline,rows
                    skylines[iline]=skyline

            #------------------------------------------------------------------
            # Get a "magnitude" for each fiber from a median on each chip.
            # Do a crude sky subtraction, calculate S/N.
            for ichip in range(3):
                if ims is None: medsky=0.
                if ims is not None: medsky=np.median(obs[fibersky,ichip])

#;                if nobj>0: obs[fiberobj,ichip]=np.median(d[ichip].flux[*,obj],axis=1)-medsky

#;                if ntelluric>0: obs[fibertelluric,ichip]=np.median(d[ichip].flux[*,telluric],axis=1)-medsky

                if nobj>0:
                    sn[fiberobj,ichip]=np.median((d[ichip].flux[*,obj]-medsky)/d[ichip].err[*,obj],axis=1)
                    if len(cframe)>1:
                        snc[fiberobj,ichip]=np.median(cframe[ichip].flux[*,obj]/cframe[ichip].err[*,obj],axis=1)

                if ntelluric>0:
#;                    sn[fibertelluric,ichip]=np.median((d[ichip].flux[*,telluric]-medsky)/d[ichip].err[*,telluric],axis=1)
                    if len(cframe)>1:
#;                        snc[fibertelluric,ichip]=np.median(cframe[ichip].flux[*,telluric]/cframe[ichip].err[*,telluric],axis=1)
#;                        medfilt=MEDFILT2D(cframe[ichip].flux[*,telluric],50,dim=1)
                        sz=shape(cframe['flux'][ichip])
                        i1=900*sz[1]/2048
                        i2=1000*sz[1]/2048
                        for itell in range(ntelluric):
#;                            p1=np.std(cframe[ichip].flux[i1:i2,telluric[itell]])
#;                            p2=np.std(cframe[ichip].flux[i1:i2,telluric[itell]]-medfilt[i1:i2,itell])
#;                            snt[fibertelluric[itell],ichip]=p1/p2

                    else:
                        snc[fibertelluric,ichip]=sn[fibertelluric,ichip]
                        medfilt=MEDFILT2D(d[ichip].flux[*,telluric],50,dim=1)
                        sz=shape(d['flux'][ichip])
                        i1=900*sz[1]/2048
                        i2=1000*sz[1]/2048
                        for itell in range(ntelluric):
#;                            p1=np.mean(d[ichip].flux[i1:i2*sz[1]/2048,telluric[itell]])
#;                            p2=np.std(d[ichip].flux[i1:i2,telluric[itell]]-medfilt[i1:i2,itell])
#;                            snt[fibertelluric[itell],ichip]=p1/p2

            #------------------------------------------------------------------
            # Calculate zeropoints from known H band mags.
            # Use a static zeropoint to calculate sky brightness.

#;            nreads=sxpar(d[0].hdr,'NFRAMES')
#;            exptime=sxpar(d[0].hdr,'EXPTIME')
            skyzero=14.75+2.5*np.log10(nreads)
            zero=0
            zerorms=0.
            faint=-1
            nfaint=0
            achievedsn=[0.,0.,0.]
            achievedsnc=[0.,0.,0.]
            altsn=[0.,0.,0.]
            nsn=0

#;            zero=np.median(fiber[[fiberobj,fibertelluric]].hmag+2.5*np.log10(obs[[fiberobj,fibertelluric],1]))
#:            zerorms=ROBUST_SIGMA(fiber[[fiberobj,fibertelluric]].hmag+2.5*alog10(obs[[fiberobj,fibertelluric],1]))
#;            faint=np.where((fiber['hmag'][fiberobj,fibertelluric]+2.5*np.log10(obs[[fiberobj,fibertelluric],1])-zero)<-0.5,nfaint)

            zeronorm=zero-2.5*np.log10(nreads)

            #------------------------------------------------------------------
            # For each star, create the exposure entry on the web page and set up the plot of the spectrum

            cfile=open(outdir+file+'.csh','w')
            jsort=np.sort(fiber['fiberid'])
            for jj in range(len(fiber)):
                j=jsort[jj]
                objhtml.write('<TR>\n')

                color='white'
                if (fiber['objtype'][j]=='SPECTROPHOTO_STD') | (fiber['objtype'][j]=='HOT_STD'): color='cyan'
                if fiber['objtype'][j]=='SKY': color='lightgreen'

#;                visitfile=APOGEE_FILENAME('Visit',plate=cplate,mjd=cmjd,fiber=fiber[j].fiberid,reduction=reduction_id,/base)
#;                visitfile=load.filename('Visit',plate=plate,mjd=mjd,field=field)

                cfib=str(fiber['fiberid'][j]).zfill(3)
                if ims is None:
                    objhtml.write('<TD><A HREF=../'+visitfile+'>'+cfib+'</A>\n')
                else:
                    objhtml.write('<TD>'+cfib+'\n')

                if ims is None:
#;                    objhtml.write('<TD BGCOLOR='+color+'><a href=../plots/'+FILE_BASENAME(visitfile,'.fits')+'.jpg>'+fiber['object'][j]+'</A>\n')
                else:
                    objhtml.write('<TD BGCOLOR='+color+'>'+fiber['object'][j]+'\n')

                rastring=str("%8.5f" % round(fiber['ra'][j],5))
                decstring=str("%8.5f" % round(fiber['dec'][j],5))

                if (fiber['object'][j]!='sky') & (fiber['fiberid'][j]>=0):
                    txt1='<BR><A HREF="http://simbad.decstring.harvard.edu/simbad/sim-basic?'
                    txt2='Ident='+rastring+'+%09'+decstring+'++&submit=SIMBAD+search"> (SIMBAD) </A>'
                    objhtml.write(txt1+txt2+'\n')

                objhtml.write('<TD>'+str("%8.3f" % round(fiber['hmag'][j],3))+'\n')
                objhtml.write('<TD>'+str("%8.2f" % round(fiber['hmag'][j]+2.5*np.log10(obs[j,1])-zero,2))+'\n')
                objhtml.write('<TD>'+str("%8.2f" % round(sn[j,1],2))+'\n)
                objhtml.write('<TD>'+str("%8.2f" % round(snc[j,1],2))+'\n)
#;                objhtml.write('<TD>'+TARGFLAG(fiber[j].target1,fiber[j].target2,fiber[j].target3,survey=survey)+'\n')

                if (ims is None) & (fiber['fiberid'][j]>=0:
#;                    vfile=APOGEE_FILENAME('Visit',plate=cplate,mjd=cmjd,fiber=fiber[j].fiberid,reduction=reduction_id)
                    if len(glob.glob(vfile))!=0:
                        h=fits.getheader(vfile)
                        if type(h)==str:
#;                            objhtml.write('<BR>'+STARFLAG(h['STARFLAG'])+'\n')

                #------------------------------------------------------------------
                # PLOT 1: spectrum 
                # https://data.sdss.org/sas/apogeework/apogee/spectro/redux/current/plates/5583/56257//plots/apPlate-5583-56257-299.jpg

                if (j%300)>-1:
                    if noplot is None:
                        print('PLOTS 1: Spectrum plots will be made here.')
                    else:
                        objhtml.write('<TD>No plots for individual exposures, see plate plots\n')

            objhtml.close()
            cfile.close()

            #------------------------------------------------------------------
            # PLOT 2: 5 panels
            # https://data.sdss.org/sas/apogeework/apogee/spectro/redux/current/plates/5583/56257/plots/ap1D-06950025.gif

            if (flat is None) & (onem is None):
                print('PLOTS 2: 5-panel plot will be made here.')
            else:
#;                achievedsn=np.median([sn[obj,*]],axis=1)

            #------------------------------------------------------------------
            # PLOTS 3-5: spatial residuals, , , 
            # 3: spatial residuals
            # https://data.sdss.org/sas/apogeework/apogee/spectro/redux/current/plates/5583/56257/plots/ap1D-06950025.jpg
            # 4: spatial sky line emission
            # https://data.sdss.org/sas/apogeework/apogee/spectro/redux/current/plates/5583/56257/plots/ap1D-06950025sky.jpg
            # 5: spatial continuum emission
            # https://data.sdss.org/sas/apogeework/apogee/spectro/redux/current/plates/5583/56257/plots/ap1D-06950025skycont.jpg

            if (starfiber is None) & (onem is None):
                print('PLOTS 3: spatial plot of residuals will be made here.\n')
                print('PLOTS 4: spatial plot of sky line emission will be made here.\n')
                print('PLOTS 5: spatial plot of continuum emission will be made here.\n')


        #------------------------------------------------------------------
        # Put all of the info and plots on the plate web page

        medsky=np.zeros(3,dtype=np.float64)
        for ichip in range(3):
            if np.median(obs[fibersky,ichip])>0:
                medsky[ichip]=-2.5*np.log10(np.median(obs[fibersky,ichip]))+skyzero
            else: 
                medsky[ichip]=99.999

        html.write('<TR><TD><A HREF=../html/'+file+'.html>',ims[i],'</A>\n')
        html.write('<TD>'+string(nreads)+'\n')
        html.write('<TD><TABLE BORDER=1><TD><TD>Red<TD>Green<TD>Blue\n')
        html.write('<TR><TD>z<TD><TD>'+str("%5.2f" % round(zero,2))+'\n')
        html.write('<TR><TD>znorm<TD><TD>'+str("%5.2f" % round(zeronorm,2))+'\n')
#;        html.write('<TR><TD>sky'+string(format='("<TD>",f5.1,"<TD>",f5.1,"<TD>",f5.1)',medsky)+'\n')
#;        html.write('<TR><TD>S/N'+string(format='("<TD>",f5.1,"<TD>",f5.1,"<TD>",f5.1)',achievedsn)+'\n')
#;        html.write('<TR><TD>S/N(c)'+string(format='("<TD>",f5.1,"<TD>",f5.1,"<TD>",f5.1)',achievedsnc)+'\n')

        if ntelluric>0:
           html.write('<TR><TD>SN(E/C)<TD<TD>'+str("%5.2f" % round(np.median(snt[telluric,1]/snc[telluric,1]),2)+'\n')
        else: 
            html.write('<TR><TD>SN(E/C)<TD<TD>\n')

        html.write('</TABLE>\n')
        html.write('<TD><IMG SRC=../plots/'+file+'.gif>\n')
        html.write('<TD> <IMG SRC=../plots/'+file+'.jpg>\n')
        html.write('<TD> <IMG SRC=../plots/'+file+'sky.jpg>\n')
        html.write('<TD> <IMG SRC=../plots/'+file+'skycont.jpg>\n')
        html.write('<TD> <IMG SRC=../plots/'+file+'telluric.jpg>\n')

        #------------------------------------------------------------------
        # Get guider info

        if onem is None:
#;            dateobs=sxpar(d[0].hdr,'DATE-OBS')
#;            mjdstart=DATE_CONV(dateobs,'MODIFIED')
#;            mjdend=mjdstart+sxpar(d[0].hdr,'EXPTIME')/86400.
            mjd0=min([mjd0,mjdstart])
            mjd1=max([mjd1,mjdend])
            nj=0
            if type(gcam)==dict:
                jcam=np.where((gcam['mjd']>mjdstart) & (gcam['mjd']<mjdend))
                nj=len(jcam[0])
            if nj>1: 
                fwhm=np.median(gcam['fwhm_median'][jcam]) 
                gdrms=np.median(gcam['gdrms'][jcam])
            else:
                fwhm=-1.
                gdrms=-1.
                print('not halted: no matching mjd range in gcam...')
        else:
            fwhm=-1
            gdrms=-1


        #------------------------------------------------------------------
        # Summary plate web page

        htmlsum.write('<TR><TD><A HREF=../html/'+file+'.html>'+str(ims[i])+'</A>\n')
#;        htmlsum.write('<TD><A HREF=../../../../plates/'+plate+'/'+mjd+'/html/'+plate+'-'+mjd+'.html>',sxpar(d[0].hdr,'PLATEID'),'</A>\n')
#;        htmlsum.write('<TD>',sxpar(d[0].hdr,'CARTID')+'\n')
#;        alt=sxpar(d[0].hdr,'ALT',count=count)
        if count>0:
            secz=1./np.cos((90.-alt)*math.pi/180.)
        else:
#;            secz=sxpar(d[0].hdr,'ARMASS')
#;        seeing=sxpar(d[0].hdr,'SEEING')
#;        ha=sxpar(d[0].hdr,'HA')
        design_ha=platedata['ha']
        dither=-99.
#;        if len(cframe)>1: dither = sxpar(cframe[0].hdr,'DITHSH') 
        htmlsum.write('<TD>'+str("%6.2f" % round(secz,2))+'\n')
        htmlsum.write('<TD>'+str("%6.2f" % round(ha,2))+'\n')
#;        htmlsum.write('<TD>'+string(format='(f6.0,",",f6.0,",",f6.0)',design_ha)+'\n')
        htmlsum.write('<TD>'+str("%6.2f" % round(seeing,2))+'\n')
        htmlsum.write('<TD>'+str("%6.2f" % round(fwhm,2))+'\n')
        htmlsum.write('<TD>'+str("%6.2f" % round(gdrms,2))+'\n')
        htmlsum.write('<TD>'+str(nreads)+'\n')
        if len(cframe)>1:
#;            htmlsum.write('<TD>'+str(format='(f8.2)',sxpar(cframe[0].hdr,'DITHSH'))+'\n')
        endif else begin
            htmlsum.write('<TD>\n)'
        endelse
        htmlsum.write('<TD>',str("%5.2f" % round(zero,2))+'\n')
        htmlsum.write('<TD>',str("%5.2f" % round(zerorms,2))+'\n')
        htmlsum.write('<TD>',str("%5.2f" % round(zeronorm,2))+'\n')
#;        htmlsum.write('<TD>',str(format='("[",f5.2,",",f5.2,",",f5.2,"]")',medsky)+'\n')
#;        htmlsum.write('<TD>',str(format='("[",f5.1,",",f5.1,",",f5.1,"]")',achievedsn)+'\n')
#;        htmlsum.write('<TD>',str(format='("[",f5.1,",",f5.1,",",f5.1,"]")',achievedsnc)+'\n')
        htmlsum.write('<TD>\n')
        for j in range(unplugged): htmlsum.write(str(300-unplugged[j])+'\n')
        htmlsum.write('<TD>\n')
        if faint[0]>0:
            for j in range(nfaint): htmlsum.write(str(fiber['fiberid'][faint][j])+'\n')
#;        allsky[i,*]=medsky
#;        allzero[i,*]=zero
#;        allzerorms[i,*]=zerorms








    html.write('</BODY></HTML>\n')
    htmlsum.write('</TABLE>\n')

    if onem is not None:
        file=mjd+'-'+starnames[0]
        htmlsum.write('<a href=../plots/apVisit-'+apred+'-'+file+'.jpg><IMG src='+'../plots/apVisit-'+apred+'-'+file+'.jpg></A>\n')

    htmlsum.write('</BODY></HTML>\n')

    html.close()
    htmlsum.close()






























