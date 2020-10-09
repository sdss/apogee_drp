import glob
import os
import sys
import subprocess
import math
import numpy as np
from pathlib import Path
from astropy.io import fits, ascii
from astropy.table import Table
from astropy.time import Time

from apogee_drp.utils import plan,apload,yanny,plugmap,platedata # platedata.getdata
from apogee_drp.utils import load as aploadplugmap
from sdss_access.path import path
import pdb

# put pdb.set_trace() wherever you want stop

sdss_path = path.Path()

#sdss_path.full('ap2D',apred=self.apred,telescope=self.telescope,instrument=self.instrument,
#                        plate=self.plate,mjd=self.mjd,prefix=self.prefix,num=0,chip='a')

#------------------------------------------------------------------------------------------------------------------------
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
'''APQA: Wrapper for running QA subprocedures                                               '''
'''-----------------------------------------------------------------------------------------'''

def apqa(field='200+45',plate='8100',mjd='57680',telescope='apo25m',apred='t14',noplot=False,verbose=True):
    #----------------------------------------------------------------------------------------
    # Use telescope, plate, mjd, and apred to load planfile into structure.
    #----------------------------------------------------------------------------------------
    load = apload.ApLoad(apred=apred,telescope=telescope)
    planfile = load.filename('Plan',plate=int(plate),mjd=mjd)
    planstr = plan.load(planfile)

    #----------------------------------------------------------------------------------------
    # Get values from plan file.
    #----------------------------------------------------------------------------------------
    fixfiberid = planstr['fixfiberid'].astype(int)
    badfiberid = planstr['badfiberid'].astype(int)
    survey =     planstr['survey']
    platetype =  planstr['platetype']
    plugmap =    planstr['plugmap']
    fluxid =     planstr['fluxid'].astype(int)
    instrument = planstr['instrument']

    #----------------------------------------------------------------------------------------
    # Establish directories.
    #----------------------------------------------------------------------------------------
    datadir = {'apo25m':os.environ['APOGEE_DATA_N'],'apo1m':os.environ['APOGEE_DATA_N'],
               'lco25m':os.environ['APOGEE_DATA_S']}[telescope]

    apodir =     os.environ('APOGEE_REDUX')
    spectrodir = apodir+'/'+apred+'/'
    caldir =     spectrodir+'cal/'
    expdir =     spectrodir+'/exposures/'+instrument+'/'

    #----------------------------------------------------------------------------------------
    # Get array of object exposures and find out how many are objects.
    #----------------------------------------------------------------------------------------
    objind = np.zeros(len(planstr['APEXP']))
    all_ims = np.empty(len(planstr['APEXP']))

    for i in range(len(planstr['APEXP'])):
        all_ims[i] = planstr['APEXP'][i]['name']
        check = np.where(planstr['APEXP'][i]['flavor']=='object')
        if len(check[0])>0: objind[i] = 1

    obj=np.where(objind==1)
    n_ims = len(obj[0])

    if n_ims<1: 
        print("No object images. You are hosed. Give up hope.")
        ims = None
    else:
        ims = all_ims[obj].astype(int)

    #----------------------------------------------------------------------------------------
    # Get mapper data.
    #----------------------------------------------------------------------------------------
    mapper_data = None
    # NOTE: need to find mapper file path.

#;    dirs=getdir(apodir,datadir=datadir)
#;    mapper_data=dirs.mapperdir

    #----------------------------------------------------------------------------------------
    # For calibration plates, measure lamp brightesses and/or line widths, etc. and write to FITS file.
    #----------------------------------------------------------------------------------------
    if platetype=='cal': x = makeCalFits(ims=all_ims,mjd=mjd,instrument=instrument)

    #----------------------------------------------------------------------------------------
    # For darks and flats, get mean and stdev of column-medianed quadrants.
    #----------------------------------------------------------------------------------------
    if platetype=='dark': x = makeDarkFits(planfile=planfile,ims=all_ims,mjd=mjd)

    #----------------------------------------------------------------------------------------
    # For normal plates, make plots and html.
    #----------------------------------------------------------------------------------------
    if platetype=='normal': 
        x = makePlotsHtml(telescope=telescope,ims=ims,plateid=plate,clobber=True,
                        mapname=plugmap,noplot=True,fixfiberid=fixfiberid,
                        badfiberid=badfiberid,survey=survey,mapper_data=mapper_data,
                        field=field,apred=apred)

        x = makePlotsHtml(telescope=telescope,ims=None,plateid=plate,mjd=mjd,clobber=True,
                        mapname=plugmap,noplot=noplot,fixfiberid=fixfiberid,
                        badfiberid=badfiberid,survey=survey,mapper_data=mapper_data,
                        field=field,apred=apred)

#;        x=plotFlux(planfile)
#;        x=makeHTMLplate(plateid=plate,mjd=mjd,fluxid=fluxid)
        platesumfile = load.filename('PlateSum',plate=int(plate),mjd=mjd,chips=True)
#;        sntab,tabs=platefile,outfile=platefile+'.dat'

    #----------------------------------------------------------------------------------------
    # For ASDAF and NMSU 1m observations, get more values and make plots and html.
    #----------------------------------------------------------------------------------------
    if platetype=='single':
        
        single = [planstr['APEXP'][i]['single'].astype(int) for i in range(n_ims)]
        sname = [planstr['APEXP'][i]['singlename'] for i in range(n_ims)]
        smag = planstr['hmag']

        x = makePlotsHtml(telescope=telescope,onem=True,ims=ims,starnames=sname,starfiber=single,
                        starmag=smag,fixfiberid=fixfiberid,clobber=True,mapname=plugmap,
                        noplot=noplot,badfiberid=badfiberid,survey=survey,apred=apred)



'''-----------------------------------------------------------------------------------------'''
''' MAKECALFITS: Make FITS file for cals (lamp brightness, line widths, etc.)               '''
'''-----------------------------------------------------------------------------------------'''

def makeCalFits(ims=None,mjd=None,instrument=None):
    n_exposures = len(ims)

    nlines = 2
    nchips = 3

    tharline = np.array([[940.,1128.,1130.],[1724.,623.,1778.]])
    uneline =  np.array([[603.,1213.,1116.],[1763.,605.,1893.]])

    if instrument=='apogee-s': tharline = np.array([[944.,1112.,1102.],[1726.,608.,1745.]])
    if instrument=='apogee-s':  uneline = np.array([[607.,1229.,1088.],[1765.,620.,1860.]])

    fibers = np.array([10,80,150,220,290])
    nfibers = len(fibers)

    #----------------------------------------------------------------------------------------
    # Make output structure.
    #----------------------------------------------------------------------------------------
    dt = np.dtype([('NAME',np.str,30),
                   ('MJD',np.str,30),
                   ('JD',np.float64),
                   ('NFRAMES',np.int32),
                   ('NREAD',np.int32),
                   ('EXPTIME',np.float64),
                   ('QRTZ',np.int32),
                   ('UNE',np.int32),
                   ('THAR',np.int32),
                   ('FLUX',np.float64,(300,nchips)),
                   ('GAUSS',np.float64,(4,nfibers,nchips,nlines)),
                   ('WAVE',np.float64,(nfibers,nchips,nlines)),
                   ('FIBERS',np.float64,(nfibers)),
                   ('LINES',np.float64,(nchips,nlines))])

    struct = np.zeros(n_exposures,dtype=dt)

    #----------------------------------------------------------------------------------------
    # Loop over exposures and get 1D images to fill structure.
    # /uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/apogee/spectro/redux/t14/exposures/apogee-n/57680/ap1D-21180073.fits
    #----------------------------------------------------------------------------------------
    for i in range(n_exposures):
        oneDfile=load.filename('1D',plate=int(plate),mjd=mjd,num=ims[i],chips=True)
        oneD = fits.getdata(oneDfile)
        oneDhdr = fits.getheader(oneDfile)

        oneD = load.ap1D(ims[i])
        hdr = oneD['a'][0].header

        if type(oneD)==dict:
            # NOTE: Not sure the below lines are needed. They won't work anyway.
            keylist = list(oneD.keys())
            if oneD.get('FLUX') is None:
                fluxid = -1
            else: 
                fluxid = np.where(keylist=='data')

            struct['NAME'][i] =    ims[i]
            struct['MJD'][i] =     mjd
            struct['JD'][i] =      oneDhdr['JD-MID']
            struct['NFRAMES'][i] = oneDhdr['NFRAMES']
            struct['NREAD'][i] =   oneDhdr['NREAD']
            struct['EXPTIME'][i] = oneDhdr['EXPTIME']
            struct['QRTZ'][i] =    oneDhdr['LAMPQRTZ']
            struct['THAR'][i] =    oneDhdr['LAMPTHAR']
            struct['UNE'][i] =     oneDhdr['LAMPUNE']

        #----------------------------------------------------------------------------------------
        # Quartz exposures.
        #----------------------------------------------------------------------------------------
        # NOTE: this won't work.
        if struct['QRTZ'][i]==1: struct['FLUX'][i] = np.median(oneD[fluxid],axis=0)

        #----------------------------------------------------------------------------------------
        # Arc lamp exposures.
        #----------------------------------------------------------------------------------------
        if struct['THAR'][i]==1 or struct['UNE'][i]==1:
            line=tharline
            if struct['THAR'][i]!=1: line = uneline

            struct['LINES'][i] = line

            sz = type(line)
            nlines = 1
            if line.shape[0]!=1: nlines = line.shape[1]

            for iline in range(nlines):
                for ichip in range(nchips):
                    print("Calling appeakfit... no, not really because it's a long IDL code.")
                    # NOTE: no translation for appeakfit
#;                    APPEAKFIT,a[ichip],linestr,fibers=fibers,nsigthresh=10
                    for ifiber in range(nfibers):
                        fibers = fibers[ifiber]
                        j = np.where(linestr['FIBER']==fiber)
                        nj = len(j)
                        if nj>0:
                            junk = np.min(np.absolute(linestr['GAUSSX'][j]-line[ichip,iline]))
                            jline = np.argmin(np.absolute(linestr['GAUSSX'][j]-line[ichip,iline])
                            struct['GAUSS'][:,ifiber,ichip,iline][i] = linestr['GPAR'][j][jline]
                            sz = a['WCOEF'][ichip].shape
                            if sz[0]==2:
                                print("pix2wave is still an IDL code, you're hosed.")
                                # NOTE: no translation for pix2wave
#;                                struct['WAVE'][i][ifiber,ichip,iline] = pix2wave(linestr['GAUSSX'][j][jline],a['WCOEF'][ichip][fiber,:])
                            struct['FLUX'][i][fiber,ichip] = linestr['SUMFLUX'][j][jline]

    outfile = load.filename('QAcal',plate=int(plate),mjd=mjd) 
    Table(struct).write(outfile)


'''-----------------------------------------------------------------------------------------'''
''' MAKEDARKFITS: Make FITS file for darks (get mean/stddev of column-medianed quadrants)   '''
'''-----------------------------------------------------------------------------------------'''

def makeDarkFits(planfile=None,ims=None,mjd=None):
    n_exposures = len(ims)

    nchips = 3
    nquad = 4

    #----------------------------------------------------------------------------------------
    # Make output structure.
    #----------------------------------------------------------------------------------------
    dt = np.dtype([('NAME',np.str,30),
                   ('MJD',np.str,30),
                   ('JD',np.float64),
                   ('NFRAMES',np.int32),
                   ('NREAD',np.int32),
                   ('EXPTIME',np.float64),
                   ('QRTZ',np.int32),
                   ('UNE',np.int32),
                   ('THAR',np.int32),
                   ('EXPTYPE',np.str,30),
                   ('MEAN',np.float64,(nchips,nquad)),
                   ('SIG',np.float64,(nchips,nquad))])

    struct = np.zeros(n_exposures,dtype=dt)

    #----------------------------------------------------------------------------------------
    # Loop over exposures and get 2D images to fill structure.
    # /uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/apogee/spectro/redux/t14/exposures/apogee-n/57680/ap2D-21180073.fits
    #----------------------------------------------------------------------------------------
    for i in range(n_exposures):
        twoDfile=load.filename('2D',plate=int(plate),mjd=mjd,num=21180073,chips=True)
        twoD = fits.getdata(twoDfile)
        twoDhdr = fits.getheader(twoDfile)

        if type(twoD)==dict:
            struct['NAME'][i] =    ims[i]
            struct['MJD'][i] =     mjd
            struct['JD'][i] =      twoDhdr['JD-MID']
            struct['NFRAMES'][i] = twoDhdr['NFRAMES']
            struct['NREAD'][i] =   twoDhdr['NREAD']
            struct['EXPTIME'][i] = twoDhdr['EXPTIME']
            struct['QRTZ'][i] =    twoDhdr['LAMPQRTZ']
            struct['THAR'][i] =    twoDhdr['LAMPTHAR']
            struct['UNE'][i] =     twoDhdr['LAMPUNE']

            for ichip in range(nchips):
                i1 = 10
                i2 = 500
                for iquad in range(quad):
                    sm = np.median(twoD['FLUX'][ichip][i1:i2,10:2000],axis=0)
                    struct['MEAN'][i,ichip,iquad] = np.mean(sm)
                    struct['SIG'][i,ichip,iquad] = np.std(sm)
                    i1 = i1+512
                    i2 = i2+512

    outfile = os.path.dirname(planfile)+'/apQAdarkflat-'+mjd+'.fits'
    Table(struct).write(outfile)


'''-----------------------------------------------------------------------------------------'''
''' MAKEPLOTSHTML: Plotmag translation                                                      '''
'''-----------------------------------------------------------------------------------------'''

def makePlotsHtml(telescope=None,ims=None,plate=None,cmjd=None,flat=None,clobber=True,starfiber=None,
                  starnames=None,noplot=None,mapname=None,starmag=None,onem=None,fixfiberid=None,
                  badfiberid=None,survey=None,mapper_data=None,field=None,apred=None):

    if ims is not None: n_exposures = len(ims)

    # I'm not sure this is needed
    if cmjd is None: cmjd = load.cmjd(ims[0])

    if type(plate)==int: plate = str(plate)

    #----------------------------------------------------------------------------------------
    # Set up directory names.
    #----------------------------------------------------------------------------------------
    # NOTE: the below commented lines probably aren't needed.

#;    dirs=GETDIR(apodir,caldir,spectrodir,vers,apred_vers=apred_vers)
#;    reddir=spectrodir+'red/'+mjd

    platedir = os.path.dirname(load.filename('Plate',plate=int(plate),mjd=mjd,chips=True))
    outdir = platedir+'/plots/'
    if len(glob.glob(outdir))==0: subprocess.call(['mkdir',outdir])

    htmldir = platedir+'/html/'
    if len(glob.glob(htmldir))==0: subprocess.call(['htmldir',outdir])

    #----------------------------------------------------------------------------------------
    # Open the output HTML file for this plate.
    #----------------------------------------------------------------------------------------
    if flat is not None: gfile = plate+'-'+mjd+'flat'
    if onem is not None: gfile = mjd+'-'+starnames[0] 
    if (flat is None) & (onem is None): gfile = plate+'-'+mjd
    platefile = gfile
    if ims is None: gfile = 'sum'+gfile

    html = open(htmldir+gfile+'.html','w')
    htmlsum = open(htmldir+gfile+'sum.html','w')

    html.write('<HTML><BODY>\n')
    htmlsum.write('<HTML><BODY>\n')
    if starfiber is None:
        txt1 = 'Left plots: red are targets, blue are telluric. Observed mags are calculated '
        txt2 = 'from median value of green chip. Zeropoint gives overall throughput: bigger number is more throughput.'
        html.write(txt1+txt2+'\n')

        txt1 = '<br>First spatial plots: circles are objects, squares are tellurics, crosses are sky fibers. '
        txt2 = 'Colors give deviation of observed mag from expected 2MASS mag using the median zeropoint; red is brighter'
        html.write(txt1+txt2+'\n')

        txt1 = '<br>Second spatial plots: circles are sky fibers. '
        txt2 = 'Colors give sky line brightness relative to plate median sky line brightness'
        html.write(txt1+txt2+'\n')

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

    txt1 = '<TR bgcolor=lightgreen><TD>Frame<TD>Plate<TD>Cart<TD>sec z<TD>HA<TD>DESIGN HA<TD>seeing<TD>FWHM<TD>GDRMS'
    txt2 = '<TD>Nreads<TD>Dither<TD>Zero<TD>Zerorms<TD>Zeronorm<TD>sky continuum<TD>S/N<TD>S/N(c)<TD>unplugged<TD>faint'
    htmlsum.write(txt1+txt2+'\n')

    #----------------------------------------------------------------------------------------
    # Get the fiber association for this plate.
    #----------------------------------------------------------------------------------------
    if ims is None: tot = load.apPlate(int(plate),mjd)
    if ims is not None: tot = load.ap1D(ims[0])
    platehdr = tot['a'][0].header

    if type(tot)!=dict:
        html.write('<FONT COLOR=red> PROBLEM/FAILURE WITH: '+str(ims[0])+'\n')
        htmlsum.write('<FONT COLOR=red> PROBLEM/FAILURE WITH: '+str(ims[0])+'\n')
        html.close()
        htmlsum.close()
        print("Error in makePlotsHtml!!!")

    if mapname is not None:
        if mapname[0]=='header':
            plugid = platehdr['NAME']
        else:
            plugid = mapname[0]
    else:
        plugid = platehdr['NAME']

    # NOTE: not sure if plate should be int for this.
    if onem is None:
        platedata = platedata(plate=int(plate),mjd=mjd,plugid=plugid,fixfiberid=fixfiberid,badfiberid=badfiberid,mapper_data=mapper_data) 
    else: 
        reduction_id = starnames[0]
        platedata = platedata(plate=int(plate),mjd=mjd,plugid=plugid,obj1m=starnames[0],starfiber=starfiber,fixfiberid=fixfiberid) 

    gd = np.where(platedata['FIBERDATA']['FIBERID']>0)
    fiber = platedata['FIBERDATA'][gd]
    nfiber = len(fiber)
    rows = 300-fiber['FIBERID']
    guide = platedata['GUIDEDATA']

    # NOTE: I don't know how to translate this to python.

#;    ADD_TAG,fiber,'sn',fltarr(n_elements(ims),3),fiber
#;    ADD_TAG,fiber,'obsmag',fltarr(n_elements(ims),3),fiber

    unplugged = np.where(fiber['FIBERID']<0)
    nunplugged = len(unplugged[0])
    if flat is not None:
        fiber['HMAG'] = 12
        fiber['OBJECT'] = 'FLAT'

    fibertelluric = np.where((fiber['OBJTYPE']=='SPECTROPHOTO_STD') | (fiber['OBJTYPE']=='HOT_STD'))
    ntelluric = len(fibertelluric[0])
    telluric = rows[fibertelluric]

    fiberobj = np.where((fiber['OBJTYPE']=='STAR_BHB') | (fiber['OBJTYPE']=='STAR') | (fiber['OBJTYPE']=='EXTOBJ'))
    nobj = len(fiberobj[0])
    obj = rows[fiberobj]

    fibersky = np.where(fiber['OBJTYPE']=='SKY')
    nsky = len(fibersky[0])
    sky = rows[fibersky]

    #----------------------------------------------------------------------------------------
    # Define skylines structure which we will use to get crude sky levels in lines.
    #----------------------------------------------------------------------------------------
    dt = np.dtype([('W1',np.float64),
                   ('W2',np.float64),
                   ('C1',np.float64),
                   ('C2',np.float64),
                   ('C3',np.float64),
                   ('C4',np.float64),
                   ('FLUX',np.float64,(nfiber)),
                   ('TYPE',np.int32)])

    skylines = np.zeros(2,dtype=dt)

    skylines['W1']   = 16230.0,15990.0
    skylines['W2']   = 16240.0,16028.0
    skylines['C1']   = 16215.0,15980.0
    skylines['C2']   = 16225.0,15990.0
    skylines['C3']   = 16245.0,0.0
    skylines['C4']   = 16255.0,0.0
    skylines['TYPE'] = 1,0

    #----------------------------------------------------------------------------------------
    # Loop through all the images for this plate, and make the plots.
    # Load up and save information for this plate in a FITS table.
    #----------------------------------------------------------------------------------------
    allsky = np.zeros((n_exposures,3),dtype=np.float64)
    allzero = np.zeros((n_exposures,3),dtype=np.float64)
    allzerorms = np.zeros((n_exposures,3),dtype=np.float64)
    ra = platehdr['RADEG']
    dec = platehdr['DECDEG']

    # NOTE: Not sure if the following lines are needed.

#;    mjd=0L
#;    READS,cmjd,mjd

    #----------------------------------------------------------------------------------------
    # Get moon information for this observation.
    #----------------------------------------------------------------------------------------
#:    MOONPOS,2400000+mjd,ramoon,decmoon
#;    GCIRC,2,ra,dec,ramoon,decmoon,moondist
#;    MPHASE,2400000+mjd,moonphase

    # NOTE: Currently hard-coding values for mjd 57680 until I find a python way.
    ramoon=57.323959
    decmoon=14.920379
    moondist=282147.24012227560
    moondist/=3600.
    moonphase=0.91864227

    #----------------------------------------------------------------------------------------
    # Get guider information.
    #----------------------------------------------------------------------------------------
    if onem is None:
        print("No translation for get_gcam! Give up hope!")
        gcam=None
        # NOTE: get_gcam has not been translated
#;        gcam=get_gcam(cmjd)
    mjd0 = 99999
    mjd1 = 0.

    #---------------------------------------------------------------------------------------- 
    # FITS table structure.
    #----------------------------------------------------------------------------------------
    dt = np.dtype([('TELESCOPE',np.str,6),
                   ('PLATE',np.str,6),
                   ('NREADS',np.int32),
                   ('DATEOBS',np.str,30),
                   ('SECZ',np.float64),
                   ('HA',np.float64),
                   ('DESIGN_HA',np.float64,3),
                   ('SEEING',np.float64),
                   ('FWHM',np.float64),
                   ('GDRMS',np.float64),
                   ('CART',np.int32),
                   ('PLUGID',np.int32),
                   ('DITHER',np.float64),
                   ('MJD',np.int32),
                   ('IM',np.int32),
                   ('ZERO',np.float64),
                   ('ZERORMS',np.float64),
                   ('ZERONORM',np.float64),
                   ('SKY',np.float64,3),
                   ('SN',np.float64,3),
                   ('SNC',np.float64,3),
                   ('ALTSN',np.float64,3),
                   ('NSN',np.int32),
                   ('SNRATIO',np.float64),
                   ('MOONDIST',np.float64),
                   ('MOONPHASE',np.float64),
                   ('TELLFIT',np.float64,(6,3))])
    
    platetab = np.zeros(n_exposures,dtype=dt)

    platetab['PLATE'] =     plate
    platetab['TELESCOPE'] = -99.0
    platetab['HA'] =        -99.0
    platetab['DESIGN_HA'] = -99.0
    platetab['PLUGID'] =    plugid
    platetab['MJD'] =       mjd
    platetab['MOONDIST'] =  moondist
    platetab['MOONPHASE'] = moonphase

    for i in range(n_exposures):
        #----------------------------------------------------------------------------------------
        # Read image.
        #----------------------------------------------------------------------------------------
        if ims is None:
            pfile = os.path.basename(load.filename('Plate',plate=int(plate),mjd=mjd,chips=True)).replace('fits','')
        else:
            pfile = os.path.basename(load.filename('1D',plate=int(plate),num=ims[0],mjd=mjd,chips=True)).replace('.fits','')

        if (clobber is True) | (len(glob.glob(outdir+pfile+'.tab'))!=0):
            if ims is None:
                d = load.apPlate(int(plate),mjd) 
            else:
                d = load.ap1D(ims[i])

            dhdr = d['a'][0].header

            if type(d)!=dict:
                if ims is None: print("Problem with apPlate!!!")
                if ims not None: print("Problem with ap1D!!!")
#;                GOTO,badim

            cframe = None
            if ims is None:
                cframe = load.apPlate(int(plate),mjd)
            else:
                cframefile = load.filename('Cframe',plate=int(plate),mjd=mjd,num=ims[i],chips='c')

                if len(glob.glob(cframefile[0]))!=0:
                    cframe = load.apCframe(field,int(plate),mjd,ims[i])

            cframehdr = cframe['a'][0].header

            obs = np.zeros((nfiber,3),dtype=np.float64)
            sn  = np.zeros((nfiber,3),dtype=np.float64)
            snc = np.zeros((nfiber,3),dtype=np.float64)
            snt = np.zeros((nfiber,3),dtype=np.float64)

            objhtml = open(htmldir+pfile+'.html','w')
            objhtml.write('<HTML>\n')
            objhtml.write('<HEAD><script type=text/javascript src=../../../../html/sorttable.js></script></head>\n')
            objhtml.write('<BODY>\n')

            if ims is not None:
                objhtml.write('<H2>'+pfile+'</H2>\n')
                # NOTE: load.filename('Plate') returns just one nonexistent file
                tmp = load.filename('Plate',plate=int(plate),mjd=mjd,chips=True)
                platefile = np.array([tmp.replace('apPlate-','apPlate-a-'),
                                    tmp.replace('apPlate-','apPlate-b-'),
                                    tmp.replace('apPlate-','apPlate-c-')]
                for ichip in range(3):
                    objhtml.write('<A HREF=../'+platefile[ichip]+'>'+platefile[ichip]+'</A>\n')
            else:
                objhtml.write('<H2>'+str(ims[i])+'</H2>\n')
                if noplot is not None:
                    objhtml.write('<A HREF=../../../../red/'+mjd+'/html/'+pfile+'.html> 1D frames </A>\n')
                    objhtml.write('<BR><A HREF=../../../../red/'+mjd+'/html/ap2D-'+str(ims[i])+'.html> 2D frames </A>\n')

            objhtml.write('<TABLE BORDER=2 CLASS=sortable>\n')
            objhtml.write('<TR><TD>Fiber<TD>Star<TD>H mag<TD>Diff<TD>S/N<TD>S/N (cframe)<TD>Target flags\n')

            #----------------------------------------------------------------------------------------
            # For each fiber, get an observed mag from a median value.
            #----------------------------------------------------------------------------------------
            for j in range(nfiber):
                for ichip in range(3):
                    obs[j,ichip] = np.median(d['FLUX'][ichip][:,rows[j]])
            endfor

            if flat is None:
                for iline in range(len(skylines)):
                    skyline = skylines[iline]
                    skyline = getflux(d=d,skyline=skyline,rows=rows)
                    skylines[iline] = skyline

            #----------------------------------------------------------------------------------------
            # Get a "magnitude" for each fiber from a median on each chip.
            # Do a crude sky subtraction, calculate S/N.
            #----------------------------------------------------------------------------------------
            for ichip in range(3):
                if ims is None: medsky = 0.
                if ims is not None: medsky = np.median(obs[fibersky,ichip])

                if nobj>0: obs[fiberobj,ichip] = np.median(d['FLUX'][ichip][:,obj],axis=0)-medsky

                if ntelluric>0: obs[fibertelluric,ichip] = np.median(d['FLUX'][ichip][:,telluric],axis=0)-medsky

                if nobj>0:
                    sn[fiberobj,ichip] = np.median((d['FLUX'][ichip][:,obj]-medsky)/d['ERR'][ichip][:,obj],axis=0)
                    if len(cframe)>1:
                        snc[fiberobj,ichip] = np.median(cframe['FLUX'][ichip][:,obj]/cframe['ERR'][ichip][:,obj],axis=0)

                if ntelluric>0:
                    sn[fibertelluric,ichip] = np.median((d['FLUX'][ichip][:,telluric]-medsky)/d['ERR'][ichip][:,telluric],axis=0)
                    if len(cframe)>1:
                        snc[fibertelluric,ichip] = np.median(cframe['FLUX'][ichip][:,telluric]/cframe['ERR'][ichip][:,telluric],axis=0)
#;                        medfilt = MEDFILT2D(cframe['FLUX'][ichip][:,telluric],50,dim=1)
                        sz = shape(cframe['FLUX'][ichip])
                        i1 = 900*sz[1]/2048
                        i2 = 1000*sz[1]/2048
                        for itell in range(ntelluric):
                            p1 = np.std(cframe['FLUX'][ichip][i1:i2,telluric[itell]])
                            # NOTE: the below will not work because medfilt not defined.
                            p2 = np.std(cframe['FLUX'][ichip][i1:i2,telluric[itell]]-medfilt[i1:i2,itell])
                            snt[fibertelluric[itell],ichip] = p1/p2

                    else:
                        snc[fibertelluric,ichip] = sn[fibertelluric,ichip]
#;                        medfilt=MEDFILT2D(d['FLUX'][ichip][:,telluric],50,dim=1)
                        sz = shape(d['FLUX'][ichip])
                        i1 = 900*sz[1]/2048
                        i2 = 1000*sz[1]/2048
                        for itell in range(ntelluric):
                            p1 = np.mean(d['FLUX'][ichip][i1:i2*(sz[1]/2048),telluric[itell]])
                            # NOTE: the below will not work because medfilt not defined.
                            p2 = np.std(d['FLUX'][ichip][i1:i2,telluric[itell]]-medfilt[i1:i2,itell])
                            snt[fibertelluric[itell],ichip] = p1/p2

            #----------------------------------------------------------------------------------------
            # Calculate zeropoints from known H band mags.
            # Use a static zeropoint to calculate sky brightness.
            #----------------------------------------------------------------------------------------
            nreads = dhdr['NFRAMES']
            exptime = dhdr['EXPTIME']
            skyzero = 14.75+(2.5*np.log10(nreads))
            zero = 0
            zerorms = 0.
            faint = -1
            nfaint = 0
            achievedsn = [0.,0.,0.]
            achievedsnc = [0.,0.,0.]
            altsn = [0.,0.,0.]
            nsn = 0

            zero = np.median(fiber['HMAG'][[fiberobj,fibertelluric]]+(2.5*np.log10(obs[[fiberobj,fibertelluric],1])))
            # NOTE: not sure what to use for robust_sigma.
#:            zerorms=ROBUST_SIGMA(fiber[[fiberobj,fibertelluric]].hmag+2.5*alog10(obs[[fiberobj,fibertelluric],1]))
            faint = np.where((fiber['HMAG'][fiberobj,fibertelluric]+(2.5*np.log10(obs[[fiberobj,fibertelluric],1])-zero)<-0.5))
            nfaint = len(faint[0])

            zeronorm = zero-(2.5*np.log10(nreads))

            #----------------------------------------------------------------------------------------
            # For each star, create the exposure entry on the web page and set up the plot of the spectrum.
            #----------------------------------------------------------------------------------------
            cfile = open(outdir+pfile+'.csh','w')
            jsort = np.sort(fiber['FIBERID'])
            for jj in range(len(fiber)):
                j = jsort[jj]
                objhtml.write('<TR>\n')

                color = 'white'
                if (fiber['OBJTYPE'][j]=='SPECTROPHOTO_STD') | (fiber['OBJTYPE'][j]=='HOT_STD'): color = 'cyan'
                if fiber['OBJTYPE'][j]=='SKY': color = 'lightgreen'

                visitfile = os.path.basename(load.filename('Visit',plate=int(plate),mjd=mjd,fiber=fiber['FIBERID'][j]))

                cfib = str(fiber['FIBERID'][j]).zfill(3)
                if ims is None:
                    objhtml.write('<TD><A HREF=../'+visitfile+'>'+cfib+'</A>\n')
                else:
                    objhtml.write('<TD>'+cfib+'\n')

                if ims is None:
                    objhtml.write('<TD BGCOLOR='+color+'><a href=../plots/'+visitfile.replace('.fits','.jpg')+'>'+fiber['OBJECT'][j]+'</A>\n')
                else:
                    objhtml.write('<TD BGCOLOR='+color+'>'+cfib+'\n')

                rastring = str("%8.5f" % round(fiber['RA'][j],5))
                decstring = str("%8.5f" % round(fiber['DEC'][j],5))

                if (fiber['OBJECT'][j]!='sky') & (fiber['FIBERID'][j]>=0):
                    txt1 = '<BR><A HREF="http://simbad.decstring.harvard.edu/simbad/sim-basic?'
                    txt2 = 'Ident='+rastring+'+%09'+decstring+'++&submit=SIMBAD+search"> (SIMBAD) </A>'
                    objhtml.write(txt1+txt2+'\n')

                objhtml.write('<TD>'+str("%8.3f" % round(fiber['HMAG'][j],3))+'\n')
                objhtml.write('<TD>'+str("%8.2f" % round(fiber['HMAG'][j]+2.5*np.log10(obs[j,1])-zero,2))+'\n')
                objhtml.write('<TD>'+str("%8.2f" % round(sn[j,1],2))+'\n)
                objhtml.write('<TD>'+str("%8.2f" % round(snc[j,1],2))+'\n)
                # NOTE: targflag.pro not converted to Python yet
                objhtml.write('<TD>Target flags should go here but you need to translate TARGFLAG.PRO!\n)
#;                objhtml.write('<TD>'+TARGFLAG(fiber[j].target1,fiber[j].target2,fiber[j].target3,survey=survey)+'\n')

                if (ims is None) & (fiber['FIBERID'][j]>=0:
                    vfile = load.filename('Visit',plate=int(plate),mjd=mjd,fiber=fiber['FIBERID'][j])
                    if len(glob.glob(vfile))!=0:
                        h = fits.getheader(vfile)
                        if type(h)==astropy.io.fits.header.Header:
                            # NOTE: starflag.pro not converted to Python yet
                            objhtml.write('<TD>Star flags should go here but you need to translate STARFLAG.PRO!\n)
#;                            objhtml.write('<BR>'+STARFLAG(h['STARFLAG'])+'\n')

                #----------------------------------------------------------------------------------------
                # PLOT 1: spectrum 
                # https://data.sdss.org/sas/apogeework/apogee/spectro/redux/current/plates/5583/56257//plots/apPlate-5583-56257-299.jpg
                #----------------------------------------------------------------------------------------

                # NOTE: Not sure if this mod statement does the trick!
                if (j%300)>-1:
                    if noplot is None:
                        print("PLOTS 1: Spectrum plots will be made here.")
                    else:
                        objhtml.write('<TD>No plots for individual exposures, see plate plots\n')

            objhtml.close()
            cfile.close()

            #----------------------------------------------------------------------------------------
            # PLOT 2: 5 panels
            # https://data.sdss.org/sas/apogeework/apogee/spectro/redux/current/plates/5583/56257/plots/ap1D-06950025.gif
            #----------------------------------------------------------------------------------------
            if (flat is None) & (onem is None):
                print('PLOTS 2: 5-panel plot will be made here.')
            else:
                achievedsn = np.median(sn[obj,:],axis=0)

            #----------------------------------------------------------------------------------------
            # PLOTS 3-5: spatial residuals, , , 
            # 3: spatial residuals
            # https://data.sdss.org/sas/apogeework/apogee/spectro/redux/current/plates/5583/56257/plots/ap1D-06950025.jpg
            # 4: spatial sky line emission
            # https://data.sdss.org/sas/apogeework/apogee/spectro/redux/current/plates/5583/56257/plots/ap1D-06950025sky.jpg
            # 5: spatial continuum emission
            # https://data.sdss.org/sas/apogeework/apogee/spectro/redux/current/plates/5583/56257/plots/ap1D-06950025skycont.jpg
            #----------------------------------------------------------------------------------------
            if (starfiber is None) & (onem is None):
                print("PLOTS 3: spatial plot of residuals will be made here.\n")
                print("PLOTS 4: spatial plot of sky line emission will be made here.\n")
                print("PLOTS 5: spatial plot of continuum emission will be made here.\n")


        #----------------------------------------------------------------------------------------
        # Put all of the info and plots on the plate web page.
        #----------------------------------------------------------------------------------------
        medsky = np.zeros(3,dtype=np.float64)
        for ichip in range(3):
            if np.median(obs[fibersky,ichip])>0:
                medsky[ichip] = -2.5*np.log10(np.median(obs[fibersky,ichip]))+skyzero
            else: 
                medsky[ichip] = 99.999

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

        #----------------------------------------------------------------------------------------
        # Get guider info.
        #----------------------------------------------------------------------------------------
        if onem is None:
            dateobs = dhdr['DATE-OBS']
            exptime = dhdr['EXPTIME']
            tt = Time(dateobs)
            mjdstart = tt.mjd
            mjdend = mjdstart+(exptime/86400.)
            mjd0 = min([mjd0,mjdstart])
            mjd1 = max([mjd1,mjdend])
            nj = 0
            # NOTE: Not sure if type will actually be dict.
            if type(gcam)==dict:
                jcam = np.where((gcam['MJD']>mjdstart) & (gcam['MJD']<mjdend))
                nj = len(jcam[0])
            if nj>1: 
                fwhm = np.median(gcam['FWHM_MEDIAN'][jcam]) 
                gdrms = np.median(gcam['GDRMS'][jcam])
            else:
                fwhm = -1.
                gdrms = -1.
                print("not halted: no matching mjd range in gcam...")
        else:
            fwhm = -1
            gdrms = -1


        #----------------------------------------------------------------------------------------
        # Summary plate web page.
        #----------------------------------------------------------------------------------------
        htmlsum.write('<TR><TD><A HREF=../html/'+file+'.html>'+str(ims[i])+'</A>\n')
        htmlsum.write('<TD><A HREF=../../../../plates/'+plate+'/'+mjd+'/html/'+plate+'-'+mjd+'.html>'+str(dhdr['PLATEID'])+'</A>\n')
        htmlsum.write('<TD>'+str(dhdr['CARTID']+'\n')
        alt = dhdr['ALT']
        # NOTE: Not sure if the below is correct
        count = len(alt)
        if count>0:
            secz = 1./np.cos((90.-alt)*(math.pi/180.))
        else:
            # NOTE: Is the below a typo? Why not "AIRMASS"?
            secz = dhdr['ARMASS']
        seeing = dhdr['SEEING']
        ha = dhdr['HA']
        design_ha = platedata['HA']
        dither = -99.
        if len(cframe)>1: dither = cframehdr['DITHSH']
        htmlsum.write('<TD>'+str("%6.2f" % round(secz,2))+'\n')
        htmlsum.write('<TD>'+str("%6.2f" % round(ha,2))+'\n')
#;        htmlsum.write('<TD>'+string(format='(f6.0,",",f6.0,",",f6.0)',design_ha)+'\n')
        htmlsum.write('<TD>'+str("%6.2f" % round(seeing,2))+'\n')
        htmlsum.write('<TD>'+str("%6.2f" % round(fwhm,2))+'\n')
        htmlsum.write('<TD>'+str("%6.2f" % round(gdrms,2))+'\n')
        htmlsum.write('<TD>'+str(nreads)+'\n')
        if len(cframe)>1:
            htmlsum.write('<TD>'+str("%f8.2" % round(cframehdr['DITHSH'],2))+'\n')
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
            for j in range(nfaint): htmlsum.write(str(fiber['FIBERID'][faint][j])+'\n')
        allsky[i,:] = medsky
        allzero[i,:] = zero
        allzerorms[i,:] = zerorms

        #----------------------------------------------------------------------------------------
        # Summary information in apPlateSum FITS file.
        #----------------------------------------------------------------------------------------
        if ims is not None:
            tellfile = load.filename('Tellstar',plate=int(plate),mjd=mjd)
            telstr = fits.getdata(tellfile)
            # Should type == dict?
            if type(telstr)==astropy.io.fits.fitsrec.FITS_rec:
                jtell = np.where(telstr['IM']==ims[i])
                ntell = len(jtell[0])
                if ntell>0: platetab['TELLFIT'][i] = telstr['FITPARS'][jtell]
            else:
                print('Error reading Tellstar file: '+tellfile)

        platetab['IM'][i] =        ims[i]
        platetab['NREADS'][i] =     nreads
        platetab['SECZ'][i] =      secz
        platetab['HA'][i] =        ha
        platetab['DESIGN_HA'][i] = design_ha
        platetab['SEEING'][i] =    seeing
        platetab['FWHM'][i] =      fwhm
        platetab['GDRMS'][i] =     gdrms
        platetab['cart'][i] =      dhdr['CARTID']
        platetab['dateobs'][i] =   dhdr['DATE-OBS']
        platetab['DITHER'][i] =    dither
        platetab['ZERO'][i] =      zero
        platetab['ZERORMS'][i] =   zerorms
        platetab['ZERONORM'][i] =  zeronorm
        platetab['SKY'][i] =       medsky
        platetab['SN'][i] =        achievedsn
        platetab['ALTSN'][i] =     altsn
        platetab['NSN'][i] =       nsn
        platetab['SNC'][i] =       achievedsnc
        if ntelluric>0: platetab['SNRATIO'][i] = np.median(snt[telluric,1]/snc[telluric,1])

        for j in range(len(fiber)):
            fiber['SN'][j][i,:] = sn[j,:]
            fiber['OBSMAG'][j][i,:] = -2.5*np.log10(obs[j,:])+zero

#;        badim:

    #----------------------------------------------------------------------------------------
    # write out the FITS table.
    #----------------------------------------------------------------------------------------
    platefile = load.filename('PlateSum',plate=int(plate),mjd=mjd)
    if ims is not None:
        # Note sure the following will work
        Table(platetab).write(platefile)
        Table(fiber).write(platefile)
#;        mwrfits,platetab,platefile,/create
#;        mwrfits,fiber,platefile

    if ims is None:
        # Note sure the following will work
        Table(platetab).write(platefile)
        Table(fiber).write(platefile)
#;        mwrfits,platetab,platefile
#;        mwrfits,fiber,platefile

    html.write('</TABLE>\n')

    #----------------------------------------------------------------------------------------
    # For individual frames, make plots of variation of sky and zeropoint.
    # For combined frames, make table of combination parameters.
    #----------------------------------------------------------------------------------------

    if onem is not None: 
        name = starnames[0]+'-'+mjd
    else:
        name = plate+'-'+mjd

    if ims is not None:
        if onem is None:
        #----------------------------------------------------------------------------------------
        # PLOT 6: guider rms plot
        #----------------------------------------------------------------------------------------
        if type(gcam)==dict:
            jcam = np.where((gcam['MJD']>mjd0) & (gcam['MJD']<mjd1))
            nj = len(jcam[0]) 
            print("PLOTS 6: Guider RMS plots will be made here.")
#;            plot,gcam[jcam].mjd,gcam[jcam].gdrms
#;            file='guider-'+name

        #----------------------------------------------------------------------------------------
        # PLOT 7: make plot of sky levels for this plate
        # https://data.sdss.org/sas/apogeework/apogee/spectro/redux/current/exposures/apogee-n/56257/plots/56257sky.gif
        #----------------------------------------------------------------------------------------
        html.write('<TABLE BORDER=2><TR>\n')
        file = 'sky-'+name
        print("PLOTS 7: Sky level plots will be made here.")
#;        xtit='Image number'
#;        ytit='Continuum sky per pixel'
#;        xr=[(ims[0] mod 10000)-1,(ims[n_elements(ims)-1] mod 10000)+1]
#;        yr=[max(allsky)+0.3,min(allsky)-0.3]
#;        plot,ims,allsky,yr=yr,ps=8,xr=xr,xtit=xtit,ytit=ytit
#;        for i=0,2 do oplot,ims mod 10000,allsky[*,i],color=i+2,ps=8
#;        ps2gif,outdir+file+'.eps',/eps,chmod='664'o,/delete

        html.write('<TD><IMG SRC=../plots/'+file+'.gif>\n)'

        #----------------------------------------------------------------------------------------
        # PLOT 8: make plot of zeropoints for this plate
        # https://data.sdss.org/sas/apogeework/apogee/spectro/redux/current/exposures/apogee-n/56257/plots/56257zero.gif
        #----------------------------------------------------------------------------------------
        file = 'zero-'+name
        print("PLOTS 8: Zeropoints plots will be made here.")
#;        device,file=outdir+file+'.eps',/encap,ysize=8,/color
#;        xtit='Image number'
#;        ytit='Zeropoint sky per pixel'
#;        xr=[(ims[0] mod 10000)-1,(ims[n_elements(ims)-1] mod 10000)+1]
#;        yr=[max(allzero)+0.3,min(allzero)-0.3]
#;        plot,ims,allzero,yr=yr,ps=8,xr=xr,xtit=xtit,ytit=ytit
#;        for i=0,2 do oplot,ims mod 10000,allzero[*,i],color=i+2,ps=8

        html.write('<TD><IMG SRC=../plots/'+file+'.gif>\n')
        html.write('</TABLE>\n')

    else:

        # The following does not give apPlate-a-plate-mjd. It gives apPlate-plate-mjd, hence the replace part
        file = load.filename('Plate',plate=int(plate),mjd=mjd,chips='a').replace('Plate-','Plate-a-')
        tmp = fits.open(file)
        shiftstr = tmp[13].data
        pairstr = tmp[14].data
        npairs = len(pairstr)
        # Should type == dict?
        if (type(pairstr==astropy.io.fits.fitsrec.FITS_rec) & (npairs>0):
            #----------------------------------------------------------------------------------------
            # Pair table.
            #----------------------------------------------------------------------------------------
            html.write('<BR><TABLE BORDER=2>\n')
            html.write('<TR><TD>IPAIR<TD>NAME<TD>SHIFT<TD>NEWSHIFT<TD>S/N\n')
            html.write('<TD>NAME<TD>SHIFT<TD>NEWSHIFT<TD>S/N\n')
            for ipair in range(npairs):
                html.write('<TR><TD>'+str(ipair)+'\n')
                for j in range(2):
                    html.write('<TD>'+str(pairstr['framename'][ipair][j])+'\n')
                    html.write('<TD>'+str(pairstr['oldshift'][ipair][j])+'\n')
                    html.write('<TD>'+str(pairstr['shift'][ipair][j])+'\n')
                    html.write('<TD>'+str(pairstr['sn'][ipair][j])+'\n')
        else:
            #----------------------------------------------------------------------------------------
            # Table of combination parameters.
            #----------------------------------------------------------------------------------------
            html.write('<BR><TABLE BORDER=2>\n')
            for iframe in range(len(shiftstr)):
                html.write('<TR><TD>'+str(shiftstr['FRAMENUM'][iframe])+'\n')
                html.write('<TD>'+str(shiftstr['SHIFT'][iframe])+'\n')
                html.write('<TD>'+str(shiftstr['SN'][iframe])+'\n')

        html.write('</TABLE>\n')

    html.write('</BODY></HTML>')
    htmlsum.write('</TABLE>\n')

    if onem is not None:
        file = mjd+'-'+starnames[0]
        htmlsum.write('<a href=../plots/apVisit-'+apred+'-'+file+'.jpg><IMG src='+'../plots/apVisit-'+apred+'-'+file+'.jpg></A>\n')

    htmlsum.write('</BODY></HTML>')

    html.close()
    htmlsum.close()

'''-----------------------------------------------------------------------------------------'''
''' GETFLUX: Translation of getflux.pro                                                     '''
'''-----------------------------------------------------------------------------------------'''

def getflux(d=None,skyline=None,rows=None):
    nnrows = len(rows)
    if skyline['W1'] gt d['WAVE'][0][2047,150]:
        ichip = 0
    else:
        if skyline['W1']>d['WAVE'][1][2047,150]:
            ichip = 1
        else:
            ichip = 2

    cont = np.zeros(nnrows)
    line = np.zeros(nnrows)
    nline = np.zeros(nnrows)

    for i in range(nnrows):
        wave = d['WAVE'][ichip][:,rows[i]]
        data = d['FLUX'][ichip][:,rows[i]]

        icont = np.where(((wave>skyline['C1']) & (wave<skyline['C2'])) | ((wave<skyline['C3']) & (wave<skyline['C4'])))

        # NOTE: not sure if the below if is correct.
        if len(icont[0])>=0: cont[i] = np.median(d['FLUX'][ichip][icont,rows[i]])

        iline = np.where((wave>skyline['W1']) & (wave<skyline['W2']))

        # NOTE: not sure if the below if is correct.
        if len(iline[0])>=0:
            tmp = d['FLUX'][ichip][iline,rows[i]]
            line[i] = np.sum(tmp,where=math.isnan(tmp) is False)
            tmp = d['FLUX'][ichip][iline,rows[i]]/d['FLUX'][ichip][iline,rows[i]]
            line[i] = np.sum(tmp,where=math.isnan(tmp) is False)

    skyline['FLUX'] = line-(nline*cont)
    if skyline['TYPE']==0: skyline['FLUX']/=cont

    return skyline

























