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
from numpy.lib.recfunctions import append_fields
from astroplan import moon_illumination
from astropy.coordinates import SkyCoord, get_moon
from astropy import units as astropyUnits
from scipy.signal import medfilt2d as ScipyMedfilt2D
from apogee_drp.utils import plan,apload,yanny,plugmap,platedata,bitmask
from apogee_drp.apred import wave
from dlnpyutils import utils as dln
from sdss_access.path import path
import pdb

sdss_path = path.Path()

# put import pdb; pdb.set_trace() wherever you want stop

#sdss_path.full('ap2D',apred=self.apred,telescope=self.telescope,instrument=self.instrument,
#                        plate=self.plate,mjd=self.mjd,prefix=self.prefix,num=0,chip='a')

# Plugmap for plate 8100 mjd 57680
# /uufs/chpc.utah.edu/common/home/sdss50/sdsswork/data/mapper/apo/57679/plPlugMapM-8100-57679-01.par

# Planfile for plate 8100 mjd 57680
# https://data.sdss.org/sas/sdss5/mwm/apogee/spectro/redux/t14/visit/apo25m/200+45/8100/57680/apPlan-8100-57680.par

#------------------------------------------------------------------------------------------------------------------------
# APQA
#
#  call routines to make "QA" plots and web pages for a plate/MJD
#  for calibration frames, measures some features and makes a apQAcal file
#    with some summary information about the calibration data
#--------------------------------------------------------------------------------------------------


'''-----------------------------------------------------------------------------------------'''
'''APQA: Wrapper for running QA subprocedures                                               '''
'''-----------------------------------------------------------------------------------------'''

def apqa(field='200+45', plate='8100', mjd='57680', telescope='apo25m', apred='t14', noplot=False):
    #----------------------------------------------------------------------------------------
    # Use telescope, plate, mjd, and apred to load planfile into structure.
    #----------------------------------------------------------------------------------------
    load = apload.ApLoad(apred=apred, telescope=telescope)
    planfile = load.filename('Plan', plate=int(plate), mjd=mjd)
    planstr = plan.load(planfile, np=True)

    #----------------------------------------------------------------------------------------
    # Get values from plan file.
    #----------------------------------------------------------------------------------------
    fixfiberid = planstr['fixfiberid']
    badfiberid = planstr['badfiberid']
    platetype =  planstr['platetype']
    plugmap =    planstr['plugmap']
    fluxid =     planstr['fluxid']
    instrument = planstr['instrument']
    survey =     planstr['survey']

    #----------------------------------------------------------------------------------------
    # Establish directories.
    #----------------------------------------------------------------------------------------
    datadir = {'apo25m':os.environ['APOGEE_DATA_N'],'apo1m':os.environ['APOGEE_DATA_N'],
               'lco25m':os.environ['APOGEE_DATA_S']}[telescope]

    apodir =     os.environ.get('APOGEE_REDUX')+'/'
    spectrodir = apodir+apred+'/'
    caldir =     spectrodir+'cal/'
    expdir =     spectrodir+'exposures/'+instrument+'/'

    #----------------------------------------------------------------------------------------
    # Get array of object exposures and find out how many are objects.
    #----------------------------------------------------------------------------------------
    flavor = planstr['APEXP']['flavor']
    all_ims = planstr['APEXP']['name']

    gd,= np.where(flavor == 'object')
    n_ims = len(gd)

    if n_ims > 0:
        ims = all_ims[gd]
    else:
        print("No object images. You are hosed. Give up hope.")
        ims = None

    #----------------------------------------------------------------------------------------
    # Get mapper data.
    #----------------------------------------------------------------------------------------
    mapper_data = {'apogee-n':os.environ['MAPPER_DATA_N'],'apogee-s':os.environ['MAPPER_DATA_S']}[instrument]

    #----------------------------------------------------------------------------------------
    # For calibration plates, measure lamp brightesses and/or line widths, etc. and write to FITS file.
    #----------------------------------------------------------------------------------------
    #if platetype == 'cal': x = makeCalFits(load=load, ims=all_ims, mjd=mjd, instrument=instrument)

    #----------------------------------------------------------------------------------------
    # For darks and flats, get mean and stdev of column-medianed quadrants.
    #----------------------------------------------------------------------------------------
    #if platetype == 'dark': x = makeDarkFits(load=load, planfile=planfile, ims=all_ims, mjd=mjd)

    #----------------------------------------------------------------------------------------
    # For normal plates, make plots and html.
    #----------------------------------------------------------------------------------------
    if platetype == 'normal': 
        q = makePlotsHtml(load=load, telescope=telescope, ims=ims, plate=plate, mjd=mjd, 
                          field=field, instrument=instrument, clobber=True, noplot=True, 
                          mapname=plugmap, survey=survey, mapper_data=mapper_data, apred=apred,
                          onem=None, starfiber=None, starnames=None, starmag=None,flat=None) 

#fixfiberid=fixfiberid,badfiberid=badfiberid)

        #x = makePlotsHtml(load=load, telescope=telescope, ims=None, plate=plate, mjd=mjd, 
        #                  field=field, instrument=instrument, clobber=True, noplot=noplot,
        #                  mapname=plugmap, survey=survey, mapper_data=mapper_data,apred=apred
        #                  onem=None, starfiber=None, starnames=None, starmag=None,flat=None) 

#fixfiberid=fixfiberid, badfiberid=badfiberid

        x = masterQApage(load=load, plate=plate, mjd=mjd, field=field, fluxid=fluxid, telescope=telescope)


        # NOTE: No translations for plotflux yet.

#;        x = plotFlux(planfile)

        #platesumfile = load.filename('PlateSum', plate=int(plate), mjd=mjd, chips=True)

        # NOTE: No python translation for sntab.
#;        sntab,tabs=platefile,outfile=platefile+'.dat'

    #----------------------------------------------------------------------------------------
    # For ASDAF and NMSU 1m observations, get more values and make plots and html.
    #----------------------------------------------------------------------------------------
    if platetype == 'single':
        
        single = [planstr['APEXP'][i]['single'].astype(int) for i in range(n_ims)]
        sname = [planstr['APEXP'][i]['singlename'] for i in range(n_ims)]
        smag = planstr['hmag']

        x = makePlotsHtml(telescope=telescope, onem=True, ims=ims, starnames=sname, starfiber=single,
                          starmag=smag, fixfiberid=fixfiberid, clobber=True, mapname=plugmap,
                          noplot=noplot, badfiberid=badfiberid, survey=survey, apred=apred)


'''-----------------------------------------------------------------------------------------'''
''' MASTERQAPAGE: mkhtmlplate translation                                                   '''
'''-----------------------------------------------------------------------------------------'''

def masterQApage(load=None, plate=None, mjd=None, field=None, fluxid=None, telescope=None):
    print("--------------------------------------------------------------------")
    print("Making MASTERQAPAGE for plate "+plate+", mjd "+mjd)
    print("--------------------------------------------------------------------\n")

    chips = np.array(['a','b','c'])
    nchips = len(chips)

    prefix = 'ap'
    if telescope == 'lco25m': prefix = 'as'

    #----------------------------------------------------------------------------------------
    # Check for existence of plateSum file
    #----------------------------------------------------------------------------------------
    platesum = load.filename('PlateSum', plate=int(plate), mjd=mjd) 
    platesumfile = os.path.basename(platesum)
    platedir = os.path.dirname(platesum)+'/'

    #print("platedir: "+platedir+"\n")
    #print(platedir+platesumfile)
    if os.path.exists(platesum) is False:
        err1 = "PROBLEM!!! "+platesumfile+" does not exist. Halting execution.\n"
        err2 = "You need to run MAKEPLOTSHTML first to make the file."
        sys.exit(err1 + err2)

    #----------------------------------------------------------------------------------------
    # Make the html directory if it doesn't already exist
    #----------------------------------------------------------------------------------------
    qafile = load.filename('QA', plate=int(plate), mjd=mjd)
    qafiledir = os.path.dirname(qafile)
    #print("opening "+platedir+"/html/apQA")
    if os.path.exists(qafiledir) is False: subprocess.call(['mkdir',qafiledir])

    html = open(qafile, 'w')
    html.write('<HTML><BODY>\n')
    html.write('<H2> PLATE: '+plate+' MJD: '+mjd+' FIELD: '+field+'</H2>\n')

    #----------------------------------------------------------------------------------------
    # Read the plateSum file
    # NOTE: apLoad.py does not have an option for loading apPlateSum, so using astropy instead
    #----------------------------------------------------------------------------------------

    tmp = fits.open(platesum)
    tab1 = tmp[1].data
    tab2 = tmp[2].data
    tab3 = tmp[3].data
#;    tab3 = mrdfits(platesum,3,status=status)

    # NOTE: just setting status=1 and hoping for the best.
    status = 1
    if status < 0:
        html.write('<FONT COLOR=red> ERROR READING TAB FILE\n')
        html.write('</BODY></HTML>\n')
        html.close()

    platefile = load.apPlate(int(plate), mjd)
    shiftstr = platefile['a'][13].data
    pairstr = platefile['a'][14].data
#;    shiftstr = mrdfits(platefile,13)
#;    pairstr = mrdfits(platefile,14,status=status)

    if status < 0:
        html.write('<FONT COLOR=red> ERROR READING apPlate FILE\n')
        html.write('</BODY></HTML>\n')
        html.close()
        return

    #----------------------------------------------------------------------------------------
    # Table of individual exposures.
    #----------------------------------------------------------------------------------------
    html.write('<TABLE BORDER=2>\n')
    html.write('<TR bgcolor=lightgreen>\n')
    html.write('<TD>Frame<TD>Cart<TD>sec z<TD>HA<TD>DESIGN HA<TD>seeing<TD>FWHM<TD>GDRMS<TD>Nreads<TD>Dither<TD>Pixshift<TD>Zero<TD>Zero rms<TD>sky continuum<TD>S/N<TD>S/N(cframe)\n')
    for i in range(len(tab1)):
        html.write('<TR>\n')
        html.write('<TD>'+str(int(round(tab1['IM'][i])))+'\n')
        html.write('<TD>'+str(int(round(tab1['CART'][i])))+'\n')
        html.write('<TD>'+str("%.2f" % round(tab1['SECZ'][i],2))+'\n')
        html.write('<TD>'+str("%.2f" % round(tab1['HA'][i],2))+'\n')
        html.write('<TD>'+str(np.round(tab1['DESIGN_HA'][i],0)).replace('[',' ')[:-1]+'\n')
        html.write('<TD>'+str("%.2f" % round(tab1['SEEING'][i],2))+'\n')
        html.write('<TD>'+str("%.2f" % round(tab1['FWHM'][i],2))+'\n')
        html.write('<TD>'+str("%.2f" % round(tab1['GDRMS'][i],2))+'\n')
        html.write('<TD>'+str(tab1['NREADS'][i])+'\n')
        j = np.where(shiftstr['FRAMENUM'] == str(tab1['IM'][i]))
        nj = len(j[0])
        if nj > 0:
            html.write('<TD>'+str("%.4f" % round(shiftstr['SHIFT'][j][0],4)).rjust(7)+'\n')
            html.write('<TD>'+str("%.2f" % round(shiftstr['PIXSHIFT'][j][0],2))+'\n')
        else:
            html.write('<TD><TD>\n')
            html.write('<TD><TD>\n')
        html.write('<TD>'+str("%.2f" % round(tab1['ZERO'][i],2))+'\n')
        html.write('<TD>'+str("%.2f" % round(tab1['ZERORMS'][i],2))+'\n')
        q = tab1['SKY'][i]
        txt = str("%.2f" % round(q[0],2))+', '+str("%.2f" % round(q[1],2))+', '+str("%.2f" % round(q[2],2))
        html.write('<TD>'+'['+txt+']\n')
        q = tab1['SN'][i]
        txt = str("%.2f" % round(q[0],2))+', '+str("%.2f" % round(q[1],2))+', '+str("%.2f" % round(q[2],2))
        html.write('<TD>'+'['+txt+']\n')
        q = tab1['SNC'][i]
        txt = str("%.2f" % round(q[0],2))+', '+str("%.2f" % round(q[1],2))+', '+str("%.2f" % round(q[2],2))
        html.write('<TD>'+'['+txt+']\n')
    html.write('</TABLE>\n')

    #----------------------------------------------------------------------------------------
    # Table of exposure pairs.
    #----------------------------------------------------------------------------------------
    npairs = len(pairstr)
    #if (type(pairstr) == 'astropy.io.fits.fitsrec.FITS_rec') & (npairs > 0):
    if npairs > 0:
        #----------------------------------------------------------------------------------------
        # Pair table.
        #----------------------------------------------------------------------------------------
        html.write('<BR><TABLE BORDER=2>\n')
        html.write('<TR bgcolor=lightgreen><TD>IPAIR<TD>NAME<TD>SHIFT<TD>NEWSHIFT<TD>S/N\n')
        html.write('<TD>NAME<TD>SHIFT<TD>NEWSHIFT<TD>S/N\n')
        for ipair in range(npairs):
            html.write('<TR><TD>'+str(ipair)+'\n')
            for j in range(2):
                html.write('<TD>'+str(pairstr['FRAMENAME'][ipair][j])+'\n')
                html.write('<TD>'+str("%.3f" % round(pairstr['OLDSHIFT'][ipair][j],3))+'\n')
                html.write('<TD>'+str("%.3f" % round(pairstr['SHIFT'][ipair][j],3))+'\n')
                html.write('<TD>'+str("%.2f" % round(pairstr['SN'][ipair][j],2))+'\n')
    else:
        #----------------------------------------------------------------------------------------
        # Table of combination parameters.
        #----------------------------------------------------------------------------------------
        html.write('<BR><TABLE BORDER=2>\n')
        for iframe in range(len(shiftstr)):
            html.write('<TR><TD>'+str(shiftstr['FRAMENUM'][iframe])+'\n')
            html.write('<TD>'+str("%.3f" % round(shiftstr['SHIFT'][iframe],3))+'\n')
            html.write('<TD>'+str("%.3f" % round(shiftstr['SN'][iframe],3))+'\n')
    html.write('</TABLE>\n')

    #----------------------------------------------------------------------------------------
    # Link to combined spectra page.
    #----------------------------------------------------------------------------------------
    html.write('<P><A HREF='+prefix+'Plate-'+plate+'-'+mjd+'.html> Visit (multiple exposure and dither combined) spectra </a><p>\n')

    #----------------------------------------------------------------------------------------
    # Flat field plots.
    #----------------------------------------------------------------------------------------
    if fluxid is not None:
        html.write('<P> Flat field relative fluxes\n')
        html.write('<TABLE BORDER=2><TR>\n')
        for chip in chips:
            fluxfile = load.filename('Flux', num=fluxid, chips=True).replace('apFlux-','apFlux-'+chip+'-')
            fluxfile = os.path.basename(fluxfile).replace('.fits','')
            html.write('<TD> <A HREF='+'../plots/'+fluxfile+'.jpg><IMG SRC=../plots/'+fluxfile+'.jpg WIDTH=400></\n')
        tmp = load.filename('Flux', num=fluxid, chips=True).replace('apFlux-','apFlux-'+chips[0]+'-')
        blockfile = os.path.basename(tmp).replace('.fits','').replace('-a-','-block-')
        html.write('<TD> <A HREF='+'../plots/'+blockfile+'.jpg><IMG SRC=../plots/'+blockfile+'.jpg WIDTH=400></A>\n')
        html.write('</TABLE>\n')

    gfile = 'guider-'+plate+'-'+mjd+'.jpg'
    html.write('<A HREF='+'../plots/'+gfile+'><IMG SRC=../plots/'+gfile+' WIDTH=400></A>\n')

    #----------------------------------------------------------------------------------------
    # Table of exposure plots.
    #----------------------------------------------------------------------------------------
    html.write('<TABLE BORDER=2>\n')

    html.write('<TR><TD>Frame<TD>Zeropoints<TD>Mag plots\n')
    html.write('<TD>Spatial mag deviation\n')
    html.write('<TD>Spatial sky telluric CH4\n')
    html.write('<TD>Spatial sky telluric CO2\n')
    html.write('<TD>Spatial sky telluric H2O\n')
    html.write('<TD>Spatial sky 16325A emission deviations (filled: sky, open: star)\n')
    html.write('<TD>Spatial sky continuum emission\n')

    for i in range(len(tab1)):
        im=tab1['IM'][i]
        oneDfile = os.path.basename(load.filename('1D', plate=int(plate), num=im, mjd=mjd, chips=True)).replace('.fits','')
        html.write('<TR><TD><A HREF=../html/'+oneDfile+'.html>'+str(im)+'</A>\n')
        html.write('<TD><TABLE BORDER=1><TD><TD>Red<TD>Green<TD>Blue\n')
        html.write('<TR><TD>z<TD><TD>'+str("%.2f" % round(tab1['ZERO'][i],2))+'\n')
        html.write('<TR><TD>znorm<TD><TD>'+str("%.2f" % round(tab1['ZERONORM'][i],2))+'\n')
        html.write('<TR><TD>sky'+str(np.round(tab1['SKY'][i],1)).replace('[',' ').replace(' ','<TD>')[:-1]+'\n')
        html.write('<TR><TD>S/N'+str(np.round(tab1['SN'][i],1)).replace('[',' ').replace(' ','<TD>')[:-1]+'\n')
        html.write('<TR><TD>S/N(c)'+str(np.round(tab1['SNC'][i],1)).replace('[',' ').replace(' ','<TD>')[:-1]+'\n')
        #if tag_exist(tab1[i],'snratio'):
        html.write('<TR><TD>SN(E/C)<TD<TD>'+str(np.round(tab1['SNRATIO'][i],2))+'\n')
        html.write('</TABLE>\n')
        html.write('<TD><IMG SRC=../plots/'+oneDfile+'.gif>\n')
        html.write('<TD> <IMG SRC=../plots/'+oneDfile+'.jpg>\n')
        cim=str(im)
        html.write('<TD> <a href=../plots/'+prefix+'telluric_'+cim+'_skyfit_CH4.jpg> <IMG SRC=../plots/'+prefix+'telluric_'+cim+'_skyfit_CH4.jpg height=400></a>\n')
        html.write('<TD> <a href=../plots/'+prefix+'telluric_'+cim+'_skyfit_CO2.jpg> <IMG SRC=../plots/'+prefix+'telluric_'+cim+'_skyfit_CO2.jpg height=400></a>\n')
        html.write('<TD> <a href=../plots/'+prefix+'telluric_'+cim+'_skyfit_H2O.jpg> <IMG SRC=../plots/'+prefix+'telluric_'+cim+'_skyfit_H2O.jpg height=400></a>\n')
        html.write('<TD> <IMG SRC=../plots/'+oneDfile+'sky.jpg>\n')
        html.write('<TD> <IMG SRC=../plots/'+oneDfile+'skycont.jpg>\n')
    html.write('</table>\n')

    html.write('</BODY></HTML>\n')
    html.close()

    print("--------------------------------------------------------------------")
    print("Done with MASTERQAPAGE for plate "+plate+", mjd "+mjd)
    print("--------------------------------------------------------------------\n")

'''-----------------------------------------------------------------------------------------'''
''' MAKEPLOTSHTML: Plotmag translation                                                      '''
'''-----------------------------------------------------------------------------------------'''

def makePlotsHtml(load=None, telescope=None, ims=None, plate=None, mjd=None, field=None, 
                  instrument=None, clobber=True, noplot=False, mapname=None, survey=None,
                  mapper_data=None, apred=None, onem=None, starfiber=None, starnames=None, 
                  starmag=None, flat=None): 
#, flat=None,
#, fixfiberid=None, badfiberid=None):

    print("--------------------------------------------------------------------")
    print("Running MAKEPLOTSHTML for plate "+plate+", mjd "+mjd)
    print("--------------------------------------------------------------------\n")

    if ims is not None: n_exposures = len(ims)

    chips = np.array(['a','b','c'])
    nchips = len(chips)

    #----------------------------------------------------------------------------------------
    # Make plot and html directories if they don't already exist.
    #----------------------------------------------------------------------------------------
    platedir = os.path.dirname(load.filename('Plate', plate=int(plate), mjd=mjd, chips=True))
    outdir = platedir+'/plots/'
    if len(glob.glob(outdir)) == 0: subprocess.call(['mkdir',outdir])

    htmldir = platedir+'/html/'
    if len(glob.glob(htmldir)) == 0: subprocess.call(['mkdir',htmldir])

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
    # Get the fiber association for this plate. Also get some other values
    #----------------------------------------------------------------------------------------
    if ims is None: tot = load.apPlate(int(plate), mjd)
    if ims is not None: tot = load.ap1D(ims[0])
    platehdr = tot['a'][0].header
    ra = platehdr['RADEG']
    dec = platehdr['DECDEG']
    DateObs = platehdr['DATE-OBS']

    if type(tot) != dict:
        html.write('<FONT COLOR=red> PROBLEM/FAILURE WITH: '+str(ims[0])+'\n')
        htmlsum.write('<FONT COLOR=red> PROBLEM/FAILURE WITH: '+str(ims[0])+'\n')
        html.close()
        htmlsum.close()
        print("Error in makePlotsHtml!!!")

    if mapname is not None:
        if mapname[0] == 'header':
            plugid = platehdr['NAME']
        else:
            plugid = mapname
    else:
        plugid = platehdr['NAME']


    if onem is None:
        plug = platedata.getdata(int(plate), int(mjd), apred, telescope, plugid=plugid) 
    else: 
        plug = platedata.getdata(int(plate), int(mjd), apred, telescope, plugid=plugid,
                                 obj1m=starnames[0], starfiber=starfiber) 

    gd, = np.where(plug['fiberdata']['fiberid'] > 0)
    fiber = plug['fiberdata'][gd]
    nfiber = len(fiber)
    rows = 300-fiber['fiberid']
    guide = plug['guidedata']

    dtype = np.dtype([('sn', np.float64, (n_exposures,3))])
    newColumn = np.zeros(nfiber, dtype=dtype)
    fiber = append_fields(fiber, 'sn', newColumn, usemask=False)
    fiber = append_fields(fiber, 'obsmag', newColumn, usemask=False)

    unplugged = np.where(fiber['fiberid'] < 0)
    nunplugged = len(unplugged[0])
    if flat is not None:
        fiber['hmag'] = 12
        fiber['object'] = 'FLAT'

    #----------------------------------------------------------------------------------------
    # Find telluric fibers.
    #----------------------------------------------------------------------------------------
    fibertelluric, = np.where((fiber['objtype'] == 'SPECTROPHOTO_STD') |
                              (fiber['objtype'] == 'HOT_STD'))
    ntelluric = len(fibertelluric)
    telluric = rows[fibertelluric]

    #----------------------------------------------------------------------------------------
    # Find science fibers.
    #----------------------------------------------------------------------------------------
    fiberobj, = np.where((fiber['objtype'] == 'STAR_BHB') | 
                         (fiber['objtype'] == 'STAR') | 
                         (fiber['objtype'] == 'EXTOBJ'))
    nobj = len(fiberobj)
    obj = rows[fiberobj]

    #----------------------------------------------------------------------------------------
    # Find sky fibers.
    #----------------------------------------------------------------------------------------
    fibersky, = np.where(fiber['objtype'] == 'SKY')
    nsky = len(fibersky)
    sky = rows[fibersky]

    #----------------------------------------------------------------------------------------
    # Find all fiber placed on stars.
    #----------------------------------------------------------------------------------------
    fiberstar = np.concatenate([fiberobj,fibertelluric]);  fiberstar.sort()

    #----------------------------------------------------------------------------------------
    # Define skylines structure which we will use to get crude sky levels in lines.
    #----------------------------------------------------------------------------------------
    dt = np.dtype([('W1',   np.float64),
                   ('W2',   np.float64),
                   ('C1',   np.float64),
                   ('C2',   np.float64),
                   ('C3',   np.float64),
                   ('C4',   np.float64),
                   ('FLUX', np.float64, (nfiber)),
                   ('TYPE', np.int32)])

    skylines = np.zeros(2,dtype=dt)

    skylines['W1']   = 16230.0, 15990.0
    skylines['W2']   = 16240.0, 16028.0
    skylines['C1']   = 16215.0, 15980.0
    skylines['C2']   = 16225.0, 15990.0
    skylines['C3']   = 16245.0, 0.0
    skylines['C4']   = 16255.0, 0.0
    skylines['TYPE'] = 1, 0

    #----------------------------------------------------------------------------------------
    # Loop through all the images for this plate, and make the plots.
    # Load up and save information for this plate in a FITS table.
    #----------------------------------------------------------------------------------------
    allsky =     np.zeros((n_exposures,3), dtype=np.float64)
    allzero =    np.zeros((n_exposures,3), dtype=np.float64)
    allzerorms = np.zeros((n_exposures,3), dtype=np.float64)

    #----------------------------------------------------------------------------------------
    # Get moon distance and phase.
    #----------------------------------------------------------------------------------------
    tt = Time(DateObs, format='fits')
    moonpos = get_moon(tt)
    moonra = moonpos.ra.deg
    moondec = moonpos.dec.deg
    c1 = SkyCoord(ra * astropyUnits.deg, dec * astropyUnits.deg)
    c2 = SkyCoord(moonra * astropyUnits.deg, moondec * astropyUnits.deg)
    sep = c1.separation(c2)
    moondist = sep.deg
    moonphase = moon_illumination(tt)

    #----------------------------------------------------------------------------------------
    # Get guider information.
    #----------------------------------------------------------------------------------------

    # NOTE: skipping gcam stuff until I get gcam_process to work
#    if onem is None:
#        expdir = os.environ.get('APOGEE_REDUX')+'/'+apred+'/'+'exposures/'+instrument+'/'
#        gcamfile = expdir+'/'+mjd+'/gcam-'+mjd+'.fits'
#        gcamfilecheck = glob.glob(gcamfile)
#        if len(gcamfilecheck) == 0:
            # NOTE: hopefully this works
#            subprocess.call(['gcam_process', '--mjd', mjd, '--instrument', instrument], shell=False)
#        gcamfilecheck = glob.glob(gcamfile)
#        if len(gcamfilecheck) != 0:
#            gcam = fits.getdata(gcamfile)
#        else:
#            print("Problem running gcam_process!")

    mjd0 = 99999
    mjd1 = 0.

    #---------------------------------------------------------------------------------------- 
    # FITS table structure.
    #----------------------------------------------------------------------------------------
    dt = np.dtype([('TELESCOPE', np.str, 6),
                   ('PLATE',     np.str, 6),
                   ('NREADS',    np.int32),
                   ('DATEOBS',   np.str, 30),
                   ('SECZ',      np.float64),
                   ('HA',        np.float64),
                   ('DESIGN_HA', np.float64, 3),
                   ('SEEING',    np.float64),
                   ('FWHM',      np.float64),
                   ('GDRMS',     np.float64),
                   ('CART',      np.int32),
                   ('PLUGID',    np.str, 30),
                   ('DITHER',    np.float64),
                   ('MJD',       np.int32),
                   ('IM',        np.int32),
                   ('ZERO',      np.float64),
                   ('ZERORMS',   np.float64),
                   ('ZERONORM',  np.float64),
                   ('SKY',       np.float64, 3),
                   ('SN',        np.float64, 3),
                   ('SNC',       np.float64, 3),
                   ('ALTSN',     np.float64, 3),
                   ('NSN',       np.int32),
                   ('SNRATIO',   np.float64),
                   ('MOONDIST',  np.float64),
                   ('MOONPHASE', np.float64),
                   ('TELLFIT',   np.float64, (6,3))])

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
        if ims is None: pfile = os.path.basename(load.filename('Plate', plate=int(plate), mjd=mjd, chips=True))
        if ims is not None: pfile = os.path.basename(load.filename('1D', plate=int(plate), num=ims[0], mjd=mjd, chips=True))
        pfile = pfile.replace('.fits','')

        if (clobber is True) | (len(glob.glob(outdir+pfile+'.tab')) != 0):
            if ims is None:     d = load.apPlate(int(plate), mjd) 
            if ims is not None: d = load.ap1D(ims[i])

            dhdr = d['a'][0].header

            if type(d)!=dict:
                if ims is None:  print("Problem with apPlate!!!")
                if ims is not None: print("Problem with ap1D!!!")

            cframe = None
            if ims is None: cframe = load.apPlate(int(plate), mjd)
            if ims is not None:
                cframefile = load.filename('Cframe', plate=int(plate), mjd=mjd, num=ims[i], chips='c')
                cframefile = cframefile.replace('apCframe-','apCframe-c-')

                if len(glob.glob(cframefile)) != 0:
                    cframe = load.apCframe(field, int(plate), mjd, ims[i])

            cframehdr = cframe['a'][0].header

            obs = np.zeros((nfiber,3), dtype=np.float64)
            sn  = np.zeros((nfiber,3), dtype=np.float64)
            snc = np.zeros((nfiber,3), dtype=np.float64)
            snt = np.zeros((nfiber,3), dtype=np.float64)

            #----------------------------------------------------------------------------------------
            # For each fiber, get an observed mag from a median value.
            #----------------------------------------------------------------------------------------
            for j in range(nfiber):
                for ichip in range(nchips): 
                    obs[j, ichip] = np.median(d[chips[ichip]][1].data[rows[j], :])

            if flat is None:
                for iline in range(len(skylines)):
                    skylines['FLUX'][iline] = getflux(d=d, skyline=skylines[iline], rows=rows)

            #----------------------------------------------------------------------------------------
            # Get a "magnitude" for each fiber from a median on each chip.
            # Do a crude sky subtraction, calculate S/N.
            #----------------------------------------------------------------------------------------
            for ichip in range(nchips):
                chip = chips[ichip]

                fluxarr = d[chip][1].data
                errarr = d[chip][2].data
                cfluxarr = cframe[chip][1].data
                cerrarr = cframe[chip][2].data

                if ims is None:     medsky = 0.
                if ims is not None: medsky = np.median(obs[fibersky, ichip])

                # NOTE: using axis=0 caused error, so trying axis=0
                if nobj > 0: obs[fiberobj, ichip] = np.median(fluxarr[obj, :], axis=1) - medsky

                if ntelluric > 0: obs[fibertelluric, ichip] = np.median(fluxarr[telluric, :], axis=1) - medsky

                if nobj > 0:
                    sn[fiberobj, ichip] = np.median((fluxarr[obj, :]-medsky) / errarr[obj, :], axis=1)
                    if len(cframe) > 1:
                        snc[fiberobj, ichip] = np.median(cfluxarr[obj, :] / cerrarr[obj, :], axis=1)

                if ntelluric > 0:
                    sn[fibertelluric, ichip] = np.median((fluxarr[telluric,:] - medsky) / errarr[telluric, :], axis=1)
                    if len(cframe) > 1:
                        snc[fibertelluric, ichip] = np.median(cfluxarr[telluric, :] / cerrarr[telluric, :], axis=1)
                        medfilt = ScipyMedfilt2D(cfluxarr[telluric, :], kernel_size=(1,49))
                        sz = cfluxarr.shape
                        i1 = int(np.floor((900 * sz[1]) / 2048))
                        i2 = int(np.floor((1000 * sz[1]) / 2048))
                        for itell in range(ntelluric):
                            p1 = np.mean(cfluxarr[telluric[itell], i1:i2])
                            p2 = np.std(cfluxarr[telluric[itell], i1:i2] - medfilt[itell, i1:i2])
                            snt[fibertelluric[itell], ichip] = p1 / p2

                    else:
                        snc[fibertelluric,ichip] = sn[fibertelluric,ichip]
                        medfilt = ScipyMedfilt2D(fluxarr[telluric, :], kernel_size=(1,49))
                        sz = fluxarr.shape
                        i1 = int(np.floor((900 * sz[1]) / 2048))
                        i2 = int(np.floor((1000 * sz[1]) / 2048))
                        for itell in range(ntelluric):
                            p1 = np.mean(fluxarr[telluric[itell], i1:i2 * (int(np.floor(sz[1] / 2048)))])
                            p2 = np.std(fluxarr[telluric[itell], i1:i2] - medfilt[itell, i1:i2])
                            snt[fibertelluric[itell], ichip] = p1 / p2

            #----------------------------------------------------------------------------------------
            # Calculate zeropoints from known H band mags.
            # Use a static zeropoint to calculate sky brightness.
            #----------------------------------------------------------------------------------------
            nreads = dhdr['NFRAMES']
            exptime = dhdr['EXPTIME']
            skyzero = 14.75 + (2.5 * np.log10(nreads))
            zero = 0
            zerorms = 0.
            faint = -1
            nfaint = 0
            achievedsn = [0.,0.,0.]
            achievedsnc = [0.,0.,0.]
            altsn = [0.,0.,0.]
            nsn = 0

            tmp = fiber['hmag'][fiberstar] + (2.5 * np.log10(obs[fiberstar,1]))
            zero = np.median(tmp)
            zerorms = dln.mad(fiber['hmag'][fiberstar] + (2.5 * np.log10(obs[fiberstar,1])))
            faint = np.where((tmp - zero) < -0.5)
            nfaint = len(faint[0])

            zeronorm = zero - (2.5 * np.log10(nreads))

            #----------------------------------------------------------------------------------------
            # For each star, create the exposure entry on the web page and set up the plot of the spectrum.
            #----------------------------------------------------------------------------------------
            objhtml = open(htmldir+pfile+'.html','w')
            objhtml.write('<HTML>\n')
            objhtml.write('<HEAD><script type=text/javascript src=../../../../html/sorttable.js></script></head>\n')
            objhtml.write('<BODY>\n')

            if ims is not None:
                objhtml.write('<H2>'+pfile+'</H2>\n')
                tmp = load.apPlate(int(plate), mjd)
                for chip in chips: 
                    objhtml.write('<A HREF=../'+tmp[chip].filename()+'>'+tmp[chip].filename()+'</A>\n')
            else:
                objhtml.write('<H2>'+str(ims[i])+'</H2>\n')
                if noplot is not None:
                    objhtml.write('<A HREF=../../../../red/'+mjd+'/html/'+pfile+'.html> 1D frames </A>\n')
                    objhtml.write('<BR><A HREF=../../../../red/'+mjd+'/html/ap2D-'+str(ims[i])+'.html> 2D frames </A>\n')

            objhtml.write('<TABLE BORDER=2 CLASS=sortable>\n')
            objhtml.write('<TR><TD>Fiber<TD>Star<TD>H mag<TD>Diff<TD>S/N<TD>S/N (cframe)<TD>Target flags\n')

            cfile = open(outdir+pfile+'.csh','w')
            jsort = np.sort(fiber['fiberid'])
            for j in range(nfiber):
                #j = jsort[jj]
                #print(str(j))
                objhtml.write('<TR>\n')

                color = 'white'
                if (fiber['objtype'][j] == 'SPECTROPHOTO_STD') | (fiber['objtype'][j] == 'HOT_STD'): color = 'cyan'
                if fiber['objtype'][j] == 'SKY': color = 'lightgreen'

                visitfile = os.path.basename(load.filename('Visit', plate=int(plate), mjd=mjd, fiber=fiber['fiberid'][j]))

                cfib = str(fiber['fiberid'][j]).zfill(3)
                if ims is None:
                    objhtml.write('<TD><A HREF=../'+visitfile+'>'+cfib+'</A>\n')
                else:
                    objhtml.write('<TD>'+cfib+'\n')

                if ims is None:
                    objhtml.write('<TD BGCOLOR='+color+'><a href=../plots/'+visitfile.replace('.fits','.jpg')+'>'+fiber['object'][j]+'</A>\n')
                else:
                    objhtml.write('<TD BGCOLOR='+color+'>'+cfib+'\n')

                rastring = str("%8.5f" % round(fiber['ra'][j],5))
                decstring = str("%8.5f" % round(fiber['dec'][j],5))

                if (fiber['objtype'][j]!='SKY') & (fiber['fiberid'][j]>=0):
                    txt1 = '<BR><A HREF="http://simbad.decstring.harvard.edu/simbad/sim-basic?'
                    txt2 = 'Ident='+rastring+'+%09'+decstring+'++&submit=SIMBAD+search"> (SIMBAD) </A>'
                    objhtml.write(txt1+txt2+'\n')

                objhtml.write('<TD>'+str("%8.3f" % round(fiber['hmag'][j],3))+'\n')
                objhtml.write('<TD>'+str("%8.2f" % round(fiber['hmag'][j]+2.5*np.log10(obs[j,1])-zero,2))+'\n')
                objhtml.write('<TD>'+str("%8.2f" % round(sn[j,1],2))+'\n')
                objhtml.write('<TD>'+str("%8.2f" % round(snc[j,1],2))+'\n')
                targflagtxt = bitmask.targflags(fiber['target1'][j], fiber['target2'][j], fiber['target3'][j], fiber['target4'][j], survey=survey)
                objhtml.write('<TD>'+targflagtxt+'\n')

                if (ims is None) & (fiber['fiberid'][j] >= 0):
                    vfile = load.filename('Visit', plate=int(plate), mjd=mjd, fiber=fiber['fiberid'][j])
                    if os.path.exists(vfile):
                        h = fits.getheader(vfile)
                        if type(h) == astropy.io.fits.header.Header:
                            objhtml.write('<BR>'+bitmask.StarBitMask().getname(h['STARFLAG'])+'\n')

                #----------------------------------------------------------------------------------------
                # PLOT 1: spectrum 
                # https://data.sdss.org/sas/apogeework/apogee/spectro/redux/current/plates/5583/56257//plots/apPlate-5583-56257-299.jpg
                #----------------------------------------------------------------------------------------

                # NOTE: Not sure if this mod statement does the trick!
                if (j%300) > -1:
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
                achievedsn = np.median(sn[obj,:], axis=0)

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
        medsky = np.zeros(3, dtype=np.float64)
        for ichip in range(nchips):
            if np.median(obs[fibersky,ichip]) > 0:
                medsky[ichip] = -2.5 * np.log10(np.median(obs[fibersky,ichip])) + skyzero
            else: 
                medsky[ichip] = 99.999

        html.write('<TR><TD><A HREF=../html/'+pfile+'.html>'+str(ims[i])+'</A>\n')
        html.write('<TD>'+str(nreads)+'\n')
        html.write('<TD><TABLE BORDER=1><TD><TD>Red<TD>Green<TD>Blue\n')
        html.write('<TR><TD>z<TD><TD>'+str("%5.2f" % round(zero,2))+'\n')
        html.write('<TR><TD>znorm<TD><TD>'+str("%5.2f" % round(zeronorm,2))+'\n')
        txt = '<TD> '+str("%5.1f" % round(medsky[0],1))+'<TD> '+str("%5.1f" % round(medsky[1],1))+'<TD> '+str("%5.1f" % round(medsky[2],1))
        html.write('<TR><TD>sky'+txt+'\n')
        txt = '<TD> '+str("%5.1f" % round(achievedsn[0],1))+'<TD> '+str("%5.1f" % round(achievedsn[1],1))+'<TD> '+str("%5.1f" % round(achievedsn[2],1))
        html.write('<TR><TD>S/N'+txt+'\n')
        txt = '<TD> '+str("%5.1f" % round(achievedsnc[0],1))+'<TD> '+str("%5.1f" % round(achievedsnc[1],1))+'<TD> '+str("%5.1f" % round(achievedsnc[2],1))
        html.write('<TR><TD>S/N(c)'+txt+'\n')

        if ntelluric > 0:
           html.write('<TR><TD>SN(E/C)<TD<TD>'+str("%5.2f" % round(np.median(snt[telluric,1] / snc[telluric,1]),2))+'\n')
        else:
            html.write('<TR><TD>SN(E/C)<TD<TD>\n')

        html.write('</TABLE>\n')
        html.write('<TD><IMG SRC=../plots/'+pfile+'.gif>\n')
        html.write('<TD> <IMG SRC=../plots/'+pfile+'.jpg>\n')
        html.write('<TD> <IMG SRC=../plots/'+pfile+'sky.jpg>\n')
        html.write('<TD> <IMG SRC=../plots/'+pfile+'skycont.jpg>\n')
        html.write('<TD> <IMG SRC=../plots/'+pfile+'telluric.jpg>\n')

        #----------------------------------------------------------------------------------------
        # Get guider info.
        #----------------------------------------------------------------------------------------
        if onem is None:
            dateobs = dhdr['DATE-OBS']
            exptime = dhdr['EXPTIME']
            tt = Time(dateobs)
            mjdstart = tt.mjd
            mjdend = mjdstart + (exptime/86400.)
            mjd0 = min([mjd0,mjdstart])
            mjd1 = max([mjd1,mjdend])
            nj = 0
        # NOTE: skipping gcam stuff until I get gcam_process to work
#            if type(gcam) == astropy.io.fits.fitsrec.FITS_rec:
#                jcam = np.where((gcam['MJD'] > mjdstart) & (gcam['MJD'] < mjdend))
#                nj = len(jcam[0])
#            if nj > 1: 
#                fwhm = np.median(gcam['FWHM_MEDIAN'][jcam]) 
#                gdrms = np.median(gcam['GDRMS'][jcam])
#            else:
#                fwhm = -1.
#                gdrms = -1.
#                print("not halted: no matching mjd range in gcam...")
#        else:
#            fwhm = -1
#            gdrms = -1

        fwhm = -1
        gdrms = -1


        #----------------------------------------------------------------------------------------
        # Summary plate web page.
        #----------------------------------------------------------------------------------------
        htmlsum.write('<TR><TD><A HREF=../html/'+pfile+'.html>'+str(ims[i])+'</A>\n')
        htmlsum.write('<TD><A HREF=../../../../plates/'+plate+'/'+mjd+'/html/'+plate+'-'+mjd+'.html>'+str(dhdr['PLATEID'])+'</A>\n')
        htmlsum.write('<TD>'+str(dhdr['CARTID'])+'\n')
        alt = dhdr['ALT']
        secz = 1. / np.cos((90.-alt) * (math.pi/180.))
        seeing = dhdr['SEEING']
        # NOTE: 'HA' is not in the ap1D header... setting ha=0 for now
#        ha = dhdr['HA']
        ha=0
        # NOTE: 'ha' is not in the plugfile, but values are ['-', '-', '-']. Setting design_ha=0 for now
#        design_ha = plug['ha']
        design_ha = [0,0,0]
        dither = -99.
        if len(cframe) > 1: dither = cframehdr['DITHSH']
        htmlsum.write('<TD>'+str("%6.2f" % round(secz,2))+'\n')
        htmlsum.write('<TD>'+str("%6.2f" % round(ha,2))+'\n')
        txt = '[ '+str(int(round(design_ha[0])))+','+str(int(round(design_ha[1])))+','+str(int(round(design_ha[2])))+']'
        htmlsum.write('<TD>'+txt+'\n')
        htmlsum.write('<TD>'+str("%6.2f" % round(seeing,2))+'\n')
        htmlsum.write('<TD>'+str("%6.2f" % round(fwhm,2))+'\n')
        htmlsum.write('<TD>'+str("%6.2f" % round(gdrms,2))+'\n')
        htmlsum.write('<TD>'+str(nreads)+'\n')
        if len(cframe) > 1:
            htmlsum.write('<TD>'+str("%f8.2" % round(dither,2))+'\n')
        else:
            htmlsum.write('<TD>\n')
        htmlsum.write('<TD>'+str("%5.2f" % round(zero,2))+'\n')
        htmlsum.write('<TD>'+str("%5.2f" % round(zerorms,2))+'\n')
        htmlsum.write('<TD>'+str("%5.2f" % round(zeronorm,2))+'\n')
        txt = '['+str("%5.2f" % round(medsky[0],2))+','+str("%5.2f" % round(medsky[1],2))+','+str("%5.2f" % round(medsky[2],2))+']'
        htmlsum.write('<TD>'+txt+'\n')
        txt = '['+str("%5.1f" % round(achievedsn[0],1))+','+str("%5.1f" % round(achievedsn[1],1))+','+str("%5.1f" % round(achievedsn[2],1))+']'
        htmlsum.write('<TD>'+txt+'\n')
        txt = '['+str("%5.1f" % round(achievedsnc[0],1))+','+str("%5.1f" % round(achievedsnc[1],1))+','+str("%5.1f" % round(achievedsnc[2],1))+']'
        htmlsum.write('<TD>'+txt+'\n')
        htmlsum.write('<TD>\n')
        for j in range(unplugged): htmlsum.write(str(300-unplugged[j])+'\n')
        htmlsum.write('<TD>\n')
        if faint[0] > 0:
            for j in range(nfaint): htmlsum.write(str(fiber['fiberid'][faint][j])+'\n')
        allsky[i,:] = medsky
        allzero[i,:] = zero
        allzerorms[i,:] = zerorms

        #----------------------------------------------------------------------------------------
        # Summary information in apPlateSum FITS file.
        #----------------------------------------------------------------------------------------
        if ims is not None:
            tellfile = load.filename('Tellstar', plate=int(plate), mjd=mjd)
            telstr = fits.getdata(tellfile)
            if type(telstr) == astropy.io.fits.fitsrec.FITS_rec:
                jtell = np.where(telstr['IM'] == ims[i])
                ntell = len(jtell[0])
                if ntell > 0: platetab['TELLFIT'][i] = telstr['FITPARS'][jtell]
            else:
                print('Error reading Tellstar file: '+tellfile)

        platetab['IM'][i] =        ims[i]
        platetab['NREADS'][i] =    nreads
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
        if ntelluric > 0: platetab['SNRATIO'][i] = np.median(snt[telluric,1] / snc[telluric,1])

        for j in range(len(fiber)):
            fiber['SN'][j][i,:] = sn[j,:]
            fiber['OBSMAG'][j][i,:] = (-2.5 * np.log10(obs[j,:])) + zero

    #----------------------------------------------------------------------------------------
    # write out the FITS table.
    #----------------------------------------------------------------------------------------
    platesum = load.filename('PlateSum', plate=int(plate), mjd=mjd)
    if ims is not None:
        # NOTE: the only different between below if statement is that if ims is none, /create is not set in mwrfits
        # ... not sure if we care.

        Table(platetab).write(platesum)
        hdulist = fits.open(platesum)
        hdu = fits.table_to_hdu(Table(fiber))
        hdulist.append(hdu)
        hdulist.writeto(platesum,overwrite=True)
        hdulist.close()
#;        mwrfits,platetab,platesum,/create
#;        mwrfits,fiber,platesum

    if ims is None:
        hdulist = fits.open(platesum)
        hdu1 = fits.table_to_hdu(Table(platetab))
        hdu2 = fits.table_to_hdu(Table(fiber))
        hdulist.append(hdu1)
        hdulist.append(hdu2)
        hdulist.writeto(platesum,overwrite=True)
        hdulist.close()
#;        mwrfits,platetab,platesum
#;        mwrfits,fiber,platesum

    html.write('</TABLE>\n')

    #----------------------------------------------------------------------------------------
    # For individual frames, make plots of variation of sky and zeropoint.
    # For combined frames, make table of combination parameters.
    #----------------------------------------------------------------------------------------
    if onem is None: name = plate+'-'+mjd
    if onem is not None: name = starnames[0]+'-'+mjd

    if ims is not None:
        # NOTE: skipping gcam stuff until I get gcam_process to work
#        if onem is None:
            #----------------------------------------------------------------------------------------
            # PLOT 6: guider rms plot
            #----------------------------------------------------------------------------------------
#            if type(gcam) == astropy.io.fits.fitsrec.FITS_rec:
#                jcam = np.where((gcam['MJD'] > mjd0) & (gcam['MJD'] < mjd1))
#                nj = len(jcam[0]) 
#                print("PLOTS 6: Guider RMS plots will be made here.")

        #----------------------------------------------------------------------------------------
        # PLOT 7: make plot of sky levels for this plate
        # https://data.sdss.org/sas/apogeework/apogee/spectro/redux/current/exposures/apogee-n/56257/plots/56257sky.gif
        #----------------------------------------------------------------------------------------
        html.write('<TABLE BORDER=2><TR>\n')
        skyfile = 'sky-'+name
        print("PLOTS 7: Sky level plots will be made here.")

        html.write('<TD><IMG SRC=../plots/'+skyfile+'.gif>\n')

        #----------------------------------------------------------------------------------------
        # PLOT 8: make plot of zeropoints for this plate
        # https://data.sdss.org/sas/apogeework/apogee/spectro/redux/current/exposures/apogee-n/56257/plots/56257zero.gif
        #----------------------------------------------------------------------------------------
        zerofile = 'zero-'+name
        print("PLOTS 8: Zeropoints plots will be made here.")

        html.write('<TD><IMG SRC=../plots/'+zerofile+'.gif>\n')
        html.write('</TABLE>\n')
    else:
        pfile = load.apPlate(int(plate), mjd)
        shiftstr = pfile['a'][13].data
        pairstr = pfile['a'][14].data
        npairs = len(pairstr)

        #----------------------------------------------------------------------------------------
        # Pair table.
        #----------------------------------------------------------------------------------------
        if (type(pairstr) == astropy.io.fits.fitsrec.FITS_rec) & (npairs > 0):

            html.write('<BR><TABLE BORDER=2>\n')
            html.write('<TR><TD>IPAIR<TD>NAME<TD>SHIFT<TD>NEWSHIFT<TD>S/N\n')
            html.write('<TD>NAME<TD>SHIFT<TD>NEWSHIFT<TD>S/N\n')
            for ipair in range(npairs):
                html.write('<TR><TD>'+str(ipair)+'\n')
                for j in range(2):
                    html.write('<TD>'+str(pairstr['FRAMENAME'][ipair][j])+'\n')
                    html.write('<TD>'+str(pairstr['OLDSHIFT'][ipair][j])+'\n')
                    html.write('<TD>'+str(pairstr['SHIFT'][ipair][j])+'\n')
                    html.write('<TD>'+str(pairstr['SN'][ipair][j])+'\n')
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
        onemfile = mjd+'-'+starnames[0]
        htmlsum.write('<a href=../plots/apVisit-'+apred+'-'+onemfile+'.jpg><IMG src='+'../plots/apVisit-'+apred+'-'+onemfile+'.jpg></A>\n')

    htmlsum.write('</BODY></HTML>')

    html.close()
    htmlsum.close()

    print("--------------------------------------------------------------------")
    print("Done with MAKEPLOTSHTML for plate "+plate+", mjd "+mjd)
    print("--------------------------------------------------------------------\n")


'''-----------------------------------------------------------------------------------------'''
''' MAKECALFITS: Make FITS file for cals (lamp brightness, line widths, etc.)               '''
'''-----------------------------------------------------------------------------------------'''

def makeCalFits(load=None, ims=None, mjd=None, instrument=None):
    print("--------------------------------------------------------------------")
    print("Running MAKECALFITS for plate "+plate+", mjd "+mjd)
    print("--------------------------------------------------------------------\n")

    n_exposures = len(ims)

    nlines = 2
    chips=np.array(['a','b','c'])
    nchips = len(chips)

    tharline = np.array([[940.,1128.,1130.],[1724.,623.,1778.]])
    uneline =  np.array([[603.,1213.,1116.],[1763.,605.,1893.]])

    if instrument == 'apogee-s': tharline = np.array([[944.,1112.,1102.],[1726.,608.,1745.]])
    if instrument == 'apogee-s':  uneline = np.array([[607.,1229.,1088.],[1765.,620.,1860.]])

    fibers = np.array([10,80,150,220,290])
    nfibers = len(fibers)

    #----------------------------------------------------------------------------------------
    # Make output structure.
    #----------------------------------------------------------------------------------------
    dt = np.dtype([('NAME',    np.str,30),
                   ('MJD',     np.str,30),
                   ('JD',      np.float64),
                   ('NFRAMES', np.int32),
                   ('NREAD',   np.int32),
                   ('EXPTIME', np.float64),
                   ('QRTZ',    np.int32),
                   ('UNE',     np.int32),
                   ('THAR',    np.int32),
                   ('FLUX',    np.float64,(300,nchips)),
                   ('GAUSS',   np.float64,(4,nfibers,nchips,nlines)),
                   ('WAVE',    np.float64,(nfibers,nchips,nlines)),
                   ('FIBERS',  np.float64,(nfibers)),
                   ('LINES',   np.float64,(nchips,nlines))])

    struct = np.zeros(n_exposures, dtype=dt)

    #----------------------------------------------------------------------------------------
    # Loop over exposures and get 1D images to fill structure.
    # /uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/apogee/spectro/redux/t14/exposures/apogee-n/57680/ap1D-21180073.fits
    #----------------------------------------------------------------------------------------
    for i in range(n_exposures):
        oneD = load.ap1D(ims[i])
        oneDhdr = oneD['a'][0].header

        if type(oneD)==dict:
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
            if struct['QRTZ'][i]==1: struct['FLUX'][i] = np.median(oneD['a'][1].data, axis=0)

            #----------------------------------------------------------------------------------------
            # Arc lamp exposures.
            #----------------------------------------------------------------------------------------
            if (struct['THAR'][i] == 1) | (struct['UNE'][i] == 1):
                if struct['THAR'][i] == 1: line = tharline
                if struct['THAR'][i] != 1: line = uneline

                struct['LINES'][i] = line

                nlines = 1
                if line.shape[0]!=1: nlines = line.shape[1]

                for iline in range(nlines):
                    for ichip in range(nchips):
                        print("Calling appeakfit... no, not really because it's a long IDL code.")
                        # NOTE: the below does not work yet... maybe use findlines instead?
                        # https://github.com/sdss/apogee/blob/master/pro/apogeereduce/appeakfit.pro
                        # https://github.com/sdss/apogee/blob/master/python/apogee/apred/wave.py

#;                        APPEAKFIT,a[ichip],linestr,fibers=fibers,nsigthresh=10
                        linestr = wave.findlines(oneD, rows=fibers, lines=line)
                        linestr = wave.peakfit(oneD[chips[ichip]][1].data)

                        for ifiber in range(nfibers):
                            fibers = fibers[ifiber]
                            j = np.where(linestr['FIBER'] == fiber)
                            nj = len(j)
                            if nj>0:
                                junk = np.min(np.absolute(linestr['GAUSSX'][j] - line[ichip,iline]))
                                jline = np.argmin(np.absolute(linestr['GAUSSX'][j] - line[ichip,iline]))
                                struct['GAUSS'][:,ifiber,ichip,iline][i] = linestr['GPAR'][j][jline]
                                sz = a['WCOEF'][ichip].shape
                                if sz[0] == 2:
                                    # NOTE: the below has not been tested
#;                                    struct['WAVE'][i][ifiber,ichip,iline] = pix2wave(linestr['GAUSSX'][j][jline],oneD['WCOEF'][ichip][fiber,:])
                                    pix = linestr['GAUSSX'][j][jline]
                                    wave0 = a['WCOEF'][ichip][fiber,:]
                                    struct['WAVE'][i][ifiber,ichip,iline] = wave.pix2wave(pix, wave0)

                                struct['FLUX'][i][fiber,ichip] = linestr['SUMFLUX'][j][jline]
        else:
            print("type(1D) does not equal dict. This is probably a problem.")

    outfile = load.filename('QAcal', plate=int(plate), mjd=mjd) 
    Table(struct).write(outfile)

    print("--------------------------------------------------------------------")
    print("Done with MAKECALFITS for plate "+plate+", mjd "+mjd)
    print("Made "+outfile)
    print("--------------------------------------------------------------------\n")


'''-----------------------------------------------------------------------------------------'''
''' MAKEDARKFITS: Make FITS file for darks (get mean/stddev of column-medianed quadrants)   '''
'''-----------------------------------------------------------------------------------------'''

def makeDarkFits(load=None, planfile=None, ims=None, mjd=None):
    print("--------------------------------------------------------------------")
    print("Running MAKEDARKFITS for plate "+plate+", mjd "+mjd)
    print("--------------------------------------------------------------------\n")

    n_exposures = len(ims)

    chips=np.array(['a','b','c'])
    nchips = len(chips)
    nquad = 4

    #----------------------------------------------------------------------------------------
    # Make output structure.
    #----------------------------------------------------------------------------------------
    dt = np.dtype([('NAME',    np.str, 30),
                   ('MJD',     np.str, 30),
                   ('JD',      np.float64),
                   ('NFRAMES', np.int32),
                   ('NREAD',   np.int32),
                   ('EXPTIME', np.float64),
                   ('QRTZ',    np.int32),
                   ('UNE',     np.int32),
                   ('THAR',    np.int32),
                   ('EXPTYPE', np.str, 30),
                   ('MEAN',    np.float64, (nchips,nquad)),
                   ('SIG',     np.float64, (nchips,nquad))])

    struct = np.zeros(n_exposures, dtype=dt)

    #----------------------------------------------------------------------------------------
    # Loop over exposures and get 2D images to fill structure.
    # /uufs/chpc.utah.edu/common/home/sdss50/sdsswork/mwm/apogee/spectro/redux/t14/exposures/apogee-n/57680/ap2D-21180073.fits
    #----------------------------------------------------------------------------------------
    for i in range(n_exposures):
        twoD = load.ap2D(ims[i])
        twoDhdr = twoD['a'][0].header

        if type(twoD) == dict:
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
                    sm = np.median(twoD[chips[ichip]][1].data[10:2000, i1:i2], axis=0)
                    struct['MEAN'][i,ichip,iquad] = np.mean(sm)
                    struct['SIG'][i,ichip,iquad] = np.std(sm)
                    i1 += 512
                    i2 += 512
        else:
            print("type(2D) does not equal dict. This is probably a problem.")

    outfile = load.filename('QAcal', plate=int(plate), mjd=mjd).replace('apQAcal','apQAdarkflat')
    Table(struct).write(outfile)

    print("--------------------------------------------------------------------")
    print("Done with MAKEDARKFITS for plate "+plate+", mjd "+mjd)
    print("Made "+outfile)
    print("--------------------------------------------------------------------\n")


'''-----------------------------------------------------------------------------------------'''
''' GETFLUX: Translation of getflux.pro                                                     '''
'''-----------------------------------------------------------------------------------------'''

def getflux(d=None, skyline=None, rows=None):

    chips = np.array(['a','b','c'])
    nnrows = len(rows)

    # NOTE: pretty sure that [2047,150] subscript won't work, but 150,2057 will. Hoping for the best.
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



