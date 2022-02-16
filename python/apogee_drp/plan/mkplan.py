import copy
import numpy as np
import os
import shutil
from glob import glob
import pdb
import subprocess
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from dlnpyutils import utils as dln
from ..utils import spectra,yanny,apload,platedata,utils,info
from ..apred import mkcal
from . import mkslurm,check
from sdss_access.path import path
from astropy.io import fits
from astropy.table import Table
from collections import OrderedDict


def args2dict(**kwargs):
    """ Dummy function used by translate_idl_mjd5_script()."""
    return kwargs

def fixidlcontinuation(lines):
    """
    Fix continuation lines.
    This is a small helper function for translate_idl_mjd5_script().
    """

    # Fix continuation lines
    if np.sum(lines.find('$')>-1):
        continueline = ''
        lines2 = []
        for i in range(len(lines)):
            line1 = lines[i]
            if line1.find('$')>-1:
                continueline += line1[0:line1.find('$')]
            else:
                lines2.append(continueline+line1)
                continueline = ''
        lines = np.char.array(lines2)
        del lines2

    return lines

def removeidlkeyword(line,key):
    """
    Remove an IDL keyword input like ,/cal in a line.
    This is a small helper function for translate_idl_mjd5_script().
    """

    out = ''
    lo = line.find(key)
    if lo==0:
        out = line[lo+len(key):]
    else:
        out = line[0:lo]+line[lo+len(key):]
    out = out.replace(',,',',')
    if out.endswith(','):
        out = out[0:-1]
    return out

def removeidlcomments(lines):
    """
    Remove IDL comments from a list of lines.
    This is a small helper function for translate_idl_mjd5_script().
    """

    flines = []
    for i in range(len(lines)):
        line1 = lines[i]
        lo = line1.lower().find(';')
        if lo==-1:
            flines.append(line1)
        else:
            # if the entire line is commented, leave it out
            if lo>0:
                flines.append(line1[0:lo])
    return np.char.array(flines)

def replaceidlcode(lines,mjd,day=None):
    """
    Replace IDL code in lines (array of strings) with the results of code
    execution. This is a small helper function for translate_idl_mjd5_script().
    """

    # day
    #  psfid=day+138
    #  domeid=day+134
    if day is not None:
        ind,nind = dln.where( (lines.lower().find('day')>-1) &
                              (lines.lower().startswith('day=')==False) )
        if nind>0:
            lines[ind] = lines[ind].replace('day',str(day))
    
    # indgen
    #  ims=day+149+indgen(2)
    ind,nind = dln.where(lines.lower().find('indgen(')>-1)
    if nind>0:
        lines[ind] = lines[ind].replace('indgen(','np.arange(')


    # Deal with assignment lines with code to execute
    ind,nind = dln.where( ((lines.lower().find('+')>-1) |
                           (lines.lower().find('-')>-1) |
                           (lines.lower().find('*')>-1) |
                           (lines.lower().find('np.arange')>-1)) &
                          (lines.lower().find('=')>-1) &
                          (lines.lower().find('mkplan')==-1) )
    for i in range(nind):
        line1 = lines[ind[i]]
        lo = line1.find('=')
        key = line1[0:lo]
        val = eval(line1[lo+1:])
        if (type(val) is int) | (type(val) is str):
            lines[ind[i]] = key+'='+str(val)
        else:
            lines[ind[i]] = key+'='+str(list(val))

    # Deal with mkplan lines with code to execute
    ind,nind = dln.where( ((lines.lower().find('+')>-1) |
                           (lines.lower().find('-')>-1) |
                           (lines.lower().find('*')>-1) |
                           (lines.lower().find('np.arange')>-1)) &
                          (lines.lower().find('=')>-1) &
                          (lines.lower().find('mkplan')>-1) )
    for i in range(nind):
        line1 = lines[ind[i]]
        raise ValueError('This has not been implemented yet')

    return lines


def translate_idl_mjd5_script(scriptfile):
    """
    Translate an IDL MJD5.pro script file to yaml.  It returns a list of strings
    that can be written to a file.

    Parameters
    ----------
    scriptfile : str
         Name of MJD5.pro script file.

    Returns
    -------
    flines : numpy char array
         The lines of the script file translated to yaml.

    Examples
    --------
    flines = mkplan.translate_idl_mjd5_script('apo25m_59085.pro')

    Example file, top part of apo25m_59085.pro
    apsetver,telescope='apo25m'
    mjd=59085
    plate=11950
    psfid=35230030
    fluxid=35230030
    ims=[35230018,35230019,35230020,35230021,35230022,35230023,35230024,35230025,35230026,35230027,35230028,35230029]
    mkplan,ims,plate,mjd,psfid,fluxid,vers=vers
    
    ;these are not sky frames
    plate = 12767
    psfid=35230015
    fluxid=35230015
    ims=[35230011,35230012,35230013,35230014]
    mkplan,ims,plate,mjd,psfid,fluxid,vers=vers;,/sky
    
    plate=12673
    psfid=35230037
    fluxid=35230037
    ims=[35230033,35230034,35230035,35230036]
    mkplan,ims,plate,mjd,psfid,fluxid,vers=vers

    By D.Nidever,  Oct 2020
    """

    # Check that the file exists
    if os.path.exists(scriptfile)==False:
        raise ValueError(scriptfile+" NOT FOUND")

    # Load the file
    lines = dln.readlines(scriptfile)
    lines = np.char.array(lines)


    # Fix continuation lines
    lines = fixidlcontinuation(lines)
    # Remove comments
    lines = removeidlcomments(lines)

    # Get telescope from apserver line
    ind,nind = dln.where(lines.strip().lower().find('apsetver')==0)
    telescope = None
    if nind==0:
        print('No APSERVER line found')
        if scriptfile.lower().find('apo25m')>-1: telescope='apo25m'
        if scriptfile.lower().find('lco25m')>-1: telescope='lcoo25m'
        if telescope is None:
            raise ValueError('Cannot find TELESCOPE')
    else:
        setverline = lines[ind[0]]
        telescope = setverline[setverline.lower().find('telescope=')+10:]
        telescope = telescope.replace("'","")
    telescopeline = "telescope: "+telescope

    # Get MJD
    ind,nind = dln.where(lines.strip().lower().find('mjd=')==0)
    if nind==0:
        raise ValueError('No MJD line found')
    mjdline = lines[ind[0]]
    mjd = int(mjdline[mjdline.find('=')+1:])
    mjdline = 'mjd: '+str(mjd)

    # Get day number
    ind,nind = dln.where(lines.lower().find('day=')>-1)
    if nind>0:
        dayline = lines[ind[0]].lower()
        # day=getnum(mjd)*10000
        if dayline.lower().find('getnum')>-1:
            dayline = dayline.replace('getnum(mjd)','(mjd-55562)')
        day = int(eval(dayline[dayline.find('=')+1:]))
    else:
        day = None

    # Remove apvers, mjd and day line
    gd,ngd = dln.where( (lines.strip('').lower().startswith('day=')==False) &
                        (lines.strip('').lower().find('apsetver')==-1) &
                        (lines.strip('').lower().startswith('mjd=')==False) )
    lines = lines[gd]

    # Deal with IDL code using day, indgen(), etc.
    lines = replaceidlcode(lines,mjd,day=day)

    # Initalize final lines
    flines = ['---']  # start of yaml file

    # Loop over mkplan blocks
    #  mkplan command is at the end of the block
    ind,nind = dln.where(lines.lower().find('mkplan')!=-1)
    for i in range(nind):
        if i==0:
            lo = 0
        else:
            lo = ind[i-1]+1
        lines1 = lines[lo:ind[i]+1]
        nlines1 = len(lines1)
        # Add TELESCOPE line
        flines.append("- "+telescopeline)
        # Add MJD line
        flines.append("  "+mjdline)
        # Assume all lines in this block except for mkplan are key: value pairs
        kvlines = lines1[0:-1]
        for kvl in kvlines:
            if kvl.strip()!='':
                lo = kvl.find('=')
                key = kvl[0:lo].strip()
                val = kvl[lo+1:].strip()
                flines.append("  "+key+": "+val)
        # Deal with mkplan lines
        planline = lines1[-1]
        # Trim off the first bit that's always the same, "mkplan,ims,plate,mjd,psfid,fluxid,"
        planline = planline[planline.lower().find('fluxid')+7:]
        # Remove vers=vers if it's there
        if planline.lower().find('vers=vers')==0:
            planline = planline[9:]

        # Deal with keywords
        if planline!='':
            if planline[0]==',':
                planline = planline[1:]
            # Add lines for sky, dark, cal
            if planline.lower().find('/sky')>-1:
                flines.append('  sky: True')
                planline = removeidlkeyword(planline,'/sky')  # Trim off /sky
            if planline.lower().find('/dark')>-1:
                flines.append('  dark: True')
                planline = removeidlkeyword(planline,'/dark')  # Trim off /dark
            if planline.lower().find('/cal')>-1:
                flines.append('  cal: True')
                planline = removeidlkeyword(planline,'/cal')  # Trim off /cal

        # Deal with remaining arguments
        if planline!='':
            # Return leftover line as a dictionary
            import pdb; pdb.set_trace()
            exec("args=args2dict("+planline+")")
            # Loop over keys and add them
            for k in args.keys():
                val = args[k]
                if (type(val) is int) | (type(val) is str):
                    flines.append("  "+k+": "+str(val))
                else:
                    flines.append("  "+k+": "+str(list(val)))

    # End of yaml file
    flines.append('...')

    return flines


def dailycals(waves=None,psf=None,lsfs=None,apred=None,telescope=None):
    """
    Create plan file for daily calibration products.

    Parameters
    ----------
    waves : numpy int array
        Array of wavecal exposure numbers.
    psf : int
        Exposure number for PSF cal.
    lsfs : numpy int array
        Array of LSF exposure numbers.
    apred : str
        APOGEE reduction version.
    telescope : str
        APOGEE telescope.

    Returns
    -------
    The dailycal.par file for the relevant instrument is updated.

    Examples
    --------
    mkplan.dailycals(lsfs=ims,psf=psfid)

    By J.Holtzman, 2011
    translated to python, D.Nidever  Oct 2020
    """

    if waves is not None and psf is None:
        raise ValueError('psf keyword must be given with waves keyword')
    if apred is None:
        raise ValueError('apred must be input')
    if telescope is None:
        raise ValueError('telescope must be input')

    load = apload.ApLoad(apred=apred,telescope=telescope)
    cal_dir = os.path.dirname(os.path.dirname(load.filename('BPM',num=0,chips='a')))+'/'
    if os.path.exists(cal_dir)==False:
        os.makedirs(cal_dir)

    parfile = cal_dir+'dailycal.par'
    with open(parfile,'a') as file:
        psf = int(psf)   # must be int
        if waves is not None:
            waves = np.array(waves)  # in case a list was input
            if waves.ndim != 2:
                waves = np.atleast_2d(waves).T
            dum,nw = waves.shape
            for i in range(nw):
                file.write('wave     99999 99999   %08i   %08i,%08i   %08i\n' % (waves[0,i],waves[0,i],waves[1,i],psf))
                file.write('lsf     99999 99999   %08i   %08i   %08i\n' % (waves[0,i],waves[0,i],psf))
                file.write('lsf     99999 99999   %08i   %08i   %08i\n' % (waves[1,i],waves[1,i],psf))
        if lsfs is not None:
            lsfs = np.atleast_1d(lsfs)
            nl = len(lsfs)
            for i in range(nl):
                file.write('lsf      99999 99999   %08i   %08i   %08i\n' % (lsfs[i],lsfs[i],psf))
    file.close()


def mkplan(ims,plate=0,mjd=None,psfid=None,fluxid=None,apred=None,telescope=None,cal=False,
           dark=False,extra=False,sky=False,plugid=None,fixfiberid=None,stars=None,
           names=None,onem=False,hmags=None,mapper_data=None,suffix=None,
           ignore=False,test=False,logger=None,configid=None,designid=None,
           fieldid=None,fps=False,force=False,fpi=None,ap3d=False,ap2d=False):
    """
    Makes plan files given input image numbers, MJD, psfid, fluxid
    includes options for dark frames, calibration frames, sky frames,
    ASDAF frames. This is called from the manually prepared MJD5.pro 
    procedures

    Parameters
    ----------
    ims : numpy int array
        List of array of exposure numbers to include in planfile.
    plate : int
        Plate number for this observation.
    mjd : int
        MJD number for this observation.
    psfid : int
        PSF cal exposure number.
    fluxid : int
        Flux cal frame exposure number.
    apred : str
        APOGEE reduction version.
    telescope : str
        APOGEE telescope.
    cal : bool, optional
        This is a calibration plan file.
    dark : bool, optional
        This is a dark sequence plan file.
    extra : bool, optional
        This is an "extra" sequence plan file.  These are "leftover" exposures that weren't
          included in any other plan files.
    sky : bool, optional
        This is a sky flat sequence plan file.
    plugid : int, optional
        Base name of the plugmap filename.
    fixfiberid : int, optional
        Fiber fixing needed (1 or 2).
    stars : numpy int array, optional
        FiberID for apo1m or ASDAF observations.
    names : numpy int array, optional
        Name of the star for apo1m or ASDAF observations.
    onem : bool, optional
        This is for a apo1m observation.
    hmags : numpy float array, optional
        2MASS H-magnitude for star in apo1m observation.
    mapper_data : str, optional
        Directory for the mapper data.
    suffix : str, optional
        Extra suffix to use (before the extension) on the planfile name.
    ignore : bool, optional
        Ignore warnings and continue.
    test : bool, optional
        Just a test.
    configid : str, optional
        The SDSS-V FPS configuration_id.
    designid : str, optional
        The SDSS-V FPS design_id.
    fieldid : str, optional
        The SDSS-V FPS field_id.
    fps : boolean, optional
        Whether the data were taken with the FPS or now.  Default is False.
    force : boolean, optional
        Force ap1dvisit to push through sky and tellurics even if there aren't
          any sky or telluric fibers.
    fpi : int, optional
        The exposure number for a full-frame FPI exposure.  Default is None.
    ap3d : boolean, optional
        This is a simple plan file for a ap3d run.
    ap2d : boolean, optional
        This is a simple plan file for a ap2d run.

    Returns
    -------
    planfile : str
         The name of the plan file created
    This creates a planfile with the given inputs and places it in
    the appropriate place in the SDSS/APOGEE directory tree.  For
    science visits this will be in $APOGEE_REDUX/{apred}/visit/{telescope}/{field}/{plate}/{mjd}/.
    Calibration, dark and sky plan files live in
    $APOGEE_REDUX/{apred}/cal/{instrument}/plan/{mjd}/

    Examples
    --------
    mkplan.mkplan(ims,plate,mjd,psfid,fluxid,apred=apred,telescope=telescope)

    By J.Holtzman, 2011?
    translated to python, D.Nidever  Oct 2020
    """

    # Logger
    if logger is None: logger=dln.basiclogger()

    if apred is None:
        raise ValueError('apred must be input')
    if telescope is None:
        raise ValueError('telescope must be input')

    if type(ims) is not list:
        ims = [ims]

    # First exposure
    im1 = dln.first_el(ims)

    if ap3d:
        logger.info('Making simple 3D plan for: '+str(im1))
    elif ap2d:
        logger.info('Making simple 2D plan for: '+str(im1))
    else:
        logger.info('Making plan for MJD: '+str(mjd))

    # Set up directories, plate, and MJD variables
    load = apload.ApLoad(apred=apred,telescope=telescope)
    caldir = os.environ['APOGEE_DRP_DIR']+'/data/cal/'
    calfile = caldir+load.instrument+'.par' 

    # Get MJD from exposure numbers if not input
    if mjd is None:
        mjd = int(load.cmjd(im1))

    # Mapper directory
    if mapper_data is None:
        if load.instrument=='apogee-n':
            mapper_data = os.environ['MAPPER_DATA_N']
        else:
            mapper_data = os.environ['MAPPER_DATA_S']


    # Planfile name and directory
    if cal:
        planfile = load.filename('CalPlan',mjd=mjd)
    elif dark:
        planfile = load.filename('DarkPlan',mjd=mjd)
    elif extra:
        planfile = load.filename('ExtraPlan',mjd=mjd)
    elif ap3d:
        planfile = load.filename('2D',num=im1,mjd=mjd,chips=True)
        planfile = os.path.dirname(planfile)+'/logs/'+os.path.basename(planfile)
        planfile = planfile.replace('2D','3DPlan').replace('.fits','.yaml')
    elif ap2d:
        planfile = load.filename('2D',num=im1,mjd=mjd,chips=True)
        planfile = os.path.dirname(planfile)+'/logs/'+os.path.basename(planfile)
        planfile = planfile.replace('2D','2DPlan').replace('.fits','.yaml')
    elif onem:
        planfile = load.filename('Plan',plate=plate,reduction=names[0],mjd=mjd) 
        if suffix is not None:
            planfile = os.path.dirname(planfile)+'/'+os.path.splitext(os.path.basename(planfile))[0]+suffix+'.yaml'
    else:
        if fps:
            planfile = load.filename('Plan',plate=plate,mjd=mjd,field=str(fieldid)) 
        else:
            planfile = load.filename('Plan',plate=plate,mjd=mjd)
    # Make sure the file ends with .yaml
    if planfile.endswith('.yaml')==False:
        planfile = planfile.replace(os.path.splitext(planfile)[-1],'.yaml')     # TEMPORARY KLUDGE!
    outdir = os.path.dirname(planfile)+'/'
    if os.path.exists(outdir)==False:
        os.makedirs(outdir)
    
    # Get calibration files for this date
    if fixfiberid is not None:
        fix0 = fixfiberid
    else:
        fix0 = None
    caldata = mkcal.getcal(calfile,mjd)
    if fix0 is not None:
        caldata['fixfiber'] = fix0

    # outplan plan file name
    if (stars is not None) & (onem==False):
        planfile = os.path.dirname(planfile)+'/'+os.path.splitext(os.path.basename(planfile))[0]+'star.yaml'
    else:
        if sky==True:
            planfile = os.path.dirname(planfile)+'/'+os.path.splitext(os.path.basename(planfile))[0]+'sky.yaml' 

    if sky==True:
        logger.info('apdailycals')
        dailycals(lsfs=ims,psf=psfid,apred=apred,telescope=telescope)
    logger.info(planfile)

    # open plan file and write header
    if os.path.exists(planfile): os.remove(planfile)
    out = {}
    out['apogee_drp_ver'] = os.environ['APOGEE_DRP_VER']
    out['telescope'] = telescope
    out['instrument'] = load.instrument
    out['plateid'] = plate
    out['fps'] = fps
    out['force'] = force
    if fpi is not None:
        out['fpi'] = str(fpi)
    else:
        out['fpi'] = 0
    if fps:
        out['configid'] = configid
        out['designid'] = designid
        out['fieldid'] = fieldid
    out['mjd'] = mjd
    out['planfile'] = os.path.basename(planfile)
    if ap3d==False and ap2d==False:
        out['logfile'] = 'apDiag-'+str(plate)+'-'+str(mjd)+'.log'
        out['plotfile'] = 'apDiag-'+str(plate)+'-'+str(mjd)+'.ps'

    # apred_vers keyword will override strict versioning using the plan file!
    out['apred_vers'] = apred

    # apo1m
    if onem==True:
        out['data_dir'] = datadir+'/'
        out['raw_dir'] = datadir+str(mjd)+'/'
        out['plate_dir'] = outdir
        out['star_dir'] = spectro_dir+'/fields/apo1m/'
        out['survey'] = 'apo1m'
        out['name'] = str(names[0]).strip()
        out['fiber'] = stars[0]
        if hmags is not None:
            out['hmag'] = hmags[0]
        out['telliter'] = 1
        if suffix!='':
            out['mjdfrac'] = 1

    # platetype
    if stars is not None or ap3d or ap2d:
        out['platetype'] = 'single'
    elif cal==True:
        out['platetype'] = 'cal'
    elif sky==True:
        out['platetype'] = 'sky'
    elif dark==True:
        out['platetype'] = 'dark'
    elif extra==True:
        out['platetype'] = 'extra'
    elif test==True:
        out['platetype'] = 'test'
    else:
        out['platetype'] = 'normal'

    sdss_path = path.Path()
    rawfile = sdss_path.full('apR',num=im1,chip='a',mjd=mjd)
    if os.path.exists(rawfile)==False:
        raise ValueError('Cannot find file '+rawfile)
    head = fits.getheader(rawfile,1)
    plateid = head.get('PLATEID')
    configid = head.get('CONFIGID')
    exptype = head.get('EXPTYPE')
    if (ignore==False) and not ap3d and not ap2d:
        if fps==False & (plate!=0) & (plate!=plateid):
            raise ValueError('plateid in header does not match plate!')

    # plugmap
    if ap3d==False and ap2d==False:
        logger.info(str(plugid))
        if plugid is None:
            rawfile = sdss_path.full('apR',num=ims[0],chip='a',mjd=mjd)
            #rawfile = load.filename('R',chip='a',num=ims[0])
            if os.path.exists(rawfile)==True:
                head = fits.getheader(rawfile,1)
                if fps==False:
                    plugid = head['NAME']
                    if type(plugid) is not str:
                        plugid = 'header'
                else:
                    plugid = 'confSummary-'+str(configid)+'.par'
            else:
                plugid = 'header'
    logger.info(str(im1))
    if (cal==False) and (dark==False) and (extra==False) and (onem==False) and (ap3d==False) and (ap2d==False):
        logger.info(str(plugid))
        tmp = plugid.split('-')
        if mjd<59556:
            if os.path.exists(mapper_data+'/'+tmp[1]+'/plPlugMapM-'+plugid+'.par')==False:
                logger.info('Cannot find plugmap file '+str(plugid))
                #spawn,'"ls" '+mapper_data+'/'+tmp[1]+'/plPlugMapA*'
                if ignore is False:
                    raise Exception
        if sky==False:
            if plate != 0:
                logger.info('getting plate data')
                plug = platedata.getdata(plate,mjd,plugid=plugid,noobject=True,mapper_data=mapper_data,apred=apred,telescope=telescope)
                loc = plug['locationid']
                spectro_dir = os.environ['APOGEE_REDUX']+'/'+apred+'/'
                if os.path.exists(spectro_dir+'fields/'+telescope+'/'+str(loc))==False:
                    os.makedirs(spectro_dir+'fields/'+telescope+'/'+str(loc))
                if fps==False:
                    field,survey,program = apload.apfield(plate,plug['locationid'])
                    out['survey'] = survey
                    out['field'] = field
                else:
                    out['survey'] = 'SDSS-V'
                    out['field'] = plug['field']
                with open(spectro_dir+'fields/'+telescope+'/'+str(loc)+'/plan-'+str(loc)+'.lis','w+') as file:
                    file.write(telescope+'/'+str(plate)+'/'+str(mjd)+'/'+os.path.basename(planfile))
                file.close()
            else:
                print('No plate/configuration information')
                plugid = 0
        out['plugmap'] = plugid

    # Use q3fix
    if 'q3fix' in caldata.keys():
        if caldata['q3fix'] is not None:
            if int(caldata['q3fix'])==1:
                out['q3fix'] = 1

    # Calibration frames to use
    calnames = ['det','bpm','littrow','persist','persistmodel','dark','flat']
    if ap3d==False:
        calnames += ['sparse','fiber','badfiber','fixfiber','response','wave','lsf']
    for c in calnames:
        val = caldata[c]
        if str(val).isdigit(): val=int(val)
        out[c+'id'] = val
    # We use multiwaveid for waveid
    if ap3d==False:
        waveid = caldata['multiwave']
        if str(waveid).isdigit(): waveid=int(waveid)
        out['waveid'] = waveid
        # Input PSFID and FLUXID
        if psfid is not None:
            out['psfid'] = psfid
        # Get PSF calibration files
        else:
            psffile = load.filename('PSF',num=0,mjd=mjd,chips=True)
            psffile = psffile.replace('PSF-','PSF-b-')
            base = ('%8d' % im1)[0:4]
            psffiles = glob(psffile.replace('-00000000','-'+base+'????'))
            if len(psffiles)==0:
                raise ValueError('No PSF files for MJD='+str(mjd))
            psfnum = [os.path.basename(p)[8:16] for p in psffiles] 
            si = np.argsort(np.abs(np.array(psfnum).astype(int)-int(im1)))
            psfid = np.array(psfnum)[si][0]
            out['psfid'] = str(psfid)
        # Flux calibration file
        if fluxid is not None:
            out['fluxid'] = fluxid
        # Get Flux calibration file
        else:
            fluxfile = load.filename('Flux',num=0,mjd=mjd,chips=True)
            fluxfile = psffile.replace('Flux-','PSF-b-')
            base = ('%8d' % im1)[0:4]
            fluxfiles = glob(psffile.replace('-00000000','-'+base+'????'))    
            if len(fluxfiles)==0:
                raise ValueError('No Flux files for MJD='+str(mjd))
            fluxnum = [os.path.basename(f)[8:16] for f in fluxfiles] 
            si = np.argsort(np.abs(np.array(fluxnum).astype(int)-int(im1)))
            fluxid = np.array(fluxnum)[si][0]
            out['fluxid'] = str(fluxid)

    # object frames
    aplist = []
    for i in range(len(ims)):
        aplist1 = {}
        if ims[i]>0:
            if stars is not None:
                star = stars[i]
                name = names[i]
            else:
                star = -1
                name = 'none'
        if ap3d:
            flavor = 'object'
            rawfile = sdss_path.full('apR',num=im1,chip='a',mjd=mjd)
            if os.path.exists(rawfile)==False:
                print(rawfile,' NOT FOUND')
            head = fits.getheader(rawfile,1)
            exptype = head.get('EXPTYPE')
            if exptype.lower()=='domeflat':
                flavor = 'psf'
        else:
            flavor = 'object'
        aplist1 = {'plateid':plate, 'mjd':mjd, 'flavor':flavor, 'name':ims[i], 'single':star, 'singlename':name}
        aplist.append(aplist1)
    out['APEXP'] = aplist

    # Write to yaml file
    with open(planfile,'w') as ofile:
        dum = yaml.dump(out,ofile,default_flow_style=False, sort_keys=False)
    os.chmod(planfile, 0o664)

    return planfile


def make_mjd5_yaml(mjd,apred,telescope,clobber=False,logger=None):
    """
    Make a MJD5 yaml file that can be used to create plan files.

    Parameters
    ----------
    mjd : int
        MJD number for this night.
    apred : str
        APOGEE reduction version.
    telescope : str
        APOGEE telescope: apo25m, apo1m, lco25m.
    clobber : bool
        Overwrite any existing files.

    Returns
    -------
    out : list of dictionaries
       The list of dictionaries that contain the information needed to
       make the plan files.
    planfiles : list of str
       The names of the plan files that would be created.
    The yaml files are also written to disk in APOGEEREDUCEPLAN product
    directory.

    Examples
    --------
    out,planfiles = mkplan.make_mjd5_yaml(57680,'t15','apo25m')

    By J.Holtzman, 2011
    translated/rewritten, D.Nidever  Oct2020
    """

    # Logger
    if logger is None: logger=dln.basiclogger()

    logger.info('Making MJD5.yaml file for MJD='+str(mjd))

    load = apload.ApLoad(apred=apred,telescope=telescope)
    datadir = {'apo25m':os.environ['APOGEE_DATA_N'],'apo1m':os.environ['APOGEE_DATA_N'],
               'lco25m':os.environ['APOGEE_DATA_S']}[telescope]
    observatory = telescope[0:3]
    chips = ['a','b','c']

    # Output file/directory
    outfile = os.environ['APOGEEREDUCEPLAN_DIR']+'/yaml/'+telescope+'/'+telescope+'_'+str(mjd)+'auto.yaml'
    if os.path.exists(os.path.dirname(outfile))==False:
        os.makedirs(os.path.dirname(outfile))
    # File already exists and clobber not set
    if os.path.exists(outfile) and clobber==False:
        logger.info(outfile+' already EXISTS and clobber==False')
        return [],[]

    # Get the exposures and info about them
    expinfo = info.expinfo(observatory=observatory,mjd5=mjd)
    if expinfo is None:
        logger.info('No exposures for MJD='+str(mjd))
        return [],[]
    expinfo = expinfo[np.argsort(expinfo['num'])]   # sort
    expinfo = Table(expinfo)
    nfiles = len(expinfo)
    logger.info(str(nfiles)+' exposures found')
    # No exposures
    if nfiles==0:
        return [],[]

    # SDSS-V FPS, use configid for plateid
    fps = False
    if int(mjd)>=59556:
        logger.info('SDSS-V FPS data.  Using configid for plateid')
        expinfo['plateid'] = expinfo['configid']
        fps = True

    # Print summary information about the data
    expindex = dln.create_index(expinfo['exptype'])
    for i in range(len(expindex['value'])):
        logger.info('  '+expindex['value'][i]+': '+str(expindex['num'][i]))
    objind, = np.where(expinfo['exptype']=='OBJECT')
    if len(objind)>0:
        plates = np.unique(expinfo['plateid'][objind])
        logger.info('Observations of '+str(len(plates))+' plates/configs')
        plateindex = dln.create_index(expinfo['plateid'][objind])
        for i in range(len(plateindex['value'])):
            logger.info('  '+plateindex['value'][i]+': '+str(plateindex['num'][i])) 

    # Do QA check of the files
    qachk = check.check(expinfo['num'],apred,telescope,verbose=False)

    # Get gang connector plugging groups
    # Loop over the exposures and mark time periods of constant dither position
    expinfo['pluggroup'] = -1
    currentgangstate = expinfo['gangstate'][0]
    pluggroup = 0
    for e in range(len(expinfo)):
        if expinfo['gangstate'][e] == currentgangstate:
            if expinfo['gangstate'][e]=='FPS':
                expinfo['pluggroup'][e] = pluggroup
        else:
            # Podium -> FPS, new plugging group
            if currentgangstate=='Podium':
                pluggroup += 1
                currentgangstate = expinfo['gangstate'][e]
                expinfo['pluggroup'][e] = pluggroup
            # FPS -> Podium
            else:
                expinfo['pluggroup'][e] = -1

    # Get domeflat and quartzflats for this night
    domeind, = np.where((expinfo['exptype']=='DOMEFLAT') & (qachk['okay']==True) )
    dome = list(expinfo['num'][domeind].astype(int))
    domepluggroup = expinfo['pluggroup'][domeind]
    quartzind, = np.where((expinfo['exptype']=='QUARTZFLAT') & (qachk['okay']==True))
    quartz = list(expinfo['num'][quartzind].astype(int))

    # Check which apPSF and apFlux files exist and can be used for calibration files
    # Which domeflat apPSF files exist
    psfdome_exist = np.zeros(len(dome),bool)
    for j in range(len(dome)):
        psffile = load.filename('PSF',num=dome[j],chips=True)
        psffiles = [psffile.replace('apPSF-','apPSF-'+ch+'-') for ch in chips]
        exist = [os.path.exists(pf) for pf in psffiles]
        if np.sum(np.array(exist))==3:
            psfdome_exist[j] = True
    gddome, = np.where(psfdome_exist == True)
    if len(gddome)>0:
        psfdome = list(np.array(dome)[gddome])
        logger.info('Available domeflat apPSF: '+str(psfdome))
    else:
        psfdome = []
        logger.info('No domeflat apPSF files exist')
    # Which quartz apPSF files exist
    psfquartz_exist = np.zeros(len(quartz),bool)
    for j in range(len(quartz)):
        psffile = load.filename('PSF',num=quartz[j],chips=True)
        psffiles = [psffile.replace('apPSF-','apPSF-'+ch+'-') for ch in chips]
        exist = [os.path.exists(pf) for pf in psffiles]
        if np.sum(np.array(exist))==3:
            psfquartz_exist[j] = True
    gdquartz, = np.where(psfquartz_exist == True)
    if len(gdquartz)>0:
        psfquartz = list(np.array(quartz)[gdquartz])
        logger.info('Available quartzflat apPSF:: '+str(psfquartz))
    else:
        psfquartz = []
        logger.info('No quartzflat apPSF files exist')
    # Which apFlux files exist
    flux_exist = np.zeros(len(dome),bool)
    for j in range(len(dome)):
        fluxfile = load.filename('Flux',num=dome[j],chips=True)
        fluxfiles = [fluxfile.replace('apFlux-','apFlux-'+ch+'-') for ch in chips]
        exist = [os.path.exists(ff) for ff in fluxfiles]
        if np.sum(np.array(exist))==3:
            flux_exist[j] = True
    gdflux, = np.where(flux_exist == True)
    if len(gdflux)>0:
        flux = list(np.array(dome)[gdflux])
        fluxpluggroup = domepluggroup[gdflux]
        logger.info('Available apFlux: '+str(flux))
    else:
        flux = []
        fluxpluggroup = []
        logger.info('No quartz apPSF files exist')

    if len(psfdome)==0 and len(psfquartz)==0:
        logger.info('No apPSF files for this night exist.  They will be created as needed')
    if len(flux)==0:
        logger.info('No apFlux files for this night exist.  They will be created as needed')

    # Check FPI cals for this night
    fpiind, = np.where((expinfo['exptype']=='FPI') & (qachk['okay']==True))
    fpinum = list(expinfo['num'][fpiind].astype(int))
    # Check that the FPI calibration file exists
    fpi_exist = np.zeros(len(fpinum),bool)
    for j in range(len(fpinum)):
        fpifile = os.path.dirname(load.filename('Wave',num=0,chips=True))+'/apWaveFPI-%s-%8d.fits' % (str(mjd),fpinum[j])
        fpifiles = [fpifile.replace('apWaveFPI-','apWaveFPI-'+ch+'-') for ch in chips]
        exist = [os.path.exists(ff) for ff in fpifiles]
        if np.sum(np.array(exist))==3:
            fpi_exist[j] = True
    gdfpi, = np.where(fpi_exist == True)
    if len(gdfpi)>0:
        fpi = list(np.array(fpinum)[gdfpi])
        logger.info('Available apWaveFPI: '+str(fpi))
    else:
        fpi = []
        logger.info('No apWaveFPI files exist')

    # Scan through all files, accumulate IDs of the various types
    dark, cal, exp, exppluggroup, sky, extra, calpsfid = [], [], [], [], [], [], None
    domeused, out, planfiles = [], [], []
    for i in range(nfiles):
        # Load image number in variable according to exptype and nreads
        #  discard images that have problems
        if qachk['okay'][i]==False:
            badbits = ', '.join(check.bitmask(qachk['mask'][i])[1])
            logger.info(str(expinfo['num'][i])+' %-13s has QA problems:  %-s ' % (expinfo['exptype'][i],badbits))
        # Dark
        elif (expinfo['exptype'][i]=='DARK') and (qachk['okay'][i]==True):
            dark.append(int(expinfo['num'][i]))
        # Internal flat
        #   reduced only to 2D, hence treated like darks
        elif (expinfo['exptype'][i]=='INTERNALFLAT') and (qachk['okay'][i]==True):
            dark.append(int(expinfo['num'][i]))
        # Quartz flat
        elif (expinfo['exptype'][i]=='QUARTZFLAT') and (qachk['okay'][i]==True):
            cal.append(int(expinfo['num'][i]))
            calpsfid = int(expinfo['num'][i])
        # Arc lamps
        elif (expinfo['exptype'][i]=='ARCLAMP') and (qachk['okay'][i]==True):
            cal.append(int(expinfo['num'][i]))
        # Sky frame
        #   identify sky frames as object frames with 10<nread<13
        #elif (expinfo['exptype'][i]=='OBJECT') and (expinfo['nread'][i]<13 and expinfo['nread'][i]>10) and (qachk['okay'][i]==True):
        elif (expinfo['exptype'][i]=='SKYFLAT') and (qachk['okay'][i]==True):
            sky.append(int(expinfo['num'][i]))
        # Object exposure, used to be >15
        elif (expinfo['exptype'][i]=='OBJECT') and (expinfo['nread'][i]>13) and (qachk['okay'][i]==True):
            exp.append(int(expinfo['num'][i]))
            exppluggroup.append(expinfo['pluggroup'][i])
        # Dome flat, dealt with above
        elif (expinfo['exptype'][i]=='DOMEFLAT'):
            pass
        # FPI, dealt with above
        elif (expinfo['exptype'][i]=='FPI'):
            pass
        else:
            print('Unknown exposure: ',expinfo['num'][i],expinfo['exptype'][i],expinfo['nread'][i],' adding to extra plan file')
            extra.append(int(expinfo['num'][i]))

        # End of this plate block
        #  if plateid changed or last exposure
        platechange = expinfo['plateid'][i] != expinfo['plateid'][np.minimum(i+1,nfiles-1)]
        # We don't need a domeflat with each field visit in the SDSS-V FPS era since
        #   we will use the domeflat lookup table
        if (platechange or i==nfiles-1) and expinfo['plateid'][i]!='' and len(exp)>0 and (len(dome)>0 or fps):
            # Object plate visit
            if fps:
                plate = expinfo['configid'][i]
                try:
                    plate = int(plate)
                except:
                    plate = 0
            else:
                plate = int(expinfo['plateid'][i])

            # Get PSF calibration file
            #  use quartz flats if possible, make sure they exist
            # Try to use ones that exist
            if len(psfquartz)>0 or len(psfdome)>0:
                if len(psfquartz)>0:
                    bestind = np.argsort(np.abs(np.array(psfquartz)-int(exp[0])))
                    psf1 = int(psfquartz[bestind[0]])
                elif len(psfdome)>0:
                    bestind = np.argsort(np.abs(np.array(psfdome)-int(exp[0])))
                    psf1 = int(psfdome[bestind[0]])
            # No apPSF files exist, they'll need to be made
            else:
                if len(quartz)>0:
                    bestind = np.argsort(np.abs(np.array(quartz)-int(exp[0])))
                    psf1 = int(quartz[bestind[0]])
                elif len(dome)>0:
                    bestind = np.argsort(np.abs(np.array(dome)-int(exp[0])))
                    psf1 = int(dome[bestind[0]])
                else:
                    psf1 = None
            # Flux calibration file
            # Use existing apFlux calibration file
            if len(flux)>0:
                # Use apFlux for this gang connector plugging
                samegroup, = np.where(np.array(fluxpluggroup)==exppluggroup[0])
                # Some apFlux for this plugging group
                if len(samegroup)>0:
                    bestind = np.argsort(np.abs(np.array(flux)[samegroup]-int(exp[0])))
                    flux1 = int(np.array(flux)[samegroup][bestind[0]])
                # No apFlux for this plugging, use closest apFlux
                else:
                    logger.info('No apFlux for this gang connector plugging. Using closest apFlux')
                    bestind = np.argsort(np.abs(np.array(flux)-int(exp[0])))
                    flux1 = int(flux[bestind[0]])
            # No apFlux calibration file exists yet, it'll need to be made
            else:
                # Use domeflat for flux calibration
                if len(dome)>0:
                    # Use apFlux for this gang connector plugging
                    samegroup, = np.where(np.array(domepluggroup)==exppluggroup[0])
                    # Some dome for this plugging group
                    if len(samegroup)>0:
                        bestind = np.argsort(np.abs(np.array(dome)[samegroup]-int(exp[0])))
                        flux1 = int(np.array(dome)[samegroup][bestind[0]])
                    # No domeflat for this plugging, use closest domeflat
                    else:
                        logger.info('No domeflat for this gang connector plugging. Using closest domeflat')
                        bestind = np.argsort(np.abs(np.array(dome)-int(exp[0])))
                        flux1 = int(dome[bestind[0]])
                else:
                    flux1 = None
            # Put everything together for this configuration/plate
            if fps:
                objplan = {'apred':str(apred), 'telescope':str(load.telescope), 'mjd':int(mjd),
                           'plate':plate, 'psfid':psf1, 'fluxid':flux1, 'ims':exp, 'fps':fps}
                if len(fpi)>0 and mjd>=59604:  # two-FPI fibers weren't used routinely until 59604
                    objplan['fpi'] = str(fpi[0])
                objplan['configid'] = str(expinfo['configid'][i])
                objplan['designid'] = str(expinfo['designid'][i])
                objplan['fieldid'] = str(expinfo['fieldid'][i])
                #print('Setting force=True for now')
                objplan['force'] = True
            else:  # plates
                objplan = {'apred':str(apred), 'telescope':str(load.telescope), 'mjd':int(mjd),
                           'plate':plate, 'psfid':psf1, 'fluxid':flux1, 'ims':exp, 'fps':fps}
            out.append(objplan)
            planfile = load.filename('Plan',plate=plate,field=expinfo['fieldid'][i],mjd=mjd)
            planfiles.append(planfile)
            exp = []

            # Sky exposures
            #   use same cals as for object
            if len(sky)>0:
                skyplan = {'apred':str(apred), 'telescope':str(load.telescope), 'mjd':int(mjd),
                           'plate':plate, 'psfid':psf1, 'fluxid':flux1, 
                           'ims':sky, 'fps':fps, 'sky':True}
                if fps:
                    skyplan['configid'] = str(expinfo['configid'][i])
                    skyplan['designid'] = str(expinfo['designid'][i])
                    skyplan['fieldid'] = str(expinfo['fieldid'][i])
                out.append(skyplan)
                skyplanfile = planfile.replace('.yaml','sky.yaml')
                planfiles.append(skyplanfile)
                sky = []

    # Some object exposures not used in visit plan files
    if len(exp)>0:
        logger.info(str(len(exp))+' unused object exposures in visit plan files.  Adding them to ExtraPlan')
        extra += exp
        exp = []
    # Some domeflat exposures not used in visit plan files
    #if len(dome)>0 and len(dome)>len(domeused):
    #    # add unused dome flats to ExtraPlan
    #    dometoadd = []
    #    for d in dome:
    #        if d not in domeused:
    #            dometoadd.append(d)
    #    print(str(len(dometoadd))+' unused dome flat exposures in visit plan files.  Adding them to ExtraPlan')
    #    extra += dometoadd
    #    dome = []

    # Dark frame information
    cplate = '0000'
    if len(dark)>0:
        darkplan = {'apred':str(apred), 'telescope':str(load.telescope), 'mjd':int(mjd),
                    'plate':0, 'psfid':0, 'fluxid':0, 'ims':dark, 'fps':fps, 'dark':True}
        if fps:
            darkplan['configid'] = str(expinfo['configid'][i])
            darkplan['designid'] = str(expinfo['designid'][i])
            darkplan['fieldid'] = str(expinfo['fieldid'][i])
        out.append(darkplan)
        planfile = load.filename('DarkPlan',mjd=mjd)
        planfiles.append(planfile)
    # Calibration frame information
    if len(cal)>0 and calpsfid is not None:
        calplan = {'apred':str(apred), 'telescope':str(load.telescope), 'mjd':int(mjd),
                   'plate':0, 'psfid':calpsfid, 'fluxid':calpsfid, 'ims':cal, 'fps':fps,
                   'cal':True}
        if fps:
            calplan['configid'] = str(expinfo['configid'][i])
            calplan['designid'] = str(expinfo['designid'][i])
            calplan['fieldid'] = str(expinfo['fieldid'][i])
        out.append(calplan)
        planfile = load.filename('CalPlan',mjd=mjd)
        planfiles.append(planfile)
    # "extra" frames information
    if len(extra)>0:
        extraplan = {'apred':str(apred), 'telescope':str(load.telescope), 'mjd':int(mjd),
                     'plate':0, 'psfid':calpsfid, 'fluxid':calpsfid, 'ims':extra, 'fps':fps,
                     'extra':True}
        if fps:
            extraplan['configid'] = str(expinfo['configid'][i])
            extraplan['designid'] = str(expinfo['designid'][i])
            extraplan['fieldid'] = str(expinfo['fieldid'][i])
        out.append(extraplan)
        planfile = load.filename('ExtraPlan',mjd=mjd)
        planfiles.append(planfile)

    # Write out the MJD5 file
    if os.path.exists(outfile): os.remove(outfile)
    logger.info('Writing MJD5.yaml file to '+outfile)
    with open(outfile,'w') as file:
        dum = yaml.dump(out,file,default_flow_style=False, sort_keys=False)
    # Copy it to the non-"auto" version
    outfile2 = outfile.replace('auto','')
    if os.path.exists(outfile2)==False:
        shutil.copyfile(outfile,outfile2)
    else:
        logger.info(outfile2+' already exists.  NOT overwriting.')

    return out, planfiles


def run_mjd5_yaml(yamlfile,logger=None):
    """
    Run the MJD5 yaml file and create the relevant plan files.

    Parameters
    ----------
    yamlfile : str
         Name of the MJD5 yaml file.

    Returns
    -------
    planfiles : list of str
         List of the created plan file names.

    Examples
    --------
    planfiles = mkplan.run_mjd5_yaml(yamlfile)

    By D.Nidever, Oct 2020
    """

    # Logger
    if logger is None: logger=dln.basiclogger()
    
    if os.path.exists(yamlfile)==False:
        raise ValueError(yamlfile+' NOT FOUND')

    # Load the yaml file
    with open(yamlfile) as file:
        data = yaml.full_load(file)

    if type(data) is not list:
        data = [data]
    ndata = len(data)
    logger.info('Information for '+str(ndata)+' plan files')

    # Loop over the plan blocks and run mkplan()
    planfiles = []
    for i in range(ndata):
        logger.info(' ')
        logger.info('Plan file '+str(i+1))
        logger.info('------------')
        pargs = data[i]
        ims = pargs.pop('ims')
        plate = pargs.pop('plate')
        mjd = pargs.pop('mjd')
        psfid = pargs.pop('psfid')
        fluxid = pargs.pop('fluxid')
        planfile = mkplan(ims,plate,mjd,psfid,fluxid,**pargs,logger=logger)
        planfiles.append(planfile)

    return planfiles
