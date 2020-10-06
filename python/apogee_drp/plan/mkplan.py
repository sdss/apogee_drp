import copy
import numpy as np
import os
import glob
import pdb
import subprocess
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from dlnpyutils import utils as dln
from apogee_drp.utils import spectra,yanny,apload,platedata
from apogee_drp.plan import mkslurm
from apogee_drp.apred import mkcal
from sdss_access.path import path
from astropy.io import fits

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
    Replace IDL code in lines with the results.
    This is a small helper function for translate_idl_mjd5_script().
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


def make_mjd5_yaml(mjd):
    """ Make a MJD5 yaml file."""
    pass


def run_mjd5_yaml(yamlfile):
    """ Run the MJD5 yaml file and create the relevant plan files."""
    pass


def mkplan(ims,plate,mjd,psfid,fluxid,cal=False,dark=False,sky=False,
           vers=None,telescope=None,plugid=None,fixfiberid=None,stars=None,
           names=None,onem=None,hmags=None,mapper_data=None,suffix=None,
           ignore=False,test=False):
    """
    Makes plan files given input image numbers, MJD, psfid, fluxid
    includes options for dark frames, calibration frames, sky frames,
    ASDAF frames. This is called from the manually prepared MJD5.pro 
    procedures
    """

    print('Making plan for MJD: ',mjd)

    # Set up directories, plate, and MJD variables
    load = apload.ApLoad(apred=vers,telescope=telescope)
    caldir = os.environ['APOGEE_DRP_DIR']+'/data/cal/'
    calfile = caldir+load.instrument+'.par' 

    # Mapper directory
    if mapper_data is None:
        if load.instrument=='apogee-n':
            mapper_data = os.environ['MAPPER_DATA_N']
        else:
            mapper_data = os.environ['MAPPER_DATA_S']

    # Planfile name and directory
    if cal==True:
        planfile = load.filename('CalPlan',mjd=mjd)
    elif dark==True:
        planfile = load.filename('DarkPlan',mjd=mjd)
    elif onem==True:
        planfile = load.filename('Plan',plate=plate,reduction=names[0],mjd=mjd) 
        if suffix is not None:
            planfile = os.path.dirname(planfile)+'/'+os.path.splitext(os.path.basename(planfile,'.yaml'))[0]+suffix+'.yaml'
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
        fixfiberid = fix0

    # outplan plan file name
    if (stars is not None) & (onem is None):
        planfile = os.path.dirname(planfile)+'/'+os.path.basename(planfile,'.yaml')+'star.yaml'
    else:
        if sky==True:
            planfile = os.path.dirname(planfile)+'/'+os.path.basename(planfile,'.yaml')+'sky.yaml' 

    if sky==True:
        print('apdailycals')
        import pdb; pdb.set_trace()
        dailycals(lsfs=ims,psf=psfid)
    print(planfile)

    # open plan file and write header
    if os.path.exists(planfile): os.remove(planfile)
    out = {}
    out['apogee_drp_ver'] = os.environ['APOGEE_DRP_VER']
    out['telescope'] = telescope
    out['instrument'] = load.instrument
    out['plateid'] = plate
    out['mjd'] = mjd
    out['planfile'] = os.path.basename(planfile)
    out['logfile'] = 'apDiag-'+str(plate)+'-'+str(mjd)+'.log'
    out['plotfile'] = 'apDiag-'+str(plate)+'-'+str(mjd)+'.ps'

    # apred_vers keyword will override strict versioning using the plan file!
    out['apred_vers'] = vers

    # apo1m
    if onem is True:
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
    if stars is not None:
        out['platetype'] = 'single'
    elif cal==True:
        out['platetype'] = 'cal'
    elif sky==True:
        out['platetype'] = 'sky'
    elif dark==True:
        out['platetype'] = 'dark'
    elif test==True:
        out['platetype'] = 'test'
    else:
        out['platetype'] = 'normal'

    # Note that q3fix is now done in ap3d.pro, not here!!
    if (mjd>56930) & (mjd<57600):
        out['q3fix'] = 1

    sdss_path = path.Path()
    rawfile = sdss_path.full('apR',num=ims[0],chip='a',mjd=mjd)
    #rawfile = load.filename('R',chip='a',num=ims[0])
    if os.path.exists(rawfile)==False:
        raise ValueError('Cannot find file '+rawfile)
    head = fits.getheader(rawfile,1)
    plateid = head['PLATEID']
    if (ignore==False):
        if (plate!=0) & (plate!=plateid):
            raise ValueError('plateid in header does not match plate!')

    # plugmap
    print(plugid)
    if plugid is None:
        rawfile = sdss_path.full('apR',num=ims[0],chip='a',mjd=mjd)
        #rawfile = load.filename('R',chip='a',num=ims[0])
        if os.path.exists(rawfile)==True:
            head = fits.getheader(rawfile,1)
            plugid = head['NAME']
            if type(plugid) is not str:
                plugid = 'header'
        else:
            plugid = 'header'
    print(ims[0])
    print(plugid)
    if (cal is None) & (dark is None) & (onem is None):
        tmp = plugid.split('=')
        if os.path.exists(mapper_data+'/'+tmp[1]+'/plPlugMapM-'+plugid+'.par')==False:
            print('Cannot find plugmap file ',plugid)
            #spawn,'"ls" '+mapper_data+'/'+tmp[1]+'/plPlugMapA*'
            if ignore is False:
                raise Exception
    if sky==False:
        print('getplatedata')
        plug = platedata.getdata(plate,mjd,plugid=plugid,noobject=True,mapper_data=mapper_data,apred=vers,telescope=telescope)
        loc = plug['locationid']
        spectro_dir = os.environ['APOGEE_REDUX']+'/'+vers+'/'
        if os.path.exists(spectro_dir+'fields/'+telescope+'/'+str(loc))==False:
            os.makedirs(spectro_dir+'fields/'+telescope+'/'+str(loc))
        field,survey,program = apload.apfield(plate,plug['locationid'])
        out['survey'] = survey
        with open(spectro_dir+'fields/'+telescope+'/'+str(loc)+'/plan-'+str(loc)+'.lis','w+') as file:
            file.write(telescope+'/'+str(plate)+'/'+str(mjd)+'/'+os.path.basename(planfile))
        file.close()
    out['plugmap'] = plugid

    # Calibration frames to use
    calnames = ['det','bpm','littrow','persist','persistmodel','dark','flat',
                'sparse','fiber','badfiber','fixfiber','response','wave','lsf']
    for c in calnames:
        val = caldata[c]
        if str(val).isdigit(): val=int(val)
        out[c+'id'] = val
    # We use multiwaveid for waveid
    waveid = caldata['multiwave']
    if str(waveid).isdigit(): waveid=int(waveid)
    out['waveid'] = waveid
    # Input PSFID and FLUXID
    out['psfid'] = psfid
    out['fluxid'] = fluxid

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
        aplist1 = {'plateid':plate, 'mjd':mjd, 'flavor':'object', 'name':ims[i], 'single':star, 'singlename':name}
        aplist.append(aplist1)
    out['APEXP'] = aplist

    # Write to yaml file
    with open(planfile,'w') as ofile:
        dum = yaml.dump(out,ofile,default_flow_style=False, sort_keys=False)
    os.chmod(planfile, 0o664)
