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
from apogee_drp.utils import spectra,yanny
from apogee_drp.plan import mkslurm

def args2dict(**kwargs):
    return kwargs

def parse_idl_line(line):
    """ This parses through a line of IDL and returns a list of 'components'."""

    # How about interpreting it as a dictionary
    # or as **kwargs for a dummy function

    # mkplan,ims,plate,mjd,psfid,fluxid,vers=vers,stars=[204,191],names=['J03322626+4603247','J03322626+4603247'] 

    pass

def fixcontinuation(lines):
    """ Fix continuation lines."""

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

def removekeyword(line,key):
    """ Remove an IDL keyword input like ,/cal in a line."""

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

def removecomments(lines):
    """ Remove IDL comments from a list of lines."""

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
    """ Replace IDL code in lines with the results."""

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
    Translate an IDL MJD5.pro script file to yaml.

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
    lines = fixcontinuation(lines)
    # Remove comments
    lines = removecomments(lines)

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
    telescopeline = "telescope:'"+telescope+"'"

    # Get MJD
    ind,nind = dln.where(lines.strip().lower().find('mjd=')==0)
    if nind==0:
        raise ValueError('No MJD line found')
    mjdline = lines[ind[0]]
    mjd = int(mjdline[mjdline.find('=')+1:])
    mjdline = mjdline.replace('=',':')

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
    flines = []

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
        flines.append(telescopeline)
        # Add MJD line
        flines.append(mjdline)
        # Assume all lines in this block except for mkplan are key: value pairs
        kvlines = lines1[0:-1]
        kvlines = kvlines.replace('=',':')        # replace = with :
        kvlines = kvlines[kvlines.strip('')!='']  # remove any blank lines
        flines += list(kvlines)
        # Deal with mkplan lines
        planline = lines1[-1]
        # Trim off the first bit that's always the same, "mkplan,ims,plate,mjd,psfid,fluxid,"
        planline = planline[planline.lower().find('fluxid')+7:]
        # Remove vers=vers if it's there
        if planline.lower().find('vers=vers')==0:
            planline = planline[9:]


        #What's this???
        #@calsetup
        #waveid=1370096

        # Deal with keywords
        if planline!='':
            if planline[0]==',':
                planline = planline[1:]
            # Add lines for sky, dark, cal
            if planline.lower().find('/sky')>-1:
                flines.append('sky: True')
                planline = removekeyword(planline,'/sky')  # Trim off /sky
            if planline.lower().find('/dark')>-1:
                flines.append('dark: True')
                planline = removekeyword(planline,'/dark')  # Trim off /dark
            if planline.lower().find('/cal')>-1:
                flines.append('cal: True')
                planline = removekeyword(planline,'/cal')  # Trim off /cal

        # Deal with remaining arguments
        if planline!='':
            # Return leftover line as a dictionary
            import pdb; pdb.set_trace()
            exec("args=args2dict("+planline+")")
            # Loop over keys and add them
            for k in args.keys():
                val = args[k]
                if (type(val) is int) | (type(val) is str):
                    flines.append(k+'='+str(val))
                else:
                    flines.append(k+'='+str(list(val)))

            #if planline.lower().find('plugid=')>-1:        
            #    # apo25m_57043.pro:mkplan,ims,plate,mjd,psfid,fluxid,vers=vers,plugid='8257-57043-50'
            #    flines.append()
            #if planline.lower().find('plug=')>-1:        
            #    # apo25m_55871.pro:mkplan,ims,plate,mjd,psfid,fluxid,vers=vers,plug='5267-55870-01B'
            #    flines.append()
            #if planline.lower().find('fixfiberid=')>-1:        
            #    # apo25m_56773.pro:mkplan,ims,plate,mjd,psfid,fluxid,vers=vers,fixfiberid=1
            #    flines.append()
            #if planline.lower().find('test=')>-1:        
            #    # apo25m_55846.pro:mkplan,ims,plate,mjd,psfid,fluxid,vers=vers,/test
            #    flines.append()
            #if planline.lower().find('stars=')>-1:
            #    # apo25m_55815.pro:mkplan,ims,plate,mjd,psfid,fluxid,vers=vers,stars=[204,191],names=['J03322626+4603247','J03322626+4603247']
            #    flines.append()
            #if planline.lower().find('names=')>-1:
            #    # apo25m_55815.pro:mkplan,ims,plate,mjd,psfid,fluxid,vers=vers,stars=[204,191],names=['J03322626+4603247','J03322626+4603247']
            #    flines.append()


        # Add blank line
        flines.append(' ')

    # Make sure the mkplan blocks are separated
    
    # add in cal, dark, sky values/lines

    import pdb; pdb.set_trace()


def make_mjd5_yaml(mjd):
    """ Make a MJD5 yaml file."""
    pass


def run_mjd5_yaml(yamlfile):
    """ Run the MJD5 yaml file and create the relevant plan files."""
    pass


def mkplan(ims,plate,mjd,psfid,fluxid,cal=False,dark=False,sky=False):
    """ Make a plan file."""
    pass
