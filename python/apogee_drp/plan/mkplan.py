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

def translate_idl_mjd5_script(scriptfile):
    """ Translate an IDL MJD5.pro script file to yaml."""

    # Check that the file exists
    if os.path.exists(scriptfile)==False:
        raise ValueError(scriptfile+" NOT FOUND")

    # Load the file
    lines = dln.readlines(scriptfile)
    lines = np.char.array(lines)

    # Example file, top part of apo25m_59085.pro
    #apsetver,telescope='apo25m'
    #mjd=59085
    #plate=11950
    #psfid=35230030
    #fluxid=35230030
    #ims=[35230018,35230019,35230020,35230021,35230022,35230023,35230024,35230025,35230026,35230027,35230028,35230029]
    #mkplan,ims,plate,mjd,psfid,fluxid,vers=vers
    #
    #;these are not sky frames
    #plate = 12767
    #psfid=35230015
    #fluxid=35230015
    #ims=[35230011,35230012,35230013,35230014]
    #mkplan,ims,plate,mjd,psfid,fluxid,vers=vers;,/sky
    #
    #plate=12673
    #psfid=35230037
    #fluxid=35230037
    #ims=[35230033,35230034,35230035,35230036]
    #mkplan,ims,plate,mjd,psfid,fluxid,vers=vers

    # Remove comment lines, apvers and mjd line
    gd,ngd = dln.where( (lines.strip('').startswith(';')==False) &
                        (lines.lower().find('apsetver')==-1) &
                        (lines.lower().startswith('mjd=')==False) )
    lines = lines[gd]

    # Replace all = with :
    gd,ngd = dln.where(lines.lower().find('mkplan')==-1)
    lines[gd] = lines[gd].replace('=',': ')

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
