import copy
import numpy as np
import os
import glob
import pdb
import subprocess
from dlnpyutils import utils as dln
#import yaml
#try:
#    from yaml import CLoader as Loader, CDumper as Dumper
#except ImportError:
#    from yaml import Loader, Dumper

#from sdss import yanny
#from apogee_drp.speclib import atmos
from ..utils import yanny,apload
#from apogee_drp.plan import mkslurm

def loadplan(planfile,verbose=False,expand=False,plugmapm=False):
    """
    Load an APOGEE plan file.

    Parameters
    ----------
    planfile  The absolute path of the plan file
    /expand   Expand the calibration ID paths and add in
                directories.

    Returns
    -------
    planstr   The plan structure with all the
    relevant information
    /verbose  Print a lot of information to the screen
    /silent   Don't print anything to the screen
    /stp      Stop at the end of the program
    =error    The error message if one occurred

    Example
    -------
    plan = loadplan(planfile)
    
    By D.Nidever  May 2010
    """

    # More than one file
    if dln.size(planefile)>1:
        raise ValueError('Only plan file at a time.')

    # Check that the plan file exists
    if os.path.exists(planfile) == False:
        raise ValueError(planfile+' NOT FOUND')

    # Load the plan file
    plandata = yanny.yanny(planfile,np=True)

    # Make sure APEXP exists
    if 'APEXP' not in plandta.keys():
        raise ValueError('No APEXP structure in plan file '+planfile)

    # Add paths to the calibration IDs and add directories
    #if expand==True:
    #
    #    load = apload.ApLoad(apred=plandata['apred_vers'].strip("'"),
    #                         telescope=plandata['telescope'].strip(","),
    #                         instrument=plandata['instrument'].strip(","))

    return plandata
