import numpy
import os
import glob
from dlnpyutils import utils as dln
import yaml
#try:
#    from yaml import CLoader as Loader, CDumper as Dumper
#except ImportError:
#    from yaml import Loader, Dumper

from ..utils import yanny,apload

def load(planfile,verbose=False,np=False,expand=False,plugmapm=False):
    """
    Load an APOGEE plan file.

    Parameters
    ----------
    planfile : str
       The absolute path of the plan file.
    expand : bool
       Expand the calibration ID paths and add in
                directories.
    np : bool
       Convert APEXP data to numpy structured array.
    verbose : bool
       Print a lot of information to the screen

    Returns
    -------
    plandata : dictionary
        The plan structure with all the relevant information.

    Example
    -------
    plan = plan.load(planfile)
    
    By D.Nidever  May 2010
    translated to python, D.Nidever Oct 2020
    """

    # More than one file
    if dln.size(planfile)>1:
        raise ValueError('Only plan file at a time.')

    # Check that the plan file exists
    if os.path.exists(planfile) == False:
        raise ValueError(planfile+' NOT FOUND')

    # Yanny file
    if planfile.endswith('.par')==True:

        # Load the plan file
        plandata = yanny.yanny(planfile,np=True)

        # Make sure APEXP exists
        if 'APEXP' not in plandata.keys():
            raise ValueError('No APEXP structure in plan file '+planfile)

    # Yaml file
    if planfile.endswith('.yaml')==True:
        # Read the file
        with open(planfile) as file:
            plandata = yaml.full_load(file)

        # Convert APEXP to numpy structured array
        if np==True:
            apexp = plandata['APEXP']
            dt = numpy.dtype([('plateid',int),('mjd',int),('flavor',numpy.str,30),
                              ('name',int),('single',int),('singlename',numpy.str,50)])
            new = numpy.zeros(len(apexp),dtype=dt)
            for i in range(len(apexp)):
                for n in dt.names:
                    new[n][i] = apexp[i][n]
            plandata['APEXP'] = new

    return plandata
