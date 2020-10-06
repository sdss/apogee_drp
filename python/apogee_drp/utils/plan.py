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


    # Expand, add paths to the calibration IDs and add directories
    if expand==True:
        load = apload.ApLoad(apred=plandata['apred_vers'],telescope=plandata['telescope'])
        calkeys = ['detid','bpmid','littrowid','persistid','persistmodelid','darkid',
                   'flatid','psfid','fluxid','responseid','waveid','lsfid']
        caltype = ['Detector','BPM','Littrow','Persist','PersistModel','Dark',
                   'Flat','PSF','Flux','Response','Wave','LSF']
        for i in range(len(calkeys)):
            val = plandata.get(calkeys[i])
            if val is not None:
                calfile = os.path.dirname(load.filename(caltype[i],num=val,chips='a'))+'/'+str(val)
                plandata[calkeys[i]] = calfile

        # Add directores
        spectro_dir = os.environ['APOGEE_REDUX']+'/'+plandata['apred_vers']+'/'
        datadir = {'apo25m':os.environ['APOGEE_DATA_N'],'apo1m':os.environ['APOGEE_DATA_N'],
                   'lco25m':os.environ['APOGEE_DATA_S']}[plandata['telescope']]+'/'
        if plandata.get('plate_dir') is None:
            plandata['plate_dir'] = spectro_dir+'visit/'+plandata['telescope']+'/'+\
                                    str(plandata['plateid'])+'/'+str(plandata['mjd'])+'/'
        if plandata.get('star_dir') is None:
            plandata['star_dir'] = spectro_dir+'fields/'+plandata['telescope']

        # Expand plugmap
        val = plandata['plugmap']
        if val is not None:
            # Check that it has no path information
            if val.find('/') == -1 and val.strip() != '':
                # Add directory
                plugfile = datadir+str(plandata['mjd'])+'/'
                # Add plPlugMap prefix if necessary
                if val.startswith('pl')==False:
                    if plugmapm==False:
                        plugfile += 'plPlugMapA-'
                    else:
                        plugfile += 'plPlugMapM-'
                plugfile += val
                # Add .par ending if necessary
                if val.endswith('.yaml')==False:
                    plugfile += '.yaml'
                plandata['plugmap'] = plugfile

    return plandata
