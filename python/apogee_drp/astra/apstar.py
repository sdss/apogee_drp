#!/usr/bin/env python

import luigi
import os
import numpy as np
import subprocess
import glob
import pickle
from astra.tasks import BaseTask
#from astra.tasks.io import ApVisitFile,ApStarFile
from sdss_access.path import path
from apogee_drp.utils import apload,yanny
from apogee_drp.apred import rv
from luigi.util import inherits
from holtztools import struct
from astropy.table import Table

# Inherit the parameters needed to define an ApStarFile, since we will need these to
# require() the correct ApVisitFile.
#@inherits(ApVisitFile)
#@inherits(ApStarFile)
class APSTAR(BaseTask):

    """ Run the RV and Visit combination part of the code."""

    # Parameters
    star = luigi.Parameter()
    apred = luigi.Parameter()
    instrument = luigi.Parameter()
    telescope = luigi.Parameter()
    field = luigi.Parameter()
    prefix = luigi.Parameter()
    release = luigi.Parameter()


    def requires(self):
        # Check that there are some Visit files for this star

        # Get all the VisitSum files for this field and concatenate them
        files = glob.glob(os.environ['APOGEE_REDUX']+'/'+self.apred+'/visit/'+self.telescope+'/'+self.field+'/apVisitSum*')
        if len(files) == 0 :
            print('no apVisitSum files found for {:s}'.format(self.field))
            return
        else:
            allvisits = struct.concat(files)
        starmask = bitmask.StarBitMask()
        gd = np.where(((allvisits['STARFLAG'] & starmask.badval()) == 0) &
                      (allvisits['APOGEE_ID'] != b'') &
                      (allvisits['SNR'] > snmin) )[0]
        allvisits = Table(allvisits)
        # Get visit files
        starvisits = np.where(allvisits['APOGEE_ID'][gd] == star)[0]
        nvisits = len(starvisits)

        return ApVisitFile(**self.get_common_param_kwargs(ApVisitFile))


    def output(self):
        # Output is similar to apStar file
        sdss_path = path.Path()
        apstarfile = sdss_path.full('apStar',apred=self.apred,telescope=self.telescope,instrument=self.instrument,
                                    field=self.field,prefix=self.prefix,obj=self.star,apstar='stars')
        output_path_prefix, ext = os.path.splitext(apstarfile)
        return luigi.LocalTarget(f"{output_path_prefix}-doneRV")


    def run(self):
        # Run doppler_rv_star()
        rv.doppler_rv_star(self.star,self.apred,self.instrument,self.field)

        # Check that the apStar file was created
        sdss_path = path.Path()
        apstarfile = sdss_path.full('apStar',apred=self.apred,telescope=self.telescope,instrument=self.instrument,
                                    field=self.field,prefix=self.prefix,obj=self.star,apstar='stars')

        # Create "done" file if apStar file exists
        if os.path.exists(apstarfile)==True:
            with open(self.output().path, "w") as fp:
                fp.write(" ")


if __name__ == "__main__":

    # The parameters for RunAP1DVISIT are the same as those needed to identify the ApPlanFile:
    # From the path definition at:
    #   https://sdss-access.readthedocs.io/en/latest/path_defs.html#dr16

    # We can see that the following parameters are needed:
    #   $APOGEE_REDUX/{apred}/visit/{telescope}/{field}/{plate}/{mjd}/{prefix}Plan-{plate}-{mjd}.par

    # Define the task.
    task = APSTAR(
        star="2M09321693+2827061",
        #obj="2M09321693+2827061",
        apred="t14",
        #apstar='stars',
        telescope="apo25m",
        instrument="apogee-n",
        field="200+45",
        #mjd='55555',
        #plate=8100,
        #fiber=1,
        prefix="ap",
        release=None
    )

    # (At least) Two ways to run this:
    # Option 1: useful for interactive debugging with %debug
    task.run()


    # Option 2: Use Luigi to build the dependency graph. Useful if you have a complex workflow, but
    #           bad if you want to interactively debug (because it doesn't allow it).
    #luigi.build(
    #    [task],
    #    local_scheduler=True
    #)

    # Option 3: Use a command line tool to run this specific task.
    # Option 4: Use a command line tool and an already-running scheduler to execute the task, and
    #           then see the progress in a web browser.
