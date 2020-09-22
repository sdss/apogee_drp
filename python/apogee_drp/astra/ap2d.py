#!/usr/bin/env python

import luigi
import os
import numpy as np
import subprocess
import glob
import pickle
from astra.tasks import BaseTask
from astra.tasks.io import ApPlanFile
from sdss_access.path import path
from apogee_drp.utils import apload,yanny
from luigi.util import inherits

# Inherit the parameters needed to define an ApPlanFile, since we will need these to
# require() the correct ApPlanFile.
@inherits(ApPlanFile)
class AP2D(BaseTask):

    """ Run the 2D to 1D portion of the APOGEE pipeline."""

    # Parameters
    apred = luigi.Parameter()
    instrument = luigi.Parameter()
    telescope = luigi.Parameter()
    field = luigi.Parameter()
    plate = luigi.IntParameter()
    mjd = luigi.Parameter()
    prefix = luigi.Parameter()
    release = luigi.Parameter()

    def requires(self):
        return ApPlanFile(**self.get_common_param_kwargs(ApPlanFile))


    def output(self):
        # Store the 1D frames in the same directory as the plan file.
        output_path_prefix, ext = os.path.splitext(self.input().path)
        return luigi.LocalTarget(f"{output_path_prefix}-done2D")


    def run(self):
        # Run the IDL program!
        cmd = "ap2d,'"+self.input().path+"'"
        ret = subprocess.call(["idl","-e",cmd],shell=False)

        # Load the plan file
        # (Note: I'd suggest moving all yanny files to YAML format and/or just supply the plan file
        # inputs as variables to the task.)
        plan = yanny.yanny(self.input().path,np=True)
        exposures = plan['APEXP']

        # Check if the ap1D files were created
        # Get the exposures directory
        sdss_path = path.Path()
        expdir = os.path.dirname(sdss_path.full('ap1D',apred=self.apred,telescope=self.telescope,instrument=self.instrument,
                                                plate=self.plate,mjd=self.mjd,prefix=self.prefix,num=0,chip='a'))
        counter = 0
        for exp in exposures['name']:
            if type(exp) is not str:  exp=exp.decode()
            exists  = [os.path.exists(expdir+"/ap1D-"+ch+"-"+str(exp)+".fits") for ch in ['a','b','c']]
            if np.sum(exists) == 3: counter += 1

        # Create "done" file if 1D frames exist
        if counter == len(exposures):
            with open(self.output().path, "w") as fp:
                fp.write(" ")


if __name__ == "__main__":

    # The parameters for RunAP3D are the same as those needed to identify the ApPlanFile:
    # From the path definition at:
    #   https://sdss-access.readthedocs.io/en/latest/path_defs.html#dr16

    # We can see that the following parameters are needed:
    #   $APOGEE_REDUX/{apred}/visit/{telescope}/{field}/{plate}/{mjd}/{prefix}Plan-{plate}-{mjd}.par

    # Define the task.
    task = AP2D(
        apred="t14",
        telescope="apo25m",
        instrument="apogee-n",
        field="200+45",
        plate=8100,   # plate must be in
        mjd="57680",
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
