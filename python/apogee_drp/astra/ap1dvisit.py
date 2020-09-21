import luigi
import os
import subprocess
import glob
import pickle
from astra.tasks import BaseTask
from astra.tasks.io import ApPlanFile
from apogee_drp.utils import apload,yanny
from luigi.util import inherits

# Inherit the parameters needed to define an ApPlanFile, since we will need these to
# require() the correct ApPlanFile.
@inherits(ApPlanFile)
class AP1DVISIT(BaseTask):

    """ Run the 1D visit portion of the APOGEE pipeline."""

    # Parameters
    apred = luigi.Parameter()
    instrument = luigi.Parameter()
    telescope = luigi.Parameter()
    field = luigi.Parameter()
    plate = luigi.IntParameter()
    mjd = luigi.Parameter()


    def requires(self):
        # We require plan files to exist!
        load = apload.ApLoad(apred=self.apred,telescope=self.telescope,instrument=self.instrument)
        planfile = load.filename('Plan',field=self.field,mjd=self.mjd,plate=self.plate)
        return ApPlanFile(**self.get_common_param_kwargs(ApPlanFile))


    def output(self):
        # Store the 1D frames in the same directory as the plan file.
        output_path_prefix, ext = os.path.splitext(self.input().path)
        return luigi.LocalTarget(f"{output_path_prefix}-done1D")


    def run(self):
        # Run the IDL program!
        subprocess.call(["idl","-e","ap1dvisit,",self.input().path])

        # Check if some apVisits have been made
        files = self.output().path+"a*Visit*.fits"
        check = glob.glob(files)

        # Create "done" file if apVisits exist
        if len(check)>140:
            with open(self.output().path, "w") as fp:
                fp.write(" ")



if __name__ == "__main__":

    # The parameters for RunAP1DVISIT are the same as those needed to identify the ApPlanFile:
    # From the path definition at:
    #   https://sdss-access.readthedocs.io/en/latest/path_defs.html#dr16

    # We can see that the following parameters are needed:
    #   $APOGEE_REDUX/{apred}/visit/{telescope}/{field}/{plate}/{mjd}/{prefix}Plan-{plate}-{mjd}.par

    # Define the task.
    task = RunAP1DVISIT(
        apred="t14",
        telescope="apo25m",
        field="203+00",
        mjd="56284",
        prefix="ap"
    )

    # (At least) Two ways to run this:
    # Option 1: useful for interactive debugging with %debug
    task.run()


    # Option 2: Use Luigi to build the dependency graph. Useful if you have a complex workflow, but
    #           bad if you want to interactively debug (because it doesn't allow it).
    luigi.build(
        [task],
        local_scheduler=True
    )

    # Option 3: Use a command line tool to run this specific task.
    # Option 4: Use a command line tool and an already-running scheduler to execute the task, and
    #           then see the progress in a web browser.
