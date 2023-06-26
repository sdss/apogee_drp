import os
import time
import numpy as np
from datetime import datetime
from dlnpyutils import utils as dln

def lock(filename,waittime=10,maxduration=3*3600,lock=False,
         unlock=False,clear=False,verbose=True):
    """
    Procedure to handle using lock files.  The default behavior is to
    check for a lock file and wait until it no longer exists.

    This is generally used for APOGEE calibration files.
    The standard usage is:
     lock(calfile)              # wait on lock file
       check if the calibration already exists. if it does, then return
     lock(calfile,lock=True)    # create the lock file
       make the calibration file
     lock(calfile,clear=True)   # clear the lock file

    Parameters
    ----------
    file : str
       Original file to be locked.  The lock file is file+'.lock'.
    waittime : int, optional
       Time to wait before checking the lock file again. Default is 10 sec.
    clear : bool, optional
      Clear/delete a lock file.  Normally at the end of processing.
    unlock : bool, optional
      If the lock file exists, then unlock/delete it.
    lock : bool, optional
      Relock the file at the end of the wait.
    maxduration : int, optional
      Maximum duration of the original file being unmodified.
      Default is 3 hours.
    verbose : bool, optional
       Do not print anything to the screen.  Default is True.

    Returns
    -------
    Nothing is returned

    Example
    -------
    lock(file,waittime=10)

    By D. Nidever June 2023
    """
  
    lockfile = filename+'.lock'
    dirname = os.path.dirname(os.path.abspath(filename))
    if os.path.exists(dirname)==False:
        os.makedirs(dirname)   # make directory if necessary 
  
    # Clear or unlock the lock file
    if clear or unlock:
        if os.path.exists(lockfile): os.remove(lockfile)
        return

    # Wait for the lockfile
    while os.path.exists(lockfile):
        if verbose: print('waiting for file lock: ', lockfile)
        curtime = datetime.now().timestamp()
        
        # How long has it been since the file has been modified
        if os.path.exists(filename): # make sure it exists 
            mtime = os.path.getmtime(filename)
            if curtime > mtime+maxduration:
                if verbose:
                    print('lock file exists but original file unchanged in over {:.2f} hours'.format(maxduration/3600))
                if os.path.exists(lockfile): os.remove(lockfile)
                if lock: dln.touch(lockfile)  # Lock it again
                return
        # Original file doesn't exist, check how long we have been waiting for it      
        else:
            lctime = os.path.getctime(lockfile)
            if curtime > lctime+maxduration:
                if verbose:
                    print('lock file exists but waiting for over {:.2f} hours'.format(maxduration/3600))
                if os.path.exists(lockfile): os.remove(lockfile)
                if lock: dln.touch(lockfile)  # lock it again
                return
        # Wait
        time.sleep(waittime)
        
    # Lock it
    if lock: dln.touch(lockfile)
