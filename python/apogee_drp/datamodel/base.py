import os
import sys
import numpy as np
import time
import socket
import platform
import getpass
from astropy.io import fits
    
class APOGEEBase(object):
    """ Base APOGEE datamodel class """

    def __init__(self,*pars,**kwargs):
        pass

    def exists(self,filename):
        pass
    
    def read(self,filename):
        pass

    def write(self,filename):
        pass

    def mainheader(self):
        header = fits.Header()
        leadstr = self.datatype+': '
        header['HISTORY'] = leadstr+time.asctime()
        header['HISTORY'] = leadstr+getpass.getuser()+' on '+socket.gethostname()
        pyvers = sys.version.split()[0]
        header['HISTORY'] = leadstr+'Python '+pyvers+' '+platform.system()+' '+platform.release()+' '+platform.architecture()[0]
        header['HISTORY'] = 'APOGEE software git hash:' +str(plan.getgitvers())
        header['HISTORY'] = leadstr+' APOGEE Reduction Pipeline Version: {:s}'.format(os.environ['APOGEE_DRP_VER'])
        return header
