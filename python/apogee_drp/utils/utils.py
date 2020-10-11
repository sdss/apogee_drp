from astropy.time import Time
import numpy as np

def getmjd5(dateobs):
    """ Convert a DATE-OBS string to 5-digit MJD number."""
    t = Time(dateobs)
    mjd = t.mjd
    # The Julian day starts at NOON, while MJD starts at midnight                                                                                             
    # For SDSS MJD we add 0.3 days                                                                                                                            
    mjd += 0.3
    # Truncate for MJD5 number                                                                                                                                
    mjd5 = int(mjd)
    return mjd5
