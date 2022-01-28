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

def writelog(logfile,line):
    """ Append lines to a logfile."""
    # Convert to list
    if type(line) is list:
        lines = line
    elif type(line) is str:
        lines = [line]
    else:
        lines = [str(line)]
    # Make sure each line ends in newline
    lines = [l+'\n' if l.endswith('\n')==False else l for l in lines]
    # Append to the file
    with open(logfile,'a') as f:
        f.writelines(lines)
