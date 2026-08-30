import copy
import numpy as np
import os
import shutil
from glob import glob
import pdb

from dlnpyutils import utils as dln
from . import apload,plugmap as plmap
from astropy.io import fits
from astropy.table import Table

import subprocess
import tempfile
import time


def file_status(filename):
    """
    Check on a file's status: exists, size, mtime, okay (not currupted/truncated)
    """

    if type(filename) is str:
        files = [filename]
    else:
        files = filename
    nfiles = len(files)
    out = np.zeros(nfiles,dtype=np.dtype([('name',(str,500)),('exists',bool),('mtime',int),('size',int),('okay',bool)]))
    out['name'] = files
    for i in range(nfiles):
        fil = files[i]
        exists = os.path.exists(fil)
        out['exists'][i] = exists
        if exists:
            out['size'][i] = os.path.getsize(fil)
            out['mtime'][i] = os.path.getmtime(fil)
            if fil.endswith('fits'):
                try:
                    hdu = fits.open(fil,checksum=True)
                    v = hdu.verify(option='exception')
                    hdu.close()
                    out['okay'][i] = True
                except:
                    out['okay'][i] = False
            else:
                # Not sure how to check non-FITS files
                out['okay'][i] = True

    return out
                
def expinfo(observatory=None,mjd5=None,files=None,expnum=None,
            logger=None,verbose=False,fieldinfo=True):
    """
    Get header information about raw APOGEE files.
    This program can be run with observatory+mjd5 or
    by giving a list of files.

    Parameters
    ----------
    observatory : str, optional
       APOGEE observatory (apo or lco).
    mjd5 : int, optional
       The MJD5 night to get exposure information for.
    files : list of str, optional
       List of APOGEE apz filenames.
    expnum : list, optional
       List of exposure numbers.
    logger : logging object, optional
       Logging object for printing logs.
    verbose : bool, optional
       Verbose output to the screen.  Default is False.
    fieldinfo : bool, optional
       Return field information.  Default is True.

    Returns
    -------
    cat : numpy structured array
       Table with information for each file grabbed from the header.

    Examples
    --------

    info = expinfo(files)

    By D.Nidever,  Oct 2020
    """

    # Types of inputs:
    #  files, observatory+mjd5, observatory+expnum
    if (files is None and observatory is None) or (files is None and mjd5 is None and expnum is None):
        raise ValueError('Either files or observatory+mjd5 or observatory+expnum must be input')
    if (mjd5 is not None and expnum is not None):
        raise ValueError('Input either observatory+mjd5 or observatory+expnum')

    if logger is None and verbose:
        logger = dln.basiclogger()

    telescope = 'apo25m'
    if observatory is not None:
        telescope = {'apo':'apo25m','lco':'lco25m'}[observatory]
    load = apload.ApLoad(apred='daily',telescope=telescope)

    # Get the exposures info for this MJD5        
    if files is None and expnum is None:
        if observatory not in ['apo','lco']:
            raise ValueError('observatory must be apo or lco')
        datadir = {'apo':os.environ['APOGEE_DATA_N'],'lco':os.environ['APOGEE_DATA_S']}[observatory]
        files = glob(datadir+'/'+str(mjd5)+'/a?R-c*.apz')
        files = np.array(files)
        nfiles = len(files)
        if nfiles==0:
            return []
        files = files[np.argsort(files)]  # sort        
    # Exposure numbers input
    if files is None and expnum is not None:
        if observatory not in ['apo','lco']:
            raise ValueError('observatory must be apo or lco')
        datadir = {'apo':os.environ['APOGEE_DATA_N'],'lco':os.environ['APOGEE_DATA_S']}[observatory]
        telescope = observatory+'25m'
        load = apload.ApLoad(apred='daily',telescope=telescope)
        if type(expnum) is not list and type(expnum) is not np.ndarray:
            expnum = [expnum]
        nfiles = len(expnum)
        files = []
        for i in range(nfiles):
            file1 = load.filename('R',num=expnum[i],chip='c')
            files.append(file1)
        files = np.array(files)
        files = files[np.argsort(files)]  # sort        
        
    nfiles = len(files)
    dtype = np.dtype([('num',int),('nread',int),('exptype',np.str,20),('arctype',np.str,20),('plateid',np.str,20),
                      ('configid',np.str,50),('designid',np.str,50),('fieldid',np.str,50),('exptime',float),
                      ('dateobs',np.str,50),('gangstate',np.str,20),('shutter',np.str,20),('calshutter',np.str,20),
                      ('mjd',int),('observatory',(np.str,10)),('dithpix',float)])
    tab = np.zeros(nfiles,dtype=dtype)
    plate2field = {}
    plate2plugmap = {}
    # Loop over the files
    for i in range(nfiles):
        if os.path.exists(files[i]):
            head = fits.getheader(files[i],1)
            base,ext = os.path.splitext(os.path.basename(files[i]))
            # apR-c-12345678.apz
            num = base.split('-')[2]
            tab['num'][i] = num
            tab['nread'][i] = head.get('nread')
            tab['exptype'][i] = head.get('exptype')
            tab['plateid'][i] = head.get('plateid')
            tab['configid'][i] = head.get('configid')
            tab['designid'][i] = head.get('designid')
            tab['fieldid'][i] = head.get('fieldid')
            tab['exptime'][i] = head.get('exptime')
            tab['dateobs'][i] = head.get('date-obs')
            mjd = int(load.cmjd(int(num)))
            tab['mjd'] = mjd
            #    tab['mjd'] = utils.getmjd5(head['date-obs'])
            plate = head.get('plateid')            
            if fieldinfo and mjd<59556 and plate is not None and str(plate) != '' and int(plate)>0:
                plugid = head.get('name')
                if plate2plugmap.get(plate) is not None:
                    plfilename = plate2plugmap.get(plate)
                else:
                    plfilename = plmap.plugmapfilename(plate,mjd,load.instrument,
                                                       plugid=plugid,verbose=verbose)
                    plate2plugmap[plate] = plfilename
                plugmap = plmap.load(plfilename)
                tab['designid'][i] = plugmap.get('designid')
                locationID = plugmap.get('locationId')
                if plate2field.get(plate) is not None:
                    fieldid = plate2field.get(plate)
                else:
                    fieldid,_,_ = apload.apfield(plate,telescope=load.telescope,fps=False)
                    plate2field[plate] = fieldid
                tab['fieldid'][i] = fieldid
            if observatory is not None:
                tab['observatory'] = observatory
            else:
                tab['observatory'] = {'p':'apo','s':'lco'}[base[1]]
            # arc types
            if tab['exptype'][i]=='ARCLAMP':
                if head.get('lampune')==1:
                    tab['arctype'][i] = 'UNE'
                elif head.get('lampthar')==1:
                    tab['arctype'][i] = 'THAR'
                else:
                    tab['arctype'][i] = 'None'
            # FPI
            if tab['exptype'][i]=='ARCLAMP' and tab['arctype'][i]=='None' and head.get('OBSCMNT').startswith('FPI'):
                tab['exptype'][i] = 'FPI'

            # Sky flat
            if tab['exptype'][i]=='OBJECT' and tab['nread'][i]>10 and tab['nread'][i]<13 and head.get('OBSCMNT').lower().replace(' ','')[0:3]=='sky':
                tab['exptype'][i] = 'SKYFLAT'

            # Dither position
            tab['dithpix'][i] = head.get('dithpix')
            # Gang state
            #  gangstat wasn't working properly until MJD=59592
            if mjd>=59592:
                tab['gangstate'][i] = head.get('gangstat')
            # APOGEE Shutter state
            #  shutter wasn't working perly until MJD=59592
            if mjd>=59592:
                tab['shutter'][i] = head.get('shutter')
            # CalBox shutter status
            lampshtr = head.get('lampshtr')
            if lampshtr is not None:
                if lampshtr:
                    tab['calshutter'][i] = 'Open'
                else:
                    tab['calshutter'][i] = 'Closed'

            if verbose:
                logger.info(tab[i])
                
    return tab

def getdithergroups(expinfo):
    """
    Calculate dither groups for a table of exposures.
    The table should have all the exposure for a given night to
    perform a proper dither group analysis.

    Parameters
    ----------
    expinfo : table
       Table of exposure information.  Must have dithpix column.

    Returns
    -------
    expinfo : table
       Table of exposure information with "dithergroup" column added.

    Example
    -------

    expinfo = getdithergroups(expinfo)

    """

    expinfo = Table(expinfo)
    if 'dithpix' not in expinfo.colnames:
        raise Exception("dithpix column not found in table")


    expinfo['dithergroup'] = -1
    currentditherpix = expinfo['dithpix'][0]
    currentmjd = expinfo['mjd'][0]
    dithergroup = 1
    for e in range(len(expinfo)):
        if np.abs(expinfo['dithpix'][e]-currentditherpix)<0.01 and (expinfo['mjd'][e]==currentmjd):
            expinfo['dithergroup'][e] = dithergroup
        else:
            dithergroup += 1
            currentditherpix = expinfo['dithpix'][e]
            currentmjd = expinfo['mjd'][e]
            expinfo['dithergroup'][e] = dithergroup

    return expinfo


def file_status(
    filenames,
    nprocs=4,
    batch_size=1000,
    sort=True,
    cached=False,
    verbose=False,
):
    """
    Check file existence and size using GNU xargs and stat.

    Parameters
    ----------
    filenames : sequence of str or path-like
        Full paths to check.
    nprocs : int, optional
        Number of parallel stat processes.
    batch_size : int, optional
        Maximum number of filenames passed to each stat invocation.
    sort : bool, optional
        Sort filenames before checking to improve metadata locality.
        Returned arrays remain aligned with the original input order.
    cached : bool, optional
        Use ``stat --cached=always``. This can return stale metadata.
    verbose : bool, optional
        Print timing and summary information.

    Returns
    -------
    exists : numpy.ndarray
        Boolean array aligned with the input filenames.
    sizes : numpy.ndarray
        File sizes in bytes. Missing or inaccessible files have size -1.
    errors : list of str
        Error messages produced by stat, bash, or xargs.

    Notes
    -----
    Requires GNU xargs, GNU stat, and bash.
    """
    filenames = [os.fspath(path) for path in filenames]
    nfiles = len(filenames)

    max_length = max((len(path) for path in filenames), default=1)
    
    fileinfo = np.empty(nfiles, dtype=[("filename", f"U{max_length}"),
                                       ("exists", bool),("size", np.int64),
                                       ("mtime_ns", np.int64),
                                       ("fingerprint", np.uint64)])
    fileinfo["filename"] = filenames
    fileinfo["exists"] = False
    fileinfo["size"] = -1
    fileinfo["mtime_ns"] = -1
    fileinfo["fingerprint"] = 0
 
    if nfiles == 0:
        return fileinfo

    if nprocs < 1:
        raise ValueError("nprocs must be at least 1")

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    for path in filenames:
        if "\n" in path or "\t" in path:
            raise ValueError(
                "Filenames containing tabs or newlines are unsupported: "
                f"{path!r}"
            )

    check_order = sorted(filenames) if sort else filenames
    status_by_path = {}

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="\n",
    ) as input_file:

        for path in check_order:
            input_file.write(path)
            input_file.write("\n")

        input_file.flush()

        with tempfile.TemporaryDirectory() as output_directory:
            script = r'''
output_directory=$1
cached_mode=$2
shift 2

output_file=$(mktemp "${output_directory}/output.XXXXXX")
error_file=$(mktemp "${output_directory}/error.XXXXXX")

if [ "$cached_mode" = "always" ]; then
    stat --cached=always --printf='%s\t%Y\t%y\t%n\n' -- "$@" \
        > "$output_file" \
        2> "$error_file"
else
    stat --printf='%s\t%Y\t%y\t%n\n' -- "$@" \
        > "$output_file" \
        2> "$error_file"
fi
'''
            
            cached_mode = "always" if cached else "default"

            command = ["xargs", "-a", input_file.name, "-d", "\n",
                       "-P", str(nprocs), "-n", str(batch_size), "bash",
                       "-c", script, "_", output_directory, cached_mode, ]

            start_time = time.monotonic()

            result = subprocess.run(command,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                    text=True, check=False)

            elapsed = time.monotonic() - start_time
            errors = []

            for name in os.listdir(output_directory):
                path = os.path.join(output_directory, name)

                if name.startswith("output."):
                    with open(path, encoding="utf-8") as output_file:
                        for line in output_file:
                            line = line.rstrip("\n")

                            if not line:
                                continue

                            try:
                                size_text, seconds_text, precise_time, filename = line.split("\t", 3)
                                # GNU stat's %y ends with something like:
                                # 2026-08-30 14:23:12.123456789 -0600
                                if "." in precise_time:
                                    fraction = precise_time.split(".", 1)[1].split(" ", 1)[0]
                                    fraction = fraction[:9].ljust(9, "0")
                                    nanoseconds = int(fraction)
                                else:
                                    nanoseconds = 0
                                mtime_ns = int(seconds_text) * 1_000_000_000 + nanoseconds
                                status_by_path[filename] =  (int(size_text), mtime_ns)
                            except (TypeError, ValueError) as error:
                                raise RuntimeError(f"Could not parse stat output: {line!r}") from error

                elif name.startswith("error."):
                    with open(path, encoding="utf-8") as error_file:
                        errors.extend(
                            line.rstrip("\n")
                            for line in error_file
                            if line.strip()
                        )

            if result.stderr:
                errors.extend(
                    line
                    for line in result.stderr.splitlines()
                    if line.strip()
                )

    with np.errstate(over="ignore"):
        for i, filename in enumerate(filenames):
            status = status_by_path.get(filename)

            if status is not None:
                size, mtime_ns = status

                # Construct a deterministic 64-bit metadata fingerprint.
                # Integer overflow is intentional.
                value = np.uint64(size) ^ (
                    np.uint64(mtime_ns)
                    + np.uint64(0x9E3779B97F4A7C15)
                )

                value ^= value >> np.uint64(30)
                value *= np.uint64(0xBF58476D1CE4E5B9)
                value ^= value >> np.uint64(27)
                value *= np.uint64(0x94D049BB133111EB)
                value ^= value >> np.uint64(31)
            
                fileinfo["exists"][i] = True
                fileinfo["size"][i] = size
                fileinfo["mtime_ns"][i] = mtime_ns
                fileinfo["fingerprint"][i] = value

    # xargs returns 123 if one or more stat commands return a nonzero
    # status, which is expected when files are missing.
    if result.returncode not in (0, 123):
        message = errors[0] if errors else "no error message"
        raise RuntimeError(
            f"xargs failed with status {result.returncode}: {message}"
        )

    if verbose:
        nfound = int(np.count_nonzero(fileinfo["exists"]))
        nmissing = nfiles - nfound
        rate = nfiles / elapsed if elapsed > 0 else float("inf")
        
        print(
            f"Checked {nfiles:,} files in {elapsed:.2f} seconds "
            f"({rate:,.0f} files/s)"
        )
        print(f"Existing: {nfound:,}")
        print(f"Missing or inaccessible: {nmissing:,}")
        
        if errors:
            print(f"stat/xargs error messages: {len(errors):,}")

    return fileinfo
