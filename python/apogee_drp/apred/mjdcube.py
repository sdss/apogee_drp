import sys
import os
import glob
import pdb
import numpy as np
from astropy.io import fits
from ..utils import apload

def mjdcube(mjd, darkid=None, write=False, apred='daily', instrument='apogee-n', clobber=False):
    """
    Make a cube for a given night with the CDS images of all frames.

    Parameters
    ----------
    mjd : int
       The MJD to process.
    darkid : int, optional
       Dark calibration file to use to correct the data.  Default
         is not to apply a dark correction.
    write : bool, optional
       Write out individual uncompressed data cubes.  Default is False.
    apred : str, optional
       Apred reduction version.  Default is 'daily'.
    instrument : str, optional
       Instrument to use.  Default is 'apogee-n'.
    clobber : bool, optional
       Overwrite any existing files.  Default is False.

    Returns
    -------
    The cube of 2D CDS images are written to apHist files.

    Example
    -------

    mjdcube(59164,apred='daily',instrument='apogee-n'))

    """

    print('mjd: ', mjd)
    print('apred: ', apred)
    print('instrument: ',instrument)
    print('write: ', write)
    print('darkid: ', darkid)

    observatory = {'apogee-n':'apo','apogee-s':'lco'}[instrument]
    telescope = {'apogee-n':'apo25m','apogee-s':'lco25m'}[instrument]
    datadir = os.getenv({'apo':'APOGEE_DATA_N','lco':'APOGEE_DATA_S'}[observatory])
    outdir = os.getenv('APOGEE_REDUX')+'/'+apred+'/exposures/'+instrument+'/'+str(mjd)+'/'
    load = apload.ApLoad(apred=apred,telescope=telescope)

    if os.path.exists(outdir)==False:
        os.makedirs(outdir)
    
    # Loop over the chips
    for chip in ['a','b','c']:
        print('chip ',chip)
        # Get all of the files
        files = glob.glob(datadir+'/'+str(mjd)+'/apR-'+chip+'-*.apz')+glob.glob(datadir+'1m/'+str(mjd)+'/apR-'+chip+'-*.apz')
        files.sort()
        print(len(files),' files')
        
        # Output file name for CDS cube
        outfile = outdir+load.prefix+'Hist-'+chip+'-'+str(mjd)+'.fits'

        # Does output file already exist?
        if not clobber and os.path.exists(outfile):
            print(outfile+' already exists and clobber==False')
            return

        out = fits.HDUList(fits.PrimaryHDU())
        
        # Get dark frame if requested
        if darkid is not None:
            darkfile = load.filename('Dark',num=darkid,chips=True).replace('Dark-','Dark-'+chip+'-')
            if os.path.exists(darkfile)==False:
                raise FileNotFoundError(darkfile)
            darkhdu = fits.open(darkfile)
            dark = darkhdu[1].data

        # Loop over all files
        for f,fil in enumerate(files):
            if write:
                # Output file name for individual uncompressed images
                outfile = os.path.basename(fil.strip('apz')+'fits')
                hduout = fits.HDUList(fits.PrimaryHDU())

            # Add filename to primary output header
            out[0].header['FILE'+str(f+1)] = fil
                
            # Open file and confirm checksums
            hdu = fits.open(fil, do_not_scale_image_data = True, uint = True, checksum = True)

            print('{:3d}   file: {:s}  objtype: {:s}'.format(f+1,fil,hdu[1].header.get('exptype')))
            
            # File has initial header, avg_dcounts, then nreads
            nreads = len(hdu)-2
            try:
                avg_dcounts = hdu[1].data
            except:
                # Fix header if there is a problem (e.g., MJD=55728, 01660046)
                hdu[1].verify('fix')
                avg_dcounts = hdu[1].data

            # First read is in extension 2
            ext = 2
            
            # Loop over reads, processing into raw reads, and appending
            for read in range(1,nreads+1):
                header = hdu[ext].header
                try:
                    raw = hdu[ext].data
                except:
                    hdu[ext].verify('fix')
                    raw = hdu[ext].data
                if read == 1:
                    data = np.copy(raw)
                else:
                    data = np.add(data,raw,dtype=np.int16)
                    data = np.add(data,avg_dcounts,dtype=np.int16)
                    if read == 2:
                        first = data
  
                if write:
                    hduout.append(fits.ImageHDU(data,header))
          
                ext += 1

            # Compute and add the cdsframe, subtract dark if we have one
            cds = (data[0:2048,0:2048] - first[0:2048,0:2048] ).astype(float)
            #print(cds.shape)
            if darkid is not None:
                #print(dark.shape,nreads)
                # If we don't have enough reads in the dark, do nothing
                try :
                    cds -= (dark[nreads-1,:,:] - dark[2,:,:])
                except:
                    print('not halting: not enough reads in dark, skipping dark subtraction for mjdcube')
                    pass
            cds = cds.astype(np.int32)     # convert back to integers
            header['EXTNAME'] = 'CDS'
            header['RAWFILE'] = fil
            out.append(fits.ImageHDU(cds,header))
            if write:
                hduout.writeto(outfile,overwrite=True, checksum = True, output_verify='fix')
                hdu.close()

        # Write out the CDS frame
        print('Writing to ',outfile)
        out.writeto(outfile,overwrite=True, checksum = True, output_verify='fix')
        out.close()
