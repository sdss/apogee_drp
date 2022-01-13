****
Data
****

There are a large number of APOGEE data products and directores.  The structure of the data products themselves
is called the "data model".  There is a new SDSS-V data model product that is now used to describe all of the
data files.

The directory structure and file names are set by the ``tree`` product.  The SDSS-V settings are available
`here <https://github.com/sdss/tree/blob/sdss5/data/sdss5.cfg>`_.  Some examples are:

apFlux = $APOGEE_REDUX/{apred}/cal/{instrument}/flux/@apgprefix|Flux-{chip}-{num:0>8}.fits
apStar = $APOGEE_REDUX/{apred}/{apstar}/{telescope}/@healpixgrp|/{healpix}/apStar-{apred}-{telescope}-{obj}.fits
apWave = $APOGEE_REDUX/{apred}/cal/{instrument}/wave/@apgprefix|Wave-{chip}-{num:0>8}.fits


Main files and directories
==========================

Since there are three APOGEE detectors, the data are almost always split into three files, one per detector:
a (red), b (green), c (blue).  For example, a 2D image for exposure 40220015 has three files with names
ap2D-a-40220015.fits, ap2D-b-40220015.fits and ap2D-c-40220015.fits.

Here's the directory layout of the main data files.


Raw files
---------

The raw files are compressed in a custom "apz" format.  Only the difference between up the ramp reads are
kept (and the first frame), and then `fpack <https://heasarc.gsfc.nasa.gov/fitsio/fpack/>`_ compressed.
These can be uncompressed with ``apunzip.pro``.

The raw files have names of ``apR-[abc]-EXPNUM8.apz`` where EXPNUM8 is the eight-digit exposure number 
(e.g., 40220015) where the first four digits are the day number (where day number zero MJD 55562) and the last
four digits are the exposure number in that day.  The 

example URL

Exposures files
---------------

Calibration files
-----------------

Visit files
-----------

Star files
----------

HEALPix grouping
