********
Overview
********

This is an overview of the APOGEE reduction pipeline.

Types of Processing
===================

There are two types of APOGEE reduction processing: (1) daily, and (2) full reduction.  The daily processing happens every afternoon
and reduces last night's data using the latest version of the pipeline software.  The full reduction is run every 6 months to a year and
reprocesses all of the data using a uniform version of the software.  These are used for official data releases.

The most complete description of the APOGEE reduction pipeline (to date) is in `Nidever et al. (2015) <https://arxiv.org/abs/1501.03742>`_.

Processing Steps
================

Below are the main processing steps.

1. Plan files
-------------

The pipeline uses human readable "plan files" which describe chunks of data.  In the past these corresponded to data taken for a single plate.
Now they correspond to exposures taken on a given field and robot configuration.  The plan files also include the names of the calibration
products to use for the reduction of the exposures.


2. Calibrations
---------------

There are master/super calibration products that are updated every year or so and daily calibration products that are produced daily.
The daily products are PSF, Flux, Wave and WaveFPI.  They are processed in that order but the jobs are parallelized on the Utah computer cluster.

3. APRED
--------

The heart of the reduction pipeline is ``apred``.  It has three main steps:

 1. AP3D: Collapse of the up-the-ramp data cube.
 2. AP2D: Extraction of spectra.
 3. AP1DVISIT: Sky and telluric correction, dither combination.

The processing is parallelized over plan files/fields.
    
**AP3D** takes the 3D up-the-ramp data cube for each exposure and detector and collapses it to a 2D image.  It tried to remove cosmic rays
and fix saturation (as much as possible).  Dark current is removed, linearit corrections are applied, and the final 2D image (ap2D file) is
flat fielded.

**AP2D** extracts the 300 fibers to 1D from the 2D images.  The extraction currently uses empiral point spread function (EPSF) profiles that
are generated from domeflats taken during the night.  During the plate era, a domeflat was taken for each plate.  During the FPS era,
we are planning to use a domeflat "library" or a more sophisticated model of the PSF.  A relative flux calibration using a daily apFlux
calibration file (generated from a domeflat) is applied to remove fiber-to-fiber throughput and relative spectral response (see
`fluxcal <fluxcal.html>`_ for more details). At the end of AP2D, a wavelength solution is "attached"
to the output ap1D file.  The wavelengths are correct/shifted using night sky emission lines if the exposure was taken on sky.  

**AP1DVISIT** first performs corrections on each exposure and then combines multiple exposure at the end.  Sky fibers are used to remove
the sky continuum and line emission from each fiber.  Hot star ("telluric") spectra are used to fit a model of the telluric absorption
from CH4, CO2 and H2O and how it changes across the field.  This is then used to divide out the telluric absorption in each spectrum.
Because APOGEE spectra are slightly undersampled, observations are taken at two different spectral dither shifts (offset by ~0.5 pixels
in the spectral dimension) to recover good sampling.  To combine these two sets of exposures, a precise dither shift is calculated between
the exposures using the data themselves.  The spectra are then combined on a fiber-by-fiber basis using the measured shifts and
sinc interpolation.  Finally, the spectra are flux calibrated using a 5th order polynomial fit to the telluric stars to get the correct
spectral shape, and the 2MASS H-magnitudes to set the absolute flux scale (see `fluxcal <fluxcal.html>`_ for more details).
The final product is apVisit files for each star.

4. Radial Velocities
--------------------

Radial velocities (RV) are determined on a star-by-star basis using the `Doppler <https://github.com/dnidever/doppler>`_ software.
This fits a Cannon model to the data,
determining Teff, logg, [Fe/H] and radial velocity in the process.  It actually fits all of the visit spectra simultaneously, using
a single set of stellar parameters, but a separate radial velocity for each spectrum.  The model spectrum is convolved with the
correct LSF (line spread profile) for each visit spectrum.  The processing is parallelized over each object.

5. Visit Combination
--------------------

After the RVs have been determined, the visit spectra are combined into a single combined spectrum (apStar) on a rest wavelength scale.
This is actually performed in the same processing step as the RVs on a star-by-star basis.

6. Quality Assurance
--------------------

After the data processes has finished, a number of quality assurance plots and webpages are generated.
