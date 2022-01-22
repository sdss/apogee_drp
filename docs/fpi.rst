**************************
Fabry-Perot Interferometer
**************************

A Fabry-Perot Interferometer (FPI) was recently installed for the APOGEE instruments to help improve our wavelengths and radial velocity precision.


APOGEE Wavelength Calibration with the FPI
==========================================

There are several processing steps to take advantage of the FPI to improve our wavelength calibration.

1. Redetermine precise FPI wavewlengths and wavelength solutions
----------------------------------------------------------------
- Determine precise wavelength solutions using a week of arclamp exposures and full-frame FPI exposure at beginning of the night.
- Fit wavelength solutions to many arclamp exposures (over a week) simultaneously
- Determine median wavelength of each FPI line using wavelength solutions and Gaussian position over the 300 fibers
- Refit wavelength solution of each fiber using the wavelengths of the FPI lines
- This is created by the ``mkfpiwave`` which calls ``fpi.dailyfpiwave)``.  This should be run every day with the daily calibration products.  **The code still needs to be tested.**
  
2. Correct wavelength solution using 2-fiber FPI data
-----------------------------------------------------
- For each science exposure use 2 FPI fibers to measure pixel shift and correct the wavelength solution
- Use 2 FPI fibers to fit linear surface and find pixel shift correction for each fiber
- Correct the wavelength solution for each fiber by offseting the x-values of each pixel and refitting the wavelengths (maybe holding the higher-order terms fixed).
  
  - from original wavelength solution get wavelengths (x array and w array).
  - offset x array, x2=x+deltax
  - refit wavelength solution using x2 and w arrays.  Maybe only refit constant and linear terms
    
- The FPI lines allow us to make a precise shift measurement relative to the FPI lines at the beginning of the night.
- Currently we use the sky lines to do this, so maybe add the FPI part to the same program, or write a similar one that is called if we have FPI data.
- This is run by the ``ap1dwavecal`` script which calls ``fpi.fpi1dwavecal()`` which is called at the end of ``ap2dproc.pro``.  **The code still needs to be tested.**
  
3. Measure Dither shift
-----------------------

- Measure dither shift between science frames like we have always done, by cross-correlating the two science spectra.
- Fit linear or quadratic polynomial to the dither shifts vs. fiber.
- These should be more accurate (on a fiber by fiber basis) than the FPI shifts which are interpolated/extrapolated to most of the fibers.
- This should be very similar to what we are already doing.
- This is done in ``apdithershift.pro``.  **This uses existing code.**
  
4. Dither combination/final wavelengths
---------------------------------------

- Use the dither shifts from #3 to perform the dither combination.
- Shift the wavelengths using the same dither shift as the spectra.
- For each fiber, average the wavelengths of the multiple exposures.
- The wavelength part of this is new code that can be written into the dither combination program.
- This is done in ``apdithercomb.pro``.  **This still needs to be implemented and tested.**
