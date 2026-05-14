*****************
Radial Velocities
*****************

The APOGEE radial velocities (RVs) are determined with `Doppler <https://doppler.readthedocs.io/en/latest/>`_ software
(the SDSS-V version is here `<https://github.com/sdss/doppler>`_)

Radial velocities (RV) are determined on a star-by-star basis using Doppler which fits a Cannon model to the data,
determining Teff, logg, [Fe/H] and radial velocity in the process.  It fits all of the visit spectra simultaneously, using
a single set of stellar parameters for the star, but a separate radial velocity for each spectrum.  The model spectrum is
convolved with the correct LSF (line spread profile) for each visit spectrum.

These are the steps taken by ``rv.py``:
 1) Each visit spectrum is fit individually by Doppler.
 2) Mean stellar parameters are calculated over the visits.
 3) All of the visit spectra are fit simultaneously with a single set of stellar parameters and a separate RV for each spectrum.
 4) Each visit spectrum is cross-correlated with the best-fit Cannon model spectrum and visit RV (these are the ``xcorr_`` values). The peak pixel of the CCF is used to determine the ``xcorr`` RV (not interpolated).
 5) Each visit CCF is Gaussian Decomposed and the number of components saved in N_COMPONENTS. If N_COMPONENTS=0, that means gausspy did not detect a peak or it crashed. 
 6) After extra quality cuts, the spectra and RVs are used to create the apStar combined spectrum.

Each time a star is observed, all of the existing visit spectra for that star are refit with Doppler. This means that there
are multiple versions of RVs for each visit spectrum based on the latest visit MJD (Modified Julian Date) and sets the "STARVER"
value.

The RV processing is parallelized over each object and log file is created in the ``stars/`` directory in the HEALPix subdirectory for that star.

RV Columns and Flags
~~~~~~~~~~~~~~~~~~~~

Visit spectra must pass through QA checks to be used in the final spectral combination:
 1) Initial QA check: STARFLAG should not have any of the BAD flags set (BAD_PIXELS,VERY_BRIGHT_NEIGHBOR,BAD_RV_COMBINATION,RV_FAIL) and must have S/N>2. These are the visits that Doppler is run on and the ones that are counted in NGOODVISITS at the STAR level.
 2) RV check: After Doppler has been run on all of the visits that passed the initial QA cut, then an additional selection is made for visits that do not have RV_REJECT set.  These visits are used to determine the mean RV, RV scatter, and make the combined spectrum.  These are the visits that have GOODVISIT=True set and are counted in NGOODRVS at the STAR level.


Relevant RV flags in STARFLAG:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Flag
     - Description

   * - RV_REJECT
     - Rejected visit because cross-correlation RV differs
       significantly from least-squares RV:

       * 50 km/s (rv_teff >= 6000)
       * 20 km/s (rv_teff < 6000 and rv_logg > 3.8)
       * 10 km/s (rv_teff < 6000 and rv_logg <= 3.8)

   * - RV_SUSPECT
     - Suspect visit (but used) because cross-correlation RV
       differs slightly from least-squares RV (> 0 km/s)

   * - MULTIPLE_SUSPECT
     - Suspect multiple components from Gaussian decomposition
       of cross-correlation (n_components > 1)

   * - RV_FAIL
     - RV failure. No good visits or RVs for this star


DR17 also had:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Flag
     - Description

   * - SUSPECT_ROTATION
     - CCFWHM > 2 * AUTOFWHM

   * - SUSPECT_BROAD_LINES
     - Cross-correlation peak with template significantly broader
       than autocorrelation of template:
       WARN (AUTOFWHM > 300) (will be restarted in 1.6)


Before DR17 we also had (when we used synthetic and combined templates):

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Flag
     - Description

   * - SUSPECT_RV_COMBINATION
     - RVs from synthetic template differ significantly
       (~2 km/s) from those from combined template: WARN

   * - BAD_RV_COMBINATION
     - RVs from synthetic template differ very significantly
       (~10 km/s) from those from combined template: BAD


Relevant Visit-level columns:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Column
     - Description

   * - STARFLAG
     - Quality flags for this visit spectrum
       
   * - STARVER
     - Version of the Doppler processing for this star.  Generally,
       the MJD of the last visit used with Doppler.
       
   * - GOODVISIT
     - Boolean flag indicating that the visit passed all the
       QA cuts and was used in determining the mean velocity
       and the combined spectrum
       
   * - VREL
     - Doppler shift (km/s)
       
   * - VRELERR
     - Uncertainty in VREL (km/s).
       
   * - VRAD
     - Barycentric radial velocity (km/s)
       
   * - BC
     - Barycentric correction for VREL (km/s)
       
   * - CHISQ
     - Chi-squared of the best-fit Cannon model
       
   * - RV_TEFF
     - Teff of the best-fit Cannon model
       
   * - RV_LOGG
     - log(g) of the best-fit Cannon model
       
   * - RV_FEH
     - Metallicity of the best-fit Cannon model
       
   * - XCORR_VREL
     - Cross-correlation doppler shift (km/s)

   * - XCORR_VRELERR
     - Uncertainty in XCORR_VREL (km/s)

   * - XCORR_VRAD
     - Barycentric radial velocity from cross-correlation (km/s)
       
   * - N_COMPONENTS
     - Number of components from the Gaussian decomposition
       of the cross-correlation function.  If N_COMPONENTS=0, then
       this means that the Gaussian decomposition did not detect a peak
       or it failed for the visit

   * - RV_COMPONENTS
     - Number of Gaussian components for each of the 3 detectors


Relevant Star-level columns:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - Column
     - Description

   * - STARFLAG
     - Quality flags OR-combined across visits

   * - ANDFLAG
     - Quality flags AND-combined across visits

   * - STARVER
     - Version of the Doppler processing for this star. Generally,
       the MJD of the last visit used with Doppler.
       
   * - NVISITS
     - Total number of visits for this star (no QA cuts)

   * - NGOODVISITS
     - Number of visits passing the initial QA cuts (no BAD STARFLAG flags and S/N>2)

   * - NGOODRVS
     - Number of visits passing all QA cuts and used in determining the
       mean RV and combined spectrum
       
   * - VRAD
     - Barycentric radial velocity (km/s)
       
   * - VSCATTER
     - Standard deviation of visit-level VRAD values (km/s)

   * - VERR
     - Uncertainty in VRAD. Standard deviation of the mean, VSCATTER/sqrt(NGOODRVS) (km/s).
       
   * - VMEDERR
     - Median uncertainty of the visit-level RVs (km/s).
       
   * - CHISQ
     - Chi-squared of the best-fit Cannon model to all visits
       
   * - RV_TEFF
     - Teff of the best-fit Cannon model (K)

   * - RV_TEFFERR
     - Uncertainty in RV_TEFF (K)
       
   * - RV_LOGG
     - log(g) of the best-fit Cannon model

   * - RV_LOGGERR
     - Uncertainty in RV_LOGG
       
   * - RV_FEH
     - Metallicity of the best-fit Cannon model

   * - RV_FEHERR
     - Uncertainty in RV_FEH

   * - RV_CCPFWHM
     - FWHM of the cross-correlation peak of the spectrum and best-fit Cannon model (km/s)

   * - RV_AUTOFWHM
     - FWHM of the cross-correlation peak of best-fit Cannon model with itself (km/s)

   * - N_COMPONENTS
     - Number of components from the Gaussian decomposition
       of the cross-correlation function. This the maximum N_COMPONENTS
       of the visits for this star.
       If N_COMPONENTS=0, then the Gaussian decomposition failed for all
       visits for this star.

   * - MEANFIB
     - Mean fiberID across the visits

   * - SIGFIB
     - Standard deviation of the fiberID across the visits.
       Note that if SIGFIB=0, then RV uncertainty is
       significantly better than normal.
