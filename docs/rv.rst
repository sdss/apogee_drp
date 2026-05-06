*****************
Radial Velocities
*****************

The APOGEE radial velocities (RVs) are determined with `Doppler <https://doppler.readthedocs.io/en/latest/>`_ software.

Radial velocities (RV) are determined on a star-by-star basis using Doppler which fits a Cannon model to the data,
determining Teff, logg, [Fe/H] and radial velocity in the process.  It actually fits all of the visit spectra simultaneously, using
a single set of stellar parameters, but a separate radial velocity for each spectrum.  The model spectrum is convolved with the
correct LSF (line spread profile) for each visit spectrum.  Each time a star is observed, all of the existing visit spectra for
that star are refit with Doppler.  This means that there will be multiple versions of RVs for each visit spectrum based on the
latest visit MJD (Modified Julian Date).  This is why there are the "visit_latest" and "star_latest" tables in the APOGEE database
(see `Data Access <access.html>`_).  They have the summary information at the visit and star level using the latest run/version
of Doppler.  The RV processing is parallelized over each object.

RV Columns and Flags:
 - Visit spectra must pass through QA checks to be used in the final spectral combination:
 1) Initial QA check: STARFLAG should not have any of the BAD flags set (BAD_PIXELS,VERY_BRIGHT_NEIGHBOR,BAD_RV_COMBINATION,RV_FAIL) and must have S/N>2. These are the visits that Doppler is run on and the ones that are counted in NGOODVISITS at the STAR level.
 2) RV check: After Doppler has been run on all of the visits that passed the initial QA cut, then an additional selection is made for visits that do not have RV_REJECT set.  These visits are used to determine the mean RV, RV scatter, and make the combined spectrum.  These are the visits that have GOODVISIT=True set and are counted in NGOODRVS at the STAR level.


Relevant flags in STARFLAG:

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
       WARN (AUTOFWHM > 300)


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
