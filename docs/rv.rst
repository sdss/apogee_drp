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
