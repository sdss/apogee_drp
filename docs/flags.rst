*************
Quality Flags
*************


STARFLAG Quality Bitmask
========================

The ``STARFLAG`` bitmask is used throughout the APOGEE reduction pipeline
to record warnings and failures associated with individual visit spectra,
radial velocity determination, and combined spectra.

``STARFLAG`` values are stored in:

- individual visit spectra (``apVisit`` files),
- combined stellar spectra (``apStar`` files),
- summary catalogs (``allVisit``, ``allStar``, etc.).

Because ``STARFLAG`` accumulates information from multiple processing
steps, a single value may contain bits that were set at different stages
of the pipeline. For example, some bits describe issues detected in an
individual visit spectrum, while others describe problems identified
during radial-velocity fitting or during construction of the combined
spectrum.

At the combined-spectrum level, the ``STARFLAG`` represents
the bitwise OR of all visit-level ``STARFLAG`` values plus any additional
flags generated during combination and RV processing.

Interpreting STARFLAG
---------------------

Each bit corresponds to a specific warning or failure condition.
Multiple conditions may be present simultaneously.

To test whether a bit is set:

.. code-block:: python

    bad = (starflag & 2**bitnum) != 0

For example:

.. code-block:: python

    low_snr = (starflag & 2**4) != 0

checks whether the ``LOW_SNR`` bit is set.

A ``STARFLAG`` value of zero indicates that none of the documented
``STARFLAG`` bits are set. It does not necessarily guarantee that the
spectrum is suitable for every science use case.


Visit and Combined STARFLAG Values
----------------------------------

At the visit level, ``STARFLAG`` describes the quality of an individual
visit spectrum and may include flags set during visit reduction and
radial-velocity analysis.

At the combined-spectrum level, ``STARFLAG`` summarizes the quality of
the combined spectrum. This value generally includes the bitwise OR of
the relevant visit-level ``STARFLAG`` values, together with any additional
bits set during radial-velocity analysis or spectral combination.

In this way, ``STARFLAG`` should be interpreted as an accumulated quality
summary. Users interested in the detailed origin of a flag should compare
the visit-level and combined-level values.


STARFLAG Bit Definitions
------------------------

.. list-table::
   :widths: 8 30 62
   :header-rows: 1

   * - Bit
     - Name
     - Description
   * - 0
     - ``BAD_PIXELS``
     - Spectrum contains a large fraction of bad pixels (>20%): BAD
   * - 1
     - ``COMMISSIONING``
     - Commissioning data (MJD<55761), non-standard configuration, poor LSF: WARN
   * - 2
     - ``BRIGHT_NEIGHBOR``
     - Star has neighbor more than 10 times brighter: WARN
   * - 3
     - ``VERY_BRIGHT_NEIGHBOR``
     - Star has neighbor more than 100 times brighter: BAD
   * - 4
     - ``LOW_SNR``
     - Spectrum has low signal-to-noise (S/N<5)
   * - 5
     - ``UNUSED``
     - Bit no longer used
   * - 6
     - ``UNUSED``
     - Bit no longer used
   * - 7
     - ``UNUSED``
     - Bit no longer used
   * - 8
     - ``UNUSED``
     - Bit no longer used
   * - 9
     - ``PERSIST_HIGH``
     - Spectrum has significant number (>20%) of pixels in high persistence region: WARN
   * - 10
     - ``PERSIST_MED``
     - Spectrum has significant number (>20%) of pixels in medium persistence region: WARN
   * - 11
     - ``PERSIST_LOW``
     - Spectrum has significant number (>20%) of pixels in low persistence region: WARN'
   * - 12
     - ``PERSIST_JUMP_POS``
     - Spectrum show obvious positive jump in blue chip: WARN
   * - 13
     - ``PERSIST_JUMP_NEG``
     - Spectrum show obvious negative jump in blue chip: WARN
   * - 14
     - ``UNUSED``
     - Bit no longer used
   * - 15
     - ``UNUSED``
     - Bit no longer used
   * - 16
     - ``SUSPECT_RV_COMBINATION``
     - RVs from synthetic template differ significantly (~2 km/s) from those from combined template: WARN
   * - 17
     - ``SUSPECT_BROAD_LINES``
     - Cross-correlation peak with template significantly broader than autocorrelation of template: WARN
   * - 18
     - ``BAD_RV_COMBINATION``
     - RVs from synthetic template differ very significatly (~10 km/s) from those from combined template: BAD
   * - 19
     - ``RV_REJECT``
     - Rejected visit because cross-correlation RV differs significantly from least squares RV
   * - 20
     - ``RV_SUSPECT``
     - Suspect visit (but used!) because cross-correlation RV differs slightly from least squares RV
   * - 21
     - ``MULTIPLE_SUSPECT``
     - Suspect multiple components from Gaussian decomposition of cross-correlation
   * - 22
     - ``RV_FAIL``
     - RV determination failed
   * - 23
     - ``SUSPECT_ROTATION``
     - Suspect rotation: cross-correlation peak with template significantly broader than autocorretion of template
   * - 24
     - ``MTPFLUX_LT_75``
     - Fiber throughput below 75 percent in MTP block
   * - 25
     - ``MTPFLUX_LT_50``
     - Fiber throughput below 50 percent in MTP block

Deprecated or Unused Bits
-------------------------

Some ``STARFLAG`` bits may be retained for backwards compatibility even
if they are no longer set by the current pipeline. These bits should not
be reused for a different purpose without careful consideration, since
older data products may still contain the historical meaning of the bit.

For bits that were used in previous reductions but are no longer active,
the recommended documentation style is:

.. code-block:: rst

   * - 17
     - ``OLD_FLAG_NAME``
     - Deprecated; no longer set by the current pipeline.

For bits that have never been assigned, the recommended documentation
style is:

.. code-block:: rst

   * - 30
     - ``RESERVED``
     - Reserved for future use.


Recommended Usage
-----------------

The appropriate use of ``STARFLAG`` depends on the science case. Some
bits indicate severe problems that usually warrant excluding a spectrum,
while others are warnings that may or may not matter for a particular
analysis.

For conservative analyses, users may wish to exclude spectra with serious
quality problems such as failed reductions, unreliable radial velocities,
severe contamination, or bad combined spectra. For less restrictive
analyses, warning-level bits such as low-level persistence or possible
throughput issues may be acceptable.

Users should inspect the individual bits rather than applying a single
``STARFLAG == 0`` requirement unless a very clean sample is required.

