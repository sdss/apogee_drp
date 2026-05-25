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

The bitmask accumulates information as the data progress through the
pipeline. Some bits originate during visit reduction, while others are
added during RV analysis or spectral combination.

At the combined-spectrum level, the ``STARFLAG`` typically represents
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
     - Spectrum contains a large fraction of bad pixels
   * - 1
     - ``COMMISSIONING``
     - Commissioning or non-standard data
   * - 2
     - ``BRIGHT_NEIGHBOR``
     - Nearby bright neighbor may contaminate spectrum
   * - 3
     - ``VERY_BRIGHT_NEIGHBOR``
     - Very bright nearby source likely contaminates spectrum
   * - 4
     - ``LOW_SNR``
     - Spectrum has low signal-to-noise
   * - 5
     - ``FAILED_REDUCTION``
     - Visit reduction failed
   * - 6
     - ``BAD_SKY_SUBTRACTION``
     - Significant sky subtraction problems
   * - 7
     - ``BAD_TELLURIC``
     - Telluric correction failure or poor correction
   * - 8
     - ``SUSPECT_FLUX``
     - Flux calibration or throughput issue
   * - 9
     - ``PERSIST_HIGH``
     - Significant high persistence contamination
   * - 10
     - ``PERSIST_MED``
     - Significant medium persistence contamination
   * - 11
     - ``PERSIST_LOW``
     - Significant low persistence contamination
   * - 12
     - ``PERSIST_JUMP_POS``
     - Positive persistence jump detected
   * - 13
     - ``PERSIST_JUMP_NEG``
     - Negative persistence jump detected
   * - 14
     - ``BAD_RADIAL_VELOCITY``
     - RV solution failed or unreliable
   * - 15
     - ``RV_REJECT``
     - Visit rejected during RV processing
   * - 16
     - ``SUSPECT_RV_COMBINATION``
     - RVs from different methods disagree
   * - 17
     - ``SUSPECT_BROAD_LINES``
     - Broad lines or rotational broadening suspected
   * - 18
     - ``MULTIPLE_SUSPECT``
     - Possible multiple stellar components
   * - 19
     - ``RV_VARIABLE``
     - Significant RV variability detected
   * - 20
     - ``VISIT_MISMATCH``
     - Visit spectra inconsistent with combined spectrum
   * - 21
     - ``COMBINATION_REJECT``
     - Visit rejected during spectral combination
   * - 22
     - ``RV_FAIL``
     - RV determination failed
   * - 23
     - ``SUSPECT_ROTATION``
     - Rotational broadening suspected from CCF width
   * - 24
     - ``MTPFLUX_LT_75``
     - Fiber throughput below 75 percent in MTP block
   * - 25
     - ``MTPFLUX_LT_50``
     - Fiber throughput below 50 percent in MTP block
   * - 26
     - ``BAD_COMBINED_SPECTRUM``
     - Combined spectrum quality failure
   * - 27
     - ``INSUFFICIENT_VISITS``
     - Too few good visits available
   * - 28
     - ``BAD_WAVELENGTH_CAL``
     - Wavelength calibration problem
   * - 29
     - ``BAD_CONTINUUM``
     - Continuum normalization failure
   * - 30
     - ``RESERVED_30``
     - Reserved for future use
   * - 31
     - ``RESERVED_31``
     - Reserved for future use

Visit-Level vs Combined-Level Flags
-----------------------------------

Some ``STARFLAG`` bits are naturally associated with individual visit
spectra:

- persistence,
- low S/N,
- sky subtraction issues,
- telluric correction problems,
- detector artifacts.

Other bits are generated during later processing stages:

- RV determination,
- visit rejection,
- spectral combination,
- variability assessment.

When new visits are added and the RV pipeline is rerun, RV-related bits
for both the visits and the combined spectrum may change. Therefore,
the combined ``STARFLAG`` should generally be considered the authoritative
summary of the current pipeline quality assessment.

Relationship to RVFLAG
----------------------

Additional RV-specific diagnostics are stored in the ``RVFLAG`` bitmask,
which provides more detailed information about radial velocity fitting
and rejection states.

In general:

- ``STARFLAG`` contains high-level user-facing quality information,
- ``RVFLAG`` contains detailed internal RV diagnostics.

Recommended Usage
-----------------

For most science applications, users should exclude spectra with severe
quality failures such as:

- ``VERY_BRIGHT_NEIGHBOR``
- ``BAD_RADIAL_VELOCITY``
- ``RV_FAIL``
- ``BAD_COMBINED_SPECTRUM``

Depending on the science goals, warning-level conditions such as
persistence or moderate RV variability may still be acceptable.
