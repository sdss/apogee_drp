*********************
Calibration Exposures
*********************

This page describes the APOGEE calibration exposures:

There several types of calibration exposures:

- darks
- internal flats
- quartz flats
- dome flats
- arcs (UNe and ThArNe)
- FPI
- sky flats (on sky, all stars offset by XX arcsec)
  
These calibrations can be taken in either the regular FullPak (illuminates all 300 fibers) or the SparsePak configuration (50 fibers, every 6th fiber). There is a separate gang cconnector port in the cal box for these two configurations.

**These pages describe the current calibrations taken during SDSS-V.**

Master calibrations
===================

On a roughly annual basis, longer calibration sequences are taken (often during engineering time or before/after summer shutdown at APO) in order to generate master calibration files.


Darks
-----

A sequence of 30x 100-read darks is taken.  It's best if there was a period of time with no light hitting the detectors to make sure there is minimal persistence in the darks.

Internal Flats
--------------

A sequence of 30x 30-read internal flats.  This is only possible at APO now, since the internal lamps are not working at LCO anymore.

Sky flat
--------

The idea is to get exposures where every fiber is on sky.  The airglow lines in the sky spectra are used to generate a new LSF master calibration file.

This is done on-sky.  Any design can be used, but the telescope needs to be offset by XX arcsec so that no fibers are on actual targets.  



Daily Calibrations
==================

Every afternoon and morning, a set of ``daily`` calibration exposures are taken.


.. parsed-literal::
   
    FILENAME	UT	NREAD	EXPTYPE	QRTZ	THAR	UNE	CONFIG	DESIGN	FIELD	SECZ	SEEING	OBSCMNT	COMMENT	COLLPIST	COLPITCH	DITHPIX	TCAMMID	TLSDETB
    56510001	19:24:45.604	60	DARK	0	0	0	10022464	745387	110537	1.0000000	1.6	None		0.0	0.0	13.496	81.948	77.395
    56510002	19:35:26.825	3	QUARTZFLAT	1	0	0	10022464	745387	110537	2.2400000	1.6			0.0	0.0	13.496	81.949	77.396
    56510003	19:36:04.588	20	ARCLAMP	0	1	0	10022464	745387	110537	2.2400000	1.6			0.0	0.0	13.496	81.949	77.395
    56510004	19:39:41.855	40	ARCLAMP	0	0	1	10022464	745387	110537	2.2400000	1.6			0.0	0.0	13.496	81.949	77.395
    56510005	19:46:59.362	20	ARCLAMP	0	1	0	10022464	745387	110537	2.2400000	1.6			0.0	0.0	12.995	81.947	77.396
    56510006	19:50:36.612	40	ARCLAMP	0	0	1	10022464	745387	110537	2.2400000	1.6			0.0	0.0	12.995	81.949	77.398
    56510007	19:57:51.935	60	DARK	0	0	0	10022464	745387	110537	2.2400000	1.6			0.0	0.0	12.995	81.949	77.398
    56510008	20:08:33.579	30	ARCLAMP	0	0	0	10022464	745387	110537	2.2400000	1.6	FPI A		0.0	0.0	12.995	81.949	77.399
    56510009	20:13:56.439	30	ARCLAMP	0	0	0	10022464	745387	110537	2.2400000	1.6	FPI B		0.0	0.0	13.496	81.949	77.4
    56510010	20:42:00.095	8	DOMEFLAT	0	0	0	10022464	745387	110537	1.0000000	1.6	None		0.0	0.0	13.496	81.947	77.402


The gang connector is connected to the full fiber cal box position for the calibration sequence.

Darks
-----
One 60 read dark starts the sequence.  Another 60 read dark is also taken between the lamp arcs and the FPI exposures.


Quartzflat
----------
A 3-read quartz flat exposure is taken.  The quartz lamp is bright and will quickly saturate the image with a longer exposure.  Fowler sampling is used to process these images, not up-the-ramp.
These illuminate 298 fibers (excluding the 2 decided FPI fibers) and the data are used to determine the PSF.

Arcs
----
UNe (Uranium-Neon) and ThArNe (Thorium-Argon-Neon) arclamp exposures are taken at both A and B dither positions.

FPI
---
Two 30 read full-frame FPI exposures are taken, A and B dithers.  This is used to generate the daily wavelength solutions. It's important that the final UNe exposure is at the **same** dither position as the first FPI exposure so that the two solutions are anchored.

Domeflat
--------
The gang connector is connected to the FPS and the dome lamps are turned on.  This exposure is used to determine the throughput of the fibers.



