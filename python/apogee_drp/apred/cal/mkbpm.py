import os

def mkbpm(bpmid, darkid=None, flatid=None, badrow=None, clobber=False, unlock=False):
    """
    Create an APOGEE bad pixel mask calibration file.

    Parameters
    ----------
    bpmid     ID8 number for this bad pixel mask.
    =darkid   ID8 number for the dark to use.
    =flatid   ID8 number for the flat to use.
    =badrow   Array of known bad rows
    /clobber  Overwrite existing files.
    /unlock   Delete lock file and start fresh 

    Returns
    -------
    BPM files (e.g. apBPM-a-ID8.fits) are created in the appropriate
    directory using the SDSS/APOGEE tree structure.

    Example
    -------

    mkbpm,12345678,darkid=00110012,flatid=00110013

    By J. Holtzman, 2011?
    Added doc strings and cleanup.  D. Nidever, Sep 2020
    """
    
    dirs = getdir()
    file = apogee_filename('BPM', num=bpmid, chip='c')
    lockfile = file + '.lock'

    # If another process is already making this file, wait!
    aplock(file, waittime=10, unlock=unlock)

    # Does the product already exist?
    # Check all three chip files
    sbpmid = str(bpmid).zfill(8)
    chips = ['a', 'b', 'c']
    bpmdir = apogee_filename('BPM', num=bpmid, chip='c', dir=True)
    allfiles = [os.path.join(bpmdir, dirs.prefix + f'BPM-{chip}-{sbpmid}.fits') for chip in chips]
    if all([os.path.exists(file) for file in allfiles]) and not clobber:
        print('BPM file:', file, 'already made')
        return
    # Delete any existing files to start fresh
    for file in allfiles:
        if os.path.exists(file):
            os.remove(file)

    print('Making BPM:', bpmid)

    # Open .lock file
    aplock(file, lock=True)

    for ichip in range(3):
        chip = chips[ichip]

        mask = np.zeros((2048, 2048), dtype=int)

        # Bad pixels from dark frame
        file = apogee_filename("Dark", chip=chip, num=darkid)
        darkmask = mrdfits(file, 3)
        bad = np.where(darkmask > 0)
        mask[bad] = mask[bad] | maskval('BADDARK')

        # Bad pixels from flat frame
        if flatid is not None:
            file = apogee_filename("Flat", chip=chip, num=flatid)
            flatmask = mrdfits(file, 3)
            bad = np.where(flatmask > 0)
            mask[bad] = mask[bad] | maskval('BADFLAT')
        else:
            flatmask = darkmask * 0

        # Flag them both as bad pixel (for historical compatibility?)
        bad = np.where((darkmask | flatmask) > 0)
        mask[bad] = mask[bad] | maskval('BADPIX')

        if badrow is not None:
            for row in badrow:
                if row.chip == ichip:
                    mask[:, row.row] = mask[:, row.row] | maskval('BADPIX')

        file = apogee_filename('BPM', chip=chip, num=bpmid)
        head1 = MRDFITS(file, 0, header=True)
        head1['EXTNAME'] = 'BPM'
        MKHDR(head1, mask)
        SXADDPAR(head1, 'EXTNAME', 'BPM')
        MWRFITS(mask, file, head1, create=True)

    aplock(file, clear=True)
