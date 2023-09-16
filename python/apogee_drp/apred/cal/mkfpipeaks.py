import os
import numpy as np

def mkfpipeaks(ims, clobber=False):
    """
    Makes APOGEE FPI Peaks master calibration.
    This is NOT run as part of the regular pipeline processing, but
    rather just once by hand.

    Parameters
    ----------
    ims: list of FPI full-frame image numbers to include in FPI peaks file.

    Returns
    -------
    An fpi_peaks-s/n.fits file.

    Example
    -------

    mkfpipeaks

    By D. Nidever, 2023
    """
    
    redux_dir = os.environ['APOGEE_REDUX'] + '/daily/'
    chips = ['a', 'b', 'c']

    # North/South loop
    for i in range(2):
        obs = ['apo', 'lco'][i]
        instrument = ['apogee-n', 'apogee-s'][i]
        telescope = ['apo25m', 'lco25m'][i]
        prefix = ['ap', 'as'][i]
        obstag = ['n', 's'][i]

        print('Making FPI Peaks file for', obs)
        print('------------------------------')

        # Get all of the FPILines files
        fpifiles = []
        for root, dirs, files in os.walk(redux_dir + 'cal/' + instrument + '/fpi/'):
            for file in files:
                if file.startswith(prefix + 'FPILines-') and file.endswith('.fits'):
                    fpifiles.append(os.path.join(root, file))

        nfpifiles = len(fpifiles)
        print(str(nfpifiles) + ' FPILines files')

        # Create giant structure to hold all the line information
        schema = [('x', float), ('row',float), ('height',float), ('flux',float),
                  ('sigma',float), ('chip',str,5), ('expnum',int), ('wave',float)]
        tab = np.zeros((nfpifiles * 150000),dtype=np.dtype(schema))

        # Load FPI lines files
        count = 0
        for j in range(min(nfpifiles, 51)):
            base = os.path.splitext(os.path.basename(fpifiles[j]))[0]
            print(j + 1, base)
            num = base.split('-')[0]
            mjd = getcmjd(int(num))
            # Load FPI lines files            
            wavefiles = redux_dir + 'cal/' + instrument + '/wave/' + prefix + 'Wave-' + ''.join(chips) + '-' + str(mjd).zfill(2) + '.fits'
            wtest = [os.path.exists(file) for file in wavefiles]
            if sum(wtest) < 3:
                print('no daily wave cal. skipping')
                continue
            fpitab = Table.read(fpifiles[j], 1, silent=True)
            fpitab['wave'] = 0.0
            fpitab['sigma'] = 0.0
            fpistr['expnum'] = int(num)
            # Get Corresponding daily wave file and add wavelengths
            x = np.arange(2048)
            for c in range(3):
                fits_read(wavefiles[c], wave, exten=2)
                ind = np.where(fpistr.chip == chips[c])[0]
                fpitab1 = fpitab[ind]
                index = dln.create_index(fpistr1['row'])
                rows = index['value']
                for k in range(len(rows)):
                    irow = rows[k]
                    rind = index['index'][index['lo'][k]:index['hi'][k]+1]
                    dwave = np.abs(slope(wave[:, irow]))
                    dwave = np.append(dwave, dwave[-1])
                    dw = dwave[fpistr1[rind].pix0]
                    ww = interpol(wave[:, irow], x, fpistr1[rind].pars[1])
                    fpistr1['wave'][rind] = ww
                    fpistr1['sigma'][rind] = fpitab1['pars'][rind,2] * dw
                fpitab[ind] = fpitab1

            # Only keep successful lines
            gd, = np.where(fpitab['success'] == True)[0]
            fpitab = fpitab[gd]
            nfpitab = len(fpitab)
            # Plug into big structure
            tab[count:count + nfpitab] = fpitab
            count += nfpitab

        tab = tab[:count]   #  trim the big structure
        print(str(count) + ' line measurements')

        # Find unique lines for each chip
        print('Finding unique peaks')
        peakstr = []

        for c in range(3):
            cind = np.where(tab['chip'] == chips[c])[0]
            cstr = tab[cind]
            hist, xhist = np.histogram(cstr['wave'], bins=range(2049))
            xhist += 0.5
            # Find peaks
            thresh = max(hist) / 5.0 > 1000
            gdpeaks = np.where(hist > thresh)[0]
            xpeaks = []
            wpeaks = []
            for p in range(len(gdpeaks)):
                wpk = xhist[gdpeaks[p]]
                if p > 0:
                    lastind = len(wpeaks) - 1
                    xdiff = xpeaks[lastind] - gdpeaks[p]
                    # merge neighboring pixels and replace with pixel of highest value                    
                    if abs(xdiff) <= 1:
                        xpeaks[lastind] = gdpeaks[p]                     # last value, not average
                        wpeaks[lastind] = (wpeaks[lastind] + wpk) * 0.5  # average the wavelengths
                    else:
                        xpeaks.append(gdpeaks[p])
                        wpeaks.append(wpk)
                else:
                    xpeaks.append(gdpeaks[p])
                    wpeaks.append(wpk)
            npeaks = len(wpeaks)
            print('chip', chips[c], str(npeaks) + ' unique peaks')

            # Now get all of the individual line measurements for each
            # unique line
            pkschema = [('id',int), ('chip',str,5), ('x',float), ('wave',float),
                        ('height',float), ('flux',float), ('sigma',float),
                        ('nfibers',int), ('nlines',int)]
            peaktab1 = np.zeros(npeaks, dtype=np.dtype(pkschema))
            peaktab1['chip'] = chips[c]
            for p in range(npeaks):
                pind = np.where(abs(cstr['wave'] - wpeaks[p]) < 1)[0]
                med = np.median(cstr['wave'][pind])
                sig = dln.mad(cstr['wave'][pind])
                pind = np.where(abs(cstr['wave'] - med) < (3 * sig > 0.1))[0]
                nfibers = len(np.unique(cstr['row'][pind]))
                peaktab1['id'][p] = p + 1
                peaktab1['x'][p] = np.mean(cstr['x'][pind])
                peaktab1['wave'][p] = np.mean(cstr['wave'][pind])
                peaktab1['height'][p] = np.median(cstr['height'][pind])
                peaktab1['flux'][p] = np.median(cstr['flux'][pind])
                peaktab1['sigma'][p] = np.median(cstr['sigma'][pind])
                peaktab1['nfibers'][p] = nfibers
                peaktab1['nlines'][p] = len(pind)
            peakstr.extend(peaktab1)
            wslp = slope(peaktab1['wave'])
            print('min/median/max wave steps:', min(wslp), np.median(wslp), max(wslp))
            print('min/median/max height:', min(peaktab1['height']), np.median(peaktab1['height']),
                  max(peaktab1['height']))
            print('min/median/max nfibers:', min(peaktab1['nfibers']), np.median(peaktab1['nfibers']),
                  max(peaktab1['nfibers']))
            print('min/median/max nlines:', min(peaktab1['nlines']), np.median(peaktab1['nlines']),
                  max(peaktab1['nlines']))

        npeaktab = len(peaktab)
        print(str(npeaktab) + ' unique FPI lines')

        # Save the table
        outfile = 'fpi_peaks-' + obstag + '.fits'
        print('Writing to', outfile)
        peakstr.write(outfile, overwrite=True)
