import os
import subprocess
import numpy as np
from astropy.io import fits
from scipy.signal import medfilt2d
from ..mkcal import getcal,readcal,getnums
from ...utils import apload
from .. import cal

chips = ['a','b','c']

_DATAMODEL_ROOTS = {
    "det": "Detector",
    "dark": "Dark",
    "flat": "Flat",
    "bpm": "BPM",
    "fiber": "Fiber",
    "sparse": "Sparse",
    "littrow": "Littrow",
    "modelpsf": "PSFModel",
    "psf": "PSF",
    "wave": "Wave",
    "multiwave": "Wave",
    "lsf": "LSF",
    "persist": "Persist",
    "response": "Response",
}

def makecal(name,caltype,apred=None,telescope=None,**kw):
    """
    This will make one or ALL of the specified calibration product types
    listed in the master calibration index file.

    Parameters
    ----------
    name : str
       ID or name of the calibration file.
    caltype : str
       Name of the calibration, e.g. "det", "dark", "wave".
         det, dark, flat, bpm, sparse, fiber, psf, modelpsf,
         fpi, littrow, persist, persistmodel, flux, 
         response, wave, multiwave, dailywave, telluric, lsf
    apred : str, optional
       APOGEE reduction version, e.g. "daily" or "1.4".
    telescope :str, optional
       Telescope name, either "apo25m" or "lco25m".
    load : ApLoad object, optional
       ApLoad object with reduction information.  Either this or 
         apred+telescope have to be input.
    calfile : str, optional
        Name of master calibration index file, if not
         specified use default cal.par in calibration directory
    Other calibration IDs can be input such as bpmid, flatid, darkid,
    that will then be passed on the other functions in the kw dictionary.

    Returns
    -------

    Calibration products are generated in places specified by the
    SDSS/APOGEE directory tree.

    Examples
    --------
    
    makecal(file=file,/dark,/flat,/wave,/lsf)
        OR
    makecal(file=file,dark=darkid,flat=flatid,wave=waveid,lsf=lsfid)
    
    Written by J.Holtzman Aug 2011
    Added doc strings and general cleanup by D. Nidever, Sep 2020
    Translated to Python  D. Nidever  2023/2024
    """

    if 'load' in kw.keys():
        load = kw['load']
    elif apred is not None and telescope is not None:
        load = apload.ApLoad(apred=apred,telescope=telescope)
        kw['load'] = load
    else:
        raise ValueError('Either apred+telescope or load have to be input')
        
    # Get default file name if file not specified
    if 'calfile' not in kw.keys() or kw['calfile'] is None or kw['calfile']=='':
        caldir = os.path.join(os.environ['APOGEE_DRP_DIR'],'data','cal')
        calfile = os.path.join(caldir,load.instrument+'.par')
    else:
        calfile = kw['calfile']
    kw['calfile'] = calfile
    # Get calibration dictionary
    if 'allcaldict' not in kw.keys():
        allcaldict = readcal(calfile)
        kw['allcaldict'] = allcaldict
    if 'verbose' not in kw.keys():
        kw['verbose'] = False
    
    funcdict = {'det':det,'dark':dark,'flat':flat,'bpm':bpm,'sparse':sparse,
                'fiber':fiber,'psf':psf,'modelpsf':modelpsf,'fpi':fpi,
                'littrow':littrow,'persist':persist,'persistmodel':persistmodel,
                'flux':flux,'response':response,'wave':wave,'multiwave':multiwave,
                'dailywave':dailywave,'telluric':telluric,'lsf':lsf}

    if caltype not in functdict.keys():
        raise ValueError(str(caltype)+' not supported')

    # Call the appropriate function
    if kw['verbose']:
        print('makecal '+str(caltype)+': '+str(name))
    functdict[caltype](name,**kw)

    
def det(name,**kw):
    """ Make Detector calibration files """
    caltype = 'det'
    calroot = _DATAMODEL_ROOTS[caltype]
    load = kw['load']
    detfile = load.filename(calroot,num=name,chips=True)
    if load.exists(calroot,num=name) and clobber==False:
        print(' detector file: ',detfile,' already made')
        return
    dettab = kw['allcaldict'][caltype] 
    ind, = np.where(dettab['name']==str(name))
    if len(ind)==0:
        print('No matching calibration line for',name)
        return
    else:
        ind = ind[0]
    cal.mkdet(dettab[ind]['name'],dettab[ind]['linid'],unlock=kw['unlock'])  # clobber?

def dark(name,**kw):
    """ Make Dark calibration files """
    caltype = 'dark'
    calroot = _DATAMODEL_ROOTS[caltype]
    load = kw['load']
    calfile = load.filename(calroot,num=name,chips=True)
    if load.exist(calroot,num=name) and clobber=False:
        print(' ',caltype,' file: ',calfile,' already made')
        return
    caltab = kw['allcaldict'][caltype]
    ind, = np.where(caltab['name']==str(name))
    if len(ind)==0:
        print('No matching calibration line for',name)
        return
    else:
        ind = ind[0]
    ims = getnums(caltab[ind]['frames'])
    mjd = int(load.cmjd(ims[0]))
    caldict = getcal(kw['calfile'],mjd)
    # Make the det calibration file exists
    makecal(caldict['detit'],'det',**kw.fromkeys(['clobber','unlock','load','calfile']))
    cal.mkdark(ims,**kw.fromkeys(['clobber','unlock']))

def flat(name,**kw):
    """ Make Flat calibration files """
    caltype = 'flat'
    calroot = _DATAMODEL_ROOTS[caltype]
    load = kw['load']
    calfile = load.filename(calroot,num=name,chips=True)
    if load.exists(calroot,num=name) and clobber==False:
        print(' ',caltype,' file: ',calfile,' already made')
        return
    caltab = kw['allcaldict'][caltype]
    ind, = np.where(caltab['name']==str(name))
    if len(ind)==0:
        print('No matching calibration line for',name)
        return
    ind = ind[0]
    ims = getnums(caltab[ind]['frames'])
    cmjd = int(load.cmjd(ims[0]))
    caldict = getcal(kw['calfile'],mjd)
    # Make sure the dark exists
    makecal(caldict['darkid'],'dark',**kw.fromkeys(['clobber','unlock','load','calfile']))
    cal.mkflat(ims,darkid=caldict['darkid'],nrep=caltab[ind]['nrep'],
               dithered=caltab[ind]['dithered'],**kw.fromkeys(['clobber','unlock']))

def bpm(name,**kw):
    """ Make BPM calibration files """
    caltype = 'bpm'
    calroot = _DATAMODEL_ROOTS[caltype]
    load = kw['load']
    calfile = load.filename(calroot,num=name,chips=True)
    if load.exists(calroot,num=name) and clobber=False:
        print(' ',caltype,' file: ',calfile, ' already made')
        return
    caltab = kw['allcaldict'][caltype]
    ind, = np.where(caltab['name']==str(name))
    if len(ind) == 0:
        print('No matching calibration line for',name)
        return
    ind = ind[0]
    # Make sure dark/flat that we need exist
    makecal(bpmtab[ind]['darkid'],'dark',**kw.fromkeys(['clobber','unlock','load','calfile']))
    makecal(bpmtab[ind]['flatid'],'flat',**kw.fromkeys(['clobber','unlock','load','calfile']))
    cal.mkbpm(bpmtab[ind]['name'],darkid=bpmtab[ind]['darkid'],flatid=bpmtab[ind]['flatid'],
              clobber=kw['clobber'],unlock=kw['unlock'])

def fiber(name,**kw):
    """ Make fiber calibration file """
    caltype = 'fiber'
    calroot = _DATAMODEL_ROOTS[caltype]
    load = kw['load']
    calfile = load.filename(calroot,num=name,chips=True)
    if load.exists(calroot,num=name) and clobber==False:
        print(' ',caltype,' file: ',calfile, ' already made')
        return
    cmjd = int(load.cmjd(name))
    caldict = getcal(kw['calfile'],mjd)
    cal.mkfiber(name,darkid=caldict['darkid'],flatid=caldict['flatid'],
                sparseid=caldict['sparseid'],unlock=kw['unlock'])

def sparse(name,**kw):
    """ Make Sparsepak PSF calibration product """
    caltype = 'sparse'
    calroot = _DATAMODEL_ROOTS[caltype]
    load = kw['load']
    calfile = load.filename(calroot,num=name,chips=True)
    if load.exists(calroot,num=name) and clobber==False:
        print(' ',caltype,' file: ',calfile,' already made')
        return
    caltab = kw['allcaldict'][caltype]
    ind, = np.where(caltab['name']==str(name))
    if len(ind) < 0:
        print('No matching calibration line for',name)
        return
    ind = ind[0]
    ims = getnums(caltab[ind]['frames'])
    mjd = int(load.cmjd(ims[0]))
    # Make sure dark/flat/bpm exist for this night
    caldict = getcal(kw['calfile'],mjd)
    makecal(caldict['darkid'],'dark',**kw.fromkeys(['clobber','unlock','load','calfile']))
    makecal(caldict['flatid'],'flat',**kw.fromkeys(['clobber','unlock','load','calfile']))
    makecal(caldict['bpmid'],'bpm',**kw.fromkeys(['clobber','unlock','load','calfile']))
    darkims = getnums(sparsetab[ind]['darkframes'])
    maxread = getnums(sparsetab[ind]['maxread'])
    if len(maxread) != 3:
        print('sparse maxread does not have 3 elements! ')
        return
    cal.mkepsf(ims,darkid=caldict['darkid'],flatid=caldict['flatid'],darkims=darkims,
               dmax=sparsetab[ind]['dmax'],maxread=maxread,clobber=kw['clobber'],
               filter=True,thresh=0.2,scat=2,unlock=kw['unlock'])
    # This creates apSparse and apEPSF files
    # Make empty apPSF files to indicate to makecal.pro that this
    #  PSF file was already made
    psffiles = [os.path.join(psfdir,load.prefix+'PSF-{:s}-{:08d}.fits'.format(ch,int(name))) for ch in chips]
    utils.touchzero(psffiles)

def psf(name,**kw):
    """ Make PSF calibration file """
    caltype = 'psf'
    calroot = _DATAMODEL_ROOTS[caltype]
    load = kw['load']
    #if keyword_set(psf) and not keyword_set(flux) and not keyword_set(wave)
    calfile = load.filename(calroot,num=name,chips=True)
    if load.exists(calroot,num=name) and clobber==False:
        print(' ',caltype,' file: ',calfile, ' already made')
        return
    cmjd = int(load.cmjd(name))
    caldict = getcal(kw['calfile'],mjd)
    makecal(caldict['littrowid'],'littrow',**kw.fromkeys(['clobber','unlock','load','calfile']))
    cal.mkpsf(psf,bpmid=caldict['bpmid'],darkid=caldict['darkid'],flatid=caldict['flatid'],
              sparseid=caldict['sparseid'],fiberid=caldict['fiberid'],
              littrowid=caldict['littrowid'],clobber=kw['clobber'],unlock=kw['unlock'])

def modelpsf(name,**kw):
    """ Make Model PSF calibration file """
    #if keyword_set(modelpsf) and (not keyword_set(fpi) and not keyword_set(flux) and not keyword_set(wave)) then begin
    caltype = 'psfmodel'
    calroot = _DATAMODEL_ROOTS[caltype]
    load = kw['load']
    calfile = load.filename(calroot,num=name,chips=True)
    if load.exists(calroot,num=name) and clobber==False:
        print(' ',caltype,' file: ',calfile, ' already made')
        return
    caltab = kw['allcaldict'][caltype]
    ind, = np.where(caltab['name']==str(name))
    if len(ind) < 0:
        print('No matching calibration line for',name)
        return
    ind = ind[0]
    makecal(caltab[ind]['sparse'],'sparse',**kw.fromkeys(['clobber','unlock','load','calfile']))
    makecal(caltab[ind]['psf'],'psf',**kw.fromkeys(['clobber','unlock','load','calfile']))
    cal.mkmodelpsf(name,sparseid=caltab[ind]['sparse'],psfid=caltab[ind]['psf'],
                   **kw.fromkeys(['clobber','unlock']))

def fpi(name,**kw):
    """ Make FPI calibration file """
    caltype = 'fpi'
    calroot = _DATAMODEL_ROOTS[caltype]
    load = kw['load']
    calfile = load.filename('WaveFPI',num=name,chips=True)
    if load.exists(calroot,num=name) and clobber==False:
        print,' ',caltype,' file: ',calfile, ' already made'
        return
    cmjd = load.cmjd(name)
    mjd = int(mjd)
    caldict = getcal(kw['calfile'],mjd)
    librarypsf = kw.get('librarypsf')
    psfid = kw.get('psfid')
    modelpsf = None
    # Use Model PSF by default
    if psfid is None and librarypsf is not True:
    #if not keyword_set(psf) and not keyword_set(librarypsf):
        psfid = None
        modelpsf = caldict.get('modelpsf')
        makecal(modelpsf,'modelpsf',**kw.fromkeys(['clobber','unlock','load','calfile']))
    # Use PSF file     
    else:
        # What PSF to use
        if keyword_set(psf):
            psfid = psf
        # Try to find a PSF from this day
        else:
            print,'Trying to automatically find a PSF calibration file'
            psfid = getpsfcal(fpi[0],psflibrary=librarypsf,unlock=kw['unlock'])
        makecal(psfid,'psf',**kw.fromkeys(['clobber','unlock','load','calfile']))
        
    makecal(caldict['fiberid'],'fiber',**kw.fromkeys(['clobber','unlock','load','calfile']))
    makecal(mjd,'dailywave',librarypsf=librarypsf,modelpsf=modelpsf,
            **kw.fromkeys(['clobber','unlock','load','calfile']))
    cal.mkfpi(name,name=name,darkid=caldict['darkid'],flatid=caldict['flatid'],psfid=psfid,
              fiberid=caldict['fiberid'],clobber=kw['clobber'],unlock=kw['unlock'],
              psflibrary=librarypsf,modelpsf=modelpsf)

def littrow(name,**kw):
    """ Make Littrow calibration file """
    caltype = 'littrow'
    calroot = _DATAMODEL_ROOTS[caltype]
    load = kw['load']
    calfile = load.filename(calroot,num=name,chips=True)
    if load.exists(calroot,num=name) and clobber==False:
        print(' ',caltype,' file: ',calfile,' already made')
        return
    cmjd = load.cmjd(name)
    mjd = int(cmjd)
    caldict = getcal(kw['calfile'],mjd)
    makecal(caldict['flatid'],'flat',**kw.fromkeys(['clobber','unlock','load','calfile']))
    cal.mklittrow(name,cmjd=cmjd,darkid=caldict['darkid'],flatid=caldict['flatid'],
                  sparseid=caldict['sparseid'],fiberid=caldict['fiberid'],
                  **kw.fromkeys(['clobber','unlock']))

def persist(name,**kw):
    """ Make Persistence calibration file """
    caltype = 'persist'
    calroot = _DATAMODEL_ROOTS[caltype]
    load = kw['load']
    calfile = load.filename(calroot,num=name,chips=True)
    if load.exists(calroot,num=name) and clobber==False:
        print(' ',caltype,' file: ',calfile, ' already made')
        return
    caltab = kw['allcaldict'][caltype] 
    ind, = np.where(caltab['name']==str(name))
    if len(ind) <= 0:
        print('No matching calibration line for',name)
        return
    ind = ind[0]
    cmd = load.cmjd(name)
    mjd = int(cmjd)
    caldict = getcal(kw['calfile'],mjd)
    cal.mkpersist(name,persisttab[ind]['darkid'],caltab[ind]['flatid'],
                  thresh=caltab[ind]['thresh'],cmjd=cmjd,darkid=caldict['darkid'],
                  flatid=caldict['flatid'],sparseid=caldict['sparseid'],
                  fiberid=caldict['fiberid'],**kw.fromkeys(['clobber','unlock']))

def persistmodel(name,**kw):
    """ Make Persistence model calibration file """
    caltype = 'persistmodel'
    calroot = _DATAMODEL_ROOTS[caltype]
    load = kw['load']
    calfile = load.filename(calroot,num=name,chips=True)
    if load.exists(calroot,num=name) and clobber==False:
        print(' ',caltype,' file: ',calfile, ' already made')
        return
    caltab = kw['allcaldict'][caltype]
    ind, = np.where(caltab['name']==str(name))
    if len(ind)<0:
        print('No matching calibration line for',name)
        return
    ind = ind[0]
    mjd = int(load.cmjd(name))
    caldict = getcal(kw['calfile'],mjd)
    cal.mkpersistmodel(name)

def flux(name,**kw):
    """ Make Flux calibration file """
    caltyp = 'flux'
    calroot = _DATAMODEL_ROOTS[caltype]
    load = kw['load']
    calfile = load.filename(calroot,num=name,chips=True)
    if load.exists(calroot,num=name) and clobber==False:
        print(' ',caltype,' file: ',calfile, ' already made')
        return
    librarypsf = kw.get('librarypsf')
    mjd = int(load.cmjd(name))
    caldict = getcal(kw['calfile'],mjd)
    psfid = kw.get('psfid')
    # Use Model PSF by default
    if psfid is None and librarypsf is not True and psfmodel is None:
    #if not keyword_set(psf) and not keyword_set(librarypsf) and keyword_set(psfmodel):
        psfid = None
        modelpsf = caldict.get('modelpsfid')
        makecal(modelpsf,'modelpsf',**kw.fromkeys(['clobber','unlock','load','calfile']))
    # Use PSF file     
    else:
        if keyword_set(psf):
            psfid = psf
        # Try to find a PSF from this day
        else:
            print('Trying to automatically find a PSF calibration file')
            psfid = getpsfcal(flux[0],psflibrary=librarypsf,unlock=kw['unlock'])
        makecal(psfid,'psf',**kw.fromkeys(['clobber','unlock','load','calfile']))
        
    makecal(littrowid,'littrow',**kw.fromkeys(['clobber','unlock','load','calfile']))
    cal.mkflux(flux,darkid=caldict['darkid'],flatid=caldict['flatid'],psfid=psfid,
               modelpsf=modelpsf,littrowid=caldict['littrowid'],waveid=caldict['waveid'],
               **kw.fromkeys(['clobber','unlock']))

def response(name,**kw):
    """ Make Response calibration file """
    caltype = 'response'
    calroot = _DATAMODEL_ROOTS[caltype]
    load = kw['load']
    calfile = load.filename(calroot,num=name,chips=True)
    if load.exists(calroot,num=name) and clobber==False:
        print(' ',caltype,' file: ',calfile, ' already made')
        return
    caltab = kw['allcaldict'][caltype]
    ind, = np.where(caltab['name']==str(name))
    if len(ind) == 0:
        print('No matching calibration line for',name)
    else:
        if nres>1:
            i=i[0]
    mjd = int(load.cmjd(name))
    caldict = getcal(kw['calfile'],mjd)
    makecal(caltab[ind]['psf'],'psf',**kw.fromkeys(['clobber','unlock','load','calfile']))
    makecal(caldict['waveid'],'wave',**kw.fromkeys(['clobber','unlock','load','calfile']))
    makecal(caldict['fiberid'],'fiber',**kw.fromkeys(['clobber','unlock','load','calfile']))
    makecal(caldict['littrowid'],'littrow',**kw.fromkeys(['clobber','unlock','load','calfile']))
    cal.mkflux(response,darkid=caldict['darkid'],flatid=caldict['flatid'],psfid=caltab[ind]['psf'],
               littrowid=caldict['littrowid'],waveid=caldict['waveid'],temp=caltab[ind]['temp'],
               **kw.fromkeys(['clobber','unlock']))

def wave(name,**kw):
    """ Make Wavelength calibration file """
    caltype = 'wave'
    calroot = _DATAMODEL_ROOTS[caltype]
    load = kw['load']
    calfile = load.filename(calroot,num=name,chip='c')
    if load.exists(calroot,num=name) and clobber==False:
        print(' ',caltype,' file: ',calfile, ' already made')
        return
    librarypsf = kw.get('librarypsf')
    psfid = kw.get('psfid')
    caltab = kw['allcaldict'][caltype]
    ind, = np.where(caltab['name']==str(name))
    if len(ind) > 0:
        ind = ind[0]
        ims = getnums(wavetab[ind]['frames'])
        name = wavetab[ind]['name']
        psfid = wavetab[ind]['psfid']
    # Use the input filename
    else:
        ims = wave
        name = ims[0]
        mjd = int(load.cmjd(name))
        caldict = getcal(kw['calfile'],mjd)
        # Use Model PSF by default
        if psfid is None and librarypsf is not True:
        #if not keyword_set(psf) and not keyword_set(librarypsf):
            psfid = None
            modelpsf = caldict.get('modelpsfid')
            makecal(modelpsf,'modelpsf',**kw.fromkeys(['clobber','unlock','load','calfile']))
        # Use PSF file     
        else:
            if keyword_set(psf):
                psfid = psf
            # Try to find a PSF from this day
            else:
                print('Trying to automatically find a PSF calibration file')
                psfid = getpsfcal(ims[0],psflibrary=librarypsf,unlock=kw['unlock'])
            makecal(psfid,'psf',**kw.fromkeys(['clobber','unlock','load','calfile']))
    mjd = int(load.cmjd(ims[0]))
    caldict = getcal(kw['calfile'],mjd)
    # Make sure we have the bpm/fiber files for this night
    makecal(caldict['bpmid'],'bpm',**kw.fromkeys(['clobber','unlock','load','calfile']))
    makecal(caldict['fiberid'],'fiber',**kw.fromkeys(['clobber','unlock','load','calfile']))
    cal.mkwave(ims,name=name,darkid=caldict['darkid'],flatid=caldict['flatid'],
               psfid=psfid,modelpsf=modelpsf,fiberid=caldict['fiberid'],
               clobber=kw['clobber'],nofit=nofit,unlock=kw['unlock'])
            
def multiwave(name,**kw):
    """ Make multi-night wavelength calibration file """
    caltype = 'multiwave'
    calroot = _DATAMODEL_ROOTS[caltype]
    load = kw['load']
    calfile = load.filename('Wave',num=name,chips=True)
    if load.exists('Wave',num=name) and clobber==False:
        print(' ',caltype,' file: ',calfile,' already made')
        return
    librarypsf = kw.get('librarypsf')
    caltab = kw['allcaldict'][caltype]
    ind, = np.where(caltab['name']==str(name))
    if len(ind)==0:
        print('No matching calibration line for',name)
        return
    ind = ind[0]
    ims = getnums(caltab[ind]['frames'])
    cal.mkmultiwave(ims,name=caltab[ind]['name'],clobber=kw['clobber'],file=file,
                    unlock=kw['unlock'],psflibrary=librarypsf)

def dailywave(name,**kw):
    """ Make daily wavelength calibration file """
    caltype = 'dailywave'
    calroot = _DATAMODEL_ROOTS[caltype]
    load = kw['load']
    calfile = load.filename('Wave',num=name,chips=True)
    wavedir = os.path.dirname(load.filename('Wave',num=name,chips=True))
    fmt = '{:s}Wave-{:s}-{:08d}.fits'
    outfiles = [os.path.join(wavedir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    if np.sum([os.path.exists(f) for f in outfiles])==3 and clobber==False:
        print(' ',caltype,' file: ',calfile,' already made')
        return
    mjd = int(name)
    caldict = getcal(kw['calfile'],mjd)
    librarypsf = kw.get('librarypsf')
    modelpsf = caldict.get('modelpsf')
    if librarypsf:
        modelpsf = None
    makecal(caldict['bpmid'],'bpm',**kw.fromkeys(['clobber','unlock','load','calfile']))
    makecal(caldict['fiberid'],'fiber',**kw.fromkeys(['clobber','unlock','load','calfile']))
    cal.mkdailywave(name,darkid=caldict['darkid'],flatid=caldict['flatid'],psfid=psfid,
                    fiberid=caldict['fiberid'],clobber=kw['clobber'],nofit=kw.get('nofit'),
                    unlock=kw['unlock'],psflibrary=librarypsf,modelpsf=modelpsf)

def telluric(name,**kw):
    """ Make daily telluric calibration file """
    caltype = 'telluric'
    calroot = _DATAMODEL_ROOTS[caltype]
    load = kw['load']
    calfile = load.filename(calroot,num=name,chips=True)
    if load.exists(calroot,num=name) and clobber==False:
        print(' ',caltype,' file: ',calfile,' already made')
        return
    waveid = int(telluric.split('-')[0])
    lsfid = int(telluric.split('-')[1])
    if waveid < 1e7:
       makecal(waveid,'dailywave',**kw.fromkeys(['clobber','unlock','load','calfile']))
    else:
       makecal(waveid,'wave',**kw.fromkeys(['clobber','unlock','load','calfile']))
    makecal(lsfid,'lsf',**kw.fromkeys(['clobber','unlock','load','calfile']))
    cal.mktelluric(name,**kw.fromkeys(['clobber','unlock']))

def lsf(name,**kw):
    """ Make LSF calibration file """
    caltype = 'lsf'
    calroot = _DATAMODEL_ROOTS[caltype]
    load = kw['load']
    calfile = load.filename(calroot,num=name,chips=True)
    if load.exists(calroot,num=name) and clobber==False:
        print(' ',caltype,' file: ',calfile,' already made')
        return
    caltab = kw['allcaldict'][caltype]
    ind, = np.where(caltab['name']==str(name))
    if len(ind) <= 0:
        print('No matching calibration line for',name)
        return
    ind = ind[0]
    ims = getnums(caltab[ind[0]]['frames'])
    mjd = int(load.cmjd(ims[0]))
    librarypsf = kw.get('librarypsf')
    modelpsf = caldict.get('modelpsf')
    if librarypsf:
        modelpsf = None
    caldict = getcal(kw['calfile'],mjd)
    makecal(caldict['multiwaveid'],'multiwave',librarypsf=librarypsf,modelpsf=modelpsf,
            **kw.fromkeys(['clobber','unlock','load','calfile']))  # librarypsf=librarypsf
    cal.mklsf(ims,name,darkid=caldict['darkid'],flatid=caldict['flatid'],
              fiberid=caldict['fiberid'],psfid=lsftab[ind]['psfid'],
              full=kw.get('full'),newwave=kw.get('newwave'),clobber=kw['clobber'],
              doplot=kw.get('dopl'),unlock=kw['unlock'])
