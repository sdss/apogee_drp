import os
import subprocess
import numpy as np
from astropy.io import fits
from scipy.signal import medfilt2d
from ..mkcal import getcal,readcal,getnums
from ...utils import apload
from .. import cal

chips = ['a','b','c']

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
    load = kw['load']
    detfile = load.filename('Detector',num=name,chips=True)
    detdir = os.path.dirname(load.filename('Detector',num=name,chips=True))
    fmt = '{:s}Detector-{:s}-{:08d}.fits'
    outfiles = [os.path.join(detdir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    if np.sum([os.path.exists(f) for f in outfiles])==3 and clobber==False:
        print(' detector file: ',detfile,' already made')
        return
    dettab = kw['allcaldict']['det'] 
    ind, = np.where(dettab['name']==str(name))
    if len(ind)==0:
        print('No matching calibration line for',name)
        return
    else:
        ind = ind[0]
    cal.mkdet(dettab[ind]['name'],dettab[ind]['linid'],unlock=kw['unlock'])  # clobber?

def dark(name,**kw):
    """ Make Dark calibration files """
    load = kw['load']
    darkfile = load.filename('Dark',num=name,chips=True)
    darkdir = os.path.dirname(load.filename('Dark',num=name,chips=True))
    fmt = '{:s}Dark-{:s}-{:08d}.fits'
    outfiles = [os.path.join(darkdir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    outfiles += [os.path.join(darkdir,load.prefix+'Dark-{:08d}.tab'.format(int(name)))]
    if np.sum([os.path.exists(f) for f in outfiles])==4 and clobber==False:
        print(' dark file: ',darkfile+'.tab',' already made')
        return
    darktab = kw['allcaldict']['dark']
    ind, = np.where(darktab['name']==str(name))
    if len(ind)==0:
        print('No matching calibration line for',name)
        return
    else:
        ind = ind[0]
    ims = getnums(darktab[ind]['frames'])
    mjd = int(load.cmjd(ims[0]))
    caldict = getcal(kw['calfile'],mjd)
    # Make the det calibration file exists
    makecal(caldict['detit'],'det',**kw.fromkeys(['clobber','unlock','load','calfile']))
    cal.mkdark(ims,clobber=kw['clobber'],unlock=kw['unlock'])

def flat(name,**kw):
    """ Make Flat calibration files """
    load = kw['load']
    flatfile = load.filename('Flat',num=name,chips=True)
    flatdir = os.path.dirname(load.filename('Flat',num=name,chips=True))
    fmt = '{:s}Flat-{:s}-{:08d}.fits'
    outfiles = [os.path.join(flatdir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    outfiles += [os.path.join(flatdir,load.prefix+'Flat-{:08d}.tab'.format(int(name)))]
    if np.sum([os.path.exists(f) for f in outfiles])==4 and clobber==False:
        print(' flat file: ',flatfile+'.tab',' already made')
        return
    flattab = kw['allcaldict']['flat']
    ind, = np.where(flattab['name']==str(name))
    if len(ind)==0:
        print('No matching calibration line for',name)
        return
    ind = ind[0]
    ims = getnums(flattab[ind]['frames'])
    cmjd = int(load.cmjd(ims[0]))
    caldict = getcal(kw['calfile'],mjd)
    # Make sure the dark exists
    makecal(caldict['darkid'],'dark',**kw.fromkeys(['clobber','unlock','load','calfile']))
    cal.mkflat(ims,darkid=caldict['darkid'],nrep=flattab[ind]['nrep'],dithered=flattab[ind]['dithered'],
               clobber=kw['clobber'],unlock=kw['unlock'])

def bpm(name,**kw):
    """ Make BPM calibration files """
    load = kw['load']
    bpmfile = load.filename('BPM',num=name,chips=True)
    bpmdir = os.path.dirname(load.filename('BPM',num=name,chips=True))
    fmt = '{:s}BPM-{:s}-{:08d}.fits'
    outfiles = [os.path.join(bpmdir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    if np.sum([os.path.exists(f) for f in outfiles])==3 and clobber==False:
        print(' bpm file: ',bpmfile, ' already made')
        return
    bpmtab = kw['allcaldict']['bpm']
    ind, = np.where(bpmtab['name']==str(name))
    if len(ind) == 0:
        print('No matching calibration line for',name)
        return
    ind = ind[0]
    # Make sure dark/flat that we need exist
    makecal(bpmtab[ind]['darkid'],'dark',**kw.fromkeys(['clobber','unlock','load','calfile']))
    makecal(bpmtab[ind]['flatid'],'flat',**kw.fromkeys(['clobber','unlock','load','calfile']))
    cal.mkbpm(bpmtab[ind]['name'],darkid=bpmtab[ind]['darkid'],flatid=bpmtab[ind]['flatid'],
              clobber=kw['clobber'],unlock=kw['unlock'])

def sparse(name,**kw):
    """ Make Sparsepak PSF calibration product """
    load = kw['load']
    sparsefile = load.filename('Sparse',num=name,chips=True)
    psfdir = os.path.dirname(load.filename('Sparse',num=name,chips=True))
    if os.path.exists(sparsefile) and clobber==False:
        print(' sparse file: ',sparsefile,' already made')
        return
    sparsetab = kw['allcaldict']['sparse'] 
    ind, = np.where(sparsetab['name']==str(name))
    if len(ind) < 0:
        print('No matching calibration line for',name)
        return
    ind = ind[0]
    ims = getnums(sparsetab[ind]['frames'])
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

def fiber(name,**kw):
    """ Make fiber calibration file """
    load = kw['load']
    psffile = load.filename('PSF',num=name,chips=True)
    psfdir = os.path.dirname(load.filename('PSF',num=name,chips=True))
    fmt = '{:s}{:s}-{:s}-{:08d}.fits'
    outfiles = [os.path.join(psfdir,fmt.format(load.prefix,'EPSF',ch,int(name))) for ch in chips]
    outfiles = [os.path.join(psfdir,fmt.format(load.prefix,'PSF',ch,int(name))) for ch in chips]
    if np.sum([os.path.exists(f) for f in outfiles])==6 and clobber==False:
        print(' psf file: ',psffile, ' already made')
        return
    cmjd = int(load.cmjd(name))
    caldict = getcal(kw['calfile'],mjd)
    cal.mkpsf(name,darkid=caldict['darkid'],flatid=caldict['flatid'],
              sparseid=caldict['sparseid'],unlock=kw['unlock'])

def psf(name,**kw):
    """ Make PSF calibration file """
    load = kw['load']
    #if keyword_set(psf) and not keyword_set(flux) and not keyword_set(wave)
    psffile = load.filename('PSF',num=name,chips=True)
    psfdir = os.path.dirname(load.filename('PSF',num=name,chips=True))
    fmt = '{:s}{:s}-{:s}-{:08d}.fits'
    outfiles = [os.path.join(psfdir,fmt.format(load.prefix,'EPSF',ch,int(name))) for ch in chips]
    outfiles += [os.path.join(psfdir,fmt.format(load.prefix,'PSF',ch,int(name))) for ch in chips]
    if np.sum([os.path.exists(f) for f in outfiles])==6 and clobber==False:
        print(' psf file: ',psffile, ' already made')
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
    load = kw['load']
    psffile = load.filename('PSFModel',num=name,chips=True)
    psfdir = os.path.dirname(load.filename('PSFModel',num=name,chips=True))
    fmt = '{:s}PSFModel-{:s}-{:08d}.fits'
    allfiles = [os.path.join(psfdir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    if np.sum([os.path.exists(f) for f in outfile])==3 and clobber==False:
        print(' modelpsf file: ',psffile, ' already made')
        return
    modelpsftab = kw['allcaldict']['modelpsf'] 
    ind, = np.where(modelpsftab['name']==str(name))
    if len(ind) < 0:
        print('No matching calibration line for',name)
        return
    ind = ind[0]
    makecal(modelpsftab[ind]['sparse'],'sparse',**kw.fromkeys(['clobber','unlock','load','calfile']))
    makecal(modelpsftab[ind]['psf'],'psf',**kw.fromkeys(['clobber','unlock','load','calfile']))
    cal.mkmodelpsf(name,sparseid=modelpsftab[ind]['sparse'],psfid=modelpsftab[ind]['psf'],
                   clobber=kw['clobber'],unlock=kw['unlock'])

def fpi(name,**kw):
    """ Make FPI calibration file """
    load = kw['load']
    wavefpifile = load.filename('WaveFPI',num=name,chips=True)
    wavefpidir = os.path.dirname(load.filename('WaveFPI',num=name,chips=True))
    cmjd = load.cmjd(name)
    mjd = int(mjd)
    fmt = '{:s}WaveFPI-{:s}-{:s}-{:08d}.fits'
    outfiles = [os.path.join(wavedir,fmt.format(load.prefix,ch,cmjd,int(name))) for ch in chips]
    if np.sum([os.path.exists(f) for f in outfiles])==3 and clobber==False:
        print,' fpi file: ',file, ' already made'
        return
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
    load = kw['load']
    litfile = load.filename('Littrow',num=name,chips=True)
    if os.path.exists(litfile) and clobber==False:
        print(' littrow file: ',litfile,' already made')
        return
    cmjd = load.cmjd(name)
    mjd = int(cmjd)
    caldict = getcal(kw['calfile'],mjd)
    makecal(caldict['flatid'],'flat',**kw.fromkeys(['clobber','unlock','load','calfile']))
    cal.mklittrow(name,cmjd=cmjd,darkid=caldict['darkid'],flatid=caldict['flatid'],
                  sparseid=caldict['sparseid'],fiberid=caldict['fiberid'],
                  clobber=kw['clobber'],unlock=kw['unlock'])

def persist(name,**kw):
    """ Make Persistence calibration file """
    load = kw['load']
    perfile = load.filename('Persist',num=name,chips=True)
    perdir = os.path.dirname(load.filename('Persist',num=name,chips=True))
    fmt = '{:s}Persist-{:s}-{:08d}.fits'
    outfiles = [os.path.join(perdir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    if np.sum([os.path.exists(f) for f in outfiles])==3 and clobber==False:
        print(' persist file: ',perfile, ' already made')
        return
    persisttab = kw['allcaldict']['persist'] 
    ind, = np.where(persisttab['name']==str(name))
    if len(ind) <= 0:
        print('No matching calibration line for',name)
        return
    ind = ind[0]
    cmd = load.cmjd(name)
    mjd = int(cmjd)
    caldict = getcal(kw['calfile'],mjd)
    cal.mkpersist(name,persisttab[ind]['darkid'],persisttab[ind]['flatid'],
                  thresh=persisttab[ind]['thresh'],cmjd=cmjd,darkid=caldict['darkid'],
                  flatid=caldict['flatid'],sparseid=caldict['sparseid'],
                  fiberid=caldict['fiberid'],clobber=kw['clobber'],unlock=kw['unlock'])

def persistmodel(name,**kw):
    """ Make Persistence model calibration file """
    load = kw['load']
    perfile = load.filename('PersistModel',num=name,chips=True)
    perdir = os.path.dirname(load.filename('PersistModel',num=name,chips=True))
    fmt = '{:s}PersistModel-{:s}-{:08d}.fits'
    outfiles = [os.path.join(perdir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    if np.sum([os.path.exists(f) for f in outfiles])==3 and clobber==False:
        print(' modelpersist file: ',file, ' already made')
        return
    persistmodeltab = kw['allcaldict']['persistmodel'] 
    ind, = np.where(persistmodeltab['name']==str(name))
    if len(ind)<0:
        print('No matching calibration line for',name)
        return
    ind = ind[0]
    mjd = int(load.cmjd(name))
    caldict = getcal(kw['calfile'],mjd)
    cal.mkpersistmodel(name)

def flux(name,**kw):
    """ Make Flux calibration file """
    load = kw['load']
    fluxfile = load.filename('Flux',num=name,chips=True)
    fluxdir = os.path.dirname(load.filename('Flux',num=name,chips=True))
    fmt = '{:s}Flux-{:s}-{:08d}.fits'
    outfiles = [os.path.join(fluxdir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    if np.sum([os.path.exists(f) for f in outfiles])==3 and clobber==False:
        print(' flux file: ',fluxfile, ' already made')
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
               clobber=kw['clobber'],unlock=kw['unlock'])

def response(name,**kw):
    """ Make Response calibration file """
    load = kw['load']
    resfile = load.filename('Response',num=name,chips=True)
    resdir = os.path.dirname(load.filename('Response',num=name,chips=True))
    fmt = '{:s}Response-{:s}-{:08d}.fits'
    outfiles = [os.path.join(resdir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    if np.sum([os.path.exists(f) for f in outfiles])==3 and clobber==False:
        print(' response file: ',resfile, ' already made')
        return
    responsetab = kw['allcaldict']['response'] 
    ind, = np.where(responsetab['name']==str(name))
    if len(ind) == 0:
        print('No matching calibration line for',name)
    else:
        if nres>1:
            i=i[0]
    mjd = int(load.cmjd(name))
    caldict = getcal(kw['calfile'],mjd)
    makecal(responsetab[ind]['psf'],'psf',**kw.fromkeys(['clobber','unlock','load','calfile']))
    makecal(caldict['waveid'],'wave',**kw.fromkeys(['clobber','unlock','load','calfile']))
    makecal(caldict['fiberid'],'fiber',**kw.fromkeys(['clobber','unlock','load','calfile']))
    makecal(caldict['littrowid'],'littrow',**kw.fromkeys(['clobber','unlock','load','calfile']))
    cal.mkflux(response,darkid=caldict['darkid'],flatid=caldict['flatid'],psfid=responsetab[ind]['psf'],
               littrowid=caldict['littrowid'],waveid=caldict['waveid'],temp=responsetab[ind]['temp'],
               clobber=kw['clobber'],unlock=kw['unlock'])

def wave(name,**kw):
    """ Make Wavelength calibration file """
    load = kw['load']
    wavefile = load.filename('Wave',num=name,chip='c')
    wavedir = os.path.dirname(load.filename('Wave',num=name,chip='c'))
    fmt = '{:s}Wave-{:s}-{:08d}.fits'
    outfiles = [os.path.join(wavedir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    if np.sum([os.path.exists(f) for f in outfiles])==3 and clobber==False:
        print(' wave file: ',wavefile, ' already made')
        return
    librarypsf = kw.get('librarypsf')
    psfid = kw.get('psfid')
    wavetab = kw['allcaldict']['wave']
    ind, = np.where(wavetab['name']==str(name))
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
    load = kw['load']
    """ Make multi-night wavelength calibration file """
    wavefile = load.filename('Wave',num=name,chips=True)
    wavedir = os.path.dirname(load.filename('Wave',num=name,chips=True))
    fmt = '{:s}Wave-{:s}-{:08d}.fits'
    outfiles = [os.path.join(wavedir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    outfiles += [os.path.join(wavedir,load.prefix+'Wave-{:08d}.py.dat'.format(int(name)))]
    if np.sum([os.path.exists(f) for f in outfiles])==4 and clobber==False:
        print(' multiwave file: ',wavefile+'.dat',' already made')
        return
    librarypsf = kw.get('librarypsf')
    multiwavetab = kw['allcaldict']['multiwave'] 
    ind, = np.where(multiwavetab['name']==str(name))
    if len(ind)==0:
        print('No matching calibration line for',name)
        return
    ind = ind[0]
    ims = getnums(multiwavetab[ind]['frames'])
    cal.mkmultiwave(ims,name=multiwavetab[ind]['name'],clobber=kw['clobber'],file=file,
                    unlock=kw['unlock'],psflibrary=librarypsf)

def dailywave(name,**kw):
    """ Make daily wavelength calibration file """
    load = kw['load']
    wavefile = load.filename('Wave',num=name,chips=True)
    wavedir = os.path.dirname(load.filename('Wave',num=name,chips=True))
    fmt = '{:s}Wave-{:s}-{:08d}.fits'
    outfiles = [os.path.join(wavedir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    if np.sum([os.path.exists(f) for f in outfiles])==3 and clobber==False:
        print(' dailywave file: ',wavefile,' already made')
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
    load = kw['load']
    telfile = load.filename('Telluric',num=name,chips=True)
    teldir = os.path.dirname(load.filename('Telluric',num=name,chips=True))
    fmt = '{:s}Telluric-{:s}-{:08d}.fits'
    outfiles = [os.path.join(teldir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    outfiles += [os.path.join(teldir,load.prefix+'Telluric-{:08d}.dat'.format(int(name)))]
    if np.sum([os.path.exists(f) for f in outfiles])==4 and clobber==False:
        print(' telluric file: ',telfile,' already made')
        return
    waveid = int(telluric.split('-')[0])
    lsfid = int(telluric.split('-')[1])
    if waveid < 1e7:
       makecal(waveid,'dailywave',**kw.fromkeys(['clobber','unlock','load','calfile']))
    else:
       makecal(waveid,'wave',**kw.fromkeys(['clobber','unlock','load','calfile']))
    makecal(lsfid,'lsf',**kw.fromkeys(['clobber','unlock','load','calfile']))
    cal.mktelluric(name,clobber=kw['clobber'],unlock=kw['unlock'])

def lsf(name,**kw):
    """ Make LSF calibration file """
    load = kw['load']
    lsffile = load.filename('LSF',num=name,chips=True)
    lsfdir = os.path.dirname(load.filename('LSF',num=name,chips=True))
    fmt = '{:s}LSF-{:s}-{:08d}.fits'
    outfiles = [os.path.join(lsfdir,fmt.format(load.prefix,ch,int(name))) for ch in chips]
    outfiles += [os.path.join(lsfdir,load.prefix+'LSF-{:08d}.sav'.format(int(name)))]
    if np.sum([os.path.exists(f) for f in outfiles])==4 and clobber==False:
        print(' lsf file: ',file+'.sav',' already made')
        return
    lsftab = kw['allcaldict']['lsf'] 
    ind, = np.where(lsftab['name']==str(name))
    if len(ind) <= 0:
        print('No matching calibration line for',name)
        return
    ind = ind[0]
    ims = getnums(lsftab[ind[0]]['frames'])
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
