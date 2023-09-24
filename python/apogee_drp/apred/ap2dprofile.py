import numpy as np
from astropy.io import fits
import matplotlib
import matplotlib.pyplot as plt
from dlnpyutils import utils as dln,plotting as pl

def gaussian2d(x,y,pars):
    """
    Two dimensional Gaussian model function.
    
    Parameters
    ----------
    x : numpy array
      Array of X-values of points for which to compute the Gaussian model.
    y : numpy array
      Array of Y-values of points for which to compute the Gaussian model.
    pars : numpy array or list
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta]
    deriv : boolean, optional
       Return the derivatives as well.
    nderiv : int, optional
       The number of derivatives to return.  The default is None
        which means that all are returned if deriv=True.

    Returns
    -------
    g : numpy array
      The Gaussian model for the input x/y values and parameters (same
        shape as x/y).

    Example
    -------

    g = gaussian2d(x,y,pars)

    """
    # pars = [amplitude, x0, y0, xsigma, ysigma, theta]

    xdiff = x - pars[1]
    ydiff = y - pars[2]
    amp = pars[0]
    xsig = pars[3]
    ysig = pars[4]
    theta = pars[5]
    cost2 = np.cos(theta) ** 2
    sint2 = np.sin(theta) ** 2
    sin2t = np.sin(2. * theta)
    xsig2 = xsig ** 2
    ysig2 = ysig ** 2
    a = ((cost2 / xsig2) + (sint2 / ysig2))
    b = ((sin2t / xsig2) - (sin2t / ysig2))    
    c = ((sint2 / xsig2) + (cost2 / ysig2))

    g = amp * np.exp(-0.5*((a * xdiff**2) + (b * xdiff * ydiff) +
                           (c * ydiff**2)))
    
    return g

def gaussian2d_integrate(x, y, pars, osamp=4):
    """
    Two dimensional Gaussian model function integrated over the pixels.

   
    Parameters
    ----------
    x : numpy array
      Array of X-values of points for which to compute the Gaussian model.
    y : numpy array
      Array of Y-values of points for which to compute the Gaussian model.
    pars : numpy array or list
       Parameter list. pars = [amplitude, x0, y0, xsigma, ysigma, theta]
    deriv : boolean, optional
       Return the derivatives as well.
    nderiv : int, optional
       The number of derivatives to return.  The default is None
        which means that all are returned if deriv=True.
    osamp : int, optional
       The oversampling of the pixel when doing the integrating.
          Default is 4.

    Returns
    -------
    g : numpy array
      The Gaussian model for the input x/y values and parameters (same
        shape as x/y).

    Example
    -------

    g = gaussian2d_integrate(x,y,pars)

    """

    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    
    # Deal with the shape, must be 1D to function properly
    shape = x.shape
    ndim = x.ndim
    if ndim>1:
        x = x.flatten()
        y = y.flatten()

    osamp2 = float(osamp)**2
    nx = x.size
    dx = (np.arange(osamp).astype(float)+1)/osamp-(1/(2*osamp))-0.5
    dx2 = np.tile(dx,(osamp,1))
    x2 = np.tile(x,(osamp,osamp,1)) + np.tile(dx2.T,(nx,1,1)).T
    y2 = np.tile(y,(osamp,osamp,1)) + np.tile(dx2,(nx,1,1)).T    
    
    # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
    theta = pars[5]
    cost2 = np.cos(theta) ** 2
    sint2 = np.sin(theta) ** 2
    sin2t = np.sin(2. * theta)
    xstd2 = pars[3] ** 2
    ystd2 = pars[4] ** 2
    xdiff = x2 - pars[1]
    ydiff = y2 - pars[2]
    a = 0.5 * ((cost2 / xstd2) + (sint2 / ystd2))
    b = 0.5 * ((sin2t / xstd2) - (sin2t / ystd2))
    c = 0.5 * ((sint2 / xstd2) + (cost2 / ystd2))
    g = pars[0] * np.exp(-((a * xdiff ** 2) + (b * xdiff * ydiff) +
                           (c * ydiff ** 2)))
    g = np.sum(np.sum(g,axis=0),axis=0)/osamp2
    if ndim>1:
        g = g.reshape(shape)
    
    return g


class PSF():

    def __init__(self,params,osamp=None):
        # Gaussian parameters
        # pars = [xsigma,ysigma,theta]
        self.params = params
        self.osamp = osamp

    @property
    def amp(self):
        """ the amplitude to use for a unit Gaussian  """
        # Volume is 2*pi*A*sigx*sigy
        return 1/(2*np.pi*self.params[0]*self.params[1])
        
    def __call__(self,xtrace=None,ytrace=None,bbox=None,osamp=None):
        """ Create PSF for an array of x/y trace positions. """

        # No positions input, just return an example
        if xtrace is None:
            return self(10,10,shape=(25,25))

        # How many positions
        if isinstance(xtrace,list) or isinstance(xtrace,np.ndarray):
            ntrace = len(xtrace)
        else:
            ntrace = 1
            xtrace = [xtrace]
            ytrace = [ytrace]

        if len(xtrace) != len(ytrace):
            raise Exception('xtrace and ytrace must have the same size')
            
        # How many pixels in the image
        if bbox is None:
            nx = (int(np.round(np.max(xtrace)))+10-(int(np.round(np.min(xtrace)))-10))
            ny = (int(np.round(np.max(ytrace)))+10-(int(np.round(np.min(ytrace)))-10))
            shape = (ny,nx)
            bbox = (0,ny,0,nx)
            npix = nx*ny
        elif len(bbox)==4:
            ny = bbox[1]-bbox[0]
            nx = bbox[3]-bbox[2]
            npix = ny*nx
        elif len(bbox)==2:
            ny,nx = bbox
            bbox = (0,ny,0,nx)
            npix = ny*nx            
        else:
            if len(bbox) != 4 and len(bbox) != 2:
                raise Exception('bbox must have 2 or 4 elements')
            
        # Loop over the trace positions
        out = np.zeros((npix,ntrace),float)
        for i in range(ntrace):
            psf1 = self.single(xtrace[i],ytrace[i],bbox,osamp=osamp)
            out[:,i] = psf1.ravel()
        if ntrace==1:
            out = out.squeeze()
            
        return out

    def single(self,x0,y0,bbox=None,osamp=None):
        """ Generate one PSF model."""
        # x: PSF x-position
        # y: PSF y position
        # bbox: bounding box [ymin,ymax,xmin,xmax]
        #    ymax/xmax are excluded
        #   can also be shape [ny,nx] and assume xmin/ymin=0
        if len(bbox)==4:
            ny = bbox[1]-bbox[0]
            nx = bbox[3]-bbox[2]
        else:
            ny,nx = bbox
            bbox = [0,ny,0,nx]
        xx,yy = np.meshgrid(np.arange(nx),np.arange(ny))
        pars = [self.amp,x0,y0]+self.params
        if osamp is None:  # use default value
            osamp = self.osamp
        if osamp is None or osamp==1:
            im = gaussian2d(xx,yy,pars)
        else:
            im = gaussian2d_integrate(xx,yy,pars,osamp=osamp)
        return im

    def model(self,xtrace,ytrace,flux=None,bbox=None,osamp=None):
        """ Simulate a full 2D image."""
        # x: PSF x-position
        # y: PSF y position
        # bbox: bounding box [ymin,ymax,xmin,xmax]
        #    ymax/xmax are excluded
        #   can also be shape [ny,nx] and assume xmin/ymin=0
        # How many positions
        if isinstance(xtrace,list) or isinstance(xtrace,np.ndarray):
            ntrace = len(xtrace)
        else:
            ntrace = 1
            xtrace = [xtrace]
            ytrace = [ytrace]        
        # Bounding Box, shape
        if bbox is None:
            nx = (int(np.round(np.max(xtrace)))+10-(int(np.round(np.min(xtrace)))-10))
            ny = (int(np.round(np.max(ytrace)))+10-(int(np.round(np.min(ytrace)))-10))
            shape = (ny,nx)
            bbox = (0,ny,0,nx)            
        elif len(bbox)==4:
            ny = bbox[1]-bbox[0]
            nx = bbox[3]-bbox[2]
            shape = (ny,nx)
        elif len(bbox)==2:
            ny,nx = bbox
            bbox = [0,ny,0,nx]
            shape = (ny,nx)
        else:
            raise Exception('bbox ',bbox,' not viable')
            
        # No flux input
        if flux is None:
            flux = np.ones(ntrace)
            
        # Loop over the positions
        im = np.zeros(shape,float)
        for i in range(len(xtrace)):
            im += flux[i]*self.single(xtrace[i],ytrace[i],bbox,osamp=osamp)

        return im

    def fit(self,im,xtrace,ytrace,err=None):
        """ Extract the flux from an image."""
        # Use the Bolton+Schlege (2010) spectroperfectionism method
        ntrace = len(xtrace)
        shape = im.shape
        npix = im.size
        if err is None:
            err = np.ones(shape,float)
        # Create the A, calibration array  [Npix,Ntrace]
        bbox = (0,shape[0],0,shape[1])
        A = self(xtrace,ytrace,bbox)   # 2D, [Npix,Ntrace]
        im1d = im.ravel()              # 1D, [Npix]
        wt = 1/err.ravel()**2          # 1D, [Npix]
        wt2d = np.identity(npix)*wt    # 2D, [Npix,Npix]
        # Solve for the flux
        # f = (A.T * N-1 * A)-1 * A.T * N-1 * p
        m1 = np.matmul(A.T,(wt*im1d))   # 1D, [Ntrace]
        m2 = np.matmul(A.T,np.matmul(wt2d,A))  # 2D [Ntrace,Ntrace]
        # Now invert m2
        m2inv = np.linalg.inv(m2)       # 2D, [Ntrace,Ntrace]
        f = np.matmul(m2inv,m1)         # 1D, [Ntrace]

        # Mask pixels that have essentially zero flux
        
        
        # Now reconvolve to get uncorrelated fluxes

        # inverse covariance matrix, C-1 = A.T * N-1 * A
        cinv = m2
        # determine the eigenbasis of C-1, taking the element-wise square root of the diagonal matrix
        #  of its eigenvalues (which will all be positive since C-1 is positive definite),
        #  and transforming this new diagonal matrix back using the unitary matrix that relates
        #  the eigenbasis to the original basis.
        #  lambda = eigen values
        #  E = eigen basis
        # Q = E I*sqrt(lambda) E-1
        eigvalues,eigbasis = np.linalg.eig(cinv)
        Q = np.matmul(eigbasis,np.matmul(np.identity(ntrace)*np.sqrt(eigvalues),np.linalg.inv(eigbasis)))
        # Next, definte a normalization vector s through
        #  s_l = Sum_l Q_lm
        s = np.sum(Q,axis=1)
        # matrix R through
        #  R_lm = s_l-1  Q_lm  (no sum)
        R = Q * 1/s
        # and a diagonal matrix C2-1 with entries given by
        #  C2_lm-1 = s_l^2
        c2inv = np.identity(ntrace) * s**2
        # By construction, we now have
        #  C-1 = R.T * C2-1 * R
        # and consequently
        #  C2 = R * C * R.T
        C = m2inv
        C2 = np.matmul(R,np.matmul(C,R.T))   # covariance matrix
        # Our extracted 1D spectrum is then
        #  flux = R*f
        f2 = np.matmul(R,f)

        return f2,C2
        
    
    def __str__(self):
        """ String representation of the PSF."""
        return self.__class__.__name__+'('+str(list(self.params))+')'

    def __repr__(self):
        """ String representation of the PSF."""
        return self.__class__.__name__+'('+str(list(self.params))+')'


def test():

    psf = PSF([1.2,0.9,0.24])
    
    xtrace = np.arange(5)*10+10
    ytrace = np.zeros(5)+10
    im = psf(xtrace,ytrace)

    shape = (21,61)
    im = np.zeros(shape,float)
    for i in range(len(xtrace)):
        im1 = psf.model(xtrace[i],ytrace[i],shape)
        im += im1

    xtrace = np.arange(10)+10
    ytrace = np.zeros(10)+10
    shape = (31,31)
    im = np.zeros(shape,float)
    for i in range(len(xtrace)):
        im1 = psf.model(xtrace[i],ytrace[i],shape)
        im += im1 

    im = psf(xtrace,ytrace,shape)
    # find pixels with any PSF in it
    tot = np.sum(im,axis=0)
    gd, = np.where(tot>1e-10)


    hdu = fits.open('/Users/nidever/sdss5/mwm/apogee/spectro/redux/daily/visit/apo25m/103120/9353/60041/apPlate-b-9353-60041.fits')
    
    
    # Flux array
    hdu = fits.open('/Users/nidever/synspec/nodegrid/grid6/flux1.fits')
    synflux = hdu[0].data
    #flux = synflux[4575:4630]
    flux = synflux[4575:4615]    

    xtrace = np.arange(len(flux))+10
    ytrace = np.zeros(len(flux))+10
    shape = (31,51)
    im = np.zeros(shape,float)
    for i in range(len(xtrace)):
        im1 = psf.model(xtrace[i],ytrace[i],shape)
        im += im1 * flux[i]

    im2d = psf(xtrace,ytrace,shape)

    flux = synflux[4575:4575+100]
    xtrace = np.arange(len(flux)) #+10
    coef = [-0.001,0.1,10]
    bbox = (0,21,0,100)
    ytrace = np.polyval(coef,xtrace)
    im2d = psf.model(xtrace,ytrace,flux,bbox)

    
# filename = '/Users/nidever/sdss5/mwm/apogee/spectro/redux/daily/exposures/apogee-s/60196/as2D-b-46340008.fits'
# hdu = fits.open(filename)
#
# pl.display(hdu[1].data,vmin=0,vmax=5000,xr=[400,450],yr=[400,450])
# im = hdu[1].data[434:489,293:489]
# err = hdu[2].data[434:489,293:489]
# from skimage.feature import peak_local_max
# coordinates = peak_local_max(im)
# yc = np.array([c[0] for c in coordinates])
# xc = np.array([c[1] for c in coordinates])
# ind, = np.where(im[yc,xc] > 2000)
# xc = xc[ind]
# yc = yc[ind]
# pl.display(im,vmin=0,vmax=5000)
# plt.scatter(xc,yc,c='r',marker='+')



from scipy.optimize import curve_fit
from prometheus.models import gaussian2d
def gauss2d(xdata,*pars):
  x,y = xdata
  return gaussian2d(x,y,pars).ravel()

def profile(xdata,*pars):
    # from Bolton+Schlegel eq. 5
    x,y = xdata
    # Unpack parameterse
    amp = pars[0]
    x0 = pars[1]
    y0 = pars[2]
    sigma = pars[3]  # Gaussian sigma
    q = pars[4]      # minor-to-major axis ratio
    theta = pars[5]  # rotation
    b = pars[6]      # fraction of flux in wing component    
    r0 = pars[7]     # characteristic size of profile wings
    cth = np.cos(theta)
    sth = np.sin(theta)
    dx = x-x0
    dy = y-y0
    # primed coordinates are translated and rotated
    xprime = dx*cth - dy*sth
    yprime = dx*sth + dy*cth
    r = np.sqrt(dx**2+dy**2)    
    rell = np.sqrt(q*xprime**2+(yprime**2)/q)
    m = (1-b)/(np.sqrt(2*np.pi)*sigma) * np.exp(-rell**2/(2*sigma**2)) + b*np.exp(-r/r0)/(2*np.pi*r0*r)
    #  goes to infinity at zero because of the 1/r term
    m *= amp
    return m

# exp(x) ~ 1+x for small x
# for small r the wing term becomes
# b*exp(-r/r0)/(2*pi*r0*r) ~ b*(1-r/r0)/(2*pi*r0*r) = b/(2*pi*r0*r) - b/(2*pi*r0^2)
# this blows up from the first 1/r term


# [amp,x0,y0,sigma,q,theta,b,r0]
#pars = [1.0,0.0,0.0,1.0,0.75,0.1,0.1,5]

# pars = [amplitude, x0, y0, xsigma, ysigma, theta]
#dt = [('x0',int),('y0',int),('x',float),('y',float),('pars',float,6),('perror',float,6),('chisq',float)]
#tab = np.zeros(len(xc),dtype=np.dtype(dt))

# Use a single Gaussian model and produce a 2D model image across the entire "image"
# -do this for all xtrace/ytrace values
# -ravel() or flatten() them, and "stack" to make the 2D "A" matrix


def apgtest():

    filename = '/Users/nidever/sdss5/mwm/apogee/spectro/redux/daily/exposures/apogee-s/60196/as2D-b-46340008.fits'
    hdu = fits.open(filename)

    #pl.display(hdu[1].data,vmin=0,vmax=5000,xr=[400,450],yr=[400,450])
    im = hdu[1].data[434:489,293:489]
    err = hdu[2].data[434:489,293:489]
    from skimage.feature import peak_local_max
    coordinates = peak_local_max(im)
    yc = np.array([c[0] for c in coordinates])
    xc = np.array([c[1] for c in coordinates])
    ind, = np.where(im[yc,xc] > 2000)
    xc = xc[ind]
    yc = yc[ind]
    #pl.display(im,vmin=0,vmax=5000)
    #plt.scatter(xc,yc,c='r',marker='+')
    
    dt = [('x0',int),('y0',int),('x',float),('y',float),('pars',float,6),('perror',float,6),('chisq',float)]
    tab = np.zeros(len(xc),dtype=np.dtype(dt))
    
    xall = np.array([])
    yall = np.array([])
    pixall = np.array([])
    for i in range(len(xc)):
        x0,y0 = xc[i],yc[i]
        xhalf = 5
        yhalf = 3
        xlo = x0-xhalf
        xhi = x0+xhalf
        ylo = y0-yhalf
        yhi = y0+yhalf
        sim = im[ylo:yhi+1,xlo:xhi+1]
        sim = np.maximum(sim,0)
        serr = err[ylo:yhi+1,xlo:xhi+1]
        serr = np.maximum(serr,1)
        xx,yy = np.meshgrid(np.arange(xhalf*2+1)-xhalf-1,np.arange(yhalf*2+1)-yhalf-1)
        xdata = [xx,yy]
        ydata = sim.ravel()
        sigma = serr.ravel()
        # pars = [amplitude, x0, y0, xsigma, ysigma, theta]
        # theta in radians
        initpars = [np.max(sim),0.0,0.0,1.2,0.9,0.26]
        mim0 = gaussian2d(xx,yy,initpars)
        resid = sim-mim0
        bounds = [np.zeros(6,float)-np.inf,np.zeros(6,float)+np.inf]
        bounds[0][0] = 0
        bounds[0][3] = 0
        bounds[0][4] = 0
        bounds[0][5] = 0.0
        bounds[1][5] = np.pi/2.
        pars,cov = curve_fit(gauss2d,xdata,ydata,sigma=sigma,p0=initpars,bounds=bounds)
        perror = np.sqrt(np.diag(cov))
        mim = gaussian2d(xx,yy,pars)
        chisq = np.sum((sim-mim)**2/serr)
        rchisq = chisq/sim.size
        tab['x0'][i] = x0
        tab['y0'][i] = y0
        tab['x'][i] = x0+pars[1]
        tab['y'][i] = y0+pars[2]
        tab['pars'][i] = pars
        tab['perror'][i] = perror
        tab['chisq'][i] = rchisq
        print(i,x0,y0,pars,rchisq)
        xall = np.append(xall,(xx-pars[1]).flatten())
        yall = np.append(yall,(yy-pars[2]).flatten())
        pixall = np.append(pixall,sim.flatten()/pars[0])

    for i in range(6): print(i,np.median(tab['pars'][:,i]))
    # 0 4530.970139828223
    # 1 -1.0878196515241454
    # 2 -1.0178424046530803
    # 3 1.2058528789253682
    # 4 0.9035683024817044
    # 5 0.24154751046780482
        
    pl.scatter(tab['x'],tab['y'],tab['pars'][:,3])

    # NICE!!!!
    pl.scatter(xall,yall,pixall,size=100)

    pl.scatter(xall,yall,pixall,size=100,log=True,vmin=0.002)


    from scipy.interpolate import CloughTocher2DInterpolator

    interp = CloughTocher2DInterpolator(list(zip(xall, yall)), pixall)
    X = np.linspace(min(xall), max(xall))
    Y = np.linspace(min(yall), max(yall))
    X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    Z = interp(X, Y)
    # not outlier resistant


    from scipy.interpolate import CloughTocher2DInterpolator
    import numpy as np
    import matplotlib.pyplot as plt
    rng = np.random.default_rng()
    x = rng.random(10) - 0.5
    y = rng.random(10) - 0.5
    z = np.hypot(x, y)
    X = np.linspace(min(x), max(x))
    Y = np.linspace(min(y), max(y))
    X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    interp = CloughTocher2DInterpolator(list(zip(x, y)), z)
    Z = interp(X, Y)
    
    # installed ndsplines, should do 2D bsplines
    # https://ndsplines.readthedocs.io/en/latest/?badge=latest

    import ndsplines
    spl = ndsplines.make_lsq_spline(samplex, sampley, ts, np.array([3,3]))

    # make_lsq_spline(x, y, knots, degrees, w=None, check_finite=True)
    # Construct a least squares regression B-spline.
    #
    # Parameters
    # ----------
    # x : array_like, shape (num_points, xdim)
    #     Abscissas.
    # y : array_like, shape (num_points, ydim)
    #     Ordinates.
    # knots : iterable of array_like, shape (n_1 + degrees[0] + 1,), ... (n_xdim, + degrees[-1] + 1)
    #     Knots and data points must satisfy Schoenberg-Whitney conditions.
    # degrees : ndarray, shape=(xdim,), dtype=np.int_                           
    # w : array_like, shape (num_points,), optional
    #     Weights for spline fitting. Must be positive. If ``None``,
    #     then weights are all equal. Default is ``None``.
