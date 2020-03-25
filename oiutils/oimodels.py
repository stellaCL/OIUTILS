import time
import copy
import multiprocessing

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import scipy.special
import scipy.interpolate
import scipy.stats

try:
    from oiutils import dpfit
except:
    import dpfit

_c = np.pi**2/180/3600/1000*1e6

def VsingleOI(oi, param, fov=None, pix=None, dx=0, dy=0):
    """
    build copy of OI, compute VIS, VIS2 and T3 for a single object parametrized
    with param

    oi: result from oiutils.loadOI, or a list of results

    param: a dictionnary with the possible keys defined below.

    fov: field of view of synthetic image (in mas). if None (default), will
        not produce a sinthetic image
    pix: pixel size (in mas) for synthetic image
    dx, dy: coordinates of the center of the image (in mas). default is 0,0

    Possible keys in the parameters dictionnary:

    position:
    ---------
    'x', 'y': define position in the field, in mas

    size (will decide the model):
    -----------------------------
    'ud': uniform disk diameter (in mas)
        or
    'fwhm': Full width at half maximum for a Gaussian (in mas)
        or
    'udout': outside diameter for a ring (in mas)
    'thick': fractional thickness of the ring (0..1)

    if none of these is given, the component will be fully resolved (V=0)

    stretching:
    -----------
    object can be stretched along one direction, using 2 additional parameters:
    'pa': projection angle from N (positive y) to E (positive x), in deg
    'i': inclination, in deg

    slant:
    ------
    'slant', 'slant pa'

    rings:
    ------
    rings are by default uniform in brightness. This can be altered by using
    different keywords:
    'profile': radial profile, can be 'uniform', 'doughnut' or 'power law'.
        if power law, 'power' gives the radial law
    'ampi', 'phii': defines the cos variation amplitude and phase for i nodes
        along the azimuth

    flux modeling:
    --------------
    if nothing is specified, flux is assume equal to 1 at all wavelengths

    'f' or 'f0': if the constant flux (as function of wavelength)
    'fi': polynomial amplitude in (wl-wl_min)**i (wl in um)
    'fpow', 'famp': famp*(wl-wl_min)**fpow (wl in um)
    spectral lines:
    'line_i_f': amplitude of line i (>0 for emission, <0 for absorption)
    'line_i_wl0': central wavelnegth of line i (um)
    'line_i_gaussian': fwhm for Gaussian profile (warning: in nm, not um!)
        or
    'line_i_lorentzian': width for Lorentzian (warning: in nm, not um!)

    """
    _param = computeLambdaParams(param)
    res = {}
    # -- what do we inherit from original data:
    for k in ['WL', 'header', 'insname', 'filename', 'fit']:
        if k in oi.keys():
            res[k] = oi[k]
    # -- model -> no telluric features
    res['TELLURICS'] = np.ones(res['WL'].shape)

    if not fov is None and not pix is None:
        # -- image coordinates in mas
        X, Y = np.meshgrid(np.linspace(-fov/2, fov/2, int(fov/pix)+1),
                           np.linspace(-fov/2, fov/2, int(fov/pix)+1))
        if not dx is None:
            X += dx
        if not dy is None:
            Y += dy
        I = np.zeros(X.shape)
    else:
        I = None

    # -- flux (spectrum)
    if 'spectrum' in _param.keys():
        f = np.zeros(res['WL'].shape)
    else:
        f = np.ones(res['WL'].shape)

    if 'f0' in _param.keys():
        f *= _param['f0']
    elif 'f' in _param.keys():
        f *= _param['f']

    As = filter(lambda x: x.startswith('f') and x[1:].isdigit(), _param.keys())
    for a in As:
        i = int(a[1:])
        if i>0:
            f += _param[a]*(res['WL']-np.min(res['WL']))**i
    if 'famp' in _param and 'fpow' in _param:
        f+= _param['famp']*(res['WL']-np.min(res['WL']))**_param['fpow']

    # -- list all the spectral lines, looking for fluxes:
    lines = filter(lambda x: x.startswith('line_') and x.endswith('_f'),
                    _param.keys())
    for l in lines:
        i = l.split('_')[1] # should not start with f!!!!
        wl0 = _param['line_'+i+'_wl0'] # in um
        if 'line_'+i+'_lorentzian' in _param.keys():
            dwl = _param['line_'+i+'_lorentzian'] # in nm
            f += _param[l]*1/(1+(res['WL']-wl0)**2/(dwl/1000)**2)
        if 'line_'+i+'_gaussian' in _param.keys():
            dwl = _param['line_'+i+'_gaussian'] # in nm
            if 'line_'+i+'_power' in _param.keys():
                _pow = _param['line_'+i+'_power']
            else:
                _pow = 2.0
            f += _param[l]*np.exp(-4*np.log(2)*np.abs(res['WL']-wl0)**_pow/
                                  (dwl/1000)**_pow)

    # -- arbitrary
    if 'spectrum' in _param.keys():
        sp = _param['spectrum']
        sp = sp.replace('WL', 'res["WL"]')
        for k in _param.keys():
            if k in sp:
                sp.replace(k, str(_param[k]))
        f += eval(sp)

    # -- check negativity of spectrum
    negativity = np.sum(f[f<0])/np.size(f)
    if 'fit' in oi and 'ignore negative flux' in oi['fit'] and \
        oi['fit']['ignore negative flux']:
        negativity = 0.0

    key = 'OI_VIS'
    if not key in oi.keys():
        key = 'OI_VIS2'
    if not 'OI_VIS2' in oi.keys():
        print('FIXME: cases with only OI_T3 not yet supported')
        return

    # -- max baseline
    Bwlmax = 0.0
    for k in oi[key].keys():
        Bwlmax = max(Bwlmax, np.max(oi[key][k]['B/wl']))

    # -- position of the element in the field
    if 'x' in _param.keys() and 'y' in _param.keys():
        x, y = _param['x'], _param['y']
    else:
        x, y = 0.0, 0.0

    # -- 'slant' i.e. linear variation of flux
    if 'slant' in list(_param.keys()) and 'slant pa' in list(_param.keys()):
        du, dv = 1e-6, 1e-6 # in meters / microns
        if _param['slant']<0:
            _param['slant'] = np.abs(_param['slant'])
            _param['slant pa'] = (_param['slant pa']+180)%360
    else:
        du, dv = 0.0, 0.0

    # -- do we need to apply a stretch?
    if 'pa' in _param.keys() and 'i' in _param.keys():
        rot = -_param['pa']*np.pi/180
        _uwl = lambda z: np.cos(_param['i']*np.pi/180)*\
                         (np.cos(rot)*z['u/wl'] + np.sin(rot)*z['v/wl'])
        _vwl = lambda z: -np.sin(rot)*z['u/wl'] + np.cos(rot)*z['v/wl']
        _Bwl = lambda z: np.sqrt(_uwl(z)**2 + _vwl(z)**2)

        if du:
            _udu = lambda z: (np.cos(rot)*z['u/wl'] +\
                              np.sin(rot)*z['v/wl'])*np.cos(np.pi*_param['i']/180)+du/oi['WL']
            _vdu = lambda z: -np.sin(rot)*z['u/wl'] + np.cos(rot)*z['v/wl']
            _Bdu = lambda z: np.sqrt(_udu(z)**2 + _vdu(z)**2)
            _udv = lambda z: (np.cos(rot)*z['u/wl'] +\
                              np.sin(rot)*z['v/wl'])*np.cos(np.pi*_param['i']/180)
            _vdv = lambda z: -np.sin(rot)*z['u/wl'] + np.cos(rot)*z['v/wl'] + dv/oi['WL']
            _Bdv = lambda z: np.sqrt(_udv(z)**2 + _vdv(z)**2)

        if not I is None:
            _X = (np.cos(rot)*(X-x) + np.sin(rot)*(Y-y))/np.cos(_param['i']*np.pi/180)
            _Y = -np.sin(rot)*(X-x) + np.cos(rot)*(Y-y)
            R = np.sqrt(_X**2+_Y**2)
    else:
        _uwl = lambda z: z['u/wl']
        _vwl = lambda z: z['v/wl']
        _Bwl = lambda z: z['B/wl']
        if du:
            _udu = lambda z: z['u/wl']+du/oi['WL']
            _vdu = lambda z: z['v/wl']
            _Bdu = lambda z: np.sqrt(_udu(z)**2 + _vdu(z)**2)
            _udv = lambda z: z['u/wl']
            _vdv = lambda z: z['v/wl']+dv/oi['WL']
            _Bdv = lambda z: np.sqrt(_udv(z)**2 + _vdv(z)**2)

        if not I is None:
            _X, _Y = X-x, Y-y
            R = np.sqrt(_X**2+_Y**2)

    # -- phase offset
    #phi = lambda z: -2j*_c*(z['u/wl']*x+z['v/wl']*y)
    PHI = lambda z: np.exp(-2j*_c*(z['u/wl']*x + z['v/wl']*y))
    if du:
        #dPHIdu = lambda z: -2j*_c*x*PHI(z)/oi['WL']
        #dPHIdv = lambda z: -2j*_c*y*PHI(z)/oi['WL']
        PHIdu = lambda z: np.exp(-2j*_c*((z['u/wl']+du/oi['WL'])*x + z['v/wl']*y))
        PHIdv = lambda z: np.exp(-2j*_c*(z['u/wl']*x + (z['v/wl']+dv/oi['WL'])*y))

    # -- guess which visibility function

    if 'ud' in _param.keys(): # == uniform disk ================================
        diamout = _param['ud']/2
        Vf = lambda z: 2*scipy.special.j1(_c*_param['ud']*_Bwl(z) + 1e-12)/(_c*_param['ud']*_Bwl(z)+ 1e-12)
        if _param['ud']<0:
            negativity += np.abs(_c*_param['ud']*Bwlmax)

        if du: # -- slanted
            Vfdu = lambda z: 2*scipy.special.j1(_c*_param['ud']*_Bdu(z))/(_c*_param['ud']*_Bdu(z))
            Vfdv = lambda z: 2*scipy.special.j1(_c*_param['ud']*_Bdv(z))/(_c*_param['ud']*_Bdv(z))
        if not I is None:
            # -- without anti aliasing
            #I = R<=_param['ud']/2
            # -- anti aliasing:
            na = 3
            for _dx in np.linspace(-pix/2, pix/2, na+2)[1:-1]:
                for _dy in np.linspace(-pix/2, pix/2, na+2)[1:-1]:
                    R2 = (_X-_dx)**2+(_Y-_dy)**2
                    I += R2<=(_param['ud']/2)**2
            I/=na**2
            if np.sum(I)==0:
                # -- unresolved -> single pixel
                R2 = _X**2+_Y**2
                I = R2==np.min(R2)

    elif 'fwhm' in _param.keys(): # == gaussian ================================
        #if _param['fwhm']<0:
        #    negativity += np.max(np.abs(_c*_param['fwhm']*_Bwl(z)))

        a = 1./(2.*(_param['fwhm']/2.355)**2)
        diamout = _param['fwhm']/2
        Vf = lambda z: np.exp(-(_c*_Bwl(z))**2/a)
        if du:
            print('WARNING: slanted gaussian does not make sense!')
            Vfdu = lambda z: np.exp(-(_c*_Bdu(z))**2/a)
            Vfdv = lambda z: np.exp(-(_c*_Bdv(z))**2/a)
        if not I is None:
            I = np.exp(-R**2*a)
            if np.sum(I)==0:
                # -- unresolved -> single pixel
                R2 = _X**2+_Y**2
                I = R2==np.min(R2)
    elif 'profile' in _param.keys(): # == F(r)*G(az) ========================
        if not 'thick' in _param:
            _param['thick'] = 1.0
        if 'Nr' in oi['fit']:
            Nr = oi['fit']['Nr']
        else:
            Nr = 10 + int(100*_param['thick']) # -- arbitrary !!!

        # -- radial profile -> depends...
        # -- amp1, pa1, ... ampi, pai
        # -- diam, thick
        udin = _param['diam']*(1-_param['thick'])

        r = np.linspace(udin/2, _param['diam']/2, Nr)
        mu = np.sqrt(1-(2*r/_param['diam'])**2)
        diamout = _param['diam']/2

        if _param['profile']=='uniform': # ring
            Ir = np.ones(r.shape)
        elif _param['profile']=='doughnut': # ring
            Ir = 1-((r-np.mean(r))/np.ptp(r)*2)**2
        #elif _param['profile']=='powerlaw': # ring
        #    Ir = r**_param['power']
        else:
            # -- generic formula
            tmp = _param['profile'].replace('R', 'r')
            tmp = tmp.replace('MU', 'mu')
            for k in _param.keys():
                tmp = tmp.replace(k, str(_param[k]))
            #print('profile:', tmp)
            Ir = eval(tmp)

        _n, _amp, _phi = [], [], []
        for k in _param.keys():
            if k.startswith('amp'):
                _n.append(int(k.split('amp')[1]))
                _phi.append(_param[k.replace('amp', 'pa')])
                _amp.append(_param[k])
        if 'pa' in _param.keys() and 'i' in _param.keys():
            stretch = [np.cos(np.pi*_param['i']/180), _param['pa']]
        else:
            stretch = [1,0]
        negativity += _negativityAzvar(_n, _phi, _amp)
        Vf = lambda z: _Vazvar(z['u/wl'], z['v/wl'], Ir, r, _n, _phi, _amp,
                                stretch=stretch)
        if du: # --slanted
            if len(_n):
                print('WARNING: slanted disk with azimutal variation not implemented properly!')
            Vfdu = lambda z: _Vazvar(_udu(z), _vdu(z), Ir, r, _n, _phi, _amp,
                                    stretch=stretch)
            Vfdv = lambda z: _Vazvar(_udv(z), _vdv(z), Ir, r, _n, _phi, _amp,
                                    stretch=stretch)
        if not I is None:
            I = _Vazvar(None, None, Ir, r, _n, _phi, _amp,
                        stretch=stretch, numerical=1, XY=(X-x,Y-y))
            if np.sum(I)==0:
                # -- unresolved -> single pixel
                R2 = _X**2+_Y**2
                I = R2==np.min(R2)

    elif 'udout' in _param.keys() and 'thick' in _param.keys(): # == ring =======
        diamout = _param['udout']/2
        udin = _param['udout']*(1-_param['thick'])
        Fout, Fin = _param['udout']**2, udin**2,
        Vf = lambda z: (Fout*2*scipy.special.j1(_c*_param['udout']*_Bwl(z))/
                        (_c*_param['udout']*_Bwl(z))-
                        Fin*2*scipy.special.j1(_c*udin*_Bwl(z))/
                                        (_c*udin*_Bwl(z)))/(Fout-Fin)
        if du: # -- slanted
            Vfdu = lambda z: (Fout*2*scipy.special.j1(_c*_param['udout']*_Bdu(z))/
                            (_c*_param['udout']*_Bdu(z))-
                            Fin*2*scipy.special.j1(_c*udin*_Bdu(z))/
                                                  (_c*udin*_Bdu(z)))/(Fout-Fin)
            Vfdv = lambda z: (Fout*2*scipy.special.j1(_c*_param['udout']*_Bdv(z))/
                            (_c*_param['udout']*_Bdv(z))-
                            Fin*2*scipy.special.j1(_c*udin*_Bdv(z))/
                                                  (_c*udin*_Bdv(z)))/(Fout-Fin)
        if not I is None:
            # -- without anti aliasing
            #I = (R<=_param['udout']/2)*(R>=udin/2)
            # -- anti aliasing:
            na = 3
            for _dx in np.linspace(-pix/2, pix/2, na+2)[1:-1]:
                for _dy in np.linspace(-pix/2, pix/2, na+2)[1:-1]:
                    R2 = (_X-_dx)**2+(_Y-_dy)**2
                    I += (R2<=(_param['udout']/2)**2)*(R2>=(udin/2)**2)
                I/=na**2
                if np.sum(I)==0:
                    # -- unresolved -> single pixel
                    R2 = _X**2+_Y**2
                    I = R2==np.min(R2)
    else:
        # -- default == fully resolved flux == zero visibility
        Vf = lambda z: np.zeros(_Bwl(z).shape)
        if du:
            Vfdu = lambda z: np.zeros(_Bdu(z).shape)
            Vfdv = lambda z: np.zeros(_Bdv(z).shape)
        # -- image is uniformly filled
        if not I is None:
            I += 1

    # -- slant in the image
    if du and not I is None:
        normI = np.sum(I)
        I *= (1.0 + np.sin(_param['slant pa']*np.pi/180)*_param['slant']/diamout*_X +
                    np.cos(_param['slant pa']*np.pi/180)*_param['slant']/diamout*_Y)
        #I *= normI/np.sum(I)

    # -- cos or sin variation
    # -- https://en.wikipedia.org/wiki/Fourier_transform#Tables_of_important_Fourier_transforms
    # -- item 115
    # ==> not implemented yet!
    if 'cos' in _param and 'cos pa' in _param:
        _C = np.cos(_param['cos pa']*np.pi/180)
        _S = np.sin(_param['cos pa']*np.pi/180)
        _xc = _S*_X + _C*_Y
        I *= 1 + _param['cos']*np.cos(_xc)
    if 'sin' in _param and 'sin pa' in _param:
        _C = np.cos(_param['sin pa']*np.pi/180)
        _S = np.sin(_param['sin pa']*np.pi/180)
        _xc = _S*_X + _C*_Y
        I *= 1 + _param['sin']*np.sin(_xc)

    # -- check that slant does not lead to negative flux
    if du and np.abs(_param['slant'])>1:
        negativity += (np.abs(_param['slant'])-1)

    res['OI_VIS'] = {}
    res['OI_VIS2'] = {}

    for k in oi[key].keys():
        # -- for each baseline
        tmp = {}
        for l in ['u', 'v', 'u/wl', 'v/wl', 'B/wl', 'MJD', 'FLAG']:
            tmp[l] = oi[key][k][l]
        if du: # -- for slanted
            V = Vf(oi[key][k])
            # compute slant from derivative of visibility
            dVdu = (Vfdu(oi[key][k]) - V)/du
            dVdv = (Vfdv(oi[key][k]) - V)/dv
            dVdu /= 2*_c/oi['WL']
            dVdv /= 2*_c/oi['WL']
            # -- see https://en.wikipedia.org/wiki/Fourier_transform#Tables_of_important_Fourier_transforms
            # -- relations 106 and 107
            V = V+1j*(np.sin(_param['slant pa']*np.pi/180)*_param['slant']/diamout*dVdu +
                     np.cos(_param['slant pa']*np.pi/180)*_param['slant']/diamout*dVdv)
            V *= PHI(oi[key][k])
        else:
            V = Vf(oi[key][k])*PHI(oi[key][k])



        tmp['|V|'] = np.abs(V)
        tmp['EV'] = np.zeros(tmp['|V|'].shape)
        # -- force -180 -> 180
        tmp['PHI'] = (np.angle(V)*180/np.pi+180)%360-180
        tmp['EPHI'] = np.zeros(tmp['PHI'].shape)
        res['OI_VIS'][k] = tmp
        if 'OI_VIS2' in oi.keys():
            tmp = {}
            for l in ['u', 'v', 'u/wl', 'v/wl', 'B/wl', 'MJD', 'FLAG']:
                tmp[l] = oi['OI_VIS2'][k][l]
            tmp['V2'] = np.abs(V)**2
            tmp['EV2'] = np.zeros(tmp['V2'].shape)
            res['OI_VIS2'][k] = tmp
    res['OI_T3'] = {}
    for k in oi['OI_T3'].keys():
        res['OI_T3'][k] = {}
        for l in ['u1', 'u2', 'v1', 'v2', 'MJD', 'formula', 'FLAG', 'Bmax/wl']:
            res['OI_T3'][k][l] = oi['OI_T3'][k][l]
    res = computeT3fromVisOI(res)

    if not I is None:
        # -- normalize image to total flux, useful for adding
        res['MODEL'] = {'image':I/np.sum(I), 'X':X, 'Y':Y}
    else:
        res['MODEL'] = {}
    res['MODEL']['totalflux'] = f
    res['MODEL']['negativity'] = negativity # 0 is image is >=0, 0<p<=1 otherwise
    return res

def VfromImageOI(oi):
    """
    oi dict must have key 'image' of 'cube' -> Vmodel with "fov" and "pix"
    """
    if type(oi)==list:
        return [VfromImageOI(o) for o in oi]

    if not 'MODEL' in oi.keys() and \
                not ('image' in oi['MODEL'].keys() or
                     'cube' in oi['MODEL'].keys()):
        print('WARNING: VfromImage cannot compute visibility from image')
        print('         run "Vmodel" with fov and pix values set before')
        return oi

    oi['IM_VIS'] = {}
    oi['IM_FLUX'] = {}

    for k in oi['OI_VIS'].keys():
        tmp = {}
        for l in ['u', 'v', 'u/wl', 'v/wl', 'B/wl', 'MJD', 'FLAG']:
            tmp[l] = oi['OI_VIS'][k][l]
        # -- dims: uv, wl, x, y
        phi = -2j*_c*(oi['MODEL']['X'][None,None,:,:]*tmp['u/wl'][:,:,None,None] +
                      oi['MODEL']['Y'][None,None,:,:]*tmp['v/wl'][:,:,None,None])
        # -- very CRUDE!
        if 'cube' in oi['MODEL'].keys():
            V = np.sum(oi['MODEL']['cube'][None,:,:,:]*np.exp(phi), axis=(2,3))/\
                       np.sum(oi['MODEL']['cube'], axis=(1,2))[None,:]
        else:
            V = np.sum(oi['MODEL']['image'][None,None,:,:]*np.exp(phi), axis=(2,3))/\
                       np.sum(oi['MODEL']['image'])
        tmp['|V|'] = np.abs(V)
        tmp['V2'] = np.abs(V)**2
        tmp['PHI'] = (np.angle(V)*180/np.pi+180)%360 - 180
        oi['IM_VIS'][k] = tmp
    for k in oi['OI_FLUX'].keys():
        oi['IM_FLUX'][k] = {'FLUX':oi['OI_FLUX'][k]['FLAG']*0 +
                                np.sum(oi['MODEL']['cube'], axis=(1,2))[None,:],
                            'MJD':oi['OI_FLUX'][k]['MJD'],
                            'FLAG':oi['OI_FLUX'][k]['FLAG'],
                            }
    oi = computeT3fromVisOI(oi)
    oi = computeDiffPhiOI(oi)
    oi = computeNormFluxOI(oi)
    return oi

def VmodelOI(oi, p, fov=None, pix=None, dx=0.0, dy=0.0, timeit=False):
    param = computeLambdaParams(p)
    if type(oi) == list:
        return [VmodelOI(o, param, fov=fov, pix=pix, dx=dx, dy=dy,
                        timeit=timeit) for o in oi]
    # -- split in components if needed
    comp = set([x.split(',')[0] for x in param.keys() if ',' in x])
    if len(comp)==0:
        return VsingleOI(oi, param, fov=fov, pix=pix, dx=dx, dy=dy)

    res = {} # -- triggers initialisation below
    t0 = time.time()
    for c in comp:
        # -- this component. Assumes all param without ',' are common to all
        _param = {k.split(',')[1]:param[k] for k in param.keys() if
                  k.startswith(c+',')}
        _param.update({k:param[k] for k in param.keys() if not ',' in k})
        if res=={}:
            res = VsingleOI(oi, _param, fov=fov, pix=pix, dx=dx, dy=dy)
            if 'image' in res['MODEL'].keys():
                res['MODEL'][c+',image'] = res['MODEL']['image']
                res['MODEL']['cube'] = res['MODEL']['image'][None,:,:]*res['MODEL']['totalflux'][:,None,None]
                res['MODEL']['image'] *= np.mean(res['MODEL']['totalflux'])
            res['MODEL'][c+',flux'] = res['MODEL']['totalflux'].copy()
            res['MODEL'][c+',negativity'] = res['MODEL']['negativity']

            res['MOD_VIS'] = {}
            for k in res['OI_VIS'].keys():
                res['MOD_VIS'][k] = res['MODEL']['totalflux'][None,:]*res['OI_VIS'][k]['|V|']*\
                                    np.exp(1j*np.pi*res['OI_VIS'][k]['PHI']/180)
            m = {}
        else:
            m = VsingleOI(oi, _param, fov=fov, pix=pix, dx=dx, dy=dy)
            if 'image' in m['MODEL'].keys():
                res['MODEL'][c+',image'] = m['MODEL']['image']
                res['MODEL']['image'] += np.mean(m['MODEL']['totalflux'])*\
                                        m['MODEL']['image']
                res['MODEL']['cube'] += m['MODEL']['image'][None,:,:]*\
                                        m['MODEL']['totalflux'][:,None,None]
            res['MODEL'][c+',flux'] = m['MODEL']['totalflux']
            res['MODEL'][c+',negativity'] = m['MODEL']['negativity']
            res['MODEL']['totalflux'] += m['MODEL']['totalflux']
            res['MODEL']['negativity'] += m['MODEL']['negativity']
            for k in res['OI_VIS'].keys():
                res['MOD_VIS'][k] += m['MODEL']['totalflux'][None,:]*m['OI_VIS'][k]['|V|']*\
                                np.exp(1j*np.pi*m['OI_VIS'][k]['PHI']/180)

    if timeit:
        print('VmodelOI > VsingleOI (x%d) %.3fms'%(len(comp),
                                                   1000*(time.time()-t0)))

    t0 = time.time()
    # -- normalise by total flux, compute OI_VIS and OI_VIS2
    for k in res['MOD_VIS'].keys():
        res['MOD_VIS'][k] /= res['MODEL']['totalflux'][None,:]
        res['OI_VIS'][k]['|V|'] = np.abs(res['MOD_VIS'][k])
        res['OI_VIS'][k]['PHI'] = (np.angle(res['MOD_VIS'][k])*180/np.pi+180)%360-180
        res['OI_VIS2'][k]['V2'] = np.abs(res['MOD_VIS'][k])**2

    res['OI_FLUX'] = {}
    if 'OI_FLUX' in oi.keys():
        for k in oi['OI_FLUX'].keys():
            res['OI_FLUX'][k] = {'FLUX': oi['OI_FLUX'][k]['FLUX']*0 +
                                 res['MODEL']['totalflux'][None,:],
                                 'EFLUX': oi['OI_FLUX'][k]['FLUX']*0,
                                 'FLAG': oi['OI_FLUX'][k]['FLAG'],
                                 'MJD': oi['OI_FLUX'][k]['MJD'],
            }
    if timeit:
        print('VmodelOI > fluxes %.3fms'%(1000*(time.time()-t0)))

    t0 = time.time()
    res['OI_T3'] = {}
    for k in oi['OI_T3'].keys():
        res['OI_T3'][k] = {}
        for l in ['MJD', 'u1', 'u2', 'v1', 'v2', 'formula', 'FLAG', 'Bmax/wl']:
            res['OI_T3'][k][l] = oi['OI_T3'][k][l]

    res = computeT3fromVisOI(res)
    if timeit:
        print('VmodelOI > T3 %.3fms'%(1000*(time.time()-t0)))

    t0 = time.time()
    if 'fit' in oi and 'obs' in oi['fit'] and 'DPHI' in oi['fit']['obs']:
        res = computeDiffPhiOI(res, param)
        if timeit:
            print('VmodelOI > dPHI %.3fms'%(1000*(time.time()-t0)))
            t0 = time.time()
    if 'fit' in oi and 'obs' in oi['fit'] and 'NFLUX' in oi['fit']['obs']:
        res = computeNormFluxOI(res, param)
        if timeit:
            print('VmodelOI > normFlux %.3fms'%(1000*(time.time()-t0)))

    res['param'] = computeLambdaParams(param)

    t0 = time.time()
    if 'fit' in res and 'spec res pix' in res['fit']:
        # -- convolve by spectral Resolution
        N = 2*int(2*oi['fit']['spec res pix'])+3
        x = np.arange(N)
        ker = np.exp(-(x-np.mean(x))**2/(2.*(oi['fit']['spec res pix']/2.355)**2))
        ker /= np.sum(ker)
        if 'NFLUX' in res.keys():
            for k in res['NFLUX'].keys():
                for i in range(len(res['NFLUX'][k]['MJD'])):
                    res['NFLUX'][k]['NFLUX'][i] = np.convolve(
                                res['NFLUX'][k]['NFLUX'][i], ker, mode='same')
        for k in res['OI_VIS'].keys():
            for i in range(len(res['OI_VIS'][k]['MJD'])):
                res['OI_VIS'][k]['|V|'][i] = np.convolve(
                            res['OI_VIS'][k]['|V|'][i], ker, mode='same')
                res['OI_VIS'][k]['PHI'][i] = np.convolve(
                            res['OI_VIS'][k]['PHI'][i], ker, mode='same')
                res['OI_VIS2'][k]['V2'][i] = np.convolve(
                            res['OI_VIS2'][k]['V2'][i], ker, mode='same')
                if 'DPHI' in res.keys():
                    res['DPHI'][k]['DPHI'][i] = np.convolve(
                                res['DPHI'][k]['DPHI'][i], ker, mode='same')
        for k in res['OI_T3'].keys():
            for i in range(len(res['OI_T3'][k]['MJD'])):
                res['OI_T3'][k]['T3PHI'][i] = np.convolve(
                            res['OI_T3'][k]['T3PHI'][i], ker, mode='same')
                res['OI_T3'][k]['T3AMP'][i] = np.convolve(
                            res['OI_T3'][k]['T3AMP'][i], ker, mode='same')
        if timeit:
            print('VmodelOI > convolve %.3fms'%(1000*(time.time()-t0)))
    return res

def computeDiffPhiOI(oi, param=None, order='auto'):
    if not param is None:
        _param = computeLambdaParams(param)
    else:
        _param = None
    if type(oi)==list:
        return [computeDiffPhiOI(o, _param, order) for o in oi]
    if 'param' in oi.keys() and param is None:
        _param = oi['param']

    if not 'OI_VIS' in oi.keys():
        #print('WARNING: computeDiffPhiOI, nothing to do')
        return oi

    # -- user-defined wavelength range
    fit = {'wl ranges':[(min(oi['WL']), max(oi['WL']))]}

    if not 'fit' in oi:
        oi['fit'] = fit
    elif not 'wl ranges' in oi['fit']:
        oi['fit'].update(fit)

    w = np.zeros(oi['WL'].shape)
    for WR in oi['fit']['wl ranges']:
        w += (oi['WL']>=WR[0])*(oi['WL']<=WR[1])
    oi['WL mask'] = np.bool_(w)

    # -- exclude where lines are in the models
    if not _param is None:
        for k in _param.keys():
            if 'line_' in k and 'wl0' in k:
                if k.replace('wl0', 'gaussian') in _param.keys():
                    dwl = 1.2*_param[k.replace('wl0', 'gaussian')]/1000.
                if k.replace('wl0', 'lorentzian') in _param.keys():
                    dwl = 6*_param[k.replace('wl0', 'lorentzian')]/1000.
                w *= (np.abs(oi['WL']-_param[k])>=dwl)

    if np.sum(w)==0:
        #print('WARNING: no continuum! using all wavelengths')
        w = oi['WL']>0

    oi['WL cont'] = np.bool_(w)
    w = np.bool_(w)

    if order=='auto':
        order = int(np.ptp(oi['WL'][oi['WL cont']])/0.2)
        order = max(order, 1)

    if np.sum(oi['WL cont'])<order+1:
        print('ERROR: not enough WL to compute continuum!')
        return oi

    oi['DPHI'] = {}
    for k in oi['OI_VIS'].keys():
        data = []
        for i,phi in enumerate(oi['OI_VIS'][k]['PHI']):
            mask = w*~oi['OI_VIS'][k]['FLAG'][i,:]
            c = np.polyfit(oi['WL'][mask], phi[mask], order)
            data.append(phi-np.polyval(c, oi['WL']))
        data = np.array(data)
        oi['DPHI'][k] = {'DPHI':data,
                         'EDPHI':oi['OI_VIS'][k]['EPHI'],
                         'FLAG':oi['OI_VIS'][k]['FLAG'],
                         'MJD':oi['OI_VIS'][k]['MJD'],
                         }
    if 'IM_VIS' in oi.keys():
        for k in oi['IM_VIS'].keys():
            data = []
            for i,phi in enumerate(oi['IM_VIS'][k]['PHI']):
                mask = w
                c = np.polyfit(oi['WL'][mask], phi[mask], order)
                data.append(phi-np.polyval(c, oi['WL']))
            data = np.array(data)
            oi['IM_VIS'][k]['DPHI'] = data
    return oi

def computeNormFluxOI(oi, param=None, order='auto'):
    if not param is None:
        _param = computeLambdaParams(param)
    else:
        _param = None

    if type(oi)==list:
        return [computeNormFluxOI(o, _param, order) for o in oi]

    if not 'OI_FLUX' in oi.keys():
        #print('WARNING: computeNormFluxOI, nothing to do')
        return oi

    if 'param' in oi.keys() and param is None:
        _param = oi['param']

    # -- user defined wavelength range
    w = oi['WL']>0
    if 'fit' in oi and 'wl ranges' in oi['fit']:
        w = np.zeros(oi['WL'].shape)
        for WR in oi['fit']['wl ranges']:
            w += (oi['WL']>=WR[0])*(oi['WL']<=WR[1])
    oi['WL mask'] = np.bool_(w)

    # -- exclude where lines are in the models
    if not _param is None:
        for k in _param.keys():
            if 'line_' in k and 'wl0' in k:
                if k.replace('wl0', 'gaussian') in _param.keys():
                    dwl = 1.2*_param[k.replace('wl0', 'gaussian')]/1000.
                if k.replace('wl0', 'lorentzian') in _param.keys():
                    dwl = 6*_param[k.replace('wl0', 'lorentzian')]/1000.
                w *= (np.abs(oi['WL']-_param[k])>=dwl)

    if np.sum(w)==0:
        #print('WARNING: no continuum! using all wavelengths')
        w = oi['WL']>0

    oi['WL cont'] = np.bool_(w)
    w = np.bool_(w)

    if order=='auto':
        order = int(np.ptp(oi['WL'][oi['WL cont']])/0.15)
        order = max(order, 1)

    if np.sum(oi['WL cont'])<order+1:
        print('ERROR: not enough WL to compute continuum!')
        return oi

    oi['NFLUX'] = {}
    # -- normalize flux in the data:
    for k in oi['OI_FLUX'].keys():
        data = []
        edata = []
        for i,flux in enumerate(oi['OI_FLUX'][k]['FLUX']):
            mask = w*~oi['OI_FLUX'][k]['FLAG'][i,:]
            c = np.polyfit(oi['WL'][w], flux[w]/oi['TELLURICS'][w], order)
            data.append(flux/oi['TELLURICS']/np.polyval(c, oi['WL']))
            edata.append(oi['OI_FLUX'][k]['EFLUX'][i]/np.polyval(c, oi['WL']))
        data = np.array(data)
        edata = np.array(edata)
        oi['NFLUX'][k] = {'NFLUX':data,
                         'ENFLUX':edata,
                         'FLAG':oi['OI_FLUX'][k]['FLAG'],
                         'MJD':oi['OI_FLUX'][k]['MJD']}

    # -- flux computed from image cube
    if 'IM_FLUX' in oi.keys():
        for k in oi['IM_FLUX'].keys():
            data = []
            for i,flux in enumerate(oi['IM_FLUX'][k]['FLUX']):
                mask = w*~oi['IM_FLUX'][k]['FLAG'][i,:]
                c = np.polyfit(oi['WL'][w], flux[w], order)
                data.append(flux/np.polyval(c, oi['WL']))
            data = np.array(data)
            oi['IM_FLUX'][k]['NFLUX'] = data

    if 'MODEL' in oi.keys() and 'totalflux' in oi['MODEL'].keys():
        mask = w
        c = np.polyfit(oi['WL'][w], oi['MODEL']['totalflux'][w], order)
        for k in list(oi['MODEL'].keys()):
            if k.endswith(',flux'):
                oi['MODEL'][k.replace(',flux', ',nflux')] = \
                    oi['MODEL'][k]/np.polyval(c, oi['WL'])
        oi['MODEL']['totalnflux'] = oi['MODEL']['totalflux']/np.polyval(c,oi['WL'])
    return oi

def computeLambdaParams(paramsI):
    params = {}
    loop = True
    nloop = 0
    while loop and nloop<10:
        loop = False
        for k in paramsI.keys():
            if type(paramsI[k])==str:
                # -- allow parameter to be expression of others
                tmp = paramsI[k]
                compute = False
                for _k in paramsI.keys():
                    if _k in paramsI[k]:
                        tmp = tmp.replace(_k, str(paramsI[_k]))
                        compute = True
                # -- are there still un-computed parameters?
                for _k in paramsI.keys():
                    if _k in tmp:
                        # -- set function to another loop
                        loop = True
                        paramsI[k] = tmp
                if compute and not loop:
                    try:
                        params[k] = eval(tmp)
                    except:
                        params[k] = tmp
                else:
                    params[k] = tmp
            else:
                params[k] = paramsI[k]
        nloop+=1


    assert nloop<10, 'too many recurences in evaluating parameters!'+str(paramsI)
    return params

def computeT3fromVisOI(oi):
    """
    oi => from oifits.loadOI()

    assumes OI contains the complex visibility as OI_VIS

    - no errors are propagated (used for modeling purposes)
    - be careful: does it in place (original OI_T3 data are erased)
    """
    if type(oi) == list:
        return [computeT3fromVisOI(o) for o in oi]
    if 'OI_T3' in oi.keys():
        for k in oi['OI_T3'].keys():
            s, t, w0, w1, w2 = oi['OI_T3'][k]['formula']
            oi['OI_T3'][k]['T3PHI'] = s[0]*oi['OI_VIS'][t[0]]['PHI'][w0,:]+\
                                      s[1]*oi['OI_VIS'][t[1]]['PHI'][w1,:]+\
                                      s[2]*oi['OI_VIS'][t[2]]['PHI'][w2,:]

            # -- force -180 -> 180 degrees
            oi['OI_T3'][k]['T3PHI'] = (oi['OI_T3'][k]['T3PHI']+180)%360-180
            oi['OI_T3'][k]['ET3PHI'] = np.zeros(oi['OI_T3'][k]['T3PHI'].shape)

            oi['OI_T3'][k]['T3AMP'] = np.abs(oi['OI_VIS'][t[0]]['|V|'][w0,:])*\
                                    np.abs(oi['OI_VIS'][t[1]]['|V|'][w1,:])*\
                                    np.abs(oi['OI_VIS'][t[2]]['|V|'][w2,:])
            oi['OI_T3'][k]['ET3AMP'] = np.zeros(oi['OI_T3'][k]['T3AMP'].shape)
    if 'IM_VIS' in oi.keys():
        oi['IM_T3'] = {}
        for k in oi['OI_T3'].keys():
            # -- inherit flags from Data,
            oi['IM_T3'][k] = {'FLAG':oi['OI_T3'][k]['FLAG'],
                               'MJD':oi['OI_T3'][k]['MJD'],
                               'Bmax/wl':oi['OI_T3'][k]['Bmax/wl']}
            s, t, w0, w1, w2 = oi['OI_T3'][k]['formula']
            oi['IM_T3'][k]['T3PHI'] = s[0]*oi['IM_VIS'][t[0]]['PHI'][w0,:]+\
                                       s[1]*oi['IM_VIS'][t[1]]['PHI'][w1,:]+\
                                       s[2]*oi['IM_VIS'][t[2]]['PHI'][w2,:]

            # -- force -180 -> 180 degrees
            oi['IM_T3'][k]['T3PHI'] = (oi['IM_T3'][k]['T3PHI']+180)%360-180

            oi['IM_T3'][k]['T3AMP'] = np.abs(oi['IM_VIS'][t[0]]['|V|'][w0,:])*\
                                       np.abs(oi['IM_VIS'][t[1]]['|V|'][w1,:])*\
                                       np.abs(oi['IM_VIS'][t[2]]['|V|'][w2,:])
    return oi


def testTelescopes(k, telescopes):
    """
    k: a telescope, baseline or triplet (str)
        eg: 'A0', 'A0G1', 'A0G1K0' etc.
    telescopes: single or list of telescopes (str)
        eg: 'G1', ['G1', 'A0'], etc.

    returns True if any telescope in k

    assumes all telescope names have same length!
    """
    if type(telescopes)==str:
        telescopes = [telescopes]
    test = False
    for t in telescopes:
        test = test or (t in k)
    return test

def testBaselines(k, baselines):
    """
    k: a telescope, baseline or triplet (str)
        eg: 'A0', 'A0G1', 'A0G1K0' etc.
    baselines: single or list of baselines (str)
        eg: 'G1A0', ['G1A0', 'A0C1'], etc.

    returns True if any baseline in k
        always False for telescopes
        order of telescope does not matter

    assumes all telescope names have same length!
    """
    if type(baselines)==str:
        baselines = [baselines]
    test = False
    for b in baselines:
        test = test or (b[:len(b)//2] in k and b[len(b)//2:] in k)
    return test

def chi2OI(oi, param):
    res = residualsOI(oi, param)

def computePrior(param, prior):
    res = []
    for p in prior.keys():
        form = p+''
        val = str(prior[p][1])+''
        for i in range(3):
            for k in param.keys():
                if k in form:
                    form = form.replace(k, str(param[k]))
                if k in val:
                    val = val.replace(k, str(param[k]))
        # -- residual
        resi = '('+form+'-'+str(val)+')/abs('+str(prior[p][2])+')'
        if prior[p][0]=='<' or prior[p][0]=='<=' or prior[p][0]=='>' or prior[p][0]=='>=':
            resi = '%s if 0'%resi+prior[p][0]+'%s else 0'%resi
        #print(resi)
        res.append(eval(resi))
    return res

def residualsOI(oi, param, timeit=False):
    """
    assumes dict OI has a key "fit" which list observable to fit:

    OI['fit']['obs'] is a list containing '|V|', 'PHI', 'DPHI', 'V2', 'T3AMP', 'T3PHI'
    OI['fit'] can have key "wl ranges" to limit fit to [(wlmin1, wlmax1), (), ...]

    """
    tt = time.time()
    res = np.array([])

    if type(oi)==list:
        for i,o in enumerate(oi):
            res = np.append(res, residualsOI(o, param, timeit=timeit))
        return res

    if 'fit' in oi:
        fit = oi['fit']
    else:
        fit = {'obs':[]}

    t0 = time.time()
    if 'DPHI' in fit['obs']:
        oi = computeDiffPhiOI(oi, param)
        if timeit:
            print('residualsOI > dPHI %.3fms'%(1000*(time.time()-t0)))
            t0 = time.time()
    if 'NFLUX' in fit['obs']:
        oi = computeNormFluxOI(oi, param)
        if timeit:
            print('residualsOI > normFlux %.3fms'%(1000*(time.time()-t0)))

    if 'ignore telescope' in fit:
        ignoreTelescope = fit['ignore telescope']
    else:
        ignoreTelescope = 'no way they would call a telescope that!'

    if 'ignore baseline' in fit:
        ignoreBaseline = fit['ignore baseline']
    else:
        ignoreBaseline = '**'

    # -- compute model
    t0 = time.time()
    m = VmodelOI(oi, param, timeit=timeit)
    if timeit:
        print('residualsOI > VmodelOI: %.3fms'%(1000*(time.time()-t0)))

    ext = {'|V|':'OI_VIS',
            'PHI':'OI_VIS',
            'DPHI':'DPHI',
            'V2':'OI_VIS2',
            'T3AMP':'OI_T3',
            'T3PHI':'OI_T3',
            'NFLUX':'NFLUX',}
    w = np.ones(oi['WL'].shape)

    if 'wl ranges' in fit:
        w = np.zeros(oi['WL'].shape)
        for WR in oi['fit']['wl ranges']:
            w += (oi['WL']>=WR[0])*(oi['WL']<=WR[1])
    w = np.bool_(w)
    t0 = time.time()
    for f in fit['obs']:
        # -- for each observable:
        if f in ext.keys():
            if 'PHI' in f:
                rf = lambda x: ((x + 180)%360 - 180)
            else:
                rf = lambda x: x
            # -- for each telescope / baseline / triangle
            for k in oi[ext[f]].keys():
                test = testTelescopes(k, ignoreTelescope) or testBaselines(k, ignoreBaseline)
                if not test:
                    mask = w[None,:]*~oi[ext[f]][k]['FLAG']
                    err = oi[ext[f]][k]['E'+f].copy()
                    if 'max error' in oi['fit'] and f in oi['fit']['max error']:
                        # -- ignore data with large error bars
                        mask *= (err<oi['fit']['max error'][f])
                    if 'min error' in fit and f in oi['fit']['min error']:
                        # -- force error to a minimum value
                        err = np.maximum(oi['fit']['min error'][f], err)
                    if 'mult error' in fit and f in oi['fit']['mult error']:
                        # -- force error to a minimum value
                        err *= oi['fit']['mult error'][f]

                    tmp = rf(oi[ext[f]][k][f][mask] -
                             m[ext[f]][k][f][mask])/err[mask]
                    res = np.append(res, tmp.flatten())
                else:
                    pass
                    #print('ignoring', ext[f], k)

        else:
            print('WARNING: unknown observable:', f)
    if timeit:
        print('residualsOI > "res": %.3fms'%(1000*(time.time()-t0)))
        print('residualsOI > total: %.3fms'%(1000*(time.time()-tt)))
        print('-'*30)
    res = np.append(res, m['MODEL']['negativity']*len(res))
    if 'fit' in oi and 'prior' in oi['fit']:
        res = np.append(res, computePrior(param, oi['fit']['prior']))
    return res

def fitOI(oi, firstGuess, fitOnly=None, doNotFit=None, verbose=2,
          maxfev=2000, ftol=1e-4, follow=None, prior=None,
          randomise=False, i=-1):
    """
    oi: a dict of list of dicts returned by oifits.loadOI

    firstGuess: a dict of parameters describring the model. can be numerical
        values or a formula in a string self-referencing to parameters. e.g.
        {'a':1, 'b':2, 'c':'3*a+b/2'}
        Parsing is very simple, parameters name should be ambiguous! for example
        using {'s':1, 'r':2, 'c':'sqrt(s*r)'} will fail, because 'c' will be
        interpreted as '1q2t(1*2)'

    fitOnly: list of parameters to fit. default is all of them

    doNotFit: list of parameters not to fit. default is none of them

    verbose: 0, 1 or 2 for various level of verbosity

    maxfev: maximum number of function evaluation (2000)

    ftol: stopping constraint for chi2 change (1e-4)

    follow: list of parameters to print in screen (for verbose>0)

    randomise: for bootstrapping

    prior: dict of priors.
        {'a+b':('=', 1.2, 0.1)} if a and b are parameters. a+b = 1.2 +- 0.1
        {'a':('<', 'sqrt(b)', 0.1)} if a and b are parameters. a<sqrt(b), and
            the penalty is 0 for a==sqrt(b) but rises significantly
            for a>sqrt(b)+0.1
    """
    if not prior is None:
        if type(oi)==dict:
            if 'fit' in oi:
                oi['fit']['prior'] = prior
            else:
                oi['fit'] = {'prior':prior}
        else:
            for o in oi:
                if 'fit' in o:
                    o['fit']['prior'] = prior
                else:
                    o['fit'] = {'prior':prior}

    if randomise is False:
        tmp = oi
    else:
        # -- randomise data, for bootstrapping
        if i>0:
            print('iteration', i)
        tmp = randomiseData(oi, randomise=randomise, verbose=i>0)
    z = residualsOI(tmp, firstGuess)*0.0
    if fitOnly is None and doNotFit is None:
        fitOnly = list(firstGuess.keys())
    fit = dpfit.leastsqFit(residualsOI, tmp, firstGuess, z, verbose=verbose,
                           maxfev=maxfev, ftol=ftol, fitOnly=fitOnly,
                           doNotFit=doNotFit, follow=follow)
    fit['prior'] = prior
    return fit

def randomiseData(oi, randomise='telescope or baseline', P=None, verbose=False):
    # -- make new sample of data
    if P is None:
        # -- evaluate reduction in amount of data
        Nall, Nt, Nb = 0, 0, 0
        for o in oi:
            for x in o['fit']['obs']:
                if x in 'NFLUX':
                    Nall += len(o['telescopes'])
                    Nt += len(o['telescopes']) - 1
                    Nb += len(o['telescopes'])
                if x in ['|V|', 'DPHI', 'V2']:
                    Nall += len(o['baselines'])
                    Nt += (len(o['telescopes'])-1)*(len(o['telescopes'])-2)//2
                    Nb += len(o['baselines']) - 1
                if x in ['T3PHI']:
                    Nall += len(o['OI_T3'])
                    Nt += (len(o['telescopes'])-2)*(len(o['telescopes'])-3)//2
                    Nb += len(o['OI_T3']) - (len(o['telescopes'])-2)

        if 'telescope' in str(randomise).lower():
            P = int(round(len(oi) * Nall/Nt))
        elif 'baseline' in str(randomise).lower():
            P = int(round(len(oi) * Nall/Nb))
        elif str(randomise).lower()=='telescope or baseline':
            P = int(round(len(oi) * 2*Nall/(Nb+Nt)))
        else:
            P = len(oi)
    tmp = []
    for k in range(P):
        i = np.random.randint(len(oi))
        tmp.append(copy.deepcopy(oi[i]))
        # -- ignore part of the data
        if 'telescope' in str(randomise).lower():
            j = np.random.randint(len(tmp[-1]['telescopes']))
            tmp[-1]['fit']['ignore telescope'] = tmp[-1]['telescopes'][j]
            tmp[-1]['fit']['ignore baseline'] = 'not a baseline name'
            if verbose:
                print('   ', i, 'ignore telescope', tmp[-1]['telescopes'][j])
        elif 'baseline' in str(randomise).lower():
            j = np.random.randint(len(tmp[-1]['baselines']))
            tmp[-1]['fit']['ignore telescope'] = 'not a telescope name'
            tmp[-1]['fit']['ignore baseline'] = tmp[-1]['baselines'][j]
            if verbose:
                print('   ', i, 'ignore baseline', tmp[-1]['baselines'][j])
        elif str(randomise).lower()=='telescope or baseline':
            if np.random.rand()<0.5:
                j = np.random.randint(len(tmp[-1]['telescopes']))
                tmp[-1]['fit']['ignore telescope'] = tmp[-1]['telescopes'][j]
                tmp[-1]['fit']['ignore baseline'] = 'not a baseline name'
                if verbose:
                    print('   ', i, 'ignore telescope', tmp[-1]['telescopes'][j])
            else:
                j = np.random.randint(len(tmp[-1]['baselines']))
                tmp[-1]['fit']['ignore telescope'] = 'not a telescope name'
                tmp[-1]['fit']['ignore baseline'] = tmp[-1]['baselines'][j]
                if verbose:
                    print('   ', i, 'ignore baseline', tmp[-1]['baselines'][j])
    return tmp

def bootstrapFitOI(oi, fit, N=None, fitOnly=None, doNotFit=None, maxfev=2000,
                   ftol=1e-5, randomise='telescope or baseline', sigmaClipping=4.5,
                   multi=True, prior=None):
    """
    """
    if N is None:
        N = 2*sum([len(o['telescopes'])**2 for o in oi])

    fitOnly = fit['fitOnly']
    doNotFit = fit['doNotFit']
    maxfev = fit['maxfev']
    ftol = fit['ftol']/10
    prior = fit['prior']
    firstGuess = fit['best']

    kwargs = {'maxfev':maxfev, 'ftol':ftol, 'verbose':False,
              'randomise':randomise, 'fitOnly':fitOnly, 'doNotFit':doNotFit,
               'prior':prior}

    res = []
    if multi:
        if type(multi)!=int:
            np = multiprocessing.cpu_count()
        else:
            np = multi
        print('pooling', np, 'processes to run', N, 'fits...')

        t = time.time()
        pool = multiprocessing.Pool(np)
        for i in range(np):
            kwargs['i'] = i
            res.append(pool.apply_async(fitOI, (oi, firstGuess, ), kwargs))
        pool.close()
        pool.join()
        res = [r.get(timeout=1) for r in res]
        t = time.time()-t
        print('one fit takes ~%.2fs'%(t/np))

        t = time.time()
        pool = multiprocessing.Pool(np)
        for i in range(N-1):
            kwargs['i'] = i
            res.append(pool.apply_async(fitOI, (oi,firstGuess, ), kwargs))
        pool.close()
        pool.join()
        res = res[:np]+[r.get(timeout=1) for r in res[np:]]
    else:
        np = 1
        t = time.time()
        res.append(fitOI(oi,firstGuess, **kwargs))
        t = time.time()-t
        print('one fit takes ~%.2fs'%t)
        t = time.time()
        for i in range(N-1):
            kwargs['i'] = i
            if i%10==0:
                print('%s | bootstrap fit %d/%d'%(time.asctime(), i, N-1))
            res.append(fitOI(oi, firstGuess, **kwargs))
    print('it took %.1fs, %.2fs per fit on average'%(time.time()-t,
                                                    (time.time()-t)/(N-np)))
    try:
        res = analyseBootstrap(res, sigmaClipping=sigmaClipping)
        res['fit to all data'] = fit
        return res
    except:
        print('analysing bootstrap failed! returning all fits')
        return res

def analyseBootstrap(Boot, sigmaClipping=4.5, verbose=2):
    """
    Boot: a list of fits (list of dict from dpfit.leastsqFit)
    """
    # -- allow to re-analyse bootstrapping results:
    if type(Boot) == dict and 'all fits' in Boot.keys():
        Boot = Boot['all fits']

    res = {'best':{}, 'uncer':{}, 'fitOnly':Boot[0]['fitOnly'],
           'all best':{}, 'all best ignored':{}, 'sigmaClipping':sigmaClipping,
           'all fits':Boot}
    mask = np.ones(len(Boot), dtype=bool)

    # -- sigma clipping and global mask
    if not sigmaClipping is None:
        for k in res['fitOnly']:
            tmp = np.ones(len(Boot), dtype=bool)
            for j in range(3):
                x = np.array([b['best'][k] for b in Boot])
                res['best'][k] = np.median(x[tmp])
                res['uncer'][k] = np.percentile(x[tmp], 84) - np.percentile(x[tmp], 16)
                tmp = np.abs(x-res['best'][k])<=sigmaClipping*res['uncer'][k]
            mask *= tmp

    for k in res['fitOnly']:
        for j in range(3):
            x = np.array([b['best'][k] for b in Boot])
            res['best'][k] = np.mean(x[mask])
            res['uncer'][k] = np.std(x[mask])
        res['all best'][k] = x[mask]
        res['all best ignored'][k] = x[~mask]
    for k in Boot[0]['best'].keys():
        if not k in res['best'].keys():
            res['best'][k] = Boot[0]['best'][k]
            res['uncer'][k] = 0.0

    res['mask'] = mask

    M = [[b['best'][k] for i,b in enumerate(Boot) if mask[i]] for k in res['fitOnly']]

    if len(res['fitOnly'])>1:
        res['cov'] = np.cov(M)
        cor = np.sqrt(np.diag(res['cov']))
        cor = cor[:,None]*cor[None,:]
    else:
        res['cov'] = np.array([[np.std(M)**2]])
        cor = np.array([[np.sqrt(res['cov'])]])

    res['cor'] = res['cov']/cor
    res['covd'] = {ki:{kj:res['cov'][i,j] for j,kj in enumerate(res['fitOnly'])}
                   for i,ki in enumerate(res['fitOnly'])}
    res['cord'] = {ki:{kj:res['cor'][i,j] for j,kj in enumerate(res['fitOnly'])}
                   for i,ki in enumerate(res['fitOnly'])}
    if verbose:
        ns = max([len(k) for k in res['best'].keys()])
        print('{', end='')
        for k in sorted(res['best'].keys()):
            if res['uncer'][k]>0:
                n = int(np.ceil(-np.log10(res['uncer'][k])+1))
                fmt = '%.'+'%d'%n+'f'
                print("'"+k+"'%s:"%(' '*(ns-len(k))), fmt%res['best'][k], end=', ')
                print('# +/-', fmt%res['uncer'][k])
            else:
                print("'"+k+"'%s:"%(' '*(ns-len(k))), res['best'][k])
        print('}')
    if verbose>1 and np.size(res['cov'])>1:
        dpfit.dispCor(res)
    return res

def sigmaClippingOI(oi, sigma=4, n=5, param=None):
    if type(oi)==list:
        return [sigmaClippingOI(o, sigma=sigma, n=n, param=param) for o in oi]

    w = oi['WL']>0
    # -- user-defined wavelength ranges
    if 'fit' in oi and 'wl ranges' in oi['fit']:
        w = np.zeros(oi['WL'].shape)
        for WR in oi['fit']['wl ranges']:
            w += (oi['WL']>=WR[0])*(oi['WL']<=WR[1])
    oi['WL mask'] = np.bool_(w)

    # -- exclude where lines are in the models
    if not param is None:
        for k in param.keys():
            if 'line_' in k and 'wl0' in k:
                if k.replace('wl0', 'gaussian') in param.keys():
                    dwl = 1.2*param[k.replace('wl0', 'gaussian')]/1000.
                if k.replace('wl0', 'lorentzian') in param.keys():
                    dwl = 6*param[k.replace('wl0', 'lorentzian')]/1000.
                w *= (np.abs(oi['WL']-param[k])>=dwl)
    if np.sum(w)==0:
        #print('WARNING: no continuum! using all wavelengths')
        w = oi['WL']>0

    oi['WL cont'] = np.bool_(w)
    w = np.bool_(w)

    # -- do sigma clipping in the continuum
    w = oi['WL mask']*oi['WL cont']
    for k in oi['OI_FLUX'].keys():
        for i in range(len(oi['OI_FLUX'][k]['MJD'])):
            oi['OI_FLUX'][k]['FLUX'][i,w] = _sigmaclip(
                        oi['OI_FLUX'][k]['FLUX'][i,w], s=sigma)
    for k in oi['OI_VIS'].keys():
        for i in range(len(oi['OI_VIS'][k]['MJD'])):
            oi['OI_VIS'][k]['|V|'][i,w] = _sigmaclip(
                        oi['OI_VIS'][k]['|V|'][i,w], s=sigma)
            oi['OI_VIS'][k]['PHI'][i,w] = _sigmaclip(
                        oi['OI_VIS'][k]['PHI'][i,w], s=sigma)
            oi['OI_VIS2'][k]['V2'][i,w] = _sigmaclip(
                        oi['OI_VIS2'][k]['V2'][i,w], s=sigma)
    for k in oi['OI_T3'].keys():
        for i in range(len(oi['OI_T3'][k]['MJD'])):
            oi['OI_T3'][k]['T3PHI'][i,w] = _sigmaclip(
                        oi['OI_T3'][k]['T3PHI'][i,w], s=sigma)
            oi['OI_T3'][k]['T3AMP'][i,w] = _sigmaclip(
                        oi['OI_T3'][k]['T3AMP'][i,w], s=sigma)
    return oi

def _sigmaclip(x, s=4.0, n=3, maxiter=5):
    N, res = len(x), np.copy(x)
    nc = True
    iter = 0
    while nc and iter<maxiter:
        nc = 0
        c = np.polyfit(np.arange(N), res, 1)
        c = [0]
        std = (np.percentile(res-np.polyval(c, np.arange(N)), 84) -
               np.percentile(res-np.polyval(c, np.arange(N)), 16))/2.
        med = np.median(res-np.polyval(c, np.arange(N)))
        for i in range(N):
            #med = np.median(res[max(0,i-n):min(N-1, i+n)])
            if abs(med-res[i])>s*std:
                res[i] = med + np.polyval(c, i)
                nc += 1
        iter += 1
    return np.array(res)

def showUV(oi, fig=0, polar=False):
    plt.close(fig)
    plt.figure(fig)
    ax = plt.subplot(111, aspect='equal', polar=polar)
    if not type(oi)==list:
        oi = [oi]
    for o in oi:
        key = 'OI_VIS'
        if not key in o.keys():
            key = 'OI_VIS2'
        for k in o[key].keys():
            R =  o[key][k]['u/wl']**2 + o[key][k]['v/wl']**2
            PA = np.arctan2(o[key][k]['v/wl'], o[key][k]['u/wl'])
            if polar:
                plt.plot(PA, R, '.k')
                plt.plot(PA+np.pi, R, '.k')
            else:
                plt.plot(o[key][k]['u/wl'], o[key][k]['v/wl'], '.k', alpha=0.5)
                plt.plot(-o[key][k]['u/wl'], -o[key][k]['v/wl'], '.k', alpha=0.5)
    #plt.xlabel('u/$\lambda$ (m/$\mu m$)', y=-0.1)
    #plt.ylabel('v/$\lambda$ (m/$\mu m$)', x=-0.5)
    if polar:
        ax.spines['polar'].set_visible(False)
        X = [0, np.pi/2, np.pi, 3*np.pi/2]
        ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
        ax.set_xticklabels(['270', '0', '90', '180'])
        #ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)

    # try:
    #     plt.tight_layout()
    # except:
    #     pass
    return

allInOneAxes = None
def showOI(oi, param=None, fig=1, obs=None, showIm=False, fov=None, pix=None,
           imPow=1., imWl0=None, cmap='bone',  dx=0.0, dy=0.0,
           showChi2=True, wlMin=None, wlMax=None, allInOne=False, imMax=None,
           figWidth=9, logB=False, logV=False, color=(1.0,0.2,0.1),
           checkImVis=False, ):
    """
    oi: result from oifits.loadOI
    param: dict of parameters for model (optional)
    obs: observable to show (default oi['fit']['obs'] if found, else all available)
        list of ('V2', '|V|', 'PHI', 'T3PHI', 'T3AMP', 'DPHI', 'FLUX')
    fig: figure number (default 1)
    figWidth: width of the figure (default 9)
    allInOne: plot all data files in a single plot, as function of baseline
        if not, show individual files in plots as function of wavelength (default)
        logB: baseline in log scale (default False)
        logV: V2 and |V| in log scale (default False)

    showIm: show image (False)
        fov: field of view (in mas)
        pix: pixel size (in mas)
        imPow: show image**imPow (default 1.0)
        imWl0: wavelength(s) at which images are shown
        imMax: max value for image
        cmap: color map ('bone')
        dx: center of FoV (in mas, default:0.0)
        dy: center of FoV (in mas, default:0.0)

    """
    global allInOneAxes
    if type(oi)==list:
        if allInOne:
            allInOneAxes = {}
        for i,o in enumerate(oi):
            if allInOne:
                f = fig
            else:
                f = fig+i
            showOI(o, param=param, fig=f, obs=obs, fov=fov, pix=pix, dx=dx, dy=dy,
                   checkImVis=checkImVis, showIm=showIm and i==(len(oi)-1),
                   imWl0=imWl0, imPow=imPow, cmap=cmap, figWidth=figWidth,
                   wlMin=wlMin, wlMax=wlMax, allInOne=allInOne, imMax=imMax,
                   logB=logB, logV=logV, color=color)
        allInOneAxes = None
        return

    if param is None:
        showChi2=False
    else:
        if type(param)==dict and 'best' in param.keys() and 'fitOnly' in param.keys():
            param = param['best']
    if allInOneAxes is None or allInOneAxes=={}:
        plt.close(fig)
    plt.figure(fig, figsize=(figWidth, 5))

    # -- user-defined wavelength range
    fit = {'wl ranges':[(min(oi['WL']), max(oi['WL']))]}
    if not 'fit' in oi:
        oi['fit'] = fit
    elif not 'wl ranges' in oi['fit']:
        oi['fit'].update(fit)

    if 'ignore telescope' in oi['fit']:
        ignoreTelescope = oi['fit']['ignore telescope']
    else:
        ignoreTelescope = 'no way they would call a telescope that!'

    if 'ignore baseline' in oi['fit']:
        ignoreBaseline = oi['fit']['ignore baseline']
    else:
        ignoreBaseline = '**'

    w = np.zeros(oi['WL'].shape)
    for WR in oi['fit']['wl ranges']:
        w += (oi['WL']>=WR[0])*(oi['WL']<=WR[1])
    oi['WL mask'] = np.bool_(w)

    if not 'obs' in oi['fit'] and obs is None:
            obs = []
            if 'OI_T3' in oi:
                obs.append('T3PHI')
            if 'OI_VIS2' in oi:
                obs.append('V2')
            if 'OI_VIS' in oi:
                obs.extend(['|V|', 'DPHI'])
            if 'OI_FLUX' in oi:
                obs.extend('NFLUX')
    elif not obs is None:
        pass
    else:
        obs = oi['fit']['obs']

    # -- force recomputing differential quantities
    if 'WL cont' in oi.keys():
        oi.pop('WL cont')
    if 'DPHI' in obs:
        oi = computeDiffPhiOI(oi, param)
    if 'NFLUX' in obs:
        oi = computeNormFluxOI(oi, param)

    if not param is None:
        #print('compute model V (analytical)')
        m = VmodelOI(oi, param, fov=fov, pix=pix, dx=dx, dy=dy)
        if not fov is None and checkImVis:
            #print('compute V from Image, fov=', fov)
            m = VfromImageOI(m)
    else:
        m = None

    c = 1 # column
    ax0 = None
    data = {
            'T3PHI':{'ext':'OI_T3', 'var':'T3PHI', 'unit':'deg', 'X':'Bmax/wl'},
            'T3AMP':{'ext':'OI_T3', 'var':'T3AMP', 'X':'Bmax/wl'},
            'DPHI':{'ext':'DPHI', 'var':'DPHI', 'unit':'deg', 'X':'B/wl'},
            '|V|':{'ext':'OI_VIS', 'var':'|V|', 'X':'B/wl'},
            'PHI':{'ext':'OI_VIS', 'var':'PHI', 'X':'B/wl'},
            'V2':{'ext':'OI_VIS2', 'var':'V2', 'X':'B/wl'},
            'FLUX':{'ext':'OI_FLUX', 'var':'FLUX'},
            'NFLUX':{'ext':'NFLUX', 'var':'NFLUX', 'unit':'normalized'},
            }
    imdata = {'T3PHI':{'ext':'IM_T3', 'var':'T3PHI', 'unit':'deg', 'X':'Bmax/wl'},
              'T3AMP':{'ext':'IM_T3', 'var':'T3AMP', 'X':'Bmax/wl'},
              'DPHI':{'ext':'IM_VIS', 'var':'DPHI', 'unit':'deg', 'X':'B/wl'},
              '|V|':{'ext':'IM_VIS', 'var':'|V|', 'X':'B/wl'},
              'PHI':{'ext':'IM_VIS', 'var':'PHI', 'X':'B/wl'},
              'V2':{'ext':'IM_VIS', 'var':'V2', 'X':'B/wl'},
              'FLUX':{'ext':'IM_FLUX', 'var':'FLUX'},
              'NFLUX':{'ext':'IM_FLUX', 'var':'NFLUX', 'unit':'normalized'},
             }
    # -- plot in a certain order
    obs = list(filter(lambda x: x in obs, ['NFLUX', 'T3PHI', 'T3AMP',
                                            'DPHI', 'PHI', '|V|', 'V2']))
    for c,l in enumerate(obs):
        # -- for each observable
        N = len(oi[data[l]['ext']].keys())

        # -- for each telescope / baseline / triplets
        for i,k in enumerate(sorted(oi[data[l]['ext']].keys())):
            X = lambda r, j: r['WL']
            Xlabel = r'wavelength ($\mu$m)'
            Xscale = 'linear'
            if allInOne:
                if 'X' in data[l]:
                    X = lambda r, j: r[data[l]['ext']][k][data[l]['X']][j,:]
                    Xlabel = data[l]['X']
                    if logB:
                        Xscale = 'log'
                if not l in allInOneAxes.keys():
                    allInOneAxes[l] = plt.subplot(1, len(obs), c+1)
                ax = allInOneAxes[l]
            else:
                if ax0 is None:
                    ax0 = plt.subplot(N, len(obs), len(obs)*i+c+1)
                    ax = ax0
                else:
                    ax = plt.subplot(N, len(obs), len(obs)*i+c+1, sharex=ax0)

            chi2 = 0.0
            resi = np.array([])
            ymin = 1e6
            ymax = -1e6
            # -- for each MJD:
            for j in range(oi[data[l]['ext']][k][data[l]['var']].shape[0]):
                mask = ~oi[data[l]['ext']][k]['FLAG'][j,:]*oi['WL mask']
                if wlMin is None:
                    wlMin = min(oi['WL'][mask])
                if wlMax is None:
                    wlMax = max(oi['WL'][mask])
                y = oi[data[l]['ext']][k][data[l]['var']][j,:]
                if 'PHI' in l:# and np.ptp(y[mask])>300:
                    y[mask] = np.mod(y[mask]+180, 360)-180
                    #y[mask] = np.unwrap(y[mask]*np.pi/180)*180/np.pi
                    #y[mask] = _smoothUnwrap(y[mask], n=5)

                err = oi[data[l]['ext']][k]['E'+data[l]['var']][j,:].copy()
                if 'max error' in oi['fit'] and \
                        data[l]['var'] in oi['fit']['max error']:
                    # -- ignore data with large error bars
                    ign = mask*(err>=oi['fit']['max error'][data[l]['var']])
                    mask *= err<oi['fit']['max error'][data[l]['var']]
                else:
                    ign = None

                if 'min error' in oi['fit'] and \
                            data[l]['var'] in oi['fit']['min error']:
                    err[mask] = np.maximum(oi['fit']['min error'][data[l]['var']], err[mask])
                if 'mult error' in oi['fit'] and \
                            data[l]['var'] in oi['fit']['mult error']:
                    err *= oi['fit']['mult error'][data[l]['var']]

                # -- show data
                test = testTelescopes(k, ignoreTelescope) or testBaselines(k, ignoreBaseline)
                if allInOne:
                    ax.plot(X(oi, j)[mask], y[mask],'.',
                            color='k' if not test else '0.5',
                            alpha=0.5, label=k if j==0 else '')
                    ax.errorbar(X(oi, j)[mask], y[mask], yerr=err[mask],
                                color='k', alpha=0.2, linestyle='None')
                else:
                    ax.step(X(oi, j)[mask], y[mask],
                            '-', color='k' if not test else '0.5',
                            alpha=0.5, label=k if j==0 else '', where='mid')
                    ax.fill_between(X(oi, j)[mask], y[mask]+err[mask], y[mask]-err[mask],
                                    step='mid', color='k',
                                    alpha=0.1 if not test else 0.05)
                if not ign is None:
                    # -- show ignored data
                    plt.plot(X(oi,j)[ign], y[ign], 'xy', alpha=0.5)

                maskp = mask*(oi['WL']>=wlMin)*(oi['WL']<=wlMax)
                ymin = min(ymin, np.percentile((y-err)[maskp], 1))
                ymax = max(ymax, np.percentile((y+err)[maskp], 99))

                # -- show model (analytical)
                if not m is None:
                    ym = m[data[l]['ext']][k][data[l]['var']][j,:]
                    ym[mask] = np.unwrap(ym[mask]*np.pi/180)*180/np.pi
                    # -- computed chi2 *in the displayed window*
                    maskc2 = mask*(oi['WL']>=wlMin)*(oi['WL']<=wlMax)
                    chi2 += np.mean((y-ym)[maskc2]**2/err[maskc2]**2
                                /len(oi[data[l]['ext']][k]['MJD']))
                    resi = np.append(resi, (y-ym)[maskc2].flatten())
                    if allInOne:
                        ax.plot(X(m,j)[maskc2], ym[maskc2],
                                '.', alpha=0.4 if not test else 0.2,
                                color=color)
                    else:
                        ax.step(X(m,j)[maskc2], ym[maskc2],
                                '-', alpha=0.4 if not test else 0.2,
                                where='mid', color=color)
                    ymin = min(ymin, np.min(ym[maskp]))
                    ymax = max(ymax, np.max(ym[maskp]))

                # -- show model (numerical from image)
                if checkImVis:
                    ax.step(X(m, j)[mask],
                             m[imdata[l]['ext']][k][imdata[l]['var']][j,mask],
                            '--b', alpha=0.4, linewidth=3, where='mid')

                # -- show continuum for differetial PHI and normalized FLUX
                if (l=='DPHI' or l=='NFLUX') and 'WL cont' in oi:
                    maskc = ~oi[data[l]['ext']][k]['FLAG'][j,:]*\
                                    oi['WL cont']*oi['WL mask']
                    _m = np.mean(oi[data[l]['ext']][k][data[l]['var']][j,maskc])
                    cont = np.ones(len(oi['WL']))*_m
                    cont[~maskc] = np.nan
                    ax.plot(X(oi, j), cont, ':', color='c', linewidth=3)

                # -- show phase based on OPL
                if l=='PHI':
                    dOPL = oi['OPL'][k[2:]] - oi['OPL'][k[:2]]
                    print(k, dOPL)
                    wl0 = 2.2
                    cn = np.polyfit(oi['WL']-wl0, oi['n_lab'], 8)
                    cn[-2:] = 0.0
                    ax.plot(oi['WL'], -360*dOPL*np.polyval(cn, oi['WL']-wl0)/(oi['WL']*1e-6),
                            ':c', linewidth=2)

            # -- end loop on MJDs
            if not allInOne:
                maskp = mask*(oi['WL']>=wlMin)*(oi['WL']<=wlMax)
                yamp = ymax-ymin
                ax.set_ylim(ymin - 0.2*yamp, ymax + 0.2*yamp)
                ax.set_xlim(wlMin, wlMax)

                if showChi2:
                    if len(resi)<24:
                        rms = np.std(resi)
                    else:
                        rms = 0.5*(np.percentile(resi, 84)-np.percentile(resi, 16))
                    fmt = '$\chi^2$=%.2f '
                    n = int(np.ceil(-np.log10(rms)+1))
                    fmt += 'rms=%.'+str(n)+'f'

                    ax.text(0.02, 0.02, fmt%(chi2, rms),
                            color=color, fontsize=6,
                            transform=ax.transAxes, ha='left', va='bottom')

                ax.text(0.02, 0.98, k, transform=ax.transAxes,
                         ha='left', va='top', fontsize=6, color='k')
                ax.tick_params(axis='x', labelsize=8)
                ax.tick_params(axis='y', labelsize=8)
                ax.grid(color=(0.2, 0.4, 0.7), alpha=0.2)
            else:
                if l in ['V2', '|V|']:
                    if logV:
                        ax.set_ylim(1e-4,1)
                        ax.set_yscale('log')
                    else:
                        ax.set_ylim(0,1)

            if i==N-1:
                ax.set_xlabel(Xlabel)
                if Xscale=='log':
                    ax.set_xscale('log')

            if i==0:
                title = l
                if 'unit' in data[l]:
                    title += ' (%s)'%data[l]['unit']
                ax.set_title(title)

    plt.subplots_adjust(hspace=0, wspace=0.2, left=0.06, right=0.99)
    if not allInOne:
        plt.suptitle(oi['filename'], fontsize=10)

    if showIm and not param is None:
        showModel(oi, param, m=m, fig=fig+1, imPow=imPow, cmap=cmap, figWidth=figWidth,
                fov=fov, pix=pix, dx=dx, dy=dy, imWl0=imWl0, imMax=imMax)
    return

def showModel(oi, param, m=None, fig=1, imPow=1.0, fov=None, pix=None,
              dx=None, dy=None, imWl0=None, cmap='bone', imMax=None, figWidth=9):
    param = computeLambdaParams(param)
    if m is None:
        m = VmodelOI(oi, param, fov=fov, pix=pix, dx=dx, dy=dy)

    # -- show synthetic images ---------------------------------------------
    plt.close(fig)
    plt.figure(fig, figsize=(figWidth,4))
    if imWl0 is None:
        imWl0 = [np.min(oi['WL'][oi['WL mask']]),
                 np.max(oi['WL'][oi['WL mask']]),
                ]
    normIm = None
    if imMax is None:
        imMax = 1.0

    # -- components
    comps = set([k.split(',')[0] for k in param.keys() if ',' in k])
    # -- peak wavelengths to show components with color code
    wlpeak = {}
    allpeaks = []
    for c in comps:
        #wlmin = m['WL'][oi['WL mask']][np.argmin(m['MODEL'][c+',nflux'][oi['WL mask']])]
        #wlmax = m['WL'][oi['WL mask']][np.argmax(m['MODEL'][c+',nflux'][oi['WL mask']])]
        lines = list(filter(lambda x: x.startswith(c+',line_') and x.endswith('wl0'), param))
        if len(lines):
            # -- weighted wavelength:
            wlpeak[c] = np.sum([param[k]*param[k.replace('_wl0', '_f')] for k in lines])/\
                        np.sum([param[k.replace('_wl0', '_f')] for k in lines])
            allpeaks.append(wlpeak[c])
        else:
            wlpeak[c] = None
    symbols = {}
    a, b, c = 0.9, 0.6, 0.1
    colors = [(a,b,c), (b,c,a), (c,a,b),
              (a,c,b), (c,b,a), (b,a,c),
              (a,c,c), (c,a,c), (c,c,a),
              (b,b,c), (b,c,b), (c,b,b),
              (b,b,b)]
    _ic = 0
    for c in comps:
        if wlpeak[c] is None:
            symbols[c] = {'m':'x', 'c':colors[_ic%len(colors)]}
            _ic+=1
        else:
            if len(allpeaks)==1:
                # -- only one componenet with line
                symbols[c] = {'m':'+', 'c':'orange'}
            else:
                symbols[c] = {'m':'+',
                              'c':matplotlib.cm.nipy_spectral(0.1+0.8*(wlpeak[c]-min(allpeaks))/np.ptp(allpeaks))
                             }

    for i,wl in enumerate(imWl0):
        # -- for each wavelength for which we need to show the image
        axi = plt.subplot(1, len(imWl0)+1, i+1, aspect='equal')
        _j = np.argmin(np.abs(oi['WL'][oi['WL mask']]-wl))
        _wl = oi['WL'][oi['WL mask']][_j]
        j = np.arange(len(oi['WL']))[oi['WL mask']][_j]
        if not imPow == 1:
            plt.title('Image$^{%.2f}$ $\lambda$=%.3f$\mu$m'%(imPow, _wl),
                    fontsize=10)
        else:
            plt.title('Image $\lambda$=%.3f$\mu$m'%(_wl), fontsize=10)

        if np.min(m['MODEL']['cube'][j,:,:])<0:
            print('WARNING: negative image! wl=%.4fum'%_wl)
            print(' "negativity" in model:', m['MODEL']['negativity'])
        im = np.maximum(m['MODEL']['cube'][j,:,:], 0)**imPow
        if normIm is None:
            normIm = np.max(im)
        im /= normIm
        pc = plt.pcolormesh(m['MODEL']['X'], m['MODEL']['Y'],
                            im, vmin=0, cmap=cmap, vmax=imMax**imPow)
        cb = plt.colorbar(pc, ax=axi)
        #Xcb = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0])*imMax
        Xcb = np.linspace(0,1,11)*imMax**imPow
        XcbL = ['%.1e'%(xcb**(1./imPow)) for xcb in Xcb]
        XcbL = [xcbl.replace('e+00', '').replace('e-0', 'e-') for xcbl in XcbL]
        cb.set_ticks(Xcb)
        cb.set_ticklabels(XcbL)
        cb.ax.tick_params(labelsize=7)
        axi.invert_xaxis()
        plt.xlabel(r'$\leftarrow$ E (mas)')
        plt.ylabel(r'N $\rightarrow$ (mas)')

        # -- show position of each components
        for c in sorted(comps):
            if c+',x' in param.keys():
                x = param[c+',x']
            else:
                x = 0.0
            if c+',y' in param.keys():
                y = param[c+',y']
            else:
                y = 0.0
            plt.plot(x, y, symbols[c]['m'], color=symbols[c]['c'], label=c)
        if i==0:
            plt.legend(fontsize=5, ncol=2)

    plt.subplot(1, len(imWl0)+1, len(imWl0)+1)
    if 'totalnflux' in m['MODEL']:
        key = 'nflux'
        plt.ylabel('spectra, normalized\nto total continuum', fontsize=8)
    else:
        key = 'flux'
        plt.ylabel('spectra', fontsize=8)

    plt.step(m['WL'][oi['WL mask']],
             m['MODEL']['total'+key][oi['WL mask']],
            '-k', label='total', where='mid')

    if 'WL cont' in oi:
        cont = np.ones(oi['WL'].shape)
        cont[~oi['WL cont']] = np.nan
        plt.step(m['WL'][oi['WL mask']],
                 (m['MODEL']['total'+key]*cont)[oi['WL mask']],
                'c', label='continuum', where='mid', alpha=0.7,
                linewidth=3, linestyle='dotted')
    # -- show spectra of each components
    for k in sorted(m['MODEL'].keys()):
        if k.endswith(','+key):
            plt.step(m['WL'][oi['WL mask']],
                      m['MODEL'][k][oi['WL mask']],
                      label=k.split(',')[0], where='mid',
                    color=symbols[k.split(',')[0]]['c'])

    plt.grid(color=(0.2, 0.4, 0.7), alpha=0.2)

    plt.legend(fontsize=5)
    plt.xlabel('wavelength ($\mu$m)')
    plt.ylim(0)
    plt.tight_layout()
    return

def _callbackAxes(ax):
    global _AX, _AY
    i = None
    for k in _AY.keys():
        if ax==_AY[k]:
            i = k
    #print('callback:', i)
    if not i is None:
        print('callback:', i, ax.get_ylim())
        _AX[i].set_xlim(ax.get_ylim())
        #_AX[i].figure.canvas.draw()
    else:
        print('could not find axes')
    return

def showBootstrap(boot, fig=1, figWidth=7,
                 showRejected=False):
    global _AX, _AY
    fig = plt.figure(fig, figsize=(figWidth, figWidth))
    _AX = {}
    fontsize = max(min(4*figWidth/len(boot['fitOnly']), 14), 6)

    color1 = 'orange'
    color2 = (0.2, 0.4, 1)
    for i1, k1 in enumerate(sorted(boot['fitOnly'])):
        _AX[i1] = plt.subplot(len(boot['fitOnly']),
                              len(boot['fitOnly']),
                              1+i1*len(boot['fitOnly'])+i1)
        bins = int(3*np.ptp(boot['all best'][k1])/boot['uncer'][k1])
        bins = min(bins, len(boot['mask'])//5)
        bins = max(bins, 5)
        h = plt.hist(boot['all best'][k1], bins=bins,
                     color='k', histtype='step', alpha=0.9)
        h = plt.hist(boot['all best'][k1], bins=bins,
                     color='k', histtype='stepfilled', alpha=0.05)
        plt.errorbar(boot['best'][k1], 0.5*max(h[0]), xerr=boot['uncer'][k1],
                    color=color2, fmt='d', capsize=fontsize/2, label='bootstrap',
                    markersize=fontsize/2)
        plt.errorbar(boot['fit to all data']['best'][k1],
                    0.4*max(h[0]), markersize=fontsize/2,
                    xerr=boot['fit to all data']['uncer'][k1], color=color1,
                    fmt='s', capsize=fontsize/2, label='fit to all data')
        plt.legend(fontsize=6)
        n = int(np.ceil(-np.log10(boot['uncer'][k1])+1))
        fmt = '%s=\n'+'%.'+'%d'%n+'f'+'$\pm$'+'%.'+'%d'%n+'f'
        plt.title(fmt%(k1, boot['best'][k1], boot['uncer'][k1]),
                    fontsize=fontsize)
        _AX[i1].yaxis.set_visible(False)
        if i1!=(len(boot['fitOnly'])-1):
            _AX[i1].xaxis.set_visible(False)
        else:
            _AX[i1].tick_params(axis='x', labelsize=fontsize*0.8)
            _AX[i1].set_xlabel(k1, fontsize=fontsize)
    _AY = {}
    for i1, k1 in enumerate(sorted(boot['fitOnly'])):
        for i2 in range(i1+1, len(boot['fitOnly'])):
            k2 = sorted(boot['fitOnly'])[i2]
            if i1==0:
                _AY[i2] = plt.subplot(len(boot['fitOnly']),
                            len(boot['fitOnly']),
                            1+i2*len(boot['fitOnly'])+i1,
                            sharex=_AX[i1])
                ax = _AY[i2]
            else:
                ax = plt.subplot(len(boot['fitOnly']),
                            len(boot['fitOnly']),
                            1+i2*len(boot['fitOnly'])+i1,
                            sharex=_AX[i1],
                            sharey=_AY[i2])

            plt.plot(boot['all best'][k1], boot['all best'][k2], '.',
                     alpha=np.sqrt(2/len(boot['mask'])), color='k')
            if showRejected:
                plt.plot(boot['all best ignored'][k1],
                         boot['all best ignored'][k2], 'xr', alpha=0.3)
            plt.plot(boot['best'][k1], boot['best'][k2], '+', color=color2)
            x, y = dpfit.errorEllipse(boot, k1, k2)
            plt.plot(x, y, '-', color=color2, label='c=%.2f'%boot['cord'][k1][k2])

            plt.plot(boot['fit to all data']['best'][k1],
                     boot['fit to all data']['best'][k2], 'x', color='0.5')
            x, y = dpfit.errorEllipse(boot['fit to all data'], k1, k2)
            plt.plot(x, y, '-', color=color1)#, label='c=%.2f'%boot['cord'][k1][k2])

            plt.legend(fontsize=6)
            if i2==(len(boot['fitOnly'])-1):
                plt.xlabel(k1, fontsize=fontsize)
                ax.tick_params(axis='x', labelsize=fontsize*0.8)
            else:
                ax.xaxis.set_visible(False)
            if i1==0:
                plt.ylabel(k2, fontsize=fontsize)
                _AY[i2].callbacks.connect('ylim_changed', _callbackAxes)
                _AY[i2].tick_params(axis='y', labelsize=fontsize*0.8)
            else:
                ax.yaxis.set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(hspace = 0, # 0.65*fig.subplotpars.hspace,
                        wspace = 0, # 0.65*fig.subplotpars.wspace
                        )
    return

def Oi2Obs(oi, obs=[]):
    """
    obs can be 'V' (complex), '|V|', 'PHI', 'DPHI', 'V2', 'T3AMP', 'CP'

    return OBS format:
    (uv, wl, t3formula, obs, data, error)
    uv = ((u1,v1), (u2,v2), ..., (un, vn)) in meters
    wl = wavelength vector in um
    mjd = date vector.
    t3mat = formalae to compute t3 from (u_i, v_i)

    limitation:
    - does not work if OI does not contain OI_VIS or OI_VIS2
    - assume each u,v (and triangle) have same MJD dimension
    """
    if type(oi)==list:
        return [Oi2Obs(o) for o in oi]
    res = []
    # -- u,v
    uv = []
    key = 'OI_VIS'
    if not key in oi:
        key = 'OI_VIS2'
    for k in oi[key].keys():
        uv.append((oi[key][k]['u'], oi[key][k]['v']))
    res.append(uv)

    # -- wavelength
    res.append(oi['WL'])

    # -- T3 formula (if needed)
    formula = None
    if 'T3AMP' in obs or 'T3PHI' in obs:
        formula = []
        for k in oi['OI_T3'].keys():
            pass

def _negativityAzvar(n, phi, amp, N=100):
    x = np.linspace(0, 2*np.pi, N)
    y = np.ones(N)
    for i in range(len(n)):
        y += amp[i]*np.cos(n[i]*(x + 3*np.pi/2 + phi[i]*np.pi/180))
    return sum(y<0)/sum(y>=0)
    #return -min(min(y), 0.)/np.max(y)


def _Vazvar(u, v, I, r, n, phi, amp, stretch=None, V0=None, numerical=False,
            XY=None, nB=30, numVis=False):
    """
    complex visibility of aritrary radial profile with cos AZ variations.

    u,v: baseline/wavelength coordinates in meters/microns (can be 1D or 2D)

    I(r)*(1+amp*cos(n*(PA+phi)))
        -> I, r: 1D arrays in mas
        -> PA: projection angle, computed from u,v
        -> n: an integer giving the number of cos max in PA
        -> phi: is a PA rotation (deg from N to E)

    n, phi, amp can be lists to have a harmonic decomposition

    stretch=[e, PA]: stretch/PA (optional, default is None) with 0<e<=1 and PA
        the rotation (N>E) of the major axis. If the stretch is defined, the
        azimutal variations rotates with it! PA in degrees

    V0 is V(u,v) for I(r). If not given, will be computed numerically (slow).
    It is the Hankel0 transform of I(r), e.g. 2J1(x)/x for UD.

    nB = 50 # number of baselines for Hankel transform

    numerical=True: compute numerically the Visibility. XY (square regular mesh
    grid in mas) can be given, or computed automatically. In that later case, the
    pitch of the image is set to ensure good accuracy in Visibility. Returns:

        ([Vis], Image, X, Y) if XY=None
            Vis: the visibility for U,V
            Image: the 2D image (total flux of 1.0)
            X, Y: the 2D meshgrid of spatial coordinates of the image (in mas)
        ([Vis], Image) is XY=X,Y where X,Y are square regular grids (in mas)
        Vis only returned is numVis is ste to True
    ---

    https://www.researchgate.net/publication/241069465_Two-Dimensional_Fourier_Transforms_in_Polar_Coordinates
    section 3.2.1 give the expression for the Fourier transform of a function
    in the form f(r,theta). It f(r, theta) can be written as g(r)*h(theta), the final
    Fourier Transform TF(rho, psi) can be written as:

    F(rho, psi) = sum_{n=-inf}^{inf} 2pi j^-n e^{j*n*psi} int_0^infty fn(r) Jn(rho*r) r dr

    with:

    fn(r) = 1/2pi g(r) int_0^2pi h(theta) e^{-j*n*theta} dtheta

    In the simple case of h(theta)=cos(n*theta), fn is non-0 only for -n and n

    fn = f-n = 1/2*g(r) and noticing that J-n = -Jn

    F(rho, psi) =  (-j)^n * e^{j*n*psi} int_0^infty[g(r)*J-n(rho*r) r*dr] +
                  - (-j)^n * e^{-j*n*psi} int_0^infty[g(r)*Jn(rho*r) r*dr] +
                 = -2*(-j)^n * cos(n*psi) * int_0^infty[g(r)*Jn(rho*r) r*dr]
                 = 2*(-j)^n * cos(n*psi) * Gn(rho)

    where Gn is the nth order Hankel transform of g.

    see Also: https://en.wikipedia.org/wiki/Hankel_transform#Relation_to_the_Fourier_transform_(general_2D_case)
    """

    c = np.pi/180./3600./1000.*1e6
    if not isinstance(n, list):
        # -- scalar case
        n = [n]
        phi = [phi]
        amp = [amp]
    if not stretch is None:
        rot = stretch[1]*np.pi/180
    else:
        rot = 0.0

    if numerical:
        if u is None or v is None:
            u, v = np.array([1.]), np.array([1])
        B = np.sqrt(u**2+v**2)
        if XY is None:
            # -- to double check
            pix = 1/(B.max()*c*20)
            Nx = int(np.ptp(r)/pix)+1
            Ny = int(np.ptp(r)/pix)+1
            Nmax = 200 # RAM requirement balloons quickly
            if Nx>Nmax:
                print("WARNING: synthetic image size is too large >Nmax=%d"%Nmax)
                return
            x = np.linspace(-np.max(r), np.max(r), Nx)
            X, Y = np.meshgrid(x, x)
            retFull = True
        else:
            X,Y = XY
            pix = np.diff(X).mean()
            Nx = X.shape[0]
            Ny = X.shape[1]
            retFull = False

        if not stretch is None:
            # -- apply stretch in direct space
            Xr = np.cos(-rot)*X + np.sin(-rot)*Y
            Yr =-np.sin(-rot)*X + np.cos(-rot)*Y
            Xr /= stretch[0]
        else:
            Xr, Yr = X, Y

        Im = np.zeros((Nx, Ny))
        #print('azvar image:', X.shape, Y.shape, Im.shape)
        r2 = np.array(r)**2
        # -- 2D intensity profile with anti-aliasing
        # ns = 3
        # for i in range(Ny):
        #     for j in range(Nx):
        #         for dx in np.linspace(-pix/2, pix/2, ns+2)[1:-1]:
        #             for dy in np.linspace(-pix/2, pix/2, ns+2)[1:-1]:
        #                 _r2 = (Xr[j,i]+dx)**2+(Yr[j,i]+dy)**2
        #                 Im[j,i] += 1./ns**2*np.interp(_r2,r2,I,right=0.0,left=0.0)
        Im = np.interp(Xr**2+Yr**2, r2, I, right=0, left=0)
        # -- azimutal variations in image
        PA = np.arctan2(Yr, Xr)

        PAvar = np.ones(PA.shape)
        for k in range(len(n)):
            PAvar += amp[k]*np.cos(n[k]*(PA + 3*np.pi/2 + phi[k]*np.pi/180))
        Im *= PAvar
        # -- normalize image to total flux
        Im /= np.sum(Im)

        # -- numerical Visibility
        if numVis:
            if len(u.shape)==2:
                Vis = Im[:,:,None,None]*np.exp(-2j*np.pi*c*(u[None,None,:,:]*X[:,:,None,None] +
                                                            v[None,None,:,:]*Y[:,:,None,None]))
            elif len(u.shape)==1:
                Vis = Im[:,:,None]*np.exp(-2j*np.pi*c*(u[None,None,:,:]*X[:,:,None] +
                                                       v[None,None,:,:]*Y[:,:,None]))
            Vis = np.trapz(np.trapz(Vis, axis=0), axis=0)/np.trapz(np.trapz(Im, axis=0), axis=0)
        else:
            Vis = None

        if retFull:
            if numVis:
                return Vis, Im, X, Y
            else:
                return Im, X, Y
        else:
            if numVis:
                return Vis, Im
            else:
                return Im

    if not stretch is None:
        # -- apply stretch in u,v space
        _u = np.cos(-rot)*u + np.sin(-rot)*v
        _v =-np.sin(-rot)*u + np.cos(-rot)*v
        _u *= stretch[0]
        _B = np.sqrt(_u**2 + _v**2)
        _PA = np.arctan2(_v,_u)
    else:
        _B = np.sqrt(u**2 + v**2)
        _PA = np.arctan(v,u)

    if _B.size>nB:
        Bm = np.linspace(np.min(_B), np.max(_B), nB)
    else:
        Bm = None

    # -- define Hankel transform of order n
    def Hankel(n):
        if not Bm is None:
            H = np.trapz(I[:,None]*r[:,None]*\
                         scipy.special.jv(n, 2*np.pi*c*Bm[None,:]*r[:,None]),
                         r, axis=0)/np.trapz(I*r, r)
            return np.interp(_B, Bm, H)
        else:
            H = np.trapz(I[:,None]*r[:,None]*\
                         scipy.special.jv(n, 2*np.pi*c*_B.flatten()[None,:]*r[:,None]),
                         r, axis=0)/np.trapz(I*r, r)
            return np.reshape(H, _B.shape)

    # -- visibility without PA variations -> Hankel0
    if V0 is None:
        Vis = Hankel(0)*(1.+0j) # -- force complex
    else:
        Vis = V0*(1.+0j) # -- force complex

    for i in range(len(n)):
        if np.abs(amp[i])>0:
            Vis += amp[i]*(-1j)**n[i]*Hankel(n[i])*\
                    np.cos(n[i]*(_PA+3*np.pi/2+phi[i]*np.pi/180))

    # -- return complex visibility
    return Vis

def testAzVar():
    """
    """
    c = np.pi/180./3600./1000.*1e6

    wl0 = 1.6 # in microns
    diam = 10 # in mas
    bmax = 140 # in meters

    # -- U,V grid
    Nuv = 51
    b = np.linspace(-bmax,bmax,Nuv)
    U, V = np.meshgrid(b, b)

    stretch = None
    # == image parameters =================================================
    # ~FS CMa
    thick = .7 # uniform disk for thick==1, uniform ring for 0<f<1
    n = [1,2,3] # int number of maxima in cos azimutal variation (0==no variations)
    phi = [90,60,30] # PA offset in azimutal var (deg)
    amp = [0.33,0.33,0.33] # amplitude of azimutal variation: 0<=amp<=1
    stretch = [.5, 90] # eccentricity and PA of large

    # n = [1] # int number of maxima in cos azimutal variation (0==no variations)
    # phi = [90] # PA offset in azimutal var (deg)
    # amp = [1.0] # amplitude of azimutal variation: 0<=amp<=1

    # =====================================================================

    # == Semi-Analytical =======================================
    t0 = time.time()
    # -- 1D intensity profiles
    Nr = min(int(30/thick), 200)
    _r = np.linspace(0, diam/2, Nr)

    if False:
        # -- uniform ring ------------------------
        _i = 1.0*(_r<=diam/2.)*(_r>=diam/2.*(1-thick))
        # -- analytical V of radial profile, and semi-analytical azimutal variation
        if not stretch is None:
            rot = stretch[1]*np.pi/180
            _u = np.cos(-rot)*U + np.sin(-rot)*V
            _v =-np.sin(-rot)*U + np.cos(-rot)*V
            _u *= stretch[0]
            x = np.pi*c*np.sqrt(_u**2+_v**2)*diam/wl0
        else:
            x = np.pi*c*np.sqrt(U**2+V**2)*diam/wl0
        VisR = (2*scipy.special.j1(x+1e-6)/(x+1e-6) -
               (1-thick)**2*2*scipy.special.j1(x*(1-thick)+1e-6)/(x*(1-thick)+1e-6))/(1-(1-thick)**2)
        #VisR = None # does not work
        Visp = _Vazvar(U/wl0, V/wl0, _i, _r, n, phi, amp, V0=VisR, stretch=stretch)
    else:
        # -- "doughnut" -------------------------
        r0 = 0.5*(1 + (1-thick))*diam/2
        width = thick*diam/4
        _i = np.maximum(1-(_r-r0)**2/width**2, 0)

        # -- truncated gaussian: r0 as FWHM = 2.355*sigma
        #_i = np.exp(-(_r/r0*2.355)**2)*(_r<=diam/2)*(_r>=(1-thick)*diam/2)

        # -- do not give analytical visibility of radial profile (slower)
        Visp = _Vazvar(U/wl0, V/wl0, _i, _r, n, phi, amp, stretch=stretch)

    tsa = time.time()-t0
    print('semi-analytical: %.4fs'%(time.time()-t0))

    # == Numerical ============================================
    t0 = time.time()
    Vis, I, X, Y = _Vazvar(U/wl0, V/wl0, _i, _r, n, phi, amp, numerical=1,
                           stretch=stretch, numVis=1)
    Nx = X.shape[0]
    tn = time.time()-t0
    print('numerical:       %.4fs'%(tn))
    print('Image min/max = %f / %f'%(np.min(I), np.max(I)))
    # == Show result =======================================
    print('speedup: x%.0f'%(tn/tsa))
    plt.figure(0, figsize=(9,5))
    plt.clf()
    plt.subplots_adjust(left=0.05, right=0.95, top=0.93,
                        bottom=0.12, hspace=0.25, wspace=0)
    ax = plt.subplot(1,4,1)
    ax.set_aspect('equal')
    plt.pcolormesh(X, Y, I, cmap='inferno', vmin=0)
    # plt.imshow(I, cmap='gist_heat', vmin=0, origin='lower',
    #             extent=[_r[0], _r[-1], _r[0], _r[-1]])
    title = r'image %dx%d, $\theta$=%.2fmas'%(Nx, Nx, diam)
    plt.title(title)
    plt.xlim(plt.xlim()[1], plt.xlim()[0])
    plt.xlabel('E <- (mas)')
    plt.ylabel('-> N (mas)')
    x, y = np.array([0,  0]), np.array([-diam/2, diam/2])

    # -- label
    # if not stretch is None:
    #     rot = stretch[1]*np.pi/180
    #     x, y = np.cos(rot)*x+np.sin(rot)*y, -np.sin(rot)*x+np.cos(rot)*y
    # else:
    #     rot = 0
    # plt.plot(x, y, 'o-w', alpha=0.5)
    # plt.text(0, 0, r'$\theta$=%.2fmas'%(diam), rotation=90+rot*180/np.pi,
    #         color='w', ha='center', va='bottom', alpha=0.5)

    ax0 = plt.subplot(2,4,2)
    plt.title('|V| numerical')
    ax0.set_aspect('equal')
    pvis = plt.pcolormesh(c*diam*U/wl0, c*diam*V/wl0, np.abs(Vis),
                          cmap='gist_stern', vmin=0, vmax=1)
    plt.colorbar(pvis)

    ax = plt.subplot(2,4,3, sharex=ax0, sharey=ax0)
    plt.title('|V| semi-analytical')
    ax.set_aspect('equal')
    plt.pcolormesh(c*diam*U/wl0, c*diam*V/wl0, np.abs(Visp),
                    cmap='gist_stern', vmin=0, vmax=1)

    ax = plt.subplot(2,4,4, sharex=ax0, sharey=ax0)
    dyn = 1. # in 1/100 visibility
    plt.title('$\Delta$|V| (1/100)')
    ax.set_aspect('equal')
    res = 100*(np.abs(Vis)-np.abs(Visp))
    pv = plt.pcolormesh(c*diam*U/wl0, c*diam*V/wl0,
                    res, cmap='RdBu',
                    vmin=-np.max(np.abs(res)),
                    vmax=np.max(np.abs(res)),
                    )
    plt.colorbar(pv)
    print('median  |V|  residual (abs.) = %.3f'%np.median(np.abs(res)), '%')
    print('90perc  |V|  residual (abs.) = %.3f'%np.percentile(np.abs(res), 90), '%')

    ax = plt.subplot(2,4,6, sharex=ax0, sharey=ax0)
    plt.xlabel(r'B$\theta$/$\lambda$ (m.rad/m)')
    ax.set_aspect('equal')
    plt.title('$\phi$ numerical')
    plt.pcolormesh(c*diam*U/wl0, c*diam*V/wl0, 180/np.pi*np.angle(Vis),
                    cmap='hsv', vmin=-180, vmax=180)
    plt.colorbar()

    ax = plt.subplot(2,4,7, sharex=ax0, sharey=ax0)
    plt.xlabel(r'B$\theta$/$\lambda$ (m.rad/m)')
    ax.set_aspect('equal')
    plt.title('$\phi$ semi-analytical')
    plt.pcolormesh(c*diam*U/wl0, c*diam*V/wl0, 180/np.pi*np.angle(Visp),
                    cmap='hsv', vmin=-180, vmax=180)

    ax = plt.subplot(2,4,8, sharex=ax0, sharey=ax0)
    dyn = 1
    plt.title('$\Delta\phi$ (deg)')
    ax.set_aspect('equal')
    res = 180/np.pi*((np.angle(Vis)-np.angle(Visp)+np.pi)%(2*np.pi)-np.pi)
    pp = plt.pcolormesh(c*diam*U/wl0, c*diam*V/wl0, res,
                        cmap='RdBu', vmin=-dyn, vmax=dyn)
    print('median phase residual (abs.) = %.3f'%np.median(np.abs(res)), 'deg')
    print('90perc phase residual (abs.) = %.3f'%np.percentile(np.abs(res), 90), 'deg')

    plt.colorbar(pp)
    ax0.set_xlim(ax0.get_xlim()[1], ax0.get_xlim()[0])
    return

def _smoothUnwrap(s, n=5):
    """
    s: signal in degrees

    unwrap if the phase jump is smooth, i.e. over many samples
    """
    offs = np.zeros(len(s)) # total offsets

    for i in np.arange(len(s))[n:-n]:
        if (np.median(s[i-n:i])-np.median(s[i:i+n]))<-180:
            s[i+1:]-=360
        if (np.median(s[i-n:i])-np.median(s[i:i+n]))>180:
            s[i+1:]+=360
    return s
