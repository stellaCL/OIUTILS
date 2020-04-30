import numpy as np
from astropy.io import fits
import scipy.signal
import copy

def loadOI(filename, insname=None, targname=None, verbose=True,
           with_header=False, medfilt=False, tellurics=None):
    """
    load OIFITS "filename" and return a dict:

    'WL': wavelength (1D array)
    'OI_VIS2': dict index by baseline. each baseline is a dict as well:
        ['V2', 'EV2', 'u', 'v', 'MJD'] v2 and ev2 1d array like wl, u and v are scalar
    'OI_VIS': dict index by baseline. each baseline is a dict as well:
        ['|V|', 'E|V|', 'PHI', 'EPHI', 'u', 'v', 'MJD'] (phases in degrees)
    'OI_T3': dict index by baseline. each baseline is a dict as well:
        [AMP', 'EAMP', 'PHI', 'EPHI', 'u1', 'v1', 'u2', 'v2', 'MJD'] (phases in degrees)

    can contains other things, *but None can start with 'OI_'*:
    'filename': name of the file
    'insname': name of the instrument (default: all of them)
    'targname': name of the instrument (default: single one if only one)
    'header': full header of the file if "with_header" is True (default False)

    All variables are keyed by the name of the baseline (or triplet). e.g.:

    oi['OI_VIS2']['V2'][baseline] is a 2D array: shape (len(u), len(wl))

    One can specify "insname" for instrument name. If None provided, returns as
    many dictionnary as they are instruments (either single dict for single
    instrument, of list of dictionnary).

    """
    if type(filename)!=str:
        res = []
        for f in filename:
            tmp = loadOI(f, insname=insname, with_header=with_header,
                         medfilt=medfilt, tellurics=tellurics, targname=targname,
                         verbose=verbose)
            if type(tmp)==list:
                res.extend(tmp)
            elif type(tmp)==dict:
                res.append(tmp)
        return res

    res = {}
    h = fits.open(filename)

    # -- how many instruments?
    instruments = []
    for hdu in h:
        if 'EXTNAME' in hdu.header and hdu.header['EXTNAME']=='OI_WAVELENGTH':
            instruments.append(hdu.header['INSNAME'])
        if 'EXTNAME' in hdu.header and hdu.header['EXTNAME']=='OI_TARGET':
            targets = {hdu.data['TARGET'][i].strip():hdu.data['TARGET_ID'][i] for
                        i in range(len(hdu.data['TARGET']))}
    if targname is None and len(targets)==1:
        targname = list(targets.keys())[0]
    assert targname in targets.keys(), 'unknown target "'+str(targname)+'", '+\
        'should be in ['+', '.join(['"'+t+'"' for t in list(targets.keys())])+']'

    if insname is None:
        if len(instruments)==1:
            insname = instruments[0]
        else:
            h.close()
            print('WARNING: insname not specified, using results for %s'%str(instruments))
            # -- return list: one dict for each insname
            return [loadOI(filename, insname=ins, with_header=with_header, medfilt=medfilt) for ins in instruments]

    assert insname in instruments, 'unknown instrument "'+insname+'", '+\
        'should be in ['+', '.join(['"'+t+'"' for t in instruments])+']'

    res['insname'] = insname
    res['filename'] = filename
    res['targname'] = targname
    if with_header:
        res['header'] = h[0].header

    if verbose:
        print('loadOI: loading', res['filename'])
        print('  > insname:', '"'+insname+'"','targname:', '"'+targname+'"')
    try:
        # -- VLTI 4T specific
        OPL = {}
        for i in range(4):
            T = h[0].header['ESO ISS CONF STATION%d'%(i+1)]
            opl = 0.5*(h[0].header['ESO DEL DLT%d OPL START'%(i+1)] +
                       h[0].header['ESO DEL DLT%d OPL END'%(i+1)])
            opl += h[0].header['ESO ISS CONF A%dL'%(i+1)]
            OPL[T] = opl
        res['OPL'] = OPL
        T = np.mean([h[0].header['ESO ISS TEMP TUN%d'%i] for i in [1,2,3,4]]) # T in C
        P = h[0].header['ESO ISS AMBI PRES'] # pressure in mbar
        H = h[0].header['ESO ISS AMBI RHUM'] # relative humidity: TODO outside == inside probably no ;(
        #print('T(C), P(mbar), H(%)', T, P, H)
        res['n_lab'] = n_JHK(res['WL'].astype(np.float64), 273.15+T, P, H)
    except:
        pass

    for hdu in h:
        if 'EXTNAME' in hdu.header and hdu.header['EXTNAME']=='OI_WAVELENGTH' and\
            hdu.header['INSNAME']==insname:
            res['WL'] = hdu.data['EFF_WAVE']*1e6

    #res['n_lab'] = n_JHK(res['WL'].astype(np.float64))#, 273.15+T, P, H)

    oiarray = dict(zip(h['OI_ARRAY'].data['STA_INDEX'],
                   np.char.strip(h['OI_ARRAY'].data['STA_NAME'])))

    # -- V2 baselines == telescopes pairs
    res['OI_VIS2'] = {}
    res['OI_VIS'] = {}
    res['OI_T3'] = {}
    res['OI_FLUX'] = {}
    for hdu in h:
        if 'EXTNAME' in hdu.header and hdu.header['EXTNAME']=='TELLURICS' and\
                    len(hdu.data['TELL_TRANS'])==len(res['WL']):
            res['TELLURICS'] = hdu.data['TELL_TRANS']
        if 'EXTNAME' in hdu.header and hdu.header['EXTNAME']=='OI_FLUX' and\
                    hdu.header['INSNAME']==insname:
            sta1 = [oiarray[s] for s in hdu.data['STA_INDEX']]
            for k in set(sta1):
                w = (np.array(sta1)==k)*(hdu.data['TARGET_ID']==targets[targname])
                try:
                    res['OI_FLUX'][k] = {'FLUX':hdu.data['FLUX'][w,:],
                                        'EFLUX':hdu.data['FLUXERR'][w,:],
                                        'FLAG':hdu.data['FLAG'][w,:],
                                        'MJD':hdu.data['MJD'][w],
                                         }
                except:
                    res['OI_FLUX'][k] = {'FLUX':hdu.data['FLUXDATA'][w,:],
                                        'EFLUX':hdu.data['FLUXERR'][w,:],
                                        'FLAG':hdu.data['FLAG'][w,:],
                                        'MJD':hdu.data['MJD'][w],
                                         }

                res['OI_FLUX'][k]['FLAG'] = np.logical_or(res['OI_FLUX'][k]['FLAG'],
                                                          ~np.isfinite(res['OI_FLUX'][k]['FLUX']))
                res['OI_FLUX'][k]['FLAG'] = np.logical_or(res['OI_FLUX'][k]['FLAG'],
                                                          ~np.isfinite(res['OI_FLUX'][k]['EFLUX']))
        elif 'EXTNAME' in hdu.header and hdu.header['EXTNAME']=='OI_VIS2' and\
                    hdu.header['INSNAME']==insname:
            sta2 = [oiarray[s[0]]+oiarray[s[1]] for s in hdu.data['STA_INDEX']]
            for k in set(sta2):
                w = (np.array(sta2)==k)*(hdu.data['TARGET_ID']==targets[targname])
                if k in res['OI_VIS2']:
                    for k1, k2 in [('V2', 'VIS2DATA'), ('EV2', 'VIS2ERR')]:
                        res['OI_VIS2'][k][k1] = np.append(res['OI_VIS2'][k][k1],
                                                        hdu.data[k2][w,:], axis=0)
                    for k1, k2 in [('u', 'UCOORD'), ('v', 'VCOORD'),('MJD','MJD')]:
                        res['OI_VIS2'][k][k1] = np.append(res['OI_VIS2'][k][k1],
                                                          hdu.data[k2][w])
                    tmp = hdu.data['UCOORD'][w][:,None]/res['WL'][None,:]
                    res['OI_VIS2'][k]['u/wl'] = np.append(res['OI_VIS2'][k]['u/wl'], tmp)
                    tmp = hdu.data['VCOORD'][w][:,None]/res['WL'][None,:]
                    res['OI_VIS2'][k]['v/wl'] = np.append(res['OI_VIS2'][k]['v/wl'], tmp)
                    res['OI_VIS2'][k]['FLAG'] = np.logical_or(res['OI_VIS2'][k]['FLAG'],
                                                              ~np.isfinite(res['OI_VIS2'][k]['V2']))
                    res['OI_VIS2'][k]['FLAG'] = np.logical_or(res['OI_VIS2'][k]['FLAG'],
                                                              ~np.isfinite(res['OI_VIS2'][k]['EV2']))
                else:
                    res['OI_VIS2'][k] = {'V2':hdu.data['VIS2DATA'][w,:],
                                         'EV2':hdu.data['VIS2ERR'][w,:],
                                         'u':hdu.data['UCOORD'][w],
                                         'v':hdu.data['VCOORD'][w],
                                         'MJD':hdu.data['MJD'][w],
                                         'u/wl': hdu.data['UCOORD'][w][:,None]/
                                                res['WL'][None,:],
                                         'v/wl': hdu.data['VCOORD'][w][:,None]/
                                                res['WL'][None,:],
                                         'FLAG':hdu.data['FLAG'][w,:]
                                        }
                    res['OI_VIS2'][k]['FLAG'] = np.logical_or(res['OI_VIS2'][k]['FLAG'],
                                                              ~np.isfinite(res['OI_VIS2'][k]['V2']))
                    res['OI_VIS2'][k]['FLAG'] = np.logical_or(res['OI_VIS2'][k]['FLAG'],
                                                              ~np.isfinite(res['OI_VIS2'][k]['EV2']))

                res['OI_VIS2'][k]['B/wl'] = np.sqrt(res['OI_VIS2'][k]['u/wl']**2+
                                                    res['OI_VIS2'][k]['v/wl']**2)
                res['OI_VIS2'][k]['PA'] = np.angle(res['OI_VIS2'][k]['v/wl']+
                                                   1j*res['OI_VIS2'][k]['u/wl'], deg=True)

        # -- V baselines == telescopes pairs
        elif 'EXTNAME' in hdu.header and hdu.header['EXTNAME']=='OI_VIS' and\
                    hdu.header['INSNAME']==insname:
            sta2 = [oiarray[s[0]]+oiarray[s[1]] for s in hdu.data['STA_INDEX']]
            for k in set(sta2):
                w = (np.array(sta2)==k)*(hdu.data['TARGET_ID']==targets[targname])
                if k in res['OI_VIS']:
                    for k1, k2 in [('|V|', 'VIS2AMP'), ('E|V|', 'VISAMPERR'),
                                    ('PHI', 'VISPHI'), ('EPHI', 'VISPHIERR')]:
                        res['OI_VIS'][k][k1] = np.append(res['OI_VIS'][k][k1],
                                                         hdu.data[k2][w,:], axis=0)
                    for k1, k2 in [('u', 'UCOORD'), ('v', 'VCOORD'), ('MJD', 'MJD')]:
                        res['OI_VIS'][k][k1] = np.append(res['OI_VIS'][k][k1],
                                                          hdu.data[k2][w])
                    tmp = hdu.data['UCOORD'][w][:,None]/res['WL'][None,:]
                    res['OI_VIS'][k]['u/wl'] = np.append(res['OI_VIS'][k]['u/wl'], tmp)
                    tmp = hdu.data['VCOORD'][w][:,None]/res['WL'][None,:]
                    res['OI_VIS'][k]['v/wl'] = np.append(res['OI_VIS'][k]['v/wl'], tmp)
                    res['OI_VIS'][k]['FLAG'] = np.logical_or(res['OI_VIS'][k]['FLAG'],
                                                             ~np.isfinite(res['OI_VIS'][k]['|V|']))
                    res['OI_VIS'][k]['FLAG'] = np.logical_or(res['OI_VIS'][k]['FLAG'],
                                                             ~np.isfinite(res['OI_VIS'][k]['E|V|']))

                else:
                    res['OI_VIS'][k] = {'|V|':hdu.data['VISAMP'][w,:],
                                        'E|V|':hdu.data['VISAMPERR'][w,:],
                                        'PHI':hdu.data['VISPHI'][w,:],
                                        'EPHI':hdu.data['VISPHIERR'][w,:],
                                        'MJD':hdu.data['MJD'][w],
                                        'u':hdu.data['UCOORD'][w],
                                        'v':hdu.data['VCOORD'][w],
                                        'u/wl': hdu.data['UCOORD'][w][:,None]/
                                               res['WL'][None,:],
                                        'v/wl': hdu.data['VCOORD'][w][:,None]/
                                               res['WL'][None,:],
                                        'FLAG':hdu.data['FLAG'][w,:]
                                        }
                res['OI_VIS'][k]['B/wl'] = np.sqrt(res['OI_VIS'][k]['u/wl']**2+
                                                    res['OI_VIS'][k]['v/wl']**2)
                res['OI_VIS'][k]['PA'] = np.angle(res['OI_VIS'][k]['v/wl']+
                                                  1j*res['OI_VIS'][k]['u/wl'], deg=True)

                res['OI_VIS'][k]['FLAG'] = np.logical_or(res['OI_VIS'][k]['FLAG'],
                                                         ~np.isfinite(res['OI_VIS'][k]['|V|']))
                res['OI_VIS'][k]['FLAG'] = np.logical_or(res['OI_VIS'][k]['FLAG'],
                                                          ~np.isfinite(res['OI_VIS'][k]['E|V|']))

    for hdu in h:
        if 'EXTNAME' in hdu.header and hdu.header['EXTNAME']=='OI_T3' and\
                    hdu.header['INSNAME']==insname:
            # -- T3 baselines == telescopes pairs
            sta3 = [oiarray[s[0]]+oiarray[s[1]]+oiarray[s[2]] for s in hdu.data['STA_INDEX']]
            # -- limitation: assumes all telescope have same number of char!
            n = len(sta3[0])//3 # number of char per telescope
            for k in set(sta3):
                w = (np.array(sta3)==k)*(hdu.data['TARGET_ID']==targets[targname])
                # -- find triangles
                t, s = [], []
                # -- first baseline
                if k[:2*n] in sta2:
                    t.append(k[:2*n])
                    s.append(1)
                elif k[n:2*n]+k[:n] in sta2:
                    t.append(k[n:2*n]+k[:n])
                    s.append(-1)

                # -- second baseline
                if k[n:] in sta2:
                    t.append(k[n:])
                    s.append(1)
                elif k[2*n:3*n]+k[n:2*n] in sta2:
                    t.append(k[2*n:3*n]+k[n:2*n])
                    s.append(-1)

                # -- third baseline
                if k[2*n:3*n]+k[:n] in sta2:
                    t.append(k[2*n:3*n]+k[:n])
                    s.append(1)
                elif k[:n]+k[2*n:3*n] in sta2:
                    t.append(k[:n]+k[2*n:3*n])
                    s.append(-1)

                if k in res['OI_T3']:
                    for k1, k2 in [('T3AMP', 'T3AMP'), ('ET3AMP', 'T3AMPERR'),
                                    ('T3PHI', 'T3PHI'), ('ET3PHI', 'T3PHIERR'),]:
                        res['OI_T3'][k][k1] = np.append(res['OI_T3'][k][k1],
                                                        hdu.data[k2][w,:], axis=0)
                    for k1, k2 in [('u1', 'U1COORD'), ('u2', 'U2COORD'),
                                   ('v1', 'V1COORD'), ('v2', 'V2COORD'),
                                   ('MJD', 'MJD')]:
                        res['OI_T3'][k][k1] = np.append(res['OI_T3'][k][k1],
                                                         hdu.data[k2][w])
                    res['OI_T3'][k]['FLAG'] = np.logical_or(res['OI_T3'][k]['FLAG'],
                                                            ~np.isfinite(res['OI_T3'][k]['T3AMP']))
                    res['OI_T3'][k]['FLAG'] = np.logical_or(res['OI_T3'][k]['FLAG'],
                                                            ~np.isfinite(res['OI_T3'][k]['ET3AMP']))
                else:
                    res['OI_T3'][k] = {'T3AMP':hdu.data['T3AMP'][w,:],
                                       'ET3AMP':hdu.data['T3AMPERR'][w,:],
                                       'T3PHI':hdu.data['T3PHI'][w,:],
                                       'ET3PHI':hdu.data['T3PHIERR'][w,:],
                                       'MJD':hdu.data['MJD'][w],
                                       'u1':hdu.data['U1COORD'][w],
                                       'v1':hdu.data['V1COORD'][w],
                                       'u2':hdu.data['U2COORD'][w],
                                       'v2':hdu.data['V2COORD'][w],
                                       'formula': (s, t),
                                       'FLAG':hdu.data['FLAG'][w,:]
                                        }
                res['OI_T3'][k]['B1'] = np.sqrt(res['OI_T3'][k]['u1']**2+
                                                res['OI_T3'][k]['v1']**2)
                res['OI_T3'][k]['B2'] = np.sqrt(res['OI_T3'][k]['u2']**2+
                                                res['OI_T3'][k]['v2']**2)
                res['OI_T3'][k]['B3'] = np.sqrt((res['OI_T3'][k]['u1']+res['OI_T3'][k]['u2'])**2+
                                                (res['OI_T3'][k]['v1']+res['OI_T3'][k]['v2'])**2)
                bmax = np.maximum(res['OI_T3'][k]['B1'], res['OI_T3'][k]['B2'])
                bmax = np.maximum(res['OI_T3'][k]['B3'], bmax)
                res['OI_T3'][k]['Bmax/wl'] = bmax[:,None]/res['WL'][None,:]
                res['OI_T3'][k]['FLAG'] = np.logical_or(res['OI_T3'][k]['FLAG'],
                                                        ~np.isfinite(res['OI_T3'][k]['T3AMP']))
                res['OI_T3'][k]['FLAG'] = np.logical_or(res['OI_T3'][k]['FLAG'],
                                                        ~np.isfinite(res['OI_T3'][k]['ET3AMP']))

    key = 'OI_VIS'
    if res['OI_VIS']=={}:
        res.pop('OI_VIS')
        key = 'OI_VIS2'
    if res['OI_VIS2']=={}:
        res.pop('OI_VIS2')
    if res['OI_FLUX']=={}:
        res.pop('OI_FLUX')
    if res['OI_T3']=={}:
        res.pop('OI_T3')
    else:
        # -- match MJDs for T3 computations:
        for k in res['OI_T3'].keys():
            s, t = res['OI_T3'][k]['formula']
            w0, w1, w2 = [], [], []
            for mjd in res['OI_T3'][k]['MJD']:
                w0.append(np.argmin(np.abs(res[key][t[0]]['MJD']-mjd)))
                w1.append(np.argmin(np.abs(res[key][t[1]]['MJD']-mjd)))
                w2.append(np.argmin(np.abs(res[key][t[2]]['MJD']-mjd)))
            res['OI_T3'][k]['formula'] = [s, t, w0, w1, w2]

    if 'OI_FLUX' in res:
        res['telescopes'] = sorted(list(res['OI_FLUX'].keys()))
    else:
        res['telescopes'] = []
        for k in res[key].keys():
            res['telescopes'].append(k[:len(k)//2])
            res['telescopes'].append(k[len(k)//2:])
        res['telescopes'] = sorted(list(set(res['telescopes'])))
    res['baselines'] = sorted(list(res[key].keys()))
    if 'OI_T3' in res.keys():
        res['triangles'] = sorted(list(res['OI_T3'].keys()))
    else:
        res['triangles'] = []

    if not 'TELLURICS' in res.keys():
        res['TELLURICS'] = np.ones(res['WL'].shape)
    if not tellurics is None:
        # -- forcing tellurics to given vector
        res['TELLURICS'] = tellurics
    if 'OI_FLUX' in res.keys():
        for k in res['OI_FLUX'].keys():
            # -- raw flux
            res['OI_FLUX'][k]['RFLUX'] = res['OI_FLUX'][k]['FLUX'].copy()
            # -- corrected from tellurics
            res['OI_FLUX'][k]['FLUX'] /= res['TELLURICS'][None,:]

    if medfilt:
        if type(medfilt) == int:
            kernel_size = 2*(medfilt//2)+1
        else:
            kernel_size = None
        res = medianFilt(res, kernel_size)

    confperMJD = {}
    for l in filter(lambda x: x in ['OI_VIS', 'OI_VIS2', 'OI_T3', 'OI_FLUX'], res.keys()):
        for k in res[l].keys():
            for mjd in res[l][k]['MJD']:
                if not mjd in confperMJD.keys():
                    confperMJD[mjd] = [k]
                else:
                    if not k in confperMJD[mjd]:
                        confperMJD[mjd].append(k)
    res['configurations per MJD'] = confperMJD

    if verbose:
        mjd = []
        for e in ['OI_VIS2', 'OI_VIS', 'OI_T3', 'OI_FLUX']:
            if e in res.keys():
                for k in res[e].keys():
                    mjd.extend(list(res[e][k]['MJD']))
        print('  > MJD:', sorted(set(mjd)))
        print('  >', '-'.join(res['telescopes']), end=' | ')
        print('WL:', res['WL'].shape, round(np.min(res['WL']), 3), 'to',
              round(np.max(res['WL']), 3), 'um', end=' | ')
        print(sorted(list(filter(lambda x: x.startswith('OI_'), res.keys()))),
                end=' | ')
        print('TELLURICS:', res['TELLURICS'].min()<1)
        # print('  >', 'telescopes:', res['telescopes'],
        #       'baselines:', res['baselines'],
        #       'triangles:', res['triangles'])

    return res

def mergeOI(OI, verbose=True, debug=False):
    """
    takes OI, a list of oifits files readouts (from loadOI), and merge them into
    a smaller number of entities based with same spectral coverage

    TODO: how polarisation in handled?
    """
    # -- create unique identifier for each setups
    setups = [oi['insname']+str(oi['WL']) for oi in OI]

    merged = [] # list of unique setup which have been merged so far
    master = [] # same length as OI, True if element hold merged data for the setup
    res = [] # result
    for i, oi in enumerate(OI):
        if debug:
            print(i, setups[i])
        if not setups[i] in merged:
            # -- this will hold the data for this setup
            merged.append(setups[i])
            master.append(True)
            if debug:
                print('super oi')
            res.append(copy.deepcopy(oi))
            continue # exit for loop, nothing else to do
        else:
            # -- find data holder for this setup
            #i0 = np.argmax([setups[j]==setups[i] and master[j] for j in range(i)])
            i0 = merged.index(setups[i])
            master.append(False)
            # -- start merging with filename
            res[i0]['filename'] += ';'+oi['filename']

        if debug:
            print('  will be merged with', i0)
            print('  merging...')
        # -- extensions to be merged:
        exts = ['OI_VIS', 'OI_VIS2', 'OI_T3', 'OI_FLUX']
        for l in filter(lambda e: e in exts, oi.keys()):
            if not l in res[i0].keys():
                # -- extension not present, using the one from oi
                res[i0][l] = copy.deepcopy(oi[l])
                if debug:
                    print('    ', i0, 'does not have', l, '!')
                # -- no additional merging needed
                dataMerge = False
            else:
                dataMerge = True

            # -- merge list of tel/base/tri per MJDs:
            for mjd in oi['configurations per MJD']:
                if not mjd in res[i0]['configurations per MJD']:
                    res[i0]['configurations per MJD'][mjd] = \
                        oi['configurations per MJD'][mjd]
                else:
                    for k in oi['configurations per MJD'][mjd]:
                        if not k in res[i0]['configurations per MJD'][mjd]:
                            res[i0]['configurations per MJD'][mjd].append(k)

            if not dataMerge:
                continue

            # -- merge data in the extension:
            for k in oi[l].keys():
                # -- for each telescpe / baseline / triangle
                if not k in res[i0][l].keys():
                    # -- unknown telescope / baseline / triangle
                    # -- just add it in the the dict
                    res[i0][l][k] = copy.deepcopy(oi[l][k])
                    if 'FLUX' in l:
                        res[i0]['telescopes'].append(k)
                    elif 'VIS' in l:
                        res[i0]['baselines'].append(k)
                    elif 'T3' in l:
                        res[i0]['triangles'].append(k)
                    if debug:
                        print('    adding', k, 'to', l, 'in', i0)
                else:
                    # -- merging data (most complicated case)
                    # -- ext1 are scalar
                    # -- ext2 are vector of length WL
                    if l=='OI_FLUX':
                        ext1 = ['MJD']
                        ext2 = ['FLUX', 'EFLUX', 'FLAG', 'RFLUX']
                    elif l=='OI_VIS2':
                        ext1 = ['u', 'v', 'MJD']
                        ext2 = ['V2', 'EV2', 'FLAG', 'u/wl', 'v/wl', 'B/wl', 'PA']
                    elif l=='OI_VIS':
                        ext1 = ['u', 'v', 'MJD']
                        ext2 = ['|V|', 'E|V|', 'PHI', 'EPHI', 'FLAG', 'u/wl', 'v/wl', 'B/wl', 'PA']
                    if l=='OI_T3':
                        ext1 = ['u1', 'v1', 'u2', 'v2', 'MJD', 'B1', 'B2', 'B3']
                        ext2 = ['T3AMP', 'ET3AMP', 'T3PHI', 'ET3PHI',
                                'FLAG','Bmax/wl']
                    if debug:
                        print(l, k, res[i0][l][k].keys())
                    for t in ext1:
                        # -- append len(MJDs) data
                        res[i0][l][k][t] = np.append(res[i0][l][k][t], oi[l][k][t])
                    for t in ext2:
                        # -- append (len(MJDs),len(WL)) data
                        s1 = res[i0][l][k][t].shape # = (len(MJDs),len(WL))
                        s2 = oi[l][k][t].shape # = (len(MJDs),len(WL))
                        res[i0][l][k][t] = np.append(res[i0][l][k][t], oi[l][k][t])
                        res[i0][l][k][t] = res[i0][l][k][t].reshape(s1[0]+s2[0], s1[1])

    for r in res:
        for k in ['telescopes', 'baselines', 'triangles']:
            if k in r:
                r[k] = list(set(r[k]))
    # -- special case for T3 formulas
    # -- match MJDs for T3 computations:
    for r in res:
        if 'OI_T3' in r.keys():
            if 'OI_VIS' in r.keys():
                key = 'OI_VIS'
            else:
                key = 'OI_VIS2'
            for k in r['OI_T3'].keys():
                s, t, w0, w1, w2 = r['OI_T3'][k]['formula']
                _w0, _w1, _w2 = [], [], []
                #print(k, np.round(r['OI_T3'][k]['MJD'], 3)-50000)
                for mjd in r['OI_T3'][k]['MJD']:
                    _w0.append(np.argmin(np.abs(r[key][t[0]]['MJD']-mjd)))
                    _w1.append(np.argmin(np.abs(r[key][t[1]]['MJD']-mjd)))
                    _w2.append(np.argmin(np.abs(r[key][t[2]]['MJD']-mjd)))
                #print(' ', t[0], np.round(r[key][t[0]]['MJD'], 3)-50000, _w0)
                #print(' ', t[1], np.round(r[key][t[1]]['MJD'], 3)-50000, _w1)
                #print(' ', t[2], np.round(r[key][t[2]]['MJD'], 3)-50000, _w2)
                r['OI_T3'][k]['formula'] = [s, t, _w0, _w1, _w2]

    # -- keep only master oi's, which contains merged data:
    if verbose:
        print('mergeOI:', len(oi), 'data files merged into', len(res), end=' ')
        if len(res)>1:
            print('dictionnaries with unique setups:')
        else:
            print('dictionnary with a unique setup:')

        for i, m in enumerate(res):
            print(' [%d]'%i, m['insname'], ' %d wavelengths:'%len(m['WL']),
                    m['WL'][0], '..', m['WL'][-1], 'um', end='\n  ')
            print('\n  '.join(m['filename'].split(';')))
    return res

def medianFilt(oi, kernel_size=None):
    """
    kernel_size is the half width
    """
    if type(oi) == list:
        return [medianFilt(o, kernel_size=kernel_size) for o in oi]

    if 'OI_FLUX' in oi.keys():
        # -- make sure the tellurics are handled properly
        if 'TELLURICS' in oi.keys():
            t = oi['TELLURICS']
        else:
            t = np.ones(np.len(oi['WL']))
        for k in oi['OI_FLUX'].keys():
            for i in range(len(oi['OI_FLUX'][k]['MJD'])):
                mask = ~oi['OI_FLUX'][k]['FLAG'][i,:]
                oi['OI_FLUX'][k]['FLUX'][i,mask] = scipy.signal.medfilt(
                    oi['OI_FLUX'][k]['FLUX'][i,mask]/t[mask],
                    kernel_size=kernel_size)*t[mask]
                oi['OI_FLUX'][k]['EFLUX'][i,mask] /= np.sqrt(kernel_size)
    if 'OI_VIS' in oi.keys():
        for k in oi['OI_VIS'].keys():
            for i in range(len(oi['OI_VIS'][k]['MJD'])):
                mask = ~oi['OI_VIS'][k]['FLAG'][i,:]
                oi['OI_VIS'][k]['|V|'][i,mask] = scipy.signal.medfilt(
                    oi['OI_VIS'][k]['|V|'][i,mask], kernel_size=kernel_size)
                oi['OI_VIS'][k]['E|V|'][i,mask] /= np.sqrt(kernel_size)

                oi['OI_VIS'][k]['PHI'][i,mask] = scipy.signal.medfilt(
                    oi['OI_VIS'][k]['PHI'][i,mask], kernel_size=kernel_size)
                oi['OI_VIS'][k]['EPHI'][i,mask] /= np.sqrt(kernel_size)

    if 'OI_VIS2' in oi.keys():
        for k in oi['OI_VIS2'].keys():
            for i in range(len(oi['OI_VIS2'][k]['MJD'])):
                mask = ~oi['OI_VIS2'][k]['FLAG'][i,:]
                oi['OI_VIS2'][k]['V2'][i,mask] = scipy.signal.medfilt(
                    oi['OI_VIS2'][k]['V2'][i,mask], kernel_size=kernel_size)
                oi['OI_VIS2'][k]['EV2'][i,mask] /= np.sqrt(kernel_size)

    if 'OI_T3' in oi.keys():
        for k in oi['OI_T3'].keys():
            for i in range(len(oi['OI_T3'][k]['MJD'])):
                mask = ~oi['OI_T3'][k]['FLAG'][i,:]
                oi['OI_T3'][k]['T3PHI'][i,mask] = scipy.signal.medfilt(
                    oi['OI_T3'][k]['T3PHI'][i,mask], kernel_size=kernel_size)
                oi['OI_T3'][k]['ET3PHI'][i,mask] /= np.sqrt(kernel_size)

                oi['OI_T3'][k]['T3AMP'][i,mask] = scipy.signal.medfilt(
                    oi['OI_T3'][k]['T3AMP'][i,mask], kernel_size=kernel_size)
                oi['OI_T3'][k]['ET3AMP'][i,mask] /= np.sqrt(kernel_size)
    return oi

def n_JHK(wl, T=None, P=None, H=None):
    """
    wl: wavelnegth in microns (only valid from 1.3 to 2.5um)
    T: temperature in K
    P: pressure in mbar
    H: relative humidity in %

    from https://arxiv.org/pdf/physics/0610256.pdf
    """
    nu = 1e4/wl
    nuref = 1e4/2.25 # cm−1

    # -- https://arxiv.org/pdf/physics/0610256.pdf
    # -- table 1
    # -- i; ciref / cmi; ciT / cmiK;  ciTT / [cmiK2]; ciH / [cmi/%]; ciHH / [cmi/%2]
    table1a=[[0, 0.200192e-3, 0.588625e-1, -3.01513, -0.103945e-7, 0.573256e-12],
             [1, 0.113474e-9, -0.385766e-7, 0.406167e-3, 0.136858e-11, 0.186367e-16],
             [2, -0.424595e-14, 0.888019e-10, -0.514544e-6, -0.171039e-14, -0.228150e-19],
             [3, 0.100957e-16, -0.567650e-13, 0.343161e-9, 0.112908e-17, 0.150947e-22],
             [4, -0.293315e-20, 0.166615e-16, -0.101189e-12, -0.329925e-21, -0.441214e-26],
             [5, 0.307228e-24, -0.174845e-20, 0.106749e-16, 0.344747e-25, 0.461209e-30]]

    # -- cip / [cmi/Pa]; cipp / [cmi/Pa2]; ciTH / [cmiK/%]; ciTp / [cmiK/Pa]; ciHp / [cmi/(% Pa)]
    table1b = [[0, 0.267085e-8, 0.609186e-17, 0.497859e-4, 0.779176e-6, -0.206567e-15],
               [1, 0.135941e-14, 0.519024e-23, -0.661752e-8, 0.396499e-12, 0.106141e-20],
               [2, 0.135295e-18, -0.419477e-27, 0.832034e-11, 0.395114e-16, -0.149982e-23],
               [3, 0.818218e-23, 0.434120e-30, -0.551793e-14, 0.233587e-20, 0.984046e-27],
               [4, -0.222957e-26, -0.122445e-33, 0.161899e-17, -0.636441e-24, -0.288266e-30],
               [5, 0.249964e-30, 0.134816e-37, -0.169901e-21, 0.716868e-28, 0.299105e-34]]

    Tref, Href, pref = 273.15+17.5, 10., 75e3
    if T is None:
        T = Tref
    if P is None:
        P = pref/100
    if H is None:
        H = Href

    n = 0.0
    p = P*100 # formula in Pa, not mbar

    for k,ca in enumerate(table1a):
        i = ca[0]
        ciref = ca[1]
        ciT = ca[2]
        ciTT = ca[3]
        ciH = ca[4]
        ciHH = ca[5]
        cb = table1b[k]
        cip = cb[1]
        cipp = cb[2]
        ciTH = cb[3]
        ciTp = cb[4]
        ciHp = cb[5]
        # -- equation 7
        ci = ciref + ciT*(1/T - 1/Tref) + ciTT*(1/T - 1/Tref)**2 +\
            ciH*(H-Href) + ciHH*(H-Href)**2 + cip*(p-pref) + cipp*(p-pref)**2 +\
            ciTH*(1/T-1/Tref)*(H-Href) + ciTp*(1/T-1/Tref)*(p-pref) +\
            ciHp*(H-Href)*(p-pref)
        # -- equation 6
        #print('mathar:', i, ciref, ci)
        n += ci*(nu - nuref)**i
    return n+1.0
