# oicandid.py

import numpy as np
import matplotlib.pyplot as plt
import time

import multiprocessing


import oimodels

def fitMap(oi, firstGuess=None, fitAlso=[], rmin=1, rmax=20, rstep=1.,
            multi=True):
    """
    firstguess: model. Star should be parametrized with '*,...' and companion 'c,...'.
        Other components can be used as well in the initial guess which can be
        passed in the variable "firstguess". Fit Only
    """

    if firstGuess is None:
        firstGuess = {'*,ud':0.0, '*,f':1.0,
                      'c,ud':0.0, 'c,f':0.1}

    fitOnly = ['c,f', 'c,x', 'c,y']
    if type(fitAlso):
        fitAlso.extend(fitAlso)

    kwargs = {'maxfev':1000, 'ftol':1e-5, 'verbose':False, 'fitOnly':fitOnly,}

    X, Y = [], []
    for r in  np.linspace(rmin, rmax, int((rmax-rmin)/rstep+1)):
        t = np.linspace(0, 2*np.pi, max(int(2*np.pi*r/rstep+1), 5))[:-1]
        X.extend(list(r*np.cos(t)))
        Y.extend(list(r*np.sin(t)))

    plt.close(0)
    plt.figure(0)
    plt.subplot(1,1,1, aspect='equal')
    plt.plot(X, Y, '+k')

    N = len(X)

    res = []
    if multi:
        if type(multi)!=int:
            Np = min(multiprocessing.cpu_count(), N)
        else:
            Np = min(multi, N)
        print('running', N, 'fits...')
        # -- estimate fitting time by running 'Np' fit in parallel
        t = time.time()
        pool = multiprocessing.Pool(Np)
        for i in range(min(Np, N)):
            tmp = firstGuess.copy()
            tmp['c,x'] = X[i]
            tmp['c,y'] = Y[i]
            res.append(pool.apply_async(oimodels.fitOI, (oi, tmp, ), kwargs))
        pool.close()
        pool.join()
        res = [r.get(timeout=1) for r in res]
        print('one fit takes ~%.2fs using %d threads'%(
                (time.time()-t)/min(Np, N), Np))

        # -- run the remaining
        if N>Np:
            pool = multiprocessing.Pool(Np)
            for i in range(max(N-Np, 0)):
                tmp = firstGuess.copy()
                tmp['c,x'] = X[Np+i]
                tmp['c,y'] = Y[Np+i]
                res.append(pool.apply_async(oimodels.fitOI, (oi, tmp, ), kwargs))
            pool.close()
            pool.join()
            res = res[:Np]+[r.get(timeout=1) for r in res[Np:]]
    else:
        Np = 1
        t = time.time()
        res.append(fitOI(oi, firstGuess, **kwargs))
        print('one fit takes ~%.2fs'%(time.time()-t))
        for i in range(N-1):
            tmp = firstGuess.copy()
            tmp['c,x'] = X[Np+i]
            tmp['c,y'] = Y[Np+i]
            res.append(oimodels.fitOI(oi, tmp, **kwargs))
    print('it took %.1fs, %.2fs per fit on average'%(time.time()-t,
                                                    (time.time()-t)/N))

    return res
