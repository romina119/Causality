import time
from math import sqrt
from scipy import linalg
import numpy as np

def var_lasso_fista(X, Y, alpha, maxit=1000, TOL=1e-6):
    soft_threshold = lambda x, l : np.sign(x) * np.maximum(np.abs(x) - l, 0.)
    _, Ndim = X.shape
    NdimT = Y.shape[1] 
    x = np.zeros((Ndim, NdimT))
    obj_fcn = []
    t = 1
    z = x.copy()
    L = linalg.norm(X) ** 2
    time_start = time.time()
    for it in range(maxit):
        xold = x.copy()
        z = z + X.T.dot(Y - X.dot(z)) / L
        x = soft_threshold(z, alpha / L)
        t0 = t
        t = (1. + sqrt(1. + 4. * t ** 2)) / 2.
        z = x + ((t0 - 1.) / t) * (x - xold)
        J = 0.5 * linalg.norm(X.dot(x) - Y) ** 2 + alpha * linalg.norm(x, 1)
        obj_fcn.append( J )
        dJ = abs(obj_fcn[it] - obj_fcn[it-1])
        print( f'iteration {it+1}, J = {J}, delta J = {dJ}' )
        if it>0 and dJ < TOL: break
    elapsed_time = time.time() - time_start
    return x, obj_fcn, elapsed_time
