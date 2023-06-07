# Based on main file of Python package  noisyopt: https://github.com/andim/noisyopt
# The MIT License (MIT)

# Copyright (c) 2016-2017 Andreas Mayer

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import warnings
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sys import getsizeof
from joblib import Parallel, delayed, parallel_backend
from sklearn.preprocessing import StandardScaler

# include OptimizeResult class for machines on which scipy version is too old
try:
    from scipy.optimize import OptimizeResult
except:
    class OptimizeResult(dict):
        """ Represents the optimization result.

        Parameters
        ----------
        x : ndarray
            The solution of the optimization.
        success : bool
            Whether or not the optimizer exited successfully.
        status : int
            Termination status of the optimizer. Its value depends on the
            underlying solver. Refer to `message` for details.
        message : str
            Description of the cause of the termination.
        fun, jac, hess, hess_inv : ndarray
            Values of objective function, Jacobian, Hessian or its inverse (if
            available). The Hessians may be approximations, see the documentation
            of the function in question.
        nfev, njev, nhev : int
            Number of evaluations of the objective functions and of its
            Jacobian and Hessian.
        nit : int
            Number of iterations performed by the optimizer.
        maxcv : float
            The maximum constraint violation.
        Notes
        -----
        There may be additional attributes not listed above depending of the
        specific solver. Since this class is essentially a subclass of dict
        with attribute accessors, one can see which attributes are available
        using the `keys()` method.
        """
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)

        __setattr__ = dict.__setitem__
        __delattr__ = dict.__delitem__

        def __repr__(self):
            if self.keys():
                m = max(map(len, list(self.keys()))) + 1
                return '\n'.join([k.rjust(m) + ': ' + repr(v)
                                  for k, v in self.items()])
            else:
                return self.__class__.__name__ + "()"


def minimizeCompass(func, x0, args=(),
            bounds=None, scaling=None,
            redfactor=2.0, deltainit=1.0, deltatol=1e-3, feps=1e-15,
            errorcontrol=True, funcNinit=30, funcmultfactor=2.0,
            paired=True, alpha=0.05, disp=False, callback=None, **kwargs):
    """
    Minimization of an objective function by a pattern search.

    The search algorithm is a simple compass search along coordinate directions.
    If the function evaluation contains a stochastic element, then the function
    is called repeatedly to average over the stochasticity in the function
    evaluation. The number of evaluations is adapted dynamically to ensure
    convergence.

    Parameters
    ----------
    func: callable
        objective function to be minimized:
        called as `func(x, *args, **kwargs)`,
        if `paired=True`, then called with keyword argument `seed` additionally
    x0: array-like
        starting point
    args: tuple
        extra arguments to be supplied to func
    bounds: array-like
        bounds on the variables
    scaling: array-like
        scaling by which to multiply step size and tolerances along different dimensions
    redfactor: float
        reduction factor by which to reduce delta if no reduction direction found 
    deltainit: float
        inital pattern size
    deltatol: float
        smallest pattern size
    feps: float
       smallest difference in function value to resolve 
    errorcontrol: boolean
        whether to control error of simulation by repeated sampling
    funcNinit: int, only for errorcontrol=True
        initial number of iterations to use for the function, do not set much lower
        than 30 as otherwise there is no sufficient statistics for function comparisons
    funcmultfactor: float, only for errorcontrol=True
        multiplication factor by which to increase number of iterations of function
    paired: boolean, only for errorcontrol=True
        compare for same random seeds
    alpha: float, only for errorcontrol=True
        significance level of tests, the higher this value the more statistics
        is acquired, which decreases the risk of taking a step in a non-descent
        direction at the expense of higher computational cost per iteration
    disp: boolean
        whether to output status updates during the optimization
    callback: callable
        called after each iteration, as callback(xk), where xk is the current parameter vector.

    Returns
    -------
    scipy.optimize.OptimizeResult object
        special entry: free
        Boolean array indicating parameters that are unconstrained at the optimum (within feps) 
    """
    #TODO: implement variable deltas for different directions (might speed up things, see review)
    if disp:
        print('minimization starting')
        print('args', args)
        print('errorcontrol', errorcontrol)
        print('paired', paired)
    # absolute tolerance for float comparisons
    floatcompatol = 1e-14
    x0 = np.asarray(x0)
    if scaling is None:
        scaling = np.ones(x0.shape)
    else:
        scaling = np.asarray(scaling)
    # ensure initial point lies within bounds
    if bounds is not None:
        bounds = np.asarray(bounds)
        np.clip(x0, bounds[:, 0], bounds[:, 1], out=x0)
    def clip(x, d):
        """clip x+d to respect bounds
        returns clipped result and effective distance"""
        xnew = x + d
        if bounds is not None:
            # if test point depasses set to boundary instead
            xclipped = np.clip(xnew, bounds[:, 0], bounds[:, 1])
            deltaeff = np.abs(x - xclipped).sum()
            return xclipped, deltaeff
        else:
            return xnew, delta
    # generate set of search directions (+- s_i e_i | i = 1, ...,  N)
    def unit(i, N):
        "return ith unit vector in R^N"
        arr = np.zeros(N)
        arr[i] = 1.0
        return arr
    N = len(x0)
    generatingset = [unit(i, N)*direction*scaling[i] for i in np.arange(N) for direction in [+1, -1]]
   
    # memoize function
    if errorcontrol:
        funcm = AveragedFunction(
            func, fargs=args, paired=paired, N=funcNinit)
        # apply Bonferroni correction to confidence level
        # (need more statistics in higher dimensions)
        alpha = alpha/len(generatingset)
    else:
        # freeze function arguments
        def funcf(x, **kwargs):
            return func(x, *args, **kwargs)
        funcm = _memoized(funcf)

    x = x0 
    delta = deltainit
    # number of iterations
    nit = 0
    # continue as long as delta is larger than tolerance
    # or if there was an update during the last iteration
    found = False
    while delta >= deltatol-floatcompatol or found:
        nit += 1
        # if delta gets close to deltatol, do iteration with step size deltatol instead
        if delta/redfactor < deltatol:
            delta = deltatol
        if disp:
            print('nit %i, Delta %g' % (nit, delta))
        found = False
        np.random.shuffle(generatingset)
        for d in generatingset:
            xtest, deltaeff = clip(x, delta*d)
            if deltaeff < floatcompatol:
                continue
            # Does xtest improve upon previous function value?
            if ((not errorcontrol and (funcm(xtest) < funcm(x)-feps))
               or (errorcontrol
                   and funcm.test(xtest, x, type_='smaller', alpha=alpha))):
                x = xtest
                found = True
                if disp:
                    print(x)
            # Is non-improvement due to too large step size or missing statistics?
            elif ((deltaeff >= deltatol*np.sum(np.abs(d))) # no refinement for boundary steps smaller than tolerance
                    and ((not errorcontrol and (funcm(xtest) > funcm(x)+feps))  # changed the sign vishal 07-01-2021
                        or (errorcontrol
                            and funcm.test(xtest, x, type_='equality', alpha=alpha)
                            and (funcm.diffse(xtest, x) > feps)))):
                # If there is no significant difference the step size might
                # correspond to taking a step to the other side of the minimum.
                # Therefore test if middle point is better
                xmid = 0.5*(x+xtest)
                if ((not errorcontrol and funcm(xmid) < funcm(x)-feps)
                    or (errorcontrol
                        and funcm.test(xmid, x, type_='smaller', alpha=alpha))):
                    x = xmid
                    delta /= redfactor
                    found = True
                    if disp:
                        print('mid', x)
                # otherwise increase accuracy of simulation to try to get to significance
                elif errorcontrol:
                    funcm.N *= funcmultfactor
                    if disp:
                        print('new N %i' % funcm.N)
                    found = True
        if callback is not None:
            callback(x)
        if not found:
            delta /= redfactor

    message = 'convergence within deltatol'
    # check if any of the directions are free at the optimum
    delta = deltatol
    free = np.zeros(x.shape, dtype=bool)
    for d in generatingset:
        dim = np.argmax(np.abs(d))
        xtest, deltaeff = clip(x, delta*d)
        if deltaeff < deltatol*np.sum(np.abs(d))-floatcompatol: # do not consider as free for boundary steps
            continue
        if not free[dim] and (((not errorcontrol and funcm(xtest) - feps < funcm(x)) or
            (errorcontrol and funcm.test(xtest, x, type_='equality', alpha=alpha)
                and (funcm.diffse(xtest, x) < feps)))):
            free[dim] = True
            message += '. dim %i is free at optimum' % dim
                
    reskwargs = dict(x=x, nit=nit, nfev=funcm.nev, message=message, free=free,
                     success=True)
    if errorcontrol:
        f, funse = funcm(x)
        res = OptimizeResult(fun=f, funse=funse, **reskwargs)
    else:
        f = funcm(x)
        res = OptimizeResult(fun=f, **reskwargs)
    if disp:
        print(res)
    return res

def minimize(*args, **kwargs):
    "deprecated: use minimizeCompass instead"
    warnings.warn("This function will be deprecated in version 1.0. Use minimizeCompass instead",
                  PendingDeprecationWarning)
    return minimizeCompass(*args, **kwargs)

def minimizeSPSA(func, x0, args=(), bounds=None, niter=100, paired=True,
                 a=1.0, c=1.0, alpha = 0.602, gamma = 0.101, reps =1,
                 disp=False, callback=None):
    """
    Minimization of an objective function by a simultaneous perturbation
    stochastic approximation algorithm.

    Parameters
    ----------
    func: callable
        objective function to be minimized
    x0: array-like
        starting point
    args: tuple
        extra arguments to be supplied to func
    bounds: array-like
        bounds on the variables
    scaling: array-like
        scaling by which to multiply step size and tolerances along different dimensions
    niter: int
        maximum number of iterations of the algorithm
    paired: boolean
        calculate gradient for same random seeds
    a: float
       algorithm scaling parameter for step size
    c: float
       algorithm scaling parameter for evaluation step size
    disp: boolean
        whether to output status updates during the optimization
    callback: callable
        called after each iteration, as callback(xk), where xk is the current parameter vector.

    Returns
    -------
    scipy.optimize.OptimizeResult object
    """
    A = 0.01 * niter

    if bounds is None:
        project = lambda x: x
    else:
        bounds = np.asarray(bounds)
        project = lambda x: np.clip(x, bounds[:, 0], bounds[:, 1])

    if args is not None:
        # freeze function arguments
        def funcf(x, **kwargs):
            return func(x, *args, **kwargs)
    else:
        def funcf(x, **kwargs):
            return func(x, **kwargs)
    N = len(x0)
    x = np.array(x0)
    for k in range(niter):
        ak = a/(k+1.0+A)**alpha
        ck = c/(k+1.0)**gamma
        
        save_grad = []
        # replace zeros with a 1 as this will raise NAN error while

        for i in range(0, reps):
            delta = np.random.choice([-1, 1], size=N)
            fkwargs = dict()
            if paired:
                fkwargs['seed'] = np.random.randint(0, np.iinfo(np.uint32).max)
            if bounds is None:
                xplus  = np.round(x + np.multiply(x+1, ck*delta))
                xminus = np.round(x - np.multiply(x+1, ck*delta))
                with parallel_backend('threading', n_jobs=2):
                    list_perturb = Parallel(n_jobs=2)(delayed(funcf)(x_v, **kwargs)
                                                                    for x_v,kwargs in (
                                                                        [xplus, {'weighted': True, 'which_perturb':'positive'}],\
                                                                        [xminus, {'weighted': True, 'which_perturb':'negative'}]))
                pos_perturb = list_perturb[0]
                neg_perturb = list_perturb[1]
                grad = np.sum(pos_perturb - neg_perturb) / (2*ck*delta)
                save_grad.append(grad)
            else:

                xplus  = np.round(x + np.multiply(x+1, ck*delta))
                xminus = np.round(x - np.multiply(x+1, ck*delta))
                with parallel_backend('threading', n_jobs=2):
                    list_perturb = Parallel(n_jobs=2)(delayed(funcf)(x_v, **kwargs)
                                                                    for x_v,kwargs in (
                                                                        [xplus, {'weighted': True, 'which_perturb':'positive'}],\
                                                                        [xminus, {'weighted': True, 'which_perturb':'negative'}]))
                pos_perturb = list_perturb[0]
                neg_perturb = list_perturb[1]
                grad = np.sum(pos_perturb - neg_perturb) / (2*ck*delta)

                save_grad.append(grad)

        # take the mean of the replications
        grad = np.mean(save_grad, axis=0)
        
        # step-size is scaled proportionally to the size
        x = project(x - ak*grad*x)
        funcf(x, eval_rmsn=True, **fkwargs)
        # print 100 status updates if disp=True
        if disp and (k%5) == 0:
            pass
            # print(x)
        if callback is not None:
            callback(x)
    message = 'terminated after reaching max number of iterations'
    return OptimizeResult(fun=funcf(x), x=x, nit=niter, nfev=2*niter, message=message, success=True)

def minimizePCSPSA(func, x0, num_od_pairs, pca, scaler, args=(), bounds=None, \
                    niter=100, paired=True, a=1.0, c=1.0, alpha = 0.602, \
                    gamma = 0.101, reps =1, disp=False, callback=None):
    """
    Minimization of an objective function by a Principal component 
    simultaneous perturbation stochastic approximation algorithm.
    In this implementation the principal directions are freshly calculated
    in each iteration. Another approach is to calculate it outside this function
    and give is an argument to this function.
    Parameters
    ----------
    func: callable
        objective function to be minimized
    x0: array-like
        starting point
    args: tuple
        extra arguments to be supplied to func
    bounds: array-like
        bounds on the variables
    scaling: array-like
        scaling by which to multiply step size and tolerances along different dimensions
    niter: int
        maximum number of iterations of the algorithm
    paired: boolean
        calculate gradient for same random seeds
    a: float
       algorithm scaling parameter for step size
    c: float
       algorithm scaling parameter for evaluation step size
    disp: boolean
        whether to output status updates during the optimization
    callback: callable
        called after each iteration, as callback(xk), where xk is the current parameter vector.

    Returns
    -------
    scipy.optimize.OptimizeResult object
    """
    A = 0.01 * niter

    if bounds is None:
        project = lambda x: x
    else:
        bounds = np.asarray(bounds)
        project = lambda x: np.clip(x, bounds[:, 0], bounds[:, 1])

    if args is not None:
        # freeze function arguments
        def funcf(x, **kwargs):
            return func(x, *args, **kwargs)
    else:
        def funcf(x, **kwargs):
            return func(x, **kwargs)
    x = np.array(x0)

    
    # covert x to shape (Time, od_pairs) since od pairs are features for the principal components
    x = x.reshape(-1, num_od_pairs)
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    
    
    z = pca.fit_transform(x)
    number_components = z.shape[1]
    print(number_components)
    pca = PCA(n_components=number_components)
    z = pca.fit_transform(x)
    z = z.flatten()
    N = len(z)

    for k in range(niter):
        ak = a/(k+1.0+A)**alpha
        ck = c/(k+1.0)**gamma
        
        save_grad = []

        for i in range(0, reps):
            delta = np.random.choice([-1, 1], size=N)
            fkwargs = dict()
            if paired:
                fkwargs['seed'] = np.random.randint(0, np.iinfo(np.uint32).max)

            if bounds is None:
                zplus  = z + np.multiply(z, ck*delta)
                zminus = z - np.multiply(z, ck*delta)
                xplus = scaler.inverse_transform(pca.inverse_transform(zplus.reshape(-1, number_components))).flatten()
                xminus = scaler.inverse_transform(pca.inverse_transform(zminus.reshape(-1, number_components))).flatten()
                grad = (funcf(xplus, **fkwargs) - funcf(xminus, **fkwargs)) / (zplus-zminus)
                save_grad.append(grad)
                print(grad)

            else:				
                zplus = project(z + np.multiply(z, ck*delta))
                zminus = project(z - np.multiply(z, ck*delta))
                xplus = scaler.inverse_transform(pca.inverse_transform(zplus.reshape(-1, number_components))).flatten()
                xminus = scaler.inverse_transform(pca.inverse_transform(zminus.reshape(-1, number_components))).flatten()
                grad = (funcf(xplus, **fkwargs) - funcf(xminus, **fkwargs)) / (zplus-zminus)
                save_grad.append(grad)

        grad = np.mean(save_grad, axis=0)
        
        z = project(z - ak*grad*z)
        print(z)
        x = scaler.inverse_transform(pca.inverse_transform(z.reshape(-1, number_components))).flatten()
        print(x)
        print("-")
        # print(x)
        funcf(x, eval_rmsn=True, **fkwargs)
        if disp and (k%5) == 0:
            pass
        if callback is not None:
            callback(x)
    message = 'terminated after reaching max number of iterations'
    return OptimizeResult(fun=funcf(x), x=x, nit=niter, nfev=2*niter, message=message, success=True)

def minimize_W_SPSA(func, x0, w_matrix, args=(), bounds=None, niter=100, paired=True,
                 a=1.0, c=1.0, alpha = 0.602, gamma = 0.101, reps =1, param_momentum=0.7,
                 network_correlation=0.02, disp=False, callback=None):
    """
    Minimization of an objective function by a Weighted simultaneous perturbation
    stochastic approximation algorithm. Lu Lu et al., 2015

    Parameters
    ----------
    func: callable
        objective function to be minimized
    x0: array-like
        starting point
    args: tuple
        extra arguments to be supplied to func
    bounds: array-like
        bounds on the variables
    scaling: array-like
        scaling by which to multiply step size and tolerances along different dimensions
    niter: int
        maximum number of iterations of the algorithm
    paired: boolean
        calculate gradient for same random seeds
    a: float
       algorithm scaling parameter for step size
    c: float
       algorithm scaling parameter for evaluation step size
    disp: boolean
        whether to output status updates during the optimization
    callback: callable
        called after each iteration, as callback(xk), where xk is the current parameter vector.

    Returns
    -------
    scipy.optimize.OptimizeResult object
    """
    A = 0.01 * niter
    momentum = param_momentum
    if bounds is None:
        project = lambda x: x
    else:
        bounds = np.asarray(bounds)
        project = lambda x: np.clip(x, bounds[:, 0], bounds[:, 1])

    if args is not None:
        # freeze function arguments
        def funcf(x, **kwargs):
            return func(x, *args, **kwargs)
    else:
        def funcf(x, **kwargs):
            return func(x, **kwargs)
    N = len(x0)
    x = np.array(x0)
    prev_change = 0
    for k in range(niter):
        ak = a/(k+1.0+A)**alpha
        ck = c/(k+1.0)**gamma
        
        save_grad = []

        delta = np.random.choice([-1, 1], size=N)
        for i in range(0, reps):
            fkwargs = dict()
            if paired:
                fkwargs['seed'] = np.random.randint(0, np.iinfo(np.uint32).max)
            if bounds is None:
                xplus  = np.round(x + np.multiply(x+1, ck*delta))
                xminus = np.round(x - np.multiply(x+1, ck*delta))
                with parallel_backend('threading', n_jobs=2):
                    list_perturb = Parallel(n_jobs=2)(delayed(funcf)(x_v, **kwargs)
                                                                    for x_v,kwargs in (
                                                                        [xplus, {'weighted': True, 'which_perturb':'positive'}],\
                                                                        [xminus, {'weighted': True, 'which_perturb':'negative'}]))
                pos_perturb = list_perturb[0]
                neg_perturb = list_perturb[1]
                grad = w_matrix @ (pos_perturb - neg_perturb) / (2*ck*delta)
                save_grad.append(grad)

            else:
                xplus  = np.round(x + np.multiply(x+1, ck*delta))
                xminus = np.round(x - np.multiply(x+1, ck*delta))
                with parallel_backend('threading', n_jobs=2):
                    list_perturb = Parallel(n_jobs=2)(delayed(funcf)(x_v, **kwargs)
                                                                    for x_v,kwargs in (
                                                                        [xplus, {'weighted': True, 'which_perturb':'positive'}],\
                                                                        [xminus, {'weighted': True, 'which_perturb':'negative'}]))
                pos_perturb = list_perturb[0]
                neg_perturb = list_perturb[1]
                grad = w_matrix @ (pos_perturb - neg_perturb) / (2*ck*delta)
                save_grad.append(grad)
        grad = np.mean(save_grad, axis=0)
        
        x = project(x - ak*grad - momentum*prev_change)
        prev_change = ak*grad

        funcf(x, eval_rmsn=True, **fkwargs)
        if disp and (k%5) == 0:
            pass
        if callback is not None:
            callback(x)
    message = 'terminated after reaching max number of iterations'
    return OptimizeResult(fun=funcf(x), spsa_a = ak, spsa_c = ck, 
                            x=x, nit=niter, nfev=2*niter, message=message, success=True)

class AverageBase(object):
    """
    Base class for averaged evaluation of noisy functions.
    """
    def __init__(self, N=30, paired=False):
        """
        Parameters
        ----------
        N: int
            number of calls to average over.
        paired: boolean
            if paired is chosen the same series of random seeds is used for different x
        """
        self._N = int(N)
        self.paired = paired
        if self.paired:
            self.uint32max = np.iinfo(np.uint32).max 
            self.seeds = list(np.random.randint(0, self.uint32max, size=int(N)))
        # cache previous iterations
        self.cache = {}
        # number of evaluations
        self.nev = 0

    @property
    def N(self):
        "number of evaluations"
        return self._N

    @N.setter
    def N(self, value):
        N = int(value)
        if self.paired and (N > self._N):
            Nadd = N - self._N
            self.seeds.extend(list(np.random.randint(0, self.uint32max, size=Nadd)))
        self._N = N

    def test0(self, x, type_='smaller', alpha=0.05, force=False, eps=1e-5, maxN=10000):
        """
        Compares the mean at x to zero.

        Parameters
        ----------
        type_: in ['smaller', 'equality']
            type of comparison to perform
        alpha: float
           significance level 
        force: boolean
            if true increase number of samples until equality rejected or meanse=eps or N > maxN
        eps: float 
        maxN: int
        """
        if force:
            while (self.test0(x, type_='equality', alpha=alpha, force=False, eps=eps)
                    and self(x)[1] > eps
                    and self.N < maxN):
                self.N *= 2.0

        mean, meanse = self(x)
        epscal = mean / meanse
        if type_ == 'smaller':
            return epscal < stats.norm.ppf(alpha)
        if type_ == 'equality':
            return np.abs(epscal) < stats.norm.ppf(1-alpha/2.0)
        raise NotImplementedError(type_)

class AveragedFunction(AverageBase):
    """Average of a function's return value over a number of runs.

        Caches previous results.
    """
    def __init__(self, func, fargs=None, **kwargs):
        """
        Parameters
        ----------
        func : callable
            function to average (called as `func(x, *fargs)`)
        fargs : tuple
            extra arguments for function
        """
        super(AveragedFunction, self).__init__(**kwargs)
        if fargs is not None:
            def funcf(x, **kwargs):
                return func(x, *fargs, **kwargs)
            self.func = funcf
        else:
            self.func = func

    def __call__(self, x):
        try:
            # convert to tuple (hashable!)
            xt = tuple(x)
        except TypeError:
            # if TypeError then likely floating point value
            xt = (x, )
        if xt in self.cache:
            Nold = len(self.cache[xt])
            if Nold < self.N:
                Nadd = self.N - Nold 
                if self.paired:
                    values = [self.func(x, seed=self.seeds[Nold+i]) for i in range(Nadd)]
                else:
                    values = [self.func(x) for i in range(Nadd)]
                self.cache[xt].extend(values)
                self.nev += Nadd
        else:
            if self.paired:
                values = [self.func(x, seed=self.seeds[i]) for i in range(self.N)]
            else:
                values = [self.func(x) for i in range(self.N)]
            self.cache[xt] = values 
            self.nev += self.N
        return np.mean(self.cache[xt]), np.std(self.cache[xt], ddof=1)/self.N**.5

    def diffse(self, x1, x2):
        """Standard error of the difference between the function values at x1 and x2""" 
        f1, f1se = self(x1)
        f2, f2se = self(x2)
        if self.paired:
            fx1 = np.array(self.cache[tuple(x1)])
            fx2 = np.array(self.cache[tuple(x2)])
            diffse = np.std(fx1-fx2, ddof=1)/self.N**.5 
            return diffse
        else:
            return (f1se**2 + f2se**2)**.5

    def test(self, xtest, x, type_='smaller', alpha=0.05):
        """
        Parameters
        ----------
        type_: in ['smaller', 'equality']
            type of comparison to perform
        alpha: float
            significance level
        """
        # call function to make sure it has been evaluated a sufficient number of times
        if type_ not in ['smaller', 'equality']:
            raise NotImplementedError(type_)
        ftest, ftestse = self(xtest)
        f, fse = self(x)
        # get function values
        fxtest = np.array(self.cache[tuple(xtest)])
        fx = np.array(self.cache[tuple(x)])
        if np.mean(fxtest-fx) == 0.0:
            if type_ == 'equality':
                return True
            if type_ == 'smaller':
                return False
        if self.paired:
            # if values are paired then test on distribution of differences
            statistic, pvalue = stats.ttest_rel(fxtest, fx, axis=None)
        else:
            statistic, pvalue = stats.ttest_ind(fxtest, fx, equal_var=False, axis=None)
        if type_ == 'smaller':
            # if paired then df=N-1, else df=N1+N2-2=2*N-2 
            df = self.N-1 if self.paired else 2*self.N-2
            pvalue = stats.t.cdf(statistic, df) 
            # return true if null hypothesis rejected
            return pvalue < alpha
        if type_ == 'equality':
            # return true if null hypothesis not rejected
            return pvalue > alpha

class DifferenceFunction(AverageBase):
    """Averages the difference of two function's return values over a number of runs
    """
    def __init__(self, func1, func2, fargs1=None, fargs2=None, **kwargs):
        """
        Parameters
        ----------
        func1,2 : callables
            functions to average (called as `func(x, *fargs)`)
        fargs1,2 : tuples
            extra arguments for functions
        kwargs: various
            accepts `AverageBase` kwargs and function kwargs
        """
        basekwargs = dict(N=kwargs.pop('N', 30),
                          paired=kwargs.pop('paired', False))
        super(DifferenceFunction, self).__init__(**basekwargs)
        if fargs1 is not None:
            def func1f(x, **kwargs):
                return func1(x, *fargs1, **kwargs)
        else:
            func1f = func1
        if fargs2 is not None:
            def func2f(x, **kwargs):
                return func2(x, *fargs2, **kwargs)
        else:
            func2f = func2
        self.funcs = [func1f, func2f]

    def __call__(self, x):
        try:
            # convert to tuple (hashable!)
            xt = tuple(x)
        except TypeError:
            # if TypeError then likely floating point value
            xt = (x, )
        for i, func in enumerate(self.funcs):
            ixt = i, xt
            if ixt in self.cache:
                Nold = len(self.cache[ixt])
                if Nold < self.N:
                    Nadd = self.N - Nold 
                    if self.paired:
                        values = [func(x, seed=self.seeds[Nold+i]) for i in range(Nadd)]
                    else:
                        values = [func(x) for i in range(Nadd)]
                    self.cache[ixt].extend(values)
                    self.nev += Nadd
            else:
                if self.paired:
                    values = [func(x, seed=self.seeds[i]) for i in range(self.N)]
                else:
                    values = [func(x) for i in range(self.N)]
                self.cache[ixt] = values 
                self.nev += self.N
        diff = np.asarray(self.cache[(0, xt)]) - np.asarray(self.cache[(1, xt)])
        return np.mean(diff), np.std(diff, ddof=1)/self.N**.5

class BisectException(Exception):
    pass

def bisect(func, a, b, xtol=1e-6, errorcontrol=True,
           testkwargs=dict(), outside='extrapolate',
           ascending=None,
           disp=False):
    """Find root by bysection search.

    If the function evaluation is noisy then use `errorcontrol=True` for adaptive
    sampling of the function during the bisection search.

    Parameters
    ----------
    func: callable
        Function of which the root should be found. If `errorcontrol=True`
        then the function should be derived from `AverageBase`.
    a, b: float
        initial interval
    xtol: float
        target tolerance for interval size
    errorcontrol: boolean
        if true, assume that function is instance of DifferenceFunction  
    testkwargs: only for `errorcontrol=True`
        see `AverageBase.test0`
    outside: ['extrapolate', 'raise']
        How to handle the case where f(a) and f(b) have same sign,
        i.e. where the root lies outside of the interval.
        If 'raise' throws a BisectException in this case.
    ascending: allow passing in directly whether function is ascending or not
        if ascending=True then it is assumed without check that f(a) < 0 and f(b) > 0
        if ascending=False then it is assumed without check that f(a) > 0 and f(b) < 0

    Returns
    -------
    float, root of function
    """
    search = True
    # check whether function is ascending or not
    if ascending is None:
        if errorcontrol:
            testkwargs.update(dict(type_='smaller', force=True))
            fa = func.test0(a, **testkwargs)
            fb = func.test0(b, **testkwargs)
        else:
            fa = func(a) < 0
            fb = func(b) < 0
        if fa and not fb:
            ascending = True
        elif fb and not fa:
            ascending =  False
        else:
            if disp:
                print('Warning: func(a) and func(b) do not have opposing signs -> no search done')
            if outside == 'raise':
                raise BisectException()
            search = False

    # refine interval until it has reached size xtol, except if root outside
    while (b-a > xtol) and search:
        mid = (a+b)/2.0
        if ascending:
            if ((not errorcontrol) and (func(mid) < 0)) or \
                    (errorcontrol and func.test0(mid, **testkwargs)):
                a = mid 
            else:
                b = mid
        else:
            if ((not errorcontrol) and (func(mid) < 0)) or \
                    (errorcontrol and func.test0(mid, **testkwargs)):
                b = mid 
            else:
                a = mid
        if disp:
            print('bisect bounds', a, b)
    # interpolate linearly to get zero
    if errorcontrol:
        ya, yb = func(a)[0], func(b)[0]
    else:
        ya, yb = func(a), func(b)
    m = (yb-ya) / (b-a)
    res = a-ya/m
    if disp:
        print('bisect final value', res)
    return res

class _memoized(object):
    """Decorator. Caches a function's return value each time it is called.

    If called later with the same arguments, the cached value is returned
    (not reevaluated).
    Can be turned of by passing `memoize=False` when calling the function.

    If the function arguments are not hashable, then no caching is attempted
    and the function is evaluated at every call. An error can instead be raised
    by passing `nothashable=raise` when calling the function.
    """
    def __init__(self, func):
        self.func = func
        self.cache = {}
        self.nev = 0

    def __call__(self, *args, **kwargs):
        # if args is not Hashable we can't cache
        # easier to ask for forgiveness than permission
        memoize = kwargs.pop('memoize', True)
        nothashable = kwargs.pop('nothashable', 'ignore')
        if memoize:
            try:
                index = ()
                for arg in args:
                    # unwrap iterable arguments
                    # (needed e.g. for np.ndarray which is not hashable)
                    try:
                        index += tuple(arg)
                    except TypeError:
                        index += (arg, ) 
                # try to also recompute if kwargs changed
                for item in kwargs.values():
                    try:
                        index += (float(item), )
                    except:
                        pass
                if index in self.cache:
                    return self.cache[index]
                else:
                    value = self.func(*args, **kwargs)
                    self.nev += 1
                    self.cache[index] = value
                    return value
            except TypeError:
                if nothashable == 'raise':
                    raise TypeError('Not hashable: %s' % str(args))
                else:
                    print('not hashable', args)
                    self.nev += 1
                    return self.func(*args, **kwargs)
        else:
            self.nev += 1
            return self.func(*args, **kwargs)
