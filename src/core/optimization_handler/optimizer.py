from scipy import optimize
import optimization_handler.noisy_opt as noisyopt


class SolutionFinder(object):
    def __init__(self, obj_func, bounds, x0):
        self.obj_func = obj_func
        self.bounds = bounds
        self.x0 = x0

    def bhop(
        self,
        path_args=None,
        method="L-BFGS-B",
        n_iter=100,
        stepsize=200,
        Temp=0,
        callback=None,
    ):
        """This calls scipy's Basin Hopping function which is a
        Stochastic Global Optimization algoorithm
        Random perturbation, Local minimization, Accept or reject the new solution
        TODO To resolve the problem when  the local pertubation results in negative demand
        which can result in SUMO Exception"""

        if path_args is None:
            minimizer_kwargs = {"method": method}
        else:
            minimizer_kwargs = {"method": method, "args": path_args}
        result = optimize.basinhopping(
            self.obj_func,
            accept_test=self.bounds,
            x0=self.x0,
            minimizer_kwargs=minimizer_kwargs,
            niter=n_iter,
            stepsize=stepsize,
            T=Temp,
            callback=callback,
        )
        return result

    def rps(self, path_args=None, **kwargs):
        """Robust Pattern Search Algorithm for noisy objective functions
        with Adaptive Sampling in Appendix of Mayer 2016. This may give exceptions
        or wrong answer when used with deeterminstic and symmetrical functional forms"""

        result = noisyopt.minimizeCompass(
            self.obj_func, x0=self.x0, args=path_args, paired=False, **kwargs
        )
        return result

    def spsa(self, path_args=None, **kwargs):
        """SPSA: for noisy objective functions. This may give exceptions
        or wrong answer when used with determinstic and symmetrical functional forms"""
        if path_args is None:
            pass
        else:
            pass
        result = noisyopt.minimizeSPSA(
            self.obj_func, x0=self.x0, args=path_args, **kwargs
        )
        return result

    def pcspsa(self, num_features, pca_model, var_scaler, path_args=None, **kwargs):
        """PC-SPSA: for noisy objective functions"""
        if path_args is None:
            pass
        else:
            pass
        result = noisyopt.minimizePCSPSA(
            self.obj_func,
            x0=self.x0,
            num_od_pairs=num_features,
            pca=pca_model,
            scaler=var_scaler,
            args=path_args,
            **kwargs
        )
        return result

    def w_spsa(self, weights_wspsa, path_args=None, **kwargs):
        """Wrapper for W-SPSA"""
        if path_args is None:
            pass
        else:
            pass
        result = noisyopt.minimize_W_SPSA(
            self.obj_func, x0=self.x0, w_matrix=weights_wspsa, args=path_args, **kwargs
        )
        return result


if __name__ == "__main__":
    pass
