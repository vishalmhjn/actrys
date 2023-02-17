import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# import ctf.functions2d

from scipy import optimize
import noisy_opt as noisyopt

import platform

sys_id = platform.system()

# Evaluate the sensitivity, time and space efficiency of the optimisationo allgorithms
# Implement Callbacks to store the results

objective_val = []


from ctf.functions2d.function2d import Function2D

class NoisyTest(Function2D):
    """ Noisy Sum Squares Function. """

    def __init__(self):
        """ Constructor. """
        self.min = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.value = 0.0
        self.domain = np.array([[-100, 100], [-100, 100], [-100, 100], [-100, 100], [-100, 100],
                            [-100, 100], [-100, 100], [-100, 100] ,[-100, 100], [-100, 100]])
        self.n = 10
        self.smooth = True
        self.info = [True, False, False]
        # Description
        self.latex_name = "Sum Squares Function"
        self.latex_type = "Bowl-Shaped"
        self.latex_cost = r"\[ f(\mathbf{x}) = \sum_{i=0}^d  (i + 1) x_i^2 \]"
        self.latex_desc = "The Sum Squares function, also referred to as the Axis Parallel Hyper-Ellipsoid" \
                          "function, has no local minimum except the global one. It is continuous, convex and unimodal."

    @staticmethod
    def cost(x,eval=False):
        """ Cost function. """
        # Cost
        c = np.zeros(x.shape[1:])
        
        # Calculate Cost
        c = np.sum([x[i]**2 for i in range(0, 10)], axis=0) + 10*np.random.rand()
        # Return Cost
        return c

def plot_synthetic_function(x_p, y_p, F):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(45, -45)
    ax.plot_surface(xgrid, ygrid, F, cmap='terrain')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('F(x, y)')
    plt.show()

def quadratic(x):
    return (x[0]**2 + x[1]**2 -100)

def noisy_quadratic(x):
    # print(x)
    # print(x[0]**2 + x[1]**2 + 2*np.random.randn())
    return x[0]**2 + x[1]**2 + 2*np.random.randn()

class MyBounds(object):
    def __init__(self, bounds):
        self.xmax = np.array([i[1] for i in bounds])
        self.xmin = np.array([i[0] for i in bounds])
    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin

def callbackF(*args):
    return objective_val.append(NoisyTest.cost(args[0]))

class SolutionFinder(object):
    def __init__(self,
                obj_func,
                bounds,
                x0):
        self.obj_func = obj_func
        self.bounds = bounds
        self.x0 = x0

    def bhop(self,
            path_args = None,
            method = "L-BFGS-B",
            n_iter = 100,
            stepsize=200,
            Temp = 0,
            callback=None):
        '''This calls scipy's Basin Hopping function which is a
        Stochastic Global Optimization algoorithm
        Random perturbation, Local minimization, Accept or reject the new solution
        TODO To resolve the problem when  the local pertubation results in negative demand
        which can result in SUMO Exception'''

        if path_args is None:
            minimizer_kwargs = {"method": method}
        else:
            minimizer_kwargs = {"method": method, "args": path_args}
        result = optimize.basinhopping(self.obj_func,
                                        accept_test=self.bounds,
                                        x0 = self.x0,
                                        minimizer_kwargs = minimizer_kwargs,
                                        niter = n_iter,
                                        stepsize=stepsize,
                                        T=Temp,
                                        callback=callback)
        return  result


    def rps(self,
            path_args=None,
            **kwargs):
        '''Robust Pattern Search Algorithm for noisy objective functions
        with Adaptive Sampling in Appendix of Mayer 2016. This may give exceptions
        or wrong answer when used with deeterminstic and symmetrical functional forms'''

        result = noisyopt.minimizeCompass(self.obj_func,
                                        x0 = self.x0,
                                        args = path_args,
                                        paired=False,
                                        **kwargs)
        return  result


    def spsa(self,
            path_args=None,
            **kwargs):
        '''SPSA: for noisy objective functions. This may give exceptions
        or wrong answer when used with determinstic and symmetrical functional forms'''
        if path_args is None:
            pass
        else:
            pass
        result = noisyopt.minimizeSPSA(self.obj_func,
                                        x0 = self.x0,
                                        args = path_args,
                                        **kwargs)
        return  result

    def pcspsa(self,
                num_features,
                pca_model,
                var_scaler,
                path_args=None,
                **kwargs):
        '''PC-SPSA: for noisy objective functions'''
        if path_args is None:
            pass
        else:
            pass
        result = noisyopt.minimizePCSPSA(self.obj_func,
                                        x0 = self.x0,
                                        num_od_pairs = num_features,
                                        pca = pca_model,
                                        scaler = var_scaler,
                                        args = path_args,
                                        **kwargs)
        return  result

    def w_spsa(self,
            weights_wspsa,
            path_args=None,
            **kwargs):
        '''Wrapper for W-SPSA'''
        if path_args is None:
            pass
        else:
            pass
        result = noisyopt.minimize_W_SPSA(self.obj_func,
                                        x0 = self.x0,
                                        w_matrix = weights_wspsa,
                                        args = path_args,
                                        **kwargs)
        return  result

if __name__ == "__main__":

    # Tried funcs

    obj_func = NoisyTest()
    # obj_func = ctf.functions2d.SumSquares() # bshop works well with this, but not rps/ spsa
    # obj_func = ctf.functions2d.ThreeHumpCamel()
    # obj_func = ctf.functions2d.Eggholder()
    # obj_func = ctf.functions2d.Beale()

    x = np.arange(obj_func.domain[0][0],obj_func.domain[0][1],0.1)
    y = np.arange(obj_func.domain[1][0], obj_func.domain[1][1],0.1)
    xgrid, ygrid = np.meshgrid(x, y)
    xy = np.stack([xgrid, ygrid])

    # plot_synthetic_function(xgrid, ygrid, quadratic(xy))


    # sf = SolutionFinder(quadratic,
    # 					MyBounds([10,10],[-10,-10]),
    # 					[100.,100.])

    # result = sf.bhop(n_iter = 100)

    # print(result)
    # print("================================")
    # result = sf.rps(bounds= [[-10,10],[-10,10]])

    # print(result)
    # print("================================")

    # result = sf.spsa(bounds= [[-10,10],[-10,10]])

    # print(result)
    # print("================================")
    # print(obj_func.domain)

    # obj_func.plot_cost()
    # plt.show()
    # plot_synthetic_function(xgrid, ygrid, obj_func.cost(xy))

    sf = SolutionFinder(NoisyTest.cost,
                    MyBounds(obj_func.domain),
                    [10,20,5,5,99,90,80,88,50,50])
    # result = sf.bhop(n_iter = 10, stepsize=1, Temp=0.1, callback=callbackF)

    # print(result)

    # plt.plot(objective_val, label='B-HOP')

    # print("================================")

    # objective_val = []
    # result = sf.rps(bounds= obj_func.domain, deltatol=0.1, callback=callbackF)
    # plt.plot(objective_val, label='RPSAS')

    # print(result)
    print("================================")


    # objective_val = []
    result = sf.spsa(bounds= obj_func.domain,
                    callback=callbackF,
                    paired=False,
                    disp=False,
                    c= 0.01, # small c if direcction finding is easy for the algorithm, or standard deviation of the noise
                    a= 0.06, # step size should be small to prevent high frequency fluctuations, depends on the scale of objective function
                    gamma = 0.085, # change these as a last resort
                    alpha= 0.57, # change this as a last resort
                    niter=500, 
                    reps=4) # there is a limit to which it ccan converge, in the end increasing iterations is the only choice
    print(objective_val)

    if sys_id=="Darwin":
        plt.plot(objective_val, label='SPSA' )
        plt.show()

    print(result)

    # plt.legend()
    # plt.show()
