from __future__ import print_function

import numpy as np

class propagator(object):

    __doc__ = """

    Propagates some variable x of dimensionality d, x[d].
    The propagation is carried out over niter itertions using
    steps of width step.

    """

    def __init__(self, x, d, step, niter):

        self.x    = x
        self.__x0 = np.copy(x)
        self.d    = d
        self.step  = step
        self.niter = niter
        self.count = 0
        self.complete = False
        if d>1:
            self.next = propagator(self.x, d-1, step, niter)

    def are_we_done(self):

        if self.d>1:
            if self.count==(self.niter[self.d-1]-1) and self.next.are_we_done():
                return True

        else:
            if self.count==(self.niter[self.d-1]-1):
                return True

        return False

    def add(self):

        if self.count<self.niter[self.d-1]-1:
            self.x[self.d-1] += self.step
            self.count += 1
        else:
            self.x[self.d-1] = self.__x0[self.d-1]
            self.count = 0
            if self.d>1:
                self.next.add()


class gist_bounds(object):

    def __init__(self, xmin, xmax, T_start=100.0, T_delta=5.0, tau=10, 
                Ptarget=0.5, Tconst=True, verbose=False):

        __doc__="""
        This class checks for bounds and is also a thermostat.
        """

        self.xmin = xmin
        self.xmax = xmax
        self.T    = T_start
        self.T_delta = T_delta
        self.tau   = tau
        self.Ptarget = Ptarget
        self.Tconst  = Tconst
        self.verbose = verbose

        self.iter  = 0
        self.P_avg = 0.

        if self.Ptarget<0. or self.Ptarget>1.:
            raise ValueError("Ptarget must fulfill 0<P<1.")
        if self.T<=0.:
            raise ValueError("T must be >0.")
        if self.tau<0:
            raise ValueError("tau must be >=0")

        self.bounds = list()
        for _xmin, _xmax in zip(self.xmin, self.xmax):

            if np.abs(_xmin) == np.inf:
                _xmin = None

            if np.abs(_xmax) == np.inf:
                _xmax = None

            self.bounds.append((_xmin, _xmax))

    def __call__(self, **kwargs):

        result = True

        x_new = kwargs["x_new"]
        f_new = kwargs["f_new"]
        x_old = kwargs["x_old"]
        f_old = kwargs["f_old"]
        tmax  = bool(np.all(x_new <= self.xmax))
        tmin  = bool(np.all(x_new >= self.xmin))

        if not tmax or not tmin:
            result = False

        elif not self.Tconst:

            df = f_old-f_new
            P  = np.exp(-df/self.T)

            if df>0:
                result = True

            else:
                r  = np.random.random()
                if P<r:
                    result = True
                else:
                    result = False

            self.iter  += 1
            self.P_avg += P
            if self.iter == self.tau:
                self.P_avg /= self.iter
                if self.P_avg > self.Ptarget:
                    if self.verbose:
                        print("Attempting temperature change %6.3f" %self.T, end=' ')
                    self.T += self.T_delta
                    if self.verbose:
                        print("-> %6.3f" %self.T)
                self.iter   = 0
                self.P_avg  = 0.

        return result

    def get_bounds_for_minimizer(self):

        bounds = list()

        for _xmin, _xmax in zip(self.xmin, self.xmax):

            if np.abs(_xmin) == np.inf:
                _xmin = None

            if np.abs(_xmax) == np.inf:
                _xmax = None

            bounds.append((_xmin, _xmax))

        return bounds


class take_step(object):

    def __init__(self, step_temp, stepsize=0.5):

        self.step_temp = step_temp
        self.stepsize  = stepsize
        self.size      = self.step_temp.shape[0]

    def __call__(self, x):

        s     = self.stepsize
        x[:]  = x[:] + np.random.uniform(-s, s, self.size)*self.step_temp
        return x