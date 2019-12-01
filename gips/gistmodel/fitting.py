from scipy import optimize
import numpy as np

from gips.datastrc.gdat_fit_lib import gdat_fit_lib
from gips.gistmodel.fit_utils import gist_bounds
from gips.gistmodel.fit_utils import propagator
from gips.gistmodel._numerical_ext import pair_difference_ext

from gips import FLOAT
from gips import DOUBLE

import pygmo

pygmo.set_serialization_backend("pickle")

class MC_fitter(gdat_fit_lib):

    def __init__(self, gdatarec_dict, 
                    gdata_dict, 
                    ref_energy=-11.108, 
                    mode=0,
                    radiusadd=[0.,3.],
                    softness=1.,
                    softcut=2.,
                    exclude=None,
                    scaling=2.0,
                    decomp_E=False,
                    decomp_S=False,
                    verbose=False):

        ### Initialize with gdat_fit_lib:
        ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        ###
        ### self.N_rec
        ### self.N_enz
        ### self.N_pos
        ### self.N_cplx
        ### self.N_lig
        ### self.N_case
        ###
        ### self.maxdim
        ###
        ### self.gdat
        ### self.name
        ###
        ### self.E
        ### self.S
        ### self.g
        ### self.w
        ### self.vol
        ### self.ind_rec
        ### self.ind_case
        ### self.dg
        ### self.dh
        ### self.ds

        super(MC_fitter, self).__init__(gdatarec_dict=gdatarec_dict,
                                        gdata_dict=gdata_dict,
                                        ref_energy=ref_energy,
                                        mode=mode,
                                        radiusadd=radiusadd,
                                        softness=softness,
                                        softcut=softcut,
                                        exclude=exclude,
                                        scaling=scaling,
                                        verbose=verbose)

        self.prepare_gdat()

        self._calc_data = None
        self._f         = None
        self._result    = None

        self._exp_data  = None

        self._functional     = None
        self._restraint      = 0.
        self._restraint_grad = None
        self._x0             = None
        self._dx             = None
        self._gradients      = None
        self.kforce          = 100.
        self.kforce_f        = None
        self.anal_grad       = True
        self.anal_boundary   = True

        self.decomp = False
        if self.mode in [1,4,6,7]:
            self.decomp   = True
            self.decomp_E = decomp_E
            self.decomp_S = decomp_S
            if self.decomp_E and self.decomp_S:
                self.decomp_S = False


    def set_selection(self, select):

        if type(select)==type(None):
            if self.pairs:
                self.select = np.arange(self.N_pairs)
            else:
                self.select = np.arange(self.N_case)
        else:
            self.select = np.array(select)
        self.N_sele = self.select.shape[0]


    def set_pairs(self):

        _name     = self.name
        self.name = list()
        self.pairidx = list()

        if type(self.pairlist) != type(None):
            for i in range(len(self.pairlist)):
                p1 = self.pairlist[i][0]
                p2 = self.pairlist[i][1]
                #if not p1 in _name:
                #    raise Warning ("%s not found." %p1)
                #if not p2 in _name:
                #    raise Warning("%s not found." %p2)
                if p1 in self.exclude:
                    continue
                if p2 in self.exclude:
                    continue

                self.pairidx.append([_name.index(p1), _name.index(p2)])
                self.name.append("%s-%s" %(p1,p2))

        else:
            for i in range(self.N_case):
                for j in range(self.N_case):
                    if j<=i:
                        continue
                    p1 = _name[i]
                    p2 = _name[j]
                    if p1 in self.exclude:
                        continue
                    if p2 in self.exclude:
                        continue

                    self.pairidx.append([_name.index(p1), _name.index(p2)])
                    self.name.append("%s-%s" %(p1,p2))

        self.pairidx = np.array(self.pairidx)
        self.N_pairs = self.pairidx.shape[0]


    def set_x0(self):

        __doc__ = """
        Set initial parameters x0 randomly. Note, that this routine must be called
        prior to fitting, otherwise no initial parameters are set and 
        consequently all fitting operations will fail.
        """

        self._x0  = np.zeros(self._parms, dtype=DOUBLE)

        for i in range(self._parms):
            self._x0[i] = np.random.uniform(self.xmin[i], self.xmax[i])


    def rmsd(self):

        return self.rmsd_select(self.select)


    def rmsd_select(self, select):

        self._calc_diff(select)

        N = select.shape[0]

        if self.decomp:
            _rmsd = np.sqrt(self.diff2[0]/N),\
                     np.sqrt(self.diff2[1]/N)
        else:
            _rmsd = np.sqrt(self.diff2[0]/N)

        return _rmsd


    def get_f(self, select):

        return self._f[select]


    def _f_error(self, x):

        return self._f_error_select(x, self.select)


    def _f_error_select(self, x, select):

        self.gist_functional(x)
        self._f_process(x)
        if self._calc_diff(select):
            return self.diff2
        else:
            raise ValueError("Something went wrong during evaulation of objective function.")


    def fitness(self, x):

        return self._f_error(x)


    def _g_select(self, x, select):

        self.gist_functional(x)
        self._f_process(x)
        if self._calc_diff(select):
            return self.grad
        else:
            raise ValueError("Something went wrong during evaulation of objective function.")


    def gradient(self, x):

        if self.anal_grad:
            grad = self._g_select(x, self.select)

            if self.decomp:
                return np.concatenate((grad[0], grad[1]))
            else:
                return self.grad[0]

        else:
            return pygmo.estimate_gradient(callable=self.fitness,
                                            x=x, 
                                            dx=self.dx) 


    def _calc_diff(self, select):

        if self.decomp:

            if self.decomp_E:
                _dh = np.copy(self._f[:,0])
                _ds = np.copy(self._f[:,1])
                self._f[:,0] = _dh + _ds
                self._f[:,1] = _dh

                if self.anal_grad:
                    _dh_g = np.copy(self._g[:,:,0])
                    _ds_g = np.copy(self._g[:,:,1])
                    self._g[:,:,0] = _dh_g + _ds_g
                    self._g[:,:,1] = _dh_g

            elif self.decomp_S:
                _dh = np.copy(self._f[:,0])
                _ds = np.copy(self._f[:,1])
                self._f[:,0] = _dh + _ds
                self._f[:,1] = _ds

                if self.anal_grad:
                    _dh_g = np.copy(self._g[:,:,0])
                    _ds_g = np.copy(self._g[:,:,1])
                    self._g[:,:,0] = _dh_g + _ds_g
                    self._g[:,:,1] = _ds_g

            try:
                self.diff2 = [self._f[select,0]-self._exp_data[select,0],\
                              self._f[select,1]-self._exp_data[select,1]]

                if self.anal_grad:
                    self.grad = [2.*np.einsum('i,ij->j', self.diff2[0], self._g[select,:,0]),\
                                 2.*np.einsum('i,ij->j', self.diff2[1], self._g[select,:,1])]
                    if self.anal_boundary:
                        self.grad[0] += self._restraint_grad
                        self.grad[1] += self._restraint_grad

                self.diff2[0] = np.sum(self.diff2[0]**2)
                self.diff2[1] = np.sum(self.diff2[1]**2)

                if self.anal_boundary:
                    self.diff2[0] += self._restraint
                    self.diff2[1] += self._restraint

                return 1

            except:
                return 0

        else:

            try:
                self.diff2 = [self._f[select]-self._exp_data[select]]

                if self.anal_grad:
                    self.grad = [2.*np.einsum('i,ij->j', self.diff2[0], self._g[select])]
                    if self.anal_boundary:
                        self.grad[0] += self._restraint_grad

                self.diff2[0] = np.sum(self.diff2[0]**2)

                if self.anal_boundary:
                    self.diff2[0] += self._restraint

                return 1

            except:
                return 0


    def has_gradient(self):

        return True


    def get_bounds(self):

        return self.xmin, self.xmax


    def get_nobj(self):

        if self.decomp:
            return 2
        else:
            return 1


    def optimize(self, niter=500, minimizer_kwargs=None, nmin=1000, 
                kforce=100., gradient=False, print_fun=None, popsize=50,
                stepsize=0.05, optimizer="evolution", seed=None):

        self.kforce = kforce
        
        if type(seed)==type(None):
            seed = np.random.randint(999999)
        else:
            seed = int(seed)
        np.random.seed(seed)
        pygmo.set_global_rng_seed(seed=seed)
        self.set_x0()

        bounds     = gist_bounds(self.xmin, self.xmax, Tconst=True)
        min_bounds = bounds.get_bounds_for_minimizer()

        if optimizer=="evolution":
            ### This works , because pygmo makes deepcopies of this object
            ### in order to remain "thread safe" during all following operations
            prob = pygmo.problem(self)
            if self.decomp:
                if (popsize%4)!=0:
                    popsize = (popsize/4)*4
                if popsize<5:
                    popsize=8

            pop  = pygmo.population(prob=prob, size=popsize)
            if self.decomp:
                ### For NSGA2, popsize must be >4 and also
                ### a multiple of four.
                algo = pygmo.algorithm(pygmo.nsga2(gen=niter))
                #algo = pygmo.algorithm(pygmo.moead(gen=niter))
            else:
                algo = pygmo.algorithm(pygmo.sade(gen=niter))
            if self.verbose:
                algo.set_verbosity(1)
            pop  = algo.evolve(pop)

            for x in pop.get_x():
                print_fun(x)
                print_fun.flush()

        elif optimizer=="brute":
            self.anal_grad     = False
            self.anal_boundary = False
            N_dim              = self._x0.size
            niter_count        = np.zeros(self._x0.size, dtype=int)

            for i in range(self._x0.size):
                self._x0[i]    = min_bounds[i][0]
                _diff          = min_bounds[i][1]-min_bounds[i][0]
                niter_count[i] = int(_diff/stepsize)

            prop  = propagator(self._x0, N_dim, stepsize, niter_count)
            stop  = False
            _stop = False

            if nmin>0:
                self.anal_grad = True
                self.anal_boundary = False
                prob = pygmo.problem(self)
                pop  = pygmo.population(prob=prob, size=1)
                algo = pygmo.algorithm(pygmo.nlopt("slsqp"))
                algo.maxeval = nmin
                if self.verbose:
                    algo.set_verbosity(1)

            while (not stop):
                if nmin>0:
                    self.anal_grad = gradient

                    if self.anal_boundary:
                        min_bounds  = None
                        bounds      = None

                    pop.set_x(0, self._x0)
                    pop  = algo.evolve(pop)
                    x    = pop.get_x()[0]

                else:
                    x = self._x0

                if print_fun != None:
                    print_fun(x)
                    print_fun.flush()

                ### propagate self._x0
                prop.add()
                if _stop:
                    stop=True
                _stop = prop.are_we_done()

        elif optimizer=="basinhopping":
            prob = pygmo.problem(self)
            pop  = pygmo.population(prob=prob, size=popsize)
            algo = pygmo.algorithm(uda = pygmo.mbh(pygmo.nlopt("slsqp"), 
                                                   stop = 100,
                                                   perturb = self.steps*0.1))
            if self.verbose:
                algo.set_verbosity(1)
            pop  = algo.evolve(pop)

            for x in pop.get_x():
                print_fun(x)
                print_fun.flush()

        else:
            raise ValueError("Optimizer %s is not known." %optimizer)