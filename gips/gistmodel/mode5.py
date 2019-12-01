import numpy as np

from gips.gistmodel.fitting import MC_fitter

from gips.gistmodel._numerical_ext import gist_functional_6p_ext
from gips.gistmodel._numerical_ext import gist_functional_5p_ext
from gips.gistmodel._numerical_ext import gist_functional_4p_ext
from gips.gistmodel._numerical_ext import gist_restraint_ext
from gips.gistmodel._numerical_ext import merge_casedata_ext
from gips.gistmodel._numerical_ext import pair_difference_ext
from gips.utils.misc import parms_error

from gips import FLOAT
from gips import DOUBLE

MODE=5

class mode5(MC_fitter):

    def __init__(self, gdatarec_dict,
                    gdata_dict,
                    ref_energy=-11.108,
                    parms=6,
                    pairs=False,
                    radiusadd=[0.,3.],
                    softness=1.0,
                    softcut=2.0,
                    boundsdict=None,
                    pairlist=None,
                    exclude=None,
                    scaling=2.0,
                    select=None,
                    decomp_E=False,
                    decomp_S=False,
                    verbose=False):

        super(mode5, self).__init__(gdatarec_dict=gdatarec_dict,
                                    gdata_dict=gdata_dict,
                                    ref_energy=ref_energy,
                                    mode=MODE,
                                    radiusadd=radiusadd,
                                    softness=softness,
                                    softcut=softcut,
                                    exclude=exclude,
                                    scaling=scaling,
                                    verbose=verbose)

        self.pairs                = pairs
        self.parms                = parms
        self._gist_functional_ext = None
        self._parms               = parms+2
        self.boundsdict           = boundsdict
        self.pairlist             = pairlist

        if self.pairs:
            self._parms = parms+1
            self.set_pairs()

        self.set_selection(select)
        self.set_functional()
        self.set_bounds()
        self.set_step()
        self.set_x0()

        self.w      = self.w.astype(DOUBLE)
        self.w_cplx = self.w_cplx.astype(DOUBLE)
        self.w_lig  = self.w_lig.astype(DOUBLE)


    def gist_functional(self, x):

        ### &PyArray_Type, &E,
        ### &PyArray_Type, &S,
        ### &PyArray_Type, &g,
        ### &PyArray_Type, &vol,
        ### &PyArray_Type, &ind,
        ### &PyArray_Type, &x,
        ### &PyArray_Type, &dx,
        ### &PyArray_Type, &fun,
        ### &PyArray_Type, &grad,
        ### &verbose

        ### without pairs:
        ### --------------
        ### x[0] = E_aff
        ### x[1] = e_co
        ### x[2] = S_aff
        ### x[3] = s_co
        ### x[4] = g_co (Rec)
        ### x[5] = g_co (Cplx)
        ### x[6] = g_co (Lig)
        ### x[7] = C

        ### with pairs:
        ### -----------
        ### x[0] = E_aff
        ### x[1] = e_co
        ### x[2] = S_aff
        ### x[3] = s_co
        ### x[4] = g_co (Cplx)
        ### x[5] = g_co (Lig)
        ### x[6] = C

        if self.pairs:
            _xc      = np.zeros(self.parms, dtype=DOUBLE)
            _xc[:-2] = x[:-3]
            _xc[-2]  = x[-3]
            _xl      = np.zeros(self.parms, dtype=DOUBLE)
            _xl[:-2] = x[:-3]
            _xl[-2]  = x[-2]
        else:
            _xr      = np.zeros(self.parms, dtype=DOUBLE)
            _xr[:-2] = x[:-4]
            _xr[-2]  = x[-4]
            _xc      = np.zeros(self.parms, dtype=DOUBLE)
            _xc[:-2] = x[:-4]
            _xc[-2]  = x[-3]
            _xl      = np.zeros(self.parms, dtype=DOUBLE)
            _xl[:-2] = x[:-4]
            _xl[-2]  = x[-2]

        ### Make sure all types are DOUBLE

        ### Calculate Receptor ###
        ### ------------------ ###
        if not self.pairs:
            if not self._gist_functional_ext(self.E, self.S, self.g, self.vol,
                                            self.ind_rec, _xr, self._dx, self._calc_data,
                                            self._gradients, 0, int(self.anal_grad)):
                raise ValueError("Something went wrong in gist functional calculation.")

        ### Calculate Complex ###
        ### ----------------- ###
        if not self._gist_functional_ext(self.E_cplx, self.S_cplx, self.g_cplx, self.vol_cplx,
                                        self.ind_rec_cplx, _xc, self._dx, self._calc_data_cplx,
                                        self._gradients_cplx, 0, int(self.anal_grad)):
            raise ValueError("Something went wrong in gist functional calculation.")

        ### Calculate Ligand ###
        ### ---------------- ###
        if not self._gist_functional_ext(self.E_lig, self.S_lig, self.g_lig, self.vol_lig,
                                        self.ind_rec_lig, _xl, self._dx, self._calc_data_lig,
                                        self._gradients_lig, 0, int(self.anal_grad)):
            raise ValueError("Something went wrong in gist functional calculation.")

        ### &PyArray_Type, &x,
        ### &PyArray_Type, &xmin,
        ### &PyArray_Type, &xmax,
        ### &k,
        ### &restraint,
        ### &PyArray_Type, &restraint_grad

        if self.anal_boundary:
            self._restraint = gist_restraint_ext(x,
                                                self.xmin,
                                                self.xmax,
                                                self.kforce_f,
                                                self.kforce,
                                                self._restraint_grad)


    def _f_process(self, x):

        __doc___= """
        returns the squared sum of residuals
        objective function is the free energy
        """

        self.gist_functional(x)

        ### x[0] = E_aff
        ### x[1] = e_co
        ### x[2] = S_aff
        ### x[3] = s_co
        ### x[4] = g_co (Rec)
        ### x[5] = g_co (Cplx)
        ### x[6] = g_co (Lig)
        ### x[7] = C

        self._f[:] = 0.
        if self.anal_grad:
            self._g[:] = 0.
            ### Complex and Ligand contributions

            ### &PyArray_Type, &source,
            ### &PyArray_Type, &assign,
            ### &PyArray_Type, &factor,
            ### &PyArray_Type, &assign_factor

            if self.pairs:
                _f          = merge_casedata_ext(self._calc_data_cplx, self.ind_case_cplx, self.w_cplx, self.ind_case_cplx)
                _f         -= merge_casedata_ext(self._calc_data_lig,  self.ind_case_lig,  self.w_lig,  self.ind_case_lig)
                self._f[:] += pair_difference_ext(_f, self.pairidx)

                for i in range(self.parms-2):
                    _g            = merge_casedata_ext(self._gradients_cplx[:,i], self.ind_case_cplx, self.w_cplx, self.ind_case_cplx)
                    _g           -= merge_casedata_ext(self._gradients_lig[:, i], self.ind_case_lig,  self.w_lig,  self.ind_case_lig)
                    self._g[:,i] += pair_difference_ext(_g, self.pairidx)
                
                _g             = merge_casedata_ext(self._gradients_cplx[:,-2], self.ind_case_cplx, self.w_cplx, self.ind_case_cplx)
                self._g[:,-3] += pair_difference_ext(_g, self.pairidx)
                _g             = merge_casedata_ext(self._gradients_lig[:, -2], self.ind_case_lig,  self.w_lig,  self.ind_case_lig)
                self._g[:,-2] -= pair_difference_ext(_g, self.pairidx)

            else:
                self._f[:] += merge_casedata_ext(self._calc_data_cplx, self.ind_case_cplx, self.w_cplx, self.ind_case_cplx)
                self._f[:] -= merge_casedata_ext(self._calc_data, self.ind_case, self.w, self.ind_rec)
                self._f[:] -= merge_casedata_ext(self._calc_data_lig,  self.ind_case_lig,  self.w_lig,  self.ind_case_lig)

                for i in range(self.parms-2):
                    self._g[:,i] += merge_casedata_ext(self._gradients_cplx[:,i], self.ind_case_cplx, self.w_cplx, self.ind_case_cplx)
                    self._g[:,i] -= merge_casedata_ext(self._gradients[:,i], self.ind_case, self.w, self.ind_rec)
                    self._g[:,i] -= merge_casedata_ext(self._gradients_lig[:, i], self.ind_case_lig,  self.w_lig,  self.ind_case_lig)
                self._g[:,-4] -= merge_casedata_ext(self._gradients[:,-2],      self.ind_case, self.w, self.ind_rec)
                self._g[:,-3] += merge_casedata_ext(self._gradients_cplx[:,-2], self.ind_case_cplx, self.w_cplx, self.ind_case_cplx)
                self._g[:,-2] -= merge_casedata_ext(self._gradients_lig[:, -2], self.ind_case_lig,  self.w_lig,  self.ind_case_lig)

            self._f[:]    += x[-1]
            self._g[:,-1]  = 1
            
        else:

            if self.pairs:
                ### Complex and Ligand contributions
                _f  = merge_casedata_ext(self._calc_data_cplx, self.ind_case_cplx, self.w_cplx, self.ind_case_cplx)
                _f -= merge_casedata_ext(self._calc_data_lig,  self.ind_case_lig,  self.w_lig,  self.ind_case_lig)
                self._f[:] += pair_difference_ext(_f, self.pairidx)

            else:
                ### Receptor contributions
                self._f[:]  = merge_casedata_ext(self._calc_data_cplx, self.ind_case_cplx, self.w_cplx, self.ind_case_cplx)
                self._f[:] -= merge_casedata_ext(self._calc_data, self.ind_case,  self.w, self.ind_rec)
                self._f[:] -= merge_casedata_ext(self._calc_data_lig,  self.ind_case_lig,  self.w_lig,  self.ind_case_lig)

            self._f[:]     += x[-1]


    def set_bounds(self):

        __doc__ = """
        Ensures that we don't run out of bounds during MC steps.
        """

        ### x[0] = E_aff
        ### x[1] = e_co
        ### x[2] = S_aff
        ### x[3] = s_co
        ### x[4] = g_co (Rec)
        ### x[5] = g_co (Cplx)
        ### x[6] = g_co (Lig)
        ### x[7] = C

        self.xmin = np.zeros(self._parms, dtype=DOUBLE)
        self.xmax = np.zeros(self._parms, dtype=DOUBLE)

        self._restraint_grad = np.zeros(self._parms, dtype=DOUBLE)
        self._restraint      = 0.

        self.kforce_f        = np.zeros(self._parms, dtype=DOUBLE)

        _E = np.min([np.min(self.E_cplx),np.min(self.E),np.min(self.E_lig)]),\
            np.max([np.max(self.E_cplx),np.max(self.E),np.max(self.E_lig)])

        _S = np.min([np.min(self.S_cplx),np.min(self.S),np.min(self.S_lig)]),\
            np.max([np.max(self.S_cplx),np.max(self.S),np.max(self.S_lig)])

        if isinstance(self.boundsdict, dict):
            if not self.pairs:
                self.xmin[-4], self.xmax[-4] = self.boundsdict['g_co'][0], self.boundsdict['g_co'][1] ### g_co (Rec)
                self.kforce_f[-4] = 10.
            self.xmin[-3], self.xmax[-3] = self.boundsdict['g_co'][0], self.boundsdict['g_co'][1] ### g_co (Cplx)
            self.xmin[-2], self.xmax[-2] = self.boundsdict['g_co'][0], self.boundsdict['g_co'][1] ### g_co (Lig)
            self.xmin[-1], self.xmax[-1] = self.boundsdict['C'][0], self.boundsdict['C'][1] ### C
            self.kforce_f[-3] = 10.
            self.kforce_f[-2] = 10.
            self.kforce_f[-1] = 1.
        else:
            _g_r = np.min(self.g),      np.max(self.g)
            _g_c = np.min(self.g_cplx), np.max(self.g_cplx)
            _g_l = np.min(self.g_lig),  np.max(self.g_lig)
            if not self.pairs:
                self.xmin[-4], self.xmax[-4] = np.min(_g_r), np.max(_g_r) ### g_co (Rec)
                self.kforce_f[-4] = 10.
            self.xmin[-3], self.xmax[-3] = np.min(_g_c), np.max(_g_c) ### g_co (Cplx)
            self.xmin[-2], self.xmax[-2] = np.min(_g_l), np.max(_g_l) ### g_co (Lig)
            self.xmin[-1], self.xmax[-1] = -10.   , 10.     ### C
            self.kforce_f[-3] = 10.
            self.kforce_f[-2] = 10.
            self.kforce_f[-1] = 1.

        if self.parms==6:

            if isinstance(self.boundsdict, dict):
                self.xmin[0], self.xmax[0] = self.boundsdict['E'][0],    self.boundsdict['E'][1]    ### E_aff
                self.xmin[1], self.xmax[1] = self.boundsdict['e_co'][0], self.boundsdict['e_co'][1] ### e_co
                self.xmin[2], self.xmax[2] = self.boundsdict['S'][0],    self.boundsdict['S'][1]    ### S_aff
                self.xmin[3], self.xmax[3] = self.boundsdict['s_co'][0], self.boundsdict['s_co'][1] ### s_co
            else:
                self.xmin[0], self.xmax[0] = -10  , 10.   ### E_aff
                self.xmin[1], self.xmax[1] = _E[0], _E[1] ### e_co
                self.xmin[2], self.xmax[2] = -10. , 10.   ### S_aff
                self.xmin[3], self.xmax[3] = _S[0], _S[1] ### s_co
            self.kforce_f[0] = 1.
            self.kforce_f[1] = 10.
            self.kforce_f[2] = 1.
            self.kforce_f[3] = 10.

        elif self.parms==5:

            if isinstance(self.boundsdict, dict):
                self.xmin[0], self.xmax[0] = self.boundsdict['E'][0],    self.boundsdict['E'][1]    ### Aff
                self.xmin[1], self.xmax[1] = self.boundsdict['e_co'][0], self.boundsdict['e_co'][1] ### e_co
                self.xmin[2], self.xmax[2] = self.boundsdict['s_co'][0], self.boundsdict['s_co'][1] ### s_co
            else:
                self.xmin[0], self.xmax[0] = -10,  10.    ### Aff
                self.xmin[1], self.xmax[1] = _E[0], _E[1] ### e_co
                self.xmin[2], self.xmax[2] = _S[0], _S[1] ### s_co
            self.kforce_f[0] = 1.
            self.kforce_f[1] = 10.
            self.kforce_f[2] = 10.

        elif self.parms==4:

            if isinstance(self.boundsdict, dict):
                self.xmin[0], self.xmax[0] = self.boundsdict['e_co'][0], self.boundsdict['e_co'][1] ### e_co
                self.xmin[1], self.xmax[1] = self.boundsdict['s_co'][0], self.boundsdict['s_co'][1] ### s_co
            else:
                self.xmin[0], self.xmax[0] = _E[0], _E[1] ### e_co
                self.xmin[1], self.xmax[1] = _S[0], _S[1] ### s_co
            self.kforce_f[0] = 10.
            self.kforce_f[1] = 10.

        else:
            parms_error(self.parms, self._parms)


    def set_step(self):

        self.steps = np.zeros(self._parms, dtype=DOUBLE)
        if not self.pairs:
            self.steps[-4] = 2.0
        self.steps[-3] = 2.0
        self.steps[-2] = 2.0
        self.steps[-1] = 1.0

        if self.parms==6:
            self.steps[0] = 1.
            self.steps[1] = 2.0
            self.steps[2] = 1.
            self.steps[3] = 2.0

        elif self.parms==5:
            self.steps[0] = 1.
            self.steps[1] = 2.0
            self.steps[2] = 2.0

        elif self.parms==4:
            self.steps[0] = 2.0
            self.steps[1] = 2.0

        else:
            parms_error(self.parms, self._parms)


    def set_functional(self):

        ### Note, all arrays which are passed to the functionals (such as 
        ### gist_functional_6p_ext), must be DOUBLE (i.e. 32bit floating
        ### point type in C). This will not checked within the C routine 
        ### (but should be implemented at some point ...).

        if self.pairs:
            self._exp_data = pair_difference_ext(self.dg.astype(DOUBLE), self.pairidx)
            self._f        = np.zeros(self.N_pairs, dtype=DOUBLE)
            self._g        = np.zeros((self.N_pairs, self._parms), dtype=DOUBLE)
        else:
            self._exp_data = np.copy(self.dg.astype(DOUBLE))
            self._f        = np.zeros(self.N_case, dtype=DOUBLE)
            self._g        = np.zeros((self.N_case, self._parms), dtype=DOUBLE)

        self._gradients      = np.zeros((self.N_pos,  self.parms), dtype=DOUBLE)
        self._gradients_cplx = np.zeros((self.N_cplx, self.parms), dtype=DOUBLE)
        self._gradients_lig  = np.zeros((self.N_lig,  self.parms), dtype=DOUBLE)
        self._calc_data      = np.zeros(self.N_pos,  dtype=DOUBLE)
        self._calc_data_cplx = np.zeros(self.N_cplx, dtype=DOUBLE)
        self._calc_data_lig  = np.zeros(self.N_lig,  dtype=DOUBLE)
        self._dx             = 0.00000001

        if self.parms == 6:
            self._gist_functional_ext = gist_functional_6p_ext
        elif self.parms == 5:
            self._gist_functional_ext = gist_functional_5p_ext
        elif self.parms == 4:
            self._gist_functional_ext = gist_functional_4p_ext

        else:
            parms_error(self.parms, self._parms)