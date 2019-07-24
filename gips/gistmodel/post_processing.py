import numpy as np
import copy

from gips import FLOAT
from gips import DOUBLE

class post_processing(object):

    def __init__(self, fitter, x, pairs=False, prefix=None):

        self.fitter = fitter
        self.x      = x
        self.pairs  = pairs
        self.case   = 0

        score_dict  = { 4 : self.parms4,
                        5 : self.parms5,
                        6 : self.parms6
                        }

        mode_dict   = { 0 : self.mode0,
                        1 : self.mode1,
                        3 : self.mode3,
                        4 : self.mode4,
                        5 : self.mode5,
                        6 : self.mode6,
                        7 : self.mode7
                        }

        self.score   = score_dict[self.fitter.parms]
        self.process = mode_dict[self.fitter.mode]

        self.prefix = prefix
        if type(self.prefix)==type(None) \
        or self.prefix=="":
            self.prefix = ""
        else:
            self.prefix = "%s" %self.prefix

        self.set_x(self.x)
        self.set_case(0)

        self.process_rec  = False
        self.process_cplx = False
        self.process_lig  = False


    def set_x(self, x):

        self.x = copy.copy(x)

        ### Apply the solution to the scoring function
        self.fitter.gist_functional(self.x)
        self.fitter._f_process(self.x)


    def set_case(self, case):

        self.case = case
        self.name = self.fitter.name[case]


    ### |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
    ### |OVERVIEW OF THE DATA STRUCTURE IN THE FITTER OBJECT|
    ### |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
    ###
    ### Experimental data stored with gdat_fit_lib
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ### self.dg  = np.zeros(self.N_case, dtype=DOUBLE)
    ### self.dh  = np.zeros(self.N_case, dtype=DOUBLE)
    ### self.ds  = np.zeros(self.N_case, dtype=DOUBLE)
    ###
    ###
    ### GIST data generated with gdat_fit_lib (receptor)
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ### self.E   = np.zeros((self.N_rec, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
    ### self.S   = np.zeros((self.N_rec, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
    ### self.g   = np.zeros((self.N_rec, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
    ### self.w   = np.zeros(self.N_pos, dtype=DOUBLE)
    ### self.vol = np.zeros((self.N_pos, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
    ### Which pose belongs to which receptor/gistdata
    ### self.ind_rec  = np.zeros(self.N_pos, dtype=np.int32)
    ### Which pose belongs to which case
    ### self.ind_case = np.zeros(self.N_pos, dtype=np.int32)
    ###
    ###
    ### GIST data generated with gdat_fit_lib (complex)
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ### self.E_cplx   = np.zeros((self.N_cplx, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
    ### self.S_cplx   = np.zeros((self.N_cplx, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
    ### self.g_cplx   = np.zeros((self.N_cplx, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
    ### self.w_cplx   = np.zeros(self.N_cplx, dtype=DOUBLE)
    ### self.vol_cplx = np.zeros((self.N_cplx, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
    ### self.ind_rec_cplx  = np.arange(self.N_cplx, dtype=np.int32)
    ### self.ind_case_cplx = np.zeros(self.N_cplx, dtype=np.int32)
    ###
    ###
    ### GIST data generated with gdat_fit_lib (ligand)
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ### self.E_lig   = np.zeros((self.N_lig, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
    ### self.S_lig   = np.zeros((self.N_lig, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
    ### self.g_lig   = np.zeros((self.N_lig, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
    ### self.w_lig   = np.zeros(self.N_lig, dtype=DOUBLE)
    ### self.vol_lig = np.zeros((self.N_lig, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
    ### self.ind_rec_lig  = np.arange(self.N_lig, dtype=np.int32)
    ### self.ind_case_lig = np.zeros(self.N_lig, dtype=np.int32)
    ###


    def mode0(self, callback=None):

        ### The receptor
        ### ~~~~~~~~~~~~
        if self.process_rec:
            valid_poses = np.where(self.fitter.ind_case==self.case)[0]
            valid_recep = self.fitter.ind_rec[valid_poses]
            i=0
            for pose, recep in zip(valid_poses, valid_recep):

                E_grid_val, S_grid_val, gv_grid_val = self.score(self.fitter.E[recep],
                                                                self.fitter.S[recep],
                                                                self.fitter.g[recep],
                                                                self.fitter.vol[pose],
                                                                self.x)

                if callback != None:

                    kwargs = { "pose"   : pose,
                               "radius" : self.fitter.radiusadd[0],
                               "C"      : self.x[-1]
                             }

                    callback(E_grid_val,
                            S_grid_val,
                            gv_grid_val,
                            self.fitter.gdat[recep],
                            self.fitter.pdat[pose],
                            prefix="%s.%d.%s" %(self.name, i, "rec"),
                            **kwargs)

                i += 1


    def mode1(self, callback=None):

        ### The receptor
        ### ~~~~~~~~~~~~
        if self.process_rec:
            valid_poses = np.where(self.fitter.ind_case==self.case)[0]
            valid_recep = self.fitter.ind_rec[valid_poses]
            i=0
            for pose, recep in zip(valid_poses, valid_recep):

                E_grid_val, S_grid_val, gv_grid_val = self.score(self.fitter.E[recep],
                                                                self.fitter.S[recep],
                                                                self.fitter.g[recep],
                                                                self.fitter.vol[pose],
                                                                self.x)

                if callback != None:
                
                    kwargs = { "pose"   : pose,
                               "radius" : self.fitter.radiusadd[0],
                               "C"      : self.x[-2:]
                             }

                    callback(E_grid_val,
                            S_grid_val,
                            gv_grid_val,
                            self.fitter.gdat[recep],
                            self.fitter.pdat[pose],
                            prefix="%s.%d.%s" %(self.name, i, "rec"),
                            **kwargs)

                i += 1


    def mode2(self, callback=None):
        pass


    def mode3(self, callback=None):

        if not self.pairs:
            ### The receptor
            ### ~~~~~~~~~~~~
            if self.process_rec:
                valid_poses = np.where(self.fitter.ind_case==self.case)[0]
                valid_recep = self.fitter.ind_rec[valid_poses]
                i=0
                for pose, recep in zip(valid_poses, valid_recep):

                    E_grid_val, S_grid_val, gv_grid_val = self.score(self.fitter.E[recep],
                                                                    self.fitter.S[recep],
                                                                    self.fitter.g[recep],
                                                                    self.fitter.vol[pose],
                                                                    self.x)

                    if callback != None:

                        kwargs = { "pose"   : pose,
                                   "radius" : self.fitter.radiusadd[0],
                                   "C"      : self.x[-1]
                                 }

                        callback(E_grid_val,
                                S_grid_val,
                                gv_grid_val,
                                self.fitter.gdat[recep],
                                self.fitter.pdat[pose],
                                prefix="%s.%d.%s" %(self.name, i, "rec"),
                                **kwargs)

                    i += 1


        ### The complex
        ### ~~~~~~~~~~~
        if self.process_cplx:
            valid_poses_cplx = np.where(self.fitter.ind_case_cplx==self.case)[0]
            valid_recep_cplx = self.fitter.ind_rec_cplx[valid_poses_cplx]
            i=0
            for pose, recep in zip(valid_poses_cplx, valid_recep_cplx):

                E_grid_val, S_grid_val, gv_grid_val = self.score(self.fitter.E_cplx[recep],
                                                                self.fitter.S_cplx[recep],
                                                                self.fitter.g_cplx[recep],
                                                                self.fitter.vol_cplx[pose],
                                                                self.x)

                if callback != None:

                    kwargs = { "pose"   : pose,
                               "radius" : self.fitter.radiusadd[1],
                               "C"      : self.x[-1]
                             }

                    callback(E_grid_val,
                            S_grid_val,
                            gv_grid_val,
                            self.fitter.gdat_cplx[recep],
                            self.fitter.pdat_cplx[pose],
                            prefix="%s.%d.%s" %(self.name, i, "cplx"),
                            **kwargs)

                i += 1


        ### The ligand
        ### ~~~~~~~~~~
        if self.process_lig:
            valid_poses_lig = np.where(self.fitter.ind_case_lig==self.case)[0]
            valid_recep_lig = self.fitter.ind_rec_lig[valid_poses_lig]
            i=0
            for pose, recep in zip(valid_poses_lig, valid_recep_lig):

                E_grid_val, S_grid_val, gv_grid_val = self.score(self.fitter.E_lig[recep],
                                                                self.fitter.S_lig[recep],
                                                                self.fitter.g_lig[recep],
                                                                self.fitter.vol_lig[pose],
                                                                self.x)

                if callback != None:

                    kwargs = { "pose"   : pose,
                               "radius" : self.fitter.radiusadd[1],
                               "C"      : self.x[-1]
                             }

                    callback(E_grid_val,
                            S_grid_val,
                            gv_grid_val,
                            self.fitter.gdat_lig[recep],
                            self.fitter.pdat_lig[pose],
                            prefix="%s.%d.%s" %(self.name, i, "lig"),
                            **kwargs)

                i += 1


    def mode4(self, callback=None):

        if not self.pairs:
            ### The receptor
            ### ~~~~~~~~~~~~
            if self.process_rec:
                valid_poses = np.where(self.fitter.ind_case==self.case)[0]
                valid_recep = self.fitter.ind_rec[valid_poses]
                i=0
                for pose, recep in zip(valid_poses, valid_recep):

                    E_grid_val, S_grid_val, gv_grid_val = self.score(self.fitter.E[recep],
                                                                    self.fitter.S[recep],
                                                                    self.fitter.g[recep],
                                                                    self.fitter.vol[pose],
                                                                    self.x)

                    if callback != None:

                        kwargs = { "pose"   : pose,
                                   "radius" : self.fitter.radiusadd[0],
                                   "C"      : self.x[-2:]
                                 }

                        callback(E_grid_val,
                                S_grid_val,
                                gv_grid_val,
                                self.fitter.gdat[recep],
                                self.fitter.pdat[pose],
                                prefix="%s.%d.%s" %(self.name, i, "rec"),
                                **kwargs)

                    i += 1


        ### The complex
        ### ~~~~~~~~~~~
        if self.process_cplx:
            valid_poses_cplx = np.where(self.fitter.ind_case_cplx==self.case)[0]
            valid_recep_cplx = self.fitter.ind_rec_cplx[valid_poses_cplx]
            i=0
            for pose, recep in zip(valid_poses_cplx, valid_recep_cplx):

                E_grid_val, S_grid_val, gv_grid_val = self.score(self.fitter.E_cplx[recep],
                                                                self.fitter.S_cplx[recep],
                                                                self.fitter.g_cplx[recep],
                                                                self.fitter.vol_cplx[pose],
                                                                self.x)

                if callback != None:

                    kwargs = { "pose"   : pose,
                               "radius" : self.fitter.radiusadd[1],
                               "C"      : self.x[-2:]
                             }

                    callback(E_grid_val,
                            S_grid_val,
                            gv_grid_val,
                            self.fitter.gdat_cplx[recep],
                            self.fitter.pdat_cplx[pose],
                            prefix="%s.%d.%s" %(self.name, i, "cplx"),
                            **kwargs)

                i += 1


        ### The ligand
        ### ~~~~~~~~~~
        if self.process_lig:
            valid_poses_lig = np.where(self.fitter.ind_case_lig==self.case)[0]
            valid_recep_lig = self.fitter.ind_rec_lig[valid_poses_lig]
            i=0
            for pose, recep in zip(valid_poses_lig, valid_recep_lig):

                E_grid_val, S_grid_val, gv_grid_val = self.score(self.fitter.E_lig[recep],
                                                                self.fitter.S_lig[recep],
                                                                self.fitter.g_lig[recep],
                                                                self.fitter.vol_lig[pose],
                                                                self.x)

                if callback != None:

                    kwargs = { "pose"   : pose,
                               "radius" : self.fitter.radiusadd[1],
                               "C"      : self.x[-2:]
                             }

                    callback(E_grid_val,
                            S_grid_val,
                            gv_grid_val,
                            self.fitter.gdat_lig[recep],
                            self.fitter.pdat_lig[pose],
                            prefix="%s.%d.%s" %(self.name, i, "lig"),
                            **kwargs)

                i += 1

    def mode5(self, callback=None):

        if not self.pairs:
            ### The receptor
            ### ~~~~~~~~~~~~
            if self.process_rec:
                valid_poses = np.where(self.fitter.ind_case==self.case)[0]
                valid_recep = self.fitter.ind_rec[valid_poses]

                _xr      = np.zeros(self.fitter.parms, dtype=DOUBLE)
                _xr[:-2] = self.x[:-4]
                _xr[-2]  = self.x[-4]
                i=0
                for pose, recep in zip(valid_poses, valid_recep):

                    E_grid_val, S_grid_val, gv_grid_val = self.score(self.fitter.E[recep],
                                                                    self.fitter.S[recep],
                                                                    self.fitter.g[recep],
                                                                    self.fitter.vol[pose],
                                                                    _xr)

                    if callback != None:

                        kwargs = { "pose"   : pose,
                                   "radius" : self.fitter.radiusadd[0],
                                   "C"      : self.x[-1]
                                 }

                        callback(E_grid_val,
                                S_grid_val,
                                gv_grid_val,
                                self.fitter.gdat[recep],
                                self.fitter.pdat[pose],
                                prefix="%s.%d.%s" %(self.name, i, "rec"),
                                **kwargs)

                    i += 1


        ### The complex
        ### ~~~~~~~~~~~
        if self.process_cplx:
            valid_poses_cplx = np.where(self.fitter.ind_case_cplx==self.case)[0]
            valid_recep_cplx = self.fitter.ind_rec_cplx[valid_poses_cplx]

            _xc      = np.zeros(self.fitter.parms, dtype=DOUBLE)
            _xc[:-2] = self.x[:-4]
            _xc[-2]  = self.x[-3]
            i=0
            for pose, recep in zip(valid_poses_cplx, valid_recep_cplx):

                E_grid_val, S_grid_val, gv_grid_val = self.score(self.fitter.E_cplx[recep],
                                                                self.fitter.S_cplx[recep],
                                                                self.fitter.g_cplx[recep],
                                                                self.fitter.vol_cplx[pose],
                                                                _xc)

                if callback != None:

                    kwargs = { "pose"   : pose,
                               "radius" : self.fitter.radiusadd[1],
                               "C"      : self.x[-1]
                             }

                    callback(E_grid_val,
                            S_grid_val,
                            gv_grid_val,
                            self.fitter.gdat_cplx[recep],
                            self.fitter.pdat_cplx[pose],
                            prefix="%s.%d.%s" %(self.name, i, "cplx"),
                            **kwargs)

                i += 1


        ### The ligand
        ### ~~~~~~~~~~
        if self.process_lig:
            _xl      = np.zeros(self.fitter.parms, dtype=DOUBLE)
            _xl[:-2] = self.x[:-4]
            _xl[-2]  = self.x[-2]
            valid_poses_lig = np.where(self.fitter.ind_case_lig==self.case)[0]
            valid_recep_lig = self.fitter.ind_rec_lig[valid_poses_lig]
            i=0
            for pose, recep in zip(valid_poses_lig, valid_recep_lig):

                E_grid_val, S_grid_val, gv_grid_val = self.score(self.fitter.E_lig[recep],
                                                                self.fitter.S_lig[recep],
                                                                self.fitter.g_lig[recep],
                                                                self.fitter.vol_lig[pose],
                                                                _xl)

                if callback != None:

                    kwargs = { "pose"   : pose,
                               "radius" : self.fitter.radiusadd[1],
                               "C"      : self.x[-1]
                             }

                    callback(E_grid_val,
                            S_grid_val,
                            gv_grid_val,
                            self.fitter.gdat_lig[recep],
                            self.fitter.pdat_lig[pose],
                            prefix="%s.%d.%s" %(self.name, i, "lig"),
                            **kwargs)

                i += 1


    def mode6(self, callback=None):

        if not self.pairs:
            ### The receptor
            ### ~~~~~~~~~~~~
            if self.process_rec:
                valid_poses = np.where(self.fitter.ind_case==self.case)[0]
                valid_recep = self.fitter.ind_rec[valid_poses]

                _xr      = np.zeros(self.fitter.parms+1, dtype=DOUBLE)
                _xr[:-3] = self.x[:-5]
                _xr[-3]  = self.x[-5]
                i=0
                for pose, recep in zip(valid_poses, valid_recep):

                    E_grid_val, S_grid_val, gv_grid_val = self.score(self.fitter.E[recep],
                                                                    self.fitter.S[recep],
                                                                    self.fitter.g[recep],
                                                                    self.fitter.vol[pose],
                                                                    _xr)

                    if callback != None:

                        kwargs = { "pose"   : pose,
                                   "radius" : self.fitter.radiusadd[0],
                                   "C"      : _xr[-2:]
                                 }

                        callback(E_grid_val,
                                S_grid_val,
                                gv_grid_val,
                                self.fitter.gdat[recep],
                                self.fitter.pdat[pose],
                                prefix="%s.%d.%s" %(self.name, i, "rec"),
                                **kwargs)

                    i += 1


        ### The complex
        ### ~~~~~~~~~~~
        if self.process_cplx:
            valid_poses_cplx = np.where(self.fitter.ind_case_cplx==self.case)[0]
            valid_recep_cplx = self.fitter.ind_rec_cplx[valid_poses_cplx]

            _xc      = np.zeros(self.fitter.parms+1, dtype=DOUBLE)
            _xc[:-3] = self.x[:-5]
            _xc[-3]  = self.x[-4]
            i=0
            for pose, recep in zip(valid_poses_cplx, valid_recep_cplx):

                E_grid_val, S_grid_val, gv_grid_val = self.score(self.fitter.E_cplx[recep],
                                                                self.fitter.S_cplx[recep],
                                                                self.fitter.g_cplx[recep],
                                                                self.fitter.vol_cplx[pose],
                                                                _xc)

                if callback != None:

                    kwargs = { "pose"   : pose,
                               "radius" : self.fitter.radiusadd[1],
                               "C"      : _xc[-2:]
                             }

                    callback(E_grid_val,
                            S_grid_val,
                            gv_grid_val,
                            self.fitter.gdat_cplx[recep],
                            self.fitter.pdat_cplx[pose],
                            prefix="%s.%d.%s" %(self.name, i, "cplx"),
                            **kwargs)

                i += 1


        ### The ligand
        ### ~~~~~~~~~~
        if self.process_lig:
            valid_poses_lig = np.where(self.fitter.ind_case_lig==self.case)[0]
            valid_recep_lig = self.fitter.ind_rec_lig[valid_poses_lig]

            _xl      = np.zeros(self.fitter.parms+1, dtype=DOUBLE)
            _xl[:-3] = self.x[:-5]
            _xl[-3]  = self.x[-3]
            i=0
            for pose, recep in zip(valid_poses_lig, valid_recep_lig):

                E_grid_val, S_grid_val, gv_grid_val = self.score(self.fitter.E_lig[recep],
                                                                self.fitter.S_lig[recep],
                                                                self.fitter.g_lig[recep],
                                                                self.fitter.vol_lig[pose],
                                                                _xl)

                if callback != None:

                    kwargs = { "pose"   : pose,
                               "radius" : self.fitter.radiusadd[1],
                               "C"      : _xl[-2:]
                             }

                    callback(E_grid_val,
                            S_grid_val,
                            gv_grid_val,
                            self.fitter.gdat_lig[recep],
                            self.fitter.pdat_lig[pose],
                            prefix="%s.%d.%s" %(self.name, i, "lig"),
                            **kwargs)

                i += 1


    def mode7(self, callback=None):

        if self.process_rec and not self.pairs:
            _xr = np.zeros(self.fitter.parms+1, dtype=DOUBLE)
        if self.process_cplx:
            _xc = np.zeros(self.fitter.parms+1, dtype=DOUBLE)
        if self.process_lig:
            _xl = np.zeros(self.fitter.parms+1, dtype=DOUBLE)        

        ###
        ### For parms=4:
        ###
        ### with pairs:
        ### -----------
        ### x[0]  = e_co (Cplx)
        ### x[1]  = e_co (Lig)
        ### x[2]  = s_co (Cplx)
        ### x[3]  = s_co (Lig)
        ### x[4]  = g_co (Cplx)
        ### x[5]  = g_co (Lig)
        ### x[6]  = C_E
        ### x[7]  = C_S
        ###
        ### without pairs:
        ### --------------
        ### x[0]  = e_co (Rec)
        ### x[1]  = e_co (Cplx)
        ### x[2]  = e_co (Lig)
        ### x[3]  = s_co (Rec)
        ### x[4]  = s_co (Cplx)
        ### x[5]  = s_co (Lig)
        ### x[6]  = g_co (Rec)
        ### x[7]  = g_co (Cplx)
        ### x[8]  = g_co (Lig)
        ### x[9]  = C_E
        ### x[10] = C_S

        if self.fitter.parms==4:
            if self.pairs:
                if self.process_cplx:
                    _xc[:-2] = self.x[[0,2,4]]
                if self.process_lig:
                    _xl[:-2] = self.x[[1,3,5]]
            else:
                if self.process_rec:
                    _xr[:-2] = self.x[[0,3,6]]
                if self.process_cplx:
                    _xc[:-2] = self.x[[1,4,7]]
                if self.process_lig:
                    _xl[:-2] = self.x[[2,5,8]]

        ###
        ### For parms=5:
        ###
        ### with pairs:
        ### -----------
        ### x[0]  = A
        ### x[1]  = e_co (Cplx)
        ### x[2]  = e_co (Lig)
        ### x[3]  = s_co (Cplx)
        ### x[4]  = s_co (Lig)
        ### x[5]  = g_co (Cplx)
        ### x[6]  = g_co (Lig)
        ### x[7]  = C_E
        ### x[8]  = C_S
        ###
        ### without pairs:
        ### --------------
        ### x[0]  = A
        ### x[1]  = e_co (Rec)
        ### x[2]  = e_co (Cplx)
        ### x[3]  = e_co (Lig)
        ### x[4]  = s_co (Rec)
        ### x[5]  = s_co (Cplx)
        ### x[6]  = s_co (Lig)
        ### x[7]  = g_co (Rec)
        ### x[8]  = g_co (Cplx)
        ### x[9]  = g_co (Lig)
        ### x[10] = C_E
        ### x[11] = C_S

        elif self.fitter.parms==5:
            
            if self.pairs:
                if self.process_cplx:
                    _xc[:-2] = self.x[[0,1,3,5]]
                if self.process_lig:
                    _xl[:-2] = self.x[[0,2,4,6]]
            else:
                if self.process_rec:
                    _xr[:-2] = self.x[[0,1,4,7]]
                if self.process_cplx:
                    _xc[:-2] = self.x[[0,2,5,8]]
                if self.process_lig:
                    _xl[:-2] = self.x[[0,3,6,9]]

        ###
        ### For parms=6:
        ###
        ### with pairs:
        ### -----------
        ### x[0]  = E_aff
        ### x[1]  = e_co (Cplx)
        ### x[2]  = e_co (Lig)
        ### x[3]  = S_aff
        ### x[4]  = s_co (Cplx)
        ### x[5]  = s_co (Lig)
        ### x[6]  = g_co (Cplx)
        ### x[7]  = g_co (Lig)
        ### x[8]  = C_E
        ### x[9]  = C_S
        ###
        ### without pairs:
        ### --------------
        ### x[0]  = E_aff
        ### x[1]  = e_co (Rec)
        ### x[2]  = e_co (Cplx)
        ### x[3]  = e_co (Lig)
        ### x[4]  = S_aff
        ### x[5]  = s_co (Rec)
        ### x[6]  = s_co (Cplx)
        ### x[7]  = s_co (Lig)
        ### x[8]  = g_co (Rec)
        ### x[9]  = g_co (Cplx)
        ### x[10] = g_co (Lig)
        ### x[11] = C_E
        ### x[12] = C_S

        elif self.fitter.parms==6:
            if self.pairs:
                if self.process_cplx:
                    _xc[:-2] = self.x[[0,1,3,4,6]]
                if self.process_lig:
                    _xl[:-2] = self.x[[0,2,3,5,7]]
            else:
                if self.process_rec:
                    _xr[:-2] = self.x[[0,1,4,5,8]]
                if self.process_cplx:
                    _xc[:-2] = self.x[[0,2,4,6,9]]
                if self.process_lig:
                    _xl[:-2] = self.x[[0,3,4,7,10]]

        if not self.pairs:
            ### The receptor
            ### ~~~~~~~~~~~~
            if self.process_rec:
                valid_poses = np.where(self.fitter.ind_case==self.case)[0]
                valid_recep = self.fitter.ind_rec[valid_poses]
                i=0
                for pose, recep in zip(valid_poses, valid_recep):

                    E_grid_val, S_grid_val, gv_grid_val = self.score(self.fitter.E[recep],
                                                                    self.fitter.S[recep],
                                                                    self.fitter.g[recep],
                                                                    self.fitter.vol[pose],
                                                                    _xr)

                    if callback != None:

                        kwargs = { "pose"   : pose,
                                   "radius" : self.fitter.radiusadd[0],
                                   "C"      : _xr[-2:]
                                 }

                        callback(E_grid_val,
                                S_grid_val,
                                gv_grid_val,
                                self.fitter.gdat[recep],
                                self.fitter.pdat[pose],
                                prefix="%s.%d.%s" %(self.name, i, "rec"),
                                **kwargs)

                    i += 1


        ### The complex
        ### ~~~~~~~~~~~
        if self.process_cplx:
            valid_poses_cplx = np.where(self.fitter.ind_case_cplx==self.case)[0]
            valid_recep_cplx = self.fitter.ind_rec_cplx[valid_poses_cplx]
            i=0
            for pose, recep in zip(valid_poses_cplx, valid_recep_cplx):

                E_grid_val, S_grid_val, gv_grid_val = self.score(self.fitter.E_cplx[recep],
                                                                self.fitter.S_cplx[recep],
                                                                self.fitter.g_cplx[recep],
                                                                self.fitter.vol_cplx[pose],
                                                                _xc)

                if callback != None:

                    kwargs = { "pose"   : pose,
                               "radius" : self.fitter.radiusadd[1],
                               "C"      : _xc[-2:]
                             }

                    callback(E_grid_val,
                            S_grid_val,
                            gv_grid_val,
                            self.fitter.gdat_cplx[recep],
                            self.fitter.pdat_cplx[pose],
                            prefix="%s.%d.%s" %(self.name, i, "cplx"),
                            **kwargs)

                i += 1


        ### The ligand
        ### ~~~~~~~~~~
        if self.process_lig:
            valid_poses_lig = np.where(self.fitter.ind_case_lig==self.case)[0]
            valid_recep_lig = self.fitter.ind_rec_lig[valid_poses_lig]
            i=0
            for pose, recep in zip(valid_poses_lig, valid_recep_lig):

                E_grid_val, S_grid_val, gv_grid_val = self.score(self.fitter.E_lig[recep],
                                                                self.fitter.S_lig[recep],
                                                                self.fitter.g_lig[recep],
                                                                self.fitter.vol_lig[pose],
                                                                _xl)

                if callback != None:

                    kwargs = { "pose"   : pose,
                               "radius" : self.fitter.radiusadd[1],
                               "C"      : _xl[-2:]
                             }

                    callback(E_grid_val,
                            S_grid_val,
                            gv_grid_val,
                            self.fitter.gdat_lig[recep],
                            self.fitter.pdat_lig[pose],
                            prefix="%s.%d.%s" %(self.name, i, "lig"),
                            **kwargs)

                i += 1


    def parms4(self, E_grid, S_grid, g_grid, vol_grid, x):

        E = np.zeros_like(E_grid)
        S = np.zeros_like(S_grid)
        g = np.zeros_like(g_grid)

        valids_E = np.where(E_grid>x[0])
        valids_S = np.where(S_grid>x[1])
        valids_g = np.where(g_grid>x[2])

        E[valids_E] = np.copy(E_grid[valids_E])
        S[valids_S] = np.copy(S_grid[valids_S])
        g[valids_g] = np.copy(g_grid[valids_g])

        E_grid_val  = np.zeros_like(E)
        S_grid_val  = np.zeros_like(S)
        gv_grid_val = np.zeros_like(g)

        ### This is probably wrong:
        #E_grid_val[valids_g]  = E[valids_g] * vol_grid[valids_g] / g[valids_g] * 0.0332
        #S_grid_val[valids_g]  = S[valids_g] * vol_grid[valids_g] / g[valids_g] * 0.0332 * -1.
        
        ### This is how it should be:
        ### Note: 0.125 is the volume of one voxel
        E_grid_val[valids_g]  = E[valids_g] * vol_grid[valids_g] * g[valids_g] * 0.0332 * 0.125
        S_grid_val[valids_g]  = S[valids_g] * vol_grid[valids_g] * g[valids_g] * 0.0332 * 0.125

        gv_grid_val[valids_g] = vol_grid[valids_g]*g[valids_g]

        return E_grid_val, S_grid_val, gv_grid_val


    def parms5(self, E_grid, S_grid, g_grid, vol_grid, x):

        E = np.zeros_like(E_grid)
        S = np.zeros_like(S_grid)
        g = np.zeros_like(g_grid)

        E[np.where(E_grid>x[1])] = 1.
        S[np.where(S_grid>x[2])] = 1.
        g[np.where(g_grid>x[3])] = 1.

        E_grid_val  = E*g*vol_grid*x[0]
        S_grid_val  = S*g*vol_grid*x[0]
        gv_grid_val = vol_grid*g

        return E_grid_val, S_grid_val, gv_grid_val


    def parms6(self, E_grid, S_grid, g_grid, vol_grid, x):

        E = np.zeros_like(E_grid)
        S = np.zeros_like(S_grid)
        g = np.zeros_like(g_grid)

        E[np.where(E_grid>x[1])] = 1.
        S[np.where(S_grid>x[3])] = 1.
        g[np.where(g_grid>x[4])] = 1.

        E_grid_val  = E*g*vol_grid*x[0]
        S_grid_val  = S*g*vol_grid*x[2]
        gv_grid_val = vol_grid*g

        return E_grid_val, S_grid_val, gv_grid_val