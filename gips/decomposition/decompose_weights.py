import numpy as np

from gips import FLOAT
from gips import DOUBLE

from gips.grid_solvent.spatial import sasa_softgrid
from gips.gistmodel.post_processing import post_processing
#from gips.gistmodel._numerical_ext import merge_casedata_ext

class weight_fitting(post_processing):

    def __init__(self, fitter, x, pairs, frag_library, prefix=None, verbose=False):

        x[-1] = 0.
        if fitter.decomp:
            x[-2] = 0.

        super(weight_fitting, self).__init__(fitter, x, pairs, prefix)

        self.frag_library = frag_library
        self.fragmatrix   = np.zeros((2,self.frag_library.N_mol, self.frag_library.N_frag), dtype=DOUBLE)

        self.prefix  = prefix
        self.verbose = verbose


    def simple_weighting(self, E_grid_val, S_grid_val, gv_grid_val, field, pmd_object, prefix, **kwargs):

        pose   = kwargs["pose"]
        radius = kwargs["radius"]
        C      = kwargs["C"]

        N_frag = len(self.frag_library.mol2frag[pose])

        vol    = np.zeros((N_frag,
                            self.fitter.maxdim[0],
                            self.fitter.maxdim[1],
                            self.fitter.maxdim[2]), dtype=DOUBLE)

        for i in range(N_frag):
                
            frag_id     = self.frag_library.mol2frag[pose][i]
            internal_id = self.frag_library.frag2mol[frag_id].index(pose)
            frag_map    = self.frag_library.frag2mol_mapping[frag_id][internal_id]

            radius_list = list()
            crds_list   = list()
            all_Hs  = 0
            frag_Hs = 0
            isH     = False
            j  = 0
            for a in pmd_object:
                if a.name.startswith("h") \
                or a.name.startswith("H"):
                    isH = True
                if j in frag_map:
                    if isH:
                        frag_Hs += 1
                    else:
                        radius_list.append(a.rmin)
                        crds_list.append(pmd_object.coordinates[j])
                if isH:
                    all_Hs += 1
                j += 1
                isH = False

            assert (len(frag_map)-frag_Hs)==len(radius_list), "Not all atoms from frag_map matched. \
frag_id:%d, internal_id:%d, mol_id:%d, frag_map:%s" %(frag_id, internal_id, pose, frag_map)

            bins   = field.bins
            vol[i] = 0.
            vol[i,:bins[0],:bins[1],:bins[2]] = sasa_softgrid(crds_list,
                                                            field,
                                                            radius_list,
                                                            radiusadd=radius,
                                                            solvent=1.4,
                                                            softness=self.fitter.softness,
                                                            cutoff=self.fitter.softcut,
                                                            verbose=self.verbose).astype(DOUBLE)

        ### Normalize the volume
        valids  = np.where(vol>0.)
        vol_sum = np.sum(vol, axis=0)
        valids  = np.where(vol_sum>0.)

        for i in range(N_frag):

            frag_id  = self.frag_library.mol2frag[pose][i]

            E = np.sum(E_grid_val[valids] * vol[i][valids] / vol_sum[valids])
            S = np.sum(S_grid_val[valids] * vol[i][valids] / vol_sum[valids])
            #E = np.sum(E_grid_val * vol[i])
            #S = np.sum(S_grid_val * vol[i])

            self.fragmatrix[0,pose,frag_id] = E
            self.fragmatrix[1,pose,frag_id] = S


    def combine(self):

        if self.process_rec:
            valid_poses = np.where(self.fitter.ind_case==self.case)[0]
            valid_recep = self.fitter.ind_rec[valid_poses]

            return self.__combine_it(valid_poses, valid_recep, self.fitter.w)

        elif self.process_cplx:
            valid_poses_cplx = np.where(self.fitter.ind_case_cplx==self.case)[0]
            valid_recep_cplx = self.fitter.ind_rec_cplx[valid_poses_cplx]

            return self.__combine_it(valid_poses_cplx, valid_recep_cplx, self.fitter.w_cplx)

        elif self.process_lig:
            valid_poses_lig = np.where(self.fitter.ind_case_lig==self.case)[0]
            valid_recep_lig = self.fitter.ind_rec_lig[valid_poses_lig]

            return self.__combine_it(valid_poses_lig, valid_recep_lig, self.fitter.w_lig)

        return None

    
    def __combine_it(self, valid_poses, valid_recep, w):

        frag_list      = self.frag_library.mol2frag[valid_poses[0]]
        N_frag         = self.frag_library.N_frag
        frag_assign    = np.zeros(N_frag, dtype=int)
        frag_assign[:] = -1
        calc_data      = np.zeros((2, N_frag), dtype=DOUBLE)

        for pose, recep in zip(valid_poses, valid_recep):
            for frag_id in frag_list:
                frag_assign[frag_id] = frag_id
                
                calc_data[0,frag_id] += self.fragmatrix[0,pose,frag_id] * w[recep]
                calc_data[1,frag_id] += self.fragmatrix[1,pose,frag_id] * w[recep]

        if self.verbose:
            print "name: (%s) %s" %(self.prefix, self.fitter.name[self.case]),
            
            print "dG(calc): %s" %(calc_data[0]+calc_data[1]),
            print "dG(sum) : %6.3f" %(np.sum(calc_data[0]+calc_data[1])),
            for f in range(N_frag):
                print "f%d: %6.3f" %(f, calc_data[0,f]+calc_data[1,f]),
            print ""
            
            print "dH(calc): %s" %(calc_data[0]),
            print "dH(sum): %6.3f" %(np.sum(calc_data[0])),
            for f in range(N_frag):
                print "f%d: %6.3f" %(f, calc_data[0,f]),
            print ""
            
            print "dS(calc): %s" %(calc_data[1]),
            print "dS(sum): %6.3f" %(np.sum(calc_data[1])),
            for f in range(N_frag):
                print "f%d: %6.3f" %(f, calc_data[1,f]),
            print ""

        return np.copy(calc_data), np.copy(frag_assign)