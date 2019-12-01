import numpy as np
#sasa_grid is depracted
#from gips.grid_solvent.spatial import sasa_grid
from gips.grid_solvent.spatial import sasa_softgrid

from gips import FLOAT
from gips import DOUBLE

class gdat_fit_lib(object):

    def __init__(self, gdatarec_dict,
                        gdata_dict,
                        ref_energy=-11.108,
                        mode=0,
                        radiusadd=[0.,3.],
                        softness=1.,
                        softcut=2.,
                        exclude=None,
                        scaling=2.,
                        verbose=False):

        self.gdatarec_dict = gdatarec_dict
        self.gdata_dict    = gdata_dict
        self.ref_energy    = ref_energy
        self.mode          = mode
        self.radiusadd     = radiusadd
        self.softness      = softness
        self.softcut       = softcut
        self.scaling       = scaling
        self.verbose       = verbose
        if type(exclude) == type(None):
            self.exclude = list()
        else:
            self.exclude = exclude


    def load_metadata(self):

        self.N_rec  = 0
        self.N_enz  = 0
        self.N_pos  = 0
        self.N_cplx = 0
        self.N_lig  = 0
        self.N_case = 0

        self.ref_dict = dict()
        self.name     = list()

        self.maxdim   = np.zeros(3, dtype=int)

        majcount = 1
        ### Count all receptor data objects
        for rec_keys in self.gdatarec_dict.keys():
            mincount = 1
            recdict  = self.gdatarec_dict[rec_keys]
            for i in range(len(recdict["receptor"])):
                if recdict["receptor"][i]["gdat"] == None:
                    continue
                for j in range(3):
                    if recdict["receptor"][i]["gdat"].bins[j]>self.maxdim[j]:
                        self.maxdim[j] = recdict["receptor"][i]["gdat"].bins[j]
                self.ref_dict["%d.%d" %(majcount, mincount)] = self.N_rec
            
                mincount   += 1
                self.N_rec += 1

            self.N_enz += 1
            majcount   += 1

        ### Count all pose data objects
        for pos_keys in self.gdata_dict.keys():
            posdict = self.gdata_dict[pos_keys]
            if posdict["title"] in self.exclude:
                continue
            self.N_case += 1
            self.name.append(posdict["title"])
            for i in range(len(posdict["pose"])):
                if posdict["pose"][i]["pmd"] == None:
                    continue
                self.N_pos += 1

        ### Count all complex data objects
        for pos_keys in self.gdata_dict.keys():
            posdict = self.gdata_dict[pos_keys]
            if posdict["title"] in self.exclude:
                continue
            if posdict["complex"] != None:
                for i in range(len(posdict["complex"])):
                    if posdict["complex"][i]["gdat"] == None:
                        continue    
                    self.N_cplx += 1
                    for j in range(3):
                        if posdict["complex"][i]["gdat"].bins[j]>self.maxdim[j]:
                            self.maxdim[j] = posdict["complex"][i]["gdat"].bins[j]

        ### Count all ligand data objects
        for pos_keys in self.gdata_dict.keys():
            posdict = self.gdata_dict[pos_keys]
            if posdict["title"] in self.exclude:
                continue
            if posdict["ligand"] != None:
                for i in range(len(posdict["ligand"])):
                    if posdict["ligand"][i]["gdat"] == None:
                        continue    
                    self.N_lig += 1
                    for j in range(3):
                        if posdict["ligand"][i]["gdat"].bins[j]>self.maxdim[j]:
                            self.maxdim[j] = posdict["ligand"][i]["gdat"].bins[j]

        if self.verbose:
            print "Maxium bin dimensions found %s ..." %self.maxdim
            print "Number of receptor found %d ..." %self.N_rec
            print "Number of poses found %d ..." %self.N_pos
            print "Number of cases found %d ..." %self.N_case
            print "Number of complexes found %d ..." %self.N_cplx
            print "Number of ligands found %d ..." %self.N_lig


    def prepare_gdat(self):

        self.load_metadata()

        ###
        ### This first if statement is for receptor gist data 
        ### preparation only.
        ###
        ### mode 0: Displacement
        ### mode 1: Displacement with energy-entropy decompostion
        if self.mode in [0,1]:

            self.gdat = list()
            self.pdat = list()

            self.E   = np.zeros((self.N_rec, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
            self.S   = np.zeros((self.N_rec, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
            self.g   = np.zeros((self.N_rec, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
            self.w   = np.zeros(self.N_rec, dtype=DOUBLE)
            self.vol = np.zeros((self.N_pos, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
            self.ind_rec  = np.zeros(self.N_pos, dtype=np.int32)
            self.ind_case = np.zeros(self.N_pos, dtype=np.int32)
            self.dg  = np.zeros(self.N_case, dtype=DOUBLE)
            self.dh  = np.zeros(self.N_case, dtype=DOUBLE)
            self.ds  = np.zeros(self.N_case, dtype=DOUBLE)

            i_ref=0
            i_pose=0
            for rec_keys in self.gdatarec_dict.keys():
                recdict = self.gdatarec_dict[rec_keys]
                w_sum   = 0.
                w_list  = list()
                for i in range(len(recdict["receptor"])):
                    if recdict["receptor"][i]["gdat"] == None:
                        continue

                    self.gdat.append(recdict["receptor"][i]["gdat"])
                    bins = self.gdat[-1].bins
                
                    self.E[i_ref,:bins[0],:bins[1],:bins[2]] = self.gdat[-1].Esw_norm + \
                                                            self.scaling*(self.gdat[-1].Eww_norm_unref - self.ref_energy)
                    self.S[i_ref,:bins[0],:bins[1],:bins[2]] = - self.gdat[-1].dTStrans_norm - \
                                                                self.gdat[-1].dTSorient_norm
                    self.g[i_ref,:bins[0],:bins[1],:bins[2]] = self.gdat[-1].gO

                    self.w[i_ref] = recdict["receptor"][i]["w"]
                    w_sum        += self.w[i_ref]
                    w_list.append(i_ref)

                    i_ref += 1
                    
                for i in w_list:
                    self.w[i] /= w_sum

#            print self.w
#            self.w /= np.sum(self.w)

            i_pose=0
            i_case=0
            for pos_keys in self.gdata_dict.keys():
                posdict = self.gdata_dict[pos_keys]
                if posdict["title"] in self.exclude:
                    continue
                self.dg[i_case] = posdict["dg"]
                self.dh[i_case] = posdict["dh"]
                self.ds[i_case] = posdict["ds"]
                for i in range(len(posdict["pose"])):
                    if posdict["pose"][i]["pmd"] == None:
                        continue

                    self.pdat.append(posdict["pose"][i]["pmd"])

                    if posdict["pose"][i]["ref"] not in self.ref_dict.keys():
                        raise IOError("Cannot find receptor reference %s" %posdict["pose"][i]["ref"] )
                    ref_id = self.ref_dict[posdict["pose"][i]["ref"]]
                    self.ind_rec[i_pose]  = ref_id
                    self.ind_case[i_pose] = i_case

                    radius_list = list()
                    crds_list   = list()
                    j = 0
                    for a in self.pdat[-1]:
                        if not (a.name.startswith("h") \
                            or a.name.startswith("H")):
                            radius_list.append(a.rmin)
                            crds_list.append(self.pdat[-1].coordinates[j])
                        j += 1

                    bins = self.gdat[ref_id].bins
                    #self.vol[i_pose] = -1.
                    self.vol[i_pose,:bins[0],:bins[1],:bins[2]] = sasa_softgrid(crds_list,
                                                                                self.gdat[ref_id],
                                                                                radius_list,
                                                                                radiusadd=self.radiusadd[0],
                                                                                solvent=1.4,
                                                                                softness=self.softness,
                                                                                cutoff=self.softcut,
                                                                                verbose=self.verbose).astype(DOUBLE)
                    i_pose += 1
                i_case += 1

        ###
        ### This if statement is for novel gist calculations
        ### that incorporate receptor, complex and ligand
        ### contributions.
        ###

        elif self.mode in [3,4,5,6,7]:

            ### ------------------ ###
            ### Load Receptor Data ###
            ### ------------------ ###

            self.gdat = list()
            self.pdat = list()

            self.E   = np.zeros((self.N_rec, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
            self.S   = np.zeros((self.N_rec, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
            self.g   = np.zeros((self.N_rec, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
            self.w   = np.zeros(self.N_rec, dtype=DOUBLE)
            self.vol = np.zeros((self.N_pos, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
            self.ind_rec  = np.zeros(self.N_pos, dtype=np.int32)
            self.ind_case = np.zeros(self.N_pos, dtype=np.int32)
            self.dg  = np.zeros(self.N_case, dtype=DOUBLE)
            self.dh  = np.zeros(self.N_case, dtype=DOUBLE)
            self.ds  = np.zeros(self.N_case, dtype=DOUBLE)

            i_ref=0
            for rec_keys in self.gdatarec_dict.keys():
                recdict = self.gdatarec_dict[rec_keys]
                w_sum   = 0.
                w_list  = list()
                for i in range(len(recdict["receptor"])):
                    if recdict["receptor"][i]["gdat"] == None:
                        continue

                    self.gdat.append(recdict["receptor"][i]["gdat"])
                    bins     = self.gdat[-1].bins
                
                    self.E[i_ref,:bins[0],:bins[1],:bins[2]] = self.gdat[-1].Esw_norm + \
                                                            self.scaling*(self.gdat[-1].Eww_norm_unref - self.ref_energy)
                    self.S[i_ref,:bins[0],:bins[1],:bins[2]] = - self.gdat[-1].dTStrans_norm - \
                                                                self.gdat[-1].dTSorient_norm
                    self.g[i_ref,:bins[0],:bins[1],:bins[2]] = self.gdat[-1].gO

                    self.w[i_ref] = recdict["receptor"][i]["w"]
                    w_sum        += self.w[i_ref]
                    w_list.append(i_ref)

                    i_ref += 1
                    
                for i in w_list:
                    self.w[i] /= w_sum

#            self.w /= np.sum(self.w)

            i_pose=0
            i_case=0
            for pos_keys in self.gdata_dict.keys():
                posdict = self.gdata_dict[pos_keys]
                if posdict["title"] in self.exclude:
                    continue
                self.dg[i_case] = posdict["dg"]
                self.dh[i_case] = posdict["dh"]
                self.ds[i_case] = posdict["ds"]
                for i in range(len(posdict["pose"])):
                    if posdict["pose"][i]["pmd"] == None:
                        continue

                    self.pdat.append(posdict["pose"][i]["pmd"])

                    if posdict["pose"][i]["ref"] not in self.ref_dict.keys():
                        raise IOError("Cannot find receptor reference %s" %posdict["pose"][i]["ref"] )
                    ref_id = self.ref_dict[posdict["pose"][i]["ref"]]
                    self.ind_rec[i_pose]  = ref_id
                    self.ind_case[i_pose] = i_case

                    radius_list = list()
                    crds_list   = list()
                    j = 0
                    for a in self.pdat[-1]:
                        if not (a.name.startswith("h") \
                            or a.name.startswith("H")):
                            radius_list.append(a.rmin)
                            crds_list.append(self.pdat[-1].coordinates[j])
                        j += 1

                    bins = self.gdat[ref_id].bins
                    #self.vol[i_pose] = -1.
                    self.vol[i_pose,:bins[0],:bins[1],:bins[2]] = sasa_softgrid(crds_list,
                                                                                self.gdat[ref_id],
                                                                                radius_list,
                                                                                radiusadd=self.radiusadd[0],
                                                                                solvent=1.4,
                                                                                softness=self.softness,
                                                                                cutoff=self.softcut,
                                                                                verbose=self.verbose).astype(DOUBLE)
                    i_pose += 1
                i_case += 1

            ### ----------------- ###
            ### Load Complex Data ###
            ### ----------------- ###

            self.gdat_cplx = list()
            self.pdat_cplx = list()

            self.E_cplx   = np.zeros((self.N_cplx, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
            self.S_cplx   = np.zeros((self.N_cplx, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
            self.g_cplx   = np.zeros((self.N_cplx, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
            self.w_cplx   = np.zeros(self.N_cplx, dtype=DOUBLE)
            self.vol_cplx = np.zeros((self.N_cplx, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
            self.ind_case_cplx = np.zeros(self.N_cplx, dtype=np.int32)
            self.ind_rec_cplx  = np.arange(self.N_cplx, dtype=np.int32)

            i_cplx=0
            i_case=0
            for cplx_keys in self.gdata_dict.keys():
                cplxdict = self.gdata_dict[cplx_keys]
                if cplxdict["title"] in self.exclude:
                    continue
                _w_sum   = 0.
                _N_dict  = len(cplxdict["complex"])
                for i in range(_N_dict):
                    if cplxdict["complex"][i]["gdat"] == None:
                        continue

                    self.pdat_cplx.append(cplxdict["complex"][i]["pmd"])

                    self.gdat_cplx.append(cplxdict["complex"][i]["gdat"])
                    bins     = self.gdat_cplx[-1].bins
                
                    self.E_cplx[i_cplx,:bins[0],:bins[1],:bins[2]] = self.gdat_cplx[-1].Esw_norm + \
                                                                    self.scaling*(self.gdat_cplx[-1].Eww_norm_unref - self.ref_energy)
                    self.S_cplx[i_cplx,:bins[0],:bins[1],:bins[2]] = - self.gdat_cplx[-1].dTStrans_norm - \
                                                                    self.gdat_cplx[-1].dTSorient_norm
                    self.g_cplx[i_cplx,:bins[0],:bins[1],:bins[2]] = self.gdat_cplx[-1].gO

                    self.w_cplx[i_cplx] = cplxdict["complex"][i]["w"]
                    _w_sum += self.w_cplx[i_cplx]

                    self.ind_case_cplx[i_cplx] = i_case

                    radius_list = list()
                    crds_list   = list()
                    j = 0
                    for a in self.pdat_cplx[-1]:
                        if not (a.name.startswith("h") \
                            or a.name.startswith("H")):
                            radius_list.append(a.rmin)
                            crds_list.append(self.pdat_cplx[-1].coordinates[j])
                        j += 1

                    #self.vol_cplx[i_cplx] = -1.
                    self.vol_cplx[i_cplx,:bins[0],:bins[1],:bins[2]] = sasa_softgrid(crds_list,
                                                                                    self.gdat_cplx[-1],
                                                                                    radius_list,
                                                                                    radiusadd=self.radiusadd[1],
                                                                                    solvent=1.4,
                                                                                    softness=self.softness,
                                                                                    cutoff=self.softcut,
                                                                                    verbose=self.verbose).astype(DOUBLE)

                    #print self.E_cplx[i_cplx].max(), self.S_cplx[i_cplx].max(), self.g_cplx[i_cplx].max()

                    i_cplx += 1
                i_case += 1
                self.w_cplx[(i_cplx-_N_dict):i_cplx] /= _w_sum

            ### ---------------- ###
            ### Load Ligand Data ###
            ### ---------------- ###

            self.gdat_lig = list()
            self.pdat_lig = list()

            self.E_lig   = np.zeros((self.N_lig, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
            self.S_lig   = np.zeros((self.N_lig, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
            self.g_lig   = np.zeros((self.N_lig, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
            self.w_lig   = np.zeros(self.N_lig, dtype=DOUBLE)
            self.vol_lig = np.zeros((self.N_lig, self.maxdim[0], self.maxdim[1], self.maxdim[2]), dtype=DOUBLE)
            self.ind_case_lig = np.zeros(self.N_lig, dtype=np.int32)
            self.ind_rec_lig  = np.arange(self.N_lig, dtype=np.int32)

            i_lig=0
            i_case=0
            for lig_keys in self.gdata_dict.keys():
                ligdict = self.gdata_dict[lig_keys]
                if ligdict["title"] in self.exclude:
                    continue
                _w_sum   = 0.
                _N_dict  = len(ligdict["ligand"])
                for i in range(_N_dict):
                    if ligdict["ligand"][i]["gdat"] == None:
                        continue

                    self.pdat_lig.append(ligdict["ligand"][i]["pmd"])

                    self.gdat_lig.append(ligdict["ligand"][i]["gdat"])
                    bins     = self.gdat_lig[-1].bins

                    self.E_lig[i_lig,:bins[0],:bins[1],:bins[2]] = self.gdat_lig[-1].Esw_norm + \
                                                                self.scaling*(self.gdat_lig[-1].Eww_norm_unref - self.ref_energy)
                    self.S_lig[i_lig,:bins[0],:bins[1],:bins[2]] = - self.gdat_lig[-1].dTStrans_norm - \
                                                                self.gdat_lig[-1].dTSorient_norm
                    self.g_lig[i_lig,:bins[0],:bins[1],:bins[2]] = self.gdat_lig[-1].gO

                    self.w_lig[i_lig] = ligdict["ligand"][i]["w"]
                    _w_sum += self.w_lig[i_lig]

                    self.ind_case_lig[i_lig] = i_case

                    radius_list = list()
                    crds_list   = list()
                    j = 0
                    for a in self.pdat_lig[-1]:
                        if not (a.name.startswith("h") \
                            or a.name.startswith("H")):
                            radius_list.append(a.rmin)
                            crds_list.append(self.pdat_lig[-1].coordinates[j])
                        j += 1

                    #self.vol_lig[i_lig] = -1.
                    self.vol_lig[i_lig,:bins[0],:bins[1],:bins[2]] = sasa_softgrid(crds_list,
                                                                                    self.gdat_lig[-1],
                                                                                    radius_list,
                                                                                    radiusadd=self.radiusadd[1],
                                                                                    solvent=1.4,
                                                                                    softness=self.softness,
                                                                                    cutoff=self.softcut,
                                                                                    verbose=self.verbose).astype(DOUBLE)

                    #print self.E_lig[i_lig].max(), self.S_lig[i_lig].max(), self.g_lig[i_lig].max()

                    i_lig += 1
                i_case += 1
                self.w_lig[(i_lig-_N_dict):i_lig] /= _w_sum

        else:
            raise IOError("self.Mode %s not understood." %self.mode)
