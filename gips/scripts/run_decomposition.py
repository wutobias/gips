import numpy as np

import copy
import pygmo

import os

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from gips.gistmodel.mode0 import mode0
from gips.gistmodel.mode1 import mode1
from gips.gistmodel.mode3 import mode3
from gips.gistmodel.mode4 import mode4
from gips.gistmodel.mode5 import mode5
from gips.gistmodel.mode6 import mode6
from gips.gistmodel.mode7 import mode7
from gips.utils.misc import mode_error
from gips.utils.misc import aux_progs
from gips.utils.misc import are_mol_same

from gips.utils.read_write import read_pairsfile
from gips.utils.read_write import read_parmsfile

from gips.decomposition.decompose_structure import frag_library
from gips.decomposition.decompose_structure import get_frag_list
from gips.decomposition.decompose_weights import weight_fitting

from gips import FLOAT
from gips import DOUBLE


def decomposition(gdatarec_lib, gdata_lib, mode, parms=6, pairs=True, parmsfile=None, 
                frag_file=None, map_file=None, radiusadd=[0.,3.], softness=1., softcut=2., 
                pairfile=None, exclude=None, paircut=0.0, prefix=None, scaling=2.0, 
                verbose=False):

    if verbose:
        print "Start mapout procedure with"
        print "mode      = %d" %mode
        print "softness  = %6.3f" %softness
        print "softcut   = %6.3f" %softcut
        print "parmsfile = %s" %parmsfile

    if verbose:
        print "Organizing and preparing data ..."

    mode_dict = dict()
    mode_dict = {0 : mode0,
                 1 : mode1,
                 3 : mode3,
                 4 : mode4,
                 5 : mode5,
                 6 : mode6,
                 7 : mode7}

    if mode in mode_dict.keys():
        fitmode = mode_dict[mode]
    else:
        mode_error(mode)

    has_cplxlig = True
    if mode in [0,1]:
        has_cplxlig = False

    fitter = fitmode(gdatarec_lib,
                    gdata_lib,
                    parms=parms,
                    pairs=False,
                    radiusadd=radiusadd,
                    softness=softness,
                    softcut=softcut,
                    scaling=scaling,
                    verbose=verbose)

    parmdict = read_parmsfile(parmsfile)
    
    ### Find position of SES in parms file
    A_SSE = -1
    B_SSE = -1
    for i, entry in enumerate(parmdict["header"]):
        if entry.startswith("SSE"):
            if entry.endswith("(A)"):
                A_SSE=i
            elif entry.endswith("(B)"):
                B_SSE=i

    
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    ### Find the best Candidate Solutions ###
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

    ### Collect all the solution candiates
    N_entries = len(parmdict.keys())-1

    A_list = list()
    B_list = list()
    x_list = list()
    A_list_tmp = list()
    B_list_tmp = list()
    x_list_tmp = list()
    
    for key, value in parmdict.items():
        if key=="header":
            continue
        A_list_tmp.append(value[A_SSE])
        B_list_tmp.append(value[B_SSE])
        x_list_tmp.append(value[:fitter._parms])

    if fitter.decomp:
        N_entries  = N_entries/2
        for i in range(N_entries):
            A_list.append([copy.copy(A_list_tmp[2*i]), copy.copy(A_list_tmp[2*i+1])])
            B_list.append([copy.copy(B_list_tmp[2*i]), copy.copy(B_list_tmp[2*i+1])])
            x_list.append(copy.copy(x_list_tmp[2*i]))
    else:
        A_list = copy.copy(A_list_tmp)
        B_list = copy.copy(B_list_tmp)
        x_list = copy.copy(x_list_tmp)

    A_list = np.array(A_list)
    B_list = np.array(B_list)

    ### Find the best candidate solution
    if fitter.decomp:
        ndf, dl, dc, ndr = pygmo.fast_non_dominated_sorting(A_list)
        ordered_ndf      = list()
        for front in ndf:
            ordered_ndf.append(pygmo.sort_population_mo(A_list[front]))

    else:
        ordered_ndf = np.argsort(A_list, axis=0)

    if fitter.decomp:
        best_x_A = np.array(x_list[ordered_ndf[0][0]])
    else:
        best_x_A = np.array(x_list[ordered_ndf[0]])


    ### ~~~~~~~~~~~~~~~~~~~~~~ ###
    ### Prepare Exclusion List ###
    ### ~~~~~~~~~~~~~~~~~~~~~~ ###

    if exclude != None \
    and exclude != "":
        exclude_list = list()
        with open(exclude, "r") as fopen:
            for line in fopen:
                l =line.rstrip().lstrip().split()
                if len(l)==0:
                    continue
                if l[0].startswith("#"):
                    continue
                for s in l:
                    exclude_list.append(s)
    else:
        exclude_list = list()


    ### ~~~~~~~~~~~~~~~~~~~~~~~ ###
    ### Prepare Pairise Fitting ###
    ### ~~~~~~~~~~~~~~~~~~~~~~~ ###

    if pairs:
        if pairfile != None \
        and pairfile != "":
            if type(pairfile) != str:
                raise TypeError("The path to pairfile must be of type str, but is of type %s" %type(pairfile))
            pairlist = read_pairsfile(pairfile, paircut)

            pairlist_idx = list()
            for pair in pairlist:
                for i in range(fitter.N_case):
                    case1 = fitter.select[i]
                    name1 = fitter.name[case1]
                    if name1 in exclude_list:
                        continue
                    for j in range(fitter.N_case):
                        if j<=i:
                            continue
                        case2 = fitter.select[j]
                        name2 = fitter.name[case2]
                        if name2 in exclude_list:
                            continue
                        if name1==pair[0] \
                        and name2==pair[1]:
                            pairlist_idx.append([case1, case2])
                        elif name1==pair[1] \
                        and name2==pair[0]:
                            pairlist_idx.append([case2, case1])

        else:
            pairlist     = None
            pairlist_idx = list()
            for i in range(fitter.N_case):
                name1 = fitter.name[i]
                if name1 in exclude_list:
                    continue
                for j in range(fitter.N_case):
                    if j<=i:
                        continue
                    name2 = fitter.name[j]
                    if name2 in exclude_list:
                        continue
                    pairlist_idx.append([i,j])

    else:
        pairlist     = None
        pairlist_idx = None


    ### ~~~~~~~~~~~~~~~~~ ###
    ### Build the Library ###
    ### ~~~~~~~~~~~~~~~~~ ###

    has_extlib = False
    ### Check for external mapping files
    if frag_file != None \
    and frag_file != "":
        has_extlib = True
        ext_frag   = list()
        ext_frag_name = list()
        with open(frag_file, "r") as fopen:
            for line in fopen:
                l = line.rstrip().lstrip().split()
                if len(l)==0:
                    continue
                if l[0].startswith("#"):
                    continue
                ext_frag.append(Chem.MolFromSmiles(l[1]))
                ext_frag_name.append(l[0])
    else:
        ext_frag=None
        ext_frag_name=None

    if map_file != None \
    and map_file != "":
        ext_map_frag = list()
        ext_map_inds = list()
        ext_map_name = list()
        with open(map_file, "r") as fopen:
            for line in fopen:
                l = line.rstrip().lstrip().split()
                if len(l)==0:
                    continue
                if l[0].startswith("#"):
                    continue
                ext_map_name.append(l[0])
                ext_map_frag.append(list())
                ext_map_inds.append(list())
                ids_list = l[1].split(",")
                if len(ids_list)==1:
                    if ids_list[0]=="-1":
                        continue
                for i in ids_list:
                    ext_map_frag[-1].append(int(i))
                for s in l[2:]:
                    ext_map_inds[-1].append(list())
                    for i in s.split(","):
                        ext_map_inds[-1][-1].append(int(i))
    else:
        ext_map_frag=None
        ext_map_inds=None
        ext_map_name=None


    if ext_frag==None \
    and ext_map_frag!=None:
        raise IOError("Must provide both, frag_file and map_file.")

    if ext_frag!=None \
    and ext_map_frag==None:
        raise IOError("Must provide both, frag_file and map_file.")

    if has_extlib:
        mol2extmol   = list()
        #frag2extfrag = list()
        if has_cplxlig:
            mol2extmol_cplx   = list()
            #frag2extfrag_cplx = list()
            mol2extmol_lig    = list()
            #frag2extfrag_lig  = list()

    if verbose:
        "Starting fragment decomposition..."
    RAND     = np.random.randint(9999)
    frag_lib = frag_library()
    if has_cplxlig:
        frag_lib_cplx = frag_library()
        frag_lib_lig  = frag_library()
    progs    = aux_progs(verbose)
    for case in range(fitter.N_case):

        valid_poses = np.where(fitter.ind_case==case)[0]
        name        = fitter.name[case]

        for pose in valid_poses:
            pmd_instance = fitter.pdat[pose]

            pmd_instance.save("p%d.mol2" %RAND)

            args = "-i p%d.mol2 -fi mol2 -o p%d_sybyl.mol2 -fo mol2 -at sybyl -pf y -dr no" %(RAND, RAND)
            progs.call(progs.ante_exe, args)

            mol  = Chem.MolFromMol2File("p%d_sybyl.mol2" %RAND, removeHs=False)

            if verbose:
                AllChem.Compute2DCoords(mol)

            if has_extlib:
                index     = ext_map_name.index(name)
                frag_list = list()
                for frag_id in ext_map_frag[index]:
                    frag_list.append(ext_frag[frag_id])
                    ### If we have an external library with mappings
                    ### we must do the refinement manually!
                mol2extmol.append(index)
            else:
                frag_list = get_frag_list(mol)

            frag_lib.add_frag_list(frag_list, mol)

            os.remove("p%d.mol2" %RAND)
            os.remove("p%d_sybyl.mol2" %RAND)

        if has_cplxlig:

            valid_poses_cplx = np.where(fitter.ind_case_cplx==case)[0]
            valid_poses_lig  = np.where(fitter.ind_case_lig==case)[0]

            for pose in valid_poses_cplx:
                pmd_instance = fitter.pdat_cplx[pose]

                pmd_instance.save("p%d.mol2" %RAND)

                args = "-i p%d.mol2 -fi mol2 -o p%d_sybyl.mol2 -fo mol2 -at sybyl -pf y -dr no" %(RAND, RAND)
                progs.call(progs.ante_exe, args)

                mol  = Chem.MolFromMol2File("p%d_sybyl.mol2" %RAND, removeHs=False)

                if verbose:
                    AllChem.Compute2DCoords(mol)

                if has_extlib:
                    index     = ext_map_name.index(name)
                    frag_list = list()
                    for frag_id in ext_map_frag[index]:
                        frag_list.append(ext_frag[frag_id])
                        ### If we have an external library with mappings
                        ### we must do the refinement manually!
                    mol2extmol_cplx.append(index)
                else:
                    frag_list = get_frag_list(mol)

                frag_lib_cplx.add_frag_list(frag_list, mol)

                os.remove("p%d.mol2" %RAND)
                os.remove("p%d_sybyl.mol2" %RAND)

            for pose in valid_poses_lig:
                pmd_instance = fitter.pdat_lig[pose]

                pmd_instance.save("p%d.mol2" %RAND)

                args = "-i p%d.mol2 -fi mol2 -o p%d_sybyl.mol2 -fo mol2 -at sybyl -pf y -dr no" %(RAND, RAND)
                progs.call(progs.ante_exe, args)

                mol  = Chem.MolFromMol2File("p%d_sybyl.mol2" %RAND, removeHs=False)

                if verbose:
                    AllChem.Compute2DCoords(mol)

                if has_extlib:
                    index     = ext_map_name.index(name)
                    frag_list = list()
                    for frag_id in ext_map_frag[index]:
                        frag_list.append(ext_frag[frag_id])
                        ### If we have an external library with mappings
                        ### we must do the refinement manually!
                    mol2extmol_lig.append(index)
                else:
                    frag_list = get_frag_list(mol)

                frag_lib_lig.add_frag_list(frag_list, mol)

                os.remove("p%d.mol2" %RAND)
                os.remove("p%d_sybyl.mol2" %RAND)

    if has_extlib:
        for frag_id in range(frag_lib.N_frag):
            frag_lib.frag2mol_mapping.append(list())
            for mol_id in frag_lib.frag2mol[frag_id]:

                frag_id_rank = frag_lib.mol2frag[mol_id].index(frag_id)
                ext_mol_id   = mol2extmol[mol_id]

                if len(ext_map_inds[ext_mol_id])==0:
                    ### If we are here, then the molecule has no fragments.
                    ### The molecule is then treated, as if itself would
                    ### be the fragment
                    mol     = frag_lib.mol_list[mol_id]
                    matches = range(mol.GetNumAtoms())
                else:
                    matches = ext_map_inds[ext_mol_id][frag_id_rank]

                frag_lib.frag2mol_mapping[-1].append(matches)

        if has_cplxlig:
            for frag_id in range(frag_lib_cplx.N_frag):
                frag_lib_cplx.frag2mol_mapping.append(list())
                for mol_id in frag_lib_cplx.frag2mol[frag_id]:

                    frag_id_rank = frag_lib_cplx.mol2frag[mol_id].index(frag_id)
                    ext_mol_id   = mol2extmol_cplx[mol_id]

                    if len(ext_map_inds[ext_mol_id])==0:
                        ### If we are here, then the molecule has no fragments.
                        ### The molecule is then treated, as if itself would
                        ### be the fragment
                        mol     = frag_lib_cplx.mol_list[mol_id]
                        matches = range(mol.GetNumAtoms())
                    else:
                        matches = ext_map_inds[ext_mol_id][frag_id_rank]

                    frag_lib_cplx.frag2mol_mapping[-1].append(matches)

            for frag_id in range(frag_lib_lig.N_frag):
                frag_lib_lig.frag2mol_mapping.append(list())
                for mol_id in frag_lib_lig.frag2mol[frag_id]:

                    frag_id_rank = frag_lib_lig.mol2frag[mol_id].index(frag_id)
                    ext_mol_id   = mol2extmol_lig[mol_id]

                    if len(ext_map_inds[ext_mol_id])==0:
                        ### If we are here, then the molecule has no fragments.
                        ### The molecule is then treated, as if itself would
                        ### be the fragment
                        mol     = frag_lib_lig.mol_list[mol_id]
                        matches = range(mol.GetNumAtoms())
                    else:
                        matches = ext_map_inds[ext_mol_id][frag_id_rank]

                    frag_lib_lig.frag2mol_mapping[-1].append(matches)

    else:
        frag_lib.refine()

        if has_cplxlig:
            frag_lib_cplx.refine()
            frag_lib_lig.refine()

    if verbose:
        print "Poses Fragments..."
        for case in range(fitter.N_case):
            name        = fitter.name[case]
            valid_poses = np.where(fitter.ind_case==case)[0]
            print name,
            for pose in valid_poses:
                print frag_lib.mol2frag[pose],
            print ""
        frag_lib.draw("pos_")

        if has_cplxlig:
            print "Cplx Fragments..."
            for case in range(fitter.N_case):
                name        = fitter.name[case]
                valid_poses = np.where(fitter.ind_case_cplx==case)[0]
                print name,
                for pose in valid_poses:
                    print frag_lib_cplx.mol2frag[pose],
                print ""
            frag_lib_cplx.draw("cplx_")

            print "Lig Fragments..."
            for case in range(fitter.N_case):
                name        = fitter.name[case]
                valid_poses = np.where(fitter.ind_case_lig==case)[0]
                print name,
                for pose in valid_poses:
                    print frag_lib_lig.mol2frag[pose],
                print ""
            frag_lib_lig.draw("lig_")


    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    ### Calculate the Fragment weightings ###
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

    if verbose:
        print "Calculate fragment weightings..."
    ### Constructor for weight_fitting:
    ### def __init__(self, fitter, x, frag_library, prefix=None, verbose=False):
    weight = weight_fitting(fitter, best_x_A, pairs, frag_lib, "pos", verbose)
    weight.process_rec  = True
    weight.process_cplx = False
    weight.process_lig  = False
    if has_cplxlig:
        weight_cplx = weight_fitting(fitter, best_x_A, pairs, frag_lib_cplx, "cplx", verbose)
        weight_cplx.process_rec  = False
        weight_cplx.process_cplx = True
        weight_cplx.process_lig  = False
        weight_lig  = weight_fitting(fitter, best_x_A, pairs, frag_lib_lig, "lig", verbose)
        weight_lig.process_rec  = False
        weight_lig.process_cplx = False
        weight_lig.process_lig  = True

    ### Make the fragment-based decomposition of the GIST grids
    for case in range(fitter.N_case):
        weight.set_case(case)
        ### Use the internal write routine as a callback for the process routine
        weight.process(weight.simple_weighting)
        if has_cplxlig:
            weight_cplx.set_case(case)
            weight_lig.set_case(case)
            weight_cplx.process(weight_cplx.simple_weighting)
            weight_lig.process(weight_lig.simple_weighting)
    
    ### Combine the individual poses and get the final
    ### contributions of the fragments
    calc_data   = np.zeros((2, fitter.N_case, frag_lib.N_frag), dtype=DOUBLE)
    frag_assign = np.zeros((fitter.N_case, frag_lib.N_frag), dtype=int)
    frag_assign[:] = -1
    if has_cplxlig:
        calc_data_cplx      = np.zeros((2, fitter.N_case, frag_lib_cplx.N_frag), dtype=DOUBLE)
        frag_assign_cplx    = np.zeros((fitter.N_case, frag_lib_cplx.N_frag), dtype=int)
        frag_assign_cplx[:] = -1
        calc_data_lig      = np.zeros((2, fitter.N_case, frag_lib_lig.N_frag), dtype=DOUBLE)
        frag_assign_lig    = np.zeros((fitter.N_case, frag_lib_lig.N_frag), dtype=int)
        frag_assign_lig[:] = -1

    for case in range(fitter.N_case):
        weight.set_case(case)
        _data, _assign = weight.combine()
        calc_data[0,case,:] = np.copy(_data[0])
        calc_data[1,case,:] = np.copy(_data[1])
        frag_assign[case,:] = np.copy(_assign)
        if has_cplxlig:
            
            weight_cplx.set_case(case)
            _data, _assign = weight_cplx.combine()
            calc_data_cplx[0,case,:] = np.copy(_data[0])
            calc_data_cplx[1,case,:] = np.copy(_data[1])
            frag_assign_cplx[case,:] = np.copy(_assign)
            
            weight_lig.set_case(case)
            _data, _assign = weight_lig.combine()
            calc_data_lig[0,case,:] = np.copy(_data[0])
            calc_data_lig[1,case,:] = np.copy(_data[1])
            frag_assign_lig[case,:] = np.copy(_assign)


    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    ### Evaluate the Fragment Properties ###
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

    if has_cplxlig:

        case2frag_cplx = np.zeros((fitter.N_case, frag_lib_cplx.N_frag), dtype=int)
        case2frag_lig  = np.zeros((fitter.N_case, frag_lib_lig.N_frag),  dtype=int)
        case2frag_cplx[:] = -1
        case2frag_lig[:]  = -1

        for case in range(fitter.N_case):

            valids      = np.where(frag_assign[case]>-1)[0]
            valids_cplx = np.where(frag_assign_cplx[case]>-1)[0]
            valids_lig  = np.where(frag_assign_lig[case]>-1)[0]

            for frag_id in frag_assign[case,valids]:

                frag_lib.qp.makeDummiesQueries      = False
                frag_lib_cplx.qp.makeDummiesQueries = False
                frag_lib_lig.qp.makeDummiesQueries  = False

                frag = Chem.AdjustQueryProperties(frag_lib.frag_list[frag_id],\
                                                    frag_lib.qp)
                   
                for frag_id_cplx in frag_assign_cplx[case,valids_cplx]:
                    frag_cplx = Chem.AdjustQueryProperties(frag_lib_cplx.frag_list[frag_id_cplx],\
                                                            frag_lib_cplx.qp)
                    if are_mol_same(frag, frag_cplx, useChirality=True):
                        case2frag_cplx[case,frag_id_cplx] = frag_id
                        break

                for frag_id_lig in frag_assign_lig[case,valids_lig]:
                    frag_lig  = Chem.AdjustQueryProperties(frag_lib_lig.frag_list[frag_id_lig],\
                                                            frag_lib_lig.qp)
                    if are_mol_same(frag, frag_lig, useChirality=True):
                        case2frag_lig[case,frag_id_lig] = frag_id
                        break
    
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    ### Write out Decompostion Data ###
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

    if prefix==None:
        prefix=""
    
    fopen_mol = open("%sdecomposition.molecules.out" %prefix, "w")

    if pairs:

        for pair_idx in pairlist_idx:
            
            case1 = pair_idx[0]
            case2 = pair_idx[1]

            fopen_mol.write("pair %s-%s; " %(fitter.name[case1], fitter.name[case2]))
            if not has_cplxlig:
                dH1 = np.sum(calc_data[0,case1,:])
                dH2 = np.sum(calc_data[0,case2,:])
                dS1 = np.sum(calc_data[1,case1,:])
                dS2 = np.sum(calc_data[1,case2,:])
            else:
                dH1 = np.sum(calc_data_cplx[0,case1,:])-np.sum(calc_data_lig[0,case1,:])
                dH2 = np.sum(calc_data_cplx[0,case2,:])-np.sum(calc_data_lig[0,case2,:])
                dS1 = np.sum(calc_data_cplx[1,case1,:])-np.sum(calc_data_lig[1,case1,:])
                dS2 = np.sum(calc_data_cplx[1,case2,:])-np.sum(calc_data_lig[1,case2,:])

            fopen_mol.write("ddG %6.3f; " %(dH1-dH2+(dS1-dS2)))
            fopen_mol.write("ddH %6.3f; " %(dH1-dH2))
            fopen_mol.write("ddS %6.3f; " %(dS1-dS2))
            fopen_mol.write("\n")

            for case in [case1, case2]:

                valids = np.where(frag_assign[case]>-1)[0]

                for frag_id in frag_assign[case,valids]:
                    delta = np.zeros(2, dtype=DOUBLE)
                    fopen_mol.write("name %s; " %fitter.name[case])
                    fopen_mol.write("frag %d; " %frag_id)
                    if not has_cplxlig:
                        fopen_mol.write("dG(Rec) %6.3f; " %(calc_data[0,case,frag_id]+\
                                                            calc_data[1,case,frag_id]))
                        fopen_mol.write("dH(Rec) %6.3f; " %calc_data[0,case,frag_id])
                        fopen_mol.write("dS(Rec) %6.3f; " %calc_data[1,case,frag_id])
                    else:
                        frag_id_cplx = np.where(case2frag_cplx[case]==frag_id)[0]
                        frag_id_lig  = np.where(case2frag_lig[case]==frag_id)[0]

                        fopen_mol.write("dG(Cplx) %6.3f; " %(calc_data_cplx[1,case,frag_id_cplx]+\
                                                             calc_data_cplx[0,case,frag_id_cplx]))
                        fopen_mol.write("dH(Cplx) %6.3f; " %calc_data_cplx[0,case,frag_id_cplx])
                        fopen_mol.write("dS(Cplx) %6.3f; " %calc_data_cplx[1,case,frag_id_cplx])

                        fopen_mol.write("dG(Lig) %6.3f; " %(calc_data_lig[0,case,frag_id_lig]+\
                                                            calc_data_lig[1,case,frag_id_lig]))
                        fopen_mol.write("dH(Lig) %6.3f; " %calc_data_lig[0,case,frag_id_lig])
                        fopen_mol.write("dS(Lig) %6.3f; " %calc_data_lig[1,case,frag_id_lig])

                        delta[0] += calc_data_cplx[0,case,frag_id_cplx]
                        delta[0] -= calc_data_lig[0,case,frag_id_lig]

                        delta[1] += calc_data_cplx[1,case,frag_id_cplx]
                        delta[1] -= calc_data_lig[1,case,frag_id_lig]

                        fopen_mol.write("dG(Delta) %6.3f; " %(delta[0]+delta[1]))
                        fopen_mol.write("dH(Delta) %6.3f; " %delta[0])
                        fopen_mol.write("dS(Delta) %6.3f; " %delta[1])

                    fopen_mol.write("\n")

    else:

        for case in range(fitter.N_case):

            #if fitter.name[case] in exclude_list:
            #    continue

            valids = np.where(frag_assign[case]>-1)[0]

            for frag_id in frag_assign[case,valids]:
                delta = np.zeros(2, dtype=DOUBLE)

                fopen_mol.write("name %s; " %fitter.name[case])
                fopen_mol.write("frag %d; " %frag_id)
                fopen_mol.write("dG(Rec) %6.3f; " %(calc_data[0,case,frag_id]+\
                                                    calc_data[1,case,frag_id]))
                fopen_mol.write("dH(Rec) %6.3f; " %calc_data[0,case,frag_id])
                fopen_mol.write("dS(Rec) %6.3f; " %calc_data[1,case,frag_id])

                if has_cplxlig:
                    frag_id_cplx = np.where(case2frag_cplx[case]==frag_id)[0]
                    frag_id_lig  = np.where(case2frag_lig[case]==frag_id)[0]
                    fopen_mol.write("dG(Cplx) %6.3f; " %(calc_data_cplx[0,case,frag_id_cplx]+\
                                                        calc_data_cplx[1,case,frag_id_cplx]))
                    fopen_mol.write("dH(Cplx) %6.3f; " %calc_data_cplx[0,case,frag_id_cplx])
                    fopen_mol.write("dS(Cplx) %6.3f; " %calc_data_cplx[1,case,frag_id_cplx])

                    fopen_mol.write("dG(Lig) %6.3f; " %(calc_data_lig[0,case,frag_id_lig]+\
                                                        calc_data_lig[1,case,frag_id_lig]))
                    fopen_mol.write("dH(Lig) %6.3f; " %calc_data_lig[0,case,frag_id_lig])
                    fopen_mol.write("dS(Lig) %6.3f; " %calc_data_lig[1,case,frag_id_lig])

                    if fitter.decomp:
                        delta[0] += calc_data_cplx[0,case,frag_id_cplx]
                        delta[0] -= calc_data[0,case,frag_id]
                        delta[0] -= calc_data_lig[0,case,frag_id_lig]

                        delta[1] += calc_data_cplx[1,case,frag_id_cplx]
                        delta[1] -= calc_data[1,case,frag_id]
                        delta[1] -= calc_data_lig[1,case,frag_id_lig]

                        fopen_mol.write("dG(Delta) %6.3f; " %(delta[0]+delta[1]))
                        fopen_mol.write("dH(Delta) %6.3f; " %delta[0])
                        fopen_mol.write("dS(Delta) %6.3f; " %delta[1])

                fopen_mol.write("\n")

    fopen_mol.close()

    
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    ### Write out Fragment Contribution Data ###
    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

    fopen_frag = open("%sdecomposition.fragments.out" %prefix, "w")

    for frag_id in range(frag_lib.N_frag):
        valids = np.where(frag_assign==frag_id)[0]
        N_mol  = valids.shape[0]

        name   = list()
            

        ### First, gather the fragment data in each molecule
        ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if has_cplxlig:
            frag_data_delta = np.zeros((2, N_mol), dtype=DOUBLE)
            if not pairs:
                frag_data = np.zeros((2, N_mol), dtype=DOUBLE)
            frag_data_cplx  = np.zeros((2, N_mol), dtype=DOUBLE)
            frag_data_lig   = np.zeros((2, N_mol), dtype=DOUBLE)
        else:
            frag_data = np.zeros((2, N_mol), dtype=DOUBLE)

        for i, case in enumerate(valids):

            #if fitter.name[case] in exclude_list:
            #    continue

            name.append(fitter.name[case])

            if has_cplxlig:
                frag_id_cplx = np.where(case2frag_cplx[case]==frag_id)[0]
                frag_id_lig  = np.where(case2frag_lig[case]==frag_id)[0]

                frag_data_cplx[0,i]  = calc_data_cplx[0,case,frag_id_cplx]
                frag_data_lig[0,i]   = calc_data_lig[0,case,frag_id_lig]
                frag_data_delta[0,i] = frag_data_cplx[0,i]-frag_data_lig[0,i]
                frag_data_cplx[1,i]  = calc_data_cplx[1,case,frag_id_cplx]
                frag_data_lig[1,i]   = calc_data_lig[1,case,frag_id_lig]
                frag_data_delta[1,i] = frag_data_cplx[1,i]-frag_data_lig[1,i]

                if not pairs:
                    frag_data[0,i]        = calc_data[0,case,frag_id]
                    frag_data_delta[0,i] -= frag_data[0,i]
                    frag_data[1,i]        = calc_data[1,case,frag_id]
                    frag_data_delta[1,i] -= frag_data[1,i]
            else:
                frag_data[0,i] = calc_data[0,case,frag_id]
                frag_data[1,i] = calc_data[1,case,frag_id]


        ### Second, combine and write out
        ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        fopen_frag.write("frag %d; " %frag_id)
        fopen_frag.write("mol; ")
        for n in name:
            fopen_frag.write("%s; " %n)
        fopen_frag.write("\n")

        if has_cplxlig:

            ### Minimum
            ### ~~~~~~~
            fopen_frag.write("frag %d; " %frag_id)
            fopen_frag.write("min; ")
            if not pairs:
                fopen_frag.write("dG(Rec) %6.3f; "   %np.min(frag_data[0]+\
                                                                frag_data[1]))
            fopen_frag.write("dG(Cplx) %6.3f; "  %np.min(frag_data_cplx[0]+\
                                                            frag_data_cplx[1]))
            fopen_frag.write("dG(Lig) %6.3f; "   %np.min(frag_data_lig[0]+\
                                                         frag_data_lig[1]))

            if not pairs:
                fopen_frag.write("dH(Rec) %6.3f; "   %np.min(frag_data[0]))
            fopen_frag.write("dH(Cplx) %6.3f; "  %np.min(frag_data_cplx[0]))
            fopen_frag.write("dH(Lig) %6.3f; "   %np.min(frag_data_lig[0]))
                
            if not pairs:
                fopen_frag.write("dS(Rec) %6.3f; "   %np.min(frag_data[1]))
            fopen_frag.write("dS(Cplx) %6.3f; "  %np.min(frag_data_cplx[1]))
            fopen_frag.write("dS(Lig) %6.3f; "   %np.min(frag_data_lig[1]))
                
            if not pairs:
                fopen_frag.write("dG(Delta) %6.3f; " %(np.min(frag_data_delta[0]+\
                                                                frag_data_delta[1])))
            fopen_frag.write("dH(Delta) %6.3f; " %np.min(frag_data_delta[0]))
            fopen_frag.write("dS(Delta) %6.3f; " %np.min(frag_data_delta[1]))

            fopen_frag.write("\n")

            
            ### Minimum at Molecule
            ### ~~~~~~~~~~~~~~~~~~~
            fopen_frag.write("frag %d; " %frag_id)
            fopen_frag.write("min-in-mol; ")
            if not pairs:
                fopen_frag.write("dG(Rec) %s; "   %name[np.argmin(frag_data[0]+\
                                                                    frag_data[1])])
            fopen_frag.write("dG(Cplx) %s; "  %name[np.argmin(frag_data_cplx[0]+\
                                                         frag_data_cplx[1])])
            fopen_frag.write("dG(Lig) %s; "   %name[np.argmin(frag_data_lig[0]+\
                                                         frag_data_lig[1])])

            if not pairs:
                fopen_frag.write("dH(Rec) %s; "   %name[np.argmin(frag_data[0])])
            fopen_frag.write("dH(Cplx) %s; "  %name[np.argmin(frag_data_cplx[0])])
            fopen_frag.write("dH(Lig) %s; "   %name[np.argmin(frag_data_lig[0])])

            if not pairs:
                fopen_frag.write("dS(Rec) %s; "   %name[np.argmin(frag_data[1])])
            fopen_frag.write("dS(Cplx) %s; "  %name[np.argmin(frag_data_cplx[1])])
            fopen_frag.write("dS(Lig) %s; "   %name[np.argmin(frag_data_lig[1])])
                
            if not pairs:
                fopen_frag.write("dG(Delta) %s; " %(name[np.argmin(frag_data_delta[0]+\
                                                                    frag_data_delta[1])]))
            fopen_frag.write("dH(Delta) %s; " %name[np.argmin(frag_data_delta[0])])
            fopen_frag.write("dS(Delta) %s; " %name[np.argmin(frag_data_delta[1])])

            fopen_frag.write("\n")


            ### Maximum
            ### ~~~~~~~
            fopen_frag.write("frag %d; " %frag_id)
            fopen_frag.write("max; ")
            if not pairs:
                fopen_frag.write("dG(Rec) %6.3f; "  %np.max(frag_data[0]+\
                                                            frag_data[1]))
            fopen_frag.write("dG(Cplx) %6.3f; " %np.max(frag_data_cplx[0]+\
                                                         frag_data_cplx[1]))
            fopen_frag.write("dG(Lig) %6.3f; "  %np.max(frag_data_lig[0]+\
                                                         frag_data_lig[1]))
            if not pairs:
                fopen_frag.write("dH(Rec) %6.3f; "   %np.max(frag_data[0]))
            fopen_frag.write("dH(Cplx) %6.3f; "  %np.max(frag_data_cplx[0]))
            fopen_frag.write("dH(Lig) %6.3f; "   %np.max(frag_data_lig[0]))
        
            if not pairs:
                fopen_frag.write("dS(Rec) %6.3f; "   %np.max(frag_data[1]))
            fopen_frag.write("dS(Cplx) %6.3f; "  %np.max(frag_data_cplx[1]))
            fopen_frag.write("dS(Lig) %6.3f; "   %np.max(frag_data_lig[1]))
                
            if not pairs:
                fopen_frag.write("dG(Delta) %6.3f; " %(np.max(frag_data_delta[0]+\
                                                                frag_data_delta[1])))
            fopen_frag.write("dH(Delta) %6.3f; " %np.max(frag_data_delta[0]))
            fopen_frag.write("dS(Delta) %6.3f; " %np.max(frag_data_delta[1]))

            fopen_frag.write("\n")

            
            ### Maximum at Molecule
            ### ~~~~~~~~~~~~~~~~~~~
            fopen_frag.write("frag %d; " %frag_id)
            fopen_frag.write("max-in-mol; ")
            if not pairs:
                fopen_frag.write("dG(Rec) %s; "   %name[np.argmax(frag_data[0]+\
                                                                    frag_data[1])])
            fopen_frag.write("dG(Cplx) %s; "  %name[np.argmax(frag_data_cplx[0]+\
                                                         frag_data_cplx[1])])
            fopen_frag.write("dG(Lig) %s; "   %name[np.argmax(frag_data_lig[0]+\
                                                         frag_data_lig[1])])

            if not pairs:
                fopen_frag.write("dH(Rec) %s; "   %name[np.argmax(frag_data[0])])
            fopen_frag.write("dH(Cplx) %s; "  %name[np.argmax(frag_data_cplx[0])])
            fopen_frag.write("dH(Lig) %s; "   %name[np.argmax(frag_data_lig[0])])
        
            if not pairs:
                fopen_frag.write("dS(Rec) %s; "   %name[np.argmax(frag_data[1])])
            fopen_frag.write("dS(Cplx) %s; "  %name[np.argmax(frag_data_cplx[1])])
            fopen_frag.write("dS(Lig) %s; "   %name[np.argmax(frag_data_lig[1])])
                
            if not pairs:
                fopen_frag.write("dG(Delta) %s; " %(name[np.argmax(frag_data_delta[0]+\
                                                                    frag_data_delta[1])]))
            fopen_frag.write("dH(Delta) %s; " %name[np.argmax(frag_data_delta[0])])
            fopen_frag.write("dS(Delta) %s; " %name[np.argmax(frag_data_delta[1])])

            fopen_frag.write("\n")


            ### Average
            ### ~~~~~~~
            fopen_frag.write("frag %d; " %frag_id)
            fopen_frag.write("avg; ")
            if not pairs:
                fopen_frag.write("dG(Rec) %6.3f; "   %np.mean(frag_data[0]+\
                                                                frag_data[1]))
            fopen_frag.write("dG(Cplx) %6.3f; "  %np.mean(frag_data_cplx[0]+\
                                                         frag_data_cplx[1]))
            fopen_frag.write("dG(Lig) %6.3f; "   %np.mean(frag_data_lig[0]+\
                                                         frag_data_lig[1]))

            if not pairs:
                fopen_frag.write("dH(Rec) %6.3f; "   %np.mean(frag_data[0]))
            fopen_frag.write("dH(Cplx) %6.3f; "  %np.mean(frag_data_cplx[0]))
            fopen_frag.write("dH(Lig) %6.3f; "   %np.mean(frag_data_lig[0]))

            if not pairs:
                fopen_frag.write("dS(Rec) %6.3f; "   %np.mean(frag_data[1]))
            fopen_frag.write("dS(Cplx) %6.3f; "  %np.mean(frag_data_cplx[1]))
            fopen_frag.write("dS(Lig) %6.3f; "   %np.mean(frag_data_lig[1]))
                
            if not pairs:
                fopen_frag.write("dG(Delta) %6.3f; " %(np.mean(frag_data_delta[0]+\
                                                                frag_data_delta[1])))
            fopen_frag.write("dH(Delta) %6.3f; " %np.mean(frag_data_delta[0]))
            fopen_frag.write("dS(Delta) %6.3f; " %np.mean(frag_data_delta[1]))

            fopen_frag.write("\n")


        else:

            ### Minimum
            ### ~~~~~~~
            fopen_frag.write("frag %d; " %frag_id)
            fopen_frag.write("min; ")
            fopen_frag.write("dG(Rec) %6.3f; "  %np.min(frag_data[0]+\
                                                        frag_data[1]))
            fopen_frag.write("dH(Rec) %6.3f; "  %np.min(frag_data[0]))
            fopen_frag.write("dS(Rec) %6.3f; "  %np.min(frag_data[1]))
            fopen_frag.write("\n")


            ### Minimum at Molecule
            ### ~~~~~~~~~~~~~~~~~~~
            fopen_frag.write("frag %d; " %frag_id)
            fopen_frag.write("min-in-mol; ")
            fopen_frag.write("dG(Rec) %s; "  %name[np.argmin(frag_data[0]+\
                                                             frag_data[1])])
            fopen_frag.write("dH(Rec) %s; "  %name[np.argmin(frag_data[0])])
            fopen_frag.write("dS(Rec) %s; "  %name[np.argmin(frag_data[1])])
            fopen_frag.write("\n")


            ### Maximum
            ### ~~~~~~~
            fopen_frag.write("frag %d; " %frag_id)
            fopen_frag.write("max; ")
            fopen_frag.write("dG(Rec) %6.3f; "  %np.max(frag_data[0]+\
                                                        frag_data[1]))
            fopen_frag.write("dH(Rec) %6.3f; "  %np.max(frag_data[0]))
            fopen_frag.write("dS(Rec) %6.3f; "  %np.max(frag_data[1]))
            fopen_frag.write("\n")


            ### Maximum at Molecule
            ### ~~~~~~~~~~~~~~~~~~~
            fopen_frag.write("frag %d; " %frag_id)
            fopen_frag.write("max-in-mol; ")
            fopen_frag.write("dG(Rec) %s; "  %name[np.argmax(frag_data[0]+\
                                                             frag_data[1])])
            fopen_frag.write("dH(Rec) %s; "  %name[np.argmax(frag_data[0])])
            fopen_frag.write("dS(Rec) %s; "  %name[np.argmax(frag_data[1])])
            fopen_frag.write("\n")


            ### Average
            ### ~~~~~~~
            fopen_frag.write("frag %d; " %frag_id)
            fopen_frag.write("avg; ")
            fopen_frag.write("dG(Rec) %6.3f; "  %np.mean(frag_data[0]+\
                                                         frag_data[1]))
            fopen_frag.write("dH(Rec) %6.3f; "  %np.mean(frag_data[0]))
            fopen_frag.write("dS(Rec) %6.3f; "  %np.mean(frag_data[1]))
            fopen_frag.write("\n")

    fopen_frag.close()