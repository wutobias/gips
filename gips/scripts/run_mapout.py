import numpy as np

import copy
import pygmo

from gips.gistmodel.mode0 import mode0
from gips.gistmodel.mode1 import mode1
from gips.gistmodel.mode3 import mode3
from gips.gistmodel.mode4 import mode4
from gips.gistmodel.mode5 import mode5
from gips.gistmodel.mode6 import mode6
from gips.gistmodel.mode7 import mode7
from gips.utils.misc import mode_error
from gips.mapout.map_processing import mapout_maps
from gips.utils.read_write import read_parmsfile
from gips.utils.read_write import write_maps


def mapout(gdatarec_lib, gdata_lib, mode, parms=6, pairs=False, 
            parmsfile=None, radiusadd=[0.,3.], softness=1., softcut=2., 
            exclude=None, prefix=None, scaling=2.0, verbose=False):

    if verbose:
        print "Start mapout procedure with"
        print "mode      = %d" %mode
        print "softness  = %6.3f" %softness
        print "softcut   = %6.3f" %softcut

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

    fitter = fitmode(gdatarec_lib,
                    gdata_lib,
                    parms=parms,
                    pairs=False,
                    radiusadd=radiusadd,
                    softness=softness,
                    softcut=softcut,
                    scaling=scaling,
                    verbose=verbose)

    ### Find position of SES in parms file
    if parmsfile==None:
        raise IOError("Must provide parmsfile.")
    parmdict = read_parmsfile(parmsfile)

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

    ### Collect all the solutions
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

    ### Find the best solution
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


    ### ~~~~~~~~~~~~~~~~~~ ###
    ### Write out the maps ###
    ### ~~~~~~~~~~~~~~~~~~ ###

    ### Write out un-processed dx grids
    if mode in [0,1,2]:
        counter = 0
        for rec_keys in fitter.gdatarec_dict.keys():
            recdict = fitter.gdatarec_dict[rec_keys]
            title   = recdict["title"]
            if title == None:
                name = "%d" %counter
            else:
                name = title
            for i in range(len(recdict["receptor"])):
                if recdict["receptor"][i]["gdat"] == None:
                    continue
                write_maps(recdict["receptor"][i]["gdat"], prefix="rec_%s_%d" %(name,i), pymol=True)
            counter += 1

    else:
        counter = 0
        for rec_keys in fitter.gdatarec_dict.keys():
            recdict = fitter.gdatarec_dict[rec_keys]
            title   = recdict["title"]
            if title == None:
                name = "%d" %counter
            else:
                name = title
            for i in range(len(recdict["receptor"])):
                if recdict["receptor"][i]["gdat"] == None:
                    continue
                write_maps(recdict["receptor"][i]["gdat"], prefix="rec_%s_%d" %(name,i), pymol=True)
            counter += 1

        counter = 0
        for cplx_keys in fitter.gdata_dict.keys():
            cplxdict = fitter.gdata_dict[cplx_keys]
            if cplxdict["title"] in fitter.exclude:
                continue
            title    = cplxdict["title"]
            if title == None:
                name = "%d" %counter
            else:
                name = title
            _N_dict  = len(cplxdict["complex"])
            for i in range(_N_dict):
                if cplxdict["complex"][i]["gdat"] == None:
                    continue
                write_maps(cplxdict["complex"][i]["gdat"], prefix="cplx_%s_%d" %(name,i), pymol=True)
            _N_dict  = len(cplxdict["ligand"])
            for i in range(_N_dict):
                if cplxdict["ligand"][i]["gdat"] == None:
                    continue
                write_maps(cplxdict["ligand"][i]["gdat"], prefix="lig_%s_%d" %(name,i), pymol=True)
            counter += 1

    ### Write out pre-processed xyz grids
    m = mapout_maps(fitter, best_x_A, pairs, prefix)

    if mode in [0,1]:
        m.process_rec  = True
        m.process_cplx = False
        m.process_lig  = False
    else:
        m.process_rec  = True
        m.process_cplx = True
        m.process_lig  = True

    for case in range(fitter.N_case):
        if fitter.name[case] in exclude_list:
            continue
        m.set_case(case)
        ### Internal write routine as a callback to the process routine
        m.process(m.write)