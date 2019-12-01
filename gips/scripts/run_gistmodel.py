import numpy as np

from collections import OrderedDict

from gips.gistmodel.mode0 import mode0
from gips.gistmodel.mode1 import mode1
from gips.gistmodel.mode3 import mode3
from gips.gistmodel.mode4 import mode4
from gips.gistmodel.mode5 import mode5
from gips.gistmodel.mode6 import mode6
from gips.gistmodel.mode7 import mode7
from gips.gistmodel.generate_output import print_fun
from gips.utils.read_write import read_boundsfile
from gips.utils.read_write import read_pairsfile
from gips.utils.read_write import read_parmsfile
from gips.utils.misc import mode_error
from gips.utils.misc import generate_ksplits
from gips.datastrc.gdat_fit_lib import gdat_fit_lib

import copy

def gistmodel(gdatarec_lib, gdata_lib, mode, parms=6, pairs=False, decomp_E=False,
            decomp_S=False, optimizer='evolution', niter=500, nmin=1000, popsize=50, 
            stepsize=0.05, verbose=False, kforce=100., gradient=False, boundary=False, 
            radiusadd=[0.,3.], boundsfile=None, softness=1., softcut=2., pairfile=None, 
            exclude=None, paircut=0.0, shuffle=False, ksplit=5, ksplitfile=None, prefix=None,
            scaling=2.0, parmsfile=None):

    if verbose:
        print "Start optimization with"
        print "mode      = %d"    %mode
        print "optimizer = %s"    %optimizer
        print "niter     = %d"    %niter
        print "nmin      = %d"    %nmin
        if optimizer=="evolution":
            print "popsize   = %d" %popsize
        print "kforce    = %6.3f" %kforce
        print "gradient  = %s"    %gradient
        print "boundary  = %s"    %boundary
        print "pairs     = %s"    %pairs
        print "decomp_E  = %s"    %decomp_E
        print "decomp_S  = %s"    %decomp_S
        if pairs:
            print "pairfile  = %s"    %pairfile
            print "paircut   = %6.3f" %paircut
        print "softness  = %6.3f" %softness
        print "softcut   = %6.3f" %softcut
        print "shuffle   = %s"    %shuffle
        print "ksplit    = %d"    %ksplit
        if ksplitfile != None \
        and ksplitfile != "":
            print "Splitfile %s" %ksplitfile
        if exclude != None \
        and exclude != "":
            print "Exclude file %s" %exclude

    optparms = OrderedDict()
    optparms["Niter               "] = niter
    optparms["optimizer           "] = optimizer
    if optimizer=="evolution":
        optparms["Population size     "] = popsize
    elif optimizer=="brute":
        optparms["Stepsize            "] = stepsize
    else:
        optparms["Nmin                "] = nmin
    optparms["k_rstr              "] = kforce
    optparms["Radius add          "] = radiusadd
    optparms["Softness            "] = softness
    optparms["Softcut             "] = softcut
    optparms["Functional          "] = mode
    optparms["Analytic Gradient   "] = gradient
    optparms["Analytic Boundaries "] = boundary
    optparms["Pairs               "] = pairs
    optparms["Scoring             "] = parms
    if boundsfile != None \
    and boundsfile != "":
        optparms["Boundsfile          "] = boundsfile
    else:
        optparms["Boundsfile          "] = "None"
    if pairfile != None \
    and pairfile != "":
        optparms["Pairfile            "] = pairfile
    else:
        optparms["Pairfile            "] = "None"
    optparms["paircut             "] = paircut
    optparms["shuffle             "] = shuffle
    optparms["ksplit              "] = ksplit
    if ksplitfile != None \
    and ksplitfile != "":
        optparms["Splitfile           "] = ksplitfile
    if exclude != None \
    and exclude != "":
        optparms["Exclude file        "] = exclude

    if verbose:
        print "Organizing and preparing data ..."

    mode_dict = dict()
    mode_dict = {0 : mode0,
                 1 : mode1,
                 3 : mode3,
                 4 : mode4,
                 5 : mode5,
                 6 : mode6,
                 7 : mode7 }

    if mode in mode_dict.keys():
        fitmode = mode_dict[mode]
    else:
        mode_error(mode)

    if boundsfile != None \
    and boundsfile != "":
        if type(boundsfile) != str:
            raise TypeError("The path to boundsfile must be of type str, but is of type %s" %type(boundsfile))
        boundsdict = read_boundsfile(boundsfile)
    else:
        boundsdict = None

    if ksplitfile != None \
    and ksplitfile != "":

        ### We need to know some information about the
        ### data in advance. Therefore, we preload just
        ### the metadata (this should be fast).

        if verbose:
            print "Preloading the gdat lib ..."
        fit_lib = gdat_fit_lib(gdatarec_dict=gdatarec_lib,
                            gdata_dict=gdata_lib,
                            ref_energy=-11.108,
                            mode=mode,
                            radiusadd=radiusadd,
                            softness=softness,
                            softcut=softcut,
                            exclude=None,
                            scaling=scaling,
                            verbose=verbose)

        fit_lib.load_metadata()

        k_groups     = list()
        include_list = list()
        exclude_list = list()
        if pairs:
            pairlist     = list()
        else:
            pairlist     = None
        with open(ksplitfile, "r") as fopen:
            for line in fopen:
                for line in fopen:
                    l =line.rstrip().lstrip().split()
                    if len(l)==0:
                        continue
                    if l[0].startswith("#"):
                        continue
                    if pairs:
                        pairlist.append([l[0], l[1]])
                        k_groups.append(int(l[2]))
                        include_list.append(l[0])
                        include_list.append(l[1])
                    else:
                        include_list.append(l[0])
                        k_groups.append(int(l[1]))

        for name in fit_lib.name:
            if not name in include_list:
                exclude_list.append(name)

        if ksplit not in k_groups:
            raise ValueError("ksplit %d value is not found in ksplit file.")

        k_groups = np.array(k_groups)

        del fit_lib

    else:

        if pairs:
            if pairfile != None \
            and pairfile != "":
                if type(pairfile) != str:
                    raise TypeError("The path to pairfile must be of type str, but is of type %s" %type(pairfile))
                pairlist = read_pairsfile(pairfile, paircut)
            else:
                pairlist = None

        else:
            pairlist = None

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
            exclude_list = None

    fitter = fitmode(gdatarec_lib,
                    gdata_lib,
                    parms=parms,
                    pairs=pairs,
                    radiusadd=radiusadd,
                    softness=softness,
                    softcut=softcut,
                    boundsdict=boundsdict,
                    pairlist=pairlist,
                    exclude=exclude_list,
                    decomp_E=decomp_E,
                    decomp_S=decomp_S,
                    verbose=verbose)

    fitter.anal_boundary = boundary

    if shuffle:
        rand = np.arange(fitter._exp_data.shape[0])
        np.random.shuffle(rand)
        _exp  = copy.copy(fitter._exp_data)
        _name = copy.copy(fitter.name)
        for i, r in enumerate(rand):
            fitter._exp_data[i] = _exp[r]
            fitter.name[i]      = _name[r]

    if ksplitfile==None \
    or ksplitfile == "":
        if pairs:
            k_groups = generate_ksplits(ksplit, fitter.N_pairs)
        else:
            k_groups = generate_ksplits(ksplit, fitter.N_case)
        ksplitlist = range(ksplit)

    else:
        ksplitlist = [ksplit]
        ### Correct for different list ordering
        ### by swapping indices in k_groups
        if not pairs:
            for k, name in enumerate(include_list):
                i           = fitter.name.index(name)
                k_groups[i] = k_groups[k]
        else:
            for k, name in enumerate(pairlist):
                i           = fitter.name.index("%s-%s" %(name[0], name[1]))
                k_groups[i] = k_groups[k]

    if parmsfile!=None:
        parmdict = read_parmsfile(parmsfile)

    for i in ksplitlist:
        test_group  = np.where(k_groups==i)[0]
        train_group = np.where(k_groups!=i)[0]

        fitter.set_selection(train_group)
        if not shuffle:
            fitter.set_functional()
        fitter.set_bounds()
        fitter.set_step()
        fitter.set_x0()

        if parmsfile!=None:
            _print_fun = print_fun(fitter=fitter, mode=mode, optimizer="brute", 
                                    optparms=optparms, selection_A=train_group, selection_B=test_group, 
                                    prefix="k%d."%i + prefix, verbose=verbose)

            for key, value in parmdict.items():
                if key=="header":
                    continue
                x = np.array(value[:fitter._parms])

                _print_fun(x)
                _print_fun.flush()

        else:
            if verbose:
                print "Start optimization for ksplit=%d ..." %i

            _print_fun = print_fun(fitter=fitter, mode=mode, optimizer=optimizer, 
                                    optparms=optparms, selection_A=train_group, selection_B=test_group, 
                                    prefix="k%d."%i + prefix, verbose=verbose)

            fitter.optimize(niter=niter,
                            nmin=nmin,
                            kforce=kforce,
                            gradient=gradient,
                            print_fun=_print_fun,
                            popsize=popsize,
                            stepsize=stepsize,
                            optimizer=optimizer)

        if verbose:
            print "Generating output ..."

        _print_fun.finish()