import pickle as pickle
import gzip
from gips.scripts.parse_args import parse_args
from gips.scripts import run_buildlib
from gips.scripts import run_gistmodel
from gips.scripts import run_split
from gips.scripts import run_mapout
from gips.scripts import run_decomposition

def main():

    args = parse_args()


    ### ~~~~~~~~~~~~~~~~~ ###
    ### Runmode: buildlib ###
    ### ~~~~~~~~~~~~~~~~~ ###

    if args.mode == "buildlib":
        if args.verbose:
            print("Attempting to build library ...")
        gistlib = run_buildlib.buildlib(args.gdat, args.verbose, args.cut)
        if args.savelib != None:
            if args.verbose:
                print("Saving library file ...")
            if not args.savelib.endswith(".gz"):
                args.savelib = args.savelib+".gz"
            output = gzip.open(args.savelib, "wb")
            pickle.dump(gistlib, output)
            output.close()

    
    ### ~~~~~~~~~~~~~~~~ ###
    ### Runmode: gistfit ###
    ### ~~~~~~~~~~~~~~~~ ###

    elif args.mode == "gistfit":
        if args.gdat != None:
            if args.verbose:
                print("Loading gdata from %s ..." %args.gdat)
            gdatarec_dict, gdata_dict = run_buildlib.buildlib(args.gdat, args.verbose, args.cut)
        elif args.loadlib != None:
            if args.verbose:
                print("Loading gdata from %s ..." %args.loadlib)
            if args.loadlib.endswith(".gz"):
                fopen = gzip.open(args.loadlib, "rb")
            else:
                fopen = open(args.loadlib, "rb")
            gdatarec_dict, gdata_dict = pickle.load(fopen)
            fopen.close()
        else:
            raise IOError("Must provide gistdata by specifing either --loadlib or --gdat")

        if args.verbose:
            print("Attempting to build a gist model ...")

        run_gistmodel.gistmodel(gdatarec_dict, gdata_dict, mode=args.fitmode, parms=args.score, pairs=args.pairs,
                                decomp_E=args.decomp_E, decomp_S=args.decomp_S, optimizer=args.optimizer, niter=args.niter, 
                                popsize=args.popsize, stepsize=args.stepsize, verbose=args.verbose, 
                                kforce=args.kforce, gradient=args.gradient, boundary=args.boundary, radiusadd=args.radiusadd, 
                                boundsfile=args.boundsfile, softness=args.softness, softcut=args.softcut, pairfile=args.pairfile, 
                                exclude=args.exclude, paircut=args.paircut, shuffle=args.shuffle, ksplit=args.ksplit, 
                                ksplitfile=args.ksplitfile, prefix=args.prefix, scaling=args.scaling, parmsfile=args.parmsfile)


    ### ~~~~~~~~~~~~~~ ###
    ### Runmode: split ###
    ### ~~~~~~~~~~~~~~ ###

    elif args.mode == "split":

        run_split.split(pairfile=args.pairfile, inclfile=args.include, exclfile=args.exclude,
                        K=args.ksplit, cut=args.paircut, prefix=args.prefix)


    ### ~~~~~~~~~~~~~~~ ###
    ### Runmode: mapout ###
    ### ~~~~~~~~~~~~~~~ ###

    elif args.mode == "mapout":

        if args.parmsfile==None:
            raise IOError("Must provide parmsfile.")

        if args.gdat != None:
            if args.verbose:
                print("Loading gdata from %s ..." %args.gdat)
            gdatarec_dict, gdata_dict = run_buildlib.buildlib(args.gdat, args.verbose, args.cut)
        elif args.loadlib != None:
            if args.verbose:
                print("Loading gdata from %s ..." %args.loadlib)
            if args.loadlib.endswith(".gz"):
                fopen = gzip.open(args.loadlib, "rb")
            else:
                fopen = open(args.loadlib, "rb")
            gdatarec_dict, gdata_dict = pickle.load(fopen)
            fopen.close()
        else:
            raise IOError("Must provide gistdata by specifing either --loadlib or --gdat")

        if args.verbose:
            print("Attempting to mapout the gistmodel ...")

        run_mapout.mapout(gdatarec_dict, gdata_dict, mode=args.fitmode, parms=args.score, pairs=args.pairs, 
                            parmsfile=args.parmsfile, radiusadd=args.radiusadd, softness=args.softness, softcut=args.softcut, 
                            exclude=args.exclude, prefix=args.prefix, scaling=args.scaling, verbose=args.verbose)


    ### ~~~~~~~~~~~~~~~~~~ ###
    ### Runmode: decompose ###
    ### ~~~~~~~~~~~~~~~~~~ ###

    elif args.mode == "decompose":

        if args.parmsfile==None:
            raise IOError("Must provide parmsfile.")

        if args.gdat != None:
            if args.verbose:
                print("Loading gdata from %s ..." %args.gdat)
            gdatarec_dict, gdata_dict = run_buildlib.buildlib(args.gdat, args.verbose, args.cut)
        elif args.loadlib != None:
            if args.verbose:
                print("Loading gdata from %s ..." %args.loadlib)
            if args.loadlib.endswith(".gz"):
                fopen = gzip.open(args.loadlib, "rb")
            else:
                fopen = open(args.loadlib, "rb")
            gdatarec_dict, gdata_dict = pickle.load(fopen)
            fopen.close()
        else:
            raise IOError("Must provide gistdata by specifing either --loadlib or --gdat")

        if args.verbose:
            print("Attempting to mapout the gistmodel ...")

        run_decomposition.decomposition(gdatarec_dict, gdata_dict, mode=args.fitmode, parms=args.score, pairs=args.pairs,
                                        parmsfile=args.parmsfile, frag_file=args.frag_file, map_file=args.map_file, radiusadd=args.radiusadd, 
                                        softness=args.softness, softcut=args.softcut, pairfile=args.pairfile, exclude=args.exclude, 
                                        paircut=args.paircut, prefix=args.prefix, scaling=args.scaling, verbose=args.verbose)


def entry_point():

    main()

if __name__ == '__main__':

    entry_point()
