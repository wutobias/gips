import argparse

def parse_args():

    run_modes = [   "buildlib",
                    "gistfit",
                    "mapout",
                    "decompose",
                    "split"
                ]

    parser = argparse.ArgumentParser(
        description="Executable for Gips (GIST-based processing of solvent functionals). \
        A program for modelling of solvation thermodynamics based on GIST data with particular focus on drug discovery. \
        For documentation and examples see the GitHub page: github.com/wutobias/gips")

    ### Required Arguments
    ### ~~~~~~~~~~~~~~~~~~

    required = parser.add_argument_group('required arguments')


    required.add_argument('-m', '--mode',
                            required=True,
                            type=str,
                            default=None,
                            help="Run mode.",
                            choices=run_modes)

    ### Optional Arguments ###
    ### ~~~~~~~~~~~~~~~~~~ ###

    mutual_group1 = parser.add_mutually_exclusive_group()

    parser._action_groups.append(mutual_group1)
    parser._action_groups.append(parser._action_groups.pop(1))

    mutual_group1.add_argument('-gd', '--gdat',
                                required=False,
                                type=str,
                                help='Path to gistdata input file.')


    mutual_group1.add_argument('-ll', '--loadlib',
                                required=False,
                                type=str,
                                help='Path to gistdata lib file.')


    parser.add_argument('-b', '--boundsfile',
                        required=False,
                        type=str,
                        help='Path to file containing bounds used in gistfit. \
                        If not provided, then the bounds will be calculated from the \
                        max and min values of the gist data.')


    parser.add_argument('-sl', '--savelib',
                        required=False,
                        type=str,
                        help='Path for saving gistdata lib file. \
                        Only required in mode=buildlib')


    parser.add_argument('-f', '--fitmode',
                        required=False,
                        type=int,
                        default=0,
                        choices=[0,1,3,4,5,6,7],
                        help='Choice of fitting procedure used for \
                        processing GIST data. Valids choices are \
                        0:Displacement; \
                        1:Displacement with energy-entropy decompostion; \
                        3:Reorganization; \
                        4:Reorganization with energy-entropy decompostion; \
                        5:Reorganization with additional density parameters; \
                        6:Reorganization with additional density parameters and energy-entropy decompostion; \
                        7:Reorganization with individual scoring functions for receptor, ligand and complex. Also energy-entropy decompostion')


    parser.add_argument('-s', '--score',
                        required=False,
                        type=int,
                        default=6,
                        choices=[4,5,6],
                        help='Choice of base GIST scoring function')


    parser.add_argument('-sc', '--scaling',
                        required=False,
                        type=float,
                        default=2.,
                        help='Energy scaling factor. Default is 2.')


    parser.add_argument('-dE', '--decomp_E',
                        action='store_true',
                        help='Activate partial decomposition mode with only energy. Note, that when both --decomp_E and --decomp_S \
are activated, only --decomp_E will be recognized.')


    parser.add_argument('-dS', '--decomp_S',
                        action='store_true',
                        help='Activate partial decomposition mode with only entropy. Note, that when both --decomp_E and --decomp_S \
are activated, only --decomp_E will be recognized.')


    parser.add_argument('-c', '--cut',
                        required=False,
                        type=float,
                        default=-1.,
                        help='Cut out range for gist boxes. Negative value \
                        means no cutting performed.')


    parser.add_argument('-r', '--radiusadd',
                        required=False,
                        type=float,
                        nargs=2,
                        default=[0.,3.],
                        help='Constant added to atomic radii during calculation of molecular volume. \
                        Can be used to calculate volume around molecule up to a certain shell of water. \
                        The first value in the list will be used for the receptor, the second value will \
                        be used for ligand and complex. It is recommended to use \'0\' for the receptor \
                        and a value of \'3\' for the ligand and complex.')


    parser.add_argument('-softness', '--softness',
                        required=False,
                        type=float,
                        default=1.,
                        help='Softness parameter \'s\' used for the calculation of soft molecular surface.')


    parser.add_argument('-softcut', '--softcut',
                        required=False,
                        type=float,
                        default=2.,
                        help='Softness cutoff parameter \'c\' used for the calculation of soft molecular surface.')


    parser.add_argument('-ni', '--niter',
                        required=False,
                        type=int,
                        default=500,
                        help='Number MC attempts in basinhopping optimization before termination or number of generations \
in evolution optimization.')


    parser.add_argument('-pop', '--popsize',
                        required=False,
                        type=int,
                        default=50,
                        help='Population size during differental evolution optimization.')


    parser.add_argument('-step', '--stepsize',
                        required=False,
                        type=float,
                        default=0.05,
                        help='Search stepsize used for brute force optimization.')


    parser.add_argument('-opt', '--optimizer',
                        required=False,
                        default='evolution',
                        type=str,
                        choices=['evolution', 'basinhopping', 'brute'],
                        help='Optimizer strategy used for optmization. \
                        For more details see PyGMO documentation.')


    parser.add_argument('-gr', '--gradient',
                        action='store_true',
                        help='Use analytical gradient. Otherwise approximate gradient using forward \
                        finite differences.')


    parser.add_argument('-bo', '--boundary',
                        action='store_true',
                        help='Use analytical boundary restraints. Otherwise treat boundaries as explicit \
                        constraints.')


    parser.add_argument('-k', '--kforce',
                        required=False,
                        type=float,
                        default=100.,
                        help='Force constant used for analytical treatment of boundaries. \
                        Default is 100.')


    parser.add_argument('-p', '--pairs',
                        action='store_true',
                        help='Activate fitting pairs of binding affinity differences \
instead of absolute binding affinities.')


    parser.add_argument('-pf', '--pairfile',
                        required=False,
                        type=str,
                        default="",
                        help="If the --pairs option is activated, then this option \
can be used to specify a file which contains all pairs that should be used in pairwise \
fitting. This file should contain a three-column row for each pair. The first two columns \
contain the title (as in the input file) of the considered datasets and the third column \
contains a value (i.e. score), which can be used for filtering (e.g. Tanimoto coefficient).")


    parser.add_argument('-pc', '--paircut',
                        required=False,
                        type=float,
                        default=0.0,
                        help="Filter the pairs given by the --pairfile option using this cutoff value. \
Each pair has a corresponding score value, as given by the third column of the --pairfile . Only \
those pairs will be considered in fitting, which have a score value, that is greater than paircut.")


    parser.add_argument('-ks', '--ksplit',
                        required=False,
                        type=int,
                        default=5,
                        help="Split the dataset into k equally-sized subsets and perform \
fitting for within all (k-1) datasets and report fitting statistics for subset k. This is \
cross validation. If --ksplitfile is provided, then --ksplit gives the number of the subset \
which is used for the testset.")


    parser.add_argument('-ksf', '--ksplitfile',
                        required=False,
                        type=str,
                        default="",
                        help="File that contains group defintions for how to split the data during \
gistfit runs. Works with pairs and without pairs. These fails can be generated with --mode=split. \
Note that, when providing the ksplitfile in --mode=gistfit, no pairfile or exclude file will be read in, \
since all information for how to select datapoints or pairs are already included in the ksplitfile.")


    parser.add_argument('-sh', '--shuffle',
                        action='store_true',
                        help='Shuffle the dG, dH, dS data before fitting.')


    parser.add_argument('-ex', '--exclude',
                        required=False,
                        type=str,
                        default="",
                        help="File that contains all datapoints (according to their title \
gistdata lib file) that shall be excluded from the gistfit procedure. Provide one datapoint \
per row or write them down separated by whitespaces.")


    parser.add_argument('-in', '--include',
                        required=False,
                        type=str,
                        default="",
                        help="File that contains all datapoints (according to their title \
gistdata lib file) that shall be included in the split procedure. Provide one datapoint \
per row or write them down separated by whitespaces.")


    parser.add_argument('-parms', '--parmsfile',
                        required=False,
                        type=str,
                        default="",
                        help="Parameter output file obtained from a gistfit run.")


    parser.add_argument('-frag', '--frag_file',
                        required=False,
                        type=str,
                        default="",
                        help="File containing fragments for external fragment library. First column \
is consecutively numbered fragment id. The second column is the smiles code of the fragment.")


    parser.add_argument('-map', '--map_file',
                        required=False,
                        type=str,
                        default="",
                        help="File containing fragment mapping instructions for external fragment library. First column \
is the name of the molecule as it appears in the gistlib file. The second column is a comma-seperated list of the fragment ids \
that it is mapped to. If it is not mapped on a fragment, then this column must contain only the entry \'-1\'. The following columns \
contain one comma-seperated list \'M\' of atom-indices per fragment, that the molecule is mapped to. In the list of atom-indices, the list \
entry at position i maps the atom at position i in the fragment on the atom at position M[i] in the molecule.")


    parser.add_argument('-pre', '--prefix',
                        required=False,
                        type=str,
                        default="",
                        help="Prefix used for filenames.")


    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        help='Verbosity output. Default is off.')


    return parser.parse_args()