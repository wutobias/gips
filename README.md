Welcome to *Gips*. *Gips* is an acronym for GIST-based Processing of Solvent Functionals. It is a program that facilitates the use of GIST (Grid Inhomogenous Solvation Theory) data in order to build models for the calculation of solvation thermodynamics. This can be applied to, but is not limited to, protein-ligand complexes, apo proteins or unbound ligand molecules. It must be noted that Gips does not calculate GIST data on its on. For the calculation of GIST data, have a look at [SSTMap](https://github.com/KurtzmanLab/SSTMap) or [CPPTRAJ](https://github.com/Amber-MD/cpptraj).

Requirements
============

* Linux Operating System
* GNU C compiler
* python 2.7 (Anaconda2 recommended)
* python packages: numpy, scipy, pygmo, matplotlib, rdkit
* python packages used for installation only: pip, setuptools

Installation
============

First, download Gips from the GitHub page
    
```
git clone https://github.com/wutobias/gips
```

Second, go into the gips directory and install the program

```
python setup.py install
```

Usage
=====

The main executable of Gips is *run_gips*. A list of all run modes and parameters necessary to run Gips, can be found using *run_gips --help*:

```
required arguments:
  -m {buildlib,gistfit,mapout,decompose,split}, --mode {buildlib,gistfit,mapout,decompose,split}
                        Run mode.

  -gd GDAT, --gdat GDAT
                        Path to gistdata input file.
  -ll LOADLIB, --loadlib LOADLIB
                        Path to gistdata lib file.

optional arguments:
  -h, --help            show this help message and exit
  -gd GDAT, --gdat GDAT
                        Path to gistdata input file.
  -ll LOADLIB, --loadlib LOADLIB
                        Path to gistdata lib file.
  -b BOUNDSFILE, --boundsfile BOUNDSFILE
                        Path to file containing bounds used in gistfit. If not
                        provided, then the bounds will be calculated from the
                        max and min values of the gist data.
  -sl SAVELIB, --savelib SAVELIB
                        Path for saving gistdata lib file. Only required in
                        mode=buildlib
  -c CUT, --cut CUT     Cut out range for gist boxes. Negative value means no
                        cutting performed.
  -ni NITER, --niter NITER
                        Number MC attempts in basinhopping optimization or
                        number of generations in evolution optimization.
  -nm NMIN, --nmin NMIN
                        Number maximum minimization steps after each MC move.
  -pop POPSIZE, --popsize POPSIZE
                        Population size during differental evolution
                        optimization.
  -step STEPSIZE, --stepsize STEPSIZE
                        Search stepsize used for brute force optimization.
  -opt {evolution,basinhopping,brute}, --optimizer {evolution,basinhopping,brute}
                        Optimizer strategy used for optmization. For more
                        details see PyGMO documentation.
  -of OUTFREQ, --outfreq OUTFREQ
                        Output frequency for gistmodelout file.
  -r RADIUSADD RADIUSADD, --radiusadd RADIUSADD RADIUSADD
                        Constant added to atomic radii during calculation of
                        molecular volume. Can be used to calculate volume
                        around molecule up to a certain shell of water. The
                        first value in the list will be used for the receptor,
                        the second value will be used for ligand and complex.
                        It is recommended to use '0' for the receptor and a
                        value of '3' for the ligand and complex.
  -soft SOFTNESS, --softness SOFTNESS
                        Softness parameter 's' used for the calculation of
                        soft molecular surface.
  -softcut SOFTCUT, --softcut SOFTCUT
                        Softness cutoff parameter 'c' used for the calculation
                        of soft molecular surface.
  -gr, --gradient       Use analytical gradient. Otherwise approximate
                        gradient using forward finite differences.
  -bo, --boundary       Use analytical boundary restraints. Otherwise treat
                        boundaries as explicit constraints.
  -k KFORCE, --kforce KFORCE
                        Force constant used for analytical treatment of
                        boundaries. Default is 100.
  -f {0,1,3,4,5,6,7}, --fitmode {0,1,3,4,5,6,7}
                        Choice of fitting procedure used for processing GIST
                        data. Valids choices are 0:Displacement;
                        1:Displacement with energy-entropy decompostion;
                        3:Reorganization; 4:Reorganization with energy-entropy
                        decompostion; 5:Reorganization with additional density
                        parameters; 6:Reorganization with additional density
                        parameters and energy-entropy decompostion;
                        7:Reorganization with individual scoring functions for
                        receptor, ligand and complex. Also energy-entropy
                        decompostion
  -s {4,5,6}, --score {4,5,6}
                        Choice of base GIST scoring function
  -sc SCALING, --scaling SCALING
                        Energy scaling factor. Default is 2.
  -dE, --decomp_E       Activate partial decomposition mode with only energy.
                        Note, that when both --decomp_E and --decomp_S are
                        activated, only --decomp_E will be recognized.
  -dS, --decomp_S       Activate partial decomposition mode with only entropy.
                        Note, that when both --decomp_E and --decomp_S are
                        activated, only --decomp_E will be recognized.
  -p, --pairs           Activate fitting pairs of binding affinity differences
                        instead of absolute binding affinities.
  -pf PAIRFILE, --pairfile PAIRFILE
                        If the --pairs option is activated, then this option
                        can be used to specify a file which contains all pairs
                        that should be used in pairwise fitting. This file
                        should contain a three-column row for each pair. The
                        first two columns contain the title (as in the input
                        file) of the considered datasets and the third column
                        contains a value (i.e. score), which can be used for
                        filtering (e.g. Tanimoto coefficient).
  -pc PAIRCUT, --paircut PAIRCUT
                        Filter the pairs given by the --pairfile option using
                        this cutoff value. Each pair has a corresponding score
                        value, as given by the third column of the --pairfile
                        . Only those pairs will be considered in fitting,
                        which have a score value, that is greater than
                        paircut.
  -ks KSPLIT, --ksplit KSPLIT
                        Split the dataset into k equally-sized subsets and
                        perform fitting for within all (k-1) datasets and
                        report fitting statistics for subset k. This is cross
                        validation. If --ksplitfile is provided, then --ksplit
                        gives the number of the subset which is used for the
                        testset.
  -ksf KSPLITFILE, --ksplitfile KSPLITFILE
                        File that contains group defintions for how to split
                        the data during gistfit runs. Works with pairs and
                        without pairs. These fails can be generated with
                        --mode=split. Note that, when providing the ksplitfile
                        in --mode=gistfit, no pairfile or exclude file will be
                        read in, since all information for how to select
                        datapoints or pairs are already included in the
                        ksplitfile.
  -sh, --shuffle        Shuffle the dG, dH, dS data before fitting.
  -ex EXCLUDE, --exclude EXCLUDE
                        File that contains all datapoints (according to their
                        title gistdata lib file) that shall be excluded from
                        the gistfit procedure. Provide one datapoint per row
                        or write them down separated by whitespaces.
  -in INCLUDE, --include INCLUDE
                        File that contains all datapoints (according to their
                        title gistdata lib file) that shall be included in the
                        split procedure. Provide one datapoint per row or
                        write them down separated by whitespaces.
  -parms PARMSFILE, --parmsfile PARMSFILE
                        Parameter output file obtained from a gistfit run.
  -frag FRAG_FILE, --frag_file FRAG_FILE
                        File containing fragments for external fragment
                        library. First column is consecutively numbered
                        fragment id. The second column is the smiles code of
                        the fragment.
  -map MAP_FILE, --map_file MAP_FILE
                        File containing fragment mapping instructions for
                        external fragment library. First column is the name of
                        the molecule as it appears in the gistlib file. The
                        second column is a comma-seperated list of the
                        fragment ids that it is mapped to. If it is not mapped
                        on a fragment, then this column must contain only the
                        entry '-1'. The following columns contain one comma-
                        seperated list 'M' of atom-indices per fragment, that
                        the molecule is mapped to. In the list of atom-
                        indices, the list entry at position i maps the atom at
                        position i in the fragment on the atom at position
                        M[i] in the molecule.
  -pre PREFIX, --prefix PREFIX
                        Prefix used for filenames.
  -v, --verbose         Verbosity output. Default is off.

```
Run mode
--------

When running the program, it is necessary to specify a run mode, which can be one of these five:

* buildlib: This mode lets you merge the GIST data into a python pickle object and save it on disk. This is much more convenient than loading the GIST grids everytime you want to do something with them. Also, you can specify a "cut out" parameter using --cut, which specifies the radius around the ligand molecules on the GIST grid that should be considered. This is step saves a lot of memory and computing time for all subsequent steps.

* gistfit: This is the most important mode of Gips, as it is used for the parameter fitting process for solvent functionas. This run modes requires several additional arguments, such as --score or --fitmode which describe the basic functional and the type of the solvent functionals used during parameter fitting.

* mapout: With this run mode, one can identify hydration sites based on the solvent functionals and save them as PDB files. The "score" of each hydration site is stored in the B factor column.

* decompose: With this mode one can spatially decompose the solvent functionals based on molecular fragmentation. The decomposition regions are defined using fragment definitions (either provided or determined internally) of the dataset. The output of the method is a detailed list of all fragment contributions in the individual molecules allowing for (A) building molecules with tailord solvation dynamics or (B) analyzing the "drivers" of solvation using fragment-based design ideas.

* split: This method merely serves for splitting the dataset into test and training sets.


Contact
=======

Main Author of the Program
--------------------------
* Tobias Hüfner (née Wulsdorf), Philipps-Universität Marburg, tobias.wulsdorf@uni-marburg.de

Scientific Supervision
----------------------
* Gerhard Klebe, Philipps-Universität Marburg, klebe@staff.uni-marburg.de
