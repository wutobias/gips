Welcome to *Gips*. *Gips* is an acronym and stands for GIST-based Processing of Solvent Functionals. It is a program that facilitates the use of GIST (Grid Inhomogenous Solvation Theory) data in order to build models for the calculation of solvation thermodynamics. This can be applied to, but is not limited to, protein-ligand complexes, apo proteins or unbound ligand molecules. It must be noted that Gips does not calculate GIST data on its on. For the calculation of GIST data, have a look at [SSTMap](https://github.com/KurtzmanLab/SSTMap) or [CPPTRAJ](https://github.com/Amber-MD/cpptraj).

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

Background
==========

The background on Gips and the demonstration of a comprehensive case study will be part of two upcoming publications. Be patient. :-)

Usage
=====

The main executable of Gips is *run_gips*. A comprehensive manual on how to use the code will be found soon in this repository.

Contact
=======

Main Author of the Program
--------------------------
* Tobias Hüfner (née Wulsdorf), Philipps-Universität Marburg, tobias.wulsdorf@uni-marburg.de

Scientific Supervision
----------------------
* Gerhard Klebe, Philipps-Universität Marburg, klebe@staff.uni-marburg.de
