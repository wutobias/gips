import numpy as np

import os
from subprocess import call as sp_call

class aux_progs(object):

    def __init__(self, verbose):

        self.AMBERHOME=os.getenv('AMBERHOME')
        if self.AMBERHOME == None:
            raise Warning("Environment variable AMBERHOME not set!")

        self.ante_exe    = self.AMBERHOME+"/bin/antechamber -pf y -dr no"
        self.respgen_exe = self.AMBERHOME+"/bin/respgen"
        self.espgen_exe  = self.AMBERHOME+"/bin/espgen"
        self.resp_exe    = self.AMBERHOME+"/bin/resp"

        self.verbose = verbose

        if self.verbose:
            self.stdout = open("stdout", 'w')
            self.stderr = open("stderr", 'w')
        else:
            self.stdout = open(os.devnull, 'w')
            self.stderr = open(os.devnull, 'w')

    def call(self, prog, args):

        args_call = list()
        args_call.append(prog)
        for arg in args.lstrip().rstrip().split():
            args_call.append(arg)
        sp_call(args_call, 
                stdout=self.stdout,
                stderr=self.stderr)


def are_mol_same(mol1, mol2, useChirality=True):

    is_same     = False
    mol1_length = mol1.GetNumAtoms()
    mol2_length = mol2.GetNumAtoms()
    if mol1_length == mol2_length:
        mol1_atoms  = list(range(mol1_length))
        ### If both have same number of atoms.
        if mol1_length == mol2_length:
            matches = mol1.GetSubstructMatches(mol2, useChirality=useChirality)
            ### If there is only one substructure match, which also has
            ### as many atoms as the other molecule.
            if len(matches) == 1 and len(matches[0]) == mol1_length:
                is_same = True
    return is_same


def generate_ksplits(k, L):

    __doc__="""
    Generate k equally-sized random splits for a dataset of length L.
    Returns an integer type array of length L. Each element is an integer
    in the range (0,k-1).
    """

    k_size = L/k
    a      = np.arange(L)

    random_assignments = np.zeros(L, dtype=int)-1
    for i in range(k-1):
        k_assigned = 0
        while k_assigned<k_size:
            r = np.random.randint(L)
            if random_assignments[r] == -1:
                random_assignments[r] = i
                k_assigned += 1
    valids = np.where(random_assignments==-1)
    random_assignments[valids] = k-1

    return random_assignments


def check_factor(N, f):
    N=int(N)
    f=int(f)
    q=N/f
    if N==f:
        return True
    elif q%f != 0:
        return False
    else:
        return check_factor(N/f,f)


def are_you_numpy(a):

    """
    Returns True if a is an instance of numpy.
    False otherwise.
    """

    return type(a).__module__ == np.__name__


def mode_error(mode):
    raise IOError("Mode %s is not understood." %mode)


def parms_error(parms, _parms):
    raise ValueError("Functional with parms=%d / _parms=%d not known." %(parms,_parms))