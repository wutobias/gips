import rdkit
from rdkit import Chem
#from rdkit.Chem import BRICS
from rdkit.Chem import Recap
from rdkit.Chem import Draw
from rdkit.Chem.Draw import DrawingOptions

DrawingOptions.atomLabelFontSize = 55
DrawingOptions.dotsPerAngstrom   = 100
DrawingOptions.bondLineWidth     = 3.0

import numpy as np

from gips.utils.misc import are_mol_same

def get_frag_list(mol):

    hierarch  = Recap.RecapDecompose(mol, minFragmentSize=5)
    frag_list = list()
    for frag_smi in hierarch.GetLeaves().keys():
        frag_mol = Chem.MolFromSmiles(frag_smi)
        frag_list.append(frag_mol)

    return frag_list


class frag_library(object):

    def __init__(self):

        self.frag_list   = list()
        self.mol_list    = list()
        self.N_frag      = 0
        self.N_mol       = 0
        self.frag2mol    = list()
        self.mol2frag    = list()

        self.max_mol2frag = 0
        self.max_frag2mol = 0

        self.qp = Chem.AdjustQueryParameters()

        self.frag2mol_mapping = list()
    
    def add_frag_list(self, frag_list, mol):

        self.mol_list.append(mol)
        self.mol2frag.append(list())
        new_mol_id  = self.N_mol
        self.N_mol += 1

        if len(frag_list)==0:
            frag_list = [mol]

        for frag in frag_list:

            new_frag_id = -1

            for frag_id, frag_db in enumerate(self.frag_list):
                self.qp.makeDummiesQueries = False
                frag_db = Chem.AdjustQueryProperties(frag_db, self.qp)
                frag    = Chem.AdjustQueryProperties(frag, self.qp)
                if are_mol_same(frag_db, frag, useChirality=True):
                    ### If we are here, then the fragment is already
                    ### in the database
                    new_frag_id = frag_id
                    break

            if new_frag_id == -1:
                ### If we are here, then the fragment is new
                self.frag_list.append(frag)
                self.frag2mol.append(list())
                new_frag_id  = self.N_frag
                self.N_frag += 1

            if new_mol_id not in self.frag2mol[new_frag_id]:
                self.frag2mol[new_frag_id].append(new_mol_id)
                if len(self.frag2mol[new_frag_id])>self.max_frag2mol:
                    self.max_frag2mol=len(self.frag2mol[new_frag_id])

            if new_frag_id not in self.mol2frag[new_mol_id]:
                self.mol2frag[new_mol_id].append(new_frag_id)
                if len(self.mol2frag[new_mol_id])>self.max_mol2frag:
                    self.max_mol2frag=len(self.mol2frag[new_mol_id])

    
    def draw(self, prefix=""):

        frag_names = list()
        for i in range(self.N_frag):
            frag_names.append("%d" %i)
        mol_names = list()
        for i in range(self.N_mol):
            mol_names.append("%d" %i)

        try:
            img=Draw.MolsToGridImage(self.frag_list,
                                     molsPerRow=3,
                                    subImgSize=(500,500),
                                    legends=frag_names,
                                    useSVG=False)
            img.save("%sfrag_list.png" %prefix)
        except:
            pass

        try:
            img=Draw.MolsToGridImage(self.mol_list,
                                     molsPerRow=3,
                                    subImgSize=(500,500),
                                    legends=mol_names,
                                    useSVG=False)
            img.save("%smol_list.png" %prefix)
        except:
            pass


    def refine(self):

        for frag_id in range(self.N_frag):
            frag        = self.frag_list[frag_id]
            mol_id_list = self.frag2mol[frag_id]

            self.frag2mol_mapping.append(list())

            for mol_id in mol_id_list:
                mol  = self.mol_list[mol_id]

                self.qp.makeDummiesQueries = True
                mol  = Chem.AdjustQueryProperties(mol, self.qp)
                frag = Chem.AdjustQueryProperties(frag, self.qp)

                matches = mol.GetSubstructMatches(frag, useChirality=True)
                if len(matches)>0:
                    self.frag2mol_mapping[-1].append(list(matches[0]))
                else:
                    self.frag2mol_mapping[-1].append(list())