import numpy as np
import copy

from gips.utils.read_write import write_files
from gips.mapout.clustering import clustering_cutoff
from gips.gistmodel.post_processing import post_processing

from gips import FLOAT
from gips import DOUBLE

class mapout_maps(post_processing):

    def __init__(self, fitter, x, pairs=False, prefix=None):

        super(mapout_maps, self).__init__(fitter, x, pairs, prefix)


    def write(self, E_grid_val, S_grid_val, gv_grid_val, field, pmd_object, prefix, **kwargs):

        ### Writes out all values on the map within cutoff parameters.
        ### Keep this for debugging purposes.
        E_grid_valids = np.where((np.abs(E_grid_val)>0.) * (np.abs(gv_grid_val)>0.))
        S_grid_valids = np.where((np.abs(S_grid_val)>0.) * (np.abs(gv_grid_val)>0.))

        E_grid_crds = np.zeros((E_grid_valids[0].shape[0],3), dtype=DOUBLE)
        S_grid_crds = np.zeros((S_grid_valids[0].shape[0],3), dtype=DOUBLE)

        E_grid_crds[:,0] = E_grid_valids[0]
        E_grid_crds[:,1] = E_grid_valids[1]
        E_grid_crds[:,2] = E_grid_valids[2]

        S_grid_crds[:,0] = S_grid_valids[0]
        S_grid_crds[:,1] = S_grid_valids[1]
        S_grid_crds[:,2] = S_grid_valids[2]

        E_real = field.get_real(E_grid_crds)
        S_real = field.get_real(S_grid_crds)

        ### write_files constructor:
        ### ~~~~~~~~~~~~~~~~~~~~~~~~
        ### def __init__(self, Delta=None, Frac2Real=None, Bins=None, Origin=None, \
        ###             Value=None, XYZ=None, X=None, Y=None, Z=None, Format='PDB', \
        ###             Filename=None, Nan_fill=-1.0):

        write_files(Value=E_grid_val[E_grid_valids], XYZ=E_real, Format='PDB',
                    Filename="%s%s.grid.energy.pdb" %(self.prefix, prefix), Nan_fill=np.nan)

        write_files(Value=S_grid_val[S_grid_valids], XYZ=S_real, Format='PDB',
                    Filename="%s%s.grid.entropy.pdb" %(self.prefix, prefix), Nan_fill=np.nan)

        pmd_object.symmetry = None
        pmd_object.save("%s%s.strc.pdb" %(self.prefix, prefix), overwrite=True)

        E_cluster_centers, g_cluster_values, E_cluster_values = clustering_cutoff(field, 
                                                                                gv_grid_val,
                                                                                E_grid_val,
                                                                                np.min(gv_grid_val),
                                                                                1.4)
        E_cluster_centers = np.array(E_cluster_centers)
        E_cluster_values  = np.array(E_cluster_values)
        write_files(Value=E_cluster_values, XYZ=E_cluster_centers, Format='PDB',
                    Filename="%s%s.sites.energy.pdb" %(self.prefix, prefix), Nan_fill=np.nan)

        S_cluster_centers, g_cluster_values, S_cluster_values = clustering_cutoff(field,
                                                                                gv_grid_val,
                                                                                S_grid_val,
                                                                                np.min(gv_grid_val),
                                                                                1.4)
        S_cluster_centers = np.array(S_cluster_centers)
        S_cluster_values  = np.array(S_cluster_values)
        write_files(Value=S_cluster_values, XYZ=S_cluster_centers, Format='PDB',
                    Filename="%s%s.sites.entropy.pdb" %(self.prefix, prefix), Nan_fill=np.nan)
