import numpy as np
from collections import OrderedDict

from gips.grid_solvent.spatial import field

from gips import FLOAT
from gips import DOUBLE

class gist(field):

### Some notes:
### ~~~~~~~~~~~
### - all coordinates are internally treated as origin-free, 
###   therefore the coordinate systems' origin is always at (0,0,0)
### - coordinate systems are right-handed
### - global translations affect everything, non-global translations
###   effect grid only
### - ...same holds for rotations.
### - all rotations and translations affect real space only
### - the effect of rotations and translations is only calculated when
###   real space information is retrieved
### - frac2real matrix as well as its inverse, internally remain unchanged,
###   true frac2real (or its inverse) is available via get_nice_frac2real()
### - all rotations are performed counter-clockwise

    def __init__(self, Bins, Origin, Delta):

        super(gist, self).__init__(Bins=Bins,
                                   Origin=Origin,
                                   Delta=Delta)

        self.Pop            = np.zeros(self.bins, dtype=FLOAT)
        self.gO             = np.zeros(self.bins, dtype=FLOAT)
        self.gH             = np.zeros(self.bins, dtype=FLOAT)
        self.dTStrans_dens  = np.zeros(self.bins, dtype=FLOAT)
        self.dTStrans_norm  = np.zeros(self.bins, dtype=FLOAT)
        self.dTSorient_dens = np.zeros(self.bins, dtype=FLOAT)
        self.dTSorient_norm = np.zeros(self.bins, dtype=FLOAT)
        self.dTSsix_dens    = np.zeros(self.bins, dtype=FLOAT)
        self.dTSsix_norm    = np.zeros(self.bins, dtype=FLOAT)
        self.Esw_dens       = np.zeros(self.bins, dtype=FLOAT)
        self.Esw_norm       = np.zeros(self.bins, dtype=FLOAT)
        self.Eww_dens       = np.zeros(self.bins, dtype=FLOAT)
        self.Eww_norm_unref = np.zeros(self.bins, dtype=FLOAT)
        self.Dipole_x_dens  = np.zeros(self.bins, dtype=FLOAT)
        self.Dipole_y_dens  = np.zeros(self.bins, dtype=FLOAT)
        self.Dipole_z_dens  = np.zeros(self.bins, dtype=FLOAT)
        self.Dipole_dens    = np.zeros(self.bins, dtype=FLOAT)
        self.Neighbor_dens  = np.zeros(self.bins, dtype=FLOAT)
        self.Neighbor_norm  = np.zeros(self.bins, dtype=FLOAT)
        self.Order_norm     = np.zeros(self.bins, dtype=FLOAT)


    def cut_round_center(self, bins):

        __doc__="""
        bins is the new self.bins attribute of this class.
        All other attributes (i.e. all scalar fields containing Gist data)
        will be reshaped and transformed such that the position of the
        grid center is preserved. Thereby all edges are cut symmetrically around
        the center.

        Example:

        bins=[20,20,20] with self.bins=[50,50,50]

        Here, the original self.bins will be reduced to bins. Thereby, the xedge,
        yedge and zedge will be truncated such that [0:15] and [35:49] are cut away.
        """

        if bins.shape[0] != 3:
          print "Target bins array must habe shape (3,)"
        elif self.bins[0] < bins[0] or self.bins[1] < bins[1] or self.bins[2] < bins[2]:
          print "Target bins array must be smaller than original one."
        else:
            cut      = (self.bins - bins)/2
            cut_bins = np.array([[cut[0], cut[0]],
                                 [cut[1], cut[1]],
                                 [cut[2], cut[2]]])
            self.cut(cut_bins)
    

    def cut(self, cut_bins):

        __doc__="""
        cut_bins must be an array of shape (3,2), with
        [[lower_cut_x, upper_cut_x],
        [lower_cut_y, upper_cut_y],
        [lower_cut_z, upper_cut_z]]
        marking the number of bins cut out at lower and upper
        boundaries of each dimension.
        """

        self.bins[0] = self.bins[0] - cut_bins[0,0] - cut_bins[0,1]
        self.bins[1] = self.bins[1] - cut_bins[1,0] - cut_bins[1,1]
        self.bins[2] = self.bins[2] - cut_bins[2,0] - cut_bins[2,1]

        self.origin = np.zeros(3)
        self.origin = self.center - self.get_real(self.bins/2)

        tmp_Pop                 = np.zeros(self.bins, dtype=FLOAT)
        tmp_gO                  = np.zeros(self.bins, dtype=FLOAT)
        tmp_gH                  = np.zeros(self.bins, dtype=FLOAT)
        tmp_dTStrans_dens       = np.zeros(self.bins, dtype=FLOAT)
        tmp_dTStrans_norm       = np.zeros(self.bins, dtype=FLOAT)
        tmp_dTSorient_dens      = np.zeros(self.bins, dtype=FLOAT)
        tmp_dTSorient_norm      = np.zeros(self.bins, dtype=FLOAT)
        tmp_dTSsix_dens         = np.zeros(self.bins, dtype=FLOAT)
        tmp_dTSsix_norm         = np.zeros(self.bins, dtype=FLOAT)
        tmp_Esw_norm            = np.zeros(self.bins, dtype=FLOAT)
        tmp_Esw_dens            = np.zeros(self.bins, dtype=FLOAT)
        tmp_Eww_dens            = np.zeros(self.bins, dtype=FLOAT)
        tmp_Eww_norm_unref      = np.zeros(self.bins, dtype=FLOAT)
        tmp_Dipole_x_dens       = np.zeros(self.bins, dtype=FLOAT)
        tmp_Dipole_y_dens       = np.zeros(self.bins, dtype=FLOAT)
        tmp_Dipole_z_dens       = np.zeros(self.bins, dtype=FLOAT)
        tmp_Dipole_dens         = np.zeros(self.bins, dtype=FLOAT)
        tmp_Neighbor_dens       = np.zeros(self.bins, dtype=FLOAT)
        tmp_Neighbor_norm       = np.zeros(self.bins, dtype=FLOAT)
        tmp_Order_norm          = np.zeros(self.bins, dtype=FLOAT)

        for x_i, x in enumerate(range(cut_bins[0,0], cut_bins[0,0]+self.bins[0])):
            for y_i, y in enumerate(range(cut_bins[1,0], cut_bins[1,0]+self.bins[1])):
                for z_i, z in enumerate(range(cut_bins[2,0], cut_bins[2,0]+self.bins[2])):

                    tmp_Pop                [x_i][y_i][z_i] = self.Pop            [x,y,z]
                    tmp_gO                 [x_i][y_i][z_i] = self.gO             [x,y,z]
                    tmp_gH                 [x_i][y_i][z_i] = self.gH             [x,y,z]
                    tmp_dTStrans_dens      [x_i][y_i][z_i] = self.dTStrans_dens  [x,y,z]
                    tmp_dTStrans_norm      [x_i][y_i][z_i] = self.dTStrans_norm  [x,y,z]
                    tmp_dTSorient_dens     [x_i][y_i][z_i] = self.dTSorient_dens [x,y,z]
                    tmp_dTSorient_norm     [x_i][y_i][z_i] = self.dTSorient_norm [x,y,z]
                    tmp_dTSsix_dens        [x_i][y_i][z_i] = self.dTSsix_dens    [x,y,z]
                    tmp_dTSsix_norm        [x_i][y_i][z_i] = self.dTSsix_norm    [x,y,z]
                    tmp_Esw_dens           [x_i][y_i][z_i] = self.Esw_dens       [x,y,z]
                    tmp_Esw_norm           [x_i][y_i][z_i] = self.Esw_norm       [x,y,z]
                    tmp_Eww_dens           [x_i][y_i][z_i] = self.Eww_dens       [x,y,z]
                    tmp_Eww_norm_unref     [x_i][y_i][z_i] = self.Eww_norm_unref [x,y,z]
                    tmp_Dipole_x_dens      [x_i][y_i][z_i] = self.Dipole_x_dens  [x,y,z]
                    tmp_Dipole_y_dens      [x_i][y_i][z_i] = self.Dipole_y_dens  [x,y,z]
                    tmp_Dipole_z_dens      [x_i][y_i][z_i] = self.Dipole_z_dens  [x,y,z]
                    tmp_Dipole_dens        [x_i][y_i][z_i] = self.Dipole_dens    [x,y,z]
                    tmp_Neighbor_dens      [x_i][y_i][z_i] = self.Neighbor_dens  [x,y,z]
                    tmp_Neighbor_norm      [x_i][y_i][z_i] = self.Neighbor_norm  [x,y,z]
                    tmp_Order_norm         [x_i][y_i][z_i] = self.Order_norm     [x,y,z]

        self.Pop            = np.copy(tmp_Pop)
        self.gO             = np.copy(tmp_gO)
        self.gH             = np.copy(tmp_gH)
        self.dTStrans_dens  = np.copy(tmp_dTStrans_dens)
        self.dTStrans_norm  = np.copy(tmp_dTStrans_norm)
        self.dTSorient_dens = np.copy(tmp_dTSorient_dens)
        self.dTSorient_norm = np.copy(tmp_dTSorient_norm)
        self.dTSsix_dens    = np.copy(tmp_dTSsix_dens)
        self.dTSsix_norm    = np.copy(tmp_dTSsix_norm)
        self.Esw_norm       = np.copy(tmp_Esw_norm)
        self.Esw_dens       = np.copy(tmp_Esw_dens)
        self.Eww_dens       = np.copy(tmp_Eww_dens)
        self.Eww_norm_unref = np.copy(tmp_Eww_norm_unref)
        self.Dipole_x_dens  = np.copy(tmp_Dipole_x_dens)
        self.Dipole_y_dens  = np.copy(tmp_Dipole_y_dens)
        self.Dipole_z_dens  = np.copy(tmp_Dipole_z_dens)
        self.Dipole_dens    = np.copy(tmp_Dipole_dens)
        self.Neighbor_dens  = np.copy(tmp_Neighbor_dens)
        self.Neighbor_norm  = np.copy(tmp_Neighbor_norm)
        self.Order_norm     = np.copy(tmp_Order_norm)

        del tmp_Pop
        del tmp_gO
        del tmp_gH
        del tmp_dTStrans_dens
        del tmp_dTStrans_norm
        del tmp_dTSorient_dens
        del tmp_dTSorient_norm
        del tmp_dTSsix_dens
        del tmp_dTSsix_norm
        del tmp_Esw_norm
        del tmp_Esw_dens
        del tmp_Eww_dens
        del tmp_Eww_norm_unref
        del tmp_Dipole_x_dens
        del tmp_Dipole_y_dens
        del tmp_Dipole_z_dens
        del tmp_Dipole_dens
        del tmp_Neighbor_dens
        del tmp_Neighbor_norm
        del tmp_Order_norm

        super(gist, self).__init__(Bins=self.bins,
                                   Origin=self.origin,
                                   Delta=self.delta)


    def get_nan(self):

        __doc__="""
        Return array that contains True whereever Population array
        is np.nan and False elsewhere.
        """
        tmp = np.zeros(self.bins, dtype=bool)
        tmp[np.isnan(self.Pop)] = True

        return tmp


    def get_pop(self):

        __doc__="""
        Return array that containes True whereever Population array
        is greater than zero.
        """

        tmp = np.zeros(self.bins, dtype=bool)
        tmp[np.where(self.Pop > 0)] = True

        return tmp


###
### The gistlib class is depracted now and is 
### replaced by simple python OrderedDict dictionaries
###
class gistlib(object):

    def __init__(self, Name):

        self.name     = Name

        self.gistN    = 0

        self.gistlist = list()
        self.gistname = list()
        self.gistids  = list()

        self.gistdg   = None
        self.gistdh   = None
        self.gistds   = None

        self.pmd      = list()
        self.sele     = list()


    def add_gist(self, gistobjects, name):

        if isinstance(gistobjects, list):
            for gistobject in gistobjects:
                if isinstance(gistobject, gist):
                    self.gistlist.append(gistobject)
                    self.gistids.append(self.gistN)
                    self.gistname.append(name)
                else:
                    raise TypeError("gistobjects must be of type %s" %gist)

        elif isinstance(gistobjects, gist):
            self.gistlist.append(gistobjects)
            self.gistids.append(self.gistN)
            self.gistname.append(name)
        else:
            raise TypeError("gistobjects must be of type %s" %gist)

        self.gistN += 1


    def get_gist(self, gist_query):

        returnlist = list()

        if type(gist_query)==int:

            for i, gistid in enumerate(self.gistids):
                if gistid==gist_query:
                    returnlist.append(i)

        elif type(gist_query)==str:

            for i, gistname in enumerate(self.gistname):
                if gistname==gist_query:
                    returnlist.append(i)

        else:
            raise TypeError("gist_query must be of type int or str but is of type %s." %type(gist_query))

        return returnlist