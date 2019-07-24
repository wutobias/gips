import numpy as np
from scipy.spatial import distance as scipy_distance
import copy
from gips.utils.constants import DEG2RAD, RAD2DEG
from gips.utils.misc import check_factor
from gips.grid_solvent._spatial_ext import sasa_vol_ext
from gips.grid_solvent._spatial_ext import sasa_grid_ext
from gips.grid_solvent._spatial_ext import sasa_softgrid_ext

from gips import FLOAT
from gips import DOUBLE

### Written by Tobias Wulsdorf @ Klebe Lab, Marburg University
### tobias.wulsdorf@gmail.com

def make_grid(arrays, out=None):
    """
    !!! Adapted from:
    !!! http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays

    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> make_grid(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
            [1, 4, 7],
            [1, 5, 6],
            [1, 5, 7],
            [2, 4, 6],
            [2, 4, 7],
            [2, 5, 6],
            [2, 5, 7],
            [3, 4, 6],
            [3, 4, 7],
            [3, 5, 6],
            [3, 5, 7]])

    """
    arrays = [np.asarray(x) for x in arrays]
    dtype  = arrays[0].dtype
    n = np.prod([x.size for x in arrays])

    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m        = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)

    if arrays[1:]:
        make_grid(arrays[1:], out=out[0:m,1:])

        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]

    return out


def sasa_softgrid(crds, field, radius, radiusadd=0, solvent=1.4, 
                softness=1., cutoff=4., verbose=False):

    vol_grid = sasa_softgrid_ext(field.frac2real,
                             field.origin,
                             field.bins,
                             np.array(crds),
                             np.array(radius),
                             radiusadd,
                             solvent,
                             softness,
                             cutoff,
                             int(verbose))

    return vol_grid


def sasa_grid(crds, field, radius, radiusadd=[0], solvent=1.4, verbose=False):

    vol_grid = sasa_grid_ext(field.frac2real,
                             field.origin,
                             field.bins,
                             np.array(crds),
                             np.array(radius),
                             np.array(radiusadd),
                             solvent, 
                             int(verbose))

    return vol_grid


def sasa_vol(crds, field, radius, solvent=1.4, usepy=False, verbose=False):

    ### Calculate a minimal bounding box that just fits the atoms
    ### plus the solvent radius plut some buffer
    crds_max      = np.copy(crds)
    crds_max[:,0] = crds_max[:,0]+radius+solvent
    crds_max[:,1] = crds_max[:,1]+radius+solvent
    crds_max[:,2] = crds_max[:,2]+radius+solvent

    crds_min      = np.copy(crds)
    crds_min[:,0] = crds_min[:,0]-radius-solvent
    crds_min[:,1] = crds_min[:,1]-radius-solvent
    crds_min[:,2] = crds_min[:,2]-radius-solvent

    crds_frac_max = field.get_frac(crds_max)
    crds_frac_min = field.get_frac(crds_min)
    bbox_max  = np.rint(np.max(crds_frac_max, axis=0))+1.
    bbox_min  = np.rint(np.min(crds_frac_min, axis=0))-1.
    bbox_max  = bbox_max.astype(int)
    bbox_min  = bbox_min.astype(int)

    bbox_edge_x = np.zeros((bbox_max[0]-bbox_min[0]+1,3), dtype=DOUBLE)
    bbox_edge_y = np.zeros((bbox_max[1]-bbox_min[1]+1,3), dtype=DOUBLE)
    bbox_edge_z = np.zeros((bbox_max[2]-bbox_min[2]+1,3), dtype=DOUBLE)

    bbox_edge_x[:,0] = np.arange(bbox_min[0], bbox_max[0]+1)
    bbox_edge_y[:,1] = np.arange(bbox_min[1], bbox_max[1]+1)
    bbox_edge_z[:,2] = np.arange(bbox_min[2], bbox_max[2]+1)

    bbox_crds_frac = make_grid((bbox_edge_x,
                                bbox_edge_y,
                                bbox_edge_z))

    bbox_edge_x = field.get_real(bbox_edge_x)
    bbox_edge_y = field.get_real(bbox_edge_y)
    bbox_edge_z = field.get_real(bbox_edge_z)

    bbox_crds_real = make_grid((bbox_edge_x,
                                bbox_edge_y,
                                bbox_edge_z))

    if usepy:
    ### This is the python routine
        radius     += solvent
        radius      = radius**2
        dists       = scipy_distance.cdist(bbox_crds_real, crds, metric="sqeuclidean")
        valids      = np.where(dists<radius)[0]
        valids      = np.unique(valids)

    else:
        ### This is the C routine
        valids      = sasa_vol_ext(bbox_crds_real, crds, radius, solvent, int(verbose))

    return bbox_crds_frac[valids]    


def merge_gist(gistlist, overlap=1):

    N_gist    = len(gistlist)
    delta_ref = gistlist[0].delta

    gist_big  = copy.copy(gistlist[0])

    lower_edge  = np.zeros((N_gist, 3))
    upper_edge  = np.zeros((N_gist, 3))

    for i, gist in enumerate(gistlist):
        delta_diff = np.abs(gist.delta-delta_ref)
        if not np.all(delta_diff<0.001):
            raise IOError("All gist objects must have same delta \
but gist object %d has delta %s, where gist object 0 has delta %s" %(i, gist.delta, delta_ref))

        lower_edge[i,:] = gist.get_real(np.zeros(3))
        upper_edge[i,:] = gist.get_real(gist.bins)

    gist_big.origin = np.min(lower_edge, axis=0)
    gist_big.bins   = np.rint(gist_big.get_frac(np.max(upper_edge, axis=0))).astype(int)

    gist_big.Pop            = np.zeros(gist_big.bins, dtype=float)
    gist_big.gO             = np.zeros(gist_big.bins, dtype=float)
    gist_big.gH             = np.zeros(gist_big.bins, dtype=float)
    gist_big.dTStrans_dens  = np.zeros(gist_big.bins, dtype=float)
    gist_big.dTStrans_norm  = np.zeros(gist_big.bins, dtype=float)
    gist_big.dTSorient_dens = np.zeros(gist_big.bins, dtype=float)
    gist_big.dTSorient_norm = np.zeros(gist_big.bins, dtype=float)
    if gist_big.gist17:
        gist_big.dTSsix_dens = np.zeros(gist_big.bins, dtype=float)
        gist_big.dTSsix_norm = np.zeros(gist_big.bins, dtype=float)
    gist_big.Esw_dens       = np.zeros(gist_big.bins, dtype=float)
    gist_big.Esw_norm       = np.zeros(gist_big.bins, dtype=float)
    gist_big.Eww_dens       = np.zeros(gist_big.bins, dtype=float)
    gist_big.Eww_norm_unref = np.zeros(gist_big.bins, dtype=float)
    gist_big.Dipole_x_dens  = np.zeros(gist_big.bins, dtype=float)
    gist_big.Dipole_y_dens  = np.zeros(gist_big.bins, dtype=float)
    gist_big.Dipole_z_dens  = np.zeros(gist_big.bins, dtype=float)
    gist_big.Dipole_dens    = np.zeros(gist_big.bins, dtype=float)
    gist_big.Neighbor_dens  = np.zeros(gist_big.bins, dtype=float)
    gist_big.Neighbor_norm  = np.zeros(gist_big.bins, dtype=float)
    gist_big.Order_norm     = np.zeros(gist_big.bins, dtype=float)

    for i, gist in enumerate(gistlist):

        lower_edge_real = gist.get_real(np.zeros(3))
        upper_edge_real = gist.get_real(gist.bins)

        lower_gist_big = np.rint(gist_big.get_frac(lower_edge_real)).astype(int)
        upper_gist_big = np.rint(gist_big.get_frac(upper_edge_real)).astype(int)

        gist_big.Pop[   lower_gist_big[0]:upper_gist_big[0],
                        lower_gist_big[1]:upper_gist_big[1],
                        lower_gist_big[2]:upper_gist_big[2] ] = gist.Pop

        gist_big.gO[ lower_gist_big[0]:upper_gist_big[0],
                     lower_gist_big[1]:upper_gist_big[1],
                     lower_gist_big[2]:upper_gist_big[2] ] = gist.gO

        gist_big.gH[ lower_gist_big[0]:upper_gist_big[0],
                     lower_gist_big[1]:upper_gist_big[1],
                     lower_gist_big[2]:upper_gist_big[2] ] = gist.gH

        gist_big.dTStrans_dens[ lower_gist_big[0]:upper_gist_big[0],
                                lower_gist_big[1]:upper_gist_big[1],
                                lower_gist_big[2]:upper_gist_big[2] ] = gist.dTStrans_dens

        gist_big.dTStrans_norm[ lower_gist_big[0]:upper_gist_big[0],
                                lower_gist_big[1]:upper_gist_big[1],
                                lower_gist_big[2]:upper_gist_big[2] ] = gist.dTStrans_norm

        gist_big.dTSorient_dens[ lower_gist_big[0]:upper_gist_big[0],
                                 lower_gist_big[1]:upper_gist_big[1],
                                 lower_gist_big[2]:upper_gist_big[2] ] = gist.dTSorient_dens

        gist_big.dTSorient_norm[ lower_gist_big[0]:upper_gist_big[0],
                                 lower_gist_big[1]:upper_gist_big[1],
                                 lower_gist_big[2]:upper_gist_big[2] ] = gist.dTSorient_norm

        if gist_big.gist17:
            gist_big.dTSsix_dens[ lower_gist_big[0]:upper_gist_big[0],
                                  lower_gist_big[1]:upper_gist_big[1],
                                  lower_gist_big[2]:upper_gist_big[2] ] = gist.dTSsix_dens
            
            gist_big.dTSsix_norm[ lower_gist_big[0]:upper_gist_big[0],
                                  lower_gist_big[1]:upper_gist_big[1],
                                  lower_gist_big[2]:upper_gist_big[2] ] = gist.dTSsix_norm
        
        gist_big.Esw_dens[ lower_gist_big[0]:upper_gist_big[0],
                           lower_gist_big[1]:upper_gist_big[1],
                           lower_gist_big[2]:upper_gist_big[2] ] = gist.Esw_dens
        
        gist_big.Esw_norm[ lower_gist_big[0]:upper_gist_big[0],
                           lower_gist_big[1]:upper_gist_big[1],
                           lower_gist_big[2]:upper_gist_big[2] ] = gist.Esw_norm
        
        gist_big.Eww_dens[ lower_gist_big[0]:upper_gist_big[0],
                           lower_gist_big[1]:upper_gist_big[1],
                           lower_gist_big[2]:upper_gist_big[2] ] = gist.Eww_dens
        
        gist_big.Eww_norm_unref[ lower_gist_big[0]:upper_gist_big[0],
                                 lower_gist_big[1]:upper_gist_big[1],
                                 lower_gist_big[2]:upper_gist_big[2] ] = gist.Eww_norm_unref
        
        gist_big.Dipole_x_dens[ lower_gist_big[0]:upper_gist_big[0],
                                lower_gist_big[1]:upper_gist_big[1],
                                lower_gist_big[2]:upper_gist_big[2] ] = gist.Dipole_x_dens
        
        gist_big.Dipole_y_dens[ lower_gist_big[0]:upper_gist_big[0],
                                lower_gist_big[1]:upper_gist_big[1],
                                lower_gist_big[2]:upper_gist_big[2] ] = gist.Dipole_y_dens
        
        gist_big.Dipole_z_dens[ lower_gist_big[0]:upper_gist_big[0],
                                lower_gist_big[1]:upper_gist_big[1],
                                lower_gist_big[2]:upper_gist_big[2] ] = gist.Dipole_z_dens
        
        gist_big.Dipole_dens[ lower_gist_big[0]:upper_gist_big[0],
                              lower_gist_big[1]:upper_gist_big[1],
                              lower_gist_big[2]:upper_gist_big[2] ] = gist.Dipole_dens
        
        gist_big.Neighbor_dens[ lower_gist_big[0]:upper_gist_big[0],
                                lower_gist_big[1]:upper_gist_big[1],
                                lower_gist_big[2]:upper_gist_big[2] ] = gist.Neighbor_dens
        
        gist_big.Neighbor_norm[ lower_gist_big[0]:upper_gist_big[0],
                                lower_gist_big[1]:upper_gist_big[1],
                                lower_gist_big[2]:upper_gist_big[2] ] = gist.Neighbor_norm
        
        gist_big.Order_norm[ lower_gist_big[0]:upper_gist_big[0],
                             lower_gist_big[1]:upper_gist_big[1],
                             lower_gist_big[2]:upper_gist_big[2] ] = gist.Order_norm

    return gist_big


class field(object):

    """
    This is a class for operation on generic
    scalar fields. 
    They are described in cartesian as well as 
    in fractional space. That means that we need 
    origin, frac2real/real2frac matrix(vector)
    and/or grid spacing vectors vector.
    """

    def __init__(self, Bins, Frac2Real=None, Delta=None, Origin=None, Center=None):

        if type(Frac2Real) == type(None) and type(Delta) == type(None):
            raise ValueError("Must provide Frac2Real or Delta.")

        if type(Frac2Real) != type(None) and type(Delta) != type(None):
            raise ValueError("Must provide either Frac2Real or Delta.")

        if type(Frac2Real) == type(None):

            self.delta     = Delta
            self.frac2real = np.eye(3,3) * self.delta

        else:

            self.frac2real = Frac2Real
            self.delta     = np.linalg.norm(self.frac2real, axis=0)

        self.real2frac = np.linalg.inv(self.frac2real)
        self.bins      = Bins

        self.rotation_matrix    = np.eye(3,3)
        self.translation_vector = np.zeros(3)


        if type(Origin) == type(None) and type(Center) == type(None):
            raise ValueError("Must provide origin or Center.")

        if type(Origin) != type(None) and type(Center) != type(None):
            raise ValueError("Must provide either origin or center.")

        if type(Center) == type(None):

            self.origin    = Origin
            self.center    = self.get_real(self.bins/2)

        else:

            self.center    = Center
            #First we need an auxiliary origin at (0,0,0)
            self.origin    = np.zeros(3)
            #Second translate origin according center displacement
            self.origin    = self.center - self.get_real(self.bins/2)

        self.dim = np.array([ np.linalg.norm(self.get_real([self.bins[0], 0., 0.])-self.origin),
                              np.linalg.norm(self.get_real([0., self.bins[1], 0.])-self.origin),
                              np.linalg.norm(self.get_real([0., 0., self.bins[2]])-self.origin)
                              ])


    def translate(self, vector=np.zeros(3)):

        """
        Translatation vector of unit cell origin
        """

        self.translation_vector += vector


    def rotate(self, matrix=np.eye(3,3)):

        """ 
        Rotate the unit cell vectors. 
        """

        rotate_check(matrix)
        self.rotation_matrix = matrix.dot(self.rotation_matrix)


    def translate_global(self, vector=np.zeros(3)):

        """
        Translate global coordinate system
        along vector.
        """

        self.origin += vector


    def rotate_global(self, reference_point=np.zeros(3), matrix=np.eye(3,3)):

        """
        Rotate global coordinate system around
        reference point.
        """

        rotate_check(matrix)
        self.origin = do_rotation(self.origin, reference_point, matrix)
        self.rotate(matrix)
        self.translation_vector = do_rotation(self.translation_vector, np.zeros(3), matrix)


    def get_nice_frac2real(self):

        return self.rotation_matrix.dot(self.frac2real)


    def get_nice_real2frac(self):

        return np.linalg.inv(self.get_nice_frac2real())


    def get_voxel_volume(self):

        """
        Returns the volume per grid voxel.
        """

        return np.absolute(np.cross(self.frac2real[:,0], self.frac2real[:,1]).dot(self.frac2real[:,2]))


    def get_frac(self, real_array):

        #Convert to initial real space by inverse translation and rotation
        initial_reals = do_rotation(real_array, self.origin + self.translation_vector, np.linalg.inv(self.rotation_matrix))
        #Remove origin
        initial_reals -= (self.origin + self.translation_vector)
        #Convert to initial fractional space
        return initial_reals.dot(self.real2frac)


    def get_real(self, frac_array):

        #Convert to real space
        reals = np.array(frac_array).dot(self.frac2real)
        #Perform rotation translation
        return do_rotation(reals, np.zeros(3), self.rotation_matrix) + self.origin + self.translation_vector


    def get_centers(self):

        return self.get_real(make_grid((np.arange(self.bins[0]),\
                                        np.arange(self.bins[1]),\
                                        np.arange(self.bins[2]))))

    def get_centers_real(self):

        return self.get_centers()


    def get_centers_frac(self):

        return make_grid((np.arange(self.bins[0]),\
                          np.arange(self.bins[1]),\
                          np.arange(self.bins[2])))


def guess_field(crds, delta=np.array([0.5,0.5,0.5])):

    _c = np.mean(crds, axis=0)

    _min = np.min((_c - crds), axis=0)
    _max = np.max((_c - crds), axis=0)

    _b = np.rint(np.abs(_max - _min)/delta + (5.0 / delta) )

    del _min, _max

    return field(Bins=_b, Delta=delta, Center=_c)


def rotate_check(matrix):

    if not (0.99 < np.linalg.det(matrix) < 1.01):

        raise Warning("Warning: Determinant of rotation matrix is %s. Should be close to +1.0." %np.linalg.det(matrix))


def do_rotation (crds, origin, rot_mat):

    return (crds - origin).dot(rot_mat) + origin

    
def set_quaternion(q_dist, theta, phi, psi):
    
    """
    Retrieve the quaternion coordinates from Euler angles theta, phi and psi.
    """

    cos_theta    = np.cos(.5*theta)
    sin_theta    = np.sin(.5*theta)
    cos_phipsi   = np.cos(.5*(phi+psi))
    sin_phipsi   = np.sin(.5*(phi+psi))

    q_dist[:, 0] = cos_theta*cos_phipsi
    q_dist[:, 1] = sin_theta*cos_phipsi
    q_dist[:, 2] = sin_theta*sin_phipsi
    q_dist[:, 3] = cos_theta*sin_phipsi

    q_norm    = np.linalg.norm(q_dist, axis=1)
    q_dist    = np.einsum('ij,i->ij', q_dist, 1./q_norm)

def set_euler(O_crds, H1_crds, H2_crds, xx, yy, zz, theta, phi, psi):
    
    """
    Retrieve Euler angles for water molecules with oxygen coordinates
    O_crds and hydrogen coordinates H1_crds,H2_crds.
    The frame of reference coordinate axes are given by xx, yy and zz.
    """
    ### For definitions of euler angle see Supporting Material in
    ### E. P. Raman, A. D. MacKerell, J. Am. Chem. Soc. 2015, 150127114301002.
    ### Further reading on Euler angle calculation:
    ### J. Diebel, 2006, "Representing attitude: Euler angles, unit quaternions, and rotation vectors".

    ### water coordinate system (all normalized):
    ### O-H1          --> X
    ### (O-H1)x(O-H2) --> Z
    ### XxZ           --> Y
    ###
    ### Line of nodes 'N' is the vector on the intersection between the xy plane of
    ### the water coordinate system and the lab coordinate system.

    xx1_wat      = H1_crds - O_crds
    xx1_norm     = np.linalg.norm(xx1_wat, axis=1)
    xx1_wat      = np.einsum('ij,i->ij', xx1_wat, 1./xx1_norm)

    xx2_wat      = H2_crds - O_crds
    xx2_norm     = np.linalg.norm(xx2_wat, axis=1)
    xx2_wat      = np.einsum('ij,i->ij', xx2_wat, 1./xx2_norm)

    zz_wat       = np.cross(xx1_wat, xx2_wat)
    zz_norm      = np.linalg.norm(zz_wat, axis=1)
    zz_wat       = np.einsum('ij,i->ij', zz_wat, 1./zz_norm)

    yy_wat       = np.cross(xx1_wat, zz_wat)
    yy_norm      = np.linalg.norm(yy_wat, axis=1)
    yy_wat       = np.einsum('ij,i->ij', yy_wat, 1./yy_norm)

    N_xy         = np.cross(zz, zz_wat)
    N_xy_norm    = np.linalg.norm(N_xy, axis=1)
    N_xy         = np.einsum('ij,i->ij', N_xy, 1./N_xy_norm)

    ### Angle theta
    ### Angle between zz-axis vector and z-axis of the lab coordinate system.
    theta_dot    = np.einsum('ij,j->i', zz_wat, zz)

    ### Angle phi
    ### Angle between line of nodes (xy lab frame / xy water frame) and xx-axis of lab frame
    phi_dot      = np.einsum('ij,j->i', N_xy, xx)

    ### Angle psi
    ### Angle between line of nodes (xy lab frame / xy water frame) and xx-axis in wat frame
    psi_dot      = np.einsum('ij,ij->i', xx1_wat, N_xy)

    ### dot products should be within [-1,+1]. However they might be slightly out of bounds (due to
    ### round-offs I guess). So bring them into bounds.
    theta_dot_check_upper     = np.where(theta_dot >  1. )
    theta_dot_check_lower     = np.where(theta_dot < -1. )
    phi_dot_check_upper       = np.where(phi_dot   >  1. )
    phi_dot_check_lower       = np.where(phi_dot   < -1. )
    psi_dot_check_upper       = np.where(psi_dot   >  1. )
    psi_dot_check_lower       = np.where(psi_dot   < -1. )

    theta_dot[theta_dot_check_upper] =  1.
    theta_dot[theta_dot_check_lower] = -1.
    phi_dot[phi_dot_check_upper]     =  1.
    phi_dot[phi_dot_check_lower]     = -1.
    psi_dot[psi_dot_check_upper]     =  1.
    psi_dot[psi_dot_check_lower]     = -1.

    ### Calculate angle from arccos function
    theta[:] = np.arccos(theta_dot)
    phi[:]   = np.arccos(phi_dot)
    psi[:]   = np.arccos(psi_dot)

    ### We must flip some of the psi and phi angles. The problem is, that we cannot
    ### make a difference between -psi and +psi, which however are two physically 
    ### different observations.
    ###
    ### Phi Correction
    zz_tmp_labframe            = np.cross(N_xy, xx)
    zz_tmp_labframe_norm       = np.linalg.norm(zz_tmp_labframe, axis=1)
    zz_tmp_labframe            = np.einsum('ij,i->ij', zz_tmp_labframe, 1./zz_tmp_labframe_norm)
    zz_tmp_labframe_dot        = np.einsum('ij,j->i', zz_tmp_labframe, zz)
    phi[np.where(zz_tmp_labframe_dot < 0.)[0]] *= -1.

    ### Psi Correction
    test1                      = np.copy(psi)
    zz_tmp_watframe            = np.cross(N_xy, xx1_wat)
    zz_tmp_watframe_norm       = np.linalg.norm(zz_tmp_watframe, axis=1)
    zz_tmp_watframe            = np.einsum('ij,i->ij', zz_tmp_watframe, 1./zz_tmp_watframe_norm)
    zz_tmp_watframe_dot        = np.einsum('ij,ij->i', zz_tmp_watframe, zz_wat)
    psi[np.where(zz_tmp_watframe_dot < 0.)[0]] *= -1.
    
    theta *= RAD2DEG
    phi   *= RAD2DEG
    psi   *= RAD2DEG

def make_grid(arrays, out=None):
    """
    !!! Adapted from:
    !!! http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays

    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> make_grid(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
            [1, 4, 7],
            [1, 5, 6],
            [1, 5, 7],
            [2, 4, 6],
            [2, 4, 7],
            [2, 5, 6],
            [2, 5, 7],
            [3, 4, 6],
            [3, 4, 7],
            [3, 5, 6],
            [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]

    dtype  = arrays[0].dtype

    n = np.prod([x.size for x in arrays])

    if out is None:

        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size

    out[:,0] = np.repeat(arrays[0], m)

    if arrays[1:]:

        make_grid(arrays[1:], out=out[0:m,1:])

        for j in xrange(1, arrays[0].size):

            out[j*m:(j+1)*m,1:] = out[0:m,1:]

    return out


def bounding_edge_frac(frac_structure, delta=np.ones(3), _buffer=0., verbose=False):

    bounding_min = np.array( [ np.min(frac_structure[:,0]),
                               np.min(frac_structure[:,1]),
                               np.min(frac_structure[:,2]) ], dtype=int )

    bounding_max = np.array( [ np.max(frac_structure[:,0]),
                               np.max(frac_structure[:,1]),
                               np.max(frac_structure[:,2]) ], dtype=int )

    bounding_min -= int(np.round(_buffer))
    bounding_max += int(np.round(_buffer))

    if verbose:
        print "Bounding min. ", bounding_min
        print "Bounding max. ", bounding_max
        print np.arange(bounding_min[2], bounding_max[2]+1, delta[2], dtype=int )

    return bounding_min, bounding_max


def bounding_box_frac(frac_structure, delta=np.ones(3), _buffer=0., verbose=False):

    """
    Input is structure in cart. or frac. coordinates as
    nx3 array (n= number of coordinates).
    Output is coordinate meshgrid array with coordinates of
    bounding box lattice as integers.
    """

    bounding_min, bounding_max = bounding_edge_frac(frac_structure, 
                                                    delta=np.ones(3), 
                                                    _buffer=0., 
                                                    verbose=False)

    return  make_grid ( [ np.arange(bounding_min[0], bounding_max[0]+1, delta[0], dtype=int ),
                          np.arange(bounding_min[1], bounding_max[1]+1, delta[1], dtype=int ),
                          np.arange(bounding_min[2], bounding_max[2]+1, delta[2], dtype=int ) ] )


def py_axis_paral(query, target, verbose=0):

    vec_paral = np.zeros(3)
    n_crds = query.shape[0]
    n_comb = 0
    for idx1 in range(n_crds):
        for idx2 in range(n_crds):
            if idx1 >= idx2:
                continue
            if verbose:
                print( "Iteration (%d %d)" %(idx1, idx2))
            v  = query[idx2] - query[idx1]
            v  /= np.linalg.norm(v)
            # First vector found gives reference orientation
            if idx1 == 0 and idx2 == 1:
                ref = v
            else:
                # Reverse vector if direction opposite to reference vector
                if np.dot(ref, v) < 0:
                    v *= -1
            vec_paral += v
            n_comb += 1

    vec_paral /= float(n_comb)
    vec_paral /= np.linalg.norm(vec_paral)

    target[0] = vec_paral[0]
    target[1] = vec_paral[1]
    target[2] = vec_paral[2]

def py_axis_ortho(query, target, verbose=0):

    vec_ortho = np.zeros(3, dtype=float)
    n_crds    = query.shape[0]
    n_comb    = 0

    for idx1 in range(n_crds):
        for idx2 in range(n_crds):
            if idx1 >= idx2:
                continue
            for idx3 in range(n_crds):
                if idx2 >= idx3:
                    continue
                v1  = query[idx3] - query[idx1]
                v2  = query[idx2] - query[idx1]
                n   = np.cross(v1, v2)
                n  /= np.linalg.norm(n)

                #First vector found gives reference orientation
                if idx2 == 1 and idx3 == 2:
                    ref = n
                else:
                    # Reverse vector if direction opposite to reference vector
                    if np.dot(ref, n) < 0:
                        n *= -1

                vec_ortho += n
                n_comb += 1

    # Bionominal coefficient, this is "ziehen ohne zuruecklegen ohne beachtung der reihenfolge"
    vec_ortho /= n_comb
    vec_ortho /= np.linalg.norm(vec_ortho)

    target[0] = vec_ortho[0]
    target[1] = vec_ortho[1]
    target[2] = vec_ortho[2]

def quaternion_symmetric_normed_difference(q1,q2):
    
    __lower  = np.linalg.norm(q1-q2, axis=-1)
    __upper  = np.linalg.norm(q1+q2, axis=-1)
    
    return 2.*np.minimum(__lower, __upper)


def euler_difference2(x,y):
    
    ### x[0]: Theta
    ### x[1]: phi
    ### x[2]: Psi

    ### Euler coordinates must be in radian units!!
    
    diff_crds        = np.zeros_like(x)
    
    diff_crds[0]     = np.absolute(np.cos(x[0])-np.cos(y[0]))
    __phi_diff_lower = np.absolute(x[1]-y[1])
    __phi_diff_upper = 2.*np.pi - __phi_diff_lower
    __psi_diff_lower = np.absolute(x[2]-y[2])
    __psi_diff_upper = 2.*np.pi - __psi_diff_lower

    diff_crds[1] = np.minimum(__phi_diff_lower, __phi_diff_upper)
    diff_crds[2] = np.minimum(__psi_diff_lower, __psi_diff_upper)

    return np.linalg.norm(diff_crds, axis=0)