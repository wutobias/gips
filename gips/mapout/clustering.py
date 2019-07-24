import numpy as np
from scipy import spatial as spatial_scipy

def clustering_cutoff(field, values_A, values_B, cutoff, size):

    cluster_centers  = list()
    cluster_values_A = list()
    cluster_values_B = list()
    valids          = np.where(values_A > cutoff)
    valuesA_copy    = np.copy(values_A)
    valuesB_copy    = np.copy(values_B)
    size2           = size**2

    while valids[0].shape[0] > 0:

        local_max_x = np.where(valuesA_copy == np.amax(valuesA_copy))
        local_max_x = field.get_real(np.array([ local_max_x[0][0],
                                                local_max_x[1][0],
                                                local_max_x[2][0]]))

        ### Reshape valids array such that
        ### [[0,0,0],
        ###  [1,1,0]
        ###  [0,1,0],
        ###  ...,
        ###  ]
        valids_reshape = np.stack([valids[0], valids[1], valids[2]], axis=1)
        d2             = spatial_scipy.distance.cdist(field.get_real(valids_reshape), 
                                                        np.array([local_max_x]),
                                                        metric="sqeuclidean")
        invalids = np.where(d2 <= size2)[0]

        cluster_values_A.append(np.sum(valuesA_copy[valids_reshape[invalids][:,0], 
                                                    valids_reshape[invalids][:,1], 
                                                    valids_reshape[invalids][:,2]]))

        cluster_values_B.append(np.sum(valuesB_copy[valids_reshape[invalids][:,0], 
                                                    valids_reshape[invalids][:,1], 
                                                    valids_reshape[invalids][:,2]]))

        valuesA_copy[valids_reshape[invalids][:,0], 
                    valids_reshape[invalids][:,1], 
                    valids_reshape[invalids][:,2]] = 0.

        valuesB_copy[valids_reshape[invalids][:,0], 
                    valids_reshape[invalids][:,1], 
                    valids_reshape[invalids][:,2]] = 0.

        cluster_centers.append(local_max_x)

        valids = np.where(valuesA_copy > cutoff)

    return cluster_centers, cluster_values_A, cluster_values_B
