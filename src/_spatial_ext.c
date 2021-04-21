/*
### Written by Tobias Wulsdorf @ Klebe Lab, Marburg University
### tobias.wulsdorf@gmail.com
*/

#define _USE_MATH_DEFINES
#include <math.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "numpy/arrayobject.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "Vec.h"
#include <math.h>

/*
/// ------------------------------------------
/// | This routine is completely untested!!! |
/// ------------------------------------------
static PyObject *
_corrected_aff(PyObject *self, PyObject *args){

    double atom_vec[3];    // atom position
    double grid_edge_x[3]; // grid edge x
    double grid_edge_y[3]; // grid edge y
    double grid_edge_z[3]; // grid edge z
    double d2; // squared distance between grid_vec and atom_vec
    PyArrayObject *grid_uc, *grid_origin;; // grid unit cell matrix (normalized!) and grid origin
    PyArrayObject *atoms; // atom coordinates
    PyArrayObject *bins; // Binning for grid edges
    PyArrayObject *corrected_aff; // Corrected affinity array

    double shortest_d; // shortest distance to atom center for some grid voxel
    double shortest_d2; // shortest distance squared to atom center for some grid voxel
    double d0; // d0 spatial decay parameter. Controls how affinity value decays over distance from atom
    double prefactor; //prefactor in normal distribution
    double d02; // d0 factor squared and multiplied by two

    int verbose=0; //are we verbose or not
    int Natoms; // Number of atoms and number of radii to add
    int i,j,k,a; // counting variables

    //Here we retrieve the data from python
    if (!PyArg_ParseTuple(args, "O!O!O!O!d|i",
        &PyArray_Type, &grid_uc,
        &PyArray_Type, &grid_origin,
        &PyArray_Type, &bins,
        &PyArray_Type, &atoms,
        &PyArray_Type, &d0,
        &verbose
        ))
    {
        return NULL;
    }

    d02       = 2. * d0 * d0;
    prefactor = 1./sqrt(M_PI * d02);

    Natoms = PyArray_DIM(atoms, 0);
    if (Natoms==0){
        PyErr_Format(PyExc_ValueError,
                     "Molecule must contain 1 or more atoms.\n"
                     );
        return NULL;
    }
    if (verbose){
        printf("Found Natoms=%d ...\n", Natoms);
    }

    if (verbose){
        printf("Making volume grid ...\n");
    }

    npy_intp dims[3];
    dims[0] = *(int *) PyArray_GETPTR1(bins, 0);
    dims[1] = *(int *) PyArray_GETPTR1(bins, 1);
    dims[2] = *(int *) PyArray_GETPTR1(bins, 2);
    corrected_aff = PyArray_EMPTY(3, dims, NPY_DOUBLE, 0);

    if (verbose){
        printf("Checking each grid voxel ...\n");
    }

    grid_edge_x[0] = *(double *) PyArray_GETPTR1(grid_origin, 0);
    grid_edge_x[1] = *(double *) PyArray_GETPTR1(grid_origin, 1);
    grid_edge_x[2] = *(double *) PyArray_GETPTR1(grid_origin, 2);
    
    for (i=0; i<*(int *) PyArray_GETPTR1(bins, 0); i++){

        grid_edge_x[0] += *(double *) PyArray_GETPTR2(grid_uc, 0, 0);
        grid_edge_x[1] += *(double *) PyArray_GETPTR2(grid_uc, 1, 0);
        grid_edge_x[2] += *(double *) PyArray_GETPTR2(grid_uc, 2, 0);

        grid_edge_y[0] = grid_edge_x[0];
        grid_edge_y[1] = grid_edge_x[1];
        grid_edge_y[2] = grid_edge_x[2];

    for (j=0; j<*(int *) PyArray_GETPTR1(bins, 1); j++){

        grid_edge_y[0] += *(double *) PyArray_GETPTR2(grid_uc, 0, 1);
        grid_edge_y[1] += *(double *) PyArray_GETPTR2(grid_uc, 1, 1);
        grid_edge_y[2] += *(double *) PyArray_GETPTR2(grid_uc, 2, 1);

        grid_edge_z[0] = grid_edge_y[0];
        grid_edge_z[1] = grid_edge_y[1];
        grid_edge_z[2] = grid_edge_y[2];

    for (k=0; k<*(int *) PyArray_GETPTR1(bins, 2); k++){

        grid_edge_z[0] += *(double *) PyArray_GETPTR2(grid_uc, 0, 2);
        grid_edge_z[1] += *(double *) PyArray_GETPTR2(grid_uc, 1, 2);
        grid_edge_z[2] += *(double *) PyArray_GETPTR2(grid_uc, 2, 2);

        // For each grid voxel, find the closest atom
        atom_vec[0] = *(double *) PyArray_GETPTR2(atoms, 0, 0);
        atom_vec[1] = *(double *) PyArray_GETPTR2(atoms, 0, 1);
        atom_vec[2] = *(double *) PyArray_GETPTR2(atoms, 0, 2);

        shortest_d2 = diff2(grid_edge_z, atom_vec);

        for (a=1; a<Natoms; a++){

        atom_vec[0] = *(double *) PyArray_GETPTR2(atoms, a, 0);
        atom_vec[1] = *(double *) PyArray_GETPTR2(atoms, a, 1);
        atom_vec[2] = *(double *) PyArray_GETPTR2(atoms, a, 2);

        d2 = diff2(grid_edge_z, atom_vec);
        if (d2<shortest_d2) shortest_d2 = d2;

        }
        shortest_d = sqrt(shortest_d2);

        //*(double *) PyArray_GETPTR3(corrected_aff, i, j, k) = exp((-1/d0) * shortest_d);
        *(double *) PyArray_GETPTR3(corrected_aff, i, j, k)   = prefactor * exp(- shortest_d2 / d02);

    }
    }
    }

    if (verbose){
        printf("Done.\n");
    }

    return PyArray_Return(corrected_aff);

}
*/

static PyObject *
_sasa_softgrid_ext(PyObject *self, PyObject *args){

    double atom_vec[3];    // atom position
    double grid_edge_x[3]; // grid edge x
    double grid_edge_y[3]; // grid edge y
    double grid_edge_z[3]; // grid edge z
    double d2; // squared distance between grid_vec and atom_vec
    PyArrayObject *grid_uc, *grid_origin;; // grid unit cell matrix (normalized!) and grid origin
    PyArrayObject *atoms; // atom coordinates
    PyArrayObject *bins; // Binning for grid edges
    PyArrayObject *radius; // list of radii for each atom
    double radius_add; // radii addition to calculate for each atom
    PyArrayObject *grid_softness; // final sasa_grid
    double solvent; //solvent radius
    double softness=1.; // surface softness parameter
    double cutoff=4.; // surface softness parameter. All grid voxels further away from the surface
                   // than cutoff (d>cutoff) are treated as completely non-enclosed by the surface.
    double shortest_d2; // shortest distance squared to atom center for some grid voxel
    double shortest_d;  // shortest distance to surface for some grid voxel
    int is_inside; // is the grid voxel inside the surface or not?
    int verbose=0; //are we verbose or not

    int Natoms; // Number of atoms and number of radii to add

    int i,j,k,a; // counting variables

    //Here we retrieve the data from python
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!dddd|i",
        &PyArray_Type, &grid_uc,
        &PyArray_Type, &grid_origin,
        &PyArray_Type, &bins,
        &PyArray_Type, &atoms,
        &PyArray_Type, &radius,
        &radius_add,
        &solvent,
        &softness,
        &cutoff,
        &verbose
        ))
    {
        return NULL;
    }

    Natoms = PyArray_DIM(atoms, 0);
    if (Natoms==0){
        PyErr_Format(PyExc_ValueError,
                     "Molecule must contain 1 or more atoms.\n"
                     );
        return NULL;
    }
    if (verbose){
        printf("Found Natoms=%d ...\n", Natoms);
    }
    double *atomradius = malloc(Natoms * sizeof(double));
    //Precompute squared atom radii
    if (verbose){
        printf("Precompute atomradius list ...\n");
    }
    for (a=0; a<Natoms; a++){
        atomradius[a]  = *(double *) PyArray_GETPTR1(radius, a);
        atomradius[a] += solvent;
        atomradius[a] += radius_add;
        atomradius[a]  = atomradius[a]*atomradius[a];
    }

    if (verbose){
        printf("Making volume grid ...\n");
    }
    npy_intp dims[3];
    dims[0] = *(int *) PyArray_GETPTR1(bins, 0);
    dims[1] = *(int *) PyArray_GETPTR1(bins, 1);
    dims[2] = *(int *) PyArray_GETPTR1(bins, 2);
    grid_softness = PyArray_EMPTY(3, dims, NPY_DOUBLE, 0);

    if (verbose){
        printf("Checking each grid voxel ...\n");
    }
    grid_edge_x[0] = *(double *) PyArray_GETPTR1(grid_origin, 0);
    grid_edge_x[1] = *(double *) PyArray_GETPTR1(grid_origin, 1);
    grid_edge_x[2] = *(double *) PyArray_GETPTR1(grid_origin, 2);
    
    for (i=0; i<*(int *) PyArray_GETPTR1(bins, 0); i++){

        grid_edge_x[0] += *(double *) PyArray_GETPTR2(grid_uc, 0, 0);
        grid_edge_x[1] += *(double *) PyArray_GETPTR2(grid_uc, 1, 0);
        grid_edge_x[2] += *(double *) PyArray_GETPTR2(grid_uc, 2, 0);

        grid_edge_y[0] = grid_edge_x[0];
        grid_edge_y[1] = grid_edge_x[1];
        grid_edge_y[2] = grid_edge_x[2];

    for (j=0; j<*(int *) PyArray_GETPTR1(bins, 1); j++){

        grid_edge_y[0] += *(double *) PyArray_GETPTR2(grid_uc, 0, 1);
        grid_edge_y[1] += *(double *) PyArray_GETPTR2(grid_uc, 1, 1);
        grid_edge_y[2] += *(double *) PyArray_GETPTR2(grid_uc, 2, 1);

        grid_edge_z[0] = grid_edge_y[0];
        grid_edge_z[1] = grid_edge_y[1];
        grid_edge_z[2] = grid_edge_y[2];

    for (k=0; k<*(int *) PyArray_GETPTR1(bins, 2); k++){

        grid_edge_z[0] += *(double *) PyArray_GETPTR2(grid_uc, 0, 2);
        grid_edge_z[1] += *(double *) PyArray_GETPTR2(grid_uc, 1, 2);
        grid_edge_z[2] += *(double *) PyArray_GETPTR2(grid_uc, 2, 2);

        is_inside   = 0;
        shortest_d  = 0;
        shortest_d2 = 0;
        for (a=0; a<Natoms; a++){

        atom_vec[0] = *(double *) PyArray_GETPTR2(atoms, a, 0);
        atom_vec[1] = *(double *) PyArray_GETPTR2(atoms, a, 1);
        atom_vec[2] = *(double *) PyArray_GETPTR2(atoms, a, 2);

        d2 = diff2(grid_edge_z, atom_vec);
        if (d2<atomradius[a]) is_inside = 1;

        if (a==0) {
            shortest_d2 = d2;
            shortest_d  = sqrt(d2)-sqrt(atomradius[a]);
            if (shortest_d<0) shortest_d *= -1;
        }
        else if ((!is_inside) && (d2<shortest_d2)) {
            shortest_d2 = d2;
            shortest_d  = sqrt(d2)-sqrt(atomradius[a]);
            if (shortest_d<0) shortest_d *= -1;
        }

        }
        
        // Value of 1. means the grid voxel is enclosed by the surface
        if (is_inside) {
            *(double *) PyArray_GETPTR3(grid_softness, i, j, k) = 1.;
        }
        // Value of 0. means the grid voxel is not enclosed by the surface
        // and also outside the cutoff region.
        else if (shortest_d > cutoff) {
            *(double *) PyArray_GETPTR3(grid_softness, i, j, k) = 0.;
        }
        else {
            *(double *) PyArray_GETPTR3(grid_softness, i, j, k) = exp((-1/softness) * shortest_d);
        }

    }
    }
    }

    if (verbose){
        printf("Done.\n");
    }
    
    free(atomradius);
    return PyArray_Return(grid_softness);

}

static PyObject *
_sasa_grid_ext(PyObject *self, PyObject *args) {

    double atom_vec[3];    // atom position
    double grid_edge_x[3]; // grid edge x
    double grid_edge_y[3]; // grid edge y
    double grid_edge_z[3]; // grid edge z
    double d2; // squared distance between grid_vec and atom_vec
    PyArrayObject *grid_uc, *grid_origin;; // grid unit cell matrix (normalized!) and grid origin
    PyArrayObject *atoms; // atom coordinates
    PyArrayObject *bins; // Binning for grid edges
    PyArrayObject *radius; // list of radii for each atom
    PyArrayObject *radius_add; // list of radii additions to calculate for each atom
    PyArrayObject *grid_sasa; // final sasa_grid
    double solvent; //solvent radius
    double smallest_radius;
    int verbose=0; //are we verbose or not

    int Natoms, Nradius_add; // Number of atoms and number of radii to add

    int i,j,k,a,r,m; // counting variables

    //Here we retrieve the data from python
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!d|i",
        &PyArray_Type, &grid_uc,
        &PyArray_Type, &grid_origin,
        &PyArray_Type, &bins,
        &PyArray_Type, &atoms,
        &PyArray_Type, &radius,
        &PyArray_Type, &radius_add,
        &solvent,
        &verbose
        ))
    {
        return NULL;
    }

    Nradius_add = PyArray_DIM(radius_add, 0);
    Natoms = PyArray_DIM(atoms, 0);
    if (verbose){
        printf("Found Nradius_add=%d and Natoms=%d ...\n",
        Nradius_add, Natoms);
    }
    double *atomradius = malloc(Nradius_add * Natoms * sizeof(double));
    //Precompute squared atom radii
    if (verbose){
        printf("Precompute atomradius list ...\n");
    }
    m=0;
    for (a=0; a<Natoms; a++){
    for (r=0; r<Nradius_add; r++){
        atomradius[m]  = *(double *) PyArray_GETPTR1(radius, a);
        atomradius[m] += solvent;
        atomradius[m] += *(double *) PyArray_GETPTR1(radius_add, r);
        atomradius[m]  = atomradius[m]*atomradius[m];
        m++;
    }
    }

    if (verbose){
        printf("Making volume grid ...\n");
    }
    npy_intp dims[3];
    dims[0] = *(int *) PyArray_GETPTR1(bins, 0);
    dims[1] = *(int *) PyArray_GETPTR1(bins, 1);
    dims[2] = *(int *) PyArray_GETPTR1(bins, 2);
    grid_sasa = PyArray_EMPTY(3, dims, NPY_DOUBLE, 0);

    if (verbose){
        printf("Checking each grid voxel ...\n");
    }
    grid_edge_x[0] = *(double *) PyArray_GETPTR1(grid_origin, 0);
    grid_edge_x[1] = *(double *) PyArray_GETPTR1(grid_origin, 1);
    grid_edge_x[2] = *(double *) PyArray_GETPTR1(grid_origin, 2);
    
    for (i=0; i<*(int *) PyArray_GETPTR1(bins, 0); i++){

        grid_edge_x[0] += *(double *) PyArray_GETPTR2(grid_uc, 0, 0);
        grid_edge_x[1] += *(double *) PyArray_GETPTR2(grid_uc, 1, 0);
        grid_edge_x[2] += *(double *) PyArray_GETPTR2(grid_uc, 2, 0);

        grid_edge_y[0] = grid_edge_x[0];
        grid_edge_y[1] = grid_edge_x[1];
        grid_edge_y[2] = grid_edge_x[2];

    for (j=0; j<*(int *) PyArray_GETPTR1(bins, 1); j++){

        grid_edge_y[0] += *(double *) PyArray_GETPTR2(grid_uc, 0, 1);
        grid_edge_y[1] += *(double *) PyArray_GETPTR2(grid_uc, 1, 1);
        grid_edge_y[2] += *(double *) PyArray_GETPTR2(grid_uc, 2, 1);

        grid_edge_z[0] = grid_edge_y[0];
        grid_edge_z[1] = grid_edge_y[1];
        grid_edge_z[2] = grid_edge_y[2];

    for (k=0; k<*(int *) PyArray_GETPTR1(bins, 2); k++){

        grid_edge_z[0] += *(double *) PyArray_GETPTR2(grid_uc, 0, 2);
        grid_edge_z[1] += *(double *) PyArray_GETPTR2(grid_uc, 1, 2);
        grid_edge_z[2] += *(double *) PyArray_GETPTR2(grid_uc, 2, 2);

        *(double *) PyArray_GETPTR3(grid_sasa, i, j, k) = -1.;
        m=0;
        smallest_radius = 99999999999.;
        for (a=0; a<Natoms; a++){

        atom_vec[0] = *(double *) PyArray_GETPTR2(atoms, a, 0);
        atom_vec[1] = *(double *) PyArray_GETPTR2(atoms, a, 1);
        atom_vec[2] = *(double *) PyArray_GETPTR2(atoms, a, 2);

        d2 = diff2(grid_edge_z, atom_vec);
        for (r=0; r<Nradius_add; r++){
            if (d2<atomradius[m]){
            if (*(double *) PyArray_GETPTR1(radius_add, r) < smallest_radius){

                smallest_radius = *(double *) PyArray_GETPTR1(radius_add, r);                
                *(double *) PyArray_GETPTR3(grid_sasa, i, j, k) = smallest_radius;
            }
            }
            m++;
        }
        }

    }
    }
    }

    if (verbose){
        printf("Done.\n");
    }
    
    free(atomradius);
    return PyArray_Return(grid_sasa);

}


static PyObject *
_sasa_vol_ext(PyObject *self, PyObject *args) {

    double grid_vec[3], atom_vec[3]; //position of grid point and atom
    double d2; //squared distance between grid_vec and atom_vec
    PyArrayObject *grid, *atoms; //grid coordiates and atom coordinates
    PyArrayObject *radius; // list of radii for each atom
    PyArrayObject *valids; // store final result with grid voxels inside sasa
    double solvent; //solvent radius
    int verbose=0; //are we verbose or not

    int Ngrid, Natoms; //Number of grid points, number of atoms
    int Ninside=0; //Number of grid voxel inside sasa

    int i,j; // counting variables

    //Here we retrieve the data from python
    if (!PyArg_ParseTuple(args, "O!O!O!d|i",
        &PyArray_Type, &grid,
        &PyArray_Type, &atoms,
        &PyArray_Type, &radius,
        &solvent,
        &verbose
        ))
    {
        return NULL;
    }

    Ngrid  = PyArray_DIM(grid, 0);
    Natoms = PyArray_DIM(atoms, 0);
    if (verbose){
        printf("Found Ngrid=%d and Natoms=%d ...\n", Ngrid, Natoms);
    }
    int *valid_voxel   = malloc(Ngrid * sizeof(int)); //valid grid voxel
    double *atomradius = malloc(Natoms * sizeof(double));
    //Precompute squared atom radii
    if (verbose){
        printf("Precompute atomradius list ...\n");
    }
    for (j=0; j<Natoms; j++){
        atomradius[j]  = *(double *) PyArray_GETPTR1(radius, j);
        atomradius[j] += solvent;
        atomradius[j]  = atomradius[j]*atomradius[j];
    }

    if (verbose){
        printf("Perform calculation for each grid voxel ...\n");
    }
    for (i=0; i<Ngrid; i++){
        grid_vec[0] = *(double *) PyArray_GETPTR2(grid, i, 0);
        grid_vec[1] = *(double *) PyArray_GETPTR2(grid, i, 1);
        grid_vec[2] = *(double *) PyArray_GETPTR2(grid, i, 2);

        for (j=0; j<Natoms; j++){
            atom_vec[0] = *(double *) PyArray_GETPTR2(atoms, j, 0);
            atom_vec[1] = *(double *) PyArray_GETPTR2(atoms, j, 1);
            atom_vec[2] = *(double *) PyArray_GETPTR2(atoms, j, 2);

            d2 = diff2(grid_vec, atom_vec);

            if (d2 < atomradius[j]){
                valid_voxel[i]=1;
                Ninside++;
                break;
            }
            else{
                valid_voxel[i]=0;
            }
        }
    }

    if (verbose){
        printf("Building final voxel list ...\n");
    }
    npy_intp dims[1];
    dims[0] = Ninside;
    valids = PyArray_SimpleNew(1, dims, NPY_INT);

    if (verbose){
        printf("Populating final voxel list ...\n");
    }
    j=0;
    for (i=0; i<Ngrid; i++){
        if (valid_voxel[i]==1){
            *(int *) PyArray_GETPTR1(valids, j) = i;
            j++;
        }
    }

    if (verbose){
        printf("Done.\n");
    }
    
    free(valid_voxel);
    free(atomradius);
    return PyArray_Return(valids);

}


static PyMethodDef _spatial_ext_methods[] = {
    {
        "sasa_softgrid_ext",
        (PyCFunction)_sasa_softgrid_ext,
        METH_VARARGS,
        "Calculates volume elements using the soft surface approach.",
    },
    {
        "sasa_vol_ext",
        (PyCFunction)_sasa_vol_ext,
        METH_VARARGS,
        "Calculates volume elements enclosed by solvent accesible surface area.",
    },
    {
        "sasa_grid_ext",
        (PyCFunction)_sasa_grid_ext,
        METH_VARARGS,
        "Calculates volume grid, which describes the solvent accesible surface area "
        "computed for varying atomic radii.",
    },
/*
/// ------------------------------------------
/// | This method is completely untested !!! |
/// ------------------------------------------
    {
        "corrected_aff_ext",
        (PyCFunction)_corrected_aff,
        METH_VARARGS,
        "Calculates corrected affinity parameters based on distances from set of hydration "
        "sites or water molecules.",
    },
*/

    { NULL, NULL, 0, NULL }
};

#if PY_MAJOR_VERSION >= 3
    #define MOD_ERROR_VAL NULL
    #define MOD_SUCCESS_VAL(val) val
    #define MOD_INIT(name) PyMODINIT_FUNC PyInit_##name(void)
    #define MOD_DEF(ob, name, doc, methods) \
            static struct PyModuleDef extmoduledef = { \
              PyModuleDef_HEAD_INIT, name, doc, -1, methods, }; \
            ob = PyModule_Create(&extmoduledef);
#else
    #define MOD_ERROR_VAL
    #define MOD_SUCCESS_VAL(val)
    #define MOD_INIT(name) void init##name(void)
    #define MOD_DEF(ob, name, doc, methods) \
            ob = Py_InitModule3(name, methods, doc);
#endif

/* Initialization function for this module
 */

MOD_INIT(_spatial_ext)
{   
    PyObject *m;

    MOD_DEF(m, "_spatial_ext", "Routines for spatial calculations.\n", _spatial_ext_methods)
    
    if (m == NULL)
        return MOD_ERROR_VAL;

    import_array();

    return MOD_SUCCESS_VAL(m);
}