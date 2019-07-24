/*
### Written by Tobias Wulsdorf @ Klebe Lab, Marburg University
### tobias.wulsdorf@gmail.com
*/

#define _USE_MATH_DEFINES
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "numpy/arrayobject.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "gist_functionals.h"
#include "gist_functionals_check.h"

static PyObject *
_merge_casedata_ext(PyObject *self, PyObject *args){

    PyArrayObject *source;
    PyArrayObject *target;
    PyArrayObject *assign;
    PyArrayObject *factor;
    PyArrayObject *assign_factor;

    int s,t,f;   // source, target and factor counting
    int N_target, N_source;

    //Here we retrieve the data from python
    if (!PyArg_ParseTuple(args, "O!O!O!O!",
        &PyArray_Type, &source,
        &PyArray_Type, &assign,
        &PyArray_Type, &factor,
        &PyArray_Type, &assign_factor
        ))
    {
        return NULL;
    }

    N_source = PyArray_DIM(source, 0);

    if (N_source!=PyArray_DIM(assign, 0)){
        PyErr_Format(PyExc_ValueError,
                     "source and assign array must have same length."
                     );
        return NULL;
    }

    if (N_source!=PyArray_DIM(assign_factor, 0)){
        PyErr_Format(PyExc_ValueError,
                     "source and assign_factor array must have same length."
                     );
        return NULL;
    }

    N_target=0;
    for (s=0;s<N_source;s++){
        t = *(int *) PyArray_GETPTR1(assign, s);
        if (t>N_target) N_target=t;
    }
    N_target++;

    npy_intp dims[1];
    dims[0] = N_target;
    target  = PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);

    for (t=0;t<N_target;t++){
        *(double *) PyArray_GETPTR1(target, t) = 0.;
    }

    for (s=0;s<N_source;s++){

        t = *(int *) PyArray_GETPTR1(assign, s);
        f = *(int *) PyArray_GETPTR1(assign_factor, s);

        *(double *) PyArray_GETPTR1(target, t) += *(double *) PyArray_GETPTR1(source, s) * *(double *) PyArray_GETPTR1(factor, f);

        /*
        printf("%d --> %d --> %d\n", s, t, f);
        printf("%10.6f --> %10.6f --> %10.6f\n", 
            *(double *) PyArray_GETPTR1(source, s),
            *(double *) PyArray_GETPTR1(factor, f),
            *(double *) PyArray_GETPTR1(target, t));
        */
    }

    return PyArray_Return(target);

}

static PyObject *
_pair_difference_ext(PyObject *self, PyObject *args){

    PyArrayObject *data, *idx; // Python arrays retrieved from python:
                              // ar1: array containing the data
                              // idx: array containing pair assignments for ar1
    PyArrayObject *diff_array; /* Array containing element-wise differences between ar1 and ar2.
                                  This is going to be returned to python.
                                */
    int N1, N2; // Length of python arrays

    int i,d1,d2; //counting variables

    //Here we retrieve the data from python
    if (!PyArg_ParseTuple(args, "O!O!",
        &PyArray_Type, &data,
        &PyArray_Type, &idx
        ))
    {
        return NULL;
    }

    N1 = PyArray_DIM(data, 0);
    N2 = PyArray_DIM(idx, 0);

    npy_intp dims[1];
    // Number of elements in lower triangle matrix
    // without diagonal entries.
    dims[0] = N2;
    diff_array = PyArray_EMPTY(1, dims, NPY_DOUBLE, 0);

    for (i=0;i<N2;i++){
        d1 = *(int *) PyArray_GETPTR2(idx, i, 0);
        d2 = *(int *) PyArray_GETPTR2(idx, i, 1);
        *(double *) PyArray_GETPTR1(diff_array, i) = *(double *) PyArray_GETPTR1(data, d1) - 
                                                     *(double *) PyArray_GETPTR1(data, d2);
    }

    return PyArray_Return(diff_array);

}

static PyObject *
_gist_restraint_ext(PyObject *self, PyObject *args) {

    PyArrayObject *x, *xmin, *xmax, 
                  *kforce_f, *restraint_grad;
    double kforce;
    double restraint;
    double diff_lower, diff_upper;
    int Ndim;

    double k; // current effective force constant

    int i; //Counting variable

    //Here we retrieve the data from python
    if (!PyArg_ParseTuple(args, "O!O!O!O!dO!",
        &PyArray_Type, &x,
        &PyArray_Type, &xmin,
        &PyArray_Type, &xmax,
        &PyArray_Type, &kforce_f,
        &kforce,
        &PyArray_Type, &restraint_grad
        ))
    {
        return NULL;
    }

    Ndim = PyArray_DIM(x, 0);

    if (Ndim != PyArray_DIM(xmin, 0)){
        PyErr_Format(PyExc_ValueError,
                     "xmin must be of shape (%d,).\n", Ndim
                     );
        return NULL;
    }

    if (Ndim != PyArray_DIM(xmax, 0)){
        PyErr_Format(PyExc_ValueError,
                     "xmax must be of shape (%d,).\n", Ndim
                     );
        return NULL;
    }

    if (Ndim != PyArray_DIM(restraint_grad, 0)){
        PyErr_Format(PyExc_ValueError,
                     "restraint_grad must be of shape (%d,).\n", Ndim
                     );
        return NULL;
    }

    if (Ndim != PyArray_DIM(kforce_f, 0)){
        PyErr_Format(PyExc_ValueError,
                     "kforce_f must be of shape (%d,).\n", Ndim
                     );
        return NULL;
    }

    restraint = 0.;
    for (i=0; i<Ndim; i++){
        *(double *) PyArray_GETPTR1(restraint_grad, i) = 0.;

        k = kforce * *(double *) PyArray_GETPTR1(kforce_f, i);

        diff_lower = *(double *) PyArray_GETPTR1(x, i) - *(double *) PyArray_GETPTR1(xmin, i);
        diff_upper = *(double *) PyArray_GETPTR1(x, i) - *(double *) PyArray_GETPTR1(xmax, i);
        if (diff_lower<0.) {
            restraint += (k * diff_lower * diff_lower);
            *(double *) PyArray_GETPTR1(restraint_grad, i) += (k * 2. * diff_lower);
        }
        else if (diff_upper>0.) {
            restraint += (k * diff_upper * diff_upper);
            *(double *) PyArray_GETPTR1(restraint_grad, i) += (k * 2. * diff_upper);
        }
    }

    return Py_BuildValue("d", restraint);

}


static PyObject *
_gist_functional_4p_ext(PyObject *self, PyObject *args) {

    PyArrayObject *E, *S, *g, *vol, *ind,
                   *x, *fun, *grad;
    double dx;
    int decomp=0; // perform decomposition of entropy and enthalpy or not
    int do_grad=1; // do gradient calculation or not
    int verbose=0; //are we verbose or not

    int score;
    score=4;

    //Here we retrieve the data from python
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!dO!O!i|ii",
        &PyArray_Type, &E,
        &PyArray_Type, &S,
        &PyArray_Type, &g,
        &PyArray_Type, &vol,
        &PyArray_Type, &ind,
        &PyArray_Type, &x,
        &dx,
        &PyArray_Type, &fun,
        &PyArray_Type, &grad,
        &decomp,
        &do_grad,
        &verbose
        ))
    {
        return NULL;
    }

    if ( !gist_functional_check(E, S, g, vol, ind,
                            x, fun, grad, &score,
                            &decomp, &verbose) ) {
        return Py_BuildValue("i", 0);
    }

    if ( gist_functional_4p(E, S, g, vol, ind,
                            x, &dx, fun, grad,
                            &decomp, &do_grad, &verbose) ) {
        return Py_BuildValue("i", 1);
    }
    else{
        return Py_BuildValue("i", 0);
    }
}


static PyObject *
_gist_functional_5p_ext(PyObject *self, PyObject *args) {

    PyArrayObject *E, *S, *g, *vol, *ind,
                   *x, *fun, *grad;
    int decomp=0;  // perform decomposition of entropy and enthalpy or not
    int do_grad=1; // do gradient calculation or not
    int verbose=0; //are we verbose or not
    double dx;

    int score;
    score=5;

    //Here we retrieve the data from python
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!dO!O!i|ii",
        &PyArray_Type, &E,
        &PyArray_Type, &S,
        &PyArray_Type, &g,
        &PyArray_Type, &vol,
        &PyArray_Type, &ind,
        &PyArray_Type, &x,
        &dx,
        &PyArray_Type, &fun,
        &PyArray_Type, &grad,
        &decomp,
        &do_grad,
        &verbose
        ))
    {
        return NULL;
    }

    if ( !gist_functional_check(E, S, g, vol, ind,
                            x, fun, grad, &score,
                            &decomp, &verbose) ) {
        return Py_BuildValue("i", 0);
    }

    if ( gist_functional_5p(E, S, g, vol, ind,
                            x, &dx, fun, grad,
                            &decomp, &do_grad, &verbose) ) {
        return Py_BuildValue("i", 1);
    }
    else{
        return Py_BuildValue("i", 0);
    }
}


static PyObject *
_gist_functional_6p_ext(PyObject *self, PyObject *args) {

    PyArrayObject *E, *S, *g, *vol, *ind,
                   *x, *fun, *grad;
    double dx;
    int decomp=0;  // perform decomposition of entropy and enthalpy or not
    int do_grad=1; // do gradient calculation or not
    int verbose=0; //are we verbose or not

    int score;
    score=6;

    //Here we retrieve the data from python
    if (!PyArg_ParseTuple(args, "O!O!O!O!O!O!dO!O!i|ii",
        &PyArray_Type, &E,
        &PyArray_Type, &S,
        &PyArray_Type, &g,
        &PyArray_Type, &vol,
        &PyArray_Type, &ind,
        &PyArray_Type, &x,
        &dx,
        &PyArray_Type, &fun,
        &PyArray_Type, &grad,
        &decomp,
        &do_grad,
        &verbose
        ))
    {
        return NULL;
    }

    if ( !gist_functional_check(E, S, g, vol, ind,
                            x, fun, grad, &score,
                            &decomp, &verbose) ) {
        return Py_BuildValue("i", 0);
    }

    if ( gist_functional_6p(E, S, g, vol, ind,
                            x, &dx, fun, grad,
                            &decomp, &do_grad, &verbose) ) {
        return Py_BuildValue("i", 1);
    }
    else{
        return Py_BuildValue("i", 0);
    }
}

static PyMethodDef _numerical_ext_methods[] = {
    {
        "gist_functional_4p_ext",
        (PyCFunction)_gist_functional_4p_ext,
        METH_VARARGS,
        "Calculates free energies from gist data using a 4 parameter functional form.",
    },
    {
        "gist_functional_5p_ext",
        (PyCFunction)_gist_functional_5p_ext,
        METH_VARARGS,
        "Calculates free energies from gist data using a 5 parameter functional form.",
    },
    {
        "gist_functional_6p_ext",
        (PyCFunction)_gist_functional_6p_ext,
        METH_VARARGS,
        "Calculates free energies from gist data using a 6 parameter functional form.",
    },
    {
        "gist_restraint_ext",
        (PyCFunction)_gist_restraint_ext,
        METH_VARARGS,
        "Calculates free energy restraints and free energy restraints gradient "
        "for quadratic restraint function with N dimensions.",
    },
    {
        "pair_difference_ext",
        (PyCFunction)_pair_difference_ext,
        METH_VARARGS,
        "Calculates the pairwise differences between for each pair in two arrays.",
    },
    {
        "merge_casedata_ext",
        (PyCFunction)_merge_casedata_ext,
        METH_VARARGS,
        "Merge data down from number of gist data points to number of case data points.",
    },
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

MOD_INIT(_numerical_ext)
{   
    PyObject *m;

    MOD_DEF(m, "_numerical_ext", "Routines for numerical calculations in gist fitting process.\n", _numerical_ext_methods)
    
    if (m == NULL)
        return MOD_ERROR_VAL;

    import_array();

    return MOD_SUCCESS_VAL(m);
}