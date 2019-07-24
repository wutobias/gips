/*
### Written by Tobias Wulsdorf @ Klebe Lab, Marburg University
### tobias.wulsdorf@gmail.com
*/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "numpy/arrayobject.h"
#include "gist_functionals_check.h"


int gist_functional_check (PyArrayObject *E,
                            PyArrayObject *S,
                            PyArrayObject *g,
                            PyArrayObject *vol,
                            PyArrayObject *ind,
                            PyArrayObject *x,
                            PyArrayObject *fun,
                            PyArrayObject *grad,
                            int          *score,
                            int          *decomp,
                            int          *verbose){

    int _score;
    int Nrec, Npose; // Number of receptors, number of poses
    int bins[3];     // Number of bins in each direction

    Nrec    = PyArray_DIM(E, 0);
    bins[0] = PyArray_DIM(E, 1);
    bins[1] = PyArray_DIM(E, 2);
    bins[2] = PyArray_DIM(E, 3);

    Npose   = PyArray_DIM(vol, 0);

    _score=*score;
    if(*decomp) _score++;

    if (*verbose){
        printf("Nrec=%d\n", Nrec);
        printf("Npose=%d\n", Npose);
        printf("bins=(%d,%d,%d)\n", bins[0], bins[1], bins[2]);
    }

    if (*verbose){
        printf("Checking Nrec ...\n");
    }
    if (Nrec != PyArray_DIM(S, 0)){
        PyErr_Format(PyExc_ValueError,
                     "E and S must have same number of receptors.\n"
                     );
        return 0;
    }
    if (Nrec != PyArray_DIM(g, 0)){
        PyErr_Format(PyExc_ValueError,
                     "E and g must have same number of receptors.\n"
                     );
        return 0;
    }

    if (*verbose){
        printf("Checking bins ...\n");
    }
    if (bins[0] != PyArray_DIM(S, 1) ||
        bins[1] != PyArray_DIM(S, 2) ||
        bins[2] != PyArray_DIM(S, 3)) {
        PyErr_Format(PyExc_ValueError,
                     "E and S grids must have same dimensions.\n"
                     );
        return 0;
    }
    if (bins[0] != PyArray_DIM(g, 1) ||
        bins[1] != PyArray_DIM(g, 2) ||
        bins[2] != PyArray_DIM(g, 3)) {
        PyErr_Format(PyExc_ValueError,
                     "E and g grids must have same dimensions.\n"
                     );
        return 0;
    }
    if (bins[0] != PyArray_DIM(vol, 1) ||
        bins[1] != PyArray_DIM(vol, 2) ||
        bins[2] != PyArray_DIM(vol, 3)) {
        PyErr_Format(PyExc_ValueError,
                     "E and vol grids must have same dimensions.\n"
                     );
        return 0;
    }

    if (*verbose){
        printf("Checking Npose ...\n");
    }
    if (Npose != PyArray_DIM(ind, 0)){
        PyErr_Format(PyExc_ValueError,
                     "vol and ind must contain same number of poses.\n"
                     );
        return 0;
    }
    if (Npose != PyArray_DIM(fun, 0)){
        PyErr_Format(PyExc_ValueError,
                     "vol and fun must contain same number of poses.\n"
                     );
        return 0;
    }

    if (Npose != PyArray_DIM(grad, 0)){
        PyErr_Format(PyExc_ValueError,
                     "vol and grad must contain same number of poses.\n"
                     );
        return 0;
    }

    if (*decomp) {
        if (_score != PyArray_DIM(grad, 1)){
            PyErr_Format(PyExc_ValueError,
                        "grad must be of shape (%d, %d, 2).\n", Npose, _score
                        );
            return 0;
        }

        if (2 != PyArray_DIM(grad, 2)){
            PyErr_Format(PyExc_ValueError,
                        "grad must be of shape (%d, %d, 2).\n", Npose, _score
                        );
            return 0;
        }

        if (2 != PyArray_DIM(fun, 1)){
            PyErr_Format(PyExc_ValueError,
                        "fun must be of shape (%d, 2).\n", Npose
                        );
            return 0;
        }
    }

    else {
        if (_score != PyArray_DIM(grad, 1)){
            PyErr_Format(PyExc_ValueError,
                        "grad must be of shape (%d, %d).\n", Npose, _score
                        );
            return 0;
        }
    }

    if (*verbose){
        printf("Checking x ...\n");
    }
    if (_score != PyArray_DIM(x, 0)){
        PyErr_Format(PyExc_ValueError,
                    "x must contain %d elements.\n", _score
                    );
        return 0;
    }

    return 1;
}