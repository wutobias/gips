/*
### Written by Tobias Wulsdorf @ Klebe Lab, Marburg University
### tobias.wulsdorf@gmail.com
*/

int gist_functional_check(PyArrayObject *E,
                        PyArrayObject *S,
                        PyArrayObject *g,
                        PyArrayObject *vol,
                        PyArrayObject *ind,
                        PyArrayObject *x,
                        PyArrayObject *fun,
                        PyArrayObject *grad,
                        int          *decomp,
                        int          *score,
                        int          *verbose);
