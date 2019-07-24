/*
### Written by Tobias Wulsdorf @ Klebe Lab, Marburg University
### tobias.wulsdorf@gmail.com
*/

int gist_functional_6p(PyArrayObject *E,
                       PyArrayObject *S,
                       PyArrayObject *g,
                       PyArrayObject *vol,
                       PyArrayObject *ind,
                       PyArrayObject *x,
                       double        *dx,
                       PyArrayObject *fun,
                       PyArrayObject *grad,
                       int          *decomp,
                       int          *do_grad,
                       int          *verbose);

int gist_functional_5p(PyArrayObject *E,
                       PyArrayObject *S,
                       PyArrayObject *g,
                       PyArrayObject *vol,
                       PyArrayObject *ind,
                       PyArrayObject *x,
                       double        *dx,
                       PyArrayObject *fun,
                       PyArrayObject *grad,
                       int          *decomp,
                       int          *do_grad,
                       int          *verbose);

int gist_functional_4p(PyArrayObject *E,
                       PyArrayObject *S,
                       PyArrayObject *g,
                       PyArrayObject *vol,
                       PyArrayObject *ind,
                       PyArrayObject *x,
                       double        *dx,
                       PyArrayObject *fun,
                       PyArrayObject *grad,
                       int          *decomp,
                       int          *do_grad,
                       int          *verbose);