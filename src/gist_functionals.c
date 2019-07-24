/*
### Written by Tobias Wulsdorf @ Klebe Lab, Marburg University
### tobias.wulsdorf@gmail.com
*/

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "numpy/arrayobject.h"
#include "gist_functionals.h"
#include "omp.h"

int gist_functional_4p (PyArrayObject *E,
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
                        int          *verbose) {

    /*
    x elements: 0: e_co
                1: s_co
                2: g_co
                3: C
    */
    int i,j,k,l,m; // some counting variables
    double upper[4]; // boundaries for 2 point finite differences
    int Npose; // Number of poses
    int bins[3]; // Number of bins in each direction
    int Nthreads;
    double dGdeco, dGdsco, dGdgcoE, dGdgcoS;
    double E_dens, E_dens_r, S_dens, S_dens_r;
    double *E_val, *S_val, *g_val; // Store values from E, S and vol during processing
    double vol_val; // Store values from vol during processing
    double rho=0.0332; // Density value for neat water
    int e_i, s_i, g_i;    // binary indicator
    int de_i, ds_i, dg_i; // differential of binary indicator
    de_i=0, ds_i=0, dg_i=0;
    e_i=0,  s_i=0,  g_i=0;

    if (*do_grad){
        upper[0] = *(double *) PyArray_GETPTR1(x, 0) + *dx;
        upper[1] = *(double *) PyArray_GETPTR1(x, 1) + *dx;
        upper[2] = *(double *) PyArray_GETPTR1(x, 2) + *dx;
        upper[3] = *(double *) PyArray_GETPTR1(x, 3) + *dx;
    }

    Npose   = PyArray_DIM(vol, 0);
    bins[0] = PyArray_DIM(E, 1);
    bins[1] = PyArray_DIM(E, 2);
    bins[2] = PyArray_DIM(E, 3);

    Nthreads = omp_get_max_threads();

    if (Nthreads>Npose){
        Nthreads=Npose;
        omp_set_num_threads(Nthreads);
    }

    #pragma omp parallel for private(i,j,k,l,m, \
                            E_val, S_val, \
                            g_val, vol_val, \
                            dGdeco, dGdsco, \
                            dGdgcoE, dGdgcoS, \
                            E_dens, E_dens_r, \
                            S_dens, S_dens_r) \
                        firstprivate(e_i, s_i, g_i, \
                            de_i, ds_i, dg_i, \
                            g, S, E, vol, \
                            x, dx, upper, ind) \
                        shared(do_grad, decomp, verbose, \
                            bins, Npose, fun, grad,\
                            rho) \
                        default(none)
    for (m=0; m<Npose; m++){

        //int ithread;
        //ithread = omp_get_thread_num();
        //if (ithread==0) printf("m: %d\n", m);

        E_dens  = 0.;
        dGdeco  = 0.;
        S_dens  = 0.;
        dGdsco  = 0.;
        dGdgcoE = 0.;
        dGdgcoS = 0.;

        for (i=0; i<bins[0]; i++){
        for (j=0; j<bins[1]; j++){
        for (k=0; k<bins[2]; k++){

            vol_val = *(double *) PyArray_GETPTR4(vol, m, i, j, k);
            if (vol_val>0.000001){

            l = *(int *) PyArray_GETPTR1(ind, m);

            g_val = (double *) PyArray_GETPTR4(g, l, i, j, k);
            if (*g_val>*(double *) PyArray_GETPTR1(x, 2)) g_i = 1;
            else g_i = 0;
            if (*do_grad){
                if (*g_val>upper[2]) dg_i = 1;
                else dg_i = 0;
                dg_i = dg_i-g_i;
            }

            if (g_i!=0 || dg_i!=0){

                E_val = (double *) PyArray_GETPTR4(E, l, i, j, k);
                if (*E_val>*(double *) PyArray_GETPTR1(x, 0)) e_i = 1;
                else e_i = 0;
                if (*do_grad){
                    if (*E_val>upper[0]) de_i = 1;
                    else de_i = 0;
                    de_i = de_i-e_i;
                }
                
                S_val = (double *) PyArray_GETPTR4(S, l, i, j, k);
                if (*S_val>*(double *) PyArray_GETPTR1(x, 1)) s_i = 1;
                else s_i = 0;
                if (*do_grad){
                    if (*S_val>upper[1]) ds_i = 1;
                    else ds_i = 0;
                    ds_i = ds_i-s_i;
                }

                /* 
                - Gradients dGdE and dGdS are exact
                - Gradients dGdeco, dGdsco and dGdgco are approximated
                  using forward finite differnce method
                */

                /// This is how it should be:
                /// Note: 0.125 is the volume of a 0.5x0.5x0.5 grid voxel!!
                E_dens_r = *E_val * vol_val * *g_val * rho * 0.125;
                S_dens_r = *S_val * vol_val * *g_val * rho * 0.125;
                //S_dens_r = *S_val * vol_val * *g_val * rho * 0.125 * -1.;

                //printf("E_dens_r: %6.3f, S_dens_r: %6.3f\n", E_dens_r, S_dens_r);

                if (g_i>0){
                    if (e_i>0) E_dens += E_dens_r;
                    if (s_i>0) S_dens += S_dens_r;

                        if (*do_grad){
                            if (de_i>0)      dGdeco += E_dens_r;
                            else if (de_i<0) dGdeco -= E_dens_r;

                        if (ds_i>0)      dGdsco += S_dens_r;
                        else if (ds_i<0) dGdsco -= S_dens_r;
                    }
                }
                if (*do_grad){
                    if (dg_i>0){
                        if (e_i>0) dGdgcoE += E_dens_r;
                        if (s_i>0) dGdgcoS += S_dens_r;
                    }

                    else if (dg_i<0){
                        if (e_i>0) dGdgcoE -= E_dens_r;
                        if (s_i>0) dGdgcoS -= S_dens_r;
                    }
                }
            }
            }
        }
        }
        }

    /*
    x elements: 0: e_co
                1: s_co
                2: g_co
                3: C
    */

        if (*decomp) {
            /*
            #################
             Energy function
            #################
            */

            // Energy term
            *(double *) PyArray_GETPTR2(fun, m, 0)  = E_dens;
            // Constant term C_E
            *(double *) PyArray_GETPTR2(fun, m, 0) += *(double *) PyArray_GETPTR1(x, 3);

            /*
            ##################
             Entropy function
            ##################
            */
            // Entropy term
            *(double *) PyArray_GETPTR2(fun, m, 1)  = S_dens;
            // Constant term C_S
            *(double *) PyArray_GETPTR2(fun, m, 1) += *(double *) PyArray_GETPTR1(x, 4);

            if (*do_grad){
            /*
            #################
             Energy gradient
            #################
            */
            // dE/deco
            *(double *) PyArray_GETPTR3(grad, m, 0, 0) = dGdeco / *dx;
            // dE/dsco 
            *(double *) PyArray_GETPTR3(grad, m, 1, 0) = 0.;
            // dE/dgco        
            *(double *) PyArray_GETPTR3(grad, m, 2, 0) = dGdgcoE / *dx;
            // dE/dC_E
            *(double *) PyArray_GETPTR3(grad, m, 3, 0) = 1.;
            *(double *) PyArray_GETPTR3(grad, m, 4, 0) = 0.;

            /*
            ##################
             Entropy gradient
            ##################
            */
            // dS/dEaff
            *(double *) PyArray_GETPTR3(grad, m, 0, 1) = 0.;
            // dS/deco 
            *(double *) PyArray_GETPTR3(grad, m, 1, 1) = dGdsco / *dx;
            // dS/dSaff
            *(double *) PyArray_GETPTR3(grad, m, 2, 1) = dGdgcoS / *dx;
            // dS/dC_S
            *(double *) PyArray_GETPTR3(grad, m, 3, 1) = 0.;
            *(double *) PyArray_GETPTR3(grad, m, 4, 1) = 1.;
            }
        } 
        else {
            // Energy term
            *(double *) PyArray_GETPTR1(fun, m)  = E_dens;
            // Entropy term
            *(double *) PyArray_GETPTR1(fun, m) += S_dens;
            // Constant term
            *(double *) PyArray_GETPTR1(fun, m) += (*(double *) PyArray_GETPTR1(x, 3));
            //printf("Rec %d: %6.3f kcal/mol\n", m, *(double *) PyArray_GETPTR1(fun, m));

            if (*do_grad){
            // dG/deco
            *(double *) PyArray_GETPTR2(grad, m, 0) = dGdeco / *dx;
            // dG/dsco 
            *(double *) PyArray_GETPTR2(grad, m, 1) = dGdsco / *dx;
            // dG/dgco        
            *(double *) PyArray_GETPTR2(grad, m, 2)  = (dGdgcoE+dGdgcoS) / *dx;
            // dG/dC
            *(double *) PyArray_GETPTR2(grad, m, 3) = 1.;

            //printf("Rec %d: [ %10.6f, %10.6f, %10.6f, %10.6f ] kcal/mol\n", m,
            //    *(double *) PyArray_GETPTR2(grad, m, 0),
            //    *(double *) PyArray_GETPTR2(grad, m, 1),
            //    *(double *) PyArray_GETPTR2(grad, m, 2),
            //    *(double *) PyArray_GETPTR2(grad, m, 3));
            }
        }
    }

    return 1;

}


int gist_functional_5p (PyArrayObject *E,
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
                        int          *verbose) {

    /*
    x elements: 0: Aff 
                1: e_co
                2: s_co
                3: g_co
                4: C
    */
    int i,j,k,l,m; // some counting variables
    double upper[5]; // boundaries for 2 point finite differences
    int Npose;   // Number of poses
    int bins[3]; // Number of bins in each direction
    int Nthreads;
    double dGdE, dGdeco, dGdS, dGdsco, dGdgcoE, dGdgcoS;
    double *E_val, *S_val, *g_val; // Store values from E, S and g during processing
    double vol_val; // Store values from vol during processing
    int e_i, s_i, g_i;    // binary indicator
    int de_i, ds_i, dg_i; // differential of binary indicator
    de_i=0, ds_i=0, dg_i=0;
    e_i=0,  s_i=0,  g_i=0;

    if (*do_grad){
        upper[0] = *(double *) PyArray_GETPTR1(x, 0) + *dx;
        upper[1] = *(double *) PyArray_GETPTR1(x, 1) + *dx;
        upper[2] = *(double *) PyArray_GETPTR1(x, 2) + *dx;
        upper[3] = *(double *) PyArray_GETPTR1(x, 3) + *dx;
        upper[4] = *(double *) PyArray_GETPTR1(x, 4) + *dx;
    }

    Npose   = PyArray_DIM(vol, 0);
    bins[0] = PyArray_DIM(E, 1);
    bins[1] = PyArray_DIM(E, 2);
    bins[2] = PyArray_DIM(E, 3);

    Nthreads = omp_get_max_threads();

    if (Nthreads>Npose){
        Nthreads=Npose;
        omp_set_num_threads(Nthreads);
    }

    #pragma omp parallel for private(i,j,k,l,m, \
                            E_val, S_val, \
                            dGdE, dGdeco, dGdS, \
                            dGdsco, dGdgcoE, dGdgcoS, \
                            g_val, vol_val) \
                        firstprivate(e_i, s_i, g_i, \
                            de_i, ds_i, dg_i, \
                            g, S, E, vol, \
                            x, dx, upper, ind) \
                        shared(do_grad, decomp, verbose, \
                            bins, Npose, fun, grad) \
                        default(none)
    for (m=0; m<Npose; m++){

        //int ithread;
        //ithread = omp_get_thread_num();
        //if (ithread==0) printf("m: %d\n", m);

        dGdE    = 0.;
        dGdeco  = 0.;
        dGdS    = 0.;
        dGdsco  = 0.;
        dGdgcoE = 0.;
        dGdgcoS = 0.;

        for (i=0; i<bins[0]; i++){
        for (j=0; j<bins[1]; j++){
        for (k=0; k<bins[2]; k++){

            vol_val = *(double *) PyArray_GETPTR4(vol, m, i, j, k);
            if (vol_val>0.000001){

            l = *(int *) PyArray_GETPTR1(ind, m);

            g_val = (double *) PyArray_GETPTR4(g, l, i, j, k);
            if (*g_val>*(double *) PyArray_GETPTR1(x, 3)) g_i = 1;
            else g_i = 0;
            if (*do_grad){
                if (*g_val>upper[3]) dg_i = 1;
                else dg_i = 0;
                dg_i = dg_i-g_i;
            }

            if (g_i!=0 || dg_i!=0){

                E_val = (double *) PyArray_GETPTR4(E, l, i, j, k);
                if (*E_val>*(double *) PyArray_GETPTR1(x, 1)) e_i = 1;
                else e_i = 0;
                if (*do_grad){
                    if (*E_val>upper[1]) de_i = 1;
                    else de_i = 0;
                    de_i = de_i-e_i;
                }
                
                S_val = (double *) PyArray_GETPTR4(S, l, i, j, k);
                if (*S_val>*(double *) PyArray_GETPTR1(x, 2)) s_i = 1;
                else s_i = 0;
                if (*do_grad){
                    if (*S_val>upper[2]) ds_i = 1;
                    else ds_i = 0;
                    ds_i = ds_i-s_i;
                }

                if (g_i>0){
                    if (e_i>0) dGdE += vol_val;
                    if (s_i>0) dGdS += vol_val;
                    if (*do_grad){
                        if (de_i>0)      dGdeco += vol_val;
                        else if (de_i<0) dGdeco -= vol_val;
                        if (ds_i>0)      dGdsco += vol_val;
                        else if (ds_i<0) dGdsco -= vol_val;
                    }
                }
                if (*do_grad){
                    if (dg_i>0){
                        if (e_i>0) dGdgcoE += vol_val;
                        if (s_i>0) dGdgcoS += vol_val;
                    }
                    else if (dg_i<0){
                        if (e_i>0) dGdgcoE -= vol_val;
                        if (s_i>0) dGdgcoS -= vol_val;
                    }
                }
            }
            }
        }
        }
        }

        if (*decomp) {
            
            /*
            #################
             Energy function
            #################
            */
            // Energy term
            *(double *) PyArray_GETPTR2(fun, m, 0)  = *(double *) PyArray_GETPTR1(x, 0) * dGdE;
            // Constant term C_E
            *(double *) PyArray_GETPTR2(fun, m, 0) += *(double *) PyArray_GETPTR1(x, 4);

            if (*do_grad){
            /*
            ##################
             Entropy function
            ##################
            */
            // Entropy term
            *(double *) PyArray_GETPTR2(fun, m, 1)  = *(double *) PyArray_GETPTR1(x, 0) * dGdS;
            // Constant term C_S
            *(double *) PyArray_GETPTR2(fun, m, 1) += *(double *) PyArray_GETPTR1(x, 5);

            /*
            #################
             Energy gradient
            #################
            */
            // dE/dAff
            *(double *) PyArray_GETPTR3(grad, m, 0, 0)  = dGdE;
            // dE/deco 
            *(double *) PyArray_GETPTR3(grad, m, 1, 0)  = *(double *) PyArray_GETPTR1(x, 0) * dGdeco;
            *(double *) PyArray_GETPTR3(grad, m, 1, 0) /= *dx;
            // dE/dsco        
            *(double *) PyArray_GETPTR3(grad, m, 2, 0)  = 0.;
            // dE/dgco
            *(double *) PyArray_GETPTR3(grad, m, 3, 0)  = *(double *) PyArray_GETPTR1(x, 0) * dGdgcoE;
            *(double *) PyArray_GETPTR3(grad, m, 3, 0) /= *dx;
            // dE/dC_E
            *(double *) PyArray_GETPTR3(grad, m, 4, 0) = 1.;
            *(double *) PyArray_GETPTR3(grad, m, 5, 0) = 0.;

            /*
            ##################
             Entropy gradient
            ##################
            */
            // dS/dAff
            *(double *) PyArray_GETPTR3(grad, m, 0, 1) = dGdS;
            // dS/deco 
            *(double *) PyArray_GETPTR3(grad, m, 1, 1) = 0.;
            // dS/dsco        
            *(double *) PyArray_GETPTR3(grad, m, 2, 1)  = *(double *) PyArray_GETPTR1(x, 0) * dGdsco;
            *(double *) PyArray_GETPTR3(grad, m, 2, 1) /= *dx;
            // dS/dgco
            *(double *) PyArray_GETPTR3(grad, m, 3, 1)  = *(double *) PyArray_GETPTR1(x, 0) * dGdgcoS;
            *(double *) PyArray_GETPTR3(grad, m, 3, 1) /= *dx;
            // dS/dC_S
            *(double *) PyArray_GETPTR3(grad, m, 4, 1) = 0.;
            *(double *) PyArray_GETPTR3(grad, m, 5, 1) = 1.;
            }
        }
        else {
            // Energy term
            *(double *) PyArray_GETPTR1(fun, m)  = *(double *) PyArray_GETPTR1(x, 0) * dGdE;
            // Entropy term
            *(double *) PyArray_GETPTR1(fun, m) += *(double *) PyArray_GETPTR1(x, 0) * dGdS;
            // Constant term
            *(double *) PyArray_GETPTR1(fun, m) += *(double *) PyArray_GETPTR1(x, 4);
            
            if (*do_grad){
            // dG/dAff
            *(double *) PyArray_GETPTR2(grad, m, 0)  = dGdE+dGdS;
            // dG/deco 
            *(double *) PyArray_GETPTR2(grad, m, 1)  = *(double *) PyArray_GETPTR1(x, 0) * dGdeco;
            *(double *) PyArray_GETPTR2(grad, m, 1) /= *dx;
            // dG/dsco        
            *(double *) PyArray_GETPTR2(grad, m, 2)  = *(double *) PyArray_GETPTR1(x, 0) * dGdsco;
            *(double *) PyArray_GETPTR2(grad, m, 2) /= *dx;
            // dG/dgco
            *(double *) PyArray_GETPTR2(grad, m, 3)  =  *(double *) PyArray_GETPTR1(x, 0) * dGdgcoE;
            *(double *) PyArray_GETPTR2(grad, m, 3) +=  *(double *) PyArray_GETPTR1(x, 0) * dGdgcoS;
            *(double *) PyArray_GETPTR2(grad, m, 3) /=  *dx;
            // dG/dC
            *(double *) PyArray_GETPTR2(grad, m, 4) = 1.;
            }
        //printf("m: %d, fun: %6.3f \n", m, *(double *) PyArray_GETPTR1(fun, m));
        }
    }

    return 1;

}


int gist_functional_6p (PyArrayObject *E,
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
                        int          *verbose) {

    /*
    x elements: 0: E_aff 
                1: e_co
                2: S_aff
                3: s_co
                4: g_co
                5: C
    */
    int i,j,k,l,m; // some counting variables
    double upper[6]; // boundaries for 2 point finite differences
    int Npose; // Number of poses
    int bins[3]; // Number of bins in each direction
    int Nthreads;
    double dGdE, dGdeco, dGdS, dGdsco, dGdgcoE, dGdgcoS;
    double *E_val, *S_val, *g_val; // Store values from E, S and g during processing
    double vol_val; // Store values from vol during processing
    int e_i, s_i, g_i;    // binary indicator
    int de_i, ds_i, dg_i; // differential of binary indicator
    de_i=0, ds_i=0, dg_i=0;
    e_i=0,  s_i=0,  g_i=0;

    if (*do_grad){
        upper[0] = *(double *) PyArray_GETPTR1(x, 0) + *dx;
        upper[1] = *(double *) PyArray_GETPTR1(x, 1) + *dx;
        upper[2] = *(double *) PyArray_GETPTR1(x, 2) + *dx;
        upper[3] = *(double *) PyArray_GETPTR1(x, 3) + *dx;
        upper[4] = *(double *) PyArray_GETPTR1(x, 4) + *dx;
        upper[5] = *(double *) PyArray_GETPTR1(x, 5) + *dx;
    }

    Npose   = PyArray_DIM(vol, 0);
    bins[0] = PyArray_DIM(E, 1);
    bins[1] = PyArray_DIM(E, 2);
    bins[2] = PyArray_DIM(E, 3);

    Nthreads = omp_get_max_threads();

    if (Nthreads>Npose){
        Nthreads=Npose;
        omp_set_num_threads(Nthreads);
    }

    #pragma omp parallel for private(i,j,k,l,m, \
                            E_val, S_val, \
                            dGdE, dGdeco, dGdS, \
                            dGdsco, dGdgcoE, dGdgcoS, \
                            g_val, vol_val) \
                        firstprivate(e_i, s_i, g_i, \
                            de_i, ds_i, dg_i, \
                            g, S, E, vol, \
                            x, dx, upper, ind) \
                        shared(do_grad, decomp, verbose, \
                            bins, Npose, fun, grad) \
                        default(none)
    for (m=0; m<Npose; m++){

        //int ithread;
        //ithread = omp_get_thread_num();
        //if (ithread==0) printf("m: %d\n", m);

        dGdE    = 0.;
        dGdeco  = 0.;
        dGdS    = 0.;
        dGdsco  = 0.;
        dGdgcoE = 0.;
        dGdgcoS = 0.;

        for (i=0; i<bins[0]; i++){
        for (j=0; j<bins[1]; j++){
        for (k=0; k<bins[2]; k++){

            vol_val = *(double *) PyArray_GETPTR4(vol, m, i, j, k);
            if (vol_val>0.000001){

            l = *(int *) PyArray_GETPTR1(ind, m);

            g_val = (double *) PyArray_GETPTR4(g, l, i, j, k);
            if (*g_val>*(double *) PyArray_GETPTR1(x, 4)) g_i = 1;
            else g_i = 0;
            if (*do_grad){
                if (*g_val>upper[4]) dg_i = 1;
                else dg_i = 0;
                dg_i = dg_i-g_i;
            }

            if (g_i!=0 || dg_i!=0){

                E_val = (double *) PyArray_GETPTR4(E, l, i, j, k);
                if (*E_val>*(double *) PyArray_GETPTR1(x, 1)) e_i = 1;
                else e_i = 0;
                if (*do_grad){
                    if (*E_val>upper[1]) de_i = 1;
                    else de_i = 0;
                    de_i = de_i-e_i;
                }
                
                S_val = (double *) PyArray_GETPTR4(S, l, i, j, k);
                if (*S_val>*(double *) PyArray_GETPTR1(x, 3)) s_i = 1;
                else s_i = 0;
                if (*do_grad){
                    if (*S_val>upper[3]) ds_i = 1;
                    else ds_i = 0;
                    ds_i = ds_i-s_i;
                }

                if (g_i>0){
                    if (e_i>0) dGdE += vol_val;
                    if (s_i>0) dGdS += vol_val;
                    if (*do_grad){
                        if (de_i>0)      dGdeco += vol_val;
                        else if (de_i<0) dGdeco -= vol_val;
                        if (ds_i>0)      dGdsco += vol_val;
                        else if (ds_i<0) dGdsco -= vol_val;
                    }
                }
                if (*do_grad){
                    if (dg_i>0){
                        if (e_i>0) dGdgcoE += vol_val;
                        if (s_i>0) dGdgcoS += vol_val;
                    }
                    else if (dg_i<0){
                        if (e_i>0) dGdgcoE -= vol_val;
                        if (s_i>0) dGdgcoS -= vol_val;
                    }
                }
            }
            }
        }
        }
        }

        if (*decomp) {
            
            /*
            #################
             Energy function
            #################
            */
            // Energy term
            *(double *) PyArray_GETPTR2(fun, m, 0) = *(double *) PyArray_GETPTR1(x, 0) * dGdE;
            // Constant term C_E
            *(double *) PyArray_GETPTR2(fun, m, 0) += *(double *) PyArray_GETPTR1(x, 5);

            if (*do_grad){
            /*
            ##################
             Entropy function
            ##################
            */
            // Entropy term
            *(double *) PyArray_GETPTR2(fun, m, 1)  = *(double *) PyArray_GETPTR1(x, 2) * dGdS;
            // Constant term C_S
            *(double *) PyArray_GETPTR2(fun, m, 1) += *(double *) PyArray_GETPTR1(x, 6);

            /*
            #################
             Energy gradient
            #################
            */
            // dE/dEaff
            *(double *) PyArray_GETPTR3(grad, m, 0, 0) = dGdE;
            // dE/deco 
            *(double *) PyArray_GETPTR3(grad, m, 1, 0)  = *(double *) PyArray_GETPTR1(x, 0) * dGdeco;
            *(double *) PyArray_GETPTR3(grad, m, 1, 0) /= *dx;
            // dE/dSaff
            *(double *) PyArray_GETPTR3(grad, m, 2, 0) = 0.;
            // dE/dsco        
            *(double *) PyArray_GETPTR3(grad, m, 3, 0) = 0.;
            // dE/dgco
            *(double *) PyArray_GETPTR3(grad, m, 4, 0)  = *(double *) PyArray_GETPTR1(x, 0) * dGdgcoE;
            *(double *) PyArray_GETPTR3(grad, m, 4, 0) /= *dx;
            // dE/dC_E
            *(double *) PyArray_GETPTR3(grad, m, 5, 0) = 1.;
            *(double *) PyArray_GETPTR3(grad, m, 6, 0) = 0.;

            /*
            ##################
             Entropy gradient
            ##################
            */
            // dS/dEaff
            *(double *) PyArray_GETPTR3(grad, m, 0, 1) = 0.;
            // dS/deco 
            *(double *) PyArray_GETPTR3(grad, m, 1, 1) = 0.;
            // dS/dSaff
            *(double *) PyArray_GETPTR3(grad, m, 2, 1) = dGdS;
            // dS/dsco        
            *(double *) PyArray_GETPTR3(grad, m, 3, 1)  = *(double *) PyArray_GETPTR1(x, 2) * dGdsco;
            *(double *) PyArray_GETPTR3(grad, m, 3, 1) /= *dx;
            // dS/dgco
            *(double *) PyArray_GETPTR3(grad, m, 4, 1)  = *(double *) PyArray_GETPTR1(x, 2) * dGdgcoS;
            *(double *) PyArray_GETPTR3(grad, m, 4, 1) /= *dx;
            // dS/dC_S
            *(double *) PyArray_GETPTR3(grad, m, 5, 1) = 0.;
            *(double *) PyArray_GETPTR3(grad, m, 6, 1) = 1.;
            }
        }
        else {
            // Energy term
            *(double *) PyArray_GETPTR1(fun, m)  = *(double *) PyArray_GETPTR1(x, 0) * dGdE;
            // Entropy term
            *(double *) PyArray_GETPTR1(fun, m) += *(double *) PyArray_GETPTR1(x, 2) * dGdS;
            // Constant term
            *(double *) PyArray_GETPTR1(fun, m) += *(double *) PyArray_GETPTR1(x, 5);
            
            if (*do_grad){
            // dG/dEaff
            *(double *) PyArray_GETPTR2(grad, m, 0) = dGdE;
            // dG/deco 
            *(double *) PyArray_GETPTR2(grad, m, 1)  = *(double *) PyArray_GETPTR1(x, 0) * dGdeco;
            *(double *) PyArray_GETPTR2(grad, m, 1) /= *dx;
            // dG/dSaff
            *(double *) PyArray_GETPTR2(grad, m, 2)  = dGdS;
            // dG/dsco        
            *(double *) PyArray_GETPTR2(grad, m, 3)  = *(double *) PyArray_GETPTR1(x, 2) * dGdsco;
            *(double *) PyArray_GETPTR2(grad, m, 3) /= *dx;
            // dG/dgco
            *(double *) PyArray_GETPTR2(grad, m, 4)  =  *(double *) PyArray_GETPTR1(x, 0) * dGdgcoE;
            *(double *) PyArray_GETPTR2(grad, m, 4) +=  *(double *) PyArray_GETPTR1(x, 2) * dGdgcoS;
            *(double *) PyArray_GETPTR2(grad, m, 4) /=  *dx;
            // dG/dC
            *(double *) PyArray_GETPTR2(grad, m, 5) = 1.;
            }
        //printf("m: %d, fun: %6.3f \n", m, *(double *) PyArray_GETPTR1(fun, m));
        }
    }

    return 1;

}