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

int strings_starts_with(char *str1, char *str2){
    if(strncmp(str1, str2, strlen(str2)) == 0) return 1;
    return 0;
}

static PyObject *
_read_gist_ext(PyObject *self, PyObject *args) {

    char *filename;
    PyArrayObject *gist_data; // This is the python array object we are going to pupulate
    int n_rows=0, n_cols=0; // number of rows and columns in gist data file
    int start_row=0; // First row that contains data
    int found_data=0; // indicates where we are in the file
    int is_data=0; // indicates if current line contains data or not
    int verbose=0; // are we verbose or not

    //Here we retrieve the data from python
    if (!PyArg_ParseTuple(args, "s|i",
        &filename,
        &verbose
        ))
    {
        return NULL;
    }

    if (verbose){
        printf("The filename is %s\n", filename);
    }

    FILE * fp;
    fp = fopen(filename, "r");
    char *row = NULL, *col;
    size_t len = 0;
    ssize_t read;

    if (fp == NULL){
        fprintf(stderr, "open of file %s failed: %s\n", filename, strerror(errno));
        exit(EXIT_FAILURE);
    }

    while ((read = getline(&row, &len, fp)) != -1) {
        is_data=1;
        if (len==0)
            is_data=0;
        if (strings_starts_with(row, "\n")) 
            is_data=0;
        if (strings_starts_with(row, "\t")) 
            is_data=0;
        if (strings_starts_with(row, "#"))
            is_data=0;
        if (strings_starts_with(row, "GIST"))
            is_data=0;
        if (strings_starts_with(row, "voxel"))
            is_data=0;
        if (is_data){
            if (!found_data){
                found_data=1;
                while ( (col = strsep(&row," ")) != NULL ){
                    if (strings_starts_with(col, "\n")) 
                        continue;
                    if (strings_starts_with(col, "\t")) 
                        continue;
                    n_cols++;
                }
            }
            n_rows++;
        }
        else {
            if (!found_data) start_row++;
        }
    }

    fclose(fp);
    row=NULL;

    if (verbose){
        printf("Found %d rows and %d columns.\n", n_rows, n_cols);
        printf("Start row is %d.\n", start_row);
    }

    npy_intp dims[2];
    dims[0] = n_rows;
    dims[1] = n_cols;
    gist_data = PyArray_SimpleNew(2, dims, NPY_DOUBLE);

    fp = fopen(filename, "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

    n_rows=0;
    while ((read = getline(&row, &len, fp)) != -1) {
        is_data=1;
        if (len==0)
            is_data=0;
        if (strings_starts_with(row, "\n")) 
            is_data=0;
        if (strings_starts_with(row, "\t")) 
            is_data=0;
        if (strings_starts_with(row, "#"))
            is_data=0;
        if (strings_starts_with(row, "GIST"))
            is_data=0;
        if (strings_starts_with(row, "voxel"))
            is_data=0;
        if (is_data){
            n_cols=0;
            while( (col = strsep(&row," ")) != NULL ){
                if (strings_starts_with(col, "\n")) 
                    continue;
                if (strings_starts_with(col, "\t")) 
                    continue;
                *(double *)PyArray_GETPTR2(gist_data, n_rows, n_cols) = atof(col);
                n_cols++;
            }
            n_rows++;
        }
    }
    
    fclose(fp);
    if (row) 
        free(row);

    return PyArray_Return(gist_data);

}


static PyMethodDef _read_ext_methods[] = {
    {
        "read_gist_ext",
        (PyCFunction)_read_gist_ext,
        METH_VARARGS,
        "Reads GIST data from file."
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

MOD_INIT(_read_ext)
{
    PyObject *m;

    MOD_DEF(m, "_read_ext", "Routines for reading GIST data.\n", _read_ext_methods)
    
    if (m == NULL)
        return MOD_ERROR_VAL;

    import_array();

    return MOD_SUCCESS_VAL(m);
}