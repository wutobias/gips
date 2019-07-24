/*
### Written by Tobias Wulsdorf @ Klebe Lab, Marburg University
### tobias.wulsdorf@gmail.com
*/

#include <math.h>
#include "Vec.h"

void crossp(double *vec1, double *vec2, double *result){

    result[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1];
    result[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2];
    result[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0];

}

double dotp(double *vec1, double *vec2){

    double prod;

    prod  = vec1[0] * vec2[0];
    prod += vec1[1] * vec2[1];
    prod += vec1[2] * vec2[2];

    return prod;

}

void norm3(double *vec1){

    double norm;

    norm = dotp(vec1, vec1);
    norm = sqrt(norm);

    vec1[0] /= norm;
    vec1[1] /= norm;
    vec1[2] /= norm;

}

double diff2(double *vec1, double *vec2){

    double d,p;
    d=0;

    p   = vec1[0] - vec2[0];
    d  += p*p;
    
    p   = vec1[1] - vec2[1];
    d  += p*p;
    
    p   = vec1[2] - vec2[2];
    d  += p*p;

    return d;

}

double diff(double *vec1, double *vec2){

    double d;

    d = diff2(vec1, vec2);
    d = sqrt(d);

    return d;

}