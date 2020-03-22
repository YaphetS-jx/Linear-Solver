/**
 * @file    CG.h
 * @brief   This file declares functions for Conjugate Gradient solver
 *
 * @author  Xin Jing  < xjing30@gatech.edu>
 *          Phanish Suryanarayana  < phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */


#ifndef CG_H
#define CG_H

#include "system.h"
#include "tools.h"

void CG(SPARC_OBJ *pSPARC, 
    void (*PoissonResidual)(DS*, double*, double*, int, int, MPI_Comm),
    void (*Precondition)(double, double *, int),
    int N, int DMnd, double* x, double *b, double tol, int max_iter, MPI_Comm comm);

#endif