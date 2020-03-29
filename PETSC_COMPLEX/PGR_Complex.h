/**
 * @file    PRG_Complex.h
 * @brief   This file declares the functions for Periodic Galerkin Richardson 
 *          solver for complex system. 
 *
 * @author  Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */

#ifndef PGR_COMPLEX_H
#define PGR_COMPLEX_H

#include "system.h"
#include "tools.h"

void PRG_Complex(Mat A, Vec x, Vec b, PetscScalar omega,
    PetscInt m, PetscInt p, PetscReal tol, int max_iter, PetscInt pc, DM da);

void Galerkin_Richardson(PetscScalar * DXres, Vec *DX, Vec *DF, Vec res, PetscInt m, double *svec, PetscScalar *DXHDF,
    lapack_complex_double *lapack_DXHDF, lapack_complex_double *lapack_DXres);

#endif
