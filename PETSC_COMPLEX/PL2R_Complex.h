/**
 * @file    PL2R_Complex.h
 * @brief   This file declares the functions for Periodic L2_Richardson 
 *          solver for complex system. 
 *
 * @author  Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */

#ifndef PL2R_COMPLEX_H
#define PL2R_COMPLEX_H

#include "system.h"
#include "tools.h"

void PL2R_Complex(Mat A, Vec x, Vec b, PetscScalar omega, PetscInt m, 
    PetscInt p, PetscReal tol, int max_iter, PetscInt pc, DM da);

void L2_Richardson(PetscScalar * DFres, Vec *DF, Vec res, PetscInt m, double *svec, PetscScalar *DFHDF,
    lapack_complex_double *lapack_DFHDF, lapack_complex_double *lapack_DFres);

#endif
