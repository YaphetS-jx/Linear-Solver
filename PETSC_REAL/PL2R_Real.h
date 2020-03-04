/**
 * @file    PL2R_Real.h
 * @brief   This file declares the functions for PL2R solver. 
 *
 * @author  Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */

#ifndef PL2R_REAL_H
#define PL2R_REAL_H

#include "system.h"

void PL2R(Mat A, Vec x, Vec b, PetscScalar omega, PetscScalar beta, 
    PetscInt m, PetscInt p, PetscScalar tol, int max_iter, PetscInt pc, DM da);

void L2_Richardson(Vec *DX, Vec *DF, Vec x, Vec x_old, Vec res, Vec f_old, PetscInt m, int k);

#endif
