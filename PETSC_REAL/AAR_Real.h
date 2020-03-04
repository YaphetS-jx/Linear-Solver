/**
 * @file    AAR_Real.h
 * @brief   This file declares the functions for AAR solver. 
 *
 * @author  Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */

#ifndef AAR_REAL_H
#define AAR_REAL_H

#include "system.h"

void AAR(Mat A, Vec x, Vec b, PetscScalar omega, PetscScalar beta, 
    PetscInt m, PetscInt p, PetscScalar tol, int max_iter, PetscInt pc, DM da);

void precondition(PC prec, Vec res, DM da, PetscInt *blockinfo, PetscScalar *local);

void Anderson(Vec *DX, Vec *DF, Vec x, Vec res, PetscInt m, PetscScalar beta);

#endif
