/**
 * @file    PRG_Real.h
 * @brief   This file declares the functions for PGR solver. 
 *
 * @author  Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */

#ifndef PGR_REAL_H
#define PGR_REAL_H

#include "system.h"

void PGR(Mat A, Vec x, Vec b, PetscScalar omega,
    PetscInt m, PetscInt p, PetscScalar tol, int max_iter, PetscInt pc, DM da);

void Galerkin_Richardson(Vec *DX, Vec *DF, Vec x, Vec x_old, Vec res, Vec f_old, PetscInt m, int k);

#endif
