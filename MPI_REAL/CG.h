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

void CG(POISSON *system,
        void (*Lap_Vec_mult)(POISSON *, double, double *, double *, MPI_Comm),
        void (*Precondition)(double, double *, int), double a,
        double *x, double *b, int max_iter, double tol, int Np, MPI_Comm comm);

#endif