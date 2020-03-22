/**
 * @file    PL2R.h
 * @brief   This file declares functions for Periodic L2_Richardson solver. 
 *
 * @author  Xin Jing  < xjing30@gatech.edu>
 *          Phanish Suryanarayana  < phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */


#ifndef PGR_H
#define PGR_H

#include "system.h"
#include "tools.h"

void PL2R(DS* pAAR,
        void (*PoissonResidual)(DS*, double*, double*, int, int, MPI_Comm),
        void (*Precondition)(double, double *, int),
        double *x, double *rhs, double omega, int m, int p, 
        int max_iter, double tol, int Np, MPI_Comm comm_dist_graph_cart);

void L2_Richardson(double **DF, double *f, int m, int Np, double **XtF, 
                    double *allredvec, double *Xtf, double *svec, MPI_Comm comm);

#endif