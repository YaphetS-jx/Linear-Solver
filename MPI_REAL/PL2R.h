/**
 * @file    AAR_Real.h
 * @brief   This file declares functions for Alternating Anderson Richardson solver
 *
 * @author  Xin Jing  < xjing30@gatech.edu>
 *          Phanish Suryanarayana  < phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */


#ifndef AAR_H
#define AAR_H

#include "system.h"
#include "tools.h"

void AAR(DS_AAR* pAAR, 
    void (*PoissonResidual)(DS_AAR*, double*, double*, int, int, MPI_Comm),
    double *x, double *rhs, double omega_aar, double beta_aar, int m_aar, int p_aar, 
    int max_iter, double tol, int Np, MPI_Comm comm_dist_graph_cart);

void AndersonExtrapolation(double **DX, double **DF, double *f, double beta_mix, int m,
    int N, double *am_vec, double **FtF, double *allredvec, double *Ftf, double *svec, double *x, double *x_old); 

#endif