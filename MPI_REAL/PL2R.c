/**
 * @file    PL2R.c
 * @brief   This file contains functions for Periodic L2_Richardson solver. 
 *
 * @author  Xin Jing  < xjing30@gatech.edu>
 *          Phanish Suryanarayana  < phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */


#include "PL2R.h"

void PL2R(DS* pAAR,
        void (*PoissonResidual)(DS*, double*, double*, int, int, MPI_Comm),
        void (*Precondition)(double, double *, int),
        double *x, double *rhs, double omega, int m, int p, 
        int max_iter, double tol, int Np, MPI_Comm comm_dist_graph_cart) 
{
    int rank, i, j, col, iter; 
    double *Ax, *Ax_old, *f, *x_old, *f_old, **DX, **DF, t0, t1;
    double **FtF, *allredvec, *Ftf, *svec, relres, rhs_norm;  
    //////////////////////////////////////////////////////////

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

    // allocate memory to store x(x) in domain
    Ax = (double*)calloc(Np, sizeof(double));  
    Ax_old = (double*)calloc(Np, sizeof(double));  
    f = (double*)calloc(Np, sizeof(double));                               // f = inv(M) * res
    x_old = (double*)calloc(Np, sizeof(double));  
    f_old = (double*)calloc(Np, sizeof(double));  
    Ftf = (double*) calloc(m, sizeof(double));                             // DF'*f 
    assert(f !=  NULL &&  x_old != NULL && f_old != NULL && Ftf != NULL); 

    // allocate memory to store DX, DF history matrices and DF'*DF
    DX = (double**) calloc(m, sizeof(double*)); 
    DF = (double**) calloc(m, sizeof(double*)); 
    FtF = (double**) calloc(m, sizeof(double*));
    assert(DF !=  NULL && DX != NULL && FtF != NULL); 

    for (i = 0; i < m; i++) {
        DX[i] = (double*)calloc(Np, sizeof(double)); 
        DF[i] = (double*)calloc(Np, sizeof(double)); 
        FtF[i] = (double*)calloc(m, sizeof(double)); 
        assert(DX[i] !=  NULL && DF[i] !=  NULL && FtF[i] !=  NULL); 
    }
    
    allredvec = (double*) calloc(m*m+m, sizeof(double));               
    svec = (double*) calloc(m, sizeof(double));                            // singular values 
    assert(allredvec != NULL && svec != NULL); 

    iter = 1; 
    relres = tol+1; 
    
    Vector2Norm(rhs, Np, &rhs_norm); 
    tol *= rhs_norm;
    PoissonResidual(pAAR, x, Ax, pAAR->np_x, pAAR->FDn, comm_dist_graph_cart); 
    for (i = 0; i < Np; i++)
        f[i] = rhs[i] - Ax[i];

#ifdef DEBUG
    Vector2Norm(f, Np, &relres);
    if(!rank) printf("Relative Residual: %g\n", relres/rhs_norm);
#endif

    t0 = MPI_Wtime(); 

    while (relres > tol && iter  <=  max_iter) {
        Precondition(-(3*pAAR->coeff_lap[0]/4/M_PI), f, Np);

        //----------Store Residual & Iterate History----------//
        for (i = 0; i < Np; i++){
            x_old[i] = x[i];
            Ax_old[i] = Ax[i];
            x[i] += (omega * f[i]);
        }

        PoissonResidual(pAAR, x, Ax, pAAR->np_x, pAAR->FDn, comm_dist_graph_cart); 
        for (i = 0; i < Np; i++)
            f[i] = rhs[i] - Ax[i];

        col = (iter-2+m) % m; 
        for (i = 0; i < Np; i++){
            DX[col][i] = x[i] - x_old[i];
            DF[col][i] = Ax[i] - Ax_old[i];
        }

        //----------Galerkin Richardson update-----------//
        if(iter % p == 0) {
            L2_Richardson(DF, f, m, Np, FtF, allredvec, Ftf, svec); 

            for (i = 0; i < m; i ++) 
                for (j = 0; j < Np; j ++){
                    x[j] = x[j] + DX[i][j] * svec[i];
                    Ax[j] = Ax[j] + DF[i][j] * svec[i];
                }

            // update DX[k] and DF[k]
            for (i = 0; i < Np; i++){
                DX[col][i] = x[i] - x_old[i];
                DF[col][i] = Ax[i] - Ax_old[i];
            }
        } 

        for (i = 0; i < Np; i++)
            f[i] = rhs[i] - Ax[i];

        if (iter % p == 0) 
            Vector2Norm(f, Np, &relres);

#ifdef DEBUG
    if (iter % p)
        Vector2Norm(f, Np, &relres);
    if(rank == 0) printf("Relative Residual: %g\n", relres/rhs_norm);
#endif        

        iter = iter+1; 
    } 

    t1 = MPI_Wtime(); 
    if(rank  ==  0) {
        if(iter < max_iter && relres  <= tol) {
            printf("PL2R converged!:  Iterations = %d, Relative Residual = %g, Time = %.4f sec\n", iter-1, relres/rhs_norm, t1-t0); 
        }
        if(iter>= max_iter) {
            printf("WARNING: PL2R exceeded maximum iterations.\n"); 
            printf("PL2R:  Iterations = %d, Relative Residual = %g, Time = %.4f sec \n", iter-1, relres/rhs_norm, t1-t0); 
        }
    }

    free(Ax);
    free(Ax_old);
    free(f); 
    free(x_old); 
    free(f_old); 
  
    for (i = 0; i < m; i++) {
      free(DX[i]); 
      free(DF[i]); 
      free(FtF[i]); 
    }
    free(DX); 
    free(DF); 
    free(FtF); 
    free(Ftf); 
    free(svec); 
    free(allredvec); 
}

/**
 * @brief   L2_Richardson update
 *
 *          x_new = x_prev + DX * (pinv(DF'*DF)*(DF'*res));
 */

void L2_Richardson( double **DF, double *f, int m, int Np, double **FtF, 
                    double *allredvec, double *Ftf, double *svec) 
{
    int i, j, k, ctr, cnt; 
    
    // ------------------- First find DF'*DF m x m matrix (local and then AllReduce) ------------------- //
    double temp_sum; 

    for (j = 0; j < m; j++) {
        for (i = 0; i <= j; i++) {
            temp_sum = 0; 
            for (k = 0; k < Np; k++) {
                temp_sum = temp_sum + DF[i][k]*DF[j][k]; 
            }
            allredvec[j*m + i] = temp_sum; 
            allredvec[i*m + j] = temp_sum; 
        }
    }

    ctr = m * m; 
    // ------------------- Compute DF'*f ------------------- //

    for (j = 0; j < m; j++) {
        temp_sum = 0; 
        for (k = 0; k < Np; k++) 
            temp_sum = temp_sum + DF[j][k]*f[k]; 
        allredvec[ctr++] = temp_sum; 
    }

    MPI_Allreduce(MPI_IN_PLACE, allredvec, m*m+m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 

    ctr = 0; 
    for (j = 0; j < m; j++) 
        for (i = 0; i < m; i++) 
            FtF[i][j] = allredvec[ctr++];
        
    cnt = 0; 
    for (j = 0; j < m; j++) 
          Ftf[cnt++] = allredvec[ctr++]; 

    PseudoInverseTimesVec(FtF, Ftf, svec, m);
}


