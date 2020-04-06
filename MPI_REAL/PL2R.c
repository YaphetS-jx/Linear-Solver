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

/**
 * @brief   Periodic L2 Richardson solver
 *
 *          This function is designed to solve linear system Ax = b
 *          PoissonResidual is for matrix vector multiplication A * x, which could be 
 *          replaced by any user defined function 
 *          Precondition is for applying precondition, which could be replaced by any
 *          user defined precondition function
 *          x        : initial guess and final solution
 *          rhs      : right hand side of linear system, i.e. b
 *          omega    : relaxation parameter (for Richardson update)
 *          m        : no. of previous iterations to be considered in extrapolation
 *          p        : Perform Anderson extrapolation at every p th iteration
 *          max_iter : maximum number of iterations
 *          tol      : convergence tolerance
 *          Np       : no. of elements in this domain
 *          comm     : MPI communicator
 */

void PL2R(POISSON *system,
        void (*Lap_Vec_mult)(POISSON *, double, double *, double *, MPI_Comm),
        void (*Precondition)(double, double *, int),
        double *x, double *rhs, double omega, int m, int p, 
        int max_iter, double tol, int Np, MPI_Comm comm) 
{
    int rank, i, j, col, iter; 
    double *Ax, *Ax_old, *f, *x_old, *f_old, **DX, **DF, t0, t1;
    double **FtF, *allredvec, *Ftf, *svec, relres, rhs_norm;  
    //////////////////////////////////////////////////////////

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

    // allocate memory to store x(x) in domain
    Ax = (double*)calloc(Np, sizeof(double));  
    Ax_old = (double*)calloc(Np, sizeof(double));  
    f = (double*)calloc(Np, sizeof(double)); 
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
    svec = (double*) calloc(m, sizeof(double)); 
    assert(allredvec != NULL && svec != NULL); 

    iter = 1; 
    relres = tol+1; 
    
    Vector2Norm(rhs, Np, &rhs_norm, comm); 
    tol *= rhs_norm;
    Lap_Vec_mult(system, -1.0/(4*M_PI), x, Ax, comm);
    for (i = 0; i < Np; i++)
        f[i] = rhs[i] - Ax[i];

#ifdef DEBUG
    Vector2Norm(f, Np, &relres, comm);
    if(!rank) printf("Relative Residual: %g\n", relres/rhs_norm);
#endif

    t0 = MPI_Wtime(); 

    while (relres > tol && iter  <=  max_iter) {
        // Apply precondition here if it is provided by user
        if (Precondition != NULL)
            Precondition(-(3*system->coeff_lap[0]/4/M_PI), f, Np);           // Jacobi

        for (i = 0; i < Np; i++){
            x_old[i] = x[i];                                               // x_old = x
            Ax_old[i] = Ax[i];                                             // Ax_old = Ax
            x[i] += (omega * f[i]);                                        // x = x + omega * f
        }

        Lap_Vec_mult(system, -1.0/(4*M_PI), x, Ax, comm);
        for (i = 0; i < Np; i++)
            f[i] = rhs[i] - Ax[i];                                         // update f = rhs - Ax

        col = (iter-2+m) % m; 
        for (i = 0; i < Np; i++){
            DX[col][i] = x[i] - x_old[i];                                  // DX[col] = x - x_old
            DF[col][i] = Ax[i] - Ax_old[i];                                // DF[col] = Ax - Ax_old
        }

        //----------Galerkin Richardson update-----------//
        if(iter % p == 0) {
            L2_Richardson(DF, f, m, Np, FtF, allredvec, Ftf, svec, comm); 

            for (i = 0; i < m; i ++) 
                for (j = 0; j < Np; j ++){
                    x[j] = x[j] + DX[i][j] * svec[i];                      // x = x + DX' * svec
                    Ax[j] = Ax[j] + DF[i][j] * svec[i];                    // Ax = Ax + DF' * svec
                }

            // update DX[k] and DF[k]
            for (i = 0; i < Np; i++){
                DX[col][i] = x[i] - x_old[i];                              // update DX[col]
                DF[col][i] = Ax[i] - Ax_old[i];                            // update DF[col]
            }
        } 

        for (i = 0; i < Np; i++)
            f[i] = rhs[i] - Ax[i];                                         // update f = rhs - Ax 

        if (iter % p == 0) 
            Vector2Norm(f, Np, &relres, comm);                             // update relative residual every p iteration

#ifdef DEBUG
    if (iter % p)
        Vector2Norm(f, Np, &relres, comm);
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

    // de-allocate memory
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
 *          svec = pinv(DF'*DF)*(DF'*res);
 */

void L2_Richardson( double **DF, double *f, int m, int Np, double **FtF, 
                    double *allredvec, double *Ftf, double *svec, MPI_Comm comm) 
{
    int i, j, k, ctr, cnt; 
    double temp_sum; 
    //////////////////////////////////////////////////////////

    for (j = 0; j < m; j++) {
        for (i = 0; i <= j; i++) {
            temp_sum = 0; 
            for (k = 0; k < Np; k++) {
                temp_sum = temp_sum + DF[i][k]*DF[j][k]; 
            }
            allredvec[j*m + i] = temp_sum;                                 // allredvec(i,j) = DF[i]' * DF[j]
            allredvec[i*m + j] = temp_sum;                                 // allredvec(j,i) = allredvec(i,j)
        }
    }

    ctr = m * m; 
    for (j = 0; j < m; j++) {
        temp_sum = 0; 
        for (k = 0; k < Np; k++) 
            temp_sum = temp_sum + DF[j][k]*f[k]; 
        allredvec[ctr++] = temp_sum;                                       // allredvec(i+m*m) = DX[k]' * f
    }

    MPI_Allreduce(MPI_IN_PLACE, allredvec, m*m+m, MPI_DOUBLE, MPI_SUM, comm); 

    ctr = 0; 
    for (j = 0; j < m; j++) 
        for (i = 0; i < m; i++) 
            FtF[i][j] = allredvec[ctr++];
        
    cnt = 0; 
    for (j = 0; j < m; j++) 
          Ftf[cnt++] = allredvec[ctr++]; 

    PseudoInverseTimesVec(FtF, Ftf, svec, m);                              // svec = inv(XtF) * Xtf
}


