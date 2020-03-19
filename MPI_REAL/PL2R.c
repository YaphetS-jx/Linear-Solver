/**
 * @file    AAR.c
 * @brief   This file contains functions for Alternating Anderson Richardson solver
 *
 * @author  Xin Jing  < xjing30@gatech.edu>
 *          Phanish Suryanarayana  < phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */


#include "AAR.h"

void AAR(DS_AAR* pAAR,
        void (*PoissonResidual)(DS_AAR*, double*, double*, int, int, MPI_Comm),
        double *x, double *rhs, double omega, double beta, int m, int p, 
        int max_iter, double tol, int Np, MPI_Comm comm_dist_graph_cart) 
{
    int rank, i, j, k, col, iter; 
    double *f, *x_old, *f_old, **DX, **DF, *am_vec, t0, t1;
    double **FtF, *allredvec, *Ftf, *svec, relf, rhs_norm;  
    //////////////////////////////////////////////////////////

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

    // allocate memory to store x(x) in domain
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

    for (k = 0; k < m; k++) {
        DX[k] = (double*)calloc(Np, sizeof(double)); 
        DF[k] = (double*)calloc(Np, sizeof(double)); 
        FtF[k] = (double*)calloc(m, sizeof(double)); 
        assert(DX[k] !=  NULL && DF[k] !=  NULL && FtF[k] !=  NULL); 
    }
    
    am_vec = (double*)calloc(Np, sizeof(double));  
    allredvec = (double*) calloc(m*m+m, sizeof(double));               
    svec = (double*) calloc(m, sizeof(double));                            // singular values 
    assert(am_vec !=  NULL && allredvec != NULL && svec != NULL); 

    iter = 1; 
    relf = tol+1; 
    // calculate norm of right hand side (required for relative preconditioned residual)
    Vector2Norm(rhs, Np, &rhs_norm); 
    tol *= rhs_norm;
    PoissonResidual(pAAR, x, f, pAAR->np_x, pAAR->FDn, comm_dist_graph_cart); 

#ifdef DEBUG
    Vector2Norm(f, Np, &relf);
    if(!rank) printf("preconditioned Relative Residual: %g\n", relf/rhs_norm);
#endif

    t0 = MPI_Wtime(); 
    
    // begin while loop
    while (relf > tol && iter  <=  max_iter) {
        //----------Store Residual & Iterate History----------//
        if(iter>1) {
            col = ((iter-2) % m); 
            for (i = 0; i < Np; i++){
                DX[col][i] = x[i] - x_old[i];
                DF[col][i] = f[i] - f_old[i];
            }
        } 

        for (i = 0; i < Np; i++){
            x_old[i] = x[i];
            f_old[i] = f[i];
        }

        //----------Anderson update-----------//
        if(iter % p == 0 && iter>1) {
            // x_new = x_prev + beta*f - (DX + beta*DF)*(pinv(DF'*DF)*(DF'*f));

            AndersonExtrapolation(DX, DF, f, beta, m, Np, am_vec, FtF, allredvec, Ftf, svec, x, x_old); 

            PoissonResidual(pAAR, x, f, pAAR->np_x, pAAR->FDn, comm_dist_graph_cart); 

            Vector2Norm(f, Np, &relf);

#ifdef DEBUG
    if(rank == 0) printf("preconditioned Relative Residual: %g\n", relf/rhs_norm);
#endif

        } else {
            //----------Richardson update-----------//
            // x = x + omega * f

            for (i = 0; i < Np; i++)
                x[i] = x_old[i] + omega * f[i];

            PoissonResidual(pAAR, x, f, pAAR->np_x, pAAR->FDn, comm_dist_graph_cart); 

#ifdef DEBUG
    Vector2Norm(f, Np, &relf);
    if(rank == 0) printf("preconditioned relf: %g\n", relf/rhs_norm);
#endif
        }

        iter = iter+1; 
    } // end while loop

    t1 = MPI_Wtime(); 
    if(rank  ==  0) {
        if(iter < max_iter && relf  <= tol) {
            printf("AAR preconditioned with Jacobi (AAJ).\n");  
            printf("AAR converged!:  Iterations = %d, preconditioned Relative Residual = %g, Time = %.4f sec\n", iter-1, relf/rhs_norm, t1-t0); 
        }
        if(iter>= max_iter) {
            printf("WARNING: AAR exceeded maximum iterations.\n"); 
            printf("AAR:  Iterations = %d, preconditioned Relative Residual = %g, Time = %.4f sec \n", iter-1, relf/rhs_norm, t1-t0); 
        }
    }

    free(f); 
    free(x_old); 
    free(f_old); 
    free(am_vec); 
  
    for (k = 0; k < m; k++) {
      free(DX[k]); 
      free(DF[k]); 
      free(FtF[k]); 
    }
    free(DX); 
    free(DF); 
    free(FtF); 
    free(Ftf); 
    free(svec); 
    free(allredvec); 
}

void AndersonExtrapolation(double **DX,  double **DF,  double *f,  double beta,  int m,  int Np,  double *am_vec,  
                            double **FtF,  double *allredvec,  double *Ftf,  double *svec, double *x, double *x_old) {
    // DX and DF are iterate and preconditioned residual history matrices of size Npxm where Np = no. of nodes in proc domain
    // am_vec is the final update vector of size Npx1
    // f is the preconditioned residual vector at current iteration of fixed point method

    int i, j, k, ctr, cnt; 
    
    // ------------------- First find DF'*DF m x m matrix (local and then AllReduce) ------------------- //
    double temp_sum = 0; 

    for (j = 0; j < m; j++) {
        for (i = 0; i  <= j; i++) {
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

    MPI_Allreduce(MPI_IN_PLACE,  allredvec,  m*m+m,  MPI_DOUBLE,  MPI_SUM,  MPI_COMM_WORLD); 

    ctr = 0; 
    for (j = 0; j < m; j++) 
        for (i = 0; i < m; i++) 
            FtF[i][j] = allredvec[ctr++];
        
    cnt = 0; 
    for (j = 0; j < m; j++) 
          Ftf[cnt++] = allredvec[ctr++]; 

    PseudoInverseTimesVec(FtF, Ftf, svec, m);

    // ------------------- Compute Anderson update vector am_vec ------------------- //
    for (k = 0; k < Np; k++) {
        temp_sum = 0; 
        for (j = 0; j < m; j++) {
            temp_sum = temp_sum + (DX[j][k] + beta*DF[j][k])*svec[j]; 
        }
        am_vec[k] = temp_sum;
    }

    // ------------------- update x ------------------- //
    for (i = 0; i < Np; i ++){
        x[i] = x_old[i] + beta * f[i] - am_vec[i];
    }
}


