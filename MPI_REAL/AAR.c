#include "AAR.h"

void AAR(DS_AAR* pAAR,
        void (*PoissonResidual)(DS_AAR*, double*, double*, int, int, MPI_Comm),
        double *x, double *rhs, double omega, double beta, int m, int p, 
        int max_iter, double tol, int Np, MPI_Comm comm_dist_graph_cart) 
{
    int rank, i, j, k, col, iter; 
    double *res, *x_old, *f_old, **DX, **DF, *am_vec, t0, t1;
    double *FtF, *allredvec, *Ftf, *svec, relres, rhs_norm;  
    //////////////////////////////////////////////////////////

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

    // allocate memory to store x(x) in domain
    res = (double*)calloc(Np, sizeof(double));  
    x_old = (double*)calloc(Np, sizeof(double));  
    f_old = (double*)calloc(Np, sizeof(double));  
    am_vec = (double*)calloc(Np, sizeof(double));  
    assert(res !=  NULL &&  x_old != NULL && f_old != NULL && am_vec != NULL); 

    // allocate memory to store DX, DF history matrices
    DX = (double**) calloc(m, sizeof(double*)); 
    DF = (double**) calloc(m, sizeof(double*)); 
    assert(DF !=  NULL && DX != NULL); 

    for (k = 0; k<m; k++) {
        DX[k] = (double*)calloc(Np, sizeof(double)); 
        DF[k] = (double*)calloc(Np, sizeof(double)); 
        assert(DX[k] !=  NULL && DF[k] !=  NULL); 
    }

    FtF = (double*) calloc(m*m, sizeof(double));                       // DF'*DF matrix in column major format
    allredvec = (double*) calloc(m*m+m, sizeof(double));               
    Ftf = (double*) calloc(m, sizeof(double));                             // DF'*res vector of size m x 1
    svec = (double*) calloc(m, sizeof(double));                            // vector to store singular values    
    

    iter = 1; 
    relres = tol+1; 
    // calculate norm of right hand side (required for relative residual)
    Vector2Norm(rhs, Np, &rhs_norm); 
    tol *= rhs_norm;
    PoissonResidual(pAAR, x, res, pAAR->np_x, pAAR->FDn, comm_dist_graph_cart); 

#ifdef DEBUG
    Vector2Norm(res, Np, &relres);
    if(!rank) printf("preconditioned reslres: %g\n", relres/rhs_norm);
#endif

    t0 = MPI_Wtime(); 
    
    // begin while loop
    while (relres > tol && iter <=  max_iter) {
        //----------Store Residual & Iterate History----------//
        if(iter>1) {
            col = ((iter-2) % m); 
            for (i = 0; i < Np; i++){
                DX[col][i] = x[i] - x_old[i];
                DF[col][i] = res[i] - f_old[i];
            }
        } 

        for (i = 0; i < Np; i++){
            x_old[i] = x[i];
            f_old[i] = res[i];
        }

        //----------Anderson update-----------//
        if(iter % p == 0 && iter>1) {
            // x_new = x_prev + beta*res - (DX + beta*DF)*(pinv(DF'*DF)*(DF'*res));

            AndersonExtrapolation(DX, DF, res, beta, m, Np, am_vec, FtF, allredvec, Ftf, svec, x, x_old); 

            PoissonResidual(pAAR, x, res, pAAR->np_x, pAAR->FDn, comm_dist_graph_cart); 

            Vector2Norm(res, Np, &relres);

#ifdef DEBUG
    if(rank == 0) printf("preconditioned reslres: %g\n", relres/rhs_norm);
#endif

        } else {
            //----------Richardson update-----------//
            // x = x + omega * res

            for (i = 0; i < Np; i++){
                x[i] = x_old[i] + omega * res[i];
            }

            PoissonResidual(pAAR, x, res, pAAR->np_x, pAAR->FDn, comm_dist_graph_cart); 

#ifdef DEBUG
    Vector2Norm(res, Np, &relres);
    if(rank == 0) printf("preconditioned relres: %g\n", relres/rhs_norm);
#endif
        }

        iter = iter+1; 
    } // end while loop

    t1 = MPI_Wtime(); 
    if(rank  ==  0) {
        if(iter < max_iter && relres <= tol) {
            printf("AAR preconditioned with Jacobi (AAJ).\n");  
            printf("AAR converged!:  Iterations = %d, preconditioned Relative Residual = %g, Time = %.4f sec\n", iter-1, relres/rhs_norm, t1-t0); 
        }
        if(iter>= max_iter) {
            printf("WARNING: AAR exceeded maximum iterations.\n"); 
            printf("AAR:  Iterations = %d, preconditioned Relative Residual = %g, Time = %.4f sec \n", iter-1, relres/rhs_norm, t1-t0); 
        }
    }

    free(res); 
    free(x_old); 
    free(f_old); 
    free(am_vec); 
  
    for (k = 0; k<m; k++) {
      free(DX[k]); 
      free(DF[k]); 
    }
    free(DX); 
    free(DF); 

    free(FtF); 
    free(Ftf); 
    free(svec); 
    free(allredvec); 
}

void AndersonExtrapolation(double **DX,  double **DF,  double *res,  double beta,  int m,  int Np,  double *am_vec,  
                            double *FtF,  double *allredvec,  double *Ftf,  double *svec, double *x, double *x_old) {
    // DX and DF are iterate and residual history matrices of size Npxm where Np = no. of nodes in proc domain and m = m
    // am_vec is the final update vector of size Npx1
    // res is the residual vector at current iteration of fixed point method

    int i, j, k, ctr, cnt; 
    
    // ------------------- First find DF'*DF m x m matrix (local and then AllReduce) ------------------- //
    double temp_sum = 0; 

    for (j = 0; j < m; j++) {
        for (i = 0; i <= j; i++) {
            temp_sum = 0; 
            for (k = 0; k < Np; k++) {
                temp_sum = temp_sum + DF[i][k]*DF[j][k]; 
            }
            ctr = j*m+i; 
            allredvec[ctr] = temp_sum; 
            ctr = i*m+j; 
            allredvec[ctr] = temp_sum; 
        }
    }

    ctr = m*m; 
    // ------------------- Compute DF'*res ------------------- //

    for (j = 0; j < m; j++) {
        temp_sum = 0; 
        for (k = 0; k < Np; k++) {
            temp_sum = temp_sum + DF[j][k]*res[k]; 
        }
        allredvec[ctr] = temp_sum; 
        ctr = ctr + 1; 
    }

    MPI_Allreduce(MPI_IN_PLACE,  allredvec,  m*m+m,  MPI_DOUBLE,  MPI_SUM,  MPI_COMM_WORLD); 

    ctr = 0; 
    for (j = 0; j < m; j++) {
        for (i = 0; i<m; i++) {
            FtF[ctr] = allredvec[ctr];  // (FtF)_ij element
            ctr = ctr + 1; 
        }
    }
    cnt = 0; 
    for (j = 0; j < m; j++) {
          Ftf[cnt] = allredvec[ctr];  // (Ftf)_j element
          ctr = ctr + 1; 
          cnt = cnt+1; 
    }

    PseudoInverseTimesVec(FtF, Ftf, svec, m);  // svec = Y      

    // ------------------- Compute Anderson update vector am_vec ------------------- //
    for (k = 0; k < Np; k++) {
        temp_sum = 0; 
        for (j = 0; j < m; j++) {
            temp_sum = temp_sum + (DX[j][k] + beta*DF[j][k])*svec[j]; 
        }
        am_vec[k] = temp_sum;  // (am_vec)_k element
    }

    for (i = 0; i < Np; i ++){
        x[i] = x_old[i] + beta * res[i] - am_vec[i];
    }
}


