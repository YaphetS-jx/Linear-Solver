#include "AAR.h"
#include "system.h"

///////////////////////////////////////////////////////////////////////////////
////////////////////////////////// AAR solver /////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
void AAR(DS_AAR* pAAR,
        void (*PoissonResidual)(DS_AAR*, double*, double*, int, int, MPI_Comm),
        double *x, double *rhs, double omega, double beta, int m, int p, 
        int max_iter, double tol, int FDn, int np_x, int np_y, int np_z, MPI_Comm comm_dist_graph_cart) {
    int rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

    int i, j, k, col; 
    double *res, *x_old, *f_old, **DX, **DF, *am_vec, t_poiss0, t_poiss1;  

    // allocate memory to store x(x) in domain
    // x_new = (double***)calloc((np_z+2*FDn), sizeof(double**)); 
    // x_old = (double***)calloc((np_z+2*FDn), sizeof(double**)); 
    // x_res = (double***)calloc((np_z+2*FDn), sizeof(double**)); 
    // x_temp = (double***)calloc((np_z+2*FDn), sizeof(double**));   
    // assert(x_new !=  NULL && x_old !=  NULL && x_res !=  NULL && x_temp !=  NULL); 

    // for (k = 0; k<np_z+2*FDn; k++) {
    //     x_new[k] = (double**)calloc((np_y+2*FDn), sizeof(double*)); 
    //     x_old[k] = (double**)calloc((np_y+2*FDn), sizeof(double*)); 
    //     x_res[k] = (double**)calloc((np_y+2*FDn), sizeof(double*)); 
    //     x_temp[k] = (double**)calloc((np_y+2*FDn), sizeof(double*)); 
    //     assert(x_new[k] !=  NULL && x_old[k] !=  NULL && x_res[k] !=  NULL && x_temp[k] !=  NULL); 

    //     for (j = 0; j<np_y+2*FDn; j++) {
    //         x_new[k][j] = (double*)calloc((np_x+2*FDn), sizeof(double)); 
    //         x_old[k][j] = (double*)calloc((np_x+2*FDn), sizeof(double)); 
    //         x_res[k][j] = (double*)calloc((np_x+2*FDn), sizeof(double)); 
    //         x_temp[k][j] = (double*)calloc((np_x+2*FDn), sizeof(double)); 
    //         assert(x_new[k][j] !=  NULL && x_old[k][j] !=  NULL && x_res[k][j] !=  NULL && x_temp[k][j] !=  NULL); 
    //     }
    // }

    // allocate memory to store x(x) in domain
    int Np = np_z * np_y * np_x;
    res = (double*)calloc(Np, sizeof(double));  
    // x_old = (double*)calloc(Np, sizeof(double)); 
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

    double *FtF, *allredvec, *Ftf, *svec;                           
    FtF = (double*) calloc(m*m, sizeof(double));                       // DF'*DF matrix in column major format
    allredvec = (double*) calloc(m*m+m, sizeof(double));               
    Ftf = (double*) calloc(m, sizeof(double));                             // DF'*res vector of size m x 1
    svec = (double*) calloc(m, sizeof(double));                            // vector to store singular values    
    int iter = 1; 
    double relres = tol+1; 

    // Initialize x_old from initial_guess
    // AXPYMvPs_3d(x_old, x, NULL, NULL, 0, 0, np_x, np_y, np_z, FDn); 
    // AXPYMvPs_3d(x_temp, x_old, NULL, NULL, 0, 0, np_x, np_y, np_z, FDn); 

    // calculate norm of right hand side (required for relative residual)
    double rhs_norm = 1.0;
    Vector2Norm(rhs, Np, &rhs_norm); 
    tol *= rhs_norm;

    PoissonResidual(pAAR, x, res, np_x, FDn, comm_dist_graph_cart); 

    Vector2Norm(res, Np, &relres);
    if(!rank) printf("reslres: %g\n", relres);

    Vector2Norm(x, Np, &relres);
    if(!rank && iter < 15) printf("phi: %g\n", relres);

    t_poiss0 = MPI_Wtime(); 
    
    // begin while loop
    while (relres > tol && iter <=  max_iter) {
        // PoissonResidual(pAAR, x, res, np_x, FDn, comm_dist_graph_cart); 

        // -------------- Update x --------------- //
        // convert_to_vector(x_res, res, np_x, np_y, np_z, FDn);
        // convert_to_vector(x_old, x_old, np_x, np_y, np_z, FDn);

        //----------Store Residual & Iterate History----------//
        if(iter>1) {
            col = ((iter-2) % m); 
            for (i = 0; i < Np; i++){
                DX[col][i] = x[i] - x_old[i];
                DF[col][i] = res[i] - f_old[i];
            }
            // AXPY(DX[col], x_old, Xold, -1.0, np_x, np_y, np_z); // DX = delta x
            // AXPY(DF[col], res, Fold, -1.0, np_x, np_y, np_z); // DF = delta res
        } 

        // AXPY(Xold, x_old, NULL, 0, np_x, np_y, np_z);   // Xold = x_old;
        // AXPY(Fold, res, NULL, 0, np_x, np_y, np_z);   // Fold = x_res;

        for (i = 0; i < Np; i++){
            x_old[i] = x[i];
            f_old[i] = res[i];
        }

        //----------Anderson update-----------//
        if(iter % p == 0 && iter>1) {
            // vec_dot_product(&relres, res, res, np_x, np_y, np_z);

            // allredvec[m * m + m] = relres; 

            AndersonExtrapolation(DX, DF, res, beta, m, Np, am_vec, FtF, allredvec, Ftf, svec); 

            for (i = 0; i < Np; i ++){
                x[i] = x_old[i] + beta * res[i] - am_vec[i];
            }
            // relres = sqrt(allredvec[m * m + m])/rhs_norm;                  // relative residual    

            PoissonResidual(pAAR, x, res, np_x, FDn, comm_dist_graph_cart); 

            Vector2Norm(res, Np, &relres);
            if(!rank && iter < 15) printf("reslres: %g\n", relres);

            // AXPYMvPs_3d(x_new, x_old, x_res, am_vec, beta, 0, np_x, np_y, np_z, FDn);

        } else {
            //----------Richardson update-----------//
            // AXPYMvPs_3d(x_new, x_old, x_res, NULL, omega, 0, np_x, np_y, np_z, FDn);    // x_k+1 = x_k + omega*f_k

            for (i = 0; i < Np; i++){
                x[i] = x_old[i] + omega * res[i];
            }

            PoissonResidual(pAAR, x, res, np_x, FDn, comm_dist_graph_cart); 
            Vector2Norm(res, Np, &relres);
            if(!rank && iter < 15) printf("reslres: %g\n", relres);

            Vector2Norm(x, Np, &relres);
            if(!rank && iter < 15) printf("phi: %g\n", relres);
        }

        // if(res<= tol || iter  ==  max_iter) {
        //     AXPYMvPs_3d(x, x_old, NULL, NULL, 0, 0, np_x, np_y, np_z, FDn);             // store the solution of Poisson's equation
        // }

        // AXPYMvPs_3d(x_old, x_new, NULL, NULL, 0, 0, np_x, np_y, np_z, FDn);                   // set x_old = x_new

        // AXPYMvPs_3d(x_temp, x_old, NULL, NULL, 0, 0, np_x+2*FDn, np_y+2*FDn, np_z+2*FDn, 0);  // update x_temp

        iter = iter+1; 
    } // end while loop

    t_poiss1 = MPI_Wtime(); 
    if(rank  ==  0) {
        if(iter<max_iter && relres<= tol) {
            printf("AAR preconditioned with Jacobi (AAJ).\n");  
            printf("AAR converged!:  Iterations = %d, Relative Residual = %g, Time = %.4f sec\n", iter-1, relres, t_poiss1-t_poiss0); 
        }
        if(iter>= max_iter) {
            printf("WARNING: AAR exceeded maximum iterations.\n"); 
            printf("AAR:  Iterations = %d, Residual = %g, Time = %.4f sec \n", iter-1, relres, t_poiss1-t_poiss0); 
        }
    }

    // shift x since it can differ by a constant
    double x_fix = 0.0; 
    double x000 = x[FDn + FDn*np_y + FDn*np_y*np_z]; 
    MPI_Bcast(&x000, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
    // AXPYMvPs_3d(x, x, NULL, NULL, 0, x_fix - x000, np_x, np_y, np_z, FDn);
    for (i = 0; i < Np; i++){
        x[i] = x[i] + x_fix - x000;
    }

    // de-allocate memory
    // for (k = 0; k<np_z+2*FDn; k++) {
    //     for (j = 0; j<np_y+2*FDn; j++) {
    //       free(x_new[k][j]); 
    //       free(x_old[k][j]); 
    //       free(x_res[k][j]); 
    //       free(x_temp[k][j]); 
    //     }
    //     free(x_new[k]); 
    //     free(x_old[k]); 
    //     free(x_res[k]); 
    //     free(x_temp[k]); 
    // }
    // free(x_new); 
    // free(x_old); 
    // free(x_res); 
    // free(x_temp); 

    free(res); 
    // free(x_old); 
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

// void convert_to_vector(double ***vector_3d, double *vector, int np_x, int np_y, int np_z, int dis){
//     int i, j, k, ctr = 0;
//     for (k = 0; k < np_z; k++) 
//         for (j = 0; j < np_y; j++) 
//             for (i = 0; i < np_x; i++)             
//                 vector[ctr++] = vector_3d[k+dis][j+dis][i+dis]; 
// }

// z = x + alpha * y
void AXPY(double *z, double *x, double *y, double alpha, int np_x, int np_y, int np_z){
    int i, j, k, ctr = 0;
    if (y != NULL && alpha == -1.0) {
        for (k = 0; k < np_z; k++) 
            for (j = 0; j < np_y; j++) 
                for (i = 0; i < np_x; i++) 
                        z[ctr++] = x[ctr] - y[ctr]; 
    }

    if (y != NULL && alpha != -1.0) {
        for (k = 0; k < np_z; k++) 
            for (j = 0; j < np_y; j++) 
                for (i = 0; i < np_x; i++) 
                        z[ctr++] = x[ctr] + alpha * y[ctr]; 
    }

    if (y == NULL) {
        for (k = 0; k < np_z; k++) 
            for (j = 0; j < np_y; j++) 
                for (i = 0; i < np_x; i++) 
                        z[ctr++] = x[ctr];
    }
}

void vec_dot_product(double *s, double *x, double *y, int np_x, int np_y, int np_z){
    int i, j, k, ctr = 0;
    *s = 0;
    for (k = 0; k < np_z; k++) 
        for (j = 0; j < np_y; j++) 
            for (i = 0; i < np_x; i++) 
                *s += x[ctr] * y[ctr++]; 
}

// z = x + alpha * y - v + s 
void AXPYMvPs_3d(double ***z, double ***x, double ***y, double *v, double alpha, double s, int np_x, int np_y, int np_z, int dis){
    int i, j, k, ctr = 0;
    if (v == NULL && y != NULL && s ==0){
        for (k = 0; k < np_z; k++) 
            for (j = 0; j < np_y; j++) 
                for (i = 0; i < np_x; i++) 
                    z[k + dis][j+dis][i+dis] = x[k+dis][j+dis][i+dis] + alpha * y[k+dis][j+dis][i+dis];
    }

    if (v == NULL && y == NULL && s ==0){
        for (k = 0; k < np_z; k++) 
            for (j = 0; j < np_y; j++) 
                for (i = 0; i < np_x; i++) 
                    z[k + dis][j+dis][i+dis] = x[k+dis][j+dis][i+dis];
    }

    if (v != NULL && y != NULL && s ==0){
        for (k = 0; k < np_z; k++) 
            for (j = 0; j < np_y; j++) 
                for (i = 0; i < np_x; i++) 
                    z[k + dis][j+dis][i+dis] = x[k+dis][j+dis][i+dis] + alpha * y[k+dis][j+dis][i+dis] - v[ctr++];
    }

    if (v == NULL && y == NULL && s !=0) {
        for (k = 0; k < np_z; k++) 
            for (j = 0; j < np_y; j++) 
                for (i = 0; i < np_x; i++) 
                    z[k + dis][j+dis][i+dis] = x[k+dis][j+dis][i+dis] + s;
    }
}


