/**
 * @file    CG.c
 * @brief   This file contains Conjugate Gradient solvers
 *
 * @author  Xin Jing  < xjing30@gatech.edu>
 *          Phanish Suryanarayana  < phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */


#include "CG.h"

/**
 * @brief   Conjugate Gradient solver
 *
 *          This function is designed to solve linear system Ax = b
 *          PoissonResidual is for matrix vector multiplication A * x, which could be 
 *          replaced by any user defined function 
 *          Precondition is for applying precondition, which could be replaced by any
 *          user defined precondition function
 *          DMnd     : no. of elements in this domain
 *          x        : initial guess and final solution
 *          b        : right hand side of linear system
 *          tol      : convergence tolerance
 *          max_iter : maximum number of iterations
 *          comm     : MPI communicator
 */


void CG(POISSON *system,
        void (*Lap_Vec_mult)(POISSON *, double, double *, double *, MPI_Comm),
        void (*Precondition)(double, double *, int),
        int DMnd, double *x, double *b, double tol, int max_iter, MPI_Comm comm)
{
    if (comm == MPI_COMM_NULL) return;

    int iter_count = 0, rank, size, i;
    double *r, *d, *s, *q, delta_new, delta_old, alpha, beta, relres, b_2norm;
    double t1, t2, tt1, tt2, tt, t, ttt0, ttt1;
    /////////////////////////////////////////////////

    tt = 0;
    t=0;
    r = (double *)malloc( DMnd * sizeof(double) );
    d = (double *)malloc( DMnd * sizeof(double) );
    s = (double *)malloc( DMnd * sizeof(double) );
    q = (double *)malloc( DMnd * sizeof(double) );    
    assert(r != NULL && d != NULL && q != NULL);
    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    t1 = MPI_Wtime();
    Vector2Norm(b, DMnd, &b_2norm, comm); 
    t2 = MPI_Wtime();
    t += t2 - t1;
#ifdef DEBUG
    if (rank == 0) printf("2-norm of RHS = %.13f, which took %.3f ms\n", b_2norm, t*1e3);
#endif

    tt1 = MPI_Wtime();
    Lap_Vec_mult(system, -1.0/(4*M_PI), x, r, comm);
    tt2 = MPI_Wtime();
    tt += (tt2 - tt1);

    for (i = 0; i < DMnd; ++i){
        r[i] = b[i] - r[i];                                                 // r = b - Ax 
        d[i] = r[i];                                                        // d = r
    }

    if (Precondition != NULL)
        Precondition(-(3*system->coeff_lap[0]/4/M_PI), d, DMnd);              // Jacobi precondition

    t1 = MPI_Wtime();

    VectorDotProduct(r, d, DMnd, &delta_new, comm);                         // delta_new = r' * d

    if (Precondition != NULL)
        Vector2Norm(r, DMnd, &relres, comm);
    else
        relres = sqrt(delta_new);                                           // relres = norm(r)

    t2 = MPI_Wtime();
    t += t2 - t1;

    tol *= b_2norm;
    ttt0 = MPI_Wtime();

    while(iter_count < max_iter && relres > tol){
        tt1 = MPI_Wtime();
        Lap_Vec_mult(system, -1.0/(4*M_PI), d, q, comm);                    // q = A * d

        tt2 = MPI_Wtime();
        tt += (tt2 - tt1);

        t1 = MPI_Wtime();
        VectorDotProduct(d, q, DMnd, &alpha, comm);                         // alpha = d' * q
        t2 = MPI_Wtime();
        t += t2 - t1;

        alpha = delta_new / alpha;                                          // alpha = (r' * d)/(d' * q)

        for (i = 0; i < DMnd; ++i)
            x[i] = x[i] + alpha * d[i];                                     // x = x + alpha * d
        
        if ((iter_count % 50) == 0)                                         // Restart every 50 cycles
        {
            tt1 = MPI_Wtime();
            Lap_Vec_mult(system, -1.0/(4*M_PI), x, r, comm);
            tt2 = MPI_Wtime();
            tt += (tt2 - tt1);

            for (i = 0; i < DMnd; ++i){
                r[i] = b[i] - r[i];                                         // r = b - A * x
                s[i] = r[i];                                                // s = r 
            }
            
        } else {
            for (i = 0; i < DMnd; ++i){
                r[i] = r[i] - alpha * q[i];                                 // r = r - alpha * q
                s[i] = r[i];                                                // s = r
            }
        }

        if (Precondition != NULL)
            Precondition(-(3*system->coeff_lap[0]/4/M_PI), s, DMnd);          // Jacobi precondition

        delta_old = delta_new;

        t1 = MPI_Wtime();
        VectorDotProduct(r, s, DMnd, &delta_new, comm);                     // delta_new = r' * s

        if (Precondition != NULL)
            Vector2Norm(r, DMnd, &relres, comm);
        else
            relres = sqrt(delta_new);                                       // relres = norm(r)

#ifdef DEBUG
    if (rank == 0) printf("norm of res: %g\n", relres);
#endif

        t2 = MPI_Wtime();
        t += t2 - t1;

        beta = delta_new / delta_old;
        for (i = 0; i < DMnd; ++i)
            d[i] = s[i] + beta * d[i];                                      // d = s + beta * d

        iter_count++;
    }

    ttt1 = MPI_Wtime();

    if(rank  ==  0) {
        if(iter_count < max_iter && relres <= tol) {
            printf("CG converged!:  Iterations = %d, Relative Residual = %g, Time = %.4f sec\n", iter_count, relres/b_2norm, ttt1-ttt0); 
        }
        if(iter_count >= max_iter) {
            printf("WARNING: CG exceeded maximum iterations.\n"); 
            printf("CG:  Iterations = %d, Relative Residual = %g, Time = %.4f sec \n", iter_count, relres/b_2norm, ttt1-ttt0); 
        }
    }

#ifdef DEBUG
    if (rank == 0) printf("Dot product and 2norm took %.3f ms, Ax took %.3f ms\n", t * 1e3, tt * 1e3);
#endif

    free(r);
    free(d);
    free(s);
    free(q);
}