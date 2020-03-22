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


void CG(DS* pAAR,
    void (*PoissonResidual)(DS*, double*, double*, int, int, MPI_Comm),
    void (*Precondition)(double, double *, int),
    int DMnd, double *x, double *b, double tol, int max_iter, MPI_Comm comm
)
{
    if (comm == MPI_COMM_NULL) return;

    int iter_count = 0, rank, size, i;
    double *r, *d, *s, *q, delta_new, delta_old, alpha, beta, err, b_2norm;

    /////////////////////////////////////////////////
    double t1, t2, tt1, tt2, tt, t;
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

    PoissonResidual(pAAR, x, r, pAAR->np_x, pAAR->FDn, comm);

    tt2 = MPI_Wtime();
    tt += (tt2 - tt1);

    for (i = 0; i < DMnd; ++i){
        r[i] = b[i] - r[i];
        d[i] = r[i];
    }

    if (Precondition != NULL)
        Precondition(-(3*pAAR->coeff_lap[0]/4/M_PI), d, DMnd);

    t1 = MPI_Wtime();

    VectorDotProduct(r, d, DMnd, &delta_new, comm);
// if (rank == 0) printf("2-norm of RHS = %.13f, which took %.3f ms\n", b_2norm, t*1e3);

    if (Precondition != NULL)
        Vector2Norm(r, DMnd, &err, comm);
    else
        err = sqrt(delta_new);

    t2 = MPI_Wtime();
    t += t2 - t1;

    tol *= b_2norm;

    while(iter_count < max_iter && err > tol){
        tt1 = MPI_Wtime();

        PoissonResidual(pAAR, d, q, pAAR->np_x, pAAR->FDn, comm);

        tt2 = MPI_Wtime();
        tt += (tt2 - tt1);

        t1 = MPI_Wtime();
        VectorDotProduct(d, q, DMnd, &alpha, comm);
        t2 = MPI_Wtime();
        t += t2 - t1;

        alpha = delta_new / alpha;

        for (i = 0; i < DMnd; ++i)
            x[i] = x[i] + alpha * d[i];
        
        // Restart every 50 cycles.
        if ((iter_count % 50) == 0)
        {
            tt1 = MPI_Wtime();
            PoissonResidual(pAAR, x, r, pAAR->np_x, pAAR->FDn, comm);
            tt2 = MPI_Wtime();
            tt += (tt2 - tt1);

            for (i = 0; i < DMnd; ++i){
                r[i] = b[i] - r[i];
                s[i] = r[i];
            }
            
        } else {
            for (i = 0; i < DMnd; ++i){
                r[i] = r[i] - alpha * q[i];
                s[i] = r[i];
            }
        }

        if (Precondition != NULL)
            Precondition(-(3*pAAR->coeff_lap[0]/4/M_PI), s, DMnd);

        delta_old = delta_new;

        t1 = MPI_Wtime();
        VectorDotProduct(r, s, DMnd, &delta_new, comm);

        if (Precondition != NULL)
            Vector2Norm(r, DMnd, &err, comm);
        else
            err = sqrt(delta_new);

#ifdef DEBUG
    if (rank == 0) printf("norm of res: %g\n", err);
#endif

        t2 = MPI_Wtime();
        t += t2 - t1;

        beta = delta_new / delta_old;
        for (i = 0; i < DMnd; ++i)
            d[i] = s[i] + beta * d[i];

        iter_count++;
    }

#ifdef DEBUG
    if (rank == 0) printf("\niter_count = %d, r_2norm = %.3e, tol = %.3e\n\n", iter_count, err/b_2norm, tol);    
    if (rank == 0) printf("Dot product and 2norm took %.3f ms, Ax took %.3f ms\n", t * 1e3, tt * 1e3);
#endif

    free(r);
    free(d);
    free(s);
    free(q);
}