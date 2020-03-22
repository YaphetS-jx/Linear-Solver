/**
 * @brief   Conjugate Gradient (CG) method for solving a general linear system Ax = b. 
 *
 *          CG() assumes that x and  b is distributed among the given communicator. 
 *          Ax is calculated by calling function Ax().
 */


#include "CG.h"


void CG(SPARC_OBJ *pSPARC, 
    void (*PoissonResidual)(DS*, double*, double*, int, int, MPI_Comm),
    void (*Precondition)(double, double *, int),
    int N, int DMnd, double* x, double *b, double tol, int max_iter, MPI_Comm comm
)
{
    if (comm == MPI_COMM_NULL) return;

    int iter_count = 0, sqrt_n = (int)sqrt(N), rank, size;
    double *r, *d, *q, delta_new, delta_old, alpha, beta, err, b_2norm;

    /////////////////////////////////////////////////
    double t1, t2, tt1, tt2, tt, t;
    /////////////////////////////////////////////////

    tt = 0;
    t=0;
    r = (double *)malloc( DMnd * sizeof(double) );
    d = (double *)malloc( DMnd * sizeof(double) );
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
    Ax(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, x, r, comm);
    tt2 = MPI_Wtime();
    tt += (tt2 - tt1);

    for (int i = 0; i < DMnd; ++i){
        r[i] = b[i] + r[i];
        d[i] = r[i];
    }

    t1 = MPI_Wtime();
    Vector2Norm(r, DMnd, &delta_new, comm);
    t2 = MPI_Wtime();
    t += t2 - t1;

    err = delta_new / b_2norm;

    while(iter_count < max_iter && err > tol){
        tt1 = MPI_Wtime();
        Ax(pSPARC, DMnd, pSPARC->DMVertices, 1, 0.0, d, q, comm);
        tt2 = MPI_Wtime();
        tt += (tt2 - tt1);

        t1 = MPI_Wtime();
        VectorDotProduct(d, q, DMnd, &alpha, comm);
        t2 = MPI_Wtime();
        t += t2 - t1;

        alpha = - delta_new * delta_new / alpha;

        for (int j = 0; j < DMnd; ++j)
            x[j] = x[j] + alpha * d[j];
        
        // Restart every sqrt_n cycles.
        if ((iter_count % sqrt_n)==0)
        {
            tt1 = MPI_Wtime();
            Ax(pSPARC, N, pSPARC->DMVertices, 1, 0.0, x, r, comm);
            tt2 = MPI_Wtime();
            tt += (tt2 - tt1);

            for (int j = 0; j < DMnd; ++j){
                r[j] = b[j] + r[j];
            }
        }
        else
        {
            for (int j = 0; j < DMnd; ++j)
                r[j] = r[j] + alpha * q[j];
        }

        delta_old = delta_new;

        t1 = MPI_Wtime();
        Vector2Norm(r, DMnd, &delta_new, comm);
        t2 = MPI_Wtime();
        t += t2 - t1;

        err = delta_new / b_2norm;
        beta = delta_new * delta_new / (delta_old * delta_old);
        for (int j = 0; j < DMnd; ++j)
            d[j] = r[j] + beta * d[j];

        iter_count++;
    }

#ifdef DEBUG
    if (rank == 0) printf("\niter_count = %d, r_2norm = %.3e, tol = %.3e\n\n", iter_count, delta_new, tol);    
    if (rank == 0) printf("Dot product and 2norm took %.3f ms, Ax took %.3f ms\n", t * 1e3, tt * 1e3);
#endif

    free(r);
    free(d);
    free(q);
}