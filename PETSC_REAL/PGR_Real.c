/**
 * @file    PGR_Real.c
 * @brief   This file contains PGR_Real solver and its required functions
 *
 * @author  Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */

#include "PGR_Real.h"

/**
 * @brief   Periodic Galerkin Richardson solver
 *
 *          This function is designed to solve linear system Ax = b
 *          A        : Matrix A
 *          x        : initial guess and final solution
 *          b        : right hand side of linear system
 *          omega    : relaxation parameter (for Richardson update)
 *          m        : no. of previous iterations to be considered in extrapolation
 *          p        : Perform Anderson extrapolation at every p th iteration
 *          tol      : convergence tolerance
 *          max_iter : maximum number of iterations
 *          pc       : Preconditioner, pc = 0 (Jacobi) or 1 (Block Jacobi)
 *          da       : PETSC data structure
 */

void PGR(Mat A, Vec x, Vec b, PetscScalar omega, 
    PetscInt m, PetscInt p, PetscScalar tol, int max_iter, PetscInt pc, DM da) 
{
    int iter = 1, k, i; 
    double b_2norm, r_2norm, *svec;
    double t0, t1, t2, t3, ta = 0, tax = 0, tp = 0, tn = 0, tpre = 0;
    t0 = MPI_Wtime();

    Vec x_old, res, res_local, pres_local, *DX, *DF, Ax, Ax_prev;
    PC prec;
    Mat Dblock;
    PetscInt blockinfo[6], Np;
    PetscScalar *local, * DXres, ***r, *DXtDF;
    /////////////////////////////////////////////////
 
    DMDAGetCorners(da, blockinfo, blockinfo+1, blockinfo+2, blockinfo+3, blockinfo+4, blockinfo+5);
    Np = blockinfo[3] * blockinfo[4] * blockinfo[5];
    local = (PetscScalar *) calloc (Np, sizeof(PetscScalar));
    DXres = (PetscScalar *) calloc (m , sizeof(PetscScalar));       // DFres = DF' * res
    svec = malloc(m * sizeof(double));
    DXtDF = malloc(m*m * sizeof(PetscScalar));                      // DFtDF = DF' * DF
    assert(local != NULL && DXres != NULL && svec != NULL && DXtDF != NULL);
    MatGetDiagonalBlock(A,&Dblock);                                 // diagonal block of matrix A

    // Set up precondition context
    PCCreate(PETSC_COMM_SELF,&prec);
    if (pc==1) {
        PCSetType(prec,PCICC);                                      // ICC(0), Block-Jacobi
        PetscPrintf(PETSC_COMM_WORLD, "PGR preconditioned with Block-Jacobi using ICC(0).\n");
    }
    else if (pc==0) {
        PCSetType(prec, PCJACOBI);                                  // Jacobi
        PetscPrintf(PETSC_COMM_WORLD, "PGR preconditioned with Jacobi.\n");
    }
    PCSetOperators(prec, Dblock, Dblock);

    // Initialize vectors
    VecDuplicate(x, &x_old); 
    VecDuplicate(x, &res); 
    VecDuplicate(x, &Ax); 
    VecDuplicate(x, &Ax_prev); 
    VecDuplicateVecs(x, m, &DX);                                    // storage for delta x
    VecDuplicateVecs(x, m, &DF);                                    // storage for delta residual 
    VecCreateSeq(PETSC_COMM_SELF, Np, &res_local);
    VecDuplicate(res_local, &pres_local);

    t2 = MPI_Wtime();
    tpre = t2 - t0;

    VecNorm(b, NORM_2, &b_2norm); 
    tol *= b_2norm;
    t3 = MPI_Wtime();
    tn += (t3-t2);   

    t2 = MPI_Wtime();
    MatMult(A, x, Ax); 
    t3 = MPI_Wtime();
    tax += (t3-t2);

    VecWAXPY(res, -1.0, Ax, b);                                     // res = b- A * x
    r_2norm = tol + 1;

#ifdef DEBUG
    VecNorm(res, NORM_2, &r_2norm); 
    PetscPrintf(PETSC_COMM_WORLD,"relres: %g\n", r_2norm/b_2norm);
#endif
    
    while (r_2norm > tol && iter <= max_iter){
        // Apply precondition here 
        t2 = MPI_Wtime();

        GetLocalVector(da, res, &res_local, blockinfo, Np, local, &r);
        PCApply(prec, res_local, pres_local);
        RestoreGlobalVector(da, res, pres_local, blockinfo, local, &r);

        t3 = MPI_Wtime();
        tp += (t3-t2);

        VecCopy(x, x_old);
        VecCopy(Ax, Ax_prev); 
    
        k = (iter-2+m) % m;

        VecAXPY(x, omega, res);                                     // x = x + omega * res
        VecWAXPY(DX[k], -1.0, x_old, x);                            // DX[k] = x - x_old        

        t2 = MPI_Wtime();
        MatMult(A, x, Ax); 
        t3 = MPI_Wtime();
        tax += (t3 - t2);

        VecWAXPY(res, -1.0, Ax, b);
        VecWAXPY(DF[k], -1.0, Ax_prev, Ax);                         // DF[k] = Ax - Ax_prev
         
        t2 = MPI_Wtime();
        if (iter % p == 0){
            /***********************
             *  Galerkin-Richardson  *
             ***********************/
            Galerkin_Richardson(DXres, DX, DF, res, m, svec, DXtDF);

            for (i=0; i<m; i++)                                     // x = x + DX' * DXres
                VecAXPY(x, DXres[i], DX[i]);                       

            VecWAXPY(DX[k], -1.0, x_old, x);                        // update DX[k] = x - x_old 

            for (i=0; i<m; i++)                                     // Ax = Ax + DF' * DXres
                VecAXPY(Ax, DXres[i], DF[i]);

            VecWAXPY(DF[k], -1.0, Ax_prev, Ax);                     // update DF[k] = Ax - Ax_prev
        }
        
        t3 = MPI_Wtime();
        ta += (t3 - t2);

        VecWAXPY(res, -1.0, Ax, b);
        
        t2 = MPI_Wtime();
        if (iter % p == 0) 
            VecNorm(res, NORM_2, &r_2norm); 

        t3 = MPI_Wtime();
        tn += (t3 - t2); 

#ifdef DEBUG  
    if (iter % p) 
        VecNorm(res, NORM_2, &r_2norm); 
    PetscPrintf(PETSC_COMM_WORLD,"relres: %g\n", r_2norm/b_2norm);
#endif        
        iter ++;
    }

    if (iter<max_iter) {
        PetscPrintf(PETSC_COMM_WORLD,"PGR converged to a relative residual of %g in %d iterations.\n", r_2norm/b_2norm, iter-1);
    } else {
        PetscPrintf(PETSC_COMM_WORLD,"PGR exceeded maximum iterations %d and converged to a relative residual of %g. \n", iter-1, r_2norm/b_2norm);
    }

    t1=MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"Time taken by PGR = %.6f seconds.\n",t1-t0);

#ifdef DEBUG        
    PetscPrintf(PETSC_COMM_WORLD,"Time taken by Galerkin-Richardson update = %.6f seconds.\n", ta);
    PetscPrintf(PETSC_COMM_WORLD,"Time taken by Matrix Vector Multiply = %.6f seconds.\n", tax);
    PetscPrintf(PETSC_COMM_WORLD,"Time taken by precondition = %.6f seconds.\n", tp);
    PetscPrintf(PETSC_COMM_WORLD,"Time taken by norm = %.6f seconds.\n", tn);
    PetscPrintf(PETSC_COMM_WORLD,"Time taken by preparation = %.6f seconds.\n", tpre);
#endif 

    // deallocate memory
    VecDestroy(&x_old);
    VecDestroy(&res);
    VecDestroy(&res_local);
    VecDestroy(&pres_local);
    VecDestroy(&Ax);
    VecDestroy(&Ax_prev);
    VecDestroyVecs(m, &DX);
    VecDestroyVecs(m, &DF);
    free(local);
    free(DXres);
    free(svec);
    free(DXtDF);
    PCDestroy(&prec);
}

/**
 * @brief   Galerkin_Richardson update
 *
 *          x_new = x_prev + DX * (pinv(DX'*DF)*(DX'*res));
 */

void Galerkin_Richardson(PetscScalar * DXres, Vec *DX, Vec *DF, Vec res, PetscInt m, double *svec, PetscScalar *DXtDF)
{
    int lprank, i, j;
    /////////////////////////////////////////////////

    for (i = 0; i < m; i++)
        for(j = 0; j < m; j++)
            VecTDotBegin(DX[i], DF[j], &DXtDF[m*i+j]);               // DFtDF(i,j) = DF[i]' * DF[j]
    
    for (i = 0; i < m; i++)
        VecTDotBegin(DX[i], res, &DXres[i]);                         // DFres(i)   = DF[i]' * res

    for (i = 0; i < m; i++)
        for(j = 0; j < m; j++)
            VecTDotEnd(DX[i], DF[j], &DXtDF[m*i+j]);   
    
    for (i = 0; i < m; i++)
        VecTDotEnd(DX[i], res, &DXres[i]); 

    // Least square problem solver. DXres = pinv(DX'*DF)*(DX'*res)
    LAPACKE_dgelsd(LAPACK_COL_MAJOR, m, m, 1, DXtDF, m, DXres, m, svec, -1.0, &lprank);
}

