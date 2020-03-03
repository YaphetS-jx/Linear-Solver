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

void PGR(Mat A, Vec x, Vec b, PetscScalar omega, 
    PetscInt m, PetscInt p, PetscScalar tol, int max_iter, PetscInt pc, DM da) 
{
    int iter = 1, k; 
    double b_2norm, r_2norm, t0, t1;
    t0 = MPI_Wtime();

    Vec x_old, res, f_old, *DX, *DF;
    PC prec;
    Mat Dblock;
    PetscInt blockinfo[6];
    PetscScalar *local;
    /////////////////////////////////////////////////
 
    DMDAGetCorners(da, blockinfo, blockinfo+1, blockinfo+2, blockinfo+3, blockinfo+4, blockinfo+5);
    local = (PetscScalar *) calloc (blockinfo[3] * blockinfo[4] * blockinfo[5], sizeof(PetscScalar));
    assert(local != NULL);
    MatGetDiagonalBlock(A,&Dblock);             // diagonal block of matrix A

    // Set up precondition context
    PCCreate(PETSC_COMM_SELF,&prec);
    if (pc==1) {
        PCSetType(prec,PCICC); 
        //ICC(0), Block-Jacobi
        PetscPrintf(PETSC_COMM_WORLD, "PGR preconditioned with Block-Jacobi using ICC(0).\n");
    }
    else if (pc==0) {
        PCSetType(prec, PCJACOBI); 
        // AAJ
        PetscPrintf(PETSC_COMM_WORLD, "PGR preconditioned with Jacobi (AAJ).\n");
    }
    PCSetOperators(prec, Dblock, Dblock);

    // Initialize vectors
    VecDuplicate(x, &x_old); 
    VecDuplicate(x, &res); 
    VecDuplicate(x, &f_old); 
    VecDuplicateVecs(x, m, &DX);                // storage for delta x
    VecDuplicateVecs(x, m, &DF);                // storage for delta residual 
 
    VecNorm(b, NORM_2, &b_2norm); 
    tol *= b_2norm;

    // res = b- A * x
    MatMult(A, x, res); 
    VecAYPX(res, -1.0, b);
    VecNorm(res, NORM_2, &r_2norm); 

#ifdef DEBUG
    PetscPrintf(PETSC_COMM_WORLD,"relres: %g\n", r_2norm/b_2norm);
    double t2, t3, ta=0;
#endif
    
    while (r_2norm > tol && iter <= max_iter){
        // Apply precondition here 
        precondition(prec, res, da, blockinfo, local);

        VecCopy(x, x_old);
        VecCopy(res, f_old); 
        
        if (iter == 1)
            k = m - 1;
        else
            k = (iter-2) % m;

        VecAXPY(x, omega, res);               // x = x + omega * res
        VecWAXPY(DX[k], -1.0, x_old, x);      // DX[k] = x - x_old        
        MatMult(A, x, res); 

        VecAYPX(res, -1.0, b);
        VecWAXPY(DF[k], -1.0, res, f_old);    // DF[k] = f_old - res

#ifdef DEBUG            
    t2 = MPI_Wtime();
#endif
        if (iter % p == 0){

            /***********************
             *  Galerkin-Richardson  *
             ***********************/
            Galerkin_Richardson(DX, DF, x, x_old, res, f_old, m, k);
        }
        
#ifdef DEBUG
    t3 = MPI_Wtime();
    ta += (t3 - t2);
#endif

        VecNorm(res, NORM_2, &r_2norm); 

#ifdef DEBUG        
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
    PetscPrintf(PETSC_COMM_WORLD,"Time taken by PGR = %.4f seconds.\n",t1-t0);

#ifdef DEBUG        
    PetscPrintf(PETSC_COMM_WORLD,"Time taken by Galerkin-Richardson update = %.4f seconds.\n",ta);
#endif 

    // deallocate memory
    VecDestroy(&res);
    VecDestroy(&f_old);
    VecDestroyVecs(m, &DX);
    VecDestroyVecs(m, &DF);
    free(local);
}

/**
 * @brief   Galerkin_Richardson update
 *
 *          x_new = x_prev + DX * (pinv(DX'*DF)*(DX'*res));
 */

void Galerkin_Richardson(Vec *DX, Vec *DF, Vec x, Vec x_old, Vec res, Vec f_old, PetscInt m, int k)
{
    int lprank, i, j;
    double *svec = malloc(m * sizeof(double));
    PetscScalar * DXtDF = malloc(m*m * sizeof(PetscScalar));    // DFtDF = DF' * DF
    PetscScalar * DXres = malloc(m * sizeof(PetscScalar));      // DFres = DF' * res
    assert(svec != NULL && DXtDF != NULL && DXres != NULL);
    /////////////////////////////////////////////////

    for (i=0; i<m; i++)
        for(j=0; j<m; j++)
            VecTDot(DX[i], DF[j], &DXtDF[m*i+j]);               // DFtDF(i,j) = DF[i]' * DF[j]
        
    for (i=0; i<m; i++)
        VecTDot(DX[i], res, &DXres[i]);                         // DFres(i)   = DF[i]' * res

    // Least square problem solver. DXres = pinv(DX'*DF)*(DX'*res)
    LAPACKE_dgelsd(LAPACK_COL_MAJOR, m, m, 1, DXtDF, m, DXres, m, svec, -1.0, &lprank);

    for (i=0; i<m; i++)                                         // x = x + DX' * DXres
        VecAXPY(x, DXres[i], DX[i]);                            

    VecWAXPY(DX[k], -1.0, x_old, x);      // update DX[k] = x - x_old 

    for (i=0; i<m; i++)                                         // x = x + DX' * DXres
        VecAXPY(res, -DXres[i], DF[i]);
    VecWAXPY(DF[k], -1.0, res, f_old);    // DF[k] = f_old - res 

    // deallocate memory
    free(svec);
    free(DXtDF);
    free(DXres);
}

