/**
 * @file    PL2R_Real.c
 * @brief   This file contains PL2R_Real solver and its required functions
 *
 * @author  Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */

#include "PL2R_Real.h"
#include "tools.h"

void PL2R(Mat A, Vec x, Vec b, PetscScalar omega, PetscScalar beta, 
    PetscInt m, PetscInt p, PetscScalar tol, int max_iter, PetscInt pc, DM da) 
{
    int iter = 1, k, i; 
    double b_2norm, r_2norm, t0, t1;
    t0 = MPI_Wtime();

    Vec x_old, res, res_local, pres_local, f_old, *DX, *DF, Ax, Ax_prev;
    PC prec;
    Mat Dblock;
    PetscInt blockinfo[6], Np;
    PetscScalar *local, *DFres, ***r;
    /////////////////////////////////////////////////
 
    DMDAGetCorners(da, blockinfo, blockinfo+1, blockinfo+2, blockinfo+3, blockinfo+4, blockinfo+5);
    Np = blockinfo[3] * blockinfo[4] * blockinfo[5];
    local = (PetscScalar *) calloc (Np, sizeof(PetscScalar));
    DFres = (PetscScalar *) calloc (m , sizeof(PetscScalar));      // DFres = DF' * res
    assert(local != NULL && DFres != NULL);
    MatGetDiagonalBlock(A,&Dblock);             // diagonal block of matrix A

    // Set up precondition context
    PCCreate(PETSC_COMM_SELF,&prec);
    if (pc==1) {
        PCSetType(prec,PCICC); 
        //ICC(0), Block-Jacobi
        PetscPrintf(PETSC_COMM_WORLD, "PL2R preconditioned with Block-Jacobi using ICC(0).\n");
    }
    else if (pc==0) {
        PCSetType(prec, PCJACOBI); 
        // AAJ
        PetscPrintf(PETSC_COMM_WORLD, "PL2R preconditioned with Jacobi (AAJ).\n");
    }
    PCSetOperators(prec, Dblock, Dblock);

    // Initialize vectors
    VecDuplicate(x, &x_old); 
    VecDuplicate(x, &res); 
    VecDuplicate(x, &f_old);
    VecDuplicate(x, &Ax); 
    VecDuplicate(x, &Ax_prev);  
    VecDuplicateVecs(x, m, &DX);                 // storage for delta x
    VecDuplicateVecs(x, m, &DF);                 // storage for delta residual 
 
    VecNorm(b, NORM_2, &b_2norm); 
    tol *= b_2norm;

    // res = b- A * x
    MatMult(A, x, Ax); 
    VecWAXPY(res, -1.0, Ax, b);
    VecNorm(res, NORM_2, &r_2norm); 

#ifdef DEBUG
    PetscPrintf(PETSC_COMM_WORLD,"relres: %g\n", r_2norm/b_2norm);
    double t2, t3, ta=0;
#endif
    
    while (r_2norm > tol && iter <= max_iter){
        // Apply precondition here 
        GetLocalVector(da, res, &res_local, blockinfo, Np, local, &r);
        VecDuplicate(res_local, &pres_local);
        PCApply(prec, res_local, pres_local);
        RestoreGlobalVector(da, res, pres_local, blockinfo, local, &r);

        VecCopy(x, x_old);
        VecCopy(res, f_old); 
        VecCopy(Ax, Ax_prev); 
        
        if (iter == 1)
            k = m - 1;
        else
            k = (iter-2) % m;

        VecAXPY(x, omega, res);                    // x = x + omega * res
        VecWAXPY(DX[k], -1.0, x_old, x);           // DX[k] = x - x_old        
        
        MatMult(A, x, Ax); 
        VecWAXPY(res, -1.0, Ax, b);
        VecWAXPY(DF[k], -1.0, Ax_prev, Ax);        // DF[k] = Ax - Ax_prev

#ifdef DEBUG            
    t2 = MPI_Wtime();
#endif
        if (iter % p == 0){
            /***********************
             *    L2_Richardson    *
             ***********************/
            L2_Richardson(DFres, DF, res, m);

            for (i=0; i<m; i++)                    // x = x + DX' * DXres
                VecAXPY(x, DFres[i], DX[i]);                            

            VecWAXPY(DX[k], -1.0, x_old, x);       // update DX[k] = x - x_old 

            for (i=0; i<m; i++)                    // x = x + DX' * DXres
                VecAXPY(Ax, DFres[i], DF[i]);
            VecWAXPY(DF[k], -1.0, Ax_prev, Ax);    // update DF[k] = Ax - Ax_prev
        }
        
#ifdef DEBUG
    t3 = MPI_Wtime();
    ta += (t3 - t2);
#endif

        VecWAXPY(res, -1.0, Ax, b);
        VecNorm(res, NORM_2, &r_2norm); 

#ifdef DEBUG        
        PetscPrintf(PETSC_COMM_WORLD,"relres: %g\n", r_2norm/b_2norm);
#endif        
        iter ++;
    }

    if (iter<max_iter) {
        PetscPrintf(PETSC_COMM_WORLD,"PL2R converged to a relative residual of %g in %d iterations.\n", r_2norm/b_2norm, iter-1);
    } else {
        PetscPrintf(PETSC_COMM_WORLD,"PL2R exceeded maximum iterations %d and converged to a relative residual of %g. \n", iter-1, r_2norm/b_2norm);
    }

    t1=MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"Time taken by PL2R = %.4f seconds.\n",t1-t0);

#ifdef DEBUG        
    PetscPrintf(PETSC_COMM_WORLD,"Time taken by Galerkin-Richardson update = %.4f seconds.\n",ta);
#endif 

    // deallocate memory
    VecDestroy(&res);
    VecDestroy(&f_old);
    VecDestroy(&res_local);
    VecDestroy(&pres_local);
    VecDestroyVecs(m, &DX);
    VecDestroyVecs(m, &DF);
    free(local);
    free(DFres);
}

/**
 * @brief   L2_Richardson update
 *
 *          x_new = x_prev + DX * (pinv(DF'*DF)*(DF'*res));
 */

void L2_Richardson(PetscScalar * DFres, Vec *DF, Vec res, PetscInt m)
{
    int lprank, i, j;
    double *svec = malloc(m * sizeof(double));
    PetscScalar * DFtDF = malloc(m*m * sizeof(PetscScalar));    // DFtDF = DF' * DF
    assert(svec != NULL && DFtDF != NULL);
    /////////////////////////////////////////////////

    for (i=0; i<m; i++)
        for(j=0; j<i+1; j++)
            VecTDotBegin(DF[i], DF[j], &DFtDF[m*i+j]);               // DFtDF(i,j) = DF[i]' * DF[j]
        
    for (i=0; i<m; i++)
        VecTDotBegin(DF[i], res, &DFres[i]);                         // DFres(i)   = DF[i]' * res

    for (i=0; i<m; i++)
        for(j=0; j<i+1; j++){
            VecTDotEnd(DF[i], DF[j], &DFtDF[m*i+j]);               // DFtDF(i,j) = DF[i]' * DF[j]
            DFtDF[i+m*j] = DFtDF[m*i+j];                        // DFtDF(j,i) = DFtDF(i,j) symmetric 
        }
        
    for (i=0; i<m; i++)
        VecTDotEnd(DF[i], res, &DFres[i]);                         // DFres(i)

    // Least square problem solver. DFres = pinv(DF'*DF)*(DF'*res)
    LAPACKE_dgelsd(LAPACK_COL_MAJOR, m, m, 1, DFtDF, m, DFres, m, svec, -1.0, &lprank);

    free(svec);
    free(DFtDF);
}