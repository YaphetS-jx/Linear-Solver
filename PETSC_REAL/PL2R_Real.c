/**
 * @file    PL2R_Real.c
 * @brief   This file contains Periodic L2_Richardson solver and its 
 *          required functions
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
    double b_2norm, r_2norm, *svec;
    double t0, t1, t2, t3, ta = 0, tax = 0, tp = 0, tn = 0, tpre = 0;
    t0 = MPI_Wtime();

    Vec x_old, res, res_local, pres_local, *DX, *DF, Ax, Ax_prev;
    PC prec;
    Mat Dblock;
    PetscInt blockinfo[6], Np;
    PetscScalar *local, *DFres, ***r, *DFtDF;
    /////////////////////////////////////////////////
 
    DMDAGetCorners(da, blockinfo, blockinfo+1, blockinfo+2, blockinfo+3, blockinfo+4, blockinfo+5);
    Np = blockinfo[3] * blockinfo[4] * blockinfo[5];
    local = (PetscScalar *) calloc (Np, sizeof(PetscScalar));
    DFres = (PetscScalar *) calloc (m , sizeof(PetscScalar));      // DFres = DF' * res
    svec = malloc(m * sizeof(double));
    DFtDF = malloc(m*m * sizeof(PetscScalar));  // DFtDF = DF' * DF
    assert(local != NULL && DFres != NULL && svec != NULL && DFtDF != NULL);
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
        PetscPrintf(PETSC_COMM_WORLD, "PL2R preconditioned with Jacobi.\n");
    }
    PCSetOperators(prec, Dblock, Dblock);

    // Initialize vectors
    VecDuplicate(x, &x_old); 
    VecDuplicate(x, &res); 
    VecDuplicate(x, &Ax); 
    VecDuplicate(x, &Ax_prev);  
    VecDuplicateVecs(x, m, &DX);                 // storage for delta x
    VecDuplicateVecs(x, m, &DF);                 // storage for delta residual 
    VecCreateSeq(PETSC_COMM_SELF, Np, &res_local);
    VecDuplicate(res_local, &pres_local);

    t2 = MPI_Wtime();
    tpre = t2 - t0;

    VecNorm(b, NORM_2, &b_2norm); 
    tol *= b_2norm;
    t3 = MPI_Wtime();
    tn += (t3-t2);   

    t2 = MPI_Wtime();
    // res = b- A * x
    MatMult(A, x, Ax); 
    t3 = MPI_Wtime();
    tax += (t3-t2);

    VecWAXPY(res, -1.0, Ax, b);
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

        VecAXPY(x, omega, res);                    // x = x + omega * res
        VecWAXPY(DX[k], -1.0, x_old, x);           // DX[k] = x - x_old        
           
        t2 = MPI_Wtime();
        MatMult(A, x, Ax); 
        t3 = MPI_Wtime();
        tax += (t3 - t2);

        VecWAXPY(res, -1.0, Ax, b);
        VecWAXPY(DF[k], -1.0, Ax_prev, Ax);        // DF[k] = Ax - Ax_prev
   
        t2 = MPI_Wtime();
        if (iter % p == 0){
            /***********************
             *    L2_Richardson    *
             ***********************/
            L2_Richardson(DFres, DF, res, m, svec, DFtDF);

            for (i=0; i<m; i++)                    // x = x + DX' * DXres
                VecAXPY(x, DFres[i], DX[i]);                            

            VecWAXPY(DX[k], -1.0, x_old, x);       // update DX[k] = x - x_old 

            for (i=0; i<m; i++)                    // x = x + DX' * DXres
                VecAXPY(Ax, DFres[i], DF[i]);
            VecWAXPY(DF[k], -1.0, Ax_prev, Ax);    // update DF[k] = Ax - Ax_prev
        }
        
        t3 = MPI_Wtime();
        ta += (t3 - t2);

        VecWAXPY(res, -1.0, Ax, b);

        t2 = MPI_Wtime();
        if (iter % p == 0) 
            VecNorm(res, NORM_2, &r_2norm);             
        
        t3 = MPI_Wtime();
        tn += (t3-t2);   

#ifdef DEBUG  
    if (iter % p) 
        VecNorm(res, NORM_2, &r_2norm);        
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
    PetscPrintf(PETSC_COMM_WORLD,"Time taken by PL2R = %.6f seconds.\n",t1-t0);

#ifdef DEBUG        
    PetscPrintf(PETSC_COMM_WORLD, "Time taken by L2-Richardson update = %.6f seconds.\n",ta);
    PetscPrintf(PETSC_COMM_WORLD, "Time taken by Matrix Vector Multiply = %.6f seconds.\n",tax);
    PetscPrintf(PETSC_COMM_WORLD, "Time taken by precondition = %.6f seconds.\n",tp);
    PetscPrintf(PETSC_COMM_WORLD, "Time taken by norm = %.6f seconds.\n", tn);
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
    free(DFres);
    free(svec);
    free(DFtDF);
    PCDestroy(&prec);
}

/**
 * @brief   L2_Richardson update
 *
 *          x_new = x_prev + DX * (pinv(DF'*DF)*(DF'*res));
 */

void L2_Richardson(PetscScalar * DFres, Vec *DF, Vec res, PetscInt m, double *svec, PetscScalar *DFtDF)
{
    int lprank, i, j;
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
}