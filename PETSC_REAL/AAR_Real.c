/**
 * @file    AAR_Real.c
 * @brief   This file contains Alternating Anderson Richardson solver and
 *          its required functions
 *
 * @author  Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */

#include "AAR_Real.h"
#include "tools.h"

void AAR(Mat A, Vec x, Vec b, PetscScalar omega, PetscScalar beta, 
    PetscInt m, PetscInt p, PetscScalar tol, int max_iter, PetscInt pc, DM da) 
{
    int iter = 1, k, i;
    double b_2norm, r_2norm, t0, t1;
    t0 = MPI_Wtime();

    Vec x_old, res, res_local, pres_local, f_old, *DX, *DF, DXDF;
    PC prec;
    Mat Dblock;
    PetscInt blockinfo[6], Np;
    PetscScalar *local, *DFres, ***r;
    /////////////////////////////////////////////////
 
    DMDAGetCorners(da, blockinfo, blockinfo+1, blockinfo+2, blockinfo+3, blockinfo+4, blockinfo+5);
    Np = blockinfo[3] * blockinfo[4] * blockinfo[5];
    local = (PetscScalar *) calloc (Np, sizeof(PetscScalar));
    DFres = (PetscScalar *) calloc (m , sizeof(PetscScalar));
    MatGetDiagonalBlock(A,&Dblock);             // diagonal block of matrix A

    // Set up precondition context
    PCCreate(PETSC_COMM_SELF,&prec);
    if (pc==1) {
        PCSetType(prec,PCICC); 
        //ICC(0), Block-Jacobi
        PetscPrintf(PETSC_COMM_WORLD,"AAR preconditioned with Block-Jacobi using ICC(0).\n");
    }
    else if (pc==0) {
        PCSetType(prec,PCJACOBI); 
        // AAJ
        PetscPrintf(PETSC_COMM_WORLD,"AAR preconditioned with Jacobi (AAJ).\n");
    }
    PCSetOperators(prec,Dblock,Dblock);

    // Initialize vectors
    VecDuplicate(x, &x_old);
    VecDuplicate(x, &res); 
    VecDuplicate(x, &f_old); 
    VecDuplicate(x, &DXDF);
    VecDuplicateVecs(x, m, &DX);                // storage for delta x
    VecDuplicateVecs(x, m, &DF);                // storage for delta residual 
 
    VecNorm(b, NORM_2, &b_2norm); 
    tol *= b_2norm;

#ifdef DEBUG
    double t2, t3, ta = 0, tax = 0;
    t2 = MPI_Wtime();
#endif

    // res = b- A * x
    MatMult(A,x,res); 
    VecAYPX(res, -1.0, b);
    VecNorm(res, NORM_2, &r_2norm); 

#ifdef DEBUG
    t3 = MPI_Wtime();
    tax += (t3-t2);
    PetscPrintf(PETSC_COMM_WORLD,"relres: %lf\n", r_2norm/b_2norm);
#endif

    while (r_2norm > tol && iter <= max_iter){
        // Apply precondition here 
        GetLocalVector(da, res, &res_local, blockinfo, Np, local, &r);
        // precondition(prec, res, da, blockinfo, local);

        VecDuplicate(res_local, &pres_local);
        PCApply(prec, res_local, pres_local);
        RestoreGlobalVector(da, res, pres_local, blockinfo, local, &r);
        
        if (iter > 1){
            k = (iter-2) % m;
            VecWAXPY(DX[k], -1.0, x_old, x);    // DX[k] = x   - x_old
            VecWAXPY(DF[k], -1.0, f_old, res);  // DF[k] = res - f_old
        }

        VecCopy(x, x_old);                      // x_old = x
        VecCopy(res, f_old);                    // f_old = res

        if (iter % p){
            /***********************
             *  Richardson update  *
             ***********************/
            VecAXPY(x, omega, res);             // x = x + omega * res
        } else {
            /***********************************
             *  Anderson extrapolation update  *
             ***********************************/
#ifdef DEBUG            
    t2 = MPI_Wtime();
#endif
            Anderson(DFres, DF, res, m);

            for (i=0; i<m; i++){                                        // x = x - (DX + beta*DF)' * DFres
                VecWAXPY(DXDF, beta, DF[i], DX[i]);
                VecAXPY(x, -DFres[i], DXDF);                            
            }

            VecAXPY(x, beta, res);                                      // x = x + beta * res

#ifdef DEBUG
    t3 = MPI_Wtime();
    ta += (t3 - t2);
#endif
        }

#ifdef DEBUG            
    t2 = MPI_Wtime();
#endif
        MatMult(A,x,res);                       // res = b- A * x
        VecAYPX(res, -1.0,b);
        VecNorm(res, NORM_2, &r_2norm); 

#ifdef DEBUG
    t3 = MPI_Wtime();
    tax += (t3 - t2);  
    PetscPrintf(PETSC_COMM_WORLD,"relres: %g\n", r_2norm/b_2norm);
#endif        
        iter ++;
    }

    if (iter<max_iter) {
        PetscPrintf(PETSC_COMM_WORLD,"AAR converged to a relative residual of %g in %d iterations.\n", r_2norm/b_2norm, iter-1);
    } else {
        PetscPrintf(PETSC_COMM_WORLD,"AAR exceeded maximum iterations %d and converged to a relative residual of %g. \n", iter-1, r_2norm/b_2norm);
    }

    t1=MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"Time taken by AAR = %.4f seconds.\n",t1-t0);
#ifdef DEBUG
    PetscPrintf(PETSC_COMM_WORLD,"Time taken by Anderson update = %.4f seconds.\n",ta);
    PetscPrintf(PETSC_COMM_WORLD,"Time taken by Matrix Vector Multiply = %.4f seconds.\n",tax);
#endif

    // deallocate memory
    VecDestroy(&x_old);
    VecDestroy(&res);
    VecDestroy(&f_old);
    VecDestroy(&DXDF);
    VecDestroy(&res_local);
    VecDestroy(&pres_local);
    VecDestroyVecs(m, &DX);
    VecDestroyVecs(m, &DF);
    free(local);
    free(DFres);
}

/**
 * @brief   precondition function
 *
 *          Apply Jacobian or ICC(0) (Block Jacobian).
 *          res = M \ res
 *          M is precondition matrix
 */

void precondition(PC prec, Vec res, DM da, PetscInt *blockinfo, PetscScalar *local)
{

    int i, j, k, t, rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    PetscInt xcor = blockinfo[0], ycor = blockinfo[1], zcor = blockinfo[2], 
              lxdim = blockinfo[3], lydim = blockinfo[4], lzdim = blockinfo[5];

    PetscInt Np = blockinfo[3] * blockinfo[4] * blockinfo[5];
    PetscScalar ***r;
    Vec localv, f;
    /////////////////////////////////////////////////

    DMDAVecGetArray(da, res, &r);                                 // r is the ghosted local vectors of res

    // Extract local values
    t = 0;
    for (k = zcor; k < lzdim+zcor; k++)
        for (j = ycor; j < lydim+ycor; j++)
            for (i = xcor; i < lxdim+xcor; i++)
                local[t++] = r[k][j][i];

    VecCreateSeqWithArray(MPI_COMM_SELF, 1, Np, local, &localv); // Create local vector with extracted value

    VecDuplicate(localv, &f);
    PCApply(prec, localv, f);                                    // f = M \ localv

    VecGetArray(f, &local);                                      // Get preconditioned residual

    // update local residual
    t = 0;
    for (k = zcor; k < lzdim+zcor; k++)
        for (j = ycor; j < lydim+ycor; j++)
            for (i = xcor; i < lxdim+xcor; i++)
                r[k][j][i] = local[t++];
    DMDAVecRestoreArray(da, res, &r);

    // deallocate memory
    VecDestroy(&localv);
    VecDestroy(&f);
}

/**
 * @brief   Anderson update
 *
 *          x_new = x_prev + beta*res - (DX + beta*DF)*(pinv(DF'*DF)*(DF'*res));
 */

void Anderson(PetscScalar *DFres, Vec *DF, Vec res, PetscInt m)
{
    int lprank, i, j;
    double *svec = malloc(m * sizeof(double));
    PetscScalar * DFtDF = malloc(m*m * sizeof(PetscScalar));    // DFtDF = DF' * DF
    assert(svec != NULL && DFtDF != NULL && DFres != NULL);
    /////////////////////////////////////////////////

    for (i=0; i<m; i++)
        for(j=0; j<i+1; j++)
            VecTDotBegin(DF[i], DF[j], &DFtDF[m*i+j]);          // DFtDF(i,j) = DF[i]' * DF[j]
        
    for (i=0; i<m; i++)
        VecTDotBegin(DF[i], res, &DFres[i]);                    // DFres(i)   = DF[i]' * res

    for (i=0; i<m; i++)
        for(j=0; j<i+1; j++){
            VecTDotEnd(DF[i], DF[j], &DFtDF[m*i+j]);            // DFtDF(i,j) = DF[i]' * DF[j]
            DFtDF[i+m*j] = DFtDF[m*i+j];                        // DFtDF(j,i) = DFtDF(i,j) symmetric 
        }
        
    for (i=0; i<m; i++)
        VecTDotEnd(DF[i], res, &DFres[i]);                      // DFres(i)   = DF[i]' * res

    // Least square problem solver. DFres = pinv(DF'*DF)*(DF'*res)
    LAPACKE_dgelsd(LAPACK_COL_MAJOR, m, m, 1, DFtDF, m, DFres, m, svec, -1.0, &lprank);

    // deallocate memory
    free(svec);
    free(DFtDF);
}



