/**
 * @file    system.c
 * @brief   This file contains functions for constructing matrix, RHS, initial guess
 *
 * @author  Xin Jing <xjing30@gatech.edu>
 *          Phanish Suryanarayana <phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */

#include "system.h"

void Read_parameters(petsc_real* system, int argc, char **argv) {    
    int rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    PetscInt p, i; 
    PetscReal Nr, Dr, val; 

    system->order = 8; 
    // store half order
    system->numPoints_x = 100; system->numPoints_y = 100; system->numPoints_z = 100; 

    if (argc < 8) {
        PetscPrintf(PETSC_COMM_WORLD, "Wrong inputs\n"); 
        exit(-1); 
    } else {
        system->solver = atoi(argv[1]); 
        system->pc = atoi(argv[2]); 
        system->solver_tol = atof(argv[3]); 
        system->m = atoi(argv[4]); 
        system->p = atoi(argv[5]); 
        system->omega = atof(argv[6]); 
        system->beta = atof(argv[7]); 
    }

    if (system->solver >2 || system->solver <0){
        PetscPrintf(PETSC_COMM_WORLD, "Nonexistent solver\n"); 
        exit(-1); 
    }

    if (system->pc >1 || system->pc <0){
        PetscPrintf(PETSC_COMM_WORLD, "Nonexistent precondition\n"); 
        exit(-1); 
    }

    //coefficients of the laplacian
    system->coeffs[0] = 0; 
    for (p=1; p<=system->order; p++)
        system->coeffs[0]+= ((PetscReal)1.0/(p*p)); 

    system->coeffs[0]*=((PetscReal)3.0); 

    for (p=1; p<=system->order; p++) {
        Nr=1; 
        Dr=1; 
        for(i=system->order-p+1; i<=system->order; i++)
            Nr*=i; 
        for(i=system->order+1; i<=system->order+p; i++)
            Dr*=i; 
        val = Nr/Dr;  
        system->coeffs[p] = (PetscReal)(-1*pow(-1, p+1)*val/(p*p*(1))); 
    }

    for (p=0; p<=system->order; p++) {
        system->coeffs[p] = system->coeffs[p]/(2*M_PI); 
        // so total (-1/4*pi) factor on fd coeffs
    }  

    PetscPrintf(PETSC_COMM_WORLD, "***************************************************************************\n"); 
    PetscPrintf(PETSC_COMM_WORLD, "                           INPUT PARAMETERS                                \n"); 
    PetscPrintf(PETSC_COMM_WORLD, "***************************************************************************\n"); 

    //PetscPrintf(PETSC_COMM_WORLD, "FD_ORDER    : %d\n", 2*system->order); 
    if (system->solver == 0)
        PetscPrintf(PETSC_COMM_WORLD, "Solver      : AAR\n"); 
    else if (system->solver == 1)
        PetscPrintf(PETSC_COMM_WORLD, "Solver      : PGR\n"); 
    else
        PetscPrintf(PETSC_COMM_WORLD, "Solver      : PL2R\n"); 
    if (system->pc == 1)
        PetscPrintf(PETSC_COMM_WORLD, "system_pc   : Block-Jacobi using ICC(0)\n"); 
    else
        PetscPrintf(PETSC_COMM_WORLD, "system_pc   : Jacobi \n"); 
    PetscPrintf(PETSC_COMM_WORLD, "solver_tol  : %e \n", system->solver_tol); 
    PetscPrintf(PETSC_COMM_WORLD, "m           : %d\n", system->m); 
    PetscPrintf(PETSC_COMM_WORLD, "p           : %d\n", system->p); 
    PetscPrintf(PETSC_COMM_WORLD, "omega       : %lf\n", system->omega); 
    PetscPrintf(PETSC_COMM_WORLD, "beta        : %lf\n", system->beta); 

    PetscPrintf(PETSC_COMM_WORLD, "***************************************************************************\n"); 

    return; 
}

/*****************************************************************************/
/*****************************************************************************/
/*********************** Setup and finalize functions ************************/
/*****************************************************************************/
/*****************************************************************************/

void Setup_and_Initialize(petsc_real* system, int argc, char **argv) {
    Read_parameters(system, argc, argv); 
    Objects_Create(system);      
}

void Objects_Create(petsc_real* system) {
    PetscInt n_x = system->numPoints_x; 
    PetscInt n_y = system->numPoints_y; 
    PetscInt n_z = system->numPoints_z; 
    PetscInt o = system->order; 
    double RHSsum; 

    PetscInt gxdim, gydim, gzdim, xcor, ycor, zcor, lxdim, lydim, lzdim, nprocx, nprocy, nprocz; 
    int i; 
    Mat A; 
    PetscMPIInt comm_size; 
    MPI_Comm_size(PETSC_COMM_WORLD, &comm_size); 

    PetscErrorCode ieer = DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_STAR, n_x, n_y, n_z, 
    PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, o, 0, 0, 0, &system->da);        // create a pattern and communication layout
    DMSetUp(system->da);

    DMCreateGlobalVector(system->da, &system->RHS);                          // using the layour of da to create vectors RHS
    VecDuplicate(system->RHS, &system->Phi);                                 // create Phi by duplicating the pattern of RHS
    VecDuplicate(system->RHS, &system->GMRES);                               // create Initial by duplicating the pattern of RHS
    VecDuplicate(system->RHS, &system->BICG);                                // create Initial by duplicating the pattern of RHS

    PetscRandom rnd; 
    unsigned long seed; 
    int rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 


    // RHS vector
    PetscRandomCreate(PETSC_COMM_WORLD, &rnd); 
    PetscRandomSetFromOptions(rnd); 
    seed=rank; 
    
    PetscRandomSetSeed(rnd, seed); 
    PetscRandomSeed(rnd); 

    VecSetRandom(system->RHS, rnd); 
    VecSum(system->RHS, &RHSsum); 
    RHSsum = -RHSsum/(n_x*n_y*n_z); 
    VecShift(system->RHS, RHSsum); 

    PetscRandomDestroy(&rnd); 

    // VecView(system->RHS, PETSC_VIEWER_STDOUT_WORLD); 

    // Initial random guess
    PetscRandomCreate(PETSC_COMM_WORLD, &rnd); 
    PetscRandomSetFromOptions(rnd); 
    seed=1;  
    PetscRandomSetSeed(rnd, seed); 
    PetscRandomSeed(rnd); 

    VecSetRandom(system->Phi, rnd); 
    VecCopy(system->Phi, system->GMRES);
    VecCopy(system->Phi, system->BICG);

    PetscRandomDestroy(&rnd); 

    // Initial all ones guess 
    // VecSet(system->Phi, 1.0); 

    if (comm_size == 1 ) {
        DMCreateMatrix(system->da, &system->poissonOpr); 
        DMSetMatType(system->da, MATSEQSBAIJ); // sequential symmetric block sparse matrices
    } else  {
        DMCreateMatrix(system->da, &system->poissonOpr); 
        DMSetMatType(system->da, MATMPISBAIJ); // distributed symmetric sparse block matrices
    }

}


// Destroy objects
void Objects_Destroy(petsc_real* system) {
    DMDestroy(&system->da); 
    VecDestroy(&system->RHS); 
    VecDestroy(&system->Phi); 
    MatDestroy(&system->poissonOpr); 

    return; 
}

/*****************************************************************************/
/*****************************************************************************/
/***************************** Matrix creation *******************************/
/*****************************************************************************/
/*****************************************************************************/

void ComputeMatrixA(petsc_real* system) {
    PetscInt i, j, k, l, colidx, gxdim, gydim, gzdim, xcor, ycor, zcor, lxdim, lydim, lzdim, nprocx, nprocy, nprocz; 
    MatStencil row; 
    MatStencil* col; 
    PetscScalar* val; 
    PetscInt o = system->order;  
    int rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

    PetscScalar Dinv_factor; 
    Dinv_factor = (system->coeffs[0]); // (-1/4pi)(Lap) = Diag term
    system->Dinv_factor = 1/Dinv_factor; 

    DMDAGetInfo(system->da, 0, &gxdim, &gydim, &gzdim, &nprocx, &nprocy, &nprocz, 0, 0, 0, 0, 0, 0);  // only get xyz dimension and number of processes in each direction
    PetscPrintf(PETSC_COMM_WORLD, "nprocx: %d, nprocy: %d, nprocz: %d\n", nprocx, nprocy, nprocz); 
    // PetscPrintf(PETSC_COMM_WORLD, "gxdim: %d, gydim: %d, gzdim: %d\n", gxdim, gydim, gzdim);  dim = number of points

    DMDAGetCorners(system->da, &xcor, &ycor, &zcor, &lxdim, &lydim, &lzdim); 
    // printf("rank: %d, xcor: %d, ycor: %d, zcor: %d, lxdim: %d, lydim: %d, lzdim: %d\n", rank, xcor, ycor, zcor, lxdim, lydim, lzdim); 
    // Returns the global (x, y, z) indices of the lower left corner and size of the local region, excluding ghost points.

    MatScale(system->poissonOpr, 0.0); 
    // MatView(system->poissonOpr, PETSC_VIEWER_STDOUT_WORLD); 

    PetscMalloc(sizeof(MatStencil)*(o*6+1), &col); 
    PetscMalloc(sizeof(PetscScalar)*(o*6+1), &val); 

    // within each local subblock e.g. from xcor to xcor+lxdim-1. using global indexing
    for(k=zcor; k<zcor+lzdim; k++) {
        for(j=ycor; j<ycor+lydim; j++) {
            for(i=xcor; i<xcor+lxdim; i++) {
                row.k = k; row.j = j, row.i = i; 

                colidx=0; 
                col[colidx].i=i; col[colidx].j=j; col[colidx].k=k; 
                val[colidx++]=system->coeffs[0]; 
                for(l=1; l<=o; l++) {
                    // z
                    col[colidx].i=i; col[colidx].j=j; col[colidx].k=k-l; 
                    val[colidx++]=system->coeffs[l]; 
                    col[colidx].i=i; col[colidx].j=j; col[colidx].k=k+l; 
                    val[colidx++]=system->coeffs[l]; 
                    //y 
                    col[colidx].i=i; col[colidx].j=j-l; col[colidx].k=k; 
                    val[colidx++]=system->coeffs[l]; 
                    col[colidx].i=i; col[colidx].j=j+l; col[colidx].k=k; 
                    val[colidx++]=system->coeffs[l]; 
                    // x
                    col[colidx].i=i-l; col[colidx].j=j; col[colidx].k=k; 
                    val[colidx++]=system->coeffs[l]; 
                    col[colidx].i=i+l; col[colidx].j=j; col[colidx].k=k; 
                    val[colidx++]=system->coeffs[l]; 

                    // 3 directions x y z and positive axis and negative axis
                }
                MatSetValuesStencil(system->poissonOpr, 1, &row, 6*o+1, col, val, ADD_VALUES); // ADD_VALUES, add values to any existing value
            }
        }
    }
    MatAssemblyBegin(system->poissonOpr, MAT_FINAL_ASSEMBLY); 
    MatAssemblyEnd(system->poissonOpr, MAT_FINAL_ASSEMBLY); 

    // MatView(system->poissonOpr, PETSC_VIEWER_STDOUT_WORLD); 
}
