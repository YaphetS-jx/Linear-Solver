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

void Read_parameters(AAR_OBJ* pAAR, int argc, char **argv) {    
    int rank; 
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    PetscInt p,i;
    PetscReal Nr, Dr, val;

    pAAR->order = 1; 
    // store half order
    pAAR->numPoints_x = 8; pAAR->numPoints_y = 8; pAAR->numPoints_z = 8;

    if (argc < 7) {
        if (rank==0) printf("Wrong inputs");
        exit(-1);
    } else {
        pAAR->pc = atoi(argv[1]);
        pAAR->solver_tol = atof(argv[2]);
        pAAR->m_aar = atoi(argv[3]);
        pAAR->p_aar = atoi(argv[4]);
        pAAR->omega_aar = atof(argv[5]);
        pAAR->beta_aar = atof(argv[6]);
    }



    //coefficients of the laplacian
    pAAR->coeffs[0] = 0;
    for (p=1; p<=pAAR->order; p++)
        pAAR->coeffs[0]+= ((PetscReal)1.0/(p*p));

    pAAR->coeffs[0]*=((PetscReal)3.0);

    for (p=1;p<=pAAR->order;p++) {
        Nr=1; 
        Dr=1;
        for(i=pAAR->order-p+1; i<=pAAR->order; i++)
            Nr*=i;
        for(i=pAAR->order+1; i<=pAAR->order+p; i++)
            Dr*=i;
        val = Nr/Dr;  
        pAAR->coeffs[p] = (PetscReal)(-1*pow(-1,p+1)*val/(p*p*(1)));
    }

    for (p=0;p<=pAAR->order;p++) {
        pAAR->coeffs[p] = pAAR->coeffs[p]/(2*M_PI); 
        // so total (-1/4*pi) factor on fd coeffs
    }  

    PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n");
    PetscPrintf(PETSC_COMM_WORLD,"                           INPUT PARAMETERS                                \n");
    PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n");

    //PetscPrintf(PETSC_COMM_WORLD,"FD_ORDER    : %d\n",2*pAAR->order);
    PetscPrintf(PETSC_COMM_WORLD,"aar_pc      : %d\n",pAAR->pc);
    PetscPrintf(PETSC_COMM_WORLD,"solver_tol  : %e \n",pAAR->solver_tol);
    PetscPrintf(PETSC_COMM_WORLD,"m_aar       : %d\n",pAAR->m_aar);
    PetscPrintf(PETSC_COMM_WORLD,"p_aar       : %d\n",pAAR->p_aar);
    PetscPrintf(PETSC_COMM_WORLD,"omega_aar   : %lf\n",pAAR->omega_aar);
    PetscPrintf(PETSC_COMM_WORLD,"beta_aar    : %lf\n",pAAR->beta_aar);

    PetscPrintf(PETSC_COMM_WORLD,"***************************************************************************\n");

    return;
}

/*****************************************************************************/
/*****************************************************************************/
/*********************** Setup and finalize functions ************************/
/*****************************************************************************/
/*****************************************************************************/

void Setup_and_Initialize(AAR_OBJ* pAAR, int argc, char **argv) {
    ObjectInitialize(pAAR);
    Read_parameters(pAAR, argc, argv);
    Objects_Create(pAAR);      
}

void ObjectInitialize(AAR_OBJ* pAAR) {
    PetscOptionsGetString(PETSC_NULL,"-name",pAAR->file,sizeof(pAAR->file),PETSC_NULL);
}

void Objects_Create(AAR_OBJ* pAAR) {
    PetscInt n_x = pAAR->numPoints_x;
    PetscInt n_y = pAAR->numPoints_y;
    PetscInt n_z = pAAR->numPoints_z;
    PetscInt o = pAAR->order;
    double RHSsum;

    PetscInt gxdim,gydim,gzdim,xcor,ycor,zcor,lxdim,lydim,lzdim,nprocx,nprocy,nprocz;
    int i;
    Mat A;
    PetscMPIInt comm_size;
    MPI_Comm_size(PETSC_COMM_WORLD,&comm_size);

    DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DM_BOUNDARY_PERIODIC,DMDA_STENCIL_STAR,n_x,n_y,n_z,
    PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,o,0,0,0,&pAAR->da);        // create a pattern and communication layout

    DMCreateGlobalVector(pAAR->da,&pAAR->RHS);                          // using the layour of da to create vectors RHS
    VecDuplicate(pAAR->RHS,&pAAR->Phi);                                 // create Phi by duplicating the pattern of RHS

    PetscRandom rnd;
    unsigned long seed;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);


    // RHS vector
    PetscRandomCreate(PETSC_COMM_WORLD,&rnd);
    PetscRandomSetFromOptions(rnd);
    seed=rank; 
    
    PetscRandomSetSeed(rnd,seed);
    PetscRandomSeed(rnd);

    VecSetRandom(pAAR->RHS,rnd);

    VecSum(pAAR->RHS,&RHSsum);
    RHSsum = -RHSsum/(n_x*n_y*n_z);
    VecShift(pAAR->RHS,RHSsum); 

    PetscRandomDestroy(&rnd);

    // Initial random guess
    PetscRandomCreate(PETSC_COMM_WORLD,&rnd);
    PetscRandomSetFromOptions(rnd);
    seed=1;  
    PetscRandomSetSeed(rnd,seed);
    PetscRandomSeed(rnd);

    VecSetRandom(pAAR->Phi,rnd);

    PetscRandomDestroy(&rnd);

    if (comm_size == 1 ) {
        DMCreateMatrix(pAAR->da,&pAAR->poissonOpr);
        DMSetMatType(pAAR->da,MATSEQSBAIJ); // sequential symmetric block sparse matrices
    } else  {
        DMCreateMatrix(pAAR->da,&pAAR->poissonOpr);
        DMSetMatType(pAAR->da,MATMPISBAIJ); // distributed symmetric sparse block matrices
    }

}


// Destroy objects
void Objects_Destroy(AAR_OBJ* pAAR) {
    DMDestroy(&pAAR->da);
    VecDestroy(&pAAR->RHS); 
    VecDestroy(&pAAR->Phi);
    MatDestroy(&pAAR->poissonOpr); 

    return;
}

/*****************************************************************************/
/*****************************************************************************/
/***************************** Matrix creation *******************************/
/*****************************************************************************/
/*****************************************************************************/

void ComputeMatrixA(AAR_OBJ* pAAR) {
    PetscInt i,j,k,l,colidx,gxdim,gydim,gzdim,xcor,ycor,zcor,lxdim,lydim,lzdim,nprocx,nprocy,nprocz;
    MatStencil row;
    MatStencil* col;
    PetscScalar* val;
    PetscInt o = pAAR->order;  
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    PetscScalar Dinv_factor;
    Dinv_factor = (pAAR->coeffs[0]); // (-1/4pi)(Lap) = Diag term
    pAAR->Dinv_factor = 1/Dinv_factor;

    DMDAGetInfo(pAAR->da,0,&gxdim,&gydim,&gzdim,&nprocx,&nprocy,&nprocz,0,0,0,0,0,0);  // only get xyz dimension and number of processes in each direction
    PetscPrintf(PETSC_COMM_WORLD,"nprocx: %d, nprocy: %d, nprocz: %d\n",nprocx,nprocy,nprocz); 
    // PetscPrintf(PETSC_COMM_WORLD,"gxdim: %d, gydim: %d, gzdim: %d\n",gxdim, gydim, gzdim);  dim = number of points

    DMDAGetCorners(pAAR->da,&xcor,&ycor,&zcor,&lxdim,&lydim,&lzdim);
    // printf("rank: %d, xcor: %d, ycor: %d, zcor: %d, lxdim: %d, lydim: %d, lzdim: %d\n",rank, xcor,ycor,zcor,lxdim,lydim,lzdim);
    // Returns the global (x,y,z) indices of the lower left corner and size of the local region, excluding ghost points.

    MatScale(pAAR->poissonOpr,0.0);
    // MatView(pAAR->poissonOpr,PETSC_VIEWER_STDOUT_WORLD);

    PetscMalloc(sizeof(MatStencil)*(o*6+1),&col);
    PetscMalloc(sizeof(PetscScalar)*(o*6+1),&val);

    // within each local subblock e.g. from xcor to xcor+lxdim-1. using global indexing
    for(k=zcor; k<zcor+lzdim; k++) {
        for(j=ycor; j<ycor+lydim; j++) {
            for(i=xcor; i<xcor+lxdim; i++) {
                row.k = k; row.j = j, row.i = i;

                colidx=0; 
                col[colidx].i=i ;col[colidx].j=j ;col[colidx].k=k ;
                val[colidx++]=pAAR->coeffs[0] ;
                for(l=1;l<=o;l++) {
                    // z
                    col[colidx].i=i ;col[colidx].j=j ;col[colidx].k=k-l ;
                    val[colidx++]=pAAR->coeffs[l];
                    col[colidx].i=i ;col[colidx].j=j ;col[colidx].k=k+l ;
                    val[colidx++]=pAAR->coeffs[l];
                    //y 
                    col[colidx].i=i ;col[colidx].j=j-l ;col[colidx].k=k ;
                    val[colidx++]=pAAR->coeffs[l];
                    col[colidx].i=i ;col[colidx].j=j+l ;col[colidx].k=k ;
                    val[colidx++]=pAAR->coeffs[l];
                    // x
                    col[colidx].i=i-l ;col[colidx].j=j ;col[colidx].k=k ;
                    val[colidx++]=pAAR->coeffs[l];
                    col[colidx].i=i+l ;col[colidx].j=j ;col[colidx].k=k ;
                    val[colidx++]=pAAR->coeffs[l];

                    // 3 directions x y z and positive axis and negative axis
                }
                MatSetValuesStencil(pAAR->poissonOpr,1,&row,6*o+1,col,val,ADD_VALUES); // ADD_VALUES, add values to any existing value
            }
        }
    }
    MatAssemblyBegin(pAAR->poissonOpr, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(pAAR->poissonOpr, MAT_FINAL_ASSEMBLY); 

    // MatView(pAAR->poissonOpr,PETSC_VIEWER_STDOUT_WORLD);
}
