    
/*=============================================================================================
| Alternating Anderson Richardson (AAR) code for real-valued systems
| Copyright (C) 2020 Material Physics & Mechanics Group at Georgia Tech.
| 
| Authors: Xin Jing, Phanish Suryanarayana
|
| Last Modified: 3 March 2020
|-------------------------------------------------------------------------------------------*/

#include "system.h"
#include "AAR_Real.h"

int main( int argc, char **argv ) {
    int ierr; 
    AAR_OBJ aar;
    PetscReal t0,t1;

    PetscInitialize(&argc,&argv,(char*)0,help);

    t0=MPI_Wtime();

    Setup_and_Initialize(&aar, argc, argv);

    // Compute RHS and Matrix for the Poisson equation
    ComputeMatrixA(&aar);     
    // Matrix, A = paar->poissonOpr
    // NOTE: For a different problem, other than Poisson equation, provide the matrix through the variable "paar->poissonOpr" and right hand side through "paar->RHS". 

    t1=MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"\nTime spent in initialization = %.4f seconds.\n",t1-t0);

    PetscPrintf(PETSC_COMM_WORLD,"*************************************************************************** \n \n");

    // -------------- AAR solver --------------------------
    AAR(aar.poissonOpr, aar.Phi, aar.RHS, aar.omega_aar, 
        aar.beta_aar, aar.m_aar, aar.p_aar, aar.solver_tol, 2000, aar.pc, aar.da);
    // AAR with preconditioning PC=0={Jacobi}, PC=1={ICC(0) block-Jacobi}     
    PetscPrintf(PETSC_COMM_WORLD,"*************************************************************************** \n \n");

    t1=MPI_Wtime();

    Objects_Destroy(&aar);  
    PetscPrintf(PETSC_COMM_WORLD,"Total wall time = %.4f seconds.\n\n",t1-t0);
    ierr = PetscFinalize();
    CHKERRQ(ierr);

    return 0;
}