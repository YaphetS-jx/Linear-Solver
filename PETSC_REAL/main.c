    
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
#include "PGR_Real.h"
#include "PL2R_Real.h"

int main( int argc, char **argv ) {
    int ierr; 
    petsc_real system;
    PetscReal t0,t1;

    PetscInitialize(&argc,&argv,(char*)0,help);

    t0 = MPI_Wtime();

    Setup_and_Initialize(&system, argc, argv);

    // Compute RHS and Matrix for the Poisson equation
    ComputeMatrixA(&system);     
    // Matrix, A = psystem->poissonOpr
    // NOTE: For a different problem, other than Poisson equation, provide the matrix through the variable "psystem->poissonOpr" and right hand side through "psystem->RHS". 

    t1 = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"\nTime spent in initialization = %.4f seconds.\n",t1-t0);

    PetscPrintf(PETSC_COMM_WORLD,"*************************************************************************** \n \n");

    // preconditioning PC=0={Jacobi}, PC=1={ICC(0) block-Jacobi}     
    if (system.solver == 0)
        // -------------- AAR solver --------------------------
        AAR(system.poissonOpr, system.Phi, system.RHS, system.omega, 
            system.beta, system.m, system.p, system.solver_tol, 2000, system.pc, system.da);
        
    else if (system.solver == 1)
        // -------------- PGR solver --------------------------
        PGR(system.poissonOpr, system.Phi, system.RHS, system.omega, 
            system.m, system.p, system.solver_tol, 2000, system.pc, system.da);
        
    else 
        // -------------- PL2R solver --------------------------
        PL2R(system.poissonOpr, system.Phi, system.RHS, system.omega, 
            system.beta, system.m, system.p, system.solver_tol, 2000, system.pc, system.da);

#ifdef DEBUG
    double x_norm;
    VecNorm(system.Phi, NORM_2, &x_norm);
    PetscPrintf(PETSC_COMM_WORLD,"test: %g\n",x_norm);
#endif
    VecView(system.Phi, PETSC_VIEWER_STDOUT_WORLD);
    PetscPrintf(PETSC_COMM_WORLD,"*************************************************************************** \n \n");

    t1 = MPI_Wtime();

    Objects_Destroy(&system);  
    PetscPrintf(PETSC_COMM_WORLD,"Total wall time = %.4f seconds.\n\n",t1-t0);
    ierr = PetscFinalize();
    CHKERRQ(ierr);

    return 0;
}