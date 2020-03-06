
/*=============================================================================================
| Alternating Anderson Richardson (AAR), Periodic Galerkin Richardson (PGR), 
| Periodic L2 Richardson (PL2R) code and tests in real-valued systems.
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
    PetscReal t0,t1, t2, t3;

    PetscInt iteration;
    KSP ksp;
    PC pc;

    PetscInitialize(&argc,&argv,(char*)0,help);

    t0 = MPI_Wtime();

    Setup_and_Initialize(&system, argc, argv);

    // Compute RHS and Matrix for the Poisson equation
    ComputeMatrixA(&system);     
    // Matrix, A = psystem->poissonOpr
    // NOTE: For a different problem, other than Poisson equation, provide the matrix through the variable "psystem->poissonOpr"
    // and right hand side through "psystem->RHS". 

    t1 = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"\nTime spent in initialization = %.4f seconds.\n",t1-t0);

    PetscPrintf(PETSC_COMM_WORLD,"*************************************************************************** \n \n");
    
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
    double A_norm, b_norm, x_norm;
    MatNorm(system.poissonOpr, NORM_FROBENIUS, &A_norm);
    VecNorm(system.RHS, NORM_2, &b_norm);
    VecNorm(system.Phi, NORM_2, &x_norm);
    PetscPrintf(PETSC_COMM_WORLD,"Norm A: %g, b: %g, x: %g\n",A_norm, b_norm, x_norm);
#endif

    // VecView(system.Phi, PETSC_VIEWER_STDOUT_WORLD);
    PetscPrintf(PETSC_COMM_WORLD,"*************************************************************************** \n \n");

    t2 = MPI_Wtime();
    KSPCreate(PETSC_COMM_WORLD, &ksp);

    KSPSetOperators(ksp, system.poissonOpr, system.poissonOpr);
    KSPGetPC(ksp,&pc);

    if (system.pc == 1) {
        PCSetType(pc, PCBJACOBI); 
        PetscPrintf(PETSC_COMM_WORLD,"GMRES preconditioned with Block-Jacobi using ILU(0).\n");
    }
    else {
        PCSetType(pc, PCJACOBI); 
        PetscPrintf(PETSC_COMM_WORLD,"GMRES preconditioned with Jacobi.\n");
    }

    KSPSetPC(ksp, pc);
    KSPSetType(ksp, KSPGMRES);
    KSPSetTolerances(ksp, 1.e-6, PETSC_DEFAULT, PETSC_DEFAULT, 2000);
    KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
    KSPSetFromOptions(ksp);

    KSPSolve(ksp, system.RHS, system.GMRES);

    KSPGetIterationNumber(ksp, &iteration);
    
    t3 = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"GMRES Iterations %D, Time: %.4f seconds.\n",iteration, (t3-t2));

    PetscPrintf(PETSC_COMM_WORLD,"*************************************************************************** \n \n");

    if (system.pc == 1) 
        PetscPrintf(PETSC_COMM_WORLD,"BICG preconditioned with Block-Jacobi using ILU(0).\n");
    else 
        PetscPrintf(PETSC_COMM_WORLD,"BICG preconditioned with Jacobi.\n");
    
    t2 = MPI_Wtime();
    KSPSetType(ksp, KSPBICG);
    KSPSolve(ksp, system.RHS, system.BICG);
    KSPGetIterationNumber(ksp, &iteration);
    t3 = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"BICG Iterations %D, Time: %.4f seconds.\n",iteration, (t3-t2));
    
    t1 = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"*************************************************************************** \n \n");
    PetscPrintf(PETSC_COMM_WORLD,"Total wall time = %.4f seconds.\n\n",t1-t0);
    KSPDestroy(&ksp);
    Objects_Destroy(&system); 
    ierr = PetscFinalize();
    CHKERRQ(ierr);

    return 0;
}