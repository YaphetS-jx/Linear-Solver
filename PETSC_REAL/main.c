
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
    int ierr, i; 
    petsc_real system;
    PetscReal t0,t1, t2, t3, rnorm, bnorm;

    PetscInt iteration;
    KSP ksp;
    PC pc;
    Vec res;
    FILE *fp;

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
    
    PetscFOpen(PETSC_COMM_WORLD,"time.txt", "w", &fp);
    int no_of_test = 1;
    for (i = 0; i < no_of_test; i ++){
        VecSet(system.AAR, 1.0);
        t2 = MPI_Wtime();
        // -------------- AAR solver --------------------------
        AAR(system.poissonOpr, system.AAR, system.RHS, system.omega, 
            system.beta, system.m, system.p, system.solver_tol, 2000, system.pc, system.da);
        t3 = MPI_Wtime();
        PetscPrintf(PETSC_COMM_WORLD,"*************************************************************************** \n \n");   
        PetscFPrintf(PETSC_COMM_WORLD, fp, "%g\n", t3 - t2);
    }


    for (i = 0; i < no_of_test; i ++){
        VecSet(system.PGR, 1.0);
        t2 = MPI_Wtime();
        // -------------- PGR solver --------------------------
        PGR(system.poissonOpr, system.PGR, system.RHS, system.omega, 
            system.m, system.p, system.solver_tol, 2000, system.pc, system.da);
        t3 = MPI_Wtime();
        PetscPrintf(PETSC_COMM_WORLD,"*************************************************************************** \n \n");
        PetscFPrintf(PETSC_COMM_WORLD, fp, "%g\n", t3 - t2);
    }


    for (i = 0; i < no_of_test; i ++){
        VecSet(system.PL2R, 1.0);
        t2 = MPI_Wtime();
        // -------------- PL2R solver --------------------------
        PL2R(system.poissonOpr, system.PL2R, system.RHS, system.omega, 
            system.beta, system.m, system.p, system.solver_tol, 2000, system.pc, system.da);
        t3 = MPI_Wtime();
        PetscPrintf(PETSC_COMM_WORLD,"*************************************************************************** \n \n");
        PetscFPrintf(PETSC_COMM_WORLD, fp, "%g\n", t3 - t2);
    }


    // VecView(system.Phi, PETSC_VIEWER_STDOUT_WORLD);
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

    for (i = 0; i < no_of_test; i ++){
        VecSet(system.GMRES, 1.0);
        t2 = MPI_Wtime();
        
        KSPSolve(ksp, system.RHS, system.GMRES);
        
        t3 = MPI_Wtime();
        PetscFPrintf(PETSC_COMM_WORLD, fp, "%g\n", t3 - t2);

        KSPGetIterationNumber(ksp, &iteration);
        VecDuplicate(system.RHS, &res);
        MatMult(system.poissonOpr, system.GMRES, res);
        VecAYPX(res, -1.0, system.RHS);
        VecNorm(res, NORM_2, &rnorm);
        VecNorm(system.RHS, NORM_2, &bnorm);

        PetscPrintf(PETSC_COMM_WORLD,"GMRES converged to a relative residual of %g in %d iterations.\n",rnorm/bnorm, iteration);
        PetscPrintf(PETSC_COMM_WORLD,"Time taken by GMRES = %.6f seconds.\n",(t3-t2));
        PetscPrintf(PETSC_COMM_WORLD,"*************************************************************************** \n \n");
    }


    if (system.pc == 1) 
        PetscPrintf(PETSC_COMM_WORLD,"BICG preconditioned with Block-Jacobi using ILU(0).\n");
    else 
        PetscPrintf(PETSC_COMM_WORLD,"BICG preconditioned with Jacobi.\n");
    
    KSPSetType(ksp, KSPBICG);
    KSPSetTolerances(ksp, 1.e-6, PETSC_DEFAULT, PETSC_DEFAULT, 2000);
    KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);

    for (i = 0; i < no_of_test; i ++){
        VecSet(system.BICG, 1.0);
        t2 = MPI_Wtime();

        KSPSolve(ksp, system.RHS, system.BICG);

        t3 = MPI_Wtime();
        PetscFPrintf(PETSC_COMM_WORLD, fp, "%g\n", t3 - t2);

        KSPGetIterationNumber(ksp, &iteration);
        MatMult(system.poissonOpr, system.BICG, res);
        VecAYPX(res, -1.0, system.RHS);
        VecNorm(res, NORM_2, &rnorm);
        
        PetscPrintf(PETSC_COMM_WORLD,"BICG converged to a relative residual of %g in %d iterations.\n",rnorm/bnorm, iteration);
        PetscPrintf(PETSC_COMM_WORLD,"Time taken by BICG = %.6f seconds.\n",(t3-t2));
        PetscPrintf(PETSC_COMM_WORLD,"*************************************************************************** \n \n");  
    }

    if (system.pc == 1) 
        PetscPrintf(PETSC_COMM_WORLD,"CG preconditioned with Block-Jacobi using ILU(0).\n");
    else 
        PetscPrintf(PETSC_COMM_WORLD,"CG preconditioned with Jacobi.\n");

    KSPSetPC(ksp, pc);
    KSPSetType(ksp, KSPCG);
    KSPSetTolerances(ksp, 1.e-6, PETSC_DEFAULT, PETSC_DEFAULT, 2000);
    KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
    KSPSetFromOptions(ksp);

    for (i = 0; i < 50; i ++){
        VecSet(system.GMRES, 1.0);
        t2 = MPI_Wtime();
        
        KSPSolve(ksp, system.RHS, system.GMRES);
        
        t3 = MPI_Wtime();
        PetscFPrintf(PETSC_COMM_WORLD, fp, "%g\n", t3 - t2);

        KSPGetIterationNumber(ksp, &iteration);
        VecDuplicate(system.RHS, &res);
        MatMult(system.poissonOpr, system.GMRES, res);
        VecAYPX(res, -1.0, system.RHS);
        VecNorm(res, NORM_2, &rnorm);
        VecNorm(system.RHS, NORM_2, &bnorm);

        PetscPrintf(PETSC_COMM_WORLD,"CG converged to a relative residual of %g in %d iterations.\n",rnorm/bnorm, iteration);
        PetscPrintf(PETSC_COMM_WORLD,"Time taken by CG = %.6f seconds.\n",(t3-t2));
        PetscPrintf(PETSC_COMM_WORLD,"*************************************************************************** \n \n");
    }

    PetscFClose(PETSC_COMM_WORLD, fp);

    t1 = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"Total wall time = %.4f seconds.\n\n",t1-t0);

#ifdef DEBUG
    double A_norm, b_norm, x1_norm, x2_norm, x3_norm, x4_norm, x5_norm;
    MatNorm(system.poissonOpr, NORM_FROBENIUS, &A_norm);
    VecNorm(system.RHS, NORM_2, &b_norm);
    VecNorm(system.AAR, NORM_2, &x1_norm);
    VecNorm(system.PGR, NORM_2, &x2_norm);
    VecNorm(system.PL2R, NORM_2, &x3_norm);
    VecNorm(system.GMRES, NORM_2, &x4_norm);
    VecNorm(system.BICG, NORM_2, &x5_norm);
    PetscPrintf(PETSC_COMM_WORLD,"Norm A: %g, b: %g, x1: %g, x2: %g, x3: %g, x4: %g, x5: %g\n",A_norm, b_norm, x1_norm, x2_norm, x3_norm, x4_norm, x5_norm);
#endif
    
    VecDestroy(&res);
    KSPDestroy(&ksp);
    Objects_Destroy(&system); 
    ierr = PetscFinalize();
    CHKERRQ(ierr);

    return 0;
}