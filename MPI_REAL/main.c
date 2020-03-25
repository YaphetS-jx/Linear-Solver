/*=============================================================================================
| Alternating Anderson Richardson (AAR), Periodic Galerkin Richardson (PGR), 
| Periodic L2 Richardson (PL2R) code and tests in real-valued systems.
| Copyright (C) 2020 Material Physics & Mechanics Group at Georgia Tech.
| 
| Authors: Xin Jing, Phanish Suryanarayana
|
| Last Modified: 24 March 2020
|-------------------------------------------------------------------------------------------*/


#include "system.h"
#include "AAR.h"
#include "PGR.h"
#include "PL2R.h"
#include "CG.h"
#include "tools.h"

int main(int argc, char ** argv) {
    DS aar = {};   

    int rank, psize, i, j; 
    FILE * fp;
    MPI_Init(&argc, &argv); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &psize); 
    aar.nproc = psize;  // no. of processors

    double t_begin, t_end, t0, t1; 
    t_begin = MPI_Wtime(); 

    CheckInputs(&aar, argc, argv); 
    Initialize(&aar);  

    t_end = MPI_Wtime(); 
    if (rank  ==  0)
        printf("Time spent in initialization = %.4f seconds. \n", t_end-t_begin); 

    MPI_Barrier(MPI_COMM_WORLD);
    if (!rank) printf("*************************************************************************** \n\n"); 
    int Np = aar.np_x * aar.np_y * aar.np_z;


    fp = fopen("time.txt", "w");

    for (j = 0; j < 10; j++){
        for (i = 0; i < Np; i ++)
            aar.phi_v[i] = 1;

        if (!rank) printf("AAR starts.\n"); 
        t0 = MPI_Wtime();
        AAR(&aar, PoissonResidual, Precondition, aar.phi_v, aar.rhs_v, aar.omega, aar.beta, aar.m, aar.p, aar.solver_maxiter, aar.solver_tol, Np, aar.comm_laplacian);  
        t1 = MPI_Wtime();
        if (!rank) printf("\n*************************************************************************** \n\n"); 
        fprintf(fp, "%g\n", t1 - t0);
    }
    
    for (j = 0; j < 10; j++){
        for (i = 0; i < Np; i ++)
            aar.phi_v2[i] = 1;

        if (!rank) printf("PGR starts.\n"); 
        t0 = MPI_Wtime();
        PGR(&aar, PoissonResidual, Precondition, aar.phi_v2, aar.rhs_v, aar.omega, aar.m, aar.p, aar.solver_maxiter, aar.solver_tol, Np, aar.comm_laplacian);  
        t1 = MPI_Wtime();
        if (!rank) printf("\n*************************************************************************** \n\n"); 
        fprintf(fp, "%g\n", t1 - t0);
    }

    for (j = 0; j < 10; j++){
        for (i = 0; i < Np; i ++)
            aar.phi_v3[i] = 1;

        if (!rank) printf("PL2R starts.\n"); 
        t0 = MPI_Wtime();
        PL2R(&aar, PoissonResidual, Precondition, aar.phi_v3, aar.rhs_v, aar.omega, aar.m, aar.p, aar.solver_maxiter, aar.solver_tol, Np, aar.comm_laplacian);  
        t1 = MPI_Wtime();
        if (!rank) printf("\n*************************************************************************** \n\n"); 
        fprintf(fp, "%g\n", t1 - t0);
    }

    for (j = 0; j < 10; j++){
        for (i = 0; i < Np; i ++)
            aar.phi_v4[i] = 1;

        if (!rank) printf("CG starts.\n"); 
        t0 = MPI_Wtime();
        CG(&aar, PoissonResidual, NULL,  Np, aar.phi_v4, aar.rhs_v, aar.solver_tol, aar.solver_maxiter, aar.comm_laplacian);  
        t1 = MPI_Wtime();
        if (!rank) printf("\n*************************************************************************** \n\n"); 
        fprintf(fp, "%g\n", t1 - t0);
    }
    
    fclose(fp);


    Deallocate_memory(&aar);   /// <  De-allocate memory.

    t_end = MPI_Wtime(); 
    if (rank  ==  0) {
        printf("Total wall time   = %.4f seconds. \n", t_end-t_begin); 
        char* c_time_str; 
        time_t current_time = time(NULL); 
        c_time_str = ctime(&current_time);   
        printf("Ending time: %s", c_time_str); 
        printf("*************************************************************************** \n \n"); 
    }
    MPI_Barrier(MPI_COMM_WORLD);    

    MPI_Finalize(); 
    return 0; 
} 
