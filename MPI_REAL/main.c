/*=============================================================================================
| Alternating Anderson Richardson (AAR), Periodic Galerkin Richardson (PGR), 
| Periodic L2 Richardson (PL2R) code and tests in real-valued systems.
| Copyright (C) 2020 Material Physics & Mechanics Group at Georgia Tech.
| 
| Authors: Xin Jing, Phanish Suryanarayana
|
| Last Modified: 19 March 2020
|-------------------------------------------------------------------------------------------*/


#include "system.h"
#include "AAR.h"
#include "PGR.h"
#include "PL2R.h"
#include "CG.h"
#include "tools.h"

int main(int argc, char ** argv) {
    DS aar = {};   

    int rank, psize; 
    MPI_Init(&argc, &argv); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &psize); 
    aar.nproc = psize;  // no. of processors

    double t_begin, t_end; 
    t_begin = MPI_Wtime(); 

    CheckInputs(&aar, argc, argv); 
    Initialize(&aar);  

    t_end = MPI_Wtime(); 
    if (rank  ==  0)
        printf("Time spent in initialization = %.4f seconds. \n", t_end-t_begin); 

    MPI_Barrier(MPI_COMM_WORLD);
    if (!rank) printf("*************************************************************************** \n\n"); 

    if (!rank) printf("AAR starts.\n"); 
    int Np = aar.np_x * aar.np_y * aar.np_z;
    AAR(&aar, PoissonResidual, Precondition, aar.phi_v, aar.rhs_v, aar.omega, aar.beta, aar.m, aar.p, aar.solver_maxiter, aar.solver_tol, Np, aar.comm_laplacian);  
    if (!rank) printf("\n*************************************************************************** \n\n"); 

    if (!rank) printf("PGR starts.\n"); 
    PGR(&aar, PoissonResidual, Precondition, aar.phi_v2, aar.rhs_v, aar.omega, aar.m, aar.p, aar.solver_maxiter, aar.solver_tol, Np, aar.comm_laplacian);  
    if (!rank) printf("\n*************************************************************************** \n\n"); 

    if (!rank) printf("PL2R starts.\n"); 
    PL2R(&aar, PoissonResidual, Precondition, aar.phi_v3, aar.rhs_v, aar.omega, aar.m, aar.p, aar.solver_maxiter, aar.solver_tol, Np, aar.comm_laplacian);  
    if (!rank) printf("\n*************************************************************************** \n\n"); 

    if (!rank) printf("CG starts.\n"); 
    CG(&aar, PoissonResidual, NULL,  Np, aar.phi_v4, aar.rhs_v, aar.solver_tol, aar.solver_maxiter, aar.comm_laplacian);  
    if (!rank) printf("\n*************************************************************************** \n\n"); 

    double x1, x2, x3, x4;
    Vector2Norm(aar.phi_v, aar.np_x*aar.np_y*aar.np_z, &x1, MPI_COMM_WORLD);
    Vector2Norm(aar.phi_v2, aar.np_x*aar.np_y*aar.np_z, &x2, MPI_COMM_WORLD);
    Vector2Norm(aar.phi_v3, aar.np_x*aar.np_y*aar.np_z, &x3, MPI_COMM_WORLD);
    Vector2Norm(aar.phi_v4, aar.np_x*aar.np_y*aar.np_z, &x4, MPI_COMM_WORLD);

    int i;
    double dot = 0;
    for(i = 0; i< aar.np_x*aar.np_y*aar.np_z; i++)
        dot += aar.rhs_v[i] * aar.rhs_v[i];
    // printf("rank: %d, dot: %g\n", rank, dot);

    // for(i = 0; i< aar.np_x*aar.np_y*aar.np_z; i++)
    //     if(rank==0) printf("%g\n", aar.rhs_v[i]);

    // FILE *fp = fopen("rhs.txt", "w");
    // for(i = 0; i< aar.np_x*aar.np_y*aar.np_z; i++)
    //     fprintf(fp, "%.8f\n", aar.rhs_v[i]);
    // fclose(fp);

#ifdef DEBUG
    if(rank==0) printf("x_norm: %g, %g, %g, %g\n", x1, x2, x3, x4);
#endif
    double rhs_norm;
    Vector2Norm(aar.rhs_v, aar.np_x*aar.np_y*aar.np_z, &rhs_norm, MPI_COMM_WORLD);
    if(rank==0) printf("rhs_norm: %g\n", rhs_norm);

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
