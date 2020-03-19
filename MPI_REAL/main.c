
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
#include "tools.h"

int main(int argc,  char ** argv) {
    DS_AAR aar = {};   

    int rank, psize; 
    MPI_Init(&argc,  &argv); 
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

    int Np = aar.np_x * aar.np_y * aar.np_z;
    AAR(&aar, PoissonResidual, aar.phi_v, aar.rhs_v, aar.omega_aar, aar.beta_aar, aar.m_aar, aar.p_aar, aar.solver_maxiter, aar.solver_tol, Np, aar.comm_laplacian);  // Jacobi Preconditioned AAR 

    if (rank  ==  0)
        printf("\n");       

    Deallocate_memory(&aar);   ///< De-allocate memory.

    t_end = MPI_Wtime(); 
    if (rank  ==  0)
        {printf("Total wall time   = %.4f seconds. \n", t_end-t_begin); 
    char* c_time_str; 
    time_t current_time = time(NULL); 
    c_time_str = ctime(&current_time);   
    printf("Ending time: %s", c_time_str); 
    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n"); 
}
MPI_Barrier(MPI_COMM_WORLD);    

MPI_Finalize(); 
return 0; 
} 