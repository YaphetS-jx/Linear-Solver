#include "system.h"
#include "AAR.h"

int main(int argc, char ** argv) {
    DS_AAR aar={};  

    int rank,psize;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&psize);
    aar.nproc = psize; // no. of processors

    double t_begin,t_end;
    t_begin = MPI_Wtime();

    CheckInputs(&aar,argc,argv);
    Initialize(&aar); 

    t_end = MPI_Wtime();
    if (rank==0)
        printf("Time spent in initialization = %.4f seconds. \n",t_end-t_begin);
    
    MPI_Barrier(MPI_COMM_WORLD);      

    AAR(&aar); // Jacobi Preconditioned AAR 

    if (rank==0)
        printf("\n");      

    Deallocate_memory(&aar);  ///< De-allocate memory.

    t_end = MPI_Wtime();
    if (rank==0)
        {printf("Total wall time   = %.4f seconds. \n",t_end-t_begin);
    char* c_time_str;
    time_t current_time=time(NULL);
    c_time_str=ctime(&current_time);  
    printf("Ending time: %s",c_time_str);
    printf("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% \n");
}
MPI_Barrier(MPI_COMM_WORLD);   

MPI_Finalize();
return 0;
} 