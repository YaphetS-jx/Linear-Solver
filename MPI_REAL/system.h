#ifndef SYSTEM_H
#define SYSTEM_H

#include <math.h>
#include <time.h>   // CLOCK
#include <stdio.h>
#include <stdlib.h> 
#include <string.h>
#include <mpi.h>
#include <assert.h>

#define SIGN(a,b) ((b) >= 0.0 ? fabs(a) : -fabs(a))
#define M_PI 3.14159265358979323846
#define max(a, b) ((a) > (b) ? (a) : (b))
#define min(a, b) ((a) < (b) ? (a) : (b))

typedef struct {
    int ereg_s[3][26],ereg_e[3][26]; // 6 faces, 12 eges, 8 corners so 26 regions disjointly make the total communication region of proc domain. 3 denotes x,y,z dirs
    int **eout_s,**eout_e,**ein_s,**ein_e,**stencil_sign,**edge_ind,*displs_send,*ncounts_send,*displs_recv,*ncounts_recv;
}DS_LapInd;

typedef struct {
    // string input_file;             ///< Input file name
    int nproc;                     ///< Total number of processors
    int nprocx,nprocy,nprocz;      ///< Number of processors in each direction of domain
    int pnode_s[3];                ///< Processor domain start nodes' indices w.r.t main (starts from 0)
    int pnode_e[3];                ///< Processor domain end nodes' indices w.r.t main (starts from 0)
    int np_x,np_y,np_z;            ///< Number of finite difference nodes in each direction of processor domain
    int FDn;                       ///< Half-the order of finite difference
    int n_int[3];                  ///< Number of finite difference intervals in each direction of main domain
    double* coeff_lap;             ///< Finite difference coefficients for Laplacian
    double ***rhs;                 ///< Right hand side of the linear equation
    DS_LapInd LapInd;              ///< Object for Laplacian communication information
    double solver_tol;             ///< Convergence tolerance for AAR solver
    int solver_maxiter;            ///< Maximum number of iterations allowed in the AAR solver
    int *neighs_lap;               ///< Array of neighboring processor ranks in the Laplacian stencil width from current processor
    double ***phi_guess;           ///< Initial guess for AAR solver
    double omega_aar;              ///< Richardson relaxation parameter
    double beta_aar;               ///< Anderson relaxation parameter
    int m_aar;                     ///< Number of iterates in Anderson mixing = m_aar+1
    int p_aar;                     ///< AAR parameter. Anderson update done every p_aar iteration of AAR solver
    int non_blocking;              ///< Option that indicates using non-blocking version of MPI command. 1=TRUE or 0=FALSE (for MPI collectives)
    double ***phi;                 ///< Unknown variable of the linear equation
    MPI_Comm comm_laplacian;       ///< Communicator topology for Laplacian
}DS_AAR;

void CheckInputs(DS_AAR* pAAR,int argc, char ** argv); 
void Initialize(DS_AAR* pAAR); 
void Read_input(DS_AAR* pAAR); 
void Processor_domain(DS_AAR* pAAR);
void Comm_topologies(DS_AAR* pAAR);
void Vector2Norm(double* Vec, int len, double* ResVal); 
void Laplacian_Comm_Indices(DS_AAR* pAAR); 
void EdgeIndicesForPoisson(DS_AAR* pAAR, int **eout_s,int **eout_e, int **ein_s,int **ein_e, int ereg_s[3][26],int ereg_e[3][26],int **stencil_sign,int **edge_ind,int *displs_send,int *displs_recv,int *ncounts_send,int *ncounts_recv);
void PoissonResidual(DS_AAR* pAAR,double ***phi_old,double ***phi_res,int iter,MPI_Comm comm_dist_graph_cart, int **eout_s,int **eout_e, int **ein_s,int **ein_e, int ereg_s[3][26],int ereg_e[3][26],int **stencil_sign,int **edge_ind,int *displs_send,int *displs_recv,int *ncounts_send,int *ncounts_recv); 
double pythag(double a, double b); 
void SingularValueDecomp(double **a,int m,int n, double *w, double **v);
void PseudoInverseTimesVec(double *Ac,double *b,double *x,int m); 
void AndersonExtrapolation(double **DX, double **DF, double *phi_res_vec, double beta_mix, int anderson_history, int N, double *am_vec, double *FtF, double *allredvec, double *Ftf, double *svec); 
void Deallocate_memory(DS_AAR* pAAR); 

#endif