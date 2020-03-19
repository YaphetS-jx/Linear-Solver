/**
 * @file    system.h
 * @brief   This file declares the functions for the linear system, residual 
 *          function and precondition function.
 *
 * @author  Xin Jing  < xjing30@gatech.edu>
 *          Phanish Suryanarayana  < phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */


#ifndef SYSTEM_H
#define SYSTEM_H

#include  <math.h>
#include  <time.h>   // CLOCK
#include  <stdio.h>
#include  <stdlib.h> 
#include  <string.h>
#include  <mpi.h>
#include  <assert.h>

#define M_PI 3.14159265358979323846
#define min(a, b) ((a) < (b) ? (a) : (b))

typedef struct {
    int ereg_s[3][26],ereg_e[3][26]; // 6 faces, 12 eges, 8 corners so 26 regions disjointly make the total communication region of proc domain. 3 denotes x,y,z dirs
    int **eout_s,**eout_e,**ein_s,**ein_e,**stencil_sign,**edge_ind,*displs_send,*ncounts_send,*displs_recv,*ncounts_recv;
}DS_LapInd;

typedef struct {
    int nproc;                     /// <  Total number of processors
    int nprocx,nprocy,nprocz;      /// <  Number of processors in each direction of domain
    int pnode_s[3];                /// <  Processor domain start nodes' indices w.r.t main (starts from 0)
    int pnode_e[3];                /// <  Processor domain end nodes' indices w.r.t main (starts from 0)
    int np_x,np_y,np_z;            /// <  Number of finite difference nodes in each direction of processor domain
    int FDn;                       /// <  Half-the order of finite difference
    int n_int[3];                  /// <  Number of finite difference intervals in each direction of main domain
    double* coeff_lap;             /// <  Finite difference coefficients for Laplacian
    double ***rhs;                 /// <  Right hand side of the linear equation
    double *rhs_v;                 /// <  Right hand side of the linear equation
    double ***res;                 /// <  residual of linear equation
    DS_LapInd LapInd;              /// <  Object for Laplacian communication information
    double solver_tol;             /// <  Convergence tolerance for AAR solver
    int solver_maxiter;            /// <  Maximum number of iterations allowed in the AAR solver
    int *neighs_lap;               /// <  Array of neighboring processor ranks in the Laplacian stencil width from current processor
    double omega;                  /// <  Richardson relaxation parameter
    double beta;                   /// <  Anderson relaxation parameter
    int m;                         /// <  Number of iterates in Anderson mixing = m+1
    int p;                         /// <  AAR parameter. Anderson update done every p iteration of AAR solver
    int non_blocking;              /// <  Option that indicates using non-blocking version of MPI command. 1=TRUE or 0=FALSE (for MPI collectives)
    double ***phi;                 /// <  Unknown variable of the linear equation
    double *phi_v;                 /// <  Unknown variable of the linear equation
    MPI_Comm comm_laplacian;       /// <  Communicator topology for Laplacian
}DS;

void CheckInputs(DS* pAAR,int argc, char ** argv); 
void Initialize(DS* pAAR); 
void Read_input(DS* pAAR); 
void Processor_domain(DS* pAAR);
void Comm_topologies(DS* pAAR);
void Laplacian_Comm_Indices(DS* pAAR); 
void EdgeIndicesForPoisson(DS* pAAR, int **eout_s,int **eout_e, int **ein_s,int **ein_e, int ereg_s[3][26],
    int ereg_e[3][26],int **stencil_sign,int **edge_ind,int *displs_send,int *displs_recv,int *ncounts_send,int *ncounts_recv);
void Deallocate_memory(DS* pAAR); 
void PoissonResidual(DS *pAAR, double *phi_v, double *res, int np, int FDn, MPI_Comm comm_dist_graph_cart);
void convert_to_vector(double ***vector_3d, double *vector, int np_x, int np_y, int np_z, int dis);
void convert_to_vector3d(double ***vector_3d, double *vector, int np_x, int np_y, int np_z, int dis);

#endif