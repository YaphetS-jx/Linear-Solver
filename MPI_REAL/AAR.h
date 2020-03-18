#ifndef AAR_H
#define AAR_H

#include "system.h"

void AAR(DS_AAR* pAAR, 
    void (*PoissonResidual)(DS_AAR*, double*, double*, int, int, MPI_Comm),
    double *x, double *rhs, double omega_aar, double beta_aar, int m_aar, int p_aar, int max_iter, double tol, 
    int FDn, int np_x, int np_y, int np_z, MPI_Comm comm_dist_graph_cart);

// void convert_to_vector(double ***vector_3d, double *vector, int np_x, int np_y, int np_z, int FDn);

void AXPY(double *z, double *x, double *y, double alpha, int np_x, int np_y, int np_z);

void vec_dot_product(double *s, double *x, double *y, int np_x, int np_y, int np_z);

void AXPYMvPs_3d(double ***z, double ***x, double ***y, double *v, double alpha, double s, int np_x, int np_y, int np_z, int dis);

#endif