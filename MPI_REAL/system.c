/**
 * @file    system.c
 * @brief   This file contains functions for linear system, residual 
 *          function and precondition function.
 *
 * @author  Xin Jing  < xjing30@gatech.edu>
 *          Phanish Suryanarayana  < phanish.suryanarayana@ce.gatech.edu>
 * 
 * Copyright (c) 2020 Material Physics & Mechanics Group at Georgia Tech.
 */


#include "system.h"
#include "tools.h"

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// Setup & Initialize
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

void CheckInputs(DS* pAAR, int argc, char ** argv)  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    char temp; 

    if (rank  ==  0) {
        printf("*************************************************************************** \n"); 
        printf("                     AAR, PGR and PL2R linear solver\n"); 
        printf("*************************************************************************** \n"); 
        char* c_time_str; 
        time_t current_time = time(NULL); 
        c_time_str = ctime(&current_time);   
        printf("Starting time: %s \n", c_time_str); 
    }

    double np_temp; 
    // check if the number of processor is a cubic number
    np_temp = pow(pAAR->nproc, 1.0/3.0); 
    if (abs(pow(round(np_temp), 3)-pAAR->nproc)>1e-14) {
        if (rank  ==  0)
            printf("Assigned number of processors:%u, is not a perfect cube. Exiting. \n", pAAR->nproc); 
        MPI_Barrier(MPI_COMM_WORLD); 
        exit(0); 
    } else {
        pAAR->nprocx = round(np_temp); pAAR->nprocy = round(np_temp); pAAR->nprocz = round(np_temp);       
    }

    if (argc !=  6) {
        if (rank  ==  0) printf("Wrong inputs"); 
        exit(-1); 
    } else {
        pAAR->solver_tol = atof(argv[1]); 
        pAAR->m = atof(argv[2]); 
        pAAR->p = atof(argv[3]); 
        pAAR->omega = atof(argv[4]); 
        pAAR->beta = atof(argv[5]); 
    }
}


void Initialize(DS* pAAR) {
    int i, j, k;
    int rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   

    /// Read from an input file, store the data in data structure  and write the data to the output file. 
    Read_input(pAAR);    

    /// Initialize quantities.
    Processor_domain(pAAR); 

    int coords[3];
    Comm_topologies(pAAR, coords);   
    Laplacian_Comm_Indices(pAAR);  // compute communication indices information for Laplacian     

    // allocate memory to store phi(x) in domain 
    pAAR->phi = (double***) calloc((pAAR->np_z+2*pAAR->FDn), sizeof(double**));  
    assert(pAAR->phi !=  NULL); 

    for (k = 0; k < pAAR->np_z+2*pAAR->FDn; k++) {
        pAAR->phi[k] = (double**) calloc((pAAR->np_y+2*pAAR->FDn), sizeof(double*));  
        assert(pAAR->phi[k] !=  NULL); 

        for (j = 0; j < pAAR->np_y+2*pAAR->FDn; j++) {
            pAAR->phi[k][j] = (double*) calloc((pAAR->np_x+2*pAAR->FDn), sizeof(double));  
            assert(pAAR->phi[k][j] !=  NULL); 
        }
    }

    pAAR->res = (double***) calloc((pAAR->np_z+2*pAAR->FDn), sizeof(double**));  
    assert(pAAR->res !=  NULL); 

    for (k = 0; k < pAAR->np_z+2*pAAR->FDn; k++) {
        pAAR->res[k] = (double**) calloc((pAAR->np_y+2*pAAR->FDn), sizeof(double*));  
        assert(pAAR->res[k] !=  NULL); 

        for (j = 0; j < pAAR->np_y+2*pAAR->FDn; j++) {
            pAAR->res[k][j] = (double*) calloc((pAAR->np_x+2*pAAR->FDn), sizeof(double));  
            assert(pAAR->res[k][j] !=  NULL); 
        }
    }

    pAAR->rhs = (double***) calloc(pAAR->np_z, sizeof(double**));   
    for (k = 0; k < pAAR->np_z; k++) {
        pAAR->rhs[k] = (double**) calloc(pAAR->np_y, sizeof(double*));  // need to de-allocate 
        for (j = 0; j < pAAR->np_y; j++) {
            pAAR->rhs[k][j] = (double*) calloc(pAAR->np_x, sizeof(double));  
        }
    }

    pAAR->phi_v = (double*) calloc(pAAR->np_x * pAAR->np_y * pAAR->np_z, sizeof(double));  
    pAAR->phi_v2 = (double*) calloc(pAAR->np_x * pAAR->np_y * pAAR->np_z, sizeof(double));  
    pAAR->phi_v3 = (double*) calloc(pAAR->np_x * pAAR->np_y * pAAR->np_z, sizeof(double));  
    pAAR->phi_v4 = (double*) calloc(pAAR->np_x * pAAR->np_y * pAAR->np_z, sizeof(double));  
    pAAR->rhs_v = (double*) calloc(pAAR->np_x * pAAR->np_y * pAAR->np_z, sizeof(double));  

    // random RHS ------------------------------- 
    int g_coords[3];
    double rhs_sum = 0, rhs_sum_global; 
    for (k = 0; k < pAAR->np_z; k++) {
        for (j = 0; j < pAAR->np_y; j++) {
            for (i = 0; i < pAAR->np_x; i++) {
                g_coords[0] = coords[0] * pAAR->np_z + k;
                g_coords[1] = coords[1] * pAAR->np_y + j;
                g_coords[2] = coords[2] * pAAR->np_x + i;
                int gidx = g_coords[0] * pAAR->n_int[0] * pAAR->n_int[1] + g_coords[1] * pAAR->n_int[0] + g_coords[2];
                srand(gidx+1); 
                pAAR->rhs[k][j][i] = (double)(rand()) / (double)(RAND_MAX); 
                rhs_sum+= pAAR->rhs[k][j][i]; 
            }
        }
    }
    MPI_Allreduce(&rhs_sum, &rhs_sum_global, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
    rhs_sum_global = rhs_sum_global/(pAAR->n_int[0]*pAAR->n_int[1]*pAAR->n_int[2]); 
    for (k = 0; k < pAAR->np_z; k++) {
        for (j = 0; j < pAAR->np_y; j++) {
            for (i = 0; i < pAAR->np_x; i++) {
                pAAR->rhs[k][j][i] = pAAR->rhs[k][j][i]-rhs_sum_global;  
            }
        }
    }

    // ---------------------------------------------

    srand(rank+pAAR->nproc);  
    for (k = 0; k < pAAR->np_z; k++) {
        for (j = 0; j < pAAR->np_z; j++) {
            for (i = 0; i < pAAR->np_z; i++) {
                pAAR->phi[k+pAAR->FDn][j+pAAR->FDn][i+pAAR->FDn] = 1;
            }
        }
    }

    convert_to_vector(pAAR->phi, pAAR->phi_v, pAAR->np_x, pAAR->np_y, pAAR->np_z, pAAR->FDn);
    convert_to_vector(pAAR->phi, pAAR->phi_v2, pAAR->np_x, pAAR->np_y, pAAR->np_z, pAAR->FDn);
    convert_to_vector(pAAR->phi, pAAR->phi_v3, pAAR->np_x, pAAR->np_y, pAAR->np_z, pAAR->FDn);
    convert_to_vector(pAAR->phi, pAAR->phi_v4, pAAR->np_x, pAAR->np_y, pAAR->np_z, pAAR->FDn);
    convert_to_vector(pAAR->rhs, pAAR->rhs_v, pAAR->np_x, pAAR->np_y, pAAR->np_z, 0);
        
}

void Read_input(DS* pAAR) {
    int rank, p, i; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    double Nr, Dr, val; 

  // ----------------- Bcast ints together and doubles together (two MPI_Bcast 's) --------------------
    int bcast_int[2] = {pAAR->m, pAAR->p}; 
    double bcast_double[3] = {pAAR->solver_tol, pAAR->omega, pAAR->beta}; 
    MPI_Bcast(bcast_int, 2, MPI_INT, 0, MPI_COMM_WORLD); 
    pAAR->m = bcast_int[0] ; 
    pAAR->p = bcast_int[1] ; 
    MPI_Bcast(bcast_double, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);   
    pAAR->solver_tol = bcast_double[0] ; 
    pAAR->omega = bcast_double[1] ; 
    pAAR->beta = bcast_double[2] ; 

    pAAR->non_blocking = 1;  // allows overlap of communication and computation in some cases
    pAAR->solver_maxiter = 1000; 
    pAAR->FDn = 6;  // store half order  
    pAAR->n_int[0] = 48;  pAAR->n_int[1] = 48;  pAAR->n_int[2] = 48; 


    if (rank  ==  0) {
        //printf("FD order    : %u \n", 2*pAAR->FDn); 
        printf("solver_tol  : %g \n", pAAR->solver_tol); 
        printf("omega   : %.2f \n", pAAR->omega); 
        printf("beta    : %.2f \n", pAAR->beta); 
        printf("m       : %d \n", pAAR->m);  
        printf("p       : %d \n", pAAR->p); 
        printf("*************************************************************************** \n"); 
        printf("nprocx = %u, nprocy = %u, nprocz = %u. \n", pAAR->nprocx, pAAR->nprocy, pAAR->nprocz); 
    }

// Compute Finite Difference coefficients for Laplacian (Del^2)
    pAAR->coeff_lap = (double*) calloc((pAAR->FDn+1), sizeof(double)); 
    pAAR->coeff_lap[0] = 0; 
    for (p = 1;  p <= pAAR->FDn;  p++)
        pAAR->coeff_lap[0]+=  -(2.0/(p*p)); 

    for (p = 1; p <= pAAR->FDn; p++) {
        Nr = 1; Dr = 1; 
        for (i = pAAR->FDn-p+1; i <= pAAR->FDn;  i++)
            Nr*= i; 
        for (i = pAAR->FDn+1; i <= pAAR->FDn+p;  i++)
            Dr*= i; 
        val = Nr/Dr; 
        pAAR->coeff_lap[p] = (2*pow(-1, p+1)*val/(p*p));  
    }

}

// function to compute end nodes of processor domain
void Processor_domain(DS* pAAR) {
    int rank, count, countx, county, countz; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    
    // Cubical domain assumption
    if (pAAR->n_int[0]!= pAAR->n_int[1] || pAAR->n_int[2]!= pAAR->n_int[1] || pAAR->n_int[0]!= pAAR->n_int[2] ) {
        printf("Error:Domain is not cubical. Current code can only perform the calculations for a cubical domain. \n"); 
        exit(0); 
    }
    // All equivalent procs assumption
    if ((pAAR->n_int[0] % pAAR->nprocx)!= 0 || (pAAR->n_int[1] % pAAR->nprocy)!= 0 || (pAAR->n_int[2] % pAAR->nprocz)!= 0) {
        printf("Error: Cannot divide the nodes equally among processors. Current code needs equal nodes in all procs. \n"); 
        exit(0); 
    }
    
    // number of nodes in each direction of processor domain (different for the end processor in that direction)
    int ndpx, ndpy, ndpz, ndpx_curr, ndpy_curr, ndpz_curr; 
    ndpx = round((double)pAAR->n_int[0]/pAAR->nprocx); 
    ndpy = round((double)pAAR->n_int[1]/pAAR->nprocy); 
    ndpz = round((double)pAAR->n_int[2]/pAAR->nprocz); 
    
    // Based on the current processor's rank, compute the processor domain end nodes, ordering the processors as z, y, x
    count = 0; 
    for (countx = 0; countx < pAAR->nprocx; countx++) { // loop over all processors count
        for (county = 0; county < pAAR->nprocy; county++){
            for (countz = 0; countz < pAAR->nprocz; countz++) {
                if (rank  ==  count) { // current processor        
                    // no. of nodes in each direction of current processor domain
                    ndpx_curr = ndpx; 
                    ndpy_curr = ndpy; 
                    ndpz_curr = ndpz; 

                    // Start and End node indices of processor domain, w.r.t main domain. Assuming main domain indexing starts from zero. 
                    pAAR->pnode_s[0] = countx*ndpx;  pAAR->pnode_e[0] = pAAR->pnode_s[0]+ndpx_curr-1; 
                    pAAR->pnode_s[1] = county*ndpy;  pAAR->pnode_e[1] = pAAR->pnode_s[1]+ndpy_curr-1; 
                    pAAR->pnode_s[2] = countz*ndpz;  pAAR->pnode_e[2] = pAAR->pnode_s[2]+ndpz_curr-1; 

                    pAAR->np_x = pAAR->pnode_e[0] - pAAR->pnode_s[0] + 1;  // no. of nodes in x-direction of processor domain
                    pAAR->np_y = pAAR->pnode_e[1] - pAAR->pnode_s[1] + 1;  // no. of nodes in y-direction of processor domain
                    pAAR->np_z = pAAR->pnode_e[2] - pAAR->pnode_s[2] + 1;  // no. of nodes in z-direction of processor domain
                }
                count = count+1; 
            }
        }
    } // end for over all processors count
}


// function to create communicator topologies
void Comm_topologies(DS* pAAR, int *pcoords) {
    int rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    int reorder = 0, j; 
    int pdims[3] = {pAAR->nprocz, pAAR->nprocy, pAAR->nprocx};  // no. of processors in each direction --> order is z, y, x
    int periodicity[3] = {1, 1, 1};  // periodic BC's in each direction
    MPI_Comm topocomm;  // declare a new communicator to give topology attribute
    MPI_Cart_create(MPI_COMM_WORLD, 3, pdims, periodicity, reorder, &topocomm);  // create Cartesian topology

    MPI_Cart_coords(topocomm, rank, 3, pcoords);  // local coordinate indices of processors
    int rank_chk; 
    MPI_Cart_rank(topocomm, pcoords, &rank_chk);  // proc rank corresponding to coords

    //////////////////////////////////////////////////////
    //// Cartesian Topology///////////////////////////////
    // This topology has atleast 6 nearest neighbors for communication in 3D
    //////////////////////////////////////////////////////

    int nneigh = 6*ceil((double)(pAAR->FDn-(1e-12))/pAAR->np_x);  // total number of neighbors
    int *neighs, count = 0; 
    neighs = (int*) calloc(nneigh, sizeof(int));  
    pAAR->neighs_lap = (int*) calloc(nneigh, sizeof(int));  

    int proc_l, proc_r, proc_u, proc_d, proc_f, proc_b;  // procs on left, right, up, down, front, back of the current proc
    for (j = 0; j < ceil((double)(pAAR->FDn-(1e-12))/pAAR->np_x); j++) { // no. of layers of nearest neighbors required for FD stencil
        MPI_Cart_shift(topocomm, 0, j+1, &proc_l, &proc_r);   // x-direction
        MPI_Cart_shift(topocomm, 1, j+1, &proc_b, &proc_f);   // y-direction
        MPI_Cart_shift(topocomm, 2, j+1, &proc_d, &proc_u);   // z-direction

        neighs[count] = proc_l;  neighs[count+1] = proc_r; 
        neighs[count+2] = proc_b;  neighs[count+3] = proc_f; 
        neighs[count+4] = proc_d;  neighs[count+5] = proc_u; 

        pAAR->neighs_lap[count] = proc_l;  pAAR->neighs_lap[count+1] = proc_r; 
        pAAR->neighs_lap[count+2] = proc_b;  pAAR->neighs_lap[count+3] = proc_f; 
        pAAR->neighs_lap[count+4] = proc_d;  pAAR->neighs_lap[count+5] = proc_u; 

        count = count+5 + 1; 
    }

    MPI_Comm comm_dist_graph_cart;  // communicator with cartesian distributed graph topology
    MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD, nneigh, neighs, (int *)MPI_UNWEIGHTED, nneigh, neighs, (int *)MPI_UNWEIGHTED, MPI_INFO_NULL, reorder, &comm_dist_graph_cart);  // creates a distributed graph topology (adjacent, cartesian)
    pAAR->comm_laplacian = comm_dist_graph_cart; 

    // de-allocate neighs
    free(neighs); 
}

// compute preconditioned relative residual

void PoissonResidual(DS *pAAR, double *phi_v, double *Ax, int np, int FDn, MPI_Comm comm_dist_graph_cart) {
    int rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    int **eout_s, **eout_e, **ein_s, **ein_e, **stencil_sign, **edge_ind, *displs_send, *displs_recv, *ncounts_send, *ncounts_recv;
    eout_s = pAAR->LapInd.eout_s; eout_e = pAAR->LapInd.eout_e; ein_s = pAAR->LapInd.ein_s; ein_e = pAAR->LapInd.ein_e; stencil_sign = pAAR->LapInd.stencil_sign; 
    edge_ind = pAAR->LapInd.edge_ind; displs_send = pAAR->LapInd.displs_send; displs_recv = pAAR->LapInd.displs_recv; ncounts_send = pAAR->LapInd.ncounts_send; 
    ncounts_recv = pAAR->LapInd.ncounts_recv;

    // STEPS TO FOLLOW:
    // Create a contiguous vector of the stencil points to send to neigh procs and receive in a similar vector the ghost point data for the edge stencil points
    // Before the communication (for non-blocking code) finishes, compute lap_phi for nodes that are inside proc domain, which are stencil width away from domain boundary
    // After the communication is done, using the ghost point data, evaluate lap_phi for the points in the stencil region
    // Find phi_new from lap_phi and phi_old

    // Assemble an array of phi at stencil points for communication.
    // This array should go over the list of neigh procs and accumulate the points, in the order x, y, z with left and rights in each direction
    int np_edge = FDn*np*np;  // no. of nodes in the stencil communication region across each face of processor domain. Six such regions communicate.
    int i, j, k, a, edge_count, neigh_count, proc_dir, proc_lr; 
    double *phi_edge_in, *phi_edge_out;  // linear array to store input and output communication data/buffer to and from the processor
    phi_edge_in = (double*) calloc(6*np_edge, sizeof(double));  
    phi_edge_out = (double*) calloc(6*np_edge, sizeof(double));   
    double lap_phi_k, TEMP_TOL = 1e-12; 
    int neigh_level, nneigh = ceil((double)(FDn-TEMP_TOL)/np); 
    double ***rhs = pAAR->rhs;
    for (k = 0; k < np+2*FDn; k++) 
        for (j = 0; j < np+2*FDn; j++) 
            for (i = 0; i < np+2*FDn; i++) {
                pAAR->phi[k][j][i] = 0;
                pAAR->res[k][j][i] = 0;
            }

    convert_to_vector3d(pAAR->phi, phi_v, np, np, np, FDn);
    double ***phi_old = pAAR->phi;
    double ***phi_res = pAAR->res;

    // Setup the outgoing array phi_edge_out with the phi values from edges of the proc domain. Order: First loop over procs x, y, z direc and then for each proc, x, y, z over nodes in the edge region
    edge_count = 0; 
    neigh_count = 0; 
    for (neigh_level = 0; neigh_level < nneigh; neigh_level++) { // loop over layers of nearest neighbors
        for (proc_dir = 0; proc_dir < 3; proc_dir++) { // loop over directions (loop over neighbor procs) ---> 0, 1, 2 = x, y, z direcs
            for (proc_lr = 0; proc_lr < 2; proc_lr++) { // loop over left & right (loop over neighbor procs) ----> 0, 1 = left, right
                // Now read phi values in the edge region into the array
                for (k = eout_s[2][neigh_count]; k <= eout_e[2][neigh_count]; k++) { // z-direction of edge region
                    for (j = eout_s[1][neigh_count]; j <= eout_e[1][neigh_count]; j++) { // y-direction of edge region
                        for (i = eout_s[0][neigh_count]; i <= eout_e[0][neigh_count]; i++) { // x-direction of edge region
                            phi_edge_out[edge_count] = phi_old[k+FDn][j+FDn][i+FDn]; 
                            edge_count = edge_count + 1; 
                        }
                    }
                }
                neigh_count = neigh_count + 1; 
            }
        }
    }

    // Using Neighborhood collective for local communication between processors
    MPI_Request request; 
    if (pAAR->non_blocking  ==  0)
        MPI_Neighbor_alltoallv(phi_edge_out, ncounts_send, displs_send, MPI_DOUBLE, phi_edge_in, ncounts_recv, displs_recv, MPI_DOUBLE, comm_dist_graph_cart); 
    else       
        MPI_Ineighbor_alltoallv(phi_edge_out, ncounts_send, displs_send, MPI_DOUBLE, phi_edge_in, ncounts_recv, displs_recv, MPI_DOUBLE, comm_dist_graph_cart, &request);  // non-blocking


    if (np > 2*FDn) {
        // Overlapping Computation with Communication when using non-blocking routines
        // Now find lap_phi and update phi using Jacobi iteration. Do this on interior and edge domains of proc domain separately so that we can use non-blocking communication
        // Update phi on the proc domain assuming it is zero outside the proc domain (will correct for this after stencil communication)
        for (k = 0+FDn; k <= np-1+FDn; k++) { // z-direction of interior region
            for (j = 0+FDn; j <= np-1+FDn; j++) { // y-direction of interior region
                for (i = 0+FDn; i <= np-1+FDn; i++) { // x-direction of interior region
                    //i, j, k indices are w.r.t proc+FDn domain
                    lap_phi_k = phi_old[k][j][i]*3*pAAR->coeff_lap[0]; 
                    for (a = 1; a <= FDn; a++) {
                        lap_phi_k +=  (phi_old[k][j][i-a] + phi_old[k][j][i+a] + phi_old[k][j-a][i] + phi_old[k][j+a][i] + phi_old[k-a][j][i] + phi_old[k+a][j][i])*pAAR->coeff_lap[a];                  
                    }
                    phi_res[k][j][i] = -(((4*M_PI)*(rhs[k-FDn][j-FDn][i-FDn]) + lap_phi_k)/(3*pAAR->coeff_lap[0]));  // Jacobi update for nodes in interior of proc domain
                } 
            }
        }
    }

    // Make sure communication has finished and then proceed to next task. This is to be done when using non-blocking routine.
    if (pAAR->non_blocking  ==  1)
        MPI_Wait(&request, MPI_STATUS_IGNORE); 

    // Store the incoming buffer data from phi_edge_in into the outer stencil regions of phi_old array
    neigh_count = 0; 
    edge_count = 0; 
    for (neigh_level = 0; neigh_level < nneigh; neigh_level++) { // loop over layers of nearest neighbors
        for (proc_dir = 0; proc_dir < 3; proc_dir++) { // loop over directions (loop over neighbor procs) ---> 0, 1, 2 = x, y, z direcs
            for (proc_lr = 0; proc_lr < 2; proc_lr++) { // loop over left & right (loop over neighbor procs) ----> 0, 1 = left, right
              // Now read phi values in the edge region into the array
                for (k = ein_s[2][neigh_count]; k <= ein_e[2][neigh_count]; k++) { // z-direction of edge region
                    for (j = ein_s[1][neigh_count]; j <= ein_e[1][neigh_count]; j++) { // y-direction of edge region
                        for (i = ein_s[0][neigh_count]; i <= ein_e[0][neigh_count]; i++) { // x-direction of edge region
                            phi_old[k][j][i] = phi_edge_in[edge_count]; 
                            edge_count = edge_count + 1; 
                        }
                    }
                }
                neigh_count = neigh_count + 1; 
            }
        }
    }


    if (np  <=  2*FDn) { // Assuming that we are not using non-blocking
        for (k = 0+FDn; k <= np-1+FDn; k++) { // z-direction of interior region
            for (j = 0+FDn; j <= np-1+FDn; j++) { // y-direction of interior region
                for (i = 0+FDn; i <= np-1+FDn; i++) { // x-direction of interior region
                    //i, j, k indices are w.r.t proc+FDn domain
                    lap_phi_k = phi_old[k][j][i]*3*pAAR->coeff_lap[0]; 
                    for (a = 1; a <= FDn; a++) {
                        lap_phi_k +=  (phi_old[k][j][i-a] + phi_old[k][j][i+a] + phi_old[k][j-a][i] + phi_old[k][j+a][i] + phi_old[k-a][j][i] + phi_old[k+a][j][i])*pAAR->coeff_lap[a];                  
                    }
                        phi_res[k][j][i] = -(((4*M_PI)*(rhs[k-FDn][j-FDn][i-FDn]) + lap_phi_k)/(3*pAAR->coeff_lap[0]));  // Jacobi update for nodes in interior of proc domain
                }
            }
        }
    }

    if (np > 2*FDn) {
        int temp = 0; 
        for (neigh_count = 0; neigh_count < 26; neigh_count++) {
            for (k = pAAR->LapInd.ereg_s[2][neigh_count]; k <= pAAR->LapInd.ereg_e[2][neigh_count]; k++) { // z-direction of edge region
                for (j = pAAR->LapInd.ereg_s[1][neigh_count]; j <= pAAR->LapInd.ereg_e[1][neigh_count]; j++) { // y-direction of edge region
                    for (i = pAAR->LapInd.ereg_s[0][neigh_count]; i <= pAAR->LapInd.ereg_e[0][neigh_count]; i++) { // x-direction of edge region
                        //i, j, k indexes are w.r.t proc domain, but phi_old and new arrays are on proc+FDn domain
                            lap_phi_k = 0.0;  
                            
                        for (a = (edge_ind[0][temp]+1); a <= FDn; a++)  // x-direction
                            lap_phi_k +=  (phi_old[k+FDn][j+FDn][i+FDn-stencil_sign[0][temp]*a])*pAAR->coeff_lap[a];  // part of the other half stencil inside domain
                        
                        for (a = (edge_ind[1][temp]+1); a <= FDn; a++)  // y-direction
                            lap_phi_k +=  (phi_old[k+FDn][j+FDn-stencil_sign[1][temp]*a][i+FDn])*pAAR->coeff_lap[a];  // part of the other half stencil inside domain

                        for (a = (edge_ind[2][temp]+1); a <= FDn; a++)  // z-direction
                            lap_phi_k +=  (phi_old[k+FDn-stencil_sign[2][temp]*a][j+FDn][i+FDn])*pAAR->coeff_lap[a];  // part of the other half stencil inside domain
                        
                        phi_res[k+FDn][j+FDn][i+FDn] +=  -((lap_phi_k)/(3*pAAR->coeff_lap[0]));  // Jacobi update for nodes in edge region of proc domain
                        temp+= 1;  
                    }
                }
            }
        }
    }

    for (k = 0; k < np; k++) 
        for (j = 0; j < np; j++) 
            for (i = 0; i < np; i++) {
                phi_res[k+FDn][j+FDn][i+FDn] *= -(3*pAAR->coeff_lap[0]/4/M_PI);
                phi_res[k+FDn][j+FDn][i+FDn] = rhs[k][j][i] - phi_res[k+FDn][j+FDn][i+FDn];
            }

    convert_to_vector(phi_res, Ax, np, np, np, FDn);

    free(phi_edge_in);  //de-allocate memory
    free(phi_edge_out);  //de-allocate memory
}

void Precondition(double diag, double *res, int Np)
{
    int i;
    for (i = 0; i < Np; i ++)
        res[i] /= diag;
}

///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// Communication functions
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////

// function to compute comm indices and arrays
void Laplacian_Comm_Indices(DS* pAAR) {
    double TEMP_TOL = 1e-12; 
  // compute edge index information to compute residual to be used in the solver
    int nneigh = 6*ceil((double)pAAR->FDn-TEMP_TOL/pAAR->np_x), k, num; 
  // allocate memory to store arrays for MPI communication
    pAAR->LapInd.displs_send = (int*) calloc(nneigh, sizeof(int)); 
    pAAR->LapInd.ncounts_send = (int*) calloc(nneigh, sizeof(int)); 
    pAAR->LapInd.displs_recv = (int*) calloc(nneigh, sizeof(int)); 
    pAAR->LapInd.ncounts_recv = (int*) calloc(nneigh, sizeof(int)); 
    pAAR->LapInd.eout_s = (int**) calloc(3, sizeof(int*));  
    pAAR->LapInd.eout_e = (int**) calloc(3, sizeof(int*));  
    pAAR->LapInd.ein_s = (int**) calloc(3, sizeof(int*)); ;  
    pAAR->LapInd.ein_e = (int**) calloc(3, sizeof(int*));  
    pAAR->LapInd.stencil_sign = (int**) calloc(3, sizeof(int*));  
    pAAR->LapInd.edge_ind = (int**) calloc(3, sizeof(int*));  
    for (k = 0; k < 3; k++) {
        pAAR->LapInd.eout_s[k] = (int*) calloc(nneigh, sizeof(int));  
        pAAR->LapInd.eout_e[k] = (int*) calloc(nneigh, sizeof(int));  
        pAAR->LapInd.ein_s[k] = (int*) calloc(nneigh, sizeof(int));  
        pAAR->LapInd.ein_e[k] = (int*) calloc(nneigh, sizeof(int));  
        num = pAAR->FDn*pAAR->FDn*pAAR->FDn*8 + pAAR->FDn*pAAR->FDn*(pAAR->np_x-2*pAAR->FDn)*12 + pAAR->FDn*(pAAR->np_x-2*pAAR->FDn)*(pAAR->np_x-2*pAAR->FDn)*6; 
        pAAR->LapInd.stencil_sign[k] = (int*) calloc(num, sizeof(int));  
        pAAR->LapInd.edge_ind[k] = (int*) calloc(num, sizeof(int)); 
    }

    EdgeIndicesForPoisson(pAAR, pAAR->LapInd.eout_s, pAAR->LapInd.eout_e, pAAR->LapInd.ein_s, pAAR->LapInd.ein_e, pAAR->LapInd.ereg_s, pAAR->LapInd.ereg_e, pAAR->LapInd.stencil_sign, pAAR->LapInd.edge_ind, pAAR->LapInd.displs_send, pAAR->LapInd.displs_recv, pAAR->LapInd.ncounts_send, pAAR->LapInd.ncounts_recv);  
}


// function to compute edge region indices
void EdgeIndicesForPoisson(DS* pAAR, int **eout_s, int **eout_e, int **ein_s, int **ein_e, int ereg_s[3][26], int ereg_e[3][26], 
                            int **stencil_sign, int **edge_ind, int *displs_send, int *displs_recv, int *ncounts_send, int *ncounts_recv) {
    // eout: start(s) and end(e) node indices w.r.t proc domain of the outgoing edges
    // ein : start(s) and end(e) node indices w.r.t proc domain of the incoming edges
    // ereg: start(s) and end(e) node indices w.r.t proc domain of the edge regions (for partial stencil updates)

    int rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    double TEMP_TOL = 1e-12; 
    int i, j, k, ii, jj, kk, edge_count = 0, neigh_count, proc_dir, proc_lr; 
    int np = pAAR->np_x;  // no. of nodes in each direction of processor domain
    int edge_s[3], edge_e[3];  // start and end nodes of the edge region in 3 directions, indicies w.r.t local processor domain
    int neigh_level, nneigh = ceil((double)(pAAR->FDn-TEMP_TOL)/pAAR->np_x); 

    // Setup the outgoing array phi_edge_out with the phi values from edges of the proc domain. Order: First loop over procs x, y, z direc and then for each proc, x, y, z over nodes in the edge region
    edge_count = 0; 
    displs_send[0] = 0; 
    for (neigh_level = 0; neigh_level < nneigh; neigh_level++) { // loop over layers of nearest neighbors
        for (proc_dir = 0; proc_dir < 3; proc_dir++) { // loop over directions (loop over neighbor procs) ---> 0, 1, 2 = x, y, z direcs
            for (proc_lr = 0; proc_lr < 2; proc_lr++) { // loop over left & right (loop over neighbor procs) ----> 0, 1 = left, right
                // store neighbor information
                ncounts_send[edge_count] = pAAR->np_x*pAAR->np_y*pAAR->np_z;  // no. of nodes to be communicated to this neighbor
                if (edge_count>0)
                    displs_send[edge_count] = displs_send[edge_count-1]+ncounts_send[edge_count-1];  // relative displacement of index in out going array 

                // for each neigh proc, compute start and end nodes of the communication region of size FDn x np x np
                edge_s[0] = 0; edge_s[1] = 0; edge_s[2] = 0;  // initialize all start nodes to start node of proc domain i.e. zero
                edge_e[0] = np-1; edge_e[1] = np-1; edge_e[2] = np-1;  // initialize all end nodes to end node of proc domain i.e. np-1

                if (neigh_level+1  ==  nneigh) { // for outermost neigh layer, need only part of the domains           
                    ncounts_send[edge_count] = pAAR->np_x*pAAR->np_x*(pAAR->FDn-floor((double)(pAAR->FDn-TEMP_TOL)/pAAR->np_x)*pAAR->np_x);  // update no. of nodes (only partial required)
                    if (edge_count>0)
                        displs_send[edge_count] = displs_send[edge_count-1]+ncounts_send[edge_count-1];  // relative displacement of index in out going array 

                    if (proc_lr  ==  1) // for right neigh proc
                        edge_s[proc_dir] = np-(pAAR->FDn-floor((double)(pAAR->FDn-TEMP_TOL)/pAAR->np_x)*pAAR->np_x)+1-1;  // update the start node for the edge region in the direction of proc dir which is only FDn width
                    if (proc_lr  ==  0) // for left neigh proc
                        edge_e[proc_dir] = (pAAR->FDn-floor((double)(pAAR->FDn-TEMP_TOL)/pAAR->np_x)*pAAR->np_x)-1;  // update the end node for the edge region in the direction of proc dir which is only FDn width
                }
    
                eout_s[0][edge_count] = edge_s[0];  eout_e[0][edge_count] = edge_e[0]; 
                eout_s[1][edge_count] = edge_s[1];  eout_e[1][edge_count] = edge_e[1]; 
                eout_s[2][edge_count] = edge_s[2];  eout_e[2][edge_count] = edge_e[2]; 

                edge_count +=  1; 
            }
        }
    }

    int **ein_s_temp, **ein_e_temp; 
    ein_s_temp = (int**) calloc(3, sizeof(int*));  
    ein_e_temp = (int**) calloc(3, sizeof(int*));  
    for (k = 0; k < 3; k++) {
        ein_s_temp[k] = (int*) calloc(6*nneigh, sizeof(int));  
        ein_e_temp[k] = (int*) calloc(6*nneigh, sizeof(int));        
    }

    int *mark_off, *send_ind_arry;  
    mark_off = (int*) calloc(6*nneigh, sizeof(int)); 
    send_ind_arry = (int*) calloc(6*nneigh, sizeof(int)); 
    int send_ind, ccnt, rep_dist, ctr, rep_dist_old = 0, rep_ind, rep_ind_old = 0; 

    // Store the incoming buffer data from phi_edge_in into the outer stencil regions of phi_old array
    edge_count = 0; 
    for (neigh_level = 0; neigh_level < nneigh; neigh_level++) { // loop over layers of nearest neighbors
        for (proc_dir = 0; proc_dir < 3; proc_dir++) { // loop over directions (loop over neighbor procs) ---> 0, 1, 2 = x, y, z direcs
            for (proc_lr = 0; proc_lr < 2; proc_lr++) { // loop over left & right (loop over neighbor procs) ----> 0, 1 = left, right
                ccnt = 0; ctr = 0; 
                for (kk = 0; kk < nneigh; kk++) { // neigh_level
                    for (jj = 0; jj < 3; jj++) { // proc_dir
                        for (ii = 0; ii < 2; ii++) { // proc_lr
                            if (pAAR->neighs_lap[ccnt]  ==  pAAR->neighs_lap[edge_count] && mark_off[ccnt]  ==  0) {
                                rep_dist = kk*6 + jj*2 + (1-ii);  // [ii, jj, kk] relative current proc would have been kk*6 + 2*jj + ii
                                rep_ind = ccnt; 

                                if (ctr  ==  0) {
                                    rep_dist_old = rep_dist; 
                                    rep_ind_old = rep_ind; 
                                }

                                if (ctr>0) {
                                    if (rep_dist < rep_dist_old) {
                                        rep_dist_old = rep_dist; 
                                        rep_ind_old = rep_ind; 
                                    }
                                }
                                ctr += 1;                 
                            }         
                            ccnt += 1; 
                        }
                    }
                }

                send_ind = rep_ind_old; 
                mark_off[send_ind] = 1;  // min distance proc marked off. Will not be considered next time.
                ncounts_recv[edge_count] = ncounts_send[send_ind]; 
                send_ind_arry[edge_count] = send_ind;  // stores the proc index to which buffer will be sent from current proc in the order of "count". So when count = 0, buffer of size ncounts_recv[0] will first be received from neigh proc send_ind_arry[0] whose corresponding proc rank is pAAR->neighs_lap[send_ind_arry[0]].

                edge_count +=  1; 
            }
        }
    }

    free(mark_off); 

    // find displs_recv
    edge_count = 0;  displs_recv[0] = 0; 
    for (neigh_level = 0; neigh_level < nneigh; neigh_level++) { // loop over layers of nearest neighbors
        for (proc_dir = 0; proc_dir < 3; proc_dir++) { // loop over directions (loop over neighbor procs) ---> 0, 1, 2 = x, y, z direcs
            for (proc_lr = 0; proc_lr < 2; proc_lr++) { // loop over left & right (loop over neighbor procs) ----> 0, 1 = left, right
                if (edge_count>0)
                    displs_recv[edge_count] = displs_recv[edge_count-1]+ncounts_recv[edge_count-1];  // relative displacement of index in out going array 
                edge_count +=  1; 
            }
        }
    }

    // Store the incoming buffer data from phi_edge_in into the outer stencil regions of phi_old array
    edge_count = 0; 
    for (neigh_level = 0; neigh_level < nneigh; neigh_level++) { // loop over layers of nearest neighbors
        for (proc_dir = 0; proc_dir < 3; proc_dir++) { // loop over directions (loop over neighbor procs) ---> 0, 1, 2 = x, y, z direcs
            for (proc_lr = 0; proc_lr < 2; proc_lr++) { // loop over left & right (loop over neighbor procs) ----> 0, 1 = left, right

                // for each neigh proc, compute start and end nodes of the communication region of size FDn x np x np
                edge_s[0] = pAAR->FDn-1+1; edge_s[1] = pAAR->FDn-1+1; edge_s[2] = pAAR->FDn-1+1;  // initialize all start nodes to start node of proc domain i.e. FDn-1+1 (w.r.t proc+FDn)
                edge_e[0] = np-1+pAAR->FDn; edge_e[1] = np-1+pAAR->FDn; edge_e[2] = np-1+pAAR->FDn;  // initialize all end nodes to end node of proc domain i.e. np-1+FDn

                if (neigh_level+1  ==  nneigh) { // for outermost neigh layer, need only part of the domain
                    if ((proc_lr)  ==  1) { // for right neigh proc
                        edge_s[proc_dir] = np+pAAR->FDn+1-1+floor((double)(pAAR->FDn-TEMP_TOL)/pAAR->np_x)*pAAR->np_x;  // update the start node for the edge region in the direction of proc dir which is only FDn width
                        edge_e[proc_dir] = np+2*pAAR->FDn-1; 
                    }
                    if ((proc_lr)  ==  0) { // for left neigh proc
                        edge_e[proc_dir] = pAAR->FDn-1-floor((double)(pAAR->FDn-TEMP_TOL)/pAAR->np_x)*pAAR->np_x;  // update the end node for the edge region in the direction of proc dir which is only FDn width
                        edge_s[proc_dir] = 0; 
                    }
                } else { // inner neigh layers
                    if ((proc_lr)  ==  1) { // for right neigh proc
                      edge_s[proc_dir] = edge_s[proc_dir]+(neigh_level+1)*(np);  // update the start node for the edge region in the direction of proc dir which is only FDn width
                      edge_e[proc_dir] = edge_e[proc_dir]+(neigh_level+1)*(np); 
                    }
                    if ((proc_lr)  ==  0) { // for left neigh proc
                        edge_e[proc_dir] = edge_e[proc_dir]-(neigh_level+1)*(np);  // update the end node for the edge region in the direction of proc dir which is only FDn width
                        edge_s[proc_dir] = edge_s[proc_dir]-(neigh_level+1)*(np); 
                    }
                }

                ein_s_temp[0][edge_count] = edge_s[0];  ein_e_temp[0][edge_count] = edge_e[0]; 
                ein_s_temp[1][edge_count] = edge_s[1];  ein_e_temp[1][edge_count] = edge_e[1]; 
                ein_s_temp[2][edge_count] = edge_s[2];  ein_e_temp[2][edge_count] = edge_e[2]; 
                edge_count +=  1; 
            }
        }
    }


    // Now loop again and find the proper order of the indices for the recv buffer based on send_ind_arry that was used to set up ncounts_recv
    // compute the start/end indices of the regions for incoming buffers. Indices w.r.t proc+FDn domain
    edge_count = 0;  
    for (neigh_level = 0; neigh_level < nneigh; neigh_level++) { // loop over layers of nearest neighbors
        for (proc_dir = 0; proc_dir < 3; proc_dir++) { // loop over directions (loop over neighbor procs) ---> 0, 1, 2 = x, y, z direcs
            for (proc_lr = 0; proc_lr < 2; proc_lr++) { // loop over left & right (loop over neighbor procs) ----> 0, 1 = left, right
                ein_s[0][edge_count] = ein_s_temp[0][send_ind_arry[edge_count]];  ein_e[0][edge_count] = ein_e_temp[0][send_ind_arry[edge_count]]; 
                ein_s[1][edge_count] = ein_s_temp[1][send_ind_arry[edge_count]];  ein_e[1][edge_count] = ein_e_temp[1][send_ind_arry[edge_count]]; 
                ein_s[2][edge_count] = ein_s_temp[2][send_ind_arry[edge_count]];  ein_e[2][edge_count] = ein_e_temp[2][send_ind_arry[edge_count]]; 
                
                edge_count +=  1; 
            }
        }
    }

    if (pAAR->np_x > 2*pAAR->FDn) {
        // Indices for boundary regions to do remaining partial stencil correction -- 26 boundary regions = (6 faces, 12 edges, 8 corners)

        edge_count = 0; 
        // loop over 6 faces = 3x2
        for (proc_dir = 0; proc_dir < 3; proc_dir++) { // loop over directions (loop over neighbor procs) ---> 0, 1, 2 = x, y, z direcs
            for (proc_lr = 0; proc_lr < 2; proc_lr++) { // loop over left & right (loop over neighbor procs) ----> 0, 1 = left, right
                edge_s[0] = 0+pAAR->FDn; edge_s[1] = 0+pAAR->FDn; edge_s[2] = 0+pAAR->FDn;  // initialize all start nodes to start node of proc domain-FDn i.e. zero+FDn
                edge_e[0] = np-1-pAAR->FDn; edge_e[1] = np-1-pAAR->FDn; edge_e[2] = np-1-pAAR->FDn;  // initialize all end nodes to end node of proc domain-FDn i.e. np-1-FDn
                if (proc_lr  ==  1) { // for right neigh proc
                    edge_s[proc_dir] = np-pAAR->FDn+1-1;  // update the start node for the edge region in the direction of proc dir which is only FDn width
                    edge_e[proc_dir] = np-1; 
                }
                
                if (proc_lr  ==  0) { // for left neigh proc
                    edge_e[proc_dir] = pAAR->FDn-1;  // update the end node for the edge region in the direction of proc dir which is only FDn width
                    edge_s[proc_dir] = 0; 
                }

                ereg_s[0][edge_count] = edge_s[0];  ereg_e[0][edge_count] = edge_e[0]; 
                ereg_s[1][edge_count] = edge_s[1];  ereg_e[1][edge_count] = edge_e[1]; 
                ereg_s[2][edge_count] = edge_s[2];  ereg_e[2][edge_count] = edge_e[2]; 

                edge_count +=  1; 
            }
        }

        // loop over 12 edges = 3x2x2 --> 3 long edge dirs (x, y, z) then 4 regs for each long edge dir --> these 4 regs split as left/right and up/down in order of x, y, z
        // long edge along x-direction
        ereg_s[0][6] = pAAR->FDn;  ereg_e[0][6] = np-1-pAAR->FDn;   ereg_s[1][6] = 0;            ereg_e[1][6] = pAAR->FDn-1;  ereg_s[2][6] = 0;            ereg_e[2][6] = pAAR->FDn-1; 
        ereg_s[0][7] = pAAR->FDn;  ereg_e[0][7] = np-1-pAAR->FDn;   ereg_s[1][7] = np-pAAR->FDn;  ereg_e[1][7] = np-1;            ereg_s[2][7] = 0;            ereg_e[2][7] = pAAR->FDn-1; 
        ereg_s[0][8] = pAAR->FDn;  ereg_e[0][8] = np-1-pAAR->FDn;   ereg_s[1][8] = 0;            ereg_e[1][8] = pAAR->FDn-1;  ereg_s[2][8] = np-pAAR->FDn;  ereg_e[2][8] = np-1; 
        ereg_s[0][9] = pAAR->FDn;  ereg_e[0][9] = np-1-pAAR->FDn;   ereg_s[1][9] = np-pAAR->FDn;  ereg_e[1][9] = np-1;            ereg_s[2][9] = np-pAAR->FDn;  ereg_e[2][9] = np-1; 
        // long edge along y-direction
        ereg_s[1][10] = pAAR->FDn;  ereg_e[1][10] = np-1-pAAR->FDn;     ereg_s[0][10] = 0;           ereg_e[0][10] = pAAR->FDn-1;     ereg_s[2][10] = 0;           ereg_e[2][10] = pAAR->FDn-1; 
        ereg_s[1][11] = pAAR->FDn;  ereg_e[1][11] = np-1-pAAR->FDn;     ereg_s[0][11] = np-pAAR->FDn; ereg_e[0][11] = np-1;           ereg_s[2][11] = 0;           ereg_e[2][11] = pAAR->FDn-1; 
        ereg_s[1][12] = pAAR->FDn;  ereg_e[1][12] = np-1-pAAR->FDn;     ereg_s[0][12] = 0;           ereg_e[0][12] = pAAR->FDn-1;     ereg_s[2][12] = np-pAAR->FDn; ereg_e[2][12] = np-1; 
        ereg_s[1][13] = pAAR->FDn;  ereg_e[1][13] = np-1-pAAR->FDn;     ereg_s[0][13] = np-pAAR->FDn; ereg_e[0][13] = np-1;           ereg_s[2][13] = np-pAAR->FDn; ereg_e[2][13] = np-1; 
        // long edge along z-direction
        ereg_s[2][14] = pAAR->FDn;  ereg_e[2][14] = np-1-pAAR->FDn;     ereg_s[1][14] = 0;            ereg_e[1][14] = pAAR->FDn-1;    ereg_s[0][14] = 0;            ereg_e[0][14] = pAAR->FDn-1; 
        ereg_s[2][15] = pAAR->FDn;  ereg_e[2][15] = np-1-pAAR->FDn;     ereg_s[1][15] = np-pAAR->FDn;  ereg_e[1][15] = np-1;          ereg_s[0][15] = 0;            ereg_e[0][15] = pAAR->FDn-1; 
        ereg_s[2][16] = pAAR->FDn;  ereg_e[2][16] = np-1-pAAR->FDn;     ereg_s[1][16] = 0;            ereg_e[1][16] = pAAR->FDn-1;    ereg_s[0][16] = np-pAAR->FDn;  ereg_e[0][16] = np-1; 
        ereg_s[2][17] = pAAR->FDn;  ereg_e[2][17] = np-1-pAAR->FDn;     ereg_s[1][17] = np-pAAR->FDn;  ereg_e[1][17] = np-1;          ereg_s[0][17] = np-pAAR->FDn;  ereg_e[0][17] = np-1; 


        // loop over 8 corners
        ereg_s[0][18] = 0;  ereg_e[0][18] = pAAR->FDn-1;    ereg_s[1][18] = 0;            ereg_e[1][18] = pAAR->FDn-1;    ereg_s[2][18] = 0;            ereg_e[2][18] = pAAR->FDn-1; 
        ereg_s[0][19] = np-pAAR->FDn;  ereg_e[0][19] = np-1;    ereg_s[1][19] = 0;            ereg_e[1][19] = pAAR->FDn-1;    ereg_s[2][19] = 0;            ereg_e[2][19] = pAAR->FDn-1; 
        ereg_s[0][20] = 0;  ereg_e[0][20] = pAAR->FDn-1;    ereg_s[1][20] = np-pAAR->FDn;  ereg_e[1][20] = np-1;          ereg_s[2][20] = 0;            ereg_e[2][20] = pAAR->FDn-1; 
        ereg_s[0][21] = np-pAAR->FDn;  ereg_e[0][21] = np-1;    ereg_s[1][21] = np-pAAR->FDn;  ereg_e[1][21] = np-1;          ereg_s[2][21] = 0;            ereg_e[2][21] = pAAR->FDn-1; 

        ereg_s[0][22] = 0;  ereg_e[0][22] = pAAR->FDn-1;    ereg_s[1][22] = 0;            ereg_e[1][22] = pAAR->FDn-1;    ereg_s[2][22] = np-pAAR->FDn;  ereg_e[2][22] = np-1; 
        ereg_s[0][23] = np-pAAR->FDn;  ereg_e[0][23] = np-1;    ereg_s[1][23] = 0;            ereg_e[1][23] = pAAR->FDn-1;    ereg_s[2][23] = np-pAAR->FDn;  ereg_e[2][23] = np-1; 
        ereg_s[0][24] = 0;  ereg_e[0][24] = pAAR->FDn-1;    ereg_s[1][24] = np-pAAR->FDn;  ereg_e[1][24] = np-1;          ereg_s[2][24] = np-pAAR->FDn;  ereg_e[2][24] = np-1; 
        ereg_s[0][25] = np-pAAR->FDn;  ereg_e[0][25] = np-1;    ereg_s[1][25] = np-pAAR->FDn;  ereg_e[1][25] = np-1;          ereg_s[2][25] = np-pAAR->FDn;  ereg_e[2][25] = np-1; 

        int temp = 0; 
        for (neigh_count = 0; neigh_count < 26; neigh_count++) {
            for (k = ereg_s[2][neigh_count]; k <= ereg_e[2][neigh_count]; k++) { // z-direction of edge region
                for (j = ereg_s[1][neigh_count]; j <= ereg_e[1][neigh_count]; j++) { // y-direction of edge region
                    for (i = ereg_s[0][neigh_count]; i <= ereg_e[0][neigh_count]; i++) { // x-direction of edge region
                        //i, j, k indices are w.r.t proc domain, but phi_old and new arrays are on proc+FDn domain
                        if (i-0  <=  np-1-i) { // index i closer to left edge
                            stencil_sign[0][temp] = 1;  // full half stencil can be updated on right side
                            edge_ind[0][temp] = min(pAAR->FDn, i-0); 
                        } else { // index i closer to right edge
                         
                            stencil_sign[0][temp] = -1;  // full half stencil can be updated on left side
                            edge_ind[0][temp] = min(pAAR->FDn, np-1-i); 
                        }

                        if (j-0  <=  np-1-j) { // index i closer to left edge
                            stencil_sign[1][temp] = 1;  // full half stencil can be updated on right side
                            edge_ind[1][temp] = min(pAAR->FDn, j-0); 
                        } else { // index i closer to right edge
                            stencil_sign[1][temp] = -1;  // full half stencil can be updated on left side
                            edge_ind[1][temp] = min(pAAR->FDn, np-1-j); 
                        }

                        if (k-0  <=  np-1-k) { // index i closer to left edge
                            stencil_sign[2][temp] = 1;  // full half stencil can be updated on right side
                            edge_ind[2][temp] = min(pAAR->FDn, k-0); 
                        } else { // index i closer to right edge
                            stencil_sign[2][temp] = -1;  // full half stencil can be updated on left side
                            edge_ind[2][temp] = min(pAAR->FDn, np-1-k); 
                        }
                        temp+= 1; 
                    }
                }
            }
        }
    } // end if condition (pAAR->np_x > 2*pAAR->FDn)

  // de-allocate memory
    for (k = 0; k < 3; k++) {
        free(ein_s_temp[k]); 
        free(ein_e_temp[k]); 
    }

    free(ein_s_temp); 
    free(ein_e_temp); 
    free(send_ind_arry); 
}



///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// Deallocate memory
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
void Deallocate_memory(DS* pAAR)  {
    int j, k; 
    int rank; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

    free(pAAR->coeff_lap); 
    
    // de-allocate rhs
    for (k = 0; k < pAAR->np_z; k++) {
        for (j = 0; j < pAAR->np_y; j++) {
            free(pAAR->rhs[k][j]); 
        }
        free(pAAR->rhs[k]); 
    }
    free(pAAR->rhs); 
    free(pAAR->rhs_v); 

    // de-allocate phi
    for (k = 0; k < pAAR->np_z+2*pAAR->FDn; k++) {
        for (j = 0; j < pAAR->np_y+2*pAAR->FDn; j++) {
            free(pAAR->phi[k][j]); 
            free(pAAR->res[k][j]);
        }
        free(pAAR->phi[k]); 
        free(pAAR->res[k]); 
    }
    free(pAAR->phi); 
    free(pAAR->res); 
    free(pAAR->phi_v); 
    free(pAAR->phi_v2); 
    free(pAAR->phi_v3); 
    free(pAAR->phi_v4); 
    
  // de-allocate comm topologies
    free(pAAR->neighs_lap); 

  // de-allocate Laplacian comm 
    for (k = 0; k < 3; k++) {
        free(pAAR->LapInd.eout_s[k]); 
        free(pAAR->LapInd.eout_e[k]); 
        free(pAAR->LapInd.ein_s[k]); 
        free(pAAR->LapInd.ein_e[k]); 
        free(pAAR->LapInd.stencil_sign[k]); 
        free(pAAR->LapInd.edge_ind[k]); 
    }

    free(pAAR->LapInd.eout_s); 
    free(pAAR->LapInd.eout_e); 
    free(pAAR->LapInd.ein_s); 
    free(pAAR->LapInd.ein_e); 
    free(pAAR->LapInd.stencil_sign); 
    free(pAAR->LapInd.edge_ind); 

    free(pAAR->LapInd.displs_send); 
    free(pAAR->LapInd.ncounts_send); 
    free(pAAR->LapInd.displs_recv); 
    free(pAAR->LapInd.ncounts_recv);   

}


void convert_to_vector(double ***vector_3d, double *vector, int np_x, int np_y, int np_z, int dis){
    int i, j, k, ctr = 0;
    for (k = 0; k < np_z; k++) 
        for (j = 0; j < np_y; j++) 
            for (i = 0; i < np_x; i++)             
                vector[ctr++] = vector_3d[k+dis][j+dis][i+dis]; 
}

void convert_to_vector3d(double ***vector_3d, double *vector, int np_x, int np_y, int np_z, int dis){
    int i, j, k, ctr = 0;
    for (k = 0; k < np_z; k++) 
        for (j = 0; j < np_y; j++) 
            for (i = 0; i < np_x; i++)             
                vector_3d[k+dis][j+dis][i+dis] = vector[ctr++]; 
}
