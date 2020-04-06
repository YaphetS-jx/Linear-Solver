/* For parallel 3-d Poisson system*/
#include  <math.h>
#include  <stdio.h>
#include  <stdlib.h> 
#include  <string.h>
#include  <mpi.h>
#include  <assert.h>

// #define M_PI 3.14159265358979323846
#define Min(a, b) (a < b ? a : b)

typedef struct {
    int FDn;
    int ssize[3];
    int psize[3];

    int np[3];
    int coords[3];
    int rem[3];

    int *send_neighs;
    int *rec_neighs;
    int *send_counts;
    int *rec_counts;
    int send_layers[6];
    int rec_layers[6];
    int sources;
    int destinations;

    double *phi;
    double *rhs;
    double *Lap_phi;

    double *coeff_lap;
    MPI_Comm comm_laplacian; 
    MPI_Comm cart;
}POISSON;

void Processor_Domain(int ssize[3], int psize[3], int np[3], int coords[3], int rem[3], MPI_Comm comm, MPI_Comm *cart);

void Comm_topologies(int FDn, int psize[3], int coords[3], int rem[3], int np[3], MPI_Comm cart, MPI_Comm *comm_laplacian,
    int *send_neighs, int *rec_neighs, int *send_counts, int *rec_counts, int send_layers[6], int rec_layers[6], int *sources, int *destinations);

void Max_layer(int ssize[3], int np[3], int FDn, int *max_layer);

void Initialize(POISSON *system, int max_layer);

void Deallocate_memory(POISSON *system);

void Vec_copy(int *a, int *b, int n);

void Lap_coefficient(double *coeff_lap, int FDn);

void Lap_Vec_mult(POISSON *system, double a, double *phi, double *Lap_phi, MPI_Comm comm_laplacian);

void Vec_2Norm(double *vec, int length, double *Norm, MPI_Comm comm);

void Find_size_dir(int rem, int coords, int psize, int *small, int *large);

void Get_block_origin_global_coords(int coords[3], int rem[3], int psize[3], int g_origin[3], MPI_Comm *cart);

int main(int argc, char ** argv) 
{
    MPI_Init(&argc, &argv); 
    int rank, np_all, i, j, k, max_layer = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &np_all); 

    POISSON *system = malloc(sizeof(POISSON));
    assert(system != NULL);

    system->ssize[0] = 6;
    system->ssize[1] = 6;
    system->ssize[2] = 6;
    system->FDn = 3;

    // Compute range of each process
    Processor_Domain(system->ssize, system->psize, system->np, system->coords, system->rem, MPI_COMM_WORLD, &system->cart); 
    
    Max_layer(system->ssize, system->np, system->FDn, &max_layer);

    Initialize(system, max_layer);
    
    Comm_topologies(system->FDn, system->psize, system->coords, system->rem, system->np, system->cart, &system->comm_laplacian,
        system->send_neighs, system->rec_neighs, system->send_counts, system->rec_counts, system->send_layers, system->rec_layers, &system->sources, &system->destinations);

    Lap_Vec_mult(system, -1.0/(4*M_PI), system->phi, system->Lap_phi, system->comm_laplacian);

    double norm;
    Vec_2Norm(system->phi, system->psize[0] * system->psize[1] * system->psize[2], &norm, system->comm_laplacian);
    if(rank == 0) printf("x_norm: %f\n", norm);

    Vec_2Norm(system->Lap_phi, system->psize[0] * system->psize[1] * system->psize[2], &norm, system->comm_laplacian);
    if(rank == 0) printf("ax_norm: %f\n", norm);
    
    Deallocate_memory(system);
    
    MPI_Finalize(); 
    return 0;
}

void Lap_Vec_mult(POISSON *system, double a, double *phi, double *Lap_phi, MPI_Comm comm_laplacian)
{
#define phi(i, j, k) phi[(i) + (j) * psize[0] + (k) * psize[0] * psize[1]]
#define Lap_phi(i, j, k) Lap_phi[(i) + (j) * psize[0] + (k) * psize[0] * psize[1]]
#define x_ghosted(i, j, k) x_ghosted[(i) + (j) * (psize[0] + 2 * FDn) + (k) * (psize[0] + 2 * FDn) * (psize[1] + 2 * FDn)]

    int i, j, k, n_in = 0, n_out = 0, sources, destinations, dir, layer;
    int count, ghosted_size, order, FDn, sum, size;
    int *sdispls, *rdispls, scale[3], *scounts, *rcounts;
    int psize[3] = {system->psize[0], system->psize[1], system->psize[2]};
    double *x_in, *x_out, *x_ghosted, *Lap_weights;
    //////////////////////////////////////////////////////////////////////

    sources = system->sources;
    destinations = system->destinations;

    FDn = system->FDn;

    MPI_Comm_size(MPI_COMM_WORLD, &size); 

    ghosted_size = (psize[0] + 2 * FDn) * (psize[1] + 2 * FDn) * (psize[2] + 2 * FDn);
    scounts = (int*) calloc(destinations, sizeof(int));
    rcounts = (int*) calloc(sources,      sizeof(int));
    sdispls = (int*) calloc(destinations, sizeof(int));
    rdispls = (int*) calloc(sources,      sizeof(int));
    x_ghosted   = (double*) calloc(ghosted_size, sizeof(double));
    Lap_weights = (double*) calloc((FDn + 1),    sizeof(double));
    assert(scounts != NULL && rcounts != NULL && sdispls != NULL && rdispls != NULL && x_ghosted != NULL && Lap_weights);

    // update coefficient
    for (i = 0; i < FDn + 1; i++){
        Lap_weights[i] = a * system->coeff_lap[i];
    }

    if (size > 1) 
    {
        scale[0] = psize[1] * psize[2];
        scale[1] = psize[0] * psize[2];
        scale[2] = psize[0] * psize[1];

        k = 0;
        for (i = 0; i < 6; i++)
            for (j = 0; j < system->send_layers[i]; j++){
                scounts[k] = system->send_counts[k] * scale[i / 2];
                n_out += scounts[k];

                if (k > 0)
                    sdispls[k] = sdispls[k - 1] + scounts[k - 1];
                k++;
            }

        k = 0;
        for (i = 0; i < 6; i++)
            for (j = 0; j < system->rec_layers[i]; j++){
                rcounts[k] = system->rec_counts[k] * scale[i / 2];
                n_in += rcounts[k];
                
                if (k > 0)
                    rdispls[k] = rdispls[k - 1] + rcounts[k - 1];
                k++;
            }

        x_in  = (double*) calloc(n_in,  sizeof(double));
        x_out = (double*) calloc(n_out, sizeof(double));
        assert(x_in != NULL && x_out != NULL);


        // assembly x_out
        count = 0;
        order = 0;
        for (dir = 0; dir < 6; dir++){
            for (layer = 0; layer < system->send_layers[dir]; layer++){

                int start[3] = {0, 0 , 0};
                int end[3]   = {psize[0], psize[1], psize[2]};

                if (dir % 2 == 0){
                    end[dir / 2] = system->send_counts[order++];
                } else {
                    start[dir / 2] = psize[dir / 2] - system->send_counts[order++];
                }

                for (k = start[2]; k < end[2]; k++)
                    for (j = start[1]; j < end[1]; j++)
                        for (i = start[0]; i < end[0]; i++)
                            x_out[count++] = phi(i, j, k);
            }   
        }

        MPI_Neighbor_alltoallv(x_out, scounts, sdispls, MPI_DOUBLE, x_in, rcounts, rdispls, MPI_DOUBLE, comm_laplacian); 


        // assembly x_in        
        count = 0;
        order = 0;
        for (dir = 0; dir < 6; dir ++){
            sum = 0;
            for (layer = 0; layer < system->rec_layers[dir]; layer ++){

                int start[3] = {FDn, FDn, FDn};
                int end[3]   = {FDn + psize[0], FDn + psize[1], FDn + psize[2]};

                sum += system->rec_counts[order];

                if (dir % 2 == 1){
                    start[dir / 2] = FDn - sum;
                    end[dir / 2]   = FDn - (sum - system->rec_counts[order++]); 
                } else {
                    start[dir / 2] = FDn + psize[dir / 2] + (sum - system->rec_counts[order++]) ;
                    end[dir / 2]   = FDn + psize[dir / 2] + sum; 
                }

                for (k = start[2]; k < end[2]; k++)
                    for (j = start[1]; j < end[1]; j++)
                        for (i = start[0]; i < end[0]; i++)
                            x_ghosted(i, j, k) = x_in[count++];
            }
        }

        for (k = 0; k < psize[2]; k++)
            for (j = 0; j < psize[1]; j++)
                for (i = 0; i < psize[0]; i++)
                    x_ghosted(i + FDn, j + FDn, k + FDn) = phi(i, j, k);


        // update phi
        for (k = FDn; k < FDn + psize[2]; k++)
            for (j = FDn; j < FDn + psize[1]; j++)
                for (i = FDn; i < FDn + psize[0]; i++){

                    Lap_phi(i - FDn, j - FDn, k - FDn) = x_ghosted(i, j, k) * 3 * Lap_weights[0];

                    for (order = 1; order < FDn + 1; order ++){

                        Lap_phi(i - FDn, j - FDn, k - FDn) += (x_ghosted(i - order, j, k) + x_ghosted(i + order, j, k) 
                                                             + x_ghosted(i, j - order, k) + x_ghosted(i, j + order, k) 
                                                             + x_ghosted(i, j, k - order) + x_ghosted(i, j, k + order)) * Lap_weights[order];
                    }
                }

        free(x_in);
        free(x_out);

    } else {
        // Sequential algorithm 
        for (k = 0; k < psize[2]; k++)
            for (j = 0; j < psize[1]; j++)
                for (i = 0; i < psize[0]; i++){

                    Lap_phi(i, j, k) = phi(i, j, k) * 3 * Lap_weights[0];

                    for (order = 1; order < FDn + 1; order ++){

                        Lap_phi(i, j, k) += (phi((i - order + order * psize[0]) % psize[0], j, k) + phi((i + order) % psize[0], j, k) 
                                           + phi(i, (j - order + order * psize[1]) % psize[1], k) + phi(i, (j + order) % psize[1], k) 
                                           + phi(i, j, (k - order + order * psize[2]) % psize[2]) + phi(i, j, (k + order) % psize[2])) * Lap_weights[order];
                    }
                }

    }

#undef phi
#undef x_ghosted
#undef Lap_phi
    free(sdispls);
    free(rdispls);
    free(scounts);
    free(rcounts);
    free(Lap_weights);
}

void Comm_topologies(int FDn, int psize[3], int coords[3], int rem[3], int np[3], MPI_Comm cart, MPI_Comm *comm_laplacian,
    int *send_neighs, int *rec_neighs, int *send_counts, int *rec_counts, int send_layers[6], int rec_layers[6], int *sources, int *destinations)
{
#define send_layers(i,j) send_layers[(i)*2+(j)]
#define rec_layers(i,j)   rec_layers[(i)*2+(j)]

    int i, j, k, sum, sign, neigh_size, large, small, reorder = 0;
    int neigh_coords[3], scale[3];
    //////////////////////////////////////////////////////////////////////

    // First sending elements to others. Left, right, back, front, down, up 
    k = 0;
    for (i = 0; i < 3; i++){
        sign = -1;
        Find_size_dir(rem[i], coords[i], psize[i], &small, &large);

        for (j = 0; j < 2; j++){
            Vec_copy(neigh_coords, coords, 3);
            sum  = FDn;
            send_layers(i, j) = 0;
            neigh_size = 0;

            while (sum > 0){
                neigh_coords[i] = (neigh_coords[i] + sign + np[i]) % np[i];
                
                send_layers(i,j)++;
                
                MPI_Cart_rank(cart, neigh_coords, send_neighs + k);
                
                *(send_counts + k) = Min(sum, psize[i]);

                if (neigh_coords[i] < rem[i])
                    neigh_size = large;
                else
                    neigh_size = small;

                sum -= neigh_size;
                k++;
            }
            sign *= (-1);
        }
    }


    // Then receiving elements from others. Right, left, front, back, up, down.
    k = 0;
    for (i = 0; i < 3; i++){
        sign = +1;
        Find_size_dir(rem[i], coords[i], psize[i], &small, &large);

        for (j = 0; j < 2; j++){
            Vec_copy(neigh_coords, coords, 3);
            sum  = FDn;
            rec_layers(i, j) = 0;

            while (sum > 0){
                neigh_coords[i] = (neigh_coords[i] + sign + np[i]) % np[i];

                if (neigh_coords[i] < rem[i])
                    neigh_size = large;
                else
                    neigh_size = small;

                MPI_Cart_rank(cart, neigh_coords, rec_neighs + k);
                
                rec_layers(i,j)++;

                sum -= neigh_size;
                
                *(rec_counts + k) = neigh_size;
                k++;
            }
            *(rec_counts + k - 1) = neigh_size + sum;

            sign *= (-1);
        }
    }

#undef send_layers
#undef rec_layers

    for (i = 0; i < 6; i++){
        *sources += rec_layers[i];
        *destinations += send_layers[i];
    }

    MPI_Dist_graph_create_adjacent(MPI_COMM_WORLD, *sources, rec_neighs, (int *)MPI_UNWEIGHTED, 
        *destinations, send_neighs, (int *)MPI_UNWEIGHTED, MPI_INFO_NULL, reorder, comm_laplacian); 
}


void Processor_Domain(int ssize[3], int psize[3], int np[3], int coords[3], int rem[3], MPI_Comm comm, MPI_Comm *cart)
{
    int i, rank, size, reorder = 0;
    int period[3] = {1, 1, 1}, neigh_coords[3];
    /////////////////////////////////////////////////////

    MPI_Comm_size(comm, &size); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 

    np[0] = 0; np[1] = 0; np[2] = 0;
    MPI_Dims_create(size, 3, np);

    MPI_Cart_create(comm, 3, np, period, reorder, cart);

    MPI_Cart_coords(*cart, rank, 3, coords);

    for (i = 0; i < 3; i++) {
        rem[i] = ssize[i] % np[i];
        if (coords[i] < rem[i]){
            psize[i] = ssize[i] / np[i] + 1;
        }
        else {
            psize[i] = ssize[i] / np[i];
            if (psize[i] == 0){
                printf("Error: System size is too small.\n");
            }

        }
    }
    // MPI_Barrier(MPI_COMM_WORLD); 
    // if (err == -1) MPI_Abort(MPI_COMM_WORLD, -1);
}

void Max_layer(int ssize[3], int np[3], int FDn, int *max_layer)
{
    int i;
    /////////////////////////////////////////////////////

    for (i = 0; i < 3; i++){
        *max_layer += 2 * (FDn / (ssize[i] / np[i]) + 1);
    }
}

void Initialize(POISSON *system, int max_layer)
{
    int i, j, k, g_origin[3], g_ind, no_nodes;
    ////////////////////////////////////////////////////////////

    system->send_neighs = (int*) calloc(max_layer, sizeof(int));
    system->rec_neighs  = (int*) calloc(max_layer, sizeof(int));
    system->send_counts = (int*) calloc(max_layer, sizeof(int));
    system->rec_counts  = (int*) calloc(max_layer, sizeof(int));
    assert(system->send_neighs != NULL && system->rec_neighs != NULL && system->send_counts != NULL && system->rec_counts != NULL);

    system->coeff_lap = (double*) calloc((system->FDn + 1), sizeof(double)); 
    assert(system->coeff_lap != NULL);

    Lap_coefficient(system->coeff_lap, system->FDn);

    no_nodes = system->psize[0] * system->psize[1] * system->psize[2];
    system->phi = (double*) calloc(no_nodes, sizeof(double)); 
    system->rhs = (double*) calloc(no_nodes, sizeof(double)); 
    system->Lap_phi = (double*) calloc(no_nodes, sizeof(double)); 
    assert(system->phi != NULL && system->rhs != NULL && system->Lap_phi != NULL);

    Get_block_origin_global_coords(system->coords, system->rem, system->psize, g_origin, &system->cart);

    for (k = 0; k < system->psize[2]; k ++)    
        for (j = 0; j < system->psize[1]; j ++)    
            for (i = 0; i < system->psize[0]; i ++) {
                g_ind = (g_origin[0] + i) + (g_origin[1] + j) * system->ssize[0] + (g_origin[2] + k) * system->ssize[0] * system->ssize[1];
                // srand(g_ind + 1);
                system->phi[i + j * system->psize[0] + k * system->psize[0] * system->psize[1]] = g_ind * 1.0;
            }

    system->sources = 0;
    system->destinations = 0;
}


void Lap_coefficient(double *coeff_lap, int FDn)
{
    int i, p;
    double  Nr, Dr, val;
    ////////////////////////////////////////////////////

    for (p = 1;  p <= FDn; p++)
        coeff_lap[0] +=  -(2.0 / (p * p)); 

    for (p = 1; p <= FDn; p++) {
        Nr = 1; Dr = 1; 
        for (i = FDn - p + 1; i <= FDn; i++)
            Nr *= i; 
        for (i = FDn + 1; i <= FDn + p; i++)
            Dr *= i; 
        val = Nr / Dr; 
        coeff_lap[p] = (2 * pow(-1, p + 1) * val / (p * p));  
    }
}

void Vec_copy(int *a, int *b, int n)
{
    for (int i = 0; i < n; i++)
        a[i] = b[i];
}


void Deallocate_memory(POISSON *system)
{
    free(system->send_neighs);
    free(system->rec_neighs);
    free(system->send_counts);
    free(system->rec_counts);

    free(system->coeff_lap);
    free(system->phi);
    free(system->rhs);
    free(system->Lap_phi);

    MPI_Comm_free(&system->comm_laplacian);
    MPI_Comm_free(&system->cart);

    free(system);
}

void Vec_2Norm(double *vec, int length, double *Norm, MPI_Comm comm)
{   
    int i;
    double norm = 0;
    /////////////////////////////////////////////////////////

    for (i = 0; i < length; i++){
        norm += vec[i] * vec[i];
    }

    MPI_Allreduce(&norm, Norm, 1, MPI_DOUBLE, MPI_SUM, comm);

    *Norm = sqrt(*Norm);
}

void Get_block_origin_global_coords(int coords[3], int rem[3], int psize[3], int g_origin[3], MPI_Comm *cart)
{
    int i, j, small, large;
    ///////////////////////////////////////////////////////////////

    for (i = 0; i < 3; i++){
        Find_size_dir(rem[i], coords[i], psize[i], &small, &large);
        g_origin[i] = 0;
        j = 0;
        while (j < coords[i]){
            if (j < rem[i])
                g_origin[i] += large;
            else
                g_origin[i] += small;
            j++;
        }
    }
}


void Find_size_dir(int rem, int coords, int psize, int *small, int *large)
{   
    if (coords < rem){
        *large = psize;
        *small = psize - 1;
    } else {
        *large = psize + 1;
        *small = psize;
    }
}
