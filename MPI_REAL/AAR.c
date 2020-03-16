#include "AAR.h"
#include "system.h"

///////////////////////////////////////////////////////////////////////////////
////////////////////////////////// AAR solver /////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
void AAR(DS_AAR* pAAR) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    double omega_aar=pAAR->omega_aar; 
    double beta_aar=pAAR->beta_aar;  
    int m_aar=pAAR->m_aar; 
    int p_aar = pAAR->p_aar; 
    int i,j,k,m;
    double ***phi_new,***phi_old,***phi_res,***phi_temp,*phi_res_vec,*phi_old_vec,*Xold,*Fold,**DX,**DF,*am_vec,t_poiss0,t_poiss1; 
    MPI_Comm comm_dist_graph_cart = pAAR->comm_laplacian; // communicator with cartesian distributed graph topology


    // allocate memory to store phi(x) in domain
    phi_new = (double***)calloc((pAAR->np_z+2*pAAR->FDn) , sizeof(double**));
    phi_old = (double***)calloc((pAAR->np_z+2*pAAR->FDn) , sizeof(double**));
    phi_res = (double***)calloc((pAAR->np_z+2*pAAR->FDn) , sizeof(double**));
    phi_temp = (double***)calloc((pAAR->np_z+2*pAAR->FDn) , sizeof(double**));  
    assert(phi_new != NULL && phi_old != NULL && phi_res != NULL && phi_temp != NULL);

    for (k=0;k<pAAR->np_z+2*pAAR->FDn;k++) {
        phi_new[k] = (double**)calloc((pAAR->np_y+2*pAAR->FDn) , sizeof(double*));
        phi_old[k] = (double**)calloc((pAAR->np_y+2*pAAR->FDn) , sizeof(double*));
        phi_res[k] = (double**)calloc((pAAR->np_y+2*pAAR->FDn) , sizeof(double*));
        phi_temp[k] = (double**)calloc((pAAR->np_y+2*pAAR->FDn) , sizeof(double*));
        assert(phi_new[k] != NULL && phi_old[k] != NULL && phi_res[k] != NULL && phi_temp[k] != NULL);

        for (j=0;j<pAAR->np_y+2*pAAR->FDn;j++) {
            phi_new[k][j] = (double*)calloc((pAAR->np_x+2*pAAR->FDn) , sizeof(double));
            phi_old[k][j] = (double*)calloc((pAAR->np_x+2*pAAR->FDn) , sizeof(double));
            phi_res[k][j] = (double*)calloc((pAAR->np_x+2*pAAR->FDn) , sizeof(double));
            phi_temp[k][j] = (double*)calloc((pAAR->np_x+2*pAAR->FDn) , sizeof(double));
            assert(phi_new[k][j] != NULL && phi_old[k][j] != NULL && phi_res[k][j] != NULL && phi_temp[k][j] != NULL);
        }
    }

    // allocate memory to store phi(x) in domain
    phi_res_vec = (double*)calloc(pAAR->np_z*pAAR->np_y*pAAR->np_x , sizeof(double)); 
    phi_old_vec = (double*)calloc(pAAR->np_z*pAAR->np_y*pAAR->np_x , sizeof(double));
    Xold = (double*)calloc(pAAR->np_z*pAAR->np_y*pAAR->np_x , sizeof(double)); 
    Fold = (double*)calloc(pAAR->np_z*pAAR->np_y*pAAR->np_x , sizeof(double)); 
    am_vec = (double*)calloc(pAAR->np_z*pAAR->np_y*pAAR->np_x , sizeof(double)); 
    assert(phi_res_vec != NULL && phi_old_vec !=NULL && Xold !=NULL && Fold !=NULL && am_vec !=NULL);

    // allocate memory to store DX, DF history matrices
    DX = (double**) calloc(m_aar , sizeof(double*));
    DF = (double**) calloc(m_aar , sizeof(double*));
    assert(DF != NULL && DX !=NULL);

    for (k=0;k<m_aar;k++) {
        DX[k] = (double*)calloc(pAAR->np_z*pAAR->np_y*pAAR->np_x , sizeof(double));
        DF[k] = (double*)calloc(pAAR->np_z*pAAR->np_y*pAAR->np_x , sizeof(double));
        assert(DX[k] != NULL && DF[k] != NULL);
    }

    // Initialize phi_old from phi_guess
    int ctr=0;
    for (k=0;k<pAAR->np_z;k++) {
        for (j=0;j<pAAR->np_z;j++) {
            for (i=0;i<pAAR->np_z;i++) {
                phi_old[k+pAAR->FDn][j+pAAR->FDn][i+pAAR->FDn] = pAAR->phi_guess[k+pAAR->FDn][j+pAAR->FDn][i+pAAR->FDn]; 
                phi_temp[k+pAAR->FDn][j+pAAR->FDn][i+pAAR->FDn] = phi_old[k+pAAR->FDn][j+pAAR->FDn][i+pAAR->FDn];
                ctr+=1;
            }
        }
    }

    // calculate norm of right hand side (required for relative residual)
    double rhs_norm=1.0,*rhs_vec;
    rhs_vec = (double*) calloc(pAAR->np_z*pAAR->np_y*pAAR->np_x , sizeof(double)); 
    assert(rhs_vec != NULL);

    ctr=0;
    for (k=0;k<pAAR->np_z;k++) {
        for (j=0;j<pAAR->np_y;j++) {
            for (i=0;i<pAAR->np_x;i++) {
                rhs_vec[ctr]=pAAR->rhs[k][j][i]; 
                ctr=ctr+1;
            }
        }
    }
    Vector2Norm(rhs_vec,pAAR->np_x*pAAR->np_y*pAAR->np_z,&rhs_norm);
    free(rhs_vec);

    int Np=pAAR->np_z*pAAR->np_y*pAAR->np_x;

    double *FtF,*allredvec; // DF'*DF matrix in column major format
    FtF = (double*) calloc(m_aar*m_aar , sizeof(double));
    allredvec = (double*) calloc(m_aar*m_aar+m_aar+1 , sizeof(double));
    double *Ftf; // DF'*phi_res_vec vector of size m x 1
    Ftf = (double*) calloc(m_aar , sizeof(double));
    double *svec; // vector to store singular values    
    svec = (double*) calloc(m_aar , sizeof(double)); 

    int max_iter=pAAR->solver_maxiter; 
    int iter=1;
    double tol = pAAR->solver_tol;
    double res = tol+1;

    t_poiss0 = MPI_Wtime();
    
    // begin while loop
    while (res>tol && iter <= max_iter) {
        PoissonResidual(pAAR,phi_temp,phi_res,iter,comm_dist_graph_cart,pAAR->LapInd.eout_s,pAAR->LapInd.eout_e, pAAR->LapInd.ein_s,pAAR->LapInd.ein_e, pAAR->LapInd.ereg_s,pAAR->LapInd.ereg_e,pAAR->LapInd.stencil_sign,pAAR->LapInd.edge_ind, pAAR->LapInd.displs_send,pAAR->LapInd.displs_recv,pAAR->LapInd.ncounts_send,pAAR->LapInd.ncounts_recv);

        // -------------- Update phi --------------- //
        ctr=0;
        for (k=0;k<pAAR->np_z;k++) {
            for (j=0;j<pAAR->np_y;j++) {
                for (i=0;i<pAAR->np_x;i++) {             
                    phi_res_vec[ctr]=phi_res[k+pAAR->FDn][j+pAAR->FDn][i+pAAR->FDn];
                    phi_old_vec[ctr]=phi_old[k+pAAR->FDn][j+pAAR->FDn][i+pAAR->FDn];
                    ctr=ctr+1;
                }
            }
        }
        //----------Store Residual & Iterate History----------//
        if(iter>1) {
            m = ((iter-2) % m_aar)+1-1; //-1 because the index starts from 0
            ctr=0;
            for (k=0;k<pAAR->np_z;k++) {
                for (j=0;j<pAAR->np_y;j++) {
                    for (i=0;i<pAAR->np_x;i++) {
                        DX[m][ctr]=phi_old_vec[ctr]-Xold[ctr];
                        DF[m][ctr]=phi_res_vec[ctr]-Fold[ctr];
                        ctr=ctr+1;
                    }
                }
            }
        } // end if
          
        ctr=0;
        for (k=0;k<pAAR->np_z;k++) {
            for (j=0;j<pAAR->np_y;j++) {
                for (i=0;i<pAAR->np_x;i++) {
                    Xold[ctr] = phi_old_vec[ctr];
                    Fold[ctr] = phi_res_vec[ctr];         
                    ctr=ctr+1;
                }
            }
        }

        //----------Anderson update-----------//
        if(iter % p_aar == 0 && iter>1) {
            ctr=0;
            res=0.0;
            for (k=0;k<pAAR->np_z;k++) {
                for (j=0;j<pAAR->np_y;j++) {
                    for (i=0;i<pAAR->np_x;i++) {
                        res = res + phi_res_vec[ctr]*phi_res_vec[ctr];
                        ctr=ctr+1;
                    }
                }
            }
            allredvec[m_aar*m_aar+m_aar]=res;

            AndersonExtrapolation(DX,DF,phi_res_vec,beta_aar,m_aar,Np,am_vec,FtF,allredvec,Ftf,svec);

            res = sqrt(allredvec[m_aar*m_aar+m_aar])/rhs_norm; // relative residual    

            ctr=0;
            for (k=0;k<pAAR->np_z;k++) {
                for (j=0;j<pAAR->np_y;j++) {
                    for (i=0;i<pAAR->np_x;i++) {
                        // phi_k+1 = phi_k + beta*f_k - dp (Anderson update)
                        phi_new[k+pAAR->FDn][j+pAAR->FDn][i+pAAR->FDn]=phi_old[k+pAAR->FDn][j+pAAR->FDn][i+pAAR->FDn]+beta_aar*phi_res[k+pAAR->FDn][j+pAAR->FDn][i+pAAR->FDn]-am_vec[ctr];
                        ctr=ctr+1;
                    }
                }
            }
        } else {
        //----------Richardson update-----------//
        // phi_k+1 = phi_k + omega*f_k
            for (k=0;k<pAAR->np_z;k++) {
                for (j=0;j<pAAR->np_y;j++) {
                    for (i=0;i<pAAR->np_x;i++) { 
                        phi_new[k+pAAR->FDn][j+pAAR->FDn][i+pAAR->FDn]=phi_old[k+pAAR->FDn][j+pAAR->FDn][i+pAAR->FDn]+omega_aar*phi_res[k+pAAR->FDn][j+pAAR->FDn][i+pAAR->FDn];
                    }
                }
            }
        }

        // set phi_old = phi_new
        for (k=0;k<pAAR->np_z;k++) {
            for (j=0;j<pAAR->np_y;j++) {
                for (i=0;i<pAAR->np_x;i++) {
                    if(res<=tol || iter==max_iter) {
                        pAAR->phi[k+pAAR->FDn][j+pAAR->FDn][i+pAAR->FDn]=phi_old[k+pAAR->FDn][j+pAAR->FDn][i+pAAR->FDn]; // store the solution of Poisson's equation
                    }
                    phi_old[k+pAAR->FDn][j+pAAR->FDn][i+pAAR->FDn]=phi_new[k+pAAR->FDn][j+pAAR->FDn][i+pAAR->FDn];            
                }
            }
        }

        for (k=0;k<pAAR->np_z+2*pAAR->FDn;k++) { // z-direction of interior region
            for (j=0;j<pAAR->np_y+2*pAAR->FDn;j++) { // y-direction of interior region
                for (i=0;i<pAAR->np_x+2*pAAR->FDn;i++) { // x-direction of interior region
                //i,j,k indices are w.r.t proc+FDn domain
                    phi_temp[k][j][i]=phi_old[k][j][i];
                }
            }
        }

        iter=iter+1;
    } // end while loop

    t_poiss1 = MPI_Wtime();
    if(rank==0) {
        if(iter<max_iter && res<=tol)
            {printf("AAR preconditioned with Jacobi (AAJ).\n"); printf("AAR converged!:  Iterations = %d, Relative Residual = %g, Time = %.4f sec\n",iter-1,res,t_poiss1-t_poiss0);}
        if(iter>=max_iter)
            {printf("WARNING: AAR exceeded maximum iterations.\n");printf("AAR:  Iterations = %d, Residual = %g, Time = %.4f sec \n",iter-1,res,t_poiss1-t_poiss0);}
    }

    // shift phi since it can differ by a constant
    double phi_fix=0.0;
    double phi000=pAAR->phi[pAAR->FDn][pAAR->FDn][pAAR->FDn];
    MPI_Bcast(&phi000,1,MPI_DOUBLE,0,MPI_COMM_WORLD);
    for (k=0;k<pAAR->np_z;k++) {
        for (j=0;j<pAAR->np_y;j++) {
            for (i=0;i<pAAR->np_x;i++) {
                pAAR->phi[k+pAAR->FDn][j+pAAR->FDn][i+pAAR->FDn] = pAAR->phi[k+pAAR->FDn][j+pAAR->FDn][i+pAAR->FDn]+phi_fix-phi000;
            }
        }
    }
  
    // de-allocate memory
    for (k=0;k<pAAR->np_z+2*pAAR->FDn;k++) {
        for (j=0;j<pAAR->np_y+2*pAAR->FDn;j++) {
          free(phi_new[k][j]);
          free(phi_old[k][j]);
          free(phi_res[k][j]);
          free(phi_temp[k][j]);
        }
        free(phi_new[k]);
        free(phi_old[k]);
        free(phi_res[k]);
        free(phi_temp[k]);
    }
    free(phi_new);
    free(phi_old);
    free(phi_res);
    free(phi_temp);

    free(phi_res_vec);
    free(phi_old_vec);
    free(Xold);
    free(Fold);
    free(am_vec);
  
    for (k=0;k<m_aar;k++) {
      free(DX[k]);
      free(DF[k]);
    }
    free(DX);
    free(DF);

    free(FtF);
    free(Ftf);
    free(svec);
    free(allredvec);
}
