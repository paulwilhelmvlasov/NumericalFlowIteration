/*
 * Copyright (C) 2022 Matthias Kirchhart and Paul Wilhelm
 *
 * This file is part of NuFI, a solver for the Vlasov–Poisson equation.
 *
 * NuFI is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3, or (at your option) any later
 * version.
 *
 * NuFI is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * NuFI; see the file COPYING.  If not see http://www.gnu.org/licenses.
 */

#include <cmath>
#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <functional>
#include <sstream>

#include <armadillo>

namespace nufi
{

namespace restart
{

// Randomized SVD with on-the-fly A*x and A^T*x evaluation
void randomized_svd_old(
    std::function<arma::vec(const arma::vec&)> apply_A,
    std::function<arma::vec(const arma::vec&)> apply_At,
    arma::uword m, arma::uword n, arma::uword k,
    arma::mat& U, arma::vec& S, arma::mat& V,
    arma::uword oversampling = 10
) {
    arma::uword l = k + oversampling;

    // Step 1: Draw a random test matrix Omega
    arma::mat Omega = arma::randn(n, l);  // shape: n × l

    // Step 2: Compute Y = A * Omega
    arma::mat Y(m, l);
    for (arma::uword i = 0; i < l; ++i)
        Y.col(i) = apply_A(Omega.col(i));

    // Step 3: Orthonormalize Y to get Q
    arma::mat Q;
    arma::mat R;
    arma::qr_econ(Q, R, Y);  // economy QR

    // Step 4: B = Q^T * A
    arma::mat B(l, n);  // Q^T * A ≈ (l x m) * (m x n) = (l x n)
    for (arma::uword i = 0; i < n; ++i) {
        arma::vec e_i = arma::zeros<arma::vec>(n);
        e_i(i) = 1.0;
        arma::vec Ai = apply_A(e_i);
        B.col(i) = Q.t() * Ai;
    }

    // Step 5: SVD of the small matrix B
    arma::mat U_tilde, V_temp;
    arma::vec S_temp;
    arma::svd(U_tilde, S_temp, V_temp, B);  // B = U_tilde * S * V_temp^T

    // Step 6: Recover U = Q * U_tilde
    U = Q * U_tilde;
    S = S_temp.head(k);
    V = V_temp.cols(0, k - 1);
    U = U.cols(0, k - 1);  // truncate U too
}

void randomized_svd_new(
    std::function<arma::vec(const arma::vec&)> apply_A,
    std::function<arma::vec(const arma::vec&)> apply_At,
    arma::uword m, arma::uword n, arma::uword k,
    arma::mat& U, arma::vec& S, arma::mat& V,
    arma::uword oversampling = 10
) {
    arma::uword l = k + oversampling;

    // Step 1: Draw a random test matrix Omega
    arma::mat Omega = arma::randn(n, l);  // shape: n × l

    // Step 2: Compute Y = A * Omega
    arma::mat Y(m, l);
    for (arma::uword i = 0; i < l; ++i)
        Y.col(i) = apply_A(Omega.col(i));

    // Step 3: Orthonormalize Y to get Q
    arma::mat Q;
    arma::mat R;
    arma::qr_econ(Q, R, Y);  // economy QR

    // Step 4: Compute B^T = A^T * Q  (transpose formulation)
    arma::mat Bt(n, l);  // B^T ∈ ℝ^{n×l}
    for (arma::uword j = 0; j < l; ++j)
        Bt.col(j) = apply_At(Q.col(j));

    // Step 5: SVD of the small matrix B^T
    arma::mat V_temp, U_tilde;
    arma::vec S_temp;
    arma::svd(V_temp, S_temp, U_tilde, Bt);  // Bt = V * S * U_tilde^T

    // Step 6: Recover U = Q * U_tilde
    U = Q * U_tilde.cols(0, k - 1);
    S = S_temp.head(k);
    V = V_temp.cols(0, k - 1);
}


class linear_interpolant_6d
{
    public:
        double xmin = 0;
        double xmax = 1;
        double ymin = 0;
        double ymax = 1;
        double zmin = 0;
        double zmax = 1;
        
        double umin = -1;
        double umax = 1;
        double vmin = -1;
        double vmax = 1;
        double wmin = -1;
        double wmax = 1;

        size_t nx_r = 8;
        size_t ny_r = 8;
        size_t nz_r = 8;
        size_t nu_r = 8;
        size_t nv_r = 8;
        size_t nw_r = 8;

        size_t size_x_r = 1;
        size_t size_v_r = 1;

        double dx_r = 1;
        double dy_r = 1;
        double dz_r = 1;
        double du_r = 1;
        double dv_r = 1;
        double dw_r = 1;

        arma::mat restart_matrix;
        arma::mat copy_mat;

        linear_interpolant_6d() { }

        linear_interpolant_6d(double xmin, double xmax, 
            double ymin, double ymax, double zmin, double zmax,
            double umin, double umax, double vmin, double vmax,
            double wmin, double wmax, size_t Nx, size_t Ny, size_t Nz,
            size_t Nu, size_t Nv, size_t Nw) 
            : xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), 
            zmin(zmin), zmax(zmax), umin(umin), umax(umax), 
            vmin(vmin), vmax(vmax), wmin(wmin), wmax(wmax),
            nx_r(Nx), ny_r(Ny), nz_r(Nz), nu_r(Nu), nv_r(Nv), nw_r(Nw)
        {
            size_x_r = (nx_r+1)*(ny_r+1)*(nz_r+1);
            size_v_r = (nu_r+1)*(nv_r+1)*(nw_r+1);

            dx_r = (xmax - xmin) / nx_r;
            dy_r = (ymax - ymin) / ny_r;
            dz_r = (zmax - zmin) / nz_r;

            du_r = (umax - umin) / nu_r;
            dv_r = (vmax - vmin) / nv_r;
            dw_r = (wmax - wmin) / nw_r;

            restart_matrix = arma::mat(size_x_r, size_v_r, arma::fill::zeros);
            copy_mat = arma::mat(size_x_r, size_v_r, arma::fill::zeros);
        }
        
        void restart_f(const std::function<double(double,double,double,double,double,double)>& eval_f)
        {
            #pragma omp parallel for collapse(6)
            for(size_t ix = 0; ix <= nx_r; ix++)
            for(size_t iy = 0; iy <= ny_r; iy++)
            for(size_t iz = 0; iz <= nz_r; iz++)
            for(size_t iu = 0; iu <= nu_r; iu++)
            for(size_t iv = 0; iv <= nv_r; iv++)
            for(size_t iw = 0; iw <= nw_r; iw++){
                double x = xmin + ix*dx_r;
                double y = ymin + iy*dy_r;
                double z = zmin + iz*dz_r;

                double u = umin + iu*du_r;
                double v = vmin + iv*dv_r;
                double w = wmin + iw*dw_r;

                size_t index_0 = ix + (nx_r+1)*(iy + (ny_r+1)*iz);
                size_t index_1 = iu + (nu_r+1)*(iv + (nv_r+1)*iw);

                copy_mat(index_0,index_1) = eval_f(x, y, z, u, v, w);
            }

            restart_matrix = copy_mat;
        }

        void restart_f_checking_boundaries(const std::function<double(double,double,double,double,double,double)>& eval_f, 
                        double& umin_new, double& umax_new, double& vmin_new, 
                        double& vmax_new, double& wmin_new, double& wmax_new, 
                        double tol, double expand = 1.5)
        {
            // This assumes that the velocity support is of the form v_{i,min} < 0
            // and v_{i,max} > 0.
            bool umin_expand = false;
            bool umax_expand = false;
            bool vmin_expand = false;
            bool vmax_expand = false;
            bool wmin_expand = false;
            bool wmax_expand = false;

            #pragma omp parallel for collapse(3)
            for(size_t ix = 0; ix <= nx_r; ix++)
            for(size_t iy = 0; iy <= ny_r; iy++)
            for(size_t iz = 0; iz <= nz_r; iz++)
            for(size_t iu = 0; iu <= nu_r; iu++)
            for(size_t iv = 0; iv <= nv_r; iv++)
            for(size_t iw = 0; iw <= nw_r; iw++){
                double x = xmin + ix*dx_r;
                double y = ymin + iy*dy_r;
                double z = zmin + iz*dz_r;

                double u = umin + iu*du_r;
                double v = vmin + iv*dv_r;
                double w = wmin + iw*dw_r;

                double f = 0;

                if(iu == 0 || iu == nu_r || iv == 0 
                    || iv == nv_r || iw == 0 || iw == nw_r){
                    f = eval_f(x,y,z,u,v,w);
                }

                // Check u_min boundary.
                if(iu == 0){
                    if(f > tol){
                        umin_expand = true;
                    }
                } 
                // Check u_max boundary.
                if(iu == nu_r){
                    if(f > tol){
                        umax_expand = true;
                    }
                }
                
                // Check v_min boundary.
                if(iv == 0){
                    if(f > tol){
                        vmin_expand = true;
                    }
                } 
                // Check v_max boundary.
                if(iv == nv_r){
                    if(f > tol){
                        vmax_expand = true;
                    }
                }

                // Check w_min boundary.
                if(iw == 0){
                    if(f > tol){
                        wmin_expand = true;
                    }
                } 
                // Check w_max boundary.
                if(iw == nu_r){
                    if(f > tol){
                        wmax_expand = true;
                    }
                }
            }

            if(umin_expand){
                umin_new = umin*expand;
            } 
            if(umax_expand){
                umax_new = umax*expand;
            } 

            if(vmin_expand){
                vmin_new = vmin*expand;
            } 
            if(vmax_expand){
                vmax_new = vmax*expand;
            } 

            if(wmin_expand){
                wmin_new = wmin*expand;
            } 
            if(wmax_expand){
                wmax_new = wmax*expand;
            } 

            // Restart with new velocity boundaries.
            double du_new = (umax_new - umin_new) / nu_r;
            double dv_new = (vmax_new - vmin_new) / nv_r;
            double dw_new = (wmax_new - wmin_new) / nw_r;

            #pragma omp parallel for collapse(6)
            for(size_t ix = 0; ix <= nx_r; ix++)
            for(size_t iy = 0; iy <= ny_r; iy++)
            for(size_t iz = 0; iz <= nz_r; iz++)
            for(size_t iu = 0; iu <= nu_r; iu++)
            for(size_t iv = 0; iv <= nv_r; iv++)
            for(size_t iw = 0; iw <= nw_r; iw++){
                double x = xmin + ix*dx_r;
                double y = ymin + iy*dy_r;
                double z = zmin + iz*dz_r;

                double u = umin_new + iu*du_new;
                double v = vmin_new + iv*dv_new;
                double w = wmin_new + iw*dw_new;

                size_t index_0 = ix + (nx_r+1)*(iy + (ny_r+1)*iz);
                size_t index_1 = iu + (nu_r+1)*(iv + (nv_r+1)*iw);

                copy_mat(index_0,index_1) = eval_f(x, y, z, u, v, w);
            }

            restart_matrix = copy_mat;

            umin = umin_new;
            umax = umax_new;

            vmin = vmin_new;
            vmax = vmax_new;

            wmin = wmin_new;
            wmax = wmax_new;

            du_r = du_new;
            dv_r = dv_new;
            dw_r = dw_new;
        }

        double eval_linear_interpolant(double x, double y, double z, 
                                    double u, double v, double w)
        {
            // Collapsing dimensions in theory should save some computational
            // effort but in practice we noticed that the benefit is neglible 
            // or the full version is even faster in some cases.

            if( u > umax || u < umin 
                || v > vmax || v < vmin 
                || w > wmax || w < wmin){
                /* std::cout << "Return u " << u << " " << umin << " " << umax << std::endl;
                std::cout << "Return v " << v << " " << vmin << " " << vmax << std::endl;
                std::cout << "Return w " << w << " " << wmin << " " << wmax << std::endl;
                std::cout << "====================================================" << std::endl; */
                return 0;
            } 

            double Lx = xmax - xmin;
            x = std::fmod(std::fmod(x - xmin, Lx) + Lx, Lx) + xmin;
            size_t x_ref_pos = std::min(static_cast<size_t>(std::floor(x / dx_r)), nx_r - 1);
            double Ly = ymax - ymin;
            y = std::fmod(std::fmod(y - ymin, Ly) + Ly, Ly) + ymin;
            size_t y_ref_pos = std::min(static_cast<size_t>(std::floor(y / dy_r)), ny_r - 1);
            double Lz = zmax - zmin;
            z = std::fmod(std::fmod(z - zmin, Lz) + Lz, Lz) + zmin;
            size_t z_ref_pos = std::min(static_cast<size_t>(std::floor(z / dz_r)), nz_r - 1);

            size_t u_ref_pos = std::min(static_cast<size_t>(std::floor((u-umin)/du_r)), nu_r - 1);
            size_t v_ref_pos = std::min(static_cast<size_t>(std::floor((v-vmin)/dv_r)), nv_r - 1);
            size_t w_ref_pos = std::min(static_cast<size_t>(std::floor((w-wmin)/dw_r)), nw_r - 1);


            double x0 = x_ref_pos*dx_r;
            double y0 = y_ref_pos*dy_r;
            double z0 = z_ref_pos*dz_r;
            double u0 = umin + u_ref_pos*du_r;    
            double v0 = vmin + v_ref_pos*dv_r;    
            double w0 = wmin + w_ref_pos*dw_r;    


            double w_x = (x - x0)/dx_r;
            double w_y = (y - y0)/dy_r;
            double w_z = (z - z0)/dz_r;
            double w_u = (u - u0)/du_r;
            double w_v = (v - v0)/dv_r;
            double w_w = (w - w0)/dw_r;

            double value = 0;
            for(int i_x = 0; i_x <= 1; i_x++)
            for(int i_y = 0; i_y <= 1; i_y++)
            for(int i_z = 0; i_z <= 1; i_z++)
            for(int i_u = 0; i_u <= 1; i_u++)
            for(int i_v = 0; i_v <= 1; i_v++)
            for(int i_w = 0; i_w <= 1; i_w++){
                double factor = ((1-w_x)*(i_x==0) + w_x*(i_x==1))
                            * ((1-w_y)*(i_y==0) + w_y*(i_y==1))
                            * ((1-w_z)*(i_z==0) + w_z*(i_z==1))
                            * ((1-w_u)*(i_u==0) + w_u*(i_u==1))
                            * ((1-w_v)*(i_v==0) + w_v*(i_v==1))
                            * ((1-w_w)*(i_w==0) + w_w*(i_w==1));
                
                size_t index_x = x_ref_pos + i_x;
                size_t index_y = y_ref_pos + i_y;
                size_t index_z = z_ref_pos + i_z;
                size_t index_u = u_ref_pos + i_u;
                size_t index_v = v_ref_pos + i_v;
                size_t index_w = w_ref_pos + i_w;

                size_t index_0 = index_x + (nx_r+1)*(index_y + (ny_r+1)*index_z);
                size_t index_1 = index_u + (nu_r+1)*(index_v + (nv_r+1)*index_w);

                value += factor * restart_matrix(index_0, index_1);
            }

            return value;
        }
};

}

}