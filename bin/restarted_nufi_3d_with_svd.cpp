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

#include <nufi/config.hpp>
#include <nufi/random.hpp>
#include <nufi/fields.hpp>
#include <nufi/poisson.hpp>
#include <nufi/rho.hpp>
#include <nufi/stopwatch.hpp>


namespace nufi
{

namespace svd_magic
{

// Randomized SVD with on-the-fly A*x and A^T*x evaluation
void randomized_svd(
    std::function<arma::vec(const arma::vec&)> apply_A,
    std::function<arma::vec(const arma::vec&)> apply_At,
    arma::uword m, arma::uword n, arma::uword k,
    arma::mat& U, arma::vec& S, arma::mat& V,
    arma::uword oversampling = 5
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


}

namespace dim3
{

const double Lx = 4*M_PI;
const double Ly = Lx;
const double Lz = Lx;

const double x_min = 0;
const double x_max = Lx;
const double y_min = 0;
const double y_max = Ly;
const double z_min = 0;
const double z_max = Lz;

const double u_min = -6;
const double u_max = 6;
const double v_min = -6;
const double v_max = 6;
const double w_min = -0.5;
const double w_max = 0.5;

const size_t nx_r = 64;
const size_t ny_r = nx_r;
const size_t nz_r = 1;

const double dx_r = Lx/ nx_r;
const double dy_r = Ly/ ny_r;
const double dz_r = Lz/ nz_r;

const size_t nu_r = nx_r;
const size_t nv_r = nu_r;
const size_t nw_r = 1;

const double du_r = (u_max - u_min)/nu_r;
const double dv_r = (v_max - v_min)/nv_r;
const double dw_r = (w_max - w_min)/nw_r;

const size_t order = 2;
const size_t Nx = nx_r;  // Number of grid points in physical space.
const size_t Ny = ny_r;  // Number of grid points in physical space.
const size_t Nz = nz_r;  // Number of grid points in physical space.
const size_t Nu = nu_r;  // Number of quadrature points in velocity space.
const size_t Nv = nv_r;  // Number of quadrature points in velocity space.
const size_t Nw = nw_r;  // Number of quadrature points in velocity space.
const double   dt = 0.1;  // Time-step size.
const size_t Nt = 100/dt;  // Number of time-steps.

size_t nt_restart = 10;

template <typename real>
real f0(real x, real y, real z, real u, real v, real w) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

    constexpr real alpha = 0.01;
    constexpr real k     = 0.5;

    // Weak Landau Damping:
    /* constexpr real c  = 0.06349363593424096978576330493464; // Weak Landau damping
    return c * ( 1. + alpha*cos(k*x) + alpha*cos(k*y) + alpha*cos(k*z)) 
             * exp( -(u*u+v*v+w*w)/2 ); */

    // 1d Two Stream Instability:
    //return 1.0 / std::sqrt(2.0 * M_PI) * u*u * std::exp(-0.5 * u*u) * (1 + alpha * std::cos(k*x)); 

    // 2d Two Stream Instability:
    return 1.0/(2.0*M_PI) * ( 1. + alpha*cos(k*x) + alpha*cos(k*y)) * u*u * exp( -(u*u+v*v)/2 );

    // 3d Two Stream Instability:
/*     constexpr real c  = 0.06349363593424096978576330493464; 
    return c * ( 1. + alpha*cos(k*x) + alpha*cos(k*y) + alpha*cos(k*z)) 
             * u*u * exp( -(u*u+v*v+w*w)/2 ); */
}

// flattening helpers
inline size_t idx_xyz(size_t ix, size_t iy, size_t iz,
                      size_t nx_r, size_t ny_r) noexcept {
    return ix + (nx_r+1) * (iy + (ny_r+1) * iz);
}
inline size_t idx_uvw(size_t iu, size_t iv, size_t iw,
                      size_t nu_r, size_t nv_r) noexcept {
    return iu + (nu_r+1) * (iv + (nv_r+1) * iw);
}

arma::mat F_r, F_r_copy;

double f_t_full(double x, double y, double z, double u, double v, double w) noexcept
{
        // This version is more stable.
    if( u > u_max || u < u_min 
        || v > v_max || v < v_min 
        || w > w_max || w < w_min){
		return 0;
	} 

    x = std::fmod(std::fmod(x, Lx) + Lx, Lx);
    size_t x_ref_pos = std::min(static_cast<size_t>(std::floor(x / dx_r)), nx_r - 1);
    y = std::fmod(std::fmod(y, Ly) + Ly, Ly);
    size_t y_ref_pos = std::min(static_cast<size_t>(std::floor(y / dy_r)), ny_r - 1);
    z = std::fmod(std::fmod(z, Lz) + Lz, Lz);
    size_t z_ref_pos = std::min(static_cast<size_t>(std::floor(z / dz_r)), nz_r - 1);

    size_t u_ref_pos = std::min(static_cast<size_t>(std::floor((u-u_min)/du_r)), nu_r - 1);
    size_t v_ref_pos = std::min(static_cast<size_t>(std::floor((v-v_min)/dv_r)), nv_r - 1);
    size_t w_ref_pos = std::min(static_cast<size_t>(std::floor((w-w_min)/dw_r)), nw_r - 1);


    double x0 = x_ref_pos*dx_r;
    double y0 = y_ref_pos*dy_r;
    double z0 = z_ref_pos*dz_r;
    double u0 = u_min + u_ref_pos*du_r;    
    double v0 = v_min + v_ref_pos*dv_r;    
    double w0 = w_min + w_ref_pos*dw_r;    


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

        value += factor * F_r(index_0, index_1);
    }

    return value;
}

template <size_t order>
void restart_with_full_matrix(size_t& nt_r_curr, size_t n, double* coeffs, config_t<double>& conf, double& total_time, double tol = 1e-2, size_t max_rank = 10, size_t oversampling = 10)
{
    std::cout << "Restart" << std::endl;
    nufi::stopwatch<double> timer_restart;

    size_t stride_t = (conf.Nx + order - 1) *
                    (conf.Ny + order - 1) *
                    (conf.Nz + order - 1);

    // Sizes (edges inclusive): nx_r,ny_r,nz_r,nu_r,nv_r,nw_r are "rightmost indices".
    // So number of nodes per dim is +1.
    size_t size_x_r = (nx_r+1)*(ny_r+1)*(nz_r+1);
    size_t size_v_r = (nu_r+1)*(nv_r+1)*(nw_r+1);

    // This is slightly (20-30%) faster than the original 6-loop. 
    // However, it is still significantly slower than the rho computation
    // as at its core it becomes a memory-bound operation.
    double* Frc = F_r_copy.memptr();
    // Parallelize across columns so each thread writes contiguous rows
    #pragma omp parallel for 
    for (size_t index_1 = 0; index_1 < size_v_r; index_1++) {
        // Decode column -> (iu, iv, iw), compute u,v,w once per column
        size_t tmp1 = index_1;
        const size_t iu = tmp1 % (nu_r+1); tmp1 /= (nu_r+1);
        const size_t iv = tmp1 % (nv_r+1); tmp1 /= (nv_r+1);
        const size_t iw = tmp1;

        const double u = u_min + iu*du_r;
        const double v = v_min + iv*dv_r;
        const double w = w_min + iw*dw_r;

        for (size_t index_0 = 0; index_0 < size_x_r; index_0++) {
            // Decode row -> (ix, iy, iz)
            size_t tmp0 = index_0;
            const size_t ix = tmp0 % (nx_r+1); tmp0 /= (nx_r+1);
            const size_t iy = tmp0 % (ny_r+1); tmp0 /= (ny_r+1);
            const size_t iz = tmp0;

            const double x = x_min + ix*dx_r;
            const double y = y_min + iy*dy_r;
            const double z = z_min + iz*dz_r;

            const double f = eval_f<double,order>(
                nt_r_curr, x, y, z, u, v, w, coeffs, conf
            );

            // Armadillo column-major: memptr()[row + n_rows * col]
            Frc[index_0 + size_x_r * index_1] = f;
        }
    }

    double timer_fill_restart_matrix = timer_restart.elapsed();
    timer_restart.reset();
    std::cout << "Filling restart matrix took " << timer_fill_restart_matrix << " s." << std::endl;

    // Test svd compression restart without svd-evaluation:
    auto A_mv_full = [&](const arma::vec& x) -> arma::vec {
        return F_r_copy * x;
    };

    auto At_mv_full = [&](const arma::vec& x) -> arma::vec {
        return F_r_copy.t() * x;
    };


    // Lazy A * x
    auto A_mv_lazy = [&](const arma::vec& x) -> arma::vec {
        arma::vec y(size_x_r, arma::fill::zeros);
        #pragma omp parallel for collapse(3)
        for (size_t ix = 0; ix <= nx_r; ++ix)
        for (size_t iy = 0; iy <= ny_r; ++iy)
        for (size_t iz = 0; iz <= nz_r; ++iz) {
            const double X = conf.x_min + ix*dx_r;
            const double Y = conf.y_min + iy*dy_r;
            const double Z = conf.z_min + iz*dz_r;

            double acc = 0.0;
            // accumulate over uvw
            for (size_t iu = 0; iu <= nu_r; ++iu) {
                const double U = conf.u_min + iu*du_r;
                for (size_t iv = 0; iv <= nv_r; ++iv) {
                    const double V = conf.v_min + iv*dv_r;
                    for (size_t iw = 0; iw <= nw_r; ++iw) {
                        const double W = conf.w_min + iw*dw_r;
                        const size_t j = idx_uvw(iu, iv, iw, nu_r, nv_r);
                        const double f = eval_f<double,order>( nt_r_curr, X, Y, Z, 
                                                U, V, W, coeffs, conf);
                        acc += f * x(j);
                    }
                }
            }
            const size_t i = idx_xyz(ix, iy, iz, nx_r, ny_r);
            y(i) = acc;
        }
        return y;
    };

    // Lazy A^T * x
    auto At_mv_lazy = [&](const arma::vec& x) -> arma::vec {
        arma::vec y(size_v_r, arma::fill::zeros);
        #pragma omp parallel for collapse(3)
        for (size_t iu = 0; iu <= nu_r; ++iu)
        for (size_t iv = 0; iv <= nv_r; ++iv)
        for (size_t iw = 0; iw <= nw_r; ++iw) {
            const double U = conf.u_min + iu*du_r;
            const double V = conf.v_min + iv*dv_r;
            const double W = conf.w_min + iw*dw_r;

            double acc = 0.0;
            // accumulate over xyz
            for (size_t ix = 0; ix <= nx_r; ++ix) {
                const double X = conf.x_min + ix*dx_r;
                for (size_t iy = 0; iy <= ny_r; ++iy) {
                    const double Y = conf.y_min + iy*dy_r;
                    for (size_t iz = 0; iz <= nz_r; ++iz) {
                        const double Z = conf.z_min + iz*dz_r;
                        const size_t i = idx_xyz(ix, iy, iz, nx_r, ny_r);
                        const double f = eval_f<double,order>( nt_r_curr, X, Y, Z, 
                                                U, V, W, coeffs, conf);
                        acc += f * x(i);
                    }
                }
            }
            const size_t j = idx_uvw(iu, iv, iw, nu_r, nv_r);
            y(j) = acc;
        }
        return y;
    };

    arma::mat U,V;
    arma::vec s;

    /* svd_magic::randomized_svd(A_mv_full, At_mv_full, size_x_r, size_v_r, max_rank, U, s, V, oversampling);
    //svd_magic::randomized_svd(A_mv_lazy, At_mv_lazy, size_x_r, size_v_r, max_rank, U, s, V, oversampling);
    F_r = U * arma::diagmat(s) * V.t(); */
    F_r = F_r_copy;
    double timer_copy_mat = timer_restart.elapsed();
    timer_restart.reset();
    std::cout << "Copying restart matrix took " << timer_copy_mat << " s." << std::endl;

    #pragma omp parallel for
    for(size_t i = 0; i < stride_t; i++){
         coeffs[i] = coeffs[nt_r_curr*stride_t + i];
    }

    conf.f0 = f_t_full;
    std::cout << n << " " << nt_r_curr << " restart " << std::endl;
    nt_r_curr = 1;
    double restart_time = timer_restart.elapsed();
    total_time += restart_time;
    std::cout << "Restart took: " << restart_time << ". Total comp time s.f.: " << total_time << std::endl;
}

arma::mat U_s_r, V_r;

double f_svd_6d(double x, double y, double z, double u, double v, double w) noexcept
{
   // Out-of-range in velocity → zero (same behavior as your original)
    if (u > u_max || u < u_min ||
        v > v_max || v < v_min ||
        w > w_max || w < w_min)
    {
        return 0.0;
    }

    // Periodic wrap in x,y,z
    x = std::fmod(std::fmod(x, Lx) + Lx, Lx);
    y = std::fmod(std::fmod(y, Ly) + Ly, Ly);
    z = std::fmod(std::fmod(z, Lz) + Lz, Lz);

    // Reference cell indices
    size_t x_ref_pos = std::min(static_cast<size_t>(std::floor(x / dx_r)), nx_r - 1);
    size_t y_ref_pos = std::min(static_cast<size_t>(std::floor(y / dy_r)), ny_r - 1);
    size_t z_ref_pos = std::min(static_cast<size_t>(std::floor(z / dz_r)), nz_r - 1);

    size_t u_ref_pos = std::min(static_cast<size_t>(std::floor((u - u_min) / du_r)), nu_r - 1);
    size_t v_ref_pos = std::min(static_cast<size_t>(std::floor((v - v_min) / dv_r)), nv_r - 1);
    size_t w_ref_pos = std::min(static_cast<size_t>(std::floor((w - w_min) / dw_r)), nw_r - 1);

    // Cell anchors
    double x0 = x_ref_pos * dx_r;
    double y0 = y_ref_pos * dy_r;
    double z0 = z_ref_pos * dz_r;
    double u0 = u_min + u_ref_pos * du_r;
    double v0 = v_min + v_ref_pos * dv_r;
    double w0 = w_min + w_ref_pos * dw_r;

    // Local barycentric weights
    double w_x = (x - x0) / dx_r;
    double w_y = (y - y0) / dy_r;
    double w_z = (z - z0) / dz_r;
    double w_u = (u - u0) / du_r;
    double w_v = (v - v0) / dv_r;
    double w_w = (w - w0) / dw_r;

    double value = 0.0;

    for (int i_x = 0; i_x <= 1; ++i_x)
    for (int i_y = 0; i_y <= 1; ++i_y)
    for (int i_z = 0; i_z <= 1; ++i_z)
    for (int i_u = 0; i_u <= 1; ++i_u)
    for (int i_v = 0; i_v <= 1; ++i_v)
    for (int i_w = 0; i_w <= 1; ++i_w)
    {
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

        // F(i,j) = sum_k U_s(i,k) * V(j,k)
        value += factor * arma::dot(U_s_r.row(index_0), V_r.row(index_1));
    }

    return value;
}

template <size_t order>
void restart_with_rsvd_compression(size_t& nt_r_curr, size_t n, double* coeffs, config_t<double>& conf, 
            double& total_time, double tol = 1e-2, size_t max_rank = 10, size_t oversampling = 10)
{
    std::cout << "Restart" << std::endl;
    nufi::stopwatch<double> timer_restart;

    size_t stride_t = (conf.Nx + order - 1) *
                    (conf.Ny + order - 1) *
                    (conf.Nz + order - 1);
    
    size_t size_x_r = (nx_r+1)*(ny_r+1)*(nz_r+1);
    size_t size_v_r = (nu_r+1)*(nv_r+1)*(nw_r+1);

    // Lazy A * x
    arma::vec y(size_x_r, arma::fill::zeros);
    auto A_mv = [&](const arma::vec& x) -> arma::vec {
        //arma::vec y(size_x_r, arma::fill::zeros);
        #pragma omp parallel for collapse(3)
        for (size_t ix = 0; ix <= nx_r; ++ix)
        for (size_t iy = 0; iy <= ny_r; ++iy)
        for (size_t iz = 0; iz <= nz_r; ++iz) {
            const double X = conf.x_min + ix*dx_r;
            const double Y = conf.y_min + iy*dy_r;
            const double Z = conf.z_min + iz*dz_r;

            double acc = 0.0;
            // accumulate over uvw
            for (size_t iu = 0; iu <= nu_r; ++iu) {
                const double U = conf.u_min + iu*du_r;
                for (size_t iv = 0; iv <= nv_r; ++iv) {
                    const double V = conf.v_min + iv*dv_r;
                    for (size_t iw = 0; iw <= nw_r; ++iw) {
                        const double W = conf.w_min + iw*dw_r;
                        const size_t j = idx_uvw(iu, iv, iw, nu_r, nv_r);
                        const double f = eval_f<double,order>( nt_r_curr, X, Y, Z, 
                                                U, V, W, coeffs, conf);
                        acc += f * x(j);
                    }
                }
            }
            const size_t i = idx_xyz(ix, iy, iz, nx_r, ny_r);
            y(i) = acc;
        }
        return y;
    };

    // Lazy A^T * x
    arma::vec yt(size_v_r, arma::fill::zeros);
    auto At_mv = [&](const arma::vec& x) -> arma::vec {
        //arma::vec yt(size_v_r, arma::fill::zeros);
        #pragma omp parallel for collapse(3)
        for (size_t iu = 0; iu <= nu_r; ++iu)
        for (size_t iv = 0; iv <= nv_r; ++iv)
        for (size_t iw = 0; iw <= nw_r; ++iw) {
            const double U = conf.u_min + iu*du_r;
            const double V = conf.v_min + iv*dv_r;
            const double W = conf.w_min + iw*dw_r;

            double acc = 0.0;
            // accumulate over xyz
            for (size_t ix = 0; ix <= nx_r; ++ix) {
                const double X = conf.x_min + ix*dx_r;
                for (size_t iy = 0; iy <= ny_r; ++iy) {
                    const double Y = conf.y_min + iy*dy_r;
                    for (size_t iz = 0; iz <= nz_r; ++iz) {
                        const double Z = conf.z_min + iz*dz_r;
                        const size_t i = idx_xyz(ix, iy, iz, nx_r, ny_r);
                        const double f = eval_f<double,order>( nt_r_curr, X, Y, Z, 
                                                U, V, W, coeffs, conf);
                        acc += f * x(i);
                    }
                }
            }
            const size_t j = idx_uvw(iu, iv, iw, nu_r, nv_r);
            yt(j) = acc;
        }
        return yt;
    };

    arma::vec s;

    svd_magic::randomized_svd(A_mv, At_mv, size_x_r, size_v_r, max_rank, U_s_r, s, V_r, oversampling);

    std::cout << "Singular values: " << std::endl;
    std::cout << s << std::endl;

    // truncate by relative tol against s(0)
    arma::uword r = arma::sum(s > tol * s(0));
    if ( r > max_rank){
        r = max_rank;
    }

    U_s_r = U_s_r.cols(0, r-1);
    s = s.rows(0, r-1);
    V_r = V_r.cols(0, r-1);
    
    // absorb singular values into U for faster evals later
    U_s_r = U_s_r * arma::diagmat(s);

    conf.f0 = f_svd_6d;

    // Copy last coeff slice
    #pragma omp parallel for
    for (size_t i = 0; i < stride_t; i++) {
        coeffs[i] = coeffs[nt_r_curr * stride_t + i];
    }

    std::cout << n << " " << nt_r_curr << " restart " << std::endl;
    nt_r_curr = 1;
    double restart_time = timer_restart.elapsed();
    total_time += restart_time;
    std::cout << "Restart took: " << restart_time
              << ". Total comp time s.f.: " << total_time << std::endl;
}

template <size_t order>
void restart_with_rsvd_compression_new(size_t& nt_r_curr, size_t n, double* coeffs,
                                   config_t<double>& conf, double& total_time,
                                   double tol = 1e-2, size_t max_rank = 10,
                                   size_t oversampling = 10)
{
    std::cout << "Restart" << std::endl;
    nufi::stopwatch<double> timer_restart;

    size_t stride_t = (conf.Nx + order - 1) *
                      (conf.Ny + order - 1) *
                      (conf.Nz + order - 1);

    size_t size_x_r = (nx_r + 1) * (ny_r + 1) * (nz_r + 1);
    size_t size_v_r = (nu_r + 1) * (nv_r + 1) * (nw_r + 1);

    arma::vec y_x(size_x_r, arma::fill::zeros);
    arma::vec y_v(size_v_r, arma::fill::zeros);

    // Lazy A * x
    auto A_mv = [&](const arma::vec& x) -> arma::vec {
        y_x.zeros();
        #pragma omp parallel for
        for (size_t idx = 0; idx < size_x_r; ++idx) {
            size_t iz = idx / ((nx_r + 1) * (ny_r + 1));
            size_t iy = (idx / (nx_r + 1)) % (ny_r + 1);
            size_t ix = idx % (nx_r + 1);

            const double X = conf.x_min + ix * dx_r;
            const double Y = conf.y_min + iy * dy_r;
            const double Z = conf.z_min + iz * dz_r;

            double acc = 0.0;
            for (size_t iu = 0; iu <= nu_r; ++iu) {
                const double U = conf.u_min + iu * du_r;
                for (size_t iv = 0; iv <= nv_r; ++iv) {
                    const double V = conf.v_min + iv * dv_r;
                    for (size_t iw = 0; iw <= nw_r; ++iw) {
                        const double W = conf.w_min + iw * dw_r;
                        const size_t j = idx_uvw(iu, iv, iw, nu_r, nv_r);
                        acc += eval_f<double, order>(nt_r_curr, X, Y, Z, U, V, W, coeffs, conf) * x(j);
                    }
                }
            }
            y_x(idx) = acc;
        }
        return y_x;
    };

    // Lazy A^T * x
    auto At_mv = [&](const arma::vec& x) -> arma::vec {
        y_v.zeros();
        #pragma omp parallel for
        for (size_t idx = 0; idx < size_v_r; ++idx) {
            size_t iw = idx % (nw_r + 1);
            size_t iv = (idx / (nw_r + 1)) % (nv_r + 1);
            size_t iu = idx / ((nv_r + 1) * (nw_r + 1));

            const double U = conf.u_min + iu * du_r;
            const double V = conf.v_min + iv * dv_r;
            const double W = conf.w_min + iw * dw_r;

            double acc = 0.0;
            for (size_t ix = 0; ix <= nx_r; ++ix) {
                const double X = conf.x_min + ix * dx_r;
                for (size_t iy = 0; iy <= ny_r; ++iy) {
                    const double Y = conf.y_min + iy * dy_r;
                    for (size_t iz = 0; iz <= nz_r; ++iz) {
                        const double Z = conf.z_min + iz * dz_r;
                        const size_t i = idx_xyz(ix, iy, iz, nx_r, ny_r);
                        acc += eval_f<double, order>(nt_r_curr, X, Y, Z, U, V, W, coeffs, conf) * x(i);
                    }
                }
            }
            y_v(idx) = acc;
        }
        return y_v;
    };

    arma::vec s;

    svd_magic::randomized_svd(A_mv, At_mv, size_x_r, size_v_r, max_rank, U_s_r, s, V_r, oversampling);

    std::cout << "Singular values: " << s.t() << std::endl;

    // Truncate by tolerance
    arma::uword r = arma::sum(s > tol * s(0));
    if ( r > max_rank){
        r = max_rank;
    }

    U_s_r = U_s_r.cols(0, r - 1);
    s = s.rows(0, r - 1);
    V_r = V_r.cols(0, r - 1);

    // Absorb singular values into U for faster eval
    U_s_r = U_s_r * arma::diagmat(s);

    conf.f0 = f_svd_6d;

    // Copy last coefficient slice
    #pragma omp parallel for
    for (size_t i = 0; i < stride_t; ++i) {
        coeffs[i] = coeffs[nt_r_curr * stride_t + i];
    }

    nt_r_curr = 1;
    double restart_time = timer_restart.elapsed();
    total_time += restart_time;

    std::cout << n << " " << nt_r_curr << " restart completed. "
              << "Restart took: " << restart_time
              << " s. Total comp time s.f.: " << total_time << std::endl;
}

template <size_t order>
void run_restarted_simulation(bool svd_compressed = false, double tolerance = 1e-8, size_t max_rank = 30, size_t oversampling = 10)
{
    config_t<double> conf(Nx, Ny, Nz, Nu, Nv, Nw, Nt, dt, x_min, x_max, y_min, y_max,
        z_min, z_max, u_min, u_max, v_min, v_max, w_min, w_max, &f0);

    size_t stride_t = (conf.Nx + order - 1) *
                    (conf.Ny + order - 1) *
                    (conf.Nz + order - 1);

    std::unique_ptr<double[]> coeffs_restart { new double[ (nt_restart+1)*stride_t ] {} };
    std::unique_ptr<double,decltype(std::free)*> rho { reinterpret_cast<double*>(std::aligned_alloc(64,
                                        sizeof(double)*conf.Nx*conf.Ny*conf.Nz)), std::free };

    poisson<double> poiss( conf );

    if(!svd_compressed){
        size_t size_x_r = (nx_r+1)*(ny_r+1)*(nz_r+1);
        size_t size_v_r = (nu_r+1)*(nv_r+1)*(nw_r+1);
        F_r = arma::mat(size_x_r, size_v_r, arma::fill::zeros);
        F_r_copy = arma::mat(size_x_r, size_v_r, arma::fill::zeros);
    }
        
    std::ofstream stat_file( "stats.txt" );
    std::ofstream coeff_file( "coeffs.txt" );
    double total_time = 0;
    size_t nt_r_curr = 0;
    for ( size_t n = 0; n <= Nt; ++n )
    {
      	nufi::stopwatch<double> timer;

        std::cout << " start of time step "<< n << " " << nt_r_curr  << std::endl; 
		nufi::stopwatch<double> rho_timer;
        #pragma omp parallel for
    	for(size_t l = 0; l<conf.Nx*conf.Ny*conf.Nz; l++)
    	{
    		rho.get()[l] = eval_rho<double,order>(nt_r_curr, l, coeffs_restart.get(), conf);
    	}
        double rho_comp_time = rho_timer.elapsed();
        std::cout << "rho comp time = " << rho_comp_time << " per dof = " <<  rho_comp_time/(conf.Nx*conf.Ny*conf.Nz) << std::endl;

        double E_energy = poiss.solve( rho.get() );
        interpolate<double,order>( coeffs_restart.get() + nt_r_curr*stride_t, rho.get(), conf );


        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;

        double t = n*conf.dt;
        stat_file << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << " " << E_energy << std::endl;
        std::cout << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << E_energy << " Comp-time: " << timer_elapsed;
        std::cout << " Total comp time s.f.: " << total_time << std::endl; 

        // Print coefficients to file.
        coeff_file << n << std::endl;
        for(size_t i = 0; i < stride_t; i++){
            coeff_file << i << " " << coeffs_restart.get()[nt_r_curr*stride_t + i ] << std::endl;
        }

        if(nt_r_curr == nt_restart)
    	{
            if(svd_compressed){
                /* restart_with_rsvd_compression<order>(nt_r_curr,n,coeffs_restart.get(),conf,
                                                total_time,tolerance,max_rank,oversampling); */
                restart_with_rsvd_compression_new<order>(nt_r_curr,n,coeffs_restart.get(),conf,
                                                total_time,tolerance,max_rank,oversampling);
            } else {
                restart_with_full_matrix<order>(nt_r_curr,n,coeffs_restart.get(),conf,
                                                total_time);
            }
            
        } else {
            nt_r_curr++;
        }
    }
}

}

}

int main()
{
    nufi::dim3::run_restarted_simulation<2>(false,1e-16,30,5);
}