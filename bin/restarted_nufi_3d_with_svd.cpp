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
const double Ly = 1;
const double Lz = 1;

const double x_min = 0;
const double x_max = Lx;
const double y_min = 0;
const double y_max = Ly;
const double z_min = 0;
const double z_max = Lz;

const double u_min = -6;
const double u_max = 6;
const double v_min = -0.5;
const double v_max = 0.5;
const double w_min = -0.5;
const double w_max = 0.5;

const size_t nx_r = 64;
const size_t ny_r = 1;
const size_t nz_r = 1;

const double dx_r = Lx/ nx_r;
const double dy_r = Ly/ ny_r;
const double dz_r = Lz/ nz_r;

const size_t nu_r = nx_r;
const size_t nv_r = 1;
const size_t nw_r = 1;

const double du_r = (u_max - u_min)/nu_r;
const double dv_r = (v_max - v_min)/nv_r;
const double dw_r = (w_max - w_min)/nw_r;

const size_t order = 2;
const size_t Nx = nx_r;  // Number of grid points in physical space.
const size_t Ny = 1;  // Number of grid points in physical space.
const size_t Nz = 1;  // Number of grid points in physical space.
const size_t Nu = Nx;  // Number of quadrature points in velocity space.
const size_t Nv = 1;  // Number of quadrature points in velocity space.
const size_t Nw = 1;  // Number of quadrature points in velocity space.
const double   dt = 0.1;  // Time-step size.
const size_t Nt = 100/dt;  // Number of time-steps.

size_t nt_restart = 5;

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
    return 1.0 / std::sqrt(2.0 * M_PI) * u*u * std::exp(-0.5 * u*u) * (1 + alpha * std::cos(k*x)); 

    // 2d Two Stream Instability:
    /* constexpr real c  = 1/(2*M_PI); 
    return c * ( 1. + alpha*cos(k*x) + alpha*cos(k*y)) 
             * u*u * exp( -(u*u+v*v)/2 ); */

    // 3d Two Stream Instability:
/*     constexpr real c  = 0.06349363593424096978576330493464; 
    return c * ( 1. + alpha*cos(k*x) + alpha*cos(k*y) + alpha*cos(k*z)) 
             * u*u * exp( -(u*u+v*v+w*w)/2 ); */
}

arma::mat U_s_r, V_r;

inline size_t idx_xyz(size_t ix, size_t iy, size_t iz,
                      size_t nx, size_t ny) noexcept {
    return ix + (nx+1)*(iy + (ny+1)*iz);
}
inline size_t idx_uvw(size_t iu, size_t iv, size_t iw,
                      size_t nu, size_t nv) noexcept {
    return iu + (nu+1)*(iv + (nv+1)*iw);
}

inline void weights_1d(double x, double x0, double dx, double& w0, double& w1) noexcept {
    const double t = (x - x0) / dx;
    w1 = t;
    w0 = 1.0 - t;
}

inline void clamp_periodic(double& x, double L) noexcept {
    x = std::fmod(std::fmod(x, L) + L, L);
}

// U is (Nxyz × r), laid out with idx_xyz(ix,iy,iz).
arma::rowvec U_interp_xyz(size_t ix, size_t iy, size_t iz,
                          double wx0, double wx1,
                          double wy0, double wy1,
                          double wz0, double wz1,
                          const arma::mat& U,
                          size_t nx_r, size_t ny_r,
                          size_t r) {
    arma::rowvec res(r, arma::fill::zeros);
    for (int dx=0; dx<=1; ++dx)
    for (int dy=0; dy<=1; ++dy)
    for (int dz=0; dz<=1; ++dz) {
        const double w =
            (dx?wx1:wx0) * (dy?wy1:wy0) * (dz?wz1:wz0);
        const size_t ix2 = ix + dx;
        const size_t iy2 = iy + dy;
        const size_t iz2 = iz + dz;
        const size_t i = idx_xyz(ix2, iy2, iz2, nx_r, ny_r);
        res += w * U.row(i);
    }
    return res;
}

// V is (Nuvw × r), laid out with idx_uvw(iu,iv,iw).
arma::rowvec V_interp_uvw(size_t iu, size_t iv, size_t iw,
                          double wu0, double wu1,
                          double vv0, double vv1,
                          double ww0, double ww1,
                          const arma::mat& V,
                          size_t nu_r, size_t nv_r,
                          size_t r) {
    arma::rowvec res(r, arma::fill::zeros);
    for (int du=0; du<=1; ++du)
    for (int dv=0; dv<=1; ++dv)
    for (int dw=0; dw<=1; ++dw) {
        const double w =
            (du?wu1:wu0) * (dv?vv1:vv0) * (dw?ww1:ww0);
        const size_t iu2 = iu + du;
        const size_t iv2 = iv + dv;
        const size_t iw2 = iw + dw;
        const size_t j = idx_uvw(iu2, iv2, iw2, nu_r, nv_r);
        res += w * V.row(j);
    }
    return res;
}

double f_svd_6d(double x, double y, double z, double u, double v, double w) noexcept
{
    // outside velocity domain -> 0 (like your code)
    if (u < u_min || u > u_max ||
        v < v_min || v > v_max ||
        w < w_min || w > w_max) {
        return 0;
    }

    // periodic wrap in x,y,z
    clamp_periodic(x, Lx);
    clamp_periodic(y, Ly);
    clamp_periodic(z, Lz);

    // left cell indices (ensure we have a +1 for right neighbor)
    const size_t ix = std::min((size_t)std::floor(x / dx_r), nx_r - 1);
    const size_t iy = std::min((size_t)std::floor(y / dy_r), ny_r - 1);
    const size_t iz = std::min((size_t)std::floor(z / dz_r), nz_r - 1);

    const size_t iu = std::min((size_t)std::floor((u - u_min) / du_r), nu_r - 1);
    const size_t iv = std::min((size_t)std::floor((v - v_min) / dv_r), nv_r - 1);
    const size_t iw = std::min((size_t)std::floor((w - w_min) / dw_r), nw_r - 1);

    // cell anchors
    const double x0 = ix * dx_r, y0 = iy * dy_r, z0 = iz * dz_r;
    const double u0 = u_min + iu * du_r, v0 = v_min + iv * dv_r, w0 = w_min + iw * dw_r;

    // 1D weights
    double wx0, wx1, wy0, wy1, wz0, wz1, wu0, wu1, vv0, vv1, ww0, ww1;
    weights_1d(x, x0, dx_r, wx0, wx1);
    weights_1d(y, y0, dy_r, wy0, wy1);
    weights_1d(z, z0, dz_r, wz0, wz1);
    weights_1d(u, u0, du_r, wu0, wu1);
    weights_1d(v, v0, dv_r, vv0, vv1);
    weights_1d(w, w0, dw_r, ww0, ww1);

    // interpolate both sides
    const size_t r = U_s_r.n_cols;
    arma::rowvec row_xyz = U_interp_xyz(ix, iy, iz, wx0, wx1, wy0, wy1, wz0, wz1,
                                        U_s_r, nx_r, ny_r, r);
    arma::rowvec row_uvw = V_interp_uvw(iu, iv, iw, wu0, wu1, vv0, vv1, ww0, ww1,
                                        V_r,   nu_r, nv_r, r);

    // final value
    return arma::dot(row_xyz, row_uvw);
}

template <size_t order>
void restart_with_rsvd_compression(size_t& nt_r_curr, size_t n, double* coeffs, config_t<double>& conf, 
            double& total_time, double tol = 1e-2, size_t max_rank = 10, size_t oversampling = 5)
{
    std::cout << "Restart" << std::endl;
    nufi::stopwatch<double> timer_restart;

    size_t stride_t = (conf.Nx + order - 1) *
                    (conf.Ny + order - 1) *
                    (conf.Nz + order - 1);

    // Sizes (edges inclusive): nx_r,ny_r,nz_r,nu_r,nv_r,nw_r are "rightmost indices".
    // So number of nodes per dim is +1.
    const size_t NX = nx_r + 1, NY = ny_r + 1, NZ = nz_r + 1;
    const size_t NU = nu_r + 1, NV = nv_r + 1, NW = nw_r + 1;
    const size_t Nxyz = NX*NY*NZ;
    const size_t Nuvw = NU*NV*NW;

    // Spacings:
    const double dx = dx_r, dy = dy_r, dz = dz_r;
    const double du = du_r, dv = dv_r, dw = dw_r;

    // Lazy A * x
    auto A_mv = [&](const arma::vec& x) -> arma::vec {
        arma::vec y(Nxyz, arma::fill::zeros);
        #pragma omp parallel for collapse(3)
        for (size_t ix = 0; ix < NX; ++ix)
        for (size_t iy = 0; iy < NY; ++iy)
        for (size_t iz = 0; iz < NZ; ++iz) {
            const double X = conf.x_min + ix*dx;
            const double Y = conf.y_min + iy*dy;
            const double Z = conf.z_min + iz*dz;

            double acc = 0.0;
            // accumulate over uvw
            for (size_t iu = 0; iu < NU; ++iu) {
                const double U = conf.u_min + iu*du;
                for (size_t iv = 0; iv < NV; ++iv) {
                    const double V = conf.v_min + iv*dv;
                    for (size_t iw = 0; iw < NW; ++iw) {
                        const double W = conf.w_min + iw*dw;
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
    auto At_mv = [&](const arma::vec& x) -> arma::vec {
        arma::vec y(Nuvw, arma::fill::zeros);
        #pragma omp parallel for collapse(3)
        for (size_t iu = 0; iu < NU; ++iu)
        for (size_t iv = 0; iv < NV; ++iv)
        for (size_t iw = 0; iw < NW; ++iw) {
            const double U = conf.u_min + iu*du;
            const double V = conf.v_min + iv*dv;
            const double W = conf.w_min + iw*dw;

            double acc = 0.0;
            // accumulate over xyz
            for (size_t ix = 0; ix < NX; ++ix) {
                const double X = conf.x_min + ix*dx;
                for (size_t iy = 0; iy < NY; ++iy) {
                    const double Y = conf.y_min + iy*dy;
                    for (size_t iz = 0; iz < NZ; ++iz) {
                        const double Z = conf.z_min + iz*dz;
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

    arma::vec s;

    svd_magic::randomized_svd(A_mv, At_mv, Nxyz, Nuvw, max_rank, U_s_r, s, V_r, oversampling);

    // truncate by relative tol against s(0)
    arma::uword r = arma::sum(s > tol * s(0));
    r = std::min<arma::uword>(r, max_rank);

    U_s_r = U_s_r.cols(0, r-1);
    V_r = V_r.cols(0, r-1);
    s = s.rows(0, r-1);

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
void run_restarted_simulation()
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
            coeff_file << i << " " << coeffs_restart.get()[n*stride_t + i ] << std::endl;
        }

        if(nt_r_curr == nt_restart)
    	{
            restart_with_rsvd_compression<order>(nt_r_curr,n,coeffs_restart.get(),conf,
                                                total_time,1e-2,30,10);
        } else {
            nt_r_curr++;
        }
    }
}

}

}

int main()
{
    nufi::dim3::run_restarted_simulation<2>();
}