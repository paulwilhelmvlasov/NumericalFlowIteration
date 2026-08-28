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

#include <nufi/restart.hpp>

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

arma::mat make_quadratic_decay_matrix(int m, int n, int r) {
    // Step 1: Generate orthonormal U and V
    arma::mat U_full, V_full;
    arma::mat temp_U = arma::randn(m, r);
    arma::mat temp_V = arma::randn(n, r);
    arma::mat R;
    arma::qr(U_full, R, temp_U);  // Orthonormal columns
    arma::qr(V_full, R, temp_V);

    // Step 2: Create singular values with quadratic decay
    arma::vec singular_values(r);
    for (int i = 0; i < r; ++i) {
        //singular_values(i) = 1.0 / ((i + 1.0) * (i + 1.0));  // quadratic
        singular_values(i) = 1.0 / ((i + 1.0));  // linear
    }

    arma::mat S = arma::diagmat(singular_values);

    // Step 3: Construct matrix A = U * S * V^T
    arma::mat A = U_full.cols(0, r - 1) * S * V_full.cols(0, r - 1).t();

    return A;
}

void test_rsvd()
{
    //arma::mat A = arma::randn(1000, 500);  // example A
    //arma::mat A = arma::randn(100, 30) * arma::randn(30, 100);
    arma::mat A = make_quadratic_decay_matrix(100,100,100);

    // Function handles (e.g., pass to randomized_svd)
    auto A_mv = [&](const arma::vec& x) -> arma::vec {
        return A * x;
    };

    auto At_mv = [&](const arma::vec& x) -> arma::vec {
        return A.t() * x;
    };

    // Output variables
    arma::mat U, V;
    arma::vec S;

    std::cout << "Start random svd. " << std::endl;
    randomized_svd(A_mv, At_mv, A.n_rows, A.n_cols, 30, U, S, V);
    std::cout << "RSVD finished." << std::endl;

    arma::mat compressed_A = U * arma::diagmat(S) * V.t();

    double l2_error = arma::norm(A - compressed_A) / arma::norm(A);
    std::cout << "L2 error = " << l2_error << std::endl;
}

}

namespace dim1
{

size_t Nx = 256;  // Number of grid points in physical space.
size_t Nu = 256;  // Number of quadrature points in velocity space.
double   dt = 1.0/10.0;  // Time-step size.
size_t Nt = 100/dt;  // Number of time-steps.

// Dimensions of physical domain.
double x_min = 0;
double x_max = 4*M_PI;
double Lx = x_max - x_min;

// Integration limits for velocity space.
double u_min = -10;
double u_max = 10;

size_t nx_r = Nx;
size_t nu_r = Nu;
size_t nt_restart = 10;
//size_t nt_restart = Nt + 2;
double dx_r = (x_max - x_min) / nx_r;
double du_r = (u_max - u_min) / nu_r;

size_t max_rank = 50;
double tol_rank = 1e-2;

std::ofstream truncation_ranks_str;

template <typename real>
real maxwellian_1d(real u, real vth) noexcept
{
    real c = 1.0 / std::sqrt(2*M_PI*vth*vth);
    return c*std::exp(-(u*u) / (2*vth*vth) );
}

template <typename real>
real f0(real x, real u) noexcept
{
	real alpha = 1e-2; // Linear Landau Damping or Two Stream instability
    //real alpha = 0.5; // Linear Landau Damping or Two Stream instability
	real k = 0.5;
    return 1.0 / std::sqrt(2.0 * M_PI) * u*u * std::exp(-0.5 * u*u) * (1 + alpha * std::cos(k*x)); // Two Stream Instability
    //return 1.0 / std::sqrt(2.0 * M_PI) * exp(-0.5 * u*u) * (1 + alpha * cos(k*x)); // Landau Damping
}

template <typename real>
real lin_interpol(real x , real y, real x1, real x2, real y1, real y2, real f_11,
		real f_12, real f_21, real f_22) noexcept
{
	real f_x_y1 = ((x2-x)*f_11 + (x-x1)*f_21)/(x2-x1);
	real f_x_y2 = ((x2-x)*f_12 + (x-x1)*f_22)/(x2-x1);

	return ((y2-y)*f_x_y1 + (y-y1)*f_x_y2) / (y2-y1);
}

arma::mat f0_r;
arma::mat U_s_r, V_r;
config_t<double> conf(64, 128, 500, 0.1, 0, 4*M_PI, -10, 10, &f0);


double f_t(double x, double u) noexcept
{
	if(u > conf.u_max || u < conf.u_min){
		return 0;
	}

/* 	size_t nx_r = f0_r.n_rows - 1;
	size_t nu_r = f0_r.n_cols - 1; */

	double dx_r = conf.Lx/ nx_r;
	double du_r = (conf.u_max - conf.u_min)/nu_r;

    x = std::fmod(std::fmod(x, conf.Lx) + conf.Lx, conf.Lx);
    size_t x_ref_pos = std::min(static_cast<size_t>(std::floor(x / dx_r)), nx_r - 1);
    size_t u_ref_pos = std::min(static_cast<size_t>(std::floor((u-conf.u_min)/du_r)), nu_r - 1);

	double x1 = x_ref_pos*dx_r;
	double x2 = x1+dx_r;
	double u1 = conf.u_min + u_ref_pos*du_r;
	double u2 = u1 + du_r;

	double f_11 = f0_r(x_ref_pos, u_ref_pos);
	double f_21 = f0_r(x_ref_pos+1, u_ref_pos);
	double f_12 = f0_r(x_ref_pos, u_ref_pos+1);
	double f_22 = f0_r(x_ref_pos+1, u_ref_pos+1);

    double value = lin_interpol<double>(x, u, x1, x2, u1, u2, f_11, f_12, f_21, f_22);

    return value;
}

// Efficient grid access using dot product
inline double f0_svd(size_t i, size_t j) noexcept {
    return arma::dot(U_s_r.row(i), V_r.row(j));
}

inline double cubic_interp(double p0, double p1, double p2, double p3, double t)
{
    double a0 = -0.5*p0 + 1.5*p1 - 1.5*p2 + 0.5*p3;
    double a1 = p0 - 2.5*p1 + 2.0*p2 - 0.5*p3;
    double a2 = -0.5*p0 + 0.5*p2;
    double a3 = p1;

    return ((a0*t + a1)*t + a2)*t + a3;
}

inline size_t clamp_index(int i, size_t N)
{
    if(i < 0) return 0;
    if(i >= static_cast<int>(N)) return N - 1;
    return static_cast<size_t>(i);
}

inline size_t periodic_index(int i, size_t N)
{
    int res = i % static_cast<int>(N);
    if(res < 0) res += N;
    return static_cast<size_t>(res);
}

double cubic_interpolation_2d(double x, double u)
{
    if(u > conf.u_max || u < conf.u_min) {
        return 0.0;
    }

    x = std::fmod(std::fmod(x, Lx) + Lx, Lx);

    size_t nx_r = U_s_r.n_rows - 1;
    size_t nu_r = V_r.n_rows - 1;

    double dx_r = Lx / nx_r;
    double du_r = (conf.u_max - conf.u_min) / nu_r;

    double gx = x / dx_r;
    double gu = (u - conf.u_min) / du_r;

    int ix = static_cast<int>(std::floor(gx));
    int iu = static_cast<int>(std::floor(gu));

    double tx = gx - ix;
    double tu = gu - iu;

    double val_u[4];

    for(int ku = -1; ku <= 2; ++ku)
    {
        int iu_idx = iu + ku;
        size_t iuc = clamp_index(iu_idx, nu_r + 1);

        double px[4];

        for(int kx = -1; kx <= 2; ++kx)
        {
            int ix_idx = ix + kx;
            size_t ixc = periodic_index(ix_idx, nx_r + 1);

            px[kx + 1] = f0_svd(ixc, iuc);
        }

        val_u[ku + 1] = cubic_interp(px[0], px[1], px[2], px[3], tx);
    }

    return cubic_interp(val_u[0], val_u[1], val_u[2], val_u[3], tu);
}

inline arma::mat f0_svd_block(size_t i, size_t j) noexcept {
    // 2xR block from U_s
    arma::mat U_block = U_s_r.rows(i, i+1);      // (2 x r)

    // 2xR block from V
    arma::mat V_block = V_r.rows(j, j+1);      // (2 x r)

    // 2x2 block of function values
    return U_block * V_block.t();              // (2 x 2)
}

double f_svd_t(double x, double u) noexcept
{
	if(u > u_max || u < u_min){
		return 0;
	}

	size_t nx_r = U_s_r.n_rows - 1;
	size_t nu_r = V_r.n_rows - 1;

    // Assuming that x_min = 0. 
    x = std::fmod(std::fmod(x, Lx) + Lx, Lx);
    size_t x_ref_pos = std::min(static_cast<size_t>(std::floor(x / dx_r)), nx_r - 1);
    size_t u_ref_pos = std::min(static_cast<size_t>(std::floor((u-conf.u_min)/du_r)), nu_r - 1);

	double x1 = x_ref_pos*dx_r;
	double x2 = x1 + dx_r;
	double u1 = conf.u_min + u_ref_pos*du_r;
	double u2 = u1 + du_r;

    // get the 2x2 block
    arma::mat F = f0_svd_block(x_ref_pos, u_ref_pos);

    // unpack into corners
    double f_11 = F(0,0);
    double f_21 = F(1,0);
    double f_12 = F(0,1);
    double f_22 = F(1,1);

    double value = lin_interpol<double>(x, u, x1, x2, u1, u2, f_11, f_12, f_21, f_22);

    return value;
}

double f_svd_t_cubic(double x, double u) noexcept
{
    return cubic_interpolation_2d(x, u);
}

template <size_t order>
void restart_with_full_matrix(size_t& nt_r_curr, size_t n, double* coeffs, config_t<double>& conf, double& total_time)
{
    const size_t stride_t = conf.Nx + order - 1;
    std::cout << "Restart" << std::endl;
    nufi::stopwatch<double> timer_restart;
    arma::mat f0_r_copy(nx_r + 1, nu_r + 1);
    #pragma omp parallel for
    for(size_t i = 0; i <= nx_r; i++ ){
    	for(size_t j = 0; j <= nu_r; j++){
    		double x = i*dx_r;
    		double u = conf.u_min + j*du_r;

            double f = periodic::eval_f<double,order>(nt_r_curr,x,u,coeffs,conf);

            f0_r_copy(i,j) = f;
    	}
    }
/* 
    // Let's try SVD compression:
    arma::mat U;
    arma::vec s;
    arma::mat V;

    // Define a threshold
    double tol = 1e-2;
    size_t max_rank = 30;

    // This was the direct SVD way:
    arma::svd_econ(U, s, V, f0_r_copy);

    // Function handles to pass to randomized_svd
    // Later the direct use of f0_r_copy would be substituted by direct 
    // evaluation of f.
    auto A_mv = [&](const arma::vec& x) -> arma::vec {
        return f0_r_copy * x;
    };

    auto At_mv = [&](const arma::vec& x) -> arma::vec {
        return f0_r_copy.t() * x;
    }; */

    //arma::svd_econ(U, s, V, f0_r_copy);

    /* std::cout << "Start random svd. " << std::endl;
    svd_magic::randomized_svd(A_mv, At_mv, f0_r_copy.n_rows, f0_r_copy.n_cols, max_rank, U, s, V);
    std::cout << "RSVD finished." << std::endl;

    std::ofstream singular_values_str("s_" + std::to_string(n*dt) + ".txt");
    for(size_t i = 0; i < s.n_elem; i++){
        singular_values_str << i << " " << s(i) << std::endl;
    } */

    // Find how many singular values are above the 
    // (relative) tolerance:
    /* arma::uword r = arma::sum(s > tol * s(0));
    if ( r > max_rank){
        r = max_rank;
    }
    std::cout << "Truncation rank = " << r << std::endl;
    truncation_ranks_str << n*dt << " " << r << std::endl;
    // Truncate U, s, V
    U = U.cols(0, r - 1);
    s = s.rows(0, r - 1);
    V = V.cols(0, r - 1);
    f0_r = U * arma::diagmat(s) * V.t(); */
    f0_r = f0_r_copy;

    conf = config_t<double>(conf.Nx, conf.Nu, conf.Nt, conf.dt, conf.x_min, 
                            conf.x_max, conf.u_min, conf.u_max, &f_t);
    // Copy last entry of coeff vector into restarted coeff vector.
    #pragma omp parallel for
    for(size_t i = 0; i < stride_t; i++){
         coeffs[i] = coeffs[nt_r_curr*stride_t + i];
    }

    std::cout << n << " " << nt_r_curr << " restart " << std::endl;
    nt_r_curr = 1;
    double restart_time = timer_restart.elapsed();
    total_time += restart_time;
    std::cout << "Restart took: " << restart_time << ". Total comp time s.f.: " << total_time << std::endl;
}

template <size_t order>
void restart_with_rsvd_compression(size_t& nt_r_curr, size_t n, double* coeffs, config_t<double>& conf, double& total_time, double tol = 1e-2, size_t max_rank = 30)
{
    const size_t stride_t = conf.Nx + order - 1;
    std::cout << "Restart" << std::endl;
    nufi::stopwatch<double> timer_restart;

//    arma::mat U;
    arma::vec s;
//    arma::mat V;

    // Define lazy matrix-vector product A * x
    auto A_mv = [&](const arma::vec& x) -> arma::vec {
        arma::vec y(nx_r + 1, arma::fill::zeros);

        // y(i) = sum_j A(i,j) * x(j)
        #pragma omp parallel for
        for (size_t i = 0; i <= nx_r; i++) {
            double x_coord = i * dx_r;
            double acc = 0.0;
            for (size_t j = 0; j <= nu_r; j++) {
                double u_coord = conf.u_min + j * du_r;
                double f = periodic::eval_f<double, order>(nt_r_curr, x_coord, u_coord, coeffs, conf);
                acc += f * x(j);
            }
            y(i) = acc;
        }

        return y;
    };

    // Define lazy matrix-vector product A^T * x
    auto At_mv = [&](const arma::vec& x) -> arma::vec {
        arma::vec y(nu_r + 1, arma::fill::zeros);

        // y(j) = sum_i A(i,j) * x(i)
        #pragma omp parallel for
        for (size_t j = 0; j <= nu_r; j++) {
            double u_coord = conf.u_min + j * du_r;
            double acc = 0.0;
            for (size_t i = 0; i <= nx_r; i++) {
                double x_coord = i * dx_r;
                double f = periodic::eval_f<double, order>(nt_r_curr, x_coord, u_coord, coeffs, conf);
                acc += f * x(i);
            }
            y(j) = acc;
        }

        return y;
    };

    std::cout << "Start random svd. " << std::endl;
    restart::randomized_svd_new(A_mv, At_mv, nx_r + 1, nu_r + 1, max_rank, U_s_r, s, V_r);
    //restart::randomized_svd_old(A_mv, At_mv, nx_r + 1, nu_r + 1, max_rank, U_s_r, s, V_r);
    //svd_magic::randomized_svd(A_mv, At_mv, nx_r + 1, nu_r + 1, max_rank, U_s_r, s, V_r);
    std::cout << "RSVD finished." << std::endl;

    // Truncate by tolerance
    arma::uword r = arma::sum(s > tol * s(0));
    if (r > max_rank) {
        r = max_rank;
    }
    std::cout << "Truncation rank = " << r << std::endl;

    U_s_r = U_s_r.cols(0, r - 1);
    s = s.rows(0, r - 1);
    V_r = V_r.cols(0, r - 1);

    U_s_r = U_s_r * arma::diagmat(s);

    conf = config_t<double>(Nx, Nu, Nt, dt, x_min, x_max, u_min, u_max, &f_svd_t_cubic /* &f_svd_t */);

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


void project_remainder_velocity_moments(
    arma::mat& V_rem,
    const arma::vec& u,
    const arma::vec& w,
    double du
)
{
    if(V_rem.n_cols == 0)
        return;

    arma::vec one(u.n_elem, arma::fill::ones);
    arma::vec u2 = u % u;

    double n0 = du * arma::dot(w, one);
    double n1 = du * arma::dot(w, u2);
    double c  = n1 / n0;

    arma::vec u2c = u2 - c;
    double n2 = du * arma::dot(w, u2c % u2c);

    arma::mat Psi(u.n_elem, 3);
    Psi.col(0) = w;
    Psi.col(1) = w % u;
    Psi.col(2) = w % u2c;

    arma::mat PhiW(u.n_elem, 3);
    PhiW.col(0) = du * one;
    PhiW.col(1) = du * u;
    PhiW.col(2) = du * u2;

    arma::mat G = PhiW.t() * Psi;
    arma::mat C = arma::solve(G, PhiW.t() * V_rem);

    V_rem -= Psi * C;
}

template <size_t order>
void restart_with_conservative_rsvd_compression(
    size_t& nt_r_curr,
    size_t n,
    double* coeffs,
    config_t<double>& conf,
    double& total_time
)
{
    const size_t stride_t = conf.Nx + order - 1;

    std::cout << "Restart with conservative RSVD" << std::endl;
    nufi::stopwatch<double> timer_restart;

    double tol = 1e-8;
    size_t max_rank = 30;

    arma::vec u(nu_r + 1);
    for(size_t j = 0; j <= nu_r; ++j)
        u(j) = conf.u_min + j * du_r;

    arma::vec one(nu_r + 1, arma::fill::ones);
    arma::vec u2 = u % u;
    arma::vec w = arma::exp(-0.5 * u2);
    arma::vec sqrt_w = arma::sqrt(w);
    arma::vec inv_sqrt_w = 1.0 / sqrt_w;

    double n0 = du_r * arma::dot(w, one);
    double n1 = du_r * arma::dot(w, u2);
    double c  = n1 / n0;

    arma::vec u2c = u2 - c;
    double n2 = du_r * arma::dot(w, u2c % u2c);

    arma::vec psi0 = w;
    arma::vec psi1 = w % u;
    arma::vec psi2 = w % u2c;

    arma::vec rho(nx_r + 1, arma::fill::zeros);
    arma::vec J(nx_r + 1, arma::fill::zeros);
    arma::vec kappa(nx_r + 1, arma::fill::zeros);

    #pragma omp parallel for
    for(size_t i = 0; i <= nx_r; ++i)
    {
        double x_coord = i * dx_r;

        double rho_i = 0.0;
        double J_i = 0.0;
        double kappa_i = 0.0;

        for(size_t j = 0; j <= nu_r; ++j)
        {
            double u_coord = conf.u_min + j * du_r;

            double f = periodic::eval_f<double, order>(
                nt_r_curr, x_coord, u_coord, coeffs, conf
            );

            rho_i   += f;
            J_i     += u_coord * f;
            kappa_i += 0.5 * u_coord * u_coord * f;
        }

        rho(i)   = du_r * rho_i;
        J(i)     = du_r * J_i;
        kappa(i) = du_r * kappa_i;
    }

    arma::vec a0 = rho / n0;
    arma::vec a1 = J / n1;
    arma::vec a2 = (2.0 * kappa - c * rho) / n2;

    auto f1_mv = [&](const arma::vec& x) -> arma::vec {
        return a0 * arma::dot(psi0, x)
             + a1 * arma::dot(psi1, x)
             + a2 * arma::dot(psi2, x);
    };

    auto f1_t_mv = [&](const arma::vec& x) -> arma::vec {
        return psi0 * arma::dot(a0, x)
             + psi1 * arma::dot(a1, x)
             + psi2 * arma::dot(a2, x);
    };

    auto A_mv = [&](const arma::vec& x) -> arma::vec {
        arma::vec x_scaled = x % inv_sqrt_w;
        arma::vec y(nx_r + 1, arma::fill::zeros);

        #pragma omp parallel for
        for(size_t i = 0; i <= nx_r; ++i)
        {
            double x_coord = i * dx_r;
            double acc = 0.0;

            for(size_t j = 0; j <= nu_r; ++j)
            {
                double u_coord = conf.u_min + j * du_r;

                double f = periodic::eval_f<double, order>(
                    nt_r_curr, x_coord, u_coord, coeffs, conf
                );

                acc += f * x_scaled(j);
            }

            y(i) = acc;
        }

        return y - f1_mv(x_scaled);
    };

    auto At_mv = [&](const arma::vec& x) -> arma::vec {
        arma::vec y(nu_r + 1, arma::fill::zeros);

        #pragma omp parallel for
        for(size_t j = 0; j <= nu_r; ++j)
        {
            double u_coord = conf.u_min + j * du_r;
            double acc = 0.0;

            for(size_t i = 0; i <= nx_r; ++i)
            {
                double x_coord = i * dx_r;

                double f = periodic::eval_f<double, order>(
                    nt_r_curr, x_coord, u_coord, coeffs, conf
                );

                acc += f * x(i);
            }

            y(j) = acc;
        }

        return (y - f1_t_mv(x)) % inv_sqrt_w;
    };

    arma::mat U_rem;
    arma::vec s;
    arma::mat V_rem;

    std::cout << "Start conservative random SVD." << std::endl;

    restart::randomized_svd_new(
        A_mv,
        At_mv,
        nx_r + 1,
        nu_r + 1,
        max_rank,
        U_rem,
        s,
        V_rem
    );

    std::cout << "RSVD finished." << std::endl;

    arma::uword r = 0;
    if(s.n_elem > 0 && s(0) > 0.0)
    {
        r = arma::sum(s > tol * s(0));
        if(r > max_rank)
            r = max_rank;
    }

    std::cout << "Remainder truncation rank = " << r << std::endl;
    std::cout << "Stored total rank = " << r + 3 << std::endl;

    U_s_r.set_size(nx_r + 1, r + 3);
    V_r.set_size(nu_r + 1, r + 3);

    U_s_r.col(0) = a0;
    U_s_r.col(1) = a1;
    U_s_r.col(2) = a2;

    V_r.col(0) = psi0;
    V_r.col(1) = psi1;
    V_r.col(2) = psi2;

    if(r > 0)
    {
        U_rem = U_rem.cols(0, r - 1);
        s     = s.rows(0, r - 1);
        V_rem = V_rem.cols(0, r - 1);

        U_rem = U_rem * arma::diagmat(s);
        V_rem.each_col() %= sqrt_w;

        U_s_r.cols(3, r + 2) = U_rem;
        V_r.cols(3, r + 2) = V_rem;
    }

    truncation_ranks_str << n * dt << " " << r + 3 << std::endl;

    conf = config_t<double>(
        Nx, Nu, Nt, dt,
        x_min, x_max,
        u_min, u_max,
        //&f_svd_t
        &f_svd_t_cubic
    );

    #pragma omp parallel for
    for(size_t i = 0; i < stride_t; ++i)
        coeffs[i] = coeffs[nt_r_curr * stride_t + i];

    std::cout << n << " " << nt_r_curr << " restart " << std::endl;

    nt_r_curr = 1;

    double restart_time = timer_restart.elapsed();
    total_time += restart_time;

    std::cout << "Restart took: " << restart_time
              << ". Total comp time s.f.: " << total_time << std::endl;
}

template <size_t order>
void restart_with_conservative_rsvd_compression_5th_order(
    size_t& nt_r_curr,
    size_t n,
    double* coeffs,
    config_t<double>& conf,
    double& total_time
)
{
    const size_t stride_t = conf.Nx + order - 1;

    std::cout << "Restart with conservative RSVD" << std::endl;
    nufi::stopwatch<double> timer_restart;

    double tol = 1e-8;
    size_t max_rank = 30;

    arma::vec u(nu_r + 1);
    for(size_t j = 0; j <= nu_r; ++j)
        u(j) = conf.u_min + j * du_r;

    arma::vec one(nu_r + 1, arma::fill::ones);
    arma::vec u2 = u % u;
    arma::vec w = arma::exp(-0.5 * u2);
    arma::vec sqrt_w = arma::sqrt(w);
    arma::vec inv_sqrt_w = 1.0 / sqrt_w;

    const double q0 = -std::sqrt(3.0 / 5.0);
    const double q1 =  0.0;
    const double q2 =  std::sqrt(3.0 / 5.0);

    const double gw0 = 5.0 / 9.0;
    const double gw1 = 8.0 / 9.0;
    const double gw2 = 5.0 / 9.0;

    auto weight_fun = [](double uu) {
        return std::exp(-0.5 * uu * uu);
    };

    double n0 = 0.0;
    double n1 = 0.0;
    double m2w = 0.0;

    for(size_t cell = 0; cell < nu_r; ++cell)
    {
        double a = conf.u_min + cell * du_r;
        double mid = a + 0.5 * du_r;
        double half = 0.5 * du_r;

        double uq[3] = {
            mid + half * q0,
            mid + half * q1,
            mid + half * q2
        };

        double wq[3] = {
            half * gw0,
            half * gw1,
            half * gw2
        };

        for(int q = 0; q < 3; ++q)
        {
            double uu = uq[q];
            double ww = weight_fun(uu);

            n0  += wq[q] * ww;
            n1  += wq[q] * uu * uu * ww;
            m2w += wq[q] * uu * uu * ww;
        }
    }

    double c = m2w / n0;

    double n2 = 0.0;
    for(size_t cell = 0; cell < nu_r; ++cell)
    {
        double a = conf.u_min + cell * du_r;
        double mid = a + 0.5 * du_r;
        double half = 0.5 * du_r;

        double uq[3] = {
            mid + half * q0,
            mid + half * q1,
            mid + half * q2
        };

        double wq[3] = {
            half * gw0,
            half * gw1,
            half * gw2
        };

        for(int q = 0; q < 3; ++q)
        {
            double uu = uq[q];
            double ww = weight_fun(uu);
            double uu2c = uu * uu - c;

            n2 += wq[q] * uu2c * uu2c * ww;
        }
    }

    arma::vec u2c = u2 - c;

    arma::vec psi0 = w;
    arma::vec psi1 = w % u;
    arma::vec psi2 = w % u2c;

    arma::vec rho(nx_r + 1, arma::fill::zeros);
    arma::vec J(nx_r + 1, arma::fill::zeros);
    arma::vec kappa(nx_r + 1, arma::fill::zeros);

    #pragma omp parallel for
    for(size_t i = 0; i <= nx_r; ++i)
    {
        double x_coord = i * dx_r;

        double rho_i = 0.0;
        double J_i = 0.0;
        double kappa_i = 0.0;

        for(size_t cell = 0; cell < nu_r; ++cell)
        {
            double a = conf.u_min + cell * du_r;
            double mid = a + 0.5 * du_r;
            double half = 0.5 * du_r;

            double uq[3] = {
                mid + half * q0,
                mid + half * q1,
                mid + half * q2
            };

            double wq[3] = {
                half * gw0,
                half * gw1,
                half * gw2
            };

            for(int q = 0; q < 3; ++q)
            {
                double u_coord = uq[q];

                double f = periodic::eval_f<double, order>(
                    nt_r_curr, x_coord, u_coord, coeffs, conf
                );

                rho_i   += wq[q] * f;
                J_i     += wq[q] * u_coord * f;
                kappa_i += 0.5 * wq[q] * u_coord * u_coord * f;
            }
        }

        rho(i)   = rho_i;
        J(i)     = J_i;
        kappa(i) = kappa_i;
    }

    arma::vec a0 = rho / n0;
    arma::vec a1 = J / n1;
    arma::vec a2 = (2.0 * kappa - c * rho) / n2;

    auto f1_mv = [&](const arma::vec& q) -> arma::vec {
        return a0 * arma::dot(psi0, q)
             + a1 * arma::dot(psi1, q)
             + a2 * arma::dot(psi2, q);
    };

    auto f1_t_mv = [&](const arma::vec& p) -> arma::vec {
        return psi0 * arma::dot(a0, p)
             + psi1 * arma::dot(a1, p)
             + psi2 * arma::dot(a2, p);
    };

    auto A_mv = [&](const arma::vec& x) -> arma::vec {
        arma::vec x_scaled = x % inv_sqrt_w;
        arma::vec y(nx_r + 1, arma::fill::zeros);

        #pragma omp parallel for
        for(size_t i = 0; i <= nx_r; ++i)
        {
            double x_coord = i * dx_r;
            double acc = 0.0;

            for(size_t j = 0; j <= nu_r; ++j)
            {
                double u_coord = conf.u_min + j * du_r;

                double f = periodic::eval_f<double, order>(
                    nt_r_curr, x_coord, u_coord, coeffs, conf
                );

                acc += f * x_scaled(j);
            }

            y(i) = acc;
        }

        return y - f1_mv(x_scaled);
    };

    auto At_mv = [&](const arma::vec& x) -> arma::vec {
        arma::vec y(nu_r + 1, arma::fill::zeros);

        #pragma omp parallel for
        for(size_t j = 0; j <= nu_r; ++j)
        {
            double u_coord = conf.u_min + j * du_r;
            double acc = 0.0;

            for(size_t i = 0; i <= nx_r; ++i)
            {
                double x_coord = i * dx_r;

                double f = periodic::eval_f<double, order>(
                    nt_r_curr, x_coord, u_coord, coeffs, conf
                );

                acc += f * x(i);
            }

            y(j) = acc;
        }

        return (y - f1_t_mv(x)) % inv_sqrt_w;
    };

    arma::mat U_rem;
    arma::vec s;
    arma::mat V_rem;

    std::cout << "Start conservative random SVD." << std::endl;

    restart::randomized_svd_new(
        A_mv,
        At_mv,
        nx_r + 1,
        nu_r + 1,
        max_rank,
        U_rem,
        s,
        V_rem
    );

    std::cout << "RSVD finished." << std::endl;

    arma::uword r = 0;
    if(s.n_elem > 0 && s(0) > 0.0)
    {
        r = arma::sum(s > tol * s(0));
        if(r > max_rank)
            r = max_rank;
    }

    std::cout << "Remainder truncation rank = " << r << std::endl;
    std::cout << "Stored total rank = " << r + 3 << std::endl;

    U_s_r.set_size(nx_r + 1, r + 3);
    V_r.set_size(nu_r + 1, r + 3);

    U_s_r.col(0) = a0;
    U_s_r.col(1) = a1;
    U_s_r.col(2) = a2;

    V_r.col(0) = psi0;
    V_r.col(1) = psi1;
    V_r.col(2) = psi2;

    if(r > 0)
    {
        U_rem = U_rem.cols(0, r - 1);
        s     = s.rows(0, r - 1);
        V_rem = V_rem.cols(0, r - 1);

        U_rem = U_rem * arma::diagmat(s);
        V_rem.each_col() %= sqrt_w;

        U_s_r.cols(3, r + 2) = U_rem;
        V_r.cols(3, r + 2) = V_rem;
    }

    truncation_ranks_str << n * dt << " " << r + 3 << std::endl;

    conf = config_t<double>(
        Nx, Nu, Nt, dt,
        x_min, x_max,
        u_min, u_max,
        &f_svd_t_cubic
    );

    #pragma omp parallel for
    for(size_t i = 0; i < stride_t; ++i)
        coeffs[i] = coeffs[nt_r_curr * stride_t + i];

    std::cout << n << " " << nt_r_curr << " restart " << std::endl;

    nt_r_curr = 1;

    double restart_time = timer_restart.elapsed();
    total_time += restart_time;

    std::cout << "Restart took: " << restart_time
              << ". Total comp time s.f.: " << total_time << std::endl;
}


std::vector<double> rho_restart;
std::vector<double> J_restart;
std::vector<double> kappa_restart;

double cons_n0 = 1.0;
double cons_n1 = 1.0;
double cons_c  = 0.0;
double cons_n2 = 1.0;

inline double cubic_periodic_vector_interp(
    const std::vector<double>& values,
    double x
)
{
    x = std::fmod(std::fmod(x, Lx) + Lx, Lx);

    double gx = x / dx_r;
    int ix = static_cast<int>(std::floor(gx));
    double tx = gx - ix;

    double p[4];

    for(int k = -1; k <= 2; ++k)
    {
        size_t idx = periodic_index(ix + k, values.size());
        p[k + 1] = values[idx];
    }

    return cubic_interp(p[0], p[1], p[2], p[3], tx);
}

inline double analytic_f1_from_moments(double x, double u)
{
    if(u > conf.u_max || u < conf.u_min)
        return 0.0;

    double rho_x   = cubic_periodic_vector_interp(rho_restart, x);
    double J_x     = cubic_periodic_vector_interp(J_restart, x);
    double kappa_x = cubic_periodic_vector_interp(kappa_restart, x);

    double u2 = u * u;
    double w  = std::exp(-0.5 * u2);

    return w * (
        rho_x / cons_n0
        + (J_x / cons_n1) * u
        + ((2.0 * kappa_x - cons_c * rho_x) / cons_n2) * (u2 - cons_c)
    );
}

double f_svd_t_cubic_f2_plus_analytic_f1(double x, double u) noexcept
{
    return analytic_f1_from_moments(x, u) + cubic_interpolation_2d(x, u);
}

template <size_t order>
void restart_with_conservative_rsvd_compression_analytic_f1(
    size_t& nt_r_curr,
    size_t n,
    double* coeffs,
    config_t<double>& conf,
    double& total_time,
    size_t max_rank,
    double tol
)
{
    const size_t stride_t = conf.Nx + order - 1;

    std::cout << "Restart with conservative RSVD, storing f2 only" << std::endl;
    nufi::stopwatch<double> timer_restart;

    arma::vec u(nu_r + 1);
    for(size_t j = 0; j <= nu_r; ++j)
        u(j) = conf.u_min + j * du_r;

    arma::vec u2 = u % u;
    arma::vec w = arma::exp(-0.5 * u2);
    arma::vec sqrt_w = arma::sqrt(w);
    arma::vec inv_sqrt_w = 1.0 / sqrt_w;

    const double q0 = -std::sqrt(3.0 / 5.0);
    const double q1 =  0.0;
    const double q2 =  std::sqrt(3.0 / 5.0);

    const double gw0 = 5.0 / 9.0;
    const double gw1 = 8.0 / 9.0;
    const double gw2 = 5.0 / 9.0;

    auto weight_fun = [](double uu) {
        return std::exp(-0.5 * uu * uu);
    };

    double n0 = 0.0;
    double n1 = 0.0;

    for(size_t cell = 0; cell < nu_r; ++cell)
    {
        double a = conf.u_min + cell * du_r;
        double mid = a + 0.5 * du_r;
        double half = 0.5 * du_r;

        double uq[3] = {
            mid + half * q0,
            mid + half * q1,
            mid + half * q2
        };

        double wq[3] = {
            half * gw0,
            half * gw1,
            half * gw2
        };

        for(int q = 0; q < 3; ++q)
        {
            double uu = uq[q];
            double ww = weight_fun(uu);

            n0 += wq[q] * ww;
            n1 += wq[q] * uu * uu * ww;
        }
    }

    double c = n1 / n0;

    double n2 = 0.0;

    for(size_t cell = 0; cell < nu_r; ++cell)
    {
        double a = conf.u_min + cell * du_r;
        double mid = a + 0.5 * du_r;
        double half = 0.5 * du_r;

        double uq[3] = {
            mid + half * q0,
            mid + half * q1,
            mid + half * q2
        };

        double wq[3] = {
            half * gw0,
            half * gw1,
            half * gw2
        };

        for(int q = 0; q < 3; ++q)
        {
            double uu = uq[q];
            double ww = weight_fun(uu);
            double u2c = uu * uu - c;

            n2 += wq[q] * u2c * u2c * ww;
        }
    }

    arma::vec u2c = u2 - c;

    arma::vec psi0 = w;
    arma::vec psi1 = w % u;
    arma::vec psi2 = w % u2c;

    arma::vec rho(nx_r + 1, arma::fill::zeros);
    arma::vec J(nx_r + 1, arma::fill::zeros);
    arma::vec kappa(nx_r + 1, arma::fill::zeros);

    #pragma omp parallel for
    for(size_t i = 0; i <= nx_r; ++i)
    {
        double x_coord = i * dx_r;

        double rho_i = 0.0;
        double J_i = 0.0;
        double kappa_i = 0.0;

        for(size_t cell = 0; cell < nu_r; ++cell)
        {
            double a = conf.u_min + cell * du_r;
            double mid = a + 0.5 * du_r;
            double half = 0.5 * du_r;

            double uq[3] = {
                mid + half * q0,
                mid + half * q1,
                mid + half * q2
            };

            double wq[3] = {
                half * gw0,
                half * gw1,
                half * gw2
            };

            for(int q = 0; q < 3; ++q)
            {
                double u_coord = uq[q];

                double f = periodic::eval_f<double, order>(
                    nt_r_curr, x_coord, u_coord, coeffs, conf
                );

                rho_i   += wq[q] * f;
                J_i     += wq[q] * u_coord * f;
                kappa_i += 0.5 * wq[q] * u_coord * u_coord * f;
            }
        }

        rho(i)   = rho_i;
        J(i)     = J_i;
        kappa(i) = kappa_i;
    }

    arma::vec a0 = rho / n0;
    arma::vec a1 = J / n1;
    arma::vec a2 = (2.0 * kappa - c * rho) / n2;

    auto f1_mv = [&](const arma::vec& q) -> arma::vec {
        return a0 * arma::dot(psi0, q)
             + a1 * arma::dot(psi1, q)
             + a2 * arma::dot(psi2, q);
    };

    auto f1_t_mv = [&](const arma::vec& p) -> arma::vec {
        return psi0 * arma::dot(a0, p)
             + psi1 * arma::dot(a1, p)
             + psi2 * arma::dot(a2, p);
    };

    auto A_mv = [&](const arma::vec& x) -> arma::vec {
        arma::vec x_scaled = x % inv_sqrt_w;
        arma::vec y(nx_r + 1, arma::fill::zeros);

        #pragma omp parallel for
        for(size_t i = 0; i <= nx_r; ++i)
        {
            double x_coord = i * dx_r;
            double acc = 0.0;

            for(size_t j = 0; j <= nu_r; ++j)
            {
                double u_coord = conf.u_min + j * du_r;

                double f = periodic::eval_f<double, order>(
                    nt_r_curr, x_coord, u_coord, coeffs, conf
                );

                acc += f * x_scaled(j);
            }

            y(i) = acc;
        }

        return y - f1_mv(x_scaled);
    };

    auto At_mv = [&](const arma::vec& x) -> arma::vec {
        arma::vec y(nu_r + 1, arma::fill::zeros);

        #pragma omp parallel for
        for(size_t j = 0; j <= nu_r; ++j)
        {
            double u_coord = conf.u_min + j * du_r;
            double acc = 0.0;

            for(size_t i = 0; i <= nx_r; ++i)
            {
                double x_coord = i * dx_r;

                double f = periodic::eval_f<double, order>(
                    nt_r_curr, x_coord, u_coord, coeffs, conf
                );

                acc += f * x(i);
            }

            y(j) = acc;
        }

        return (y - f1_t_mv(x)) % inv_sqrt_w;
    };

    arma::mat U_rem;
    arma::vec s;
    arma::mat V_rem;

    std::cout << "Start conservative random SVD." << std::endl;

    restart::randomized_svd_new(
        A_mv,
        At_mv,
        nx_r + 1,
        nu_r + 1,
        max_rank,
        U_rem,
        s,
        V_rem
    );

    std::cout << "RSVD finished." << std::endl;

    arma::uword r = 0;

    if(s.n_elem > 0 && s(0) > 0.0)
    {
        r = arma::sum(s > tol * s(0)); // Relative tolerance
        if(r > max_rank)
            r = max_rank;
    }

    std::cout << "Stored f2 rank = " << r << std::endl;

    if(r > 0)
    {
        U_rem = U_rem.cols(0, r - 1);
        s     = s.rows(0, r - 1);
        V_rem = V_rem.cols(0, r - 1);

        U_rem = U_rem * arma::diagmat(s);
        V_rem.each_col() %= sqrt_w;

        //project_remainder_velocity_moments(V_rem, u, w, du_r);

        U_s_r = U_rem;
        V_r   = V_rem;
    }
    else
    {
        U_s_r.set_size(nx_r + 1, 0);
        V_r.set_size(nu_r + 1, 0);
    }

    cons_n0 = n0;
    cons_n1 = n1;
    cons_c  = c;
    cons_n2 = n2;

    rho_restart.assign(rho.begin(), rho.end());
    J_restart.assign(J.begin(), J.end());
    kappa_restart.assign(kappa.begin(), kappa.end());

    truncation_ranks_str << n * dt << " " << r << std::endl;

    conf = config_t<double>(
        Nx, Nu, Nt, dt,
        x_min, x_max,
        u_min, u_max,
        &f_svd_t_cubic_f2_plus_analytic_f1
    );

    #pragma omp parallel for
    for(size_t i = 0; i < stride_t; ++i)
        coeffs[i] = coeffs[nt_r_curr * stride_t + i];

    std::cout << n << " " << nt_r_curr << " restart " << std::endl;

    nt_r_curr = 1;

    double restart_time = timer_restart.elapsed();
    total_time += restart_time;

    std::cout << "Restart took: " << restart_time
              << ". Total comp time s.f.: " << total_time << std::endl;
}

template <size_t order>
void run_restarted_simulation(bool with_svd_compression = true)
{
    truncation_ranks_str.open("truncation_ranks.txt");
	using std::exp;
	using std::sin;
	using std::cos;
    using std::abs;
    using std::max;

    //omp_set_num_threads(1);

    if(!with_svd_compression){
        f0_r.resize(nx_r+1, nu_r+1);
    }
    
    conf = config_t<double>(Nx, Nu, Nt, dt, x_min, x_max, u_min, u_max, &f0);
    const size_t stride_t = conf.Nx + order - 1;

    std::unique_ptr<double[]> coeffs_restart { new double[ (nt_restart+1)*stride_t ] {} };
    std::unique_ptr<double,decltype(std::free)*> rho { reinterpret_cast<double*>(std::aligned_alloc(64,sizeof(double)*conf.Nx)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};

    poisson<double> poiss( conf );
    
    std::ofstream stat_file( "stats.txt" );
    std::ofstream stat_full_file( "stats_full.txt" );
    //std::ofstream coeff_str("coeff_restart.txt");
    double total_time = 0;
    size_t nt_r_curr = 0;
    for ( size_t n = 0; n <= Nt; ++n )
    {
    	nufi::stopwatch<double> timer;

        std::cout << " start of time step "<< n << " " << nt_r_curr  << std::endl; 
    	// Compute rho:
		#pragma omp parallel for
    	for(size_t i = 0; i<conf.Nx; i++)
    	{
    		rho.get()[i] = periodic::eval_rho<double,order>(nt_r_curr, i, coeffs_restart.get(), conf);
    	}

        double elec_energy = poiss.solve( rho.get() );

        // Interpolation of Poisson solution.
        periodic::interpolate<double,order>( coeffs_restart.get() + nt_r_curr*stride_t, rho.get(), conf );

        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;

        double Emax = 0;
        size_t plot_n_x = 512;
        double dx_plot = conf.Lx / plot_n_x;
        for ( size_t i = 0; i <= plot_n_x; ++i )
        {
            double x = conf.x_min + i*dx_plot;
            double E = periodic::eval<double,order,1>(x,coeffs_restart.get()+nt_r_curr*stride_t,conf);
            Emax = max( Emax, std::abs(E) );
        }

	    double t = n*conf.dt;
        stat_file << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << Emax  << " " << elec_energy << std::endl;
        std::cout << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << Emax << " Comp-time: " << timer_elapsed;
        std::cout << " Total comp time s.f.: " << total_time << std::endl; 

        if(n % 10 == 0 && false){
            arma::mat f0_r_copy(nx_r + 1, nu_r + 1);
            #pragma omp parallel for
            for(size_t i = 0; i <= nx_r; i++ ){
                for(size_t j = 0; j <= nu_r; j++){
                    double x = i*dx_r;
                    double u = conf.u_min + j*du_r;

                    double f = periodic::eval_f<double,order>(nt_r_curr,x,u,coeffs_restart.get(),conf);

                    f0_r_copy(i,j) = f;
                }
            }

            arma::mat U;
            arma::vec s;
            arma::mat V;

            // Define a threshold
            double tol = 1e-4;

            arma::svd_econ(U, s, V, f0_r_copy);
/*             arma::uword r = arma::sum(s > tol * s(0));
            std::cout << "Truncation rank = " << r << std::endl;
            truncation_ranks_str << n*dt << " " << r << std::endl;
 */
            std::ofstream singular_value_str("s_" + std::to_string(n*dt) + ".txt");
            double max_singular_value = s(0);
            for(size_t i = 0; i < s.n_elem; i++ ){
                singular_value_str << i << " " << s(i)/max_singular_value << std::endl;
            }
        }

        if(n % (10) == 0 && true){
            size_t plot_n_u = plot_n_x;
            double du_plot = (conf.u_max - conf.u_min) / plot_n_u;

            double kinetic_energy = 0;
            double entropy = 0;
            double l1_norm = 0;
            double l2_norm = 0;
            double max_norm = 0;

            bool plot_f_now = (n % (10*10) == 0);
            arma::mat f_values;
            if(plot_f_now){
                f_values = arma::mat(plot_n_x,plot_n_u,arma::fill::zeros);
            }

            for(size_t i = 0; i < plot_n_x; i++){
                for(size_t j = 0; j < plot_n_u; j++){
                    double x = conf.x_min + i*dx_plot;
                    double u = conf.u_min + j*du_plot;
                    double f = periodic::eval_f<double,order>(nt_r_curr, x, u, coeffs_restart.get(), conf);
                    kinetic_energy += u*u*f;
                    if(f > 0){
                        entropy -= f*std::log(f);
                    } 
                    l1_norm += f;
                    l2_norm += f*f;
                    max_norm = std::max(f, max_norm);

                    if(plot_f_now){
                        f_values(i,j) = f;
                    }
                }
            }

            if(plot_f_now){
                std::ofstream f_str("f_" + std::to_string(t) + ".txt");
                for(size_t i = 0; i < plot_n_x; i++){
                    for(size_t j = 0; j < plot_n_u; j++){
                        double x = conf.x_min + i*dx_plot;
                        double u = conf.u_min + j*du_plot;
                        double f = f_values(i,j);
                        f_str << x << " " << u << " " << f << std::endl;
                    }
                    f_str << std::endl;
                }
            }
            

            double weight = dx_plot*du_plot;
            kinetic_energy *= weight;
            entropy *= weight;
            l1_norm *= weight;
            l2_norm *= weight;
            double total_energy = kinetic_energy + elec_energy;
            stat_full_file << std::setprecision(16) << t << "; "
                            << l1_norm              << "; "
                            << l2_norm              << "; "
                            << elec_energy          << "; "
                            << kinetic_energy       << "; "
                            << total_energy         << "; "
                            << entropy              << ";" << std::endl;
        }

        if(nt_r_curr == nt_restart)
    	{
            if(with_svd_compression)
            {
                restart_with_conservative_rsvd_compression_analytic_f1<order>(nt_r_curr,n,coeffs_restart.get(),conf,total_time,max_rank,tol_rank);
                //restart_with_conservative_rsvd_compression_5th_order<order>(nt_r_curr,n,coeffs_restart.get(),conf,total_time);
                //restart_with_rsvd_compression<order>(nt_r_curr,n,coeffs_restart.get(),conf,total_time, tol_rank, max_rank);
                //restart_with_conservative_rsvd_compression<order>(nt_r_curr,n,coeffs_restart.get(),conf,total_time);
            } else {
                restart_with_full_matrix<order>(nt_r_curr,n,coeffs_restart.get(),conf,total_time);
            }
    	} else {
            nt_r_curr++;
        }
    }
    std::cout << "Total time: " << total_time << std::endl;

}

}
}



int main()
{
	nufi::dim1::run_restarted_simulation<4>(true);

    //nufi::svd_magic::test_rsvd();
}

