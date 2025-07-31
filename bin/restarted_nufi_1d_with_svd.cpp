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
	real k = 0.5;
    return 1.0 / std::sqrt(2.0 * M_PI) * u*u * std::exp(-0.5 * u*u) * (1 + alpha * std::cos(k*x)); // Two Stream Instability
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
config_t<double> conf(64, 128, 500, 0.1, 0, 4*M_PI, -10, 10, &f0);

double f_t(double x, double u) noexcept
{
	if(u > conf.u_max || u < conf.u_min){
		return 0;
	}

	size_t nx_r = f0_r.n_rows - 1;
	size_t nu_r = f0_r.n_cols - 1;

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

template <size_t order>
void run_restarted_simulation()
{
	using std::exp;
	using std::sin;
	using std::cos;
    using std::abs;
    using std::max;

    //omp_set_num_threads(1);

    size_t Nx = 256;  // Number of grid points in physical space.
    size_t Nu = Nx;  // Number of quadrature points in velocity space.
    double   dt = 0.1;  // Time-step size.
    size_t Nt = 100/dt;  // Number of time-steps.

    // Dimensions of physical domain.
    double x_min = 0;
    double x_max = 4*M_PI;
    conf.x_min = x_min;
    conf.x_max = x_max; // Actually I should also set Lx etc.

    // Integration limits for velocity space.
    double u_min = -6;
    double u_max = 6;
    conf.u_min = u_min;
    conf.u_max = u_max;

    // We use conf.Nt as restart timer for now.
    size_t nx_r = Nx;
	size_t nu_r = nx_r;
    size_t nt_restart = 100;
    double dx_r = conf.Lx / nx_r;
    double du_r = (conf.u_max - conf.u_min)/ nu_r;
    f0_r.resize(nx_r+1, nu_r+1);
    conf = config_t<double>(Nx, Nu, Nt, dt, x_min, x_max, u_min, u_max, &f0);
    config_t<double> conf_full(Nx, Nu, Nt, dt, x_min, x_max, u_min, u_max, &f0);
    const size_t stride_t = conf.Nx + order - 1;

    std::unique_ptr<double[]> coeffs { new double[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<double[]> coeffs_restart { new double[ (nt_restart+1)*stride_t ] {} };
    std::unique_ptr<double,decltype(std::free)*> rho { reinterpret_cast<double*>(std::aligned_alloc(64,sizeof(double)*conf.Nx)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};

    poisson<double> poiss( conf );

    std::cout << f0_r.n_rows << " " << f0_r.n_cols << std::endl;
    std::cout << nx_r << " " << nu_r << std::endl;

    
    std::ofstream stat_file( "stats.txt" );
    std::ofstream stat_full_file( "stats_full.txt" );
    std::ofstream coeff_str("coeff_restart.txt");
    std::ofstream coeff_r_str("coeff_r_restart.txt");
    double total_time = 0;
    size_t restart_counter = 0;
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

/*         std::ofstream rho_str("rho_" + std::to_string(n*conf.dt) + ".txt");
        for(size_t i = 0; i < conf.Nx; i++){
            rho_str << i*conf.dx << " " << rho.get()[i] << std::endl;
        }  */

        double elec_energy = poiss.solve( rho.get() );

        // Interpolation of Poisson solution.
        periodic::interpolate<double,order>( coeffs_restart.get() + nt_r_curr*stride_t, rho.get(), conf );
        // Copy solution also into global coeffs-vector.
        
        #pragma omp parallel for
        for(size_t i = 0; i < stride_t; i++){
            coeffs.get()[n*stride_t + i ] = coeffs_restart.get()[nt_r_curr*stride_t + i];
        }

        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;

        double Emax = 0;
        size_t plot_n_x = 128;
        double dx_plot = conf.Lx / plot_n_x;
        //std::ofstream E_str("E_" + std::to_string(n*conf.dt) + ".txt");
        for ( size_t i = 0; i <= plot_n_x; ++i )
        {
            double x = conf.x_min + i*dx_plot;
            double E = periodic::eval<double,order,1>(x,coeffs_restart.get()+nt_r_curr*stride_t,conf);
            Emax = max( Emax, std::abs(E) );
            //E_str << x << " " << E << std::endl;
        }

	    double t = n*conf.dt;
        stat_file << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << Emax  << " " << elec_energy << std::endl;
        std::cout << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << Emax << " Comp-time: " << timer_elapsed;
        std::cout << " Total comp time s.f.: " << total_time << std::endl; 

        if(n % (5*16) == 0 && false){
            size_t plot_n_u = plot_n_x;
            double du_plot = (conf.u_max - conf.u_min) / plot_n_u;

            double kinetic_energy = 0;
            double entropy = 0;
            double l1_norm = 0;
            double l2_norm = 0;
            double max_norm = 0;

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
                            << elec_energy                 << "; "
                            << kinetic_energy       << "; "
                            << total_energy         << "; "
                            << entropy              << ";" << std::endl;
        }

        if(nt_r_curr == nt_restart)
    	{
            std::cout << "Restart" << std::endl;
            nufi::stopwatch<double> timer_restart;
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

            // Let's try SVD compression:
            arma::mat U;
            arma::vec s;
            arma::mat V;

            arma::svd_econ(U, s, V, f0_r_copy);
            // Define a threshold
            double tol = 1e-2;
            size_t max_rank = 30;

            // Find how many singular values are above the 
            // (relative) tolerance:
            arma::uword r = arma::sum(s > tol * s(0));
            if ( r > max_rank){
                r = max_rank;
            }
            std::cout << "Truncation rank = " << r << std::endl;
            // Truncate U, s, V
            U = U.cols(0, r - 1);
            s = s.rows(0, r - 1);
            V = V.cols(0, r - 1);

            f0_r = U * arma::diagmat(s) * V.t();
            //f0_r = f0_r_copy;

            conf = config_t<double>(Nx, Nu, Nt, dt, x_min, x_max, u_min, u_max, &f_t);

            // Copy last entry of coeff vector into restarted coeff vector.
            #pragma omp parallel for
            for(size_t i = 0; i < stride_t; i++){
                coeffs_restart.get()[i] = coeffs_restart.get()[nt_r_curr*stride_t + i];
            }

            std::cout << n << " " << nt_r_curr << " restart " << std::endl;
            nt_r_curr = 1;
            restart_counter++;
            double restart_time = timer_restart.elapsed();
            total_time += restart_time;
            std::cout << "Restart took: " << restart_time << ". Total comp time s.f.: " << total_time << std::endl;
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
	//nufi::dim1::run_restarted_simulation<2>();

    nufi::svd_magic::test_rsvd();
}

