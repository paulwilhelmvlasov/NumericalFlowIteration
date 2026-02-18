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
	//real alpha = 1e-2; // Linear Landau Damping or Two Stream instability
	real alpha = 0.5; // Strong Landau Damping
	real k = 0.5;
    //return 1.0 / std::sqrt(2.0 * M_PI) * u*u * std::exp(-0.5 * u*u) * (1 + alpha * std::cos(k*x)); // Two Stream Instability
	return 1.0 / std::sqrt(2.0 * M_PI) * exp(-0.5 * u*u) * (1 + alpha * cos(k*x)); // Landau Damping

    // Bump on tail Instability
    /* return 1.0 / std::sqrt(2.0 * M_PI) * (1 + 0.04 * std::cos(0.3*x)) 
            *  ( 0.9 * std::exp(-0.5 * u*u)  + 0.2 * std::exp(-0.5/(0.5*0.5) * (u-4.5)*(u-4.5)) );  */
}

template <typename real>
real lin_interpol(real x , real y, real x1, real x2, real y1, real y2, real f_11,
		real f_12, real f_21, real f_22) noexcept
{
	real f_x_y1 = ((x2-x)*f_11 + (x-x1)*f_21)/(x2-x1);
	real f_x_y2 = ((x2-x)*f_12 + (x-x1)*f_22)/(x2-x1);

	return ((y2-y)*f_x_y1 + (y-y1)*f_x_y2) / (y2-y1);
}

config_t<double> conf(64, 128, 500, 0.1, 0, 4*M_PI, -10, 10, &f0);

double eval_interpolant(double x, double u, const arma::mat& value_mat) noexcept
{
	// Velocity boundary treatment: If u gets close to a boundary, then
    // f(u) ~ 0 anyway, so we can safely assume u = umin or 
    // u = umax respectively.
    if(u < conf.u_min){
        u = conf.u_min;
    } else if(u > conf.u_max){
        u = conf.u_max;
    }

	size_t nx_r = value_mat.n_rows - 1;
	size_t nu_r = value_mat.n_cols - 1;

	double dx_r = conf.Lx / nx_r;
	double du_r = (conf.u_max - conf.u_min)/nu_r;

    x = std::fmod(std::fmod(x, conf.Lx) + conf.Lx, conf.Lx);
    size_t x_ref_pos = std::min(static_cast<size_t>(std::floor(x / dx_r)), nx_r - 1);
    size_t u_ref_pos = std::min(static_cast<size_t>(std::floor((u-conf.u_min)/du_r)), nu_r - 1);

	double x1 = x_ref_pos*dx_r;
	double x2 = x1+dx_r;
	double u1 = conf.u_min + u_ref_pos*du_r;
	double u2 = u1 + du_r;

	double f_11 = value_mat(x_ref_pos, u_ref_pos);
	double f_21 = value_mat(x_ref_pos+1, u_ref_pos);
	double f_12 = value_mat(x_ref_pos, u_ref_pos+1);
	double f_22 = value_mat(x_ref_pos+1, u_ref_pos+1);

    return lin_interpol<double>(x, u, x1, x2, u1, u2, f_11, f_12, f_21, f_22);
}

size_t restart_counter = 0;

std::vector<arma::mat> char_map_x;
std::vector<arma::mat> char_map_v;

double eval_f_cmm_linear(double x, double v)
{
    for(size_t i = restart_counter; i > 0; i--){
        double x0 = x, v0 = v;
        x = eval_interpolant(x0, v0, char_map_x[i]);
        v = eval_interpolant(x0, v0, char_map_v[i]);
    }

    return f0(x,v);
}

template <size_t order>
void cmm_nufi_linear()
{
	using std::exp;
	using std::sin;
	using std::cos;
    using std::abs;
    using std::max;

    size_t Nx = 128;  // Number of grid points in physical space.
    size_t Nu = 256;  // Number of quadrature points in velocity space.
    double   dt = 0.1;  // Time-step size.
    size_t Nt = 100/dt;  // Number of time-steps.

    // Dimensions of physical domain.
    double x_min = 0;
    double x_max = 4*M_PI;
    //double x_max = 2*M_PI/0.3;
    conf.x_min = x_min;
    conf.x_max = x_max; // Actually I should also set Lx etc.

    // Integration limits for velocity space.
    double u_min = -6;
    double u_max = 6;
    conf.u_min = u_min;
    conf.u_max = u_max;

    // Set up CMM restart.
    size_t nx_r = Nx;
	size_t nu_r = Nu;
    size_t nt_restart = 50;
    double dx_r = conf.Lx / nx_r;
    double du_r = (conf.u_max - conf.u_min)/ nu_r;
    conf = config_t<double>(Nx, Nu, Nt, dt, x_min, x_max, u_min, u_max, &f0);
    config_t<double> conf_full(Nx, Nu, Nt, dt, x_min, x_max, u_min, u_max, &f0);
    const size_t stride_t = conf.Nx + order - 1;

    char_map_x.resize(Nt/nt_restart + 1);
    char_map_v.resize(Nt/nt_restart + 1);
    restart_counter = 0;

    std::unique_ptr<double[]> coeffs { new double[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<double[]> coeffs_restart { new double[ (nt_restart+1)*stride_t ] {} };
    std::unique_ptr<double,decltype(std::free)*> rho { reinterpret_cast<double*>(std::aligned_alloc(64,sizeof(double)*conf.Nx)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};

    poisson<double> poiss( conf );

    std::cout << nx_r << " " << nu_r << std::endl;
    
    std::ofstream stat_file( "stats.txt" );
    std::ofstream stat_full_file( "stats_full.txt" );
    std::ofstream coeff_str("coeff_restart.txt");
    std::ofstream coeff_r_str("coeff_r_restart.txt");
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

        if(n % (5*16) == 0){
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
            char_map_x[restart_counter + 1].resize(nx_r + 1, nu_r + 1);
            char_map_v[restart_counter + 1].resize(nx_r + 1, nu_r + 1);
            #pragma omp parallel for
    		for(size_t i = 0; i <= nx_r; i++ ){
    			for(size_t j = 0; j <= nu_r; j++){
    				double x = i*dx_r;
    				double u = conf.u_min + j*du_r;

                    periodic::eval_char_map<double,order>(nt_r_curr,x,u,coeffs_restart.get(),conf);

                    char_map_x[restart_counter + 1](i,j) = x;
                    char_map_v[restart_counter + 1](i,j) = u;
    			}
    		}

            conf = config_t<double>(Nx, Nu, Nt, dt, x_min, x_max, u_min, u_max, &eval_f_cmm_linear);

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


double bogus(double x, double y, double u, double v)
{
    return -1;
}

dim2::config_t<double> spline_conf(1, 1, 1, 1, 1, 1, 
                    0, 1, 0, 1, -1, 1, -1, 1, 
                    &bogus);

std::vector<std::vector<double>> char_map_x_coeff;
std::vector<std::vector<double>> char_map_v_coeff;

template <size_t order>
double eval_BSpline_interpolant(double x, double u, double* coeff)
{
    // Velocity boundary treatment: If u gets close to a boundary, then
    // f(u) ~ 0 anyway, so we can safely assume u = umin or 
    // u = umax respectively.
    if(u < conf.u_min){
        u = conf.u_min;
    } else if(u > conf.u_max){
        u = conf.u_max;
    }

    return dim2::eval<double,order>(x, u, coeff, spline_conf);
}

template <size_t order>
double eval_f_cmm_spline(double x, double v)
{
    for(size_t i = restart_counter; i > 0; i--){
        double x0 = x, v0 = v;
        x = eval_BSpline_interpolant<order>(x0, v0, char_map_x_coeff[i].data());
        v = eval_BSpline_interpolant<order>(x0, v0, char_map_v_coeff[i].data());
    }

    return f0(x,v);
}


template <size_t order>
void cmm_nufi_spline()
{
	using std::exp;
	using std::sin;
	using std::cos;
    using std::abs;
    using std::max;

    size_t Nx = 128;  // Number of grid points in physical space.
    size_t Nu = 256;  // Number of quadrature points in velocity space.
    double   dt = 0.1;  // Time-step size.
    size_t Nt = 100/dt;  // Number of time-steps.

    // Dimensions of physical domain.
    double x_min = 0;
    double x_max = 4*M_PI;
    //double x_max = 2*M_PI/0.3;
    conf.x_min = x_min;
    conf.x_max = x_max; // Actually I should also set Lx etc.

    // Integration limits for velocity space.
    double u_min = -6;
    double u_max = 6;
    conf.u_min = u_min;
    conf.u_max = u_max;

    // Set up CMM restart.
    size_t nx_r = Nx;
	size_t nu_r = Nu;
    size_t nt_restart = 50;
    double dx_r = conf.Lx / nx_r;
    double du_r = (conf.u_max - conf.u_min)/ nu_r;
    conf = config_t<double>(Nx, Nu, Nt, dt, x_min, x_max, u_min, u_max, &f0);
    config_t<double> conf_full(Nx, Nu, Nt, dt, x_min, x_max, u_min, u_max, &f0);
    const size_t stride_t = conf.Nx + order - 1;

    char_map_x_coeff.resize(Nt/nt_restart + 1);
    char_map_v_coeff.resize(Nt/nt_restart + 1);
    // For spline-conf only "x" coordinates and sizes are relevant. Rest are placeholders with bogus values.
    spline_conf = dim2::config_t<double>(nx_r,nu_r, 1, 1, 1, 1, x_min, x_max, u_min, u_max, -1, 1, -1, 1, &bogus); 
    size_t stride_spline = (spline_conf.Nx + order - 1) * (spline_conf.Ny + order - 1);
    std::vector<double> map_values_x(nx_r*nu_r, 0);
    std::vector<double> map_values_v(nx_r*nu_r, 0);
    restart_counter = 0;

    std::unique_ptr<double[]> coeffs { new double[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<double[]> coeffs_restart { new double[ (nt_restart+1)*stride_t ] {} };
    std::unique_ptr<double,decltype(std::free)*> rho { reinterpret_cast<double*>(std::aligned_alloc(64,sizeof(double)*conf.Nx)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};

    poisson<double> poiss( conf );

    std::cout << nx_r << " " << nu_r << std::endl;
    
    std::ofstream stat_file( "stats.txt" );
    std::ofstream stat_full_file( "stats_full.txt" );
    std::ofstream coeff_str("coeff_restart.txt");
    std::ofstream coeff_r_str("coeff_r_restart.txt");
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

        if(n % (5*16) == 0){
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

            char_map_x_coeff[restart_counter + 1] = std::vector<double>(stride_spline,0);
            char_map_v_coeff[restart_counter + 1] = std::vector<double>(stride_spline,0);

            #pragma omp parallel for
    		for(size_t i = 0; i < nx_r; i++ ){
    			for(size_t j = 0; j < nu_r; j++){
                    size_t l = i + j*nx_r;
    				double x = i*dx_r;
    				double u = conf.u_min + j*du_r;

                    periodic::eval_char_map<double,order>(nt_r_curr,x,u,coeffs_restart.get(),conf);

                    map_values_x[l] = x;
                    map_values_v[l] = u;
    			}
    		}

            dim2::interpolate<double,order>(char_map_x_coeff[restart_counter+1].data(), map_values_x.data(),spline_conf);
            dim2::interpolate<double,order>(char_map_v_coeff[restart_counter+1].data(), map_values_v.data(),spline_conf);

            conf = config_t<double>(Nx, Nu, Nt, dt, x_min, x_max, u_min, u_max, &eval_f_cmm_spline<order>);

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
	//nufi::dim1::cmm_nufi_linear<4>();
    nufi::dim1::cmm_nufi_spline<4>();

    return 0;
}

