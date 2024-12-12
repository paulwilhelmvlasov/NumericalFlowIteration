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
real f0_electron(real x, real u) noexcept
{
    real M_r = 1000;
    real U_e = -2;
	real alpha = 0.5;
	real k = 0.5;
	return 1.0 / std::sqrt(2.0 * M_PI) * exp(-0.5*(u-U_e)*(u-U_e)) * (1 + alpha * cos(k*x));
}

template <typename real>
real f0_ion(real x, real u) noexcept
{
    real M_r = 1000;
	real alpha = 0;
	real k = 0.5;
    return std::sqrt( M_r / (2.0 * M_PI)) * exp(-0.5*M_r*u*u) * (1 + alpha * cos(k*x));
}

template <typename real>
real lin_interpol(real x , real y, real x1, real x2, real y1, real y2, real f_11,
		real f_12, real f_21, real f_22) noexcept
{
	real f_x_y1 = ((x2-x)*f_11 + (x-x1)*f_21)/(x2-x1);
	real f_x_y2 = ((x2-x)*f_12 + (x-x1)*f_22)/(x2-x1);

	return ((y2-y)*f_x_y1 + (y-y1)*f_x_y2) / (y2-y1);
}

bool debug = false;
bool write = false;

arma::mat f0_r_electron;
arma::mat f0_r_ion;
config_t<double> conf_electron(64, 128, 500, 0.1, 0, 4*M_PI, -10, 10, &f0_electron);
config_t<double> conf_ion(64, 128, 500, 0.1, 0, 4*M_PI, -10, 10, &f0_ion);

double f_t_electron(double x, double u) noexcept
{
	if(u > conf_electron.u_max || u < conf_electron.u_min){
		return 0;
	}

	size_t nx_r = f0_r_electron.n_rows - 1;
	size_t nu_r = f0_r_electron.n_cols - 1;

	double dx_r = conf_electron.Lx/ nx_r;
	double du_r = (conf_electron.u_max - conf_electron.u_min)/nu_r;

	// Compute periodic reference position of x. Assume x_min = 0 to this end.
     if(x < 0 || x > conf_electron.x_max){
	    x -= conf_electron.Lx * std::floor(x*conf_electron.Lx_inv);
    } 

	size_t x_ref_pos = std::floor(x/dx_r);
    x_ref_pos = x_ref_pos % nx_r;
	size_t u_ref_pos = std::floor((u-conf_electron.u_min)/du_r);

	double x1 = x_ref_pos*dx_r;
	double x2 = x1+dx_r;
	double u1 = conf_electron.u_min + u_ref_pos*du_r;
	double u2 = u1 + du_r;

	double f_11 = f0_r_electron(x_ref_pos, u_ref_pos);
	double f_21 = f0_r_electron(x_ref_pos+1, u_ref_pos);
	double f_12 = f0_r_electron(x_ref_pos, u_ref_pos+1);
	double f_22 = f0_r_electron(x_ref_pos+1, u_ref_pos+1);

    double value = lin_interpol<double>(x, u, x1, x2, u1, u2, f_11, f_12, f_21, f_22);

    return value;
}

double f_t_ion(double x, double u) noexcept
{
	if(u > conf_ion.u_max || u < conf_ion.u_min){
		return 0;
	}

	size_t nx_r = f0_r_ion.n_rows - 1;
	size_t nu_r = f0_r_ion.n_cols - 1;

	double dx_r = conf_ion.Lx/ nx_r;
	double du_r = (conf_ion.u_max - conf_ion.u_min)/nu_r;

	// Compute periodic reference position of x. Assume x_min = 0 to this end.
     if(x < 0 || x > conf_ion.x_max){
	    x -= conf_ion.Lx * std::floor(x*conf_ion.Lx_inv);
    } 

	size_t x_ref_pos = std::floor(x/dx_r);
    x_ref_pos = x_ref_pos % nx_r;
	size_t u_ref_pos = std::floor((u-conf_ion.u_min)/du_r);

	double x1 = x_ref_pos*dx_r;
	double x2 = x1+dx_r;
	double u1 = conf_ion.u_min + u_ref_pos*du_r;
	double u2 = u1 + du_r;

	double f_11 = f0_r_ion(x_ref_pos, u_ref_pos);
	double f_21 = f0_r_ion(x_ref_pos+1, u_ref_pos);
	double f_12 = f0_r_ion(x_ref_pos, u_ref_pos+1);
	double f_22 = f0_r_ion(x_ref_pos+1, u_ref_pos+1);

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

    size_t Nx = 256;  // Number of grid points in physical space. Same for both species!
    size_t Nu_electron = 1024;
    size_t Nu_ion = 1024;
    double dt = 0.0625;  // Time-step size.
    size_t Nt = 5000/dt;  // Number of time-steps.

    // Dimensions of physical domain.
    double x_min = 0;
    double x_max = 4*M_PI;

    // Integration limits for velocity space.
    double u_min_electron = -10;
    double u_max_electron = 10;
    double u_min_ion = -0.4;
    double u_max_ion = 0.4;

    conf_electron = config_t<double>(Nx, Nu_electron, Nt, dt, x_min, x_max, 
                                        u_min_electron, u_max_electron, &f0_electron);
    conf_ion = config_t<double>(Nx, Nu_ion, Nt, dt, x_min, x_max, u_min_ion, 
                                        u_max_ion, &f0_ion);

    const size_t stride_t = Nx + order - 1;

    // Restart parameters.
    size_t nx_r = 2048;
	size_t nu_r = nx_r;
    size_t nt_restart = 200;
    double dx_r = conf_electron.Lx / nx_r;
    double du_r_electron = (conf_electron.u_max - conf_electron.u_min)/ nu_r;
    double du_r_ion = (conf_ion.u_max - conf_ion.u_min)/ nu_r;
    f0_r_electron.resize(nx_r+1, nu_r+1);
    f0_r_ion.resize(nx_r+1, nu_r+1);

    std::unique_ptr<double[]> coeffs_restart { new double[ (nt_restart+1)*stride_t ] {} };
    std::unique_ptr<double,decltype(std::free)*> rho { reinterpret_cast<double*>(std::aligned_alloc(64,sizeof(double)*Nx)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};

    poisson<double> poiss( conf_electron );


    std::ofstream stat_file( "stats.txt" );
    std::ofstream coeff_str("coeff_restart.txt");
    double total_time = 0;
    size_t restart_counter = 0;
    size_t nt_r_curr = 0;
    for ( size_t n = 0; n <= Nt; ++n )
    {
    	nufi::stopwatch<double> timer;

        std::cout << " start of time step "<< n << " " << nt_r_curr  << std::endl; 

        // Compute rho:
		#pragma omp parallel for
    	for(size_t i = 0; i < Nx; i++)
    	{
            double rho_electron = periodic::eval_rho_single_species<double,order>(nt_r_curr, i, 
                                            coeffs_restart.get(), conf_electron, true);
            double rho_ion = periodic::eval_rho_single_species<double,order>(nt_r_curr, i, 
                                            coeffs_restart.get(), conf_ion, false);
    		rho.get()[i] = rho_ion - rho_electron;
    	}

        poiss.solve( rho.get() );

        // Interpolation of Poisson solution.
        periodic::interpolate<double,order>( coeffs_restart.get() + nt_r_curr*stride_t, rho.get(), conf_electron );

        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;

        coeff_str << n << std::endl;
        for(size_t i = 0; i < stride_t; i++){
            coeff_str << i << " " << coeffs_restart.get()[nt_r_curr*stride_t + i] << std::endl; 
        }

        double Emax = 0;
	    double electric_energy = 0; // Electric energy
        size_t plot_n_x = 256;
        double dx_plot = conf_electron.Lx / plot_n_x;
        for ( size_t i = 0; i < plot_n_x; ++i )
        {
            double x = x_min + i*dx_plot;
            double E_abs = abs( periodic::eval<double,order,1>(x,coeffs_restart.get()+nt_r_curr*stride_t,conf_electron));
            Emax = max( Emax, E_abs );
	        electric_energy += E_abs*E_abs;
        }
	    electric_energy =  0.5*dx_plot*electric_energy;

	    double t = n*dt;
        stat_file << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << Emax  << " " << electric_energy << std::endl;
        std::cout << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << Emax << " Comp-time: " << timer_elapsed;
        std::cout << " Total comp time s.f.: " << total_time << std::endl; 

        if(nt_r_curr == nt_restart)
    	{
            std::cout << "Restart" << std::endl;
            nufi::stopwatch<double> timer_restart;
            arma::mat f0_r_copy(nx_r + 1, nu_r + 1);
            // Restart ions.
            #pragma omp parallel for
    		for(size_t i = 0; i <= nx_r; i++ ){
    			for(size_t j = 0; j <= nu_r; j++){
    				double x = i*dx_r;
    				double u = conf_ion.u_min + j*du_r_ion;

                    double f = periodic::eval_f<double,order>(nt_r_curr,x,u,coeffs_restart.get(),conf_ion,false);

                    f0_r_copy(i,j) = f;
    			}
    		}

            f0_r_ion = f0_r_copy;
            conf_ion = config_t<double>(Nx, Nu_ion, Nt, dt, x_min, x_max, u_min_ion, u_max_ion, &f_t_ion);

            // Restart electrons.
            #pragma omp parallel for
    		for(size_t i = 0; i <= nx_r; i++ ){
    			for(size_t j = 0; j <= nu_r; j++){
    				double x = i*dx_r;
    				double u = conf_electron.u_min + j*du_r_electron;

                    double f = periodic::eval_f<double,order>(nt_r_curr,x,u,coeffs_restart.get(),conf_electron,true);

                    f0_r_copy(i,j) = f;
    			}
    		}

            f0_r_electron = f0_r_copy;
            conf_electron = config_t<double>(Nx, Nu_electron, Nt, dt, x_min, x_max, u_min_electron, u_max_electron, &f_t_electron);

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

        if(n % (50*16) == 0){
            size_t nx_plot = 512;
            size_t nv_plot = nx_plot;
            double dx_plot = (x_max - x_min)/nx_plot;
            double dv_plot_electron = (u_max_electron - u_min_electron) / nv_plot;
            double dv_plot_ion = (u_max_ion - u_min_ion) / nv_plot;

            std::ofstream f_electron_str("f_electron_" + std::to_string(t) + ".txt");
            std::ofstream f_ion_str("f_ion_" + std::to_string(t) + ".txt");
            for(size_t i = 0; i <= nx_plot; i++){
                for(size_t j = 0; j <= nv_plot; j++){
                    double x = i*dx_plot;
                    double u_elec = u_min_electron + j*dv_plot_electron;
                    double u_ion = u_min_ion + j*dv_plot_ion;

                    double f_elec = periodic::eval_f<double,order>(nt_r_curr,x,u_elec,coeffs_restart.get(),conf_electron,true); 
                    double f_ion = periodic::eval_f<double,order>(nt_r_curr,x,u_ion,coeffs_restart.get(),conf_ion,false); 

                    f_electron_str << x << " " << u_elec << " " << f_elec << std::endl;
                    f_ion_str << x << " " << u_ion << " " << f_ion << std::endl;
                }
                f_electron_str << std::endl;
                f_ion_str << std::endl;
            } 
        }
    }
    
    std::cout << "Total time: " << total_time << std::endl; 
}


}
}



int main()
{
	nufi::dim1::run_restarted_simulation<4>();
}