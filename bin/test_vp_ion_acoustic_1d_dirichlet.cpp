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
#include <iomanip>
#include <fstream>
#include <sstream>
#include <random>
#include <math.h>
#include <thread>
#include <vector>

#include <armadillo>

#include <nufi/config.hpp>
#include <nufi/random.hpp>
#include <nufi/fields.hpp>
#include <nufi/poisson.hpp>
#include <nufi/rho.hpp>
#include <nufi/stopwatch.hpp>
#include <nufi/finite_difference_poisson.hpp>

namespace nufi
{

namespace dim1
{

namespace dirichlet
{


arma::mat f0_r_electron;
arma::mat f0_r_ion;
config_t<double> conf(&f0_electron, &f0_ion);


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

double f_t_electron(double x, double u) noexcept
{
	if(u > conf.u_electron_max || u < conf.u_electron_min){
		return 0;
	}

	size_t nx_r = f0_r_electron.n_rows - 1;
	size_t nu_r = f0_r_electron.n_cols - 1;

	double dx_r = conf.Lx / nx_r;
	double du_r = (conf.u_electron_max - conf.u_electron_min)/nu_r;

	size_t x_ref_pos = std::floor((x-conf.x_min)/dx_r);
    x_ref_pos = x_ref_pos % nx_r;
	size_t u_ref_pos = std::floor((u-conf.u_electron_min)/du_r);

	double x1 = x_ref_pos*dx_r;
	double x2 = x1+dx_r;
	double u1 = conf.u_electron_min + u_ref_pos*du_r;
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
	if(u > conf.u_ion_max || u < conf.u_ion_min){
		return 0;
	}

	size_t nx_r = f0_r_ion.n_rows - 1;
	size_t nu_r = f0_r_ion.n_cols - 1;

	double dx_r = conf.Lx / nx_r;
	double du_r = (conf.u_ion_max - conf.u_ion_min)/nu_r;

	size_t x_ref_pos = std::floor((x-conf.x_min)/dx_r);
    x_ref_pos = x_ref_pos % nx_r;
	size_t u_ref_pos = std::floor((u-conf.u_ion_min)/du_r);

	double x1 = x_ref_pos*dx_r;
	double x2 = x1+dx_r;
	double u1 = conf.u_ion_min + u_ref_pos*du_r;
	double u2 = u1 + du_r;

	double f_11 = f0_r_ion(x_ref_pos, u_ref_pos);
	double f_21 = f0_r_ion(x_ref_pos+1, u_ref_pos);
	double f_12 = f0_r_ion(x_ref_pos, u_ref_pos+1);
	double f_22 = f0_r_ion(x_ref_pos+1, u_ref_pos+1);

    double value = lin_interpol<double>(x, u, x1, x2, u1, u2, f_11, f_12, f_21, f_22);

    return value;
}

template <typename real, size_t order>
void nufi_two_species_ion_acoustic_with_reflecting_dirichlet_boundary(bool plot = true)
{
	conf.dt = 0.05;
	conf.Nt = 1000/conf.dt;
	conf.Nu_electron = 64;
	conf.Nu_ion = 1024;
    conf.u_electron_min = -50;
    conf.u_electron_max =  50;
    conf.u_ion_min = -0.7;
    conf.u_ion_max =  0.7;
    conf.du_electron = (conf.u_electron_max - conf.u_electron_min)/conf.Nu_electron;
    conf.du_ion = (conf.u_ion_max - conf.u_ion_min)/conf.Nu_ion;
    conf.Nx = 2048;
    conf.l = conf.Nx - 1;
    conf.dx = (conf.x_max - conf.x_min)/conf.Nx;
    conf.dx_inv = 1.0/conf.dx;

    const size_t stride_x = 1;
    const size_t stride_t = conf.Nx + order - 2; // conf.Nx = l+2, therefore: conf.Nx +order-2 = order+l

    // Restart parameters.
    size_t nx_r = 8192;
	size_t nu_r = nx_r;
    size_t nt_restart = 400;
    double dx_r = conf_electron.Lx / nx_r;
    double du_r_electron = (conf_electron.u_max - conf_electron.u_min)/ nu_r;
    double du_r_ion = (conf_ion.u_max - conf_ion.u_min)/ nu_r;
    f0_r_electron.resize(nx_r+1, nu_r+1);
    f0_r_ion.resize(nx_r+1, nu_r+1);

    std::unique_ptr<double[]> coeffs_restart { new double[ (nt_restart+1)*stride_t ] {} };
    std::unique_ptr<real[]> coeffs { new real[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*conf.Nx)), std::free };
    std::unique_ptr<real,decltype(std::free)*> rho_dir { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*(conf.Nx+1))), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};
    std::unique_ptr<real,decltype(std::free)*> phi_ext { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*stride_t)), std::free };//rename, this is the rhs of the interpolation task, but with 2 additional entries.


    //poisson_fd_dirichlet<double> poiss(conf);
    poisson_fd_pure_neumann_dirichlet poiss(conf);

    //cuda_scheduler<real,order> sched { conf };

    std::cout << "u_electron_min = " <<  conf.u_electron_min << std::endl;
    std::cout << "u_electron_max = " <<  conf.u_electron_max << std::endl;
    std::cout << "Nu_electron = " << conf.Nu_electron <<std::endl;
    std::cout << "du_electron = " << conf.du_electron <<std::endl;
    std::cout << "u_ion_min = " <<  conf.u_ion_min << std::endl;
    std::cout << "u_ion_max = " <<  conf.u_ion_max << std::endl;
    std::cout << "Nu_ion = " << conf.Nu_ion <<std::endl;
    std::cout << "du_ion = " << conf.du_ion <<std::endl;
    std::cout << "x_min = " << conf.x_min << " x_max = " << conf.x_max << std::endl;
    std::cout << "Nx = " << conf.Nx <<std::endl;
    std::cout << "dt = " << conf.dt << std::endl;
    std::cout << "Nt = " << conf.Nt <<std::endl;

    arma::vec rho_tot(conf.Nx,arma::fill::zeros);
    arma::vec rho_e(conf.Nx,arma::fill::zeros);
    arma::vec rho_i(conf.Nx,arma::fill::zeros);
    arma::vec rho_phi(conf.Nx+1,arma::fill::zeros);

    double total_time = 0;

    std::ofstream stats_file( "stats.txt" );
    std::ofstream coeffs_str( "coeffs_Nt_" + std::to_string(conf.dt) + "_Nx_ "
				+ std::to_string(conf.Nx) + "_stride_t_" + std::to_string(stride_t) + ".txt" );

	size_t restart_counter = 0;
    size_t nt_r_curr = 0;
    for ( size_t n = 0; n <= conf.Nt; n++ )
    {
		std::cout << " start of time step "<< n << " " << nt_r_curr  << std::endl; 
        nufi::stopwatch<double> timer;

    	//Compute rho:
    	#pragma omp parallel for
    	for(size_t i = 0; i<conf.Nx; i++)
    	 {
    		real x = conf.x_min + i*conf.dx;
    		rho_e(i) = eval_rho_adaptive_trapezoidal_rule<real,order>(n,x,coeffs.get(),conf,conf.u_electron_min,
					conf.u_electron_max, true, true, false);
    		rho_i(i) = eval_rho_adaptive_trapezoidal_rule<real,order>(n,x,coeffs.get(),conf,conf.u_ion_min,
					conf.u_ion_max, false, true, false);

    		rho_tot(i) = rho_i(i) - rho_e(i);
    		rho_phi(i) = rho_tot(i);
    		rho.get()[i] = rho_tot(i);
    	 }

		//std::memset( rho.get(), 0, conf.Nx*sizeof(real) );
        //sched.compute_rho ( n, 0, conf.Nx*conf.Nu_electron ); // Works only if Nu_i=Nu_e !!!
        //sched.download_rho( rho.get() );

    	// Set rho_dir:
    	/*
    	rho_dir.get()[0] = 0; // Left phi value.
    	for(size_t i = 1; i < conf.Nx; i++)
    	{
    		rho_dir.get()[i] = rho.get()[i];
    	}
    	rho_dir.get()[conf.Nx] = 0; // Left phi value.
    	*/

    	// Solve for phi:
    	//poiss.solve(rho_dir.get());
    	poiss.solve(rho_phi);


        // this is the value to the left of a.
        phi_ext.get()[0]=rho_phi(0);

        // this is the value to the right of b.
        for(size_t i = 0; i<order-3;i++){
            phi_ext.get()[stride_t-1-i] = rho_phi(conf.Nx);
        }
        //these are all values in [a,b]
        for(size_t i = 1; i<conf.Nx+1;i++){
            //phi_ext.get()[i] = rho_dir.get()[i-1];
        	phi_ext.get()[i] = rho_phi(i-1);
        }

        dim1::dirichlet::interpolate<real,order>( coeffs.get() + n*stride_t,phi_ext.get(), conf );
        // Output coeffs to reproduce results later.
        for(size_t i = 0; i < stride_t; i++){
        	coeffs_str << coeffs.get()[n*stride_t + i] << std::endl;
        }

        //sched.upload_phi( n, coeffs.get() );
        double time_elapsed = timer.elapsed();
        total_time += time_elapsed;

        std::cout << "t = " << n*conf.dt << " Iteration " << n << " Time for step: " << time_elapsed << " and total time s.f.: " << total_time << std::endl;

		// Restart
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

                    double f = eval_f_ion_acoustic<double,order>(nt_r_curr,x,u,coeffs_restart.get(),conf,false);

                    f0_r_copy(i,j) = f;
    			}
    		}

            f0_r_ion = f0_r_copy;

            // Restart electrons.
            #pragma omp parallel for
    		for(size_t i = 0; i <= nx_r; i++ ){
    			for(size_t j = 0; j <= nu_r; j++){
    				double x = i*dx_r;
    				double u = conf_electron.u_min + j*du_r_electron;

                    double f = eval_f_ion_acoustic<double,order>(nt_r_curr,x,u,coeffs_restart.get(),conf,true);

                    f0_r_copy(i,j) = f;
    			}
    		}

            f0_r_electron = f0_r_copy;
            conf = config_t<double>(&f_t_electron, &f_t_ion);

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

        // Output
        real t = n*conf.dt;
        real x_min_plot = conf.x_min;
        real x_max_plot = conf.x_max;
        real u_min_electron_plot = conf.u_electron_min;
        real u_max_electron_plot = conf.u_electron_max;
        real u_min_ion_plot = conf.u_ion_min;
        real u_max_ion_plot = conf.u_ion_max;
        size_t plot_x = 512;
        size_t plot_u = 512;

        real dx_plot = (x_max_plot - x_min_plot) / plot_x;
        real du_electron_plot = (u_max_electron_plot - u_min_electron_plot) / plot_u;
        real du_ion_plot = (u_max_ion_plot - u_min_ion_plot) / plot_u;

        if(n % 2 == 0)
        {
        	real E_l2 = 0;
        	real E_max = 0;
			for(size_t i = 0; i < plot_x; i++)
			{
				real x = x_min_plot + (i+0.5)*dx_plot;
				//real E = -dim1::dirichlet::eval<real,order,1>( x, coeffs.get() + n*stride_t, conf );
				real E = -dim1::dirichlet::eval_E<real,order>( x, coeffs.get() + n*stride_t, conf );

				E_max = std::max( E_max, abs(E) );
				E_l2 += E*E;
			}
			E_l2 *=  dx_plot;
			stats_file << std::setw(20) << t << std::setw(20) << std::setprecision(8)
					   << std::scientific << E_max << std::setw(20)
					   << std::setprecision(8) << std::scientific << E_l2
					   << std::endl;


			if(n % (20*5) == 0 && plot)
			{
				/*
				std::ofstream f_electron_file("f_electron_"+ std::to_string(t) + ".txt");
				for(size_t i = 0; i <= plot_x; i++)
				{
					for(size_t j = 0; j <= plot_u; j++)
					{
						double x = x_min_plot + i*dx_plot;
						double u = u_min_electron_plot + j*du_electron_plot;
						double f = eval_f_ion_acoustic<real,order>(n, x, u, coeffs.get(), conf,
									true, true, false);
						f_electron_file << x << " " << u << " " << f << std::endl;
					}
					f_electron_file << std::endl;
				}
				std::ofstream f_ion_file("f_ion_"+ std::to_string(t) + ".txt");
				for(size_t i = 0; i <= plot_x; i++)
				{
					for(size_t j = 0; j <= plot_u; j++)
					{
						double x = x_min_plot + i*dx_plot;
						double u = u_min_ion_plot + j*du_ion_plot;
						double f = eval_f_ion_acoustic<real,order>(n, x, u, coeffs.get(), conf,
									false, true, false);
						f_ion_file << x << " " << u << " " << f << std::endl;
					}
					f_ion_file << std::endl;
				}
				*/

				std::ofstream E_file("E_"+ std::to_string(t) + ".txt");
				std::ofstream rho_electron_file("rho_electron_"+ std::to_string(t) + ".txt");
				std::ofstream rho_ion_file("rho_ion_"+ std::to_string(t) + ".txt");
				std::ofstream rho_file("rho_"+ std::to_string(t) + ".txt");
				std::ofstream phi_file("phi_"+ std::to_string(t) + ".txt");
				for(size_t i = 0; i <= plot_x; i++)
				{
					real x = x_min_plot + i*dx_plot;
					//real E = -dim1::dirichlet::eval<real,order,1>( x, coeffs.get() + n*stride_t, conf );
					real E = -dim1::dirichlet::eval_E<real,order>( x, coeffs.get() + n*stride_t, conf );
					real phi = dim1::dirichlet::eval<real,order>( x, coeffs.get() + n*stride_t, conf );
					//real rho_electron = eval_rho_electron<real,order>(n, x, coeffs.get(), conf);
					//real rho_ion = eval_rho_ion<real,order>(n, x, coeffs.get(), conf);
					//real rho = rho_ion - rho_electron;

					E_file << x << " " << E << std::endl;
					//rho_electron_file << x << " " << rho_electron << std::endl;
					//rho_ion_file << x << " " << rho_ion << std::endl;
					//rho_file << x << " " << rho << std::endl;
					phi_file << x << " " << phi << std::endl;
				}

				for(size_t i = 0; i < conf.Nx; i++)
				{
					real x = x_min_plot + i*conf.dx;
					rho_electron_file << x << " " << rho_e(i) << std::endl;
					rho_ion_file << x << " " << rho_i(i) << std::endl;
					rho_file << x << " " << rho_tot(i) << std::endl;
				}
			}
        }
    }
}

}
}
}

int main()
{
	nufi::dim1::dirichlet::nufi_two_species_ion_acoustic_with_reflecting_dirichlet_boundary<double,4>();

	//nufi::dim1::dirichlet::read_coeff_and_plot<double,4>(std::string("../coeffs_Nt_0.050000_Nx_ 2048_stride_t_2050.txt"));

}


