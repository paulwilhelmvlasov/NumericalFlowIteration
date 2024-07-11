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
#include <nufi/cuda_scheduler.hpp>
#include <nufi/finite_difference_poisson.hpp>

namespace nufi
{

namespace dim1
{

namespace dirichlet
{

template <typename real, size_t order>
void read_coeff_and_plot(const std::string& i_str)
{
	config_t<real> conf;

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

    std::unique_ptr<real[]> coeffs {new real[ (conf.Nt+1)*stride_t ] {} };

    size_t Nt_plot = 100*20;

	std::ifstream coeff_str(i_str);
	for(size_t i = 0; i <= Nt_plot*stride_t; i++){
		coeff_str >> coeffs[i];
	}

	// Plotting:
    for ( size_t n = 0; n <= Nt_plot; n+=20 )
    {
    	real t = n*conf.dt;

    	std::cout << "Plot t = " << t << "." << std::endl;

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

        auto loop1 = [&]() {
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
        };
        auto loop2 = [&]() {
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
        };

        auto loop3 = [&]() {
			std::ofstream phi_file("phi_"+ std::to_string(t) + ".txt");
			for(size_t i = 0; i <= plot_x; i++)
			{
				real x = x_min_plot + i*dx_plot;
				real phi = dim1::dirichlet::eval<real,order>( x, coeffs.get() + n*stride_t, conf );

				phi_file << x << " " << phi << std::endl;
			}
        };

        auto loop4 = [&]() {
			std::ofstream E_file("E_"+ std::to_string(t) + ".txt");
			for(size_t i = 0; i <= plot_x; i++)
			{
				real x = x_min_plot + i*dx_plot;
				real E = -dim1::dirichlet::eval_E<real,order>( x, coeffs.get() + n*stride_t, conf );

				E_file << x << " " << E << std::endl;
			}
        };

        auto loop5 = [&]() {
			std::ofstream rho_ion_file("rho_ion_"+ std::to_string(t) + ".txt");
			size_t plot_x_rho = 2048;
			real plot_dx_rho = (conf.x_max - conf.x_min)/plot_x_rho;
			for(size_t i = 0; i <= plot_x_rho; i++)
			{
				real x = conf.x_min + i*plot_dx_rho;

				real rho_ion = eval_rho_adaptive_trapezoidal_rule<real,order>(n,x,coeffs.get(),conf,conf.u_ion_min,
						conf.u_ion_max, false, true, false);

				rho_ion_file << x << " " << rho_ion << std::endl;
			}
        };

        auto loop6 = [&]() {
			std::ofstream rho_electron_file("rho_electron_"+ std::to_string(t) + ".txt");
			size_t plot_x_rho = 2048;
			real plot_dx_rho = (conf.x_max - conf.x_min)/plot_x_rho;
			for(size_t i = 0; i <= plot_x_rho; i++)
			{
				real x = conf.x_min + i*plot_dx_rho;
				real rho_electron = eval_rho_adaptive_trapezoidal_rule<real,order>(n,x,coeffs.get(),conf,conf.u_electron_min,
						conf.u_electron_max, true, true, false);

				rho_electron_file << x << " " << rho_electron << std::endl;
			}
        };

        std::thread t1(loop1);
        std::thread t2(loop2);
        std::thread t3(loop3);
        std::thread t4(loop4);
        std::thread t5(loop5);
        std::thread t6(loop6);

        t1.join();
        t2.join();
        t3.join();
        t4.join();
        t5.join();
        t6.join();
    }
}

template <typename real, size_t order>
void nufi_two_species_ion_acoustic_with_reflecting_dirichlet_boundary()
{
	config_t<real> conf;

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
    for ( size_t n = 0; n <= conf.Nt; n++ )
    {
        nufi::stopwatch<double> timer;

    	//Compute rho:
    	#pragma omp parallel for
    	for(size_t i = 0; i<conf.Nx; i++)
    	 {
    		/*
    	 	rho.get()[i] = eval_rho_ion<real,order>(n, i, coeffs.get(), conf)
    	 				  - eval_rho_electron<real,order>(n, i, coeffs.get(), conf);
    		*/
    		real x = conf.x_min + i*conf.dx;
    		rho_e(i) = eval_rho_adaptive_trapezoidal_rule<real,order>(n,x,coeffs.get(),conf,conf.u_electron_min,
					conf.u_electron_max, true, true, false);
    		rho_i(i) = eval_rho_adaptive_trapezoidal_rule<real,order>(n,x,coeffs.get(),conf,conf.u_ion_min,
					conf.u_ion_max, false, true, false);

    		//rho_i(i) = 1;
    		rho_tot(i) = rho_i(i) - rho_e(i);
    		rho_phi(i) = rho_tot(i);
    		rho.get()[i] = rho_tot(i);
    		/*
    		rho.get()[i] = eval_rho_adaptive_trapezoidal_rule<real,order>(n,x,coeffs.get(),conf,conf.u_ion_min,
    											conf.u_ion_max, false, true, false)
						- eval_rho_adaptive_trapezoidal_rule<real,order>(n,x,coeffs.get(),conf,conf.u_electron_min,
								conf.u_electron_max, true, true, false);
    		 */
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

        if(n%1 == 0)
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


			if(n % (20*5) == 0 && true)
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
	//nufi::dim1::dirichlet::nufi_two_species_ion_acoustic_with_reflecting_dirichlet_boundary<double,4>();

	nufi::dim1::dirichlet::read_coeff_and_plot<double,4>(std::string("../coeffs_Nt_0.050000_Nx_ 2048_stride_t_2050.txt"));

}


