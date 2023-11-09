/*
 * Copyright (C) 2022 Matthias Kirchhart and Paul Wilhelm
 *
 * This file is part of Der Gerät, a solver for the Vlasov–Poisson equation.
 *
 * Der Gerät is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3, or (at your option) any later
 * version.
 *
 * Der Gerät is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * Der Gerät; see the file COPYING.  If not see http://www.gnu.org/licenses.
 */

#include <cmath>
#include <memory>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <random>
#include <math.h>
#include <vector>

#include <armadillo>

#include <dergeraet/config.hpp>
#include <dergeraet/random.hpp>
#include <dergeraet/fields.hpp>
#include <dergeraet/poisson.hpp>
#include <dergeraet/rho.hpp>
#include <dergeraet/stopwatch.hpp>
#include <dergeraet/cuda_scheduler.hpp>
#include <dergeraet/finite_difference_poisson.hpp>

namespace dergeraet
{

namespace dim1
{

namespace dirichlet
{

template <typename real, size_t order>
void test()
{
	config_t<real> conf;
	conf.dt = conf.lambda/conf.c * 5 * 1e-3;
	conf.Nt = 100/conf.dt;
	//conf.Nt = 0;
	conf.Nu_electron = 128;
	conf.Nu_ion = conf.Nu_electron;
    conf.u_electron_min = -400*conf.m_e*conf.c;
    conf.u_electron_max =  400*conf.m_e*conf.c;
    conf.u_ion_min = -0.0003*conf.m_i*conf.c;
    conf.u_ion_max =  0.0003*conf.m_i*conf.c;
    conf.du_electron = (conf.u_electron_max - conf.u_electron_min)/conf.Nu_electron;
    conf.du_ion = (conf.u_ion_max - conf.u_ion_min)/conf.Nu_ion;

    const size_t stride_x = 1;
    const size_t stride_t = conf.Nx + order - 2; // conf.Nx = l+2, therefore: conf.Nx +order-2 = order+l

    std::unique_ptr<real[]> coeffs { new real[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*conf.Nx)), std::free };
    std::unique_ptr<real,decltype(std::free)*> rho_dir { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*(conf.Nx+1))), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};
    std::unique_ptr<real,decltype(std::free)*> phi_ext { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*stride_t)), std::free };//rename, this is the rhs of the interpolation task, but with 2 additional entries.


    poisson_fd_dirichlet<double> poiss(conf);

    //cuda_scheduler<real,order> sched { conf }; // Unused atm.!

    std::cout << "u_electron_min = " <<  conf.u_electron_min << std::endl;
    std::cout << "u_electron_max = " <<  conf.u_electron_max << std::endl;
    std::cout << "Nu_electron = " << conf.Nu_electron <<std::endl;
    std::cout << "du_electron = " << conf.du_electron <<std::endl;
    std::cout << "u_ion_min = " <<  conf.u_ion_min << std::endl;
    std::cout << "u_ion_max = " <<  conf.u_ion_max << std::endl;
    std::cout << "Nu_ion = " << conf.Nu_ion <<std::endl;
    std::cout << "du_ion = " << conf.du_ion <<std::endl;
    std::cout << "x_min = " << conf.x_min << " x_max = " << conf.x_max << std::endl;
    std::cout << conf.Nx <<std::endl;
    std::cout << conf.dt << std::endl;
    std::cout << conf.Nt <<std::endl;

    std::cout << "nc = " << conf.n_c << std::endl;

    double total_time = 0;


    for ( size_t n = 0; n <= conf.Nt; n++ )
    {
        dergeraet::stopwatch<double> timer;
        std::memset( rho.get(), 0, conf.Nx*sizeof(real) );

    	// Compute rho:
    	//#pragma omp parallel for
    	for(size_t i = 0; i<conf.Nx; i++)
    	{
    		rho.get()[i] = eval_rho_ion<real,order>(n, i, coeffs.get(), conf)
    					  - eval_rho_electron<real,order>(n, i, coeffs.get(), conf);
    	}

    	// Set rho_dir:
    	rho_dir.get()[0] = 0; // Left phi value.
    	for(size_t i = 1; i < conf.Nx; i++)
    	{
    		rho_dir.get()[i] = rho.get()[i];
    	}
    	rho_dir.get()[conf.Nx] = 0; // Left phi value.

    	// Solve for phi:
    	poiss.solve(rho_dir.get());

        // this is the value to the left of a.
        phi_ext.get()[0]=0;

        // this is the value to the right of b.
        for(size_t i = 0; i<order-3;i++){
            phi_ext.get()[stride_t-1-i] = 0;
        }
        //these are all values in [a,b]
        for(size_t i = 1; i<conf.Nx+1;i++){
            phi_ext.get()[i] = rho_dir.get()[i-1];

        }

        dim1::dirichlet::interpolate<real,order>( coeffs.get() + n*stride_t,phi_ext.get(), conf );
        //sched.upload_phi( n, coeffs.get() ); // Unused atm.!
        double time_elapsed = timer.elapsed();
        total_time += time_elapsed;

        std::cout << "Iteration " << n << " Time for step: " << time_elapsed << " and total time s.f.: " << total_time << std::endl;

        // Output
        if(n % 10 == 0)
        {
        	real x_min_plot = -2;
        	real x_max_plot = 18;
        	size_t plot_x = 256;
        	size_t plot_u = 256;
        	real dx_plot = (x_max_plot - x_min_plot) / plot_x;
        	real du_electron_plot = (conf.u_electron_max - conf.u_electron_min) / plot_u;
        	real du_ion_plot = (conf.u_ion_max - conf.u_ion_min) / plot_u;

        	real t = n*conf.dt;
			std::ofstream f_electron_file("f_electron_"+ std::to_string(t) + ".txt");
			for(size_t i = 0; i <= plot_x; i++)
			{
				for(size_t j = 0; j <= plot_u; j++)
				{
					double x = x_min_plot + i*dx_plot;
					double u = conf.u_electron_min + j*du_electron_plot;
					double f = eval_f_ion_acoustic<real,order>(n, x, u, coeffs.get(), conf);
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
					double u = conf.u_ion_min + j*du_ion_plot;
					double f = eval_f_ion_acoustic<real,order>(n, x, u, coeffs.get(), conf, false);
					f_ion_file << x << " " << u << " " << f << std::endl;
				}
				f_ion_file << std::endl;
			}

			std::ofstream E_file("E_"+ std::to_string(t) + ".txt");
			std::ofstream rho_electron_file("rho_electron_"+ std::to_string(t) + ".txt");
			std::ofstream rho_ion_file("rho_ion_"+ std::to_string(t) + ".txt");
			std::ofstream phi_file("phi_"+ std::to_string(t) + ".txt");
			for(size_t i = 0; i <= plot_x; i++)
			{
				real x = x_min_plot + i*dx_plot;
				real E = -dim1::dirichlet::eval<real,order,1>( x, coeffs.get() + n*stride_t, conf );
				real phi = dim1::dirichlet::eval<real,order>( x, coeffs.get() + n*stride_t, conf );
				real rho_electron = eval_rho_electron<real,order>(n, x, coeffs.get(), conf);
				real rho_ion = eval_rho_ion<real,order>(n, x, coeffs.get(), conf);

				E_file << x << " " << E << std::endl;
				rho_electron_file << x << " " << rho_electron << std::endl;
				rho_ion_file << x << " " << rho_ion << std::endl;
				phi_file << x << " " << phi << std::endl;
			}
        }

    }

}

}
}
}

int main()
{
    dergeraet::dim1::dirichlet::test<double,4>();
}


