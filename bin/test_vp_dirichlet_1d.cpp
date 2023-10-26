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
    const size_t stride_x = 1;
    const size_t stride_t = conf.Nx + order - 2; // conf.Nx = l+2, therefore: conf.Nx +order-2 = order+l

    std::unique_ptr<real[]> coeffs { new real[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*conf.Nx)), std::free };
    std::unique_ptr<real,decltype(std::free)*> rho_dir { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*(conf.Nx+1))), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};
    std::unique_ptr<real,decltype(std::free)*> phi_ext { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*stride_t)), std::free };//rename, this is the rhs of the interpolation task, but with 2 additional entries.


    poisson_fd_dirichlet<double> poiss(conf);

    cuda_scheduler<real,order> sched { conf };

    std::cout<<conf.Nt<<std::endl;
    double total_time = 0;

    for ( size_t n = 0; n <= conf.Nt; n++ )
    //for ( size_t n = 0; n <= 1; n++ )
    {
        dergeraet::stopwatch<double> timer;
        std::memset( rho.get(), 0, conf.Nx*sizeof(real) );

    	// Compute rho:
    	#pragma omp parallel for
    	for(size_t i = 0; i<conf.Nx; i++)
    	{
    		rho.get()[i] = eval_rho<real,order>(n, i, coeffs.get(), conf);
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
        sched.upload_phi( n, coeffs.get() );
        double time_elapsed = timer.elapsed();
        total_time += time_elapsed;

        std::ofstream f_quer("f_quer_"+ std::to_string(n) + ".txt");
        for(size_t j = 0; j <= conf.Nu; j++)
        {
        	double x = 0;
        	double u = conf.u_min + j*conf.du;
        	double f = eval_f<real,order>(n, x, u, coeffs.get(), conf);
        	f_quer << u << " " << f << std::endl;
        }

        std::ofstream f_file("f_"+ std::to_string(n) + ".txt");
        double f_max = 0;
        for(size_t i = 0; i <= conf.Nx; i++)
        {
            for(size_t j = 0; j <= conf.Nu; j++)
            {
            	double x = conf.x_min + i*conf.dx;
            	double u = conf.u_min + j*conf.du;
            	double f = eval_f<real,order>(n, x, u, coeffs.get(), conf);
            	f_file << x << " " << u << " " << f << std::endl;

            	f_max = std::max(f_max, f);
            }
            f_file << std::endl;
        }

        std::cout << f_max << std::endl;

        std::ofstream E_file("E_"+ std::to_string(n) + ".txt");
        std::ofstream rho_file_new("rho_new_"+ std::to_string(n) + ".txt");
        std::ofstream phi_file("phi_"+ std::to_string(n) + ".txt");
		for(size_t i = 0; i <= conf.Nx; i++)
		{
			real x = conf.x_min + i*conf.dx;
			real E = -dim1::dirichlet::eval<real,order,1>( x, coeffs.get() + n*stride_t, conf );
			real rho = -dim1::dirichlet::eval<real,order,2>( x, coeffs.get() + n*stride_t, conf );
			real phi = dim1::dirichlet::eval<real,order>( x, coeffs.get() + n*stride_t, conf );

			E_file << x << " " << E << std::endl;
			rho_file_new << x << " " << rho << std::endl;
			phi_file << x << " " << phi << std::endl;
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


