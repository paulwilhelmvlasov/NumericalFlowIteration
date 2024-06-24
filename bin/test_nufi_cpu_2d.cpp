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


#include <nufi/config.hpp>
#include <nufi/random.hpp>
#include <nufi/fields.hpp>
#include <nufi/poisson.hpp>
#include <nufi/rho.hpp>
#include <nufi/stopwatch.hpp>

namespace nufi
{

namespace dim2
{

template <typename real>
real f0(real x, real y, real u, real v) noexcept
{
	real alpha = 1e-2;
	real k = 0.5;
    return 1.0 / (2.0 * M_PI) * exp(-0.5 * u*u) * (1 + alpha * cos(k*x));
}


template <typename real, size_t order>
void test()
{
    using std::abs;
    using std::max;

    // Number of grid points in physical space.
    size_t Nx = 32;
    size_t Ny = 32;
    // Number of quadrature points in velocity space.
    size_t Nu = 64;
    size_t Nv = 64;
    real   dt = 0.1;   // Time-step size.
    size_t Nt = 50/dt; // Number of time-steps.

    // Dimensions of physical domain.
    real x_min = 0, x_max = 4*M_PI;
    real y_min = x_min, y_max = x_max;

    // Integration limits for velocity space.
    real u_min = -10, u_max = 10;
    real v_min = -u_min, v_max = -u_max;

    config_t<real> conf(Nx, Ny, Nu, Nv, Nt, dt, x_min, x_max, y_min, y_max,
    					u_min, u_max, v_min, v_max, &f0);
    size_t stride_t = (conf.Nx + order - 1) *
                      (conf.Ny + order - 1);


    std::unique_ptr<real[]> coeffs { new real[ conf.Nt*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,
    													sizeof(real)*conf.Nx*conf.Ny)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};

    poisson<real> poiss( conf );

    std::ofstream Emax_file( "Emax.txt" );
    double total_time = 0;
    double total_time_with_plotting = 0;
    for ( size_t n = 0; n < conf.Nt; ++n )
    {
    	nufi::stopwatch<double> timer;

    	// Compute rho:
		#pragma omp parallel for
    	for(size_t l = 0; l<conf.Nx*conf.Ny; l++)
    	{
    		rho.get()[l] = eval_rho<real,order>(n, l, coeffs.get(), conf);
    	}

        poiss.solve( rho.get() );
        interpolate<real,order>( coeffs.get() + n*stride_t, rho.get(), conf );

        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;
        nufi::stopwatch<double> timer_plots;

/*
        real Emax = 0;
	    real E_l2 = 0;
        for ( size_t i = 0; i < conf.Nx; ++i )
        for ( size_t j = 0; j < conf.Ny; ++j )
        {
            real x = conf.x_min + i*conf.dx;
            real y = conf.y_min + j*conf.dy;
            real E_abs = std::hypot( eval<real,order,1,0>(x,y,coeffs.get()+n*stride_t,conf),
            						 eval<real,order,0,1>(x,y,coeffs.get()+n*stride_t,conf));
            Emax = max( Emax, E_abs );
	        E_l2 += E_abs*E_abs;
        }
	    E_l2 *=  conf.dx;

	    double t = n*conf.dt;
        Emax_file << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << Emax << std::endl;
        */
        //std::cout << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << Emax << " Comp-time: " << timer_elapsed << std::endl;
        std::cout << "n = " << n << " t = " << n*conf.dt << " Comp-time: " << timer_elapsed << std::endl;

    }
    std::cout << "Total time: " << total_time << std::endl;
    std::cout << "Total time with plotting: " << total_time_with_plotting << std::endl;
}


}
}


int main()
{
	nufi::dim2::test<double,4>();
}

