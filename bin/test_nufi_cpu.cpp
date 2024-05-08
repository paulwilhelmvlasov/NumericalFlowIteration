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
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>


#include <dergeraet/config.hpp>
#include <dergeraet/random.hpp>
#include <dergeraet/fields.hpp>
#include <dergeraet/poisson.hpp>
#include <dergeraet/rho.hpp>
#include <dergeraet/stopwatch.hpp>

namespace dergeraet
{

namespace dim1
{

template <typename real>
real f0(real x, real u) noexcept
{
	real alpha = 1e-2;
	real k = 0.5;
    return 1.0 / (2.0 * M_PI) * u*u * exp(-0.5 * u*u) * (1 + alpha * cos(k*x));
}

template <typename real, size_t order>
void test()
{
	using std::exp;
	using std::sin;
	using std::cos;
    using std::abs;
    using std::max;

    size_t Nx = 64;  // Number of grid points in physical space.
    size_t Nu = 128;  // Number of quadrature points in velocity space.
    real   dt = 0.1;  // Time-step size.
    size_t Nt = 50/dt;  // Number of time-steps.

    // Dimensions of physical domain.
    real x_min = 0;
    real x_max = 4*M_PI;

    // Integration limits for velocity space.
    real u_min = -10;
    real u_max = 10;

    config_t<real> conf(Nx, Nu, Nt, dt, x_min, x_max, u_min, u_max, &f0);
    const size_t stride_t = conf.Nx + order - 1;

    std::unique_ptr<real[]> coeffs { new real[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*conf.Nx)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};

    poisson<real> poiss( conf );

    std::ofstream Emax_file( "Emax.txt" );
    std::ofstream coeffs_str( "coeffs_Nt_" + std::to_string(conf.Nt) + "_Nx_"
    						+ std::to_string(conf.Nx) + "_stride_t_" + std::to_string(stride_t) + ".txt" );
    double total_time = 0;
    for ( size_t n = 0; n <= conf.Nt; ++n )
    {
    	dergeraet::stopwatch<double> timer;

    	// Compute rho:
		#pragma omp parallel for
    	for(size_t i = 0; i<conf.Nx; i++)
    	{
    		rho.get()[i] = periodic::eval_rho<real,order>(n, i, coeffs.get(), conf);
    	}

        poiss.solve( rho.get() );
        periodic::interpolate<real,order>( coeffs.get() + n*stride_t, rho.get(), conf );

        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;

        real Emax = 0;
	    real E_l2 = 0;
        for ( size_t i = 0; i < conf.Nx; ++i )
        {
            real x = conf.x_min + i*conf.dx;
            real E_abs = abs( periodic::eval<real,order,1>(x,coeffs.get()+n*stride_t,conf));
            Emax = max( Emax, E_abs );
	        E_l2 += E_abs*E_abs;
        }
	    E_l2 *=  conf.dx;

	    double t = n*conf.dt;
        Emax_file << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << Emax << std::endl;
        std::cout << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << Emax << " Comp-time: " << timer_elapsed << std::endl;

        for(size_t i = 0; i < stride_t; i++){
        	coeffs_str << coeffs.get()[n*stride_t + i] << std::endl;
        }

    }
    std::cout << "Total time: " << total_time << std::endl;
}


}
}


int main()
{
	dergeraet::dim1::test<double,4>();
}

