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
    return 1.0 / (2.0 * M_PI) * exp(-0.5 * u*u) * (1 + alpha * cos(k*x));
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

    std::ofstream timings_file( "timings.txt" );
    std::ofstream Emax_file( "Emax.txt" );
    std::ofstream str_E_max_err("E_max_error.txt");
    std::ofstream str_E_l2_err("E_l2_error.txt");
    std::ofstream str_E_max_rel_err("E_max_rel_error.txt");
    std::ofstream str_E_l2_rel_err("E_l2_rel_error.txt");
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
        if(n % (10*16) == 0)
        {
        	timings_file << "Total time until t = " << n*conf.dt << " was "
        			<< total_time << "s." << std::endl;
        }

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

		if(n % (10*16) == 0)
		{
			std::ifstream E_str( "../TestRes/E_" + std::to_string(t) + ".txt" );
			size_t plot_res_e = 256;
			double dx = conf.Lx / plot_res_e;
			double E_max_error = 0;
			double E_l2_error = 0;
			double E_max_exact = 0;
			for(size_t i = 0; i <= plot_res_e; i++)
			{
				double x = i * dx;
				double E_exact = 0;
				E_str >> x >> E_exact;
				double E = -periodic::eval<real,order,1>(x,coeffs.get()+n*stride_t,conf);

				double dist = std::abs(E - E_exact);
				E_max_error = std::max(E_max_error, dist);
				E_l2_error += (dist*dist);
				E_max_exact = std::max(E_max_exact, E_exact);
			}
			E_l2_error *= dx;
			str_E_max_err << t << " " << E_max_error << std::endl;
			str_E_l2_err << t << " " << E_l2_error << std::endl;
			str_E_max_rel_err << t << " " << E_max_error/E_max_exact << std::endl;
			str_E_l2_rel_err << t << " " << E_l2_error/E_max_exact << std::endl;
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

