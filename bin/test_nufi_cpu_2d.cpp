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
#include <dergeraet/cuda_scheduler.hpp>

namespace dergeraet
{

namespace dim2
{


template <typename real, size_t order>
void test()
{
    using std::abs;
    using std::max;

    config_t<real> conf;
    conf.Nt = 70;
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
    	dergeraet::stopwatch<double> timer;

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
        dergeraet::stopwatch<double> timer_plots;

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
	dergeraet::dim2::test<double,4>();
}

