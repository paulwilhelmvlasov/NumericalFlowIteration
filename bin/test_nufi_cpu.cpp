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

namespace dim1
{

template <typename real, size_t order>
void test()
{
    using std::abs;
    using std::max;

    config_t<real> conf;
    const size_t stride_t = conf.Nx + order - 1;

    std::unique_ptr<real[]> coeffs { new real[ conf.Nt*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*conf.Nx)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};

    poisson<real> poiss( conf );

    std::ofstream stats_file( "stats.txt" );
    double total_time = 0;
    double total_time_with_plotting = 0;
    for ( size_t n = 0; n <= conf.Nt; ++n )
    {
    	nufi::stopwatch<double> timer;

    	// Compute rho:
		#pragma omp parallel for
    	for(size_t i = 0; i<conf.Nx; i++)
    	{
    		rho.get()[i] = eval_rho<real,order>(n, i, coeffs.get(), conf);
    		//rho.get()[i] = eval_rho_simpson<real,order>(n, i, coeffs.get(), conf);
    	}

        poiss.solve( rho.get() );
        dim1::interpolate<real,order>( coeffs.get() + n*stride_t, rho.get(), conf );

        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;
        nufi::stopwatch<double> timer_plots;


		real t = n*conf.dt;
        // Plotting:
        if(n % 2 == 0)
        {
			size_t plot_x = 256;
			//size_t plot_v = plot_x;
			real dx_plot = conf.Lx/plot_x;
			real Emax = 0;
			real E_l2 = 0;

			if(n % (10*16) == 0)
			{
				std::ofstream file_E( "E_" + std::to_string(t) + ".txt" );
				for ( size_t i = 0; i < plot_x; ++i )
				{
					real x = conf.x_min + i*dx_plot;
					real E = -dim1::eval<real,order,1>(x,coeffs.get()+n*stride_t,conf);
					Emax = max( Emax, abs(E) );
					E_l2 += E*E;

					file_E << x << " " << E << std::endl;

				}
				E_l2 *=  conf.dx;
				double t = n*conf.dt;
				stats_file << std::setw(20) << t << std::setw(20) << std::setprecision(8) << std::scientific << Emax
							<< std::setw(20) << std::setprecision(8) << std::scientific << E_l2 << std::endl;
			} else {
				for ( size_t i = 0; i < plot_x; ++i )
				{
					real x = conf.x_min + i*dx_plot;
					real E = -dim1::eval<real,order,1>(x,coeffs.get()+n*stride_t,conf);
					Emax = max( Emax, abs(E) );
					E_l2 += E*E;
				}
				E_l2 *=  conf.dx;
				double t = n*conf.dt;
				stats_file << std::setw(20) << t << std::setw(20) << std::setprecision(8) << std::scientific << Emax
							<< std::setw(20) << std::setprecision(8) << std::scientific << E_l2 << std::endl;
			}
        }
        std::cout << std::setw(15) << t << " Comp-time: " << timer_elapsed << " Total time: " << total_time << std::endl;

    }
    std::cout << "Total time: " << total_time << std::endl;
}


}

}


int main()
{
	nufi::dim1::test<double,4>();
}

