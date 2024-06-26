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


template <typename real, size_t order>
void test()
{
    using std::abs;
    using std::max;

    config_t<real> conf;
    size_t stride_t = (conf.Nx + order - 1) *
                      (conf.Ny + order - 1);


    std::unique_ptr<real[]> coeffs { new real[ conf.Nt*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,
    													sizeof(real)*conf.Nx*conf.Ny)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};

    poisson<real> poiss( conf );

    //std::ofstream E_energy_file( "E_energy.txt" );
    double total_time = 0;
    for ( size_t n = 0; n < conf.Nt; ++n )
    {
    	nufi::stopwatch<double> timer;

    	// Compute rho:
		#pragma omp parallel for
    	for(size_t l = 0; l<conf.Nx*conf.Ny; l++)
    	{
    		rho.get()[l] = eval_rho<real,order>(n, l, coeffs.get(), conf);
    	}

        real E_energy = poiss.solve( rho.get() );
        interpolate<real,order>( coeffs.get() + n*stride_t, rho.get(), conf );

        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;
/*
        if(n % 8 == 0)
        {
        	real t = n*conf.dt;
        	std::ofstream file_phi( "phi_" + std::to_string(t) + ".txt" );
			std::ofstream file_E( "E_" + std::to_string(t) + ".txt" );
        	for ( size_t i = 0; i < conf.Nx; ++i )
        	{
				for ( size_t j = 0; j < conf.Ny; ++j )
				{
					real x = conf.x_min + i*conf.dx;
					real y = conf.y_min + j*conf.dy;
					real phi = eval<real,order>(x,y,coeffs.get()+n*stride_t,conf);
					real Ex = -eval<real,order,1,0>(x,y,coeffs.get()+n*stride_t,conf);
					real Ey = -eval<real,order,0,1>(x,y,coeffs.get()+n*stride_t,conf);

					file_phi << x << " " << y << " " << phi << std::endl;
					file_E << x << " " << y << " " << Ex << " " << Ey << std::endl;
				}
				file_phi << std::endl;
        	}
        }
*/
	    double t = n*conf.dt;
//	    E_energy_file << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << E_energy << std::endl;
        std::cout << "n = " << n << " t = " << n*conf.dt << " Comp-time: " << timer_elapsed << ". Total time s.f.: " << total_time << std::endl;

    }
    std::cout << "Total time: " << total_time << std::endl;
}
}
}


int main()
{
	nufi::dim2::test<double,4>();
}

