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

namespace dim3
{


template <typename real, size_t order>
void test()
{
    using std::abs;
    using std::max;

    config_t<real> conf;
    size_t stride_t = (conf.Nx + order - 1) *
                      (conf.Ny + order - 1) *
					  (conf.Nz + order - 1);


    std::unique_ptr<real[]> coeffs { new real[ conf.Nt*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,
    													sizeof(real)*conf.Nx*conf.Ny*conf.Nz)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};

    poisson<real> poiss( conf );

    double total_time = 0;
    for ( size_t n = 0; n < conf.Nt; ++n )
    {
    	dergeraet::stopwatch<double> timer;

    	// Compute rho:
		#pragma omp parallel for
    	for(size_t l = 0; l<conf.Nx*conf.Ny*conf.Nz; l++)
    	{
    		rho.get()[l] = eval_rho<real,order>(n, l, coeffs.get(), conf);
    	}

        real E_energy = poiss.solve( rho.get() );
        interpolate<real,order>( coeffs.get() + n*stride_t, rho.get(), conf );

        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;
	    double t = n*conf.dt;
        std::cout << "n = " << n << " t = " << n*conf.dt << " Comp-time: " << timer_elapsed << ". Total time s.f.: " << total_time << std::endl;

    }
    std::cout << "Total time: " << total_time << std::endl;
}
}
}


int main()
{
	dergeraet::dim3::test<double,4>();
}

