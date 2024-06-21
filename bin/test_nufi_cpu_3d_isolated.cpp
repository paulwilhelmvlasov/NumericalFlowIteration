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
#include <random>
#include <sstream>


#include <nufi/config.hpp>
#include <nufi/random.hpp>
#include <nufi/fields.hpp>
#include <nufi/poisson.hpp>
#include <nufi/rho.hpp>
#include <nufi/stopwatch.hpp>

namespace nufi
{

namespace dim3
{



template <typename real, size_t order>
void write_quasi_random_potentials()
{
    using std::abs;
    using std::max;

    config_t<real> conf;
    size_t stride_t = (conf.Nx + order - 1) *
                      (conf.Ny + order - 1) *
					  (conf.Nz + order - 1);


    std::unique_ptr<real[]> coeffs { new real[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,
    													sizeof(real)*conf.Nx*conf.Ny*conf.Nz)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};



    std::ofstream write_potential("quasi_random_bop_potential_periodic_3d.txt" );

    write_potential << "Nt = " << std::to_string(conf.Nt) << std::endl;
    write_potential << "dt = " << std::to_string(conf.dt) << std::endl;
    write_potential << "Nx = " << std::to_string(conf.Nx) << std::endl;
    write_potential << "Ny = " << std::to_string(conf.Ny) << std::endl;
    write_potential << "Nz = " << std::to_string(conf.Nz) << std::endl;
    write_potential << "order = " << std::to_string(order) << std::endl;
    write_potential << std::endl;
    write_potential << std::setprecision(16);


    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(-1e-4,1e-4);
	real fac_x = 2*M_PI/conf.Lx;
	real fac_y = 2*M_PI/conf.Ly;
	real fac_z = 2*M_PI/conf.Lz;

    for ( size_t n = 0; n <= conf.Nt; ++n )
    {
    	real eps = dis(gen);
    	for(size_t i = 0; i < conf.Nx; i++)
    	for(size_t j = 0; j < conf.Ny; j++)
    	for(size_t k = 0; k < conf.Nz; k++)
    	{
    		real x = conf.x_min + i * conf.dx;
    		real y = conf.y_min + j * conf.dy;
    		real z = conf.z_min + k * conf.dz;

    		size_t l = i + conf.Nx*(j + k*conf.Ny);

    		rho.get()[l] = std::sin(x*fac_x) * std::sin(y*fac_y) * std::sin(z*fac_z) * eps;
    	}

        interpolate<real,order>( coeffs.get() + n*stride_t, rho.get(), conf );

        for(size_t l = 0; l < stride_t; l++)
        {
        	write_potential << coeffs.get()[n*stride_t + l] << std::endl;
        }
    }
}

template <typename real, size_t order>
void simulation_to_write_potentials()
{
    using std::abs;
    using std::max;

    config_t<real> conf;
    size_t stride_t = (conf.Nx + order - 1) *
                      (conf.Ny + order - 1) *
					  (conf.Nz + order - 1);


    std::unique_ptr<real[]> coeffs { new real[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,
    													sizeof(real)*conf.Nx*conf.Ny*conf.Nz)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};

    poisson<real> poiss( conf );

	//std::ofstream write_potential("quasi_random_bop_potential_periodic_3d.txt" );
    std::ofstream write_potential("bump_on_tail_potential_periodic_3d.txt" );

    write_potential << "Nt = " << std::to_string(conf.Nt) << std::endl;
    write_potential << "dt = " << std::to_string(conf.dt) << std::endl;
    write_potential << "Nx = " << std::to_string(conf.Nx) << std::endl;
    write_potential << "Ny = " << std::to_string(conf.Ny) << std::endl;
    write_potential << "Nz = " << std::to_string(conf.Nz) << std::endl;
    write_potential << "order = " << std::to_string(order) << std::endl;
    write_potential << std::endl;
    write_potential << std::setprecision(16);

    double total_time = 0;
    for ( size_t n = 0; n <= conf.Nt; ++n )
    {
    	nufi::stopwatch<double> timer;

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

        //write_potential << "n = " << n << std::endl;
        for(size_t l = 0; l < stride_t; l++)
        {
        	write_potential << coeffs.get()[n*stride_t + l] << std::endl;
        }
        //write_potential << std::endl;
    }
    std::cout << "Total time: " << total_time << std::endl;
}

template <typename real, size_t order>
void compute_isolated_time_step(size_t n)
{
    config_t<real> conf;
    size_t stride_t = (conf.Nx + order - 1) *
                      (conf.Ny + order - 1) *
					  (conf.Nz + order - 1);

	if(n > conf.Nt)
	{
		throw std::runtime_error("n > Nt");
	}


    std::unique_ptr<real[]> coeffs { new real[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,
    													sizeof(real)*conf.Nx*conf.Ny*conf.Nz)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};

    poisson<real> poiss( conf );

    // Read the coefficients in:
    char dump[256];
    std::ifstream read_potential("bump_on_tail_potential_periodic_3d.txt" );
    std::cout << "Open file = " << read_potential.is_open() << std::endl;
    std::ofstream write_debug_potential("debug.txt");
    read_potential >> std::setprecision(16);
    for(size_t i = 0; i < 7; i++)
    {
    	read_potential.getline(dump, 256);
    }
    for(size_t i = 0; i <= conf.Nt; i++)
    {
		//read_potential.getline(dump, 256);
    	write_debug_potential << "n = " << i << std::endl;

    	for(size_t l = 0; l < stride_t; l++)
    	{
    		size_t index = i*stride_t + l;
        	read_potential >> coeffs.get()[index];
        	write_debug_potential << coeffs.get()[index] << std::endl;
    	}
    }

    // Run requested time-step:
	nufi::stopwatch<double> timer;
	// Compute rho:
	#pragma omp parallel for
	for(size_t l = 0; l<conf.Nx*conf.Ny*conf.Nz; l++)
	{
		rho.get()[l] = eval_rho<real,order>(n, l, coeffs.get(), conf);
	}

	// Solve Poisson equation:
    real E_energy = poiss.solve( rho.get() );
    // Save the new coefficients:
    interpolate<real,order>( coeffs.get() + n*stride_t, rho.get(), conf );
    double timer_elapsed = timer.elapsed();
    std::cout << "n = " << n << " Comp-time: " << timer_elapsed << std::endl;
}


}
}


int main()
{
//	nufi::dim3::write_quasi_random_potentials<double,4>();
	//nufi::dim3::simulation_to_write_potentials<double,4>();
	nufi::dim3::compute_isolated_time_step<double,4>(30);
}



