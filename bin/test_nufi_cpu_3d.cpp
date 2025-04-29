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
 
 #include <armadillo>
 
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

template <typename real>
real f0(real x, real y, real z, real u, real v, real w) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

    constexpr real alpha = 0.01;
    constexpr real k     = 0.5;

    // Weak Landau Damping in x direction:
    constexpr real c  = 1.0 / std::pow(2.0 * M_PI, 3.0/2.0); 
    return c * ( 1. + alpha*cos(k*x)) * exp( -(u*u + v*v + w*w)/2 );
}

const double Lx = 4*M_PI;
const double umin = -5;
const double umax = 5;
const size_t Nx = 32;  
const size_t Ny = 1;  
const size_t Nz = 1;  
const size_t Nu = 32;
const size_t Nv = 8;
const size_t Nw = 8;  
const double   dt = 0.1;  
const size_t Nt = 30/dt;  
config_t<double> conf(Nx, Ny, Nz, Nu, Nv, Nw, Nt, dt, 
                    0, Lx, 0, Lx, 0, Lx, umin, umax, 
                    umin, umax, umin, umax,  &f0);

template <typename real, size_t order>
void run_simulation()
{
    poisson<real> poiss( conf );

    size_t stride_t = (conf.Nx + order - 1) *
                        (conf.Ny + order - 1) *
                        (conf.Nz + order - 1);

    std::unique_ptr<double[]> coeffs_full { new double[ (Nt+1)*stride_t ] {} };
    std::unique_ptr<double,decltype(std::free)*> rho { reinterpret_cast<double*>(std::aligned_alloc(64,
        sizeof(double)*conf.Nx*conf.Ny*conf.Nz)), std::free };

    std::ofstream stats_file( "stats.txt" );
    std::ofstream coeff_file( "coeffs.txt" );
    double total_time = 0;
    double total_time_with_plotting = 0;
    for ( size_t n = 0; n <= conf.Nt; ++n )
    {
    	nufi::stopwatch<double> timer;

    	// Compute rho:
		#pragma omp parallel for
    	for(size_t l = 0; l<conf.Nx*conf.Ny*conf.Nz; l++)
    	{
    		rho.get()[l] = eval_rho<real,order>(n, l, coeffs_full.get(), conf);
    	}

        double electric_energy = poiss.solve( rho.get() );
        interpolate<real,order>( coeffs_full.get() + n*stride_t, rho.get(), conf );

        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;
        nufi::stopwatch<double> timer_plots;

        // Print coefficients to file.
        coeff_file << n << std::endl;
        for(size_t i = 0; i < stride_t; i++){
            coeff_file << i << " " << coeffs_full.get()[n*stride_t + i ] << std::endl;
        }
        stats_file << n*conf.dt << " " << electric_energy << std::endl;

        std::cout << "n = " << n << " t = " << n*conf.dt << " Comp-time: " << timer_elapsed << std::endl;

    }
    std::cout << "Total time: " << total_time << std::endl;
    std::cout << "Total time with plotting: " << total_time_with_plotting << std::endl;

}

}
}

int main()
{
    nufi::dim3::run_simulation<double,4>();

    return 0;
}