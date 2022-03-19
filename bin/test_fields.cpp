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
#include <dergeraet/config.hpp>
#include <dergeraet/random.hpp>
#include <dergeraet/fields.hpp>
#include <dergeraet/stopwatch.hpp>

namespace dergeraet
{


template <typename real>
real f( real x, real y = 0, real z = 0 )
{
    using std::sin;
    return sin(3*x) + sin(1*y) + sin(37*z);
}

template <typename real>
void test()
{
    using std::abs;
    using std::max;
    using std::hypot;

    config_t<real> conf;
    conf.Nx     = 128;
    conf.dx     = real(conf.Lx) / conf.Nx;
    conf.dx_inv = real(1) / conf.dx;

    conf.Ny     = 128;
    conf.dy     = real(conf.Ly) / conf.Ny;
    conf.dy_inv = real(1) / conf.dy;

    conf.Nz     = 128;
    conf.dz     = real(conf.Lz) / conf.Nz;
    conf.dz_inv = real(1) / conf.dz;
    
    
    std::unique_ptr<real[]> mem { new real[ 2*conf.Nx*conf.Ny*conf.Nz ] };
    real *coeffs = mem.get();
    real *values = coeffs + conf.Nx*conf.Ny*conf.Nz;

    random_real<real> rand( 0, 2*3.1415926535 );
    for ( size_t l = 0; l < conf.Nx*conf.Ny*conf.Nz; ++l )
    {
        size_t k   = l / ( conf.Nx*conf.Ny );
        size_t tmp = l % ( conf.Nx*conf.Ny );
        size_t j = tmp / conf.Nx;
        size_t i = tmp % conf.Nx;

        values[ l ] = f( i*conf.dx, j*conf.dy, k*conf.dz );
        coeffs[ l ] = 0;
    }

    stopwatch<real> clock;
    dim3::interpolate<real,4>( coeffs, values, conf ); 
    real elapsed = clock.elapsed();
    std::cout << "Time for solving system: " << elapsed << ".\n"; 

    real sum = 0, err = 0;
    for ( size_t l = 0; l < conf.Nx*conf.Ny*conf.Nz; ++l )
    {
        size_t k   = l / ( conf.Nx*conf.Ny );
        size_t tmp = l % ( conf.Nx*conf.Ny );
        size_t j = tmp / conf.Nx;
        size_t i = tmp % conf.Nx;
        real val = dim3::eval<real,4>( i*conf.dx, j*conf.dy, k*conf.dz, coeffs, conf );
        err = hypot(err,values[ l ] - val);
        sum = hypot(sum,values[ l ]);
    }
    std::cout << "Absolute l²-Error: " << err << ". "
              << "Relative l²-Error: " << err/sum << ".\n";

    real max_err = 0;
    for ( size_t i = 0; i < 4096; ++i )
    {
        real x = rand(), y = rand(), z = rand();
        real approx = dim3::eval<real,4>( x, y, z, coeffs, conf );
        real exact  = f(x,y,z);
        max_err = max( max_err, abs(approx-exact) );
    }
    std::cout << "Max error: " << max_err << std::endl;
}

}

int main()
{
    dergeraet::test<double>();
}

