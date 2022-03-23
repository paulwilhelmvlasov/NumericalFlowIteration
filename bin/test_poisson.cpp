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

#include <iostream>
#include <fstream>

#include <iomanip>

#include <dergeraet/poisson.hpp>
#include <dergeraet/stopwatch.hpp>


namespace dergeraet
{

namespace dim3
{

template <typename real>
real rho( real x, real y, real z )
{
    using std::sin;
    using std::cos;
    return cos(2*z) + sin(8*y) + cos(42*x);
}

template <typename real>
real phi( real x, real y, real z )
{
    using std::sin;
    using std::cos;
    return cos(2*z)/4 + sin(8*y)/64 + cos(42*x)/(42*42);
}

template <typename real>
void test()
{
    using std::abs;

    std::cout << "Testing dim = 3.\n";
    config_t<real> conf;

    conf.Nx     = 128;
    conf.x_min  = 0; conf.x_max = 2*M_PI; 
    conf.Lx     = conf.x_max - conf.x_min;
    conf.Lx_inv = 1/conf.Lx;
    conf.dx     = conf.Lx / conf.Nx;
    conf.dx_inv = 1 / conf.dx;

    conf.Ny     = 64;
    conf.y_min  = 0; conf.y_max = 2*M_PI; 
    conf.Ly     = conf.y_max - conf.y_min;
    conf.Ly_inv = 1/conf.Ly;
    conf.dy     = conf.Ly / conf.Ny;
    conf.dy_inv = 1 / conf.dy;

    conf.Nz     = 32;
    conf.z_min  = 0; conf.z_max = 2*M_PI; 
    conf.Lz     = conf.z_max - conf.z_min;
    conf.Lz_inv = 1/conf.Lz;
    conf.dz     = conf.Lz / conf.Nz;
    conf.dz_inv = 1 / conf.dz;

    stopwatch<real> clock;
    poisson<real> pois(conf);
    std::cout << "Initialisation time = " << clock.elapsed() << "[s]" << std::endl;

    real *data = (real*) std::aligned_alloc( pois.alignment, sizeof(real)*conf.Nx*conf.Ny*conf.Nz );

    for ( size_t k = 0; k < conf.Nz; ++k )
    for ( size_t j = 0; j < conf.Ny; ++j )
    for ( size_t i = 0; i < conf.Nx; ++i )
    {
        real x = conf.x_min + i*conf.dx;
        real y = conf.y_min + j*conf.dy;
        real z = conf.z_min + k*conf.dz;
        data[ i + j*conf.Nx + k*conf.Nx*conf.Ny ] = rho(x,y,z);
    }

    
    clock.reset();
    pois.solve(data);
    std::cout << "Computation time = " << clock.elapsed() << " [s]" << std::endl;

    real l1_err = 0, l1_norm = 0;
    for ( size_t k = 0; k < conf.Nz; ++k )
    for ( size_t j = 0; j < conf.Ny; ++j )
    for ( size_t i = 0; i < conf.Nx; ++i )
    {
        real x = conf.x_min + i*conf.dx;
        real y = conf.y_min + j*conf.dy;
        real z = conf.z_min + k*conf.dz;
        real approx = data[ i + j*conf.Nx + k*conf.Nx*conf.Ny ];
        real exact  = phi(x,y,z);
        real err = approx - exact;
        l1_err  += abs(err);
        l1_norm += abs(exact);
    }
    std::cout << u8"Relative l¹-error = " << l1_err/l1_norm << std::endl;

    std::free(data);
}

}

}

int main()
{
    std::cout << "Testing float.\n";
    dergeraet::dim3::test<float>();

    std::cout << "Testing double.\n";
    dergeraet::dim3::test<double>();
}

