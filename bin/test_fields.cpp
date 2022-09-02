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

/*
template <typename real, size_t order>
void test_3d()
{
    using std::abs;
    using std::max;
    using std::hypot;

    std::cout << "Testing dim = 3.\n";

    dim3::config_t<real> conf;
    conf.Nx     = 128;
    conf.x_min  = 0; conf.x_max = 4*M_PI;
    conf.Lx     = conf.x_max - conf.x_min;
    conf.Lx_inv = 1 / conf.Lx;
    conf.dx     = real(conf.Lx) / conf.Nx;
    conf.dx_inv = real(1) / conf.dx;

    conf.Ny     = 128;
    conf.y_min  = 0; conf.y_max = 2*M_PI;
    conf.Ly     = conf.y_max - conf.y_min;
    conf.Ly_inv = 1 / conf.Ly;
    conf.dy     = real(conf.Ly) / conf.Ny;
    conf.dy_inv = real(1) / conf.dy;

    conf.Nz     = 512;
    conf.z_min  = -M_PI; conf.z_max = M_PI;
    conf.Lz     = conf.z_max - conf.z_min;
    conf.Lz_inv = 1 / conf.Lz;
    conf.dz     = conf.Lz / conf.Nz;
    conf.dz_inv = real(1) / conf.dz;
    
    std::unique_ptr<real[]> values { new real[ conf.Nx*conf.Ny*conf.Nz ] };
    std::unique_ptr<real[]> coeffs { new real[ (conf.Nx + order - 1)*
                                               (conf.Ny + order - 1)*
                                               (conf.Nz + order - 1) ] };
   

    for ( size_t l = 0; l < conf.Nx*conf.Ny*conf.Nz; ++l )
    {
        size_t k   = l / ( conf.Nx*conf.Ny );
        size_t tmp = l % ( conf.Nx*conf.Ny );
        size_t j = tmp / conf.Nx;
        size_t i = tmp % conf.Nx;

        real x = conf.x_min + i*conf.dx;
        real y = conf.y_min + j*conf.dy;
        real z = conf.z_min + k*conf.dz;
        values[ l ] = f( x, y, z );
    }

    stopwatch<real> clock;
    dim3::interpolate<real,order>( coeffs.get(), values.get(), conf ); 
    real elapsed = clock.elapsed();
    std::cout << "Time for solving system: " << elapsed << ".\n"; 

    real sum = 0, err = 0;
    for ( size_t l = 0; l < conf.Nx*conf.Ny*conf.Nz; ++l )
    {
        size_t k   = l / ( conf.Nx*conf.Ny );
        size_t tmp = l % ( conf.Nx*conf.Ny );
        size_t j = tmp / conf.Nx;
        size_t i = tmp % conf.Nx;

        real x = conf.x_min + i*conf.dx;
        real y = conf.y_min + j*conf.dy;
        real z = conf.z_min + k*conf.dz;

        real val = dim3::eval<real,order>( x, y, z, coeffs.get(), conf );
        err = hypot(err,values[ l ] - val);
        sum = hypot(sum,values[ l ]);
    }
    std::cout << "Absolute l²-Error: " << err << ". "
              << "Relative l²-Error: " << err/sum << ".\n";

    real max_err = 0, err_sum = 0;
    random_real<real> randx( conf.x_min, conf.x_max );
    random_real<real> randy( conf.y_min, conf.y_max );
    random_real<real> randz( conf.z_min, conf.z_max );
    for ( size_t i = 0; i < 4096*4096; ++i )
    {
        real x = randx(), y = randy(), z = randz();
        real approx = dim3::eval<real,order>( x, y, z, coeffs.get(), conf );
        real exact  = f(x,y,z);
        max_err = max( max_err, abs(approx-exact) );
        err_sum += abs(approx-exact);
    }
    std::cout << "Max error: " << max_err << std::endl;
    std::cout << "Avg error: " << err_sum / (4096*4096) << std::endl;
}
*/

template <typename real, size_t order>
void test_2d()
{
    using std::abs;
    using std::max;
    using std::hypot;

    std::cout << "Testing dim = 2.\n";

    dim2::config_t<real> conf;
    conf.Nx     = 128;
    conf.x_min  = 0; conf.x_max = 4*M_PI;
    conf.Lx     = conf.x_max - conf.x_min;
    conf.Lx_inv = 1 / conf.Lx;
    conf.dx     = real(conf.Lx) / conf.Nx;
    conf.dx_inv = real(1) / conf.dx;

    conf.Ny     = 128;
    conf.y_min  = -M_PI; conf.y_max = M_PI;
    conf.Ly     = conf.y_max - conf.y_min;
    conf.Ly_inv = 1 / conf.Ly;
    conf.dy     = real(conf.Ly) / conf.Ny;
    conf.dy_inv = real(1) / conf.dy;

    
    std::unique_ptr<real[]> values { new real[ conf.Nx*conf.Ny ] };
    std::unique_ptr<real[]> coeffs { new real[ (conf.Nx + order - 1)*
                                               (conf.Ny + order - 1) ] };
   

    random_real<real> rand(0,1);
    for ( size_t l = 0; l < conf.Nx*conf.Ny; ++l )
    {
        size_t j = l / conf.Nx;
        size_t i = l % conf.Nx;
        real x = conf.x_min + i*conf.dx;
        real y = conf.y_min + j*conf.dy;

//        values[ l ] = f(x,y);
        values[ l ] = rand();
    }

    stopwatch<real> clock;
    dim2::interpolate<real,order>( coeffs.get(), values.get(), conf ); 
    real elapsed = clock.elapsed();
    std::cout << "Time for solving system: " << elapsed << ".\n"; 

    real sum = 0, err = 0;
    for ( size_t l = 0; l < conf.Nx*conf.Ny; ++l )
    {
        size_t j = l / conf.Nx;
        size_t i = l % conf.Nx;
        real x = conf.x_min + i*conf.dx;
        real y = conf.y_min + j*conf.dy;
        real val = dim2::eval<real,order>( x, y, coeffs.get(), conf );
        err = hypot(err,values[ l ] - val);
        sum = hypot(sum,values[ l ]);
    }
    std::cout << "Absolute l²-Error: " << err << ". "
              << "Relative l²-Error: " << err/sum << ".\n";

    /*
    real max_err = 0, err_sum = 0;
    random_real<real> randx( conf.x_min, conf.x_max );
    random_real<real> randy( conf.y_min, conf.y_max );
    for ( size_t i = 0; i < 4096*4096; ++i )
    {
        real x = randx(), y = randy();
        real approx = dim2::eval<real,order>( x, y, coeffs.get(), conf );
        real exact  = f(x,y);
        max_err = max( max_err, abs(approx-exact) );
        err_sum += abs(approx-exact);
    }
    std::cout << "Max error: " << max_err << std::endl;
    std::cout << "Avg error: " << err_sum/(4096*4096) << std::endl;
    */
}

}

int main()
{
    dergeraet::test_2d<float,4>();
//    dergeraet::test_3d<double,4>();
}

