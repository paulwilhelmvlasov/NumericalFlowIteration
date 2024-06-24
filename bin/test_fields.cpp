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
#include <iomanip>
#include <nufi/config.hpp>
#include <nufi/random.hpp>
#include <nufi/fields.hpp>
#include <nufi/stopwatch.hpp>

namespace nufi
{


template <typename real>
real f( real x, real y = 0, real z = 0 )
{
	// Periodic.
    using std::sin;
    return sin(3*x) + sin(3*y) + sin(3*z);
}

template <typename real>
real g( real x, real y = 0, real z = 0 )
{
	// Non-periodic
    using std::sin;
    using std::cos;
    return sin(0.1*x)*exp(0.2*x) + sin(3*y)*y*y + sin(3*z)*z;
}

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
    conf.Lx_inv = real(1) / conf.Lx;
    conf.dx     = conf.Lx / real(conf.Nx);
    conf.dx_inv = real(1) / conf.dx;

    conf.Ny     = 128;
    conf.y_min  = 0; conf.y_max = 4*M_PI;
    conf.Ly     = conf.y_max - conf.y_min;
    conf.Ly_inv = real(1) / conf.Ly;
    conf.dy     = conf.Ly / real(conf.Ny);
    conf.dy_inv = real(1) / conf.dy;

    conf.Nz     = 128;
    conf.z_min  = 0; conf.z_max = 4*M_PI;
    conf.Lz     = conf.z_max - conf.z_min;
    conf.Lz_inv = real(1) / conf.Lz;
    conf.dz     = conf.Lz / real(conf.Nz);
    conf.dz_inv = real(1) / conf.dz;
    
    std::unique_ptr<real[]> values { new real[ conf.Nx*conf.Ny*conf.Nz ] };
    std::unique_ptr<real[]> coeffs { new real[ (conf.Nx + order - 1)*
                                               (conf.Ny + order - 1)*
                                               (conf.Nz + order - 1) ] {} };

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

        values[ l ] = f(x,y);
    }

    stopwatch<real> clock;
    dim2::interpolate<real,order>( coeffs.get(), values.get(), conf ); 
    real elapsed = clock.elapsed();
    std::cout << "Time for solving system: " << elapsed << ".\n"; 

    real sum = 0, err_sum = 0, err = 0, max_err = 0;
    for ( size_t l = 0; l < conf.Nx*conf.Ny; ++l )
    {
        size_t j = l / conf.Nx;
        size_t i = l % conf.Nx;
        real x = conf.x_min + i*conf.dx;
        real y = conf.y_min + j*conf.dy;
        real val = dim2::eval<real,order>( x, y, coeffs.get(), conf );
        max_err = max(abs(val-values[l]),max_err);
        err = hypot(err,values[ l ] - val);
        sum = hypot(sum,values[ l ]);
    }
    std::cout << "Max error: " << max_err << ". "
              << "Absolute l²-Error: " << err << ". "
              << "Relative l²-Error: " << err/sum << ".\n";

    max_err = err_sum = 0;
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
}

template <typename real, size_t order>
void test_1d()
{
    using std::abs;
    using std::max;
    using std::hypot;

    std::cout << "Testing dim = 1.\n";

    dim1::config_t<real> conf;
    size_t Nx = 128 + 1;
    conf.Nx     = Nx;
    conf.x_min  = 0; conf.x_max = 4*M_PI;
    conf.Lx     = conf.x_max - conf.x_min;
    conf.Lx_inv = 1 / conf.Lx;
    conf.dx     = real(conf.Lx) / (conf.Nx-1);
    conf.dx_inv = real(1) / conf.dx;

//    std::unique_ptr<real[]> values { new real[ conf.Nx ] };
//    std::unique_ptr<real[]> coeffs { new real[ conf.Nx + order - 1 ] };
    std::unique_ptr<real[]> values { new real[ Nx ] };
    std::unique_ptr<real[]> coeffs { new real[ Nx + order - 1 ] };

    // Bemerkung: Für den periodischen Fall hat obige Anzahl an Werten gereicht,
    // weil man sie in der "zweiten Periode" als Stützstellen nutzen konnte.
    // Im nicht-periodischen Fall ist das leider nicht einfach so möglich und
    // der Interpolationscode muss nochmal neu geschrieben werden. Ggfs kann man
    // die Auswertung aber beibehalten.



    random_real<real> rand(0,1);
//    for ( size_t i = 0; i < conf.Nx; ++i )
    for ( size_t i = 0; i < Nx; ++i )
    {
        real x = conf.x_min + i*conf.dx;

        //values[ i ] = f(x);
        values[ i ] = g(x);
    }

    stopwatch<real> clock;
    dim1::fd_dirichlet::interpolate<real,order>( coeffs.get(), values.get(), conf );
    real elapsed = clock.elapsed();
    std::cout << "Time for solving system: " << elapsed << ".\n";

    std::cout << "Interpolation points:" << std::endl;
    real sum = 0, err_sum = 0, err = 0, max_err = 0;
//    for ( size_t i = 0; i < conf.Nx; ++i )
    for ( size_t i = 0; i < Nx; ++i )
    {
        real x = conf.x_min + i*conf.dx;
        real val = dim1::fd_dirichlet::eval<real,order>( x, coeffs.get(), conf );
        max_err = max(abs(val-values[i]),max_err);
        err = hypot(err,values[ i ] - val);
        sum = hypot(sum,values[ i ]);
    }
    std::cout << "Max error: " << max_err << ". "
              << "Absolute l²-Error: " << err << ". "
              << "Relative l²-Error: " << err/sum << ".\n";

    std::cout << "Random points:" << std::endl;
    max_err = err_sum = 0;
    random_real<real> randx( conf.x_min, conf.x_max );
    for ( size_t i = 0; i < 4096; ++i )
    {
        real x = randx();
        real approx = dim1::fd_dirichlet::eval<real,order>( x, coeffs.get(), conf );
        //real exact  = f(x);
        real exact  = g(x);
        max_err = max( max_err, abs(approx-exact) );
        err_sum += abs(approx-exact);
    }
    std::cout << "Max error: " << max_err << std::endl;
    std::cout << "Avg error: " << err_sum/(4096) << std::endl;

    std::cout << conf.x_min << " " << conf.x_max << std::endl;
    std::cout << "S(x_min) = " << dim1::fd_dirichlet::eval<real,order>( conf.x_min, coeffs.get(), conf ) << std::endl;
    //std::cout << "f(x_min) = " << f(conf.x_min) << std::endl;
    std::cout << "g(x_min) = " << g(conf.x_min) << std::endl;

    std::cout << "S(x_max) = " << dim1::fd_dirichlet::eval<real,order>( conf.x_max, coeffs.get(), conf ) << std::endl;
    //std::cout << "f(x_max) = " << f(conf.x_max) << std::endl;
    std::cout << "g(x_max) = " << g(conf.x_max) << std::endl;
}


}

int main()
{
    std::cout << std::scientific;
    nufi::test_1d<double,4>();
    //nufi::test_2d<double,4>();
    //nufi::test_3d<double,4>();
}

