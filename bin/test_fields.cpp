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
#include <nufi/stopwatch.hpp>

namespace nufi
{


template <typename real>
real f( real x, real y = 0, real z = 0 )
{
    using std::sin;
    return sin(3*x) + sin(3*y) + sin(3*z);
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

    std::ofstream error_stream("error_stream_order" + std::to_string(order) + ".txt");
    std::ofstream diff_error_stream("diff_error_stream_order" + std::to_string(order) + ".txt");
    std::cout << "Testing dim = 1.\n";
    for(size_t N = 0; N <= 8; N++){
        dim1::config_t<real> conf;
        conf.Nx     = 16*std::pow(2,N);
        conf.x_min  = 0; conf.x_max = 4*M_PI;
        conf.Lx     = conf.x_max - conf.x_min;
        conf.Lx_inv = 1 / conf.Lx;
        conf.dx     = real(conf.Lx) / conf.Nx;
        conf.dx_inv = real(1) / conf.dx;
        
        std::unique_ptr<real[]> values { new real[ conf.Nx ] };
        std::unique_ptr<real[]> coeffs { new real[ (conf.Nx + order - 1)] };

        for ( size_t i = 0; i < conf.Nx; i++ )
        {
            real x = conf.x_min + i*conf.dx;

            values[ i ] = f(x);
        }


        dim1::interpolate<real,order>( coeffs.get(), values.get(), conf ); 

        double l2_error = 0;
        double max_error = 0;
        double diff_l2_error = 0;
        double diff_max_error = 0;
        size_t plot_x = 2048;
        double dx_plot = conf.Lx/plot_x;
        for(size_t i = 0; i < plot_x; i++){
            double x = conf.x_min + i*dx_plot;

            double f_exact = f(x);
            double f_num = dim1::eval<real,order>(x, coeffs.get(), conf);

            double diff_f_exact = 3*cos(3*x);
            double diff_f_num = dim1::eval<real,order,1>(x, coeffs.get(), conf);

            double dist = abs(f_exact - f_num);
            double dist_diff = abs(diff_f_exact - diff_f_num);

            l2_error += dist*dist;
            max_error = max(dist, max_error);

            diff_l2_error += dist_diff*dist_diff;
            diff_max_error = max(dist_diff, diff_max_error);
        }

        l2_error = dx_plot*sqrt(l2_error);
        diff_l2_error = dx_plot*sqrt(diff_l2_error);

/*         std::cout << "L2 error = " << l2_error << std::endl;
        std::cout << "Max error = " << max_error << std::endl; */

        error_stream << N << " " << l2_error << " " << max_error << std::endl;
        diff_error_stream << N << " " << diff_l2_error << " " << diff_max_error << std::endl;
    }
}


}

int main()
{
    std::cout << std::scientific;

    std::cout << 2 << std::endl;
    nufi::test_1d<double,2>();

    std::cout << 3 << std::endl;
    nufi::test_1d<double,3>();

    std::cout << 4 << std::endl;
    nufi::test_1d<double,4>();

    std::cout << 6 << std::endl;
    nufi::test_1d<double,6>();

    std::cout << 8 << std::endl;
    nufi::test_1d<double,8>();
}

