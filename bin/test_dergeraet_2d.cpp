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
#include <fstream>
#include <sstream>


template <typename real>
real f0( real x, real y, real u, real v ) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

	constexpr real alpha = 0.01;
	constexpr real k     = 0.5;
    return (7/(4*M_PI)) * sin(u/3) * sin(u/3) * (1+0.05*cos(0.3*x)) * exp( -(u*u + 4*v*v)/8. );
}


#include <dergeraet/config.hpp>
#include <dergeraet/random.hpp>
#include <dergeraet/fields.hpp>
#include <dergeraet/poisson.hpp>
#include <dergeraet/rho.hpp>
#include <dergeraet/stopwatch.hpp>

namespace dergeraet
{

template <typename real, size_t order>
void do_test()
{
    using std::hypot;
    using std::max;

    config_t<real> conf;
    conf.Nx = 32;
    conf.Ny = 32;
    conf.Nu = 256; conf.u_max = 10;
    conf.Nv = 256; conf.v_max = 10;
    conf.dt = 0.4;
    conf.Nt = 100 / conf.dt;
    
    conf.L0x = 0;
    conf.L1x = 20*M_PI/3;
    conf.Lx  = 20*M_PI/3;         conf.Lx_inv = 1 / conf.Lx;
    conf.dx  = conf.Lx / conf.Nx; conf.dx_inv = 1 / conf.dx;

    conf.L0y = 0;
    conf.L1y = 20*M_PI/3;
    conf.Ly  = 20*M_PI/3;         conf.Ly_inv = 1 / conf.Ly;
    conf.dy  = conf.Ly / conf.Ny; conf.dy_inv = 1 / conf.dy;

    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_t = stride_y*(conf.Ny + order - 1);

    std::unique_ptr<real[]> coeffs { new real[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*conf.Nx*conf.Ny)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};

    dim2::poisson<real> poiss( conf );

    for ( size_t n = 0; n <= conf.Nt; ++n )
    {
        #pragma omp parallel for
        for ( size_t l = 0; l < conf.Nx * conf.Ny; ++l )
        {
            size_t i = l % conf.Nx;
            size_t j = l / conf.Nx;
            rho.get()[ l ] = dim2::eval_rho<real,order>( n, i, j, coeffs.get(), conf );
        }

        poiss.solve( rho.get() );

        if ( n )
        {
            for ( size_t l = 0; l < stride_t; ++l )
                coeffs[ n*stride_t + l ] = coeffs[ (n-1)*stride_t + l ];
        }
        dim2::interpolate<real,order>( coeffs.get() + n*stride_t, rho.get(), conf );

        real Emax = 0;
        for ( size_t l = 0; l < conf.Nx * conf.Ny; ++l )
        {
            size_t i = l % conf.Nx;
            size_t j = l / conf.Nx;

            real Ex = -dim2::eval<real,order,1,0>( i*conf.dx, j*conf.dy, coeffs.get() + n*stride_t, conf ) * conf.dx_inv;
            real Ey = -dim2::eval<real,order,0,1>( i*conf.dx, j*conf.dy, coeffs.get() + n*stride_t, conf ) * conf.dy_inv;

            Emax = max( Emax, hypot(Ex,Ey) );
        }

        std::cout << std::setw(15) << n*conf.dt << std::setw(15) << std::setprecision(5) << std::scientific << Emax << std::endl; 
    }

    std::ofstream file( "f12.txt" );
    for ( double u = -10; u <= 10; u += 20./256. )
    {
        for ( double x = -conf.Lx/2; x <= conf.Lx/2; x += conf.Lx / 256 )
            file << x + conf.Lx/2  << " " << u << " " << dim2::eval_f<real,order>( 30, x, 0, u, 0, coeffs.get(), conf ) << std::endl;
        file << std::endl;
    }
}

}


int main()
{
    dergeraet::do_test<double,4>();
}

