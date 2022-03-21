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
real f0( real x, real u ) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

    //double np  = 0.9, nb = 0.2, vb = 4.5, vt = 0.5, alpha = 0.04, k = 0.3;    
    //double fac = 1.0/std::sqrt(2*3.141592653589793238462643);
    //return fac*( np*exp(-0.5*v*v) + nb*exp(-0.5*(v-vb)*(v-vb)/(vt*vt)) ) * (1+alpha*std::cos(k*x));

	constexpr real alpha = 0.01;
	constexpr real k     = 0.5;
    return 0.39894228040143267793994 * ( 1. + alpha*cos(k*x) ) * exp( -u*u/2. ) * u*u;
    //return 0.39894228040143267793994 * ( 1. + alpha*cos(k*x) ) * exp( -u*u/2 );
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
    using std::abs;
    using std::max;

    config_t<real> conf;
    conf.Nx = 128;
    conf.Nu = 1024*4; conf.u_max = 10;
    conf.Nt = 100*16;
    
    conf.L0x = 0;
    conf.L1x = 4*M_PI;
    conf.Lx  = 4*M_PI;            conf.Lx_inv = 1 / conf.Lx;
    conf.dx  = conf.Lx / conf.Nx; conf.dx_inv = 1 / conf.dx;

    conf.dt = 1./16.;

    const size_t stride_x = 1;
    const size_t stride_t = conf.Nx + order - 1;

    std::unique_ptr<real[]> coeffs { new real[ conf.Nt*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*conf.Nx)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};

    dim1::poisson<real> poiss( conf );

    for ( size_t n = 0; n < conf.Nt; ++n )
    {
        #pragma omp parallel for
        for ( size_t i = 0; i < conf.Nx; ++i )
        {
            rho.get()[ i ] = dim1::eval_rho<real,order>( n, i, coeffs.get(), conf );
        }

        poiss.solve( rho.get() );
        dim1::interpolate<real,order>( coeffs.get() + n*stride_t, rho.get(), conf );

        real Emax = 0;
        for ( size_t i = 0; i < conf.Nx; ++i )
            Emax = max( Emax, abs( dim1::eval<real,order,1>( i*conf.dx, coeffs.get() + n*stride_t, conf ) ) ); 
        Emax *= conf.dx_inv;

        std::stringstream filename; filename << "E" << n << ".txt";
        std::ofstream file( filename.str() ); file << std::scientific << std::setprecision(8);
        for ( size_t i = 0; i < conf.Nx; ++i )
        {
            real E = -dim1::eval<real,order,1>( i*conf.dx, coeffs.get() + n*stride_t, conf )*conf.dx_inv;
            file << std::setw(20) << i*conf.dx << std::setw(20) << E << '\n';
        }
        file.close();
   
        std::cout << std::setw(15) << n*conf.dt << std::setw(15) << std::setprecision(5) << std::scientific << Emax << std::endl; 
    }
}

}


int main()
{
    dergeraet::do_test<double,4>();
}

