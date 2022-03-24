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


#include <dergeraet/config.hpp>
#include <dergeraet/random.hpp>
#include <dergeraet/fields.hpp>
#include <dergeraet/poisson.hpp>
#include <dergeraet/rho.hpp>
#include <dergeraet/stopwatch.hpp>
#include <dergeraet/cuda_kernel.hpp>

namespace dergeraet
{

namespace dim1
{

template <typename real, size_t order>
void test()
{
    using std::abs;
    using std::max;

    config_t<real> conf;
    const size_t stride_x = 1;
    const size_t stride_t = conf.Nx + order - 1;

    std::unique_ptr<real[]> coeffs { new real[ conf.Nt*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*conf.Nx)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};

    poisson<real> poiss( conf );

    #ifdef HAVE_CUDA
    cuda_kernel<real,order> kernel { conf };
    #endif

    for ( size_t n = 0; n < conf.Nt; ++n )
    {
        #ifdef HAVE_CUDA
        kernel.compute_rho( n, coeffs.get(), rho.get() );
        #else  
        #pragma omp parallel for
        for ( size_t i = 0; i < conf.Nx; ++i )
        {
            rho.get()[ i ] = dim1::eval_rho<real,order>( n, i, coeffs.get(), conf );
        }
        #endif

        poiss.solve( rho.get() );
        dim1::interpolate<real,order>( coeffs.get() + n*stride_t, rho.get(), conf );

        real Emax = 0;
        for ( size_t i = 0; i < conf.Nx; ++i )
        {
            real x = conf.x_min + i*conf.dx;
            Emax = max( Emax, abs( dim1::eval<real,order,1>(x,coeffs.get()+n*stride_t,conf))); 
        }

        std::stringstream filename; filename << "E" << n << ".txt";
        std::ofstream file( filename.str() ); file << std::scientific << std::setprecision(8);
        for ( size_t i = 0; i < conf.Nx; ++i )
        {
            real x = conf.x_min + i*conf.dx;
            real E = -dim1::eval<real,order,1>( x, coeffs.get() + n*stride_t, conf );
            file << std::setw(20) << x << std::setw(20) << E << '\n';
        }
        file.close();
   
        std::cout << std::setw(15) << n*conf.dt << std::setw(15) << std::setprecision(5) << std::scientific << Emax << std::endl; 
    }
}

}

}


int main()
{
    dergeraet::dim1::test<double,4>();
}

