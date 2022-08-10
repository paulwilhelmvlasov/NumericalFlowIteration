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
#include <dergeraet/cuda_scheduler.hpp>

namespace dergeraet
{

namespace dim2
{

template <typename real, size_t order>
void test()
{
    using std::hypot;
    using std::max;

    config_t<real> conf;
    poisson<real> poiss( conf );

    cuda_scheduler<real,order> sched { conf,0, conf.Nx*conf.Ny };

    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_t = stride_y*(conf.Ny + order - 1);

    std::unique_ptr<real[]> coeffs { new real[ (conf.Nt+1)*stride_t ] {} };
    void *tmp = std::aligned_alloc( poiss.alignment, sizeof(real)*conf.Nx*conf.Ny );
    if ( tmp == nullptr ) throw std::bad_alloc {};
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(tmp), std::free };

    std::ofstream Emax_file( "Emax2d.txt" );
    for ( size_t n = 0; n <= conf.Nt; ++n )
    {
    	stopwatch<real> clock;
        #ifdef HAVE_CUDA
        sched.compute_rho( n, coeffs.get(), rho.get() );
        #else

        #pragma omp parallel for
        for ( size_t l = 0; l < conf.Nx * conf.Ny; ++l )
            rho.get()[ l ] = dim2::eval_rho<real,order>( n, l, coeffs.get(), conf );

        #endif

        poiss.solve( rho.get() );

        if ( n )
        {
            for ( size_t l = 0; l < stride_t; ++l )
                coeffs[ n*stride_t + l ] = coeffs[ (n-1)*stride_t + l ];
        }
        interpolate<real,order>( coeffs.get() + n*stride_t, rho.get(), conf );

        real Emax = 0;
        for ( size_t l = 0; l < conf.Nx * conf.Ny; ++l )
        {
            size_t i = l % conf.Nx;
            size_t j = l / conf.Nx;
            real x = conf.x_min + i*conf.dx;
            real y = conf.y_min + j*conf.dy;

            real Ex = -eval<real,order,1,0>( x, y, coeffs.get() + n*stride_t, conf );
            real Ey = -eval<real,order,0,1>( x, y, coeffs.get() + n*stride_t, conf );

            Emax = max( Emax, hypot(Ex,Ey) );
        }
        real elapsed = clock.elapsed();
        Emax_file << std::setw(15) << n*conf.dt << std::setw(15) << std::setprecision(5) << std::scientific << Emax << std::endl; 
        std::cout << std::setw(15) << n*conf.dt << std::setw(15) << std::setprecision(5) << std::scientific << Emax << std::endl; 

        std::cout << "This time step took: " << elapsed << "s." <<std::endl;

        /*
        std::stringstream filename; filename << 'f' << n << ".txt"; 
        std::ofstream file( filename.str() );
        const size_t plotNu = 512, plotNx = 512;
        for ( size_t i = 0; i <= plotNu; ++i )
        {
            real u = conf.u_min + i*(conf.u_max-conf.u_min)/plotNu;
            for ( size_t j = 0; j <= plotNx; ++j )
            {
                real x = conf.x_min + j*(conf.x_max-conf.x_min)/plotNx;
                file << x << " " << u << " " << dim2::eval_f<real,order>( n, x, 0, u, 0, coeffs.get(), conf ) << std::endl;
            }
            file << std::endl;
        }
		*/
    }

}

}

}


int main()
{
    dergeraet::dim2::test<float,4>();
}

