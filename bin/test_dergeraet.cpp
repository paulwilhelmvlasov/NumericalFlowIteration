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

namespace dim1
{

template <typename real, size_t order>
void test()
{
    using std::abs;
    using std::max;
    using std::sqrt;

    config_t<real> conf;
    const size_t stride_x = 1;
    const size_t stride_t = conf.Nx + order - 1;

    std::unique_ptr<real[]> coeffs { new real[ conf.Nt*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*conf.Nx)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};

    poisson<real> poiss( conf );

    cuda_scheduler<real,order> sched { conf };

    std::ofstream statistics_file( "statistics.csv" );
    statistics_file << R"("Time"; "L1-Norm"; "L2-Norm"; "Electric Energy"; "Kinetic Energy"; "Total Energy"; "Entropy")";
    statistics_file << std::endl;

    statistics_file << std::scientific;
          std::cout << std::scientific;

    for ( size_t n = 0; n < conf.Nt; ++n )
    {
        std::memset( rho.get(), 0, conf.Nx*sizeof(real) );
        sched.compute_rho ( n, 0, conf.Nx*conf.Nu );
        sched.download_rho( rho.get() );

        real electric_energy = poiss.solve( rho.get() );
        dim1::interpolate<real,order>( coeffs.get() + n*stride_t, rho.get(), conf );

        sched.upload_phi( n, coeffs.get() );

        real metrics[4] = { 0, 0, 0, 0 };
        sched.compute_metrics( n, 0, conf.Nx*conf.Nu );
        sched.download_metrics( metrics );

        real kinetic_energy = metrics[2];
        real   total_energy = electric_energy + kinetic_energy;

        metrics[1]  = sqrt(metrics[1]);

        for ( size_t i = 0; i < 80; ++i )
            std::cout << '=';
        std::cout << std::endl;

        std::cout << "t = " << conf.dt*n  << ".\n";

        std::cout << "L¹-Norm:      " << std::setw(20) << metrics[0]      << std::endl;
        std::cout << "L²-Norm:      " << std::setw(20) << metrics[1]      << std::endl;
        std::cout << "Total energy: " << std::setw(20) << total_energy    << std::endl;
        std::cout << "Entropy:      " << std::setw(20) << metrics[3]      << std::endl;

        std::cout << std::endl;

        statistics_file << conf.dt*n       << "; "
                        << metrics[0]      << "; "
                        << metrics[1]      << "; "
                        << electric_energy << "; "
                        <<  kinetic_energy << "; "
                        <<    total_energy << "; "
                        << metrics[3]      << std::endl;
                       

        /*
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
   
        Emax_file << std::setw(15) << n*conf.dt << std::setw(15) << std::setprecision(5) << std::scientific << Emax << std::endl; 
        std::cout << std::setw(15) << n*conf.dt << std::setw(15) << std::setprecision(5) << std::scientific << Emax << std::endl; 

        std::stringstream filename = std::stringstream {}; filename << 'f' << n << ".txt"; 
        std::ofstream file( filename.str() );
        const size_t plotNu = 512, plotNx = 512;
        for ( size_t i = 0; i <= plotNu; ++i )
        {
            real u = conf.u_min + i*(conf.u_max-conf.u_min)/plotNu;
            for ( size_t j = 0; j <= plotNx; ++j )
            {
                real x = conf.x_min + j*(conf.x_max-conf.x_min)/plotNx;
                file << x << " " << u << " " << eval_f<real,order>( n, x, u, coeffs.get(), conf ) << std::endl;
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
    dergeraet::dim1::test<double,4>();
}

