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
#include <fstream>
#include <sstream>
#include <random>
#include <math.h>

#include <armadillo>

#include <nufi/config.hpp>
#include <nufi/random.hpp>
#include <nufi/fields.hpp>
#include <nufi/poisson.hpp>
#include <nufi/rho.hpp>
#include <nufi/stopwatch.hpp>
#include <nufi/cuda_scheduler.hpp>
#include <nufi/finite_difference_poisson.hpp>

namespace nufi
{

namespace dim1
{
namespace periodic
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

    double total_time = 0;
    for ( size_t n = 0; n <= conf.Nt; ++n )
    {
        nufi::stopwatch<double> timer;
        std::memset( rho.get(), 0, conf.Nx*sizeof(real) );
        sched.compute_rho ( n, 0, conf.Nx*conf.Nu );
        sched.download_rho( rho.get() );

        real electric_energy = poiss.solve( rho.get() );
        interpolate<real,order>( coeffs.get() + n*stride_t, rho.get(), conf );

        sched.upload_phi( n, coeffs.get() );
        double time_elapsed = timer.elapsed();
        total_time += time_elapsed;

        if( n % 1 == 0 )
        {
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

            if(n % (10*16) == 0)
            {
				size_t Nx_plot = 256;
				real dx_plot = conf.Lx / Nx_plot;
				size_t Nv_plot = Nx_plot;
				real dv_plot = (conf.u_max - conf.u_min) / Nv_plot;
				real t = n * conf.dt;
				std::ofstream file_E( "E_" + std::to_string(t) + ".txt" );
				std::ofstream file_rho( "rho_" + std::to_string(t) + ".txt" );
				std::ofstream file_rho_electron( "rho_electron_" + std::to_string(t) + ".txt" );
				std::ofstream file_u( "u_" + std::to_string(t) + ".txt" );
				std::ofstream file_p( "p_" + std::to_string(t) + ".txt" );
				std::ofstream file_q( "q_" + std::to_string(t) + ".txt" );
				std::ofstream file_R( "R_" + std::to_string(t) + ".txt" );
				std::ofstream file_f( "f_" + std::to_string(t) + ".txt" );
				std::vector<real> f_cross(Nv_plot + 1);
				for(size_t i = 0; i <= Nx_plot; i++)
				{
					real x = conf.x_min + i*dx_plot;
					real E = -dim1::periodic::eval<real,order,1>( x, coeffs.get() + n*stride_t, conf );
					real rho = -dim1::periodic::eval<real,order,2>( x, coeffs.get() + n*stride_t, conf );
					file_E << x << " " << E << std::endl;
					file_rho << x << " " << rho << std::endl;
					real u = 0;
					real p = 0;
					real q = 0;
					real R = 0;
					for(size_t j = 0; j <= Nv_plot; j++)
					{
						real v = conf.u_min + j*dv_plot;
						real f = eval_f<real, order>(n, x, v, coeffs.get(), conf);
						u += v*f;

						f_cross[j] = f;

						file_f << x << " " << v << " " << f << std::endl;
					}
					file_f << std::endl;
					u *= 1/(1-rho)*dv_plot;
					for(size_t j = 0; j <= Nv_plot; j++)
					{
						real v = conf.u_min + j*dv_plot;
						real c = v-u;
						real f = f_cross[j];

						p += c*c*f;
						q += c*c*c*f;
						R += c*c*c*c*f;
					}
					p *= dv_plot;
					q *= dv_plot;
					R *= dv_plot;

					file_rho_electron << x << " " << 1-rho << std::endl;
					file_u << x << " " << u << std::endl;
					file_p << x << " " << p << std::endl;
					file_q << x << " " << q << std::endl;
					file_R << x << " " << R << std::endl;
				}

            }
        }
    }
      

    std::cout << "Elapsed time = " << total_time << std::endl;
}
}
namespace dirichlet{

template <typename real, size_t order>
void test()
{
    
    using std::abs;
    using std::max;
    using std::sqrt;

    config_t<real> conf;
    const size_t stride_x = 1;
    const size_t stride_t = conf.Nx + order - 2; // conf.Nx = l+2, therefore: conf.Nx +order-2 = order+l
    
    const size_t n_total = conf.l + order;
    std::unique_ptr<real[]> coeffs { new real[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*conf.Nx)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};
    std::unique_ptr<real,decltype(std::free)*> rho_ext { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*n_total)), std::free };//rename, this is the rhs of the interpolation task, but with 2 additional entries.

     
    poisson_fd_dirichlet<double> poiss(conf);

    cuda_scheduler<real,order> sched { conf };
    
    std::ofstream statistics_file( "statistics.csv" );
    statistics_file << R"("Time"; "L1-Norm"; "L2-Norm"; "Electric Energy"; "Kinetic Energy"; "Total Energy"; "Entropy")";
    statistics_file << std::endl;

    statistics_file << std::scientific;
          std::cout << std::scientific;
    std::cout<<conf.Nt<<std::endl;
    double total_time = 0;
  
  //  for ( size_t n = 0; n <= conf.Nt; n++ )
    for ( size_t n = 0; n <= 1; n++ )
    {
    
    nufi::stopwatch<double> timer;
    std::memset( rho.get(), 0, conf.Nx*sizeof(real) );
    //sched.compute_rho ( n, 0, conf.Nx*conf.Nu ); // Auf Dirichlet kernel umstellen.
    //sched.download_rho( rho.get() );

	// Compute rho:
	#pragma omp parallel for
	for(size_t i = 0; i<conf.Nx; i++)
	{
		rho.get()[i] = eval_rho<real,order>(n, i, coeffs.get(), conf);
	}


    real electric_energy = 1;//poiss.solve( rho.get() ); // Auf Dirichlet Poisson umstellen.
    // phi am Rand auf 0 setzen. (rho_ext auf 0 setzen)
   
    // this is the value to the left of a.
    rho_ext.get()[0]=0;

    // this is the value to the right of b.
    for(size_t i = 0; i<order-3;i++){
        rho_ext.get()[n_total-1-i] = 0;
    }
    //these are all values in [a,b)
    for(size_t i = 1; i<conf.Nx+1;i++){
        rho_ext.get()[i] = rho.get()[i-1];
        
    }
    //this is the value for b. so far, its the last value of the rho array.
    rho_ext.get()[n_total-2] = rho.get()[conf.Nx-1];

    if(n== 0){
        std::ofstream dirfile("rho_dir.txt");
        for(size_t i = 0; i<conf.Nx+1;i++){
            dirfile<< conf.x_min+i*conf.dx << " " <<std::setprecision(3)<<rho.get()[i]<<"\n";
        }           
        dirfile.close();
    }
    for(int i = 0; i<n_total; i++){
        
        std::cout << rho_ext.get()[i]<<"\n";
    }
    dim1::dirichlet::interpolate<real,order>( coeffs.get() + n*stride_t,rho_ext.get(), conf );
    sched.upload_phi( n, coeffs.get() );
    double time_elapsed = timer.elapsed();
    total_time += time_elapsed;

        if( n % 2 == 0 )
        {
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

            if(n %(100*16) == 0)
            {
				size_t Nx_plot = 128;
				real dx_plot = conf.Lx / Nx_plot;
				real t = n * conf.dt;
				std::ofstream file_E( "E_" + std::to_string(t) + ".txt" );
				std::ofstream file_rho( "rho_" + std::to_string(t) + ".txt" );
                std::ofstream outputfile("result.txt");
                std::cout<< conf.x_min + Nx_plot * dx_plot<<std::endl;
				for(size_t i = 0; i <= Nx_plot; i++)
				{
                    
					real x = conf.x_min + i*dx_plot;
					real E = -dim1::dirichlet::eval<real,order,1>( x, coeffs.get() + n*stride_t, conf ); //maybe have to change stride?
					real rho = -dim1::dirichlet::eval<real,order,2>( x, coeffs.get() + n*stride_t, conf );
					real phi = dim1::dirichlet::eval<real,order>( x, coeffs.get() + n*stride_t, conf );
                    //outputfile<<x<<"\t"<< -dim1::eval<real,order,1>( x, coeffs.get() + n*stride_t, conf )<<"\n";
					file_E << x << " " << E << std::endl;
					file_rho << x << " " << rho << std::endl;
					// file_phi ....
                    //outputfile<<x<<"\t"<< dim1::dirichlet::eval<real,order,0>( x, coeffs.get() + n*stride_t, conf )<<"\n";

					
				}
                for(int i = 0; i<conf.l + order  ; i++){
                    real x = conf.x_min + (i-1)*conf.dx;
                    outputfile<<i<<"\t"<<x<<"\t"<<rho_ext.get()[i]<<"\t"<<dim1::dirichlet::eval<real,order,0>( x, coeffs.get() + n*stride_t, conf )<<"\n";
                                    }


            }
        }
    }

    std::cout << "Elapsed time = " << total_time << std::endl;
}
}
}

}


int main()
{
    nufi::dim1::periodic::test<double,4>();
    //nufi::dim1::dirichlet::test<double,4>();
}

