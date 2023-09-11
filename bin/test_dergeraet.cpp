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
#include <random>
#include <math.h>

#include <armadillo>

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
    for ( size_t n = 0; n < conf.Nt; ++n )
    {
        dergeraet::stopwatch<double> timer;
        std::memset( rho.get(), 0, conf.Nx*sizeof(real) );
        sched.compute_rho ( n, 0, conf.Nx*conf.Nu );
        sched.download_rho( rho.get() );

        real electric_energy = poiss.solve( rho.get() );
        interpolate<real,order>( coeffs.get() + n*stride_t, rho.get(), conf );

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

            if(n % (100*16) == 0)
            {
				size_t Nx_plot = 256;
				real dx_plot = conf.Lx / Nx_plot;
				real t = n * conf.dt;
				std::ofstream file_E( "E_" + std::to_string(t) + ".txt" );
				std::ofstream file_rho( "rho_" + std::to_string(t) + ".txt" );
				for(size_t i = 0; i <= Nx_plot; i++)
				{
					real x = conf.x_min + i*dx_plot;
					real E = -dim1::periodic::eval<real,order,1>( x, coeffs.get() + n*stride_t, conf );
					real rho = -dim1::periodic::eval<real,order,2>( x, coeffs.get() + n*stride_t, conf );
					file_E << x << " " << E << std::endl;
					file_rho << x << " " << rho << std::endl;
				}

            }
        }
    }
      

    std::cout << "Elapsed time = " << total_time << std::endl;
}

template <typename real, size_t order>
void test_dirichlet()
{
    using std::abs;
    using std::max;
    using std::sqrt;

    config_t<real> conf;
    const size_t stride_x = 1;
    const size_t stride_t = conf.Nx + order - 2; // conf.Nx = l+2, therefore: conf.Nx +order-2 = order+l
    
    const size_t n_total = conf.l + order;
    std::unique_ptr<real[]> coeffs { new real[ conf.Nt*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*conf.Nx)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};
    std::unique_ptr<real,decltype(std::free)*> rho_ext { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*n_total)), std::free };//rename, this is the rhs of the interpolation task, but with 2 additional entries.

    poisson<real> poiss( conf );

    cuda_scheduler<real,order> sched { conf };

    std::ofstream statistics_file( "statistics.csv" );
    statistics_file << R"("Time"; "L1-Norm"; "L2-Norm"; "Electric Energy"; "Kinetic Energy"; "Total Energy"; "Entropy")";
    statistics_file << std::endl;

    statistics_file << std::scientific;
          std::cout << std::scientific;
    std::cout<<conf.Nt<<std::endl;
    double total_time = 0;
  
    for ( size_t n = 0; n < 1;n++)//conf.Nt; n++ )
    {
    
    dergeraet::stopwatch<double> timer;
    std::memset( rho.get(), 0, conf.Nx*sizeof(real) );
    sched.compute_rho ( n, 0, conf.Nx*conf.Nu );
    sched.download_rho( rho.get() );

    real electric_energy = poiss.solve( rho.get() );
   
    
    
    // this is the value to the left of a.
    rho_ext.get()[0]=rho.get()[0];

    // this is the value to the right of b.
    for(size_t i = 0; i<order-3;i++){
        rho_ext.get()[n_total-1-i] = rho.get()[conf.Nx-1];
    }
    //these are all values in [a,b)
    for(size_t i = 1; i<conf.Nx+1;i++){
        rho_ext.get()[i] = rho.get()[i-1];
        
    }
    //this is the value for b. so far, its the last value of the rho array.
    rho_ext.get()[n_total-2] = rho.get()[conf.Nx-1];

std::cout<<"test1"<<std::endl;

    std::ofstream f_file("f.txt");
    for(int i = 0; i<n_total; i++){
        
        std::cout << rho_ext.get()[i]<<"\n";
    }
    f_file.close();
    std::cout<<"test2"<<std::endl;
    dim1::dirichlet::interpolate<real,order>( coeffs.get() + n*stride_t,rho_ext.get(), conf );
    if(n==0){
        std::ofstream coeff_file ("coeffs.txt");
        for(int j = 0; j<n_total; j++){
            coeff_file<<coeffs[j+n*stride_t]<<"\n";
        }
        coeff_file.close();
    }
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
				size_t Nx_plot = 10000;
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
                    //outputfile<<x<<"\t"<< -dim1::eval<real,order,1>( x, coeffs.get() + n*stride_t, conf )<<"\n";
					file_E << x << " " << E << std::endl;
					file_rho << x << " " << rho << std::endl;
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
    //dergeraet::dim1::periodic::test<double,4>();
    dergeraet::dim1::periodic::test_dirichlet<double,4>();

	const double a = 0;
	const double b = 2*M_PI;
	const size_t Nx = 8;
	const double dx = (b-a)/(Nx);
	const size_t order = 4;
	const size_t l = Nx -1;
	const size_t n = l + order;
	std::vector<double> N_vec(order);
	dergeraet::splines1d::N<double,order>(0, N_vec.data());

	dergeraet::dim1::config_t<double> conf;
	conf.Nx = Nx;
	conf.x_min = a;
	conf.x_max = b;
	conf.dx = dx;
	conf.dx_inv = 1.0/dx;
	conf.Lx = (b-a);
	conf.Lx_inv = 1.0/conf.Lx;

	arma::vec rhs(n, arma::fill::zeros);


	for(int i = 0; i <= l+1; i++)
	{
		double x = a + i*dx;
		rhs(i+1) = std::sin(x);

		std::cout << x << " " << rhs(i+1) << std::endl;
	}
	rhs(0) = rhs(1);
	std::cout << a-dx << " " << rhs(0) << std::endl;
	rhs(n-1) = rhs(n-2);
	std::cout << b+dx << " " << rhs(n-1) << std::endl;


	arma::mat A(n, n, arma::fill::zeros);

    for ( size_t i = 0; i < n; ++i )
    {
        if ( i == 0 )
        {
            for ( size_t ii = 0; ii < order-1; ++ii ){
            	A(i, i+ii) = N_vec[ii+1];
            }
        }
        else if(i<l+3){
            for ( size_t ii = 0; ii < order-1; ++ii )
            	A(i, i + ii -1) = N_vec[ii];
        }
        else
        {
            for ( size_t ii = 0; ii < order-2; ++ii )
            {
            	A(i,i+ii-1) = N_vec[ii];
            }
        }
    }

    std::cout << A << std::endl;

    arma::vec coeffs(n);
    arma::solve(coeffs, A, rhs);


	for(int i = 0; i < n; i++)
	{
		double x = a + (i-1)*dx;
		double approx_result = dergeraet::dim1::dirichlet::eval<double, order>(x, coeffs.memptr(), conf);
		double exact = rhs(i);
		std::cout << x << " " << approx_result << " " << exact << " " << std::abs(approx_result - exact) << std::endl;
	}

	std::cout << rhs << std::endl;
	std::cout << coeffs << std::endl;

	std::cout << std::floor(-1) << std::endl;
	std::cout << std::floor(-1.01) << std::endl;
	std::cout << std::floor(-0.99) << std::endl;

	double x_knot = std::floor((a-dx)*conf.dx_inv);

	std::cout << x_knot << std::endl;
	std::cout << static_cast<int>(x_knot) << std::endl;
}

