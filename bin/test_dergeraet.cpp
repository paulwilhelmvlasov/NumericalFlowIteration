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

    config_t<real> conf_metrics;
    // To compute metrics use higher amount of quadrature points:
    conf_metrics.Nx = 128;//1024;
    conf_metrics.Nu = 128;//1024;
    conf_metrics.dx = (conf_metrics.x_max - conf_metrics.x_min) / conf_metrics.Nx;
    conf_metrics.du = (conf_metrics.u_max - conf_metrics.u_min) / conf_metrics.Nu;


    std::unique_ptr<real[]> coeffs { new real[ conf.Nt*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*conf.Nx)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};

    poisson<real> poiss( conf );

    cuda_scheduler<real,order> sched { conf, conf_metrics };

    std::ofstream statistics_file( "statistics.csv" );
    statistics_file << R"("Time"; "L1-Norm"; "L2-Norm"; "Electric Energy"; "Kinetic Energy"; "Total Energy"; "Entropy")";
    statistics_file << std::endl;

    statistics_file << std::scientific;
          std::cout << std::scientific;

    double total_time = 0;
    for ( size_t n = 0; n <= conf.Nt; ++n )
    {
        dergeraet::stopwatch<double> timer;
        std::memset( rho.get(), 0, conf.Nx*sizeof(real) );
        sched.compute_rho ( n, 0, conf.Nx*conf.Nu );
        sched.download_rho( rho.get() );

        real electric_energy = poiss.solve( rho.get() );
        dim1::interpolate<real,order>( coeffs.get() + n*stride_t, rho.get(), conf );

        sched.upload_phi( n, coeffs.get() );
        double time_elapsed = timer.elapsed();
        total_time += time_elapsed;

        std::cout << std::setw(15) << n << " Comp-time: " << time_elapsed << " Total time: " << total_time << std::endl;

        /*
        if( n % 16 == 0 )
        {
            real metrics[4] = { 0, 0, 0, 0 };
            sched.compute_metrics( n, 0, conf_metrics.Nx*conf_metrics.Nu );
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

            statistics_file << std::setprecision(16) << conf.dt*n       << "; "
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
				real t = n * conf.dt;
				std::ofstream file_E( "E_" + std::to_string(t) + ".txt" );
				std::ofstream file_rho( "rho_" + std::to_string(t) + ".txt" );
				for(size_t i = 0; i <= Nx_plot; i++)
				{
					real x = conf.x_min + i*dx_plot;
					real E = -dim1::eval<real,order,1>( x, coeffs.get() + n*stride_t, conf );
					real rho = -dim1::eval<real,order,2>( x, coeffs.get() + n*stride_t, conf );

					file_E << x << " " << E << std::endl;
					file_rho << x << " " << rho << std::endl;
				}

				size_t Nv_plot = Nx_plot;
				real dv_plot = (conf.u_max - conf.u_min)/Nv_plot;

				std::ofstream file_f( "f_" + std::to_string(t) + ".txt" );
				for(size_t i = 0; i<=Nx_plot; i++)
				{
					for(size_t j = 0; j <= Nv_plot; j++)
					{
						real x = conf.x_min + i*dx_plot;
						real v = conf.u_min + j*dv_plot;
						real f = eval_f<real, order>(n, x, v, coeffs.get(), conf);
						
						file_f << x << " " << v << " " << f << std::endl;
					}
					file_f << std::endl;
				}
            }
        }
        */
    }
    
    std::cout << "Elapsed time = " << total_time << std::endl;


    // Debugging/Testing:
    /*
    std::cout << "Let's try something out: " << std::endl;
	size_t Nx_plot = 256;
	real dx_plot = conf.Lx / Nx_plot;
    std::vector<real> exact_rho(Nx_plot);
    real rho_max = 0;
    real rho_l2 = 0;
	for(size_t i = 0; i < Nx_plot; i++)
	{
		real x = conf.x_min + i*dx_plot;
		exact_rho[i] = -dim1::eval<real,order,2>( x, coeffs.get() + conf.Nt*stride_t, conf );
		rho_max = max(abs(exact_rho[i]),rho_max);
		rho_l2 = exact_rho[i]*exact_rho[i];
	}

	rho_l2 = sqrt(rho_l2)*dx_plot;
	std::cout << "rho_exact: L2-norm =" << rho_l2 << " Max-norm = " << rho_max << std::endl;

	size_t Nv_plot = 128;
	real dv_plot = (conf.u_max - conf.u_min)/Nv_plot;
	std::vector<real> num_rho_midpoint(Nx_plot, 0);
	std::vector<real> num_rho_simpson(Nx_plot, 0);


	real max_error_mp = 0;
	real max_error_simpson = 0;
	real l2_error_mp = 0;
	real l2_error_simpson = 0;
	#pragma omp parallel for
	for(size_t i = 0; i < Nx_plot; i++)
	{
		real x = conf.x_min + i*dx_plot;

		// Compute rho with mid-point rule:
		for(size_t j = 0; j < Nv_plot; j++)
		{
			real v = conf.u_min + j*dv_plot;
			num_rho_midpoint[i] += eval_ftilda<real,order>( conf.Nt, x, v, coeffs.get(), conf );
		}
		num_rho_midpoint[i] = 1 - dv_plot*num_rho_midpoint[i];

		// Compute rho with simpson-rule:
		for(size_t j = 0; j < Nv_plot; j++)
		{
		    real left = conf.u_min + j*dv_plot;
			real mid = left + 0.5*dv_plot;
			real right = left + dv_plot;
		    real f_left = eval_ftilda<real,order>( conf.Nt, x, left, coeffs.get(), conf );
		    real f_mid = eval_ftilda<real,order>( conf.Nt, x, mid, coeffs.get(), conf );
		    real f_right = eval_ftilda<real,order>( conf.Nt, x, right, coeffs.get(), conf );

		    num_rho_simpson[i] += 1.0/6.0*f_left + 4.0/6.0*f_mid + 1.0/6.0*f_right;
		}
		num_rho_simpson[i] = 1 - dv_plot*num_rho_simpson[i];

		real dist_mp = abs(num_rho_midpoint[i] - exact_rho[i]);
		real dist_simpson = abs(num_rho_simpson[i] - exact_rho[i]);

		max_error_mp = max(dist_mp, max_error_mp);
		max_error_simpson = max(dist_simpson, max_error_simpson);

		l2_error_mp += dist_mp*dist_mp;
		l2_error_simpson += dist_simpson*dist_simpson;
	}

	l2_error_mp = sqrt(l2_error_mp)*dx_plot;
	l2_error_simpson = sqrt(l2_error_simpson)*dx_plot;

	std::cout << "MP: Max-error = " << max_error_mp << " L2-error = " << l2_error_mp << std::endl;
	std::cout << "Simpson: Max-error = " << max_error_simpson << " L2-error = " << l2_error_simpson << std::endl;

	std::cout << "MP: Rel-Max-error = " << max_error_mp/rho_max << " Rel-L2-error = " << l2_error_mp/rho_l2 << std::endl;
	std::cout << "Simpson: Rel-Max-error = " << max_error_simpson/rho_max << " Rel-L2-error = " << l2_error_simpson/rho_l2 << std::endl;
     */
}

}
}


int main()
{
    dergeraet::dim1::test<double,4>();
}

