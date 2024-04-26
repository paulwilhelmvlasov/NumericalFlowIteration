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

template<typename real>
void transform_to_polar_coordinates(real x, real y, real& r, real& phi)
{
	r = std::sqrt(x*x + y*y);

	if(std::abs(x) < 1e-16){
		if(y > 0){
			phi = M_PI/2.0;
		}else{
			phi = -M_PI/2.0;
		}
	}else{
		if(x > 0){
			phi = std::atan(y/x);
		}else{
			if(y < 0){
				phi = std::atan(y/x) - M_PI;
			}else{
				phi = std::atan(y/x) + M_PI;
			}
		}
	}
}

template<typename real>
void transform_to_plain_coordinates(real r, real phi, real& x, real& y)
{
	x = r*std::cos(phi);
	y = r*std::sin(phi);
}

void transform_to_fourier_space(size_t Nx, size_t Nv, const config_t<double>& conf, std::vector<double>& F)
{
    std::unique_ptr<double,decltype(std::free)*> data { reinterpret_cast<double*>(
    		std::aligned_alloc(64,sizeof(double)*Nx*Nv)), std::free };
    if ( data == nullptr ) throw std::bad_alloc {};

    for(size_t i = 0; i < Nx; i++){
    	for(size_t j = 0; j < Nv; j++){
    		size_t pos = i + j*Nx;
    		data[pos] = F[pos];
    	}
    }


    using memptr = std::unique_ptr<double,decltype(std::free)*>;

    size_t mem_size  = sizeof(double) * Nx * Nv;
    void *tmp = std::aligned_alloc( 64, mem_size ); // alignement_double = 64
    if ( tmp == nullptr ) throw std::bad_alloc {};
    memptr mem { reinterpret_cast<double*>(tmp), &std::free };

    fftw_plan plan = fftw_plan_r2r_2d( Nx, Nv, mem.get(), mem.get(),
                             FFTW_DHT, FFTW_DHT, FFTW_MEASURE );

    fftw_execute_r2r( plan, data, data );

    for(size_t i = 0; i < Nx; i++){
    	for(size_t j = 0; j < Nv; j++){
    		size_t pos = i + j*Nx;
    		F[pos] = data[pos];
    	}
    }
}


template <typename real, size_t order>
void read_in_coeffs(const std::string& i_str)
{
    config_t<real> conf;
    const size_t stride_x = 1;
    const size_t stride_t = conf.Nx + order - 1;

    size_t n = conf.Nt;

    std::unique_ptr<real[]> coeffs { new real[ conf.Nt*stride_t ] {} };

	std::ifstream coeff_str(i_str);
	for(size_t i = 0; i <= conf.Nt*stride_t; i++){
		coeff_str >> coeffs[i];
	}

	size_t Nx_plot = 1024;
    real dx_plot = conf.Lx / Nx_plot;
	real t = n * conf.dt;
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

	std::ofstream file_phase_flow_x( "phase_flow_x_" + std::to_string(t) + ".txt" );
	std::ofstream file_phase_flow_v( "phase_flow_v_" + std::to_string(t) + ".txt" );
	for(size_t i = 0; i<=Nx_plot; i++)
	{
		for(size_t j = 0; j <= Nv_plot; j++)
		{
			real x = conf.x_min + i*dx_plot;
			real v = conf.u_min + j*dv_plot;
			real x_new = x;
			real v_new = v;
			eval_phase_flow<real, order>(n, x_new, v_new, coeffs.get(), conf);

			file_phase_flow_x << x << " " << v << " " << x_new << std::endl;
			file_phase_flow_v << x << " " << v << " " << v_new << std::endl;
		}
		file_phase_flow_x << std::endl;
		file_phase_flow_v << std::endl;
	}

	// Plot in polar coordinates:
	/*
	size_t N_r_plot = 1024;
	size_t N_phi_plot = N_r_plot;

	real r_max = conf.Lx / 2.0;
	real phi_min = -M_PI;
	real phi_max = M_PI;
	real dr_plot = r_max/N_r_plot;
	real dphi_plot = (phi_max-phi_min)/N_phi_plot;

	std::ofstream file_f_polar( "f_polar_" + std::to_string(t) + ".txt" );
	std::ofstream file_phase_flow_r( "phase_flow_r_" + std::to_string(t) + ".txt" );
	std::ofstream file_phase_flow_phi( "phase_flow_phi_" + std::to_string(t) + ".txt" );
	for(size_t i = 0; i <= N_r_plot; i++){
		for(size_t j = 0; j <= N_phi_plot; j++){
			real r = i*dr_plot;
			real phi = phi_min + j*dphi_plot;
			real x,v;
			transform_to_plain_coordinates<real>(r, phi, x, v);

			real f = eval_f<real, order>(n, x, v, coeffs.get(), conf);
			file_f_polar << r << " " << phi << " " << f << std::endl;

			real x_new = x;
			real v_new = v;
			eval_phase_flow<real, order>(n, x_new, v_new, coeffs.get(), conf);
			real r_new, phi_new;
			transform_to_polar_coordinates<real>(x_new, v_new, r_new, phi_new);

			file_phase_flow_r << r << " " << phi << " " << r_new << std::endl;
			file_phase_flow_phi << r << " " << phi << " " << phi_new << std::endl;
		}
		file_f_polar << std::endl;
		file_phase_flow_r << std::endl;
		file_phase_flow_phi << std::endl;
	}
	*/

}

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
    conf_metrics.Nx = conf.Nx;//1024;
    conf_metrics.Nu = conf.Nu;//1024;
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

    std::ofstream coeffs_str( "coeffs_Nt_" + std::to_string(conf.Nt) + "_Nx_ " + std::to_string(conf.Nx) + "_stride_t_" + std::to_string(stride_t) + ".txt" );
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


        if( n % 1 == 0 )
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

            if(n % (10*16) == 0 )
            {
				size_t Nx_plot = 1024;
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

			    arma::mat pf_x(Nx_plot+1, Nv_plot+1);
			    arma::mat pf_v(Nx_plot+1, Nv_plot+1);
			    arma::mat f_mat(Nx_plot+1, Nv_plot+1);

				std::ofstream file_f( "f_" + std::to_string(t) + ".txt" );
				for(size_t i = 0; i<=Nx_plot; i++)
				{
					for(size_t j = 0; j <= Nv_plot; j++)
					{
						real x = conf.x_min + i*dx_plot;
						real v = conf.u_min + j*dv_plot;
						real f = eval_f<real, order>(n, x, v, coeffs.get(), conf);
						
						file_f << x << " " << v << " " << f << std::endl;
						f_mat(i,j) = f;
					}
					file_f << std::endl;
				}

				std::ofstream file_phase_flow_x( "phase_flow_x_" + std::to_string(t) + ".txt" );
				std::ofstream file_phase_flow_v( "phase_flow_v_" + std::to_string(t) + ".txt" );
				for(size_t i = 0; i<=Nx_plot; i++)
				{
					for(size_t j = 0; j <= Nv_plot; j++)
					{
						real x = conf.x_min + i*dx_plot;
						real v = conf.u_min + j*dv_plot;
						real x_new = x;
						real v_new = v;
						eval_phase_flow<real, order>(n, x_new, v_new, coeffs.get(), conf);

						file_phase_flow_x << x << " " << v << " " << x_new << std::endl;
						file_phase_flow_v << x << " " << v << " " << v_new << std::endl;

						pf_x(i,j) = x_new;
					    pf_v(i,j) = v_new;
					}
					file_phase_flow_x << std::endl;
					file_phase_flow_v << std::endl;
				}

				arma::vec s_x = arma::svd(pf_x);
				arma::vec s_v = arma::svd(pf_v);
				arma::vec s_f = arma::svd(f_mat);

				std::ofstream file_s_x("s_x_" + std::to_string(t) + ".txt" );
				file_s_x << s_x;
				std::ofstream file_s_v("s_v_" + std::to_string(t) + ".txt" );
				file_s_v << s_v;
				std::ofstream file_s_f("s_f_" + std::to_string(t) + ".txt" );
				file_s_f << s_f;
            }
        }

        for(size_t i = 0; i < stride_t; i++){
        	coeffs_str << coeffs.get()[n*stride_t + i] << std::endl;
        }
    }
    
    std::cout << "Elapsed time = " << total_time << std::endl;
}

}
}


int main()
{
    //dergeraet::dim1::test<float,4>();
	dergeraet::dim1::read_in_coeffs<float,4>(std::string("../coeffs_Nt_1600_Nx_ 256_stride_t_259.txt"));
}

