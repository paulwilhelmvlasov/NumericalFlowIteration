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

#include <algorithm>
#include <cmath>
#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>


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
void read_coeffs(const config_t<real>& conf, std::vector<real>& coeffs_old, size_t old_n, const std::string& i_str)
{
    const size_t stride_t = conf.Nx + order - 1;
	std::ifstream coeff_str(i_str);
	for(size_t i = 0; i <= old_n*stride_t; i++){
		coeff_str >> coeffs_old[i];
		//std::cout << i << " of " << old_n*stride_t << " coeff[i] = " << coeffs_old[i] << std::endl;
	}
}

template <typename real, size_t order>
void ion_acoustic_restart(const config_t<real>& conf, const std::vector<real>& coeffs_old, size_t old_n)
{
    using std::abs;
    using std::max;

	// This method allows to read in an old set of coefficients and start from that rather than
	// rerunning the entire simulation with the same parameters.
    const size_t stride_t = conf.Nx + order - 1;

    std::unique_ptr<real[]> coeffs { new real[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*conf.Nx)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};
    std::vector<real> rho_electron(conf.Nx);
    std::vector<real> rho_ion(conf.Nx);

    // Copy old coefficients into coeff vector:
    // Also start writing the coefficients onto disc to continue writing
    // at the right position later on.
    std::ofstream coeffs_str( "restart_coeffs_Nt_" + std::to_string(conf.Nt) + "_Nx_ " + std::to_string(conf.Nx) + "_stride_t_" + std::to_string(stride_t) + ".txt" );
    for(size_t i = 0; i <= old_n*stride_t; i++){
    	coeffs.get()[i] = coeffs_old[i];
    	coeffs_str << coeffs.get()[i] << std::endl;
		//std::cout << i << " of " << old_n*stride_t << " coeff[i] = " << coeffs.get()[i] << std::endl;
    }

    // Maybe it would be nice to compute these initial bounds automatically instead of setting
    // them manually. This would give some additional robustness to the approach.
    std::vector<real> velocity_support_electron_lower_bound(conf.Nx, -10);
    std::vector<real> velocity_support_electron_upper_bound(conf.Nx, 8);
    std::vector<real> velocity_support_ion_lower_bound(conf.Nx, -0.35);
    std::vector<real> velocity_support_ion_upper_bound(conf.Nx, 0.35);

    poisson<real> poiss( conf );

    bool write_stats = true;
    std::ofstream stats_file( "restart_stats.txt" );
    double total_time = 0;
    for ( size_t n = old_n+1; n <= conf.Nt; ++n )
    {
    	std::cout << "Start " << n << std::endl;
    	dergeraet::stopwatch<double> timer;
    	// Compute rho:
		//#pragma omp parallel for
    	for(size_t i = 0; i<conf.Nx; i++)
    	{
    		// Careful: This seems to be buggy. Probably there is an issue with the way I read in and store the
    		// coefficients...
    		std::cout << i;
    		real x = conf.x_min + i*conf.dx;
    		rho_ion[i] = eval_rho_adaptive_trapezoidal_rule<real,order>(n, x, coeffs.get(), conf, velocity_support_ion_lower_bound[i],
					velocity_support_ion_upper_bound[i], false);
    		std::cout << " ion";
    		rho_electron[i] = eval_rho_adaptive_trapezoidal_rule<real,order>(n, x, coeffs.get(), conf, velocity_support_electron_lower_bound[i],
					velocity_support_electron_upper_bound[i], true);
    		std::cout << " electron" << std::endl;
    		rho.get()[i] = rho_ion[i] - rho_electron[i];
    	}

    	std::cout << "rho finished" << std::endl;
    	// Let's try out setting the minimum and maximum velocity bound to their respective
    	// min/max after each time-step to stop "folding" of the distribution function
    	// from producing wholes in the integration domain which cannot be detected by the current
    	// algorithm.
        typename std::vector<real>::iterator result_el;
        result_el = std::min_element(velocity_support_electron_lower_bound.begin(), velocity_support_electron_lower_bound.end());
    	real electron_lower_bound = *result_el;
    	std::fill(velocity_support_electron_lower_bound.begin(), velocity_support_electron_lower_bound.end(), electron_lower_bound);

    	typename std::vector<real>::iterator result_eu = std::max_element(velocity_support_electron_upper_bound.begin(), velocity_support_electron_upper_bound.end());
    	real electron_upper_bound = *result_eu;
    	std::fill(velocity_support_electron_upper_bound.begin(), velocity_support_electron_upper_bound.end(), electron_upper_bound);

    	typename std::vector<real>::iterator result_il = std::min_element(velocity_support_ion_lower_bound.begin(), velocity_support_ion_lower_bound.end());
    	real ion_lower_bound = *result_il;
    	std::fill(velocity_support_ion_lower_bound.begin(), velocity_support_ion_lower_bound.end(), ion_lower_bound);

    	typename std::vector<real>::iterator result_iu = std::max_element(velocity_support_ion_upper_bound.begin(), velocity_support_ion_upper_bound.end());
    	real ion_upper_bound = *result_iu;
    	std::fill(velocity_support_ion_upper_bound.begin(), velocity_support_ion_upper_bound.end(), ion_upper_bound);

    	// Solve Poisson's equation:
        poiss.solve( rho.get() );

        std::cout << "Poisson finished" << std::endl;

        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;
        double t = n*conf.dt;
        std::cout << std::setw(15) << t << " Comp-time: " << timer_elapsed << " Total time: " << total_time << std::endl;

        if(write_stats){
			size_t plot_x = 256;
			size_t plot_v = plot_x;
			real dx_plot = conf.Lx/plot_x;
			real Emax = 0;
			real E_l2 = 0;

			for ( size_t i = 0; i < plot_x; ++i )
			{
				real x = conf.x_min + i*dx_plot;
				real E = -dim1::eval<real,order,1>(x,coeffs.get()+n*stride_t,conf);
				Emax = max( Emax, abs(E) );
				E_l2 += E*E;
			}
			E_l2 *=  dx_plot;
			stats_file << std::setw(20) << t << std::setw(20) << std::setprecision(8) << std::scientific << Emax
						<< std::setw(20) << std::setprecision(8) << std::scientific << E_l2 << std::endl;
        }
		// Write coeffs to disc:
        for(size_t i = 0; i < stride_t; i++){
        	coeffs_str << coeffs.get()[n*stride_t + i] << std::endl;
        }
    }


}

template <typename real, size_t order>
void test()
{
    using std::abs;
    using std::max;

    // Set config:
    config_t<real> conf;
    conf.Nx = 32;
    conf.x_min = 0;
    //conf.x_max = 4*M_PI;
    conf.x_max = 40*M_PI;
    conf.dt = 1./8.0;
    conf.Nt = 2000.0 / conf.dt;
    conf.Lx = conf.x_max - conf.x_min;
    conf.Lx_inv = 1/conf.Lx;
    conf.dx = conf.Lx/conf.Nx;
    conf.dx_inv = 1/conf.dx;

    conf.tol_cut_off_velocity_supp = 1e-7;
    conf.tol_integral = 1e-5;
    conf.max_depth_integration = 3;
    conf.Nu = 32;

    const size_t stride_t = conf.Nx + order - 1;

    std::unique_ptr<real[]> coeffs { new real[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*conf.Nx)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};
    std::vector<real> rho_electron(conf.Nx);
    std::vector<real> rho_ion(conf.Nx);

    // Maybe it would be nice to compute these initial bounds automatically instead of setting
    // them manually. This would give some additional robustness to the approach.
    std::vector<real> velocity_support_electron_lower_bound(conf.Nx, -10);
    std::vector<real> velocity_support_electron_upper_bound(conf.Nx, 8);
    std::vector<real> velocity_support_ion_lower_bound(conf.Nx, -0.35);
    std::vector<real> velocity_support_ion_upper_bound(conf.Nx, 0.35);

    poisson<real> poiss( conf );

    std::ofstream stats_file( "stats.txt" );
    std::ofstream coeffs_str( "coeffs_Nt_" + std::to_string(conf.Nt) + "_Nx_ " + std::to_string(conf.Nx) + "_stride_t_" + std::to_string(stride_t) + ".txt" );
    double total_time = 0;
    for ( size_t n = 0; n <= conf.Nt; ++n )
    {
    	dergeraet::stopwatch<double> timer;
    	// Compute rho:
		#pragma omp parallel for
    	for(size_t i = 0; i<conf.Nx; i++)
    	{
    		real x = conf.x_min + i*conf.dx;
    		rho_ion[i] = eval_rho_adaptive_trapezoidal_rule<real,order>(n, x, coeffs.get(), conf, velocity_support_ion_lower_bound[i],
					velocity_support_ion_upper_bound[i], false);
    		rho_electron[i] = eval_rho_adaptive_trapezoidal_rule<real,order>(n, x, coeffs.get(), conf, velocity_support_electron_lower_bound[i],
					velocity_support_electron_upper_bound[i], true);
    		rho.get()[i] = rho_ion[i] - rho_electron[i];
    	}

    	// Let's try out setting the minimum and maximum velocity bound to their respective
    	// min/max after each time-step to stop "folding" of the distribution function
    	// from producing wholes in the integration domain which cannot be detected by the current
    	// algorithm.
        typename std::vector<real>::iterator result_el;
        result_el = std::min_element(velocity_support_electron_lower_bound.begin(), velocity_support_electron_lower_bound.end());
    	real electron_lower_bound = *result_el;
    	std::fill(velocity_support_electron_lower_bound.begin(), velocity_support_electron_lower_bound.end(), electron_lower_bound);

    	typename std::vector<real>::iterator result_eu = std::max_element(velocity_support_electron_upper_bound.begin(), velocity_support_electron_upper_bound.end());
    	real electron_upper_bound = *result_eu;
    	std::fill(velocity_support_electron_upper_bound.begin(), velocity_support_electron_upper_bound.end(), electron_upper_bound);

    	typename std::vector<real>::iterator result_il = std::min_element(velocity_support_ion_lower_bound.begin(), velocity_support_ion_lower_bound.end());
    	real ion_lower_bound = *result_il;
    	std::fill(velocity_support_ion_lower_bound.begin(), velocity_support_ion_lower_bound.end(), ion_lower_bound);

    	typename std::vector<real>::iterator result_iu = std::max_element(velocity_support_ion_upper_bound.begin(), velocity_support_ion_upper_bound.end());
    	real ion_upper_bound = *result_iu;
    	std::fill(velocity_support_ion_upper_bound.begin(), velocity_support_ion_upper_bound.end(), ion_upper_bound);

    	// Solve Poisson's equation:
        poiss.solve( rho.get() );
        dim1::interpolate<real,order>( coeffs.get() + n*stride_t, rho.get(), conf );

        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;const size_t stride_t = conf.Nx + order - 1;

		real t = n*conf.dt;
        // Plotting:
        if(n % 1 == 0)
        {
			size_t plot_x = 256;
			size_t plot_v = plot_x;
			real dx_plot = conf.Lx/plot_x;
			real Emax = 0;
			real E_l2 = 0;

			if(n % (10*8) == 0)
			{
				/*
				std::ofstream file_v_min_max( "v_min_max_" + std::to_string(t) + ".txt" );
				for ( size_t i = 0; i < conf.Nx; ++i )
				{
					real x = conf.x_min + i*conf.dx;
					file_v_min_max << x << " " << velocity_support_electron_lower_bound[i]
										<< " " << velocity_support_electron_upper_bound[i]
										<< " " << velocity_support_ion_lower_bound[i]
										<< " " << velocity_support_ion_upper_bound[i] << std::endl;
				}
				*/
				std::ofstream file_E( "E_" + std::to_string(t) + ".txt" );
				for ( size_t i = 0; i < plot_x; ++i )
				{
					real x = conf.x_min + i*dx_plot;
					real E = -dim1::eval<real,order,1>(x,coeffs.get()+n*stride_t,conf);
					Emax = max( Emax, abs(E) );
					E_l2 += E*E;

					file_E << x << " " << E << std::endl;
				}

				E_l2 *=  dx_plot;
				stats_file << std::setw(20) << t << std::setw(20) << std::setprecision(8) << std::scientific << Emax
							<< std::setw(20) << std::setprecision(8) << std::scientific << E_l2 << std::endl;

				std::ofstream file_rho_ion( "rho_ion_" + std::to_string(t) + ".txt" );
				std::ofstream file_rho_electron( "rho_electron_" + std::to_string(t) + ".txt" );
				for ( size_t i = 0; i < conf.Nx; ++i )
				{
					real x = conf.x_min + i*conf.dx;
					file_rho_ion << std::setw(20) << x << std::setw(20) << std::setprecision(8) << std::scientific << rho_ion[i] << std::endl;
					file_rho_electron << std::setw(20) <<  x << std::setw(20) << std::setprecision(8) << std::scientific << rho_electron[i] << std::endl;
				}



				real v_min_electron = -10;
				real v_max_electron = 10;
				real dv_electron = (v_max_electron - v_min_electron) / plot_v;
				real v_min_ion = -0.4;
				real v_max_ion = 0.4;
				real dv_ion = (v_max_ion - v_min_ion) / plot_v;
				std::ofstream file_f_electron( "f_electron_" + std::to_string(t) + ".txt" );
				for ( size_t i = 0; i <= plot_x; ++i )
				{
					for ( size_t j = 0; j <= plot_v; ++j )
					{
						real x = conf.x_min + i*dx_plot;
						real v = v_min_electron + j*dv_electron;

						real f = eval_f_ion_acoustic<real,order>(n,x,v,coeffs.get(),conf,true);
						file_f_electron << x << " " << v << " " << f << std::endl;
					}
					file_f_electron << std::endl;
				}

				std::ofstream file_f_ion( "f_ion_" + std::to_string(t) + ".txt" );
				for ( size_t i = 0; i <= plot_x; ++i )
				{
					for ( size_t j = 0; j <= plot_v; ++j )
					{
						real x = conf.x_min + i*dx_plot;
						real v = v_min_ion + j*dv_ion;

						real f = eval_f_ion_acoustic<real,order>(n,x,v,coeffs.get(),conf,false);
						file_f_ion << x << " " << v << " " << f << std::endl;
					}
					file_f_ion << std::endl;
				}

			} else {
				for ( size_t i = 0; i < plot_x; ++i )
				{
					real x = conf.x_min + i*dx_plot;
					real E = -dim1::eval<real,order,1>(x,coeffs.get()+n*stride_t,conf);
					Emax = max( Emax, abs(E) );
					E_l2 += E*E;
				}
				E_l2 *=  dx_plot;
				stats_file << std::setw(20) << t << std::setw(20) << std::setprecision(8) << std::scientific << Emax
							<< std::setw(20) << std::setprecision(8) << std::scientific << E_l2 << std::endl;
			}
        }
        std::cout << std::setw(15) << t << " Comp-time: " << timer_elapsed << " Total time: " << total_time << std::endl;

		// Write coeffs to disc:
        for(size_t i = 0; i < stride_t; i++){
        	coeffs_str << coeffs.get()[n*stride_t + i] << std::endl;
        }
    }
}


}

}


int main()
{
	//dergeraet::dim1::test<double,4>();

	constexpr size_t order = 4;
	dergeraet::dim1::config_t<double> conf;
    conf.Nx = 32;
    conf.x_min = 0;
    //conf.x_max = 4*M_PI;
    conf.x_max = 40*M_PI;
    conf.dt = 1./8.0;
    conf.Nt = 2000.0 / conf.dt;
    conf.Lx = conf.x_max - conf.x_min;
    conf.Lx_inv = 1/conf.Lx;
    conf.dx = conf.Lx/conf.Nx;
    conf.dx_inv = 1/conf.dx;

    conf.tol_cut_off_velocity_supp = 1e-7;
    conf.tol_integral = 1e-5;
    conf.max_depth_integration = 1;//3;
    conf.Nu = 32;

    const size_t stride_t = conf.Nx + order - 1;
    size_t old_n = 1000*8;

	std::vector<double> coeffs_old((old_n+1)*stride_t);
	dergeraet::dim1::read_coeffs<double,order>(conf, coeffs_old, old_n, std::string("../coeffs_Nt_16000_Nx_ 32_stride_t_35.txt"));
	dergeraet::dim1::ion_acoustic_restart<double,order>(conf, coeffs_old, old_n);
}

