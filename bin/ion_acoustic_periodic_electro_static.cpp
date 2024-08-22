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

#include <algorithm>
#include <cmath>
#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <vector>


#include <nufi/config.hpp>
#include <nufi/random.hpp>
#include <nufi/fields.hpp>
#include <nufi/poisson.hpp>
#include <nufi/rho.hpp>
#include <nufi/stopwatch.hpp>
#include <nufi/cuda_scheduler.hpp>

namespace nufi
{

namespace dim1
{

template <typename real, size_t order>
void read_coeffs(const config_t<real>& conf, std::vector<real>& coeffs_old, size_t old_n, const std::string& i_str)
{
    const size_t stride_t = conf.Nx + order - 1;
	std::ifstream coeff_str(i_str);
	for(size_t i = 0; i < old_n*stride_t; i++){
		coeff_str >> coeffs_old[i];
		//std::cout << i << " of " << old_n*stride_t << " coeff[i] = " << coeffs_old[i] << std::endl;
	}
}

template <typename real, size_t order>
void ion_acoustic_restart(const config_t<real>& conf_electron, const config_t<real>& conf_ion, const std::vector<real>& coeffs_old, size_t old_n)
{
    using std::abs;
    using std::max;

	// This method allows to read in an old set of coefficients and start from that rather than
	// rerunning the entire simulation with the same parameters.
    const size_t stride_t = conf_electron.Nx + order - 1;

    std::unique_ptr<real[]> coeffs { new real[ (conf_electron.Nt+1)*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*conf_electron.Nx)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};
    std::vector<real> rho_electron(conf_electron.Nx);
    std::vector<real> rho_ion(conf_electron.Nx);

    // Copy old coefficients into coeff vector:
    // Also start writing the coefficients onto disc to continue writing
    // at the right position later on.
    std::ofstream coeffs_str( "restart_coeffs_Nt_" + std::to_string(conf_electron.Nt) + "_Nx_ " + std::to_string(conf_electron.Nx) + "_stride_t_" + std::to_string(stride_t) + ".txt" );
    for(size_t i = 0; i < old_n*stride_t; i++){
    	coeffs.get()[i] = coeffs_old[i];
    	coeffs_str << coeffs.get()[i] << std::endl;
    }

    // Maybe it would be nice to compute these initial bounds automatically instead of setting
    // them manually. This would give some additional robustness to the approach.
    /*
    std::vector<real> velocity_support_electron_lower_bound(conf.Nx, -10);
    std::vector<real> velocity_support_electron_upper_bound(conf.Nx, 8);
    std::vector<real> velocity_support_ion_lower_bound(conf.Nx, -0.35);
    std::vector<real> velocity_support_ion_upper_bound(conf.Nx, 0.35);

    poisson<real> poiss( conf );
    */

    // Plot:
    size_t n_plot = 1000*8;
    real t = n_plot*conf_electron.dt;
    size_t plot_x_electron = 2048;
    size_t plot_x_ion = 1024;
	real dx_plot_electron = conf_electron.Lx/plot_x_electron;
	real dx_plot_ion = conf_electron.Lx/plot_x_ion;
    //real x_min_plot = 7.5;
    //real x_max_plot = 7.54;
    //real dx_plot = (x_max_plot - x_min_plot)/plot_x;
	size_t plot_v_electron = 2048;
    size_t plot_v_ion = 1024;
	real v_min_electron = -8;
	real v_max_electron = 5;
	real dv_electron = (v_max_electron - v_min_electron) / plot_v_electron;
	real v_min_ion = -0.2;
	real v_max_ion = 0.2;
	real dv_ion = (v_max_ion - v_min_ion) / plot_v_ion;
	//std::ofstream file_f_electron( "f_electron_zoom_" + std::to_string(t) + ".txt" );
	std::ofstream file_f_electron( "f_electron_" + std::to_string(t) + ".txt" );
	for ( size_t i = 0; i <= plot_x_electron; ++i )
	{
		for ( size_t j = 0; j <= plot_v_electron; ++j )
		{
			real x = conf_electron.x_min + i*dx_plot_electron;
			//real x = x_min_plot + i*dx_plot;
			real v = v_min_electron + j*dv_electron;

			real f = eval_f_ion_acoustic<real,order>(n_plot,x,v,coeffs.get(),conf_electron,true);
			file_f_electron << x << " " << v << " " << f << std::endl;
		}
		file_f_electron << std::endl;
	}

	std::ofstream file_f_ion( "f_ion_" + std::to_string(t) + ".txt" );
	for ( size_t i = 0; i <= plot_x_ion; ++i )
	{
		for ( size_t j = 0; j <= plot_v_ion; ++j )
		{
			real x = conf_electron.x_min + i*dx_plot_ion;
			real v = v_min_ion + j*dv_ion;

			real f = eval_f_ion_acoustic<real,order>(n_plot,x,v,coeffs.get(),conf_ion,false);
			file_f_ion << x << " " << v << " " << f << std::endl;
		}
		file_f_ion << std::endl;
	}

    /*
    bool write_stats = true;
    std::ofstream stats_file( "restart_stats.txt" );
    double total_time = 0;
    for ( size_t n = old_n-10; n <= conf.Nt; ++n )
    {
    	nufi::stopwatch<double> timer;
    	// Compute rho:
		#pragma omp parallel for
    	for(size_t i = 0; i<conf.Nx; i++)
    	{
    		// Careful: This seems to be buggy. Probably there is an issue with the way I read in and store the
    		// coefficients...
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
	*/

}

template <typename real, size_t order>
void ion_acoustic(bool plot = true, bool only_stats = true)
{
    using std::abs;
    using std::max;

    // Set config:
    // Use different Nu for electrons and ions as the dynamics of
    // electrons tend to develop fast towards filamentation than for
    // ions, i.e., one needs higher resolution for electrons than ions.
    // However, be careful to keep Lx, Nx, dt, Nt, etc. the same for
    // electrons and ions to not break the method.
    config_t<real> conf_electron;
    //conf_electron.Nx = N;
    conf_electron.Nx = 128;
    conf_electron.x_min = 0;
    //conf_electron.x_max = 40*M_PI;
    conf_electron.x_max = 4*M_PI;
    conf_electron.dt = 1./4.0;
    conf_electron.Nt = 200.0 / conf_electron.dt;
    conf_electron.Lx = conf_electron.x_max - conf_electron.x_min;
    conf_electron.Lx_inv = 1/conf_electron.Lx;
    conf_electron.dx = conf_electron.Lx/conf_electron.Nx;
    conf_electron.dx_inv = 1/conf_electron.dx;

    conf_electron.tol_cut_off_velocity_supp = 1e-7;
    conf_electron.tol_integral_1 = 1e-10;
    conf_electron.tol_integral_2 = 1e-5;
    conf_electron.max_depth_integration = 5;
    //conf_electron.max_depth_integration = 1;
    //conf_electron.Nu = N*16;
    conf_electron.Nu = 32;

    config_t<real> conf_ion;
    conf_ion.Nx = conf_electron.Nx;
    conf_ion.x_min = conf_electron.x_min;
    conf_ion.x_max = conf_electron.x_max;
    conf_ion.dt = conf_electron.dt;
    conf_ion.Nt = conf_electron.Nt;
    conf_ion.Lx = conf_electron.Lx;
    conf_ion.Lx_inv = conf_electron.Lx_inv;
    conf_ion.dx = conf_electron.dx;
    conf_ion.dx_inv = conf_electron.dx_inv;

    conf_ion.tol_cut_off_velocity_supp = 1e-7;
    conf_ion.tol_integral_1 = conf_electron.tol_integral_1;
    conf_ion.tol_integral_2 = conf_electron.tol_integral_2;
    conf_ion.max_depth_integration = 3;
    //conf_ion.max_depth_integration = 1;
    //conf_ion.Nu = N*4;
    conf_ion.Nu = 32;

    const size_t stride_t = conf_electron.Nx + order - 1;

    std::unique_ptr<real[]> coeffs { new real[ (conf_electron.Nt+1)*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*conf_electron.Nx)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};
    std::vector<real> rho_electron(conf_electron.Nx);
    std::vector<real> rho_ion(conf_electron.Nx);

    // Maybe it would be nice to compute these initial bounds automatically instead of setting
    // them manually. This would give some additional robustness to the approach.
    std::vector<real> velocity_support_electron_lower_bound(conf_electron.Nx, -10);
    std::vector<real> velocity_support_electron_upper_bound(conf_electron.Nx, 10);
    std::vector<real> velocity_support_ion_lower_bound(conf_electron.Nx, -0.35);
    std::vector<real> velocity_support_ion_upper_bound(conf_electron.Nx, 0.35);

    poisson<real> poiss( conf_electron );

    std::ofstream stats_file( "stats.txt" );
    std::ofstream coeffs_str( "coeffs_Nt_" + std::to_string(conf_electron.Nt) + "_Nx_ "
    						+ std::to_string(conf_electron.Nx) + "_stride_t_" + std::to_string(stride_t) + ".txt" );
    double total_time = 0;
    for ( size_t n = 0; n <= conf_electron.Nt; ++n )
    {
    	nufi::stopwatch<double> timer;
    	// Compute rho:
		#pragma omp parallel for
    	for(size_t i = 0; i<conf_electron.Nx; i++)
    	{
    		real x = conf_electron.x_min + i*conf_electron.dx;
    		rho_ion[i] = eval_rho_adaptive_trapezoidal_rule<real,order>(n, x, coeffs.get(), conf_ion, velocity_support_ion_lower_bound[i],
					velocity_support_ion_upper_bound[i], false);
    		rho_electron[i] = eval_rho_adaptive_trapezoidal_rule<real,order>(n, x, coeffs.get(), conf_electron, velocity_support_electron_lower_bound[i],
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
        dim1::interpolate<real,order>( coeffs.get() + n*stride_t, rho.get(), conf_electron );

        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;

		real t = n*conf_electron.dt;
        // Plotting:
        if(n % 1 == 0 && plot)
        {
			size_t plot_x = 256;
			size_t plot_v = plot_x;
			real dx_plot = conf_electron.Lx/plot_x;
			real Emax = 0;
			real E_l2 = 0;

			if(n % (5*4) == 0 && (!only_stats))
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
					real x = conf_electron.x_min + i*dx_plot;
					real E = -dim1::eval<real,order,1>(x,coeffs.get()+n*stride_t,conf_electron);
					Emax = max( Emax, abs(E) );
					E_l2 += E*E;

					file_E << x << " " << E << std::endl;
				}

				E_l2 *=  dx_plot;
				stats_file << std::setw(20) << t << std::setw(20) << std::setprecision(8) << std::scientific << Emax
							<< std::setw(20) << std::setprecision(8) << std::scientific << E_l2 << std::endl;

				std::ofstream file_rho_ion( "rho_ion_" + std::to_string(t) + ".txt" );
				std::ofstream file_rho_electron( "rho_electron_" + std::to_string(t) + ".txt" );
				for ( size_t i = 0; i < conf_electron.Nx; ++i )
				{
					real x = conf_electron.x_min + i*conf_electron.dx;
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
						real x = conf_electron.x_min + i*dx_plot;
						real v = v_min_electron + j*dv_electron;

						real f = eval_f_ion_acoustic<real,order>(n,x,v,coeffs.get(),conf_electron,true);
						file_f_electron << x << " " << v << " " << f << std::endl;
					}
					file_f_electron << std::endl;
				}

				std::ofstream file_f_ion( "f_ion_" + std::to_string(t) + ".txt" );
				for ( size_t i = 0; i <= plot_x; ++i )
				{
					for ( size_t j = 0; j <= plot_v; ++j )
					{
						real x = conf_electron.x_min + i*dx_plot;
						real v = v_min_ion + j*dv_ion;

						real f = eval_f_ion_acoustic<real,order>(n,x,v,coeffs.get(),conf_ion,false);
						file_f_ion << x << " " << v << " " << f << std::endl;
					}
					file_f_ion << std::endl;
				}

			} else {
				for ( size_t i = 0; i < plot_x; ++i )
				{
					real x = conf_electron.x_min + i*dx_plot;
					real E = -dim1::eval<real,order,1>(x,coeffs.get()+n*stride_t,conf_electron);
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
	nufi::dim1::ion_acoustic<double,4>(true,true);

	/*
	constexpr size_t order = 4;
	nufi::dim1::config_t<double> conf_electron;
    conf_electron.Nx = 128;
    conf_electron.x_min = 0;
    conf_electron.x_max = 40*M_PI;
    //conf_electron.x_max = 4*M_PI;
    conf_electron.dt = 1./8.0;
    conf_electron.Nt = 2000.0 / conf_electron.dt;
    conf_electron.Lx = conf_electron.x_max - conf_electron.x_min;
    conf_electron.Lx_inv = 1/conf_electron.Lx;
    conf_electron.dx = conf_electron.Lx/conf_electron.Nx;
    conf_electron.dx_inv = 1/conf_electron.dx;

    conf_electron.tol_cut_off_velocity_supp = 1e-7;
    conf_electron.tol_integral = 1e-5;
    conf_electron.max_depth_integration = 3;
    conf_electron.Nu = 128;

    nufi::dim1::config_t<double> conf_ion;
    conf_ion.Nx = conf_electron.Nx;
    conf_ion.x_min = conf_electron.x_min;
    conf_ion.x_max = conf_electron.x_max;
    conf_ion.dt = conf_electron.dt;
    conf_ion.Nt = conf_electron.Nt;
    conf_ion.Lx = conf_electron.Lx;
    conf_ion.Lx_inv = conf_electron.Lx_inv;
    conf_ion.dx = conf_electron.dx;
    conf_ion.dx_inv = conf_electron.dx_inv;

    conf_ion.tol_cut_off_velocity_supp = 1e-7;
    conf_ion.tol_integral = 1e-5;
    conf_ion.max_depth_integration = 3;
    conf_ion.Nu = 128;

    const size_t stride_t = conf_electron.Nx + order - 1;
    size_t old_n = 2000*8;

	std::vector<double> coeffs_old((old_n+1)*stride_t);
	nufi::dim1::read_coeffs<double,order>(conf_electron, coeffs_old, old_n, std::string("../coeffs_Nt_16000_Nx_ 128_stride_t_131.txt"));
	nufi::dim1::ion_acoustic_restart<double,order>(conf_electron, conf_ion, coeffs_old, old_n);
	*/
}

