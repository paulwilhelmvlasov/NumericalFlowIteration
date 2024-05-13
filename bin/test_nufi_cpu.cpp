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
#include <iostream>
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

namespace dergeraet
{

namespace dim1
{

template <typename real>
real f0(real x, real u) noexcept
{
	real alpha = 1e-2;
	real k = 0.5;
    //return 1.0 / std::sqrt(2.0 * M_PI) * u*u * exp(-0.5 * u*u) * (1 + alpha * cos(k*x));
	return 1.0 / std::sqrt(2.0 * M_PI) * exp(-0.5 * u*u) * (1 + alpha * cos(k*x));
}

template <typename real>
real lin_interpol(real x , real y, real x1, real x2, real y1, real y2, real f_11,
		real f_12, real f_21, real f_22) noexcept
{
	real f_x_y1 = ((x2-x)*f_11 + (x-x1)*f_21)/(x2-x1);
	real f_x_y2 = ((x2-x)*f_12 + (x-x1)*f_22)/(x2-x1);

	return ((y2-y)*f_x_y1 * (y-y1)*f_x_y2) / (y2-y1);
}

arma::mat f0_r;
config_t<double> conf(64, 128, 500, 0.1, 0, 4*M_PI, -10, 10, &f0);

double f_t(double x, double u) noexcept
{
	size_t nx_r = f0_r.n_rows;
	size_t nu_r = f0_r.n_cols;

	double dx_r = conf.Lx/ nx_r;
	double du_r = (conf.u_max - conf.u_min)/nu_r;

	// Compute periodic reference position of x. Assume x_min = 0 to this end.
	x -= conf.Lx * std::floor(x*conf.Lx_inv);

	if(u > conf.u_max || u < conf.u_min){
		return 0;
	}

	size_t x_ref_pos = std::floor(x*dx_r);
	size_t u_ref_pos = std::floor((u-conf.u_min)*du_r);

	double x1 = x_ref_pos*dx_r;
	double x2 = x1+dx_r;
	double u1 = conf.u_min + u_ref_pos*du_r;
	double u2 = u1 + du_r;

	double f_11 = f0_r(x_ref_pos, u_ref_pos);
	double f_21 = f0_r(x_ref_pos+1, u_ref_pos);
	double f_12 = f0_r(x_ref_pos, u_ref_pos+1);
	double f_22 = f0_r(x_ref_pos+1, u_ref_pos+1);

	return lin_interpol<double>(x, u, x1, x2, u1, u2, f_11, f_21, f_12, f_22);
}

template <size_t order>
void run_restarted_simulation()
{
	using std::exp;
	using std::sin;
	using std::cos;
    using std::abs;
    using std::max;

    size_t Nx = 64;  // Number of grid points in physical space.
    size_t Nu = 128;  // Number of quadrature points in velocity space.
    double   dt = 0.1;  // Time-step size.
    size_t Nt = 50/dt;  // Number of time-steps.

    // Dimensions of physical domain.
    double x_min = 0;
    double x_max = 4*M_PI;

    // Integration limits for velocity space.
    double u_min = -10;
    double u_max = 10;

    // We use conf.Nt as restart timer for now.
    size_t nt_restart = 100;
    size_t nx_r = 256;
    size_t nu_r = nx_r;
    double dx_r = conf.Lx / nx_r;
    double du_r = (conf.u_max - conf.u_min)/ nu_r;
    f0_r.resize(nx_r+1, nu_r+1);
    conf = config_t<double>(Nx, Nu, nt_restart, dt, x_min, x_max, u_min, u_max, &f0);
    const size_t stride_t = conf.Nx + order - 1;

    std::unique_ptr<double[]> coeffs { new double[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<double,decltype(std::free)*> rho { reinterpret_cast<double*>(std::aligned_alloc(64,sizeof(double)*conf.Nx)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};

    poisson<double> poiss( conf );

    std::ofstream Emax_file( "Emax.txt" );
    double total_time = 0;
    size_t nt_r_curr = 0;
    for ( size_t n = 0; n <= Nt; ++n )
    {
    	dergeraet::stopwatch<double> timer;

    	if(nt_r_curr == nt_restart)
    	{
    		for(size_t i = 0; i <= nx_r; i++ ){
    			for(size_t j = 0; j <= nu_r; j++){
    				double x = i*dx_r;
    				double u = conf.u_min + j*du_r;
    				f0_r(i,j) = periodic::eval_f<double,order>(nt_r_curr,x,u,coeffs.get(),conf);
    			}

    			conf = config_t<double>(Nx, Nu, nt_restart, dt, x_min, x_max,
    					u_min, u_max, &f_t);
    		}

    		nt_r_curr = 0;
    	}

    	// Compute rho:
		#pragma omp parallel for
    	for(size_t i = 0; i<conf.Nx; i++)
    	{
    		rho.get()[i] = periodic::eval_rho<double,order>(nt_r_curr, i, coeffs.get(), conf);
    	}

        poiss.solve( rho.get() );
        periodic::interpolate<double,order>( coeffs.get() + nt_r_curr*stride_t, rho.get(), conf );

        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;

        double Emax = 0;
	    double E_l2 = 0;
        for ( size_t i = 0; i < conf.Nx; ++i )
        {
            double x = conf.x_min + i*conf.dx;
            double E_abs = abs( periodic::eval<double,order,1>(x,coeffs.get()+nt_r_curr*stride_t,conf));
            Emax = max( Emax, E_abs );
	        E_l2 += E_abs*E_abs;
        }
	    E_l2 *=  conf.dx;

	    double t = n*conf.dt;
        Emax_file << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << Emax << std::endl;
        std::cout << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << Emax << " Comp-time: " << timer_elapsed << std::endl;

        nt_r_curr++;
    }
    std::cout << "Total time: " << total_time << std::endl;

}

template <typename real, size_t order>
void run_simulation()
{
	using std::exp;
	using std::sin;
	using std::cos;
    using std::abs;
    using std::max;

    size_t Nx = 128;  // Number of grid points in physical space.
    size_t Nu = 256;  // Number of quadrature points in velocity space.
    real   dt = 0.1;  // Time-step size.
    size_t Nt = 100/dt;  // Number of time-steps.

    // Dimensions of physical domain.
    real x_min = 0;
    real x_max = 4*M_PI;

    // Integration limits for velocity space.
    real u_min = -6;
    real u_max = 6;

    config_t<real> conf(Nx, Nu, Nt, dt, x_min, x_max, u_min, u_max, &f0);
    const size_t stride_t = conf.Nx + order - 1;

    std::unique_ptr<real[]> coeffs { new real[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,sizeof(real)*conf.Nx)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};

    poisson<real> poiss( conf );

    std::ofstream Emax_file( "Emax.txt" );
    std::ofstream coeffs_str( "coeffs_Nt_" + std::to_string(conf.Nt) + "_Nx_"
    						+ std::to_string(conf.Nx) + "_stride_t_" + std::to_string(stride_t) + ".txt" );
    double total_time = 0;
    for ( size_t n = 0; n <= conf.Nt; ++n )
    {
    	dergeraet::stopwatch<double> timer;

    	// Compute rho:
		#pragma omp parallel for
    	for(size_t i = 0; i<conf.Nx; i++)
    	{
    		rho.get()[i] = periodic::eval_rho<real,order>(n, i, coeffs.get(), conf);
    	}

        poiss.solve( rho.get() );
        periodic::interpolate<real,order>( coeffs.get() + n*stride_t, rho.get(), conf );

        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;

        real Emax = 0;
	    real E_l2 = 0;
        for ( size_t i = 0; i < conf.Nx; ++i )
        {
            real x = conf.x_min + i*conf.dx;
            real E_abs = abs( periodic::eval<real,order,1>(x,coeffs.get()+n*stride_t,conf));
            Emax = max( Emax, E_abs );
	        E_l2 += E_abs*E_abs;
        }
	    E_l2 *=  conf.dx;

	    double t = n*conf.dt;
        Emax_file << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << Emax << std::endl;
        std::cout << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << Emax << " Comp-time: " << timer_elapsed << std::endl;

        for(size_t i = 0; i < stride_t; i++){
        	coeffs_str << coeffs.get()[n*stride_t + i] << std::endl;
        }

    }
    std::cout << "Total time: " << total_time << std::endl;
}



}
}


int main()
{
	dergeraet::dim1::run_simulation<double,4>();
	//dergeraet::dim1::run_restarted_simulation<4>();
}

