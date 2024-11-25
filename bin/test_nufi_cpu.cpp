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
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include <armadillo>

#include <nufi/config.hpp>
#include <nufi/random.hpp>
#include <nufi/fields.hpp>
#include <nufi/poisson.hpp>
#include <nufi/rho.hpp>
#include <nufi/stopwatch.hpp>

namespace nufi
{

namespace dim1
{

template <typename real>
real f0(real x, real u) noexcept
{
	real alpha = 1e-2;
	//real alpha = 0.5; // Strong Landau Damping
	real k = 0.5;
    return 1.0 / std::sqrt(2.0 * M_PI) * u*u * exp(-0.5 * u*u) * (1 + alpha * cos(k*x));
	//return 1.0 / std::sqrt(2.0 * M_PI) * exp(-0.5 * u*u) * (1 + alpha * cos(k*x));
}

template <typename real>
real lin_interpol(real x , real y, real x1, real x2, real y1, real y2, real f_11,
		real f_12, real f_21, real f_22) noexcept
{
	real f_x_y1 = ((x2-x)*f_11 + (x-x1)*f_21)/(x2-x1);
	real f_x_y2 = ((x2-x)*f_12 + (x-x1)*f_22)/(x2-x1);

	return ((y2-y)*f_x_y1 + (y-y1)*f_x_y2) / (y2-y1);
}

bool debug = false;
bool write = false;

arma::mat f0_r;
config_t<double> conf(64, 128, 500, 0.1, 0, 4*M_PI, -10, 10, &f0);

double f_t(double x, double u) noexcept
{
	if(u > conf.u_max || u < conf.u_min){
		return 0;
	}

	size_t nx_r = f0_r.n_rows - 1;
	size_t nu_r = f0_r.n_cols - 1;

	double dx_r = conf.Lx/ nx_r;
	double du_r = (conf.u_max - conf.u_min)/nu_r;

	// Compute periodic reference position of x. Assume x_min = 0 to this end.
     if(x < 0 || x > conf.x_max){
	    x -= conf.Lx * std::floor(x*conf.Lx_inv);
    } 

/*     if(debug && write){
        std::cout << "ft: " << x << " " << u << " ";
    } */

	size_t x_ref_pos = std::floor(x/dx_r);
    x_ref_pos = x_ref_pos % nx_r;
	size_t u_ref_pos = std::floor((u-conf.u_min)/du_r);

/*     if(debug && write){
        std::cout << x_ref_pos << " " << u_ref_pos << " ";
    }
 */
	double x1 = x_ref_pos*dx_r;
	double x2 = x1+dx_r;
	double u1 = conf.u_min + u_ref_pos*du_r;
	double u2 = u1 + du_r;

/*     if(debug && write){
        std::cout << x1 << " " << x2 << " " << u1 << " " << u2 << " ";
    }
 */
	double f_11 = f0_r(x_ref_pos, u_ref_pos);
	double f_21 = f0_r(x_ref_pos+1, u_ref_pos);
	double f_12 = f0_r(x_ref_pos, u_ref_pos+1);
	double f_22 = f0_r(x_ref_pos+1, u_ref_pos+1);

    double value = lin_interpol<double>(x, u, x1, x2, u1, u2, f_11, f_12, f_21, f_22);

/*     if(debug && write){
        std::cout << value << std::endl;
    } */

    return value;
}

template <size_t order>
void run_restarted_simulation()
{
	using std::exp;
	using std::sin;
	using std::cos;
    using std::abs;
    using std::max;

    //omp_set_num_threads(1);

    size_t Nx = 64;  // Number of grid points in physical space.
    size_t Nu = 2*Nx;  // Number of quadrature points in velocity space.
    double   dt = 0.0625;  // Time-step size.
    //double   dt = 0.1;  // Time-step size.
    size_t Nt = 500/dt;  // Number of time-steps.

    // Dimensions of physical domain.
    double x_min = 0;
    double x_max = 4*M_PI;
    conf.x_min = x_min;
    conf.x_max = x_max; // Actually I should also set Lx etc.

    // Integration limits for velocity space.
    double u_min = -8;
    double u_max = 8;
    conf.u_min = u_min;
    conf.u_max = u_max;

    // We use conf.Nt as restart timer for now.
    size_t nx_r = 2048;
	size_t nu_r = nx_r;
    size_t nt_restart = 1600;
    double dx_r = conf.Lx / nx_r;
    double du_r = (conf.u_max - conf.u_min)/ nu_r;
    f0_r.resize(nx_r+1, nu_r+1);
    conf = config_t<double>(Nx, Nu, Nt, dt, x_min, x_max, u_min, u_max, &f0);
    config_t<double> conf_full(Nx, Nu, Nt, dt, x_min, x_max, u_min, u_max, &f0);
    const size_t stride_t = conf.Nx + order - 1;

    std::unique_ptr<double[]> coeffs { new double[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<double[]> coeffs_restart { new double[ (nt_restart+1)*stride_t ] {} };
    std::unique_ptr<double,decltype(std::free)*> rho { reinterpret_cast<double*>(std::aligned_alloc(64,sizeof(double)*conf.Nx)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};

    poisson<double> poiss( conf );

    std::cout << f0_r.n_rows << " " << f0_r.n_cols << std::endl;
    std::cout << nx_r << " " << nu_r << std::endl;

    
/*     std::ofstream Emax_file( "Emax_correct.txt" );
    std::ofstream coeff_str("coeff_correct.txt"); */
    std::ofstream stat_file( "stats.txt" );
    std::ofstream coeff_str("coeff_restart.txt");
    std::ofstream coeff_r_str("coeff_r_restart.txt");
    double total_time = 0;
    size_t restart_counter = 0;
    size_t nt_r_curr = 0;
    for ( size_t n = 0; n <= Nt; ++n )
    {
        debug = false;
    	nufi::stopwatch<double> timer;

        std::cout << " start of time step "<< n << " " << nt_r_curr  << std::endl; 
    	// Compute rho:
		#pragma omp parallel for
    	for(size_t i = 0; i<conf.Nx; i++)
    	{
    		rho.get()[i] = periodic::eval_rho<double,order>(nt_r_curr, i, coeffs_restart.get(), conf);
    	}

/*         std::ofstream rho_str("rho_restart_" + std::to_string(n) + ".txt");
        //std::ofstream rho_str("rho_correct_" + std::to_string(n) + ".txt");
        for(size_t i = 0; i < conf.Nx; i++){
            rho_str << i*conf.dx << " " << rho.get()[i] << std::endl;
        }  */

        poiss.solve( rho.get() );

        //std::ofstream phi_str("phi_correct_" + std::to_string(n) + ".txt");
/*         std::ofstream phi_str("phi_restart_" + std::to_string(n) + ".txt");
        for(size_t i = 0; i < conf.Nx; i++){
            phi_str << i*conf.dx << " " << rho.get()[i] << std::endl;
        } */

        // Interpolation of Poisson solution.
        periodic::interpolate<double,order>( coeffs_restart.get() + nt_r_curr*stride_t, rho.get(), conf );
        // Copy solution also into global coeffs-vector.
        
/*         coeff_str << n << std::endl;
        coeff_r_str << n << std::endl; */
        #pragma omp parallel for
        for(size_t i = 0; i < stride_t; i++){
            coeffs.get()[n*stride_t + i ] = coeffs_restart.get()[nt_r_curr*stride_t + i];
/*             coeff_r_str << i << " " << coeffs_restart.get()[nt_r_curr*stride_t + i] << std::endl;
            coeff_str << i << " " << coeffs.get()[n*stride_t + i ] << std::endl; */
        }

        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;

        double Emax = 0;
	    double E_l2 = 0; // Electric energy
        size_t plot_n_x = 256;
        double dx_plot = conf.Lx / plot_n_x;
        for ( size_t i = 0; i < plot_n_x; ++i )
        {
            double x = conf.x_min + i*dx_plot;
            double E_abs = abs( periodic::eval<double,order,1>(x,coeffs_restart.get()+nt_r_curr*stride_t,conf));
            Emax = max( Emax, E_abs );
	        E_l2 += E_abs*E_abs;
        }
	    E_l2 =  0.5*dx_plot*E_l2;

	    double t = n*conf.dt;
        stat_file << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << Emax  << " " << E_l2 << std::endl;
        std::cout << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << Emax << " Comp-time: " << timer_elapsed;
        std::cout << " Total comp time s.f.: " << total_time << std::endl; 

        if(nt_r_curr == nt_restart)
    	{
            debug = true;
            std::cout << "Restart" << std::endl;
            nufi::stopwatch<double> timer_restart;
            arma::mat f0_r_copy(nx_r + 1, nu_r + 1);
            //std::ofstream f_str("f_restart" + std::to_string(n) + ".txt");
            #pragma omp parallel for
    		for(size_t i = 0; i <= nx_r; i++ ){
    			for(size_t j = 0; j <= nu_r; j++){
    				double x = i*dx_r;
    				double u = conf.u_min + j*du_r;

                    double f = periodic::eval_f<double,order>(nt_r_curr,x,u,coeffs_restart.get(),conf);
                    //double f = periodic::eval_f<double,order>(n,x,u,coeffs.get(),conf_full);

                    f0_r_copy(i,j) = f;
                    //f_str << x << " " << u << " " << f << std::endl;
    			}
                //f_str << std::endl;
    		}

            f0_r = f0_r_copy;

            conf = config_t<double>(Nx, Nu, Nt, dt, x_min, x_max, u_min, u_max, &f_t);

            // Copy last entry of coeff vector into restarted coeff vector.
            #pragma omp parallel for
            for(size_t i = 0; i < stride_t; i++){
                coeffs_restart.get()[i] = coeffs_restart.get()[nt_r_curr*stride_t + i];
            }

            std::cout << n << " " << nt_r_curr << " restart " << std::endl;
            nt_r_curr = 1;
            restart_counter++;
            double restart_time = timer_restart.elapsed();
            total_time += restart_time;
            std::cout << "Restart took: " << restart_time << ". Total comp time s.f.: " << total_time << std::endl;
    	} else {
/*             if((n % (5)) == 0){
                //std::ofstream f_str("f_normal_correct" + std::to_string(n*dt) + ".txt");
                std::ofstream f_str("f_normal_restart" + std::to_string(n) + ".txt");
                for(size_t i = 0; i <= nx_r; i++ ){
                    for(size_t j = 0; j <= nu_r; j++){
                        double x = i*dx_r;
                        double u = conf.u_min + j*du_r;
                        double f = periodic::eval_f<double,order>(nt_r_curr,x,u,coeffs_restart.get(),conf);
                        f_str << x << " " << u << " " << f << std::endl;
                    }
                    f_str << std::endl;
                }
            } */
            nt_r_curr++;
        }
    }
    std::cout << "Total time: " << total_time << std::endl;

}

void read_in_coeff()
{
    std::ifstream coeff_str("coeff_correct.txt");
    //std::ifstream coeff_str("coeff_restart.txt");
    
    size_t Nx = 32;  // Number of grid points in physical space.
    size_t Nu = 2*Nx;  // Number of quadrature points in velocity space.
    double   dt = 0.1;  // Time-step size.
    size_t Nt = 5/dt;  // Number of time-steps.

	size_t nx_r = 64;
	size_t nu_r = nx_r;

    // Dimensions of physical domain.
    double x_min = 0;
    double x_max = 4*M_PI;

    // Integration limits for velocity space.
    double u_min = -10;
    double u_max = 10;

    // We use conf.Nt as restart timer for now.
    double dx_r = conf.Lx / nx_r;
    double du_r = (conf.u_max - conf.u_min)/ nu_r;

    conf = config_t<double>(Nx, Nu, Nt, dt, x_min, x_max, u_min, u_max, &f0);

    const size_t order = 4;
    const size_t stride_t = conf.Nx + order - 1;
    std::unique_ptr<double[]> coeffs { new double[ (Nt+1)*stride_t ] {} };

    double a, b;
    for(size_t n = 0; n <= 30; n++){
        coeff_str >> a;
        for(size_t i = 0; i < stride_t; i++){
            coeff_str >> a >> b;
            coeffs.get()[n*stride_t + i] = b;
        }
    }
    double x = 0;
    double u = -1.875;

    std::cout << x << " " << u << std::endl;

    for(int n = 20; n > 10; n--){
        u -= 0.5*dt*periodic::eval<double,order,1>( x, coeffs.get()+n*stride_t, conf );
        x -= dt*u;
        u -= 0.5*dt*periodic::eval<double,order,1>( x, coeffs.get()+(n-1)*stride_t, conf );
        std::cout << x << " " << u << std::endl;
    }

    //std::cout << x << " " << u << std::endl;
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
    	nufi::stopwatch<double> timer;

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
	//nufi::dim1::run_simulation<double,4>();
	nufi::dim1::run_restarted_simulation<4>();

    //nufi::dim1::read_in_coeff();
}

