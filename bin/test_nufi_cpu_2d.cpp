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

namespace dim2
{

const size_t dim = 4;
int32_t d = dim;
auto dPtr = &d;
const size_t n_r = 128;
const size_t size_tensor = n_r + 1;

arma::mat restart_matrix;

const double Lx = 4*M_PI;
const double Ly = Lx;
const double dx_r = Lx/ n_r;
const double dy_r = Ly/ n_r;

const double umin = -6;
const double umax = 6;
const double vmin = umin;
const double vmax = umax;
const double wmin = umin;

const double du_r = (umax - umin)/n_r;
const double dv_r = (vmax - vmin)/n_r;

double linear_interpolation_4d(double x, double y, double u, double v)
{
    
	if( u > umax || u < umin 
        || v > vmax || v < vmin ){
		return 0;
	}

	// Compute periodic reference position of x. Assume x_min = 0 to this end.
    if(x < 0 || x > Lx){
	    x -= Lx * std::floor(x/Lx);
    } 
    // Compute periodic reference position of y. Assume y_min = 0 to this end.
    if(y < 0 || y > Ly){
	    y -= Ly * std::floor(y/Ly);
    } 

	size_t x_ref_pos = std::floor(x/dx_r);
    /* x_ref_pos = x_ref_pos % htensor::n_r; */

	size_t y_ref_pos = std::floor(y/dy_r);
    /* y_ref_pos = y_ref_pos % htensor::n_r; */

	size_t u_ref_pos = std::floor((u-umin)/du_r);
    size_t v_ref_pos = std::floor((v-vmin)/dv_r);


    // Right now only works excluding the right boundary!
    double x0 = x_ref_pos*dx_r;
    double y0 = y_ref_pos*dy_r;
    double u0 = umin + u_ref_pos*du_r;
    double v0 = vmin + v_ref_pos*dv_r;

    double w_x = (x - x0)/dx_r;
    double w_y = (y - y0)/dy_r;
    double w_u = (u - u0)/du_r;
    double w_v = (v - v0)/dv_r;

    nufi::stopwatch<double> timer_main_loop;
    double value = 0;
    for(int i_x = 0; i_x <= 1; i_x++)
    for(int i_y = 0; i_y <= 1; i_y++)
    for(int i_u = 0; i_u <= 1; i_u++)
    for(int i_v = 0; i_v <= 1; i_v++){
        double factor = ((1-w_x)*(i_x==0) + w_x*(i_x==1))
                    * ((1-w_y)*(i_y==0) + w_y*(i_y==1))
                    * ((1-w_u)*(i_u==0) + w_u*(i_u==1))
                    * ((1-w_v)*(i_v==0) + w_v*(i_v==1));

        size_t index_x = x_ref_pos + i_x;
        size_t index_y = y_ref_pos + i_y;
        size_t index_u = u_ref_pos + i_u;
        size_t index_v = v_ref_pos + i_v;
        double f = restart_matrix(index_x + size_tensor * index_y, 
                                  index_u + size_tensor * index_v); 

        value += factor * f;
    }

    return value;
}


template <typename real>
real f0(real x, real y,real u, real v) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

    constexpr real alpha = 0.01;
    constexpr real k     = 0.5;

    // Weak Landau Damping:
/*     constexpr real c  = 1.0 / (2.0 * M_PI); 
    return c * ( 1. + alpha*cos(k*x) + alpha*cos(k*y)) 
             * exp( -(u*u+v*v)/2 );
 */
    // Two Stream instability (v_x direction):
    constexpr real c  = 1.0 / (2.0 * M_PI); 
    return c * ( 1. + alpha*cos(k*x) + alpha*cos(k*y)) 
             * u*u * exp( -(u*u+v*v)/2 );
}

size_t restart_counter = 0;
bool restarted = false;

const size_t order = 4;
const size_t Nx = 32;  // Number of grid points in physical space.
const size_t Nu = 2*Nx;  // Number of quadrature points in velocity space.
const double   dt = 0.1;  // Time-step size.
const size_t Nt = 500/dt;  // Number of time-steps.
config_t<double> conf(Nx, Nx, Nu, Nu, Nt, dt, 
                    0, Lx, 0, Ly, umin, umax, vmin, vmax, 
                    &f0);
size_t stride_t = (conf.Nx + order - 1) *
                  (conf.Ny + order - 1) ;
const size_t nt_restart = 30*10;
const size_t max_rank = 500;
const double tol = 5e-2;
std::unique_ptr<double[]> coeffs_full { new double[ (Nt+1)*stride_t ] {} };
std::unique_ptr<double[]> coeffs_restart { new double[ (nt_restart+1)*stride_t ] {} };

std::unique_ptr<double,decltype(std::free)*> rho { reinterpret_cast<double*>(std::aligned_alloc(64,
                                        sizeof(double)*conf.Nx*conf.Ny)), std::free };

size_t test_n = 0;


template <typename real, size_t order>
void test()
{
    poisson<real> poiss( conf );

    std::ofstream stats_file( "stats.txt" );
    std::ofstream coeff_file( "coeffs.txt" );
    double total_time = 0;
    double total_time_with_plotting = 0;
    for ( size_t n = 0; n <= conf.Nt; ++n )
    {
    	nufi::stopwatch<double> timer;

    	// Compute rho:
		#pragma omp parallel for
    	for(size_t l = 0; l<conf.Nx*conf.Ny; l++)
    	{
    		rho.get()[l] = eval_rho<real,order>(n, l, coeffs_full.get(), conf);
    	}

        double electric_energy = poiss.solve( rho.get() );
        interpolate<real,order>( coeffs_full.get() + n*stride_t, rho.get(), conf );

        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;
        nufi::stopwatch<double> timer_plots;

        // Print coefficients to file.
        coeff_file << n << std::endl;
        for(size_t i = 0; i < stride_t; i++){
            coeff_file << i << " " << coeffs_full.get()[n*stride_t + i ] << std::endl;
        }
        stats_file << n*conf.dt << " " << electric_energy << std::endl;

        std::cout << "n = " << n << " t = " << n*conf.dt << " Comp-time: " << timer_elapsed << std::endl;

    }
    std::cout << "Total time: " << total_time << std::endl;
    std::cout << "Total time with plotting: " << total_time_with_plotting << std::endl;
}


void restart_with_svd()
{
    std::cout << "Read in coeffs." << std::endl;

    nufi::stopwatch<double> timer;
    std::ifstream coeff_str(std::string("../coeffs.txt"));
    int a = 0;
    for(size_t n = 0; n <= conf.Nt; n++){
        coeff_str >> a;
        for(size_t i = 0; i < stride_t; i++){
            coeff_str >> a >> coeffs_full.get()[n*stride_t + i];
        }
    }
    std::cout << "Coeff read in took = " << timer.elapsed() << std::endl;
    timer.reset();
    
    restart_matrix.resize(size_tensor*size_tensor,size_tensor*size_tensor);    

    #pragma omp parallel for
    for(size_t ix = 0; ix <= n_r; ix++)
    for(size_t iy = 0; iy <= n_r; iy++)
    for(size_t iu = 0; iu <= n_r; iu++)
    for(size_t iv = 0; iv <= n_r; iv++){
        size_t lx = ix + size_tensor*iy;
        size_t lu = iu + size_tensor*iv;

        double x = ix * dx_r;
        double y = iy * dy_r;
        double u = umin + iu * du_r;
        double v = vmin + iv * dv_r;

        restart_matrix(lx,lu) = eval_f<double,order>(300,x,y,u,v,coeffs_full.get(),conf);
    }
    std::cout << "Building restart matrix took = " << timer.elapsed() << std::endl;
    timer.reset();

    std::ofstream restart_matrix_str("restart_matrix.txt");
    restart_matrix_str << restart_matrix;
    std::cout << "Write restart matrix to disk = " << timer.elapsed() << std::endl;
    timer.reset();
    
    #pragma omp parallel for
    for(size_t i = 0; i < stride_t; i++){
        coeffs_restart.get()[i] = coeffs_full.get()[300 * stride_t + i];
    }
    std::cout << "Copy coeffs took = " << timer.elapsed() << std::endl;
    timer.reset();

    // SVD decomposition of restart-matrix.
    arma::mat U,V;
    arma::vec s;
    arma::svd_econ(U,s,V,restart_matrix);

    #pragma omp parallel for
    for(size_t i = 1; i < s.n_elem; i++){
        if(s(i)/s(0) < tol || i >= max_rank){
            s(i) = 0;
        }
    }

    restart_matrix = U * arma::diagmat(s) * V.t();

    std::cout << "Truncating matrix with svd took = " << timer.elapsed() << std::endl;
    timer.reset();

    config_t<double> conf_new(Nx, Nx, Nu, Nu, Nt, dt, 
        0, Lx, 0, Ly, umin, umax, vmin, vmax, 
        &linear_interpolation_4d);

    poisson<double> poiss( conf_new );

    std::ofstream stats_restart_file( "stats_restart.txt" );
    std::ofstream coeff_restart_file( "coeffs_restart.txt" );
    double total_time = 0;
    size_t nt_r_curr = 1;
    for(size_t n = 301; n <= conf.Nt; n++){
        timer.reset();

    	// Compute rho:
		#pragma omp parallel for
    	for(size_t l = 0; l<conf.Nx*conf.Ny; l++)
    	{
    		rho.get()[l] = eval_rho<double,order>(nt_r_curr, l, coeffs_restart.get(), conf_new);
    	}

        double electric_energy = poiss.solve( rho.get() );
        interpolate<double,order>( coeffs_restart.get() + nt_r_curr*stride_t, rho.get(), conf_new );

        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;

        // Print coefficients to file.
        coeff_restart_file << nt_r_curr << std::endl;
        for(size_t i = 0; i < stride_t; i++){
            coeff_restart_file << i << " " << coeffs_restart.get()[nt_r_curr*stride_t + i ] << std::endl;
        }
        stats_restart_file << n*conf.dt << " " << electric_energy << std::endl;

        std::cout << "n = " << nt_r_curr << " t = " << n*conf.dt << " Comp-time: " << timer_elapsed << std::endl;
        if(n % nt_restart == 0){
            timer.reset();
            arma::mat copy_mat(size_tensor*size_tensor,size_tensor*size_tensor);
            #pragma omp parallel for
            for(size_t ix = 0; ix <= n_r; ix++)
            for(size_t iy = 0; iy <= n_r; iy++)
            for(size_t iu = 0; iu <= n_r; iu++)
            for(size_t iv = 0; iv <= n_r; iv++){
                size_t lx = ix + size_tensor*iy;
                size_t lu = iu + size_tensor*iv;
        
                double x = ix * dx_r;
                double y = iy * dy_r;
                double u = umin + iu * du_r;
                double v = vmin + iv * dv_r;
        
                copy_mat(lx,lu) = eval_f<double,order>(nt_restart,x,y,u,v,coeffs_restart.get(),conf_new);
            }

            arma::svd_econ(U,s,V,copy_mat);

            #pragma omp parallel for
            for(size_t i = 1; i < s.n_elem; i++){
                if(s(i)/s(0) < tol || i >= max_rank){
                    s(i) = 0;
                }
            }

            restart_matrix = U * arma::diagmat(s) * V.t();

            // Copy last entry of coeff vector into restarted coeff vector.            
            #pragma omp parallel for
            for(size_t i = 0; i < stride_t; i++){
                coeffs_restart.get()[i] = coeffs_restart.get()[nt_restart*stride_t + i];
            }

            nt_r_curr = 1;
            restart_counter++;
            restarted = true;
            double time_restart = timer.elapsed();
            std::cout << "Restart took: " << time_restart << std::endl;
            timer_elapsed += time_restart;
        } else {
            nt_r_curr++;
        }
    }

    std::cout << "Total time: " << total_time << std::endl;
}


void test_read_in()
{
    std::cout << "Read in coeffs." << std::endl;
    std::ifstream coeff_str(std::string("../coeffs.txt"));
    int a = 0;
    for(size_t n = 0; n <= conf.Nt; n++){
        coeff_str >> a;
        for(size_t i = 0; i < stride_t; i++){
            coeff_str >> a >> coeffs_full.get()[n*stride_t + i];
        }
    }

    size_t nx_plot = 32;
    size_t nu_plot = nx_plot;
    double dx_plot = Lx/nx_plot;

    std::ofstream stat_str("read_in_stat.txt");
    for(size_t n = 0; n <= conf.Nt; n++){
        double max_val = 0;
        double l2_norm = 0;

        std::ofstream phi_exact_str("phi_exact" + std::to_string(n) + ".txt");
        for(size_t iy = 0; iy <= nx_plot; iy++){
            for(size_t ix = 0; ix <= nx_plot; ix++){            
                double x = ix*dx_plot;
                double y = iy*dx_plot;
    
                double phi_exact = eval<double,order>(x,y,coeffs_full.get() + n*stride_t, conf);
    
                phi_exact_str << x << " " << y << " " << phi_exact << std::endl;

                max_val = std::max(max_val,std::abs(phi_exact));
                l2_norm += phi_exact*phi_exact;
            }
            phi_exact_str << std::endl;
        }

        l2_norm *= dx_plot*dx_plot;

        stat_str << n*conf.dt << " " << max_val << " " << l2_norm << std::endl;  // norms of phi, not E!!!
        std::cout << n*conf.dt << " " << max_val << " " << l2_norm << std::endl;
    }

    arma::mat rho_mat(conf.Nx,conf.Ny);
    std::ofstream rho_300("rho_300.txt");
    for(size_t iy = 0; iy < conf.Ny; iy++){
        for(size_t ix = 0; ix < conf.Nx; ix++){            
            double x = ix*conf.dx;
            double y = iy*conf.dy;

            size_t l = ix + conf.Nx*iy;

            double rho_val = eval_rho<double,order>(300,l,coeffs_full.get(), conf);

            rho_300 << x << " " << y << " " << rho_val << std::endl;
            rho_mat(ix,iy) = rho_val;
        }
        rho_300 << std::endl;
    }

    std::ofstream rho_mat_str("rho_300_mat.txt");
    rho_mat_str << rho_mat;

    std::ofstream rho_301("rho_301.txt");
    for(size_t iy = 0; iy < conf.Ny; iy++){
        for(size_t ix = 0; ix < conf.Nx; ix++){            
            double x = ix*conf.dx;
            double y = iy*conf.dy;

            size_t l = ix + conf.Nx*iy;

            double rho_val = eval_rho<double,order>(301,l,coeffs_full.get(), conf);

            rho_301 << x << " " << y << " " << rho_val << std::endl;
            rho_mat(ix,iy) = rho_val;
            rho.get()[l] = rho_val;
        }
        rho_301 << std::endl;
    }

    std::ofstream rho_301_mat_str("rho_301_mat.txt");
    rho_301_mat_str << rho_mat;

    std::unique_ptr<double[]> coeffs_test { new double[ stride_t ] {} };

    poisson<double> poiss( conf );

    poiss.solve(rho.get());

    std::ofstream phi_pre_recomp_str("phi_pre_recomp" + std::to_string(301) + ".txt");
    for(size_t iy = 0; iy < conf.Ny; iy++){
        for(size_t ix = 0; ix < conf.Nx; ix++){            
            double x = ix*conf.dx;
            double y = iy*conf.dy;
            
            size_t l = ix + conf.Nx*iy;

            double phi_exact = rho.get()[l];

            phi_pre_recomp_str << x << " " << y << " " << phi_exact << std::endl;
        }
        phi_pre_recomp_str << std::endl;
    }

    interpolate<double,order>(coeffs_test.get(), rho.get(), conf);

    std::ofstream phi_recomp_str("phi_recomp" + std::to_string(301) + ".txt");
    for(size_t iy = 0; iy <= nx_plot; iy++){
        for(size_t ix = 0; ix <= nx_plot; ix++){            
            double x = ix*dx_plot;
            double y = iy*dx_plot;

            double phi_exact = eval<double,order>(x,y,coeffs_test.get(), conf);

            phi_recomp_str << x << " " << y << " " << phi_exact << std::endl;
        }
        phi_recomp_str << std::endl;
    }
}

void test_svd_linear_interpol()
{
    test_n = 300;

    std::cout << "Read in coeffs." << std::endl;
    std::ifstream coeff_str(std::string("../../coeffs.txt"));
	for(size_t i = 0; i <= conf.Nt*stride_t; i++){
        int a = 0;
		coeff_str >> a >> coeffs_full.get()[i];
	}

    std::cout << "Store coeffs." << std::endl;
    #pragma omp parallel for
    for(size_t i = 0; i < stride_t; i++){
        coeffs_restart.get()[i] = coeffs_full.get()[test_n*stride_t + i];
    }

    std::cout << "Init and read in matrices." << std::endl;
    arma::mat U(size_tensor*size_tensor,size_tensor*size_tensor);
    arma::mat V(size_tensor*size_tensor,size_tensor*size_tensor);
    arma::vec S(size_tensor);

    U.load(std::string("../U.txt"));
    V.load(std::string("../V.txt"));
    //S.load(std::string("../s_rank.txt"));
    S.load(std::string("../s.txt"));

    std::cout << "Compute restart matrix." << std::endl;
    restart_matrix = U * arma::diagmat(S) * V.t();

    config_t<double> conf_new(Nx, Nx, Nu, Nu, Nt, dt, 
        0, Lx, 0, Ly, umin, umax, vmin, vmax, 
        &linear_interpolation_4d);

    // Test current time-step:
    std::cout << "Plotting n=300." << std::endl;
    size_t nx_plot = 32;
    size_t nu_plot = nx_plot;
    double dx_plot = Lx/nx_plot;
    double du_plot = (umax - umin)/nu_plot;
    size_t iy_plot = nx_plot/2;
    size_t iv_plot = iy_plot;

    std::ofstream f_exact_str("f_exact.txt");
    std::ofstream f_approx_str("f_approx.txt");
    std::ofstream f_error_str("f_error.txt");
    for(size_t ix = 0; ix <= nx_plot; ix++){
        for(size_t iu = 0; iu <= nu_plot; iu++){
            double x = ix*dx_plot;
            double y = iy_plot*dx_plot;
            double u = umin + iu*du_plot;
            double v = vmin + iv_plot*du_plot;
            double f_approx = eval_f<double,4>(0,x,y,u,v, coeffs_restart.get(),conf_new);
            double f_exact = eval_f<double,4>(test_n,x,y,u,v, coeffs_full.get(),conf);
            double f_error = std::abs(f_approx - f_exact);

            f_approx_str << x << " " << u << " " << f_approx << std::endl;
            f_exact_str << x << " " << u << " " << f_exact << std::endl;
            f_error_str << x << " " << u << " " << f_error << std::endl;
        }
        f_approx_str << std::endl;
        f_exact_str << std::endl;
        f_error_str << std::endl;
    }

    // Test density and field computation:
    poisson<double> poiss( conf );

    std::unique_ptr<double,decltype(std::free)*> rho_approx { reinterpret_cast<double*>(std::aligned_alloc(64,
                                            sizeof(double)*conf.Nx*conf.Ny)), std::free };
    std::unique_ptr<double,decltype(std::free)*> rho_error { reinterpret_cast<double*>(std::aligned_alloc(64,
                                                sizeof(double)*conf.Nx*conf.Ny)), std::free };
    #pragma omp parallel for
    for(size_t l = 0; l<conf.Nx*conf.Ny; l++)
    {
        rho_approx.get()[l] = eval_rho<double,order>(1, l, coeffs_restart.get(), conf_new);
        rho.get()[l] = eval_rho<double,order>(test_n+1, l, coeffs_full.get(), conf);
        rho_error.get()[l] = std::abs(rho_approx.get()[l] - rho.get()[l])/std::abs(rho.get()[l]);
    }

    std::ofstream rho_approx_str("rho_approx.txt");
    std::ofstream rho_exact_str("rho_exact.txt");
    std::ofstream rho_error_str("rho_error.txt");
    for(size_t ix = 0; ix < conf.Nx; ix++){
        for(size_t iy = 0; iy < conf.Ny; iy++){
            size_t l = ix + conf.Nx * iy;
            double x = ix*conf.dx;
            double y = iy*conf.dy;
            double rho_value_approx = rho_approx.get()[l];
            double rho_value_exact = rho.get()[l];

            rho_approx_str << x << " " << y << " " << rho_value_approx << std::endl;
            rho_exact_str << x << " " << y << " " << rho_value_exact << std::endl;
            rho_error_str << x << " " << y << " " << rho_error.get()[l] << std::endl;
        }
        rho_approx_str << std::endl;
        rho_exact_str << std::endl;
        rho_error_str << std::endl;
    }

    for(size_t iy = 0; iy < conf.Ny; iy+=5){
        std::ofstream rho_approx_n_str(std::string("rho_approx_" + std::to_string(iy) + ".txt"));
        std::ofstream rho_exact_n_str(std::string("rho_exact_" + std::to_string(iy) + ".txt"));
        for(size_t ix = 0; ix < conf.Nx; ix++){
            size_t l = ix + conf.Nx * iy;
            double x = ix*conf.dx;
            double y = iy*conf.dy;
            double rho_value_approx = rho_approx.get()[l];
            double rho_value_exact = rho.get()[l];

            rho_approx_n_str << x << " " << rho_value_approx << std::endl;
            rho_exact_n_str << x << " " << rho_value_exact << std::endl;
        }
    }

    double E_energy_approx = poiss.solve( rho_approx.get() );
    double E_energy_exact = poiss.solve( rho.get() );

    std::ofstream after_poiss_solve_approx_str("after_poiss_solve_approx.txt");
    std::ofstream after_poiss_solve_exact_str("after_poiss_solve_exact.txt");
    std::ofstream after_poiss_solve_error_str("after_poiss_solve_error.txt");
    for(size_t ix = 0; ix < conf.Nx; ix++){
        for(size_t iy = 0; iy < conf.Ny; iy++){
            size_t l = ix + conf.Nx * iy;
            double x = ix*conf.dx;
            double y = iy*conf.dy;
            double phi_value_approx = rho_approx.get()[l];
            double phi_value_exact = rho.get()[l];

            after_poiss_solve_approx_str << x << " " << y << " " << phi_value_approx << std::endl;
            after_poiss_solve_exact_str << x << " " << y << " " << phi_value_exact << std::endl;
            after_poiss_solve_error_str << x << " " << y << " " << std::abs(phi_value_exact - phi_value_approx) << std::endl;
        }
        after_poiss_solve_approx_str << std::endl;
        after_poiss_solve_exact_str << std::endl;
        after_poiss_solve_error_str << std::endl;
    }

    std::cout << "Energy exact " << E_energy_exact << ". E energy approx " << E_energy_approx << std::endl;
    interpolate<double,order>( coeffs_restart.get() + stride_t, rho_approx.get(), conf );
    //interpolate<double,order>( coeffs_full.get() + 301*stride_t, rho.get(), conf );

    std::ofstream phi_approx_str("phi_approx.txt");
    std::ofstream phi_exact_str("phi_exact.txt");
    for(size_t ix = 0; ix <= nx_plot; ix++){
        for(size_t iy = 0; iy <= nx_plot; iy++){
            double x = ix*dx_plot;
            double y = iy*dx_plot;

            double phi_approx = eval<double,order>(x,y,coeffs_restart.get() + 1*stride_t, conf_new);
            double phi_exact = eval<double,order>(x,y,coeffs_full.get() + 301*stride_t, conf);

            phi_approx_str << x << " " << y << " " << phi_approx << std::endl;
            phi_exact_str << x << " " << y << " " << phi_exact << std::endl;
        }
        phi_approx_str << std::endl;
        phi_exact_str << std::endl;
    }

    for(size_t iy = 0; iy <= nx_plot; iy += 5){
        std::ofstream phi_approx_n_str(std::string("phi_approx_" + std::to_string(iy) + ".txt"));
        std::ofstream phi_exact_n_str(std::string("phi_exact_" + std::to_string(iy) + ".txt"));
        for(size_t ix = 0; ix < nx_plot; ix++){
            double x = ix*dx_plot;
            double y = iy*dx_plot;

            double phi_approx = eval<double,order>(x,y,coeffs_restart.get() + 1*stride_t, conf_new);
            double phi_exact = eval<double,order>(x,y,coeffs_full.get() + 301*stride_t, conf);

            phi_approx_n_str << x << " " << phi_approx << std::endl;
            phi_exact_n_str << x << " " << phi_exact << std::endl;
        }

    }


    // Test next time-step:
    std::cout << "Plotting n=301." << std::endl;
    std::ofstream f_exact_next_str("f_exact_next.txt");
    std::ofstream f_approx_next_str("f_approx_next.txt");
    for(size_t ix = 0; ix <= nx_plot; ix++){
        for(size_t iu = 0; iu <= nu_plot; iu++){
            double x = ix*dx_plot;
            double y = iy_plot*dx_plot;
            double u = umin + iu*du_plot;
            double v = vmin + iv_plot*du_plot;
            double f_approx = eval_f<double,order>(1,x,y,u,v,coeffs_restart.get()+1*stride_t,conf_new);
            double f_exact = eval_f<double,order>(test_n+1,x,y,u,v,coeffs_full.get(),conf);

            f_approx_next_str << x << " " << u << " " << f_approx << std::endl;
            f_exact_next_str << x << " " << u << " " << f_exact << std::endl;
        }
        f_approx_next_str << std::endl;
        f_exact_next_str << std::endl;
    }
}
}
}


int main()
{
    //nufi::dim2::test_svd_linear_interpol();

    //nufi::dim2::test_read_in();

    nufi::dim2::restart_with_svd();

	//nufi::dim2::test<double,4>();

    return 0;
}

