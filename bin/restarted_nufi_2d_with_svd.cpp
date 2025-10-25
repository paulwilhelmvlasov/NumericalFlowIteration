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
#include <functional>
#include <sstream>

#include <armadillo>

#include <nufi/config.hpp>
#include <nufi/random.hpp>
#include <nufi/fields.hpp>
#include <nufi/poisson.hpp>
#include <nufi/rho.hpp>
#include <nufi/stopwatch.hpp>
#include <nufi/restart.hpp>

namespace nufi
{

namespace dim2
{
const double Lx = 10*M_PI;//4*M_PI;
const double Ly = Lx;

const double x_min = 0;
const double x_max = Lx;
const double y_min = 0;
const double y_max = Ly;

const double u_min = -6;
const double u_max = 6;
const double v_min = -6;
const double v_max = 6;

const size_t nx_r = 64;
const size_t ny_r = nx_r;

const double dx_r = Lx/ nx_r;
const double dy_r = Ly/ ny_r;

const size_t nu_r = nx_r;
const size_t nv_r = nu_r;

const double du_r = (u_max - u_min)/nu_r;
const double dv_r = (v_max - v_min)/nv_r;

const size_t order = 2;
const size_t Nx = nx_r;  // Number of grid points in physical space.
const size_t Ny = ny_r;  // Number of grid points in physical space.
const size_t Nu = nu_r;  // Number of quadrature points in velocity space.
const size_t Nv = nv_r;  // Number of quadrature points in velocity space.
const double   dt = 0.1;  // Time-step size.
const size_t Nt = 100/dt;  // Number of time-steps.

size_t nt_restart = 10;

template <typename real>
real f0(real x, real y, real u, real v) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

    real alpha = 0.01;
    real k     = 0.5;

    // Weak Landau Damping:
    /* constexpr real c  = 0.06349363593424096978576330493464; // Weak Landau damping
    return c * ( 1. + alpha*cos(k*x) + alpha*cos(k*y) + alpha*cos(k*z)) 
             * exp( -(u*u+v*v+w*w)/2 ); */

    // 1d Two Stream Instability:
    //return 1.0 / std::sqrt(2.0 * M_PI) * u*u * std::exp(-0.5 * u*u) * (1 + alpha * std::cos(k*x)); 

    // 2d Two Stream Instability (pseudo-1d):
    //return 1.0/(2.0*M_PI) * ( 1. + alpha*cos(k*x) + alpha*cos(k*y)) * u*u * exp( -(u*u+v*v)/2 );

    // 2d Two Stream Instability (Kormann paper):
    alpha = 0.001;
    k     = 0.2;
    constexpr real speed = 2.4;
    return 1.0/(8.0*M_PI) * ( 1. + alpha*cos(k*x) + alpha*cos(k*y)) 
            * ( exp( -(u-speed)*(u-speed)/2 ) + exp( -(u+speed)*(u+speed)/2 ) )
            * ( exp( -(v-speed)*(v-speed)/2 ) + exp( -(v+speed)*(v+speed)/2 ) );
}

arma::mat F_r, F_r_copy;

double f_t_full(double x, double y, double u, double v) noexcept
{
        // This version is more stable.
    if( u > u_max || u < u_min 
        || v > v_max || v < v_min){
		return 0;
	} 

    x = std::fmod(std::fmod(x, Lx) + Lx, Lx);
    size_t x_ref_pos = std::min(static_cast<size_t>(std::floor(x / dx_r)), nx_r - 1);
    y = std::fmod(std::fmod(y, Ly) + Ly, Ly);
    size_t y_ref_pos = std::min(static_cast<size_t>(std::floor(y / dy_r)), ny_r - 1);

    size_t u_ref_pos = std::min(static_cast<size_t>(std::floor((u-u_min)/du_r)), nu_r - 1);
    size_t v_ref_pos = std::min(static_cast<size_t>(std::floor((v-v_min)/dv_r)), nv_r - 1);


    double x0 = x_ref_pos*dx_r;
    double y0 = y_ref_pos*dy_r;
    double u0 = u_min + u_ref_pos*du_r;    
    double v0 = v_min + v_ref_pos*dv_r;    

    double w_x = (x - x0)/dx_r;
    double w_y = (y - y0)/dy_r;
    double w_u = (u - u0)/du_r;
    double w_v = (v - v0)/dv_r;

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

        size_t index_0 = index_x + (nx_r+1)*index_y;
        size_t index_1 = index_u + (nu_r+1)*index_v;

        value += factor * F_r(index_0, index_1);
    }

    return value;
}

template <size_t order>
void restart_with_full_matrix(size_t& nt_r_curr, size_t n, double* coeffs, config_t<double>& conf, double& total_time, double tol = 1e-2, size_t max_rank = 10, size_t oversampling = 10)
{
    std::cout << "Restart" << std::endl;
    nufi::stopwatch<double> timer_restart;

    size_t stride_t = (conf.Nx + order - 1) *
                    (conf.Ny + order - 1);

    size_t size_x_r = (nx_r+1)*(ny_r+1);
    size_t size_v_r = (nu_r+1)*(nv_r+1);

    
    // Parallelize across columns so each thread writes contiguous rows
    // This way is slightly faster (20-30%) than a 4-loop over all dimensions. 
    double* Frc = F_r_copy.memptr();
    #pragma omp parallel for 
    for (size_t index_1 = 0; index_1 < size_v_r; index_1++) {
        size_t tmp1 = index_1;
        const size_t iu = index_1 % (nu_r+1);
        const size_t iv = index_1 / (nu_r+1);

        const double u = u_min + iu*du_r;
        const double v = v_min + iv*dv_r;

        for (size_t index_0 = 0; index_0 < size_x_r; index_0++) {
            const size_t ix = index_0 % (nx_r+1);
            const size_t iy = index_0 / (nx_r+1);

            const double x = x_min + ix*dx_r;
            const double y = y_min + iy*dy_r;

            const double f = eval_f<double,order>(nt_r_curr, x, y, u, v, coeffs, conf);

            // Armadillo column-major: memptr()[row + n_rows * col]
            //Frc[index_0 + size_x_r * index_1] = f;
            F_r_copy(index_0,index_1) = f;
        }
    }

    double timer_fill_restart_matrix = timer_restart.elapsed();
    timer_restart.reset();
    std::cout << "Filling restart matrix took " << timer_fill_restart_matrix << " s." << std::endl;

    // Test svd compression restart without svd-evaluation:
/*     auto A_mv_full = [&](const arma::vec& x) -> arma::vec {
        return F_r_copy * x;
    };

    auto At_mv_full = [&](const arma::vec& x) -> arma::vec {
        return F_r_copy.t() * x;
    };

    arma::mat U,V;
    arma::vec s;
    restart::randomized_svd_new(A_mv_full, At_mv_full, size_x_r, size_v_r, max_rank, U, s, V, oversampling);
    F_r = U * arma::diagmat(s) * V.t(); */
    // Full restart:
    F_r = F_r_copy;
    double timer_copy_mat = timer_restart.elapsed();
    timer_restart.reset();
    std::cout << "Copying restart matrix took " << timer_copy_mat << " s." << std::endl;

    #pragma omp parallel for
    for(size_t i = 0; i < stride_t; i++){
         coeffs[i] = coeffs[nt_r_curr*stride_t + i];
    }

    conf.f0 = f_t_full;
    std::cout << n << " " << nt_r_curr << " restart " << std::endl;
    nt_r_curr = 1;
    double restart_time = timer_restart.elapsed();
    total_time += restart_time;
    std::cout << "Restart took: " << restart_time << ". Total comp time s.f.: " << total_time << std::endl;
}

template <size_t order>
void run_restarted_simulation(bool svd_compressed = false, double tolerance = 1e-8, size_t max_rank = 30, size_t oversampling = 10)
{
    config_t<double> conf(Nx, Ny, Nu, Nv, Nt, dt, x_min, x_max, y_min, y_max, 
                        u_min, u_max, v_min, v_max, &f0);

    size_t stride_t = (conf.Nx + order - 1) *
                    (conf.Ny + order - 1);

    std::unique_ptr<double[]> coeffs_restart { new double[ (nt_restart+1)*stride_t ] {} };
    std::unique_ptr<double,decltype(std::free)*> rho { reinterpret_cast<double*>(std::aligned_alloc(64,
                                        sizeof(double)*conf.Nx*conf.Ny)), std::free };

    poisson<double> poiss( conf );

    if(!svd_compressed){
        size_t size_x_r = (nx_r+1)*(ny_r+1);
        size_t size_v_r = (nu_r+1)*(nv_r+1);
        F_r = arma::mat(size_x_r, size_v_r, arma::fill::zeros);
        F_r_copy = arma::mat(size_x_r, size_v_r, arma::fill::zeros);
    }
        
    std::ofstream stat_file( "stats.txt" );
    std::ofstream coeff_file( "coeffs.txt" );
    double total_time = 0;
    size_t nt_r_curr = 0;
    for ( size_t n = 0; n <= Nt; ++n )
    {
      	nufi::stopwatch<double> timer;

        std::cout << " start of time step "<< n << " " << nt_r_curr  << std::endl; 
		nufi::stopwatch<double> rho_timer;
        #pragma omp parallel for
    	for(size_t l = 0; l<conf.Nx*conf.Ny; l++)
    	{
    		rho.get()[l] = eval_rho<double,order>(nt_r_curr, l, coeffs_restart.get(), conf);
    	}
        double rho_comp_time = rho_timer.elapsed();
        std::cout << "rho comp time = " << rho_comp_time << " per dof = " <<  rho_comp_time/(conf.Nx*conf.Ny) << std::endl;

        double E_energy = poiss.solve( rho.get() );
        interpolate<double,order>( coeffs_restart.get() + nt_r_curr*stride_t, rho.get(), conf );


        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;

        double t = n*conf.dt;
        stat_file << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << " " << E_energy << std::endl;
        std::cout << std::setw(15) << t << std::setw(15) << std::setprecision(5) << std::scientific << E_energy << " Comp-time: " << timer_elapsed;
        std::cout << " Total comp time s.f.: " << total_time << std::endl; 

        // Print coefficients to file.
        coeff_file << n << std::endl;
        for(size_t i = 0; i < stride_t; i++){
            coeff_file << i << " " << coeffs_restart.get()[nt_r_curr*stride_t + i ] << std::endl;
        }

        if(nt_r_curr == nt_restart)
    	{
            if(svd_compressed){
                /* restart_with_rsvd_compression<order>(nt_r_curr,n,coeffs_restart.get(),conf,
                                                total_time,tolerance,max_rank,oversampling); */
            } else {
                restart_with_full_matrix<order>(nt_r_curr,n,coeffs_restart.get(),conf,
                                                total_time,tolerance,max_rank,oversampling);
            }
            
        } else {
            nt_r_curr++;
        }
    }
}

}
}


int main()
{
    nufi::dim2::run_restarted_simulation<2>(false,1e-16,10,5);
}