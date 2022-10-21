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

#include <mpi.h>

#include <dergeraet/config.hpp>
#include <dergeraet/random.hpp>
#include <dergeraet/fields.hpp>
#include <dergeraet/poisson.hpp>
#include <dergeraet/rho.hpp>
#include <dergeraet/stopwatch.hpp>
#include <dergeraet/cuda_scheduler.hpp>
#include <dergeraet/mpi.hpp>

inline
void mpi_guard( int errcode )
{
    char buf[ MPI_MAX_ERROR_STRING + 1 ];
    int  len;
    if ( errcode != MPI_SUCCESS )
    {
        MPI_Error_string( errcode, buf, &len );
        buf[ len ] = '\0';
        throw std::runtime_error { std::string { buf } };
    }
}

namespace dergeraet
{

namespace dim2
{

template <typename real, size_t order>
void test()
{
    using std::hypot;
    using std::max;

    config_t<real> conf;
    poisson<real> poiss( conf );

    size_t stride_t = (conf.Nx + order - 1) *
                      (conf.Ny + order - 1);

    int iworld_size, irank;
    mpi::comm_size(MPI_COMM_WORLD,&iworld_size);
    mpi::comm_rank(MPI_COMM_WORLD,&irank);
   
    size_t world_size = iworld_size;
    size_t rank       = irank; 

    size_t Nrho = conf.Nx * conf.Ny;
    size_t chunk_size = Nrho / world_size;
    size_t remainder  = Nrho % world_size;
    std::vector<size_t> rank_boundaries( world_size + 1 );
    rank_boundaries[ 0 ] = 0;
    for ( size_t i = 0; i < world_size; ++i )
    {
        if ( i < remainder )
        {
            rank_boundaries[ i + 1 ] = rank_boundaries[ i ] + chunk_size + 1;
        }
        else
        {
            rank_boundaries[ i + 1 ] = rank_boundaries[ i ] + chunk_size;
        }
    }

    std::vector<int> rank_count( world_size ), rank_offset( world_size );
    for ( size_t i = 0; i < world_size; ++i )
    {
        rank_count [ i ] = rank_boundaries[i+1] - rank_boundaries[i];
        rank_offset[ i ] = rank_boundaries[ i ];
    }

    if ( rank == 0 )
    {
        std::cout << u8"Running Der Gerät on " << world_size << " ranks. " << std::endl;
        std::cout << u8"Loads are as follows: " << std::endl;
        for ( size_t i = 0; i < world_size; ++i )
            std::cout << "Rank " << std::setw(5) << i 
                      << ": "    << std::setw(5) << rank_count[ i ] << ", "
                      << "from: "    << std::setw(5) << rank_boundaries[ i ] << " "
                      << "to:   "    << std::setw(5) << rank_boundaries[ i + 1 ] << std::endl;
    }

    cuda_scheduler<real,order> sched { conf, rank_boundaries[ rank ], rank_boundaries[ rank + 1 ] };


    std::unique_ptr<real[]> coeffs { new real[ (conf.Nt+1)*stride_t ] {} };
    void *tmp = std::aligned_alloc( poiss.alignment, sizeof(real)*conf.Nx*conf.Ny );
    if ( tmp == nullptr ) throw std::bad_alloc {};
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(tmp), std::free };

    void *tmp_f = std::aligned_alloc( poiss.alignment, sizeof(real)*conf.Nx*conf.Ny*conf.Nv*conf.Nu );
    if ( tmp_f == nullptr ) throw std::bad_alloc {};
    std::unique_ptr<real,decltype(std::free)*> f_values { reinterpret_cast<real*>(tmp_f), std::free };

    bool plot_f = true;

    std::ofstream E_max_file;
    if ( rank == 0 )
        E_max_file.open( "E_max.txt" );
    std::ofstream E_l2_file;
    if ( rank == 0 )
        E_l2_file.open( "E_l2.txt" );
    std::ofstream E_max_err_file;
    if ( rank == 0 )
        E_max_err_file.open( "E_max_err.txt" );
    std::ofstream E_l2_err_file;
    if ( rank == 0 )
        E_l2_err_file.open( "E_l2_err.txt" );
    std::ofstream entropy_file;
    if ( rank == 0 )
        entropy_file.open( "entropy.txt" );
    std::ofstream total_energy_file;
    if ( rank == 0 )
    	total_energy_file.open( "total_energy.txt" );
    std::ofstream l1_norm_f_file;
    if ( rank == 0 )
    	l1_norm_f_file.open( "l1_norm_f.txt" );
    std::ofstream l2_norm_f_file;
    if ( rank == 0 )
    	l2_norm_f_file.open( "l2_norm_f.txt" );

    double t_total = 0;
    for ( size_t n = 0; n <= conf.Nt; ++n )
    {
        double t1 = MPI_Wtime();
        // Without f metrics:
        //sched.compute_rho( n, coeffs.get(), rho.get() );
        // With f metrics:
        sched.compute_rho( n, coeffs.get(), rho.get(), f_values.get() );

        mpi::allgatherv( MPI_IN_PLACE, 0, 
                         rho.get(), rank_count.data(), rank_offset.data(),
                         MPI_COMM_WORLD );

        poiss.solve( rho.get() );

        if ( n )
        {
            // Set initial guess as previous time step solution.
            for ( size_t l = 0; l < stride_t; ++l )
                coeffs[ n*stride_t + l ] = coeffs[ (n-1)*stride_t + l ];
        }
        interpolate<real,order>( coeffs.get() + n*stride_t, rho.get(), conf );

        if ( rank == 0 )
        {
            real Emax = 0;
            for ( size_t l = 0; l < conf.Nx * conf.Ny; ++l )
            {
                size_t i = l % conf.Nx;
                size_t j = l / conf.Nx;
                real x = conf.x_min + i*conf.dx;
                real y = conf.y_min + j*conf.dy;

                real Ex = -eval<real,order,1,0>( x, y, coeffs.get() + n*stride_t, conf );
                real Ey = -eval<real,order,0,1>( x, y, coeffs.get() + n*stride_t, conf );

                Emax = max( Emax, hypot(Ex,Ey) );
            }
            E_max_file << std::setw(15) << n*conf.dt
                      << std::setw(15) << std::setprecision(5) << std::scientific << Emax << std::endl;

            std::cout << "t    = " << std::setw(15) << n*conf.dt  << ", "
                      << "Emax = " << std::setw(15) << std::setprecision(5) << std::scientific << Emax << ". "; 
            double t2 = MPI_Wtime();
            double t_time_step = t2-t1;
            std::cout << "Elapsed time: " << t_time_step << std::endl;
            t_total += t_time_step;
        }

        bool do_plots = (n%10 == 0);
    // Plotting E, rho and related metrics.
		size_t t = n*conf.dt;
        real E_l2 = 0;
        real E_max = 0;
        real E_l2_error = 0;
        real E_max_error = 0;
    if(do_plots)
	{
	        std::stringstream E_filename; E_filename << 'E' << t << ".txt";
        	std::ofstream E_file( E_filename.str() );
		std::stringstream rho_filename; rho_filename << 'p' << t << ".txt";
	        std::ofstream rho_file(rho_filename.str() );
		std::ifstream E_exact_file("../comparison/E" + std::to_string(t) + ".txt");

        	const size_t plotNx = 256, plotNy = 256;
		const real plot_dx = (conf.x_max - conf.x_min)/plotNx;
                const real plot_dy = (conf.y_max - conf.y_min)/plotNy;
		for ( size_t i = 0; i <= plotNx; ++i )
        	{
	            real x = conf.x_min + i*plot_dx;
	            for ( size_t j = 0; j <= plotNy; ++j )
        	    {
                	real y = conf.y_min + j*plot_dy;
	                size_t l = j + i * plotNy;
			real Ex = -eval<real,order,1,0>( x, y, coeffs.get() + n*stride_t, conf );
	                real Ey = -eval<real,order,0,1>( x, y, coeffs.get() + n*stride_t, conf );
			real a  = eval<real,order,2,0>(x,y, coeffs.get() + n*stride_t, conf);
			real b  = eval<real,order,0,2>(x,y, coeffs.get() + n*stride_t, conf); 
			real rho = -a*a - b*b;
	                E_file << x << " " << y << " " << Ex << " " << Ey << std::endl;
			rho_file << x << " " << y << " " << rho << std::endl;

			real E_abs = std::sqrt(Ex*Ex + Ey*Ey);
			E_l2 += E_abs;

			real Ex_exact = 0;
			real Ey_exact = 0;
			E_exact_file >> x >> y >> Ex_exact >> Ey_exact;

			real dist_x = Ex_exact - Ex;
			real dist_y = Ey_exact - Ey;
			real dist = std::sqrt( dist_x*dist_x + dist_y*dist_y);
			E_l2_error += dist;
			E_max_error = std::max(dist, E_max_error);
        	    }
	            E_file << std::endl;
	            rho_file << std::endl;
		    E_exact_file.ignore();
        	}
		
		E_l2_error *= plot_dx*plot_dy;
		E_l2 *= plot_dx*plot_dy;

		E_l2_file << t << " " << E_l2 << std::endl;
		E_max_err_file << t << " " << E_max_error << std::endl;
		E_l2_err_file << t << " " << E_l2_error << std::endl;
	}
    // Plotting of f and f-related metrics:
    if(plot_f && do_plots)
    {
		const size_t plot_nx = conf.Nx;
		const size_t plot_ny = conf.Ny;
		const size_t plot_nv = conf.Nv;
		const size_t plot_nu = conf.Nu;

    	const real plot_dx = (conf.x_max - conf.x_min)/plot_nx;
    	const real plot_dy = (conf.y_max - conf.y_min)/plot_ny;
    	const real plot_dv = (conf.v_max - conf.v_min)/plot_nv;
    	const real plot_du = (conf.u_max - conf.u_min)/plot_nu;

    	real entropy = 0;
    	real kinetic_energy = 0;
    	real l1_norm_f = 0;
    	real l2_norm_f = 0;

    	for(size_t i = 0; i < plot_nx; i++)
    		for(size_t j = 0; j < plot_ny; j++)
    			for(size_t k = 0; k < plot_nv; k++)
    				for(size_t l = 0; l < plot_nu; l++)
    				{
    					real x = conf.x_min + i * plot_dx;
    					real y = conf.y_min + j * plot_dy;
    					real v = conf.v_min + k * plot_dv;
    					real u = conf.u_min + l * plot_du;
    					real f = f_values.get()[i + conf.Nx*(j + conf.Ny*(k + conf.Nv*l))];
    					//real f = eval_ftilda<real,order>( n, x, y, u, v, coeffs.get(), conf );

    					if(f > 1e-10)
    					{
    						entropy += f * std::log(f);
    					}

    					kinetic_energy += (v*v + u*u) * f;

    					l1_norm_f += f;
    					l2_norm_f += f*f;
    				}
    	entropy *= plot_dx*plot_dy*plot_dv*plot_du;
    	kinetic_energy *= plot_dx*plot_dy*plot_dv*plot_du;
    	l1_norm_f *= plot_dx*plot_dy*plot_dv*plot_du;
    	l2_norm_f *= plot_dx*plot_dy*plot_dv*plot_du;
    	entropy_file << t << " " << entropy << std::endl;
    	total_energy_file << t << " " << kinetic_energy + E_l2 << std::endl;
    	l1_norm_f_file << t << " " << entropy << std::endl;
    	l2_norm_f_file << t << " " << entropy << std::endl;
    }
    }
    std::cout << "Total time = " << t_total << std::endl;
}

}

}


int main( int argc, char *argv[] )
{
    dergeraet::mpi::programme prog(&argc,&argv);
    dergeraet::dim2::test<float,4>();

}

