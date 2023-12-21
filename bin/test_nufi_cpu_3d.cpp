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


#include <dergeraet/config.hpp>
#include <dergeraet/random.hpp>
#include <dergeraet/fields.hpp>
#include <dergeraet/poisson.hpp>
#include <dergeraet/rho.hpp>
#include <dergeraet/stopwatch.hpp>
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

namespace dim3
{

template <typename real, size_t order>
void do_stats( size_t n, size_t rank, size_t my_begin, size_t my_end,
               config_t<real> conf, real electric_energy,
			   const real *coeffs, std::ofstream& statistics_file );

template <typename real, size_t order>
void test()
{
    using std::abs;
    using std::max;

    config_t<real> conf;
    size_t stride_t = (conf.Nx + order - 1) *
                      (conf.Ny + order - 1) *
					  (conf.Nz + order - 1);


    int iworld_size, irank;
    mpi::comm_size(MPI_COMM_WORLD,&iworld_size);
    mpi::comm_rank(MPI_COMM_WORLD,&irank);

    size_t world_size = iworld_size;
    size_t rank       = irank;

    size_t Nquad = conf.Nx * conf.Ny * conf.Nz *conf.Nu * conf.Nv * conf.Nw;
    size_t chunk_size = Nquad / world_size;
    size_t remainder  = Nquad % world_size;
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

    const size_t my_begin = rank_boundaries[ rank ];
    const size_t my_end   = rank_boundaries[ rank + 1 ];

    if ( rank == 0 )
    {
        std::cout << u8"Running NuFI on " << world_size << " ranks. " << std::endl;
        std::cout << u8"Loads are as follows: " << std::endl;
        for ( size_t i = 0; i < world_size; ++i )
            std::cout << "Rank "     << std::setw(5) << i
                      << ": "        << std::setw(5) << rank_count[ i ] << ", "
                      << "from: "    << std::setw(5) << rank_boundaries[ i ] << " "
                      << "to:   "    << std::setw(5) << rank_boundaries[ i + 1 ] << std::endl;
    }

    if(rank == 0)
    {
		std::cout << "Nx = " << conf.Nx << std::endl;
		std::cout << "Ny = " << conf.Ny << std::endl;
		std::cout << "Nz = " << conf.Nz << std::endl;
		std::cout << "Nu = " << conf.Nu << std::endl;
		std::cout << "Nv = " << conf.Nv << std::endl;
		std::cout << "Nw = " << conf.Nw << std::endl;
		std::cout << "dt = " << conf.dt << std::endl;
    }

    std::unique_ptr<real[]> coeffs { new real[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(std::aligned_alloc(64,
    													sizeof(real)*conf.Nx*conf.Ny*conf.Nz)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};

    std::ofstream statistics_file( "statistics.csv" );
    statistics_file << R"("Time"; "L1-Norm"; "L2-Norm"; "Electric Energy"; "Kinetic Energy"; "Total Energy"; "Entropy")";
    statistics_file << std::endl;

    statistics_file << std::scientific;
          std::cout << std::scientific;


    poisson<real> poiss( conf );

    double total_time = 0;
    for ( size_t n = 0; n <= conf.Nt; ++n )
    {
    	dergeraet::stopwatch<double> timer;

        std::memset( rho.get(), 0, sizeof(real)*conf.Nx*conf.Ny*conf.Nz );

        // Compute rho:
		#pragma omp parallel for
    	for(size_t l = my_begin; l < my_end; l++)
    	{
    		// Note that l \in {0,...,N} with N = Nx*Ny*Nz*Nu*Nv*Nw.
    		// Thus one has to first convert l into the correct indices.
    	    size_t tmp = l;
    	    const size_t iz  = tmp / ( conf.Ny*conf.Nx*conf.Nw*conf.Nv*conf.Nu );
    	                 tmp = tmp % ( conf.Ny*conf.Nx*conf.Nw*conf.Nv*conf.Nu );
    	    const size_t iy  = tmp / ( conf.Nx*conf.Nw*conf.Nv*conf.Nu );
    	                 tmp = tmp % ( conf.Nx*conf.Nw*conf.Nv*conf.Nu );
    	    const size_t ix  = tmp / ( conf.Nw*conf.Nv*conf.Nu );
    	                 tmp = tmp % ( conf.Nw*conf.Nv*conf.Nu );
    	    const size_t iw  = tmp / ( conf.Nv*conf.Nu );
    	                 tmp = tmp % ( conf.Nv*conf.Nu );
    	    const size_t iv  = tmp / ( conf.Nu );
    	    const size_t iu  = tmp % ( conf.Nu );

    	    const real x = conf.x_min + ix*conf.dx;
    	    const real y = conf.y_min + iy*conf.dy;
    	    const real z = conf.z_min + iz*conf.dz;
    	    const real u = conf.u_min + iu*conf.du + conf.du/2;
    	    const real v = conf.v_min + iv*conf.dv + conf.dv/2;
    	    const real w = conf.w_min + iw*conf.dw + conf.dw/2;

    	    const real weight = conf.du*conf.dv*conf.dw;
    	    const real f = eval_ftilda<real,order>( n, x, y, z, u, v, w,coeffs.get(),conf);

			#pragma omp atomic
    		rho.get()[iz*conf.Nx*conf.Ny + iy*conf.Nx + ix ] -= weight*f;
    	}

        mpi::allreduce_add( MPI_IN_PLACE, rho.get(), conf.Nx*conf.Ny*conf.Nz, MPI_COMM_WORLD );

        real E_energy = poiss.solve( rho.get() );
        interpolate<real,order>( coeffs.get() + n*stride_t, rho.get(), conf );

        if(rank == 0)
        {
			double timer_elapsed = timer.elapsed();
			total_time += timer_elapsed;
			double t = n*conf.dt;
			std::cout << "n = " << n << " t = " << n*conf.dt << " Comp-time: " << timer_elapsed << ". Total time s.f.: " << total_time << std::endl;
        }
        // After this, we can output statistics, etc.
        if(n % 4 == 0 )
        {
        	do_stats<real,order>(n,rank,my_begin,my_end,conf,E_energy,coeffs.get(),statistics_file);
        }

    }
    std::cout << "Total time: " << total_time << std::endl;
}

template <typename real, size_t order>
void do_stats( size_t n, size_t rank, size_t my_begin, size_t my_end,
                       config_t<real> conf, real electric_energy,
					   const real *coeffs, std::ofstream& statistics_file)
{
    real metrics[4] { 0, 0, 0, 0 };

    // Do metrics computations.
	const real weight = conf.du*conf.dv*conf.dw;
	#pragma omp parallel for
	for(size_t l = my_begin; l < my_end; l++)
	{
		// Note that l \in {0,...,N} with N = Nx*Ny*Nz*Nu*Nv*Nw.
		// Thus one has to first convert l into the correct indices.
		size_t tmp = l;
		const size_t iz  = tmp / ( conf.Ny*conf.Nx*conf.Nw*conf.Nv*conf.Nu );
					 tmp = tmp % ( conf.Ny*conf.Nx*conf.Nw*conf.Nv*conf.Nu );
		const size_t iy  = tmp / ( conf.Nx*conf.Nw*conf.Nv*conf.Nu );
					 tmp = tmp % ( conf.Nx*conf.Nw*conf.Nv*conf.Nu );
		const size_t ix  = tmp / ( conf.Nw*conf.Nv*conf.Nu );
					 tmp = tmp % ( conf.Nw*conf.Nv*conf.Nu );
		const size_t iw  = tmp / ( conf.Nv*conf.Nu );
					 tmp = tmp % ( conf.Nv*conf.Nu );
		const size_t iv  = tmp / ( conf.Nu );
		const size_t iu  = tmp % ( conf.Nu );

		const real x = conf.x_min + ix*conf.dx;
		const real y = conf.y_min + iy*conf.dy;
		const real z = conf.z_min + iz*conf.dz;
		const real u = conf.u_min + iu*conf.du + conf.du/2;
		const real v = conf.v_min + iv*conf.dv + conf.dv/2;
		const real w = conf.w_min + iw*conf.dw + conf.dw/2;

		const real f = eval_ftilda<real,order>( n, x, y, z, u, v, w, coeffs, conf );

		#pragma omp atomic
		metrics[0] += f;
		#pragma omp atomic
		metrics[1] += f*f;
		#pragma omp atomic
		metrics[2] += (u*u+v*v+w*w)*f/2;
		#pragma omp atomic
		metrics[3] += (f>0) ? -f*log(f) : 0;
	}

    mpi::allreduce_add( MPI_IN_PLACE, metrics, 4, MPI_COMM_WORLD );

    if(rank == 0)
    {
		metrics[0] *= weight;
		metrics[1] *= weight;
		metrics[2] *= weight;
		metrics[3] *= weight;
		metrics[1]  = std::sqrt(metrics[1]); // Take square-root for L²-norm

		real kinetic_energy = metrics[2];
		real total_energy   = kinetic_energy + electric_energy;

        for ( size_t i = 0; i < 80; ++i )
            std::cout << '=';
        std::cout << std::endl;

        std::cout << std::setprecision(7) << std::scientific;
        std::cout << u8"t = " << n*conf.dt << '.' << std::endl;
        std::cout << u8"L¹-norm:      " << std::setw(20) << metrics[0]   << std::endl;
        std::cout << u8"L²-norm:      " << std::setw(20) << metrics[1]   << std::endl;
        std::cout << u8"Total Energy: " << std::setw(20) << total_energy << std::endl;
        std::cout << u8"Entropy:      " << std::setw(20) << metrics[3]   << std::endl;

        std::cout << std::endl;

        statistics_file << conf.dt*n       << "; "
                        << metrics[0]      << "; "
                        << metrics[1]      << "; "
                        << electric_energy << "; "
                        <<  kinetic_energy << "; "
                        <<    total_energy << "; "
                        << metrics[3]      << std::endl;
    }
}

}
}


int main( int argc, char *argv[] )
{
    dergeraet::mpi::programme prog(&argc,&argv);
	dergeraet::dim3::test<double,4>();
}

