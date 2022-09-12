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


    std::ofstream Emax_file;
    if ( rank == 0 )
        Emax_file.open( "Emax2d.txt" );

    for ( size_t n = 0; n <= conf.Nt; ++n )
    {
    	stopwatch<real> clock;
        sched.compute_rho( n, coeffs.get(), rho.get() );

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
            Emax_file << std::setw(15) << n*conf.dt
                      << std::setw(15) << std::setprecision(5) << std::scientific << Emax << std::endl; 

            std::cout << "t    = " << std::setw(15) << n*conf.dt  << ", "
                      << "Emax = " << std::setw(15) << std::setprecision(5) << std::scientific << Emax << std::endl; 
    
        }
        /*
        std::stringstream filename; filename << 'f' << n << ".txt"; 
        std::ofstream file( filename.str() );
        const size_t plotNu = 512, plotNx = 512;
        for ( size_t i = 0; i <= plotNu; ++i )
        {
            real u = conf.u_min + i*(conf.u_max-conf.u_min)/plotNu;
            for ( size_t j = 0; j <= plotNx; ++j )
            {
                real x = conf.x_min + j*(conf.x_max-conf.x_min)/plotNx;
                file << x << " " << u << " " << dim2::eval_f<real,order>( n, x, 0, u, 0, coeffs.get(), conf ) << std::endl;
            }
            file << std::endl;
        }
		*/

        std::stringstream filename; filename << 'p' << n*conf.dt << ".txt";
        std::ofstream file( filename.str() );
        const size_t plotNx = conf.Nx, plotNy = conf.Ny;
        for ( size_t i = 0; i <= plotNx; ++i )
        {
            real x = conf.x_min + i*(conf.x_max-conf.x_min)/plotNx;
            for ( size_t j = 0; j <= plotNy; ++j )
            {
                real y = conf.y_min + j*(conf.y_max-conf.y_min)/plotNy;
                size_t l = j + i * plotNy;
                file << x << " " << y << " " <<
                		dim2::eval_rho<real,order>( n, l, coeffs.get(), conf )
					 << std::endl;

            }
            file << std::endl;
        }

    }

}

}

}


int main( int argc, char *argv[] )
{
    dergeraet::mpi::programme prog(&argc,&argv);
    dergeraet::dim2::test<float,4>();

}

