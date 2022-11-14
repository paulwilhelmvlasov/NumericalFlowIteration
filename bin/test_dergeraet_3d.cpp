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

namespace dim3
{

template <typename real, size_t order>
void do_matthias_shit( size_t n, size_t rank, size_t my_begin, size_t my_end,
                       config_t<real> conf, cuda_scheduler<real,order> &sched, real electric_energy );

template <typename real, size_t order>
void test()
{
    using std::hypot;
    using std::max;

    config_t<real> conf;
    poisson<real> poiss( conf );

    size_t stride_t = (conf.Nx + order - 1) *
                      (conf.Ny + order - 1) *
					  (conf.Nz + order - 1);

    int iworld_size, irank;
    mpi::comm_size(MPI_COMM_WORLD,&iworld_size);
    mpi::comm_rank(MPI_COMM_WORLD,&irank);

    size_t world_size = iworld_size;
    size_t rank       = irank;

    size_t Nrho = conf.Nx * conf.Ny * conf.Nz;
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

    cuda_scheduler<real,order> sched { conf };

    std::unique_ptr<real[]> coeffs { new real[ (conf.Nt+1)*stride_t ] {} };
    void *tmp = std::aligned_alloc( poiss.alignment, sizeof(real)*conf.Nx*conf.Ny*conf.Nz );
    if ( tmp == nullptr ) throw std::bad_alloc {};
    std::unique_ptr<real,decltype(std::free)*> rho { reinterpret_cast<real*>(tmp), std::free };

    std::cout << "Nx = " << conf.Nx << std::endl;
    std::cout << "Ny = " << conf.Ny << std::endl;
    std::cout << "Nz = " << conf.Nz << std::endl;
    std::cout << "Nu = " << conf.Nu << std::endl;
    std::cout << "Nv = " << conf.Nv << std::endl;
    std::cout << "Nw = " << conf.Nw << std::endl;
    std::cout << "dt = " << conf.dt << std::endl;

    real t_total = 0;
    for ( size_t n = 0; n <= conf.Nt; ++n )
    {
        // The actual NuFI loop.
        sched. compute_rho( n, my_begin, my_end );
        sched.download_rho( rho.get(), my_begin, my_end );

        mpi::allgatherv( MPI_IN_PLACE, 0, 
                         rho.get(), rank_count.data(), rank_offset.data(),
                         MPI_COMM_WORLD );

        const real electric_energy = poiss.solve( rho.get() );
        interpolate<real,order>( coeffs.get() + n*stride_t, rho.get(), conf );
        sched.upload_phi( n, coeffs.get() );

        // After this, we can do shit to output statistics, etc.
        do_matthias_shit(n,rank,my_begin,my_end,conf,sched,electric_energy);
        //do_pauls_shit();
    }

    std::cout << "Total time = " << t_total << std::endl;
}

template <typename real, size_t order>
void do_matthias_shit( size_t n, size_t rank, size_t my_begin, size_t my_end,
                       config_t<real> conf, cuda_scheduler<real,order> &sched, real electric_energy )
{
    real metrics[4];
    sched.compute_metrics( n, my_begin, my_end );
    sched.download_metrics( metrics );
    mpi::allreduce_add( MPI_IN_PLACE, metrics, 4, MPI_COMM_WORLD );
    metrics[1]  = std::sqrt(metrics[1]); // Take square-root for L²-norm
    //metrics[2] += electric_energy;       // Total energy = kinetic + electric energy

    if ( rank == 0 )
    {
        for ( size_t i = 0; i < 80; ++i )
            std::cout << '=';
        std::cout << std::endl;

        std::cout << std::setprecision(7) << std::scientific;
        std::cout << u8"t = " << n*conf.dt << '.' << std::endl;
        std::cout << u8"L¹-norm:      " << std::setw(20) << metrics[0] << std::endl;
        std::cout << u8"L²-norm:      " << std::setw(20) << metrics[1] << std::endl;
        std::cout << u8"Total Energy: " << std::setw(20) << metrics[2] << std::endl;
        std::cout << u8"Entropy:      " << std::setw(20) << metrics[3] << std::endl;

        std::cout << std::endl;
    }
}

}

}

int main( int argc, char *argv[] )
{
    dergeraet::mpi::programme prog(&argc,&argv);
    dergeraet::dim3::test<double,4>();
}

