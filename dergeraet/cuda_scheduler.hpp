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
#ifndef DERGERAET_CUDA_SCHEDULER_HPP
#define DERGERAET_CUDA_SCHEDULER_HPP

#include <cmath>
#include <dergeraet/stopwatch.hpp>
#include <dergeraet/cuda_kernel.hpp>

#ifdef HAVE_CUDA

namespace dergeraet
{

namespace dim1
{

template <typename real, size_t order>
class cuda_scheduler
{
public:
    cuda_scheduler() = delete;
    cuda_scheduler( const cuda_scheduler&  ) = delete;
    cuda_scheduler(       cuda_scheduler&& ) = default;
    cuda_scheduler& operator=( const cuda_scheduler&  ) = delete;
    cuda_scheduler& operator=(       cuda_scheduler&& ) = default;

    cuda_scheduler( const config_t<real> &p_conf, size_t p_begin, size_t p_end ):
    conf { p_conf }, begin { p_begin }, end { p_end }
    {
        size_t n_dev = cuda::device_count();
        kernels.reserve(n_dev);

         for ( size_t i =0; i< n_dev; i++ )
        {
            try 
            {
                kernels.emplace_back( cuda_kernel<real,order>(conf,i) );
            }
            catch ( cuda::exception &ex )
            {
                // Do not use this device.
            }
        }

        std::cout << "Running on " << kernels.size() << " CUDA devices.\n";

        if ( kernels.size() == 0 )
            throw cuda::exception( cudaErrorUnknown, "cuda_scheduler: Failed to create kernels." );
    }

    void compute_rho( size_t n, const real *coeffs, real *rho )
    {
        if ( begin == end ) return;

        size_t n_cards = kernels.size();
        size_t N = end - begin;
        size_t chunk_size = N / n_cards;

        for ( size_t i = 0; i < n_cards - 1; ++i )
            kernels[i].compute_rho( n, coeffs, begin + i*chunk_size, begin + (i+1)*chunk_size );
        kernels.back().compute_rho( n, coeffs, begin + (n_cards-1)*chunk_size, end);

        for ( size_t i = 0; i < n_cards-1; ++i )
            kernels[i].load_rho( rho, begin + i*chunk_size, begin + (i+1)*chunk_size ); 
        kernels.back().load_rho( rho, begin + (n_cards-1)*chunk_size, end);
        }

private:
    config_t<real> conf;

    size_t begin;
    size_t end;

    std::vector< cuda_kernel<real,order> > kernels;
};



}

namespace dim2
{

template <typename real, size_t order>
class cuda_scheduler
{
public:
    cuda_scheduler() = delete;
    cuda_scheduler( const cuda_scheduler&  ) = delete;
    cuda_scheduler(       cuda_scheduler&& ) = default;
    cuda_scheduler& operator=( const cuda_scheduler&  ) = delete;
    cuda_scheduler& operator=(       cuda_scheduler&& ) = default;

    cuda_scheduler( const config_t<real> &p_conf, size_t p_begin, size_t p_end ):
    conf { p_conf }, begin { p_begin }, end { p_end }
    {
        size_t n_dev = cuda::device_count();
        kernels.reserve(n_dev);
        std::cout<<" number of devices: "<<n_dev<<std::endl;

       
        for ( size_t i =0; i< n_dev; i++)
        {
            try 
            {

                kernels.emplace_back( cuda_kernel<real,order>(conf,i) );
		        std::cout << "Added dövice " << i << ".\n"; std::cout.flush();
            }
            catch ( cuda::exception &ex )
            {
                
                std::cout<<" did not add "<<i<<std::endl; std::cout.flush();
                // Do not use this device.
            }
        }
        std::cout << "Running on " << kernels.size() << " CUDA devices.\n";

        if ( kernels.size() == 0 )
            throw cuda::exception( cudaErrorUnknown, "cuda_scheduler: Failed to create kernels." );
    }

    void compute_rho( size_t n, const real *coeffs, real *rho )
    {
        if ( begin == end ) return;

        size_t n_cards = kernels.size();
        size_t N = end - begin;
       // size_t testval = 2;
        size_t chunk_size = N /n_cards;

        //std::cout<<"begin: "<<begin<<"begin + chunksize: "<<begin+chunk_size<<"end: "<<end<<std::endl;
        for ( size_t i = 0; i < n_cards; ++i )
        {
            stopwatch<real> clock;
           // std::cout<<" next stop: compute rho, we are on "<<n_cards<<" cards, our index is "<<i<<std::endl;
            kernels[i].compute_rho( n, coeffs, begin + i*chunk_size, begin + (i+1)*chunk_size );
            

            //kernels[i].compute_rho( n, coeffs,begin, end);
            real elapsed = clock.elapsed();
            std::cout << "Time for launching kernel " << i << ": " << elapsed << "." << std::endl;
        }

        for ( size_t i = 0; i < n_cards; ++i )
        {
            stopwatch<real> clock;
            kernels[i].load_rho( rho, begin + i*chunk_size, begin + (i+1)*chunk_size ); 
            real elapsed = clock.elapsed();
            std::cout << "Waiting time for card " << i << ": " << elapsed << std::endl;
           // kernels[i].load_rho( rho, begin + i*chunk_size, begin + (i+1)*chunk_size ); 
           //kernels[i].load_rho( rho, begin ,end ); 
            //real elapsed = clock.elapsed();
            //std::cout << "Time for loading from kernel " << i << ": " << elapsed << "." << std::endl;

        }
        // Leftovers.
        kernels[0].compute_rho( n, coeffs, begin + n_cards*chunk_size, end );
        kernels[0].load_rho( rho, begin + n_cards*chunk_size, end );
    }

private:
    config_t<real> conf;
    size_t begin, end;
    std::vector< cuda_kernel<real,order> > kernels;
};

}

}

#endif
#endif

