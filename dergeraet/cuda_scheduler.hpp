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

        for ( size_t i = 0; i < n_dev; i++ )
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

        // Remainder 
        kernels.back().compute_rho( n, coeffs, begin + (n_cards-1)*chunk_size, end);

        for ( size_t i = 0; i < n_cards - 1; ++i )
            kernels[i].load_rho( rho, begin + i*chunk_size, begin + (i+1)*chunk_size ); 

        // Remainder
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
       
        for ( size_t i = 0; i < n_dev; i++)
        {
            try 
            {
                kernels.emplace_back( cuda_kernel<real,order>(conf,i) );
	        std::cout << "Added device " << i << ".\n"; std::cout.flush();
            }
            catch ( cuda::exception &ex )
            {
                // Do not use this device.
            }
        }

        if ( kernels.size() == 0 )
            throw cuda::exception( cudaErrorUnknown, "cuda_scheduler: Failed to create kernels." );
    }

    void compute_rho( size_t n, const real *coeffs, real *rho )
    {
        if ( begin == end ) return;

        size_t n_cards = kernels.size();
        size_t N = end - begin;
        size_t chunk_size = N / n_cards;
        size_t remainder  = N % n_cards;

        // Launch tasks.
        size_t curr = begin;
        for ( size_t i = 0; i < n_cards; ++i )
        { 
            if ( i < remainder )
            {  
                kernels[i].compute_rho( n, coeffs, curr, curr + chunk_size + 1 );
                curr += chunk_size + 1;
            }
            else
            {
                kernels[i].compute_rho( n, coeffs, curr, curr + chunk_size );
                curr += chunk_size;
            }
        }

        // Load results.
        curr = begin;
        for ( size_t i = 0; i < n_cards; ++i )
        {
            if ( i < remainder )
            {
                kernels[i].load_rho( rho, curr, curr + chunk_size + 1 ); 
                curr += chunk_size + 1;
            }
            else
            {
                kernels[i].load_rho( rho, curr, curr + chunk_size ); 
                curr += chunk_size;
            }
        }
    }

    // compute_rho version which also saves the values of f for further use (like
    // computing entropy).

    void compute_rho( size_t n, const real *coeffs, real *rho, real *f_values )
    {
        if ( begin == end ) return;

        size_t n_cards = kernels.size();
        size_t N = end - begin;
        size_t chunk_size = N / n_cards;
        size_t remainder  = N % n_cards;

        real l1_norm_f = 0;
        real *l1_norm_array = new real[n_cards];

        // Launch tasks.
        size_t curr = begin;
        for ( size_t i = 0; i < n_cards; ++i )
        {
            if ( i < remainder )
            {
                kernels[i].compute_rho( n, coeffs, curr, curr + chunk_size + 1, l1_norm_array[i] );
                curr += chunk_size + 1;
            }
            else
            {
                kernels[i].compute_rho( n, coeffs, curr, curr + chunk_size, l1_norm_array[i] );
                curr += chunk_size;
            }
        }

        for( size_t i = 0; i < n_cards; ++i )
        {
        	l1_norm_f += l1_norm_array[i];
        }

        real dv = (conf.v_max - conf.v_min) / conf.Nv;
        real du = (conf.u_max - conf.u_min) / conf.Nu;
        l1_norm_f *= conf.dx*conf.dy*dv*du;
        std::cout << n << " " << l1_norm_f << std::endl;

        // Load results
        curr = begin;
        for ( size_t i = 0; i < n_cards; ++i )
        {
            if ( i < remainder )
            {
                kernels[i].load_rho( rho, curr, curr + chunk_size + 1 );
                curr += chunk_size + 1;
            }
            else
            {
                kernels[i].load_rho( rho, curr, curr + chunk_size );
                curr += chunk_size;
            }
        }
    }


private:
    config_t<real> conf;
    size_t begin, end;
    std::vector< cuda_kernel<real,order> > kernels;
};

}

}

#endif

