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

    cuda_scheduler( const config_t<real> &p_conf ):
    conf { p_conf }
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

    void compute_rho( size_t n, size_t begin, size_t end )
    {
        if ( begin == end ) return;

        size_t n_cards = kernels.size();
        size_t N = end - begin;
        size_t chunk_size = N / n_cards;
        size_t remainder  = N % n_cards;

        size_t current = begin;
        for ( size_t i = 0; i < n_cards; ++i )
        {
            if ( i < remainder )
            {
                kernels[i].compute_rho( n, current, current + chunk_size + 1 );
                current = current + chunk_size + 1;
            }
            else
            {
                kernels[i].compute_rho( n, current, current + chunk_size );
                current = current + chunk_size;
            }
        }
    }

    void download_rho( real *rho, size_t begin, size_t end )
    {
        if ( begin == end ) return;

        size_t n_cards = kernels.size();
        size_t N = end - begin;
        size_t chunk_size = N / n_cards;
        size_t remainder  = N % n_cards;

        size_t current = begin;
        for ( size_t i = 0; i < n_cards; ++i )
        {
            if ( i < remainder )
            {
                kernels[i].download_rho( rho, current, current + chunk_size + 1 );
                current = current + chunk_size + 1;
            }
            else
            {
                kernels[i].download_rho( rho, current, current + chunk_size );
                current = current + chunk_size;
            }
        }
    }

    void upload_phi( size_t n, const real *coeffs ) 
    {
        size_t n_cards = kernels.size();

        for ( size_t i = 0; i < n_cards; ++i )
            kernels[i].upload_phi(n,coeffs);
    }


    void compute_metrics( size_t n, size_t begin, size_t end )
    {
        if ( begin == end ) return;

        size_t n_cards = kernels.size();
        size_t N = end - begin;
        size_t chunk_size = N / n_cards;
        size_t remainder  = N % n_cards;

        size_t current = begin;
        for ( size_t i = 0; i < n_cards; ++i )
        {
            if ( i < remainder )
            {
                kernels[i].compute_metrics( n, current, current + chunk_size + 1 );
                current = current + chunk_size + 1;
            }
            else
            {
                kernels[i].compute_metrics( n, current, current + chunk_size );
                current = current + chunk_size;
            }
        }
    }

    void download_metrics( real *metrics )
    {
        real tmp[ 4 ];

        metrics[ 0 ] = 0;
        metrics[ 1 ] = 0;
        metrics[ 2 ] = 0;
        metrics[ 3 ] = 0;

        for ( size_t i = 0; i < kernels.size(); ++i )
        {
            kernels[i].download_metrics(tmp);
            metrics[ 0 ] += tmp[ 0 ];
            metrics[ 1 ] += tmp[ 1 ];
            metrics[ 2 ] += tmp[ 2 ];
            metrics[ 3 ] += tmp[ 3 ];
        }
    }

private:
    config_t<real> conf;

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

    cuda_scheduler( const config_t<real> &p_conf ):
    conf { p_conf }
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

    void compute_rho( size_t n, size_t begin, size_t end )
    {
        if ( begin == end ) return;

        size_t n_cards = kernels.size();
        size_t N = end - begin;
        size_t chunk_size = N / n_cards;
        size_t remainder  = N % n_cards;

        size_t current = begin;
        for ( size_t i = 0; i < n_cards; ++i )
        {
            if ( i < remainder )
            {
                kernels[i].compute_rho( n, current, current + chunk_size + 1 );
                current = current + chunk_size + 1;
            }
            else
            {
                kernels[i].compute_rho( n, current, current + chunk_size );
                current = current + chunk_size;
            }
        }
    }

    void download_rho( real *rho, size_t begin, size_t end )
    {
        if ( begin == end ) return;

        size_t n_cards = kernels.size();
        size_t N = end - begin;
        size_t chunk_size = N / n_cards;
        size_t remainder  = N % n_cards;

        size_t current = begin;
        for ( size_t i = 0; i < n_cards; ++i )
        {
            if ( i < remainder )
            {
                kernels[i].download_rho( rho, current, current + chunk_size + 1 );
                current = current + chunk_size + 1;
            }
            else
            {
                kernels[i].download_rho( rho, current, current + chunk_size );
                current = current + chunk_size;
            }
        }
    }

    void upload_phi( size_t n, const real *coeffs ) 
    {
        size_t n_cards = kernels.size();
        for ( size_t i = 0; i < n_cards; ++i )
            kernels[i].upload_phi(n,coeffs);
    }


    void compute_metrics( size_t n, size_t begin, size_t end )
    {
        if ( begin == end ) return;

        size_t n_cards = kernels.size();
        size_t N = end - begin;
        size_t chunk_size = N / n_cards;
        size_t remainder  = N % n_cards;

        size_t current = begin;
        for ( size_t i = 0; i < n_cards; ++i )
        {
            if ( i < remainder )
            {
                kernels[i].compute_metrics( n, current, current + chunk_size + 1 );
                current = current + chunk_size + 1;
            }
            else
            {
                kernels[i].compute_metrics( n, current, current + chunk_size );
                current = current + chunk_size;
            }
        }
    }

    void download_metrics( real *metrics )
    {
        real tmp[ 4 ];

        metrics[ 0 ] = 0;
        metrics[ 1 ] = 0;
        metrics[ 2 ] = 0;
        metrics[ 3 ] = 0;

        for ( size_t i = 0; i < kernels.size(); ++i )
        {
            kernels[i].download_metrics(tmp);
            metrics[ 0 ] += tmp[ 0 ];
            metrics[ 1 ] += tmp[ 1 ];
            metrics[ 2 ] += tmp[ 2 ];
            metrics[ 3 ] += tmp[ 3 ];
        }
    }

private:
    config_t<real> conf;

    std::vector< cuda_kernel<real,order> > kernels;
};

}

namespace dim3
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

    cuda_scheduler( const config_t<real> &p_conf ):
    conf { p_conf }
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

    void compute_rho( size_t n, size_t begin, size_t end )
    {
        if ( begin == end ) return;

        size_t n_cards = kernels.size();
        size_t N = end - begin;
        size_t chunk_size = N / n_cards;
        size_t remainder  = N % n_cards;

        size_t current = begin;
        for ( size_t i = 0; i < n_cards; ++i )
        {
            if ( i < remainder )
            {
                kernels[i].compute_rho( n, current, current + chunk_size + 1 );
                current = current + chunk_size + 1;
            }
            else
            {
                kernels[i].compute_rho( n, current, current + chunk_size );
                current = current + chunk_size;
            }
        }
    }

    void download_rho( real *rho, size_t begin, size_t end )
    {
        if ( begin == end ) return;

        size_t n_cards = kernels.size();
        size_t N = end - begin;
        size_t chunk_size = N / n_cards;
        size_t remainder  = N % n_cards;

        size_t current = begin;
        for ( size_t i = 0; i < n_cards; ++i )
        {
            if ( i < remainder )
            {
                kernels[i].download_rho( rho, current, current + chunk_size + 1 );
                current = current + chunk_size + 1;
            }
            else
            {
                kernels[i].download_rho( rho, current, current + chunk_size );
                current = current + chunk_size;
            }
        }
    }

    void upload_phi( size_t n, const real *coeffs ) 
    {
        size_t n_cards = kernels.size();

        for ( size_t i = 0; i < n_cards; ++i )
            kernels[i].upload_phi(n,coeffs);
    }

    void compute_metrics( size_t n, size_t begin, size_t end )
    {
        if ( begin == end ) return;

        size_t n_cards = kernels.size();
        size_t N = end - begin;
        size_t chunk_size = N / n_cards;
        size_t remainder  = N % n_cards;

        size_t current = begin;
        for ( size_t i = 0; i < n_cards; ++i )
        {
            if ( i < remainder )
            {
                kernels[i].compute_metrics( n, current, current + chunk_size + 1 );
                current = current + chunk_size + 1;
            }
            else
            {
                kernels[i].compute_metrics( n, current, current + chunk_size );
                current = current + chunk_size;
            }
        }
    }

    void download_metrics( real *metrics )
    {
        real tmp[ 4 ];

        metrics[ 0 ] = 0;
        metrics[ 1 ] = 0;
        metrics[ 2 ] = 0;
        metrics[ 3 ] = 0;

        for ( size_t i = 0; i < kernels.size(); ++i )
        {
            kernels[i].download_metrics(tmp);
            metrics[ 0 ] += tmp[ 0 ];
            metrics[ 1 ] += tmp[ 1 ];
            metrics[ 2 ] += tmp[ 2 ];
            metrics[ 3 ] += tmp[ 3 ];
        }
    }

private:
    config_t<real> conf;

    std::vector< cuda_kernel<real,order> > kernels;
};

}

}

#endif

