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
#ifndef DERGERAET_DEVICE_SCHEDULER_HPP
#define DERGERAET_DEVICE_SCHEDULER_HPP

#include <cmath>
#include <vector>
#include <stdexcept>
#include <sycl/sycl.hpp>

#include <dergeraet/stopwatch.hpp>
#include <dergeraet/device_kernel.hpp>

namespace dergeraet
{

namespace dim1
{

template <typename real, size_t order>
class device_scheduler
{
public:
    device_scheduler() = delete;
    device_scheduler( const device_scheduler&  ) = delete;
    device_scheduler(       device_scheduler&& ) = default;
    device_scheduler& operator=( const device_scheduler&  ) = delete;
    device_scheduler& operator=(       device_scheduler&& ) = default;

    device_scheduler( const config_t<real> &p_conf, std::vector<sycl::device> devs ):
    conf { p_conf }
    {
        if ( devs.size() == 0 )
            throw std::runtime_error { "device_scheduler: Empty list of devices."  };

        for ( size_t i = 0; i < devs.size(); ++i )
            kernels.emplace_back( device_kernel<real,order>(conf,devs[i]) );
    }

    void compute_rho( size_t n, size_t q_begin, size_t q_end )
    {
        if ( q_begin == q_end ) return;

        size_t n_cards = kernels.size();
        size_t N = q_end - q_begin;
        size_t chunk_size = N / n_cards;
        size_t remainder  = N % n_cards;

        size_t current = q_begin;
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

    void download_rho( real *rho )
    {
        size_t n_cards = kernels.size();
        for ( size_t i = 0; i < n_cards; ++i )
            kernels[ i ].download_rho( rho );
    }

    void upload_phi( size_t n, const real *coeffs ) 
    {
        size_t n_cards = kernels.size();
        for ( size_t i = 0; i < n_cards; ++i )
            kernels[i].upload_phi(n,coeffs);
    }


    void compute_metrics( size_t n, size_t q_begin, size_t q_end )
    {
        if ( q_begin == q_end ) return;

        size_t n_cards = kernels.size();
        size_t N = q_end - q_begin;
        size_t chunk_size = N / n_cards;
        size_t remainder  = N % n_cards;

        size_t current = q_begin;
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
        for ( size_t i = 0; i < kernels.size(); ++i )
            kernels[i].download_metrics(metrics);
    }

private:
    config_t<real> conf;

    std::vector< device_kernel<real,order> > kernels;
};

}

namespace dim2
{

template <typename real, size_t order>
class device_scheduler
{
public:
    device_scheduler() = delete;
    device_scheduler( const device_scheduler&  ) = delete;
    device_scheduler(       device_scheduler&& ) = default;
    device_scheduler& operator=( const device_scheduler&  ) = delete;
    device_scheduler& operator=(       device_scheduler&& ) = default;

    device_scheduler( const config_t<real> &p_conf, std::vector<sycl::device> devs ):
    conf { p_conf }
    {
        if ( devs.size() == 0 )
            throw std::runtime_error { "device_scheduler: Empty list of devices."  };

        for ( size_t i = 0; i < devs.size(); ++i )
            kernels.emplace_back( device_kernel<real,order>(conf,devs[i]) );
    }

    void compute_rho( size_t n, size_t q_begin, size_t q_end )
    {
        if ( q_begin == q_end ) return;

        size_t n_cards = kernels.size();
        size_t N = q_end - q_begin;
        size_t chunk_size = N / n_cards;
        size_t remainder  = N % n_cards;

        size_t current = q_begin;
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

    void download_rho( real *rho )
    {
        size_t n_cards = kernels.size();
        for ( size_t i = 0; i < n_cards; ++i )
            kernels[ i ].download_rho( rho );
    }

    void upload_phi( size_t n, const real *coeffs ) 
    {
        size_t n_cards = kernels.size();
        for ( size_t i = 0; i < n_cards; ++i )
            kernels[i].upload_phi(n,coeffs);
    }


    void compute_metrics( size_t n, size_t q_begin, size_t q_end )
    {
        if ( q_begin == q_end ) return;

        size_t n_cards = kernels.size();
        size_t N = q_end - q_begin;
        size_t chunk_size = N / n_cards;
        size_t remainder  = N % n_cards;

        size_t current = q_begin;
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
        for ( size_t i = 0; i < kernels.size(); ++i )
            kernels[i].download_metrics(metrics);
    }

private:
    config_t<real> conf;

    std::vector< device_kernel<real,order> > kernels;
};

}

namespace dim3
{

template <typename real, size_t order>
class device_scheduler
{
public:
    device_scheduler() = delete;
    device_scheduler( const device_scheduler&  ) = delete;
    device_scheduler(       device_scheduler&& ) = default;
    device_scheduler& operator=( const device_scheduler&  ) = delete;
    device_scheduler& operator=(       device_scheduler&& ) = default;

    device_scheduler( const config_t<real> &p_conf, std::vector<sycl::device> devs ):
    conf { p_conf }
    {
        if ( devs.size() == 0 )
            throw std::runtime_error { "device_scheduler: Empty list of devices."  };

        for ( size_t i = 0; i < devs.size(); ++i )
            kernels.emplace_back( device_kernel<real,order>(conf,devs[i]) );
    }

    void compute_rho( size_t n, size_t q_begin, size_t q_end )
    {
        if ( q_begin == q_end ) return;

        size_t n_cards = kernels.size();
        size_t N = q_end - q_begin;
        size_t chunk_size = N / n_cards;
        size_t remainder  = N % n_cards;

        size_t current = q_begin;
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

    void download_rho( real *rho )
    {
        size_t n_cards = kernels.size();
        for ( size_t i = 0; i < n_cards; ++i )
            kernels[ i ].download_rho( rho );
    }

    void upload_phi( size_t n, const real *coeffs ) 
    {
        size_t n_cards = kernels.size();
        for ( size_t i = 0; i < n_cards; ++i )
            kernels[i].upload_phi(n,coeffs);
    }


    void compute_metrics( size_t n, size_t q_begin, size_t q_end )
    {
        if ( q_begin == q_end ) return;

        size_t n_cards = kernels.size();
        size_t N = q_end - q_begin;
        size_t chunk_size = N / n_cards;
        size_t remainder  = N % n_cards;

        size_t current = q_begin;
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
        for ( size_t i = 0; i < kernels.size(); ++i )
            kernels[i].download_metrics(metrics);
    }

private:
    config_t<real> conf;

    std::vector< device_kernel<real,order> > kernels;
};

}

}

#endif

