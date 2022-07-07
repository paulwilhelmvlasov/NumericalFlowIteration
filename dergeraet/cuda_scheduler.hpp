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

    cuda_scheduler( const config_t<real> &p_conf ):
    conf { p_conf }
    {
        size_t n_dev = cuda::device_count();
        kernels.reserve(n_dev);
        cuda_kernel<real,order> *k;

        for ( size_t i = 0; i < n_dev; ++i )
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
            throw cuda::exception( cudaErrorUnknown, "cuda_scheduler: could not create kernels on any devices." );
    }

    void compute_rho( size_t n, const real *coeffs, real *rho )
    {
        using std::min;

        size_t n_cards = kernels.size();
        size_t N = conf.Nx;
        size_t chunk_size = N / n_cards;

        for ( size_t i = 0; i < n_cards; ++i )
            kernels[i].compute_rho( n, coeffs, i*chunk_size, min(N,(i+1)*chunk_size) ); 

        for ( size_t i = 0; i < n_cards; ++i )
            kernels[i].load_rho( rho, i*chunk_size, min(N,(i+1)*chunk_size) ); 
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
        cuda_kernel<real,order> *k;

        for ( size_t i = 0; i < n_dev; ++i )
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
    }

    void compute_rho( size_t n, const real *coeffs, real *rho )
    {
        using std::min;

        size_t n_cards = kernels.size();
        size_t N = conf.Nx*conf.Ny;
        size_t chunk_size = N / n_cards;

        for ( size_t i = 0; i < n_cards; ++i )
            kernels[i].compute_rho( n, coeffs, i*chunk_size, min(N,(i+1)*chunk_size) ); 

        for ( size_t i = 0; i < n_cards; ++i )
            kernels[i].load_rho( rho, i*chunk_size, min(N,(i+1)*chunk_size) ); 
    }

private:
    config_t<real> conf;
    std::vector< cuda_kernel<real,order> > kernels;
};

}

}

#endif
#endif


