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
#include <dergeraet/cuda_kernel.hpp>
#include <dergeraet/stopwatch.hpp>
#include <dergeraet/rho.hpp>

#include <stdexcept>

namespace dergeraet
{

namespace dim1
{

template <typename real, size_t order>
__global__
void cuda_eval_rho( size_t n, const real *coeffs, const config_t<real> conf, real *rho,
                    size_t l_min, size_t l_end )
{
  
    const size_t N     = l_end - l_min;
    const size_t i     = l_min + (blockDim.x*blockIdx.x + threadIdx.x) % N;
    const real   x     = conf.x_min + i*conf.dx; 
    const real   du    = (conf.u_max - conf.u_min) / conf.Nu;
    const real   u_min = conf.u_min + 0.5*du;

    real result = 0;
    for ( size_t ii = 0; ii < conf.Nu; ++ii )
    {
        real u = u_min + ii*du;
        result += eval_ftilda<real,order>( n, x, u, coeffs, conf );
    }
    result = 1 - du*result;

    rho[ i ] = result;
}

template <typename real, size_t order>
cuda_kernel<real,order>::cuda_kernel( const config_t<real> &p_conf, int dev ):
conf { p_conf }, device_number { dev }
{
    size_t coeff_size = sizeof(real)*(conf.Nt+1)*(conf.Nx+order-1);
    size_t   rho_size = sizeof(real)*(conf.Nx);

    
    cuda::set_device( device_number );
    cuda_coeffs.reset( cuda::malloc(coeff_size) );
    cuda_rho   .reset( cuda::malloc(  rho_size) );
}

template <typename real, size_t order>
void cuda_kernel<real,order>::compute_rho( size_t n, const real *coeffs, size_t l_min, size_t l_end )
{

    if ( l_min == l_end ) return;

    if ( n > conf.Nt )
        throw std::range_error { "Time-step out of range." };

    cuda::set_device( device_number );
    real *cu_coeffs = reinterpret_cast<real*>( cuda_coeffs.get() );
    real *cu_rho    = reinterpret_cast<real*>( cuda_rho   .get() );

    if ( n )
    {
        size_t stride_n = (conf.Nx + order - 1);
        cuda::memcpy_to_device( cu_coeffs + (n-1)*stride_n,
                                   coeffs + (n-1)*stride_n,
                                sizeof(real)*stride_n );
    }

    size_t N = conf.Nx;
    size_t block_size = 64;
    size_t Nblocks = 1 +  ( (N-1) / (block_size) );
    stopwatch<real> clock;

    cuda_eval_rho<real,order><<<Nblocks,block_size>>>( n, cu_coeffs, conf, cu_rho, l_min, l_end );
    real elapsed = clock.elapsed();
    std::cout << "Time for computing rho on kernel: " << device_number << ": " << elapsed << "." << std::endl;
}

template <typename real, size_t order>
void cuda_kernel<real,order>::load_rho( real *rho, size_t l_min, size_t l_end )
{
    if ( l_min == l_end ) return;

    size_t N = l_end - l_min;
    cuda::set_device( device_number );
    real *cu_rho    = reinterpret_cast<real*>( cuda_rho.get() );
    cuda::memcpy_to_host(rho + l_min, cu_rho + l_min, N*sizeof(real) );
}

template class cuda_kernel<double,3>;
template class cuda_kernel<double,4>;
template class cuda_kernel<double,5>;
template class cuda_kernel<double,6>;
template class cuda_kernel<double,7>;
template class cuda_kernel<double,8>;

template class cuda_kernel<float,3>;
template class cuda_kernel<float,4>;
template class cuda_kernel<float,5>;
template class cuda_kernel<float,6>;
template class cuda_kernel<float,7>;
template class cuda_kernel<float,8>;

}

namespace dim2
{

template <typename real, size_t order>
__global__
void cuda_eval_rho( size_t n, const real *coeffs, const config_t<real> conf, real *rho,
                    size_t l_min, size_t l_end )
{
    const size_t N = l_end - l_min;
    const size_t l = l_min + (blockDim.x*blockIdx.x + threadIdx.x) % N;

    const size_t i = l % conf.Nx;
    const size_t j = l / conf.Nx;

    const real   x = conf.x_min + i*conf.dx; 
    const real   y = conf.y_min + j*conf.dy; 

    const real du = (conf.u_max - conf.u_min) / conf.Nu;
    const real dv = (conf.v_max - conf.v_min) / conf.Nv;

    const real u_min = conf.u_min + 0.5*du;
    const real v_min = conf.v_min + 0.5*dv;

    real result = 0;
    for ( size_t jj = 0; jj < conf.Nv; ++jj )
    for ( size_t ii = 0; ii < conf.Nu; ++ii )
    {
        real u = u_min + ii*du;
        real v = v_min + jj*dv;

        result += eval_ftilda<real,order>( n, x, y, u, v, coeffs, conf );
    }
    result = 1 - du*dv*result;

    rho[ l ] = result;
}

template <typename real, size_t order>
cuda_kernel<real,order>::cuda_kernel( const config_t<real> &p_conf, int dev ):
conf { p_conf }, device_number { dev }
{
    size_t coeff_size = sizeof(real)*(conf.Nt+1)*(conf.Nx+order-1)*
                                                 (conf.Ny+order-1);
    size_t   rho_size = sizeof(real)*conf.Nx*conf.Ny;

    cuda::set_device(dev);
    cuda_coeffs.reset( cuda::malloc(coeff_size), dev );
    cuda_rho   .reset( cuda::malloc(  rho_size), dev );
}

template <typename real, size_t order>
void cuda_kernel<real,order>::compute_rho( size_t n, const real *coeffs,
                                           size_t l_min, size_t l_end )
{

    stopwatch<real> clock;
    if ( n > conf.Nt )
        throw std::range_error { "Time-step out of range." };

    if ( l_min == l_end ) return;

    cuda::set_device(device_number);
    real *cu_coeffs = reinterpret_cast<real*>( cuda_coeffs.get() );
    real *cu_rho    = reinterpret_cast<real*>( cuda_rho   .get() );
    if ( n )
    {
        size_t stride_n = (conf.Nx + order - 1)*(conf.Ny + order - 1);
        cuda::memcpy_to_device( cu_coeffs + (n-1)*stride_n,
                                  coeffs + (n-1)*stride_n, sizeof(real)*stride_n );
    }
    size_t N = l_end - l_min;
    size_t block_size = 64;
    size_t Nblocks = 1 + (N-1) / block_size;
    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);
    // cudaEventRecord(start);
    cuda_eval_rho<real,order><<<Nblocks,block_size>>>( n, cu_coeffs, conf, cu_rho, l_min, l_end );
    //cudaDeviceSynchronize();
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    //cudaDeviceSynchronize();
    //real elapsed = clock.elapsed();
}

template <typename real, size_t order>
void cuda_kernel<real,order>::load_rho(real *rho, size_t l_min, size_t l_end )
{
    if ( l_min == l_end ) return;

    cuda::set_device(device_number);
    real *cu_rho = reinterpret_cast<real*>( cuda_rho.get() );

    size_t N = l_end - l_min;
    cuda::memcpy_to_host( rho + l_min, cu_rho + l_min, N*sizeof(real) );
}


template class cuda_kernel<double,3>;
template class cuda_kernel<double,4>;
template class cuda_kernel<double,5>;
template class cuda_kernel<double,6>;
template class cuda_kernel<double,7>;
template class cuda_kernel<double,8>;

template class cuda_kernel<float,3>;
template class cuda_kernel<float,4>;
template class cuda_kernel<float,5>;
template class cuda_kernel<float,6>;
template class cuda_kernel<float,7>;
template class cuda_kernel<float,8>;

}

}
