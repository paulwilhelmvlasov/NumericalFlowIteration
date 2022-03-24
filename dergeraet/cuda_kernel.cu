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

#include <dergeraet/rho.hpp>

#include <stdexcept>

namespace dergeraet
{

namespace dim1
{

template <typename real, size_t order>
__global__
void cuda_eval_rho( size_t n, const real *coeffs, const config_t<real> conf, real *rho  )
{
    const size_t i     = blockDim.x*blockIdx.x + threadIdx.x;
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
cuda_kernel<real,order>::cuda_kernel( const config_t<real> &p_conf ): conf { p_conf }
{
    cudaError_t err = cudaMalloc( &cuda_coeffs, sizeof(real) * (conf.Nt + 1) *
                                                               (conf.Nx + order - 1) );
                                                 
    if ( err == cudaErrorMemoryAllocation ) throw std::bad_alloc {};

    err = cudaMalloc ( &cuda_rho, sizeof(real) * conf.Nx );
    if ( err == cudaErrorMemoryAllocation )
    {
        cudaFree( &cuda_coeffs );
        throw std::bad_alloc {};
    }
}

template <typename real, size_t order>
cuda_kernel<real,order>::~cuda_kernel()
{
    cudaFree( cuda_rho );
    cudaFree( cuda_coeffs );
}

template <typename real, size_t order>
void cuda_kernel<real,order>::compute_rho( size_t n, const real *coeffs, real *rho )
{
    if ( n > conf.Nt )
        throw std::range_error { "Time-step out of range." };

    if ( n )
    {
        size_t stride_n = (conf.Nx + order - 1);
        cudaMemcpy( cuda_coeffs + (n-1)*stride_n,
                         coeffs + (n-1)*stride_n, sizeof(real)*stride_n,
                    cudaMemcpyHostToDevice );
    }


    size_t N = conf.Nx;
    size_t block_size = 32;

    size_t Nblocks = N / block_size;
    size_t Ncpu    = N % block_size;
    size_t Ncuda   = N - Ncpu;
    
    cuda_eval_rho<real,order><<<Nblocks,block_size>>>( n, cuda_coeffs, conf, cuda_rho );

    #pragma omp parallel for
    for ( size_t l = Ncuda; l < N; ++l )
        rho[ l ] = eval_rho<real,order>( n, l, coeffs, conf );

    cudaMemcpy( rho, cuda_rho, Ncuda*sizeof(real), cudaMemcpyDeviceToHost );
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
void cuda_eval_rho( size_t n, const real *coeffs, const config_t<real> conf, real *rho  )
{
    const size_t l = blockDim.x*blockIdx.x + threadIdx.x;

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
cuda_kernel<real,order>::cuda_kernel( const config_t<real> &p_conf ): conf { p_conf }
{
    cudaError_t err = cudaMalloc( &cuda_coeffs, sizeof(real) * (conf.Nt + 1) *
                                                               (conf.Nx + order - 1) *
                                                               (conf.Ny + order - 1) );
    if ( err == cudaErrorMemoryAllocation ) throw std::bad_alloc {};

    err = cudaMalloc ( &cuda_rho, sizeof(real) * ( conf.Nx * conf.Ny ) );
    if ( err == cudaErrorMemoryAllocation )
    {
        cudaFree( &cuda_coeffs );
        throw std::bad_alloc {};
    }
}

template <typename real, size_t order>
cuda_kernel<real,order>::~cuda_kernel()
{
    cudaFree( cuda_rho );
    cudaFree( cuda_coeffs );
}

template <typename real, size_t order>
void cuda_kernel<real,order>::compute_rho( size_t n, const real *coeffs, real *rho )
{
    if ( n > conf.Nt )
        throw std::range_error { "Time-step out of range." };

    if ( n )
    {
        size_t stride_n = (conf.Nx + order - 1)*(conf.Ny + order - 1);
        cudaMemcpy( cuda_coeffs + (n-1)*stride_n,
                         coeffs + (n-1)*stride_n, sizeof(real)*stride_n,
                    cudaMemcpyHostToDevice );
    }


    size_t N = conf.Nx*conf.Ny;
    size_t block_size = 128;

    size_t Nblocks = N / block_size;
    size_t Ncpu    = N % block_size;
    size_t Ncuda   = N - Ncpu;
    
    cuda_eval_rho<real,order><<<Nblocks,block_size>>>( n, cuda_coeffs, conf, cuda_rho );

    #pragma omp parallel for
    for ( size_t l = Ncuda; l < N; ++l )
        rho[ l ] = eval_rho<real,order>( n, l, coeffs, conf );

    cudaMemcpy( rho, cuda_rho, Ncuda*sizeof(real), cudaMemcpyDeviceToHost );
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

