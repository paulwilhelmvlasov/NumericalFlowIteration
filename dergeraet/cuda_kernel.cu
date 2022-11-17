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
void cuda_eval_rho( size_t n, const real *coeffs, const config_t<real> conf, real *rho,
                    size_t q_begin, size_t q_end )
{
    // Number of my quadrature node.
    const size_t q = q_begin +
                     size_t(blockDim.x)*size_t(blockIdx.x) + size_t(threadIdx.x);

    const size_t ix = q / conf.Nu;
    const size_t iu = q % conf.Nu;
    
    const real   x = conf.x_min + ix*conf.dx; 
    const real   u = conf.u_min + iu*conf.du + conf.du/2;
    real *my_rho = rho + ix;

    const real f = eval_ftilda<real,order>( n, x, u, coeffs, conf );
    const real weight = conf.du;

    if ( q < q_end ) atomicAdd( my_rho, -weight*f );  
}

template <typename real, size_t order>
__global__
void cuda_eval_metrics( size_t n, const real *coeffs, const config_t<real> conf, real *metrics,
                        size_t q_begin, size_t q_end )
{
    // Number of my quadrature node.
    const size_t q = q_begin +
                     size_t(blockDim.x)*size_t(blockIdx.x) + size_t(threadIdx.x);

    const size_t ix = q / conf.Nu;
    const size_t iu = q % conf.Nu;
    
    const real x = conf.x_min + ix*conf.dx; 
    const real u = conf.u_min + iu*conf.du + conf.du/2;

    const real f = eval_f<real,order>( n, x, u, coeffs, conf );
    const real weight = conf.du*conf.dx;

    if ( q < q_end )
    {
        atomicAdd( metrics + 0, weight*f );
        atomicAdd( metrics + 1, weight*f*f );
        atomicAdd( metrics + 2, weight*(u*u*f/2) );
        atomicAdd( metrics + 3, (f>0) ? -weight*f*log(f) : 0 );
    }
}

template <typename real, size_t order>
cuda_kernel<real,order>::cuda_kernel( const config_t<real> &p_conf, int dev ):
conf { p_conf }, device_number { dev } 
{
    size_t  coeff_size  = sizeof(real)*(conf.Nt+1)*(conf.Nx+order-1);
    size_t     rho_size = sizeof(real)*conf.Nx;
    size_t metrics_size = sizeof(real)*4;
    
    cuda::set_device( device_number );
    cuda_coeffs .reset( cuda::malloc(  coeff_size), dev );
    cuda_rho    .reset( cuda::malloc(    rho_size), dev );
    cuda_metrics.reset( cuda::malloc(metrics_size), dev );
    tmp_rho.reset( new real[ conf.Nx ] );
}

template <typename real, size_t order>
void cuda_kernel<real,order>::compute_rho( size_t n, size_t q_begin, size_t q_end )
{
    if ( n > conf.Nt )
        throw std::range_error { "Time-step out of range." };

    if ( q_begin == q_end ) return;

    real *cu_coeffs = reinterpret_cast<real*>( cuda_coeffs.get() );
    real *cu_rho    = reinterpret_cast<real*>( cuda_rho   .get() );

    size_t N          = q_end - q_begin;
    size_t block_size = 64;
    size_t Nblocks    = 1 + ( (N-1) / (block_size) );

    size_t rho_size = conf.Nx*sizeof(real);

    cuda::set_device( device_number );
    cuda::memset( cu_rho, 0, rho_size );
    cuda_eval_rho<real,order><<<Nblocks,block_size>>>( n, cu_coeffs, conf, cu_rho, q_begin, q_end );
}

template <typename real, size_t order>
void cuda_kernel<real,order>::download_rho( real *rho )
{
    real *cu_rho    = reinterpret_cast<real*>( cuda_rho.get() );
    size_t N        = conf.Nx;

    cuda::set_device(device_number);
    cuda::memcpy_to_host( tmp_rho.get(), cu_rho, sizeof(real)*N );

    for ( size_t i = 0; i < N; ++i )
        rho[ i ] += tmp_rho[ i ];
}

template <typename real, size_t order>
void cuda_kernel<real,order>::upload_phi( size_t n, const real *coeffs )
{
    size_t stride_n = (conf.Nx + order - 1);
    real *cu_coeffs = reinterpret_cast<real*>( cuda_coeffs.get() );

    cuda::set_device(device_number);
    cuda::memcpy_to_device( cu_coeffs + n*stride_n,
                               coeffs + n*stride_n, sizeof(real)*stride_n );
}

template <typename real, size_t order>
void cuda_kernel<real,order>::compute_metrics( size_t n, size_t q_begin, size_t q_end )
{
    if ( q_begin == q_end ) return;

    real *cu_coeffs  = reinterpret_cast<real*>( cuda_coeffs .get() );
    real *cu_metrics = reinterpret_cast<real*>( cuda_metrics.get() );

    size_t N = q_end - q_begin;
    size_t block_size = 64;
    size_t Nblocks = 1 + (N-1) / block_size;

    cuda::set_device(device_number);
    cuda::memset( cu_metrics, 0, 4*sizeof(real) );
    cuda_eval_metrics<real,order><<<Nblocks,block_size>>>( n, cu_coeffs, conf, cu_metrics, q_begin, q_end );
}

template <typename real, size_t order>
void cuda_kernel<real,order>::download_metrics( real *metrics )
{
    real *cu_metrics = reinterpret_cast<real*>( cuda_metrics.get() );
    real tmp_metrics[ 4 ];

    cuda::set_device(device_number);
    cuda::memcpy_to_host( tmp_metrics, cu_metrics, 4*sizeof(real) );

    metrics[ 0 ] += tmp_metrics[ 0 ];
    metrics[ 1 ] += tmp_metrics[ 1 ];
    metrics[ 2 ] += tmp_metrics[ 2 ];
    metrics[ 3 ] += tmp_metrics[ 3 ];
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
                    size_t q_begin, size_t q_end )
{
    // Number of my quadrature node.
    const size_t q = q_begin +
                     size_t(blockDim.x)*size_t(blockIdx.x) + size_t(threadIdx.x);

    size_t tmp = q;
    const size_t iy  = tmp / ( conf.Nx*conf.Nv*conf.Nu );
                 tmp = tmp % ( conf.Nx*conf.Nv*conf.Nu );
    const size_t ix  = tmp / ( conf.Nv*conf.Nu );
                 tmp = tmp % ( conf.Nv*conf.Nu );
    const size_t iv  = tmp / ( conf.Nu ); 
    const size_t iu  = tmp % ( conf.Nu );
    
    const real x = conf.x_min + ix*conf.dx; 
    const real y = conf.y_min + iy*conf.dy; 
    const real u = conf.u_min + iu*conf.du + conf.du/2;
    const real v = conf.v_min + iv*conf.dv + conf.dv/2;
    real *my_rho = rho + iy*conf.Nx + ix;

    const real f = eval_ftilda<real,order>( n, x, y, u, v, coeffs, conf );
    const real weight = conf.du*conf.dv;

    if ( q < q_end ) atomicAdd( my_rho, -weight*f );  
}

template <typename real, size_t order>
__global__
void cuda_eval_metrics( size_t n, const real *coeffs, const config_t<real> conf, real *metrics,
                        size_t q_begin, size_t q_end )
{
    // Number of my quadrature node.
    const size_t q = q_begin +
                     size_t(blockDim.x)*size_t(blockIdx.x) + size_t(threadIdx.x);

    size_t tmp = q;
    const size_t iy  = tmp / ( conf.Nx*conf.Nv*conf.Nu );
                 tmp = tmp % ( conf.Nx*conf.Nv*conf.Nu );
    const size_t ix  = tmp / ( conf.Nv*conf.Nu );
                 tmp = tmp % ( conf.Nv*conf.Nu );
    const size_t iv  = tmp / ( conf.Nu ); 
    const size_t iu  = tmp % ( conf.Nu );
    
    const real x = conf.x_min + ix*conf.dx; 
    const real y = conf.y_min + iy*conf.dy; 
    const real u = conf.u_min + iu*conf.du + conf.du/2;
    const real v = conf.v_min + iv*conf.dv + conf.dv/2;

    const real f = eval_f<real,order>( n, x, y, u, v, coeffs, conf );
    const real weight = conf.dx*conf.dy*conf.du*conf.dv;

    if ( q < q_end )
    {
        atomicAdd( metrics + 0, weight*f );
        atomicAdd( metrics + 1, weight*f*f );
        atomicAdd( metrics + 2, weight*(u*u+v*v)*f/2 );
        atomicAdd( metrics + 3, (f>0) ? -weight*f*log(f) : 0 );
    }
}

template <typename real, size_t order>
cuda_kernel<real,order>::cuda_kernel( const config_t<real> &p_conf, int dev ):
conf { p_conf }, device_number { dev }
{
    size_t   coeff_size = sizeof(real)*(conf.Nt+1)*(conf.Nx+order-1)*
                                                   (conf.Ny+order-1);
    size_t     rho_size = sizeof(real)*conf.Nx*conf.Ny;
    size_t metrics_size = sizeof(real)*4;

    cuda::set_device(dev);
    cuda_coeffs .reset( cuda::malloc(  coeff_size), dev );
    cuda_rho    .reset( cuda::malloc(    rho_size), dev );
    cuda_metrics.reset( cuda::malloc(metrics_size), dev );
    tmp_rho.reset( new real[ conf.Nx*conf.Ny ] );
}

template <typename real, size_t order>
void cuda_kernel<real,order>::compute_rho( size_t n, size_t q_begin, size_t q_end )
{
    if ( n > conf.Nt )
        throw std::range_error { "Time-step out of range." };

    if ( q_begin == q_end ) return;

    real *cu_coeffs = reinterpret_cast<real*>( cuda_coeffs.get() );
    real *cu_rho    = reinterpret_cast<real*>( cuda_rho   .get() );

    size_t N = q_end - q_begin;
    size_t block_size = 64;
    size_t Nblocks = 1 + (N-1) / block_size;

    size_t rho_size = conf.Nx*conf.Ny*sizeof(real);

    if ( Nblocks >= (size_t(1)<<size_t(30)) )
        throw std::range_error { "dim2::cuda_kernel::compute_rho: Too many blocks for one kernel call." };

    cuda::set_device(device_number);
    cuda::memset( cu_rho, 0, rho_size );
    cuda_eval_rho<real,order><<<Nblocks,block_size>>>( n, cu_coeffs, conf, cu_rho, q_begin, q_end );
}

template <typename real, size_t order>
void cuda_kernel<real,order>::download_rho( real *rho )
{
    real *cu_rho = reinterpret_cast<real*>( cuda_rho.get() );
    size_t N     = conf.Nx*conf.Ny;

    cuda::set_device(device_number);
    cuda::memcpy_to_host( tmp_rho.get(), cu_rho, sizeof(real)*N );

    for ( size_t i = 0; i < N; ++i )
        rho[ i ] += tmp_rho[ i ];
}

template <typename real, size_t order>
void cuda_kernel<real,order>::upload_phi( size_t n, const real *coeffs )
{
    size_t stride_n = (conf.Nx + order - 1)*(conf.Ny + order - 1);
    real *cu_coeffs = reinterpret_cast<real*>( cuda_coeffs.get() );

    cuda::set_device(device_number);
    cuda::memcpy_to_device( cu_coeffs + n*stride_n,
                               coeffs + n*stride_n, sizeof(real)*stride_n );
}

template <typename real, size_t order>
void cuda_kernel<real,order>::compute_metrics( size_t n, size_t q_begin, size_t q_end )
{
    if ( q_begin == q_end ) return;

    real *cu_coeffs  = reinterpret_cast<real*>( cuda_coeffs .get() );
    real *cu_metrics = reinterpret_cast<real*>( cuda_metrics.get() );

    size_t N = q_end - q_begin;
    size_t block_size = 64;
    size_t Nblocks = 1 + (N-1) / block_size;

    if ( Nblocks >= (size_t(1)<<size_t(30)) )
        throw std::range_error { "dim2::cuda_kernel::compute_metrics: Too many blocks for one kernel call." };

    cuda::set_device(device_number);
    cuda::memset( cu_metrics, 0, 4*sizeof(real) );
    cuda_eval_metrics<real,order><<<Nblocks,block_size>>>( n, cu_coeffs, conf, cu_metrics, q_begin, q_end );
}

template <typename real, size_t order>
void cuda_kernel<real,order>::download_metrics( real *metrics )
{
    real *cu_metrics = reinterpret_cast<real*>( cuda_metrics.get() );
    real tmp_metrics[ 4 ];

    cuda::set_device(device_number);
    cuda::memcpy_to_host( tmp_metrics, cu_metrics, 4*sizeof(real) );

    metrics[ 0 ] += tmp_metrics[ 0 ];
    metrics[ 1 ] += tmp_metrics[ 1 ];
    metrics[ 2 ] += tmp_metrics[ 2 ];
    metrics[ 3 ] += tmp_metrics[ 3 ];
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

namespace dim3
{

// Normal version of cuda_eval_rho.
template <typename real, size_t order>
__global__
void cuda_eval_rho( size_t n, const real *coeffs, const config_t<real> conf, real *rho,
                    size_t l_min, size_t l_end )
{
    const size_t N = l_end - l_min;
    const size_t l = l_min + (blockDim.x*blockIdx.x + threadIdx.x) % N;

    size_t k   = l / ( conf.Nx * conf.Ny );
    size_t tmp = l % ( conf.Nx * conf.Ny );
    size_t j   = tmp / conf.Nx;
    size_t i   = tmp % conf.Nx;

    const real  x = conf.x_min + i*conf.dx;
    const real  y = conf.y_min + j*conf.dy;
    const real  z = conf.z_min + k*conf.dz;

    const real du = (conf.u_max - conf.u_min) / conf.Nu;
    const real dv = (conf.v_max - conf.v_min) / conf.Nv;
    const real dw = (conf.w_max - conf.w_min) / conf.Nw;

    const real u_min = conf.u_min + 0.5*du;
    const real v_min = conf.v_min + 0.5*dv;
    const real w_min = conf.w_min + 0.5*dw;

    real result = 0;
    for ( size_t kk = 0; kk < conf.Nw; ++kk )
    for ( size_t jj = 0; jj < conf.Nv; ++jj )
    for ( size_t ii = 0; ii < conf.Nu; ++ii )
    {
        real u = u_min + ii*du;
        real v = v_min + jj*dv;
        real w = w_min + kk*dw;

        result += eval_ftilda<real,order>( n, x, y, z, u, v, w, coeffs, conf );
    }
    result = 1 - du*dv*dw*result;

    rho[ l ] = result;
}

template <typename real, size_t order>
__global__
void cuda_eval_metrics( size_t n, const real *coeffs, const config_t<real> conf, real *metrics,
                        size_t l_min, size_t l_end )
{
    const size_t N = l_end - l_min;
    const size_t l = l_min + (blockDim.x*blockIdx.x + threadIdx.x) % N;

    size_t k   = l / ( conf.Nx * conf.Ny );
    size_t tmp = l % ( conf.Nx * conf.Ny );
    size_t j   = tmp / conf.Nx;
    size_t i   = tmp % conf.Nx;

    const real x = conf.x_min + i*conf.dx; 
    const real y = conf.y_min + j*conf.dy; 
    const real z = conf.z_min + k*conf.dz; 

    const real du = (conf.u_max - conf.u_min) / conf.Nu;
    const real dv = (conf.v_max - conf.v_min) / conf.Nv;
    const real dw = (conf.w_max - conf.w_min) / conf.Nw;

    const real u_min = conf.u_min + 0.5*du;
    const real v_min = conf.v_min + 0.5*dv;
    const real w_min = conf.w_min + 0.5*dw;

    real tmp_metrics[ 4 ] = { 0, 0, 0, 0 };
    for ( size_t kk = 0; kk < conf.Nw; ++kk )
    for ( size_t jj = 0; jj < conf.Nv; ++jj )
    for ( size_t ii = 0; ii < conf.Nu; ++ii )
    {
        real u = u_min + ii*du;
        real v = v_min + jj*dv;
        real w = w_min + kk*dw;

        real f = eval_f<real,order>( n, x, y, z, u, v, w, coeffs, conf );

        tmp_metrics[ 0 ] += f;                       // L¹-Norm
        tmp_metrics[ 1 ] += f*f;                     // L²-Norm
        tmp_metrics[ 2 ] += (u*u + v*v + w*w)*f;     // Kinetic Energy
        tmp_metrics[ 3 ] += (f>0) ? -f * log(f) : 0; // Entropy
    }

    real quadrature_weight = du*dv*dw*conf.dx*conf.dy*conf.dz;
    tmp_metrics[ 0 ] *= quadrature_weight;
    tmp_metrics[ 1 ] *= quadrature_weight;
    tmp_metrics[ 2 ] *= quadrature_weight/2;
    tmp_metrics[ 3 ] *= quadrature_weight;

    if ( blockDim.x*blockIdx.x + threadIdx.x < N )
    {
        atomicAdd( metrics + 0, tmp_metrics[ 0 ] );
        atomicAdd( metrics + 1, tmp_metrics[ 1 ] );
        atomicAdd( metrics + 2, tmp_metrics[ 2 ] );
        atomicAdd( metrics + 3, tmp_metrics[ 3 ] );
    }
}

template <typename real, size_t order>
cuda_kernel<real,order>::cuda_kernel( const config_t<real> &p_conf, int dev ):
conf { p_conf }, device_number { dev }
{
    size_t  coeff_size  = sizeof(real)*(conf.Nt+1)*(conf.Nx+order-1)*
                                                   (conf.Ny+order-1)*
                                                   (conf.Nz+order-1);
    size_t    rho_size  = sizeof(real)*conf.Nx*conf.Ny*conf.Nz;
    size_t metrics_size = sizeof(real)*4;

    cuda::set_device(dev);
    cuda_coeffs .reset( cuda::malloc(  coeff_size), dev );
    cuda_rho    .reset( cuda::malloc(    rho_size), dev );
    cuda_metrics.reset( cuda::malloc(metrics_size), dev );
}

template <typename real, size_t order>
void cuda_kernel<real,order>::compute_rho( size_t n, size_t l_min, size_t l_end )
{
    if ( n > conf.Nt )
        throw std::range_error { "Time-step out of range." };

    if ( l_min == l_end ) return;

    real *cu_coeffs = reinterpret_cast<real*>( cuda_coeffs.get() );
    real *cu_rho    = reinterpret_cast<real*>( cuda_rho   .get() );

    size_t N = l_end - l_min;
    size_t block_size = 64;
    size_t Nblocks = 1 + (N-1) / block_size;

    cuda::set_device(device_number);
    cuda_eval_rho<real,order><<<Nblocks,block_size>>>( n, cu_coeffs, conf, cu_rho, l_min, l_end );
}

template <typename real, size_t order>
void cuda_kernel<real,order>::download_rho(real *rho, size_t l_min, size_t l_end)
{
    if ( l_min == l_end ) return;
    size_t N = l_end - l_min;

    real *cu_rho = reinterpret_cast<real*>( cuda_rho.get() );

    // Copying rho to normal RAM:
    cuda::set_device(device_number);
    cuda::memcpy_to_host( rho + l_min, cu_rho + l_min, N*sizeof(real) );
}

template <typename real, size_t order>
void cuda_kernel<real,order>::upload_phi( size_t n, const real *coeffs )
{
    size_t stride_n = (conf.Nx + order - 1)*
                      (conf.Ny + order - 1)*
                      (conf.Nz + order - 1);
    real *cu_coeffs = reinterpret_cast<real*>( cuda_coeffs.get() );

    cuda::set_device(device_number);
    cuda::memcpy_to_device( cu_coeffs + n*stride_n,
                               coeffs + n*stride_n, sizeof(real)*stride_n );
}

template <typename real, size_t order>
void cuda_kernel<real,order>::compute_metrics( size_t n, size_t l_min, size_t l_end )
{
    if ( l_min == l_end ) return;

    real *cu_coeffs  = reinterpret_cast<real*>( cuda_coeffs .get() );
    real *cu_metrics = reinterpret_cast<real*>( cuda_metrics.get() );
    real zero_metrics[ 4 ] = { 0, 0, 0, 0 };

    size_t N = l_end - l_min;
    size_t block_size = 64;
    size_t Nblocks = 1 + (N-1) / block_size;

    cuda::set_device(device_number);
    cuda::memcpy_to_device( cu_metrics, zero_metrics, 4*sizeof(real) ); // Initialise to zero on device.
    cuda_eval_metrics<real,order><<<Nblocks,block_size>>>( n, cu_coeffs, conf, cu_metrics, l_min, l_end );
}

template <typename real, size_t order>
void cuda_kernel<real,order>::download_metrics( real *metrics )
{
    real *cu_metrics = reinterpret_cast<real*>( cuda_metrics.get() );

    cuda::set_device(device_number);
    cuda::memcpy_to_host( metrics, cu_metrics, 4*sizeof(real) );
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

