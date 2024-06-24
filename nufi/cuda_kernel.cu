/*
 * Copyright (C) 2022 Matthias Kirchhart and Paul Wilhelm
 *
 * This file is part of NuFI, a solver for the Vlasov–Poisson equation.
 *
 * NuFI is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3, or (at your option) any later
 * version.
 *
 * NuFI is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * NuFI; see the file COPYING.  If not see http://www.gnu.org/licenses.
 */
#include <nufi/cuda_kernel.hpp>
#include <nufi/rho.hpp>

#include <stdexcept>

namespace nufi
{

namespace dim1
{

namespace periodic
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

namespace dirichlet
{
template <typename real, size_t order>
__global__
void cuda_eval_rho_ion_acoustic( size_t n, const real *coeffs, const config_t<real> conf, real *rho,
                    size_t q_begin, size_t q_end )
{
    // Number of my quadrature node.
    const size_t q = q_begin +
                     size_t(blockDim.x)*size_t(blockIdx.x) + size_t(threadIdx.x);

    const size_t ix = q / conf.Nu_electron;
    const size_t iu = q % conf.Nu_electron;
    
    const real   x = conf.x_min + ix*conf.dx; 
    const real   u_ion = conf.u_ion_min + iu*conf.du_ion + conf.du_ion/2;
    const real   u_electron = conf.u_electron_min + iu*conf.du_electron + conf.du_electron/2;
    real *my_rho = rho + ix;

    const real f_ion = eval_ftilda_ion_acoustic<real,order>( n, x, u_ion, coeffs, conf, false );
	const real f_electron = eval_ftilda_ion_acoustic<real,order>( n, x, u_electron, coeffs, conf, true );
    const real weight_ion = conf.du_ion;
    const real weight_electron = conf.du_electron;

    if ( q < q_end ) atomicAdd( my_rho, weight_ion*f_ion - weight_electron*f_electron );
}
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
    size_t  coeff_size  = sizeof(real)*(conf.Nt+1)*(conf.l+order);
    size_t     rho_size = sizeof(real)*(conf.Nx);
    size_t metrics_size = sizeof(real)*4;
    
    cuda::set_device( device_number );
    cuda_coeffs .reset( cuda::malloc(  coeff_size), dev );
    cuda_rho    .reset( cuda::malloc(    rho_size), dev );
    cuda_metrics.reset( cuda::malloc(metrics_size), dev );
    tmp_rho.reset( new real[ conf.Nx] );
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

    size_t rho_size = (conf.Nx)*sizeof(real);

    cuda::set_device( device_number );
    cuda::memset( cu_rho, 0, rho_size );
    //cuda_eval_rho<real,order><<<Nblocks,block_size>>>( n, cu_coeffs, conf, cu_rho, q_begin, q_end );
    cuda_eval_rho_ion_acoustic<real,order><<<Nblocks,block_size>>>( n, cu_coeffs, conf, cu_rho, q_begin, q_end );
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
    size_t stride_n = (conf.l + order );
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



}

namespace dim2
{

namespace periodic
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

    if ( Nblocks >= (size_t(1)<<size_t(31)) )
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

    if ( Nblocks >= (size_t(1)<<size_t(31)) )
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

namespace dirichlet{

template <typename real, size_t order>
__global__
void cuda_eval_rho( size_t n, const real *coeffs, const config_t<real> conf, real *rho,
                    size_t q_begin, size_t q_end )
{
    // Number of my quadrature node.
    const size_t q = q_begin +
                     size_t(blockDim.x)*size_t(blockIdx.x) + size_t(threadIdx.x);

    size_t tmp = q;
    const size_t iy  = tmp / ( (conf.Nx+1)*conf.Nv*conf.Nu ); //GERATEN
                 tmp = tmp % ( (conf.Nx+1)*conf.Nv*conf.Nu );
    const size_t ix  = tmp / ( conf.Nv*conf.Nu );
                 tmp = tmp % ( conf.Nv*conf.Nu );
    const size_t iv  = tmp / ( conf.Nu ); 
    const size_t iu  = tmp % ( conf.Nu );
    
    const real x = conf.x_min + ix*conf.dx; 
    const real y = conf.y_min + iy*conf.dy; 
    const real u = conf.u_min + iu*conf.du + conf.du/2;
    const real v = conf.v_min + iv*conf.dv + conf.dv/2;
    real *my_rho = rho + iy*(conf.Nx+1) + ix;

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
    const size_t iy  = tmp / ( (conf.Nx+1)*conf.Nv*conf.Nu );
                 tmp = tmp % ( (conf.Nx+1)*conf.Nv*conf.Nu );
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
    size_t   coeff_size = sizeof(real)*(conf.Nt+1)*(conf.lx+order)*
                                                   (conf.ly+order);
    size_t     rho_size = sizeof(real)*(conf.Nx+1)*(conf.Ny+1);
    size_t metrics_size = sizeof(real)*4;

    cuda::set_device(dev);
    cuda_coeffs .reset( cuda::malloc(  coeff_size), dev );
    cuda_rho    .reset( cuda::malloc(    rho_size), dev );
    cuda_metrics.reset( cuda::malloc(metrics_size), dev );
    tmp_rho.reset( new real[ rho_size  ] );
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

    size_t rho_size = (conf.Nx+1)*(conf.Ny+1)*sizeof(real);

    if ( Nblocks >= (size_t(1)<<size_t(31)) )
        throw std::range_error { "dim2::cuda_kernel::compute_rho: Too many blocks for one kernel call." };

    cuda::set_device(device_number);
    cuda::memset( cu_rho, 0, rho_size );
    cuda_eval_rho<real,order><<<Nblocks,block_size>>>( n, cu_coeffs, conf, cu_rho, q_begin, q_end );
}

template <typename real, size_t order>
void cuda_kernel<real,order>::download_rho( real *rho )
{
    real *cu_rho = reinterpret_cast<real*>( cuda_rho.get() );
    size_t N     = (conf.Nx+1)*(conf.Ny+1);

    cuda::set_device(device_number);
    cuda::memcpy_to_host( tmp_rho.get(), cu_rho, sizeof(real)*N );

    for ( size_t i = 0; i < N; ++i )
        rho[ i ] += tmp_rho[ i ];
}

template <typename real, size_t order>
void cuda_kernel<real,order>::upload_phi( size_t n, const real *coeffs )
{
    size_t stride_n = (conf.lx + order)*(conf.ly + order );
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

    if ( Nblocks >= (size_t(1)<<size_t(31)) )
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

}

namespace dim3
{

// Normal version of cuda_eval_rho.
template <typename real, size_t order>
__global__
void cuda_eval_rho( size_t n, const real *coeffs, const config_t<real> conf, real *rho,
                    size_t q_begin, size_t q_end )
{
    // Number of my quadrature node.
    const size_t q = q_begin +
                     size_t(blockDim.x)*size_t(blockIdx.x) + size_t(threadIdx.x);

    size_t tmp = q;
    const size_t iz  = tmp / ( conf.Ny*conf.Nx*conf.Nw*conf.Nv*conf.Nu );
                 tmp = tmp % ( conf.Ny*conf.Nx*conf.Nw*conf.Nv*conf.Nu );
    const size_t iy  = tmp / ( conf.Nx*conf.Nw*conf.Nv*conf.Nu );
                 tmp = tmp % ( conf.Nx*conf.Nw*conf.Nv*conf.Nu );
    const size_t ix  = tmp / ( conf.Nw*conf.Nv*conf.Nu );
                 tmp = tmp % ( conf.Nw*conf.Nv*conf.Nu );
    const size_t iw  = tmp / ( conf.Nv*conf.Nu );
                 tmp = tmp % ( conf.Nv*conf.Nu );
    const size_t iv  = tmp / ( conf.Nu ); 
    const size_t iu  = tmp % ( conf.Nu );
    
    const real x = conf.x_min + ix*conf.dx; 
    const real y = conf.y_min + iy*conf.dy; 
    const real z = conf.z_min + iz*conf.dz; 
    const real u = conf.u_min + iu*conf.du + conf.du/2;
    const real v = conf.v_min + iv*conf.dv + conf.dv/2;
    const real w = conf.w_min + iw*conf.dw + conf.dw/2;
    real *my_rho = rho + iz*conf.Nx*conf.Ny + iy*conf.Nx + ix;

    const real f = eval_ftilda<real,order>( n, x, y, z, u, v, w, coeffs, conf );
    const real weight = conf.du*conf.dv*conf.dw;

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
    const size_t iz  = tmp / ( conf.Ny*conf.Nx*conf.Nw*conf.Nv*conf.Nu );
                 tmp = tmp % ( conf.Ny*conf.Nx*conf.Nw*conf.Nv*conf.Nu );
    const size_t iy  = tmp / ( conf.Nx*conf.Nw*conf.Nv*conf.Nu );
                 tmp = tmp % ( conf.Nx*conf.Nw*conf.Nv*conf.Nu );
    const size_t ix  = tmp / ( conf.Nw*conf.Nv*conf.Nu );
                 tmp = tmp % ( conf.Nw*conf.Nv*conf.Nu );
    const size_t iw  = tmp / ( conf.Nv*conf.Nu );
                 tmp = tmp % ( conf.Nv*conf.Nu );
    const size_t iv  = tmp / ( conf.Nu ); 
    const size_t iu  = tmp % ( conf.Nu );
    
    const real x = conf.x_min + ix*conf.dx; 
    const real y = conf.y_min + iy*conf.dy; 
    const real z = conf.z_min + iz*conf.dz; 
    const real u = conf.u_min + iu*conf.du + conf.du/2;
    const real v = conf.v_min + iv*conf.dv + conf.dv/2;
    const real w = conf.w_min + iw*conf.dw + conf.dw/2;

    const real f = eval_f<real,order>( n, x, y, z, u, v, w, coeffs, conf );
    const real weight = conf.du*conf.dv*conf.dw;

    if ( q < q_end )
    {
        atomicAdd( metrics + 0, weight*f );
        atomicAdd( metrics + 1, weight*f*f );
        atomicAdd( metrics + 2, weight*(u*u+v*v+w*w)*f/2 );
        atomicAdd( metrics + 3, (f>0) ? -weight*f*log(f) : 0 );
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
    tmp_rho.reset( new real[ conf.Nx*conf.Ny*conf.Nz ] );
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

    size_t rho_size = conf.Nx*conf.Ny*conf.Nz*sizeof(real);

    if ( Nblocks >= (size_t(1)<<size_t(31)) )
    {
        throw std::range_error { "dim3::cuda_kernel::compute_rho: Too many blocks for one kernel call." };
    }

    cuda::set_device(device_number);
    cuda::memset( cu_rho, 0, rho_size );
    cuda_eval_rho<real,order><<<Nblocks,block_size>>>( n, cu_coeffs, conf, cu_rho, q_begin, q_end );
}

template <typename real, size_t order>
void cuda_kernel<real,order>::download_rho( real *rho )
{
    real *cu_rho = reinterpret_cast<real*>( cuda_rho.get() );
    size_t N     = conf.Nx*conf.Ny*conf.Nz;

    cuda::set_device(device_number);
    cuda::memcpy_to_host( tmp_rho.get(), cu_rho, sizeof(real)*N );

    for ( size_t i = 0; i < N; ++i )
        rho[ i ] += tmp_rho[ i ];
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
void cuda_kernel<real,order>::compute_metrics( size_t n, size_t q_begin, size_t q_end )
{
    if ( q_begin == q_end ) return;

    real *cu_coeffs  = reinterpret_cast<real*>( cuda_coeffs .get() );
    real *cu_metrics = reinterpret_cast<real*>( cuda_metrics.get() );

    size_t N = q_end - q_begin;
    size_t block_size = 64;
    size_t Nblocks = 1 + (N-1) / block_size;

    if ( Nblocks >= (size_t(1)<<size_t(31)) )
    {
        throw std::range_error { "dim3::cuda_kernel::compute_metrics: Too many blocks for one kernel call." };
    }

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


}

