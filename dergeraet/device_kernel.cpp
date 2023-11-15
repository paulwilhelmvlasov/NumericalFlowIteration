/*
 * Copyright (C) 2022, 2023 Matthias Kirchhart and Paul Wilhelm
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
#include <dergeraet/device_kernel.hpp>
#include <dergeraet/numerical_flow.hpp>

#include <stdexcept>

namespace
{

template <typename real>
using relaxed_atomic = sycl::atomic_ref<real,sycl::memory_order::relaxed,
                                             sycl::memory_scope::device>;
}

namespace dergeraet
{

namespace dim1
{

template <typename real, size_t order>
void device_eval_rho( size_t n, const real *coeffs, const config_t<real> conf, real *rho,
                      size_t q_begin, size_t q_end, sycl::nd_item<1> item )
{
    // Number of my quadrature node.
    const size_t q = q_begin + item.get_global_id(0);
    if ( q >= q_end ) return;

    const size_t ix = q / conf.Nu;
    const size_t iu = q % conf.Nu;
    
    real x = conf.x_min + ix*conf.dx; 
    real u = conf.u_min + iu*conf.du + conf.du/2;
    tilda_numerical_flow<real,order>(n,x,u,coeffs,conf);

    const real f = config_t<real>::f0(x,u);

    relaxed_atomic<real> my_rho( *(rho+ix) );
    const real weight = conf.du;

    my_rho.fetch_add( -weight*f );
}

template <typename real, size_t order>
void device_eval_metrics( size_t n, const real *coeffs, const config_t<real> conf, real *metrics,
                          size_t q_begin, size_t q_end, sycl::nd_item<1> item )
{
    using sycl::log;

    // Number of my quadrature node.
    const size_t q = q_begin + item.get_global_id(0);
    if ( q >= q_end ) return;

    const size_t ix = q / conf.Nu;
    const size_t iu = q % conf.Nu;
    
    real x = conf.x_min + ix*conf.dx; 
    real u = conf.u_min + iu*conf.du + conf.du/2;
    numerical_flow<real,order>(n,x,u,coeffs,conf);

    const real f = config_t<real>::f0(x,u);
    const real weight = conf.du*conf.dx;

    u = conf.u_min + iu*conf.du + conf.du/2;
    relaxed_atomic<real> l1_norm ( metrics[0] ); l1_norm .fetch_add( weight*f   );
    relaxed_atomic<real> l2_norm ( metrics[1] ); l2_norm .fetch_add( weight*f*f );
    relaxed_atomic<real> kineticE( metrics[2] ); kineticE.fetch_add( weight*(u*u*f*real(0.5)) );
    relaxed_atomic<real> entropy ( metrics[3] ); entropy .fetch_add( (f>real(0)) ? -weight*f*log(f) : real(0) );
}

template <typename real, size_t order>
device_kernel<real,order>::device_kernel( const config_t<real> &p_conf, sycl::device dev ):
conf { p_conf }, Q { dev, sycl::property::queue::in_order() }
{
    size_t  coeff_size  = (conf.Nt+1)*(conf.Nx+order-1);
    size_t     rho_size = conf.Nx;
    size_t metrics_size = 4;

    try
    {
        tmp_rho = new real[ rho_size ];
        dev_coeffs  = sycl::malloc_device<real>(   coeff_size, Q );
        dev_rho     = sycl::malloc_device<real>(     rho_size, Q );
        dev_metrics = sycl::malloc_device<real>( metrics_size, Q );
    }
    catch ( ... )
    {
        sycl::free(dev_metrics,Q);
        sycl::free(dev_rho    ,Q);
        sycl::free(dev_coeffs ,Q);
        delete [] tmp_rho;
        throw;
    }

    if ( dev_coeffs == nullptr || dev_rho == nullptr || dev_metrics == nullptr )
    {
        sycl::free(dev_metrics,Q);
        sycl::free(dev_rho    ,Q);
        sycl::free(dev_coeffs ,Q);
        delete [] tmp_rho;
        throw std::bad_alloc {};
    }
}

template <typename real, size_t order>
device_kernel<real,order>::~device_kernel()
{
    sycl::free(dev_metrics,Q);
    sycl::free(dev_rho    ,Q);
    sycl::free(dev_coeffs ,Q);
    delete [] tmp_rho;
}

template <typename real, size_t order>
device_kernel<real,order>::device_kernel( device_kernel<real,order> &&rhs ) noexcept:
conf { rhs.conf }, Q { rhs.Q },
tmp_rho { rhs.tmp_rho }, dev_coeffs  { rhs.dev_coeffs },
dev_rho { rhs.dev_rho }, dev_metrics { rhs.dev_metrics }
{
    rhs.tmp_rho     = nullptr;
    rhs.dev_rho     = nullptr;
    rhs.dev_coeffs  = nullptr;
    rhs.dev_metrics = nullptr;
}

template <typename real, size_t order>
device_kernel<real,order>&
device_kernel<real,order>::operator=( device_kernel<real,order> &&rhs ) noexcept
{
    using std::swap;

    swap(conf       ,rhs.conf);
    swap(Q          ,rhs.Q);
    swap(tmp_rho    ,rhs.tmp_rho);
    swap(dev_rho    ,rhs.dev_rho);
    swap(dev_coeffs ,rhs.dev_coeffs);
    swap(dev_metrics,rhs.dev_metrics);

    return *this;
}

template <typename real, size_t order>
void device_kernel<real,order>::compute_rho( size_t n, size_t q_begin, size_t q_end )
{
    if ( n > conf.Nt )
        throw std::range_error { "Time-step out of range." };

    if ( q_begin == q_end ) return;

    size_t N          = q_end - q_begin;
    size_t block_size = 64;
    size_t Nblocks    = 1 + ( (N-1) / (block_size) );

    sycl::nd_range<1> myrange( sycl::range<1>( Nblocks*block_size ),
                               sycl::range<1>( block_size) );

    size_t rho_bytes = conf.Nx*sizeof(real);

    config_t<real> tmp_conf = conf;
    real *tmp_dev_coeffs = dev_coeffs;
    real *tmp_dev_rho    = dev_rho;

    Q.memset( dev_rho, 0, rho_bytes );
    Q.parallel_for( myrange, [=]( sycl::nd_item<1> item )
    {
        device_eval_rho<real,order>( n, tmp_dev_coeffs, tmp_conf, tmp_dev_rho, q_begin, q_end, item );
    });
}

template <typename real, size_t order>
void device_kernel<real,order>::download_rho( real *rho )
{
    size_t N = conf.Nx;

    Q.memcpy( tmp_rho, dev_rho, sizeof(real)*N ).wait();
    for ( size_t i = 0; i < N; ++i )
        rho[ i ] += tmp_rho[ i ];
}

template <typename real, size_t order>
void device_kernel<real,order>::upload_phi( size_t n, const real *coeffs )
{
    size_t stride_n = (conf.Nx + order - 1);
    Q.memcpy( dev_coeffs + n*stride_n, coeffs + n*stride_n, sizeof(real)*stride_n ).wait();
}

template <typename real, size_t order>
void device_kernel<real,order>::compute_metrics( size_t n, size_t q_begin, size_t q_end )
{
    if ( q_begin == q_end ) return;

    size_t N = q_end - q_begin;
    size_t block_size = 64;
    size_t Nblocks = 1 + (N-1) / block_size;

    sycl::nd_range<1> myrange( sycl::range<1>( Nblocks*block_size ),
                               sycl::range<1>( block_size) );

    config_t<real> tmp_conf = conf;
    real *tmp_dev_coeffs  = dev_coeffs;
    real *tmp_dev_metrics = dev_metrics;

    Q.memset( dev_metrics, 0, 4*sizeof(real) );
    Q.parallel_for( myrange, [=]( sycl::nd_item<1> item )
    {
        device_eval_metrics<real,order>( n, tmp_dev_coeffs, tmp_conf, tmp_dev_metrics, q_begin, q_end, item );
    });
}

template <typename real, size_t order>
void device_kernel<real,order>::download_metrics( real *metrics )
{
    real tmp_metrics[ 4 ];

    Q.memcpy( tmp_metrics, dev_metrics, 4*sizeof(real) ).wait();

    metrics[ 0 ] += tmp_metrics[ 0 ];
    metrics[ 1 ] += tmp_metrics[ 1 ];
    metrics[ 2 ] += tmp_metrics[ 2 ];
    metrics[ 3 ] += tmp_metrics[ 3 ];
}

template class device_kernel<double,3>;
template class device_kernel<double,4>;
template class device_kernel<double,5>;
template class device_kernel<double,6>;
template class device_kernel<double,7>;
template class device_kernel<double,8>;

template class device_kernel<float,3>;
template class device_kernel<float,4>;
template class device_kernel<float,5>;
template class device_kernel<float,6>;
template class device_kernel<float,7>;
template class device_kernel<float,8>;

}

namespace dim2
{

template <typename real, size_t order>
void device_eval_rho( size_t n, const real *coeffs, const config_t<real> conf, real *rho,
                      size_t q_begin, size_t q_end, sycl::nd_item<1> it )
{
    // Number of my quadrature node.
    const size_t q = q_begin + it.get_global_id(0);
    if ( q >= q_end ) return;

    size_t tmp = q;
    const size_t iy  = tmp / ( conf.Nx*conf.Nv*conf.Nu );
                 tmp = tmp % ( conf.Nx*conf.Nv*conf.Nu );
    const size_t ix  = tmp / ( conf.Nv*conf.Nu );
                 tmp = tmp % ( conf.Nv*conf.Nu );
    const size_t iv  = tmp / ( conf.Nu ); 
    const size_t iu  = tmp % ( conf.Nu );
    
    real x = conf.x_min + ix*conf.dx; 
    real y = conf.y_min + iy*conf.dy; 
    real u = conf.u_min + iu*conf.du + conf.du/2;
    real v = conf.v_min + iv*conf.dv + conf.dv/2;
    tilda_numerical_flow<real,order>( n, x, y, u, v, coeffs, conf );

    const real f = config_t<real>::f0(x,y,u,v);
    const real weight = conf.du*conf.dv;

    relaxed_atomic<real> my_rho( *(rho + iy*conf.Nx + ix) );
    my_rho.fetch_add( -weight*f );
}

template <typename real, size_t order>
void device_eval_metrics( size_t n, const real *coeffs, const config_t<real> conf, real *metrics,
                          size_t q_begin, size_t q_end, sycl::nd_item<1> item )
{
    // Number of my quadrature node.
    const size_t q = q_begin + item.get_global_id(0);
    if ( q >= q_end ) return;

    size_t tmp = q;
    const size_t iy  = tmp / ( conf.Nx*conf.Nv*conf.Nu );
                 tmp = tmp % ( conf.Nx*conf.Nv*conf.Nu );
    const size_t ix  = tmp / ( conf.Nv*conf.Nu );
                 tmp = tmp % ( conf.Nv*conf.Nu );
    const size_t iv  = tmp / ( conf.Nu ); 
    const size_t iu  = tmp % ( conf.Nu );
    
    real x = conf.x_min + ix*conf.dx; 
    real y = conf.y_min + iy*conf.dy; 
    real u = conf.u_min + iu*conf.du + conf.du/2;
    real v = conf.v_min + iv*conf.dv + conf.dv/2;
    numerical_flow<real,order>( n, x, y, u, v, coeffs, conf );

    const real f = config_t<real>::f0(x,y,u,v);
    const real weight = conf.dx*conf.dy*conf.du*conf.dv;

    u = conf.u_min + iu*conf.du + conf.du/2;
    v = conf.v_min + iv*conf.dv + conf.dv/2;
    relaxed_atomic<real> l1_norm ( metrics[0] ); l1_norm .fetch_add( weight*f   );
    relaxed_atomic<real> l2_norm ( metrics[1] ); l2_norm .fetch_add( weight*f*f );
    relaxed_atomic<real> kineticE( metrics[2] ); kineticE.fetch_add( weight*((u*u+v*v)*f*real(0.5)) );
    relaxed_atomic<real> entropy ( metrics[3] ); entropy .fetch_add( (f>real(0)) ? -weight*f*log(f) : real(0) );
}

template <typename real, size_t order>
device_kernel<real,order>::device_kernel( const config_t<real> &p_conf, sycl::device dev ):
conf { p_conf }, Q { dev, sycl::property::queue::in_order() }
{
    size_t   coeff_size = (conf.Nt+1)*(conf.Nx+order-1)*
                                      (conf.Ny+order-1);
    size_t     rho_size = conf.Nx*conf.Ny;
    size_t metrics_size = 4;

    try
    {
        tmp_rho = new real[ rho_size ];
        dev_coeffs  = sycl::malloc_device<real>(   coeff_size, Q );
        dev_rho     = sycl::malloc_device<real>(     rho_size, Q );
        dev_metrics = sycl::malloc_device<real>( metrics_size, Q );
    }
    catch ( ... )
    {
        sycl::free(dev_metrics,Q);
        sycl::free(dev_rho    ,Q);
        sycl::free(dev_coeffs ,Q);
        delete [] tmp_rho;
        throw;
    }

    if ( dev_coeffs == nullptr || dev_rho == nullptr || dev_metrics == nullptr )
    {
        sycl::free(dev_metrics,Q);
        sycl::free(dev_rho    ,Q);
        sycl::free(dev_coeffs ,Q);
        delete [] tmp_rho;
        throw std::bad_alloc {};
    }
}

template <typename real, size_t order>
device_kernel<real,order>::~device_kernel()
{
    sycl::free(dev_metrics,Q);
    sycl::free(dev_rho    ,Q);
    sycl::free(dev_coeffs ,Q);
    delete [] tmp_rho;
}

template <typename real, size_t order>
device_kernel<real,order>::device_kernel( device_kernel<real,order> &&rhs ) noexcept:
conf { rhs.conf }, Q { rhs.Q },
tmp_rho { rhs.tmp_rho }, dev_coeffs  { rhs.dev_coeffs },
dev_rho { rhs.dev_rho }, dev_metrics { rhs.dev_metrics }
{
    rhs.tmp_rho     = nullptr;
    rhs.dev_rho     = nullptr;
    rhs.dev_coeffs  = nullptr;
    rhs.dev_metrics = nullptr;
}

template <typename real, size_t order>
device_kernel<real,order>&
device_kernel<real,order>::operator=( device_kernel<real,order> &&rhs ) noexcept
{
    using std::swap;

    swap(conf       ,rhs.conf);
    swap(Q          ,rhs.Q);
    swap(tmp_rho    ,rhs.tmp_rho);
    swap(dev_rho    ,rhs.dev_rho);
    swap(dev_coeffs ,rhs.dev_coeffs);
    swap(dev_metrics,rhs.dev_metrics);

    return *this;
}


template <typename real, size_t order>
void device_kernel<real,order>::compute_rho( size_t n, size_t q_begin, size_t q_end )
{
    if ( n > conf.Nt )
        throw std::range_error { "Time-step out of range." };

    if ( q_begin == q_end ) return;

    size_t N = q_end - q_begin;
    size_t block_size = 64;
    size_t Nblocks = 1 + (N-1) / block_size;

    sycl::nd_range<1> myrange( sycl::range<1>( Nblocks*block_size ),
                               sycl::range<1>( block_size) );

    size_t rho_bytes = conf.Nx*conf.Ny*sizeof(real);

    if ( Nblocks >= (size_t(1)<<size_t(31)) )
        throw std::range_error { "dim2::device_kernel::compute_rho: Too many blocks for one kernel call." };

    config_t<real> tmp_conf = conf;
    real *tmp_dev_coeffs = dev_coeffs;
    real *tmp_dev_rho    = dev_rho;

    Q.memset( dev_rho, 0, rho_bytes );
    Q.parallel_for( myrange, [=]( sycl::nd_item<1> item )
    {
        device_eval_rho<real,order>( n, tmp_dev_coeffs, tmp_conf, tmp_dev_rho, q_begin, q_end, item );
    });
}

template <typename real, size_t order>
void device_kernel<real,order>::download_rho( real *rho )
{
    size_t N     = conf.Nx*conf.Ny;

    Q.memcpy( tmp_rho, dev_rho, sizeof(real)*N ).wait();
    for ( size_t i = 0; i < N; ++i )
        rho[ i ] += tmp_rho[ i ];
}

template <typename real, size_t order>
void device_kernel<real,order>::upload_phi( size_t n, const real *coeffs )
{
    size_t stride_n = (conf.Nx + order - 1)*(conf.Ny + order - 1);
    Q.memcpy( dev_coeffs + n*stride_n, coeffs + n*stride_n, sizeof(real)*stride_n ).wait();
}

template <typename real, size_t order>
void device_kernel<real,order>::compute_metrics( size_t n, size_t q_begin, size_t q_end )
{
    if ( n > conf.Nt )
        throw std::range_error { "Time-step out of range." };

    if ( q_begin == q_end ) return;

    size_t N = q_end - q_begin;
    size_t block_size = 64;
    size_t Nblocks = 1 + (N-1) / block_size;

    sycl::nd_range<1> myrange( sycl::range<1>( Nblocks*block_size ),
                               sycl::range<1>( block_size) );

    if ( Nblocks >= (size_t(1)<<size_t(31)) )
        throw std::range_error { "dim2::device_kernel::compute_rho: Too many blocks for one kernel call." };

    config_t<real> tmp_conf = conf;
    real *tmp_dev_coeffs  = dev_coeffs;
    real *tmp_dev_metrics = dev_metrics;

    Q.memset( dev_metrics, 0, 4*sizeof(real) );
    Q.parallel_for( myrange, [=]( sycl::nd_item<1> item )
    {
        device_eval_metrics<real,order>( n, tmp_dev_coeffs, tmp_conf, tmp_dev_metrics, q_begin, q_end, item );
    });
}

template <typename real, size_t order>
void device_kernel<real,order>::download_metrics( real *metrics )
{
    real tmp_metrics[ 4 ];

    Q.memcpy( tmp_metrics, dev_metrics, 4*sizeof(real) ).wait();

    metrics[ 0 ] += tmp_metrics[ 0 ];
    metrics[ 1 ] += tmp_metrics[ 1 ];
    metrics[ 2 ] += tmp_metrics[ 2 ];
    metrics[ 3 ] += tmp_metrics[ 3 ];
}

template class device_kernel<double,3>;
template class device_kernel<double,4>;
template class device_kernel<double,5>;
template class device_kernel<double,6>;
template class device_kernel<double,7>;
template class device_kernel<double,8>;

template class device_kernel<float,3>;
template class device_kernel<float,4>;
template class device_kernel<float,5>;
template class device_kernel<float,6>;
template class device_kernel<float,7>;
template class device_kernel<float,8>;

}

namespace dim3
{

// Normal version of cuda_eval_rho.
template <typename real, size_t order>
void device_eval_rho( size_t n, const real *coeffs, const config_t<real> conf, real *rho,
                      size_t q_begin, size_t q_end, sycl::nd_item<1> item )
{
    // Number of my quadrature node.
    const size_t q = q_begin + item.get_global_id(0);
    if ( q >= q_end ) return;

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
    
    real x = conf.x_min + ix*conf.dx; 
    real y = conf.y_min + iy*conf.dy; 
    real z = conf.z_min + iz*conf.dz; 
    real u = conf.u_min + iu*conf.du + conf.du*real(0.5);
    real v = conf.v_min + iv*conf.dv + conf.dv*real(0.5);
    real w = conf.w_min + iw*conf.dw + conf.dw*real(0.5);
    tilda_numerical_flow<real,order>(n,x,y,z,u,v,w,coeffs,conf);

    const real f = config_t<real>::f0(x,y,z,u,v,w);
    const real weight = conf.du*conf.dv*conf.dw;

    relaxed_atomic<real> my_rho( *( rho + iz*conf.Nx*conf.Ny + iy*conf.Nx + ix ) );
    my_rho.fetch_add( -weight*f );
}

template <typename real, size_t order>
void device_eval_metrics( size_t n, const real *coeffs, const config_t<real> conf, real *metrics,
                          size_t q_begin, size_t q_end, sycl::nd_item<1> item )
{
    // Number of my quadrature node.
    const size_t q = q_begin + item.get_global_id(0);
    if ( q >= q_end ) return;

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
    
    real x = conf.x_min + ix*conf.dx; 
    real y = conf.y_min + iy*conf.dy; 
    real z = conf.z_min + iz*conf.dz; 
    real u = conf.u_min + iu*conf.du + conf.du*real(0.5);
    real v = conf.v_min + iv*conf.dv + conf.dv*real(0.5);
    real w = conf.w_min + iw*conf.dw + conf.dw*real(0.5);
    numerical_flow<real,order>(n,x,y,z,u,v,w,coeffs,conf);

    const real f = config_t<real>::f0(x,y,z,u,v,w);
    const real weight = conf.du*conf.dv*conf.dw;

    u = conf.u_min + iu*conf.du + conf.du*real(0.5);
    v = conf.v_min + iv*conf.dv + conf.dv*real(0.5);
    w = conf.w_min + iw*conf.dw + conf.dw*real(0.5);
    relaxed_atomic<real> l1_norm ( metrics[0] ); l1_norm .fetch_add( weight*f   );
    relaxed_atomic<real> l2_norm ( metrics[1] ); l2_norm .fetch_add( weight*f*f );
    relaxed_atomic<real> kineticE( metrics[2] ); kineticE.fetch_add( weight*((u*u+v*v+w*w)*f*real(0.5)) );
    relaxed_atomic<real> entropy ( metrics[3] ); entropy .fetch_add( (f>real(0)) ? -weight*f*log(f) : real(0) );
}

template <typename real, size_t order>
device_kernel<real,order>::device_kernel( const config_t<real> &p_conf, sycl::device dev ):
conf { p_conf }, Q { dev, sycl::property::queue::in_order() }
{
    size_t  coeff_size  = (conf.Nt+1)*(conf.Nx+order-1)*
                                      (conf.Ny+order-1)*
                                       (conf.Nz+order-1);
    size_t    rho_size  = conf.Nx*conf.Ny*conf.Nz;
    size_t metrics_size = 4;

    try
    {
        tmp_rho = new real[ rho_size ];
        dev_coeffs  = sycl::malloc_device<real>(   coeff_size, Q );
        dev_rho     = sycl::malloc_device<real>(     rho_size, Q );
        dev_metrics = sycl::malloc_device<real>( metrics_size, Q );
    }
    catch ( ... )
    {
        sycl::free(dev_metrics,Q);
        sycl::free(dev_rho    ,Q);
        sycl::free(dev_coeffs ,Q);
        delete [] tmp_rho;
        throw;
    }

    if ( dev_coeffs == nullptr || dev_rho == nullptr || dev_metrics == nullptr )
    {
        sycl::free(dev_metrics,Q);
        sycl::free(dev_rho    ,Q);
        sycl::free(dev_coeffs ,Q);
        delete [] tmp_rho;
        throw std::bad_alloc {};
    }
}

template <typename real, size_t order>
device_kernel<real,order>::~device_kernel()
{
    sycl::free(dev_metrics,Q);
    sycl::free(dev_rho    ,Q);
    sycl::free(dev_coeffs ,Q);
    delete [] tmp_rho;
}

template <typename real, size_t order>
device_kernel<real,order>::device_kernel( device_kernel<real,order> &&rhs ) noexcept:
conf { rhs.conf }, Q { rhs.Q },
tmp_rho { rhs.tmp_rho }, dev_coeffs  { rhs.dev_coeffs },
dev_rho { rhs.dev_rho }, dev_metrics { rhs.dev_metrics }
{
    rhs.tmp_rho     = nullptr;
    rhs.dev_rho     = nullptr;
    rhs.dev_coeffs  = nullptr;
    rhs.dev_metrics = nullptr;
}

template <typename real, size_t order>
device_kernel<real,order>&
device_kernel<real,order>::operator=( device_kernel<real,order> &&rhs ) noexcept
{
    using std::swap;

    swap(conf       ,rhs.conf);
    swap(Q          ,rhs.Q);
    swap(tmp_rho    ,rhs.tmp_rho);
    swap(dev_rho    ,rhs.dev_rho);
    swap(dev_coeffs ,rhs.dev_coeffs);
    swap(dev_metrics,rhs.dev_metrics);

    return *this;
}

template <typename real, size_t order>
void device_kernel<real,order>::compute_rho( size_t n, size_t q_begin, size_t q_end )
{
    if ( n > conf.Nt )
        throw std::range_error { "Time-step out of range." };

    if ( q_begin == q_end ) return;

    size_t N          = q_end - q_begin;
    size_t block_size = 64;
    size_t Nblocks    = 1 + ( (N-1) / (block_size) );

    sycl::nd_range<1> myrange( sycl::range<1>( Nblocks*block_size ),
                               sycl::range<1>( block_size) );

    size_t rho_bytes = conf.Nx*conf.Ny*conf.Nz*sizeof(real);

    config_t<real> tmp_conf = conf;
    real *tmp_dev_coeffs = dev_coeffs;
    real *tmp_dev_rho    = dev_rho;

    Q.memset( dev_rho, 0, rho_bytes );
    Q.parallel_for( myrange, [=]( sycl::nd_item<1> item )
    {
        device_eval_rho<real,order>( n, tmp_dev_coeffs, tmp_conf, tmp_dev_rho, q_begin, q_end, item );
    });
}

template <typename real, size_t order>
void device_kernel<real,order>::download_rho( real *rho )
{
    size_t N     = conf.Nx*conf.Ny*conf.Nz;

    Q.memcpy( tmp_rho, dev_rho, sizeof(real)*N ).wait();
    for ( size_t i = 0; i < N; ++i )
        rho[ i ] += tmp_rho[ i ];
}

template <typename real, size_t order>
void device_kernel<real,order>::upload_phi( size_t n, const real *coeffs )
{
    size_t stride_n = (conf.Nx + order - 1)*
                      (conf.Ny + order - 1)*
                      (conf.Nz + order - 1);

    Q.memcpy( dev_coeffs + n*stride_n, coeffs + n*stride_n, sizeof(real)*stride_n ).wait();
}

template <typename real, size_t order>
void device_kernel<real,order>::compute_metrics( size_t n, size_t q_begin, size_t q_end )
{
    if ( q_begin == q_end ) return;

    size_t N = q_end - q_begin;
    size_t block_size = 64;
    size_t Nblocks = 1 + (N-1) / block_size;

    sycl::nd_range<1> myrange( sycl::range<1>( Nblocks*block_size ),
                               sycl::range<1>( block_size) );

    if ( Nblocks >= (size_t(1)<<size_t(31)) )
    {
        throw std::range_error { "dim3::device_kernel::compute_metrics: Too many blocks for one kernel call." };
    }

    config_t<real> tmp_conf = conf;
    real *tmp_dev_coeffs  = dev_coeffs;
    real *tmp_dev_metrics = dev_metrics;

    Q.memset( dev_metrics, 0, 4*sizeof(real) );
    Q.parallel_for( myrange, [=]( sycl::nd_item<1> item )
    {
        device_eval_metrics<real,order>( n, tmp_dev_coeffs, tmp_conf, tmp_dev_metrics, q_begin, q_end, item );
    });
}

template <typename real, size_t order>
void device_kernel<real,order>::download_metrics( real *metrics )
{
    real tmp_metrics[ 4 ];

    Q.memcpy( tmp_metrics, dev_metrics, 4*sizeof(real) ).wait();

    metrics[ 0 ] += tmp_metrics[ 0 ];
    metrics[ 1 ] += tmp_metrics[ 1 ];
    metrics[ 2 ] += tmp_metrics[ 2 ];
    metrics[ 3 ] += tmp_metrics[ 3 ];
}

template class device_kernel<double,3>;
template class device_kernel<double,4>;
template class device_kernel<double,5>;
template class device_kernel<double,6>;
template class device_kernel<double,7>;
template class device_kernel<double,8>;

template class device_kernel<float,3>;
template class device_kernel<float,4>;
template class device_kernel<float,5>;
template class device_kernel<float,6>;
template class device_kernel<float,7>;
template class device_kernel<float,8>;

}

}

