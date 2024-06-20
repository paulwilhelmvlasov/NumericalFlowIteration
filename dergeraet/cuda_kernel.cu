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
}

namespace dim_1_half
{

template <typename real>
__host__ __device__
void matrix_exp_jb_times_v(real B, real& v1, real& v2, real tol=1e-16)
{
	// Evaluates exp(-dt*J_B)*v.
	// If B=0, then exp(J_B)=I.

	if(std::abs(B) > tol)
	{
		real fac1 = std::sin(B);
		if(B<0){
			fac1*=-1;
		}
		real fac2 = std::cos(B) - 1;

		real old_v1 = v1;
		real old_v2 = v2;

		v1 = old_v1 + fac1*old_v2 + fac2*old_v1;
		v2 = old_v2 - fac1*old_v1 + fac2*old_v2;
	}
}

template <typename real, size_t order>
__host__ __device__
real eval_f_vm_lie(size_t nt, real x, real u, real v, const real *coeffs_E_1,
		const real *coeffs_E_2, const real *coeffs_B_3, const real *coeffs_J_hf_1,
		const real *coeffs_J_hf_2, const config_t<real> conf)
{
    const size_t stride_t = conf.Nx + order - 1;

	for(;nt > 0; nt--)
	{
	    real B = eval<real,order>(x, coeffs_B_3 + (nt-1)*stride_t, conf);
	    real alpha_1 = u;
	    real alpha_2 = v;
	    matrix_exp_jb_times_v(conf.dt*B,alpha_1,alpha_2);

	    real beta_1 = eval<real,order>(x, coeffs_E_1 + (nt-1)*stride_t, conf)
	    				- conf.dt*eval<real,order>(x, coeffs_j_hf_1 + (nt-1)*stride_t, conf);
		real beta_2 = eval<real,order>(x, coeffs_E_2 + (nt-1)*stride_t, conf)
	    	    		- conf.dt*eval<real,order>(x, coeffs_j_hf_2 + (nt-1)*stride_t, conf)
						- conf.dt*eval<real,order,1>(x, coeffs_B_3 + (nt-1)*stride_t, conf);
		u = alpha_1 - conf.dt*beta_1;
		v = alpha_2 - conf.dt*beta_2;
		x -= conf.dt*u;
	}

	return conf.f0(x, u, v);
}

template <typename real, size_t order>
__global__
void cuda_eval_j_hf( size_t nt, const real *coeffs_E_1, const real *coeffs_E_2,
					const real *coeffs_B_3, const config_t<real> conf, real *j_hf_1,
					real *j_hf_2, size_t q_begin, size_t q_end )
{
	// J_Hf has to be computed to advance from t to t+dt. Thus we use J_Hf(nt-1) to evaluate
	// f(nt+1).
    // Number of my quadrature node.
    const size_t q = q_begin + size_t(blockDim.x)*size_t(blockIdx.x) + size_t(threadIdx.x);

    const size_t ix = q % conf.Nx;
    const size_t iu = size_t(q/conf.Nx) % conf.Nu;
    const size_t iv = q / (conf.Nx*conf.Nu);

    const real   x = conf.x_min + ix*conf.dx;
    const real   u = conf.u_min + (iu+0.5)*conf.du;
    const real   v = conf.v_min + (iv+0.5)*conf.dv;
    real *my_j_hf_1 = j_hf_1 + ix;
    real *my_j_hf_2 = j_hf_2 + ix;

    const real f = eval_f_vm_lie<real,order>( nt, x-0.5*conf.dt*u, u, v, coeffs_E_1, coeffs_E_2,
    										coeffs_B_3, j_hf_1, j_hf_2, conf );
    const real weight = conf.du*conf.dv;

    if ( q < q_end ){
    	atomicAdd( my_j_hf_1, weight*u*f );
    	atomicAdd( my_j_hf_2, weight*v*f );
    }
}

template <typename real, size_t order>
__global__
void cuda_eval_metrics( size_t nt, const real *coeffs_E_1, const real *coeffs_E_2,
						const real *coeffs_B_3, const config_t<real> conf,
						real *metrics, size_t q_begin, size_t q_end )
{
    // Number of my quadrature node.
    const size_t q = q_begin +
                     size_t(blockDim.x)*size_t(blockIdx.x) + size_t(threadIdx.x);

    const size_t ix = q % conf.Nx;
    const size_t iu = size_t(q/conf.Nx) % conf.Nu;
    const size_t iv = q / (conf.Nx*conf.Nu);

    const real x = conf.x_min + (ix+0.5)*conf.dx;
    const real u = conf.u_min + (iu+0.5)*conf.du;
    const real v = conf.v_min + (iv+0.5)*conf.dv;

    const real f = eval_f_vm_lie<real,order>( nt, x, u, v, coeffs_E_1, coeffs_E_2,
											coeffs_B_3, conf );
    const real weight = conf.dx*conf.du*conf.dv;

    if ( q < q_end )
    {
        atomicAdd( metrics + 0, weight*f );
        atomicAdd( metrics + 1, weight*f*f ); // This ...
        atomicAdd( metrics + 2, weight*((u*u+v*v)*f/2) ); // This...
        atomicAdd( metrics + 3, (f>0) ? -weight*f*log(f) : 0 ); // ...and this is wrong...
    }
}

template <typename real, size_t order>
cuda_kernel_vm<real,order>::cuda_kernel_vm( const config_t<real> &p_conf, int dev ):
conf { p_conf }, device_number { dev }
{
    size_t coeff_size  = sizeof(real)*(conf.Nt+1)*(conf.Nx+order-1);
    size_t j_size = sizeof(real)*conf.Nx;
    size_t metrics_size = sizeof(real)*4;

    cuda::set_device( device_number );
    cuda_coeffs_E_1.reset( cuda::malloc(  coeff_size), dev );
    cuda_coeffs_E_2.reset( cuda::malloc(  coeff_size), dev );
    cuda_coeffs_B_3.reset( cuda::malloc(  coeff_size), dev );
    cuda_j_1.reset( cuda::malloc( j_size), dev );
    cuda_j_2.reset( cuda::malloc( j_size), dev );
    cuda_metrics.reset( cuda::malloc(metrics_size), dev );
    tmp_j_1.reset( new real[ conf.Nx ] );
    tmp_j_2.reset( new real[ conf.Nx ] );
}

template <typename real, size_t order>
void cuda_kernel_vm<real,order>::compute_j_hf( size_t n, size_t q_begin, size_t q_end )
{
    if ( n > conf.Nt )
        throw std::range_error { "Time-step out of range." };

    if ( q_begin == q_end ) return;

    real *cu_coeffs_E_1 = reinterpret_cast<real*>( cuda_coeffs_E_1.get() );
    real *cu_coeffs_E_2 = reinterpret_cast<real*>( cuda_coeffs_E_2.get() );
    real *cu_coeffs_B_3 = reinterpret_cast<real*>( cuda_coeffs_B_3.get() );
    real *cu_j_1    = reinterpret_cast<real*>( cuda_j_1.get() );
    real *cu_j_2    = reinterpret_cast<real*>( cuda_j_2.get() );

    size_t N          = q_end - q_begin;
    size_t block_size = 64;
    size_t Nblocks    = 1 + ( (N-1) / (block_size) );

    size_t j_size = conf.Nx*sizeof(real);

    cuda::set_device( device_number );
    cuda::memset( cu_j_1, 0, j_size );
    cuda::memset( cu_j_2, 0, j_size );

    cuda_eval_j_hf<real,order><<<Nblocks,block_size>>>( n, cu_coeffs_E_1, cu_coeffs_E_2,
    					cu_coeffs_B_3, conf, cu_j_1, cu_j_2, q_begin, q_end );
}

template <typename real, size_t order>
void cuda_kernel_vm<real,order>::download_j_hf( real *j_1, real *j_2 )
{
    real *cu_j_1    = reinterpret_cast<real*>( cuda_j_1.get() );
    real *cu_j_2    = reinterpret_cast<real*>( cuda_j_2.get() );
    size_t N        = conf.Nx;

    cuda::set_device(device_number);
    cuda::memcpy_to_host( tmp_j_1.get(), cu_j_1, sizeof(real)*N );
    cuda::memcpy_to_host( tmp_j_2.get(), cu_j_2, sizeof(real)*N );

    for ( size_t i = 0; i < N; ++i )
    {
    	// To paralellize on several GPU's, we have to
    	// spread j on several GPU's and later add up.
    	// Thus this is not a "=" but a "+=".
        j_1[ i ] += tmp_j_1[ i ];
    	j_2[ i ] += tmp_j_2[ i ];
    }
}

template <typename real, size_t order>
void cuda_kernel_vm<real,order>::upload_coeffs( size_t n, const real *coeffs_E_1,
							const real *coeffs_E_2, const real *coeffs_B_3)
{
    size_t stride_n = (conf.Nx + order - 1);
    real *cu_coeffs_E_1 = reinterpret_cast<real*>( cuda_coeffs_E_1.get() );
    real *cu_coeffs_E_2 = reinterpret_cast<real*>( cuda_coeffs_E_2.get() );
    real *cu_coeffs_B_3 = reinterpret_cast<real*>( cuda_coeffs_B_3.get() );

    cuda::set_device(device_number);
    cuda::memcpy_to_device( cu_coeffs_E_1 + n*stride_n,
    				coeffs_E_1 + n*stride_n, sizeof(real)*stride_n );

    cuda::memcpy_to_device( cu_coeffs_E_2 + n*stride_n,
    				coeffs_E_2 + n*stride_n, sizeof(real)*stride_n );

    cuda::memcpy_to_device( cu_coeffs_B_3 + n*stride_n,
    				coeffs_B_3 + n*stride_n, sizeof(real)*stride_n );
}

template <typename real, size_t order>
void cuda_kernel_vm<real,order>::compute_metrics( size_t n, size_t q_begin, size_t q_end )
{
    if ( q_begin == q_end ) return;

    real *cu_coeffs_E_1  = reinterpret_cast<real*>( cuda_coeffs_E_1.get() );
    real *cu_coeffs_E_2  = reinterpret_cast<real*>( cuda_coeffs_E_2.get() );
    real *cu_coeffs_B_3  = reinterpret_cast<real*>( cuda_coeffs_B_3.get() );
    real *cu_metrics = reinterpret_cast<real*>( cuda_metrics.get() );

    size_t N = q_end - q_begin;
    size_t block_size = 64;
    size_t Nblocks = 1 + (N-1) / block_size;

    cuda::set_device(device_number);
    cuda::memset( cu_metrics, 0, 4*sizeof(real) );

    cuda_eval_metrics<real,order><<<Nblocks,block_size>>>( n, cu_coeffs_E_1,
    			cu_coeffs_E_2, cu_coeffs_B_3, conf, cu_metrics, q_begin, q_end );
}

template <typename real, size_t order>
void cuda_kernel_vm<real,order>::download_metrics( real *metrics )
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

template class cuda_kernel_vm<double,3>;
template class cuda_kernel_vm<double,4>;
template class cuda_kernel_vm<double,5>;
template class cuda_kernel_vm<double,6>;
template class cuda_kernel_vm<double,7>;
template class cuda_kernel_vm<double,8>;

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

