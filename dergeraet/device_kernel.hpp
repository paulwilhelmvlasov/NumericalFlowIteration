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

#ifndef DERGERAET_CUDA_KERNEL_HPP
#define DERGERAET_CUDA_KERNEL_HPP

#include <sycl/sycl.hpp>
#include <dergeraet/config.hpp>

namespace dergeraet
{

namespace dim1
{

template <typename real, size_t order>
class device_kernel
{
public:
    device_kernel( const config_t<real> &conf, sycl::device dev );
    ~device_kernel();

    device_kernel( const device_kernel  &rhs ) = delete;
    device_kernel(       device_kernel &&rhs ) noexcept;
    device_kernel& operator=( const device_kernel  &rhs ) = delete;
    device_kernel& operator=(       device_kernel &&rhs ) noexcept;

    void  compute_rho( size_t n,  size_t q_min, size_t q_max );
    void download_rho( real *rho );
    void   upload_phi( size_t n, const real *coeffs );
    void  compute_metrics( size_t n, size_t q_min, size_t q_max );
    void download_metrics( real *metrics );

private:
    config_t<real> conf;
    sycl::queue Q;

    real *tmp_rho { nullptr }; 
    real *dev_coeffs { nullptr }, *dev_rho { nullptr }, *dev_metrics { nullptr };
};

extern template class device_kernel<double,3>;
extern template class device_kernel<double,4>;
extern template class device_kernel<double,5>;
extern template class device_kernel<double,6>;
extern template class device_kernel<double,7>;
extern template class device_kernel<double,8>;

extern template class device_kernel<float,3>;
extern template class device_kernel<float,4>;
extern template class device_kernel<float,5>;
extern template class device_kernel<float,6>;
extern template class device_kernel<float,7>;
extern template class device_kernel<float,8>;

}

namespace dim2
{

template <typename real, size_t order>
class device_kernel
{
public:
    device_kernel( const config_t<real> &conf, sycl::device dev );
    ~device_kernel();

    device_kernel( const device_kernel  &rhs ) = delete;
    device_kernel(       device_kernel &&rhs ) noexcept;
    device_kernel& operator=( const device_kernel  &rhs ) = delete;
    device_kernel& operator=(       device_kernel &&rhs ) noexcept;

    void  compute_rho( size_t n,  size_t q_min, size_t q_max );
    void download_rho( real *rho );
    void   upload_phi( size_t n, const real *coeffs );
    void  compute_metrics( size_t n, size_t q_min, size_t q_max );
    void download_metrics( real *metrics );

private:
    config_t<real> conf;
    sycl::queue Q;

    real *tmp_rho { nullptr }; 
    real *dev_coeffs { nullptr }, *dev_rho { nullptr }, *dev_metrics { nullptr };
};

extern template class device_kernel<double,3>;
extern template class device_kernel<double,4>;
extern template class device_kernel<double,5>;
extern template class device_kernel<double,6>;
extern template class device_kernel<double,7>;
extern template class device_kernel<double,8>;

extern template class device_kernel<float,3>;
extern template class device_kernel<float,4>;
extern template class device_kernel<float,5>;
extern template class device_kernel<float,6>;
extern template class device_kernel<float,7>;
extern template class device_kernel<float,8>;

}

namespace dim3
{

template <typename real, size_t order>
class device_kernel
{
public:
    device_kernel( const config_t<real> &conf, sycl::device dev );
    ~device_kernel();

    device_kernel( const device_kernel  &rhs ) = delete;
    device_kernel(       device_kernel &&rhs ) noexcept;
    device_kernel& operator=( const device_kernel  &rhs ) = delete;
    device_kernel& operator=(       device_kernel &&rhs ) noexcept;

    void  compute_rho( size_t n,  size_t q_min, size_t q_max );
    void download_rho( real *rho );
    void   upload_phi( size_t n, const real *coeffs );
    void  compute_metrics( size_t n, size_t q_min, size_t q_max );
    void download_metrics( real *metrics );

private:
    config_t<real> conf;
    sycl::queue Q;

    real *tmp_rho { nullptr }; 
    real *dev_coeffs { nullptr }, *dev_rho { nullptr }, *dev_metrics { nullptr };
};

extern template class device_kernel<double,3>;
extern template class device_kernel<double,4>;
extern template class device_kernel<double,5>;
extern template class device_kernel<double,6>;
extern template class device_kernel<double,7>;
extern template class device_kernel<double,8>;

extern template class device_kernel<float,3>;
extern template class device_kernel<float,4>;
extern template class device_kernel<float,5>;
extern template class device_kernel<float,6>;
extern template class device_kernel<float,7>;
extern template class device_kernel<float,8>;

}

}

#endif

