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

#include <dergeraet/config.hpp>
#include <dergeraet/cuda_runtime.hpp>

namespace dergeraet
{

namespace dim1
{

template <typename real, size_t order>
class cuda_kernel
{
public:
    cuda_kernel( const config_t<real> &conf, int dev = cuda::get_device() );

    cuda_kernel( const cuda_kernel  &rhs ) = delete;
    cuda_kernel(       cuda_kernel &&rhs ) = default;
    cuda_kernel& operator=( const cuda_kernel  &rhs ) = delete;
    cuda_kernel& operator=(       cuda_kernel &&rhs ) = default;

    void compute_rho( size_t n, const real *coeffs, size_t l_min, size_t l_max );    
    void    load_rho( real *rho, size_t l_min, size_t l_max );    

private:
    config_t<real> conf; int device_number;
    cuda::autoptr cuda_coeffs, cuda_rho;
};

extern template class cuda_kernel<double,3>;
extern template class cuda_kernel<double,4>;
extern template class cuda_kernel<double,5>;
extern template class cuda_kernel<double,6>;
extern template class cuda_kernel<double,7>;
extern template class cuda_kernel<double,8>;

extern template class cuda_kernel<float,3>;
extern template class cuda_kernel<float,4>;
extern template class cuda_kernel<float,5>;
extern template class cuda_kernel<float,6>;
extern template class cuda_kernel<float,7>;
extern template class cuda_kernel<float,8>;

}

namespace dim2
{

template <typename real, size_t order>
class cuda_kernel
{
public:
    cuda_kernel( const config_t<real> &conf, int dev = cuda::get_device() );

    cuda_kernel( const cuda_kernel  &rhs ) = delete;
    cuda_kernel(       cuda_kernel &&rhs ) = default;
    cuda_kernel& operator=( const cuda_kernel  &rhs ) = delete;
    cuda_kernel& operator=(       cuda_kernel &&rhs ) = default;

    void compute_rho( size_t n, const real *coeffs, size_t l_min, size_t l_max );
    void    load_rho( real *rho, size_t l_min, size_t l_max );
    void    load_rho( real *rho, size_t l_min, size_t l_max,
    		real *f_metric_l1_norm, real *f_metric_l2_norm,
    		real *f_metric_entropy, real *f_metric_kinetic_energy);

private:
    config_t<real> conf; int device_number;
    cuda::autoptr cuda_coeffs, cuda_rho;
    cuda::autoptr cuda_f_metric_l1_norm;
    cuda::autoptr cuda_f_metric_l2_norm;
    cuda::autoptr cuda_f_metric_entropy;
    cuda::autoptr cuda_f_metric_kinetic_energy;
};

extern template class cuda_kernel<double,3>;
extern template class cuda_kernel<double,4>;
extern template class cuda_kernel<double,5>;
extern template class cuda_kernel<double,6>;
extern template class cuda_kernel<double,7>;
extern template class cuda_kernel<double,8>;

extern template class cuda_kernel<float,3>;
extern template class cuda_kernel<float,4>;
extern template class cuda_kernel<float,5>;
extern template class cuda_kernel<float,6>;
extern template class cuda_kernel<float,7>;
extern template class cuda_kernel<float,8>;

}

}

#endif

