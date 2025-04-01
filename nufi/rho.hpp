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

#ifndef NUFI_RHO_HPP
#define NUFI_RHO_HPP

#include <armadillo>

#include <nufi/fields.hpp>
#include <nufi/stopwatch.hpp>

namespace nufi
{
namespace dim1
{
namespace periodic
{
template <typename real, size_t order>
real eval_ftilda( size_t n, real x, real u,
                  const real *coeffs, const config_t<real> &conf, 
                  bool is_electron = true )
{
    if ( n == 0 ) return conf.f0(x,u);

    const size_t stride_x = 1;
    const size_t stride_t = stride_x*(conf.Nx + order - 1);

    real Ex;
    const real *c;

    real q = (-1)*(is_electron) + (!is_electron)/conf.Mr;

    // We omit the initial half-step.

    while ( --n )
    {
        x  = x - conf.dt*u;
        c  = coeffs + n*stride_t;
        Ex = q*eval<real,order,1>(x,c,conf);
        u  = u + conf.dt*Ex;
    }

    // The final half-step.
    x -= conf.dt*u;
    c  = coeffs + n*stride_t;
    Ex = q*eval<real,order,1>(x,c,conf);
    u += 0.5*conf.dt*Ex;

    return conf.f0(x,u);
}

template <typename real, size_t order>
real eval_f( size_t n, real x, real u, 
             const real *coeffs, const config_t<real> &conf, 
             bool is_electron = true )
{
    if ( n == 0 ) return conf.f0(x,u);

    const size_t stride_x = 1;
    const size_t stride_t = stride_x*(conf.Nx + order - 1);

    real Ex;
    const real *c;

    real q = (-1)*(is_electron) + (!is_electron)/conf.Mr;

    // Initial half-step.
    c  = coeffs + n*stride_t;
    Ex = q*eval<real,order,1>( x, c, conf );
    u += 0.5*conf.dt*Ex;

    while ( --n )
    {
        x -= conf.dt*u;
        c  = coeffs + n*stride_t;
        Ex = q*eval<real,order,1>( x, c, conf );
        u += conf.dt*Ex;
    }

    // Final half-step.
    x -= conf.dt*u;
    c  = coeffs + n*stride_t;
    Ex = q*eval<real,order,1>( x, c, conf );
    u += 0.5*conf.dt*Ex;

    return conf.f0(x,u);
}

template <typename real, size_t order>
real eval_f_on_grid( size_t n, size_t index_x, size_t index_u,
             const real *coeffs, const config_t<real> &conf, bool is_electron = true )
{
	real x = conf.x_min + index_x*conf.dx;
	real u = conf.u_min + index_u*conf.du;

    if ( n == 0 ) return conf.f0(x,u);

    const size_t stride_x = 1;
    const size_t stride_t = stride_x*(conf.Nx + order - 1);

    real Ex;
    const real *c;

    real q = (-1)*(is_electron) + (!is_electron)/conf.Mr;

    // Initial half-step.
    c  = coeffs + n*stride_t;
    Ex = q*eval<real,order,1>( x, c, conf );
    u += 0.5*conf.dt*Ex;

    while ( --n )
    {
        x -= conf.dt*u;
        c  = coeffs + n*stride_t;
        Ex = q*eval<real,order,1>( x, c, conf );
        u += conf.dt*Ex;
    }

    // Final half-step.
    x -= conf.dt*u;
    c  = coeffs + n*stride_t;
    Ex = q*eval<real,order,1>( x, c, conf );
    u += 0.5*conf.dt*Ex;

    return conf.f0(x,u);
}


template <typename real, size_t order>
real eval_rho( size_t n, size_t i, const real *coeffs, const config_t<real> &conf )
{
    const real x = conf.x_min + i*conf.dx; 
    const real du = (conf.u_max-conf.u_min) / conf.Nu;
    const real u_min = conf.u_min + 0.5*du;

    real rho = 0;
    for ( size_t ii = 0; ii < conf.Nu; ++ii )
        rho += eval_ftilda<real,order>( n, x, u_min + ii*du, coeffs, conf );
    rho = 1 - du*rho; 

    return rho;
}

template <typename real, size_t order>
real eval_rho_single_species( size_t n, size_t i, const real *coeffs, const config_t<real> &conf, 
                                bool is_electron = true )
{
    const real x = conf.x_min + i*conf.dx; 
    const real du = (conf.u_max-conf.u_min) / conf.Nu;
    const real u_min = conf.u_min + 0.5*du;

    real rho = 0;
    for ( size_t ii = 0; ii < conf.Nu; ++ii )
    {
        rho += eval_ftilda<real,order>( n, x, u_min + ii*du, coeffs, conf, is_electron );
    }
    rho *= du; 

    return rho;
}

}
}


namespace dim2
{


template <typename real, size_t order>
real eval_ftilda( size_t n, real x, real y, real u, real v,
                  const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return conf.f0(x,y,u,v);

    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_t = stride_y*(conf.Ny + order - 1);

    real Ex, Ey;
    const real *c;

    // We omit the initial half-step.

    while ( --n )
    {
        x -= conf.dt*u;
        y -= conf.dt*v;

        c  = coeffs + n*stride_t;
        Ex = -eval<real,order,1,0>( x, y, c, conf );
        Ey = -eval<real,order,0,1>( x, y, c, conf );

        u += conf.dt*Ex;
        v += conf.dt*Ey;
    }

    // The final half-step.
    x -= conf.dt*u;
    y -= conf.dt*v;

    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0>( x, y, c, conf );
    Ey = -eval<real,order,0,1>( x, y, c, conf );

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;

    return conf.f0(x,y,u,v);
}

template <typename real, size_t order>
real eval_f( size_t n, real x, real y, real u, real v,
             const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return conf.f0(x,y,u,v);

    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_t = stride_y*(conf.Ny + order - 1);

    real Ex, Ey;
    const real *c;

    // Initial half-step.
    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0>( x, y, c, conf );
    Ey = -eval<real,order,0,1>( x, y, c, conf );

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;

    while ( --n )
    {
        x -= conf.dt*u;
        y -= conf.dt*v;

        c  = coeffs + n*stride_t;
        Ex = -eval<real,order,1,0>( x, y, c, conf );
        Ey = -eval<real,order,0,1>( x, y, c, conf );

        u += conf.dt*Ex;
        v += conf.dt*Ey;
    }

    // Final half-step.
    x -= conf.dt*u;
    y -= conf.dt*v;

    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0>( x, y, c, conf );
    Ey = -eval<real,order,0,1>( x, y, c, conf );

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;

    return conf.f0(x,y,u,v);
}

template <typename real, size_t order>
real eval_f_on_grid( size_t n, size_t index_x, size_t index_y, size_t index_u, size_t index_v,
             const real *coeffs, const config_t<real> &conf )
{
	real x = conf.x_min + index_x*conf.dx;
	real y = conf.y_min + index_y*conf.dy;
	real u = conf.u_min + index_u*conf.du;
	real v = conf.v_min + index_v*conf.dv;

    if ( n == 0 ) return conf.f0(x,y,u,v);

    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_t = stride_y*(conf.Ny + order - 1);

    real Ex, Ey;
    const real *c;

    // Initial half-step.
    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0>( x, y, c, conf );
    Ey = -eval<real,order,0,1>( x, y, c, conf );

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;

    while ( --n )
    {
        x -= conf.dt*u;
        y -= conf.dt*v;

        c  = coeffs + n*stride_t;
        Ex = -eval<real,order,1,0>( x, y, c, conf );
        Ey = -eval<real,order,0,1>( x, y, c, conf );

        u += conf.dt*Ex;
        v += conf.dt*Ey;
    }

    // Final half-step.
    x -= conf.dt*u;
    y -= conf.dt*v;

    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0>( x, y, c, conf );
    Ey = -eval<real,order,0,1>( x, y, c, conf );

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;

    return conf.f0(x,y,u,v);
}

template <typename real, size_t order>
real eval_rho( size_t n, size_t l, const real *coeffs, const config_t<real> &conf )
{
    const size_t i = l % conf.Nx;
    const size_t j = l / conf.Nx;

    const real   x = conf.x_min + i*conf.dx; 
    const real   y = conf.y_min + j*conf.dy; 

    const real du = (conf.u_max - conf.u_min) / conf.Nu;
    const real dv = (conf.v_max - conf.v_min) / conf.Nv;

    const real u_min = conf.u_min + 0.5*du;
    const real v_min = conf.v_min + 0.5*dv;

    real rho = 0;
    for ( size_t jj = 0; jj < conf.Nv; ++jj )
    for ( size_t ii = 0; ii < conf.Nu; ++ii )
    {
        real u = u_min + ii*du;
        real v = v_min + jj*dv;

        rho += eval_ftilda<real,order>( n, x, y, u, v, coeffs, conf );
    }
    rho = 1 - du*dv*rho;

    return rho;
}

}

namespace dim3
{

template <typename real, size_t order>
real eval_ftilda( size_t n, real x, real y, real z,
                            real u, real v, real w,
                  const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return conf.f0(x,y,z,u,v,w);

    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_z = stride_y*(conf.Ny + order - 1);
    const size_t stride_t = stride_z*(conf.Nz + order - 1);

    real Ex, Ey, Ez;
    const real *c;

    // We omit the initial half-step.

    while ( --n )
    {
        x -= conf.dt*u;
        y -= conf.dt*v;
        z -= conf.dt*w;

        c  = coeffs + n*stride_t;
        Ex = -eval<real,order,1,0,0>( x, y, z, c, conf );
        Ey = -eval<real,order,0,1,0>( x, y, z, c, conf );
        Ez = -eval<real,order,0,0,1>( x, y, z, c, conf );

        u += conf.dt*Ex;
        v += conf.dt*Ey;
        w += conf.dt*Ez;
    }

    // The final half-step.
    x -= conf.dt*u;
    y -= conf.dt*v;
    z -= conf.dt*w;

    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0,0>( x, y, z, c, conf );
    Ey = -eval<real,order,0,1,0>( x, y, z, c, conf );
    Ez = -eval<real,order,0,0,1>( x, y, z, c, conf );

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;
    w += 0.5*conf.dt*Ez;

    return conf.f0( x, y, z, u, v, w );
}

template <typename real, size_t order>
real eval_f( size_t n, real x, real y, real z,
                       real u, real v, real w,
             const real *coeffs, const config_t<real> &conf )
{
    if ( n == 0 ) return conf.f0(x,y,z,u,v,w);

    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_z = stride_y*(conf.Ny + order - 1);
    const size_t stride_t = stride_z*(conf.Nz + order - 1);

    real Ex, Ey, Ez;
    const real *c;

    // Initial half-step.
    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0,0>( x, y, z, c, conf );
    Ey = -eval<real,order,0,1,0>( x, y, z, c, conf );
    Ez = -eval<real,order,0,0,1>( x, y, z, c, conf );

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;
    w += 0.5*conf.dt*Ez;

    while ( --n )
    {
        x -= conf.dt*u;
        y -= conf.dt*v;
        z -= conf.dt*w;

        c  = coeffs + n*stride_t;
        Ex = -eval<real,order,1,0,0>( x, y, z, c, conf );
        Ey = -eval<real,order,0,1,0>( x, y, z, c, conf );
        Ez = -eval<real,order,0,0,1>( x, y, z, c, conf );

        u += conf.dt*Ex;
        v += conf.dt*Ey;
        w += conf.dt*Ez;
    }

    // Final half-step.
    x -= conf.dt*u;
    y -= conf.dt*v;
    z -= conf.dt*w;

    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0,0>( x, y, z, c, conf );
    Ey = -eval<real,order,0,1,0>( x, y, z, c, conf );
    Ez = -eval<real,order,0,0,1>( x, y, z, c, conf );

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;
    w += 0.5*conf.dt*Ez;

    return conf.f0(x,y,z,u,v,w);
}

template <typename real, size_t order>
real eval_f( size_t n, size_t index_x, size_t index_y, size_t index_z,
                       size_t index_u, size_t index_v, size_t index_w,
             const real *coeffs, const config_t<real> &conf )
{
	real x = conf.x_min + index_x*conf.dx;
	real y = conf.y_min + index_y*conf.dy;
	real z = conf.z_min + index_z*conf.dz;
	real u = conf.u_min + index_u*conf.du;
	real v = conf.v_min + index_v*conf.dv;
	real w = conf.w_min + index_w*conf.dw;

    if ( n == 0 ) return conf.f0(x,y,z,u,v,w);

    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_z = stride_y*(conf.Ny + order - 1);
    const size_t stride_t = stride_z*(conf.Nz + order - 1);

    real Ex, Ey, Ez;
    const real *c;

    // Initial half-step.
    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0,0>( x, y, z, c, conf );
    Ey = -eval<real,order,0,1,0>( x, y, z, c, conf );
    Ez = -eval<real,order,0,0,1>( x, y, z, c, conf );

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;
    w += 0.5*conf.dt*Ez;

    while ( --n )
    {
        x -= conf.dt*u;
        y -= conf.dt*v;
        z -= conf.dt*w;

        c  = coeffs + n*stride_t;
        Ex = -eval<real,order,1,0,0>( x, y, z, c, conf );
        Ey = -eval<real,order,0,1,0>( x, y, z, c, conf );
        Ez = -eval<real,order,0,0,1>( x, y, z, c, conf );

        u += conf.dt*Ex;
        v += conf.dt*Ey;
        w += conf.dt*Ez;
    }

    // Final half-step.
    x -= conf.dt*u;
    y -= conf.dt*v;
    z -= conf.dt*w;

    c  = coeffs + n*stride_t;
    Ex = -eval<real,order,1,0,0>( x, y, z, c, conf );
    Ey = -eval<real,order,0,1,0>( x, y, z, c, conf );
    Ez = -eval<real,order,0,0,1>( x, y, z, c, conf );

    u += 0.5*conf.dt*Ex;
    v += 0.5*conf.dt*Ey;
    w += 0.5*conf.dt*Ez;

    return conf.f0(x,y,z,u,v,w);
}


template <typename real, size_t order>
real eval_rho( size_t n, size_t l, const real *coeffs, const config_t<real> &conf )
{
    const size_t k   = l   / (conf.Nx * conf.Ny);
    const size_t tmp = l   % (conf.Nx * conf.Ny);
    const size_t j   = tmp / conf.Nx;
    const size_t i   = tmp % conf.Nx;

    const real x = conf.x_min + i*conf.dx; 
    const real y = conf.y_min + j*conf.dy; 
    const real z = conf.z_min + k*conf.dz; 

    const real du = (conf.u_max-conf.u_min) / conf.Nu;
    const real dv = (conf.v_max-conf.v_min) / conf.Nv;
    const real dw = (conf.w_max-conf.w_min) / conf.Nw;

    const real u_min = conf.u_min + 0.5*du;
    const real v_min = conf.v_min + 0.5*dv;
    const real w_min = conf.w_min + 0.5*dw;

    real rho = 0;
    for ( size_t kk = 0; kk < conf.Nw; ++kk )
    for ( size_t jj = 0; jj < conf.Nv; ++jj )
    for ( size_t ii = 0; ii < conf.Nu; ++ii )
    {
        real u = u_min + ii*du;
        real v = v_min + jj*dv;
        real w = w_min + kk*dw;

        rho += eval_ftilda<real,order>( n, x, y, z, u, v, w, coeffs, conf );
    }
    rho = 1 - du*dv*dw*rho;
    
    return rho;
}


template <typename real>
arma::Mat<real> exp_J(real vx, real vy, real vz, real tol = 1e-16)
{
    arma::Col<real> v({vx,vy,vz});

    real theta = arma::norm(v);
    if(theta < tol){
        return arma::Mat<real>(3,3,arma::fill::eye);
    }

    arma::mat J_v(3,3,arma::fill::zeros);
    J_v(0,1) = v(2);
    J_v(0,2) = -v(1);
    J_v(1,0) = -v(2);
    J_v(1,2) = v(0);
    J_v(2,0) = v(1);
    J_v(2,1) = -v(0);

    return arma::Mat<real>(3,3,arma::fill::eye) + std::sin(theta)/theta * J_v + (1-std::cos(theta))/theta * J_v * J_v;
}

template <typename real>
arma::Mat<real> exp_J(const arma::Col<real>& v, real tol = 1e-16)
{
    real theta = arma::norm(v);
    if(theta < tol){
        return arma::Mat<real>(3,3,arma::fill::eye);
    }

    arma::mat J_v(3,3,arma::fill::zeros);
    J_v(0,1) = v(2);
    J_v(0,2) = -v(1);
    J_v(1,0) = -v(2);
    J_v(1,2) = v(0);
    J_v(2,0) = v(1);
    J_v(2,1) = -v(0);

    return arma::Mat<real>(3,3,arma::fill::eye) + std::sin(theta)/theta * J_v + (1-std::cos(theta))/theta * J_v * J_v;
}

template <typename real>
void exp_J(arma::Mat<real>& J_v, const arma::Col<real>& v, real tol = 1e-16)
{
    real theta = arma::norm(v);
    if(theta < tol){
        J_v = arma::Mat<real>(3,3,arma::fill::eye);
    } else {
        J_v(0,0) = 0;
        J_v(0,1) = v(2);
        J_v(0,2) = -v(1);
        J_v(1,0) = -v(2);
        J_v(1,1) = 0;
        J_v(1,2) = v(0);
        J_v(2,0) = v(1);
        J_v(2,1) = -v(0);
        J_v(2,2) = 0;

        J_v = std::sin(theta)/theta * J_v + (1-std::cos(theta))/theta * J_v * J_v;
        J_v(0,0) += 1;
        J_v(1,1) += 1;
        J_v(2,2) += 1;
    }
}

template <typename real, size_t order> 
real eval_f_lie_fBE(size_t n, real x, real y, real z,
    real u, real v, real w, const std::vector<std::vector<real>>& coeffs_E, 
    const std::vector<std::vector<real>>& coeffs_B, const std::vector<std::vector<real>>& coeffs_j_hat, 
    const config_t<real> &conf )
{
    const size_t stride_x = 1;
    const size_t stride_y = stride_x*(conf.Nx + order - 1);
    const size_t stride_z = stride_y*(conf.Ny + order - 1);
    const size_t stride_t = stride_z*(conf.Nz + order - 1);

    // Introduce helper variables.
    arma::Col<real> E2(3,arma::fill::zeros);
    arma::Col<real> B0(3,arma::fill::zeros);
    arma::Col<real> j_hat(3,arma::fill::zeros);

    arma::Col<real> x_vec({x,y,z});
    arma::Col<real> v_vec({u,v,w});

    for(; n > 0; n--){
        stopwatch<double> timer_loop;
        B0(0) = eval<real,order>(x_vec(0), x_vec(1), x_vec(2), coeffs_B[0].data() + (n-1)*stride_t, conf);
        B0(1) = eval<real,order>(x_vec(0), x_vec(1), x_vec(2), coeffs_B[1].data() + (n-1)*stride_t, conf);
        B0(2) = eval<real,order>(x_vec(0), x_vec(1), x_vec(2), coeffs_B[2].data() + (n-1)*stride_t, conf);

        j_hat(0) = eval<real,order>(x_vec(0), x_vec(1), x_vec(2), coeffs_j_hat[0].data() + (n-1)*stride_t, conf);
        j_hat(1) = eval<real,order>(x_vec(0), x_vec(1), x_vec(2), coeffs_j_hat[1].data() + (n-1)*stride_t, conf);
        j_hat(2) = eval<real,order>(x_vec(0), x_vec(1), x_vec(2), coeffs_j_hat[2].data() + (n-1)*stride_t, conf);

        E2(0) = eval<real,order>(x_vec(0), x_vec(1), x_vec(2), coeffs_E[0].data() + (n-1)*stride_t, conf);
        E2(1) = eval<real,order>(x_vec(0), x_vec(1), x_vec(2), coeffs_E[1].data() + (n-1)*stride_t, conf);
        E2(2) = eval<real,order>(x_vec(0), x_vec(1), x_vec(2), coeffs_E[2].data() + (n-1)*stride_t, conf);

        E2 = E2 - conf.dt * conf.q/conf.m * j_hat;

        E2(0) = E2(0) + conf.dt * ( eval<real,order,0,1,0>(x_vec(0), x_vec(1), x_vec(2),coeffs_B[2].data()+(n-1)*stride_t,conf) 
                                    - eval<real,order,0,0,1>(x_vec(0), x_vec(1), x_vec(2),coeffs_B[1].data()+(n-1)*stride_t,conf));
        E2(1) = E2(1) + conf.dt * ( eval<real,order,0,0,1>(x_vec(0), x_vec(1), x_vec(2),coeffs_B[0].data()+(n-1)*stride_t,conf) 
                                    - eval<real,order,1,0,0>(x_vec(0), x_vec(1), x_vec(2),coeffs_B[2].data()+(n-1)*stride_t,conf));
        E2(2) = E2(2) + conf.dt * ( eval<real,order,1,0,0>(x_vec(0), x_vec(1), x_vec(2),coeffs_B[1].data()+(n-1)*stride_t,conf) 
                                    - eval<real,order,0,1,0>(x_vec(0), x_vec(1), x_vec(2),coeffs_B[0].data()+(n-1)*stride_t,conf));

        arma::Mat<real> J_B = exp_J<real>(-conf.dt*conf.q/conf.m*B0);

        v_vec = J_B * (v_vec - conf.dt * conf.q/conf.m * E2);
        x_vec = x_vec - conf.dt * v_vec;
    }

    return conf.f0(x_vec(0), x_vec(1), x_vec(2), v_vec(0), v_vec(1), v_vec(2));
}

template <typename real, size_t order>
void eval_j_hat(size_t n, std::vector<std::vector<real>>& j_hat, 
    const std::vector<std::vector<real>>& coeffs_E, const std::vector<std::vector<real>>& coeffs_B,
    const std::vector<std::vector<real>>& coeffs_j_hat, const config_t<real> &conf )
{
    #pragma omp parallel for
    for(size_t l = 0; l < conf.Nx*conf.Ny*conf.Nz; l++){
        
        size_t iz   = l   / (conf.Nx * conf.Ny);
        size_t tmp  = l   % (conf.Nx * conf.Ny);
        size_t iy   = tmp / conf.Nx;
        size_t ix   = tmp % conf.Nx;
    
        real x = conf.x_min + ix*conf.dx; 
        real y = conf.y_min + iy*conf.dy; 
        real z = conf.z_min + iz*conf.dz; 
        
        j_hat[0][l] = 0;
        j_hat[1][l] = 0;
        j_hat[2][l] = 0;
        for(size_t iu = 0; iu < conf.Nu; iu++)
        for(size_t iv = 0; iv < conf.Nv; iv++)
        for(size_t iw = 0; iw < conf.Nw; iw++){
            real u = conf.u_min + (iu + 0.5) * conf.du;
            real v = conf.v_min + (iv + 0.5) * conf.dv;
            real w = conf.w_min + (iw + 0.5) * conf.dw;

            real f_half = eval_f_lie_fBE<real,order>(n, x - 0.5*conf.dt*u, y - 0.5*conf.dt*v, z - 0.5*conf.dt*w, 
                                                        u, v, w, coeffs_E, coeffs_B, coeffs_j_hat, conf );

            j_hat[0][l] += u*f_half;
            j_hat[1][l] += v*f_half;
            j_hat[2][l] += w*f_half;
        }
        j_hat[0][l] *= conf.du*conf.dv*conf.dw;
        j_hat[1][l] *= conf.du*conf.dv*conf.dw;
        j_hat[2][l] *= conf.du*conf.dv*conf.dw;
    }
}


template<typename real> 
real trilinear_interpolation(real x, real y, real z, 
                            real x0, real y0, real z0, 
                            real dx_inv, real dy_inv, real dz_inv,
                            real c000, real c001, 
                            real c010, real c011, 
                            real c100, real c101,
                            real c110, real c111)
{
    // C_{ix, iy, iz}
    real xd = (x-x0) * dx_inv;
    real yd = (y-y0) * dy_inv;
    real zd = (z-z0) * dz_inv;

    real c00 = c000 * (1-xd) + c100*xd;
    real c01 = c001 * (1-xd) + c101*xd;
    real c10 = c010 * (1-xd) + c110*xd;
    real c11 = c011 * (1-xd) + c111*xd;

    real c0 = c00 * (1-yd) + c10*yd;
    real c1 = c01 * (1-yd) + c11*yd;

    return c0 * (1-zd) + c1*zd;
}

template <typename real>
real eval_field(size_t n, size_t dir, real x, real y, real z, 
                const std::vector<real>& field_values, const config_t<real>& conf,
                bool node_storage )
{
    // We store the field values in an array (5 dim tensor) with the sorting:
    // l = n + Nt * (dir + d * (ix + Ny * (iy + Ny * iz))).
    // Furthermore as we use a staggered grid for the fields the electric field 
    // is stored on the nodes while magnetic field is stored on the cell centers.
    
    // Shift to a box that starts at 0.
    x -= conf.x_min;
    y -= conf.y_min;
    z -= conf.z_min;

    // Get "periodic position" in box at origin.
    x = x - conf.Lx * floor( x*conf.Lx_inv ); 
    y = y - conf.Ly * floor( y*conf.Ly_inv ); 
    z = z - conf.Lz * floor( z*conf.Lz_inv ); 

    size_t l_base = n + conf.Nt * dir;

    if(node_storage){
        // E:
        real x_knot = floor( x*conf.dx_inv ); 
        real y_knot = floor( y*conf.dy_inv ); 
        real z_knot = floor( z*conf.dz_inv );

        size_t ix = static_cast<size_t>(x_knot);
        size_t iy = static_cast<size_t>(y_knot);
        size_t iz = static_cast<size_t>(z_knot);

        size_t ix_1 = (ix+1) % conf.Nx;
        size_t iy_1 = (iy+1) % conf.Ny;
        size_t iz_1 = (iz+1) % conf.Nz;

        // C_{ix, iy, iz}.
        real c000 = field_values[n + conf.Nt * (dir +  3 * (ix + conf.Nx * (iy + conf.Ny * iz)))];
        real c001 = field_values[n + conf.Nt * (dir +  3 * (ix + conf.Nx * (iy + conf.Ny * iz_1)))];
        real c010 = field_values[n + conf.Nt * (dir +  3 * (ix + conf.Nx * (iy_1 + conf.Ny * iz)))];
        real c011 = field_values[n + conf.Nt * (dir +  3 * (ix + conf.Nx * (iy_1 + conf.Ny * iz_1)))];
        real c100 = field_values[n + conf.Nt * (dir +  3 * (ix_1 + conf.Nx * (iy + conf.Ny * iz)))];
        real c101 = field_values[n + conf.Nt * (dir +  3 * (ix_1 + conf.Nx * (iy + conf.Ny * iz_1)))];
        real c110 = field_values[n + conf.Nt * (dir +  3 * (ix_1 + conf.Nx * (iy_1 + conf.Ny * iz)))];
        real c111 = field_values[n + conf.Nt * (dir +  3 * (ix_1 + conf.Nx * (iy_1 + conf.Ny * iz_1)))];

        return trilinear_interpolation<real>(x, y, z, ix*conf.dx, iy*conf.dy, iz*conf.dz,
                                            conf.dx_inv, conf.dy_inv, conf.dz_inv, 
                                            c000, c001, c010, c011, c100, c101, c110, c111);

    } else {
        // B: 
        real x_knot = floor( (x + conf.dx/2.0)*conf.dx_inv ); 
        real y_knot = floor( (y + conf.dy/2.0)*conf.dy_inv ); 
        real z_knot = floor( (z + conf.dz/2.0)*conf.dz_inv );

        int ix = static_cast<int>(x_knot);
        int iy = static_cast<int>(y_knot);
        int iz = static_cast<int>(z_knot);

        size_t ix_1 = (ix+1) % conf.Nx;
        size_t iy_1 = (iy+1) % conf.Ny;
        size_t iz_1 = (iz+1) % conf.Nz;

        // ix could be also (-1) if x is in between 0 and x_{1/2}. 
        // For this case we have to take the modulo but only
        // after computing the correct x0.
        real x0 = (ix + 0.5) * conf.dx;
        real y0 = (iy + 0.5) * conf.dy;
        real z0 = (iz + 0.5) * conf.dz;

        ix = ix % conf.Nx;
        iy = iy % conf.Ny;
        iz = iz % conf.Nz;

        // C_{ix, iy, iz}.
        real c000 = field_values[n + conf.Nt * (dir +  3 * (ix + conf.Nx * (iy + conf.Ny * iz)))];
        real c001 = field_values[n + conf.Nt * (dir +  3 * (ix + conf.Nx * (iy + conf.Ny * iz_1)))];
        real c010 = field_values[n + conf.Nt * (dir +  3 * (ix + conf.Nx * (iy_1 + conf.Ny * iz)))];
        real c011 = field_values[n + conf.Nt * (dir +  3 * (ix + conf.Nx * (iy_1 + conf.Ny * iz_1)))];
        real c100 = field_values[n + conf.Nt * (dir +  3 * (ix_1 + conf.Nx * (iy + conf.Ny * iz)))];
        real c101 = field_values[n + conf.Nt * (dir +  3 * (ix_1 + conf.Nx * (iy + conf.Ny * iz_1)))];
        real c110 = field_values[n + conf.Nt * (dir +  3 * (ix_1 + conf.Nx * (iy_1 + conf.Ny * iz)))];
        real c111 = field_values[n + conf.Nt * (dir +  3 * (ix_1 + conf.Nx * (iy_1 + conf.Ny * iz_1)))];

        return trilinear_interpolation<real>(x, y, z, x0, y0, z0,
                                            conf.dx_inv, conf.dy_inv, conf.dz_inv, 
                                            c000, c001, c010, c011, c100, c101, c110, c111);
    }
}

// Does it make sense to use the above routine to compute derivatives or wouldn't it rather be better to 
// compute and store the derivatives on the grid? 
// Think about how an efficient implemention for the derivatives could look like, keeping in mind that
// I will need to interpolate values inbetween nodes/cell-centers. 

}

}

#endif

