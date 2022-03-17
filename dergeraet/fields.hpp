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
#ifndef DERGERAET_FIELDS_HPP
#define DERGERAET_FIELDS_HPP

#include <dergeraet/gmres.hpp>
#include <dergeraet/config.hpp>
#include <dergeraet/splines.hpp>

namespace dergeraet
{

namespace dim1
{

template <typename real, size_t order>
real eval( real x, const real *coeffs, const config_t<real> &config ) noexcept
{
    using std::floor;
    auto div_ceil = []( size_t a, size_t b ) noexcept { return a/b + (a%b!=0); };

    // Get "periodic position" in box at origin.
    x = x - config.Lx * floor( x*config.Lx_inv ); 

    // Knot number
    real x_knot = floor( x*config.dx_inv ); 

    // Index of lowest node required for evaluating the spline might be negative.
    // The addition of a multiple of config.nx ensures that we stay positive.
    size_t ii_offset = config.Nx * div_ceil(order-1,config.Nx) + 1 - order;
    size_t ii = static_cast<size_t>(x_knot) + ii_offset;

    real c[ order ];
    for ( size_t i = 0; i < order; ++i )
        c[ i ] = coeffs[ ( (ii+i) % config.Nx ) ];

    // Convert to reference coordinates.
    x = x*config.dx_inv - x_knot;
    return splines1d::eval<real,order>( x, c );
}

template <typename real, size_t order>
void interpolate( real *coeffs, const real *values, const config_t<real> &config )
{
    struct mat_t
    {
        const config_t<real> &config;

        void operator()( const real *in,  size_t stride_in,
                               real *out, size_t stride_out ) const
        {
            if ( stride_in != 1 || stride_out != 1 )
                throw std::runtime_error { "dergeraet::fields::interpolate: Expected stride 1." };

            #pragma omp parallel for
            for ( size_t i = 0; i < config.Nx; ++i )
                out[i] = eval<real,order>( i*config.dx, in, config );
        }
    };

    mat_t M { config };
    gmres_config<real> opt;
    gmres<real,mat_t>( config.Nx, coeffs, 1, values, 1, M, opt ); 
}

}

namespace dim2
{

template <typename real, size_t order>
real eval( real x, real y, const real *coeffs, const config_t<real> &config )
{
    using std::floor;
    auto div_ceil = []( size_t a, size_t b ) noexcept { return a/b + (a%b!=0); };

    // Get "periodic position" in box at origin.
    x = x - config.Lx * floor( x*config.Lx_inv ); 
    y = y - config.Ly * floor( y*config.Ly_inv ); 

    // Knot number
    real x_knot = floor( x*config.dx_inv ); 
    real y_knot = floor( y*config.dy_inv ); 

    // Index of lowest node required for evaluating the spline might be negative.
    // The addition of a multiple of config.nx ensures that we stay positive.
    size_t ii_offset = config.Nx * div_ceil(order-1,config.Nx) + 1 - order;
    size_t jj_offset = config.Ny * div_ceil(order-1,config.Ny) + 1 - order;
    
    size_t ii = static_cast<size_t>(x_knot) + ii_offset;
    size_t jj = static_cast<size_t>(y_knot) + jj_offset;

    real c[ order*order ];
    for ( size_t j = 0; j < order; ++j )
    for ( size_t i = 0; i < order; ++i )
        c[ j*order + i ] = coeffs[ ( (jj+j) % config.Ny )*config.Nx +
                                   ( (ii+i) % config.Nx )  ];

    // Convert x to reference coordinates.
    x = x*config.dx_inv - x_knot;
    y = y*config.dy_inv - y_knot;
    return splines2d::eval<real,order>( x, y, c, order, 1 );
}

template <typename real, size_t order>
void interpolate( real *coeffs, const real *values, const config_t<real> &config )
{
    struct mat_t
    {
        const config_t<real> &config;

        void operator()( const real *in,  size_t stride_in,
                               real *out, size_t stride_out ) const
        {
            if ( stride_in != 1 || stride_out != 1 )
                throw std::runtime_error { "dergeraet::fields::interpolate: Expected stride 1." };

            #pragma omp parallel for
            for ( size_t l = 0; l < config.Nx*config.Ny; ++l )
            {
                size_t j = l / config.Nx;
                size_t i = l % config.Nx;
                out[ l ] = eval<real,order>( i*config.dx, j*config.dy, in, config );
            }
        }
    };

    mat_t M { config };

    gmres_config<real> opt; opt.max_iter = 500; opt.target_residual = 1e-12;
    gmres<real,mat_t>( config.Nx*config.Ny, coeffs, 1, values, 1, M, opt ); 
}

}


namespace dim3
{

template <typename real, size_t order>
real eval( real x, real y, real z, const real *coeffs, const config_t<real> &config )
{
    using std::floor;
    auto div_ceil = []( size_t a, size_t b ) noexcept { return a/b + (a%b!=0); };

    // Get "periodic position" in box at origin.
    x = x - config.Lx * floor( x*config.Lx_inv ); 
    y = y - config.Ly * floor( y*config.Ly_inv ); 
    z = z - config.Lz * floor( z*config.Lz_inv ); 

    // Knot number
    real x_knot = floor( x*config.dx_inv ); 
    real y_knot = floor( y*config.dy_inv ); 
    real z_knot = floor( z*config.dz_inv ); 

    // Index of lowest node required for evaluating the spline might be negative.
    // The addition of a multiple of config.nx ensures that we stay positive.
    size_t ii_offset = config.Nx * div_ceil(order-1,config.Nx) + 1 - order;
    size_t jj_offset = config.Ny * div_ceil(order-1,config.Ny) + 1 - order;
    size_t kk_offset = config.Nz * div_ceil(order-1,config.Nz) + 1 - order;
    
    size_t ii = static_cast<size_t>(x_knot) + ii_offset;
    size_t jj = static_cast<size_t>(y_knot) + jj_offset;
    size_t kk = static_cast<size_t>(z_knot) + kk_offset;

    real c[ order*order*order ];
    for ( size_t k = 0; k < order; ++k )
    for ( size_t j = 0; j < order; ++j )
    for ( size_t i = 0; i < order; ++i )
        c[ k*order*order + j*order + i ] = coeffs[ ( (kk+k) % config.Nz )*config.Nx*config.Ny + 
                                                   ( (jj+j) % config.Ny )*config.Nx +
                                                   ( (ii+i) % config.Nx ) ];

    // Convert x to reference coordinates.
    x = x*config.dx_inv - x_knot;
    y = y*config.dy_inv - y_knot;
    z = z*config.dz_inv - z_knot;
    return splines3d::eval<real,order>( x, y, z, c, order*order, order, 1 );
}

template <typename real, size_t order>
void interpolate( real *coeffs, const real *values, const config_t<real> &config )
{
    struct mat_t
    {
        const config_t<real> &config;

        void operator()( const real *in,  size_t stride_in,
                               real *out, size_t stride_out ) const
        {
            if ( stride_in != 1 || stride_out != 1 )
                throw std::runtime_error { "dergeraet::fields::interpolate: Expected stride 1." };

            #pragma omp parallel for
            for ( size_t l = 0; l < config.Nx*config.Ny*config.Nz; ++l )
            {
                size_t k   = l / (config.Nx*config.Ny);
                size_t tmp = l % (config.Nx*config.Ny);
                size_t j =  tmp / config.Nx;
                size_t i =  tmp % config.Nx;
                out[ l ] = eval<real,order>( i*config.dx, j*config.dy, k*config.dz, in, config );
            }
        }
    };

    mat_t M { config };

    gmres_config<real> opt;
    gmres<real,mat_t>( config.Nx*config.Ny*config.Nz, coeffs, 1, values, 1, M, opt ); 
}

}

}

#endif

