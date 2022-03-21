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

#include <limits>

#include <dergeraet/gmres.hpp>
#include <dergeraet/config.hpp>
#include <dergeraet/splines.hpp>

namespace dergeraet
{

namespace dim1
{

template <typename real, size_t order, size_t dx = 0>
real eval( real x, const real *coeffs, const config_t<real> &config ) noexcept
{
    using std::floor;

    // Get "periodic position" in box at origin.
    x = x - config.Lx * floor( x*config.Lx_inv ); 

    // Knot number
    real x_knot = floor( x*config.dx_inv ); 

    size_t ii = static_cast<size_t>(x_knot);

    // Convert x to reference coordinates.
    x = x*config.dx_inv - x_knot;

    return splines1d::eval<real,order,dx>( x, coeffs + ii );
}

template <typename real, size_t order>
void interpolate( real *coeffs, const real *values, const config_t<real> &config )
{
    std::unique_ptr<real[]> tmp { new real[ config.Nx ] };

    for ( size_t i = 0; i < config.Nx; ++i )
        tmp[ i ] = coeffs[ i ];

    struct mat_t
    {
        const config_t<real> &config;
        real  N[ order ];

        mat_t( const config_t<real> &conf ): config { conf }
        {
            splines1d::N<real,order>(0,N);
        }

        void operator()( const real *in,  size_t stride_in,
                               real *out, size_t stride_out ) const
        {
            if ( stride_in != 1 || stride_out != 1 )
                throw std::runtime_error { "dergeraet::fields::interpolate: Expected stride 1." };

            #pragma omp parallel for 
            for ( size_t i = 0; i < config.Nx; ++i )
            {
                real result = 0;
                if ( i + order <= config.Nx )
                {
                    for ( size_t ii = 0; ii < order; ++ii )
                        result += N[ii] * in[ i + ii ];
                }
                else
                {
                    for ( size_t ii = 0; ii < order; ++ii )
                        result += N[ii]*in[ (i+ii) % config.Nx ];
                }
                out[ i ] = result;
            }
        }
    };

    mat_t M { config };
    gmres_config<real> opt; opt.print_frequency = 0; opt.max_iter = 64;
    gmres<real,mat_t>( config.Nx, tmp.get(), 1, values, 1, M, opt ); 

    for ( size_t i = 0; i < config.Nx + order - 1; ++i )
        coeffs[ i ] = tmp[ i % config.Nx ];
}

}

namespace dim2
{

template <typename real, size_t order, size_t dx = 0, size_t dy = 0>
real eval( real x, real y, const real *coeffs, const config_t<real> &config )
{
    using std::floor;

    // Get "periodic position" in box at origin.
    x = x - config.Lx * floor( x*config.Lx_inv ); 
    y = y - config.Ly * floor( y*config.Ly_inv ); 

    // Knot number
    real x_knot = floor( x*config.dx_inv ); 
    real y_knot = floor( y*config.dy_inv ); 

    size_t ii = static_cast<size_t>(x_knot);
    size_t jj = static_cast<size_t>(y_knot);

    // Convert x to reference coordinates.
    x = x*config.dx_inv - x_knot;
    y = y*config.dy_inv - y_knot;

    const size_t stride_x = 1;
    const size_t stride_y = config.Nx + order - 1;
    coeffs += jj*stride_y + ii;
     
    return splines2d::eval<real,order,dx,dy>( x, y, coeffs, stride_y, stride_x );
}

template <typename real, size_t order>
void interpolate( real *coeffs, const real *values, const config_t<real> &config )
{
    std::unique_ptr<real[]> tmp { new real[ config.Nx * config.Ny ] };

    size_t stride_x = 1;
    size_t stride_y =  config.Nx + order - 1;

    for ( size_t j = 0; j < config.Ny; ++j )
    for ( size_t i = 0; i < config.Nx; ++i )
    {
        tmp [ j*config.Nx + i ] = coeffs[ j*stride_y + i*stride_x ];
    }

    struct mat_t
    {
        const config_t<real> &config;
        real  N[ order ];

        mat_t( const config_t<real> &conf ): config { conf }
        {
            splines1d::N<real,order>(0,N);
        }

        void operator()( const real *in,  size_t stride_in,
                               real *out, size_t stride_out ) const
        {
            if ( stride_in != 1 || stride_out != 1 )
                throw std::runtime_error { "dergeraet::fields::interpolate: Expected stride 1." };

            #pragma omp parallel for 
            for ( size_t l = 0; l < config.Nx*config.Ny; ++l )
            {
                size_t j =  l / config.Nx;
                size_t i =  l % config.Nx;

                if ( i + order <= config.Nx && j + order <= config.Ny )
                {
                    const real *c = in + j*config.Nx + i;
                    real result = 0;
                    for ( size_t jj = 0; jj < order; ++jj )
                    for ( size_t ii = 0; ii < order; ++ii )
                    {
                        result += N[jj]*N[ii]*c[ jj*config.Nx + ii ];
                    }
                    out[ l ] = result;
                }
                else
                {
                    real result = 0;
                    for ( size_t jj = 0; jj < order; ++jj )
                    for ( size_t ii = 0; ii < order; ++ii )
                    {
                        result += N[jj]*N[ii]*in[ ( (j+jj) % config.Ny )*config.Nx+
                                                  ( (i+ii) % config.Nx ) ];
                    }
                    out[ l ] = result;
                }
            }
        }
    };


    mat_t M { config };

    gmres_config<real> opt; opt.max_iter = 500; opt.target_residual = 1e-12;
    gmres<real,mat_t>( config.Nx*config.Ny, tmp.get(), 1, values, 1, M, opt ); 

    for ( size_t j = 0; j < config.Ny + order - 1; ++j )
    for ( size_t i = 0; i < config.Nx + order - 1; ++i )
    {
        coeffs[ j*stride_y + i*stride_x ] = tmp[ (j%config.Ny)*config.Nx +
                                                 (i%config.Nx) ];
    }
}

}


namespace dim3
{

template <typename real, size_t order, size_t dx = 0, size_t dy = 0, size_t dz = 0>
real eval( real x, real y, real z, const real *coeffs, const config_t<real> &config )
{
    using std::floor;

    // Get "periodic position" in box at origin.
    x = x - config.Lx * floor( x*config.Lx_inv ); 
    y = y - config.Ly * floor( y*config.Ly_inv ); 
    z = z - config.Lz * floor( z*config.Lz_inv ); 

    // Knot number
    real x_knot = floor( x*config.dx_inv ); 
    real y_knot = floor( y*config.dy_inv ); 
    real z_knot = floor( z*config.dz_inv ); 

    size_t ii = static_cast<size_t>(x_knot);
    size_t jj = static_cast<size_t>(y_knot);
    size_t kk = static_cast<size_t>(z_knot);

    // Convert x to reference coordinates.
    x = x*config.dx_inv - x_knot;
    y = y*config.dy_inv - y_knot;
    z = z*config.dz_inv - z_knot;

    const size_t stride_x = 1;
    const size_t stride_y =  config.Nx + order - 1;
    const size_t stride_z = (config.Ny + order - 1)*stride_y;

    coeffs += kk*stride_z + jj*stride_y + ii*stride_x;
    return splines3d::eval<real,order,dx,dy,dz>( x, y, z, coeffs, stride_z, stride_y, stride_x );
}

template <typename real, size_t order>
void interpolate( real *coeffs, const real *values, const config_t<real> &config )
{
    std::unique_ptr<real[]> tmp { new real[ config.Nx * config.Ny * config.Nz ] };

    size_t stride_x = 1;
    size_t stride_y =  config.Nx + order - 1;
    size_t stride_z = (config.Ny + order - 1)*stride_y;

    for ( size_t k = 0; k < config.Nz; ++k )
    for ( size_t j = 0; j < config.Ny; ++j )
    for ( size_t i = 0; i < config.Nx; ++i )
    {
        tmp [ k*config.Ny*config.Nx +
              j*config.Nx + i ] = coeffs[ k*stride_z + j*stride_y + i*stride_x ];
    }

    struct mat_t
    {
        const config_t<real> &config;
        real  N[ order ];

        mat_t( const config_t<real> &conf ): config { conf }
        {
            splines1d::N<real,order>(0,N);
        }

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

                if ( i + order <= config.Nx && j + order <= config.Ny && k + order <= config.Nz )
                {
                    const real *c = in + k*config.Ny*config.Nx + j*config.Nx + i;
                    real result = 0;
                    for ( size_t kk = 0; kk < order; ++kk )
                    for ( size_t jj = 0; jj < order; ++jj )
                    for ( size_t ii = 0; ii < order; ++ii )
                    {
                        result += N[kk]*N[jj]*N[ii]*c[ kk*config.Ny*config.Nx + jj*config.Nx + ii ];
                    }
                    out[ l ] = result;
                }
                else
                {
                    real result = 0;
                    for ( size_t kk = 0; kk < order; ++kk )
                    for ( size_t jj = 0; jj < order; ++jj )
                    for ( size_t ii = 0; ii < order; ++ii )
                    {
                        result += N[kk]*N[jj]*N[ii]*in[ ( (k+kk) % config.Nz )*config.Ny*config.Nx + 
                                                        ( (j+jj) % config.Ny )*config.Nx+
                                                        ( (i+ii) % config.Nx ) ];
                    }
                    out[ l ] = result;
                }
            }
        }
    };

    mat_t M { config };

    gmres_config<real> opt;
    opt.target_residual = std::numeric_limits<real>::epsilon() * 128;
    gmres<real,mat_t>( config.Nx*config.Ny*config.Nz, tmp.get(), 1, values, 1, M, opt ); 

    for ( size_t k = 0; k < config.Nz + order - 1; ++k )
    for ( size_t j = 0; j < config.Ny + order - 1; ++j )
    for ( size_t i = 0; i < config.Nx + order - 1; ++i )
    {
        coeffs[ k*stride_z + j*stride_y + i*stride_x ] = tmp[ (k%config.Nz)*config.Ny*config.Nx + 
                                                              (j%config.Ny)*config.Nx +
                                                              (i%config.Nx) ];
       
    }
}

}

}

#endif

