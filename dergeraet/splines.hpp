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
#ifndef DERGERAET_SPLINES_HPP
#define DERGERAET_SPLINES_HPP

#include <cstddef>
#include <dergeraet/cuda_runtime.hpp>

namespace dergeraet
{

namespace splines1d
{

template <typename real>
__host__ __device__
constexpr real faculty( size_t n ) noexcept
{
   return  (n > 1) ? real(n)*faculty<real>(n-1) : real(1);
}

template <typename real, size_t order, size_t derivative = 0>
__host__ __device__
void N( real x, real *result, size_t stride = 1 ) noexcept
{
    static_assert( order > 0, "Splines must have order greater than zero." );
    constexpr int n { order      };
    constexpr int d { derivative };

    if ( derivative >= order )
        for ( size_t i = 0; i < order; ++i )
            result[ i*stride ] = 0;

    if ( n == 1 )
    {
        *result = 1;
        return;
    }

    real v[n]; v[n-1] = 1;
    for ( int k = 1; k < n - d; ++k )
    {
        v[n-k-1] = (1-x)*v[n-k];

        for ( int i = 1-k; i < 0; ++i )
            v[n-1+i] = (x-i)*v[n-1+i] + (k+1+i-x)*v[n+i];

        v[n-1] *= x;
    }

    // Differentiate if necessary.
    for ( size_t j = derivative; j-- > 0;  )
    {
        v[j] = -v[j+1];
        for ( size_t i = j + 1; i < order - 1; ++i )
            v[i] = v[i] - v[i+1];
    }

    constexpr real factor = real(1) / faculty<real>(order-derivative-1);
    for ( size_t i = 0; i < order; ++i )
        result[i*stride] = v[i]*factor;
}

template <typename real, size_t order, size_t derivative = 0>
__host__ __device__
real eval( real x, const real *coefficients, size_t stride = 1 ) noexcept
{
    static_assert( order > 0, "Splines must have order greater than zero." );
    static_assert( order > derivative, "Too high derivative requested." );
    constexpr size_t n { order };
    constexpr size_t d { derivative };

    if ( d >= n ) return 0;
    if ( n == 1 ) return *coefficients;

    // Gather coefficients.
    real c[ order ];
    for ( size_t j = 0; j < order; ++j )
        c[j] = coefficients[ stride * j ];

    // Differentiate if necessary.
    for ( size_t j = 1; j <= d; ++j )
        for ( size_t i = n; i-- > j; )
            c[i] = c[i] - c[i-1];

    // Evaluate using de Boor’s algorithm.
    for ( size_t j = 1; j < n-d; ++j )
        for ( size_t i = n-d; i-- > j; )
            c[d+i] = (x+n-d-1-i)*c[d+i] + (i-j+1-x)*c[d+i-1];

    constexpr real factor = real(1) / faculty<real>(order-derivative-1);
    return factor*c[n-1];
}

}

namespace splines2d
{

template <typename real, size_t order, size_t dx = 0, size_t dy = 0>
__host__ __device__
real eval( real x, real y, const real *coefficients, size_t stride_y, size_t stride_x = 1 ) noexcept
{
    static_assert( order > 0, "Splines must have order greater than zero." );
    constexpr size_t n { order };

    if ( dx >= n ) return 0;
    if ( dy >= n ) return 0;
    if ( n  == 1 ) return *coefficients;

    real c [ order ] {};
    real Nx[ order ];

    splines1d::N<real,order,dx>(x,Nx);
    for ( size_t j = 0; j < order; ++j )
    for ( size_t i = 0; i < order; ++i )
        c[ j ] += coefficients[ j*stride_y + i*stride_x ] * Nx[ i ];

    return splines1d::eval<real,order,dy>(y,c);
}

}

namespace splines3d
{

template <typename real, size_t order, size_t dx = 0, size_t dy = 0, size_t dz = 0>
__host__ __device__
real eval( real x, real y, real z, const real *coefficients, size_t stride_z, size_t stride_y, size_t stride_x = 1 ) noexcept
{
    static_assert( order > 0, "Splines must have order greater than zero." );
    constexpr size_t n { order };

    if ( dx >= n ) return 0;
    if ( dy >= n ) return 0;
    if ( dz >= n ) return 0;
    if ( n  == 1 ) return *coefficients;

    real czy[ order*order ] {};
    real cz [ order ] {};
    real N  [ order ];

    splines1d::N<real,order,dx>(x,N);
    for ( size_t k = 0; k < order; ++k )
    for ( size_t j = 0; j < order; ++j )
    for ( size_t i = 0; i < order; ++i )
        czy[ k*order + j ] += coefficients[ k*stride_z + j*stride_y + i*stride_x ]*N[i];

    splines1d::N<real,order,dy>(y,N);
    for ( size_t k = 0; k < order; ++k )
    for ( size_t j = 0; j < order; ++j )
        cz[ k ]  += czy[ k*order + j ]*N[j];

    return splines1d::eval<real,order,dz>(z,cz);
}

}

}

#endif

