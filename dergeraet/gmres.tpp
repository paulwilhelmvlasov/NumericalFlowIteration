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

namespace dergeraet
{

namespace gmres_impl
{

template <typename real> inline
real generate_givens_rotation( real a, real b )
{
    using std::abs;
    using std::hypot;

    real c, s, rho;
    if ( b == 0 )
    {
        c = 1; s = 0;
    }
    else
    {
        real rinv = 1./hypot(a,b);
        c = a*rinv; s = -b*rinv;
    }

    if ( c == 0 )
    {
        rho = 1;
    }
    else if ( abs(s) < abs(c) )
    {
        rho = (c>0) ? s/2 : -s/2;
    }
    else
    {
        rho = (s>0) ? 2/c : -2/c;
    }
    return rho;
}

template <typename real> inline
void givens_rotation_from_rho( real rho, real &c, real &s )
{
    using std::abs;
    using std::sqrt;

    if ( rho == 1 )
    {
        c = 0; s = 1;
    }
    else if ( abs(rho) < 1 )
    {
        s = 2*rho; c = sqrt(1-s*s);
    }
    else
    {
        c = 2/rho; s = sqrt(1-c*c);
    }
}

}

template <typename real, typename matmul_t>
void gmres( size_t n,       real *x, size_t stride_x,
                      const real *b, size_t stride_b,
            matmul_t matmul, gmres_config<real> &config )
{
    using std::abs;
    using std::swap;
    using std::hypot;
    using std::memcpy;
    using std::memset;
    using dergeraet::blas  :: gemv;
    using dergeraet::lapack::larfg;
    using dergeraet::lapack::larfx;
    using dergeraet::lapack::trtrs;
    using dergeraet::gmres_impl::givens_rotation_from_rho;
    using dergeraet::gmres_impl::generate_givens_rotation;

    const size_t max_iter = std::min(n,config.max_iter);

    real norm_b = 0;
    for ( size_t i = 0; i < n; ++i )
        norm_b = hypot( norm_b, b[i*stride_b] );

    if ( norm_b == 0 )
    {
        for ( size_t i = 0; i < n; ++i )
            x[i*stride_x] = 0;
        config.iterations = 0;
        config.residual   = 0;
        return;
    }

    real r, abs_r, c, s, aa, bb;

    std::unique_ptr<real[]> data { new real[ n*(max_iter+3) + 2*max_iter + 2 ] };

    real *qr    = data.get();          // QR work array, dimension n x max_iter
    real *w     = qr + n*max_iter;     // Vector, dimension n.
    real *v     = w  + n;              // Vector, dimension n.
    real *work  = v  + n;              // Vector, dimension n.
    real *tau   = work + n;            // Multipliers of Householder projections.
    real *rho   = tau + max_iter + 1;  // Givens rotations.

    matmul( x, stride_x, qr, 1 );
    for ( size_t i = 0; i < n; ++i )
        qr[i] = b[i*stride_b] - qr[i];

    larfg( n, qr, qr + 1, 1, tau );
    w[0] = qr[0];

    abs_r = abs( w[0] );
        r = abs_r / norm_b;

    if ( config.print_frequency )
    {
        if ( config.relative_residual )
            std::cout << "GMRES: Iteration:  " << std::setw(4)  << 0 << ", "
                      << "Residual:  "         << std::setw(12) << std::scientific << r << ".\n";
        else
            std::cout << "GMRES: Iteration:  " << std::setw(4)  << 0 << ", "
                      << "Residual:  "         << std::setw(12) << std::scientific << abs_r << ".\n";
    }

    if ( config.relative_residual )
    {
        if ( r < config.target_residual )
        {
            config.residual   = r;
            config.iterations = 0;
            return;
        }
        else if ( abs_r < config.target_residual )
        {
            config.residual   = abs_r;
            config.iterations = 0;
            return;
        }
    }

    size_t i; // Iteration number.
    for ( i = 1; i <= max_iter; ++i )
    {
        memset( v, 0, sizeof(real)*n );
        v[i-1] = 1;
        for ( size_t j = i; j-- > 0; )
        {
            r = qr[ j + j*n ]; qr[ j + j*n ] = 1;
            larfx( 'L', n - j, 1, qr + j + j*n, tau[j], v + j, n, work );
            qr[ j + j*n ] = r;
        }
        matmul( v, 1, work, 1 ); swap(work,v); // v = A*v;
        for ( size_t j = 0; j < i; ++j )
        {
            r = qr[ j + j*n ]; qr[ j + j*n ] = 1;
            larfx( 'L', n - j, 1, qr + j + j*n, tau[j], v + j, n, work );
            qr[ j + j*n ] = r;
        }

        if ( i != n        ) larfg( n-i, v+i, v+i+1, 1, tau + i );
        if ( i != max_iter ) memcpy( qr + i*n, v, sizeof(real)*n );

        for ( size_t j = 1; j < i; ++j )
        {
            givens_rotation_from_rho(rho[j-1],c,s);
            aa = v[j-1]; bb = v[j];
            v[j-1] = c*aa - s*bb;
            v[j  ] = s*aa + c*bb;
        }

        if ( i != n )
        {
            rho[i-1] = generate_givens_rotation(v[i-1],v[i]);
            givens_rotation_from_rho(rho[i-1],c,s);
            aa = v[i-1]; bb = v[i];
            v[i-1] = c*aa - s*bb;
            v[i  ] = s*aa + c*bb;

            aa = w[i-1]; bb = w[i];
            w[i-1] = c*aa - s*bb;
            w[i]   = s*aa + c*bb;
            abs_r = std::abs( w[i] );
                r = abs_r / norm_b;
        }
        else
        {
            r = abs_r = 0;
        }

        memcpy( qr + (i-1)*n, v, sizeof(real)*i );

        if ( (r     < config.target_residual &&  config.relative_residual) ||
             (abs_r < config.target_residual && !config.relative_residual)  )
        {
            if ( config.print_frequency )
            {
                if ( config.relative_residual )
                    std::cout << "GMRES: Iteration:  " << std::setw(4)  << i << ", "
                              << "Residual:  "         << std::setw(12) << std::scientific << r << ".\n";
                else 
                    std::cout << "GMRES: Iteration:  " << std::setw(4)  << i << ", "
                              << "Residual:  "         << std::setw(12) << std::scientific << abs_r << ".\n";
            }
            break;
        }

        if ( config.print_frequency && (i % config.print_frequency == 0 || i == max_iter) )
        {
            if ( config.relative_residual )
                std::cout << "GMRES: Iteration:  " << std::setw(4)  << i << ", "
                          << "Residual:  "         << std::setw(12) << std::scientific << r << ".\n";
            else
                std::cout << "GMRES: Iteration:  " << std::setw(4)  << i << ", "
                          << "Residual:  "         << std::setw(12) << std::scientific << abs_r << ".\n";
        }
    }

    if ( i == max_iter + 1 ) --i;

    trtrs( 'U', 'N', 'N', i, 1, qr, n, w, n );


    for ( size_t j = i-1; j < n; ++j )
        qr[ j + (i-1)*n ] *= -tau[i-1];

    memset( qr + (i-1)*n, 0, sizeof(real)*i );
    qr[ i-1 + (i-1)*n ] = 1 - tau[i-1];

    for ( size_t j = i-1; j-- > 0; )
    {
        qr[ j + j*n ] = 1;
        larfx( 'L', n - j, i-(j+1), qr + j + j*n, tau[j], qr + j + (j+1)*n, n, work );

        for ( size_t k = j; k < n; ++k )
            qr[ k + j*n ] *= -tau[j];

        memset( qr+ j*n, 0, sizeof(real)*j );
        qr[ j + j*n ] = 1 - tau[j];

    }

    gemv( 'N', n, i, 1.0, qr, n, w, 1, 1.0, x, stride_x );

    config.iterations = i;
    config.residual   = config.relative_residual ? r : abs_r;
}

}

