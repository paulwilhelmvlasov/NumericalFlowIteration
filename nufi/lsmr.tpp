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

namespace nufi
{

namespace lsmr_impl
{

template <typename real>
real norm( size_t n, const real *x )
{
    /*
    using std::hypot;

    real result = 0;
    for ( size_t i = 0; i < n; ++i )
        result = hypot(result,x[i]);

    
    return result;
    */

    using blas::nrm2;
    return nrm2(n, x, 1);
}

// Reorthognalise u with respect to the previous vectors in buffer,
// using the modified Gram–Schmidt process. Overwrite the oldest vector
// in buffer when full.
template <typename real>
void reorthogonalise( real *buf, size_t n, size_t buffer_max,  
                      real *u,   size_t iter )
{
    using std::min;
    using blas::dot;
    using blas::axpy;
    using blas::scal;
    using blas::copy;

    size_t n_buffered = min( iter+1, buffer_max );
    for ( size_t i = 0; i < n_buffered; ++i )
    {
        real fac = -dot( n, u, 1, buf + i*n, 1 );
        axpy( n, fac, buf + i*n, 1, u, 1 );
    }

    scal( n, 1/norm(n,u), u, 1 );
    copy( n, u, 1, buf + ((iter+1)%buffer_max)*n, 1 );
}

}

template <typename real, typename mat, typename transposed_mat>
void lsmr( size_t m, size_t n, const mat& A, const transposed_mat& At,
           const real *b, real *x, lsmr_options<real> &S )
{
    using std::min;
    using std::max;
    using std::abs;
    using std::swap;
    using std::hypot;
    using blas::axpy; 
    using blas::scal; 
    using blas::copy;
    using lsmr_impl::norm;
    using lsmr_impl::reorthogonalise;


    // Allocation of buffers.
    size_t max_buf = min(n,m)-1;
    S.reorthogonalise_u = min(S.reorthogonalise_u,max_buf);
    S.reorthogonalise_v = min(S.reorthogonalise_v,max_buf);
    size_t u_buffer_size = max( S.reorthogonalise_u, size_t(1) );
    size_t v_buffer_size = max( S.reorthogonalise_v, size_t(1) );

    std::unique_ptr<real[]> data { new real[ n*( 4 + v_buffer_size )  +
                                             m*( 2 + u_buffer_size ) ] {} };

    real *u     = data.get();
    real *utmp  = u     + m;
    real *ubuf  = utmp  + m;
    real *v     = ubuf  + m*u_buffer_size;
    real *vtmp  = v     + n;
    real *h     = vtmp  + n;
    real *h_bar = h     + n;
    real *vbuf  = h_bar + n;

    At(b,v);
    const real norm_ATb = norm(n,v);


    A(x,u); axpy(m,real(-1),b,1,u,1); 
    scal(m, real(-1), u, 1 );         // u = b - Ax;
   
    real alpha = 0; 
    real beta  = norm(m,u); 

    if ( beta > real(0) )
    {
        scal(m, real(1)/beta, u, 1 ); // u = b - Ax / norm(b-Ax)
        At(u,v);                      // v = At*u
        alpha = norm(n,v);
    }

    if ( alpha > real(0) )
        scal(n, real(1)/alpha, v, 1 ); // v = At*u/norm(At*u)

    copy(n,u,1,ubuf,1);                // u_buf.col(0) = u_buf
    copy(n,v,1,vbuf,1);                // v_buf.col(0) = v
    copy(n,v,1,h,1);                   // h = v

    if ( alpha * beta == real(0) ) return;

    
    real alpha_bar = alpha, zeta_bar = alpha*beta;
    real rho = 1, rho_bar = 1, c_bar = 1, s_bar = 0;
    real c, s, theta, zeta, theta_bar, rho_prev, rho_bar_prev;

    // For estimating the condition number.
    real   sigma_max = 0,   sigma_min = std::numeric_limits<real>::max();
    real rho_bar_max = 0, rho_bar_min = std::numeric_limits<real>::max();

    S.norm_A_estimate = 0;
    for ( S.iter = 0; S.iter < S.max_iter; ++S.iter )
    {
        // Continue the bidiagonalisation.
        A(v,utmp); axpy(m,-alpha,u,1,utmp,1); swap(u,utmp); // u = A*v - alpha*u
        beta = norm(m,u);

        if ( beta > 0 )
        {
            scal(m, real(1)/beta, u, 1 );
            if ( S.reorthogonalise_u )
                reorthogonalise( ubuf, m, u_buffer_size, u, S.iter );

            S.norm_A_estimate = hypot( alpha, S.norm_A_estimate );
            S.norm_A_estimate = hypot( beta , S.norm_A_estimate );

            At(u,vtmp); axpy(n,-beta,v,1,vtmp,1); swap(v,vtmp); // v = At*u - beta*v
            alpha = norm(n,v);

            if ( alpha > 0 )
            {
                scal(n,real(1)/alpha, v, 1 );
                if ( S.reorthogonalise_v )
                    reorthogonalise( vbuf, n, v_buffer_size, v, S.iter );
            }
        }

        // Construct and apply rotation P_k
        rho_prev     = rho;
        rho          = hypot(alpha_bar,beta);
        c            = alpha_bar/rho;
        s            = beta/rho;
        theta        = s*alpha;
        alpha_bar    = c*alpha;

        // Construct and apply rotation \bar{P}_k
        rho_bar_prev = rho_bar;
        if ( S.iter )
        {
            rho_bar_max = max( rho_bar, rho_bar_max );
            rho_bar_min = min( rho_bar, rho_bar_min );
        }
        theta_bar    = s_bar*rho;
        rho_bar      = hypot( c_bar*rho, theta );
        if ( S.iter )
        {
            sigma_max    = max( rho_bar_max, c_bar*rho );
            sigma_min    = min( rho_bar_min, c_bar*rho );
        }
        c_bar        = c_bar * rho/rho_bar;
        s_bar        = theta/rho_bar;
        zeta         = c_bar * zeta_bar;
        zeta_bar     = -s_bar*zeta_bar;


        // Update h, h_bar, x
        scal(n, -(theta_bar*rho)/(rho_prev*rho_bar_prev), h_bar, 1 ) ;
        axpy(n, real(1), h, 1, h_bar, 1 );             // h_bar = h - factor*h_bar

        axpy( n, zeta/(rho*rho_bar), h_bar, 1, x, 1 ); // x += factor * h_bar

        scal(n, -theta/rho, h, 1 );
        axpy(n, real(1), v, 1, h, 1 );                 // h = v - factor*h;

        // Estimate quantities.
        if ( S.relative_residual ) S.residual = abs(zeta_bar)/norm_ATb;
        else                       S.residual = abs(zeta_bar);
        S.cond_estimate = sigma_max / sigma_min;

        if ( S.residual <= S.target_residual )
        {
            if ( S.silent == false )
            {
                std::cout << "LSMR: Iteration:  " << std::setw(4)  << S.iter << ", "
                          << "Residual:  "        << std::setw(12) << std::scientific << S.residual << ", "
                          << "cond estimate:  "   << std::setw(12) << std::scientific << S.cond_estimate << ".\n";
            }
            return;
        }

        if ( S.silent == false && (S.iter%10) == 0 )
        {
            std::cout << "LSMR: Iteration:  " << std::setw(4)  << S.iter << ", "
                      << "Residual:  "        << std::setw(12) << std::scientific << S.residual << ", "
                      << "cond estimate:  "   << std::setw(12) << std::scientific << S.cond_estimate << ".\n";
        }
    }
}

}

