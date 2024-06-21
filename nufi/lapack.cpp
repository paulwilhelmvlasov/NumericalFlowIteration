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

#include <nufi/lapack.hpp>

#include <lapacke.h>

namespace nufi
{

namespace lapack
{

int larfg( size_t n, double *alpha, double *x, size_t incx, double *tau )
{
    return LAPACKE_dlarfg(n,alpha,x,incx,tau);
}

int larfg( size_t n, float *alpha, float *x, size_t incx, float *tau )
{
    return LAPACKE_slarfg(n,alpha,x,incx,tau);
}

int larfx( char side, size_t m, size_t n, double *v, double tau, double *c, size_t ldc, double *work )
{
    return LAPACKE_dlarfx( LAPACK_COL_MAJOR, side, m, n, v, tau, c, ldc, work );
}

int larfx( char side, size_t m, size_t n, float  *v, float  tau, float  *c, size_t ldc, float  *work )
{
    return LAPACKE_slarfx( LAPACK_COL_MAJOR, side, m, n, v, tau, c, ldc, work );
}

int trtrs( char uplo, char trans, char diag, size_t n, size_t nrhs,
            const double *a, size_t lda, double *b, size_t ldb )
{
    return LAPACKE_dtrtrs( LAPACK_COL_MAJOR, uplo, trans, diag, n, nrhs, a, lda, b, ldb );
}

int trtrs( char uplo, char trans, char diag, size_t n, size_t nrhs,
            const float *a, size_t lda, float *b, size_t ldb )
{
    return LAPACKE_strtrs( LAPACK_COL_MAJOR, uplo, trans, diag, n, nrhs, a, lda, b, ldb );
}

int geqp3( size_t m, size_t n, double *A, size_t lda, int *jpvt, double *tau )
{
	return LAPACKE_dgeqp3( LAPACK_COL_MAJOR, m, n, A, lda, jpvt, tau );
}

int geqp3( size_t m, size_t n, float *A, size_t lda, int *jpvt, float *tau )
{
	return LAPACKE_sgeqp3( LAPACK_COL_MAJOR, m, n, A, lda, jpvt, tau );
}

int ormqr( char side, char trans, size_t m, size_t n, size_t k, double *a, size_t lda, double *tau, double *c, size_t ldc )
{
	return LAPACKE_dormqr( LAPACK_COL_MAJOR, side, trans, m, n, k, a, lda, tau, c, ldc );
}

int ormqr( char side, char trans, size_t m, size_t n, size_t k, float  *a, size_t lda, float  *tau, float  *c, size_t ldc )
{
	return LAPACKE_sormqr( LAPACK_COL_MAJOR, side, trans, m, n, k, a, lda, tau, c, ldc );
}

}

}

