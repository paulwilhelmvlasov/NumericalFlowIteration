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
#include <dergeraet/blas.hpp>

#include <cblas.h>

namespace dergeraet
{

namespace blas
{

void ger( const size_t M, const size_t N, const double alpha,
          const double *X, const size_t incX, const double *Y, const size_t incY,
          double *A, const size_t lda)
{
    cblas_dger( CblasColMajor, M, N, alpha, X, incX, Y, incY, A, lda );
}

void ger( const size_t M, const size_t N, const float alpha,
          const float *X, const size_t incX, const float *Y, const size_t incY,
          float *A, const size_t lda)
{
    cblas_sger( CblasColMajor, M, N, alpha, X, incX, Y, incY, A, lda );
}

void gemv( const char trans, size_t m, size_t n,
		   double alpha, const double *a, size_t lda,
           const double *x, size_t incx, double beta,
           double *y, size_t incy )
{
    if ( trans == 'T' || trans == 'Y' )
    {
        cblas_dgemv( CblasColMajor, CblasTrans, m, n, alpha, a, lda, x, incx, beta, y, incy );
    }
    else
    {
        cblas_dgemv( CblasColMajor, CblasNoTrans, m, n, alpha, a, lda, x, incx, beta, y, incy );
    }
}

void gemv( const char trans, size_t m, size_t n,
		   float alpha, const float *a, size_t lda,
           const float *x, size_t incx, float beta,
           float *y, size_t incy )
{
    if ( trans == 'T' || trans == 'Y' )
    {
        cblas_sgemv( CblasColMajor, CblasTrans, m, n, alpha, a, lda, x, incx, beta, y, incy );
    }
    else
    {
        cblas_sgemv( CblasColMajor, CblasNoTrans, m, n, alpha, a, lda, x, incx, beta, y, incy );
    }
}

}

}

