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
#include <nufi/blas.hpp>

#include <cblas.h>

namespace nufi
{

namespace blas
{

double dot( const size_t n, const double *x, size_t incx,
                            const double *y, size_t incy )
{
    return cblas_ddot(n,x,incx,y,incy);
}

float dot( const size_t n, const float *x, size_t incx,
                           const float *y, size_t incy )
{
    return cblas_sdot(n,x,incx,y,incy);
}

void axpy( size_t n, double alpha, const double *x, size_t incx,
                                         double *y, size_t incy )
{
    cblas_daxpy(n,alpha,x,incx,y,incy);
}

void axpy( size_t n, float  alpha, const float  *x, size_t incx,
                                         float  *y, size_t incy )
{
    cblas_saxpy(n,alpha,x,incx,y,incy);
}

void scal( size_t n, double alpha, double *x, size_t incx )
{
    cblas_dscal(n,alpha,x,incx);
}

void scal( size_t n, float  alpha, float  *x, size_t incx )
{
    cblas_sscal(n,alpha,x,incx);
}

void copy( size_t n, const double *x, size_t incx, double *y, size_t incy )
{
    cblas_dcopy(n,x,incx,y,incy);
}

void copy( size_t n, const float  *x, size_t incx, float  *y, size_t incy )
{
    cblas_scopy(n,x,incx,y,incy);
}

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

double nrm2( size_t n, const double *x, size_t incx )
{
    return cblas_dnrm2(n, x, incx);
}

float nrm2( size_t n, const float *x, size_t incx )
{
    return cblas_snrm2(n, x, incx);
}

}

}

