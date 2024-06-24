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
#ifndef NUFI_LAPACK_HPP
#define NUFI_LAPACK_HPP

#include <cstddef>

namespace nufi
{

/*!
 * \brief Convenience wrappers for LAPACK, with overloads for single and double
 *        precision.
 */
namespace lapack
{

int larfg( size_t n, double *alpha, double *x, size_t incx, double *tau );
int larfg( size_t n, float  *alpha,  float *x, size_t incx, float  *tau );

int larfx( char side, size_t m, size_t n, double *v, double tau, double *c, size_t ldc, double *work );
int larfx( char side, size_t m, size_t n, float  *v, float  tau, float  *c, size_t ldc, float  *work );

int trtrs( char uplo, char trans, char diag, size_t n, size_t nrhs,
           const double *a, size_t lda, double *b, size_t ldb );
int trtrs( char uplo, char trans, char diag, size_t n, size_t nrhs,
           const float *a, size_t lda, float *b, size_t ldb );

int geqp3( size_t m, size_t n, double *A, size_t lda, int *jpvt, double *tau );
int geqp3( size_t m, size_t n, float  *A, size_t lda, int *jpvt, float  *tau );

int ormqr( char side, char trans, size_t m, size_t n, size_t k, double *a, size_t lda, double *tau, double *c, size_t ldc );
int ormqr( char side, char trans, size_t m, size_t n, size_t k, float  *a, size_t lda, float  *tau, float  *c, size_t ldc );
}

}

#endif

