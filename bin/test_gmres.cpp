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

#include <memory>
#include <iomanip>
#include <iostream>
#include <nufi/gmres.hpp>
#include <nufi/random.hpp>

int main()
{
    constexpr size_t N = 100;
    std::unique_ptr<double[]> mem { new double[ N*(N+3) ] };

    nufi::random_real<double> rand(-1,1);

    double *A = mem.get();
    double *b = A + N*N;
    double *x = b + N;
    double *test = x + N;

    for ( size_t i = 0; i < N*N; ++i )
        A[ i ] = rand();

    for ( size_t i = 0; i < N; ++i )
    {
        x[ i ] = 0;
        b[ i ] = rand();
        A[ i + i*N ] += 10;
    }

    struct matmul_t
    {
        double *A;

        void operator()( const double *x, size_t stride_x,
                               double *v, size_t stride_y ) const noexcept
        {
            for ( size_t i = 0; i < N; ++i )
            {
                double val = 0;
                for ( size_t j = 0; j < N; ++j )
                {
                    val += A[ i + j*N ]*x[ j*stride_x ];
                }
                v[ i*stride_y ] = val;
            }
        }
    };

    matmul_t mat { A };
    nufi::gmres_config<double> opt; opt.target_residual = 1e-9;
    nufi::gmres<double,matmul_t>( N, x, 1, b, 1, mat, opt );

    mat( x, 1, test, 1 );

    double resid = 0, norm_b = 0;
    for ( size_t i = 0; i < N; ++i )
    {
        resid  = hypot( resid, b[i]-test[i] );
        norm_b = hypot( norm_b, b[i] );
    }

    std::cout << "Actual residuals: absolute: " << std::setw(15) << resid << ' '
              << "relative: " << std::setw(15) << resid/norm_b << std::endl;
}

