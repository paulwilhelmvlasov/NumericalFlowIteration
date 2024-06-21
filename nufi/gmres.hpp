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

#ifndef NUFI_GMRES_HPP
#define NUFI_GMRES_HPP

#include <cmath>
#include <memory>
#include <utility>
#include <cstring>
#include <iomanip>
#include <iostream>

#include <nufi/blas.hpp>
#include <nufi/gmres.hpp>
#include <nufi/lapack.hpp>

namespace nufi
{

template <typename real>
struct gmres_config
{
    bool   relative_residual { true };
    real   target_residual   { 1e-6 };
    size_t max_iter          { 150  };
    size_t print_frequency   { 10 };

    real   residual;
    size_t iterations;
};

template <typename real, typename matmul_t>
void gmres( size_t n,       real *x, size_t stride_x,
                      const real *b, size_t stride_b,
            matmul_t matmul, gmres_config<real> &c );

}

#include <nufi/gmres.tpp>
#endif

