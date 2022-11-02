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
#ifndef DERGERAET_LSMR_H
#define DERGERAET_LSMR_H

#include <cmath>
#include <limits>
#include <iomanip>
#include <iostream>
#include <dergeraet/blas.hpp>

namespace dergeraet
{

template <typename real>
struct lsmr_options
{
    ///////////
    // INPUT //
    ///////////

    // Whether to print messages to std::cout.
    bool silent            = false;

    // Residual of normal equations AᵀAx = Aᵀb
    bool relative_residual  = true;
    real target_residual    = std::numeric_limits<real>::epsilon();
    size_t      max_iter    = 1000;

    // How many Lánczos vectors to keep for local reorthogonalisation.
    // Choose zero for no reorthogonalisation, pure LSMR.
    // Choose a large value for complete reorthognalisation.
    //
    // In an ideal world without roundoff errors, this would have no effect
    // at all, as the Lánczos vectors would be perfectly orthogonal. In practice
    // this property is lost rather quickly.  One may choose to store some of
    // the most recent Lánczos vectors to enforce this property manually. This
    // increase convergence speed at the cost of additional memory requirements.
    size_t reorthogonalise_u = 50;
    size_t reorthogonalise_v = 50;

    ////////////
    // OUTPUT //
    ////////////

    // Iteration count and reached residual.
    // Estimates of ‖A‖ and cond(A)
    size_t iter; real residual;
    real norm_A_estimate, cond_estimate;
};

template <typename real, typename mat, typename transposed_mat>
void lsmr( size_t m, size_t n, const mat& A, const transposed_mat& At,
           const real *b, real *x, lsmr_options<real> &S );

}

#include <dergeraet/lsmr.tpp>
#endif

