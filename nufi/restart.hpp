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

#include <cmath>
#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <functional>
#include <sstream>

#include <armadillo>

namespace nufi
{

namespace restart
{

// Randomized SVD with on-the-fly A*x and A^T*x evaluation
void randomized_svd_old(
    std::function<arma::vec(const arma::vec&)> apply_A,
    std::function<arma::vec(const arma::vec&)> apply_At,
    arma::uword m, arma::uword n, arma::uword k,
    arma::mat& U, arma::vec& S, arma::mat& V,
    arma::uword oversampling = 10
) {
    arma::uword l = k + oversampling;

    // Step 1: Draw a random test matrix Omega
    arma::mat Omega = arma::randn(n, l);  // shape: n × l

    // Step 2: Compute Y = A * Omega
    arma::mat Y(m, l);
    for (arma::uword i = 0; i < l; ++i)
        Y.col(i) = apply_A(Omega.col(i));

    // Step 3: Orthonormalize Y to get Q
    arma::mat Q;
    arma::mat R;
    arma::qr_econ(Q, R, Y);  // economy QR

    // Step 4: B = Q^T * A
    arma::mat B(l, n);  // Q^T * A ≈ (l x m) * (m x n) = (l x n)
    for (arma::uword i = 0; i < n; ++i) {
        arma::vec e_i = arma::zeros<arma::vec>(n);
        e_i(i) = 1.0;
        arma::vec Ai = apply_A(e_i);
        B.col(i) = Q.t() * Ai;
    }

    // Step 5: SVD of the small matrix B
    arma::mat U_tilde, V_temp;
    arma::vec S_temp;
    arma::svd(U_tilde, S_temp, V_temp, B);  // B = U_tilde * S * V_temp^T

    // Step 6: Recover U = Q * U_tilde
    U = Q * U_tilde;
    S = S_temp.head(k);
    V = V_temp.cols(0, k - 1);
    U = U.cols(0, k - 1);  // truncate U too
}

void randomized_svd_new(
    std::function<arma::vec(const arma::vec&)> apply_A,
    std::function<arma::vec(const arma::vec&)> apply_At,
    arma::uword m, arma::uword n, arma::uword k,
    arma::mat& U, arma::vec& S, arma::mat& V,
    arma::uword oversampling = 10
) {
    arma::uword l = k + oversampling;

    // Step 1: Draw a random test matrix Omega
    arma::mat Omega = arma::randn(n, l);  // shape: n × l

    // Step 2: Compute Y = A * Omega
    arma::mat Y(m, l);
    for (arma::uword i = 0; i < l; ++i)
        Y.col(i) = apply_A(Omega.col(i));

    // Step 3: Orthonormalize Y to get Q
    arma::mat Q;
    arma::mat R;
    arma::qr_econ(Q, R, Y);  // economy QR

    // Step 4: Compute B^T = A^T * Q  (transpose formulation)
    arma::mat Bt(n, l);  // B^T ∈ ℝ^{n×l}
    for (arma::uword j = 0; j < l; ++j)
        Bt.col(j) = apply_At(Q.col(j));

    // Step 5: SVD of the small matrix B^T
    arma::mat V_temp, U_tilde;
    arma::vec S_temp;
    arma::svd(V_temp, S_temp, U_tilde, Bt);  // Bt = V * S * U_tilde^T

    // Step 6: Recover U = Q * U_tilde
    U = Q * U_tilde.cols(0, k - 1);
    S = S_temp.head(k);
    V = V_temp.cols(0, k - 1);
}


}

}