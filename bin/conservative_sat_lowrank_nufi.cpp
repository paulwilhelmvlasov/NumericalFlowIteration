/*
 * Conservative SAT low-rank Vlasov--Poisson solver, NuFI-style driver.
 *
 * This file is meant to sit next to the existing NuFI example code.  It uses
 * one config_t<double> object, constructs poisson<double> from that config, and
 * writes output in the same text-file style as run_restarted_simulation().
 *
 * The phase-space density is stored as
 *
 *     f(x_i,u_j) ~= Ux(i,:) * diag(C) * Uu(j,:)^T
 *
 * throughout the time stepping.  Full matrices are only reconstructed through
 * explicit output/evaluation loops.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <complex>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <armadillo>

#include <nufi/config.hpp>
#include <nufi/poisson.hpp>
#include <nufi/stopwatch.hpp>

namespace nufi
{
namespace dim1
{

// -----------------------------------------------------------------------------
// Parameters: same style as the NuFI-LR run_restarted_simulation() driver.
// -----------------------------------------------------------------------------

size_t Nx = 256;                 // Number of grid points in physical space.
size_t Nu = 256;                 // Number of quadrature points in velocity space.
double dt = 5e-3;                // SAT is more restrictive than the restarted NuFI run.
size_t Nt = static_cast<size_t>(100.0 / dt);

double x_min = 0.0;
double x_max = 4.0 * M_PI;
double Lx = x_max - x_min;

double u_min = -10.0;
double u_max = 10.0;

size_t max_rank = 50;
double tol_rank = 1e-2;

size_t stat_every = 10;
size_t f_output_every = 25*200;
size_t console_every = 100;

std::ofstream truncation_ranks_str;

enum class initial_condition_t
{
    landau,
    twostream,
    twostream_1
};

initial_condition_t initial_condition = initial_condition_t::twostream_1;
double alpha_landau = 1e-2;
double alpha_twostream = 1e-3;
double alpha_twostream_1 = 1e-2;
double k_test = 0.5;
double twostream_v0 = 2.4;

template <typename real>
real f0(real x, real u) noexcept
{
    using std::cos;
    using std::exp;
    using std::sqrt;

    const real pi = static_cast<real>(M_PI);
    const real k = static_cast<real>(k_test);

    switch(initial_condition)
    {
        case initial_condition_t::landau:
        {
            const real alpha = static_cast<real>(alpha_landau);
            return (real(1) + alpha * cos(k * x))
                 * exp(-real(0.5) * u * u) / sqrt(real(2) * pi);
        }

        case initial_condition_t::twostream:
        {
            const real alpha = static_cast<real>(alpha_twostream);
            const real v0 = static_cast<real>(twostream_v0);
            return (real(1) + alpha * cos(k * x))
                 * (exp(-real(0.5) * (u - v0) * (u - v0))
                  + exp(-real(0.5) * (u + v0) * (u + v0)))
                 / (real(2) * sqrt(real(2) * pi));
        }

        case initial_condition_t::twostream_1:
        default:
        {
            const real alpha = static_cast<real>(alpha_twostream_1);
            return (real(1) + alpha * cos(k * x))
                 * u * u * exp(-real(0.5) * u * u) / sqrt(real(2) * pi);
        }
    }
}

// A dummy initial condition is not needed here.  The config uses f0 above, in
// the same style as the NuFI-LR file.
config_t<double> conf(64, 128, 500, 0.1, 0.0, 4.0 * M_PI, -10.0, 10.0, &f0);

// -----------------------------------------------------------------------------
// Low-rank data structures.
// -----------------------------------------------------------------------------

struct low_rank_state
{
    arma::vec C;     // r
    arma::mat Ux;    // Nx x r
    arma::mat Uu;    // Nu x r

    size_t rank() const noexcept
    {
        return static_cast<size_t>(C.n_elem);
    }
};

struct operators_t
{
    arma::vec x;
    arma::vec u;
    arma::vec u_plus;
    arma::vec u_minus;
    arma::mat mom_u;      // columns: 1, u, 0.5*u^2
    double du = 0.0;
};

struct moments_t
{
    arma::vec rho;
    arma::vec J;
    arma::vec kappa;
};

struct conservative_truncation_data
{
    arma::vec w;
    arma::vec sqrt_w;
    arma::vec inv_sqrt_w;
    arma::mat Uu1;

    double n1 = 0.0;
    double nu = 0.0;
    double c = 0.0;
    double nq3 = 0.0;
};

// -----------------------------------------------------------------------------
// Grid and operator helpers.
// -----------------------------------------------------------------------------

operators_t precompute_operators(const config_t<double>& param)
{
    operators_t ops;

    ops.x.set_size(param.Nx);
    for(size_t i = 0; i < param.Nx; ++i)
        ops.x(i) = param.x_min + static_cast<double>(i) * param.dx;

    // Match the MATLAB SAT code exactly: Nu denotes the number of velocity
    // nodes, including both endpoints.  Therefore the SAT quadrature/stencil
    // width is (u_max-u_min)/(Nu-1), not config_t::du.  The latter is used by
    // NuFI interpolation/restart routines, but this SAT discretisation is the
    // direct MATLAB port.
    if(param.Nu < 2)
        throw std::runtime_error("SAT velocity grid requires Nu >= 2.");

    ops.du = (param.u_max - param.u_min) / static_cast<double>(param.Nu - 1);
    ops.u.set_size(param.Nu);
    for(size_t j = 0; j < param.Nu; ++j)
        ops.u(j) = param.u_min + static_cast<double>(j) * ops.du;

    ops.u_plus = arma::clamp(ops.u, 0.0, arma::datum::inf);
    ops.u_minus = arma::clamp(ops.u, -arma::datum::inf, 0.0);

    ops.mom_u.set_size(param.Nu, 3);
    ops.mom_u.col(0).ones();
    ops.mom_u.col(1) = ops.u;
    ops.mom_u.col(2) = 0.5 * arma::square(ops.u);

    return ops;
}

conservative_truncation_data precompute_conservative_truncation_data(
    const operators_t& ops
) {
    conservative_truncation_data data;

    const arma::vec one(ops.u.n_elem, arma::fill::ones);
    const arma::vec u2 = arma::square(ops.u);

    data.w = arma::exp(-0.5 * u2);
    data.sqrt_w = arma::sqrt(data.w);
    data.inv_sqrt_w = 1.0 / data.sqrt_w;

    const arma::vec wq = data.w * ops.du;

    data.n1 = arma::dot(one, wq);
    data.nu = arma::dot(u2, wq);
    data.c = data.nu / data.n1;

    const arma::vec q3 = u2 - data.c;
    data.nq3 = arma::dot(arma::square(q3), wq);

    if(data.n1 <= 0.0 || data.nu <= 0.0 || data.nq3 <= 0.0)
        throw std::runtime_error("Invalid conservative truncation projection norms.");

    data.Uu1.set_size(ops.u.n_elem, 3);
    data.Uu1.col(0) = data.w;
    data.Uu1.col(1) = data.w % ops.u;
    data.Uu1.col(2) = data.w % q3;

    return data;
}

inline arma::uword relative_rank(
    const arma::vec& s,
    double rel_tol,
    arma::uword rmax
) noexcept {
    if(s.n_elem == 0 || s(0) == 0.0 || rmax == 0)
        return 0;

    const double threshold = rel_tol * s(0);
    arma::uword r = 0;

    for(arma::uword i = 0; i < s.n_elem; ++i)
    {
        if(s(i) > threshold)
            r = i + 1;
    }

    return std::min(r, rmax);
}

// -----------------------------------------------------------------------------
// Low-rank algebra.
// -----------------------------------------------------------------------------

void assert_finite(const arma::mat& A, const char* name)
{
    if(!A.is_finite())
        throw std::runtime_error(std::string(name) + " contains NaN or Inf.");
}

void assert_finite(const arma::vec& v, const char* name)
{
    if(!v.is_finite())
        throw std::runtime_error(std::string(name) + " contains NaN or Inf.");
}

low_rank_state standard_truncation_lr(
    const low_rank_state& in,
    double tol,
    size_t rmax
) {
    const arma::uword nx = in.Ux.n_rows;
    const arma::uword nu = in.Uu.n_rows;

    if(in.C.n_elem == 0 || rmax == 0)
        return low_rank_state{arma::vec(), arma::mat(nx, 0), arma::mat(nu, 0)};

    assert_finite(in.Ux, "standard_truncation_lr::Ux");
    assert_finite(in.Uu, "standard_truncation_lr::Uu");
    assert_finite(in.C,  "standard_truncation_lr::C");

    arma::mat L = in.Ux;
    L.each_row() %= in.C.t();
    const arma::mat& R = in.Uu;

    arma::mat QL, RL, QR, RR;
    if(!arma::qr_econ(QL, RL, L) || !arma::qr_econ(QR, RR, R))
        throw std::runtime_error("QR failed in standard_truncation_lr.");

    const arma::mat small = RL * RR.t();
    assert_finite(small, "standard_truncation_lr::small");

    arma::mat Us, Vs;
    arma::vec s;
    if(!arma::svd_econ(Us, s, Vs, small))
        throw std::runtime_error("SVD failed in standard_truncation_lr.");

    const arma::uword r = relative_rank(s, tol, static_cast<arma::uword>(rmax));

    if(r == 0)
        return low_rank_state{arma::vec(), arma::mat(nx, 0), arma::mat(nu, 0)};

    low_rank_state out;
    out.C = s.head(r);
    out.Ux = QL * Us.cols(0, r - 1);
    out.Uu = QR * Vs.cols(0, r - 1);
    return out;
}

inline void copy_lr_block(
    low_rank_state& out,
    arma::uword pos,
    const low_rank_state& in,
    double alpha
) {
    const arma::uword r = in.C.n_elem;
    const arma::uword nx = in.Ux.n_rows;
    const arma::uword nu = in.Uu.n_rows;

    for(arma::uword c = 0; c < r; ++c)
    {
        out.C(pos + c) = alpha * in.C(c);
        std::copy_n(in.Ux.colptr(c), nx, out.Ux.colptr(pos + c));
        std::copy_n(in.Uu.colptr(c), nu, out.Uu.colptr(pos + c));
    }
}

void lr_combine_into(
    low_rank_state& out,
    const low_rank_state& a, double alpha_a,
    const low_rank_state& b, double alpha_b
) {
    const arma::uword total_rank = a.C.n_elem + b.C.n_elem;
    out.C.set_size(total_rank);
    out.Ux.set_size(a.Ux.n_rows, total_rank);
    out.Uu.set_size(a.Uu.n_rows, total_rank);

    copy_lr_block(out, 0, a, alpha_a);
    copy_lr_block(out, a.C.n_elem, b, alpha_b);
}

void lr_combine_into(
    low_rank_state& out,
    const low_rank_state& a, double alpha_a,
    const low_rank_state& b, double alpha_b,
    const low_rank_state& c, double alpha_c
) {
    const arma::uword rb = b.C.n_elem;
    const arma::uword total_rank = a.C.n_elem + rb + c.C.n_elem;
    out.C.set_size(total_rank);
    out.Ux.set_size(a.Ux.n_rows, total_rank);
    out.Uu.set_size(a.Uu.n_rows, total_rank);

    copy_lr_block(out, 0, a, alpha_a);
    copy_lr_block(out, a.C.n_elem, b, alpha_b);
    copy_lr_block(out, a.C.n_elem + rb, c, alpha_c);
}

moments_t moments_lr(const low_rank_state& state, const operators_t& ops)
{
    moments_t m;
    m.rho.zeros(state.Ux.n_rows);
    m.J.zeros(state.Ux.n_rows);
    m.kappa.zeros(state.Ux.n_rows);

    if(state.C.n_elem == 0)
        return m;

    arma::mat B = state.Uu.t() * ops.mom_u;  // r x 3
    B.each_col() %= state.C;                 // diag(C) * B

    const arma::mat macro = ops.du * (state.Ux * B);

    m.rho = macro.col(0);
    m.J = macro.col(1);
    m.kappa = macro.col(2);

    return m;
}

arma::vec electric_field_from_potential(
    const arma::vec& phi,
    const config_t<double>& param
) {
    assert_finite(phi, "electric_field_from_potential::phi");

    const arma::uword n = phi.n_elem;
    arma::cx_vec phi_hat = arma::fft(phi);
    arma::cx_vec E_hat(n, arma::fill::zeros);

    const std::complex<double> I(0.0, 1.0);
    const double k0 = 2.0 * M_PI / param.Lx;

    for(arma::uword ell = 0; ell < n; ++ell)
    {
        // MATLAB ordering for even Nx: [0:Nx/2, -Nx/2+1:-1].
        const long mode = (ell <= n / 2)
                        ? static_cast<long>(ell)
                        : static_cast<long>(ell) - static_cast<long>(n);

        const double k = k0 * static_cast<double>(mode);

        // poisson<double>::solve() is used in NuFI-LR as follows:
        //   poiss.solve(rho.get());
        //   periodic::interpolate(..., rho.get(), conf);
        //   E = periodic::eval<double,order,1>(...);
        // Thus the overwritten buffer is the Poisson potential.  The SAT
        // MATLAB RHS, however, needs E = -d_x phi.
        E_hat(ell) = -I * k * phi_hat(ell);
    }

    arma::vec E = arma::real(arma::ifft(E_hat));
    E -= arma::mean(E);
    assert_finite(E, "electric_field_from_potential::E");
    return E;
}

std::pair<arma::vec, double> poisson_lr(
    const low_rank_state& state,
    const operators_t& ops,
    const config_t<double>& param,
    const poisson<double>& poiss
) {
    arma::vec phi = moments_lr(state, ops).rho;
    assert_finite(phi, "poisson_lr::rho");

    const double electric_energy = poiss.solve(phi.memptr());
    assert_finite(phi, "poisson_lr::phi");

    arma::vec E = electric_field_from_potential(phi, param);

    if(!std::isfinite(electric_energy))
        throw std::runtime_error("poisson_lr::electric_energy is NaN or Inf.");

    return {E, electric_energy};
}

// -----------------------------------------------------------------------------
// Fifth-order upwind finite-volume differentiation of the basis matrices.
// -----------------------------------------------------------------------------

void upwind5_x_periodic_basis_into(
    const arma::mat& U,
    double h,
    bool plus,
    arma::mat& D,
    arma::uword col_offset,
    const std::vector<arma::uword>& im2,
    const std::vector<arma::uword>& im1,
    const std::vector<arma::uword>& ip1,
    const std::vector<arma::uword>& ip2,
    const std::vector<arma::uword>& ip3
) {
    const arma::uword n = U.n_rows;
    const arma::uword r = U.n_cols;

    // D is first used as the numerical flux F and then differentiated in place.
    // Columns are contiguous in Armadillo, so no row temporaries are created.
    for(arma::uword c = 0; c < r; ++c)
    {
        const double* u = U.colptr(c);
        double* d = D.colptr(col_offset + c);

        if(plus)
        {
            for(arma::uword i = 0; i < n; ++i)
            {
                d[i] =  (1.0 / 30.0) * u[im2[i]]
                      - (13.0 / 60.0) * u[im1[i]]
                      + (47.0 / 60.0) * u[i]
                      + (9.0 / 20.0)  * u[ip1[i]]
                      - (1.0 / 20.0)  * u[ip2[i]];
            }
        }
        else
        {
            for(arma::uword i = 0; i < n; ++i)
            {
                d[i] = -(1.0 / 20.0)  * u[im1[i]]
                      + (9.0 / 20.0)  * u[i]
                      + (47.0 / 60.0) * u[ip1[i]]
                      - (13.0 / 60.0) * u[ip2[i]]
                      + (1.0 / 30.0)  * u[ip3[i]];
            }
        }

        double previous = d[n - 1];
        for(arma::uword i = 0; i < n; ++i)
        {
            const double current = d[i];
            d[i] = (current - previous) / h;
            previous = current;
        }
    }
}

void upwind5_u_zero_boundary_basis_into(
    const arma::mat& U,
    double h,
    bool plus,
    arma::mat& D,
    arma::uword col_offset,
    std::vector<double>& flux
) {
    const arma::uword n = U.n_rows;
    const arma::uword r = U.n_cols;

    for(arma::uword c = 0; c < r; ++c)
    {
        const double* u = U.colptr(c);
        double* d = D.colptr(col_offset + c);

        auto zero = [&](long j_matlab) noexcept -> double
        {
            if(j_matlab < 1 || j_matlab > static_cast<long>(n))
                return 0.0;
            return u[static_cast<arma::uword>(j_matlab - 1)];
        };

        if(plus)
        {
            for(arma::uword iface = 0; iface <= n; ++iface)
            {
                const long i = static_cast<long>(iface);
                flux[iface] =  (1.0 / 30.0) * zero(i - 2)
                              - (13.0 / 60.0) * zero(i - 1)
                              + (47.0 / 60.0) * zero(i)
                              + (9.0 / 20.0)  * zero(i + 1)
                              - (1.0 / 20.0)  * zero(i + 2);
            }
        }
        else
        {
            for(arma::uword iface = 0; iface <= n; ++iface)
            {
                const long i = static_cast<long>(iface);
                flux[iface] = -(1.0 / 20.0)  * zero(i - 1)
                              + (9.0 / 20.0)  * zero(i)
                              + (47.0 / 60.0) * zero(i + 1)
                              - (13.0 / 60.0) * zero(i + 2)
                              + (1.0 / 30.0)  * zero(i + 3);
            }
        }

        for(arma::uword i = 0; i < n; ++i)
            d[i] = (flux[i + 1] - flux[i]) / h;
    }
}

// -----------------------------------------------------------------------------
// Vlasov RHS and SSPRK3.
// -----------------------------------------------------------------------------

// Reused SSPRK3 storage.  The largest concatenation is
// state + stage + RHS(stage), hence at most 6*rmax columns.
struct ssprk3_workspace
{
    low_rank_state rhs;
    low_rank_state combined;
    low_rank_state stage;

    std::vector<arma::uword> im2;
    std::vector<arma::uword> im1;
    std::vector<arma::uword> ip1;
    std::vector<arma::uword> ip2;
    std::vector<arma::uword> ip3;
    std::vector<double> flux_u;

    ssprk3_workspace(arma::uword nx, arma::uword nu, size_t rmax)
        : im2(nx), im1(nx), ip1(nx), ip2(nx), ip3(nx), flux_u(nu + 1)
    {
        const arma::uword r = static_cast<arma::uword>(rmax);

        // Preallocate the maximum sizes once.  Subsequent set_size() calls reuse
        // Armadillo's allocation as long as the requested size stays below it.
        rhs.C.set_size(4 * r);
        rhs.Ux.set_size(nx, 4 * r);
        rhs.Uu.set_size(nu, 4 * r);

        combined.C.set_size(6 * r);
        combined.Ux.set_size(nx, 6 * r);
        combined.Uu.set_size(nu, 6 * r);

        stage.C.set_size(r);
        stage.Ux.set_size(nx, r);
        stage.Uu.set_size(nu, r);

        for(arma::uword i = 0; i < nx; ++i)
        {
            im2[i] = (i + nx - 2) % nx;
            im1[i] = (i + nx - 1) % nx;
            ip1[i] = (i + 1) % nx;
            ip2[i] = (i + 2) % nx;
            ip3[i] = (i + 3) % nx;
        }
    }
};

void vlasov_rhs_lr_into(
    const low_rank_state& state,
    const operators_t& ops,
    const config_t<double>& param,
    const poisson<double>& poiss,
    low_rank_state& rhs,
    ssprk3_workspace& work
) {
    const auto poisson_result = poisson_lr(state, ops, param, poiss);
    const arma::vec& E = poisson_result.first;

    const arma::uword r = state.C.n_elem;
    const arma::uword nx = state.Ux.n_rows;
    const arma::uword nu = state.Uu.n_rows;

    rhs.C.set_size(4 * r);
    rhs.Ux.set_size(nx, 4 * r);
    rhs.Uu.set_size(nu, 4 * r);

    // x-advection factors go straight into RHS columns [0,2r).
    upwind5_x_periodic_basis_into(
        state.Ux, param.dx, true, rhs.Ux, 0,
        work.im2, work.im1, work.ip1, work.ip2, work.ip3
    );
    upwind5_x_periodic_basis_into(
        state.Ux, param.dx, false, rhs.Ux, r,
        work.im2, work.im1, work.ip1, work.ip2, work.ip3
    );

    // u-advection factors are simple diagonal row scalings.
    for(arma::uword c = 0; c < r; ++c)
    {
        const double* uu = state.Uu.colptr(c);
        double* up = rhs.Uu.colptr(c);
        double* um = rhs.Uu.colptr(r + c);

        for(arma::uword j = 0; j < nu; ++j)
        {
            up[j] = ops.u_plus(j) * uu[j];
            um[j] = ops.u_minus(j) * uu[j];
        }
    }

    // Electric-field factors are also written directly into the final RHS.
    for(arma::uword c = 0; c < r; ++c)
    {
        const double* ux = state.Ux.colptr(c);
        double* ep = rhs.Ux.colptr(2 * r + c);
        double* em = rhs.Ux.colptr(3 * r + c);

        for(arma::uword i = 0; i < nx; ++i)
        {
            const double e = E(i);
            ep[i] = ((e > 0.0) ? e : 0.0) * ux[i];
            em[i] = ((e < 0.0) ? e : 0.0) * ux[i];
        }
    }

    // Velocity derivatives go straight into RHS columns [2r,4r).
    upwind5_u_zero_boundary_basis_into(
        state.Uu, ops.du, true, rhs.Uu, 2 * r, work.flux_u
    );
    upwind5_u_zero_boundary_basis_into(
        state.Uu, ops.du, false, rhs.Uu, 3 * r, work.flux_u
    );

    for(arma::uword block = 0; block < 4; ++block)
        for(arma::uword c = 0; c < r; ++c)
            rhs.C(block * r + c) = -state.C(c);
}

void conservative_truncation_lr_into(
    const low_rank_state& in,
    const operators_t& ops,
    const conservative_truncation_data& data,
    double tol,
    size_t rmax,
    low_rank_state& out
) {
    const arma::uword nx = in.Ux.n_rows;
    const arma::uword nu = in.Uu.n_rows;

    assert_finite(in.Ux, "conservative_truncation_lr::Ux input");
    assert_finite(in.Uu, "conservative_truncation_lr::Uu input");
    assert_finite(in.C,  "conservative_truncation_lr::C input");

    const moments_t m = moments_lr(in, ops);

    const arma::vec M1 = m.rho / data.n1;
    const arma::vec M2 = m.J / data.nu;
    const arma::vec M3 = (2.0 * m.kappa - data.c * m.rho) / data.nq3;

    arma::mat Ux1(nx, 3);
    Ux1.col(0) = M1;
    Ux1.col(1) = M2;
    Ux1.col(2) = M3;

    const arma::mat& Uu1 = data.Uu1;
    const arma::vec C1(3, arma::fill::ones);

    arma::mat Uu_scaled = in.Uu;
    Uu_scaled.each_col() %= data.inv_sqrt_w;

    arma::mat Uu1_scaled = Uu1;
    Uu1_scaled.each_col() %= data.inv_sqrt_w;

    const arma::mat AL = arma::join_rows(in.Ux, Ux1);
    const arma::mat AR = arma::join_rows(Uu_scaled, Uu1_scaled);
    const arma::vec AC = arma::join_cols(in.C, -C1);

    arma::mat L = AL;
    L.each_row() %= AC.t();
    const arma::mat& R = AR;

    arma::mat QL, RL, QR, RR;
    if(!arma::qr_econ(QL, RL, L) || !arma::qr_econ(QR, RR, R))
        throw std::runtime_error("QR failed in conservative_truncation_lr.");

    const arma::mat small = RL * RR.t();
    assert_finite(small, "conservative_truncation_lr::small");

    arma::mat Us, Vs;
    arma::vec s;
    if(!arma::svd_econ(Us, s, Vs, small))
        throw std::runtime_error("SVD failed in conservative_truncation_lr.");

    const arma::uword max_r2 = (rmax > 3) ? static_cast<arma::uword>(rmax - 3) : 0;
    const arma::uword r2 = relative_rank(s, tol, max_r2);

    const arma::uword rout = 3 + r2;
    out.C.set_size(rout);
    out.Ux.set_size(nx, rout);
    out.Uu.set_size(nu, rout);

    out.C.head(3).ones();
    out.Ux.cols(0, 2) = Ux1;
    out.Uu.cols(0, 2) = Uu1;

    if(r2 > 0)
    {
        out.C.subvec(3, rout - 1) = s.head(r2);
        out.Ux.cols(3, rout - 1) = QL * Us.cols(0, r2 - 1);
        out.Uu.cols(3, rout - 1) = QR * Vs.cols(0, r2 - 1);
        out.Uu.cols(3, rout - 1).each_col() %= data.sqrt_w;
    }
}

void ssprk3_step_lr(
    low_rank_state& state,
    const operators_t& ops,
    const conservative_truncation_data& trunc_data,
    const config_t<double>& param,
    const poisson<double>& poiss,
    double tol,
    size_t rmax,
    ssprk3_workspace& work
) {
    // Stage 1.
    vlasov_rhs_lr_into(state, ops, param, poiss, work.rhs, work);
    lr_combine_into(work.combined, state, 1.0, work.rhs, param.dt);
    conservative_truncation_lr_into(work.combined, ops, trunc_data, tol, rmax, work.stage);

    // Stage 2.  rhs/combined/stage are reused rather than reconstructed.
    vlasov_rhs_lr_into(work.stage, ops, param, poiss, work.rhs, work);
    lr_combine_into(
        work.combined,
        state, 3.0 / 4.0,
        work.stage, 1.0 / 4.0,
        work.rhs, param.dt / 4.0
    );
    conservative_truncation_lr_into(work.combined, ops, trunc_data, tol, rmax, work.stage);

    // Stage 3.
    vlasov_rhs_lr_into(work.stage, ops, param, poiss, work.rhs, work);
    lr_combine_into(
        work.combined,
        state, 1.0 / 3.0,
        work.stage, 2.0 / 3.0,
        work.rhs, 2.0 * param.dt / 3.0
    );
    conservative_truncation_lr_into(work.combined, ops, trunc_data, tol, rmax, work.stage);

    // Swap rather than copy.  work.stage receives the old state allocation and
    // can reuse it on the next timestep.
    std::swap(state.C, work.stage.C);
    std::swap(state.Ux, work.stage.Ux);
    std::swap(state.Uu, work.stage.Uu);
}

// -----------------------------------------------------------------------------
// Initial condition.  This uses the analytic separable form of the same f0 used
// in the NuFI-LR file, so no full initial matrix is formed.
// -----------------------------------------------------------------------------

low_rank_state initial_condition_lr(
    const config_t<double>& param,
    const operators_t& ops,
    double tol,
    size_t rmax
) {
    arma::vec g(ops.u.n_elem);
    double alpha = alpha_twostream_1;

    switch(initial_condition)
    {
        case initial_condition_t::landau:
            alpha = alpha_landau;
            g = arma::exp(-0.5 * arma::square(ops.u)) / std::sqrt(2.0 * M_PI);
            break;

        case initial_condition_t::twostream:
            alpha = alpha_twostream;
            g = (arma::exp(-0.5 * arma::square(ops.u - twostream_v0))
               + arma::exp(-0.5 * arma::square(ops.u + twostream_v0)))
              / (2.0 * std::sqrt(2.0 * M_PI));
            break;

        case initial_condition_t::twostream_1:
        default:
            alpha = alpha_twostream_1;
            g = arma::square(ops.u) % arma::exp(-0.5 * arma::square(ops.u))
              / std::sqrt(2.0 * M_PI);
            break;
    }

    low_rank_state state;
    state.C.set_size(2);
    state.C(0) = 1.0;
    state.C(1) = alpha;

    state.Ux.set_size(param.Nx, 2);
    state.Ux.col(0).ones();
    state.Ux.col(1) = arma::cos(k_test * ops.x);

    state.Uu.set_size(param.Nu, 2);
    state.Uu.col(0) = g;
    state.Uu.col(1) = g;

    return standard_truncation_lr(state, tol, rmax);
}

// -----------------------------------------------------------------------------
// Low-rank point evaluation for output/statistics.  This does not construct the
// full phase-space matrix.
// -----------------------------------------------------------------------------

inline size_t periodic_index(long i, size_t n) noexcept
{
    long r = i % static_cast<long>(n);
    if(r < 0)
        r += static_cast<long>(n);
    return static_cast<size_t>(r);
}

inline size_t clamp_index(long i, size_t n) noexcept
{
    if(i < 0)
        return 0;
    if(i >= static_cast<long>(n))
        return n - 1;
    return static_cast<size_t>(i);
}

inline double cubic_interp(double p0, double p1, double p2, double p3, double t) noexcept
{
    const double a0 = -0.5 * p0 + 1.5 * p1 - 1.5 * p2 + 0.5 * p3;
    const double a1 = p0 - 2.5 * p1 + 2.0 * p2 - 0.5 * p3;
    const double a2 = -0.5 * p0 + 0.5 * p2;
    const double a3 = p1;
    return ((a0 * t + a1) * t + a2) * t + a3;
}

arma::rowvec cubic_periodic_matrix_row(
    const arma::mat& A,
    double x,
    const config_t<double>& param
) {
    x = std::fmod(std::fmod(x - param.x_min, param.Lx) + param.Lx, param.Lx);

    const double gx = x * param.dx_inv;
    const long ix = static_cast<long>(std::floor(gx));
    const double tx = gx - static_cast<double>(ix);

    arma::rowvec out(A.n_cols, arma::fill::zeros);

    for(arma::uword c = 0; c < A.n_cols; ++c)
    {
        const double p0 = A(periodic_index(ix - 1, A.n_rows), c);
        const double p1 = A(periodic_index(ix,     A.n_rows), c);
        const double p2 = A(periodic_index(ix + 1, A.n_rows), c);
        const double p3 = A(periodic_index(ix + 2, A.n_rows), c);
        out(c) = cubic_interp(p0, p1, p2, p3, tx);
    }

    return out;
}

arma::rowvec cubic_clamped_velocity_matrix_row(
    const arma::mat& A,
    double u,
    const config_t<double>& param
) {
    if(u < param.u_min || u > param.u_max)
        return arma::rowvec(A.n_cols, arma::fill::zeros);

    // MATLAB SAT endpoint velocity nodes: u_j = u_min + j*hu.
    const double hu = (param.u_max - param.u_min) / static_cast<double>(A.n_rows - 1);
    const double gu = (u - param.u_min) / hu;
    const long iu = static_cast<long>(std::floor(gu));
    const double tu = gu - static_cast<double>(iu);

    arma::rowvec out(A.n_cols, arma::fill::zeros);

    for(arma::uword c = 0; c < A.n_cols; ++c)
    {
        const double p0 = A(clamp_index(iu - 1, A.n_rows), c);
        const double p1 = A(clamp_index(iu,     A.n_rows), c);
        const double p2 = A(clamp_index(iu + 1, A.n_rows), c);
        const double p3 = A(clamp_index(iu + 2, A.n_rows), c);
        out(c) = cubic_interp(p0, p1, p2, p3, tu);
    }

    return out;
}

inline double eval_f_grid_lr(
    const low_rank_state& state,
    size_t i,
    size_t j
) noexcept {
    if(state.C.n_elem == 0)
        return 0.0;
    return arma::dot(state.Ux.row(i) % state.C.t(), state.Uu.row(j));
}

double eval_f_lr(
    const low_rank_state& state,
    double x,
    double u,
    const config_t<double>& param
) {
    if(state.C.n_elem == 0)
        return 0.0;

    const arma::rowvec ux = cubic_periodic_matrix_row(state.Ux, x, param);
    const arma::rowvec uu = cubic_clamped_velocity_matrix_row(state.Uu, u, param);

    return arma::dot(ux % state.C.t(), uu);
}

double cubic_periodic_vector_interp(
    const arma::vec& values,
    double x,
    const config_t<double>& param
) {
    x = std::fmod(std::fmod(x - param.x_min, param.Lx) + param.Lx, param.Lx);

    const double gx = x * param.dx_inv;
    const long ix = static_cast<long>(std::floor(gx));
    const double tx = gx - static_cast<double>(ix);

    const double p0 = values(periodic_index(ix - 1, values.n_elem));
    const double p1 = values(periodic_index(ix,     values.n_elem));
    const double p2 = values(periodic_index(ix + 1, values.n_elem));
    const double p3 = values(periodic_index(ix + 2, values.n_elem));

    return cubic_interp(p0, p1, p2, p3, tx);
}

// -----------------------------------------------------------------------------
// Output/statistics, mirroring the NuFI-LR do_stats block.
// -----------------------------------------------------------------------------

template <size_t order>
void do_stats(
    size_t n,
    const low_rank_state& state,
    const operators_t& ops,
    const config_t<double>& param,
    const poisson<double>& poiss,
    std::ofstream& stat_file,
    std::ofstream& stat_full_file
) {
    (void)order;

    const auto poisson_result = poisson_lr(state, ops, param, poiss);
    const arma::vec& E_grid = poisson_result.first;
    const double electric_energy = poisson_result.second;

    const double t = static_cast<double>(n) * param.dt;

    double Emax = 0.0;
    const size_t plot_n_x = 512;
    const double dx_plot = param.Lx / static_cast<double>(plot_n_x);

    for(size_t i = 0; i <= plot_n_x; ++i)
    {
        const double x = param.x_min + static_cast<double>(i) * dx_plot;
        const double E = cubic_periodic_vector_interp(E_grid, x, param);
        Emax = std::max(Emax, std::abs(E));
    }

    stat_file << std::setw(15) << t
              << std::setw(15) << std::setprecision(5) << std::scientific << Emax
              << " " << electric_energy
              << " " << state.rank()
              << std::endl;

    if(n % stat_every == 0)
    {
        const size_t plot_n_u = plot_n_x;
        const double du_plot = (param.u_max - param.u_min)
                             / static_cast<double>(plot_n_u);

        double kinetic_energy = 0.0;
        double entropy = 0.0;
        double l1_norm = 0.0;
        double l2_norm = 0.0;
        double max_norm = 0.0;

        const bool plot_f_now = (f_output_every > 0 && n % f_output_every == 0);
        std::ofstream f_str;

        if(plot_f_now)
        {
            f_str.open("f_" + std::to_string(t) + ".txt");
            f_str << std::setprecision(16);
        }

        for(size_t i = 0; i < plot_n_x; ++i)
        {
            const double x = param.x_min + static_cast<double>(i) * dx_plot;

            for(size_t j = 0; j < plot_n_u; ++j)
            {
                const double u = param.u_min + static_cast<double>(j) * du_plot;
                const double f = eval_f_lr(state, x, u, param);

                kinetic_energy += u * u * f;
                if(f > 0.0)
                    entropy -= f * std::log(f);
                l1_norm += f;
                l2_norm += f * f;
                max_norm = std::max(max_norm, f);

                if(plot_f_now)
                    f_str << x << " " << u << " " << f << '\n';
            }

            if(plot_f_now)
                f_str << '\n';
        }

        const double weight = dx_plot * du_plot;
        kinetic_energy *= weight;
        entropy *= weight;
        l1_norm *= weight;
        l2_norm *= weight;
        const double total_energy = kinetic_energy + electric_energy;

        stat_full_file << std::setprecision(16) << t << "; "
                       << l1_norm         << "; "
                       << l2_norm         << "; "
                       << electric_energy << "; "
                       << kinetic_energy  << "; "
                       << total_energy    << "; "
                       << entropy         << "; "
                       << max_norm        << "; "
                       << state.rank()    << ";"
                       << std::endl;
    }
}

void write_low_rank_state(
    const low_rank_state& state,
    const std::string& prefix
) {
    state.Ux.save(prefix + "_Ux.bin", arma::arma_binary);
    state.C.save(prefix + "_C.bin", arma::arma_binary);
    state.Uu.save(prefix + "_Uu.bin", arma::arma_binary);
}

// -----------------------------------------------------------------------------
// Driver with the same outer shape as run_restarted_simulation().
// -----------------------------------------------------------------------------

template <size_t order>
void run_conservative_sat_simulation()
{
    using std::abs;
    using std::max;

    conf = config_t<double>(Nx, Nu, Nt, dt, x_min, x_max, u_min, u_max, &f0);

    // Keep the globals synchronized with config in case they are changed above.
    Lx = conf.Lx;

    operators_t ops = precompute_operators(conf);
    conservative_truncation_data trunc_data = precompute_conservative_truncation_data(ops);

    poisson<double> poiss(conf);

    truncation_ranks_str.open("truncation_ranks.txt");
    std::ofstream stat_file("stats.txt");
    std::ofstream stat_full_file("stats_full.txt");

    stat_file << "# t Emax electric_energy rank\n";
    stat_full_file << "# t; l1; l2; electric_energy; kinetic_energy; total_energy; entropy; max_norm; rank;\n";

    nufi::stopwatch<double> total_timer;
    double total_time = 0.0;

    low_rank_state state = initial_condition_lr(conf, ops, tol_rank, max_rank);
    ssprk3_workspace rk_work(conf.Nx, conf.Nu, max_rank);

    if(state.rank() == 0)
        throw std::runtime_error("Initial condition was truncated to zero. Decrease tol_rank.");

    write_low_rank_state(state, "low_rank_initial");

    const moments_t m0 = moments_lr(state, ops);
    const double electric_energy0 = poisson_lr(state, ops, conf, poiss).second;
    const double mass0 = arma::sum(m0.rho) * conf.dx;
    const double momentum0 = arma::sum(m0.J) * conf.dx;
    const double energy0 = arma::sum(m0.kappa) * conf.dx + electric_energy0;

    std::ofstream invariant_file("invariants.txt");
    invariant_file << std::setprecision(16)
                   << "mass0 " << mass0 << '\n'
                   << "momentum0 " << momentum0 << '\n'
                   << "energy0 " << energy0 << '\n'
                   << "electric_energy0 " << electric_energy0 << '\n';

    for(size_t n = 0; n <= conf.Nt; ++n)
    {
        

        //std::cout << " start of time step " << n << std::endl;

        do_stats<order>(n, state, ops, conf, poiss, stat_file, stat_full_file);
        truncation_ranks_str << n * conf.dt << " " << state.rank() << std::endl;

        const moments_t m = moments_lr(state, ops);
        const double electric_energy = poisson_lr(state, ops, conf, poiss).second;
        const double mass = arma::sum(m.rho) * conf.dx;
        const double momentum = arma::sum(m.J) * conf.dx;
        const double energy = arma::sum(m.kappa) * conf.dx + electric_energy;

        if(n == conf.Nt)
            break;

        nufi::stopwatch<double> timer;

        ssprk3_step_lr(
            state,
            ops,
            trunc_data,
            conf,
            poiss,
            tol_rank,
            max_rank,
            rk_work
        );

        const double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;

        assert_finite(state.Ux, "state.Ux after step");
        assert_finite(state.Uu, "state.Uu after step");
        assert_finite(state.C,  "state.C after step");

        if(console_every > 0 && n % console_every == 0)
        {
            std::cout << std::setw(15) << n * conf.dt
                      << " rank: " << state.rank()
                      << " mass err: " << std::scientific << abs(mass - mass0)
                      << " mom err: " << abs(momentum - momentum0)
                      << " energy err: " << abs(energy - energy0)
                      << " Comp-time: " << timer_elapsed
                      << " Total comp time s.f.: " << total_time
                      << std::endl;
        }
    }

    write_low_rank_state(state, "low_rank_final");

    std::cout << "Total time: " << total_time << std::endl;
    std::cout << "Wall time: " << total_timer.elapsed() << std::endl;
}

} // namespace dim1
} // namespace nufi

int main()
{
    nufi::dim1::run_conservative_sat_simulation<4>();
}
