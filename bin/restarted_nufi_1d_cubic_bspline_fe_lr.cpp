/*
 * NuFI 1x1v with conservative L2-Galerkin cubic B-spline restarts.
 * Port of vlasov_poisson_cubic_bspline_fe_restart_verlet.m.
 * Optional weighted randomized SVD of the moment-free remainder.
 * No global moment correction.
 *
 * Copyright (C) 2022 Matthias Kirchhart and Paul Wilhelm.
 * Distributed under the GNU General Public License, version 3 or later.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <stdexcept>
#include <vector>

#include <armadillo>

#include <nufi/config.hpp>
#include <nufi/fields.hpp>
#include <nufi/poisson.hpp>
#include <nufi/rho.hpp>
#include <nufi/stopwatch.hpp>

namespace nufi::dim1 {

constexpr double pi = 3.14159265358979323846264338327950288;
constexpr size_t Nx = 256, Nu = 256;
constexpr double dt = 0.1, T = 100.0;
constexpr size_t Nt = size_t(T/dt + 0.5);
constexpr double x_min = 0.0, x_max = 4.0*pi, u_min = -10.0, u_max = 10.0;
constexpr double alpha = 0.01, wave_number = 0.5;

constexpr size_t restart_every = Nt+1;
constexpr size_t fe_velocity_nodes = 256, fe_gauss_order = 32;
constexpr size_t measurement_every = 1, nx_meas = 512;
constexpr size_t measurement_gauss_order = 8, nv_meas_uniform = 2048;
constexpr bool use_aligned_measurement = true;
constexpr bool recompute_restart_field = false;

// Set to true to store f1 plus a low-rank approximation of f2.
constexpr bool use_low_rank_compression = true;
constexpr size_t lr_rank = 50, lr_oversampling = 6, lr_power_iterations = 1;
constexpr double lr_relative_tol = 1e-2;  // 0 disables singular-value cutoff.
constexpr size_t lr_seed = 12345;
constexpr bool lr_preserve_reflection_symmetry = false;

inline double wrap_x(double x, double length) noexcept
{
    x = std::fmod(x,length);
    return x < 0.0 ? x+length : x;
}

inline double f0(double x, double v) noexcept
{
    return (1.0+alpha*std::cos(wave_number*x))*v*v*std::exp(-0.5*v*v)/std::sqrt(2.0*pi);
}

struct GaussRule {
    arma::vec x, w;
};

GaussRule gauss_legendre(size_t n)
{
    if(n == 0) throw std::invalid_argument("Gauss quadrature order must be positive");
    arma::mat J(n,n,arma::fill::zeros);
    for(size_t i = 1; i < n; ++i) {
        const double b = double(i)/std::sqrt(4.0*i*i-1.0);
        J(i-1,i) = J(i,i-1) = b;
    }
    arma::vec x;
    arma::mat Q;
    if(!arma::eig_sym(x,Q,J)) throw std::runtime_error("Gauss-Legendre eigensolve failed");
    arma::vec w = 2.0*arma::square(Q.row(0).t());
    for(size_t i = 0; i < n/2; ++i) {
        const size_t j = n-1-i;
        const double a = 0.5*(std::abs(x(i))+std::abs(x(j)));
        const double b = 0.5*(w(i)+w(j));
        x(i) = -a; x(j) = a;
        w(i) = b; w(j) = b;
    }
    if(n%2) x(n/2) = 0.0;
    return {std::move(x),std::move(w)};
}

struct Quadrature {
    arma::vec v, w;
};

Quadrature velocity_quadrature(const arma::vec& nodes, size_t order)
{
    GaussRule g = gauss_legendre(order);
    const size_t cells = nodes.n_elem-1, nq = cells*order;
    Quadrature out{arma::vec(nq),arma::vec(nq)};
    for(size_t cell = 0; cell < cells; ++cell) {
        const double mid = 0.5*(nodes(cell)+nodes(cell+1));
        const double half = 0.5*(nodes(cell+1)-nodes(cell));
        for(size_t j = 0; j < order; ++j) {
            const size_t q = cell*order+j;
            out.v(q) = mid+half*g.x(j);
            out.w(q) = half*g.w(j);
        }
    }
    // Pair global quadrature points exactly, preserving velocity reflection.
    for(size_t q = 0; q < nq/2; ++q) {
        const size_t r = nq-1-q;
        const double a = 0.5*(std::abs(out.v(q))+std::abs(out.v(r)));
        const double w = 0.5*(out.w(q)+out.w(r));
        out.v(q) = -a; out.v(r) = a;
        out.w(q) = w; out.w(r) = w;
    }
    return out;
}

struct Basis4 {
    size_t first;
    std::array<double,4> w;
};

Basis4 cubic_bspline(double v, const arma::vec& knots, size_t n, double vmin, double h)
{
    // Uniform not-a-knot knot indexing, identical to the MATLAB implementation.
    const double z = std::floor((v-(vmin+2.0*h))/h)+4.0;
    const size_t span = size_t(std::max(3.0,std::min(double(n-1),z)));
    std::array<double,4> N{1.0,0.0,0.0,0.0};
    double left[4]{}, right[4]{};
    for(size_t j = 1; j <= 3; ++j) {
        left[j] = v-knots(span+1-j);
        right[j] = knots(span+j)-v;
        double saved = 0.0;
        for(size_t r = 0; r < j; ++r) {
            const double denom = right[r+1]+left[j-r];
            const double temp = denom != 0.0 ? N[r]/denom : 0.0;
            N[r] = saved+right[r+1]*temp;
            saved = left[j-r]*temp;
        }
        N[j] = saved;
    }
    return {span-3,N};
}

struct VelocityFE {
    size_t nv;
    double vmin, vmax, h;
    arma::vec nodes, knots, vq, wq, mass_trap;
    arma::mat cholL, L, B, G;
    std::vector<Basis4> basis_q;

    VelocityFE(double lo, double hi, size_t n, size_t ng, size_t ntrap):
        nv(n), vmin(lo), vmax(hi), h((hi-lo)/(n-1))
    {
        if(n < 6 || ng < 3 || hi <= lo || std::abs(lo+hi) > 1e-12)
            throw std::invalid_argument("Expected symmetric velocity interval and at least six FE nodes");
        nodes.set_size(n);
        for(size_t i = 0; i < n; ++i) nodes(i) = lo+i*h;
        for(size_t i = 0; i < n/2; ++i) nodes(n-1-i) = -nodes(i);

        // Four clamped end knots and interior knots at nodes[2] ... nodes[n-3].
        knots.set_size(n+4);
        for(size_t i = 0; i < 4; ++i) knots(i) = lo;
        for(size_t i = 2; i <= n-3; ++i) knots(i+2) = nodes(i);
        for(size_t i = n; i < n+4; ++i) knots(i) = hi;

        Quadrature q = velocity_quadrature(nodes,ng);
        vq = std::move(q.v);
        wq = std::move(q.w);
        basis_q.reserve(vq.n_elem);
        arma::mat mass(n,n,arma::fill::zeros);
        arma::mat rhs(n,3,arma::fill::zeros);
        const double W0 = std::sqrt(2.0*pi)*std::erf(hi/std::sqrt(2.0));
        const double W2 = W0-2.0*hi*std::exp(-0.5*hi*hi);
        const double c0 = W2/W0;
        for(size_t qid = 0; qid < vq.n_elem; ++qid) {
            const double v = vq(qid), wt = wq(qid);
            const Basis4 b = cubic_bspline(v,knots,n,lo,h);
            basis_q.push_back(b);
            const double gaussian = std::exp(-0.5*v*v);
            const double p[3] = {gaussian,v*gaussian,(v*v-c0)*gaussian};
            for(size_t j = 0; j < 4; ++j) {
                const size_t ii = b.first+j;
                const double wb = wt*b.w[j];
                for(size_t k = 0; k < 4; ++k)
                    mass(ii,b.first+k) += wb*b.w[k];
                for(size_t k = 0; k < 3; ++k) rhs(ii,k) += wb*p[k];
            }
        }
        mass = 0.5*(mass+mass.t());
        if(!arma::chol(cholL,mass,"lower"))
            throw std::runtime_error("FE mass matrix Cholesky factorisation failed");

        // Exact integrals of B_i, v*B_i, 0.5*v^2*B_i.
        L.set_size(n,3);
        for(size_t i = 0; i < n; ++i) {
            double s1 = 0.0, s2 = 0.0;
            for(size_t j = 0; j < 5; ++j) {
                const double t = knots(i+j);
                s1 += t; s2 += t*t;
            }
            const double beta0 = (knots(i+4)-knots(i))/4.0;
            L(i,0) = beta0;
            L(i,1) = beta0*s1/5.0;
            L(i,2) = 0.5*beta0*(s1*s1+s2)/30.0;
        }

        B = solve_mass(rhs);
        for(size_t i = 0; i < n/2; ++i) {
            const size_t j = n-1-i;
            for(size_t k = 0; k < 3; ++k) {
                const double sign = k == 1 ? -1.0 : 1.0;
                const double value = 0.5*(B(i,k)+sign*B(j,k));
                B(i,k) = value;
                B(j,k) = sign*value;
            }
        }
        G = L.t()*B;
        if(arma::rcond(G) < 1e-10) throw std::runtime_error("FE moment carrier basis ill-conditioned");

        // Reproduce 1, v, v^2, including the no-knot regions near boundaries.
        double reproduction = 0.0;
        for(size_t qid = 0; qid < vq.n_elem; ++qid) {
            const Basis4& b = basis_q[qid];
            double p0 = 0.0, p1 = 0.0, p2 = 0.0;
            for(size_t j = 0; j < 4; ++j) {
                const size_t i = b.first+j;
                const double a = knots(i+1), c = knots(i+2), d = knots(i+3);
                p0 += b.w[j];
                p1 += b.w[j]*(a+c+d)/3.0;
                p2 += b.w[j]*(a*c+a*d+c*d)/3.0;
            }
            reproduction = std::max(reproduction,std::abs(p0-1.0));
            reproduction = std::max(reproduction,std::abs(p1-vq(qid)));
            reproduction = std::max(reproduction,std::abs(p2-vq(qid)*vq(qid)));
        }
        if(reproduction > 3e-11) throw std::runtime_error("FE polynomial reproduction check failed");

        mass_trap.zeros(n);
        const double dv = (hi-lo)/(ntrap-1);
        for(size_t j = 0; j < ntrap; ++j) {
            const double v = lo+j*dv;
            const double wt = dv*(j == 0 || j+1 == ntrap ? 0.5 : 1.0);
            const Basis4 b = cubic_bspline(v,knots,n,lo,h);
            for(size_t k = 0; k < 4; ++k) mass_trap(b.first+k) += wt*b.w[k];
        }
    }

    arma::mat solve_mass(const arma::mat& rhs) const
    {
        arma::mat y = arma::solve(arma::trimatl(cholL),rhs);
        return arma::solve(arma::trimatu(cholL.t()),y);
    }
};

arma::mat periodic_second_derivatives(const arma::mat& C, double dx)
{
    const size_t nx = C.n_rows;
    arma::mat rhs(nx,C.n_cols);
    for(size_t i = 0; i < nx; ++i)
        rhs.row(i) = 6.0*(C.row((i+1)%nx)-2.0*C.row(i)+C.row((i+nx-1)%nx))/(dx*dx);
    arma::cx_mat freq = arma::fft(rhs);
    for(size_t k = 0; k < nx; ++k)
        freq.row(k) /= 4.0+2.0*std::cos(2.0*pi*k/nx);
    return arma::real(arma::ifft(freq));
}

struct LowRankFactors {
    arma::mat X, XX, V; // spatial factors, second derivatives, velocity spline factors
    size_t remainder_rank = 0;
    double relative_error = 0.0;
    bool symmetric = false;
};

struct RSVD {
    arma::mat U, V;
    arma::vec s;
};

RSVD randomized_svd(const arma::mat& F, size_t rank, size_t oversampling, size_t npower)
{
    const size_t ell = std::min(size_t(std::min(F.n_rows,F.n_cols)),rank+oversampling);
    arma::mat Q, R;
    if(!arma::qr_econ(Q,R,F*arma::randn<arma::mat>(F.n_cols,ell)))
        throw std::runtime_error("RSVD initial QR failed");
    for(size_t i = 0; i < npower; ++i) {
        arma::mat P;
        if(!arma::qr_econ(P,R,F.t()*Q)) throw std::runtime_error("RSVD right QR failed");
        if(!arma::qr_econ(Q,R,F*P)) throw std::runtime_error("RSVD left QR failed");
    }
    arma::mat Us, V;
    arma::vec s;
    if(!arma::svd_econ(Us,s,V,Q.t()*F)) throw std::runtime_error("RSVD small SVD failed");
    const size_t r = std::min(rank,size_t(s.n_elem));
    if(r == 0) return {};
    return {Q*Us.cols(0,r-1),V.cols(0,r-1),s.head(r)};
}

LowRankFactors compress_remainder(const arma::mat& C2, const VelocityFE& fe,
                                  const arma::mat& A, size_t rank, size_t oversampling,
                                  size_t power_iterations, double tol, double dx)
{
    // Weighted SVD: ||C2*L||_F is the velocity L2 norm, M = L*L^T.
    LowRankFactors result;
    const size_t nx = C2.n_rows, nv = C2.n_cols;
    const size_t requested = std::min(rank,std::min(nx,nv));
    result.X = A;
    result.V = fe.B;
    const arma::mat F = C2*fe.cholL;
    if(requested == 0) {
        result.relative_error = arma::norm(F,"fro") > 0.0 ? 1.0 : 0.0;
        result.XX = periodic_second_derivatives(result.X,dx);
        return result;
    }

    arma::mat X2(nx,requested,arma::fill::zeros), V2(nv,requested,arma::fill::zeros);
    size_t r = 0;
    // With the physical reflection f(x,v) = f(Lx-x,-v), the even and odd
    // spatial parts live in orthogonal subspaces. Compress independently and
    // retain the largest singular values across both sectors.
    arma::mat reflected(nx,nv);
    for(size_t i = 0; i < nx; ++i)
        for(size_t j = 0; j < nv; ++j)
            reflected(i,j) = C2((nx-i)%nx,nv-1-j);
    const double symmetry_error = arma::norm(C2-reflected,"fro")/
        std::max(arma::norm(C2+A*fe.B.t(),"fro"),std::numeric_limits<double>::epsilon());
    result.symmetric = lr_preserve_reflection_symmetry && symmetry_error < 1e-10;
    if(result.symmetric) {
        const arma::mat sym = 0.5*(C2+reflected);
        arma::mat even(nx,nv), odd(nx,nv);
        for(size_t i = 0; i < nx; ++i) {
            even.row(i) = 0.5*(sym.row(i)+sym.row((nx-i)%nx));
            odd.row(i) = 0.5*(sym.row(i)-sym.row((nx-i)%nx));
        }
        const RSVD e = randomized_svd(even*fe.cholL,requested,oversampling,power_iterations);
        const RSVD o = randomized_svd(odd*fe.cholL,requested,oversampling,power_iterations);
        size_t ne = 0, no = 0;
        const double leading = std::max(e.s.n_elem ? e.s(0) : 0.0,o.s.n_elem ? o.s(0) : 0.0);
        while(r < requested && (ne < e.s.n_elem || no < o.s.n_elem)) {
            const bool choose_even = no == o.s.n_elem ||
                (ne < e.s.n_elem && e.s(ne) >= o.s(no));
            const RSVD& part = choose_even ? e : o;
            const size_t idx = choose_even ? ne++ : no++;
            const double sigma = part.s(idx);
            if(tol > 0.0 && sigma <= tol*leading) break;
            const double sign = choose_even ? 1.0 : -1.0;
            arma::vec x = sigma*part.U.col(idx);
            arma::vec v = arma::solve(arma::trimatu(fe.cholL.t()),part.V.col(idx));
            // Project away roundoff components of the opposite parity.
            for(size_t i = 0; i < nx/2+1; ++i) {
                const size_t j = (nx-i)%nx;
                if(i > j) continue;
                const double value = 0.5*(x(i)+sign*x(j));
                x(i) = value; x(j) = sign*value;
            }
            for(size_t i = 0; i < nv/2; ++i) {
                const size_t j = nv-1-i;
                const double value = 0.5*(v(i)+sign*v(j));
                v(i) = value; v(j) = sign*value;
            }
            X2.col(r) = x;
            V2.col(r) = v;
            ++r;
        }
    } else {
        const RSVD s = randomized_svd(F,requested,oversampling,power_iterations);
        if(s.s.n_elem > 0) {
            r = std::min(requested,size_t(s.s.n_elem));
            if(tol > 0.0 && s.s(0) > 0.0)
                while(r > 0 && s.s(r-1) <= tol*s.s(0)) --r;
            if(r > 0) {
                X2.cols(0,r-1) = s.U.cols(0,r-1)*arma::diagmat(s.s.head(r));
                V2.cols(0,r-1) = arma::solve(arma::trimatu(fe.cholL.t()),s.V.cols(0,r-1));
            }
        }
    }
    result.remainder_rank = r;
    if(r > 0) {
        X2 = X2.cols(0,r-1);
        V2 = V2.cols(0,r-1);
        result.X = arma::join_rows(A,X2);
        result.V = arma::join_rows(fe.B,V2);
        result.relative_error = arma::norm(F-(X2*V2.t())*fe.cholL,"fro")/
            std::max(arma::norm(F,"fro"),std::numeric_limits<double>::epsilon());
    } else {
        result.relative_error = arma::norm(F,"fro") > 0.0 ? 1.0 : 0.0;
    }
    result.XX = periodic_second_derivatives(result.X,dx);
    return result;
}

struct RestartState {
    const VelocityFE* fe = nullptr;
    arma::mat C, mx;     // full FE state when compression is disabled
    LowRankFactors lr;   // f1 + compressed f2 when enabled
    bool compressed = false;
    double x0 = 0.0, Lx = 0.0, dx = 0.0;
    double M = 0.0, J = 0.0, K = 0.0;
    double Msource = 0.0, Jsource = 0.0, Ksource = 0.0;
    double Mtrap = 0.0, symmetry = 0.0, symmetry1 = 0.0, symmetry2 = 0.0;
    double moment_free_defect = 0.0;

    double eval(double x, double v) const noexcept
    {
        if(!fe || v < fe->vmin || v > fe->vmax) return 0.0;
        x = wrap_x(x-x0,Lx);
        const double z = x/dx;
        const size_t nx = compressed ? lr.X.n_rows : C.n_rows;
        const size_t i = std::min(size_t(std::floor(z)),nx-1);
        const size_t i2 = (i+1)%nx;
        const double t = z-i, a = 1.0-t;
        const double s1 = (a*a*a-a)*dx*dx/6.0;
        const double s2 = (t*t*t-t)*dx*dx/6.0;
        const Basis4 b = cubic_bspline(v,fe->knots,fe->nv,fe->vmin,fe->h);
        double result = 0.0;
        if(compressed) {
            for(size_t k = 0; k < lr.X.n_cols; ++k) {
                double vk = 0.0;
                for(size_t j = 0; j < 4; ++j) vk += b.w[j]*lr.V(b.first+j,k);
                const double xk = a*lr.X(i,k)+t*lr.X(i2,k)+s1*lr.XX(i,k)+s2*lr.XX(i2,k);
                result += xk*vk;
            }
        } else {
            for(size_t j = 0; j < 4; ++j) {
                const size_t k = b.first+j;
                const double value = a*C(i,k)+t*C(i2,k)+s1*mx(i,k)+s2*mx(i2,k);
                result += b.w[j]*value;
            }
        }
        return result;
    }
};

RestartState restart_state;
double f_restart(double x, double v) noexcept { return restart_state.eval(x,v); }

// The Galerkin projection preserves its velocity moments. With compression,
// keep f1 explicitly and apply a weighted randomized SVD only to f2.
template <typename Eval>
RestartState build_restart(const VelocityFE& fe, size_t nx, double x0, double Lx, Eval&& eval,
                           bool compress = use_low_rank_compression)
{
    const double dx = Lx/nx;
    arma::mat rhs(fe.nv,nx,arma::fill::zeros), source(nx,3,arma::fill::zeros);
    #pragma omp parallel for schedule(static)
    for(std::ptrdiff_t ii = 0; ii < std::ptrdiff_t(nx); ++ii) {
        const size_t i = size_t(ii);
        const double x = x0+i*dx;
        double M = 0.0, J = 0.0, K = 0.0;
        for(size_t q = 0; q < fe.vq.n_elem; ++q) {
            const double v = fe.vq(q);
            const double f = eval(x,v);
            const double fw = fe.wq(q)*f;
            M += fw; J += v*fw; K += 0.5*v*v*fw;
            const Basis4& b = fe.basis_q[q];
            for(size_t j = 0; j < 4; ++j) rhs(b.first+j,i) += fw*b.w[j];
        }
        source(i,0) = M;
        source(i,1) = J;
        source(i,2) = K;
    }
    RestartState state;
    state.fe = &fe;
    state.x0 = x0;
    state.Lx = Lx;
    state.dx = dx;
    state.C = fe.solve_mass(rhs).t();
    const arma::mat moments = state.C*fe.L;
    state.Msource = dx*arma::accu(source.col(0));
    state.Jsource = dx*arma::accu(source.col(1));
    state.Ksource = dx*arma::accu(source.col(2));

    // f1 carries all three local moments, f2 has none in FE moment integrals.
    const arma::mat A = arma::solve(fe.G,moments.t()).t();
    const arma::mat C1 = A*fe.B.t();
    const arma::mat C2 = state.C-C1;
    const auto symmetry_error = [nx](const arma::mat& C) {
        arma::mat reflection(C.n_rows,C.n_cols);
        for(size_t i = 0; i < nx; ++i)
            for(size_t j = 0; j < C.n_cols; ++j)
                reflection(i,j) = C((nx-i)%nx,C.n_cols-1-j);
        return arma::norm(C-reflection,"fro")/
            std::max(arma::norm(C,"fro"),std::numeric_limits<double>::epsilon());
    };
    if(compress) {
        state.compressed = true;
        state.lr = compress_remainder(C2,fe,A,lr_rank,lr_oversampling,
                                      lr_power_iterations,lr_relative_tol,dx);
        arma::mat C2r(nx,fe.nv,arma::fill::zeros);
        if(state.lr.remainder_rank > 0)
            C2r = state.lr.X.cols(3,state.lr.X.n_cols-1)*
                  state.lr.V.cols(3,state.lr.V.n_cols-1).t();
        const arma::mat Cr = C1+C2r;
        const arma::mat compressed_moments = state.lr.X*(state.lr.V.t()*fe.L);
        state.M = dx*arma::accu(compressed_moments.col(0));
        state.J = dx*arma::accu(compressed_moments.col(1));
        state.K = dx*arma::accu(compressed_moments.col(2));
        state.Mtrap = dx*arma::accu(state.lr.X*(state.lr.V.t()*fe.mass_trap));
        state.moment_free_defect = arma::norm(C2r*fe.L,"fro");
        state.symmetry = symmetry_error(Cr);
        state.symmetry1 = symmetry_error(C1);
        state.symmetry2 = symmetry_error(C2r);
        state.C.reset();  // Only (3+rank) separable terms survive the restart.
    } else {
        state.mx = periodic_second_derivatives(state.C,dx);
        state.M = dx*arma::accu(moments.col(0));
        state.J = dx*arma::accu(moments.col(1));
        state.K = dx*arma::accu(moments.col(2));
        state.Mtrap = dx*arma::accu(state.C*fe.mass_trap);
        state.moment_free_defect = arma::norm(C2*fe.L,"fro");
        state.symmetry = symmetry_error(state.C);
        state.symmetry1 = symmetry_error(C1);
        state.symmetry2 = symmetry_error(C2);
    }
    return state;
}

template <size_t order>
void apply_restart(size_t& nt, size_t n, double* coeffs, config_t<double>& conf,
                   const VelocityFE& fe, std::ostream& log)
{
    nufi::stopwatch<double> timer;
    const size_t stride = conf.Nx+order-1;
    RestartState next = build_restart(fe,Nx,conf.x_min,conf.Lx,
        [&](double x, double v) { return periodic::eval_f<double,order>(nt,x,v,coeffs,conf); });
    restart_state = std::move(next);
    log << std::setprecision(16) << n*dt << ' '
        << restart_state.M-restart_state.Msource << ' '
        << restart_state.J-restart_state.Jsource << ' '
        << restart_state.K-restart_state.Ksource << ' '
        << restart_state.Msource-conf.Lx << ' '
        << restart_state.M-conf.Lx << ' ' << restart_state.J << ' '
        << restart_state.Mtrap-restart_state.M << ' '
        << restart_state.symmetry << ' ' << restart_state.symmetry1 << ' '
        << restart_state.symmetry2 << ' '
        << restart_state.lr.remainder_rank << ' '
        << restart_state.lr.relative_error << ' '
        << restart_state.moment_free_defect << ' '
        << restart_state.lr.symmetric << '\n';
    std::cout << std::setprecision(9) << "t=" << n*dt
              << " FE projection: dM=" << restart_state.M-restart_state.Msource
              << " dJ=" << restart_state.J-restart_state.Jsource
              << " dK=" << restart_state.K-restart_state.Ksource
              << " M-M0=" << restart_state.M-conf.Lx
              << " J=" << restart_state.J
              << " sym=" << restart_state.symmetry
              << " rank=" << restart_state.lr.remainder_rank
              << " LRrel=" << restart_state.lr.relative_error
              << " f2mom=" << restart_state.moment_free_defect
              << " LR-sym=" << restart_state.lr.symmetric << '\n';
    // Retain the potential at restart time as history entry zero.
    for(size_t i = 0; i < stride; ++i) coeffs[i] = coeffs[nt*stride+i];
    conf = config_t<double>(Nx,Nu,Nt,dt,x_min,x_max,u_min,u_max,&f_restart);
    nt = 1;
    std::cout << "Restart time: " << timer.elapsed() << " s\n";
}

struct MeasuredMoments { double M = 0.0, J = 0.0, kinetic = 0.0; };

template <size_t order>
MeasuredMoments measure(size_t nt, double* coeffs, const config_t<double>& conf,
                        const arma::vec& v, const arma::vec& w)
{
    const double dx = conf.Lx/nx_meas;
    long double M = 0.0, J = 0.0, K = 0.0;
    #pragma omp parallel for reduction(+:M,J,K) schedule(static)
    for(std::ptrdiff_t ii = 0; ii < std::ptrdiff_t(nx_meas); ++ii) {
        const double x = conf.x_min+ii*dx;
        long double mx = 0.0, jx = 0.0, kx = 0.0;
        for(size_t j = 0; j < v.n_elem; ++j) {
            const double u = v(j);
            const double f = periodic::eval_f<double,order>(nt,x,u,coeffs,conf);
            const long double wf = w(j)*f;
            mx += wf; jx += u*wf; kx += 0.5*u*u*wf;
        }
        M += mx; J += jx; K += kx;
    }
    return {double(dx*M),double(dx*J),double(dx*K)};
}

template <size_t order>
void run()
{
    static_assert(Nx%2 == 0 && fe_velocity_nodes >= 6,"Invalid spline grid");
    static_assert(restart_every > 0 && measurement_every > 0,"Invalid output period");
    VelocityFE fe(u_min,u_max,fe_velocity_nodes,fe_gauss_order,nv_meas_uniform);
    Quadrature measurement;
    if(use_aligned_measurement) {
        measurement = velocity_quadrature(fe.nodes,measurement_gauss_order);
    } else {
        measurement.v.set_size(nv_meas_uniform);
        measurement.w.set_size(nv_meas_uniform);
        const double dv = (u_max-u_min)/(nv_meas_uniform-1);
        for(size_t j = 0; j < nv_meas_uniform; ++j) {
            measurement.v(j) = u_min+j*dv;
            measurement.w(j) = (j == 0 || j+1 == nv_meas_uniform ? 0.5 : 1.0)*dv;
        }
    }
    config_t<double> conf(Nx,Nu,Nt,dt,x_min,x_max,u_min,u_max,&f0);
    const size_t stride = conf.Nx+order-1;
    std::unique_ptr<double[]> coeffs(new double[(restart_every+1)*stride]{});
    const size_t aligned_bytes = ((sizeof(double)*Nx+63)/64)*64;
    std::unique_ptr<double,decltype(std::free)*> rho(
        static_cast<double*>(std::aligned_alloc(64,aligned_bytes)),std::free);
    if(!rho) throw std::bad_alloc();
    poisson<double> poiss(conf);
    std::ofstream stats("stats.txt"), measured("conservation.txt"), restarts("restart_defects.txt");
    if(!stats || !measured || !restarts) throw std::runtime_error("Cannot open output files");
    stats << "# t Emax electric_energy\n";
    measured << "# t M J kinetic total_energy relative_M_error absolute_J_error relative_H_error\n";
    restarts << "# t dM_restart dJ_restart dK_restart source_M_minus_M0 M_minus_M0 J trap_M_minus_M symmetry symmetry1 symmetry2 lr_rank lr_relative_error f2_moment_defect lr_symmetry_mode\n";
    if(use_low_rank_compression) arma::arma_rng::set_seed(lr_seed);
    std::cout << "FE restart compression: " << (use_low_rank_compression ? "RSVD" : "none") << '\n';
    size_t nt = 0;
    double H0 = std::numeric_limits<double>::quiet_NaN();
    const double initial_tail = conf.Lx*(std::erfc(u_max/std::sqrt(2.0))
        +std::sqrt(2.0/pi)*u_max*std::exp(-0.5*u_max*u_max));
    std::cout << "Initial excluded velocity-tail mass: " << initial_tail << '\n';
    for(size_t n = 0; n <= Nt; ++n) {
        nufi::stopwatch<double> timer;
        #pragma omp parallel for schedule(static)
        for(std::ptrdiff_t i = 0; i < std::ptrdiff_t(conf.Nx); ++i)
            rho.get()[i] = periodic::eval_rho<double,order>(nt,size_t(i),coeffs.get(),conf);
        double electric_energy = poiss.solve(rho.get());
        periodic::interpolate<double,order>(coeffs.get()+nt*stride,rho.get(),conf);
        const bool at_restart = n > 0 && n%restart_every == 0;
        if(at_restart) {
            apply_restart<order>(nt,n,coeffs.get(),conf,fe,restarts);
            if(recompute_restart_field) {
                #pragma omp parallel for schedule(static)
                for(std::ptrdiff_t i = 0; i < std::ptrdiff_t(conf.Nx); ++i)
                    rho.get()[i] = periodic::eval_rho<double,order>(0,size_t(i),coeffs.get(),conf);
                electric_energy = poiss.solve(rho.get());
                periodic::interpolate<double,order>(coeffs.get(),rho.get(),conf);
            }
        } else {
            ++nt;
        }
        double Emax = 0.0;
        const size_t field_step = at_restart ? 0 : nt-1;
        for(size_t i = 0; i < 512; ++i) {
            const double x = conf.x_min+i*conf.Lx/512;
            const double E = -periodic::eval<double,order,1>(x,coeffs.get()+field_step*stride,conf);
            Emax = std::max(Emax,std::abs(E));
        }
        stats << std::setprecision(16) << n*dt << ' ' << Emax << ' ' << electric_energy << '\n';
        if(n%measurement_every == 0 || n == Nt || at_restart) {
            // At a restart time, evaluate the newly projected base (nt=0).
            const size_t eval_nt = at_restart ? 0 : nt-1;
            MeasuredMoments m = measure<order>(eval_nt,coeffs.get(),conf,measurement.v,measurement.w);
            const double H = m.kinetic+electric_energy;
            if(n == 0) H0 = H;
            const double errH = std::abs(H-H0)/std::abs(H0);
            measured << std::setprecision(16) << n*dt << ' ' << m.M << ' ' << m.J
                     << ' ' << m.kinetic << ' ' << H << ' '
                     << std::abs(m.M-conf.Lx)/conf.Lx << ' ' << std::abs(m.J) << ' ' << errH << '\n';
            std::cout << std::setprecision(9) << "t=" << n*dt
                      << " measured dM/M=" << std::abs(m.M-conf.Lx)/conf.Lx
                      << " |J|=" << std::abs(m.J) << " dH/H=" << errH << '\n';
        }
        std::cout << "step " << n << "/" << Nt << ": " << timer.elapsed() << " s\n";
    }
}

} // namespace nufi::dim1

int main()
{
    nufi::dim1::run<4>();
    return 0;
}
