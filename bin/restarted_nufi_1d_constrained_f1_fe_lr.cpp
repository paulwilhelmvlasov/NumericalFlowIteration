/*
 * NuFI 1x1v: constrained analytic f1 plus cubic FE low-rank f2 restarts.
 * Port of nufi_vp_split_fe_lr_constrained_f1_MJ_fixed.m.
 * Gaussian carriers in f1, exact FE moment-nullspace f2,
 * spatial-and-velocity FE-mass-weighted rank-r SVD of f2.
 * Global f1 mass and momentum constrained to their initial values.
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
constexpr double dt = 0.1, T = 50.0;
constexpr size_t Nt = size_t(T/dt + 0.5);
constexpr double x_min = 0.0, x_max = 4.0*pi, u_min = -10.0, u_max = 10.0;
constexpr double alpha = 0.01, wave_number = 0.5;

constexpr size_t restart_every = 100;
constexpr size_t fe_velocity_nodes = 256, fe_gauss_order = 4;
constexpr size_t measurement_every = 1, nx_meas = 128;
constexpr size_t measurement_gauss_order = 4, nv_meas_uniform = 512;
constexpr bool use_aligned_measurement = true;
constexpr bool recompute_restart_field = false;

// Matches the MATLAB constrained-f1 calculation; disable either independently.
constexpr bool enforce_f1_initial_MJ = true;
constexpr bool use_low_rank_compression = true;
constexpr size_t lr_rank = 16; // Rank of f2; analytic f1 contributes three more terms.

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
    double vmin, vmax, h, c0;
    arma::vec nodes, knots, vq, wq, mass_trap, carrier_mass_trap;
    arma::mat cholL, L, G, H, moment_basis;
    std::vector<Basis4> basis_q;

    VelocityFE(double lo, double hi, size_t n, size_t ng, size_t ntrap):
        nv(n), vmin(lo), vmax(hi), h((hi-lo)/(n-1))
    {
        if(n < 6 || ng < 4 || hi <= lo || std::abs(lo+hi) > 1e-12)
            throw std::invalid_argument("Expected symmetric velocity interval, cubic FE and Gauss order >= 4");
        nodes.set_size(n);
        for(size_t i = 0; i < n; ++i) nodes(i) = lo+i*h;
        for(size_t i = 0; i < n/2; ++i) nodes(n-1-i) = -nodes(i);
        knots.set_size(n+4);
        for(size_t i = 0; i < 4; ++i) knots(i) = lo;
        for(size_t i = 2; i <= n-3; ++i) knots(i+2) = nodes(i);
        for(size_t i = n; i < n+4; ++i) knots(i) = hi;

        Quadrature q = velocity_quadrature(nodes,ng);
        vq = std::move(q.v);
        wq = std::move(q.w);
        const double W0 = std::sqrt(2.0*pi)*std::erf(hi/std::sqrt(2.0));
        const double W2 = W0-2.0*hi*std::exp(-0.5*hi*hi);
        c0 = W2/W0;

        arma::mat mass(n,n,arma::fill::zeros);
        G.zeros(3,3);
        H.zeros(n,3);
        basis_q.reserve(vq.n_elem);
        for(size_t iq = 0; iq < vq.n_elem; ++iq) {
            const double v = vq(iq), wt = wq(iq);
            const Basis4 b = cubic_bspline(v,knots,n,lo,h);
            basis_q.push_back(b);
            const double g = std::exp(-0.5*v*v);
            const double carrier[3] = {g,v*g,(v*v-c0)*g};
            const double p[3] = {1.0,v,0.5*v*v};
            for(size_t k = 0; k < 3; ++k)
                for(size_t j = 0; j < 3; ++j) G(k,j) += wt*p[k]*carrier[j];
            for(size_t j = 0; j < 4; ++j) {
                const size_t ii = b.first+j;
                const double wb = wt*b.w[j];
                for(size_t k = 0; k < 4; ++k) mass(ii,b.first+k) += wb*b.w[k];
                for(size_t k = 0; k < 3; ++k) H(ii,k) += wb*carrier[k];
            }
        }
        mass = 0.5*(mass+mass.t());
        if(!arma::chol(cholL,mass,"lower"))
            throw std::runtime_error("Velocity FE mass factorisation failed");

        // Exact integrals of B_j(v) against 1, v, v^2/2.
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
            L(i,2) = beta0*(s1*s1+s2)/60.0;
        }
        const arma::mat Z = solve_mass(L);
        moment_basis = arma::solve((L.t()*Z).t(),Z.t()).t(); // Z*inv(L^T Z)

        mass_trap.zeros(n);
        carrier_mass_trap.zeros(3);
        const double dv = (hi-lo)/(ntrap-1);
        for(size_t j = 0; j < ntrap; ++j) {
            const double v = lo+j*dv;
            const double wt = dv*(j == 0 || j+1 == ntrap ? 0.5 : 1.0);
            const Basis4 b = cubic_bspline(v,knots,n,lo,h);
            for(size_t k = 0; k < 4; ++k) mass_trap(b.first+k) += wt*b.w[k];
            const double g = std::exp(-0.5*v*v);
            carrier_mass_trap(0) += wt*g;
            carrier_mass_trap(1) += wt*v*g;
            carrier_mass_trap(2) += wt*(v*v-c0)*g;
        }
    }

    arma::mat solve_mass(const arma::mat& rhs) const
    {
        const arma::mat y = arma::solve(arma::trimatl(cholL),rhs);
        return arma::solve(arma::trimatu(cholL.t()),y);
    }
};

struct SpatialFE {
    size_t n;
    double dx;
    arma::mat cholL;

    SpatialFE(size_t nx, double spacing): n(nx), dx(spacing)
    {
        arma::mat mass(n,n,arma::fill::zeros);
        const GaussRule g = gauss_legendre(4);
        for(size_t i = 0; i < n; ++i) {
            for(size_t q = 0; q < 4; ++q) {
                const double t = 0.5*(g.x(q)+1.0);
                const double a = 1.0-t;
                const double w[4] = {a*a*a,3*t*t*t-6*t*t+4,
                                     -3*t*t*t+3*t*t+3*t+1,t*t*t};
                const double wt = 0.5*dx*g.w(q)/36.0;
                for(size_t j = 0; j < 4; ++j)
                    for(size_t k = 0; k < 4; ++k)
                        mass((i+j+n-1)%n,(i+k+n-1)%n) += wt*w[j]*w[k];
            }
        }
        mass = 0.5*(mass+mass.t());
        if(!arma::chol(cholL,mass,"lower"))
            throw std::runtime_error("Spatial FE mass factorisation failed");
    }

    arma::mat coefficients(const arma::mat& values) const
    {
        // Cardinal periodic cubic interpolation on uniform grid.
        arma::cx_mat spectrum = arma::fft(values);
        for(size_t k = 0; k < n; ++k)
            spectrum.row(k) /= (4.0+2.0*std::cos(2.0*pi*k/n))/6.0;
        return arma::real(arma::ifft(spectrum));
    }

    arma::mat nodal(const arma::mat& C) const
    {
        arma::mat Y(C.n_rows,C.n_cols);
        for(size_t i = 0; i < n; ++i)
            Y.row(i) = (C.row((i+n-1)%n)+4.0*C.row(i)+C.row((i+1)%n))/6.0;
        return Y;
    }
};

Basis4 periodic_cubic(double x, double dx, size_t nx) noexcept
{
    const double z = x/dx;
    const size_t i = std::min(size_t(std::floor(z)),nx-1);
    const double t = std::max(0.0,std::min(1.0,z-double(i)));
    const double a = 1.0-t;
    return {(i+nx-1)%nx,{a*a*a/6.0,(3*t*t*t-6*t*t+4)/6.0,
            (-3*t*t*t+3*t*t+3*t+1)/6.0,t*t*t/6.0}};
}

struct RestartState {
    const VelocityFE* fe = nullptr;
    arma::mat carrier_coeff, U, V, C2; // Cubic spatial coefficients for all x factors.
    size_t nx = 0, rank = 0;
    bool compressed = false;
    double x0 = 0.0, Lx = 0.0, dx = 0.0;
    double M = 0.0, J = 0.0, K = 0.0, M1 = 0.0, J1 = 0.0, K1 = 0.0;
    double M2 = 0.0, J2 = 0.0, K2 = 0.0;
    double Msource = 0.0, Jsource = 0.0, Ksource = 0.0;
    double Mprojected = 0.0, Jprojected = 0.0, Kprojected = 0.0;
    double M1before = 0.0, J1before = 0.0;
    double Mtrap = 0.0, symmetry = 0.0, symmetry1 = 0.0, symmetry2 = 0.0;
    double moment_free_defect = 0.0, relative_error = 0.0;

    double eval(double x, double v) const noexcept
    {
        if(!fe || v < fe->vmin || v > fe->vmax) return 0.0;
        const Basis4 bx = periodic_cubic(wrap_x(x-x0,Lx),dx,nx);
        const Basis4 bv = cubic_bspline(v,fe->knots,fe->nv,fe->vmin,fe->h);
        const double g = std::exp(-0.5*v*v);
        const double carriers[3] = {g,v*g,(v*v-fe->c0)*g};
        double f1 = 0.0, f2 = 0.0;
        for(size_t k = 0; k < 3; ++k) {
            double a = 0.0;
            for(size_t j = 0; j < 4; ++j)
                a += bx.w[j]*carrier_coeff((bx.first+j)%nx,k);
            f1 += a*carriers[k];
        }
        if(compressed) {
            for(size_t k = 0; k < rank; ++k) {
                double ux = 0.0, vv = 0.0;
                for(size_t j = 0; j < 4; ++j) {
                    ux += bx.w[j]*U((bx.first+j)%nx,k);
                    vv += bv.w[j]*V(bv.first+j,k);
                }
                f2 += ux*vv;
            }
        } else {
            for(size_t j = 0; j < 4; ++j)
                for(size_t k = 0; k < 4; ++k)
                    f2 += bx.w[j]*bv.w[k]*C2((bx.first+j)%nx,bv.first+k);
        }
        return f1+f2;
    }
};

RestartState restart_state;
double f_restart(double x, double v) noexcept { return restart_state.eval(x,v); }

double reflection_error(const arma::mat& F)
{
    double d2 = 0.0, n2 = 0.0;
    const size_t nx = F.n_rows, nv = F.n_cols;
    for(size_t i = 0; i < nx; ++i)
        for(size_t j = 0; j < nv; ++j) {
            const double a = F(i,j), b = F((nx-i)%nx,nv-j-1);
            d2 += (a-b)*(a-b);
            n2 += a*a;
        }
    return std::sqrt(d2/std::max(n2,std::numeric_limits<double>::epsilon()));
}

template <typename Eval>
RestartState build_restart(const VelocityFE& fe, const SpatialFE& spatial, double x0, double Lx,
                           Eval&& eval, bool compress = use_low_rank_compression)
{
    const size_t nx = spatial.n;
    const double dx = Lx/nx;
    arma::mat rhs(fe.nv,nx,arma::fill::zeros), source(nx,3,arma::fill::zeros);
    #pragma omp parallel for schedule(static)
    for(std::ptrdiff_t ii = 0; ii < std::ptrdiff_t(nx); ++ii) {
        const size_t i = size_t(ii);
        const double x = x0+i*dx;
        double M = 0.0, J = 0.0, K = 0.0;
        for(size_t q = 0; q < fe.vq.n_elem; ++q) {
            const double v = fe.vq(q), f = eval(x,v), wf = fe.wq(q)*f;
            M += wf; J += v*wf; K += 0.5*v*v*wf;
            const Basis4& b = fe.basis_q[q];
            for(size_t j = 0; j < 4; ++j) rhs(b.first+j,i) += wf*b.w[j];
        }
        source(i,0) = M;
        source(i,1) = J;
        source(i,2) = K;
    }

    RestartState state;
    state.fe = &fe;
    state.nx = nx;
    state.x0 = x0;
    state.Lx = Lx;
    state.dx = dx;
    const arma::rowvec initial = {Lx,0.0};
    const arma::rowvec src_global = dx*arma::sum(source,0);
    state.Msource = src_global(0);
    state.Jsource = src_global(1);
    state.Ksource = src_global(2);

    // f2 Galerkin RHS uses UNCORRECTED local source moments, as in MATLAB.
    // H(:,k) = Q4[B_j*g_k], G(k,l) = Q4[p_k*g_l].
    const arma::mat Araw = arma::solve(fe.G,source.t()).t();
    const arma::mat rhs2 = rhs-fe.H*Araw.t();
    const arma::mat C2values = fe.solve_mass(rhs2).t();

    // Global constrained spatial f1 solve. For a periodic cubic basis on a
    // uniform grid, Mx*1 = dx*1, so the KKT solution is a uniform coefficient
    // shift equal to (target - integral)/Lx for each constrained moment field.
    arma::mat moment_coeff = spatial.coefficients(source);
    const arma::rowvec before = dx*arma::sum(moment_coeff,0);
    state.M1before = before(0);
    state.J1before = before(1);
    if(enforce_f1_initial_MJ)
        for(size_t k = 0; k < 2; ++k)
            moment_coeff.col(k) += (initial(k)-before(k))/Lx;

    // Store the analytic Gaussian amplitudes, not FE projections of them.
    state.carrier_coeff = arma::solve(fe.G,moment_coeff.t()).t();

    // Exact velocity moment-nullspace projection before and after truncation.
    const arma::mat C2 = spatial.coefficients(C2values);
    const arma::mat C2free = C2-(C2*fe.L)*fe.moment_basis.t();
    if(compress) {
        state.compressed = true;
        const arma::mat weighted = spatial.cholL.t()*C2free*fe.cholL;
        arma::mat P, Q;
        arma::vec s;
        if(!arma::svd_econ(P,s,Q,weighted)) throw std::runtime_error("FE-weighted SVD failed");
        state.rank = std::min(lr_rank,size_t(s.n_elem));
        if(s.n_elem > state.rank)
            state.relative_error = arma::norm(s.tail(s.n_elem-state.rank),2)/
                                   std::max(arma::norm(s,2),std::numeric_limits<double>::epsilon());
        if(state.rank) {
            state.U = arma::solve(arma::trimatu(spatial.cholL.t()),
                                  P.cols(0,state.rank-1)*arma::diagmat(s.head(state.rank)));
            state.V = arma::solve(arma::trimatu(fe.cholL.t()),Q.cols(0,state.rank-1));
            state.V -= fe.moment_basis*(fe.L.t()*state.V);
        }
    } else {
        state.C2 = C2free;
    }

    const arma::mat projected = fe.solve_mass(rhs).t();
    const arma::rowvec mp = dx*arma::sum(projected*fe.L,0);
    state.Mprojected = mp(0);
    state.Jprojected = mp(1);
    state.Kprojected = mp(2);
    const arma::rowvec m1 = dx*arma::sum(moment_coeff,0);
    arma::mat m2rows(nx,3,arma::fill::zeros);
    if(compress && state.rank) m2rows = state.U*(state.V.t()*fe.L);
    else if(!compress) m2rows = state.C2*fe.L;
    const arma::rowvec m2 = dx*arma::sum(m2rows,0);
    state.M1 = m1(0); state.J1 = m1(1); state.K1 = m1(2);
    state.M2 = m2(0); state.J2 = m2(1); state.K2 = m2(2);
    state.M = state.M1+state.M2;
    state.J = state.J1+state.J2;
    state.K = state.K1+state.K2;
    state.moment_free_defect = arma::norm(m2rows,"fro");
    state.Mtrap = dx*arma::accu(state.carrier_coeff*fe.carrier_mass_trap);
    if(compress && state.rank) state.Mtrap += dx*arma::accu(state.U*(state.V.t()*fe.mass_trap));
    else state.Mtrap += dx*arma::accu(state.C2*fe.mass_trap);

    // Reflection diagnostics at physical spline nodes, matching MATLAB.
    arma::mat Gnodes(fe.nv,3), Qnodes(fe.nv,fe.nv,arma::fill::zeros);
    for(size_t j = 0; j < fe.nv; ++j) {
        const double v = fe.nodes(j), g = std::exp(-0.5*v*v);
        Gnodes(j,0) = g; Gnodes(j,1) = v*g; Gnodes(j,2) = (v*v-fe.c0)*g;
        const Basis4 b = cubic_bspline(v,fe.knots,fe.nv,fe.vmin,fe.h);
        for(size_t k = 0; k < 4; ++k) Qnodes(j,b.first+k) += b.w[k];
    }
    const arma::mat F1 = spatial.nodal(state.carrier_coeff)*Gnodes.t();
    arma::mat F2(nx,fe.nv,arma::fill::zeros);
    if(compress && state.rank) F2 = spatial.nodal(state.U)*(Qnodes*state.V).t();
    else if(!compress) F2 = spatial.nodal(state.C2)*Qnodes.t();
    state.symmetry = reflection_error(F1+F2);
    state.symmetry1 = reflection_error(F1);
    state.symmetry2 = reflection_error(F2);
    return state;
}

template <size_t order>
void apply_restart(size_t& nt, size_t n, double* coeffs, config_t<double>& conf,
                   const VelocityFE& fe, const SpatialFE& spatial, std::ostream& log)
{
    nufi::stopwatch<double> timer;
    const size_t stride = conf.Nx+order-1;
    RestartState next = build_restart(fe,spatial,conf.x_min,conf.Lx,
        [&](double x, double v) { return periodic::eval_f<double,order>(nt,x,v,coeffs,conf); });
    restart_state = std::move(next);
    const auto& s = restart_state;
    log << std::setprecision(16) << n*dt << ' '
        << s.Msource-conf.Lx << ' ' << s.Jsource << ' '
        << s.M1before-conf.Lx << ' ' << s.J1before << ' '
        << s.M1-conf.Lx << ' ' << s.J1 << ' '
        << s.M2 << ' ' << s.J2 << ' ' << s.K2 << ' '
        << s.M-conf.Lx << ' ' << s.J << ' '
        << s.Mprojected-s.Msource << ' ' << s.Jprojected-s.Jsource << ' '
        << s.Kprojected-s.Ksource << ' ' << s.rank << ' '
        << s.relative_error << ' ' << s.moment_free_defect << ' '
        << s.Mtrap-s.M << ' ' << s.symmetry << ' ' << s.symmetry1 << ' '
        << s.symmetry2 << '\n';
    std::cout << std::setprecision(9) << "t=" << n*dt
              << " source(M-M0)=" << s.Msource-conf.Lx
              << " f1(M-M0)=" << s.M1-conf.Lx
              << " f1_J=" << s.J1 << " f2_M=" << s.M2
              << " f2_J=" << s.J2 << " f2_K=" << s.K2
              << " total(M-M0)=" << s.M-conf.Lx
              << " J=" << s.J << " rank=" << s.rank
              << " LRrel=" << s.relative_error
              << " f2mom=" << s.moment_free_defect << '\n';
    // Keep the potential at restart time as history entry zero.
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
void measure_convergence(size_t n, size_t ntr, size_t nx_baseline, size_t nv_baseline,
                         double* coeffs, const config_t<double>& conf,
                         double electric_energy, double H0, std::ostream& out,
                         bool refine_x = false)
{
    constexpr size_t levels = 12;
    const size_t last_restart = (n/ntr)*ntr;
    const size_t local_nt = n-last_restart;

    const double M0 = conf.Lx;
    const double J0 = 0.0;

    out << std::setprecision(16);
    out << "# timestep " << n << ", time " << n*conf.dt
        << ", last restart " << last_restart << '\n';
    out << "# Nx Nv M J K H relative_mass_error momentum_error relative_energy_error\n";

    for(size_t level = 0; level < levels; ++level) {
        const size_t factor = size_t(1)<<level;
        const size_t nx = refine_x ? nx_baseline*factor : nx_baseline;
        const size_t nv = nv_baseline*factor;

        const double dx = conf.Lx/nx;
        const double dv = (u_max-u_min)/nv;

        long double M = 0.0, J = 0.0, K = 0.0;

        #pragma omp parallel for reduction(+:M,J,K) schedule(static)
        for(std::ptrdiff_t ii = 0; ii < std::ptrdiff_t(nx); ++ii) {
            const double x = conf.x_min+ii*dx;
            long double mx = 0.0, jx = 0.0, kx = 0.0;

            for(size_t j = 0; j < nv; ++j) {
                const double v = u_min+(j+0.5)*dv;
                const double f = periodic::eval_f<double,order>(
                    local_nt,x,v,coeffs,conf);

                mx += f;
                jx += v*f;
                kx += 0.5*v*v*f;
            }

            M += mx;
            J += jx;
            K += kx;
        }

        const double mass = dx*dv*double(M);
        const double momentum = dx*dv*double(J);
        const double kinetic = dx*dv*double(K);
        const double H = kinetic+electric_energy;

        out << nx << ' ' << nv << ' '
            << mass << ' ' << momentum << ' ' << kinetic << ' ' << H << ' '
            << (mass-M0)/M0 << ' '
            << momentum-J0 << ' '
            << (H-H0)/H0 << '\n';
    }
}

template <size_t order>
void run()
{
    static_assert(Nx%2 == 0 && fe_velocity_nodes >= 6,"Invalid spline grid");
    static_assert(restart_every > 0 && measurement_every > 0,"Invalid output period");
    VelocityFE fe(u_min,u_max,fe_velocity_nodes,fe_gauss_order,nv_meas_uniform);
    SpatialFE spatial(Nx,(x_max-x_min)/Nx);
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
    restarts << "# t source_dM source_J before_f1_dM before_f1_J f1_dM f1_J f2_M f2_J f2_K total_dM total_J proj_dM proj_dJ proj_dK rank lr_relative_error f2_moment_defect trap_dM symmetry symmetry1 symmetry2\n";
    std::cout << "Analytic f1 / cubic FE f2, constrained MJ=" << enforce_f1_initial_MJ
              << ", rank(f2)=" << (use_low_rank_compression ? lr_rank : fe_velocity_nodes)
              << ", ntr=" << restart_every << '\n';
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
            apply_restart<order>(nt,n,coeffs.get(),conf,fe,spatial,restarts);
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
        if(n==50 || n == 150 || n == 250 || n==350 || n == 450) {
            std::ofstream conv_anal("conv_anal_" + std::to_string(n*conf.dt) + ".txt");
            measure_convergence<order>(
                n,restart_every,8192,64,coeffs.get(),conf,
                electric_energy,H0,conv_anal);
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
