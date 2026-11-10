#include <vector>
#include <utility>
#include <stdexcept>
#include <limits>
#include <cstddef>
#include <array>
#include <algorithm>
#include <cmath>
#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include <armadillo>

#include <nufi/config.hpp>
#include <nufi/fields.hpp>
#include <nufi/Maxwell.hpp>
#include <nufi/poisson.hpp>
#include <nufi/random.hpp>
#include <nufi/rho.hpp>
#include <nufi/stopwatch.hpp>

namespace nufi
{

namespace dim1
{

namespace electron_ion_shock
{

    const double mi = 1;
    //const double me = 1.0/100.;
    const double me = 1.0/25.;
    const double vth_e = 1e-2;
    const double vth_i = vth_e / 5 /* 1e-3 */;
    const double B0z = 2.8e-2;
    const double v_drift = 0.1;
    const double Lx = 30;
    const double xmin = 0;
    const double xmax = Lx;
    const double Ly = 1;
    const double umin_e = -0.5;
    const double umax_e = 0.5;
    const double vmin_e = -0.5;
    const double vmax_e = 0.5;
    const double umin_i = -0.5;
    const double umax_i = 0.5;
    const double vmin_i = -0.25;
    const double vmax_i = 0.25;

    const double boundary_buffer = 5;
    const double buffer_strength = 0.05;

    double density_profile(double x)
    {
        double A_left = 0.5 * (1.0 + std::tanh((x - boundary_buffer) / buffer_strength));
        double A_right = 0.5 * (1.0 + std::tanh((Lx - boundary_buffer - x) / buffer_strength));
        return A_left * A_right;
    }   

    double drift_velocity(double x)
    {
        return -v_drift * std::tanh((x - 0.5 * Lx) / buffer_strength);
    }

    double E0_y(double x)
    {
        double density = density_profile(x);
        double ux = drift_velocity(x);
        return density * ux * B0z;
    }  

    double f0_e_1x2v(double x, double u, double v)
    {
        return density_profile(x) * maxwellian_1d(u-drift_velocity(x),vth_e) 
                                    * maxwellian_1d(v,vth_e); 
    }

    double f0_i_1x2v(double x, double u, double v)
    {
        return density_profile(x) * maxwellian_1d(u-drift_velocity(x),vth_i) 
                                    * maxwellian_1d(v,vth_i); 
    }

}



// Kormann's Streaming Weibel Instability
//const double Lx = 2*M_PI/0.2;
const double umin = -0.5;
const double umax = 0.5;
const double vmin = -1.2;
const double vmax = 1.2;

// Weak Landau
/* const double Lx = 2*M_PI/0.5;
const double umin = -5;
const double umax = 5;
const double vmin = -5;
const double vmax = 5; */

// TSI
/* const double Lx = 2*M_PI/0.5;
const double umin = -8;
const double umax = 8;
const double vmin = -5;
const double vmax = 5; */

// Paul & Fabio magnetic TSI (Filamentation instability) 
/* const double trigger_k = 2;
const double Lx = 2*M_PI/trigger_k;
const double umin = -1;
const double umax = 1;
const double vmin = -1.2;
const double vmax = 1.2; */

// Fabio 1x2v shock
const double Lx = electron_ion_shock::Lx;
const double umin_e = electron_ion_shock::umin_e;
const double umax_e = electron_ion_shock::umax_e;
const double vmin_e = electron_ion_shock::vmin_e;
const double vmax_e = electron_ion_shock::vmax_e;
const double umin_i = electron_ion_shock::umin_i;
const double umax_i = electron_ion_shock::umax_i;
const double vmin_i = electron_ion_shock::vmin_i;
const double vmax_i = electron_ion_shock::vmax_i;

const size_t Nx = 128;

// Single species
const size_t Nu = 64;
const size_t Nv = 64;

// For multi-species:
const size_t Nu_e = 128;
const size_t Nv_e = 128;
const size_t Nu_i = 2048;
const size_t Nv_i = 1024;

const size_t steps_per_1 = 20;
const double   dt = 1.0 / steps_per_1;
const size_t Nt = 100/dt;

bool strang_split = true;
bool gauss_clean = true;
bool with_filter = true;

size_t nt_restart = 20;

// Tensor-product cubic B-spline Galerkin restarts with optional weighted RSVD.
namespace fe_lr_1x2v {

constexpr bool use_low_rank_compression = true;
constexpr size_t lr_rank = 16;
constexpr size_t lr_oversampling = 6;
constexpr size_t lr_power_iterations = 1;
constexpr double lr_relative_tol = 0.0;
constexpr size_t lr_seed = 12345;
constexpr size_t fe_gauss_order = 32;
// FE degrees of freedom are independent of the velocity quadrature in NuFI.
// The defaults match the existing restart grids. Large multispecies grids
// require considerable memory and projection work, especially with 32^2 quadrature.
constexpr size_t fe_nu = Nu+1, fe_nv = Nv+1;
constexpr size_t fe_nu_e = Nu_e+1, fe_nv_e = Nv_e+1;
constexpr size_t fe_nu_i = Nu_i+1, fe_nv_i = Nv_i+1;

struct Basis4 {
    size_t first;
    std::array<double,4> w;
};

struct GaussRule { arma::vec x, w; };

GaussRule gauss_legendre(size_t n)
{
    if(n < 4) throw std::invalid_argument("Gauss order must be at least four");
    arma::mat J(n,n,arma::fill::zeros);
    for(size_t k = 1; k < n; ++k) {
        const double a = double(k)/std::sqrt(4.0*k*k-1.0);
        J(k-1,k) = J(k,k-1) = a;
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

Basis4 cubic_bspline(double x, const arma::vec& knots, size_t n, double xmin, double h) noexcept
{
    const double z = std::floor((x-(xmin+2.0*h))/h)+4.0;
    const size_t span = size_t(std::max(3.0,std::min(double(n-1),z)));
    std::array<double,4> N{1.0,0.0,0.0,0.0};
    double left[4]{}, right[4]{};
    for(size_t j = 1; j <= 3; ++j) {
        left[j] = x-knots(span+1-j);
        right[j] = knots(span+j)-x;
        double saved = 0.0;
        for(size_t r = 0; r < j; ++r) {
            const double den = right[r+1]+left[j-r];
            const double temp = den != 0.0 ? N[r]/den : 0.0;
            N[r] = saved+right[r+1]*temp;
            saved = left[j-r]*temp;
        }
        N[j] = saved;
    }
    return {span-3,N};
}

struct AxisFE {
    size_t n;
    double lo, hi, h;
    arma::vec nodes, knots, xq, wq;
    arma::mat cholL, mass, polynomial;
    std::vector<Basis4> basis_q;

    AxisFE(double lower, double upper, size_t count, size_t gauss_order):
        n(count), lo(lower), hi(upper), h((upper-lower)/(count-1))
    {
        if(n < 6 || hi <= lo) throw std::invalid_argument("Invalid FE velocity grid");
        nodes.set_size(n);
        for(size_t i = 0; i < n; ++i) nodes(i) = lo+i*h;
        knots.set_size(n+4);
        for(size_t i = 0; i < 4; ++i) knots(i) = lo;
        for(size_t i = 2; i <= n-3; ++i) knots(i+2) = nodes(i);
        for(size_t i = n; i < n+4; ++i) knots(i) = hi;

        const GaussRule g = gauss_legendre(gauss_order);
        xq.set_size((n-1)*gauss_order);
        wq.set_size(xq.n_elem);
        basis_q.reserve(xq.n_elem);
        mass.zeros(n,n);
        for(size_t cell = 0; cell+1 < n; ++cell) {
            const double mid = 0.5*(nodes(cell)+nodes(cell+1));
            const double half = 0.5*(nodes(cell+1)-nodes(cell));
            for(size_t j = 0; j < gauss_order; ++j) {
                const size_t q = cell*gauss_order+j;
                xq(q) = mid+half*g.x(j);
                wq(q) = half*g.w(j);
                const Basis4 b = cubic_bspline(xq(q),knots,n,lo,h);
                basis_q.push_back(b);
                for(size_t a = 0; a < 4; ++a)
                    for(size_t c = 0; c < 4; ++c)
                        mass(b.first+a,b.first+c) += wq(q)*b.w[a]*b.w[c];
            }
        }
        mass = 0.5*(mass+mass.t());
        if(!arma::chol(cholL,mass,"lower"))
            throw std::runtime_error("B-spline FE mass matrix is not SPD");

        // Exact B-spline coefficients of 1, coordinate and coordinate^2.
        polynomial.set_size(n,3);
        for(size_t i = 0; i < n; ++i) {
            const double a = knots(i+1), b = knots(i+2), c = knots(i+3);
            polynomial(i,0) = 1.0;
            polynomial(i,1) = (a+b+c)/3.0;
            polynomial(i,2) = (a*b+a*c+b*c)/3.0;
        }
    }

    arma::mat solve_mass(const arma::mat& rhs) const
    {
        const arma::mat y = arma::solve(arma::trimatl(cholL),rhs);
        return arma::solve(arma::trimatu(cholL.t()),y);
    }
};

struct VelocityFE {
    AxisFE u, v;
    size_t nu, nv, size;
    arma::mat carrier, moments, G;

    VelocityFE(double ulo, double uhi, double vlo, double vhi, size_t nu_, size_t nv_, size_t ng):
        u(ulo,uhi,nu_,ng), v(vlo,vhi,nv_,ng), nu(nu_), nv(nv_), size(nu_*nv_)
    {
        carrier.set_size(size,4);
        carrier.col(0) = arma::kron(v.polynomial.col(0),u.polynomial.col(0));
        carrier.col(1) = arma::kron(v.polynomial.col(0),u.polynomial.col(1));
        carrier.col(2) = arma::kron(v.polynomial.col(1),u.polynomial.col(0));
        carrier.col(3) = 0.5*(arma::kron(v.polynomial.col(0),u.polynomial.col(2))+
                              arma::kron(v.polynomial.col(2),u.polynomial.col(0)));
        const arma::mat pu = u.mass*u.polynomial, pv = v.mass*v.polynomial;
        moments.set_size(size,4);
        moments.col(0) = arma::kron(pv.col(0),pu.col(0));
        moments.col(1) = arma::kron(pv.col(0),pu.col(1));
        moments.col(2) = arma::kron(pv.col(1),pu.col(0));
        moments.col(3) = 0.5*(arma::kron(pv.col(0),pu.col(2))+
                              arma::kron(pv.col(2),pu.col(0)));
        G = carrier.t()*moments;
        if(arma::rcond(G) < 1e-12)
            throw std::runtime_error("Velocity-moment carrier Gram matrix ill-conditioned");
    }

    // Transform coefficients to velocity-L2 Euclidean coordinates:
    // vec(F) = (Lv^T kron Lu^T) vec(C), with Mu=Lu Lu^T, Mv=Lv Lv^T.
    arma::mat weight(const arma::mat& coeff) const
    {
        arma::mat result(coeff.n_rows,size);
        for(size_t i = 0; i < coeff.n_rows; ++i) {
            const arma::mat C = arma::reshape(coeff.row(i).t(),nu,nv);
            const arma::mat F = u.cholL.t()*C*v.cholL;
            result.row(i) = arma::vectorise(F).t();
        }
        return result;
    }

    arma::mat unweight_factors(const arma::mat& weighted) const
    {
        arma::mat result(size,weighted.n_cols);
        for(size_t k = 0; k < weighted.n_cols; ++k) {
            const arma::mat W = arma::reshape(weighted.col(k),nu,nv);
            const arma::mat X = arma::solve(arma::trimatu(u.cholL.t()),W);
            const arma::mat V = arma::solve(arma::trimatu(v.cholL.t()),X.t()).t();
            result.col(k) = arma::vectorise(V);
        }
        return result;
    }
};

arma::mat periodic_second_derivatives(const arma::mat& C, double dx)
{
    const size_t nx = C.n_rows;
    arma::mat rhs(nx,C.n_cols);
    for(size_t i = 0; i < nx; ++i)
        rhs.row(i) = 6.0*(C.row((i+1)%nx)-2.0*C.row(i)+C.row((i+nx-1)%nx))/(dx*dx);
    arma::cx_mat fft = arma::fft(rhs);
    for(size_t k = 0; k < nx; ++k)
        fft.row(k) /= 4.0+2.0*std::cos(2.0*M_PI*k/nx);
    return arma::real(arma::ifft(fft));
}

struct RSVD { arma::mat U, V; arma::vec s; };

RSVD randomized_svd(const arma::mat& F, size_t rank, size_t oversampling, size_t power)
{
    const size_t ell = std::min(size_t(std::min(F.n_rows,F.n_cols)),rank+oversampling);
    if(!ell) return {};
    arma::mat Q, R;
    if(!arma::qr_econ(Q,R,F*arma::randn<arma::mat>(F.n_cols,ell)))
        throw std::runtime_error("RSVD initial QR failed");
    for(size_t i = 0; i < power; ++i) {
        arma::mat P;
        if(!arma::qr_econ(P,R,F.t()*Q)) throw std::runtime_error("RSVD right QR failed");
        if(!arma::qr_econ(Q,R,F*P)) throw std::runtime_error("RSVD left QR failed");
    }
    arma::mat Us, V;
    arma::vec s;
    if(!arma::svd_econ(Us,s,V,Q.t()*F)) throw std::runtime_error("RSVD small SVD failed");
    const size_t r = std::min(rank,size_t(s.n_elem));
    if(!r) return {};
    return {Q*Us.cols(0,r-1),V.cols(0,r-1),s.head(r)};
}

struct RestartState {
    const VelocityFE* fe = nullptr;
    arma::mat C, Cxx;          // Full coefficients when LR disabled.
    arma::mat X, Xxx, V;       // Four moment factors plus compressed remainder.
    bool compressed = false;
    size_t remainder_rank = 0;
    double x0 = 0.0, Lx = 0.0, dx = 0.0, relative_error = 0.0;
    arma::rowvec source_moments, stored_moments;
    double f2_moment_defect = 0.0;

    double eval(double x, double u, double v) const noexcept
    {
        if(!fe || u < fe->u.lo || u > fe->u.hi || v < fe->v.lo || v > fe->v.hi) return 0.0;
        x = std::fmod(x-x0,Lx);
        if(x < 0.0) x += Lx;
        const size_t nx = compressed ? X.n_rows : C.n_rows;
        const double z = x/dx;
        const size_t i = std::min(size_t(std::floor(z)),nx-1), j = (i+1)%nx;
        const double t = z-i, a = 1.0-t;
        const double b0 = (a*a*a-a)*dx*dx/6.0, b1 = (t*t*t-t)*dx*dx/6.0;
        const Basis4 bu = cubic_bspline(u,fe->u.knots,fe->nu,fe->u.lo,fe->u.h);
        const Basis4 bv = cubic_bspline(v,fe->v.knots,fe->nv,fe->v.lo,fe->v.h);
        double result = 0.0;
        if(compressed) {
            for(size_t k = 0; k < X.n_cols; ++k) {
                double vk = 0.0;
                for(size_t jv = 0; jv < 4; ++jv)
                    for(size_t ju = 0; ju < 4; ++ju)
                        vk += bu.w[ju]*bv.w[jv]*V(bu.first+ju+fe->nu*(bv.first+jv),k);
                const double xk = a*X(i,k)+t*X(j,k)+b0*Xxx(i,k)+b1*Xxx(j,k);
                result += xk*vk;
            }
        } else {
            for(size_t jv = 0; jv < 4; ++jv)
                for(size_t ju = 0; ju < 4; ++ju) {
                    const size_t k = bu.first+ju+fe->nu*(bv.first+jv);
                    result += bu.w[ju]*bv.w[jv]*(a*C(i,k)+t*C(j,k)+b0*Cxx(i,k)+b1*Cxx(j,k));
                }
        }
        return result;
    }
};

RestartState single_state, electron_state, ion_state;

double eval_single(double x, double u, double v) noexcept { return single_state.eval(x,u,v); }
double eval_electron(double x, double u, double v) noexcept { return electron_state.eval(x,u,v); }
double eval_ion(double x, double u, double v) noexcept { return ion_state.eval(x,u,v); }

template <typename Eval>
RestartState build_restart(const VelocityFE& fe, size_t nx, double x0, double length, Eval&& eval,
                           bool compress = use_low_rank_compression)
{
    RestartState out;
    out.fe = &fe;
    out.x0 = x0;
    out.Lx = length;
    out.dx = length/nx;
    out.C.set_size(nx,fe.size);
    arma::mat source(nx,4,arma::fill::zeros);
    #pragma omp parallel for schedule(static)
    for(std::ptrdiff_t ii = 0; ii < std::ptrdiff_t(nx); ++ii) {
        const double x = x0+ii*out.dx;
        arma::mat rhs(fe.nu,fe.nv,arma::fill::zeros);
        long double m0 = 0.0, mx = 0.0, my = 0.0, energy = 0.0;
        for(size_t qu = 0; qu < fe.u.xq.n_elem; ++qu) {
            const Basis4& bu = fe.u.basis_q[qu];
            const double u = fe.u.xq(qu), wu = fe.u.wq(qu);
            for(size_t qv = 0; qv < fe.v.xq.n_elem; ++qv) {
                const Basis4& bv = fe.v.basis_q[qv];
                const double v = fe.v.xq(qv);
                const double wf = (wu*fe.v.wq(qv))*eval(x,u,v);
                m0 += wf; mx += u*wf; my += v*wf; energy += 0.5*(u*u+v*v)*wf;
                for(size_t ju = 0; ju < 4; ++ju)
                    for(size_t jv = 0; jv < 4; ++jv)
                        rhs(bu.first+ju,bv.first+jv) += wf*bu.w[ju]*bv.w[jv];
            }
        }
        const arma::mat projected_u = fe.u.solve_mass(rhs);
        const arma::mat projected = fe.v.solve_mass(projected_u.t()).t();
        out.C.row(ii) = arma::vectorise(projected).t();
        source(ii,0) = double(m0);
        source(ii,1) = double(mx);
        source(ii,2) = double(my);
        source(ii,3) = double(energy);
    }
    out.source_moments = out.dx*arma::sum(source,0);
    const arma::mat projected_moments = out.C*fe.moments;
    if(compress) {
        const arma::mat A = arma::solve(fe.G,projected_moments.t()).t();
        arma::mat C2 = std::move(out.C);
        for(size_t i = 0; i < nx; ++i) C2.row(i) -= A.row(i)*fe.carrier.t();
        arma::mat F = fe.weight(C2);
        C2.reset();
        const double normF = arma::norm(F,"fro");
        const size_t rank = std::min(lr_rank,size_t(std::min(F.n_rows,F.n_cols)));
        const RSVD s = normF > 0.0 ? randomized_svd(F,rank,lr_oversampling,lr_power_iterations) : RSVD{};
        size_t r = s.s.n_elem;
        if(r && lr_relative_tol > 0.0 && s.s(0) > 0.0)
            while(r && s.s(r-1) <= lr_relative_tol*s.s(0)) --r;
        out.compressed = true;
        out.remainder_rank = r;
        out.X = A;
        out.V = fe.carrier;
        if(r) {
            const arma::mat spatial = s.U.cols(0,r-1)*arma::diagmat(s.s.head(r));
            const arma::mat velocity = fe.unweight_factors(s.V.cols(0,r-1));
            out.X = arma::join_rows(A,spatial);
            out.V = arma::join_rows(fe.carrier,velocity);
            const arma::mat W = fe.weight(velocity.t());  // r x (Nu*Nv)
            long double err2 = 0.0;
            for(size_t i = 0; i < nx; ++i) {
                const arma::rowvec residual = F.row(i)-spatial.row(i)*W;
                err2 += arma::dot(residual,residual);
            }
            out.relative_error = std::sqrt(double(err2))/normF;
            out.f2_moment_defect = arma::norm(spatial*(velocity.t()*fe.moments),"fro");
        } else {
            out.relative_error = normF > 0.0 ? 1.0 : 0.0;
        }
        out.stored_moments = out.dx*arma::sum(out.X*(out.V.t()*fe.moments),0);
        out.Xxx = periodic_second_derivatives(out.X,out.dx);
        out.C.reset();
    } else {
        out.stored_moments = out.dx*arma::sum(projected_moments,0);
        out.Cxx = periodic_second_derivatives(out.C,out.dx);
    }
    return out;
}

template <typename Eval>
void update(RestartState& state, const VelocityFE& fe, const char* label, size_t step,
            size_t nx, double x0, double length, Eval&& eval, std::ostream& log)
{
    RestartState next = build_restart(fe,nx,x0,length,std::forward<Eval>(eval));
    state = std::move(next);
    const arma::rowvec defect = state.stored_moments-state.source_moments;
    log << std::setprecision(16) << step*dt << ' ' << label << ' '
        << defect(0) << ' ' << defect(1) << ' ' << defect(2) << ' ' << defect(3) << ' '
        << state.remainder_rank << ' ' << state.relative_error << ' ' << state.f2_moment_defect << '\n';
    std::cout << std::setprecision(9) << "t=" << step*dt << ' ' << label
              << " FE-LR: dM=" << defect(0) << " dJu=" << defect(1)
              << " dJv=" << defect(2) << " dK=" << defect(3)
              << " rank=" << state.remainder_rank << " LRerr=" << state.relative_error
              << " f2mom=" << state.f2_moment_defect << '\n';
}

} // namespace fe_lr_1x2v

template <typename real>
real f0(real x, real u) noexcept
{
    // Careful! Only placeholder!
    return -1;
}

template <typename real>
real f0_1x2v(real x, real u, real v) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

    // Kormann Streaming Weibel instability 
    real omega = 0.1/std::sqrt(2);
    real theta = 0.2;
    real beta = 1e-3;
    real v0_1 = 0.5;
    real v0_2 = -0.1;
    real delta = 1.0/6.0;

    return maxwellian_1d(u,omega) 
            * ( delta*maxwellian_1d(v-v0_1,omega) 
            + (1-delta)*maxwellian_1d(v-v0_2,omega) );

    // Weak Landau Damping
    /* real alpha = 0.01;
    real k = 0.5;
    return (1 + alpha * std::cos(k*x)) * maxwellian_1d(u,1.0) * maxwellian_1d(v,1.0); */

    // TSI
    /* real alpha = 0.01;
    real k = 0.5;
    return (1 + alpha * std::cos(k*x)) * u*u * maxwellian_1d(u,1.0) * maxwellian_1d(v,1.0); */

    // Paul & Fabio magnetic TSI (Filamentation instability) 
    /* real v_beam = 0.4;
    real vth = 0.1;
    return 0.5 * (maxwellian_2d<real>(u,v-v_beam,vth) + maxwellian_2d<real>(u,v+v_beam,vth)); */
}

template <typename real>
real f0_electron_1x2v(real x, real u, real v) noexcept
{
    return electron_ion_shock::f0_e_1x2v(x,u,v);
}

template <typename real>
real f0_ion_1x2v(real x, real u, real v) noexcept
{
    return electron_ion_shock::f0_i_1x2v(x,u,v);
}

template <typename real>
arma::Col<real> E0(real x, real y, real z)
{
    // Kormann Streaming & Filamentation instability
    //return  arma::Col<real>({0, 0, 0});

    // Weak Landau & TSI
    /* constexpr real alpha = 1e-2;
    constexpr real k     = 0.5;
    return  arma::Col<real>({-alpha / k * std::sin(k*x), 0, 0}); */

    // Fabio (non-relativistic) electron-ion shock
    return  arma::Col<real>({0, electron_ion_shock::E0_y(x), 0});
}

template <typename real>
arma::Col<real> B0(real x, real y, real z)
{
    // Kormann's Streaming Weibel instability
    /* constexpr real theta = 0.2;
    constexpr real beta = 1e-3;
    return arma::Col<real>({0, 0, beta*std::sin(theta*x)}); */

    // Filamentation instability
    /* constexpr real beta = 1e-3;
    return arma::Col<real>({0, 0, beta*std::sin(trigger_k*x)}); */

    // Electro-static: Landau Damping & TSI
    //return arma::Col<real>({0, 0, 0});

    // Electro-static: Landau Damping & TSI
    return arma::Col<real>({0, 0, electron_ion_shock::B0z});
}


template<typename real, size_t order>
void do_stats_1x2v(size_t nt, double kinetic_energy, double entropy, std::ofstream& stat_file, 
    const std::vector<real>& coeffs_Ex, const std::vector<real>& coeffs_Ey, 
    const std::vector<real>& coeffs_Bz, const config_t<double>& conf, 
    bool restarted = false, size_t n_full = 0, size_t nx_plot = 128, 
    bool with_plot = false)
{
    const size_t stride_t = (conf.Nx + order - 1);

    const size_t dim = 1;
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Nspace = Nx_ext;

    double dx_plot = conf.Lx/nx_plot; 
    double electric_energy = 0;
    double magnetic_energy = 0;

    double electric_x_energy = 0;
    double electric_y_energy = 0;
    double magnetic_z_energy = 0;

    double current_time = 0;
    if(restarted){
        current_time = n_full * conf.dt;
    } else {
        current_time = nt * conf.dt;
    }

    if(with_plot){
        std::ofstream Ex_str("Ex_" + std::to_string(current_time) + ".txt");
        std::ofstream dxEx_str("dxEx_" + std::to_string(current_time) + ".txt");
        std::ofstream Ey_str("Ey_" + std::to_string(current_time) + ".txt");
        std::ofstream Bz_str("Bz_" + std::to_string(current_time) + ".txt");
        std::ofstream dxBz_str("dxBz_" + std::to_string(current_time) + ".txt");
        std::ofstream dxdxBz_str("dxdxBz_" + std::to_string(current_time) + ".txt");

        for(size_t ix = 0; ix < nx_plot; ix++){
            double x = (ix+0.5)*dx_plot;

            double Ex = periodic::eval<real,order>(x,coeffs_Ex.data() + nt*stride_t,conf);
            double dxEx = periodic::eval<real,order,1>(x,coeffs_Ex.data() + nt*stride_t,conf);
            double Ey = periodic::eval<real,order>(x,coeffs_Ey.data() + nt*stride_t,conf);
            double Bz = periodic::eval<real,order>(x,coeffs_Bz.data() + nt*stride_t,conf);
            double dxBz = periodic::eval<real,order,1>(x,coeffs_Bz.data() + nt*stride_t,conf);
            double dxdxBz = periodic::eval<real,order,2>(x,coeffs_Bz.data() + nt*stride_t,conf);

            electric_energy += Ex*Ex + Ey*Ey;
            magnetic_energy += Bz*Bz;

            electric_x_energy += Ex*Ex;
            electric_y_energy += Ey*Ey;

            magnetic_z_energy += Bz*Bz;

            Ex_str << x << " " << Ex << std::endl;
            dxEx_str << x << " " << dxEx << std::endl;
            Ey_str << x << " " << Ey << std::endl;
            Bz_str << x << " " << Bz << std::endl;
            dxBz_str << x << " " << dxBz << std::endl;
            dxdxBz_str << x << " " << dxdxBz << std::endl;
        }
    } else {
        #pragma omp parallel for reduction(+:electric_energy,magnetic_energy,electric_x_energy,electric_y_energy,magnetic_z_energy)
        for(size_t ix = 0; ix < nx_plot; ix++){
            double x = (ix+0.5)*dx_plot;

            double Ex = periodic::eval<real,order>(x,coeffs_Ex.data() + nt*stride_t,conf);
            double dxEx = periodic::eval<real,order,1>(x,coeffs_Ex.data() + nt*stride_t,conf);
            double Ey = periodic::eval<real,order>(x,coeffs_Ey.data() + nt*stride_t,conf);
            double Bz = periodic::eval<real,order>(x,coeffs_Bz.data() + nt*stride_t,conf);

            electric_energy += Ex*Ex + Ey*Ey;
            magnetic_energy += Bz*Bz;

            electric_x_energy += Ex*Ex;
            electric_y_energy += Ey*Ey;

            magnetic_z_energy += Bz*Bz;
        }
    }

    electric_energy *= 0.5*dx_plot;
    electric_x_energy *= 0.5*dx_plot;
    electric_y_energy *= 0.5*dx_plot;
    magnetic_energy *= 0.5*dx_plot;
    magnetic_z_energy *= 0.5*dx_plot;

    double total_energy = electric_energy + magnetic_energy + kinetic_energy;
    stat_file << std::setprecision(15) << current_time << " " << electric_energy << " " << magnetic_energy  << " " 
        << kinetic_energy << " " << total_energy << " " << entropy << " "
        << electric_x_energy << " " << electric_y_energy << " " << magnetic_z_energy << " "
        << std::endl;
    std::cout << current_time << " " << electric_energy << " " << magnetic_energy << std::endl;
}

template<typename real, size_t order>
void kinetic_energy_and_entropy_1x2v(size_t nt, std::ofstream& stat_file, 
    const std::vector<real>& coeffs_Ex, const std::vector<real>& coeffs_Ey, 
    const std::vector<real>& coeffs_Bz, const std::vector<real>& coeffs_jx_hat, 
    const std::vector<real>& coeffs_jy_hat, const config_t<double>& conf, 
    double& kin_energy, double& entropy, bool restarted = false, size_t n_full = 0, 
    size_t nx_plot = 128, size_t nu_plot = 128, size_t nv_plot = 128)
{
    double t = nt*dt;
    if(restarted){
        t = n_full*dt;
    }

    double dx_plot = Lx/nx_plot;
    double du_plot = (umax - umin)/nu_plot;
    double dv_plot = (vmax - vmin)/nv_plot;

    kin_energy = 0;
    entropy = 0;
    double l1_norm = 0;
    double l2_norm = 0;

    #pragma omp parallel for collapse(3) reduction(+:kin_energy,entropy,l1_norm,l2_norm)
    for(size_t ix = 0; ix < nx_plot; ix++)
    for(size_t iu = 0; iu < nu_plot; iu++)
    for(size_t iv = 0; iv < nv_plot; iv++){
        double x = (ix + 0.5)*dx_plot;
        double u = umin + (iu + 0.5)*du_plot;
        double v = vmin + (iv + 0.5)*dv_plot;

        double f = periodic::redux_1x2v::eval_f_nufi_ham_lie_fBE_1x2v<real,order>(nt, x, u, v, coeffs_Ex, coeffs_Ey, coeffs_Bz, coeffs_jx_hat, coeffs_jy_hat, conf );

        kin_energy += (u*u + v*v) * f;
        if(f > 1e-16){
            entropy += f * std::log(f);
        }

        l1_norm += std::abs(f);
        l2_norm += f*f;
    }

    double dplot = dx_plot*du_plot*dv_plot;
    kin_energy *= 0.5*dplot;
    entropy *= dplot;
    l1_norm *= dplot;
    l2_norm = std::sqrt(dplot*l2_norm);

    stat_file << std::setprecision(15) << t << " " << kin_energy << " " << entropy << " " << l1_norm << " " << l2_norm << std::endl;
}

template<typename real, size_t order>
void kinetic_energy_and_entropy_1x2v(size_t nt, std::ofstream& stat_file, 
    const std::vector<real>& coeffs_Ex, const std::vector<real>& coeffs_Ey, 
    const std::vector<real>& coeffs_Bz, const config_t<double>& conf, 
    double& kin_energy, double& entropy, bool restarted = false, size_t n_full = 0, 
    size_t nx_plot = 128, size_t nu_plot = 128, size_t nv_plot = 128, bool plot_f = false, std::string name_add = "")
{
    double t = nt*dt;
    if(restarted){
        t = n_full*dt;
    }

    double dx_plot = Lx/nx_plot;
    double du_plot = (umax - umin)/nu_plot;
    double dv_plot = (vmax - vmin)/nv_plot;

    std::vector<double> f_values;
    if(plot_f){
        f_values.resize(nx_plot*nu_plot*nv_plot);
    }

    kin_energy = 0;
    entropy = 0;
    double l1_norm = 0;
    double l2_norm = 0;

    #pragma omp parallel for collapse(3) reduction(+:kin_energy,entropy,l1_norm,l2_norm)
    for(size_t ix = 0; ix < nx_plot; ix++)
    for(size_t iu = 0; iu < nu_plot; iu++)
    for(size_t iv = 0; iv < nv_plot; iv++){
        double x = (ix + 0.5)*dx_plot;
        double u = umin + (iu + 0.5)*du_plot;
        double v = vmin + (iv + 0.5)*dv_plot;

        double f = 0;
        if(strang_split){
            f = periodic::redux_1x2v::strang_2nd_order::eval_f_nufi_ham_strang_HE_HB_Hf_HB_HE_1x2v<real,order>(nt, x, u, v, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf );
        }else{
            f = periodic::redux_1x2v::eval_f_nufi_ham_lie_Hf_HB_HE_1x2v<real,order>(nt, x, u, v, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf );
        }

        kin_energy += (u*u + v*v) * f;
        if(f > 1e-16){
            entropy += f * std::log(f);
        }

        l1_norm += std::abs(f);
        l2_norm += f*f;

        if(plot_f){
            f_values[ix + nx_plot*(iu + nu_plot*iv)] = f;
        }
    }

    double dplot = dx_plot*du_plot*dv_plot;
    kin_energy *= 0.5*dplot;
    entropy *= dplot;
    l1_norm *= dplot;
    l2_norm = std::sqrt(dplot*l2_norm);

    stat_file << std::setprecision(15) << t << " " << kin_energy << " " << entropy << " " << l1_norm << " " << l2_norm << std::endl;

    if(plot_f){
        std::ofstream f_str("f_" + std::to_string(t) + name_add + ".txt");
        for(size_t l = 0; l < nx_plot*nu_plot*nv_plot; l++){
            f_str << f_values[l] << std::endl;
        }
    }
}

config_t<double> conf(Nx, Nu, Nt, dt, 0, Lx, umin, umax, &f0);

template<size_t order>
void periodically_restarted_nufi_maxwell_lie_fBE_aligned()
{
    // Storage of coefficients now via: 
    // index = nt + Nt * (d + dim * (ix + Nx * (iy + Ny * iz)))
    size_t stride_t = (conf.Nx + order - 1);

    const size_t dim = 1;
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Nspace = Nx_ext;
    
    std::cout << "Start NuFI-Ham Vlasov-Maxwell-Solver with 1x2v redux." << std::endl;
    std::cout << "Init helper variables." << std::endl;
    arma::arma_rng::set_seed(fe_lr_1x2v::lr_seed);

    // Flattened 1D coefficient storage
    std::vector<double> coeffs_Ex((conf.Nt + 1) * stride_t, 0);
    std::vector<double> coeffs_Ey((conf.Nt + 1) * stride_t, 0);
    std::vector<double> coeffs_Bz((conf.Nt + 1) * stride_t, 0);
    std::vector<double> coeffs_jx_hat((conf.Nt + 1) * stride_t, 0);
    std::vector<double> coeffs_jy_hat((conf.Nt + 1) * stride_t, 0);
    std::vector<double> Ex(conf.Nx, 0);
    std::vector<double> Ey(conf.Nx, 0);
    std::vector<double> Bz(conf.Nx, 0);
    std::vector<double> jx_hat(conf.Nx, 0);
    std::vector<double> jy_hat(conf.Nx, 0);


    std::vector<double> coeffs_phi(stride_t, 0);
    std::vector<double> rho(Nx, 0);
    std::vector<double> rho_test(Nx, 0);
    std::vector<double> g(Nx, 0);
    poisson<double> poiss( conf );


    // Set up config.
    conf = config_t<double>(Nx, Nu, Nt, dt, 0, Lx, umin, umax, &f0);
    conf.Nv = Nv;
    conf.v_min = vmin;
    conf.v_max = vmax;
    conf.dv = (vmax - vmin)/Nv;
    conf.f0_1x2v = f0_1x2v;
    fe_lr_1x2v::VelocityFE velocity_fe(conf.u_min,conf.u_max,conf.v_min,conf.v_max,
        fe_lr_1x2v::fe_nu,fe_lr_1x2v::fe_nv,fe_lr_1x2v::fe_gauss_order);
    config_t<double> conf_test(Nx, 2*Nu, Nt, dt, 0, Lx,umin, umax, &f0);
    conf_test.Nv = 2*Nv;
    conf_test.v_min = vmin;
    conf_test.v_max = vmax;
    conf_test.dv = (vmax - vmin)/conf_test.Nv;
    conf_test.f0_1x2v = f0_1x2v;

    // Compute E(0) and B(0).
    std::cout << "Compute E(0) and B(0)." << std::endl;
    #pragma omp parallel for
    for(size_t l = 0; l < conf.Nx; l++){
        double x = conf.x_min + l*conf.dx; 
        
        // Normal initialization:        
        arma::Col<double> E0_vec = E0(x,0.0,0.0);
        arma::Col<double> B0_vec = B0(x,0.0,0.0);

        Ex[l] = E0_vec(0);
        Ey[l] = E0_vec(1);
        Bz[l] = B0_vec(2);
    }

    // Interpolate E(0) and B(0).
    std::cout << "Interpolate Ex(0)." << std::endl;
    periodic::interpolate<double,order>(coeffs_Ex.data(), Ex.data(), conf);
    std::cout << "Interpolate Ey(0)." << std::endl;
    periodic::interpolate<double,order>(coeffs_Ey.data(), Ey.data(), conf);
    std::cout << "Interpolate Bz(0)." << std::endl;
    periodic::interpolate<double,order>(coeffs_Bz.data(), Bz.data(), conf);
    
    // Compute j_hat(0).
    std::cout << "Compute j_hat(0)." << std::endl;

    periodic::redux_1x2v::eval_j_hat<double,order>(0, jx_hat, jy_hat, coeffs_Ex, coeffs_Ey, coeffs_Bz, coeffs_jx_hat, coeffs_jy_hat, conf);

    // Interpolate j_hat(0).
    std::cout << "Interpolate j_hat(0)." << std::endl;
    periodic::interpolate<double,order>(coeffs_jx_hat.data(), jx_hat.data(), conf);
    periodic::interpolate<double,order>(coeffs_jy_hat.data(), jy_hat.data(), conf);
    
    std::cout << "First output." << std::endl;
    std::ofstream stat_file( "stats.txt" );
    std::ofstream lr_restart_file("restart_defects.txt");
    lr_restart_file << "# t species dM dJu dJv dK rank rel_L2_error f2_moment_defect\n";
    std::ofstream kin_energy_entropy_file( "kinetic_energy_and_entropy.txt" );
    double kinetic_energy = 0;
    double entropy = 0;
    // Output stats (Electric/magnetic energy).
    kinetic_energy_and_entropy_1x2v<double,order>(0,kin_energy_entropy_file,coeffs_Ex,coeffs_Ey,coeffs_Bz,coeffs_jx_hat,coeffs_jy_hat,conf, kinetic_energy, entropy,false,0,64,64,64);
    do_stats_1x2v<double,order>(0, kinetic_energy, entropy, stat_file,coeffs_Ex, coeffs_Ey, coeffs_Bz, conf, false, 0, 128, true);

    std::ofstream gle_file( "gle.txt" );
    double rho_integration_error = 0;
    gle_file << 0 << " " << 0 << " " << rho_integration_error << std::endl;

    std::cout << "Restart time-loop." << std::endl;    
    std::cout << " ---------------------------------- " << std::endl;
    double total_time = 0;
    size_t nt_r_curr = 1;
    for(size_t n = 1; n <= conf.Nt; n++)
    {
        nufi::stopwatch<double> timer;
        // Compute E(n) and B(n).
        double Ex_mean = 0, Ey_mean = 0;
        #pragma omp parallel for reduction(+:Ex_mean,Ey_mean)
        for(size_t l = 0; l < conf.Nx; l++){
        
            double x = conf.x_min + l*conf.dx; 
    
            double ex = periodic::eval<double,order>(x,coeffs_Ex.data() + (nt_r_curr-1)*stride_t,conf);
            double ey = periodic::eval<double,order>(x,coeffs_Ey.data() + (nt_r_curr-1)*stride_t,conf);
            double bz = periodic::eval<double,order>(x,coeffs_Bz.data() + (nt_r_curr-1)*stride_t,conf);
            
            double jx = periodic::eval<double,order>(x,coeffs_jx_hat.data() + (nt_r_curr-1)*stride_t,conf);
            double jy = periodic::eval<double,order>(x,coeffs_jy_hat.data() + (nt_r_curr-1)*stride_t,conf);

            ex = ex - conf.dt*jx;
            ey = ey - conf.dt*jy - conf.dt*periodic::eval<double,order,1>(x,coeffs_Bz.data() + (nt_r_curr-1)*stride_t,conf);

            bz = bz - conf.dt*periodic::eval<double,order,1>(x,coeffs_Ey.data()+(nt_r_curr-1)*stride_t,conf)
                - conf.dt*conf.dt*periodic::eval<double,order,1>(x,coeffs_jy_hat.data()+(nt_r_curr-1)*stride_t,conf)
                - conf.dt*conf.dt*periodic::eval<double,order,2>(x,coeffs_Bz.data()+(nt_r_curr-1)*stride_t,conf);

            Ex[l] = ex;
            Ey[l] = ey;
            Bz[l] = bz;

            Ex_mean += ex;
            Ey_mean += ey;
        }
        double time_compute_EB = timer.elapsed();
        std::cout << "Compute EB took " << time_compute_EB << " s." << std::endl;
        timer.reset();

        // Gauge fix!!!
        // I must gauge-fix E here! This means E -= mean(E).        
        Ex_mean /= conf.Nx;
        Ey_mean /= conf.Nx;

        #pragma omp parallel for
        for(size_t l = 0; l < conf.Nx; l++){
            Ex[l] -= Ex_mean;
            Ey[l] -= Ey_mean;
        }

        // Interpolate E(n) and B(n).
        //std::cout << "Interpolate E(n) and B(n)." << std::endl;
        periodic::interpolate<double,order>(coeffs_Ex.data() + nt_r_curr*stride_t, Ex.data(), conf);
        periodic::interpolate<double,order>(coeffs_Ey.data() + nt_r_curr*stride_t, Ey.data(), conf);
        periodic::interpolate<double,order>(coeffs_Bz.data() + nt_r_curr*stride_t, Bz.data(), conf);

        double time_interpolate_EB = timer.elapsed();
        std::cout << "EB interpolation took " << time_interpolate_EB << " s." << std::endl;
        timer.reset();

        // Gauss clean & integration error test
        periodic::redux_1x2v::eval_rho<double,order>(nt_r_curr, rho, coeffs_Ex, coeffs_Ey, coeffs_Bz, 
                        coeffs_jx_hat, coeffs_jy_hat, conf);
        double rho_mean = 0;
        #pragma omp parallel for reduction(+:rho_mean)
        for(size_t i = 0; i < rho.size(); i++){
            // Ions are a uniform background with q = 1, while electrons have q = -1.
            rho[i] = 1 + rho[i];
            rho_mean += rho[i];
        }
        rho_mean /= conf.Nx;
        #pragma omp parallel for
        for(size_t i = 0; i < rho.size(); i++){
            // Ions are a uniform background with q = 1, while electrons have q = -1.
            rho[i] -= rho_mean;
        }

        double gle = periodic::maxwell::E_clean_gauss_law_1x2v<double,order>(nt_r_curr,coeffs_Ex,Ex,rho,g,coeffs_phi,conf,poiss,!gauss_clean);

        if(n % (5*steps_per_1) == 0){
            periodic::redux_1x2v::eval_rho<double,order>(nt_r_curr, rho_test, coeffs_Ex, coeffs_Ey, coeffs_Bz, 
                        coeffs_jx_hat, coeffs_jy_hat, conf_test);
            rho_integration_error = 0;
            double rho_l2_norm = 0;
            rho_mean = 0;
            #pragma omp parallel for reduction(+:rho_integration_error,rho_l2_norm,rho_mean)
            for(size_t i = 0; i < rho.size(); i++){
                rho_test[i] = 1 + rho_test[i];
                rho_mean += rho_test[i];
                rho_l2_norm += rho_test[i]*rho_test[i];
            }

            rho_mean /= conf.Nx;

            #pragma omp parallel for reduction(+:rho_integration_error,rho_l2_norm)
            for(size_t i = 0; i < rho.size(); i++){
                rho_test[i] -= rho_mean;
                double error = rho[i] - rho_test[i];
                rho_integration_error += error*error;
            }

            rho_l2_norm = std::sqrt(conf.dx*rho_l2_norm);
            //rho_integration_error = std::sqrt(conf.dx*rho_integration_error) / rho_l2_norm;
            rho_integration_error = std::sqrt(conf.dx*rho_integration_error);

            if(n % (5*steps_per_1) == 0){
                // Careful: Hardcoded for d = 1! 
                std::ofstream rho_str("rho_" + std::to_string(n*conf.dt) + ".txt");
                for(size_t i = 0; i < conf.Nx; i++){
                    double x = conf.x_min + i * conf.dx;
                    rho_str << x << " " << rho[i] << " " << rho_test[i] << std::endl;
                }

                std::ofstream j_str("j_" + std::to_string(n*conf.dt) + ".txt");
                for(size_t i = 0; i < conf.Nx; i++){
                    double x = conf.x_min + i * conf.dx;
                    j_str << x << " " << jx_hat[i] << " " << jy_hat[i] << std::endl;
                }

            }
        }

        gle_file << n*conf.dt << " " << gle << " " << rho_integration_error << std::endl;

        // Compute j_hat(n).
        periodic::redux_1x2v::eval_j_hat<double,order>(nt_r_curr, jx_hat, jy_hat, coeffs_Ex, coeffs_Ey, coeffs_Bz, 
                    coeffs_jx_hat, coeffs_jy_hat, conf);
        double time_eval_j_hat = timer.elapsed();
        std::cout << "Eval j_hat took " << time_eval_j_hat << " s." << std::endl;
        timer.reset();
        
        // Interpolate j_hat(n).
        periodic::interpolate<double,order>(coeffs_jx_hat.data() + nt_r_curr*stride_t,jx_hat.data(),conf);
        periodic::interpolate<double,order>(coeffs_jy_hat.data() + nt_r_curr*stride_t,jy_hat.data(),conf);
        double time_interpolate_j_hat = timer.elapsed();
        std::cout << "Interpolate j_hat took " << time_interpolate_j_hat << " s." << std::endl;
        timer.reset();

        // Analyze data if wanted.
        // Compute time measurement.
        double time_for_step = time_compute_EB + time_interpolate_EB + time_eval_j_hat + time_interpolate_j_hat;
        total_time += time_for_step;
        std::cout << "Time step " << n << " took a total of " << time_for_step << " s." << std::endl;

        if(n % (steps_per_1) == 0 ){
            kinetic_energy_and_entropy_1x2v<double,order>(nt_r_curr,kin_energy_entropy_file,coeffs_Ex,coeffs_Ey,coeffs_Bz,coeffs_jx_hat,coeffs_jy_hat,conf,kinetic_energy,entropy,true, n,64,64,64);
        }
        do_stats_1x2v<double,order>(nt_r_curr, kinetic_energy, entropy, stat_file, coeffs_Ex,coeffs_Ey,coeffs_Bz,conf, true, n, 128, (n % (5*steps_per_1) == 0));
        //do_stats_1x2v<double,order>(nt_r_curr, kinetic_energy, entropy, stat_file, coeffs_Ex,coeffs_Ey,coeffs_Bz,conf, true, n, 128, true);

        std::cout << "Do stats took: " << double(timer.elapsed()) << " s." << std::endl;
        std::cout << " ---------------------------------- " << std::endl;

        if(nt_r_curr == nt_restart) {
            timer.reset();
            fe_lr_1x2v::update(fe_lr_1x2v::single_state,velocity_fe,"single",n,conf.Nx,
                conf.x_min,conf.Lx,[&](double x,double u,double v) {
                    return periodic::redux_1x2v::eval_f_nufi_ham_lie_fBE_1x2v<double,order>(
                        nt_r_curr,x,u,v,coeffs_Ex,coeffs_Ey,coeffs_Bz,coeffs_jx_hat,coeffs_jy_hat,conf);
                },lr_restart_file);
            for(size_t ix = 0; ix < stride_t; ++ix) {
                coeffs_Ex[ix] = coeffs_Ex[nt_r_curr*stride_t+ix];
                coeffs_Ey[ix] = coeffs_Ey[nt_r_curr*stride_t+ix];
                coeffs_Bz[ix] = coeffs_Bz[nt_r_curr*stride_t+ix];
                coeffs_jx_hat[ix] = coeffs_jx_hat[nt_r_curr*stride_t+ix];
                coeffs_jy_hat[ix] = coeffs_jy_hat[nt_r_curr*stride_t+ix];
            }
            conf.f0_1x2v = fe_lr_1x2v::eval_single;
            conf_test.f0_1x2v = fe_lr_1x2v::eval_single;
            nt_r_curr = 1;
            total_time += timer.elapsed();
        } else {
            nt_r_curr++;
        }
    }

    std::cout << "Total simulation time " << total_time << " s." << std::endl;
}


inline void spectral_filter_hat_1d(
    std::vector<double>& hat_re,
    std::vector<double>& hat_im,
    double alpha = 30.0,
    int p = 8
)
{
    const size_t Nx = hat_re.size();
    const int kmax = static_cast<int>(Nx) / 2;

    for (size_t m = 0; m < Nx; ++m) {
        int kmode = (m <= Nx/2) ? static_cast<int>(m)
                                : static_cast<int>(m) - static_cast<int>(Nx);

        double eta = std::abs(kmode) / static_cast<double>(kmax);
        double sigma = std::exp(-alpha * std::pow(eta, p));

        hat_re[m] *= sigma;
        hat_im[m] *= sigma;
    }
}

template<size_t order>
void periodically_restarted_nufi_maxwell_lie_exact_fourier_integral_aligned()
{
    // Storage of coefficients now via:
    // index = nt + Nt * (d + dim * (ix + Nx * (iy + Ny * iz)))
    const size_t stride_t = (conf.Nx + order - 1);

    const size_t dim = 1;
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Nspace = Nx_ext;

    std::cout << "Start NuFI-Ham Vlasov-Maxwell-Solver with 1x2v redux." << std::endl;
    std::cout << "Init helper variables." << std::endl;
    arma::arma_rng::set_seed(fe_lr_1x2v::lr_seed);

    // Flattened 1D coefficient storage
    std::vector<double> coeffs_Ex((conf.Nt + 1) * stride_t, 0.0);
    std::vector<double> coeffs_Ey((conf.Nt + 1) * stride_t, 0.0);
    std::vector<double> coeffs_Bz((conf.Nt + 1) * stride_t, 0.0);

    std::vector<double> Ex(conf.Nx, 0.0);
    std::vector<double> Ey(conf.Nx, 0.0);
    std::vector<double> Bz(conf.Nx, 0.0);
    std::vector<double> jx_hat(conf.Nx, 0.0);
    std::vector<double> jy_hat(conf.Nx, 0.0);

    std::vector<double> coeffs_phi(stride_t, 0.0);
    std::vector<double> rho(Nx, 0.0);
    std::vector<double> rho_test(Nx, 0.0);
    std::vector<double> g(Nx, 0.0);
    poisson<double> poiss(conf);

    // FFTW work arrays and plans for field update.
    fftw_complex* fft_in  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * conf.Nx);
    fftw_complex* fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * conf.Nx);
    if (fft_in == nullptr || fft_out == nullptr) {
        if (fft_in  != nullptr) fftw_free(fft_in);
        if (fft_out != nullptr) fftw_free(fft_out);
        throw std::runtime_error("FFTW allocation failed in periodically_restarted_nufi_maxwell_lie_exact_fourier_integral_aligned.");
    }

    fftw_plan plan_fwd = fftw_plan_dft_1d(static_cast<int>(conf.Nx), fft_in, fft_out, FFTW_FORWARD, FFTW_MEASURE);
    fftw_plan plan_bwd = fftw_plan_dft_1d(static_cast<int>(conf.Nx), fft_out, fft_in, FFTW_BACKWARD, FFTW_MEASURE);
    if (plan_fwd == nullptr || plan_bwd == nullptr) {
        if (plan_fwd != nullptr) fftw_destroy_plan(plan_fwd);
        if (plan_bwd != nullptr) fftw_destroy_plan(plan_bwd);
        fftw_free(fft_in);
        fftw_free(fft_out);
        throw std::runtime_error("FFTW plan creation failed in periodically_restarted_nufi_maxwell_lie_exact_fourier_integral_aligned.");
    }

    std::vector<double> Ex_hat_re(conf.Nx, 0.0), Ex_hat_im(conf.Nx, 0.0);
    std::vector<double> Ey_hat_re(conf.Nx, 0.0), Ey_hat_im(conf.Nx, 0.0);
    std::vector<double> Bz_hat_re(conf.Nx, 0.0), Bz_hat_im(conf.Nx, 0.0);
    std::vector<double> jx_hat_re(conf.Nx, 0.0), jx_hat_im(conf.Nx, 0.0);
    std::vector<double> jy_hat_re(conf.Nx, 0.0), jy_hat_im(conf.Nx, 0.0);

    const double two_pi_over_Lx = 2.0 * M_PI / conf.Lx;
    const double invNx = 1.0 / static_cast<double>(conf.Nx);


    // Set up config.
    conf = config_t<double>(Nx, Nu, Nt, dt, 0, Lx, umin, umax, &f0);
    conf.Nv = Nv;
    conf.v_min = vmin;
    conf.v_max = vmax;
    conf.dv = (vmax - vmin) / Nv;
    conf.f0_1x2v = f0_1x2v;
    fe_lr_1x2v::VelocityFE velocity_fe(conf.u_min,conf.u_max,conf.v_min,conf.v_max,
        fe_lr_1x2v::fe_nu,fe_lr_1x2v::fe_nv,fe_lr_1x2v::fe_gauss_order);

    config_t<double> conf_test(Nx, 2 * Nu, Nt, dt, 0, Lx, umin, umax, &f0);
    conf_test.Nv = 2 * Nv;
    conf_test.v_min = vmin;
    conf_test.v_max = vmax;
    conf_test.dv = (vmax - vmin) / conf_test.Nv;
    conf_test.f0_1x2v = f0_1x2v;

    // Compute E(0) and B(0).
    std::cout << "Compute E(0) and B(0)." << std::endl;
    #pragma omp parallel for
    for (size_t l = 0; l < conf.Nx; l++) {
        const double x = conf.x_min + l * conf.dx;

        arma::Col<double> E0_vec = E0(x, 0.0, 0.0);
        arma::Col<double> B0_vec = B0(x, 0.0, 0.0);

        Ex[l] = E0_vec(0);
        Ey[l] = E0_vec(1);
        Bz[l] = B0_vec(2);
    }

    // Interpolate E(0) and B(0).
    std::cout << "Interpolate Ex(0)." << std::endl;
    periodic::interpolate<double, order>(coeffs_Ex.data(), Ex.data(), conf);
    std::cout << "Interpolate Ey(0)." << std::endl;
    periodic::interpolate<double, order>(coeffs_Ey.data(), Ey.data(), conf);
    std::cout << "Interpolate Bz(0)." << std::endl;
    periodic::interpolate<double, order>(coeffs_Bz.data(), Bz.data(), conf);

    // Compute j_hat(0).
    std::cout << "Compute j_hat(0)." << std::endl;
    periodic::redux_1x2v::eval_j_time_integral_Hf_exact_1x2v_fourier<order>(0, jx_hat_re, jx_hat_im, jy_hat_re, jy_hat_im, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf);

    std::cout << "First output." << std::endl;
    std::ofstream stat_file("stats.txt");
    std::ofstream lr_restart_file("restart_defects.txt");
    lr_restart_file << "# t species dM dJu dJv dK rank rel_L2_error f2_moment_defect\n";
    std::ofstream kin_energy_entropy_file("kinetic_energy_and_entropy.txt");
    double kinetic_energy = 0.0;
    double entropy = 0.0;
    kinetic_energy_and_entropy_1x2v<double,order>(0,kin_energy_entropy_file,coeffs_Ex,coeffs_Ey,coeffs_Bz,conf, kinetic_energy, entropy,false,0,64,64,64);
    do_stats_1x2v<double, order>(0, kinetic_energy, entropy, stat_file, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf, false, 0, 128, true);

    std::ofstream gle_file("gle.txt");
    double rho_integration_error = 0.0;
    gle_file << 0 << " " << 0 << " " << rho_integration_error << std::endl;

    std::cout << "Restart time-loop." << std::endl;
    std::cout << " ---------------------------------- " << std::endl;
    double total_time = 0.0;
    size_t nt_r_curr = 1;

    for (size_t n = 1; n <= conf.Nt; n++) {
        nufi::stopwatch<double> timer;

        // Read fields/current integrals at time level nt_r_curr-1 on the physical grid.
        #pragma omp parallel for
        for (size_t l = 0; l < conf.Nx; l++) {
            const double x = conf.x_min + l * conf.dx;

            Ex[l] = periodic::eval<double, order>(x, coeffs_Ex.data() + (nt_r_curr - 1) * stride_t, conf);
            Ey[l] = periodic::eval<double, order>(x, coeffs_Ey.data() + (nt_r_curr - 1) * stride_t, conf);
            Bz[l] = periodic::eval<double, order>(x, coeffs_Bz.data() + (nt_r_curr - 1) * stride_t, conf);
        }

        // FFT Ex
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_in[l][0] = Ex[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Ex_hat_re[l] = fft_out[l][0];
            Ex_hat_im[l] = fft_out[l][1];
        }

        // FFT Ey
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_in[l][0] = Ey[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Ey_hat_re[l] = fft_out[l][0];
            Ey_hat_im[l] = fft_out[l][1];
        }

        // FFT Bz
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_in[l][0] = Bz[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Bz_hat_re[l] = fft_out[l][0];
            Bz_hat_im[l] = fft_out[l][1];
        }

        // Fourier field update analogous to the Matlab code:
        // 1) H_f: Ex_hat -= Jx_int_hat, Ey_hat -= Jy_int_hat
        // 2) H_B: Ey_hat -= dt * (i k) * Bz_hat
        // 3) H_E: Bz_hat -= dt * (i k) * Ey_hat   (using updated Ey_hat from H_B)
        for (size_t m = 0; m < conf.Nx; m++) {
            Ex_hat_re[m] -= jx_hat_re[m];
            Ex_hat_im[m] -= jx_hat_im[m];

            Ey_hat_re[m] -= jy_hat_re[m];
            Ey_hat_im[m] -= jy_hat_im[m];
        }

        // Gauge fix: Enforce zero mean of Ex in Fourier space
        Ex_hat_re[0] = 0.0;
        Ex_hat_im[0] = 0.0;

        for (size_t m = 0; m < conf.Nx; m++) {
            const int kmode = (m <= conf.Nx / 2) ? static_cast<int>(m) : static_cast<int>(m) - static_cast<int>(conf.Nx);
            const double k = two_pi_over_Lx * static_cast<double>(kmode);

            const double bz_re = Bz_hat_re[m];
            const double bz_im = Bz_hat_im[m];

            // i*k*Bz_hat = (-k*bz_im) + i*(k*bz_re)
            Ey_hat_re[m] -= conf.dt * (-k * bz_im);
            Ey_hat_im[m] -= conf.dt * ( k * bz_re);
        }

        // Gauge fix: Enforce zero mean of Ex in Fourier space 
        Ey_hat_re[0] = 0.0;
        Ey_hat_im[0] = 0.0;

        for (size_t m = 0; m < conf.Nx; m++) {
            const int kmode = (m <= conf.Nx / 2) ? static_cast<int>(m) : static_cast<int>(m) - static_cast<int>(conf.Nx);
            const double k = two_pi_over_Lx * static_cast<double>(kmode);

            const double ey_re = Ey_hat_re[m];
            const double ey_im = Ey_hat_im[m];

            // i*k*Ey_hat = (-k*ey_im) + i*(k*ey_re)
            Bz_hat_re[m] -= conf.dt * (-k * ey_im);
            Bz_hat_im[m] -= conf.dt * ( k * ey_re);
        }

        if(with_filter){
            spectral_filter_hat_1d(Ey_hat_re, Ey_hat_im);
            spectral_filter_hat_1d(Bz_hat_re, Bz_hat_im);
            spectral_filter_hat_1d(Ex_hat_re, Ex_hat_im);
        }

        // IFFT Ex
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_out[l][0] = Ex_hat_re[l];
            fft_out[l][1] = Ex_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Ex[l] = fft_in[l][0] * invNx;
        }

        // IFFT Ey
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_out[l][0] = Ey_hat_re[l];
            fft_out[l][1] = Ey_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Ey[l] = fft_in[l][0] * invNx;
        }

        // IFFT Bz
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_out[l][0] = Bz_hat_re[l];
            fft_out[l][1] = Bz_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Bz[l] = fft_in[l][0] * invNx;
        }

        double time_compute_EB = timer.elapsed();
        std::cout << "Compute EB took " << time_compute_EB << " s." << std::endl;
        timer.reset();

        // Interpolate E(n) and B(n).
        periodic::interpolate<double, order>(coeffs_Ex.data() + nt_r_curr * stride_t, Ex.data(), conf);
        periodic::interpolate<double, order>(coeffs_Ey.data() + nt_r_curr * stride_t, Ey.data(), conf);
        periodic::interpolate<double, order>(coeffs_Bz.data() + nt_r_curr * stride_t, Bz.data(), conf);

        double time_interpolate_EB = timer.elapsed();
        std::cout << "EB interpolation took " << time_interpolate_EB << " s." << std::endl;
        timer.reset();

        // Gauss clean & integration error test
        periodic::redux_1x2v::eval_rho_ham_lie_Hf_HB_HE_1x2v<double, order>(nt_r_curr, rho, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf);
        double rho_mean = 0.0;
        #pragma omp parallel for reduction(+:rho_mean)
        for (size_t i = 0; i < rho.size(); i++) {
            rho[i] = 1 + rho[i];
            rho_mean += rho[i];
        }
        rho_mean /= conf.Nx;

        #pragma omp parallel for
        for (size_t i = 0; i < rho.size(); i++) {
            rho[i] -= rho_mean;
        }

        double gle = periodic::maxwell::E_clean_gauss_law_1x2v<double, order>(
            nt_r_curr, coeffs_Ex, Ex, rho, g, coeffs_phi, conf, poiss, !gauss_clean
        );

        if (n % (steps_per_1) == 0) {
        //if (true) {
            periodic::redux_1x2v::eval_rho_ham_lie_Hf_HB_HE_1x2v<double, order>(nt_r_curr, rho_test, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf_test);
            rho_integration_error = 0.0;
            double rho_l2_norm = 0.0;
            rho_mean = 0.0;

            #pragma omp parallel for reduction(+:rho_integration_error,rho_l2_norm,rho_mean)
            for (size_t i = 0; i < rho.size(); i++) {
                rho_test[i] = 1 + rho_test[i];
                rho_mean += rho_test[i];
                rho_l2_norm += rho_test[i] * rho_test[i];
            }

            rho_mean /= conf.Nx;

            #pragma omp parallel for reduction(+:rho_integration_error)
            for (size_t i = 0; i < rho.size(); i++) {
                rho_test[i] -= rho_mean;
                const double error = rho[i] - rho_test[i];
                rho_integration_error += error * error;
            }

            rho_l2_norm = std::sqrt(conf.dx * rho_l2_norm);
            //rho_integration_error = std::sqrt(conf.dx * rho_integration_error) / rho_l2_norm;
            rho_integration_error = std::sqrt(conf.dx * rho_integration_error);

            /* if (n % (steps_per_1) == 0) {
            //if (true) {
                std::ofstream rho_str("rho_" + std::to_string(n * conf.dt) + ".txt");
                for (size_t i = 0; i < conf.Nx; i++) {
                    const double x = conf.x_min + i * conf.dx;
                    rho_str << x << " " << rho[i] << " " << rho_test[i] << std::endl;
                }

                std::ofstream j_str("j_" + std::to_string(n * conf.dt) + ".txt");
                for (size_t i = 0; i < conf.Nx; i++) {
                    const double x = conf.x_min + i * conf.dx;
                    j_str << x << " " << jx_hat[i] << " " << jy_hat[i] << std::endl;
                }
            } */
        }

        gle_file << n * conf.dt << " " << gle << " " << rho_integration_error << std::endl;

        // Compute j_hat(n).
        periodic::redux_1x2v::eval_j_time_integral_Hf_exact_1x2v_fourier<order>(
                nt_r_curr,jx_hat_re, jx_hat_im,jy_hat_re, jy_hat_im,
                coeffs_Ex, coeffs_Ey, coeffs_Bz, conf
        );
        double time_eval_j_hat = timer.elapsed();
        std::cout << "Eval j_hat took " << time_eval_j_hat << " s." << std::endl;
        timer.reset();

        double time_for_step = time_compute_EB + time_interpolate_EB + time_eval_j_hat;
        total_time += time_for_step;
        std::cout << "Time step " << n << " took a total of " << time_for_step << " s." << std::endl;

        //if(n % (steps_per_1) == 0){
        if(true){
            if(n % (50*steps_per_1) == 0){
                kinetic_energy_and_entropy_1x2v<double,order>(nt_r_curr,kin_energy_entropy_file,coeffs_Ex,coeffs_Ey,coeffs_Bz,conf, kinetic_energy, entropy,true,n,1,512,512,true,"uv");
                kinetic_energy_and_entropy_1x2v<double,order>(nt_r_curr,kin_energy_entropy_file,coeffs_Ex,coeffs_Ey,coeffs_Bz,conf, kinetic_energy, entropy,true,n,512,512,1,true,"xu");
                kinetic_energy_and_entropy_1x2v<double,order>(nt_r_curr,kin_energy_entropy_file,coeffs_Ex,coeffs_Ey,coeffs_Bz,conf, kinetic_energy, entropy,true,n,512,1,512,true,"xv");
            } else {
                kinetic_energy_and_entropy_1x2v<double,order>(nt_r_curr,kin_energy_entropy_file,coeffs_Ex,coeffs_Ey,coeffs_Bz,conf, kinetic_energy, entropy,true,n,64,64,64);
            }
        }
        do_stats_1x2v<double, order>(nt_r_curr, kinetic_energy, entropy, stat_file, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf, true, n, 128, (n % (50 * steps_per_1) == 0));
        //do_stats_1x2v<double, order>(nt_r_curr, kinetic_energy, entropy, stat_file, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf, true, n, 128, true);

        std::cout << "Do stats took: " << double(timer.elapsed()) << " s." << std::endl;
        std::cout << " ---------------------------------- " << std::endl;

        if(nt_r_curr == nt_restart) {
            timer.reset();
            fe_lr_1x2v::update(fe_lr_1x2v::single_state,velocity_fe,"single",n,conf.Nx,
                conf.x_min,conf.Lx,[&](double x,double u,double v) {
                    return periodic::redux_1x2v::eval_f_nufi_ham_lie_Hf_HB_HE_1x2v<double,order>(
                        nt_r_curr,x,u,v,coeffs_Ex,coeffs_Ey,coeffs_Bz,conf);
                },lr_restart_file);
            for(size_t ix = 0; ix < stride_t; ++ix) {
                coeffs_Ex[ix] = coeffs_Ex[nt_r_curr*stride_t+ix];
                coeffs_Ey[ix] = coeffs_Ey[nt_r_curr*stride_t+ix];
                coeffs_Bz[ix] = coeffs_Bz[nt_r_curr*stride_t+ix];
            }
            conf.f0_1x2v = fe_lr_1x2v::eval_single;
            conf_test.f0_1x2v = fe_lr_1x2v::eval_single;
            nt_r_curr = 1;
            total_time += timer.elapsed();
        } else {
            nt_r_curr++;
        }
    }

    fftw_destroy_plan(plan_fwd);
    fftw_destroy_plan(plan_bwd);
    fftw_free(fft_in);
    fftw_free(fft_out);

    std::cout << "Total simulation time " << total_time << " s." << std::endl;
}

template<size_t order>
void periodically_restarted_nufi_maxwell_strang_exact_fourier_integral_aligned()
{
    // Storage of coefficients now via:
    // index = nt + Nt * (d + dim * (ix + Nx * (iy + Ny * iz)))
    const size_t stride_t = (conf.Nx + order - 1);

    const size_t dim = 1;
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Nspace = Nx_ext;

    std::cout << "Start NuFI-Ham Vlasov-Maxwell-Solver with 1x2v redux." << std::endl;
    std::cout << "Init helper variables." << std::endl;
    arma::arma_rng::set_seed(fe_lr_1x2v::lr_seed);

    // Flattened 1D coefficient storage
    std::vector<double> coeffs_Ex((conf.Nt + 1) * stride_t, 0.0);
    std::vector<double> coeffs_Ey((conf.Nt + 1) * stride_t, 0.0);
    std::vector<double> coeffs_Bz((conf.Nt + 1) * stride_t, 0.0);

    std::vector<double> Ex(conf.Nx, 0.0);
    std::vector<double> Ey(conf.Nx, 0.0);
    std::vector<double> Bz(conf.Nx, 0.0);
    std::vector<double> jx_hat(conf.Nx, 0.0);
    std::vector<double> jy_hat(conf.Nx, 0.0);

    std::vector<double> coeffs_phi(stride_t, 0.0);
    std::vector<double> rho(Nx, 0.0);
    std::vector<double> rho_test(Nx, 0.0);
    std::vector<double> g(Nx, 0.0);
    poisson<double> poiss(conf);

    // FFTW work arrays and plans for field update.
    fftw_complex* fft_in  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * conf.Nx);
    fftw_complex* fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * conf.Nx);
    if (fft_in == nullptr || fft_out == nullptr) {
        if (fft_in  != nullptr) fftw_free(fft_in);
        if (fft_out != nullptr) fftw_free(fft_out);
        throw std::runtime_error("FFTW allocation failed in periodically_restarted_nufi_maxwell_strang_exact_fourier_integral_aligned.");
    }

    fftw_plan plan_fwd = fftw_plan_dft_1d(static_cast<int>(conf.Nx), fft_in, fft_out, FFTW_FORWARD, FFTW_MEASURE);
    fftw_plan plan_bwd = fftw_plan_dft_1d(static_cast<int>(conf.Nx), fft_out, fft_in, FFTW_BACKWARD, FFTW_MEASURE);
    if (plan_fwd == nullptr || plan_bwd == nullptr) {
        if (plan_fwd != nullptr) fftw_destroy_plan(plan_fwd);
        if (plan_bwd != nullptr) fftw_destroy_plan(plan_bwd);
        fftw_free(fft_in);
        fftw_free(fft_out);
        throw std::runtime_error("FFTW plan creation failed in periodically_restarted_nufi_maxwell_strang_exact_fourier_integral_aligned.");
    }

    std::vector<double> Ex_hat_re(conf.Nx, 0.0), Ex_hat_im(conf.Nx, 0.0);
    std::vector<double> Ey_hat_re(conf.Nx, 0.0), Ey_hat_im(conf.Nx, 0.0);
    std::vector<double> Bz_hat_re(conf.Nx, 0.0), Bz_hat_im(conf.Nx, 0.0);
    std::vector<double> jx_hat_re(conf.Nx, 0.0), jx_hat_im(conf.Nx, 0.0);
    std::vector<double> jy_hat_re(conf.Nx, 0.0), jy_hat_im(conf.Nx, 0.0);

    const double two_pi_over_Lx = 2.0 * M_PI / conf.Lx;
    const double invNx = 1.0 / static_cast<double>(conf.Nx);


    // Set up config.
    conf = config_t<double>(Nx, Nu, Nt, dt, 0, Lx, umin, umax, &f0);
    conf.Nv = Nv;
    conf.v_min = vmin;
    conf.v_max = vmax;
    conf.dv = (vmax - vmin) / Nv;
    conf.f0_1x2v = f0_1x2v;
    fe_lr_1x2v::VelocityFE velocity_fe(conf.u_min,conf.u_max,conf.v_min,conf.v_max,
        fe_lr_1x2v::fe_nu,fe_lr_1x2v::fe_nv,fe_lr_1x2v::fe_gauss_order);

    config_t<double> conf_test(Nx, 2 * Nu, Nt, dt, 0, Lx, umin, umax, &f0);
    conf_test.Nv = 2 * Nv;
    conf_test.v_min = vmin;
    conf_test.v_max = vmax;
    conf_test.dv = (vmax - vmin) / conf_test.Nv;
    conf_test.f0_1x2v = f0_1x2v;

    // Compute E(0) and B(0).
    std::cout << "Compute E(0) and B(0)." << std::endl;
    #pragma omp parallel for
    for (size_t l = 0; l < conf.Nx; l++) {
        const double x = conf.x_min + l * conf.dx;

        arma::Col<double> E0_vec = E0(x, 0.0, 0.0);
        arma::Col<double> B0_vec = B0(x, 0.0, 0.0);

        Ex[l] = E0_vec(0);
        Ey[l] = E0_vec(1);
        Bz[l] = B0_vec(2);
    }

    // Interpolate E(0) and B(0).
    std::cout << "Interpolate Ex(0)." << std::endl;
    periodic::interpolate<double, order>(coeffs_Ex.data(), Ex.data(), conf);
    std::cout << "Interpolate Ey(0)." << std::endl;
    periodic::interpolate<double, order>(coeffs_Ey.data(), Ey.data(), conf);
    std::cout << "Interpolate Bz(0)." << std::endl;
    periodic::interpolate<double, order>(coeffs_Bz.data(), Bz.data(), conf);

    std::cout << "First output." << std::endl;
    std::ofstream stat_file("stats.txt");
    std::ofstream lr_restart_file("restart_defects.txt");
    lr_restart_file << "# t species dM dJu dJv dK rank rel_L2_error f2_moment_defect\n";
    std::ofstream kin_energy_entropy_file("kinetic_energy_and_entropy.txt");
    double kinetic_energy = 0.0;
    double entropy = 0.0;
    kinetic_energy_and_entropy_1x2v<double,order>(0,kin_energy_entropy_file,coeffs_Ex,coeffs_Ey,coeffs_Bz,conf, kinetic_energy, entropy,false,0,64,64,64);
    do_stats_1x2v<double, order>(0, kinetic_energy, entropy, stat_file, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf, false, 0, 128, true);

    std::ofstream gle_file("gle.txt");
    double rho_integration_error = 0.0;
    gle_file << 0 << " " << 0 << " " << rho_integration_error << std::endl;

    std::cout << "Restart time-loop." << std::endl;
    std::cout << " ---------------------------------- " << std::endl;
    double total_time = 0.0;
    size_t nt_r_curr = 1;

    for (size_t n = 1; n <= conf.Nt; n++) {
        nufi::stopwatch<double> timer;

        // Compute j_hat(nt_r_curr), associated with the step
        // t_{nt_r_curr-1} -> t_{nt_r_curr}.
        periodic::redux_1x2v::strang_2nd_order::eval_j_time_integral_Hf_exact_1x2v_fourier<order>(
            nt_r_curr, jx_hat_re, jx_hat_im, jy_hat_re, jy_hat_im,
            coeffs_Ex, coeffs_Ey, coeffs_Bz, conf
        );

        // Read fields at time level nt_r_curr-1 on the physical grid.
        #pragma omp parallel for
        for (size_t l = 0; l < conf.Nx; l++) {
            const double x = conf.x_min + l * conf.dx;

            Ex[l] = periodic::eval<double, order>(x, coeffs_Ex.data() + (nt_r_curr - 1) * stride_t, conf);
            Ey[l] = periodic::eval<double, order>(x, coeffs_Ey.data() + (nt_r_curr - 1) * stride_t, conf);
            Bz[l] = periodic::eval<double, order>(x, coeffs_Bz.data() + (nt_r_curr - 1) * stride_t, conf);
        }

        // FFT Ex
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_in[l][0] = Ex[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Ex_hat_re[l] = fft_out[l][0];
            Ex_hat_im[l] = fft_out[l][1];
        }

        // FFT Ey
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_in[l][0] = Ey[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Ey_hat_re[l] = fft_out[l][0];
            Ey_hat_im[l] = fft_out[l][1];
        }

        // FFT Bz
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_in[l][0] = Bz[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Bz_hat_re[l] = fft_out[l][0];
            Bz_hat_im[l] = fft_out[l][1];
        }

        // ------------------------------------------------------------
        // Strang field update:
        //   H_E(dt/2) H_B(dt/2) H_f(dt) H_B(dt/2) H_E(dt/2)
        //
        // 1) H_E(dt/2): Bz^(1) = Bz^(n-1) - dt/2 * (i k) * Ey^(n-1)
        // 2) H_B(dt/2): Ey^(2) = Ey^(n-1) - dt/2 * (i k) * Bz^(1)
        // 3) H_f(dt):   Ex^(3) = Ex^(n-1) - Jx_int
        //                Ey^(3) = Ey^(2)   - Jy_int
        // 4) H_B(dt/2): Ey^(4) = Ey^(3) - dt/2 * (i k) * Bz^(1)
        // 5) H_E(dt/2): Bz^(n) = Bz^(1) - dt/2 * (i k) * Ey^(4)
        //
        // Final fields:
        //   Ex^(n) = Ex^(3)
        //   Ey^(n) = Ey^(4)
        //   Bz^(n) from step 5
        // ------------------------------------------------------------

        std::vector<double> Bz1_hat_re(conf.Nx, 0.0), Bz1_hat_im(conf.Nx, 0.0);
        std::vector<double> Ey2_hat_re(conf.Nx, 0.0), Ey2_hat_im(conf.Nx, 0.0);
        std::vector<double> Ex3_hat_re(conf.Nx, 0.0), Ex3_hat_im(conf.Nx, 0.0);
        std::vector<double> Ey3_hat_re(conf.Nx, 0.0), Ey3_hat_im(conf.Nx, 0.0);
        std::vector<double> Ey4_hat_re(conf.Nx, 0.0), Ey4_hat_im(conf.Nx, 0.0);

        // Step 1: H_E(dt/2)
        for (size_t m = 0; m < conf.Nx; m++) {
            const int kmode = (m <= conf.Nx / 2) ? static_cast<int>(m) : static_cast<int>(m) - static_cast<int>(conf.Nx);
            const double k = two_pi_over_Lx * static_cast<double>(kmode);

            const double ey_re = Ey_hat_re[m];
            const double ey_im = Ey_hat_im[m];

            // i*k*Ey_hat = (-k*ey_im) + i*(k*ey_re)
            Bz1_hat_re[m] = Bz_hat_re[m] - 0.5 * conf.dt * (-k * ey_im);
            Bz1_hat_im[m] = Bz_hat_im[m] - 0.5 * conf.dt * ( k * ey_re);
        }

        // Step 2: H_B(dt/2)
        for (size_t m = 0; m < conf.Nx; m++) {
            const int kmode = (m <= conf.Nx / 2) ? static_cast<int>(m) : static_cast<int>(m) - static_cast<int>(conf.Nx);
            const double k = two_pi_over_Lx * static_cast<double>(kmode);

            const double bz_re = Bz1_hat_re[m];
            const double bz_im = Bz1_hat_im[m];

            // i*k*Bz_hat = (-k*bz_im) + i*(k*bz_re)
            Ey2_hat_re[m] = Ey_hat_re[m] - 0.5 * conf.dt * (-k * bz_im);
            Ey2_hat_im[m] = Ey_hat_im[m] - 0.5 * conf.dt * ( k * bz_re);
        }

        // Step 3: H_f(dt)
        for (size_t m = 0; m < conf.Nx; m++) {
            Ex3_hat_re[m] = Ex_hat_re[m] - jx_hat_re[m];
            Ex3_hat_im[m] = Ex_hat_im[m] - jx_hat_im[m];

            Ey3_hat_re[m] = Ey2_hat_re[m] - jy_hat_re[m];
            Ey3_hat_im[m] = Ey2_hat_im[m] - jy_hat_im[m];
        }

        // Gauge fix: Enforce zero mean of Ex in Fourier space
        Ex3_hat_re[0] = 0.0;
        Ex3_hat_im[0] = 0.0;

        // Step 4: H_B(dt/2)
        for (size_t m = 0; m < conf.Nx; m++) {
            const int kmode = (m <= conf.Nx / 2) ? static_cast<int>(m) : static_cast<int>(m) - static_cast<int>(conf.Nx);
            const double k = two_pi_over_Lx * static_cast<double>(kmode);

            const double bz_re = Bz1_hat_re[m];
            const double bz_im = Bz1_hat_im[m];

            // i*k*Bz_hat = (-k*bz_im) + i*(k*bz_re)
            Ey4_hat_re[m] = Ey3_hat_re[m] - 0.5 * conf.dt * (-k * bz_im);
            Ey4_hat_im[m] = Ey3_hat_im[m] - 0.5 * conf.dt * ( k * bz_re);
        }

        // Gauge fix: Enforce zero mean of Ey in Fourier space
        Ey4_hat_re[0] = 0.0;
        Ey4_hat_im[0] = 0.0;

        // Step 5: H_E(dt/2)
        for (size_t m = 0; m < conf.Nx; m++) {
            const int kmode = (m <= conf.Nx / 2) ? static_cast<int>(m) : static_cast<int>(m) - static_cast<int>(conf.Nx);
            const double k = two_pi_over_Lx * static_cast<double>(kmode);

            const double ey_re = Ey4_hat_re[m];
            const double ey_im = Ey4_hat_im[m];

            // i*k*Ey_hat = (-k*ey_im) + i*(k*ey_re)
            Bz_hat_re[m] = Bz1_hat_re[m] - 0.5 * conf.dt * (-k * ey_im);
            Bz_hat_im[m] = Bz1_hat_im[m] - 0.5 * conf.dt * ( k * ey_re);
        }

        // Final electric field
        Ex_hat_re = Ex3_hat_re;
        Ex_hat_im = Ex3_hat_im;
        Ey_hat_re = Ey4_hat_re;
        Ey_hat_im = Ey4_hat_im;

        if(with_filter){
            spectral_filter_hat_1d(Ey_hat_re, Ey_hat_im);
            spectral_filter_hat_1d(Bz_hat_re, Bz_hat_im);
            spectral_filter_hat_1d(Ex_hat_re, Ex_hat_im);
        }

        // IFFT Ex
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_out[l][0] = Ex_hat_re[l];
            fft_out[l][1] = Ex_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Ex[l] = fft_in[l][0] * invNx;
        }

        // IFFT Ey
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_out[l][0] = Ey_hat_re[l];
            fft_out[l][1] = Ey_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Ey[l] = fft_in[l][0] * invNx;
        }

        // IFFT Bz
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_out[l][0] = Bz_hat_re[l];
            fft_out[l][1] = Bz_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Bz[l] = fft_in[l][0] * invNx;
        }

        double time_compute_EB = timer.elapsed();
        std::cout << "Compute EB took " << time_compute_EB << " s." << std::endl;
        timer.reset();

        // Interpolate E(n) and B(n).
        periodic::interpolate<double, order>(coeffs_Ex.data() + nt_r_curr * stride_t, Ex.data(), conf);
        periodic::interpolate<double, order>(coeffs_Ey.data() + nt_r_curr * stride_t, Ey.data(), conf);
        periodic::interpolate<double, order>(coeffs_Bz.data() + nt_r_curr * stride_t, Bz.data(), conf);

        double time_interpolate_EB = timer.elapsed();
        std::cout << "EB interpolation took " << time_interpolate_EB << " s." << std::endl;
        timer.reset();

        // Gauss clean & integration error test
        periodic::redux_1x2v::strang_2nd_order::eval_rho_ham_strang_HE_HB_Hf_HB_HE_1x2v<double, order>(
            nt_r_curr, rho, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf
        );
        double rho_mean = 0.0;
        #pragma omp parallel for reduction(+:rho_mean)
        for (size_t i = 0; i < rho.size(); i++) {
            rho[i] = 1 + rho[i];
            rho_mean += rho[i];
        }
        rho_mean /= conf.Nx;

        #pragma omp parallel for
        for (size_t i = 0; i < rho.size(); i++) {
            rho[i] -= rho_mean;
        }

        double gle = periodic::maxwell::E_clean_gauss_law_1x2v<double, order>(
            nt_r_curr, coeffs_Ex, Ex, rho, g, coeffs_phi, conf, poiss, !gauss_clean
        );

        if (n % (steps_per_1) == 0) {
            periodic::redux_1x2v::strang_2nd_order::eval_rho_ham_strang_HE_HB_Hf_HB_HE_1x2v<double, order>(
                nt_r_curr, rho_test, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf_test
            );
            rho_integration_error = 0.0;
            double rho_l2_norm = 0.0;
            rho_mean = 0.0;

            #pragma omp parallel for reduction(+:rho_integration_error,rho_l2_norm,rho_mean)
            for (size_t i = 0; i < rho.size(); i++) {
                rho_test[i] = 1 + rho_test[i];
                rho_mean += rho_test[i];
                rho_l2_norm += rho_test[i] * rho_test[i];
            }

            rho_mean /= conf.Nx;

            #pragma omp parallel for reduction(+:rho_integration_error)
            for (size_t i = 0; i < rho.size(); i++) {
                rho_test[i] -= rho_mean;
                const double error = rho[i] - rho_test[i];
                rho_integration_error += error * error;
            }

            rho_l2_norm = std::sqrt(conf.dx * rho_l2_norm);
            //rho_integration_error = std::sqrt(conf.dx * rho_integration_error) / rho_l2_norm;
            rho_integration_error = std::sqrt(conf.dx * rho_integration_error);

            /* if (n % (steps_per_1) == 0) {
                std::ofstream rho_str("rho_" + std::to_string(n * conf.dt) + ".txt");
                for (size_t i = 0; i < conf.Nx; i++) {
                    const double x = conf.x_min + i * conf.dx;
                    rho_str << x << " " << rho[i] << " " << rho_test[i] << std::endl;
                }

                std::ofstream j_str("j_" + std::to_string(n * conf.dt) + ".txt");
                for (size_t i = 0; i < conf.Nx; i++) {
                    const double x = conf.x_min + i * conf.dx;
                    j_str << x << " " << jx_hat[i] << " " << jy_hat[i] << std::endl;
                }
            } */
        }

        gle_file << n * conf.dt << " " << gle << " " << rho_integration_error << std::endl;

        double time_eval_j_hat = timer.elapsed();
        std::cout << "Eval j_hat took " << time_eval_j_hat << " s." << std::endl;
        timer.reset();

        double time_for_step = time_compute_EB + time_interpolate_EB + time_eval_j_hat;
        total_time += time_for_step;
        std::cout << "Time step " << n << " took a total of " << time_for_step << " s." << std::endl;

        kinetic_energy_and_entropy_1x2v<double,order>(nt_r_curr,kin_energy_entropy_file,coeffs_Ex,coeffs_Ey,coeffs_Bz,conf, kinetic_energy, entropy,true,n,64,64,64);
        do_stats_1x2v<double, order>(nt_r_curr, kinetic_energy, entropy, stat_file, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf, true, n, 128, (n % (steps_per_1) == 0));

        std::cout << "Do stats took: " << double(timer.elapsed()) << " s." << std::endl;
        std::cout << " ---------------------------------- " << std::endl;

        if(nt_r_curr == nt_restart) {
            timer.reset();
            fe_lr_1x2v::update(fe_lr_1x2v::single_state,velocity_fe,"single",n,conf.Nx,
                conf.x_min,conf.Lx,[&](double x,double u,double v) {
                    return periodic::redux_1x2v::strang_2nd_order::eval_f_nufi_ham_strang_HE_HB_Hf_HB_HE_1x2v<double,order>(
                        nt_r_curr,x,u,v,coeffs_Ex,coeffs_Ey,coeffs_Bz,conf);
                },lr_restart_file);
            for(size_t ix = 0; ix < stride_t; ++ix) {
                coeffs_Ex[ix] = coeffs_Ex[nt_r_curr*stride_t+ix];
                coeffs_Ey[ix] = coeffs_Ey[nt_r_curr*stride_t+ix];
                coeffs_Bz[ix] = coeffs_Bz[nt_r_curr*stride_t+ix];
            }
            conf.f0_1x2v = fe_lr_1x2v::eval_single;
            conf_test.f0_1x2v = fe_lr_1x2v::eval_single;
            nt_r_curr = 1;
            total_time += timer.elapsed();
        } else {
            nt_r_curr++;
        }
    }

    fftw_destroy_plan(plan_fwd);
    fftw_destroy_plan(plan_bwd);
    fftw_free(fft_in);
    fftw_free(fft_out);

    std::cout << "Total simulation time " << total_time << " s." << std::endl;
}


template<typename real, size_t order>
void kinetic_energy_and_entropy_1x2v_multispecies_with_plotting(
    size_t nt, std::ofstream& stat_file, 
    const std::vector<real>& coeffs_Ex, const std::vector<real>& coeffs_Ey, 
    const std::vector<real>& coeffs_Bz, 
    const config_t<double>& conf_electron, 
    const config_t<double>& conf_ion,
    double& kin_energy_electron, double& entropy_electron, 
    double& kin_energy_ion, double& entropy_ion, 
    bool restarted = false, size_t n_full = 0, 
    size_t nx_plot = 128, 
    size_t nu_e_plot = 128, size_t nv_e_plot = 128, 
    size_t nu_i_plot = 128, size_t nv_i_plot = 128, 
    bool plot_f = false, std::string name_add = "")
{
    double t = nt*dt;
    if(restarted){
        t = n_full*dt;
    }

    double dx_plot = Lx/nx_plot;

    double du_e_plot = (conf_electron.u_max - conf_electron.u_min)/nu_e_plot;
    double dv_e_plot = (conf_electron.v_max - conf_electron.v_min)/nv_e_plot;

    double du_i_plot = (conf_ion.u_max - conf_ion.u_min) / nu_i_plot;
    double dv_i_plot = (conf_ion.v_max - conf_ion.v_min) / nv_i_plot;

    std::vector<double> f_e_values;
    std::vector<double> f_i_values;
    if(plot_f){
        f_e_values.resize(nx_plot*nu_e_plot*nv_e_plot);
        f_i_values.resize(nx_plot*nu_i_plot*nv_i_plot);
    }

    kin_energy_electron = 0;
    entropy_electron = 0;
    double l1_norm_electron = 0;
    double l2_norm_electron = 0;

    #pragma omp parallel for collapse(3) reduction(+:kin_energy_electron,entropy_electron,l1_norm_electron,l2_norm_electron)
    for(size_t ix = 0; ix < nx_plot; ix++)
    for(size_t iu = 0; iu < nu_e_plot; iu++)
    for(size_t iv = 0; iv < nv_e_plot; iv++){
        double x = (ix + 0.5)*dx_plot;
        double u = conf_electron.u_min + (iu + 0.5)*du_e_plot;
        double v = conf_electron.v_min + (iv + 0.5)*dv_e_plot;

        double f = periodic::redux_1x2v::strang_2nd_order::eval_f_nufi_ham_strang_HE_HB_Hf_HB_HE_1x2v<real,order>(nt, x, u, v, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf_electron );

        kin_energy_electron += (u*u + v*v) * f;
        if(f > 1e-16){
            entropy_electron += f * std::log(f);
        }

        l1_norm_electron += std::abs(f);
        l2_norm_electron += f*f;

        if(plot_f){
            f_e_values[ix + nx_plot*(iu + nu_e_plot*iv)] = f;
        }
    }

    double dplot = dx_plot*du_e_plot*dv_e_plot;
    kin_energy_electron *= 0.5*conf_electron.m*dplot;
    entropy_electron *= dplot;
    l1_norm_electron *= dplot;
    l2_norm_electron = std::sqrt(dplot*l2_norm_electron);

    kin_energy_ion = 0;
    entropy_ion = 0;
    double l1_norm_ion = 0;
    double l2_norm_ion = 0;

    #pragma omp parallel for collapse(3) reduction(+:kin_energy_ion,entropy_ion,l1_norm_ion,l2_norm_ion)
    for(size_t ix = 0; ix < nx_plot; ix++)
    for(size_t iu = 0; iu < nu_i_plot; iu++)
    for(size_t iv = 0; iv < nv_i_plot; iv++){
        double x = (ix + 0.5)*dx_plot;
        double u = conf_ion.u_min + (iu + 0.5)*du_i_plot;
        double v = conf_ion.v_min + (iv + 0.5)*dv_i_plot;

        double f = periodic::redux_1x2v::strang_2nd_order::eval_f_nufi_ham_strang_HE_HB_Hf_HB_HE_1x2v<real,order>(nt, x, u, v, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf_ion );

        kin_energy_ion += (u*u + v*v) * f;
        if(f > 1e-16){
            entropy_ion += f * std::log(f);
        }

        l1_norm_ion += std::abs(f);
        l2_norm_ion += f*f;

        if(plot_f){
            f_i_values[ix + nx_plot*(iu + nu_i_plot*iv)] = f;
        }
    }

    dplot = dx_plot*du_i_plot*dv_i_plot;
    kin_energy_ion *= 0.5*conf_ion.m*dplot;
    entropy_ion *= dplot;
    l1_norm_ion *= dplot;
    l2_norm_ion = std::sqrt(dplot*l2_norm_ion);

    stat_file << std::setprecision(15) << t << " " 
            << kin_energy_electron << " " << kin_energy_ion << " " 
            << entropy_electron << " " << entropy_ion << " " 
            << l1_norm_electron << " " << l1_norm_ion << " " 
            << l2_norm_electron << " " << l2_norm_ion << std::endl;

    if(plot_f){
        #pragma omp parallel sections
        {
            #pragma omp section
            {
                std::ofstream f_e_str("f_e_" + std::to_string(t) + name_add + ".txt");

                for (size_t l = 0; l < nx_plot * nu_e_plot * nv_e_plot; ++l) {
                    f_e_str << f_e_values[l] << '\n';
                }
            }

            #pragma omp section
            {
                std::ofstream f_i_str("f_i_" + std::to_string(t) + name_add + ".txt");

                for (size_t l = 0; l < nx_plot * nu_i_plot * nv_i_plot; ++l) {
                    f_i_str << f_i_values[l] << '\n';
                }
            }
        }
    }
}

template<size_t order>
void periodically_restarted_nufi_maxwell_strang_exact_fourier_integral_aligned_multispecies()
{
    // Storage of coefficients now via:
    // index = nt + Nt * (d + dim * (ix + Nx * (iy + Ny * iz)))
    const size_t stride_t = (conf.Nx + order - 1);

    const size_t dim = 1;
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Nspace = Nx_ext;

    std::cout << "Start NuFI-Ham Vlasov-Maxwell-Solver with 1x2v redux." << std::endl;
    std::cout << "Init helper variables." << std::endl;
    arma::arma_rng::set_seed(fe_lr_1x2v::lr_seed);

    // Flattened 1D coefficient storage
    std::vector<double> coeffs_Ex((conf.Nt + 1) * stride_t, 0.0);
    std::vector<double> coeffs_Ey((conf.Nt + 1) * stride_t, 0.0);
    std::vector<double> coeffs_Bz((conf.Nt + 1) * stride_t, 0.0);

    std::vector<double> Ex(conf.Nx, 0.0);
    std::vector<double> Ey(conf.Nx, 0.0);
    std::vector<double> Bz(conf.Nx, 0.0);
    std::vector<double> jx_hat(conf.Nx, 0.0);
    std::vector<double> jy_hat(conf.Nx, 0.0);

    std::vector<double> coeffs_phi(stride_t, 0.0);
    std::vector<double> rho(Nx, 0.0);
    std::vector<double> rho_test(Nx, 0.0);
    std::vector<double> g(Nx, 0.0);
    poisson<double> poiss(conf);

    // FFTW work arrays and plans for field update.
    fftw_complex* fft_in  = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * conf.Nx);
    fftw_complex* fft_out = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * conf.Nx);
    if (fft_in == nullptr || fft_out == nullptr) {
        if (fft_in  != nullptr) fftw_free(fft_in);
        if (fft_out != nullptr) fftw_free(fft_out);
        throw std::runtime_error("FFTW allocation failed in periodically_restarted_nufi_maxwell_strang_exact_fourier_integral_aligned.");
    }

    fftw_plan plan_fwd = fftw_plan_dft_1d(static_cast<int>(conf.Nx), fft_in, fft_out, FFTW_FORWARD, FFTW_MEASURE);
    fftw_plan plan_bwd = fftw_plan_dft_1d(static_cast<int>(conf.Nx), fft_out, fft_in, FFTW_BACKWARD, FFTW_MEASURE);
    if (plan_fwd == nullptr || plan_bwd == nullptr) {
        if (plan_fwd != nullptr) fftw_destroy_plan(plan_fwd);
        if (plan_bwd != nullptr) fftw_destroy_plan(plan_bwd);
        fftw_free(fft_in);
        fftw_free(fft_out);
        throw std::runtime_error("FFTW plan creation failed in periodically_restarted_nufi_maxwell_strang_exact_fourier_integral_aligned.");
    }

    std::vector<double> Ex_hat_re(conf.Nx, 0.0), Ex_hat_im(conf.Nx, 0.0);
    std::vector<double> Ey_hat_re(conf.Nx, 0.0), Ey_hat_im(conf.Nx, 0.0);
    std::vector<double> Bz_hat_re(conf.Nx, 0.0), Bz_hat_im(conf.Nx, 0.0);

    std::vector<double> jx_e_hat_re(conf.Nx, 0.0), jx_e_hat_im(conf.Nx, 0.0);
    std::vector<double> jy_e_hat_re(conf.Nx, 0.0), jy_e_hat_im(conf.Nx, 0.0);

    std::vector<double> jx_i_hat_re(conf.Nx, 0.0), jx_i_hat_im(conf.Nx, 0.0);
    std::vector<double> jy_i_hat_re(conf.Nx, 0.0), jy_i_hat_im(conf.Nx, 0.0);

    std::vector<double> jx_hat_re(conf.Nx, 0.0), jx_hat_im(conf.Nx, 0.0);
    std::vector<double> jy_hat_re(conf.Nx, 0.0), jy_hat_im(conf.Nx, 0.0);

    const double two_pi_over_Lx = 2.0 * M_PI / conf.Lx;
    const double invNx = 1.0 / static_cast<double>(conf.Nx);

    // Init restart matrices.
    std::cout << "Initialize restart matrices." << std::endl;
    fe_lr_1x2v::VelocityFE fe_e(umin_e,umax_e,vmin_e,vmax_e,
        fe_lr_1x2v::fe_nu_e,fe_lr_1x2v::fe_nv_e,fe_lr_1x2v::fe_gauss_order);
    fe_lr_1x2v::VelocityFE fe_i(umin_i,umax_i,vmin_i,vmax_i,
        fe_lr_1x2v::fe_nu_i,fe_lr_1x2v::fe_nv_i,fe_lr_1x2v::fe_gauss_order);

    // Set up config.
    config_t<double> conf_electron(Nx, Nu_e, Nt, dt, 0, Lx, umin_e, umax_e, &f0);
    conf_electron.Nv = Nv_e;
    conf_electron.v_min = vmin_e;
    conf_electron.v_max = vmax_e;
    conf_electron.dv = (vmax_e - vmin_e) / Nv_e;
    conf_electron.m = electron_ion_shock::me;
    conf_electron.q = -1;
    conf_electron.f0_1x2v = f0_electron_1x2v;

    config_t<double> conf_ion(Nx, Nu_i, Nt, dt, 0, Lx, umin_i, umax_i, &f0);
    conf_ion.Nv = Nv_i;
    conf_ion.v_min = vmin_i;
    conf_ion.v_max = vmax_i;
    conf_ion.dv = (vmax_i - vmin_i) / Nv_i;
    conf_ion.m = electron_ion_shock::mi;
    conf_ion.q = 1;
    conf_ion.f0_1x2v = f0_ion_1x2v;

    // Compute E(0) and B(0).
    std::cout << "Compute E(0) and B(0)." << std::endl;
    #pragma omp parallel for
    for (size_t l = 0; l < conf.Nx; l++) {
        const double x = conf.x_min + l * conf.dx;

        arma::Col<double> E0_vec = E0(x, 0.0, 0.0);
        arma::Col<double> B0_vec = B0(x, 0.0, 0.0);

        Ex[l] = E0_vec(0);
        Ey[l] = E0_vec(1);
        Bz[l] = B0_vec(2);
    }

    // Interpolate E(0) and B(0).
    std::cout << "Interpolate Ex(0)." << std::endl;
    periodic::interpolate<double, order>(coeffs_Ex.data(), Ex.data(), conf);
    std::cout << "Interpolate Ey(0)." << std::endl;
    periodic::interpolate<double, order>(coeffs_Ey.data(), Ey.data(), conf);
    std::cout << "Interpolate Bz(0)." << std::endl;
    periodic::interpolate<double, order>(coeffs_Bz.data(), Bz.data(), conf);

    std::cout << "First output." << std::endl;
    std::ofstream stat_file("stats.txt");
    std::ofstream lr_restart_file("restart_defects.txt");
    lr_restart_file << "# t species dM dJu dJv dK rank rel_L2_error f2_moment_defect\n";
    std::ofstream kin_energy_entropy_file("kinetic_energy_and_entropy.txt");
    double kinetic_energy = 0.0;
    double kinetic_energy_electron = 0.0;
    double kinetic_energy_ion = 0.0;
    double entropy = 0.0;
    double entropy_electron = 0.0;
    double entropy_ion = 0.0;
    //kinetic_energy_and_entropy_1x2v<double,order>(0,kin_energy_entropy_file,coeffs_Ex,coeffs_Ey,coeffs_Bz,conf, kinetic_energy, entropy,false,0,64,64,64);
    kinetic_energy_and_entropy_1x2v_multispecies_with_plotting<double,order>(0,kin_energy_entropy_file,coeffs_Ex,coeffs_Ey,coeffs_Bz,conf_electron,conf_ion,kinetic_energy_electron,entropy_electron,kinetic_energy_ion,entropy_ion,false,0,Nx,Nu_e,Nv_e,Nu_i,Nv_i,false);
    kinetic_energy = kinetic_energy_electron + kinetic_energy_ion;
    entropy = entropy_electron + entropy_ion;
    std::cout << "Finished kin energy" << std::endl;
    do_stats_1x2v<double, order>(0, kinetic_energy, entropy, stat_file, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf, false, 0, 128, true);

    std::ofstream gle_file("gle.txt");
    double rho_integration_error = 0.0;
    gle_file << 0 << " " << 0 << " " << rho_integration_error << std::endl;

    std::cout << "Restart time-loop." << std::endl;
    std::cout << " ---------------------------------- " << std::endl;
    double total_time = 0.0;
    size_t nt_r_curr = 1;

    auto eval_f_t_electron = [&](double x, double u, double v) {
        return periodic::redux_1x2v::strang_2nd_order::eval_f_nufi_ham_strang_HE_HB_Hf_HB_HE_1x2v<double,order>(
            nt_r_curr, x, u, v, 
            coeffs_Ex, coeffs_Ey, coeffs_Bz, 
            conf_electron 
        );
    };
    auto eval_f_t_ion = [&](double x, double u, double v) {
        return periodic::redux_1x2v::strang_2nd_order::eval_f_nufi_ham_strang_HE_HB_Hf_HB_HE_1x2v<double,order>(
            nt_r_curr, x, u, v, 
            coeffs_Ex, coeffs_Ey, coeffs_Bz, 
            conf_ion 
        );
    };

    for (size_t n = 1; n <= conf.Nt; n++) {
        nufi::stopwatch<double> timer;

        // Compute j_hat(nt_r_curr), associated with the step
        // t_{nt_r_curr-1} -> t_{nt_r_curr}.
        periodic::redux_1x2v::strang_2nd_order::eval_j_time_integral_Hf_exact_1x2v_fourier<order>(
            nt_r_curr, jx_e_hat_re, jx_e_hat_im, jy_e_hat_re, jy_e_hat_im,
            coeffs_Ex, coeffs_Ey, coeffs_Bz, conf_electron
        );

        periodic::redux_1x2v::strang_2nd_order::eval_j_time_integral_Hf_exact_1x2v_fourier<order>(
            nt_r_curr, jx_i_hat_re, jx_i_hat_im, jy_i_hat_re, jy_i_hat_im,
            coeffs_Ex, coeffs_Ey, coeffs_Bz, conf_ion
        );

        #pragma omp parallel for
        for(size_t l = 0; l < Nx; l++){
            jx_hat_re[l] = jx_e_hat_re[l] + jx_i_hat_re[l];
            jx_hat_im[l] = jx_e_hat_im[l] + jx_i_hat_im[l];

            jy_hat_re[l] = jy_e_hat_re[l] + jy_i_hat_re[l];
            jy_hat_im[l] = jy_e_hat_im[l] + jy_i_hat_im[l];
        }

        double time_eval_j_hat = timer.elapsed();
        std::cout << "Eval j_hat took " << time_eval_j_hat << " s." << std::endl;
        timer.reset();


        // Read fields at time level nt_r_curr-1 on the physical grid.
        #pragma omp parallel for
        for (size_t l = 0; l < conf.Nx; l++) {
            const double x = conf.x_min + l * conf.dx;

            Ex[l] = periodic::eval<double, order>(x, coeffs_Ex.data() + (nt_r_curr - 1) * stride_t, conf);
            Ey[l] = periodic::eval<double, order>(x, coeffs_Ey.data() + (nt_r_curr - 1) * stride_t, conf);
            Bz[l] = periodic::eval<double, order>(x, coeffs_Bz.data() + (nt_r_curr - 1) * stride_t, conf);
        }

        // FFT Ex
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_in[l][0] = Ex[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Ex_hat_re[l] = fft_out[l][0];
            Ex_hat_im[l] = fft_out[l][1];
        }

        // FFT Ey
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_in[l][0] = Ey[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Ey_hat_re[l] = fft_out[l][0];
            Ey_hat_im[l] = fft_out[l][1];
        }

        // FFT Bz
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_in[l][0] = Bz[l];
            fft_in[l][1] = 0.0;
        }
        fftw_execute(plan_fwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Bz_hat_re[l] = fft_out[l][0];
            Bz_hat_im[l] = fft_out[l][1];
        }

        // ------------------------------------------------------------
        // Strang field update:
        //   H_E(dt/2) H_B(dt/2) H_f(dt) H_B(dt/2) H_E(dt/2)
        //
        // 1) H_E(dt/2): Bz^(1) = Bz^(n-1) - dt/2 * (i k) * Ey^(n-1)
        // 2) H_B(dt/2): Ey^(2) = Ey^(n-1) - dt/2 * (i k) * Bz^(1)
        // 3) H_f(dt):   Ex^(3) = Ex^(n-1) - Jx_int
        //                Ey^(3) = Ey^(2)   - Jy_int
        // 4) H_B(dt/2): Ey^(4) = Ey^(3) - dt/2 * (i k) * Bz^(1)
        // 5) H_E(dt/2): Bz^(n) = Bz^(1) - dt/2 * (i k) * Ey^(4)
        //
        // Final fields:
        //   Ex^(n) = Ex^(3)
        //   Ey^(n) = Ey^(4)
        //   Bz^(n) from step 5
        // ------------------------------------------------------------

        std::vector<double> Bz1_hat_re(conf.Nx, 0.0), Bz1_hat_im(conf.Nx, 0.0);
        std::vector<double> Ey2_hat_re(conf.Nx, 0.0), Ey2_hat_im(conf.Nx, 0.0);
        std::vector<double> Ex3_hat_re(conf.Nx, 0.0), Ex3_hat_im(conf.Nx, 0.0);
        std::vector<double> Ey3_hat_re(conf.Nx, 0.0), Ey3_hat_im(conf.Nx, 0.0);
        std::vector<double> Ey4_hat_re(conf.Nx, 0.0), Ey4_hat_im(conf.Nx, 0.0);

        // Step 1: H_E(dt/2)
        for (size_t m = 0; m < conf.Nx; m++) {
            const int kmode = (m <= conf.Nx / 2) ? static_cast<int>(m) : static_cast<int>(m) - static_cast<int>(conf.Nx);
            const double k = two_pi_over_Lx * static_cast<double>(kmode);

            const double ey_re = Ey_hat_re[m];
            const double ey_im = Ey_hat_im[m];

            // i*k*Ey_hat = (-k*ey_im) + i*(k*ey_re)
            Bz1_hat_re[m] = Bz_hat_re[m] - 0.5 * conf.dt * (-k * ey_im);
            Bz1_hat_im[m] = Bz_hat_im[m] - 0.5 * conf.dt * ( k * ey_re);
        }

        // Step 2: H_B(dt/2)
        for (size_t m = 0; m < conf.Nx; m++) {
            const int kmode = (m <= conf.Nx / 2) ? static_cast<int>(m) : static_cast<int>(m) - static_cast<int>(conf.Nx);
            const double k = two_pi_over_Lx * static_cast<double>(kmode);

            const double bz_re = Bz1_hat_re[m];
            const double bz_im = Bz1_hat_im[m];

            // i*k*Bz_hat = (-k*bz_im) + i*(k*bz_re)
            Ey2_hat_re[m] = Ey_hat_re[m] - 0.5 * conf.dt * (-k * bz_im);
            Ey2_hat_im[m] = Ey_hat_im[m] - 0.5 * conf.dt * ( k * bz_re);
        }

        // Step 3: H_f(dt)
        for (size_t m = 0; m < conf.Nx; m++) {
            Ex3_hat_re[m] = Ex_hat_re[m] - jx_hat_re[m];
            Ex3_hat_im[m] = Ex_hat_im[m] - jx_hat_im[m];

            Ey3_hat_re[m] = Ey2_hat_re[m] - jy_hat_re[m];
            Ey3_hat_im[m] = Ey2_hat_im[m] - jy_hat_im[m];
        }

        // Gauge fix: Enforce zero mean of Ex in Fourier space
        Ex3_hat_re[0] = 0.0;
        Ex3_hat_im[0] = 0.0;

        // Step 4: H_B(dt/2)
        for (size_t m = 0; m < conf.Nx; m++) {
            const int kmode = (m <= conf.Nx / 2) ? static_cast<int>(m) : static_cast<int>(m) - static_cast<int>(conf.Nx);
            const double k = two_pi_over_Lx * static_cast<double>(kmode);

            const double bz_re = Bz1_hat_re[m];
            const double bz_im = Bz1_hat_im[m];

            // i*k*Bz_hat = (-k*bz_im) + i*(k*bz_re)
            Ey4_hat_re[m] = Ey3_hat_re[m] - 0.5 * conf.dt * (-k * bz_im);
            Ey4_hat_im[m] = Ey3_hat_im[m] - 0.5 * conf.dt * ( k * bz_re);
        }

        // Gauge fix: Enforce zero mean of Ey in Fourier space
        Ey4_hat_re[0] = 0.0;
        Ey4_hat_im[0] = 0.0;

        // Step 5: H_E(dt/2)
        for (size_t m = 0; m < conf.Nx; m++) {
            const int kmode = (m <= conf.Nx / 2) ? static_cast<int>(m) : static_cast<int>(m) - static_cast<int>(conf.Nx);
            const double k = two_pi_over_Lx * static_cast<double>(kmode);

            const double ey_re = Ey4_hat_re[m];
            const double ey_im = Ey4_hat_im[m];

            // i*k*Ey_hat = (-k*ey_im) + i*(k*ey_re)
            Bz_hat_re[m] = Bz1_hat_re[m] - 0.5 * conf.dt * (-k * ey_im);
            Bz_hat_im[m] = Bz1_hat_im[m] - 0.5 * conf.dt * ( k * ey_re);
        }

        // Final electric field
        Ex_hat_re = Ex3_hat_re;
        Ex_hat_im = Ex3_hat_im;
        Ey_hat_re = Ey4_hat_re;
        Ey_hat_im = Ey4_hat_im;

        if(with_filter){
            spectral_filter_hat_1d(Ey_hat_re, Ey_hat_im);
            spectral_filter_hat_1d(Bz_hat_re, Bz_hat_im);
            spectral_filter_hat_1d(Ex_hat_re, Ex_hat_im);
        }

        // IFFT Ex
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_out[l][0] = Ex_hat_re[l];
            fft_out[l][1] = Ex_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Ex[l] = fft_in[l][0] * invNx;
        }

        // IFFT Ey
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_out[l][0] = Ey_hat_re[l];
            fft_out[l][1] = Ey_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Ey[l] = fft_in[l][0] * invNx;
        }

        // IFFT Bz
        for (size_t l = 0; l < conf.Nx; l++) {
            fft_out[l][0] = Bz_hat_re[l];
            fft_out[l][1] = Bz_hat_im[l];
        }
        fftw_execute(plan_bwd);
        for (size_t l = 0; l < conf.Nx; l++) {
            Bz[l] = fft_in[l][0] * invNx;
        }

        double time_compute_EB = timer.elapsed();
        std::cout << "Compute EB took " << time_compute_EB << " s." << std::endl;
        timer.reset();

        // Interpolate E(n) and B(n).
        periodic::interpolate<double, order>(coeffs_Ex.data() + nt_r_curr * stride_t, Ex.data(), conf);
        periodic::interpolate<double, order>(coeffs_Ey.data() + nt_r_curr * stride_t, Ey.data(), conf);
        periodic::interpolate<double, order>(coeffs_Bz.data() + nt_r_curr * stride_t, Bz.data(), conf);

        double time_interpolate_EB = timer.elapsed();
        std::cout << "EB interpolation took " << time_interpolate_EB << " s." << std::endl;
        timer.reset();

        // No Gauss clean here!

        double time_for_step = time_compute_EB + time_interpolate_EB + time_eval_j_hat;
        total_time += time_for_step;
        std::cout << "Time step " << n << " took a total of " << time_for_step << " s." << std::endl;

        kinetic_energy_and_entropy_1x2v_multispecies_with_plotting<double,order>(nt_r_curr,kin_energy_entropy_file,coeffs_Ex,coeffs_Ey,coeffs_Bz,conf_electron,conf_ion,kinetic_energy_electron,entropy_electron,kinetic_energy_ion,entropy_ion,true,n,Nx,Nu_e,Nv_e,Nu_i,Nv_i,false);
        kinetic_energy = kinetic_energy_electron + kinetic_energy_ion;
        entropy = entropy_electron + entropy_ion;
        do_stats_1x2v<double, order>(nt_r_curr, kinetic_energy, entropy, stat_file, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf, true, n, 128, (n % (steps_per_1) == 0));

        std::cout << "Do stats took: " << double(timer.elapsed()) << " s." << std::endl;
        std::cout << " ---------------------------------- " << std::endl;

        if(nt_r_curr == nt_restart) {
            timer.reset();
            fe_lr_1x2v::update(fe_lr_1x2v::electron_state,fe_e,"electron",n,conf.Nx,
                conf.x_min,conf.Lx,eval_f_t_electron,lr_restart_file);
            fe_lr_1x2v::update(fe_lr_1x2v::ion_state,fe_i,"ion",n,conf.Nx,
                conf.x_min,conf.Lx,eval_f_t_ion,lr_restart_file);
            for(size_t ix = 0; ix < stride_t; ++ix) {
                coeffs_Ex[ix] = coeffs_Ex[nt_r_curr*stride_t+ix];
                coeffs_Ey[ix] = coeffs_Ey[nt_r_curr*stride_t+ix];
                coeffs_Bz[ix] = coeffs_Bz[nt_r_curr*stride_t+ix];
            }
            conf_electron.f0_1x2v = fe_lr_1x2v::eval_electron;
            conf_ion.f0_1x2v = fe_lr_1x2v::eval_ion;
            nt_r_curr = 1;
            total_time += timer.elapsed();
        } else {
            nt_r_curr++;
        }
    }

    fftw_destroy_plan(plan_fwd);
    fftw_destroy_plan(plan_bwd);
    fftw_free(fft_in);
    fftw_free(fft_out);

    std::cout << "Total simulation time " << total_time << " s." << std::endl;
}


}
}

int main(int argc, char** argv)
{
    //nufi::dim1::periodically_restarted_nufi_maxwell_lie_fBE_aligned<4>();

    /* if(nufi::dim1::strang_split){
        nufi::dim1::periodically_restarted_nufi_maxwell_strang_exact_fourier_integral_aligned<4>();
    }else{
        nufi::dim1::periodically_restarted_nufi_maxwell_lie_exact_fourier_integral_aligned<4>();
    } */

    nufi::dim1::periodically_restarted_nufi_maxwell_strang_exact_fourier_integral_aligned_multispecies<4>();

    return 0;
}
