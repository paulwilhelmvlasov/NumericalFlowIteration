/*
 * Conservative low-rank restarts for the 1x1v NuFI Vlasov-Poisson solver.
 * MATLAB conservation correction, using the original C++ cubic interpolation.
 *
 * NuFI is distributed under the GNU General Public License, version 3
 * or (at your option) any later version.
 *
 * Requires the existing NuFI headers and Armadillo (with BLAS/LAPACK support).
 */

#include <armadillo>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

 #include <nufi/config.hpp>
#include <nufi/fields.hpp>
#include <nufi/poisson.hpp>
#include <nufi/rho.hpp>
#include <nufi/stopwatch.hpp>



namespace nufi::dim1 {

// Simulation parameters. The restart parameters follow the MATLAB script.
constexpr double pi = 3.14159265358979323846264338327950288;
constexpr double dt = 0.1;
constexpr double T = 50.0;
constexpr size_t Nx = 256;
constexpr size_t Nu = 256;
constexpr size_t Nt = size_t(T/dt + 0.5);
constexpr double x_min = 0.0, x_max = 4.0*pi;
constexpr double u_min = -10.0, u_max = 10.0;
constexpr double alpha = 0.01, wave_number = 0.5;

constexpr size_t restart_every = 20;
constexpr size_t nx_r = Nx, nv_r = Nu; // nx_r periodic spatial nodes, nv_r+1 velocity nodes.
constexpr size_t lr_rank = 16;
constexpr size_t lr_oversampling = 6;
constexpr size_t lr_power_iterations = 1;
constexpr double lr_relative_tol = 0.0;
constexpr size_t lr_block_rows = 16;
constexpr bool lr_matrix_free = false; // Set true to sample remainder rows on demand.
constexpr size_t measurement_every = 50;
constexpr size_t nx_meas = 1024, nv_meas = 1024;

inline double wrap_x(double x, double length) noexcept
{
    x = std::fmod(x, length);
    return x < 0.0 ? x + length : x;
}

// Original C++ Catmull-Rom interpolation and index handling.
inline double cubic_interp(double p0, double p1, double p2, double p3, double t) noexcept
{
    const double a0 = -0.5*p0 + 1.5*p1 - 1.5*p2 + 0.5*p3;
    const double a1 = p0 - 2.5*p1 + 2.0*p2 - 0.5*p3;
    const double a2 = -0.5*p0 + 0.5*p2;
    return ((a0*t + a1)*t + a2)*t + p1;
}

inline size_t clamp_index(int i, size_t n) noexcept
{
    if(i < 0) return 0;
    if(i >= int(n)) return n-1;
    return size_t(i);
}

inline size_t periodic_index(int i, size_t n) noexcept
{
    int j = i % int(n);
    return size_t(j < 0 ? j + int(n) : j);
}

// Exact cell integral of the same Catmull-Rom cubic used by cubic_interp.
inline double cubic_integral(double p0, double p1, double p2, double p3) noexcept
{
    return (-p0 + 13.0*p1 + 13.0*p2 - p3)/24.0;
}

// Exact integral of t*cubic_interp(...,t) over t in [0,1].
inline double cubic_first_integral(double p0, double p1, double p2, double p3) noexcept
{
    return -p0/60.0 + 7.0*p1/40.0 + 11.0*p2/30.0 - p3/40.0;
}

// Integrate the original Catmull-Rom interpolant on the distinct periodic x nodes.
// Unlike the old driver, do not duplicate the endpoint when wrapping indices.
double integrate_cubic_x(const arma::vec& values, double dx)
{
    double sum = 0.0;
    for(size_t i = 0; i < values.n_elem; ++i) {
        const double p0 = values(periodic_index(int(i)-1,values.n_elem));
        const double p1 = values(i);
        const double p2 = values(periodic_index(int(i)+1,values.n_elem));
        const double p3 = values(periodic_index(int(i)+2,values.n_elem));
        sum += cubic_integral(p0,p1,p2,p3);
    }
    return dx*sum;
}

// Integrate the clamped velocity interpolation, including its first moment.
std::pair<double,double> integrate_cubic_v(const arma::vec& values, double vmin, double dv)
{
    double m0 = 0.0, m1 = 0.0;
    for(size_t j = 0; j+1 < values.n_elem; ++j) {
        const double p0 = values(clamp_index(int(j)-1,values.n_elem));
        const double p1 = values(j);
        const double p2 = values(j+1);
        const double p3 = values(clamp_index(int(j)+2,values.n_elem));
        const double q0 = cubic_integral(p0,p1,p2,p3);
        const double q1 = cubic_first_integral(p0,p1,p2,p3);
        m0 += q0;
        m1 += (vmin+j*dv)*q0 + dv*q1;
    }
    return {dv*m0,dv*m1};
}

struct ConservativeRestart {
    arma::mat moment, U_s, V; // Sampled cubic interpolation data (not B-spline coefficients).
    double Lx = 0.0, dx = 0.0, dv = 0.0, vmin = 0.0, vmax = 0.0;
    double c0 = 0.0, W0 = 0.0, W2 = 0.0;
    double delta_M2 = 0.0, delta_J2 = 0.0;
    double M = 0.0, J = 0.0;

    // Original periodic interpolation routine, acting on one moment column.
    double cubic_periodic_vector_interp(size_t k, double x) const noexcept
    {
        x = wrap_x(x,Lx);
        const double gx = x/dx;
        const int ix = int(std::floor(gx));
        const double t = gx-ix;
        double p[4];
        for(int i = -1; i <= 2; ++i)
            p[i+1] = moment(periodic_index(ix+i,moment.n_rows),k);
        return cubic_interp(p[0],p[1],p[2],p[3],t);
    }

    // Original cubic_interpolation_2d: cubic in x, then cubic in velocity.
    double cubic_interpolation_2d(double x, double v) const noexcept
    {
        if(U_s.n_cols == 0) return 0.0;
        x = wrap_x(x,Lx);
        const double gx = x/dx, gv = (v-vmin)/dv;
        const int ix = int(std::floor(gx)), iv = int(std::floor(gv));
        const double tx = gx-ix, tv = gv-iv;
        double val_v[4];
        for(int kv = -1; kv <= 2; ++kv) {
            const size_t j = clamp_index(iv+kv,V.n_rows);
            double val_x[4];
            for(int kx = -1; kx <= 2; ++kx) {
                const size_t i = periodic_index(ix+kx,U_s.n_rows);
                val_x[kx+1] = arma::dot(U_s.row(i),V.row(j));
            }
            val_v[kv+1] = cubic_interp(val_x[0],val_x[1],val_x[2],val_x[3],tx);
        }
        return cubic_interp(val_v[0],val_v[1],val_v[2],val_v[3],tv);
    }

    double eval(double x, double v) const noexcept
    {
        if(v < vmin || v > vmax) return 0.0;
        const double c1 = cubic_periodic_vector_interp(0,x);
        const double c2 = cubic_periodic_vector_interp(1,x);
        const double c3 = cubic_periodic_vector_interp(2,x);
        const double w = std::exp(-0.5*v*v);
        const double f1 = w*(c1+c2*v+c3*(v*v-c0));
        const double f2 = cubic_interpolation_2d(x,v);
        return f1+f2-w*(delta_M2/(Lx*W0)+v*delta_J2/(Lx*W2));
    }
};

ConservativeRestart restart_state;

double f0(double x, double v) noexcept
{
    return (1.0+alpha*std::cos(wave_number*x))*v*v*std::exp(-0.5*v*v)/std::sqrt(2.0*pi);
}

double f_restart(double x, double v) noexcept { return restart_state.eval(x,v); }

// Access the restart remainder in blocks, allowing either cached or on-demand sampling.
using RowEvaluator = std::function<arma::mat(size_t,size_t)>;

arma::mat remainder_product(const RowEvaluator& rows, const arma::mat& X,
                            size_t nx, size_t nv, bool transpose)
{
    arma::mat Y(transpose ? nv : nx,X.n_cols,arma::fill::zeros);
    for(size_t i = 0; i < nx; i += lr_block_rows) {
        const size_t last = std::min(nx,i+lr_block_rows);
        arma::mat R = rows(i,last);
        if(transpose) Y += R.t()*X.rows(i,last-1);
        else Y.rows(i,last-1) = R*X;
    }
    return Y;
}

void randomized_svd_rows(const RowEvaluator& rows, size_t nx, size_t nv,
                         arma::mat& U, arma::vec& s, arma::mat& V)
{
    const size_t r = std::min(lr_rank,std::min(nx,nv));
    if(r == 0) { U.set_size(nx,0); V.set_size(nv,0); s.reset(); return; }
    const size_t ell = std::min(std::min(nx,nv),r+lr_oversampling);
    arma::mat Omega = arma::randn<arma::mat>(nv,ell);
    arma::mat Q,R;
    arma::mat Y = remainder_product(rows,Omega,nx,nv,false);
    arma::qr_econ(Q,R,Y);
    for(size_t it = 0; it < lr_power_iterations; ++it) {
        arma::mat Z = remainder_product(rows,Q,nx,nv,true);
        arma::mat P;
        arma::qr_econ(P,R,Z);
        Y = remainder_product(rows,P,nx,nv,false);
        arma::qr_econ(Q,R,Y);
    }
    arma::mat B = remainder_product(rows,Q,nx,nv,true).t();
    arma::mat Us, Vfull;
    arma::vec sfull;
    if(!arma::svd_econ(Us,sfull,Vfull,B)) throw std::runtime_error("RSVD small SVD failed");
    size_t nkeep = std::min(r,size_t(sfull.n_elem));
    if(lr_relative_tol > 0.0 && sfull.n_elem && sfull(0) > 0.0)
        nkeep = std::min(nkeep,size_t(arma::accu(sfull > lr_relative_tol*sfull(0))));
    if(nkeep == 0) { U.set_size(nx,0); V.set_size(nv,0); s.reset(); return; }
    U = Q*Us.cols(0,nkeep-1);
    V = Vfull.cols(0,nkeep-1);
    s = sfull.head(nkeep);
}

template <size_t order>
ConservativeRestart build_restart(size_t nt, double* coeffs, config_t<double>& conf)
{
    const double length = conf.Lx;
    if(std::abs(conf.u_min+conf.u_max) > 1e-12)
        throw std::invalid_argument("The analytic Gaussian moments require symmetric velocity limits");
    const double vmax = conf.u_max;
    const double dx = length/nx_r, dv = (conf.u_max-conf.u_min)/nv_r;
    arma::vec v(nv_r+1);
    for(size_t j = 0; j <= nv_r; ++j) v(j) = conf.u_min+j*dv;

    // Original cubic interpolation, with distinct periodic x nodes and inclusive velocity endpoints.
    arma::mat F;
    if(!lr_matrix_free) {
        F.set_size(nx_r,nv_r+1);
        #pragma omp parallel for
        for(size_t i = 0; i < nx_r; ++i)
            for(size_t j = 0; j <= nv_r; ++j)
                F(i,j) = periodic::eval_f<double,order>(nt,conf.x_min+i*dx,v(j),coeffs,conf);
    }
    auto sample_rows = [&](size_t first, size_t last) -> arma::mat {
        if(!lr_matrix_free) return F.rows(first,last-1);
        arma::mat out(last-first,nv_r+1);
        #pragma omp parallel for
        for(size_t i = first; i < last; ++i)
            for(size_t j = 0; j <= nv_r; ++j)
                out(i-first,j) = periodic::eval_f<double,order>(nt,conf.x_min+i*dx,v(j),coeffs,conf);
        return out;
    };

    arma::vec rho(nx_r,arma::fill::zeros), J(nx_r,arma::fill::zeros);
    arma::vec kappa(nx_r,arma::fill::zeros);
    for(size_t first = 0; first < nx_r; first += lr_block_rows) {
        const size_t last = std::min(nx_r,first+lr_block_rows);
        arma::mat values = sample_rows(first,last);
        for(size_t i = first; i < last; ++i)
            for(size_t j = 0; j <= nv_r; ++j) {
                const double f = values(i-first,j)*dv;
                rho(i) += f;
                J(i) += v(j)*f;
                kappa(i) += 0.5*v(j)*v(j)*f;
            }
    }

    // Correct the continuous integrals of the interpolated moment functions.
    const double M0 = length, J0 = 0.0;
    rho += (M0-integrate_cubic_x(rho,dx))/length;
    J += (J0-integrate_cubic_x(J,dx))/length;

    const double gauss = std::exp(-0.5*vmax*vmax);
    const double W0 = std::sqrt(2.0*pi)*std::erf(vmax/std::sqrt(2.0));
    const double W2 = W0-2.0*vmax*gauss;
    const double W4 = 3.0*W2-2.0*vmax*vmax*vmax*gauss;
    const double c0 = W2/W0;
    arma::mat moment(nx_r,3);
    moment.col(0) = rho/W0;
    moment.col(1) = J/W2;
    moment.col(2) = (2.0*kappa-c0*rho)/(W4-c0*W2);

    const double M1 = W0*integrate_cubic_x(moment.col(0),dx);
    const double J1 = W2*integrate_cubic_x(moment.col(1),dx);
    arma::rowvec w0(nv_r+1), w1(nv_r+1), w2(nv_r+1);
    for(size_t j = 0; j <= nv_r; ++j) {
        const double u = v(j), w = std::exp(-0.5*u*u);
        w0(j) = w;
        w1(j) = u*w;
        w2(j) = (u*u-c0)*w;
    }
    RowEvaluator remainder_rows = [&](size_t first, size_t last) -> arma::mat {
        arma::mat out = sample_rows(first,last);
        arma::mat a = moment.rows(first,last-1);
        out -= a.col(0)*w0+a.col(1)*w1+a.col(2)*w2;
        return out;
    };

    arma::mat U,V;
    arma::vec s;
    randomized_svd_rows(remainder_rows,nx_r,nv_r+1,U,s,V);
    ConservativeRestart state;
    state.Lx = length;
    state.dx = dx;
    state.dv = dv;
    state.vmin = conf.u_min;
    state.vmax = conf.u_max;
    state.c0 = c0;
    state.W0 = W0;
    state.W2 = W2;
    state.moment = std::move(moment);
    state.U_s = U;
    state.U_s.each_row() %= s.t();
    state.V = std::move(V);

    // Exact continuous moments of the SAME cubic interpolants used for f2.
    for(arma::uword k = 0; k < s.n_elem; ++k) {
        const double ix = integrate_cubic_x(state.U_s.col(k),dx);
        const auto [iv0,iv1] = integrate_cubic_v(state.V.col(k),conf.u_min,dv);
        state.delta_M2 += ix*iv0;
        state.delta_J2 += ix*iv1;
    }
    // One untruncated Gaussian correction cancels the LR remainder's moments.
    state.M = M1 + state.delta_M2 - state.delta_M2/(length*W0)*length*W0;
    state.J = J1 + state.delta_J2 - state.delta_J2/(length*W2)*length*W2;
    std::cout << std::setprecision(12) << "restart: rank=" << s.n_elem
              << " M2=" << state.delta_M2 << " J2=" << state.delta_J2
              << " M-M0=" << state.M-M0 << " J-J0=" << state.J-J0 << '\n';
    return state;
}

template <size_t order>
void apply_restart(size_t& nt, size_t n, double* coeffs, config_t<double>& conf, std::ostream& log)
{
    nufi::stopwatch<double> timer;
    const size_t stride = conf.Nx+order-1;
    ConservativeRestart next = build_restart<order>(nt,coeffs,conf);
    restart_state = std::move(next);
    log << std::setprecision(16) << n*dt << ' ' << restart_state.U_s.n_cols
        << ' ' << restart_state.delta_M2 << ' ' << restart_state.delta_J2
        << ' ' << restart_state.M-conf.Lx << ' ' << restart_state.J << '\n';
    for(size_t i = 0; i < stride; ++i) coeffs[i] = coeffs[nt*stride+i];
    conf = config_t<double>(Nx,Nu,Nt,dt,x_min,x_max,u_min,u_max,&f_restart);
    nt = 1;
    std::cout << "t=" << n*dt << " restart time=" << timer.elapsed() << " s\n";
}

struct MeasuredMoments {
    double M = 0.0, J = 0.0, kinetic = 0.0;
};

template <size_t order>
MeasuredMoments measure(size_t nt, double* coeffs, config_t<double>& conf)
{
    const double dx = conf.Lx/nx_meas;
    const double dv = (conf.u_max-conf.u_min)/(nv_meas-1);
    double M = 0.0, J = 0.0, kinetic = 0.0;
    // Trapezoidal velocity measurement; independent of restart moment quadrature.
    #pragma omp parallel for reduction(+:M,J,kinetic)
    for(size_t i = 0; i < nx_meas; ++i) {
        const double x = conf.x_min+i*dx;
        for(size_t j = 0; j < nv_meas; ++j) {
            const double v = conf.u_min+j*dv;
            const double f = periodic::eval_f<double,order>(nt,x,v,coeffs,conf);
            const double weight = dx*dv*(j == 0 || j == nv_meas-1 ? 0.5 : 1.0);
            M += weight*f;
            J += weight*v*f;
            kinetic += 0.5*weight*v*v*f;
        }
    }
    return {M,J,kinetic};
}

template <size_t order>
void run()
{
    static_assert(nx_r >= 3 && nv_r >= 3,"Cubic interpolation requires at least four nodes");
    static_assert(Nx % 2 == 0,"NuFI periodic grid must have even Nx");
    static_assert(restart_every > 0,"restart_every must be positive");
    static_assert(measurement_every > 0,"measurement_every must be positive");
    arma::arma_rng::set_seed(12345);
    config_t<double> conf(Nx,Nu,Nt,dt,x_min,x_max,u_min,u_max,&f0);
    const size_t stride = conf.Nx+order-1;
    std::unique_ptr<double[]> coeffs(new double[(restart_every+1)*stride]{});
    // Keep the alignment requirement used by the original NuFI driver.
    std::unique_ptr<double,decltype(std::free)*> rho(
        static_cast<double*>(std::aligned_alloc(64,((sizeof(double)*Nx+63)/64)*64)),std::free);
    if(!rho) throw std::bad_alloc();
    poisson<double> poiss(conf);
    std::ofstream stats("stats.txt"), measured("conservation.txt"), restarts("restart_defects.txt");
    stats << "# t Emax electric_energy\n";
    measured << "# t M J kinetic total_energy relative_M_error absolute_J_error relative_H_error\n";
    restarts << "# t rank remainder_M remainder_J analytic_M_defect analytic_J_defect\n";
    size_t nt = 0;
    double H0 = std::numeric_limits<double>::quiet_NaN();

    for(size_t n = 0; n <= Nt; ++n) {
        nufi::stopwatch<double> timer;
        #pragma omp parallel for
        for(size_t i = 0; i < conf.Nx; ++i)
            rho.get()[i] = periodic::eval_rho<double,order>(nt,i,coeffs.get(),conf);
        const double electric_energy = poiss.solve(rho.get());
        periodic::interpolate<double,order>(coeffs.get()+nt*stride,rho.get(),conf);

        double Emax = 0.0;
        for(size_t i = 0; i < 512; ++i) {
            const double x = conf.x_min+i*conf.Lx/512;
            // The Poisson buffer contains phi, so E = -d_x phi.
            const double E = -periodic::eval<double,order,1>(x,coeffs.get()+nt*stride,conf);
            Emax = std::max(Emax,std::abs(E));
        }
        stats << std::setprecision(16) << n*dt << ' ' << Emax << ' ' << electric_energy << '\n';

        const bool is_restart = n > 0 && n%restart_every == 0;
        size_t eval_nt = nt;
        if(is_restart) {
            apply_restart<order>(nt,n,coeffs.get(),conf,restarts);
            eval_nt = 0; // Evaluate the new base at its restart time.
        }

        if(n%measurement_every == 0 || n == Nt || is_restart) {
            MeasuredMoments m = measure<order>(eval_nt,coeffs.get(),conf);
            const double H = m.kinetic+electric_energy;
            if(n == 0) H0 = H;
            const double errH = std::abs(H-H0)/std::abs(H0);
            measured << std::setprecision(16) << n*dt << ' ' << m.M << ' ' << m.J
                     << ' ' << m.kinetic << ' ' << H << ' '
                     << std::abs(m.M-conf.Lx)/conf.Lx << ' ' << std::abs(m.J)
                     << ' ' << errH << '\n';
            std::cout << "t=" << n*dt << " measured dM/M=" << std::abs(m.M-conf.Lx)/conf.Lx
                      << " |J|=" << std::abs(m.J) << " dH/H=" << errH << '\n';
        }
        if(!is_restart) ++nt;
        std::cout << "step " << n << "/" << Nt << ": " << timer.elapsed() << " s\n";
    }
    std::cout << "Excluded initial mass tail: "
              << (x_max-x_min)*(std::erfc(u_max/std::sqrt(2.0))
              +std::sqrt(2.0/pi)*u_max*std::exp(-0.5*u_max*u_max)) << '\n';
}

} // namespace nufi::dim1

int main()
{
    nufi::dim1::run<4>();
}
