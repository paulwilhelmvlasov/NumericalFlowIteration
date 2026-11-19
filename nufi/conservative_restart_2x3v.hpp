#ifndef NUFI_CONSERVATIVE_RESTART_2X3V_HPP
#define NUFI_CONSERVATIVE_RESTART_2X3V_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <armadillo>

namespace nufi::restart {

// The 1D basis consists of the nodal Catmull-Rom functions, with the same
// clamped velocity indices as cubic_interpolant_2x3v. Four-point GL integrates
// the L2 Gram matrix (degree six) and moments up to quadratic order exactly.
struct cubic_1d_fe_operators {
    arma::mat mass;
    arma::mat moments; // columns: integrals of phi, v*phi, v^2*phi

    cubic_1d_fe_operators(double a, double b, size_t n)
        : mass(n+1,n+1,arma::fill::zeros), moments(n+1,3,arma::fill::zeros)
    {
        if (n<3) throw std::invalid_argument("Cubic restart requires at least three cells per velocity direction.");
        constexpr double nodes[4] = {-0.8611363115940526,-0.3399810435848563,
                                      0.3399810435848563, 0.8611363115940526};
        constexpr double weights[4] = {0.3478548451374539,0.6521451548625461,
                                       0.6521451548625461,0.3478548451374539};
        const double h=(b-a)/n;
        for (size_t j=0; j<n; ++j)
        for (int q=0; q<4; ++q) {
            const double s=0.5*(nodes[q]+1), s2=s*s, s3=s2*s, v=a+(j+s)*h;
            const double phi[4]={-0.5*s+s2-0.5*s3,1-2.5*s2+1.5*s3,
                                  0.5*s+2*s2-1.5*s3,-0.5*s2+0.5*s3};
            const double weight=0.5*h*weights[q];
            std::array<size_t,4> idx;
            for (int k=0; k<4; ++k) idx[k]=static_cast<size_t>(std::clamp(static_cast<int>(j)+k-1,0,static_cast<int>(n)));
            for (int k=0; k<4; ++k) {
                const size_t i=idx[k];
                moments(i,0)+=weight*phi[k];
                moments(i,1)+=weight*phi[k]*v;
                moments(i,2)+=weight*phi[k]*v*v;
                for (int l=0; l<4; ++l) mass(i,idx[l])+=weight*phi[k]*phi[l];
            }
        }
    }
};

// L2-orthogonal constrained correction of nodal cubic interpolation values.
// A: five exact velocity-moment functionals. M: exact tensor-product L2 mass
// matrix of the Catmull-Rom velocity basis. The correction is
//     dc = M^{-1} A^T (A M^{-1} A^T)^{-1} (target - A*c).
// All expensive matrix factorizations are reduced to 1D and performed ONCE.
struct cubic_l2_moment_projection_2x3v {
    std::vector<std::array<double,5>> basis, correction;

    template<class Interpolant>
    explicit cubic_l2_moment_projection_2x3v(const Interpolant& interp)
    {
        const cubic_1d_fe_operators U(interp.umin,interp.umax,interp.nu_r);
        const cubic_1d_fe_operators V(interp.vmin,interp.vmax,interp.nv_r);
        const cubic_1d_fe_operators W(interp.wmin,interp.wmax,interp.nw_r);
        const arma::mat DU=arma::solve(U.mass,U.moments,arma::solve_opts::likely_sympd);
        const arma::mat DV=arma::solve(V.mass,V.moments,arma::solve_opts::likely_sympd);
        const arma::mat DW=arma::solve(W.mass,W.moments,arma::solve_opts::likely_sympd);
        basis.resize(interp.size_v_r);
        correction.resize(interp.size_v_r);
        std::vector<std::array<double,5>> dual(interp.size_v_r);

        for (size_t iw=0; iw<=interp.nw_r; ++iw)
        for (size_t iv=0; iv<=interp.nv_r; ++iv)
        for (size_t iu=0; iu<=interp.nu_r; ++iu) {
            const size_t j=iu+(interp.nu_r+1)*(iv+(interp.nv_r+1)*iw);
            const double u0=U.moments(iu,0),v0=V.moments(iv,0),w0=W.moments(iw,0);
            const double du0=DU(iu,0),dv0=DV(iv,0),dw0=DW(iw,0);
            basis[j]={u0*v0*w0, U.moments(iu,1)*v0*w0, u0*V.moments(iv,1)*w0,
                      u0*v0*W.moments(iw,1),
                      0.5*(U.moments(iu,2)*v0*w0+u0*V.moments(iv,2)*w0+u0*v0*W.moments(iw,2))};
            dual[j]={du0*dv0*dw0, DU(iu,1)*dv0*dw0, du0*DV(iv,1)*dw0,
                     du0*dv0*DW(iw,1),
                     0.5*(DU(iu,2)*dv0*dw0+du0*DV(iv,2)*dw0+du0*dv0*DW(iw,2))};
        }

        arma::Mat<double>::fixed<5,5> gram(arma::fill::zeros);
        for (size_t j=0; j<basis.size(); ++j)
        for (int a=0; a<5; ++a)
        for (int b=0; b<5; ++b) gram(a,b)+=basis[j][a]*dual[j][b];
        gram=0.5*(gram+gram.t());
        arma::Mat<double>::fixed<5,5> inv_gram;
        if (!arma::inv_sympd(inv_gram,gram))
            throw std::runtime_error("Five-moment L2 constraint Gram matrix is singular.");
        for (size_t j=0; j<basis.size(); ++j)
        for (int a=0; a<5; ++a)
            for (int b=0; b<5; ++b) correction[j][a]+=dual[j][b]*inv_gram(b,a);
    }

    template<class Matrix>
    std::array<double,5> evaluate(const Matrix& values, size_t row) const
    {
        std::array<double,5> out{};
        for (size_t j=0; j<basis.size(); ++j) {
            const double c=values(row,j);
            for (int k=0; k<5; ++k) out[k]+=basis[j][k]*c;
        }
        return out;
    }

    template<class Matrix>
    void enforce(Matrix& values, size_t row, const std::array<double,5>& target) const
    {
        const auto initial=evaluate(values,row);
        std::array<double,5> rhs{};
        for (int k=0; k<5; ++k) rhs[k]=target[k]-initial[k];
        for (size_t j=0; j<correction.size(); ++j) {
            double delta=0;
            for (int k=0; k<5; ++k) delta+=correction[j][k]*rhs[k];
            values(row,j)+=delta;
        }
    }
};

// Compensated summation avoids accumulating O(n_xy) rounding error when
// computing the global mass from the periodic cubic spatial interpolant.
inline double compensated_sum(const std::vector<std::array<double,5>>& moments, int k)
{
    double sum=0, compensation=0;
    for (const auto& m : moments) {
        const double v=m[k]-compensation, t=sum+v;
        compensation=(t-sum)-v;
        sum=t;
    }
    return sum;
}

// Sample the evolved distribution BEFORE activating the new restart.
// The reference moment targets use configurable midpoint integration; the
// coefficients reproduce the first target rescaled to the EXACT initial mass,
// and the other four measured targets, in the exact FE moment functionals.
// Requires periodic stencil indices modulo nx_r,ny_r (not nx_r+1,ny_r+1).
template<class Interpolant, class Evaluate>
void conservative_restart_2x3v(Interpolant& interp, const cubic_l2_moment_projection_2x3v& projection,
                               const Evaluate& eval_f, size_t nq_u, size_t nq_v, size_t nq_w,
                               double initial_mass)
{
    if (!nq_u || !nq_v || !nq_w || !(initial_mass>0))
        throw std::invalid_argument("Invalid conservative restart resolution or target mass.");
    const size_t nx=interp.nx_r,ny=interp.ny_r,ns=nx*ny;
    std::vector<std::array<double,5>> targets(ns),outputs(ns);
    const double hu=(interp.umax-interp.umin)/nq_u;
    const double hv=(interp.vmax-interp.vmin)/nq_v;
    const double hw=(interp.wmax-interp.wmin)/nq_w;
    const double dvol=hu*hv*hw;

    // The previous restart_matrix must remain unchanged while eval_f traces
    // through the old initial distribution, including at subsequent restarts.
    #pragma omp parallel for collapse(2)
    for (size_t iy=0; iy<ny; ++iy)
    for (size_t ix=0; ix<nx; ++ix) {
        const size_t row=ix+(nx+1)*iy, i=ix+nx*iy;
        const double x=interp.xmin+ix*interp.dx_r,y=interp.ymin+iy*interp.dy_r;
        std::array<double,5> target{};
        for (size_t iw=0; iw<nq_w; ++iw)
        for (size_t iv=0; iv<nq_v; ++iv)
        for (size_t iu=0; iu<nq_u; ++iu) {
            const double u=interp.umin+(iu+0.5)*hu;
            const double v=interp.vmin+(iv+0.5)*hv;
            const double w=interp.wmin+(iw+0.5)*hw;
            const double f=eval_f(x,y,u,v,w)*dvol;
            target[0]+=f;
            target[1]+=u*f;
            target[2]+=v*f;
            target[3]+=w*f;
            target[4]+=0.5*(u*u+v*v+w*w)*f;
        }
        targets[i]=target;

        for (size_t iw=0; iw<=interp.nw_r; ++iw)
        for (size_t iv=0; iv<=interp.nv_r; ++iv)
        for (size_t iu=0; iu<=interp.nu_r; ++iu) {
            const double u=interp.umin+iu*interp.du_r;
            const double v=interp.vmin+iv*interp.dv_r;
            const double w=interp.wmin+iw*interp.dw_r;
            const size_t j=iu+(interp.nu_r+1)*(iv+(interp.nv_r+1)*iw);
            interp.copy_mat(row,j)=eval_f(x,y,u,v,w);
        }
    }

    // With the corrected periodic Catmull-Rom stencil, the integral of the
    // spatial interpolant is exactly dx*dy times the sum over independent
    // spatial nodes. Preserve the species' analytic initial finite-box mass.
    const double dxy=interp.dx_r*interp.dy_r;
    const double measured_mass=dxy*compensated_sum(targets,0);
    if (!(measured_mass>0)) throw std::runtime_error("Nonpositive measured mass before restart.");
    const double density_scale=initial_mass/measured_mass;

    #pragma omp parallel for collapse(2)
    for (size_t iy=0; iy<ny; ++iy)
    for (size_t ix=0; ix<nx; ++ix) {
        const size_t i=ix+nx*iy,row=ix+(nx+1)*iy;
        targets[i][0]*=density_scale;
        projection.enforce(interp.copy_mat,row,targets[i]);
        outputs[i]=projection.evaluate(interp.copy_mat,row);
    }

    for (size_t iy=0; iy<ny; ++iy)
        interp.copy_mat.row(nx+(nx+1)*iy)=interp.copy_mat.row((nx+1)*iy);
    for (size_t ix=0; ix<=nx; ++ix)
        interp.copy_mat.row(ix+(nx+1)*ny)=interp.copy_mat.row(ix);

    // Nothing in the old interpolant is changed before all measurements finish.
    std::swap(interp.restart_matrix,interp.copy_mat);
    interp.non_negative_enforce=false; // Clipping violates all five constraints.

    double max_relative_residual=0;
    for (size_t i=0; i<ns; ++i)
    for (int k=0; k<5; ++k)
        max_relative_residual=std::max(max_relative_residual,
            std::abs(targets[i][k]-outputs[i][k])/(1+std::abs(targets[i][k])));
    const double actual_mass=dxy*compensated_sum(outputs,0);
    std::cout << std::setprecision(16)
              << "Conservative FE-L2 five-moment restart: mass before=" << measured_mass
              << ", initial=" << initial_mass << ", after=" << actual_mass
              << ", density scale=" << density_scale
              << ", rel. mass error=" << (actual_mass-initial_mass)/initial_mass
              << ", max local moment residual=" << max_relative_residual << '\n';
}

} // namespace nufi::restart

#endif
