#ifndef NUFI_BSPLINE_GALERKIN_RESTART_2X3V_HPP
#define NUFI_BSPLINE_GALERKIN_RESTART_2X3V_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <vector>
#include <utility>

#include <armadillo>

namespace nufi::restart {

struct bspline_quad_point {
    double x, weight;
    std::array<size_t,4> index;
    std::array<double,4> basis;
};

struct cubic_bspline_axis {
    double a, b, h;
    size_t cells, n;
    bool periodic;
    std::vector<double> knots;
    arma::mat mass_inverse, collocation_inverse;
    std::array<arma::mat,5> discrete_mass_inverse;
    std::vector<std::array<double,3>> moments;

    cubic_bspline_axis(double a_, double b_, size_t cells_, bool periodic_)
        : a(a_), b(b_), h((b_-a_)/cells_), cells(cells_), n(cells_+(periodic_?0:3)), periodic(periodic_)
    {
        if (!(b>a) || cells<4) throw std::invalid_argument("Cubic B-splines require at least four cells and a positive interval.");
        if (!periodic) {
            knots.resize(n+4);
            for (size_t i=0; i<n+4; ++i)
                knots[i]=i<=3 ? 0.0 : (i>=n ? double(cells) : double(i-3));
        }
        const auto q=quadrature(4);
        arma::mat M(n,n,arma::fill::zeros);
        moments.resize(n);
        for (const auto& pt : q)
        for (int j=0; j<4; ++j) {
            const size_t i=pt.index[j];
            const double v=pt.weight*pt.basis[j];
            moments[i][0]+=v;
            moments[i][1]+=v*pt.x;
            moments[i][2]+=v*pt.x*pt.x;
            for (int k=0; k<4; ++k) M(i,pt.index[k])+=v*pt.basis[k];
        }
        if (!arma::inv_sympd(mass_inverse,M)) throw std::runtime_error("Cubic B-spline mass matrix is not SPD.");
        discrete_mass_inverse[4]=mass_inverse;
        // Matching quadrature on both sides makes the discrete Galerkin
        // projection idempotent even with fewer than four Gauss points.
        for (int nq=2; nq<=3; ++nq) {
            arma::mat Mq(n,n,arma::fill::zeros);
            for (const auto& pt : quadrature(nq))
                for (int i=0; i<4; ++i)
                    for (int j=0; j<4; ++j)
                        Mq(pt.index[i],pt.index[j])+=pt.weight*pt.basis[i]*pt.basis[j];
            if (!arma::inv_sympd(discrete_mass_inverse[nq],Mq))
                throw std::runtime_error("Under-integrated B-spline Gram matrix is singular.");
        }
        if (periodic) {
            arma::mat C(n,n,arma::fill::zeros);
            for (size_t i=0; i<n; ++i) {
                const auto pt=basis_at(a+i*h);
                for (int k=0; k<4; ++k) C(i,pt.index[k])+=pt.basis[k];
            }
            if (!arma::inv(collocation_inverse,C)) throw std::runtime_error("Periodic cubic B-spline collocation matrix is singular.");
        }
    }

    static std::pair<std::array<double,4>,std::array<double,4>> gauss(int nq)
    {
        if (nq==1) return {{{0,0,0,0}},{{2,0,0,0}}};
        if (nq==2) return {{{-0.5773502691896257645,0.5773502691896257645,0,0}},{{1,1,0,0}}};
        if (nq==3) return {{{-0.774596669241483377,0,0.774596669241483377,0}},{{0.555555555555555556,0.888888888888888889,0.555555555555555556,0}}};
        if (nq==4) return {{{-0.861136311594052575,-0.339981043584856265,0.339981043584856265,0.861136311594052575}},
                             {{0.347854845137453857,0.652145154862546143,0.652145154862546143,0.347854845137453857}}};
        throw std::invalid_argument("Supported Gauss orders are 1, 2, 3 and 4.");
    }

    bspline_quad_point basis_at(double x) const
    {
        if (periodic) {
            double t=std::fmod(x-a,b-a);
            if (t<0) t+=b-a;
            const size_t cell=std::min(cells-1,size_t(t/h));
            const double s=t/h-cell, s2=s*s, s3=s2*s;
            bspline_quad_point pt{x,0,{(cell+cells-1)%cells,cell,(cell+1)%cells,(cell+2)%cells},
                {(1-3*s+3*s2-s3)/6,(4-6*s2+3*s3)/6,(1+3*s+3*s2-3*s3)/6,s3/6}};
            return pt;
        }
        if (x<a || x>b) throw std::out_of_range("Velocity outside B-spline interval.");
        const double t=(x-a)/h;
        const size_t cell=std::min(cells-1,size_t(t));
        const size_t span=cell+3;
        std::array<double,4> B{1,0,0,0},left{},right{};
        for (size_t degree=1; degree<=3; ++degree) {
            left[degree]=t-knots[span+1-degree];
            right[degree]=knots[span+degree]-t;
            double saved=0;
            for (size_t j=0; j<degree; ++j) {
                const double denom=right[j+1]+left[degree-j];
                const double v=denom!=0 ? B[j]/denom : 0;
                B[j]=saved+right[j+1]*v;
                saved=left[degree-j]*v;
            }
            B[degree]=saved;
        }
        return {x,0,{cell,cell+1,cell+2,cell+3},B};
    }

    std::vector<bspline_quad_point> quadrature(int nq) const
    {
        const auto [nodes,weights]=gauss(nq);
        std::vector<bspline_quad_point> pts;
        pts.reserve(cells*nq);
        for (size_t cell=0; cell<cells; ++cell)
        for (int k=0; k<nq; ++k) {
            const double x=a+(cell+(nodes[k]+1)*0.5)*h;
            auto pt=basis_at(x);
            pt.weight=0.5*h*weights[k];
            pts.push_back(pt);
        }
        return pts;
    }
};

// Full tensor-product periodic x,y and open-knot cubic B-spline u,v,w space.
// spatial_gauss=2..4 gives a tensor-product 5D Galerkin projection using composite
// quadrature for its mass matrix AND right-hand side. With four Gauss
// points these matrices are the exact continuous L2 mass matrices; two or
// three points define an idempotent quadrature-discrete Galerkin projection.
// spatial_gauss=0 uses the less expensive
// velocity-L2 Galerkin projection at spatial nodes, followed by periodic spline
// interpolation in x,y (as in the user's NuFI-LR manuscript).
class bspline_galerkin_2x3v {
public:
    double xmin,xmax,ymin,ymax,umin,umax,vmin,vmax,wmin,wmax;
    const cubic_bspline_axis X,Y,U,V,W;
    size_t nx_r,ny_r,nu_r,nv_r,nw_r,size_x_r,size_v_r;
    arma::mat restart_matrix, copy_mat;

    bspline_galerkin_2x3v(double xmin_,double xmax_,double ymin_,double ymax_,
                          double umin_,double umax_,double vmin_,double vmax_,double wmin_,double wmax_,
                          size_t nx,size_t ny,size_t nu,size_t nv,size_t nw)
        : xmin(xmin_),xmax(xmax_),ymin(ymin_),ymax(ymax_),umin(umin_),umax(umax_),
          vmin(vmin_),vmax(vmax_),wmin(wmin_),wmax(wmax_),
          X(xmin,xmax,nx,true),Y(ymin,ymax,ny,true),U(umin,umax,nu,false),
          V(vmin,vmax,nv,false),W(wmin,wmax,nw,false),
          nx_r(nx),ny_r(ny),nu_r(nu),nv_r(nv),nw_r(nw),size_x_r(X.n*Y.n),size_v_r(U.n*V.n*W.n),
          restart_matrix(size_x_r,size_v_r,arma::fill::zeros),copy_mat(size_x_r,size_v_r,arma::fill::zeros) {}

    double cubic_interpolation_5d_streaming(double x,double y,double u,double v,double w) const
    {
        if (u<umin || u>umax || v<vmin || v>vmax || w<wmin || w>wmax) return 0;
        const auto bx=X.basis_at(x), by=Y.basis_at(y), bu=U.basis_at(u), bv=V.basis_at(v), bw=W.basis_at(w);
        double result=0;
        for (int k=0; k<4; ++k)
        for (int j=0; j<4; ++j) {
            const double wyz=by.basis[j]*bw.basis[k];
            for (int m=0; m<4; ++m)
            for (int n=0; n<4; ++n) {
                const double vweight=wyz*bu.basis[m]*bv.basis[n];
                const size_t col=bu.index[m]+U.n*(bv.index[n]+V.n*bw.index[k]);
                for (int i=0; i<4; ++i)
                    result+=bx.basis[i]*vweight*restart_matrix(bx.index[i]+X.n*by.index[j],col);
            }
        }
        return result;
    }

    // Exact FE moments of the stored 5D B-spline representation; no quadrature.
    std::array<double,5> global_moments(const arma::mat& coeff) const
    {
        std::array<double,5> out{};
        const double xyweight=X.h*Y.h;
        for (size_t iw=0; iw<W.n; ++iw)
        for (size_t iv=0; iv<V.n; ++iv)
        for (size_t iu=0; iu<U.n; ++iu) {
            const size_t col=iu+U.n*(iv+V.n*iw);
            const double u0=U.moments[iu][0],v0=V.moments[iv][0],w0=W.moments[iw][0];
            const double factors[5]={u0*v0*w0,U.moments[iu][1]*v0*w0,u0*V.moments[iv][1]*w0,
                                     u0*v0*W.moments[iw][1],0.5*(U.moments[iu][2]*v0*w0+
                                     u0*V.moments[iv][2]*w0+u0*v0*W.moments[iw][2])};
            // Sum spatial degrees of freedom; integration of each periodic spline = dx*dy.
            double cs=0;
            for (size_t row=0; row<size_x_r; ++row) cs+=coeff(row,col);
            for (int k=0; k<5; ++k) out[k]+=xyweight*cs*factors[k];
        }
        return out;
    }

    template<class Evaluate>
    void project(const Evaluate& eval_f, double initial_mass, int spatial_gauss=2, int velocity_gauss=2)
    {
        if (!(initial_mass>0) || spatial_gauss<0 || spatial_gauss>4 || spatial_gauss==1 || velocity_gauss<2 || velocity_gauss>4)
            throw std::invalid_argument("Invalid spline Galerkin parameters: use spatial Gauss 0 or 2-4, velocity Gauss 2-4.");
        const auto qu=U.quadrature(velocity_gauss),qv=V.quadrature(velocity_gauss),qw=W.quadrature(velocity_gauss);
        arma::mat rhs(size_x_r,size_v_r,arma::fill::zeros);
        const arma::mat& sxinv=spatial_gauss ? X.discrete_mass_inverse[spatial_gauss] : X.collocation_inverse;
        const arma::mat& syinv=spatial_gauss ? Y.discrete_mass_inverse[spatial_gauss] : Y.collocation_inverse;
        std::array<double,5> measured{};

        // Sum-factorised composite Gauss quadrature for the velocity Galerkin RHS.
        // All calls to eval_f happen before restart_matrix is overwritten.
        auto velocity_rhs=[&](double x,double y,std::vector<double>& out,std::array<double,5>& mom) {
            const size_t nqu=qu.size(),nqv=qv.size(),nqw=qw.size();
            std::vector<double> rhs_u(U.n*nqv*nqw,0.0);
            std::vector<double> rhs_uv(U.n*V.n*nqw,0.0);
            out.assign(size_v_r,0.0);
            mom={};
            for (size_t kw=0; kw<nqw; ++kw)
            for (size_t kv=0; kv<nqv; ++kv)
            for (size_t ku=0; ku<nqu; ++ku) {
                const auto& pu=qu[ku]; const auto& pv=qv[kv]; const auto& pw=qw[kw];
                const double f=eval_f(x,y,pu.x,pv.x,pw.x);
                const double tmp=f*pu.weight;
                const double dvw=pv.weight*pw.weight;
                const double d=f*pu.weight*dvw;
                mom[0]+=d;
                mom[1]+=pu.x*d;
                mom[2]+=pv.x*d;
                mom[3]+=pw.x*d;
                mom[4]+=0.5*(pu.x*pu.x+pv.x*pv.x+pw.x*pw.x)*d;
                for (int i=0; i<4; ++i) rhs_u[pu.index[i]+U.n*(kv+nqv*kw)]+=tmp*pu.basis[i];
            }
            for (size_t kw=0; kw<nqw; ++kw)
            for (size_t kv=0; kv<nqv; ++kv) {
                const auto& pv=qv[kv];
                for (size_t iu=0; iu<U.n; ++iu) {
                    const double tmp=rhs_u[iu+U.n*(kv+nqv*kw)]*pv.weight;
                    for (int j=0; j<4; ++j) rhs_uv[iu+U.n*(pv.index[j]+V.n*kw)]+=tmp*pv.basis[j];
                }
            }
            for (size_t kw=0; kw<nqw; ++kw) {
                const auto& pw=qw[kw];
                for (size_t iv=0; iv<V.n; ++iv)
                for (size_t iu=0; iu<U.n; ++iu) {
                    const double tmp=rhs_uv[iu+U.n*(iv+V.n*kw)]*pw.weight;
                    for (int k=0; k<4; ++k)
                        out[iu+U.n*(iv+V.n*pw.index[k])]+=tmp*pw.basis[k];
                }
            }
        };

        if (spatial_gauss==0) {
            // Much cheaper: Galerkin in velocity, interpolation in x,y.
            #pragma omp parallel for
            for (size_t row=0; row<size_x_r; ++row) {
                const size_t ix=row%X.n,iy=row/X.n;
                std::vector<double> values;
                std::array<double,5> target{};
                velocity_rhs(xmin+ix*X.h,ymin+iy*Y.h,values,target);
                for (size_t j=0; j<size_v_r; ++j) rhs(row,j)=values[j];
                #pragma omp critical(bspline_moments)
                for (int k=0; k<5; ++k) measured[k]+=target[k]*X.h*Y.h;
            }
        } else {
            // Genuine tensor-product 5D L2-Galerkin projection in phase space.
            // Spatial element contributions are assembled with row-level locks.
            const auto qx=X.quadrature(spatial_gauss),qy=Y.quadrature(spatial_gauss);
            std::vector<std::mutex> locks(size_x_r);
            #pragma omp parallel for collapse(2) schedule(dynamic)
            for (size_t cy=0; cy<Y.cells; ++cy)
            for (size_t cx=0; cx<X.cells; ++cx) {
                std::vector<double> local(16*size_v_r,0.0),values;
                std::array<double,5> local_mom{};
                std::array<size_t,16> rows{};
                for (int j=0; j<4; ++j)
                for (int i=0; i<4; ++i)
                    rows[i+4*j]=(cx+X.cells-1+i)%X.cells+X.n*((cy+Y.cells-1+j)%Y.cells);
                for (int jy=0; jy<spatial_gauss; ++jy)
                for (int jx=0; jx<spatial_gauss; ++jx) {
                    const auto& py=qy[cy*spatial_gauss+jy];
                    const auto& px=qx[cx*spatial_gauss+jx];
                    std::array<double,5> target{};
                    velocity_rhs(px.x,py.x,values,target);
                    const double dxy=px.weight*py.weight;
                    for (int k=0; k<5; ++k) local_mom[k]+=dxy*target[k];
                    for (int j=0; j<4; ++j)
                    for (int i=0; i<4; ++i) {
                        const double fac=dxy*px.basis[i]*py.basis[j];
                        double* dest=local.data()+(i+4*j)*size_v_r;
                        for (size_t k=0; k<size_v_r; ++k) dest[k]+=fac*values[k];
                    }
                }
                for (int k=0; k<5; ++k) {
                    #pragma omp atomic
                    measured[k]+=local_mom[k];
                }
                for (int q=0; q<16; ++q) {
                    const size_t row=rows[q];
                    std::lock_guard<std::mutex> lock(locks[row]);
                    const double* source=local.data()+q*size_v_r;
                    for (size_t k=0; k<size_v_r; ++k) rhs(row,k)+=source[k];
                }
            }
        }

        // Apply inverses of the two 1D spatial operators as tensor contractions.
        const arma::mat syT=syinv.t();
        #pragma omp parallel for schedule(static)
        for (size_t k=0; k<size_v_r; ++k) {
            arma::mat S(rhs.colptr(k),X.n,Y.n,false,true);
            arma::mat result=sxinv*S*syT;
            std::copy(result.begin(),result.end(),copy_mat.colptr(k));
        }

        // Apply the three 1D velocity mass-matrix inverses.
        const arma::mat vT=V.discrete_mass_inverse[velocity_gauss].t(),wT=W.discrete_mass_inverse[velocity_gauss].t();
        #pragma omp parallel for schedule(static)
        for (size_t row=0; row<size_x_r; ++row) {
            arma::mat raw(U.n,V.n*W.n);
            for (size_t j=0; j<size_v_r; ++j) raw[j]=copy_mat(row,j);
            arma::mat uu=U.discrete_mass_inverse[velocity_gauss]*raw;
            arma::mat vv(U.n,V.n*W.n);
            for (size_t iw=0; iw<W.n; ++iw) {
                arma::mat slab(uu.colptr(iw*V.n),U.n,V.n,false,true);
                arma::mat updated=slab*vT;
                std::copy(updated.begin(),updated.end(),vv.colptr(iw*V.n));
            }
            arma::mat ww=arma::mat(vv.memptr(),U.n*V.n,W.n,false,true)*wT;
            for (size_t j=0; j<size_v_r; ++j) copy_mat(row,j)=ww[j];
        }

        const auto before=global_moments(copy_mat);
        if (!(before[0]>0) || !std::isfinite(before[0])) throw std::runtime_error("Nonpositive projected species mass.");
        const double scale=initial_mass/before[0];
        copy_mat*=scale;
        const auto after=global_moments(copy_mat);
        std::swap(copy_mat,restart_matrix);

        const double referenceK=measured[4];
        std::cout << std::setprecision(16)
                  << "B-spline " << (spatial_gauss ? "full-5D" : "velocity-only") << " Galerkin restart: "
                  << "quadrature mass=" << measured[0] << ", projected mass=" << before[0]
                  << ", exact initial mass=" << initial_mass << ", scale=" << scale
                  << ", final mass=" << after[0] << ", relative mass defect=" << (after[0]-initial_mass)/initial_mass
                  << ", quadrature kinetic=" << referenceK << ", FE kinetic before=" << before[4]
                  << ", FE kinetic after=" << after[4] << '\n';
    }
};

} // namespace nufi::restart
#endif
