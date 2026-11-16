#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
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
#include <nufi/restart.hpp>
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


arma::mat restart_matrix;

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

const size_t nx_r = Nx;
const size_t nu_r = Nu;
const size_t nv_r = Nv;

// For multi-species:
const size_t nu_e_r = Nu_e;
const size_t nv_e_r = Nv_e;
const size_t nu_i_r = Nu_i;
const size_t nv_i_r = Nv_i;

size_t nt_restart = 20; // Set Nt+1 for a run without restarts.

const double dx_r = Lx / nx_r;

const double du_r = (umax - umin) / nu_r;
const double dv_r = (vmax - vmin) / nv_r;

// Conservative analytic-f1 / tensor-product cubic-FE f2 restarts for 1x2v.
// The mass of each species is invariant; its particle momentum is NOT invariant
// under Maxwell coupling. Hence only global species mass is constrained.
namespace conservative_fe_1x2v {

constexpr size_t gauss_order = 4;
constexpr size_t rank_f2 = 16;
constexpr bool low_rank = true;
constexpr bool randomized_svd = true; // Reuse NuFI restart::randomized_svd_new.
constexpr bool residual_from_corrected_f1 = true;
// Independent FE restart grids. For the narrow shock distributions, these
// should be refined in a convergence study; they are not the transport grids.
constexpr size_t nu_e = 128, nv_e = 128;
constexpr size_t nu_i = 256, nv_i = 128;
constexpr double sigma_ue = 0.075, sigma_ve = 0.025;
constexpr double sigma_ui = 0.075, sigma_vi = 0.015;

struct Basis {
    size_t first;
    std::array<double,4> w;
};

struct Gauss {
    arma::vec nodes, weights;
};

Gauss gauss_legendre(size_t n)
{
    arma::mat T(n,n,arma::fill::zeros);
    for(size_t i=1; i<n; ++i) T(i-1,i)=T(i,i-1)=double(i)/std::sqrt(4.0*i*i-1.0);
    arma::vec z;
    arma::mat U;
    arma::eig_sym(z,U,T);
    arma::vec w=2.0*arma::square(U.row(0).t());
    for(size_t i=0; i<n/2; ++i) {
        size_t j=n-1-i;
        double a=0.5*(std::abs(z(i))+std::abs(z(j)));
        double b=0.5*(w(i)+w(j));
        z(i)=-a; z(j)=a; w(i)=w(j)=b;
    }
    if(n%2) z(n/2)=0.0;
    return {std::move(z),std::move(w)};
}

Basis bspline(double x, const arma::vec& knots, size_t n, double lo, double h)
{
    const double z=std::floor((x-(lo+2*h))/h)+4;
    const size_t span=size_t(std::max(3.0,std::min(double(n-1),z)));
    std::array<double,4> N{1.0,0.0,0.0,0.0};
    double left[4]{}, right[4]{};
    for(size_t j=1; j<=3; ++j) {
        left[j]=x-knots(span+1-j);
        right[j]=knots(span+j)-x;
        double saved=0.0;
        for(size_t k=0; k<j; ++k) {
            const double d=right[k+1]+left[j-k];
            const double temp=d!=0.0 ? N[k]/d : 0.0;
            N[k]=saved+right[k+1]*temp;
            saved=left[j-k]*temp;
        }
        N[j]=saved;
    }
    return {span-3,N};
}

struct Axis {
    size_t n;
    double lo, hi, h;
    arma::vec knots, q, w, m0, m1, m2;
    arma::mat chol;
    std::vector<Basis> bq;

    Axis(double a,double b,size_t num): n(num),lo(a),hi(b),h((b-a)/(num-1))
    {
        if(num<6 || b<=a) throw std::runtime_error("Invalid cubic FE velocity grid");
        arma::vec nodes=arma::linspace<arma::vec>(a,b,n);
        knots.set_size(n+4);
        for(size_t i=0; i<4; ++i) knots(i)=lo;
        for(size_t i=2; i<=n-3; ++i) knots(i+2)=nodes(i);
        for(size_t i=n; i<n+4; ++i) knots(i)=hi;
        Gauss rule=gauss_legendre(gauss_order);
        const size_t nq=(n-1)*gauss_order;
        q.set_size(nq); w.set_size(nq);
        arma::mat mass(n,n,arma::fill::zeros);
        bq.reserve(nq);
        for(size_t cell=0; cell<n-1; ++cell)
            for(size_t j=0; j<gauss_order; ++j) {
                size_t k=cell*gauss_order+j;
                q(k)=0.5*(nodes(cell)+nodes(cell+1))+0.5*h*rule.nodes(j);
                w(k)=0.5*h*rule.weights(j);
                Basis B=bspline(q(k),knots,n,lo,h);
                bq.push_back(B);
                for(size_t s=0; s<4; ++s)
                    for(size_t t=0; t<4; ++t)
                        mass(B.first+s,B.first+t)+=w(k)*B.w[s]*B.w[t];
            }
        mass=0.5*(mass+mass.t());
        if(!arma::chol(chol,mass,"lower")) throw std::runtime_error("Velocity FE mass factor failed");
        m0.set_size(n); m1.set_size(n); m2.set_size(n);
        for(size_t i=0; i<n; ++i) {
            double s1=0.0,s2=0.0;
            for(size_t j=0; j<5; ++j) { s1+=knots(i+j); s2+=knots(i+j)*knots(i+j); }
            double b0=(knots(i+4)-knots(i))/4;
            m0(i)=b0;
            m1(i)=b0*s1/5;
            m2(i)=b0*(s1*s1+s2)/30;
        }
    }

    arma::mat solve(const arma::mat& rhs) const
    {
        arma::mat tmp=arma::solve(arma::trimatl(chol),rhs);
        return arma::solve(arma::trimatu(chol.t()),tmp);
    }
};

struct Spatial {
    size_t nx;
    double dx;
    arma::mat chol;

    Spatial(size_t n,double h):nx(n),dx(h)
    {
        arma::mat mass(nx,nx,arma::fill::zeros);
        Gauss g=gauss_legendre(4);
        for(size_t i=0;i<nx;++i)
            for(size_t iq=0;iq<4;++iq) {
                double t=0.5*(g.nodes(iq)+1),a=1-t;
                double B[4]={a*a*a,(3*t*t*t-6*t*t+4),(-3*t*t*t+3*t*t+3*t+1),t*t*t};
                double wt=0.5*dx*g.weights(iq)/36;
                for(size_t j=0;j<4;++j)
                    for(size_t k=0;k<4;++k)
                        mass((i+j+nx-1)%nx,(i+k+nx-1)%nx)+=wt*B[j]*B[k];
            }
        if(!arma::chol(chol,mass,"lower")) throw std::runtime_error("Spatial FE mass factor failed");
    }

    arma::mat interpolate(const arma::mat& nodal) const
    {
        arma::cx_mat a=arma::fft(nodal);
        for(size_t k=0;k<nx;++k)
            a.row(k)/=(4+2*std::cos(2*M_PI*k/nx))/6;
        return arma::real(arma::ifft(a));
    }
};

Basis periodic_basis(double x,double dx,size_t nx)
{
    const double z=x/dx;
    size_t i=std::min(size_t(std::floor(z)),nx-1);
    double t=std::max(0.0,std::min(1.0,z-i)),a=1-t;
    return {(i+nx-1)%nx,{a*a*a/6,(3*t*t*t-6*t*t+4)/6,
        (-3*t*t*t+3*t*t+3*t+1)/6,t*t*t/6}};
}

struct Velocity {
    Axis u,v;
    size_t n;
    double sig_u,sig_v,c0;
    arma::mat G, H, L, projector;

    Velocity(double umin,double umax,double vmin,double vmax,size_t nu,size_t nv,double su,double sv):
        u(umin,umax,nu),v(vmin,vmax,nv),n(nu*nv),sig_u(su),sig_v(sv)
    {
        double denom=0.0, numerator=0.0;
        for(size_t iu=0;iu<u.q.n_elem;++iu)
            for(size_t iv=0;iv<v.q.n_elem;++iv) {
                const double x=u.q(iu),y=v.q(iv);
                const double w=u.w(iu)*v.w(iv)*std::exp(-0.5*(x*x/(su*su)+y*y/(sv*sv)));
                denom+=w; numerator+=w*(x*x+y*y);
            }
        c0=numerator/denom;
        G.zeros(4,4); H.zeros(n,4); L.zeros(n,4);
        for(size_t j=0;j<v.n;++j)
            for(size_t i=0;i<u.n;++i) {
                size_t k=i+u.n*j;
                L(k,0)=u.m0(i)*v.m0(j);
                L(k,1)=u.m1(i)*v.m0(j);
                L(k,2)=u.m0(i)*v.m1(j);
                L(k,3)=0.5*(u.m2(i)*v.m0(j)+u.m0(i)*v.m2(j));
            }
        for(size_t iu=0;iu<u.q.n_elem;++iu)
            for(size_t iv=0;iv<v.q.n_elem;++iv) {
                double x=u.q(iu),y=v.q(iv),w=u.w(iu)*v.w(iv);
                auto g=carriers(x,y);
                double p[4]={1,x,y,0.5*(x*x+y*y)};
                for(size_t i=0;i<4;++i)
                    for(size_t j=0;j<4;++j) G(i,j)+=w*p[i]*g[j];
                const Basis& bu=u.bq[iu];
                const Basis& bv=v.bq[iv];
                for(size_t a=0;a<4;++a)
                    for(size_t b=0;b<4;++b) {
                        size_t k=(bu.first+a)+u.n*(bv.first+b);
                        double wb=w*bu.w[a]*bv.w[b];
                        for(size_t j=0;j<4;++j) H(k,j)+=wb*g[j];
                    }
            }
        if(arma::rcond(G)<1e-12) throw std::runtime_error("Gaussian carrier moment matrix ill-conditioned");
        const arma::mat Z=solve_mass(L);
        projector=arma::solve((L.t()*Z).t(),Z.t()).t(); // M^{-1}L (L^T M^{-1} L)^{-1}
    }

    std::array<double,4> carriers(double x,double y) const
    {
        const double g=std::exp(-0.5*(x*x/(sig_u*sig_u)+y*y/(sig_v*sig_v)));
        return {g,x*g,y*g,(x*x+y*y-c0)*g};
    }

    arma::mat solve_mass(const arma::mat& RHS) const
    {
        arma::mat result(RHS.n_rows,RHS.n_cols);
        for(size_t k=0;k<RHS.n_cols;++k) {
            arma::mat A=arma::reshape(RHS.col(k),u.n,v.n);
            A=u.solve(A);
            A=v.solve(A.t()).t();
            result.col(k)=arma::vectorise(A);
        }
        return result;
    }

    // Multiply on the right by the tensor Cholesky factor kron(chol(v),chol(u)).
    arma::mat weight(const arma::mat& C) const
    {
        arma::mat out(C.n_rows,n);
        for(size_t i=0;i<C.n_rows;++i) {
            arma::mat A=arma::reshape(C.row(i).t(),u.n,v.n);
            out.row(i)=arma::vectorise(u.chol.t()*A*v.chol).t();
        }
        return out;
    }

    // Inverse transpose of tensor Cholesky for right singular vectors.
    arma::mat unweight_vectors(const arma::mat& vectors) const
    {
        arma::mat out(n,vectors.n_cols);
        for(size_t i=0;i<vectors.n_cols;++i) {
            arma::mat A=arma::reshape(vectors.col(i),u.n,v.n);
            A=arma::solve(arma::trimatu(u.chol.t()),A);
            A=arma::solve(arma::trimatu(v.chol.t()),A.t()).t();
            out.col(i)=arma::vectorise(A);
        }
        return out;
    }
};

struct State {
    const Velocity* vel=nullptr;
    arma::mat carrier_coeff, X, V, C2;
    size_t nx=0,rank=0;
    double x0=0,Lx=0,dx=0,M0=0,source_mass=0,mass=0,relative_error=0;
    arma::rowvec moments_source, moments_final, f2_global;
    double f2_defect=0;

    double eval(double x,double u,double v) const noexcept
    {
        if(!vel || u<vel->u.lo || u>vel->u.hi || v<vel->v.lo || v>vel->v.hi) return 0.0;
        x=std::fmod(x-x0,Lx); if(x<0) x+=Lx;
        Basis bx=periodic_basis(x,dx,nx);
        auto gu=vel->carriers(u,v);
        double value=0.0;
        for(size_t k=0;k<4;++k) {
            double a=0.0;
            for(size_t j=0;j<4;++j) a+=bx.w[j]*carrier_coeff((bx.first+j)%nx,k);
            value+=a*gu[k];
        }
        Basis bu=bspline(u,vel->u.knots,vel->u.n,vel->u.lo,vel->u.h);
        Basis bv=bspline(v,vel->v.knots,vel->v.n,vel->v.lo,vel->v.h);
        if(V.n_cols) {
            for(size_t r=0;r<rank;++r) {
                double ax=0.0, av=0.0;
                for(size_t j=0;j<4;++j) ax+=bx.w[j]*X((bx.first+j)%nx,r);
                for(size_t j=0;j<4;++j)
                    for(size_t k=0;k<4;++k)
                        av+=bu.w[j]*bv.w[k]*V(bu.first+j+vel->u.n*(bv.first+k),r);
                value+=ax*av;
            }
        } else if(C2.n_elem) {
            for(size_t j=0;j<4;++j)
                for(size_t k=0;k<4;++k)
                    for(size_t a=0;a<4;++a)
                        value+=bx.w[j]*bu.w[k]*bv.w[a]*C2((bx.first+j)%nx,bu.first+k+vel->u.n*(bv.first+a));
        }
        return value;
    }
};

// States are swapped only after an entire restart has been constructed.
State electron_state, ion_state;
double f_electron(double x,double u,double v) { return electron_state.eval(x,u,v); }
double f_ion(double x,double u,double v) { return ion_state.eval(x,u,v); }

// Exact truncated-Maxwellian mass for the currently selected shock setup.
inline double normal_mass(double lo,double hi,double mean,double sigma)
{
    const double d=std::sqrt(2.0)*sigma;
    return 0.5*(std::erf((hi-mean)/d)-std::erf((lo-mean)/d));
}

double initial_mass_shock(bool electron,size_t nx)
{
    const double mass_v=normal_mass(electron ? vmin_e : vmin_i,
        electron ? vmax_e : vmax_i,0.0,
        electron ? electron_ion_shock::vth_e : electron_ion_shock::vth_i);
    const double sig=electron ? electron_ion_shock::vth_e : electron_ion_shock::vth_i;
    const double xstep=Lx/nx;
    double total=0.0;
    for(size_t i=0;i<nx;++i) {
        const double x=i*xstep;
        const double mu=electron_ion_shock::drift_velocity(x);
        const double prob_u=normal_mass(electron ? umin_e : umin_i,
            electron ? umax_e : umax_i,mu,sig);
        total+=electron_ion_shock::density_profile(x)*prob_u*mass_v;
    }
    return xstep*total;
}

template<typename Eval>
State make_restart(const Velocity& vel,const Spatial& spatial,double x0,double Lx,
                   double initial_mass,Eval&& evaluate)
{
    const size_t nx=spatial.nx;
    arma::mat rhs(nx,vel.n,arma::fill::zeros), moments(nx,4,arma::fill::zeros);
    #pragma omp parallel for schedule(static)
    for(std::ptrdiff_t ix=0;ix<std::ptrdiff_t(nx);++ix) {
        arma::mat local(vel.u.n,vel.v.n,arma::fill::zeros);
        arma::rowvec m(4,arma::fill::zeros);
        const double x=x0+size_t(ix)*spatial.dx;
        for(size_t iu=0;iu<vel.u.q.n_elem;++iu)
            for(size_t iv=0;iv<vel.v.q.n_elem;++iv) {
                const double u=vel.u.q(iu),v=vel.v.q(iv);
                const double wf=vel.u.w(iu)*vel.v.w(iv)*evaluate(x,u,v);
                m(0)+=wf; m(1)+=u*wf; m(2)+=v*wf;
                m(3)+=0.5*(u*u+v*v)*wf;
                const Basis& bu=vel.u.bq[iu];
                const Basis& bv=vel.v.bq[iv];
                for(size_t a=0;a<4;++a)
                    for(size_t b=0;b<4;++b)
                        local(bu.first+a,bv.first+b)+=wf*bu.w[a]*bv.w[b];
            }
        rhs.row(size_t(ix))=arma::vectorise(local).t();
        moments.row(size_t(ix))=m;
    }

    State state;
    state.vel=&vel; state.nx=nx; state.x0=x0; state.Lx=Lx;
    state.dx=spatial.dx; state.M0=initial_mass;
    state.moments_source=state.dx*arma::sum(moments,0);
    state.source_mass=state.moments_source(0);
    arma::mat corrected=spatial.interpolate(moments);
    const double M1=state.dx*arma::accu(corrected.col(0));
    corrected.col(0)+=(initial_mass-M1)/Lx;
    state.carrier_coeff=arma::solve(vel.G,corrected.t()).t();

    // In C++ the residual can be formed with corrected f1, unlike the earlier
    // 1x1v prototype, and the zero-moment projection remains explicit.
    const arma::mat raw_coeff=arma::solve(vel.G,moments.t()).t();
    const arma::mat carriers= residual_from_corrected_f1 ?
        arma::solve(vel.G,corrected.t()).t() : raw_coeff;
    const arma::mat rhs2=rhs-carriers*vel.H.t();
    arma::mat C2val=vel.solve_mass(rhs2.t()).t();
    arma::mat C2=spatial.interpolate(C2val);
    C2-=(C2*vel.L)*vel.projector.t();

    if(low_rank) {
        arma::mat F=spatial.chol.t()*vel.weight(C2);
        arma::mat U,V;
        arma::vec s;
        if(randomized_svd) {
            const auto Av=[&F](const arma::vec& z) { return arma::vec(F*z); };
            const auto Atv=[&F](const arma::vec& z) { return arma::vec(F.t()*z); };
            restart::randomized_svd_new(Av,Atv,F.n_rows,F.n_cols,rank_f2,U,s,V);
        } else if(!arma::svd_econ(U,s,V,F)) {
            throw std::runtime_error("Weighted FE SVD failed");
        }
        state.rank=std::min(rank_f2,size_t(s.n_elem));
        if(state.rank)
            state.relative_error=arma::norm(F-U.cols(0,state.rank-1)*
                arma::diagmat(s.head(state.rank))*V.cols(0,state.rank-1).t(),"fro")/
                std::max(arma::norm(F,"fro"),std::numeric_limits<double>::epsilon());
        else state.relative_error=arma::norm(F,"fro")>0 ? 1.0 : 0.0;
        if(state.rank) {
            state.X=arma::solve(arma::trimatu(spatial.chol.t()),
                U.cols(0,state.rank-1)*arma::diagmat(s.head(state.rank)));
            state.V=vel.unweight_vectors(V.cols(0,state.rank-1));
            state.V-=vel.projector*(vel.L.t()*state.V);
        }
    } else state.C2=std::move(C2);

    arma::mat f2mom(nx,4,arma::fill::zeros);
    if(low_rank && state.rank) f2mom=state.X*(state.V.t()*vel.L);
    if(!low_rank) f2mom=state.C2*vel.L;
    state.f2_defect=arma::norm(f2mom,"fro");
    state.f2_global=state.dx*arma::sum(f2mom,0);
    state.moments_final=state.dx*arma::sum(corrected,0)+state.f2_global;
    state.mass=state.moments_final(0);
    return state;
}

void log_restart(const char* species,double t,const State& s,std::ostream& out)
{
    out << std::setprecision(16) << t << ' ' << species << ' '
        << s.M0 << ' ' << s.source_mass << ' ' << s.mass << ' '
        << s.moments_source(1) << ' ' << s.moments_final(1) << ' '
        << s.moments_source(2) << ' ' << s.moments_final(2) << ' '
        << s.moments_source(3) << ' ' << s.moments_final(3) << ' '
        << s.rank << ' ' << s.relative_error << ' ' << s.f2_defect << '\n';
    std::cout << std::setprecision(10) << t << " " << species
              << " source dM=" << s.source_mass-s.M0 << " restart dM=" << s.mass-s.M0
              << " dJu=" << s.moments_final(1)-s.moments_source(1)
              << " dJv=" << s.moments_final(2)-s.moments_source(2)
              << " f2mom=" << s.f2_defect << " rank=" << s.rank
              << " LRrel=" << s.relative_error << '\n';
}

} // namespace conservative_fe_1x2v

double linear_interpolation_3d(double x, double u, double v)
{
    // This version is more stable.
    if( u > umax || u < umin 
        || v > vmax || v < vmin ){
		return 0;
	} 

    x = std::fmod(std::fmod(x, Lx) + Lx, Lx);
    size_t x_ref_pos = std::min(static_cast<size_t>(std::floor(x / dx_r)), nx_r - 1);

    size_t u_ref_pos = std::min(static_cast<size_t>(std::floor((u-umin)/du_r)), nu_r - 1);
    size_t v_ref_pos = std::min(static_cast<size_t>(std::floor((v-vmin)/dv_r)), nv_r - 1);


    double x0 = x_ref_pos*dx_r;
    double u0 = umin + u_ref_pos*du_r;    
    double v0 = vmin + v_ref_pos*dv_r;    


    double w_x = (x - x0)/dx_r;
    double w_u = (u - u0)/du_r;
    double w_v = (v - v0)/dv_r;

    double value = 0;
    for(int i_x = 0; i_x <= 1; i_x++)
    for(int i_u = 0; i_u <= 1; i_u++)
    for(int i_v = 0; i_v <= 1; i_v++){
        double factor = ((1-w_x)*(i_x==0) + w_x*(i_x==1))
                    * ((1-w_u)*(i_u==0) + w_u*(i_u==1))
                    * ((1-w_v)*(i_v==0) + w_v*(i_v==1));
        
        size_t index_x = x_ref_pos + i_x;
        size_t index_u = u_ref_pos + i_u;
        size_t index_v = v_ref_pos + i_v;

        size_t index_0 = index_x;
        size_t index_1 = index_u + (nu_r+1)*index_v;

        value += factor * restart_matrix(index_0, index_1);
    }

    return value;
}

inline double cubic_interp(double p0, double p1, double p2, double p3, double t)
{
    // Catmull-Rom spline
    double a0 = -0.5*p0 + 1.5*p1 - 1.5*p2 + 0.5*p3;
    double a1 = p0 - 2.5*p1 + 2.0*p2 - 0.5*p3;
    double a2 = -0.5*p0 + 0.5*p2;
    double a3 = p1;

    return ((a0*t + a1)*t + a2)*t + a3;
}

inline size_t clamp_index(int i, size_t N)
{
    if (i < 0) return 0;
    if (i >= static_cast<int>(N)) return N - 1;
    return static_cast<size_t>(i);
}

inline size_t periodic_index(int i, size_t N)
{
    int res = i % static_cast<int>(N);
    if (res < 0) res += N;
    return static_cast<size_t>(res);
}

double cubic_interpolation_3d(double x, double u, double v)
{
    if (u > umax || u < umin || v > vmax || v < vmin) {
        return 0.0;
    }

    // periodic x
    x = std::fmod(std::fmod(x, Lx) + Lx, Lx);

    double gx = x / dx_r;
    double gu = (u - umin) / du_r;
    double gv = (v - vmin) / dv_r;

    int ix = static_cast<int>(std::floor(gx));
    int iu = static_cast<int>(std::floor(gu));
    int iv = static_cast<int>(std::floor(gv));

    double tx = gx - ix;
    double tu = gu - iu;
    double tv = gv - iv;

    double val_u_v[4][4];

    // Loop over stencil in v and u
    for (int kv = -1; kv <= 2; kv++) {
        int iv_idx = iv + kv;
        size_t ivc = clamp_index(iv_idx, nv_r + 1);

        for (int ku = -1; ku <= 2; ku++) {
            int iu_idx = iu + ku;
            size_t iuc = clamp_index(iu_idx, nu_r + 1);

            double px[4];

            // interpolate along x first
            for (int kx = -1; kx <= 2; kx++) {
                int ix_idx = ix + kx;
                size_t ixc = periodic_index(ix_idx, nx_r + 1);

                size_t index_0 = ixc;
                size_t index_1 = iuc + (nu_r + 1) * ivc;

                px[kx + 1] = restart_matrix(index_0, index_1);
            }

            val_u_v[ku + 1][kv + 1] = cubic_interp(px[0], px[1], px[2], px[3], tx);
        }
    }

    double val_v[4];

    // interpolate along u
    for (int kv = 0; kv < 4; kv++) {
        val_v[kv] = cubic_interp(
            val_u_v[0][kv],
            val_u_v[1][kv],
            val_u_v[2][kv],
            val_u_v[3][kv],
            tu
        );
    }

    // interpolate along v
    double result = cubic_interp(val_v[0], val_v[1], val_v[2], val_v[3], tv);

    return result;
}

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

    // Init restart matrices.
    std::cout << "Initialize restart matrices." << std::endl;
    size_t size_x_r = (nx_r+1);
    size_t size_v_r = (nu_r+1)*(nv_r+1);
    restart_matrix = arma::mat(size_x_r, size_v_r, arma::fill::zeros);
    arma::mat copy_mat(size_x_r, size_v_r, arma::fill::zeros);

    // Set up config.
    conf = config_t<double>(Nx, Nu, Nt, dt, 0, Lx, umin, umax, &f0);
    conf.Nv = Nv;
    conf.v_min = vmin;
    conf.v_max = vmax;
    conf.dv = (vmax - vmin)/Nv;
    conf.f0_1x2v = f0_1x2v;
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

        if(nt_r_curr == nt_restart){
            timer.reset();
            std::cout << "Restart simulation. " << std::endl;
            std::cout << "Min value restart_matrix " << restart_matrix.min() << std::endl;
            std::cout << "Max value restart_matrix " << restart_matrix.max() << std::endl;
            // Compute first restart matrix.
            #pragma omp parallel for collapse(3)
            for(size_t ix = 0; ix <= nx_r; ix++)
            for(size_t iu = 0; iu <= nu_r; iu++)
            for(size_t iv = 0; iv <= nv_r; iv++){
                double x = conf.x_min + ix*dx_r;
                double u = conf.u_min + iu*du_r;
                double v = conf.v_min + iv*dv_r;

                size_t index_0 = ix;
                size_t index_1 = iu + (nu_r+1)*iv;

                double f = periodic::redux_1x2v::eval_f_nufi_ham_lie_fBE_1x2v<double,order>(nt_r_curr, x, u, v, coeffs_Ex, coeffs_Ey, coeffs_Bz, coeffs_jx_hat, coeffs_jy_hat, conf );
                copy_mat(index_0,index_1) = f;
            }

            double timer_fill_restart_matrix = timer.elapsed();
            timer.reset();
            std::cout << "Filling restart matrix took " << timer_fill_restart_matrix << " s." << std::endl;

            restart_matrix = copy_mat;
            double timer_copy_mat = timer.elapsed();
            timer.reset();
            std::cout << "Copying restart matrix took " << timer_copy_mat << " s." << std::endl;

            std::cout << "After restart: " << std::endl;
            std::cout << "Min value restart_matrix " << restart_matrix.min() << std::endl;
            std::cout << "Max value restart_matrix " << restart_matrix.max() << std::endl;

            // Copy last entries of coeff vectors.
            #pragma omp parallel 
            for(size_t ix = 0; ix < Nx_ext; ix++){
                    coeffs_Ex[ix] = coeffs_Ex[nt_r_curr*stride_t + ix];
                    coeffs_Ey[ix] = coeffs_Ey[nt_r_curr*stride_t + ix];
                    coeffs_Bz[ix] = coeffs_Bz[nt_r_curr*stride_t + ix];
                    coeffs_jx_hat[ix] = coeffs_jx_hat[nt_r_curr*stride_t + ix];
                    coeffs_jy_hat[ix] = coeffs_jy_hat[nt_r_curr*stride_t + ix];
            }
            
            conf.f0_1x2v = linear_interpolation_3d;
            conf_test.f0_1x2v = linear_interpolation_3d;

            nt_r_curr = 1;
            double timer_copy_coeff = timer.elapsed();
            double timer_restart = timer_fill_restart_matrix + timer_copy_mat + timer_copy_coeff;
            std::cout << "Restart took: " << timer_restart << std::endl;
            total_time += timer_restart;
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

    // Init restart matrices.
    std::cout << "Initialize restart matrices." << std::endl;
    const size_t size_x_r = (nx_r + 1);
    const size_t size_v_r = (nu_r + 1) * (nv_r + 1);
    restart_matrix = arma::mat(size_x_r, size_v_r, arma::fill::zeros);
    arma::mat copy_mat(size_x_r, size_v_r, arma::fill::zeros);

    // Set up config.
    conf = config_t<double>(Nx, Nu, Nt, dt, 0, Lx, umin, umax, &f0);
    conf.Nv = Nv;
    conf.v_min = vmin;
    conf.v_max = vmax;
    conf.dv = (vmax - vmin) / Nv;
    conf.f0_1x2v = f0_1x2v;

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

        if (nt_r_curr == nt_restart) {
            timer.reset();
            std::cout << "Restart simulation. " << std::endl;
            std::cout << "Min value restart_matrix " << restart_matrix.min() << std::endl;
            std::cout << "Max value restart_matrix " << restart_matrix.max() << std::endl;

            #pragma omp parallel for collapse(3)
            for (size_t ix = 0; ix <= nx_r; ix++)
            for (size_t iu = 0; iu <= nu_r; iu++)
            for (size_t iv = 0; iv <= nv_r; iv++) {
                const double x = conf.x_min + ix * dx_r;
                const double u = conf.u_min + iu * du_r;
                const double v = conf.v_min + iv * dv_r;

                const size_t index_0 = ix;
                const size_t index_1 = iu + (nu_r + 1) * iv;

                double f = periodic::redux_1x2v::eval_f_nufi_ham_lie_Hf_HB_HE_1x2v<double, order>(
                    nt_r_curr, x, u, v, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf
                );
                copy_mat(index_0, index_1) = f;
            }

            double timer_fill_restart_matrix = timer.elapsed();
            timer.reset();
            std::cout << "Filling restart matrix took " << timer_fill_restart_matrix << " s." << std::endl;

            restart_matrix = copy_mat;
            std::ofstream mat_str("mat_" + std::to_string(n*dt) + ".txt");
            mat_str << restart_matrix;
            double timer_copy_mat = timer.elapsed();
            timer.reset();
            std::cout << "Copying restart matrix took " << timer_copy_mat << " s." << std::endl;

            std::cout << "After restart: " << std::endl;
            std::cout << "Min value restart_matrix " << restart_matrix.min() << std::endl;
            std::cout << "Max value restart_matrix " << restart_matrix.max() << std::endl;

            #pragma omp parallel for
            for (size_t ix = 0; ix < stride_t; ix++) {
                coeffs_Ex[ix] = coeffs_Ex[nt_r_curr * stride_t + ix];
                coeffs_Ey[ix] = coeffs_Ey[nt_r_curr * stride_t + ix];
                coeffs_Bz[ix] = coeffs_Bz[nt_r_curr * stride_t + ix];
            }

            //conf.f0_1x2v = linear_interpolation_3d;
            //conf_test.f0_1x2v = linear_interpolation_3d;

            conf.f0_1x2v = cubic_interpolation_3d;
            conf_test.f0_1x2v = cubic_interpolation_3d;

            nt_r_curr = 1;
            double timer_copy_coeff = timer.elapsed();
            double timer_restart = timer_fill_restart_matrix + timer_copy_mat + timer_copy_coeff;
            std::cout << "Restart took: " << timer_restart << std::endl;
            total_time += timer_restart;
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

    // Init restart matrices.
    std::cout << "Initialize restart matrices." << std::endl;
    const size_t size_x_r = (nx_r + 1);
    const size_t size_v_r = (nu_r + 1) * (nv_r + 1);
    restart_matrix = arma::mat(size_x_r, size_v_r, arma::fill::zeros);
    arma::mat copy_mat(size_x_r, size_v_r, arma::fill::zeros);

    // Set up config.
    conf = config_t<double>(Nx, Nu, Nt, dt, 0, Lx, umin, umax, &f0);
    conf.Nv = Nv;
    conf.v_min = vmin;
    conf.v_max = vmax;
    conf.dv = (vmax - vmin) / Nv;
    conf.f0_1x2v = f0_1x2v;

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

        if (nt_r_curr == nt_restart) {
            timer.reset();
            std::cout << "Restart simulation. " << std::endl;
            std::cout << "Min value restart_matrix " << restart_matrix.min() << std::endl;
            std::cout << "Max value restart_matrix " << restart_matrix.max() << std::endl;

            #pragma omp parallel for collapse(3)
            for (size_t ix = 0; ix <= nx_r; ix++)
            for (size_t iu = 0; iu <= nu_r; iu++)
            for (size_t iv = 0; iv <= nv_r; iv++) {
                const double x = conf.x_min + ix * dx_r;
                const double u = conf.u_min + iu * du_r;
                const double v = conf.v_min + iv * dv_r;

                const size_t index_0 = ix;
                const size_t index_1 = iu + (nu_r + 1) * iv;

                double f = periodic::redux_1x2v::strang_2nd_order::eval_f_nufi_ham_strang_HE_HB_Hf_HB_HE_1x2v<double, order>(
                    nt_r_curr, x, u, v, coeffs_Ex, coeffs_Ey, coeffs_Bz, conf
                );
                copy_mat(index_0, index_1) = f;
            }

            double timer_fill_restart_matrix = timer.elapsed();
            timer.reset();
            std::cout << "Filling restart matrix took " << timer_fill_restart_matrix << " s." << std::endl;

            restart_matrix = copy_mat;
            /* std::ofstream mat_str("mat_" + std::to_string(n*dt) + ".txt");
            mat_str << restart_matrix; */
            double timer_copy_mat = timer.elapsed();
            timer.reset();
            std::cout << "Copying restart matrix took " << timer_copy_mat << " s." << std::endl;

            std::cout << "After restart: " << std::endl;
            std::cout << "Min value restart_matrix " << restart_matrix.min() << std::endl;
            std::cout << "Max value restart_matrix " << restart_matrix.max() << std::endl;

            #pragma omp parallel for
            for (size_t ix = 0; ix < stride_t; ix++) {
                coeffs_Ex[ix] = coeffs_Ex[nt_r_curr * stride_t + ix];
                coeffs_Ey[ix] = coeffs_Ey[nt_r_curr * stride_t + ix];
                coeffs_Bz[ix] = coeffs_Bz[nt_r_curr * stride_t + ix];
            }

            //conf.f0_1x2v = linear_interpolation_3d;
            //conf_test.f0_1x2v = linear_interpolation_3d;

            conf.f0_1x2v = cubic_interpolation_3d;
            conf_test.f0_1x2v = cubic_interpolation_3d;

            nt_r_curr = 1;
            double timer_copy_coeff = timer.elapsed();
            double timer_restart = timer_fill_restart_matrix + timer_copy_mat + timer_copy_coeff;
            std::cout << "Restart took: " << timer_restart << std::endl;
            total_time += timer_restart;
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

nufi::restart::cubic_interpolant_1x2v interpolant_electron;
nufi::restart::cubic_interpolant_1x2v interpolant_ion;

double eval_f_electron_with_interpolant(double x, double u, double v)
{
    return interpolant_electron.cubic_interpolation_3d(x,u,v);
}

double eval_f_ion_with_interpolant(double x, double u, double v)
{
    return interpolant_ion.cubic_interpolation_3d(x,u,v);
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

    // Conservative cubic FE restart objects, independent of the transport grids.
    namespace fe = conservative_fe_1x2v;
    fe::Spatial restart_x(Nx,Lx/Nx);
    fe::Velocity restart_e(umin_e,umax_e,vmin_e,vmax_e,
        fe::nu_e,fe::nv_e,fe::sigma_ue,fe::sigma_ve);
    fe::Velocity restart_i(umin_i,umax_i,vmin_i,vmax_i,
        fe::nu_i,fe::nv_i,fe::sigma_ui,fe::sigma_vi);
    const double target_mass_e=fe::initial_mass_shock(true,Nx);
    const double target_mass_i=fe::initial_mass_shock(false,Nx);
    std::ofstream restart_log("maxwell_restart_moments.txt");
    restart_log << "# t species M0 sourceM restartM sourceJu restartJu sourceJv restartJv "
                   "sourceK restartK rank relativeLRerror f2momentDefect\n";
    std::cout << "Conservative 1x2v FE restarts: ntr=" << nt_restart
              << ", ranks=" << fe::rank_f2 << ", e=(" << fe::nu_e << "," << fe::nv_e
              << "), i=(" << fe::nu_i << "," << fe::nv_i << ")\n";

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

        if (nt_r_curr == nt_restart) {
            timer.reset();
            // Build both states using the old characteristics, then switch bases.
            auto next_e=fe::make_restart(restart_e,restart_x,conf.x_min,conf.Lx,
                target_mass_e,eval_f_t_electron);
            auto next_i=fe::make_restart(restart_i,restart_x,conf.x_min,conf.Lx,
                target_mass_i,eval_f_t_ion);
            fe::electron_state=std::move(next_e);
            fe::ion_state=std::move(next_i);
            fe::log_restart("electrons",n*dt,fe::electron_state,restart_log);
            fe::log_restart("ions",n*dt,fe::ion_state,restart_log);

            double timer_fill_restart_matrix = timer.elapsed();
            timer.reset();
            std::cout << "Filling and copying restart matrix took " << timer_fill_restart_matrix << " s." << std::endl;

            #pragma omp parallel for
            for (size_t ix = 0; ix < stride_t; ix++) {
                coeffs_Ex[ix] = coeffs_Ex[nt_r_curr * stride_t + ix];
                coeffs_Ey[ix] = coeffs_Ey[nt_r_curr * stride_t + ix];
                coeffs_Bz[ix] = coeffs_Bz[nt_r_curr * stride_t + ix];
            }

            conf_electron.f0_1x2v = fe::f_electron;
            conf_ion.f0_1x2v = fe::f_ion;

            nt_r_curr = 1;
            double timer_copy_coeff = timer.elapsed();
            double timer_restart = timer_fill_restart_matrix + timer_copy_coeff;
            std::cout << "Restart took: " << timer_restart << std::endl;
            total_time += timer_restart;
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
