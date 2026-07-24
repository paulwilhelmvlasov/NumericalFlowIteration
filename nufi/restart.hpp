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
#include <omp.h>
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
    //arma::svd(V_temp, S_temp, U_tilde, Bt);  // Bt = V * S * U_tilde^T
    arma::svd_econ(V_temp, S_temp, U_tilde, Bt);  // Bt = V * S * U_tilde^T

    // Step 6: Recover U = Q * U_tilde
    U = Q * U_tilde.cols(0, k - 1);
    S = S_temp.head(k);
    V = V_temp.cols(0, k - 1);
}


class cubic_rsvd_interpolant_3x3v
{
public:
    double xmin = 0, xmax = 1, ymin = 0, ymax = 1, zmin = 0, zmax = 1;
    double umin = -1, umax = 1, vmin = -1, vmax = 1, wmin = -1, wmax = 1;

    double Lx = 1, Ly = 1, Lz = 1;

    size_t nx_r = 8, ny_r = 8, nz_r = 8;
    size_t nu_r = 8, nv_r = 8, nw_r = 8;

    size_t size_x_r = 1, size_v_r = 1;

    double dx_r = 1, dy_r = 1, dz_r = 1;
    double du_r = 1, dv_r = 1, dw_r = 1;

    bool non_negative_enforce = true;

    // Active low-rank representation F ~= U_s V^T
    arma::mat U_s, V;

    cubic_rsvd_interpolant_3x3v() {}

    cubic_rsvd_interpolant_3x3v(
        double xmin, double xmax, double ymin, double ymax, double zmin, double zmax,
        double umin, double umax, double vmin, double vmax, double wmin, double wmax,
        size_t Nx, size_t Ny, size_t Nz, size_t Nu, size_t Nv, size_t Nw,
        bool non_negative = true)
        : xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), zmin(zmin), zmax(zmax),
          umin(umin), umax(umax), vmin(vmin), vmax(vmax), wmin(wmin), wmax(wmax),
          nx_r(Nx), ny_r(Ny), nz_r(Nz), nu_r(Nu), nv_r(Nv), nw_r(Nw),
          non_negative_enforce(non_negative)
    {
        Lx = xmax - xmin;
        Ly = ymax - ymin;
        Lz = zmax - zmin;

        dx_r = Lx/nx_r;
        dy_r = Ly/ny_r;
        dz_r = Lz/nz_r;

        du_r = (umax - umin)/nu_r;
        dv_r = (vmax - vmin)/nv_r;
        dw_r = (wmax - wmin)/nw_r;

        size_x_r = (nx_r+1)*(ny_r+1)*(nz_r+1);
        size_v_r = (nu_r+1)*(nv_r+1)*(nw_r+1);
    }

    inline size_t index_xyz(size_t ix, size_t iy, size_t iz) const
    {
        return ix + (nx_r+1)*(iy + (ny_r+1)*iz);
    }

    inline size_t index_uvw(size_t iu, size_t iv, size_t iw) const
    {
        return iu + (nu_r+1)*(iv + (nv_r+1)*iw);
    }

    inline size_t clamp_index(int i, size_t N) const
    {
        if (i < 0) return 0;
        if (i >= static_cast<int>(N)) return N-1;
        return static_cast<size_t>(i);
    }

    inline size_t periodic_index(int i, size_t N) const
    {
        int res = i % static_cast<int>(N);
        if (res < 0) res += N;
        return static_cast<size_t>(res);
    }

    inline std::array<double,4> cubic_weights(double t) const
    {
        double t2 = t*t, t3 = t2*t;
        return {
            -0.5*t + t2 - 0.5*t3,
             1.0 - 2.5*t2 + 1.5*t3,
             0.5*t + 2.0*t2 - 1.5*t3,
            -0.5*t2 + 0.5*t3
        };
    }

    void restart_f(
        const std::function<double(double,double,double,double,double,double)>& eval_f,
        size_t max_rank, double tol = 1e-8, size_t oversampling = 10)
    {
        arma::uword m = size_x_r;
        arma::uword n = size_v_r;
        arma::uword k = std::min<arma::uword>(max_rank, std::min(m,n));
        arma::uword l = std::min<arma::uword>(k + oversampling, std::min(m,n));

        // Omega^T is stored so each random vector block is contiguous.
        arma::mat OmegaT = arma::randn(l,n);
        arma::mat YT(l,m,arma::fill::zeros);

        // Y = A Omega. Each value of f is evaluated only once.
        #pragma omp parallel for
        for (size_t i = 0; i < size_x_r; ++i) {
            size_t tmp = i;
            size_t ix = tmp % (nx_r+1); tmp /= nx_r+1;
            size_t iy = tmp % (ny_r+1); tmp /= ny_r+1;
            size_t iz = tmp;

            double x = xmin + ix*dx_r;
            double y = ymin + iy*dy_r;
            double z = zmin + iz*dz_r;
            double* yi = YT.colptr(i);

            for (size_t j = 0; j < size_v_r; ++j) {
                size_t tmpv = j;
                size_t iu = tmpv % (nu_r+1); tmpv /= nu_r+1;
                size_t iv = tmpv % (nv_r+1); tmpv /= nv_r+1;
                size_t iw = tmpv;

                double u = umin + iu*du_r;
                double v = vmin + iv*dv_r;
                double w = wmin + iw*dw_r;
                double f = eval_f(x,y,z,u,v,w);

                const double* omega = OmegaT.colptr(j);
                for (arma::uword a = 0; a < l; ++a)
                    yi[a] += f*omega[a];
            }
        }

        OmegaT.reset();

        arma::mat Y = YT.t();
        YT.reset();

        arma::mat Q, R;
        if (!arma::qr_econ(Q,R,Y))
            throw std::runtime_error("QR decomposition failed in RSVD.");
        Y.reset();
        R.reset();

        // B = Q^T A. Again each value of f is evaluated only once.
        arma::mat B(l,n,arma::fill::zeros);

        #pragma omp parallel for
        for (size_t j = 0; j < size_v_r; ++j) {
            size_t tmpv = j;
            size_t iu = tmpv % (nu_r+1); tmpv /= nu_r+1;
            size_t iv = tmpv % (nv_r+1); tmpv /= nv_r+1;
            size_t iw = tmpv;

            double u = umin + iu*du_r;
            double v = vmin + iv*dv_r;
            double w = wmin + iw*dw_r;
            double* bj = B.colptr(j);

            for (size_t i = 0; i < size_x_r; ++i) {
                size_t tmp = i;
                size_t ix = tmp % (nx_r+1); tmp /= nx_r+1;
                size_t iy = tmp % (ny_r+1); tmp /= ny_r+1;
                size_t iz = tmp;

                double x = xmin + ix*dx_r;
                double y = ymin + iy*dy_r;
                double z = zmin + iz*dz_r;
                double f = eval_f(x,y,z,u,v,w);

                for (arma::uword a = 0; a < l; ++a)
                    bj[a] += f*Q(i,a);
            }
        }

        arma::mat U_tilde, V_temp;
        arma::vec s;
        if (!arma::svd_econ(U_tilde,s,V_temp,B))
            throw std::runtime_error("SVD failed in RSVD.");
        B.reset();

        s = s.head(k);
        arma::uword r = 0;
        if (!s.empty() && s(0) > 0)
            r = arma::sum(s > tol*s(0));

        if (r == 0) {
            // Keep old factors if the new decomposition is empty.
            return;
        }

        r = std::min(r,k);

        // Construct new factors locally. Active U_s/V remain untouched.
        arma::mat new_V = V_temp.cols(0,r-1);
        arma::mat new_U_s = Q * U_tilde.cols(0,r-1);
        new_U_s.each_row() %= s.head(r).t();

        // Only now replace the previous restart representation.
        U_s = std::move(new_U_s);
        V   = std::move(new_V);
    }

    double cubic_interpolation_6d(
        double x, double y, double z,
        double u, double v, double w) const
    {
        if (u < umin || u > umax ||
            v < vmin || v > vmax ||
            w < wmin || w > wmax)
            return 0.0;

        if (U_s.n_cols == 0)
            return 0.0;

        x = xmin + std::fmod(std::fmod(x-xmin,Lx)+Lx,Lx);
        y = ymin + std::fmod(std::fmod(y-ymin,Ly)+Ly,Ly);
        z = zmin + std::fmod(std::fmod(z-zmin,Lz)+Lz,Lz);

        double gx = (x-xmin)/dx_r;
        double gy = (y-ymin)/dy_r;
        double gz = (z-zmin)/dz_r;
        double gu = (u-umin)/du_r;
        double gv = (v-vmin)/dv_r;
        double gw = (w-wmin)/dw_r;

        int ix = static_cast<int>(std::floor(gx));
        int iy = static_cast<int>(std::floor(gy));
        int iz = static_cast<int>(std::floor(gz));
        int iu = static_cast<int>(std::floor(gu));
        int iv = static_cast<int>(std::floor(gv));
        int iw = static_cast<int>(std::floor(gw));

        auto wx = cubic_weights(gx-ix);
        auto wy = cubic_weights(gy-iy);
        auto wz = cubic_weights(gz-iz);
        auto wu = cubic_weights(gu-iu);
        auto wv = cubic_weights(gv-iv);
        auto ww = cubic_weights(gw-iw);

        size_t idx_x[64], idx_v[64];
        double weight_x[64], weight_v[64];

        size_t q = 0;
        for (int kz = 0; kz < 4; ++kz)
        for (int ky = 0; ky < 4; ++ky)
        for (int kx = 0; kx < 4; ++kx) {
            size_t ixc = periodic_index(ix+kx-1,nx_r);
            size_t iyc = periodic_index(iy+ky-1,ny_r);
            size_t izc = periodic_index(iz+kz-1,nz_r);
            idx_x[q] = index_xyz(ixc,iyc,izc);
            weight_x[q++] = wx[kx]*wy[ky]*wz[kz];
        }

        q = 0;
        for (int kw = 0; kw < 4; ++kw)
        for (int kv = 0; kv < 4; ++kv)
        for (int ku = 0; ku < 4; ++ku) {
            size_t iuc = clamp_index(iu+ku-1,nu_r+1);
            size_t ivc = clamp_index(iv+kv-1,nv_r+1);
            size_t iwc = clamp_index(iw+kw-1,nw_r+1);
            idx_v[q] = index_uvw(iuc,ivc,iwc);
            weight_v[q++] = wu[ku]*wv[kv]*ww[kw];
        }

        double value = 0.0;

        for (arma::uword a = 0; a < U_s.n_cols; ++a) {
            const double* uc = U_s.colptr(a);
            const double* vc = V.colptr(a);

            double u_interp = 0.0;
            double v_interp = 0.0;

            for (size_t i = 0; i < 64; ++i) {
                u_interp += weight_x[i]*uc[idx_x[i]];
                v_interp += weight_v[i]*vc[idx_v[i]];
            }

            value += u_interp*v_interp;
        }

        return non_negative_enforce ? std::max(0.0,value) : value;
    }

    size_t rank() const
    {
        return U_s.n_cols;
    }
};

class linear_interpolant_6d
{
    public:
        double xmin = 0;
        double xmax = 1;
        double ymin = 0;
        double ymax = 1;
        double zmin = 0;
        double zmax = 1;
        
        double umin = -1;
        double umax = 1;
        double vmin = -1;
        double vmax = 1;
        double wmin = -1;
        double wmax = 1;

        size_t nx_r = 8;
        size_t ny_r = 8;
        size_t nz_r = 8;
        size_t nu_r = 8;
        size_t nv_r = 8;
        size_t nw_r = 8;

        size_t size_x_r = 1;
        size_t size_v_r = 1;

        double dx_r = 1;
        double dy_r = 1;
        double dz_r = 1;
        double du_r = 1;
        double dv_r = 1;
        double dw_r = 1;

        arma::mat restart_matrix;
        arma::mat copy_mat;

        linear_interpolant_6d() { }

        linear_interpolant_6d(double xmin, double xmax, 
            double ymin, double ymax, double zmin, double zmax,
            double umin, double umax, double vmin, double vmax,
            double wmin, double wmax, size_t Nx, size_t Ny, size_t Nz,
            size_t Nu, size_t Nv, size_t Nw) 
            : xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), 
            zmin(zmin), zmax(zmax), umin(umin), umax(umax), 
            vmin(vmin), vmax(vmax), wmin(wmin), wmax(wmax),
            nx_r(Nx), ny_r(Ny), nz_r(Nz), nu_r(Nu), nv_r(Nv), nw_r(Nw)
        {
            size_x_r = (nx_r+1)*(ny_r+1)*(nz_r+1);
            size_v_r = (nu_r+1)*(nv_r+1)*(nw_r+1);

            dx_r = (xmax - xmin) / nx_r;
            dy_r = (ymax - ymin) / ny_r;
            dz_r = (zmax - zmin) / nz_r;

            du_r = (umax - umin) / nu_r;
            dv_r = (vmax - vmin) / nv_r;
            dw_r = (wmax - wmin) / nw_r;

            restart_matrix = arma::mat(size_x_r, size_v_r, arma::fill::zeros);
            copy_mat = arma::mat(size_x_r, size_v_r, arma::fill::zeros);
        }
        
        void restart_f(const std::function<double(double,double,double,double,double,double)>& eval_f)
        {
            #pragma omp parallel for collapse(6)
            for(size_t ix = 0; ix <= nx_r; ix++)
            for(size_t iy = 0; iy <= ny_r; iy++)
            for(size_t iz = 0; iz <= nz_r; iz++)
            for(size_t iu = 0; iu <= nu_r; iu++)
            for(size_t iv = 0; iv <= nv_r; iv++)
            for(size_t iw = 0; iw <= nw_r; iw++){
                double x = xmin + ix*dx_r;
                double y = ymin + iy*dy_r;
                double z = zmin + iz*dz_r;

                double u = umin + iu*du_r;
                double v = vmin + iv*dv_r;
                double w = wmin + iw*dw_r;

                size_t index_0 = ix + (nx_r+1)*(iy + (ny_r+1)*iz);
                size_t index_1 = iu + (nu_r+1)*(iv + (nv_r+1)*iw);

                copy_mat(index_0,index_1) = eval_f(x, y, z, u, v, w);
            }

            restart_matrix = copy_mat;
        }

        void restart_f_MPI(const std::function<double(double,double,double,double,double,double)>& eval_f)
        {
            int rank, size;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);

            const size_t NxTot = (nx_r+1)*(ny_r+1)*(nz_r+1);
            const size_t NvTot = (nu_r+1)*(nv_r+1)*(nw_r+1);

            arma::mat local(size_x_r, size_v_r, arma::fill::zeros);

            // -------------------------
            // MPI decomposition in configuration space
            // -------------------------
            size_t base = NxTot / size;
            size_t rem  = NxTot % size;

            size_t Nloc = base + (rank < rem ? 1 : 0);
            size_t i0   = base * rank + std::min((size_t)rank, rem);
            size_t iend = i0 + Nloc;

            // -------------------------
            // distributed configuration loop
            // -------------------------
            for(size_t I = i0; I < iend; I++)
            {
                size_t iz = I / ((nx_r+1)*(ny_r+1));
                size_t tmp = I % ((nx_r+1)*(ny_r+1));
                size_t iy = tmp / (nx_r+1);
                size_t ix = tmp % (nx_r+1);

                double x = xmin + ix*dx_r;
                double y = ymin + iy*dy_r;
                double z = zmin + iz*dz_r;

                // -------------------------
                // OpenMP over velocity space
                // -------------------------
                #pragma omp parallel for collapse(3)
                for(size_t iu = 0; iu <= nu_r; iu++)
                for(size_t iv = 0; iv <= nv_r; iv++)
                for(size_t iw = 0; iw <= nw_r; iw++)
                {
                    double u = umin + iu*du_r;
                    double v = vmin + iv*dv_r;
                    double w = wmin + iw*dw_r;

                    size_t index_1 =
                        iu + (nu_r+1)*(iv + (nv_r+1)*iw);

                    local(I, index_1) =
                        eval_f(x, y, z, u, v, w);
                }
            }

            // -------------------------
            // reconstruct restart_matrix
            // -------------------------
            MPI_Allreduce(local.memptr(),
                        restart_matrix.memptr(),
                        size_x_r * size_v_r,
                        MPI_DOUBLE,
                        MPI_SUM,
                        MPI_COMM_WORLD);
        }


        void restart_f_checking_boundaries(const std::function<double(double,double,double,double,double,double)>& eval_f, 
                        double& umin_new, double& umax_new, double& vmin_new, 
                        double& vmax_new, double& wmin_new, double& wmax_new, 
                        double tol, double expand = 1.5)
        {
            // This assumes that the velocity support is of the form v_{i,min} < 0
            // and v_{i,max} > 0.
            bool umin_expand = false;
            bool umax_expand = false;
            bool vmin_expand = false;
            bool vmax_expand = false;
            bool wmin_expand = false;
            bool wmax_expand = false;

            #pragma omp parallel for collapse(3)
            for(size_t ix = 0; ix <= nx_r; ix++)
            for(size_t iy = 0; iy <= ny_r; iy++)
            for(size_t iz = 0; iz <= nz_r; iz++)
            for(size_t iu = 0; iu <= nu_r; iu++)
            for(size_t iv = 0; iv <= nv_r; iv++)
            for(size_t iw = 0; iw <= nw_r; iw++){
                double x = xmin + ix*dx_r;
                double y = ymin + iy*dy_r;
                double z = zmin + iz*dz_r;

                double u = umin + iu*du_r;
                double v = vmin + iv*dv_r;
                double w = wmin + iw*dw_r;

                double f = 0;

                if(iu == 0 || iu == nu_r || iv == 0 
                    || iv == nv_r || iw == 0 || iw == nw_r){
                    f = eval_f(x,y,z,u,v,w);
                }

                // Check u_min boundary.
                if(iu == 0){
                    if(f > tol){
                        umin_expand = true;
                    }
                } 
                // Check u_max boundary.
                if(iu == nu_r){
                    if(f > tol){
                        umax_expand = true;
                    }
                }
                
                // Check v_min boundary.
                if(iv == 0){
                    if(f > tol){
                        vmin_expand = true;
                    }
                } 
                // Check v_max boundary.
                if(iv == nv_r){
                    if(f > tol){
                        vmax_expand = true;
                    }
                }

                // Check w_min boundary.
                if(iw == 0){
                    if(f > tol){
                        wmin_expand = true;
                    }
                } 
                // Check w_max boundary.
                if(iw == nu_r){
                    if(f > tol){
                        wmax_expand = true;
                    }
                }
            }

            if(umin_expand){
                umin_new = umin*expand;
            } 
            if(umax_expand){
                umax_new = umax*expand;
            } 

            if(vmin_expand){
                vmin_new = vmin*expand;
            } 
            if(vmax_expand){
                vmax_new = vmax*expand;
            } 

            if(wmin_expand){
                wmin_new = wmin*expand;
            } 
            if(wmax_expand){
                wmax_new = wmax*expand;
            } 

            // Restart with new velocity boundaries.
            double du_new = (umax_new - umin_new) / nu_r;
            double dv_new = (vmax_new - vmin_new) / nv_r;
            double dw_new = (wmax_new - wmin_new) / nw_r;

            #pragma omp parallel for collapse(6)
            for(size_t ix = 0; ix <= nx_r; ix++)
            for(size_t iy = 0; iy <= ny_r; iy++)
            for(size_t iz = 0; iz <= nz_r; iz++)
            for(size_t iu = 0; iu <= nu_r; iu++)
            for(size_t iv = 0; iv <= nv_r; iv++)
            for(size_t iw = 0; iw <= nw_r; iw++){
                double x = xmin + ix*dx_r;
                double y = ymin + iy*dy_r;
                double z = zmin + iz*dz_r;

                double u = umin_new + iu*du_new;
                double v = vmin_new + iv*dv_new;
                double w = wmin_new + iw*dw_new;

                size_t index_0 = ix + (nx_r+1)*(iy + (ny_r+1)*iz);
                size_t index_1 = iu + (nu_r+1)*(iv + (nv_r+1)*iw);

                copy_mat(index_0,index_1) = eval_f(x, y, z, u, v, w);
            }

            restart_matrix = copy_mat;

            umin = umin_new;
            umax = umax_new;

            vmin = vmin_new;
            vmax = vmax_new;

            wmin = wmin_new;
            wmax = wmax_new;

            du_r = du_new;
            dv_r = dv_new;
            dw_r = dw_new;
        }

        double compute_kinetic_energy()
        {
            double kin_energy = 0;
            #pragma omp parallel for collapse(3) reduction(+:kin_energy)
            for(size_t ix = 0; ix < nx_r; ix++)
            for(size_t iy = 0; iy < ny_r; iy++)
            for(size_t iz = 0; iz < nz_r; iz++)
            for(size_t iu = 0; iu <= nu_r; iu++)
            for(size_t iv = 0; iv <= nv_r; iv++)
            for(size_t iw = 0; iw <= nw_r; iw++){
                double x = xmin + ix*dx_r;
                double y = ymin + iy*dy_r;
                double z = zmin + iz*dz_r;

                double u = umin + iu*du_r;
                double v = vmin + iv*dv_r;
                double w = wmin + iw*dw_r;

                size_t index_0 = ix + (nx_r+1)*(iy + (ny_r+1)*iz);
                size_t index_1 = iu + (nu_r+1)*(iv + (nv_r+1)*iw);

                double f = restart_matrix(index_0,index_1);

                kin_energy += (u*u + v*v + w*w)*f;
            }

            kin_energy *= 0.5 * dx_r * dy_r * dz_r * du_r * dv_r * dw_r;

            return kin_energy;
        }

        double eval_linear_interpolant(double x, double y, double z, 
                                    double u, double v, double w)
        {
            // Collapsing dimensions in theory should save some computational
            // effort but in practice we noticed that the benefit is neglible 
            // or the full version is even faster in some cases.

            if( u > umax || u < umin 
                || v > vmax || v < vmin 
                || w > wmax || w < wmin){
                /* std::cout << "Return u " << u << " " << umin << " " << umax << std::endl;
                std::cout << "Return v " << v << " " << vmin << " " << vmax << std::endl;
                std::cout << "Return w " << w << " " << wmin << " " << wmax << std::endl;
                std::cout << "====================================================" << std::endl; */
                return 0;
            } 

            double Lx = xmax - xmin;
            x = std::fmod(std::fmod(x - xmin, Lx) + Lx, Lx) + xmin;
            size_t x_ref_pos = std::min(static_cast<size_t>(std::floor(x / dx_r)), nx_r - 1);
            double Ly = ymax - ymin;
            y = std::fmod(std::fmod(y - ymin, Ly) + Ly, Ly) + ymin;
            size_t y_ref_pos = std::min(static_cast<size_t>(std::floor(y / dy_r)), ny_r - 1);
            double Lz = zmax - zmin;
            z = std::fmod(std::fmod(z - zmin, Lz) + Lz, Lz) + zmin;
            size_t z_ref_pos = std::min(static_cast<size_t>(std::floor(z / dz_r)), nz_r - 1);

            size_t u_ref_pos = std::min(static_cast<size_t>(std::floor((u-umin)/du_r)), nu_r - 1);
            size_t v_ref_pos = std::min(static_cast<size_t>(std::floor((v-vmin)/dv_r)), nv_r - 1);
            size_t w_ref_pos = std::min(static_cast<size_t>(std::floor((w-wmin)/dw_r)), nw_r - 1);


            double x0 = x_ref_pos*dx_r;
            double y0 = y_ref_pos*dy_r;
            double z0 = z_ref_pos*dz_r;
            double u0 = umin + u_ref_pos*du_r;    
            double v0 = vmin + v_ref_pos*dv_r;    
            double w0 = wmin + w_ref_pos*dw_r;    


            double w_x = (x - x0)/dx_r;
            double w_y = (y - y0)/dy_r;
            double w_z = (z - z0)/dz_r;
            double w_u = (u - u0)/du_r;
            double w_v = (v - v0)/dv_r;
            double w_w = (w - w0)/dw_r;

            double value = 0;
            for(int i_x = 0; i_x <= 1; i_x++)
            for(int i_y = 0; i_y <= 1; i_y++)
            for(int i_z = 0; i_z <= 1; i_z++)
            for(int i_u = 0; i_u <= 1; i_u++)
            for(int i_v = 0; i_v <= 1; i_v++)
            for(int i_w = 0; i_w <= 1; i_w++){
                double factor = ((1-w_x)*(i_x==0) + w_x*(i_x==1))
                            * ((1-w_y)*(i_y==0) + w_y*(i_y==1))
                            * ((1-w_z)*(i_z==0) + w_z*(i_z==1))
                            * ((1-w_u)*(i_u==0) + w_u*(i_u==1))
                            * ((1-w_v)*(i_v==0) + w_v*(i_v==1))
                            * ((1-w_w)*(i_w==0) + w_w*(i_w==1));
                
                size_t index_x = x_ref_pos + i_x;
                size_t index_y = y_ref_pos + i_y;
                size_t index_z = z_ref_pos + i_z;
                size_t index_u = u_ref_pos + i_u;
                size_t index_v = v_ref_pos + i_v;
                size_t index_w = w_ref_pos + i_w;

                size_t index_0 = index_x + (nx_r+1)*(index_y + (ny_r+1)*index_z);
                size_t index_1 = index_u + (nu_r+1)*(index_v + (nv_r+1)*index_w);

                value += factor * restart_matrix(index_0, index_1);
            }

            return value;
        }
};

class flow_map_linear_interpolant_2x3v
{
    public:
    linear_interpolant_6d flow_x;
    linear_interpolant_6d flow_y;
    linear_interpolant_6d flow_u;
    linear_interpolant_6d flow_v;
    linear_interpolant_6d flow_w;

    flow_map_linear_interpolant_2x3v();
    flow_map_linear_interpolant_2x3v(double xmin, double xmax, 
            double ymin, double ymax, double zmin, double zmax,
            double umin, double umax, double vmin, double vmax,
            double wmin, double wmax, size_t Nx, size_t Ny, size_t Nz,
            size_t Nu, size_t Nv, size_t Nw, 
            const std::function<void(double&,double&,double&,double&,double&,double&)>& eval_flow_map)
    {
        flow_x = linear_interpolant_6d(xmin, xmax, ymin, ymax, zmin, zmax, umin, umax,
                             vmin, vmax, wmin, wmax, Nx, Ny, Nz, Nu, Nv, Nw);
        flow_y = linear_interpolant_6d(xmin, xmax, ymin, ymax, zmin, zmax, umin, umax,
                             vmin, vmax, wmin, wmax, Nx, Ny, Nz, Nu, Nv, Nw);
        flow_u = linear_interpolant_6d(xmin, xmax, ymin, ymax, zmin, zmax, umin, umax,
                             vmin, vmax, wmin, wmax, Nx, Ny, Nz, Nu, Nv, Nw);
        flow_v = linear_interpolant_6d(xmin, xmax, ymin, ymax, zmin, zmax, umin, umax,
                             vmin, vmax, wmin, wmax, Nx, Ny, Nz, Nu, Nv, Nw);
        flow_w = linear_interpolant_6d(xmin, xmax, ymin, ymax, zmin, zmax, umin, umax,
                             vmin, vmax, wmin, wmax, Nx, Ny, Nz, Nu, Nv, Nw);

        double dx_r = flow_x.dx_r;
        double dy_r = flow_x.dy_r;
        double dz_r = flow_x.dz_r;
        double du_r = flow_x.du_r;
        double dv_r = flow_x.dv_r;
        double dw_r = flow_x.dw_r;

        #pragma omp parallel for collapse(6)
        for(size_t ix = 0; ix <= Nx; ix++)
        for(size_t iy = 0; iy <= Ny; iy++)
        for(size_t iz = 0; iz <= Nz; iz++)
        for(size_t iu = 0; iu <= Nu; iu++)
        for(size_t iv = 0; iv <= Nv; iv++)
        for(size_t iw = 0; iw <= Nw; iw++){
            double x = xmin + ix*dx_r;
            double y = ymin + iy*dy_r;
            double z = zmin + iz*dz_r;

            double u = umin + iu*du_r;
            double v = vmin + iv*dv_r;
            double w = wmin + iw*dw_r;

            size_t index_0 = ix + (Nx+1)*(iy + (Ny+1)*iz);
            size_t index_1 = iu + (Nu+1)*(iv + (Nv+1)*iw);

            eval_flow_map(x, y, z, u, v, w);

            flow_x.restart_matrix(index_0,index_1) = x;
            flow_y.restart_matrix(index_0,index_1) = y;
            flow_u.restart_matrix(index_0,index_1) = u;
            flow_v.restart_matrix(index_0,index_1) = v;
            flow_w.restart_matrix(index_0,index_1) = w;
        }
    }

    double eval_flow_map(size_t d, double x, double y, double z, double u, double v, double w)
    {
        switch (d)
        {
        case 0:
            // x
            return flow_x.eval_linear_interpolant(x,y,z,u,v,w);
            break;
        case 1:
            // y
            return flow_y.eval_linear_interpolant(x,y,z,u,v,w);
            break;
        case 2: 
            // z
            return z;
            break;
        case 3:
            // u
            return flow_u.eval_linear_interpolant(x,y,z,u,v,w);
            break;
        case 4:
            // v
            return flow_v.eval_linear_interpolant(x,y,z,u,v,w);
            break;
        case 5:
            // w
            return flow_w.eval_linear_interpolant(x,y,z,u,v,w);
            break;
        default:
            break;
        }

        return 0;
    }

};

class cubic_interpolant_1x2v
{
    public:
        double xmin = 0;
        double xmax = 1;

        double Lx = 1;
        
        double umin = -1;
        double umax = 1;
        double vmin = -1;
        double vmax = 1;

        size_t nx_r = 8;
        size_t ny_r = 8;
        size_t nu_r = 8;
        size_t nv_r = 8;
        size_t nw_r = 8;

        size_t size_x_r = 1;
        size_t size_v_r = 1;

        double dx_r = 1;
        double dy_r = 1;
        double du_r = 1;
        double dv_r = 1;
        double dw_r = 1;

        bool non_negative_enforce = true;

        arma::mat restart_matrix; // size: (Nx_r+1)*(Ny_r+1) \times (Nu_r+1)*(Nv_r+1)*(Nw_r+1)
        arma::mat copy_mat;

        cubic_interpolant_1x2v() { }

        cubic_interpolant_1x2v(double xmin, double xmax, 
            double umin, double umax, double vmin, double vmax,
            size_t Nx, size_t Nu, size_t Nv, bool non_negative = true) 
            : xmin(xmin), xmax(xmax), umin(umin), umax(umax), vmin(vmin), vmax(vmax),
            nx_r(Nx), nu_r(Nu), nv_r(Nv), non_negative_enforce(non_negative)
        {
            size_x_r = (nx_r+1);
            size_v_r = (nu_r+1)*(nv_r+1);

            dx_r = (xmax - xmin) / nx_r;

            du_r = (umax - umin) / nu_r;
            dv_r = (vmax - vmin) / nv_r;

            Lx = (xmax - xmin);

            restart_matrix = arma::mat(size_x_r, size_v_r, arma::fill::zeros);
            copy_mat = arma::mat(size_x_r, size_v_r, arma::fill::zeros);
        }

        void restart_f(const std::function<double(double,double,double)>& eval_f)
        {
            #pragma omp parallel for collapse(3)
            for(size_t ix = 0; ix <= nx_r; ix++)
            for(size_t iu = 0; iu <= nu_r; iu++)
            for(size_t iv = 0; iv <= nv_r; iv++){
                double x = xmin + ix*dx_r;

                double u = umin + iu*du_r;
                double v = vmin + iv*dv_r;

                size_t index_0 = ix;
                size_t index_1 = iu + (nu_r+1)*iv;

                copy_mat(index_0,index_1) = eval_f(x, u, v);
            }

            restart_matrix = copy_mat;
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

            if(non_negative_enforce){
                return std::max(0.0,result);
            }
            return result;
        }

};

class cubic_interpolant_2x3v
{
    public:
        double xmin = 0;
        double xmax = 1;
        double ymin = 0;
        double ymax = 1;

        double Lx = 1;
        double Ly = 1;
        
        double umin = -1;
        double umax = 1;
        double vmin = -1;
        double vmax = 1;
        double wmin = -1;
        double wmax = 1;

        size_t nx_r = 8;
        size_t ny_r = 8;
        size_t nz_r = 8;
        size_t nu_r = 8;
        size_t nv_r = 8;
        size_t nw_r = 8;

        size_t size_x_r = 1;
        size_t size_v_r = 1;

        double dx_r = 1;
        double dy_r = 1;
        double dz_r = 1;
        double du_r = 1;
        double dv_r = 1;
        double dw_r = 1;

        bool non_negative_enforce = true;

        arma::mat restart_matrix; // size: (Nx_r+1)*(Ny_r+1) \times (Nu_r+1)*(Nv_r+1)*(Nw_r+1)
        arma::mat copy_mat;

        cubic_interpolant_2x3v() { }

        cubic_interpolant_2x3v(double xmin, double xmax, 
            double ymin, double ymax, double umin, double umax, 
            double vmin, double vmax, double wmin, double wmax, 
            size_t Nx, size_t Ny, size_t Nu, size_t Nv, size_t Nw,
            bool non_negative = true) 
            : xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), 
            umin(umin), umax(umax), vmin(vmin), vmax(vmax), wmin(wmin), wmax(wmax),
            nx_r(Nx), ny_r(Ny), nu_r(Nu), nv_r(Nv), nw_r(Nw), non_negative_enforce(non_negative)
        {
            size_x_r = (nx_r+1)*(ny_r+1);
            size_v_r = (nu_r+1)*(nv_r+1)*(nw_r+1);

            dx_r = (xmax - xmin) / nx_r;
            dy_r = (ymax - ymin) / ny_r;

            du_r = (umax - umin) / nu_r;
            dv_r = (vmax - vmin) / nv_r;
            dw_r = (wmax - wmin) / nw_r;

            Lx = (xmax - xmin);
            Ly = (ymax - ymin);

            restart_matrix = arma::mat(size_x_r, size_v_r, arma::fill::zeros);
            copy_mat = arma::mat(size_x_r, size_v_r, arma::fill::zeros);
        }

        void restart_f(const std::function<double(double,double,double,double,double)>& eval_f)
        {
            #pragma omp parallel for collapse(5)
            for(size_t ix = 0; ix <= nx_r; ix++)
            for(size_t iy = 0; iy <= ny_r; iy++)
            for(size_t iu = 0; iu <= nu_r; iu++)
            for(size_t iv = 0; iv <= nv_r; iv++)
            for(size_t iw = 0; iw <= nw_r; iw++){
                double x = xmin + ix*dx_r;
                double y = ymin + iy*dy_r;

                double u = umin + iu*du_r;
                double v = vmin + iv*dv_r;
                double w = wmin + iw*dw_r;

                size_t index_0 = ix + (nx_r+1)*iy;
                size_t index_1 = iu + (nu_r+1)*(iv + (nv_r+1)*iw);

                copy_mat(index_0,index_1) = eval_f(x, y, u, v, w);
            }

            restart_matrix = copy_mat;
        }

        void restart_f_MPI(const std::function<double(double,double,double,double,double)>& eval_f)
        {
            int rank, size;
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);
            MPI_Comm_size(MPI_COMM_WORLD, &size);

            const size_t NxTot = (nx_r+1)*(ny_r+1);
            const size_t NvTot = (nu_r+1)*(nv_r+1)*(nw_r+1);

            arma::mat local(size_x_r, size_v_r, arma::fill::zeros);

            // -------------------------
            // MPI decomposition in configuration space
            // -------------------------
            size_t base = NxTot / size;
            size_t rem  = NxTot % size;

            size_t Nloc = base + (rank < rem ? 1 : 0);
            size_t i0   = base * rank + std::min((size_t)rank, rem);
            size_t iend = i0 + Nloc;

            // -------------------------
            // distributed configuration loop
            // -------------------------
            for(size_t I = i0; I < iend; I++)
            {
                size_t iy = I / (nx_r+1);
                size_t ix = I % (nx_r+1);

                double x = xmin + ix*dx_r;
                double y = ymin + iy*dy_r;

                // -------------------------
                // OpenMP over velocity space
                // -------------------------
                #pragma omp parallel for collapse(3)
                for(size_t iu = 0; iu <= nu_r; iu++)
                for(size_t iv = 0; iv <= nv_r; iv++)
                for(size_t iw = 0; iw <= nw_r; iw++)
                {
                    double u = umin + iu*du_r;
                    double v = vmin + iv*dv_r;
                    double w = wmin + iw*dw_r;

                    size_t index_1 =
                        iu + (nu_r+1)*(iv + (nv_r+1)*iw);

                    local(I, index_1) =
                        eval_f(x, y, u, v, w);
                }
            }

            // -------------------------
            // reconstruct restart_matrix
            // -------------------------
            MPI_Allreduce(local.memptr(),
                        restart_matrix.memptr(),
                        size_x_r * size_v_r,
                        MPI_DOUBLE,
                        MPI_SUM,
                        MPI_COMM_WORLD);
        }

        inline size_t index_xy(size_t ix, size_t iy)
        {
            return ix + (nx_r + 1) * iy;
        }

        inline size_t index_uvw(size_t iu, size_t iv, size_t iw)
        {
            return iu + (nu_r + 1) * (iv + (nv_r + 1) * iw);
        }

        inline double restart_value_2x3v(
            size_t ix, size_t iy,
            size_t iu, size_t iv, size_t iw)
        {
            return restart_matrix(
                index_xy(ix, iy),
                index_uvw(iu, iv, iw)
            );
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

        double cubic_interpolation_5d_streaming(
            double x, double y,
            double u, double v, double w)
        {
            if (u < umin || u > umax ||
                v < vmin || v > vmax ||
                w < wmin || w > wmax) {
                return 0.0;
            }

            // periodic space
            x = std::fmod(std::fmod(x, Lx) + Lx, Lx);
            y = std::fmod(std::fmod(y, Ly) + Ly, Ly);

            const double gx = x / dx_r;
            const double gy = y / dy_r;
            const double gu = (u - umin) / du_r;
            const double gv = (v - vmin) / dv_r;
            const double gw = (w - wmin) / dw_r;

            const int ix = static_cast<int>(std::floor(gx));
            const int iy = static_cast<int>(std::floor(gy));
            const int iu = static_cast<int>(std::floor(gu));
            const int iv = static_cast<int>(std::floor(gv));
            const int iw = static_cast<int>(std::floor(gw));

            const double tx = gx - ix;
            const double ty = gy - iy;
            const double tu = gu - iu;
            const double tv = gv - iv;
            const double tw = gw - iw;

            double buf_w[4];

            for (int kw = -1; kw <= 2; ++kw) {
                const size_t iwc = clamp_index(iw + kw, nw_r + 1);

                double buf_v[4];

                for (int kv = -1; kv <= 2; ++kv) {
                    const size_t ivc = clamp_index(iv + kv, nv_r + 1);

                    double buf_u[4];

                    for (int ku = -1; ku <= 2; ++ku) {
                        const size_t iuc = clamp_index(iu + ku, nu_r + 1);

                        double buf_y[4];

                        for (int ky = -1; ky <= 2; ++ky) {
                            const size_t iyc = periodic_index(iy + ky, ny_r + 1);

                            double px[4];

                            for (int kx = -1; kx <= 2; ++kx) {
                                const size_t ixc = periodic_index(ix + kx, nx_r + 1);

                                px[kx + 1] = restart_value_2x3v(
                                    ixc, iyc, iuc, ivc, iwc
                                );
                            }

                            buf_y[ky + 1] =
                                cubic_interp(px[0], px[1], px[2], px[3], tx);
                        }

                        buf_u[ku + 1] =
                            cubic_interp(buf_y[0], buf_y[1], buf_y[2], buf_y[3], ty);
                    }

                    buf_v[kv + 1] =
                        cubic_interp(buf_u[0], buf_u[1], buf_u[2], buf_u[3], tu);
                }

                buf_w[kw + 1] =
                    cubic_interp(buf_v[0], buf_v[1], buf_v[2], buf_v[3], tv);
            }

            double value = cubic_interp(
                buf_w[0], buf_w[1], buf_w[2], buf_w[3], tw
            );

            if(non_negative_enforce){
                return std::max(0.0,value);
            }
            return value;
        }

};


class KDTree
{
public:
    KDTree() = default;

    KDTree(const arma::mat& points)
    {
        build(points);
    }

    void build(const arma::mat& points)
    {
        points_ = &points;
        dim_ = points.n_cols;

        indices_.resize(points.n_rows);
        std::iota(indices_.begin(), indices_.end(), arma::uword(0));

        nodes_.clear();
        nodes_.reserve(points.n_rows);

        root_ = build_recursive(0, indices_.size(), 0);
    }

    void radius_search(const arma::rowvec& q,
                       double radius,
                       std::vector<arma::uword>& result) const
    {
        const double radius2 = radius * radius;
        search_recursive(root_, q, radius2, result);
    }

private:
    struct Node
    {
        arma::uword index = 0;
        arma::uword axis = 0;
        int left = -1;
        int right = -1;
    };

    const arma::mat* points_ = nullptr;
    arma::uword dim_ = 0;
    int root_ = -1;

    std::vector<arma::uword> indices_;
    std::vector<Node> nodes_;

    int build_recursive(std::size_t begin, std::size_t end, arma::uword depth)
    {
        if (begin >= end)
            return -1;

        const arma::uword axis = depth % dim_;
        const std::size_t mid = begin + (end - begin) / 2;

        std::nth_element(indices_.begin() + begin,
                         indices_.begin() + mid,
                         indices_.begin() + end,
                         [&](arma::uword a, arma::uword b)
                         {
                             return (*points_)(a, axis) < (*points_)(b, axis);
                         });

        const int node_id = static_cast<int>(nodes_.size());
        nodes_.push_back(Node{indices_[mid], axis, -1, -1});

        nodes_[node_id].left  = build_recursive(begin, mid, depth + 1);
        nodes_[node_id].right = build_recursive(mid + 1, end, depth + 1);

        return node_id;
    }

    void search_recursive(int node_id,
                          const arma::rowvec& q,
                          double radius2,
                          std::vector<arma::uword>& result) const
    {
        if (node_id < 0)
            return;

        const Node& node = nodes_[node_id];
        const arma::rowvec p = points_->row(node.index);

        const double d2 = arma::accu(arma::square(q - p));

        if (d2 <= radius2)
            result.push_back(node.index);

        const double diff = q(node.axis) - p(node.axis);
        const double diff2 = diff * diff;

        const int near_child = diff <= 0.0 ? node.left : node.right;
        const int far_child  = diff <= 0.0 ? node.right : node.left;

        search_recursive(near_child, q, radius2, result);

        if (diff2 <= radius2)
            search_recursive(far_child, q, radius2, result);
    }
};


class SparseWendlandMap
{
public:
    SparseWendlandMap() = default;

    SparseWendlandMap(const arma::mat& final_pos,
                      const arma::mat& initial_pos,
                      double radius,
                      double regularization = 0.0,
                      double cg_tol = 1e-10,
                      int cg_max_iter = 1000)
    {
        initialize(final_pos, initial_pos, radius, regularization, cg_tol, cg_max_iter);
    }

    void initialize(const arma::mat& final_pos,
                    const arma::mat& initial_pos,
                    double radius,
                    double regularization = 0.0,
                    double cg_tol = 1e-10,
                    int cg_max_iter = 1000)
    {
        centers_ = final_pos;
        values_ = final_pos - initial_pos;

        radius_ = radius;
        regularization_ = regularization;
        dim_ = final_pos.n_cols;

        ell_ = static_cast<int>(dim_ / 2) + 2;

        tree_.build(centers_);

        arma::sp_mat A = assemble_matrix();

        alpha_.set_size(centers_.n_rows, dim_);

        /* #pragma omp parallel for schedule(static)
        for (arma::sword d = 0; d < static_cast<arma::sword>(dim_); ++d)
            alpha_.col(d) = conjugate_gradient(A, values_.col(d), cg_tol, cg_max_iter); */

        bool ok = arma::spsolve(alpha_, A, values_);

        if (!ok)
        {
            std::cerr << "spsolve failed\n";
        }
    }

    arma::rowvec displacement(const arma::rowvec& z) const
    {
        std::vector<arma::uword> nbrs;
        nbrs.reserve(64);

        tree_.radius_search(z, radius_, nbrs);

        arma::rowvec out(dim_, arma::fill::zeros);

        for (arma::uword j : nbrs)
        {
            const double r =
                std::sqrt(arma::accu(arma::square(z - centers_.row(j)))) / radius_;

            out += wendland(r) * alpha_.row(j);
        }

        return out;
    }

    arma::rowvec inverse_map(const arma::rowvec& z) const
    {
        return z - displacement(z);
    }

    arma::mat displacements(const arma::mat& points) const
    {
        arma::mat out(points.n_rows, dim_, arma::fill::zeros);

        for (arma::uword i = 0; i < points.n_rows; ++i)
            out.row(i) = displacement(arma::rowvec(points.row(i)));

        return out;
    }

    arma::mat inverse_maps(const arma::mat& points) const
    {
        arma::mat out(points.n_rows, dim_, arma::fill::zeros);

        for (arma::uword i = 0; i < points.n_rows; ++i)
            out.row(i) = inverse_map(arma::rowvec(points.row(i)));

        return out;
    }

private:
    arma::mat centers_; // z_i^n
    arma::mat values_;  // z_i^n - z_i^0
    arma::mat alpha_;   // RBF coefficients

    double radius_ = 0.0;
    double regularization_ = 0.0;
    arma::uword dim_ = 0;
    int ell_ = 0;

    KDTree tree_;

    double wendland(double r) const
    {
        if (r >= 1.0)
            return 0.0;

        const double s = 1.0 - r;
        return std::pow(s, ell_ + 1) * ((ell_ + 1) * r + 1.0);
    }

    arma::sp_mat assemble_matrix() const
    {
        const arma::uword N = centers_.n_rows;
        const int nthreads = omp_get_max_threads();

        std::vector<std::vector<arma::uword>> rows_t(nthreads);
        std::vector<std::vector<arma::uword>> cols_t(nthreads);
        std::vector<std::vector<double>> vals_t(nthreads);

        #pragma omp parallel
        {
            const int tid = omp_get_thread_num();

            auto& rows = rows_t[tid];
            auto& cols = cols_t[tid];
            auto& vals = vals_t[tid];

            rows.reserve(64 * N / nthreads);
            cols.reserve(64 * N / nthreads);
            vals.reserve(64 * N / nthreads);

            std::vector<arma::uword> nbrs;
            nbrs.reserve(128);

            #pragma omp for schedule(dynamic, 32)
            for (arma::sword ii = 0; ii < static_cast<arma::sword>(N); ++ii)
            {
                const arma::uword i = static_cast<arma::uword>(ii);

                nbrs.clear();

                const arma::rowvec zi = centers_.row(i);
                tree_.radius_search(zi, radius_, nbrs);

                for (arma::uword j : nbrs)
                {
                    const double r =
                        std::sqrt(arma::accu(arma::square(zi - centers_.row(j)))) / radius_;

                    double aij = wendland(r);

                    if (i == j)
                        aij += regularization_;

                    if (aij != 0.0)
                    {
                        rows.push_back(i);
                        cols.push_back(j);
                        vals.push_back(aij);
                    }
                }
            }
        }

        std::size_t nnz = 0;

        for (int t = 0; t < nthreads; ++t)
            nnz += vals_t[t].size();

        arma::umat locations(2, nnz);
        arma::vec values(nnz);

        std::size_t k = 0;

        for (int t = 0; t < nthreads; ++t)
        {
            for (std::size_t q = 0; q < vals_t[t].size(); ++q)
            {
                locations(0, k) = rows_t[t][q];
                locations(1, k) = cols_t[t][q];
                values(k) = vals_t[t][q];
                ++k;
            }
        }

        return arma::sp_mat(locations, values, N, N);
    }

    static arma::vec conjugate_gradient(const arma::sp_mat& A,
                                        const arma::vec& b,
                                        double tol,
                                        int max_iter)
    {
        arma::vec x(b.n_elem, arma::fill::zeros);

        arma::vec r = b - A * x;
        arma::vec p = r;

        double rsold = arma::dot(r, r);
        const double bnorm = std::sqrt(arma::dot(b, b)) + 1e-30;

        for (int iter = 0; iter < max_iter; ++iter)
        {
            arma::vec Ap = A * p;

            const double denom = arma::dot(p, Ap);
            const double alpha = rsold / denom;

            x += alpha * p;
            r -= alpha * Ap;

            const double rsnew = arma::dot(r, r);

            if (std::sqrt(rsnew) / bnorm < tol)
                break;

            p = r + (rsnew / rsold) * p;
            rsold = rsnew;
        }

        return x;
    }
};


}

}