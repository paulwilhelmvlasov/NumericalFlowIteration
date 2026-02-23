#pragma once
#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <memory>
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <fstream>
#include <sstream>

#include <armadillo>

#include <nufi/config.hpp>

namespace nufi
{

template <typename real>
real maxwellian(real u, real v, real w, real vth) noexcept
{
    real c = 1.0 / std::pow(2*M_PI*vth*vth, 3.0/2.0);
    return c*std::exp(-(u*u + v*v + w*w) / (2*vth*vth) );
}

template <typename real>
real maxwellian_2d(real u, real v, real vth) noexcept
{
    real c = 1.0 / (2*M_PI*vth*vth);
    return c*std::exp(-(u*u + v*v) / (2*vth*vth) );
}

template <typename real>
real maxwellian_1d(real u, real vth) noexcept
{
    real c = 1.0 / std::sqrt(2*M_PI*vth*vth);
    return c*std::exp(-(u*u) / (2*vth*vth) );
}


namespace dim3
{

inline size_t idx_base(size_t t, size_t d, size_t ix, size_t iy, size_t iz,
    size_t Nx_ext, size_t Ny_ext, size_t Nz_ext, size_t Nt) {
    size_t spatial_idx = ix + Nx_ext * (iy + Ny_ext * iz);
    size_t Nspace = Nx_ext * Ny_ext * Nz_ext;
    return d * Nspace + spatial_idx + 3 * Nspace * t;
}


template<typename real, size_t order>
void interpolate_fields_aligned(size_t n, std::vector<real>& coeffs, 
                                std::vector<real>& values, 
                                const config_t<real>& conf)
{
    // It is assumed that E and B are precomputed correctly already.
    // Storage of coefficients now via: 
    // index = nt + Nt * (d + dim * (ix + Nx * (iy + Ny * iz)))
    const size_t stride_t = (conf.Nx + order - 1) *
                        (conf.Ny + order - 1) *
                        (conf.Nz + order - 1);

    const size_t dim = 3;
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Ny_ext = conf.Ny + order - 1;
    const size_t Nz_ext = conf.Nz + order - 1;
    const size_t Nspace = Nx_ext * Ny_ext * Nz_ext;

    #pragma omp parallel for
    for(size_t d = 0; d < 3; d++){
        interpolate<real,order>(coeffs.data() + idx_base(n,d,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),
                                                values.data() + d*conf.Nx*conf.Ny*conf.Nz,conf);
    }
}

template<typename real>
void mpi_allgather_field(std::vector<real>& field,
                         const config_t<real>& conf)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const size_t Nspace = conf.Nx * conf.Ny * conf.Nz;
    const size_t dim = field.size() / Nspace;   // 1 for rho, 3 for j

    size_t base = Nspace / size;
    size_t rem  = Nspace % size;

    std::vector<int> counts(size), displs(size);

    size_t offset = 0;
    for(int r = 0; r < size; r++){
        size_t Nloc = base + (r < rem ? 1 : 0);
        counts[r] = dim * Nloc;
        displs[r] = dim * offset;
        offset += Nloc;
    }

    std::vector<real> tmp = field;

    MPI_Allgatherv(tmp.data() + displs[rank],
                   counts[rank],
                   MPI_DOUBLE,
                   field.data(),
                   counts.data(),
                   displs.data(),
                   MPI_DOUBLE,
                   MPI_COMM_WORLD);
}


namespace analysis
{

template<typename real, size_t order>
void do_stats(size_t nt, size_t nx_plot, size_t ny_plot, size_t nz_plot, std::ofstream& stat_file, 
    const std::vector<real>& coeffs_E, const std::vector<real>& coeffs_B, 
    const config_t<double>& conf, bool plot_E_B = false, bool restarted = false, 
    size_t n_full = 0)
{
    // Storage of coefficients now via: 
    // index = nt + Nt * (d + dim * (ix + Nx * (iy + Ny * iz)))
    const size_t stride_t = (conf.Nx + order - 1) *
                        (conf.Ny + order - 1) *
                        (conf.Nz + order - 1);

    const size_t dim = 3;
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Ny_ext = conf.Ny + order - 1;
    const size_t Nz_ext = conf.Nz + order - 1;
    const size_t Nspace = Nx_ext * Ny_ext * Nz_ext;

    double dx_plot = (conf.x_max - conf.x_min)/nx_plot; 
    double dy_plot = (conf.y_max - conf.y_min)/ny_plot; 
    double dz_plot = (conf.z_max - conf.z_min)/nz_plot; 
    double electric_energy = 0;
    double magnetic_energy = 0;

    arma::vec Ex(nx_plot*ny_plot*nz_plot,arma::fill::zeros);
    arma::vec Ey(nx_plot*ny_plot*nz_plot,arma::fill::zeros);
    arma::vec Ez(nx_plot*ny_plot*nz_plot,arma::fill::zeros);

    arma::vec Bx(nx_plot*ny_plot*nz_plot,arma::fill::zeros);
    arma::vec By(nx_plot*ny_plot*nz_plot,arma::fill::zeros);
    arma::vec Bz(nx_plot*ny_plot*nz_plot,arma::fill::zeros);

    double electric_x_energy = 0;
    double electric_y_energy = 0;
    double electric_z_energy = 0;
    double magnetic_x_energy = 0;
    double magnetic_y_energy = 0;
    double magnetic_z_energy = 0;

    double current_time = 0;
    if(restarted){
        current_time = n_full * conf.dt;
    } else {
        current_time = nt * conf.dt;
    }

    #pragma omp parallel for collapse(3)
    for(size_t ix = 0; ix < nx_plot; ix++)
    for(size_t iy = 0; iy < ny_plot; iy++)
    for(size_t iz = 0; iz < nz_plot; iz++){
        double x = (ix+0.5)*dx_plot;
        double y = (iy+0.5)*dy_plot;
        double z = (iz+0.5)*dz_plot;

        size_t index = ix + nx_plot*iy + nx_plot*ny_plot*iz;

        Ex(index) = eval<real,order>(x,y,z,coeffs_E.data() + idx_base(nt,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
        Ey(index) = eval<real,order>(x,y,z,coeffs_E.data() + idx_base(nt,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
        Ez(index) = eval<real,order>(x,y,z,coeffs_E.data() + idx_base(nt,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);

        Bx(index) = eval<real,order>(x,y,z,coeffs_B.data() + idx_base(nt,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
        By(index) = eval<real,order>(x,y,z,coeffs_B.data() + idx_base(nt,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
        Bz(index) = eval<real,order>(x,y,z,coeffs_B.data() + idx_base(nt,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,conf.Nt),conf);
    }

    electric_x_energy = 0.5 * dx_plot*dy_plot*dz_plot * arma::sum(Ex%Ex);
    electric_y_energy = 0.5 * dx_plot*dy_plot*dz_plot * arma::sum(Ey%Ey);
    electric_z_energy = 0.5 * dx_plot*dy_plot*dz_plot * arma::sum(Ez%Ez);

    magnetic_x_energy = 0.5 * dx_plot*dy_plot*dz_plot * arma::sum(Bx%Bx);
    magnetic_y_energy = 0.5 * dx_plot*dy_plot*dz_plot * arma::sum(By%By);
    magnetic_z_energy = 0.5 * dx_plot*dy_plot*dz_plot * arma::sum(Bz%Bz);

    electric_energy = electric_x_energy + electric_y_energy + electric_z_energy;
    magnetic_energy = magnetic_x_energy + magnetic_y_energy + magnetic_z_energy;

    stat_file << current_time << " " << electric_energy << " " << magnetic_energy << " "
        << electric_x_energy << " " << electric_y_energy << " " << electric_z_energy << " "
        << magnetic_x_energy << " " << magnetic_y_energy << " " << magnetic_z_energy << " "
        << std::endl;
    std::cout << current_time << " " << electric_energy << " " << magnetic_energy << std::endl;


    if(plot_E_B){
        std::ofstream Ex_str("Ex_" + std::to_string(current_time) + ".txt");
        std::ofstream Ey_str("Ey_" + std::to_string(current_time) + ".txt");
        std::ofstream Ez_str("Ez_" + std::to_string(current_time) + ".txt");
        std::ofstream Bx_str("Bx_" + std::to_string(current_time) + ".txt");
        std::ofstream By_str("By_" + std::to_string(current_time) + ".txt");
        std::ofstream Bz_str("Bz_" + std::to_string(current_time) + ".txt");

        Ex_str << Ex;
        Ey_str << Ey;
        Ez_str << Ez;
        Bx_str << Bx;
        By_str << By;
        Bz_str << Bz;
    }
}

void compute_j_l2(size_t nt_full, std::ofstream& stat_file, const config_t<double>& conf, 
        const std::vector<double>& j, const std::vector<double>& j_e, 
        const std::vector<double> j_i)
{
    double jx_l2 = 0;
    double jy_l2 = 0;
    double jz_l2 = 0;

    double jx_e_l2 = 0;
    double jy_e_l2 = 0;
    double jz_e_l2 = 0;

    double jx_i_l2 = 0;
    double jy_i_l2 = 0;
    double jz_i_l2 = 0;

    #pragma omp parallel for reduction(+:jx_l2,jy_l2,jz_l2)
    for(size_t l = 0; l < conf.Nx*conf.Ny*conf.Nz; l++){
        jx_l2 += j[l]*j[l];
        jy_l2 += j[l + conf.Nx*conf.Ny*conf.Nz]*j[l + conf.Nx*conf.Ny*conf.Nz];
        jz_l2 += j[l + 2*conf.Nx*conf.Ny*conf.Nz]*j[l + 2*conf.Nx*conf.Ny*conf.Nz];
    }

    #pragma omp parallel for reduction(+:jx_e_l2,jy_e_l2,jz_e_l2)
    for(size_t l = 0; l < conf.Nx*conf.Ny*conf.Nz; l++){
        jx_e_l2 += j_e[l]*j_e[l];
        jy_e_l2 += j_e[l + conf.Nx*conf.Ny*conf.Nz]*j_e[l + conf.Nx*conf.Ny*conf.Nz];
        jz_e_l2 += j_e[l + 2*conf.Nx*conf.Ny*conf.Nz]*j_e[l + 2*conf.Nx*conf.Ny*conf.Nz];
    }

    #pragma omp parallel for reduction(+:jx_i_l2,jy_i_l2,jz_i_l2)
    for(size_t l = 0; l < conf.Nx*conf.Ny*conf.Nz; l++){
        jx_i_l2 += j_i[l]*j_i[l];
        jy_i_l2 += j_i[l + conf.Nx*conf.Ny*conf.Nz]*j_i[l + conf.Nx*conf.Ny*conf.Nz];
        jz_i_l2 += j_i[l + 2*conf.Nx*conf.Ny*conf.Nz]*j_i[l + 2*conf.Nx*conf.Ny*conf.Nz];
    }

    jx_l2 = conf.dx*conf.dy*conf.dz * std::sqrt(jx_l2);
    jy_l2 = conf.dx*conf.dy*conf.dz * std::sqrt(jy_l2);
    jz_l2 = conf.dx*conf.dy*conf.dz * std::sqrt(jz_l2);

    jx_e_l2 = conf.dx*conf.dy*conf.dz * std::sqrt(jx_e_l2);
    jy_e_l2 = conf.dx*conf.dy*conf.dz * std::sqrt(jy_e_l2);
    jz_e_l2 = conf.dx*conf.dy*conf.dz * std::sqrt(jz_e_l2);

    jx_i_l2 = conf.dx*conf.dy*conf.dz * std::sqrt(jx_i_l2);
    jy_i_l2 = conf.dx*conf.dy*conf.dz * std::sqrt(jy_i_l2);
    jz_i_l2 = conf.dx*conf.dy*conf.dz * std::sqrt(jz_i_l2);

    double j_tot_l2 = std::sqrt(jx_l2*jx_l2 + jy_l2*jy_l2 + jz_l2*jz_l2);
    double j_e_tot_l2 = std::sqrt(jx_e_l2*jx_e_l2 + jy_e_l2*jy_e_l2 + jz_e_l2*jz_e_l2);
    double j_i_tot_l2 = std::sqrt(jx_i_l2*jx_i_l2 + jy_i_l2*jy_i_l2 + jz_i_l2*jz_i_l2);

    stat_file << nt_full*conf.dt << " " << j_tot_l2 << " " << j_e_tot_l2 << " " << j_i_tot_l2 
                << " " << jx_l2 << " " << jy_l2 << " " << jz_l2 
                << jx_e_l2 << " " << jy_e_l2 << " " << jz_e_l2 
                << jx_e_l2 << " " << jy_e_l2 << " " << jz_e_l2 << std::endl;
}

template<typename real, size_t order>
void plot_full_f_3x3v_parallelized(size_t nx_plot, size_t ny_plot, size_t nz_plot, 
    size_t nu_plot, size_t nv_plot, size_t nw_plot, double xmin_p, double xmax_p, double ymin_p, 
    double ymax_p, double zmin_p, double zmax_p, double umin_p, double umax_p, 
    double vmin_p, double vmax_p, double wmin_p, double wmax_p,  
    const std::function<double(double,double,double,double,double,double)>& eval_f,
    std::ofstream& file)
{
    double dx_plot = (xmax_p - xmin_p) / nx_plot;
    double dy_plot = (ymax_p - ymin_p) / ny_plot;
    double dz_plot = (zmax_p - zmin_p) / nz_plot; 
    double du_plot = (umax_p - umin_p) / nu_plot;
    double dv_plot = (vmax_p - vmin_p) / nv_plot;
    double dw_plot = (wmax_p - wmin_p) / nw_plot;

    arma::mat matrix(nx_plot*ny_plot*nz_plot,nu_plot*nv_plot*nw_plot);

    #pragma omp parallel for collapse(6) 
    for(size_t ix = 0; ix < nx_plot; ix++)
    for(size_t iy = 0; iy < ny_plot; iy++)
    for(size_t iz = 0; iz < nz_plot; iz++)
    for(size_t iu = 0; iu < nu_plot; iu++)
    for(size_t iv = 0; iv < nv_plot; iv++)
    for(size_t iw = 0; iw < nw_plot; iw++){
        double x = xmin_p + (ix+0.5) * dx_plot;
        double y = ymin_p + (iy+0.5) * dy_plot;
        double z = zmin_p + (iz+0.5) * dz_plot;

        double u = umin_p + (iu+0.5) * du_plot;
        double v = vmin_p + (iv+0.5) * dv_plot;
        double w = wmin_p + (iw+0.5) * dw_plot;

        size_t index_0 = ix + nx_plot*(iy + ny_plot*iz);
        size_t index_1 = iu + nu_plot*(iv + nv_plot*iw);

        matrix(index_0, index_1) = eval_f(x, y, z, u, v, w);
    }
    
    file << matrix;
}



template<typename real, size_t order>
void write_coeffs_nufi_pc(size_t n, const std::vector<real>& coeffs_E, 
    const std::vector<real>& coeffs_B, const config_t<real>& conf, 
    std::ofstream& coeff_file_E, std::ofstream& coeff_file_B)
{
    // Storage of coefficients now via: 
    // index = nt + Nt * (d + dim * (ix + Nx * (iy + Ny * iz)))
    const size_t stride_t = (conf.Nx + order - 1) *
                        (conf.Ny + order - 1) *
                        (conf.Nz + order - 1);

    const size_t dim = 3;
    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Ny_ext = conf.Ny + order - 1;
    const size_t Nz_ext = conf.Nz + order - 1;
    const size_t Nspace = Nx_ext * Ny_ext * Nz_ext;

    #pragma omp parallel
    {
        #pragma omp sections
        {    
            #pragma omp section
            for(size_t ix = 0; ix < conf.Nx; ix++)
            for(size_t iy = 0; iy < conf.Ny; iy++)
            for(size_t iz = 0; iz < conf.Nz; iz++)
            {
                coeff_file_E << std::scientific << std::setprecision(16) << coeffs_E[idx_base(n,0,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] << " "
                            << coeffs_E[idx_base(n,1,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] << " "
                            << coeffs_E[idx_base(n,2,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] << " "
                            << std::endl;
            }
            
            #pragma omp section
            for(size_t ix = 0; ix < conf.Nx; ix++)
            for(size_t iy = 0; iy < conf.Ny; iy++)
            for(size_t iz = 0; iz < conf.Nz; iz++)
            {
                coeff_file_B << std::scientific << std::setprecision(16) << coeffs_B[idx_base(n,0,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] << " "
                            << coeffs_B[idx_base(n,1,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] << " "
                            << coeffs_B[idx_base(n,2,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] << " "
                            << std::endl;
            }
        }
    }

}    


template <typename real, size_t order>
void read_in_coeff_and_plot_aligned_nufi_pc(const config_t<double>& conf)
{
    // This is still somehow broken. The "multiply trick" from the other
    // Lie Splitting doesn't work here (the same way). Check this if you have time...

    conf.print_config(std::cout);

    std::ifstream coeff_str_E("../coeff_E.txt");
    std::ifstream coeff_str_B("../coeff_B.txt");

    size_t stride_t = (conf.Nx + order - 1) *
                        (conf.Ny + order - 1) *
                        (conf.Nz + order - 1);

    const size_t Nx_ext = conf.Nx + order - 1;
    const size_t Ny_ext = conf.Ny + order - 1;
    const size_t Nz_ext = conf.Nz + order - 1;
    const size_t Nspace = Nx_ext * Ny_ext * Nz_ext;

    std::vector<double> coeffs_E(3 * (conf.Nt + 1) * stride_t, 0);
    std::vector<double> coeffs_B(3 * (conf.Nt + 1) * stride_t, 0);

    std::cout << "Read in coeffs." << std::endl;

    size_t end_n = 50*50;

    for(size_t n = 0; n <= end_n; n++){
        for(size_t ix = 0; ix < conf.Nx; ix++)
        for(size_t iy = 0; iy < conf.Ny; iy++)
        for(size_t iz = 0; iz < conf.Nz; iz++)
        {
            coeff_str_E >> coeffs_E[idx_base(n,0,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] 
                        >> coeffs_E[idx_base(n,1,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] 
                        >> coeffs_E[idx_base(n,2,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)];
            
            coeffs_E[idx_base(n,0,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] *= 2e3;
            coeffs_E[idx_base(n,1,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] *= 2e3;
            coeffs_E[idx_base(n,2,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] *= 2e3;
        }
            

        for(size_t ix = 0; ix < conf.Nx; ix++)
        for(size_t iy = 0; iy < conf.Ny; iy++)
        for(size_t iz = 0; iz < conf.Nz; iz++)
        {
            coeff_str_B >> coeffs_B[idx_base(n,0,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)]
                         >> coeffs_B[idx_base(n,1,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)]
                         >> coeffs_B[idx_base(n,2,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)];

            coeffs_B[idx_base(n,0,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] *= 2e3;
            coeffs_B[idx_base(n,1,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] *= 2e3;
            coeffs_B[idx_base(n,2,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,conf.Nt)] *= 2e3;
        }
    }

    std::cout << "Analyze data." << std::endl;

    size_t nt_plot = end_n;

    size_t nx_plot = 64;
    size_t ny_plot = 1;
    size_t nz_plot = 1;
    size_t nu_plot = nx_plot;
    size_t nv_plot = nx_plot;
    size_t nw_plot = 1;

    double dx_plot = conf.Lx/nx_plot;
    double dy_plot = conf.Ly/ny_plot;
    double dz_plot = conf.Lz/nz_plot;
    double du_plot = (conf.u_max - conf.u_min)/nu_plot;
    double dv_plot = (conf.v_max - conf.v_min)/nv_plot;
    double dw_plot = (conf.w_max - conf.w_min)/nw_plot;

    /* 
    * Do analysis or something...
    */
}


}

}

namespace BSpline
{

struct BSpline1D {
  int p;                    // degree
  std::vector<double> t;    // knot vector, nondecreasing
  int nBasis;               // number of basis functions

  BSpline1D() = default;
  BSpline1D(int degree, std::vector<double> knots)
      : p(degree), t(std::move(knots)) {
    if (p < 0) throw std::invalid_argument("Degree must be >= 0.");
    if ((int)t.size() < 2 * (p + 1))
      throw std::invalid_argument("Knot vector too short for degree.");
    for (size_t i = 1; i < t.size(); ++i)
      if (t[i] < t[i - 1])
        throw std::invalid_argument("Knot vector must be nondecreasing.");
    // Standard relation: m+1 knots, n basis => m = n + p + 1
    // => n = (m+1) - p - 1 = (#knots) - p - 1
    nBasis = (int)t.size() - p - 1;
    if (nBasis <= 0) throw std::invalid_argument("No basis functions.");
  }

  // Find span i such that t[i] <= x < t[i+1], with special handling at right end.
  int findSpan(double x) const {
    const int n = nBasis - 1;         // last basis index
    const int m = (int)t.size() - 1;  // last knot index

    // Clamp x into [t[p], t[m-p]] for safety (open knot vector domain).
    const double left = t[p];
    const double right = t[m - p];
    if (x <= left) return p;
    if (x >= right) return n; // rightmost span

    // Binary search in [p, n]
    int low = p, high = n + 1;
    int mid = (low + high) / 2;
    while (!(x >= t[mid] && x < t[mid + 1])) {
      if (x < t[mid]) high = mid;
      else low = mid;
      mid = (low + high) / 2;
    }
    return mid;
  }

  // Evaluate all nonzero basis functions at x.
  // Returns arrays (idx, N) such that basis indices are idx = span-p ... span
  // and N[k] = B_{idx+k}^{p}(x).
void basisFuns(double x, int &firstIdx, std::vector<double> &N) const {
  const int m = (int)t.size() - 1;
  const double a = t[p];
  const double b = t[m - p];

  // Robust endpoint handling for open/clamped knots
  const double eps = 64.0 * std::numeric_limits<double>::epsilon() * (std::abs(b) + std::abs(a) + 1.0);

  if (x <= a + eps) {
    firstIdx = 0;
    N.assign(p + 1, 0.0);
    N[0] = 1.0;
    return;
  }
  if (x >= b - eps) {
    firstIdx = nBasis - (p + 1);
    N.assign(p + 1, 0.0);
    N[p] = 1.0;
    return;
  }

  const int span = findSpan(x);
  firstIdx = span - p;
  N.assign(p + 1, 0.0);

  std::vector<double> left(p + 1), right(p + 1);
  N[0] = 1.0;

  for (int j = 1; j <= p; ++j) {
    left[j]  = x - t[span + 1 - j];
    right[j] = t[span + j] - x;
    double saved = 0.0;
    for (int r = 0; r < j; ++r) {
      const double denom = right[r + 1] + left[j - r];
      double temp = 0.0;
      if (denom != 0.0) temp = N[r] / denom;
      const double val = temp * right[r + 1];
      N[r] = saved + val;
      saved = temp * left[j - r];
    }
    N[j] = saved;
  }
}

  // Convenience: evaluate the full basis vector of length nBasis at x (sparse fill).
  arma::rowvec evalAll(double x) const {
    arma::rowvec v(nBasis, arma::fill::zeros);
    int first = 0;
    std::vector<double> N;
    basisFuns(x, first, N);
    for (int k = 0; k <= p; ++k) {
      const int idx = first + k;
      if (0 <= idx && idx < nBasis) v((arma::uword)idx) = N[(size_t)k];
    }
    return v;
  }
};

// Open/clamped uniform knot vector on [a,b] with nBasis basis functions, degree p.
// Creates nBasis = nCtrl basis functions, knots length = nBasis + p + 1.
// Interior knots are uniform (or as uniform as possible).
static std::vector<double> makeOpenUniformKnots(int nBasis, int p, double a, double b) {
  if (nBasis <= p) throw std::invalid_argument("Need nBasis > p for open uniform knots.");
  if (!(b > a)) throw std::invalid_argument("Require b > a.");

  const int mPlus1 = nBasis + p + 1;            // number of knots
  std::vector<double> t(mPlus1);

  // First p+1 are a, last p+1 are b
  for (int i = 0; i <= p; ++i) t[i] = a;
  for (int i = mPlus1 - p - 1; i < mPlus1; ++i) t[i] = b;

  // Number of interior knots (not counting endpoint repeats)
  // Indices: p+1 ... mPlus1-p-2 inclusive
  const int nInterior = mPlus1 - 2 * (p + 1);
  if (nInterior > 0) {
    // Uniform interior knots in (a,b): a + k * (b-a)/(nInterior+1), k=1..nInterior
    for (int k = 1; k <= nInterior; ++k) {
      t[p + k] = a + (b - a) * (double)k / (double)(nInterior + 1);
    }
  }
  return t;
}

// Build 1D collocation matrix A where A(i,j) = B_j(x_i), size nPts x nBasis.
// For interpolation we typically set nPts == nBasis and choose x_i as data nodes.
static arma::mat buildCollocation(const BSpline1D &spl, const arma::vec &x) {
  arma::mat A(x.n_elem, (arma::uword)spl.nBasis, arma::fill::zeros);
  for (arma::uword i = 0; i < x.n_elem; ++i) {
    arma::rowvec r = spl.evalAll(x[i]);
    A.row(i) = r;
  }
  return A;
}

struct TensorSpline2D {
  BSpline1D sx, sy;
  arma::mat C;   // coefficients, size nxBasis x nyBasis

  // Fit interpolant on tensor grid (x nodes, y nodes), with F size nx x ny.
  // Requirement for interpolation: x.n_elem == sx.nBasis and y.n_elem == sy.nBasis.
  static TensorSpline2D interpolate(const arma::vec &x, const arma::vec &y,
                                    const arma::mat &F,
                                    int px, int py,
                                    double ax, double bx,
                                    double ay, double by) {
    const int nxBasis = (int)x.n_elem; // choose basis count = number of nodes
    const int nyBasis = (int)y.n_elem;

    if ((int)F.n_rows != nxBasis || (int)F.n_cols != nyBasis)
      throw std::invalid_argument("F must have size (x.n_elem, y.n_elem).");

    TensorSpline2D s;
    s.sx = BSpline1D(px, makeOpenUniformKnots(nxBasis, px, ax, bx));
    s.sy = BSpline1D(py, makeOpenUniformKnots(nyBasis, py, ay, by));

    // Collocation matrices
    arma::mat Ax = buildCollocation(s.sx, x); // nx x nx
    std::cout << "cond(Ax) = " << arma::cond(Ax) << std::endl;
    arma::mat Ay = buildCollocation(s.sy, y); // ny x ny
    std::cout << "cond(Ay) = " << arma::cond(Ay) << std::endl;

    // Solve Ax * G = F  (G is nx x ny)
    arma::mat G = arma::solve(Ax, F); // uses LU/QR depending on properties

    // Solve Ay * (C^T) = (G^T)  => C^T = solve(Ay, G^T)
    arma::mat Ct = arma::solve(Ay, G.t());
    s.C = Ct.t();
    return s;
  }

  // Evaluate s(xq, yq)
  double eval(double xq, double yq) const {
    // Evaluate sparse basis vectors via local support
    int ix0 = 0, iy0 = 0;
    std::vector<double> Nx, Ny;
    sx.basisFuns(xq, ix0, Nx);
    sy.basisFuns(yq, iy0, Ny);

    double val = 0.0;
    for (int a = 0; a <= sx.p; ++a) {
      const int ia = ix0 + a;
      if (ia < 0 || ia >= sx.nBasis) continue;
      const double bx = Nx[(size_t)a];
      if (bx == 0.0) continue;
      for (int b = 0; b <= sy.p; ++b) {
        const int ib = iy0 + b;
        if (ib < 0 || ib >= sy.nBasis) continue;
        val += C((arma::uword)ia, (arma::uword)ib) * bx * Ny[(size_t)b];
      }
    }
    return val;
  }

  // Vectorized evaluation on a grid (xq size M, yq size N) -> MxN matrix
  arma::mat evalGrid(const arma::vec &xq, const arma::vec &yq) const {
    arma::mat S(xq.n_elem, yq.n_elem, arma::fill::zeros);
    for (arma::uword i = 0; i < xq.n_elem; ++i) {
      for (arma::uword j = 0; j < yq.n_elem; ++j) {
        S(i, j) = eval(xq[i], yq[j]);
      }
    }
    return S;
  }
};

static arma::vec grevillePoints(const BSpline1D& spl) {
  // Greville abscissae: g_i = (t_{i+1}+...+t_{i+p})/p, i=0..nBasis-1
  // For p=0, the "Greville" points can be taken as midpoints of knot intervals.
  arma::vec g(spl.nBasis);

  if (spl.p == 0) {
    for (int i = 0; i < spl.nBasis; ++i) {
      g[i] = 0.5 * (spl.t[i] + spl.t[i+1]);
    }
    return g;
  }

  for (int i = 0; i < spl.nBasis; ++i) {
    double s = 0.0;
    for (int k = 1; k <= spl.p; ++k) s += spl.t[i + k];
    g[i] = s / (double)spl.p;
  }
  return g;
}    

}

}