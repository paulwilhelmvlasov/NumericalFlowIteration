/*
 * Kernel-CMM NuFI 1x1v prototype with tensor-product periodic Wendland kernel.
 *
 * Changes relative to the previous periodic prototype:
 *   - one restart particle per controlled sample point;
 *   - no copied periodic particles;
 *   - no angular output representation;
 *   - tensor-product Wendland kernel:
 *
 *       K((x,u),(y,w)) =
 *           b( d_per(x,y) / sigma_x ) * b( |u-w| / sigma_u )
 *
 *   - periodic x-displacement is stored in [-Lx/2,Lx/2];
 *   - the sparse interpolation system is solved directly with arma::spsolve.
 */

#include <cmath>
#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>

#include <armadillo>

#include <nufi/config.hpp>
#include <nufi/fields.hpp>
#include <nufi/poisson.hpp>
#include <nufi/rho.hpp>
#include <nufi/stopwatch.hpp>
#include <nufi/restart.hpp>

namespace nufi
{
namespace restart
{

class KDTree1x1v
{
public:
    KDTree1x1v() = default;

    KDTree1x1v(const arma::mat& points)
    {
        build(points);
    }

    void build(const arma::mat& points)
    {
        points_ = &points;

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
    int root_ = -1;

    std::vector<arma::uword> indices_;
    std::vector<Node> nodes_;

    int build_recursive(std::size_t begin, std::size_t end, arma::uword depth)
    {
        if (begin >= end)
            return -1;

        const arma::uword axis = depth % 2;
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

        const double dx = q(0) - (*points_)(node.index, 0);
        const double du = q(1) - (*points_)(node.index, 1);
        const double d2 = dx*dx + du*du;

        if (d2 <= radius2)
            result.push_back(node.index);

        const double diff = q(node.axis) - (*points_)(node.index, node.axis);
        const double diff2 = diff * diff;

        const int near_child = diff <= 0.0 ? node.left : node.right;
        const int far_child  = diff <= 0.0 ? node.right : node.left;

        search_recursive(near_child, q, radius2, result);

        if (diff2 <= radius2)
            search_recursive(far_child, q, radius2, result);
    }
};


class PeriodicTensorWendlandMap1x1v
{
public:
    PeriodicTensorWendlandMap1x1v() = default;

    PeriodicTensorWendlandMap1x1v(const arma::mat& final_pos,
                                  const arma::mat& initial_pos,
                                  double sigma_x,
                                  double sigma_u,
                                  double x_min,
                                  double x_max,
                                  double regularization = 0.0)
    {
        initialize(final_pos,
                   initial_pos,
                   sigma_x,
                   sigma_u,
                   x_min,
                   x_max,
                   regularization);
    }

    void initialize(const arma::mat& final_pos,
                    const arma::mat& initial_pos,
                    double sigma_x,
                    double sigma_u,
                    double x_min,
                    double x_max,
                    double regularization = 0.0)
    {
        centers_ = final_pos;

        sigma_x_ = sigma_x;
        sigma_u_ = sigma_u;
        search_radius_ = std::sqrt(sigma_x_*sigma_x_ + sigma_u_*sigma_u_);

        regularization_ = regularization;

        x_min_ = x_min;
        x_max_ = x_max;
        Lx_ = x_max_ - x_min_;
        Lx_inv_ = 1.0 / Lx_;

        values_.set_size(final_pos.n_rows, 2);

        for (arma::uword i = 0; i < final_pos.n_rows; ++i)
        {
            values_(i, 0) = periodic_signed_dx(final_pos(i, 0), initial_pos(i, 0));
            values_(i, 1) = final_pos(i, 1) - initial_pos(i, 1);
        }

        tree_.build(centers_);

        arma::sp_mat A = assemble_matrix();

        bool ok = arma::spsolve(alpha_, A, values_);

        if (!ok)
            std::cerr << "Warning: arma::spsolve failed in PeriodicTensorWendlandMap1x1v.\n";
    }

    arma::rowvec displacement(const arma::rowvec& z) const
    {
        std::vector<arma::uword> nbrs;
        nbrs.reserve(128);

        radius_search_periodic(z, nbrs);

        arma::rowvec out(2, arma::fill::zeros);

        for (arma::uword j : nbrs)
            out += kernel_point_row(z, j) * alpha_.row(j);

        return out;
    }

    arma::rowvec inverse_map(const arma::rowvec& z) const
    {
        arma::rowvec g = displacement(z);

        arma::rowvec z0(2);
        z0(0) = wrap_x(z(0) - g(0));
        z0(1) = z(1) - g(1);

        return z0;
    }

private:
    arma::mat centers_; // z_i^n
    arma::mat values_;  // periodic displacement z_i^n - z_i^0
    arma::mat alpha_;   // RBF coefficients

    double sigma_x_ = 0.0;
    double sigma_u_ = 0.0;
    double search_radius_ = 0.0;
    double regularization_ = 0.0;

    double x_min_ = 0.0;
    double x_max_ = 0.0;
    double Lx_ = 0.0;
    double Lx_inv_ = 0.0;

    KDTree1x1v tree_;

    static double wendland(double r)
    {
        if (r >= 1.0)
            return 0.0;

        // 1D Wendland b^W_{1,2}; this matches the tensor-product
        // kernels used in the earlier RBF particle implementation.
        const double s = 1.0 - r;
        return s*s*s*s*s * (8.0*r*r + 5.0*r + 1.0);
    }

    double wrap_x(double x) const
    {
        return x_min_ + (x - x_min_)
             - Lx_ * std::floor((x - x_min_) * Lx_inv_);
    }

    double periodic_signed_dx(double x, double y) const
    {
        double dx = x - y;
        dx -= Lx_ * std::round(dx * Lx_inv_);
        return dx;
    }

    double periodic_abs_dx(double x, double y) const
    {
        return std::abs(periodic_signed_dx(x, y));
    }

    double kernel_rows(arma::uword i, arma::uword j) const
    {
        const double dx = periodic_abs_dx(centers_(i, 0), centers_(j, 0));
        const double du = std::abs(centers_(i, 1) - centers_(j, 1));

        return wendland(dx / sigma_x_) * wendland(du / sigma_u_);
    }

    double kernel_point_row(const arma::rowvec& z, arma::uword j) const
    {
        const double dx = periodic_abs_dx(z(0), centers_(j, 0));
        const double du = std::abs(z(1) - centers_(j, 1));

        return wendland(dx / sigma_x_) * wendland(du / sigma_u_);
    }

    void radius_search_periodic(const arma::rowvec& z,
                                std::vector<arma::uword>& nbrs) const
    {
        nbrs.clear();

        arma::rowvec q = z;

        tree_.radius_search(q, search_radius_, nbrs);

        q(0) = z(0) - Lx_;
        tree_.radius_search(q, search_radius_, nbrs);

        q(0) = z(0) + Lx_;
        tree_.radius_search(q, search_radius_, nbrs);

        std::sort(nbrs.begin(), nbrs.end());
        nbrs.erase(std::unique(nbrs.begin(), nbrs.end()), nbrs.end());
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
                const arma::rowvec zi = centers_.row(i);

                radius_search_periodic(zi, nbrs);

                for (arma::uword j : nbrs)
                {
                    double aij = kernel_rows(i, j);

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
};

}
}


namespace nufi
{
namespace dim1
{

template <typename real>
real f0(real x, real u) noexcept
{
    const real alpha = 0.01;
    const real k = 0.5;

    return 1.0 / std::sqrt(2.0*M_PI)
         * std::exp(-0.5*u*u)
         * (1.0 + alpha*std::cos(k*x));
}

config_t<double> conf(64, 128, 500, 0.1, 0, 4*M_PI, -6, 6, &f0);

size_t restart_counter = 0;

std::vector<std::unique_ptr<restart::PeriodicTensorWendlandMap1x1v>> char_maps;

double wrap_x(double x) noexcept
{
    return conf.x_min + (x - conf.x_min)
         - conf.Lx * std::floor((x - conf.x_min) * conf.Lx_inv);
}

double clamp_u(double u) noexcept
{
    if (u < conf.u_min) return conf.u_min;
    if (u > conf.u_max) return conf.u_max;
    return u;
}

double eval_f_cmm_kernel(double x, double u)
{
    for (size_t r = restart_counter; r > 0; --r)
    {
        x = wrap_x(x);
        u = clamp_u(u);

        arma::rowvec z(2);
        z(0) = x;
        z(1) = u;

        arma::rowvec z0 = char_maps[r]->inverse_map(z);

        x = z0(0);
        u = z0(1);
    }

    x = wrap_x(x);
    u = clamp_u(u);

    return f0(x, u);
}

template <size_t order>
void cmm_nufi_kernel()
{
    using std::abs;
    using std::max;

    const size_t Nx = 64;
    const size_t Nu = 64;
    const double dt = 0.1;
    const size_t Nt = static_cast<size_t>(50.0 / dt);

    const double x_min = 0.0;
    const double x_max = 4.0*M_PI;
    const double u_min = -8.0;
    const double u_max =  8.0;

    conf = config_t<double>(Nx, Nu, Nt, dt, x_min, x_max, u_min, u_max, &f0);
    const size_t stride_t = conf.Nx + order - 1;

    const size_t nx_r = Nx;
    const size_t nu_r = Nu;
    const size_t nt_restart = 50;

    const double dx_r = conf.Lx / nx_r;
    const double du_r = (conf.u_max - conf.u_min) / nu_r;

    // Tensor-product Wendland scales.
    // These are absolute scales, not multiples of h. For weak Landau damping
    // this mirrors the parameter regime used in the previous RBF particle work.
    const double kernel_sigma_x = 3.0;
    const double kernel_sigma_u = 1.0;
    const double kernel_regularization = 1e-12;

    char_maps.clear();
    char_maps.resize(Nt / nt_restart + 2);
    restart_counter = 0;

    std::unique_ptr<double[]> coeffs_restart
    {
        new double[(nt_restart + 1) * stride_t]{}
    };

    std::unique_ptr<double, decltype(std::free)*> rho
    {
        reinterpret_cast<double*>(std::aligned_alloc(64, sizeof(double)*conf.Nx)),
        std::free
    };

    if (rho == nullptr)
        throw std::bad_alloc{};

    poisson<double> poiss(conf);

    std::ofstream stat_file("stats.txt");
    std::ofstream stat_full_file("stats_full.txt");

    double total_time = 0.0;
    size_t nt_r_curr = 0;

    for (size_t n = 0; n <= Nt; ++n)
    {
        nufi::stopwatch<double> timer;

        std::cout << "start of time step " << n
                  << " restart-local step " << nt_r_curr << std::endl;

        #pragma omp parallel for
        for (size_t i = 0; i < conf.Nx; ++i)
        {
            rho.get()[i] =
                periodic::eval_rho<double, order>
                (
                    nt_r_curr,
                    i,
                    coeffs_restart.get(),
                    conf
                );
        }

        const double elec_energy = poiss.solve(rho.get());

        periodic::interpolate<double, order>
        (
            coeffs_restart.get() + nt_r_curr*stride_t,
            rho.get(),
            conf
        );

        const double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;

        const double t = n * conf.dt;

        double Emax = 0.0;
        const size_t plot_n_x = 128;
        const double dx_plot = conf.Lx / plot_n_x;

        for (size_t i = 0; i <= plot_n_x; ++i)
        {
            const double x = conf.x_min + i*dx_plot;

            const double E =
                periodic::eval<double, order, 1>
                (
                    x,
                    coeffs_restart.get() + nt_r_curr*stride_t,
                    conf
                );

            Emax = max(Emax, abs(E));
        }

        stat_file << std::setw(15) << t
                  << std::setw(15) << std::setprecision(5)
                  << std::scientific << Emax
                  << " " << elec_energy << std::endl;

        std::cout << std::setw(15) << t
                  << std::setw(15) << std::setprecision(5)
                  << std::scientific << Emax
                  << " Comp-time: " << timer_elapsed
                  << " Total comp time s.f.: " << total_time
                  << std::endl;

        if (n % 80 == 0)
        {
            const size_t plot_n_u = plot_n_x;
            const double du_plot = (conf.u_max - conf.u_min) / plot_n_u;

            double kinetic_energy = 0.0;
            double entropy = 0.0;
            double l1_norm = 0.0;
            double l2_norm = 0.0;

            #pragma omp parallel for reduction(+:kinetic_energy,entropy,l1_norm,l2_norm)
            for (size_t i = 0; i < plot_n_x; ++i)
            {
                for (size_t j = 0; j < plot_n_u; ++j)
                {
                    const double x = conf.x_min + i*dx_plot;
                    const double u = conf.u_min + j*du_plot;

                    const double f =
                        periodic::eval_f<double, order>
                        (
                            nt_r_curr,
                            x,
                            u,
                            coeffs_restart.get(),
                            conf
                        );

                    kinetic_energy += u*u*f;

                    if (f > 0.0)
                        entropy -= f*std::log(f);

                    l1_norm += f;
                    l2_norm += f*f;
                }
            }

            const double weight = dx_plot * du_plot;

            kinetic_energy *= weight;
            entropy *= weight;
            l1_norm *= weight;
            l2_norm *= weight;

            const double total_energy = kinetic_energy + elec_energy;

            stat_full_file << std::setprecision(16)
                           << t              << "; "
                           << l1_norm        << "; "
                           << l2_norm        << "; "
                           << elec_energy    << "; "
                           << kinetic_energy << "; "
                           << total_energy   << "; "
                           << entropy        << ";"
                           << std::endl;
        }

        if (nt_r_curr == nt_restart)
        {
            std::cout << "Restart" << std::endl;
            nufi::stopwatch<double> timer_restart;

            const size_t n_total = nx_r * (nu_r + 1);

            arma::mat final_pos(n_total, 2, arma::fill::zeros);
            arma::mat initial_pos(n_total, 2, arma::fill::zeros);

            #pragma omp parallel for
            for (size_t i = 0; i < nx_r; ++i)
            {
                for (size_t j = 0; j <= nu_r; ++j)
                {
                    const double x_final = conf.x_min + i*dx_r;
                    const double u_final = conf.u_min + j*du_r;

                    double x_initial = x_final;
                    double u_initial = u_final;

                    periodic::eval_char_map<double, order>
                    (
                        nt_r_curr,
                        x_initial,
                        u_initial,
                        coeffs_restart.get(),
                        conf
                    );

                    x_initial = wrap_x(x_initial);

                    const size_t row = i*(nu_r + 1) + j;

                    final_pos(row, 0) = x_final;
                    final_pos(row, 1) = u_final;

                    initial_pos(row, 0) = x_initial;
                    initial_pos(row, 1) = u_initial;
                }
            }

            nufi::stopwatch<double> timer_kernel;

            char_maps[restart_counter + 1] =
                std::make_unique<restart::PeriodicTensorWendlandMap1x1v>
                (
                    final_pos,
                    initial_pos,
                    kernel_sigma_x,
                    kernel_sigma_u,
                    x_min,
                    x_max,
                    kernel_regularization
                );

            const double kernel_time = timer_kernel.elapsed();
            std::cout << "Kernel map construction took: "
                      << kernel_time << std::endl;

            conf = config_t<double>
            (
                Nx,
                Nu,
                Nt,
                dt,
                x_min,
                x_max,
                u_min,
                u_max,
                &eval_f_cmm_kernel
            );

            #pragma omp parallel for
            for (size_t i = 0; i < stride_t; ++i)
                coeffs_restart.get()[i] =
                    coeffs_restart.get()[nt_r_curr*stride_t + i];

            std::cout << n << " " << nt_r_curr
                      << " restart finished" << std::endl;

            nt_r_curr = 1;
            ++restart_counter;

            const double restart_time = timer_restart.elapsed();
            total_time += restart_time;

            std::cout << "Restart took: " << restart_time
                      << ". Total comp time s.f.: "
                      << total_time << std::endl;
        }
        else
        {
            ++nt_r_curr;
        }
    }

    std::cout << "Total time: " << total_time << std::endl;
}

}
}


int main()
{
    nufi::dim1::cmm_nufi_kernel<4>();
    return 0;
}
