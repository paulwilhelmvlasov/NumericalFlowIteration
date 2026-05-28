// sparse_wendland_map.cpp
//
// Compile with:
//   g++ -O3 -std=c++17 sparse_wendland_map.cpp -o sparse_wendland_map -larmadillo
//
// Data convention:
//   final_pos(i, :)   = z_i^n
//   initial_pos(i, :) = z_i^0
//
// The interpolated quantity is
//   g(z) = z - Phi^{-1}(z)
//
// At particles:
//   g(z_i^n) = z_i^n - z_i^0
//
// Then:
//   Phi^{-1}(z) ~= z - g_h(z)

#include <armadillo>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <omp.h>

#include <nufi/stopwatch.hpp>

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


// Minimal test.
int main()
{
    nufi::stopwatch<double> timer;
    constexpr arma::uword N = 5000;
    constexpr arma::uword dim = 3; // e.g. 1x2v: z = (x, v1, v2)

    arma::mat z0(N, dim, arma::fill::randu);
    arma::mat zn = z0;

    #pragma omp parallel for
    for (arma::uword i = 0; i < N; ++i)
    {
        const double x  = z0(i, 0);
        const double v1 = z0(i, 1);
        const double v2 = z0(i, 2);

        zn(i, 0) += 0.1 * std::sin(2.0 * arma::datum::pi * x);
        zn(i, 1) += 0.2 * std::cos(2.0 * arma::datum::pi * v1);
        zn(i, 2) += 0.3 * std::sin(2.0 * arma::datum::pi * v2);
    }

    const double radius = 0.5;
    const double reg = 1e-12;

    SparseWendlandMap map(zn, z0, radius, reg);

    std::cout << "Assemble took: " << timer.elapsed() << std::endl;
    timer.reset();
    
    arma::rowvec z = zn.row(10); 
    arma::rowvec z0_approx = map.inverse_map(z);

    std::cout << "Eval took: " << timer.elapsed() << std::endl;

    std::cout << "z              = " << z;
    std::cout << "Phi^{-1}(z)    = " << z0_approx;
    std::cout << "exact z0       = " << z0.row(10);
    std::cout << "error          = " << arma::norm(z0_approx - z0.row(10), 2) << "\n";



    return 0;
}