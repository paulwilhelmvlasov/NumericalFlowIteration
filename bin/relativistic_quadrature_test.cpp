/*
 * Offline momentum-quadrature convergence study using saved potential coefficients.
 * Reconstructs characteristics through the full field history without CMM maps.
 *
 * Usage:
 *   ./quadrature <time_step> <Nx> <Nu_e> <Nu_i> <rule> [nx_test] [coeffs_file]
 *
 * Rules: midpoint, simpson, boole, gauss4, gauss8
 */

#include <cmath>
#include <memory>
#include <vector>
#include <algorithm>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <stdexcept>

#include <armadillo>

#include <nufi/config.hpp>
#include <nufi/random.hpp>
#include <nufi/fields.hpp>
#include <nufi/poisson.hpp>
#include <nufi/rho.hpp>
#include <nufi/stopwatch.hpp>


namespace nufi
{
namespace dim1
{
namespace periodic
{
namespace relativistic
{

template <typename real>
real maxwell_juttner_1d(real p, real m, real c, real theta) noexcept
{
    real s = p*p/(m*m*c*c);
    real gamma = std::sqrt(1 + s);
    if (theta < 0.01){
        real gamma_m1 = s/(gamma + 1);
        real K1_scaled = std::sqrt(M_PI*theta/2)
                       * (1 + 3*theta/8 - 15*theta*theta/128);
        return std::exp(-gamma_m1/theta)/(2*m*c*K1_scaled);
    }
    real norm = 2*m*c*std::cyl_bessel_k(1,1/theta);
    return std::exp(-gamma/theta)/norm;
}

template <typename real>
real p_max_juttner(real m, real c, real theta, real eps = 1e-15)
{
    real gamma_max = 1 + theta*std::log(1/eps);
    return m*c*std::sqrt(gamma_max*gamma_max - 1);
}

const double light_speed = 1;
const double me = 1;
const double mi = 1836;
const double qe = -1;
const double qi = 1;

const double gamma_e0 = 2;
const double gamma_i0 = 1 + (me/mi)*(gamma_e0 - 1);
const double p0_e = me*light_speed*std::sqrt(gamma_e0*gamma_e0 - 1);
const double p0_i = mi*light_speed*std::sqrt(gamma_i0*gamma_i0 - 1);

const double theta_e = 1e-1;
const double theta_i = theta_e*me/mi;
const double alpha = 1e-4;
const double k0 = 0.05;
const double x_min = 0;
const double x_max = 2*M_PI/k0;

const size_t n_modes = 16;
const size_t n_profile = 4096;

std::vector<double> make_random_profile(unsigned int seed)
{
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> phase(0,2*M_PI);
    std::vector<double> phases(n_modes), values(n_profile);
    for (double &p : phases) p = phase(rng);

    double amplitude = alpha*std::sqrt(2.0/n_modes);
    for (size_t i = 0; i < n_profile; ++i){
        double angle = 2*M_PI*i/n_profile;
        double value = 0;
        for (size_t j = 0; j < n_modes; ++j)
            value += std::cos((j+1)*angle + phases[j]);
        values[i] = amplitude*value;
    }
    return values;
}

const std::vector<double> perturb_e = make_random_profile(12345);
const std::vector<double> perturb_i = make_random_profile(67890);

double eval_random_profile(double x, const std::vector<double> &values) noexcept
{
    double y = (x-x_min)/(x_max-x_min);
    y -= std::floor(y);
    double pos = y*n_profile;
    size_t i = static_cast<size_t>(pos);
    double t = pos-i;

    size_t im1 = (i+n_profile-1)%n_profile;
    size_t ip1 = (i+1)%n_profile;
    size_t ip2 = (i+2)%n_profile;
    double a = values[im1], b = values[i], c = values[ip1], d = values[ip2];
    return 0.5*(2*b + (-a+c)*t + (2*a-5*b+4*c-d)*t*t
               + (-a+3*b-3*c+d)*t*t*t);
}

template <typename real>
real f0_e(real x, real p) noexcept
{
    return (1 + eval_random_profile(x,perturb_e))
         * 0.5*(maxwell_juttner_1d<real>(p-p0_e,me,light_speed,theta_e)
              + maxwell_juttner_1d<real>(p+p0_e,me,light_speed,theta_e));
}

template <typename real>
real f0_i(real x, real p) noexcept
{
    return (1 + eval_random_profile(x,perturb_i))
         * 0.5*(maxwell_juttner_1d<real>(p-p0_i,mi,light_speed,theta_i)
              + maxwell_juttner_1d<real>(p+p0_i,mi,light_speed,theta_i));
}

std::vector<double> read_coeffs(size_t n, size_t stride_t, const std::string &filename)
{
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("Cannot open " + filename);

    std::vector<double> coeffs((n+1)*stride_t);
    for (double &c : coeffs)
        if (!(in >> c)) throw std::runtime_error("Coefficient file incomplete");
    return coeffs;
}

enum class QuadratureRule { midpoint, simpson, boole, gauss4, gauss8 };

QuadratureRule parse_rule(const std::string &name)
{
    if (name == "midpoint") return QuadratureRule::midpoint;
    if (name == "simpson") return QuadratureRule::simpson;
    if (name == "boole") return QuadratureRule::boole;
    if (name == "gauss4") return QuadratureRule::gauss4;
    if (name == "gauss8") return QuadratureRule::gauss8;
    throw std::invalid_argument("Unknown quadrature rule: " + name);
}

std::string rule_name(QuadratureRule rule)
{
    switch (rule){
        case QuadratureRule::midpoint: return "midpoint";
        case QuadratureRule::simpson: return "simpson";
        case QuadratureRule::boole: return "boole";
        case QuadratureRule::gauss4: return "gauss4";
        case QuadratureRule::gauss8: return "gauss8";
    }
    return "unknown";
}

void check_resolution(size_t np, QuadratureRule rule)
{
    if (np == 0) throw std::invalid_argument("Np must be positive");
    if (rule == QuadratureRule::simpson && np%2 != 0)
        throw std::invalid_argument("Simpson requires even Np (subintervals)");
    if ((rule == QuadratureRule::boole || rule == QuadratureRule::gauss4) && np%4 != 0)
        throw std::invalid_argument("Boole and Gauss4 require Np divisible by 4");
    if (rule == QuadratureRule::gauss8 && np%8 != 0)
        throw std::invalid_argument("Gauss8 requires Np divisible by 8");
}

// Composite Gauss-Legendre, with np total nodes and np/Nnodes panels.
template <size_t order, size_t Nnodes>
double density_gauss(size_t n, double x, size_t np, const double *coeffs,
                     const config_t<double> &conf,
                     const std::array<double,Nnodes> &nodes,
                     const std::array<double,Nnodes> &weights)
{
    size_t panels = np/Nnodes;
    double dp = (conf.u_max-conf.u_min)/panels;
    long double sum = 0;
    for (size_t i = 0; i < panels; ++i){
        double center = conf.u_min+(i+0.5)*dp;
        for (size_t j = 0; j < Nnodes; ++j){
            double p = center+0.5*dp*nodes[j];
            sum += weights[j]*eval_f<double,order>(n,x,p,coeffs,conf);
        }
    }
    return static_cast<double>(sum*(0.5*dp));
}

template <size_t order>
double density_quadrature(size_t n, double x, size_t np, const double *coeffs,
                          const config_t<double> &conf, QuadratureRule rule)
{
    static constexpr std::array<double,4> x4 = {
        -0.8611363115940526, -0.3399810435848563,
         0.3399810435848563,  0.8611363115940526
    };
    static constexpr std::array<double,4> w4 = {
        0.3478548451374539, 0.6521451548625461,
        0.6521451548625461, 0.3478548451374539
    };
    static constexpr std::array<double,8> x8 = {
        -0.9602898564975363, -0.7966664774136267,
        -0.5255324099163290, -0.1834346424956498,
         0.1834346424956498,  0.5255324099163290,
         0.7966664774136267,  0.9602898564975363
    };
    static constexpr std::array<double,8> w8 = {
        0.1012285362903763, 0.2223810344533745,
        0.3137066458778873, 0.3626837833783620,
        0.3626837833783620, 0.3137066458778873,
        0.2223810344533745, 0.1012285362903763
    };

    if (rule == QuadratureRule::gauss4)
        return density_gauss<order>(n,x,np,coeffs,conf,x4,w4);
    if (rule == QuadratureRule::gauss8)
        return density_gauss<order>(n,x,np,coeffs,conf,x8,w8);

    double dp = (conf.u_max-conf.u_min)/np;
    long double sum = 0;
    if (rule == QuadratureRule::midpoint){
        for (size_t j = 0; j < np; ++j){
            double p = conf.u_min+(j+0.5)*dp;
            sum += eval_f<double,order>(n,x,p,coeffs,conf);
        }
        return static_cast<double>(sum*dp);
    }

    // Newton-Cotes composite rules share endpoints between adjacent panels.
    for (size_t j = 0; j <= np; ++j){
        double p = conf.u_min+j*dp;
        double weight;
        if (rule == QuadratureRule::simpson){
            weight = (j == 0 || j == np) ? 1 : (j%2 ? 4 : 2);
        } else {
            weight = (j == 0 || j == np) ? 7 : (j%4 == 0 ? 14 : (j%2 ? 32 : 12));
        }
        sum += weight*eval_f<double,order>(n,x,p,coeffs,conf);
    }
    return static_cast<double>(sum*dp*(rule == QuadratureRule::simpson ? 1.0/3 : 2.0/45));
}

// Save f(x,p) at four of the spatial points used in the quadrature test.
// Each species has its own momentum range and therefore its own output file.
template <size_t order>
void plot_f_cross_sections(size_t n, size_t nx_test, size_t np_plot,
                           const double *coeffs, const config_t<double> &conf_e,
                           const config_t<double> &conf_i)
{
    if (!np_plot) throw std::invalid_argument("np_plot must be positive");

    std::vector<size_t> indices = {0, nx_test/4, nx_test/2, 3*nx_test/4};
    std::sort(indices.begin(),indices.end());
    indices.erase(std::unique(indices.begin(),indices.end()),indices.end());

    std::string suffix = "_n" + std::to_string(n);
    std::string gp_name = "f_cross_sections" + suffix + ".gp";
    std::ofstream gp(gp_name);
    if (!gp) throw std::runtime_error("Cannot write " + gp_name);

    for (int species = 0; species < 2; ++species){
        const auto &conf = species == 0 ? conf_e : conf_i;
        std::string tag = species == 0 ? "e" : "i";
        std::string data_name = "f_cross_sections_" + tag + suffix + ".txt";
        std::string png_name = "f_cross_sections_" + tag + suffix + ".png";
        double dp = (conf.u_max-conf.u_min)/np_plot;

        std::vector<std::vector<double>> f_values(indices.size(),std::vector<double>(np_plot));
        #pragma omp parallel for collapse(2)
        for (size_t i = 0; i < indices.size(); ++i){
            for (size_t j = 0; j < np_plot; ++j){
                double x = conf.x_min+(indices[i]+0.5)*conf.Lx/nx_test;
                double p = conf.u_min+(j+0.5)*dp;
                f_values[i][j] = eval_f<double,order>(n,x,p,coeffs,conf);
            }
        }

        std::ofstream out(data_name);
        if (!out) throw std::runtime_error("Cannot write " + data_name);
        out << std::setprecision(17);
        out << "# p";
        for (size_t ix : indices){
            double x = conf.x_min+(ix+0.5)*conf.Lx/nx_test;
            out << " f_at_x=" << x << "_ix=" << ix;
        }
        out << '\n';
        for (size_t j = 0; j < np_plot; ++j){
            double p = conf.u_min+(j+0.5)*dp;
            out << p;
            for (size_t i = 0; i < indices.size(); ++i) out << ' ' << f_values[i][j];
            out << '\n';
        }

        gp << "set terminal pngcairo size 1200,700\n";
        gp << "set output '" << png_name << "'\n";
        gp << "set xlabel 'p'\nset ylabel 'f_" << tag << "(x,p)'\n";
        gp << "set title '" << (species == 0 ? "Electron" : "Ion")
           << " momentum cross-sections, t=" << n*conf.dt << "'\n";
        gp << "set key outside right\n";
        gp << "plot ";
        for (size_t i = 0; i < indices.size(); ++i){
            double x = conf.x_min+(indices[i]+0.5)*conf.Lx/nx_test;
            if (i) gp << ", \\\n     ";
            gp << "'" << data_name << "' using 1:" << i+2
               << " with lines title 'x=" << std::setprecision(8) << x << "'";
        }
        gp << "\n";

        std::cout << "Saved " << data_name << "\n";
    }
    std::cout << "Run 'gnuplot " << gp_name << "' to generate the two PNG plots.\n";
}

template <size_t order>
void quadrature_error_from_coeffs(size_t n, size_t Nx, size_t Nu_e, size_t Nu_i,
                                  QuadratureRule rule, size_t nx_test = 32,
                                  const std::string &filename = "coeffs_cmm.txt",
                                  size_t np_plot = 4096)
{
    const double dt = 0.1;
    const size_t Nt = static_cast<size_t>(1000/dt);
    if (n > Nt || !Nx || !Nu_e || !Nu_i || !nx_test)
        throw std::runtime_error("Invalid simulation parameters");

    double p_max_e = p0_e+p_max_juttner(me,light_speed,theta_e);
    double p_max_i = p0_i+p_max_juttner(mi,light_speed,theta_i);

    config_t<double> conf_e(Nx,Nu_e,Nt,dt,x_min,x_max,-p_max_e,p_max_e,&f0_e<double>);
    config_t<double> conf_i(Nx,Nu_i,Nt,dt,x_min,x_max,-p_max_i,p_max_i,&f0_i<double>);
    conf_e.q = qe;
    conf_e.m = me;
    conf_e.light_speed = light_speed;
    conf_i.q = qi;
    conf_i.m = mi;
    conf_i.light_speed = light_speed;

    const size_t stride_t = Nx+order-1;
    std::vector<double> coeffs = read_coeffs(n,stride_t,filename);

    // Write the cross-sections first; the convergence sweep may take a long time.
    plot_f_cross_sections<order>(n,nx_test,np_plot,coeffs.data(),conf_e,conf_i);

    size_t baseline = 64;
    const std::vector<size_t> np_values = {
                                            baseline,
                                         4*baseline,
                                         8*baseline,
                                         32*baseline,
                                         128*baseline,
                                         512*baseline,
                                         //2048*baseline,
                                         //8192*baseline,
                                         //4*8192*baseline,
                                         //16*8192*baseline
                                         };
    for (size_t np : np_values) check_resolution(np,rule);

    std::vector<std::vector<double>> ne(np_values.size()), ni(np_values.size());
    for (size_t j = 0; j < np_values.size(); ++j){
        ne[j].resize(nx_test);
        ni[j].resize(nx_test);
        #pragma omp parallel for
        for (size_t ix = 0; ix < nx_test; ++ix){
            double x = x_min+(ix+0.5)*(x_max-x_min)/nx_test;
            ne[j][ix] = density_quadrature<order>(n,x,np_values[j],coeffs.data(),conf_e,rule);
            ni[j][ix] = density_quadrature<order>(n,x,np_values[j],coeffs.data(),conf_i,rule);
        }
    }

    std::string output_name = "quadrature_error_" + rule_name(rule)
                            + "_n" + std::to_string(n) + ".txt";
    std::ofstream out(output_name);
    if (!out) throw std::runtime_error("Cannot write " + output_name);
    out << std::setprecision(17) << "# Np max_err_ne max_err_ni max_err_charge\n";

    for (size_t j = 0; j+1 < np_values.size(); ++j){
        double err_e = 0, err_i = 0, err_charge = 0;
        for (size_t ix = 0; ix < nx_test; ++ix){
            double de = ne[j][ix]-ne.back()[ix];
            double di = ni[j][ix]-ni.back()[ix];
            err_e = std::max(err_e,std::abs(de));
            err_i = std::max(err_i,std::abs(di));
            err_charge = std::max(err_charge,std::abs(qe*de+qi*di));
        }
        out << np_values[j] << " " << err_e << " " << err_i << " " << err_charge << "\n";
        std::cout << np_values[j] << " " << err_e << " " << err_i << " " << err_charge << "\n";
    }
    std::cout << "Quadrature convergence (" << rule_name(rule) << ") at t=" << n*dt
              << " saved to " << output_name << "\n";
}

} // namespace relativistic
} // namespace periodic
} // namespace dim1
} // namespace nufi

int main(int argc, char **argv)
{
    if (argc < 6 || argc > 9){
        std::cerr << "Usage: " << argv[0]
                  << " <time_step> <Nx> <Nu_e> <Nu_i> <rule> [nx_test] [coeffs_file] [np_plot]\n"
                  << "Rules: midpoint, simpson, boole, gauss4, gauss8\n";
        return 1;
    }
    try {
        size_t nx_test = argc > 6 ? std::stoul(argv[6]) : 32;
        std::string filename = argc > 7 ? argv[7] : "coeffs_cmm.txt";
        size_t np_plot = argc > 8 ? std::stoul(argv[8]) : 4096;
        auto rule = nufi::dim1::periodic::relativistic::parse_rule(argv[5]);
        nufi::dim1::periodic::relativistic::quadrature_error_from_coeffs<4>(
            std::stoul(argv[1]),std::stoul(argv[2]),std::stoul(argv[3]),
            std::stoul(argv[4]),rule,nx_test,filename,np_plot);
    } catch (const std::exception &e){
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}

