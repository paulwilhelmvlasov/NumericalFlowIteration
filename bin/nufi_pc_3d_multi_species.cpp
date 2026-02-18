#include <cmath>
#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include <armadillo>

#include <nufi/config.hpp>
#include <nufi/random.hpp>
#include <nufi/fields.hpp>
#include <nufi/Maxwell.hpp>
#include <nufi/poisson.hpp>
#include <nufi/restart.hpp>
#include <nufi/rho.hpp>
#include <nufi/stopwatch.hpp>


namespace pezzini
{
// The following sets up the simulation in ion scale units:
double mass_ratio = 183.6 /10.0;

double me = 1.0/mass_ratio;
double mi = 1;

double uth_elec = 0.02071;
double vth_elec = 0.02071;
double wth_elec = 0.02071;

double uth_ion_core = 0.00186;
double vth_ion_core = 0.00163;
double wth_ion_core = 0.00163;

double u_drift_ion_core = -0.00053;

double uth_ion_beam = 0.00292;
double vth_ion_beam = 0.00230;
double wth_ion_beam = 0.00230;

double u_drift_ion_beam = 0.00339;

double nc = 0.864;
double nb = 1 - nc;

// Because I use rationalized CGS instead of plain CGS the initial B0 has to be 
// divived by 4*pi.
double B0 = 0.00270 / (4 * M_PI); 

double xmin = 0;
double xmax = 64;
double ymin = 0; 
double ymax = 256;
double zmin = 0;
double zmax = 1;
}

namespace electro_static_bump_on_tail
{
// The following sets up the simulation in ion scale units:
double mass_ratio = 183.6 ;

/* double me = 1.0/mass_ratio;
double mi = 1; */

double me = 1.0;
double mi = mass_ratio;

double uth_elec_core = 1;
double vth_elec_core = 1;
double wth_elec_core = 1;

double uth_elec_beam = 0.5;
double vth_elec_beam = 0.5;
double wth_elec_beam = 0.5;

double u_d = 4.5;

double uth_ion_core = 1;
double vth_ion_core = 1;
double wth_ion_core = 1;

double nc = 0.9;
double nb = 0.2;

double k = 0.3;
//double k = 0.5;

double xmin = 0;
double xmax = 2*M_PI/k;
double ymin = 0; 
//double ymax = 1;
double ymax = xmax; // For electro-magnetic effects
double zmin = 0;
double zmax = 1;
}

namespace double_harris_magnetic_reconnection
{
// The following sets up the simulation in ion scale units:
double mass_ratio = 25;

double me = 1.0;
double mi = mass_ratio;

double uth_elec_core = 1;

double uth_ion_core = std::sqrt(0.2);

double k = 0.3;
//double k = 0.5;

double Lx = 25.6;
double Ly = 12.8;
double xmin = 0;
double xmax = Lx;
double ymin = 0; 
double ymax = Ly; 
double zmin = 0;
double zmax = 1;

// Sheets
double l0 = 0.5;
double y1 = Ly/4.0;
double y2 = 3 * Ly / 4.0;

// Fields:
double B0 = 1;

// Density 
double n0 = 1;
double nb = 0.2;

double sech2(double x){
    double y = 1.0 / std::cosh(x);
    return y*y;
}

double init_density(double x, double y) {
    return nb + n0 * (sech2( (y - y1)/l0 ) + sech2( (y-y2)/l0 ));
}

double Bx(double x, double y) {
    return B0 * (std::tanh( (y - y1)/l0 ) - tanh( (y-y2)/l0 ) - 1);
}

arma::Col<double> delta_B(double x, double y){
    double kx = 2.0*M_PI / Lx;
    double ky = 2.0*M_PI / Ly;

    double eps = 1e-2;

    double Ap = eps * B0 * (std::min(Lx, Ly) / (2.0*M_PI)); // eps=1e-3..1e-2
    return arma::Col<double>({
        -Ap * ky * std::cos(kx*x) * std::sin(ky*y),
        Ap * kx * std::sin(kx*x) * std::cos(ky*y),
        0});
}

double Jz(double x, double y){
    return -B0/l0 * ( sech2( (y - y1)/l0 ) - sech2( (y-y2)/l0 ) );
}

double uze(double x, double y){
    return B0/l0/init_density(x,y) * ( sech2( (y - y1)/l0 ) - sech2( (y-y2)/l0 ) );
}

}

namespace nufi
{

namespace dim3
{

// Electro-static case
/* double xmin = 0;
double xmax = 4*M_PI;
double ymin = 0;
double ymax = 1;
double zmin = 0;
double zmax = 1;

double umin_e = -5;
double umax_e = 5;
double vmin_e = -0.5;
double vmax_e = 0.5;
double wmin_e = -0.5;
double wmax_e = 0.5;

double umin_i = -5;
double umax_i = 5;
double vmin_i = -0.5;
double vmax_i = 0.5;
double wmin_i = -0.5;
double wmax_i = 0.5;
 */
/* double xmin = pezzini::xmin;
double xmax = pezzini::xmax;
double ymin = pezzini::ymin;
double ymax = pezzini::ymax;
double zmin = pezzini::zmin;
double zmax = pezzini::zmax;

double umin_e = -5*pezzini::vth_elec;
double umax_e = 5*pezzini::vth_elec;
double vmin_e = -5*pezzini::vth_elec;
double vmax_e = 5*pezzini::vth_elec;
double wmin_e = -5*pezzini::vth_elec;
double wmax_e = 5*pezzini::vth_elec;

double umin_i = -8*pezzini::uth_ion_beam;
double umax_i = 8*pezzini::uth_ion_beam;
double vmin_i = -5*pezzini::vth_ion_beam;
double vmax_i = 5*pezzini::vth_ion_beam;
double wmin_i = -5*pezzini::wth_ion_beam;
double wmax_i = 5*pezzini::wth_ion_beam;
 */
/* double xmin = electro_static_bump_on_tail::xmin;
double xmax = electro_static_bump_on_tail::xmax;
double ymin = electro_static_bump_on_tail::ymin;
double ymax = electro_static_bump_on_tail::ymax;
double zmin = electro_static_bump_on_tail::zmin;
double zmax = electro_static_bump_on_tail::zmax;

double umin_e = -10*electro_static_bump_on_tail::uth_elec_core;
double umax_e = 10*electro_static_bump_on_tail::uth_elec_core;
double vmin_e = -5*electro_static_bump_on_tail::vth_elec_core;
double vmax_e = 5*electro_static_bump_on_tail::vth_elec_core;
double wmin_e = -5*electro_static_bump_on_tail::wth_elec_core;
double wmax_e = 5*electro_static_bump_on_tail::wth_elec_core;

double umin_i = -5*electro_static_bump_on_tail::uth_ion_core;
double umax_i = 5*electro_static_bump_on_tail::uth_ion_core;
double vmin_i = -5*electro_static_bump_on_tail::uth_ion_core;
double vmax_i = 5*electro_static_bump_on_tail::uth_ion_core;
double wmin_i = -5*electro_static_bump_on_tail::uth_ion_core;
double wmax_i = 5*electro_static_bump_on_tail::uth_ion_core; */

double xmin = double_harris_magnetic_reconnection::xmin;
double xmax = double_harris_magnetic_reconnection::xmax;
double ymin = double_harris_magnetic_reconnection::ymin;
double ymax = double_harris_magnetic_reconnection::ymax;
double zmin = double_harris_magnetic_reconnection::zmin;
double zmax = double_harris_magnetic_reconnection::zmax;

double umin_e = -8*double_harris_magnetic_reconnection::uth_elec_core;
double umax_e = 8*double_harris_magnetic_reconnection::uth_elec_core;
double vmin_e = -8*double_harris_magnetic_reconnection::uth_elec_core;
double vmax_e = 8*double_harris_magnetic_reconnection::uth_elec_core;
double wmin_e = -5*double_harris_magnetic_reconnection::uth_elec_core;
double wmax_e = 5*double_harris_magnetic_reconnection::uth_elec_core;

double umin_i = -5*double_harris_magnetic_reconnection::uth_ion_core;
double umax_i = 5*double_harris_magnetic_reconnection::uth_ion_core;
double vmin_i = -5*double_harris_magnetic_reconnection::uth_ion_core;
double vmax_i = 5*double_harris_magnetic_reconnection::uth_ion_core;
double wmin_i = -5*double_harris_magnetic_reconnection::uth_ion_core;
double wmax_i = 5*double_harris_magnetic_reconnection::uth_ion_core;

// Spatial grid must be the same for both species!!!
size_t Nx = 32;
size_t Ny = 32;
size_t Nz = 1;

size_t Nu_e = 32;
size_t Nv_e = 32;
size_t Nw_e = 16;

size_t Nu_i = 32;
size_t Nv_i = 32;
size_t Nw_i = 16;

size_t steps_per_1 = 10;
double   dt = 1.0 / steps_per_1;
size_t Nt = 1000*steps_per_1;

size_t nx_r = Nx;
size_t ny_r = Ny;
size_t nz_r = Nz;

size_t nu_r_e = Nu_e;
size_t nv_r_e = Nv_e;
size_t nw_r_e = Nw_e;

size_t nu_r_i = Nu_i;
size_t nv_r_i = Nv_i;
size_t nw_r_i = Nw_i;

size_t nt_restart = 20;

template <typename real>
real f0_electron(real x, real y, real z, real u, real v, real w) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

    // Weak Landau Damping
    //return (1+0.01*cos(0.5*x))*maxwellian_1d<real>(u,1);

    // Pezzini
    //return maxwellian<double>(u,v,w,pezzini::uth_elec);

    // Electro-static bump-on-tail
    /* return 1.0 / std::sqrt(2.0 * M_PI) * (1 + 0.04 * std::cos(0.3*x)) 
            *  ( 0.9 * std::exp(-0.5 * u*u)  + 0.2 * std::exp(-0.5/(0.5*0.5) * (u-4.5)*(u-4.5)) );  */
    /* return (1 + 0.04 * std::cos( electro_static_bump_on_tail::k * x )) 
            * (electro_static_bump_on_tail::nc * maxwellian_1d<real>(u, electro_static_bump_on_tail::uth_elec_core) 
            + electro_static_bump_on_tail::nb * maxwellian_1d<real>(u - electro_static_bump_on_tail::u_d, electro_static_bump_on_tail::uth_elec_beam))
            * maxwellian_2d<real>(v, w, electro_static_bump_on_tail::vth_elec_core); */

    // Double Harris Magnetic Reconnection
    return double_harris_magnetic_reconnection::init_density(x,y) 
            * maxwellian<double>(u,v,w - double_harris_magnetic_reconnection::uze(x,y),
                    double_harris_magnetic_reconnection::uth_elec_core);
}

template <typename real>
real f0_ion(real x, real y, real z, real u, real v, real w) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

    // Weak Landau Damping
    //return maxwellian_1d<real>(u,1);

    // Pezzini
    /* return pezzini::nc * maxwellian_1d<double>(u-pezzini::u_drift_ion_core,pezzini::uth_ion_core)
                        * maxwellian_1d<double>(v,pezzini::vth_ion_core)
                        * maxwellian_1d<double>(w,pezzini::wth_ion_core)
        + pezzini::nb * maxwellian_1d<double>(u-pezzini::u_drift_ion_beam,pezzini::uth_ion_beam)
                        * maxwellian_1d<double>(v,pezzini::vth_ion_beam)
                        * maxwellian_1d<double>(w,pezzini::wth_ion_beam); */

    // Electro-static bump on tail.
    //return maxwellian_1d<double>(u,1);
    //return maxwellian<real>(u,v,w, electro_static_bump_on_tail::uth_ion_core);

    // Double Harris Magnetic Reconnection
    return double_harris_magnetic_reconnection::init_density(x,y)
            * maxwellian<double>(u,v,w,double_harris_magnetic_reconnection::uth_ion_core);
}


template <typename real>
arma::Col<real> E0(real x, real y, real z)
{
    // Electro-Static setup for Weak Landau or TSI:
    //return  arma::Col<real>({-0.02 * std::sin(0.5*x), 0, 0});

    // Pezzini
    return  arma::Col<real>({0, 0, 0});
}

template <typename real>
arma::Col<real> B0(real x, real y, real z)
{
    // Electro-Static
    //return  arma::Col<real>({0, 0, 0});

    // Pezzini
    /* double Lx = xmax - xmin;
    double Ly = ymax - ymin;
    double kx = 2*M_PI / Lx;
    double ky = 2*M_PI / Ly;
    double perturb = (1 + 0.1 * std::cos(kx*x)*std::sin(ky*y));
    return  arma::Col<real>({pezzini::B0*perturb, 0, 0}); */
    //return  arma::Col<real>({pezzini::B0, 0, 0});


    // Electro-static bump on tail.
    //return arma::Col<real>({0,0,0});

    // Double Harris Magnetic Reconnection
    arma::Col<real> B_init({
        double_harris_magnetic_reconnection::Bx(x,y),
        0, 
        0
    });
    return B_init + double_harris_magnetic_reconnection::delta_B(x,y);
}

void random_perturbation_2d_f0_electron(arma::mat& restart_mat, double eps = 1e-3)
{
    double Lx = xmax - xmin;
    double Ly = ymax - ymin;
    std::vector<double> x_pertb = generateRandomSmoothFunction(Lx, 100, nx_r);
    std::vector<double> y_pertb = generateRandomSmoothFunction(Ly, 100, ny_r);

    size_t size_x_r = (nx_r+1)*(ny_r+1)*(nz_r+1);
    size_t size_v_r = (nu_r_e+1)*(nv_r_e+1)*(nw_r_e+1);

    double dx_r = (xmax - xmin) / nx_r;
    double dy_r = (ymax - ymin) / ny_r;
    double dz_r = (zmax - zmin) / nz_r;

    double du_r = (umax_e - umin_e) / nu_r_e;
    double dv_r = (vmax_e - vmin_e) / nv_r_e;
    double dw_r = (wmax_e - wmin_e) / nw_r_e;

    std::cout << "Electron restart matrix " << restart_mat.n_rows << " " << restart_mat.n_cols << std::endl;

    #pragma omp parallel for collapse(3)
    for(size_t iu = 0; iu <= nu_r_e; iu++)
    for(size_t iv = 0; iv <= nv_r_e; iv++)
    for(size_t iw = 0; iw <= nw_r_e; iw++)
    for(size_t ix = 0; ix <= nx_r; ix++)
    for(size_t iy = 0; iy <= ny_r; iy++)
    for(size_t iz = 0; iz <= nz_r; iz++){
        double x = xmin + ix*dx_r;
        double y = ymin + iy*dy_r;
        double z = zmin + iz*dz_r;

        double u = umin_e + iu*du_r;
        double v = vmin_e + iv*dv_r;
        double w = wmin_e + iw*dw_r;

        size_t index_0 = ix + (nx_r+1)*(iy + (ny_r+1)*iz);
        size_t index_1 = iu + (nu_r_e+1)*(iv + (nv_r_e+1)*iw);

        if(ix == 0 || ix == nx_r || iy == 0 || iy == ny_r){
            restart_mat(index_0, index_1) = f0_electron<double>(x,y,z,u,v,w);
        } else {
            restart_mat(index_0, index_1) = (1 + eps * x_pertb[ix]) 
                                        * (1 + eps * y_pertb[iy]) 
                                        * f0_electron<double>(x,y,z,u,v,w);
        }
    }
}

void random_perturbation_2d_f0_ion(arma::mat& restart_mat, double eps = 1e-3)
{
    double Lx = xmax - xmin;
    double Ly = ymax - ymin;
    std::vector<double> x_pertb = generateRandomSmoothFunction(Lx, 100, nx_r);
    std::vector<double> y_pertb = generateRandomSmoothFunction(Ly, 100, ny_r);

    size_t size_x_r = (nx_r+1)*(ny_r+1)*(nz_r+1);
    size_t size_v_r = (nu_r_i+1)*(nv_r_i+1)*(nw_r_i+1);

    double dx_r = (xmax - xmin) / nx_r;
    double dy_r = (ymax - ymin) / ny_r;
    double dz_r = (zmax - zmin) / nz_r;

    double du_r = (umax_i - umin_i) / nu_r_i;
    double dv_r = (vmax_i - vmin_i) / nv_r_i;
    double dw_r = (wmax_i - wmin_i) / nw_r_i;

    #pragma omp parallel for collapse(3)
    for(size_t iu = 0; iu <= nu_r_i; iu++)
    for(size_t iv = 0; iv <= nv_r_i; iv++)
    for(size_t iw = 0; iw <= nw_r_i; iw++)
    for(size_t ix = 0; ix <= nx_r; ix++)
    for(size_t iy = 0; iy <= ny_r; iy++)
    for(size_t iz = 0; iz <= nz_r; iz++){
        double x = xmin + ix*dx_r;
        double y = ymin + iy*dy_r;
        double z = zmin + iz*dz_r;

        double u = umin_i + iu*du_r;
        double v = vmin_i + iv*dv_r;
        double w = wmin_i + iw*dw_r;

        size_t index_0 = ix + (nx_r+1)*(iy + (ny_r+1)*iz);
        size_t index_1 = iu + (nu_r_i+1)*(iv + (nv_r_i+1)*iw);

        if(ix == 0 || ix == nx_r || iy == 0 || iy == ny_r){
            restart_mat(index_0, index_1) = f0_ion<double>(x,y,z,u,v,w);
        } else {
            restart_mat(index_0, index_1) = (1 + eps * x_pertb[ix]) 
                                            * (1 + eps * y_pertb[iy]) 
                                            * f0_ion<double>(x,y,z,u,v,w);
        }
    }
}


nufi::restart::linear_interpolant_6d interpolant_electron;
nufi::restart::linear_interpolant_6d interpolant_ion;

double eval_f_electron_with_linear_interpolant(double x, double y, double z, double u, double v, double w)
{
    return interpolant_electron.eval_linear_interpolant(x,y,z,u,v,w);
}

double eval_f_ion_with_linear_interpolant(double x, double y, double z, double u, double v, double w)
{
    return interpolant_ion.eval_linear_interpolant(x,y,z,u,v,w);
}

template<size_t order>
void periodically_restarted_nufi_maxwell_lie_EBf_predictor_corrector_aligned()
{
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    if(mpi_rank == 0){
        std::cout << "Start Simulation with MPI support. Number of MPI processes: " << mpi_size << std::endl;
    }

    // Set up config.
    config_t<double> conf_electron (Nx, Ny, Nz, Nu_e, Nv_e, Nw_e, Nt, dt, 
                            xmin, xmax, ymin, ymax, zmin, zmax, umin_e, umax_e, 
                            vmin_e, vmax_e, wmin_e, wmax_e, &f0_electron, electro_static_bump_on_tail::me, -1);
    conf_electron.print_config(std::cout);
    config_t<double> conf_ion (Nx, Ny, Nz, Nu_i, Nv_i, Nw_i, Nt, dt, 
                            xmin, xmax, ymin, ymax, zmin, zmax, umin_i, umax_i, 
                            vmin_i, vmax_i, wmin_i, wmax_i, &f0_ion, electro_static_bump_on_tail::mi, 1);
    poisson<double> poiss( conf_electron );
    conf_ion.print_config(std::cout);

    if(conf_electron.Nx != conf_ion.Nx){
        throw std::runtime_error("Nx must be equal for all species!");
    }
    if(conf_electron.Ny != conf_ion.Ny){
        throw std::runtime_error("Ny must be equal for all species!");
    }
    if(conf_electron.Nz != conf_ion.Nz){
        throw std::runtime_error("Nz must be equal for all species!");
    }

    // Storage of coefficients now via: 
    // index = nt + Nt * (d + dim * (ix + Nx * (iy + Ny * iz)))
    size_t stride_t = (Nx + order - 1) *
                        (Ny + order - 1) *
                        (Nz + order - 1);

    const size_t dim = 3;
    const size_t Nx_ext = Nx + order - 1;
    const size_t Ny_ext = Ny + order - 1;
    const size_t Nz_ext = Nz + order - 1;
    const size_t Nspace = Nx_ext * Ny_ext * Nz_ext;

    std::cout << "Start NuFI Vlasov-Maxwell-Solver with fBE-Lie-Splitting." << std::endl;
    std::cout << "Init helper variables." << std::endl;

    // Flattened 1D coefficient storage
    std::vector<double> coeffs_E(3 * (Nt + 1) * stride_t, 0);
    std::vector<double> coeffs_B_staggered(3 * (Nt + 2) * stride_t, 0); // 0 = -1/2, 1 = 1/2, 2 = 3/2, ... (index = n - 1/2)
    std::vector<double> coeffs_B(3 * (Nt + 1) * stride_t, 0);
    std::vector<double> E(3 * Nx * Ny * Nz, 0);
    std::vector<double> B(3 * Nx * Ny * Nz, 0);
    std::vector<double> j_0(3 * Nx * Ny * Nz, 0);
    std::vector<double> j_electron(3 * Nx * Ny * Nz, 0);
    std::vector<double> j_ion(3 * Nx * Ny * Nz, 0);
    std::vector<double> j_1(3 * Nx * Ny * Nz, 0);

    std::ofstream kin_energy_file( "kin_energy.txt" );

    // Init restart matrices.
    std::cout << "Initialize restart matrices." << std::endl;
    interpolant_electron = nufi::restart::linear_interpolant_6d (xmin, xmax, ymin, ymax, zmin, zmax, 
                                        umin_e, umax_e, vmin_e, vmax_e, wmin_e, wmax_e, 
                                        nx_r, ny_r, nz_r, nu_r_e, nv_r_e, nw_r_e);
    interpolant_ion = nufi::restart::linear_interpolant_6d (xmin, xmax, ymin, ymax, zmin, zmax,
                                        umin_i, umax_i, vmin_i, vmax_i, wmin_i, wmax_i, 
                                        nx_r, ny_r, nz_r, nu_r_i, nv_r_i, nw_r_i);

    // Init f0 with random perturbation.
    //std::cout << "Random f0_electron perturbation. " << std::endl;
    //random_perturbation_2d_f0_electron(interpolant_electron.restart_matrix, 1e-2);
    /* std::cout << "Random f0_ion perturbation. " << std::endl;
    random_perturbation_2d_f0_ion(interpolant_ion.restart_matrix, 1e-2); */
    /* {
        std::ofstream restart_mat_electron_str("restart_mat_electron_" + std::to_string(0*conf_electron.dt) + ".txt");
        restart_mat_electron_str << interpolant_electron.restart_matrix;

        std::ofstream restart_mat_ion_str("restart_mat_ion_" + std::to_string(0*conf_electron.dt) + ".txt");
        restart_mat_ion_str << interpolant_ion.restart_matrix;
    } */

/*     double kin_energy_electron = interpolant_electron.compute_kinetic_energy();
    double kin_energy_ion = interpolant_electron.compute_kinetic_energy();
    double kin_energy = kin_energy_electron + kin_energy_ion;

    kin_energy_file << 0*conf_electron.dt << " " << kin_energy << " " << kin_energy_electron << " " << kin_energy_ion << std::endl;
 */
    //conf_electron.f0 = &eval_f_electron_with_linear_interpolant;
    //conf_ion.f0 = &eval_f_ion_with_linear_interpolant;

    std::vector<double> coeffs_phi(stride_t, 0);
    std::vector<double> rho_e(Nx*Ny*Nz, 0);
    std::vector<double> rho_i(Nx*Ny*Nz, 0);
    std::vector<double> rho(Nx*Ny*Nz, 0);
    std::vector<double> g(Nx*Ny*Nz, 0);
    
    std::cout << "Compute rho_0. " << std::endl;
    eval_rho_full_EBf<double,order>(0, rho_e, coeffs_E, coeffs_B, conf_electron);
    eval_rho_full_EBf<double,order>(0, rho_i, coeffs_E, coeffs_B, conf_ion);

    std::ofstream rho_str("rho.txt");
    #pragma omp parallel for
    for(size_t i = 0; i < rho.size(); i++){
        rho[i] = rho_i[i] + rho_e[i];
    }
    for(size_t i = 0; i < Nx; i++){
        rho_str << i*conf_electron.dx << " " << rho[i] << " " << rho_e[i] << " " << rho_i[i] << std::endl;
    }

    poiss.solve( rho.data() );

    std::cout << "Compute phi_0. " << std::endl;
    interpolate<double,order>(coeffs_phi.data(), rho.data(), conf_electron );

    auto eval_E0_random_pertb = [&](double x, double y, double z, size_t d) {
        if(d == 0){
            return -eval<double,order,1,0,0>(x,y,z, coeffs_phi.data(), conf_electron);
        } 
        if(d == 1 ){
            return -eval<double,order,0,1,0>(x,y,z, coeffs_phi.data(), conf_electron);
        }
        return -eval<double,order,0,0,1>(x,y,z, coeffs_phi.data(), conf_electron);
    };

    /* std::ofstream init_E0_str("init_E.txt");
    for(size_t i = 0; i < Nx; i++){
        double x = i*conf_electron.dx;
        init_E0_str << x << " " << eval_E0_random_pertb(x, 0, 0, 0) 
                    << " " << eval<double,order>(x,0,0,coeffs_phi.data(),conf_electron)    << std::endl;
    } */

    double tolerance = 1e-6;
    double tolerance_velocity_electron = tolerance * interpolant_electron.restart_matrix.max();
    double tolerance_velocity_ion = tolerance * interpolant_ion.restart_matrix.max();

    size_t nt_r_curr = 1;

    auto eval_f_electron = [&](double x, double y, double z, double u, double v, double w) {
        return eval_f_lie_EBf<double, order>( nt_r_curr, x, y, z, u, v, w, coeffs_E, coeffs_B, conf_electron);
    };
    auto eval_f_ion = [&](double x, double y, double z, double u, double v, double w) {
        return eval_f_lie_EBf<double, order>( nt_r_curr, x, y, z, u, v, w, coeffs_E, coeffs_B, conf_ion);
    };

    //std::random_device dev;
    //std::mt19937 rng(dev());
    //std::uniform_int_distribution<std::mt19937::result_type> dist(1e-8,1e-6);    

    // Compute E(0) and B(0).
    #pragma omp parallel for
    for(size_t l = 0; l < Nx*Ny*Nz; l++){
        size_t iz   = l   / (Nx * Ny);
        size_t tmp  = l   % (Nx * Ny);
        size_t iy   = tmp / Nx;
        size_t ix   = tmp % Nx;
    
        double x = xmin + ix*conf_electron.dx; 
        double y = ymin + iy*conf_electron.dy; 
        double z = zmin + iz*conf_electron.dz; 
        
        // Normal initialization:        
        arma::Col<double> E0_vec = E0(x,y,z);
        arma::Col<double> B0_vec = B0(x,y,z);

         for(size_t d = 0; d < 3; d++){
            size_t index = d*Nx*Ny*Nz + l;
            //E[index] = E0_vec(d);
            E[index] = eval_E0_random_pertb(x,y,z,d);
            B[index] = B0_vec(d);

            // Small random perturbation.
            /* if(d == 0){
                B[index] *= (1 + dist(rng)); 
            }  */
        }
    }

    std::ofstream init_E0_1_str("init_E0_1.txt");
    for(size_t i = 0; i < Nx; i++){
        init_E0_1_str << i*conf_electron.dx << " " << E[i] << std::endl;
    }

    // Interpolate E(0) and B(0).
    interpolate_fields_aligned<double,order>(0, coeffs_E, E, conf_electron);
    interpolate_fields_aligned<double,order>(0, coeffs_B, B, conf_electron);

    std::ofstream coeff_E_str("coeff_E.txt");
    std::ofstream coeff_B_str("coeff_B.txt");
    analysis::write_coeffs_nufi_pc<double,order>(0,coeffs_E,coeffs_B,conf_electron,coeff_E_str,coeff_B_str);

    // Compute B_{-1/2}.
    #pragma omp parallel for
    for(size_t l = 0; l < Nx*Ny*Nz; l++){
        size_t iz   = l   / (Nx * Ny);
        size_t tmp  = l   % (Nx * Ny);
        size_t iy   = tmp / Nx;
        size_t ix   = tmp % Nx;
    
        double x = xmin + ix*conf_electron.dx; 
        double y = ymin + iy*conf_electron.dy; 
        double z = zmin + iz*conf_electron.dz; 
        
        // Normal initialization:        
        arma::Col<double> B0_vec ({
                                eval<double,order>(x,y,z,coeffs_B.data() + idx_base(0,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,Nt),conf_electron),
                                eval<double,order>(x,y,z,coeffs_B.data() + idx_base(0,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,Nt),conf_electron),
                                eval<double,order>(x,y,z,coeffs_B.data() + idx_base(0,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,Nt),conf_electron)
                            });
        arma::Col<double> rot_E0_vec = rot<double,order>(0,x,y,z,coeffs_E,conf_electron);
        B0_vec = B0_vec + 0.5*conf_electron.dt*rot_E0_vec;

        for(size_t d = 0; d < 3; d++){
            size_t index = d*Nx*Ny*Nz + l;
            B[index] = B0_vec(d);
        }
    }

    // Interpolate B(-1/2).
    interpolate_fields_aligned<double,order>(0, coeffs_B_staggered, B, conf_electron);

    // Compute j(0).
    eval_j_full_EBf<double,order>(0, j_electron, coeffs_E, coeffs_B, conf_electron);
    mpi_allgather_field(j_electron, conf_electron);
    eval_j_full_EBf<double,order>(0, j_ion, coeffs_E, coeffs_B, conf_ion);
    mpi_allgather_field(j_ion, conf_ion);

    #pragma omp parallel for
    for(size_t i = 0; i < j_0.size(); i++){
        j_0[i] = j_ion[i] + j_electron[i];
    }

    // Compute B(1/2).
    maxwell::B_step_predictor_corrector<double,order>(1,coeffs_E,coeffs_B_staggered,B,conf_electron);

    // Do first output.
    std::ofstream stat_file( "stats.txt" );
    std::ofstream j_stat_file( "j_stats.txt" );
    analysis::do_stats<double,order>(0, 64, 1, 1, stat_file, coeffs_E, coeffs_B, conf_electron, true, true, 0);
    analysis::compute_j_l2(0,j_stat_file,conf_electron,j_0,j_electron,j_ion);
    {
        std::ofstream mat_e_str("f_e_full_" + std::to_string(0*conf_electron.dt) + ".txt" );
        std::ofstream mat_i_str("f_i_full_" + std::to_string(0*conf_electron.dt) + ".txt" );
/*         analysis::plot_full_f_3x3v_parallelized<double,order>(4,4,1,128,128,8,xmin,xmax,ymin,ymax,zmin,zmax,umin_e,umax_e,vmin_e,vmax_e,wmin_e,wmax_e,eval_f_electron,mat_e_str);
        analysis::plot_full_f_3x3v_parallelized<double,order>(4,4,1,128,128,8,xmin,xmax,ymin,ymax,zmin,zmax,umin_i,umax_i,vmin_i,vmax_i,wmin_i,wmax_i,eval_f_ion,mat_i_str);
 */
        analysis::plot_full_f_3x3v_parallelized<double,order>(32,32,1,64,64,1,xmin,xmax,ymin,ymax,zmin,zmax,umin_e,umax_e,vmin_e,vmax_e,wmin_e,wmax_e,eval_f_electron,mat_e_str);
        analysis::plot_full_f_3x3v_parallelized<double,order>(32,32,1,64,64,1,xmin,xmax,ymin,ymax,zmin,zmax,umin_i,umax_i,vmin_i,vmax_i,wmin_i,wmax_i,eval_f_ion,mat_i_str);

        std::ofstream j_e_str("j_e_" + std::to_string(0*conf_electron.dt) + ".txt" );
        for(size_t i = 0; i < j_electron.size(); i++){
            j_e_str << j_electron[i] << std::endl;
        }
        std::ofstream j_i_str("j_i_" + std::to_string(0*conf_electron.dt) + ".txt" );
        for(size_t i = 0; i < j_ion.size(); i++){
            j_i_str << j_ion[i] << std::endl;
        }

        std::ofstream j_str("j_" + std::to_string(0*conf_electron.dt) + ".txt" );
        for(size_t i = 0; i < j_0.size(); i++){
            j_str << j_0[i] << std::endl;
        }

        std::ofstream rho_str("rho_" + std::to_string(0*conf_electron.dt) + ".txt" );
        for(size_t i = 0; i < rho.size(); i++){
            rho_str << rho[i] << std::endl;
        }
    }

    std::cout << "Time-loop." << std::endl;    
    std::cout << " ---------------------------------- " << std::endl;
    std::ofstream gauss_law_error_str("gauss_law_error.txt");
    double total_time, total_time_with_plot = 0;
    for(size_t n = 1; n <= Nt; n++)
    {
        nufi::stopwatch<double> timer;
        
        // Compute j(n).
        eval_j_full_EBf<double,order>(nt_r_curr, j_electron, coeffs_E, coeffs_B, conf_electron);
        mpi_allgather_field(j_electron, conf_electron);
        eval_j_full_EBf<double,order>(nt_r_curr, j_ion, coeffs_E, coeffs_B, conf_ion);
        mpi_allgather_field(j_ion, conf_ion);

        #pragma omp parallel for
        for(size_t i = 0; i < j_1.size(); i++){
            j_1[i] = j_ion[i] + j_electron[i];
        }

        // Compute E(n).
        maxwell::E_step_predictor_corrector<double,order>(nt_r_curr,coeffs_E,coeffs_B_staggered,E,j_0,j_1,conf_electron);

        // Clean gauss-law:
        eval_rho_full_EBf<double,order>(nt_r_curr, rho_e, coeffs_E, coeffs_B, conf_electron);
        mpi_allgather_field(rho_e, conf_electron);
        eval_rho_full_EBf<double,order>(nt_r_curr, rho_i, coeffs_E, coeffs_B, conf_ion);
        mpi_allgather_field(rho_i, conf_ion);

        #pragma omp parallel for
        for(size_t i = 0; i < rho.size(); i++){
            rho[i] = rho_i[i] + rho_e[i];
        }
        double gauss_law_error = maxwell::E_clean_gauss_law<double,order>(nt_r_curr,coeffs_E, 
                                                    E, rho, g, coeffs_phi, conf_electron, poiss);
        gauss_law_error_str << n*conf_electron.dt << " " << gauss_law_error << std::endl;


        // Compute B(n+1/2).
        maxwell::B_step_predictor_corrector<double,order>(nt_r_curr+1,coeffs_E, coeffs_B_staggered, B,conf_electron);

        // Compute B(n).
        maxwell::B_average<double,order>(nt_r_curr,coeffs_B,coeffs_B_staggered,B,conf_electron);

        // Shuffle j_1 into j_0.
        j_0 = j_1;

        double time_for_step = timer.elapsed();
        timer.reset();
        
        bool plot_EB = (n % (5*steps_per_1) == 0);
        bool plot_f_j = (n % (5*steps_per_1) == 0);
        // Careful: Currently stats only evaluates in x not y!!!
        analysis::do_stats<double,order>(nt_r_curr, 64, 1, 1, stat_file, coeffs_E, coeffs_B, conf_electron, plot_EB, true, n);
        analysis::compute_j_l2(n, j_stat_file, conf_electron, j_0, j_electron, j_ion);
        if(plot_f_j){
            std::ofstream mat_e_str("f_e_full_" + std::to_string(n*conf_electron.dt) + ".txt" );
            analysis::plot_full_f_3x3v_parallelized<double,order>(32,32,1,64,64,1,xmin,xmax,ymin,ymax,zmin,zmax,umin_e,umax_e,vmin_e,vmax_e,wmin_e,wmax_e,eval_f_electron,mat_e_str);
            
            std::ofstream mat_i_str("f_i_full_" + std::to_string(n*conf_electron.dt) + ".txt" );
            analysis::plot_full_f_3x3v_parallelized<double,order>(32,32,1,64,64,1,xmin,xmax,ymin,ymax,zmin,zmax,umin_i,umax_i,vmin_i,vmax_i,wmin_i,wmax_i,eval_f_ion,mat_i_str);

            std::ofstream j_e_str("j_e_" + std::to_string(n*conf_electron.dt) + ".txt" );
            for(size_t i = 0; i < j_electron.size(); i++){
                j_e_str << j_electron[i] << std::endl;
            }
            std::ofstream j_i_str("j_i_" + std::to_string(n*conf_electron.dt) + ".txt" );
            for(size_t i = 0; i < j_ion.size(); i++){
                j_i_str << j_ion[i] << std::endl;
            }

            std::ofstream j_str("j_" + std::to_string(n*conf_electron.dt) + ".txt" );
            for(size_t i = 0; i < j_0.size(); i++){
                j_str << j_0[i] << std::endl;
            }

            std::ofstream rho_str("rho_" + std::to_string(n*conf_electron.dt) + ".txt" );
            for(size_t i = 0; i < rho.size(); i++){
                rho_str << rho[i] << std::endl;
            }
        }
        
        /* analysis::do_stats<double,order>(nt_r_curr, 32, 32, 1, stat_file, coeffs_E, coeffs_B, conf_electron, true, true, n);
        std::ofstream j_e_str("j_e_" + std::to_string(n*conf_electron.dt) + ".txt" );
        for(size_t i = 0; i < j_electron.size(); i++){
            j_e_str << j_electron[i] << std::endl;
        }
        std::ofstream j_i_str("j_i_" + std::to_string(n*conf_electron.dt) + ".txt" );
        for(size_t i = 0; i < j_ion.size(); i++){
            j_i_str << j_ion[i] << std::endl;
        } */
        
        double time_for_plot = timer.elapsed();
        std::cout << "Plotting took " << time_for_plot << " s." << std::endl;
        analysis::write_coeffs_nufi_pc<double,order>(nt_r_curr,coeffs_E,coeffs_B,conf_electron,coeff_E_str,coeff_B_str);

        total_time += time_for_step;
        total_time_with_plot += time_for_step + time_for_plot;
        std::cout << "Time step " << n << " took a total of " << time_for_step << " s. So far total time = " << total_time << " s." << std::endl;

        if(nt_r_curr == nt_restart){
            timer.reset();
            std::cout << "Restart simulation. " << std::endl;
            //interpolant_electron.restart_f(eval_f_electron);
            interpolant_electron.restart_f_MPI(eval_f_electron);
            //std::ofstream restart_mat_electron_str("restart_mat_electron_" + std::to_string(n*conf_electron.dt) + ".txt");
            //restart_mat_electron_str << interpolant_electron.restart_matrix;

            //interpolant_ion.restart_f(eval_f_ion);
            interpolant_ion.restart_f_MPI(eval_f_ion);
            //std::ofstream restart_mat_ion_str("restart_mat_ion_" + std::to_string(n*conf_electron.dt) + ".txt");
            //restart_mat_ion_str << interpolant_ion.restart_matrix;
            
            /* interpolant_electron.restart_f_checking_boundaries(eval_f_electron, umin_e, umax_e, vmin_e, vmax_e, wmin_e, wmax_e, tolerance_velocity_electron);
            conf_electron.u_min = umin_e;
            conf_electron.u_max = umax_e;
            conf_electron.v_min = vmin_e;
            conf_electron.v_max = vmax_e;
            conf_electron.w_min = wmin_e;
            conf_electron.w_max = wmax_e;

            interpolant_ion.restart_f_checking_boundaries(eval_f_ion, umin_i, umax_i, vmin_i, vmax_i, wmin_i, wmax_i, tolerance_velocity_ion);
            conf_ion.u_min = umin_i;
            conf_ion.u_max = umax_i;
            conf_ion.v_min = vmin_i;
            conf_ion.v_max = vmax_i;
            conf_ion.w_min = wmin_i;
            conf_ion.w_max = wmax_i; */

            double timer_fill_restart_matrix = timer.elapsed();
            timer.reset();
            std::cout << "Filling restart matrix took " << timer_fill_restart_matrix << " s." << std::endl;

            /* kin_energy_electron = interpolant_electron.compute_kinetic_energy();
            kin_energy_ion = interpolant_electron.compute_kinetic_energy();
            kin_energy = kin_energy_electron + kin_energy_ion;
            kin_energy_file << n*conf_electron.dt << " " << kin_energy << " " << kin_energy_electron << " " << kin_energy_ion << std::endl; */

            // Copy last entries of coeff vectors.
            #pragma omp parallel for collapse(2)
            for(size_t k = 0; k < 3; k++){
                for(size_t ix = 0; ix < Nx_ext; ix++)
                for(size_t iy = 0; iy < Ny_ext; iy++)
                for(size_t iz = 0; iz < Nz_ext; iz++){
                    coeffs_E[idx_base(0,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,Nt)] 
                                    = coeffs_E[idx_base(nt_r_curr,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,Nt)];
                    coeffs_B[idx_base(0,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,Nt)] 
                                    = coeffs_B[idx_base(nt_r_curr,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,Nt)];
                    coeffs_B_staggered[idx_base(0,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,Nt)] 
                                    = coeffs_B_staggered[idx_base(nt_r_curr,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,Nt)];
                    coeffs_B_staggered[idx_base(1,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,Nt)] 
                                    = coeffs_B_staggered[idx_base(nt_r_curr+1,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,Nt)];
                }
            }


            conf_electron.f0 = &eval_f_electron_with_linear_interpolant;
            conf_ion.f0 = &eval_f_ion_with_linear_interpolant;

            nt_r_curr = 1;
            double timer_copy_coeff = timer.elapsed();
            double timer_restart = timer_fill_restart_matrix + timer_copy_coeff;
            std::cout << "Restart took: " << timer_restart << std::endl;
            total_time += timer_restart;
        } else {
            nt_r_curr++;
        }
    }

    std::cout << "Total simulation time (pure) " << total_time << " s." << std::endl;
    std::cout << "Total simulation time (with plotting) " << total_time_with_plot << " s." << std::endl;
}


std::vector<nufi::restart::flow_map_linear_interpolant_2x3v> char_maps_electron;
std::vector<nufi::restart::flow_map_linear_interpolant_2x3v> char_maps_ion;

size_t restart_counter = 0;

// This functions are general placeholds for the case that I want a function pointer as above
// or a restart with a linear interpolant of some f.
std::function<double(double,double,double,double,double,double)> restart_f0_electron;
std::function<double(double,double,double,double,double,double)> restart_f0_ion;

double eval_f_electron_cmm(double x, double y, double z, double u, double v, double w)
{
    for(size_t i = restart_counter; i > 0; i--){
        x = char_maps_electron[i].eval_flow_map(0, x, y, z, u, v, w);
        y = char_maps_electron[i].eval_flow_map(1, x, y, z, u, v, w);
        //z = char_maps_electron[i].eval_flow_map(2, x, y, z, u, v, w);
        u = char_maps_electron[i].eval_flow_map(3, x, y, z, u, v, w);
        v = char_maps_electron[i].eval_flow_map(4, x, y, z, u, v, w);
        w = char_maps_electron[i].eval_flow_map(5, x, y, z, u, v, w);
    }

    return restart_f0_electron(x,y,z,u,v,w);
}

double eval_f_ion_cmm(double x, double y, double z, double u, double v, double w)
{
    for(size_t i = restart_counter; i > 0; i--){
        x = char_maps_ion[i].eval_flow_map(0, x, y, z, u, v, w);
        y = char_maps_ion[i].eval_flow_map(1, x, y, z, u, v, w);
        //z = char_maps_ion[i].eval_flow_map(2, x, y, z, u, v, w);
        u = char_maps_ion[i].eval_flow_map(3, x, y, z, u, v, w);
        v = char_maps_ion[i].eval_flow_map(4, x, y, z, u, v, w);
        w = char_maps_ion[i].eval_flow_map(5, x, y, z, u, v, w);
    }

    return restart_f0_ion(x,y,z,u,v,w);
}


template<size_t order>
void nufi_cmm_maxwell_lie_EBf_predictor_corrector_aligned()
{
    // Set up config.
    config_t<double> conf_electron (Nx, Ny, Nz, Nu_e, Nv_e, Nw_e, Nt, dt, 
                            xmin, xmax, ymin, ymax, zmin, zmax, umin_e, umax_e, 
                            vmin_e, vmax_e, wmin_e, wmax_e, &f0_electron, pezzini::me, -1);
    conf_electron.print_config(std::cout);
    config_t<double> conf_ion (Nx, Ny, Nz, Nu_i, Nv_i, Nw_i, Nt, dt, 
                            xmin, xmax, ymin, ymax, zmin, zmax, umin_i, umax_i, 
                            vmin_i, vmax_i, wmin_i, wmax_i, &f0_ion, pezzini::mi, 1);
    conf_ion.print_config(std::cout);

    if(conf_electron.Nx != conf_ion.Nx){
        throw std::runtime_error("Nx must be equal for all species!");
    }
    if(conf_electron.Ny != conf_ion.Ny){
        throw std::runtime_error("Ny must be equal for all species!");
    }
    if(conf_electron.Nz != conf_ion.Nz){
        throw std::runtime_error("Nz must be equal for all species!");
    }

    // Storage of coefficients now via: 
    // index = nt + Nt * (d + dim * (ix + Nx * (iy + Ny * iz)))
    size_t stride_t = (Nx + order - 1) *
                        (Ny + order - 1) *
                        (Nz + order - 1);

    const size_t dim = 3;
    const size_t Nx_ext = Nx + order - 1;
    const size_t Ny_ext = Ny + order - 1;
    const size_t Nz_ext = Nz + order - 1;
    const size_t Nspace = Nx_ext * Ny_ext * Nz_ext;

    std::cout << "Start NuFI Vlasov-Maxwell-Solver with fBE-Lie-Splitting." << std::endl;
    std::cout << "Init helper variables." << std::endl;

    // Flattened 1D coefficient storage
    std::vector<double> coeffs_E(3 * (Nt + 1) * stride_t, 0);
    std::vector<double> coeffs_B_staggered(3 * (Nt + 2) * stride_t, 0); // 0 = -1/2, 1 = 1/2, 2 = 3/2, ... (index = n - 1/2)
    std::vector<double> coeffs_B(3 * (Nt + 1) * stride_t, 0);
    std::vector<double> E(3 * Nx * Ny * Nz, 0);
    std::vector<double> B(3 * Nx * Ny * Nz, 0);
    std::vector<double> j_0(3 * Nx * Ny * Nz, 0);
    std::vector<double> j_electron(3 * Nx * Ny * Nz, 0);
    std::vector<double> j_ion(3 * Nx * Ny * Nz, 0);
    std::vector<double> j_1(3 * Nx * Ny * Nz, 0);

    std::ofstream kin_energy_file( "kin_energy.txt" );

    // Init restart matrices.
    std::cout << "Initialize restart matrices." << std::endl;
    interpolant_electron = nufi::restart::linear_interpolant_6d (xmin, xmax, ymin, ymax, zmin, zmax, 
                                        umin_e, umax_e, vmin_e, vmax_e, wmin_e, wmax_e, 
                                        nx_r, ny_r, nz_r, nu_r_e, nv_r_e, nw_r_e);
    interpolant_ion = nufi::restart::linear_interpolant_6d (xmin, xmax, ymin, ymax, zmin, zmax,
                                        umin_i, umax_i, vmin_i, vmax_i, wmin_i, wmax_i, 
                                        nx_r, ny_r, nz_r, nu_r_i, nv_r_i, nw_r_i);

    // Init f0 with random perturbation.
    std::cout << "Random f0_electron perturbation. " << std::endl;
    random_perturbation_2d_f0_electron(interpolant_electron.restart_matrix, 1e-2);
    std::cout << "Random f0_ion perturbation. " << std::endl;
    random_perturbation_2d_f0_ion(interpolant_ion.restart_matrix, 1e-2);

    double kin_energy_electron = interpolant_electron.compute_kinetic_energy();
    double kin_energy_ion = interpolant_electron.compute_kinetic_energy();
    double kin_energy = kin_energy_electron + kin_energy_ion;

    kin_energy_file << 0*conf_electron.dt << " " << kin_energy << " " << kin_energy_electron << " " << kin_energy_ion << std::endl;

    conf_electron.f0 = &eval_f_electron_with_linear_interpolant;
    restart_f0_electron = eval_f_electron_with_linear_interpolant;
    conf_ion.f0 = &eval_f_ion_with_linear_interpolant;
    restart_f0_ion = eval_f_ion_with_linear_interpolant;

    std::vector<double> coeffs_phi(stride_t, 0);
    std::vector<double> rho_e(Nx*Ny*Nz, 0);
    std::vector<double> rho_i(Nx*Ny*Nz, 0);
    std::vector<double> rho(Nx*Ny*Nz, 0);
    
    std::cout << "Compute rho_0. " << std::endl;
    eval_rho_full_EBf<double,order>(0, rho_e, coeffs_E, coeffs_B, conf_electron);
    eval_rho_full_EBf<double,order>(0, rho_i, coeffs_E, coeffs_B, conf_ion);

    #pragma omp parallel for
    for(size_t i = 0; i < rho.size(); i++){
        rho[i] = rho_i[i] + rho_e[i];
    }

    std::cout << "Compute phi_0. " << std::endl;
    interpolate<double,order>(coeffs_phi.data(), rho.data(), conf_electron );

    auto eval_E0_random_pertb = [&](double x, double y, double z, size_t d) {
        if(d == 0){
            return -eval<double,order,1,0,0>(x,y,z, coeffs_phi.data(), conf_electron);
        } 
        if(d == 1 ){
            return -eval<double,order,0,1,0>(x,y,z, coeffs_phi.data(), conf_electron);
        }
        return -eval<double,order,0,0,1>(x,y,z, coeffs_phi.data(), conf_electron);
    };

    double tolerance = 1e-6;
    double tolerance_velocity_electron = tolerance * interpolant_electron.restart_matrix.max();
    double tolerance_velocity_ion = tolerance * interpolant_ion.restart_matrix.max();

    size_t nt_r_curr = 1;

    auto eval_f_electron = [&](double x, double y, double z, double u, double v, double w) {
        return eval_f_lie_EBf<double, order>( nt_r_curr, x, y, z, u, v, w, coeffs_E, coeffs_B, conf_electron);
    };
    auto eval_f_ion = [&](double x, double y, double z, double u, double v, double w) {
        return eval_f_lie_EBf<double, order>( nt_r_curr, x, y, z, u, v, w, coeffs_E, coeffs_B, conf_ion);
    };

    //std::random_device dev;
    //std::mt19937 rng(dev());
    //std::uniform_int_distribution<std::mt19937::result_type> dist(1e-8,1e-6);    

    // Compute E(0) and B(0).
    #pragma omp parallel for
    for(size_t l = 0; l < Nx*Ny*Nz; l++){
        size_t iz   = l   / (Nx * Ny);
        size_t tmp  = l   % (Nx * Ny);
        size_t iy   = tmp / Nx;
        size_t ix   = tmp % Nx;
    
        double x = xmin + ix*conf_electron.dx; 
        double y = ymin + iy*conf_electron.dy; 
        double z = zmin + iz*conf_electron.dz; 
        
        // Normal initialization:        
        //arma::Col<double> E0_vec = E0(x,y,z);
        arma::Col<double> B0_vec = B0(x,y,z);

         for(size_t d = 0; d < 3; d++){
            size_t index = d*Nx*Ny*Nz + l;
            //E[index] = E0_vec(d);
            E[index] = eval_E0_random_pertb(x,y,z,d);
            B[index] = B0_vec(d);
        }
    }

    // Interpolate E(0) and B(0).
    interpolate_fields_aligned<double,order>(0, coeffs_E, E, conf_electron);
    interpolate_fields_aligned<double,order>(0, coeffs_B, B, conf_electron);

    std::ofstream coeff_E_str("coeff_E.txt");
    std::ofstream coeff_B_str("coeff_B.txt");
    analysis::write_coeffs_nufi_pc<double,order>(0,coeffs_E,coeffs_B,conf_electron,coeff_E_str,coeff_B_str);

    // Compute B_{-1/2}.
    #pragma omp parallel for
    for(size_t l = 0; l < Nx*Ny*Nz; l++){
        size_t iz   = l   / (Nx * Ny);
        size_t tmp  = l   % (Nx * Ny);
        size_t iy   = tmp / Nx;
        size_t ix   = tmp % Nx;
    
        double x = xmin + ix*conf_electron.dx; 
        double y = ymin + iy*conf_electron.dy; 
        double z = zmin + iz*conf_electron.dz; 
        
        // Normal initialization:        
        arma::Col<double> B0_vec ({
                                eval<double,order>(x,y,z,coeffs_B.data() + idx_base(0,0,0,0,0,Nx_ext,Ny_ext,Nz_ext,Nt),conf_electron),
                                eval<double,order>(x,y,z,coeffs_B.data() + idx_base(0,1,0,0,0,Nx_ext,Ny_ext,Nz_ext,Nt),conf_electron),
                                eval<double,order>(x,y,z,coeffs_B.data() + idx_base(0,2,0,0,0,Nx_ext,Ny_ext,Nz_ext,Nt),conf_electron)
                            });
        arma::Col<double> rot_E0_vec = rot<double,order>(0,x,y,z,coeffs_E,conf_electron);
        B0_vec = B0_vec + 0.5*conf_electron.dt*rot_E0_vec;

        for(size_t d = 0; d < 3; d++){
            size_t index = d*Nx*Ny*Nz + l;
            B[index] = B0_vec(d);
        }
    }

    // Interpolate B(-1/2).
    interpolate_fields_aligned<double,order>(0, coeffs_B_staggered, B, conf_electron);

    // Compute j(0).
    eval_j_full_EBf<double,order>(0, j_electron, coeffs_E, coeffs_B, conf_electron);
    eval_j_full_EBf<double,order>(0, j_ion, coeffs_E, coeffs_B, conf_ion);

    #pragma omp parallel for
    for(size_t i = 0; i < j_0.size(); i++){
        j_0[i] = j_ion[i] + j_electron[i];
    }

    // Compute B(1/2).
    maxwell::B_step_predictor_corrector<double,order>(1,coeffs_E,coeffs_B_staggered,B,conf_electron);

    // Do first output.
    std::ofstream stat_file( "stats.txt" );
    std::ofstream j_stat_file( "j_stats.txt" );
    analysis::do_stats<double,order>(0, 64, 64, 1, stat_file, coeffs_E, coeffs_B, conf_electron, true, true, 0);
    analysis::compute_j_l2(0,j_stat_file,conf_electron,j_0,j_electron,j_ion);
    {
        std::ofstream mat_e_str("f_e_full_" + std::to_string(0*conf_electron.dt) + ".txt" );
        std::ofstream mat_i_str("f_i_full_" + std::to_string(0*conf_electron.dt) + ".txt" );
        analysis::plot_full_f_3x3v_parallelized<double,order>(2,2,1,128,128,8,xmin,xmax,ymin,ymax,zmin,zmax,umin_e,umax_e,vmin_e,vmax_e,wmin_e,wmax_e,eval_f_electron,mat_e_str);
        analysis::plot_full_f_3x3v_parallelized<double,order>(2,2,1,128,128,8,xmin,xmax,ymin,ymax,zmin,zmax,umin_i,umax_i,vmin_i,vmax_i,wmin_i,wmax_i,eval_f_ion,mat_i_str);

        std::ofstream j_e_str("j_e_" + std::to_string(0*conf_electron.dt) + ".txt" );
        for(size_t i = 0; i < j_electron.size(); i++){
            j_e_str << j_electron[i] << std::endl;
        }
        std::ofstream j_i_str("j_i_" + std::to_string(0*conf_electron.dt) + ".txt" );
        for(size_t i = 0; i < j_ion.size(); i++){
            j_i_str << j_ion[i] << std::endl;
        }
    }

    std::cout << "Time-loop." << std::endl;    
    std::cout << " ---------------------------------- " << std::endl;
    double total_time, total_time_with_plot = 0;
    for(size_t n = 1; n <= Nt; n++)
    {
        nufi::stopwatch<double> timer;
        
        // Compute j(n).
        eval_j_full_EBf<double,order>(nt_r_curr, j_electron, coeffs_E, coeffs_B, conf_electron);
        eval_j_full_EBf<double,order>(nt_r_curr, j_ion, coeffs_E, coeffs_B, conf_ion);

        #pragma omp parallel for
        for(size_t i = 0; i < j_1.size(); i++){
            j_1[i] = j_ion[i] + j_electron[i];
        }

        // Compute E(n).
        maxwell::E_step_predictor_corrector<double,order>(nt_r_curr,coeffs_E,coeffs_B_staggered,E,j_0,j_1,conf_electron);

        // Compute B(n+1/2).
        maxwell::B_step_predictor_corrector<double,order>(nt_r_curr+1,coeffs_E, coeffs_B_staggered, B,conf_electron);

        // Compute B(n).
        maxwell::B_average<double,order>(nt_r_curr,coeffs_B,coeffs_B_staggered,B,conf_electron);

        // Shuffle j_1 into j_0.
        j_0 = j_1;

        double time_for_step = timer.elapsed();
        timer.reset();
        
        bool plot_EB = (n % (5*steps_per_1) == 0);
        bool plot_f_j = (n % (1*steps_per_1) == 0);
        analysis::do_stats<double,order>(nt_r_curr, 64, 64, 1, stat_file, coeffs_E, coeffs_B, conf_electron, plot_EB, true, n);
        analysis::compute_j_l2(n, j_stat_file, conf_electron, j_0, j_electron, j_ion);
        if(plot_f_j){
            std::ofstream mat_e_str("f_e_full_" + std::to_string(n*conf_electron.dt) + ".txt" );
            analysis::plot_full_f_3x3v_parallelized<double,order>(2,2,1,128,128,8,xmin,xmax,ymin,ymax,zmin,zmax,umin_e,umax_e,vmin_e,vmax_e,wmin_e,wmax_e,eval_f_electron,mat_e_str);
            std::ofstream velo_supp_e_str("v_supp_e_" + std::to_string(n*conf_electron.dt) + ".txt" );
            velo_supp_e_str << umin_e << " " << umax_e << " " << vmin_e << " " << vmax_e << " " << wmin_e << " " << wmax_e;
            
            std::ofstream mat_i_str("f_i_full_" + std::to_string(n*conf_electron.dt) + ".txt" );
            analysis::plot_full_f_3x3v_parallelized<double,order>(2,2,1,128,128,8,xmin,xmax,ymin,ymax,zmin,zmax,umin_i,umax_i,vmin_i,vmax_i,wmin_i,wmax_i,eval_f_ion,mat_i_str);
            std::ofstream velo_supp_i_str("v_supp_i_" + std::to_string(n*conf_electron.dt) + ".txt" );
            velo_supp_i_str << umin_i << " " << umax_i << " " << vmin_i << " " << vmax_i << " " << wmin_i << " " << wmax_i;

            std::ofstream j_e_str("j_e_" + std::to_string(n*conf_electron.dt) + ".txt" );
            for(size_t i = 0; i < j_electron.size(); i++){
                j_e_str << j_electron[i] << std::endl;
            }
            std::ofstream j_i_str("j_i_" + std::to_string(n*conf_electron.dt) + ".txt" );
            for(size_t i = 0; i < j_ion.size(); i++){
                j_i_str << j_ion[i] << std::endl;
            }
        }
                
        double time_for_plot = timer.elapsed();
        std::cout << "Plotting took " << time_for_plot << " s." << std::endl;
        analysis::write_coeffs_nufi_pc<double,order>(nt_r_curr,coeffs_E,coeffs_B,conf_electron,coeff_E_str,coeff_B_str);

        total_time += time_for_step;
        total_time_with_plot += time_for_step + time_for_plot;
        std::cout << "Time step " << n << " took a total of " << time_for_step << " s. So far total time = " << total_time << " s." << std::endl;

        if(nt_r_curr == nt_restart){
            timer.reset();
            std::cout << "Restart simulation. " << std::endl;
            auto flow_electron = [&](double x, double y, double z, double u, double v, double w){
                eval_flow_map_EBf<double,order>(nt_r_curr, x, y, z, u, v, w, 
                                        coeffs_E, coeffs_B, conf_electron);
            };
            char_maps_electron.push_back(nufi::restart::flow_map_linear_interpolant_2x3v(xmin, xmax, 
                        ymin, ymax, zmin, zmax, umin_e, umax_e, vmin_e, vmax_e, wmin_e, wmax_e,
                        nx_r, ny_r, nz_r, nu_r_e, nv_r_e, nw_r_e, flow_electron));

            auto flow_ion = [&](double x, double y, double z, double u, double v, double w){
                eval_flow_map_EBf<double,order>(nt_r_curr, x, y, z, u, v, w, 
                                        coeffs_E, coeffs_B, conf_ion);
            };
            char_maps_ion.push_back(nufi::restart::flow_map_linear_interpolant_2x3v(xmin, xmax, 
                        ymin, ymax, zmin, zmax, umin_i, umax_i, vmin_i, vmax_i, wmin_i, wmax_i,
                        nx_r, ny_r, nz_r, nu_r_i, nv_r_i, nw_r_i, flow_ion));


            double timer_fill_restart_matrix = timer.elapsed();
            timer.reset();
            std::cout << "Filling restart matrix took " << timer_fill_restart_matrix << " s." << std::endl;

            // Copy last entries of coeff vectors.
            #pragma omp parallel for collapse(2)
            for(size_t k = 0; k < 3; k++){
                for(size_t ix = 0; ix < Nx_ext; ix++)
                for(size_t iy = 0; iy < Ny_ext; iy++)
                for(size_t iz = 0; iz < Nz_ext; iz++){
                    coeffs_E[idx_base(0,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,Nt)] 
                                    = coeffs_E[idx_base(nt_r_curr,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,Nt)];
                    coeffs_B[idx_base(0,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,Nt)] 
                                    = coeffs_B[idx_base(nt_r_curr,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,Nt)];
                    coeffs_B_staggered[idx_base(0,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,Nt)] 
                                    = coeffs_B_staggered[idx_base(nt_r_curr,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,Nt)];
                    coeffs_B_staggered[idx_base(1,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,Nt)] 
                                    = coeffs_B_staggered[idx_base(nt_r_curr+1,k,ix,iy,iz,Nx_ext,Ny_ext,Nz_ext,Nt)];
                }
            }


            conf_electron.f0 = &eval_f_electron_cmm;
            conf_ion.f0 = &eval_f_ion_cmm;

            nt_r_curr = 1;
            restart_counter++;
            double timer_copy_coeff = timer.elapsed();
            double timer_restart = timer_fill_restart_matrix + timer_copy_coeff;
            std::cout << "Restart took: " << timer_restart << std::endl;
            total_time += timer_restart;
        } else {
            nt_r_curr++;
        }
    }

    std::cout << "Total simulation time (pure) " << total_time << " s." << std::endl;
    std::cout << "Total simulation time (with plotting) " << total_time_with_plot << " s." << std::endl;
}



}

}

int main(int argc, char** argv){

    // Todo:
    // Add MPI parallelization.
    // Add compressed restart.
    // Add CMM-restart.
    // Add initialition through init-file.

    /* MPI_Init(&argc, &argv);
    nufi::dim3::periodically_restarted_nufi_maxwell_lie_EBf_predictor_corrector_aligned<4>();
    MPI_Finalize(); */
    nufi::dim3::nufi_cmm_maxwell_lie_EBf_predictor_corrector_aligned<4>();

    return 0;
}