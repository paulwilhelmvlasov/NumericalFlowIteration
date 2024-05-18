#include <vector>
#include <iostream>
#include <cmath>

// for testing: remove #include <dergeraet/config.hpp> from CalculateMoments_<<theory>>.cpp and add the following statement
// #include "test_def.hpp"

// terminal commands:
/*
g++ -c -std=c++17 -O2 CalculateMoments_<<theory>>.cpp
g++ -c -std=c++17 -O2 TestPiInverse.cpp
g++ TestPiInverse.o CalculateMoments_<<theory>>.o -o TestInverse
./TestInverse

e.g. for 3333333:
g++ -c -std=c++17 -O2 CalculateMoments_3333333.cpp
g++ -c -std=c++17 -O2 TestPiInverse.cpp
g++ TestPiInverse.o CalculateMoments_3333333.o -o TestInverse
./TestInverse
*/

#include "../CalculateMoments_generated/CalculateMoments_Full2.cpp"
#include "../CalculateMoments_generated/CalculateMoments_3333333.cpp"
#include "../CalculateMoments_generated/CalculateMoments_444444444.cpp"
#include "../CalculateMoments_generated/CalculateMoments_Full10.cpp"

void calculate (int theory) {
    std::vector<double> coeffs; // aktuell egal
    std::vector<double> moments;

    config_t<double> conf;
    conf.Nx = conf.Ny = conf.Nz = 1;  // Number of grid points in physical space.
    conf.Nu = conf.Nv = conf.Nw = 80;  // Number of quadrature points in velocity space.
    //conf.Nt = 50;  // Number of time-steps.
    //conf.dt = 1;  // Time-step size.

    // Dimensions of physical domain.
    conf.x_min = conf.y_min = conf.z_min = 0; // aktuell egal
    conf.x_max = conf.y_max = conf.z_max = 1; // aktuell egal

    // Integration limits for velocity space.
    conf.u_min = conf.v_min = conf.w_min = -20; // ausprobieren
    conf.u_max = conf.v_max = conf.w_max = 20;

    // Grid-sizes and their reciprocals.
    conf.dx = conf.dy = conf.dz = 1; // aktuell egal
    //conf.dx_inv = conf.dy_inv = conf.dz_inv = 1;
    //conf.Lx = conf.Ly = conf.Lz = 1;
    //conf.Lx_inv = conf.Ly_inv = conf.Lz_inv = 1;
    
    conf.du = (conf.u_max-conf.u_min) / conf.Nu; 
	conf.dv = (conf.v_max-conf.v_min) / conf.Nv;
	conf.dw = (conf.w_max-conf.w_min) / conf.Nw;

    size_t n = 1; // aktuell egal
    size_t l = 1; // aktuell egal

    constexpr size_t order = 1; // aktuell egal

    switch (theory)
    {
    case 1:
        moments.resize(40);
        cm_21100::calculate_moments<double, order>(n, l, &coeffs[0], conf, &moments[0]);
        break;
    case 2:
        moments.resize(201);
        cm_3333333::calculate_moments<double, order>(n, l, &coeffs[0], conf, &moments[0]);
        break;
    case 3:
        moments.resize(410);
        cm_444444444::calculate_moments<double, order>(n, l, &coeffs[0], conf, &moments[0]);
        break;
    case 4:
        moments.resize(1776);
        cm_1099887766554433221100::calculate_moments<double, order>(n, l, &coeffs[0], conf, &moments[0]);
        break;
    default:
        moments.resize(40);
        cm_21100::calculate_moments<double, order>(n, l, &coeffs[0], conf, &moments[0]);
        break;
    }

    const double du = (conf.u_max-conf.u_min) / conf.Nu; 
	const double dv = (conf.v_max-conf.v_min) / conf.Nv;
	const double dw = (conf.w_max-conf.w_min) / conf.Nw;

    const double u_min = conf.u_min + 0.5*du;
	const double v_min = conf.v_min + 0.5*dv;
	const double w_min = conf.w_min + 0.5*dw;

    double accum = 0.;
    
    for ( size_t kk = 0; kk < conf.Nw; ++kk )
	for ( size_t jj = 0; jj < conf.Nv; ++jj )
	for ( size_t ii = 0; ii < conf.Nu; ++ii )
	{
		double u = u_min + ii*du;
		double v = v_min + jj*dv;
		double w = w_min + kk*dw;

        double func;
        switch (theory)
        {
        case 1:
            func = eval_f<double, order> (n, 0, 0, 0, u, v , w , &coeffs[0], conf ) - cm_21100::pi_inverse<double>(u, v, w, &moments[0]);
    
            break;
        case 2:
            func = eval_f<double, order> (n, 0, 0, 0, u, v , w , &coeffs[0], conf ) - cm_3333333::pi_inverse<double>(u, v, w, &moments[0]);
            break;
        case 3:
            func = eval_f<double, order> (n, 0, 0, 0, u, v , w , &coeffs[0], conf ) - cm_444444444::pi_inverse<double>(u, v, w, &moments[0]);
            break;
        case 4:
            func = eval_f<double, order> (n, 0, 0, 0, u, v , w , &coeffs[0], conf ) - cm_1099887766554433221100::pi_inverse<double>(u, v, w, &moments[0]);
            break;
        default:
            func = eval_f<double, order> (n, 0, 0, 0, u, v , w , &coeffs[0], conf ) - cm_21100::pi_inverse<double>(u, v, w, &moments[0]);
            break;
        }

		accum += func * func * du * dv * dw;  
	}
    
    accum = std::sqrt(accum);

    double accumf = 0.;
    
    for ( size_t kk = 0; kk < conf.Nw; ++kk )
	for ( size_t jj = 0; jj < conf.Nv; ++jj )
	for ( size_t ii = 0; ii < conf.Nu; ++ii )
	{
		double u = u_min + ii*du;
		double v = v_min + jj*dv;
		double w = w_min + kk*dw;
		double func = eval_f<double, order> (n, 0, 0, 0, u, v, w, &coeffs[0], conf );

		accumf += func * func * du * dv * dw;  
	}

    accumf = std::sqrt(accumf);

    accum /= accumf;

    std::cout << "L_2-Norm: " << accum << std::endl;
}

int main () {
    bool cont = true;
    int choice; 
    char con;
    while (cont) {
        std::cout << "Select a theory (select number): 1) Full 2; 2) 3333333; 3) 444444444; 4) Full10" << std::endl;
        std::cin >> choice;
        calculate(choice);
        std::cout << "Continue? [y/n]" << std::endl;
        std::cin >> con;
        if (con == 'n') cont = false;
    }
    return 0;
}