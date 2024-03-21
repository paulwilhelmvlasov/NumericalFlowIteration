#include <vector>
#include <iostream>
#include <cmath>

// for testing: remove #include <dergeraet/config.hpp> from CalculateMoments_<<theory>>.cpp and add the following statement
// #include "test_def.hpp"

// terminal commands:
/*
g++ -c -std=c++20 CalculateMoments_<<theory>>.cpp
g++ -c -std=c++20 TestPiInverse.cpp
g++ TestPiInverse.o CalculateMoments_<<theory>>.o -o Test
./Test
*/

// Select theory
#include "CalculateMoments_3333333.cpp"
// #include "CalculateMoments_444444444.cpp"
// #include "CalculateMoments_Full10.cpp"

int main () {
    std::vector<double> coeffs; // aktuell egal
    std::vector<double> moments;
    moments.resize(201);

    config_t<double> conf;
    conf.Nx = conf.Ny = conf.Nz = 1;  // Number of grid points in physical space.
    conf.Nu = conf.Nv = conf.Nw = 50;  // Number of quadrature points in velocity space.
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
    conf.du = conf.dv = conf.dw = 0.1; // ausprobieren

    size_t n = 1; // aktuell egal
    size_t l = 1; // aktuell egal

    constexpr size_t order = 1; // aktuell egal

    eval_moments<double, order>(n, l, &coeffs[0], conf, &moments[0]);

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
		double func = eval_f<double, order> (n, 0, 0, 0, u * moments[200] + moments[197], v * moments[200] + moments[198], w * moments[200] + moments[199], &coeffs[0], conf ) - pi_inverse<double, order>(u, v, w, &moments[0]);
        
        double zsp = accum; 
		accum += func * func * du * dv * dw;

        if(accum - zsp > 1) {
            std::cout << "(" << u * moments[200] + moments[197] << ", " << v * moments[200] + moments[198] << ", " << w * moments[200] + moments[199] << ") " << func << std::endl;
        }
        
	}

    accum = std::sqrt(accum);

    double u = 0;
    double v = 10;
    double w = 5;
    std::cout << accum << std::endl;
    std::cout << pi_inverse<double, order>(u, v, w, &moments[0]) << std::endl;
    std::cout << eval_f<double, order> (n, 0, 0, 0, u, v, w, &coeffs[0], conf ) << std::endl;
    return 0;
}