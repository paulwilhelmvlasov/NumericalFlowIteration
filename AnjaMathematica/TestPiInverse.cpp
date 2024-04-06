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
    conf.u_min = conf.v_min = conf.w_min = -10; // ausprobieren
    conf.u_max = conf.v_max = conf.w_max = 10;

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

    eval_moments<double, order>(1, 1, &coeffs[0], conf, &moments[0] );

    for (int i = 0; i < 199; i++) {
			moments[i] = 0;
	}

    moments[0] = 1.;
    moments[2] = -0.1369306394;
    moments[3] = -0.08451542547;
    moments[18] = 0.8660254038;
    moments[23] = 0.4629100499;
    moments[28] = 0.1735912687;
    moments[33] = 0.04273521617;
    moments[68] = 0.2195775164;
    moments[77] = 0.1872563352;
    moments[86] = 0.1201009893;
    moments[95] = 0.06569386817;
    moments[150] = 0.03310255611;
    moments[163] = 0.03626203338;
    moments[176] = 0.02968256789;
    moments[189] = 0.02079872197;

    moments[196] = 1;
    moments[200] = 1;


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
		double func = eval_f<double, order> (n, 0, 0, 0, u, v , w , &coeffs[0], conf ) - pi_inverse<double, order>(u, v, w, &moments[0]);

		accum += func * func * du * dv * dw;  
	}
    std::cout << "pi_inv(0, 0, 0) = " << pi_inverse<double, order>(0, 0, 0, &moments[0]) << std::endl;
    std::cout << "pi_inv(0, 0, sqrt(3/2)) = " << pi_inverse<double, order>(0, 0, 1.22474, &moments[0]) << std::endl;
    std::cout << "pi_inv(0, 0, -sqrt(3/2)) = " << pi_inverse<double, order>(0, 0, -1.22474, &moments[0]) << std::endl;
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
    return 0;
}