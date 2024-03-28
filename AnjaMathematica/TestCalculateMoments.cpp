#include <vector>
#include <iostream>
#include <cmath>

// for testing: remove #include <dergeraet/config.hpp> from CalculateMoments_<<theory>>.cpp and add the following statement
// #include "test_def.hpp"

// terminal commands:
/*
g++ -c -std=c++20 CalculateMoments_<<theory>>.cpp
g++ -c -std=c++20 TestCalculateMoments.cpp
g++ TestCalculateMoments.o CalculateMoments_<<theory>>.o -o Test
./Test
*/

// Select theory
#include "CalculateMoments_3333333.cpp"
// #include "CalculateMoments_444444444.cpp"
// #include "CalculateMoments_Full10.cpp"

double runde(double value, int precision)
{
    double multiplier = std::pow(10, precision);
    return (double)((int)(value * multiplier + 0.5f))/multiplier;
}

void gauss3333333 () {
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
    conf.u_min = conf.v_min = conf.w_min = -5; // ausprobieren
    conf.u_max = conf.v_max = conf.w_max = 5;

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

    eval_moments<double, order>(n, l, &coeffs[0], conf, &moments[0] );
    for(int i = 0; i < moments.size(); i++) {
        std::cout << "moments[" << i << "] = ";
        if(abs(moments[i]) < 0.5) {
            std::cout << runde(moments[i],6)  << std::endl;
        } else {
            std::cout << moments[i]  << std::endl;
        }
    }
}

void gauss444444444 () {
    std::vector<double> coeffs; // aktuell egal
    std::vector<double> moments;
    moments.resize(410);

    config_t<double> conf;
    conf.Nx = conf.Ny = conf.Nz = 1;  // Number of grid points in physical space.
    conf.Nu = conf.Nv = conf.Nw = 20;  // Number of quadrature points in velocity space.
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

    eval_moments<double, order>(n, l, &coeffs[0], conf, &moments[0] );

    for(int i = 0; i < moments.size(); i++) {
        std::cout << "moments[" << i << "] = " << runde(moments[i],6)  << std::endl;
    }
}

void Full10 () {
    std::vector<double> coeffs; // aktuell egal
    std::vector<double> moments;
    moments.resize(1776);

    config_t<double> conf;
    conf.Nx = conf.Ny = conf.Nz = 1;  // Number of grid points in physical space.
    conf.Nu = conf.Nv = conf.Nw = 20;  // Number of quadrature points in velocity space.
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

    eval_moments<double, order>(n, l, &coeffs[0], conf, &moments[0] );

    for(int i = 0; i < moments.size(); i++) {
        std::cout << "moments[" << i << "] = " << runde(moments[i],6)  << std::endl;
    }
}

int main () {

    gauss3333333();

    // gauss444444444();

    // Full10();

    return 0;
}