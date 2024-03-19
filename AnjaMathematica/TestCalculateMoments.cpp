//#include "CalculateMoments_444444444.cpp" // 4444444444
//#include "CalculateMoments_3333333.cpp" // 3333333
#include "CalculateMoments_Full10.cpp" // Full10
#include <vector>
#include <iostream>
#include <cmath>

double runde(double value, int precision)
{
  double multiplier = std::pow(10, precision);
  return (double)((int)(value * multiplier + 0.5f))/multiplier;
}

int main () {
    std::vector<double> coeffs; // aktuell egal
    std::vector<double> moments;
    //moments.resize(410); // 4444444444
    //moments.resize(201); // 3333333
    moments.resize(1776); // Full10

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
    conf.du = conf.dv = conf.dw = 0.5; // ausprobieren

    size_t n = 1; // aktuell egal
    size_t l = 1; // aktuell egal

    constexpr size_t order = 1; // aktuell egal

    eval_moments<double, order>(n, l, &coeffs[0], conf, &moments[0] );

    for(int i = 0; i < moments.size(); i++) {
        std::cout << "moments[" << i << "] = " << runde(moments[i],6)  << std::endl;
    }
}