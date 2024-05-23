#include <vector>
#include <iostream>
#include <cmath>

// for testing: remove #include <dergeraet/config.hpp> from CalculateMoments_<<theory>>.cpp and add the following statement
// #include "test_def.hpp"

// terminal commands:
/*
g++ -c -std=c++17 -O2 CalculateMoments_<<theory>>.cpp
g++ -c -std=c++17 -O2 TestCalculateMoments.cpp
g++ TestCalculateMoments.o CalculateMoments_<<theory>>.o -o TestMoments
./TestMoments

e.g. for 3333333:
g++ -c -std=c++17 -O2 CalculateMoments_3333333.cpp
g++ -c -std=c++17 -O2 TestCalculateMoments.cpp
g++ TestCalculateMoments.o CalculateMoments_3333333.o -o TestMoments
./TestMoments
*/

#include "../AnjaMathematica/CalculateMoments_generated/CalculateMoments_Full2.cpp"
/*
#include "../AnjaMathematica/CalculateMoments_3333333.cpp"
#include "../AnjaMathematica/CalculateMoments_444444444.cpp"
#include "../AnjaMathematica/CalculateMoments_Full10.cpp"
*/

double runde (double value, int precision)
{
    double multiplier = std::pow(10, precision);
    return (double)((int)(value * multiplier + 0.5f))/multiplier;
}

void max_norm (const std::vector<double> moments, const std::vector<double> orig) {
    double max = 0;
    double pos;
    for(int i = 0; i < orig.size(); i++) {
        double zsp = abs(orig[i]-moments[i]);
        if(zsp > max) {
            max = zsp;
            pos = i;
        }
    }
    std::cout << "max_norm (discontinuous f only!) = " << max << std::endl;
    std::cout << "pos = " << pos << std::endl;
}

void calculate_moments (const std::vector<double> orig, int theory) {
    std::vector<double> moments;
    moments.resize(orig.size() + 5);

    std::vector<double> coeffs; // aktuell egal
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

    const size_t order = 1; // aktuell egal

    switch (theory)
    {
    case 1:
        cm_21100::calculate_moments<double, order>(n, l, &coeffs[0], conf, &moments[0]);
        break;
        /*
    case 2:
        cm_3333333::calculate_moments<double, order>(n, l, &coeffs[0], conf, &moments[0]);
        break;
    case 3:
        cm_444444444::calculate_moments<double, order>(n, l, &coeffs[0], conf, &moments[0]);
        break;
    case 4:
        cm_1099887766554433221100::calculate_moments<double, order>(n, l, &coeffs[0], conf, &moments[0]);
        break;
        */
    default:
        cm_21100::calculate_moments<double, order>(n, l, &coeffs[0], conf, &moments[0]);
        break;
    }

    for(int i = 0; i < moments.size(); i++) {
        std::cout << "moments[" << i << "] = ";
        if(abs(moments[i]) < 1) {
            std::cout << runde(moments[i],6) << ";" << std::endl;
        } else {
            std::cout << moments[i] << ";" << std::endl;
        }
    }

    if(orig[0] == 1.0) {
        max_norm(moments, orig);
    }
}

void Full2 () {
    std::vector<double> orig;
    orig.resize(35);
    orig[0] = 1.0;
    orig[1] = 0;
    orig[2] = -0.1914778361;
    orig[3] = 0;
    orig[4] = 0;
    orig[5] = 0;
    orig[6] = -0.2797288653;
    orig[7] = -0.02153353809;
    orig[8] = -0.03936924771;
    orig[9] = -0.02205882353;
    orig[10] = 0;
    orig[11] = -0.6749903882;
    orig[12] = -0.2292420186;
    orig[13] = -0.8823529412;
    orig[14] = 0.008669806723;
    orig[15] = 0.002702978232;
    orig[16] = -0.3849241224;
    orig[17] = -0.3207534169;
    orig[18] = -0.281075134;
    orig[19] = -0.232727199;
    orig[20] = -0.03404751323;
    orig[21] = -0.1554049252;
    orig[22] = -0.02637309034;
    orig[23] = -0.03522511637;
    orig[24] = 0.03404751323; 
    orig[25] = -0.232727199;
    orig[26] = 0.03371206157;
    orig[27] = -0.006193301176;
    orig[28] = 0.01667073307;
    orig[29] = 0.002340847815;
    orig[30] = 0.3411191802;
    orig[31] = 0.02809017378;
    orig[32] = 0.5398344391;
    orig[33] = -0.03715980706;
    orig[34] = 0.6971373398;

    calculate_moments(orig, 1);
}


int main ()
{
    Full2();

    return 0;
}
