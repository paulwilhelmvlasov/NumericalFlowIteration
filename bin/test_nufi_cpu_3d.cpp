/*
 * Copyright (C) 2022 Matthias Kirchhart and Paul Wilhelm
 *
 * This file is part of Der Gerät, a solver for the Vlasov–Poisson equation.
 *
 * Der Gerät is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3, or (at your option) any later
 * version.
 *
 * Der Gerät is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
 * details.
 *
 * You should have received a copy of the GNU General Public License along with
 * Der Gerät; see the file COPYING.  If not see http://www.gnu.org/licenses.
 */

#include <cmath>
#include <memory>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>


#include <dergeraet/config.hpp>
#include <dergeraet/random.hpp>
#include <dergeraet/fields.hpp>
#include <dergeraet/poisson.hpp>
#include <dergeraet/rho.hpp>
#include <dergeraet/stopwatch.hpp>

namespace dergeraet
{

namespace dim3
{

template <typename real, size_t order>
void calculate_moments( size_t n, size_t l, const real *coeffs, const config_t<real> &conf, real* moments )
{
	const size_t k   = l   / (conf.Nx * conf.Ny);
	const size_t tmp = l   % (conf.Nx * conf.Ny);
	const size_t j   = tmp / conf.Nx;
	const size_t i   = tmp % conf.Nx;

	const real x = conf.x_min + i*conf.dx;
	const real y = conf.y_min + j*conf.dy;
	const real z = conf.z_min + k*conf.dz;

	const real du = (conf.u_max-conf.u_min) / conf.Nu;
	const real dv = (conf.v_max-conf.v_min) / conf.Nv;
	const real dw = (conf.w_max-conf.w_min) / conf.Nw;

	const real u_min = conf.u_min + 0.5*du;
	const real v_min = conf.v_min + 0.5*dv;
	const real w_min = conf.w_min + 0.5*dw;

	real rho = 0;
	real j_x = 0;
	real j_y = 0;
	real j_z = 0;
	real std_dev = 0;

	std::vector<std::vector<std::vector<real> > > f(conf.Nu, std::vector<std::vector<real> >(conf.Nv, std::vector<real>(conf.Nw)));

	for ( size_t kk = 0; kk < conf.Nw; ++kk )
	for ( size_t jj = 0; jj < conf.Nv; ++jj )
	for ( size_t ii = 0; ii < conf.Nu; ++ii )
	{
		real u = u_min + ii*du;
		real v = v_min + jj*dv;
		real w = w_min + kk*dw;
		real func = eval_f<real,order>( n, x, y, z, u, v, w, coeffs, conf );

		rho += func * du * dv * dw;
		j_x += func * u * du * dv * dw;
		j_y += func * v * du * dv * dw;
		j_z += func * w * du * dv * dw;

		f[ii][jj][kk] = func;
	}

	// Calculate vmean
	j_x /= rho;
	j_y /= rho;
	j_z /= rho;

	// Calculate standard deviation
	for ( size_t kk = 0; kk < conf.Nw; ++kk )
	for ( size_t jj = 0; jj < conf.Nv; ++jj )
	for ( size_t ii = 0; ii < conf.Nu; ++ii )
	{
		// vmean = 0
		real unew = u_min + ii*du - j_x;
		real vnew = v_min + jj*dv - j_y;
		real wnew = w_min + kk*dw - j_z;

		std_dev += f[ii][jj][kk] * 1/3 * (pow(unew, 2) + pow(vnew, 2) + pow(wnew, 2)) * du * dv * dw;
	}

	std_dev = sqrt(std_dev / rho); // Standard deviation

	if(rho < pow(10,-6)) {
		for (int i = 0; i < 35; i++) {
			moments[i] = 0;
		}
	} else {

		// Calculate moments
		for ( size_t kk = 0; kk < conf.Nw; ++kk )
		for ( size_t jj = 0; jj < conf.Nv; ++jj )
		for ( size_t ii = 0; ii < conf.Nu; ++ii )
		{
			// std_dev = 1
			real unew = (u_min + ii*du - j_x)/std_dev;
			real vnew = (v_min + jj*dv - j_y)/std_dev;
			real wnew = (w_min + kk*dw - j_z)/std_dev;

			moments[0] += f[ii][jj][kk]/rho * du * dv * dw * 1.;
			moments[1] += f[ii][jj][kk]/rho * du * dv * dw * 0.816496580927726*(1.5 - 0.5*pow(unew,2) - 0.5*pow(vnew,2) - 0.5*pow(wnew,2));
			moments[2] += f[ii][jj][kk]/rho * du * dv * dw * 0.3651483716701107*(3.75 - 2.5*pow(unew,2) + 0.25*pow(unew,4) - 2.5*pow(vnew,2) + 0.5*pow(unew,2)*pow(vnew,2) + 0.25*pow(vnew,4) - 2.5*pow(wnew,2) + 0.5*pow(unew,2)*pow(wnew,2) + 0.5*pow(vnew,2)*pow(wnew,2) + 0.25*pow(wnew,4));
			moments[3] += f[ii][jj][kk]/rho * du * dv * dw * vnew;
			moments[4] += f[ii][jj][kk]/rho * du * dv * dw * wnew;
			moments[5] += f[ii][jj][kk]/rho * du * dv * dw * -1.*unew;
			moments[6] += f[ii][jj][kk]/rho * du * dv * dw * 0.6324555320336759*(2.5*vnew - 0.5*pow(unew,2)*vnew - 0.5*pow(vnew,3) - 0.5*vnew*pow(wnew,2));
			moments[7] += f[ii][jj][kk]/rho * du * dv * dw * 0.6324555320336759*(2.5*wnew - 0.5*pow(unew,2)*wnew - 0.5*pow(vnew,2)*wnew - 0.5*pow(wnew,3));
			moments[8] += f[ii][jj][kk]/rho * du * dv * dw * 0.6324555320336759*(-2.5*unew + 0.5*pow(unew,3) + 0.5*unew*pow(vnew,2) + 0.5*unew*pow(wnew,2));
			moments[9] += f[ii][jj][kk]/rho * du * dv * dw * unew*vnew;
			moments[10] += f[ii][jj][kk]/rho * du * dv * dw * vnew*wnew;
			moments[11] += f[ii][jj][kk]/rho * du * dv * dw * 0.2886751345948129*(-1.*pow(unew,2) - 1.*pow(vnew,2) + 2.*pow(wnew,2));
			moments[12] += f[ii][jj][kk]/rho * du * dv * dw * -1.*unew*wnew;
			moments[13] += f[ii][jj][kk]/rho * du * dv * dw * 0.5*(pow(unew,2) - 1.*pow(vnew,2));
			moments[14] += f[ii][jj][kk]/rho * du * dv * dw * 0.5345224838248488*(3.5*unew*vnew - 0.5*pow(unew,3)*vnew - 0.5*unew*pow(vnew,3) - 0.5*unew*vnew*pow(wnew,2));
			moments[15] += f[ii][jj][kk]/rho * du * dv * dw * 0.5345224838248488*(3.5*vnew*wnew - 0.5*pow(unew,2)*vnew*wnew - 0.5*pow(vnew,3)*wnew - 0.5*vnew*pow(wnew,3));
			moments[16] += f[ii][jj][kk]/rho * du * dv * dw * 0.1543033499620919*(-3.5*pow(unew,2) + 0.5*pow(unew,4) - 3.5*pow(vnew,2) + pow(unew,2)*pow(vnew,2) + 0.5*pow(vnew,4) + 7.*pow(wnew,2) - 0.5*pow(unew,2)*pow(wnew,2) - 0.5*pow(vnew,2)*pow(wnew,2) - 1.*pow(wnew,4));
			moments[17] += f[ii][jj][kk]/rho * du * dv * dw * 0.5345224838248488*(-3.5*unew*wnew + 0.5*pow(unew,3)*wnew + 0.5*unew*pow(vnew,2)*wnew + 0.5*unew*pow(wnew,3));
			moments[18] += f[ii][jj][kk]/rho * du * dv * dw * 0.2672612419124244*(3.5*pow(unew,2) - 0.5*pow(unew,4) - 3.5*pow(vnew,2) + 0.5*pow(vnew,4) - 0.5*pow(unew,2)*pow(wnew,2) + 0.5*pow(vnew,2)*pow(wnew,2));
			moments[19] += f[ii][jj][kk]/rho * du * dv * dw * 0.20412414523193154*(3.*pow(unew,2)*vnew - 1.*pow(vnew,3));
			moments[20] += f[ii][jj][kk]/rho * du * dv * dw * unew*vnew*wnew;
			moments[21] += f[ii][jj][kk]/rho * du * dv * dw * 0.15811388300841897*(-1.*pow(unew,2)*vnew - 1.*pow(vnew,3) + 4.*vnew*pow(wnew,2));
			moments[22] += f[ii][jj][kk]/rho * du * dv * dw * 0.12909944487358055*(-3.*pow(unew,2)*wnew - 3.*pow(vnew,2)*wnew + 2.*pow(wnew,3));
			moments[23] += f[ii][jj][kk]/rho * du * dv * dw * 0.15811388300841897*(pow(unew,3) + unew*pow(vnew,2) - 4.*unew*pow(wnew,2));
			moments[24] += f[ii][jj][kk]/rho * du * dv * dw * 0.5*(pow(unew,2)*wnew - 1.*pow(vnew,2)*wnew);
			moments[25] += f[ii][jj][kk]/rho * du * dv * dw * 0.20412414523193154*(-1.*pow(unew,3) + 3.*unew*pow(vnew,2));
			moments[26] += f[ii][jj][kk]/rho * du * dv * dw * 0.2886751345948129*(pow(unew,3)*vnew - 1.*unew*pow(vnew,3));
			moments[27] += f[ii][jj][kk]/rho * du * dv * dw * 0.20412414523193154*(3.*pow(unew,2)*vnew*wnew - 1.*pow(vnew,3)*wnew);
			moments[28] += f[ii][jj][kk]/rho * du * dv * dw * 0.1091089451179962*(-1.*pow(unew,3)*vnew - 1.*unew*pow(vnew,3) + 6.*unew*vnew*pow(wnew,2));
			moments[29] += f[ii][jj][kk]/rho * du * dv * dw * 0.07715167498104596*(-3.*pow(unew,2)*vnew*wnew - 3.*pow(vnew,3)*wnew + 4.*vnew*pow(wnew,3));
			moments[30] += f[ii][jj][kk]/rho * du * dv * dw * 0.012198750911856666*(3.*pow(unew,4) + 6.*pow(unew,2)*pow(vnew,2) + 3.*pow(vnew,4) - 24.*pow(unew,2)*pow(wnew,2) - 24.*pow(vnew,2)*pow(wnew,2) + 8.*pow(wnew,4));
			moments[31] += f[ii][jj][kk]/rho * du * dv * dw * 0.07715167498104596*(3.*pow(unew,3)*wnew + 3.*unew*pow(vnew,2)*wnew - 4.*unew*pow(wnew,3));
			moments[32] += f[ii][jj][kk]/rho * du * dv * dw * 0.0545544725589981*(-1.*pow(unew,4) + pow(vnew,4) + 6.*pow(unew,2)*pow(wnew,2) - 6.*pow(vnew,2)*pow(wnew,2));
			moments[33] += f[ii][jj][kk]/rho * du * dv * dw * 0.20412414523193154*(-1.*pow(unew,3)*wnew + 3.*unew*pow(vnew,2)*wnew);
			moments[34] += f[ii][jj][kk]/rho * du * dv * dw * 0.07216878364870323*(pow(unew,4) - 6.*pow(unew,2)*pow(vnew,2) + pow(vnew,4));
		}
	}
	moments[35] = rho;
	moments[36] = j_x;
	moments[37] = j_y;
	moments[38] = j_z;
	moments[39] = std_dev;
}
template <typename real>
real pi_inverse( real u, real v, real w, real* moments )
{
	real value = 0;
	u = (u - moments[36]) / moments[39];
	v = (v - moments[37]) / moments[39];
	w = (w - moments[38]) / moments[39];
	value += moments[0] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 1.;
	value += moments[1] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.816496580927726*(1.5 - 0.5*pow(u,2) - 0.5*pow(v,2) - 0.5*pow(w,2));
	value += moments[2] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.3651483716701107*(3.75 - 2.5*pow(u,2) + 0.25*pow(u,4) - 2.5*pow(v,2) + 0.5*pow(u,2)*pow(v,2) + 0.25*pow(v,4) - 2.5*pow(w,2) + 0.5*pow(u,2)*pow(w,2) + 0.5*pow(v,2)*pow(w,2) + 0.25*pow(w,4));
	value += moments[3] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * v;
	value += moments[4] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * w;
	value += moments[5] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * -1.*u;
	value += moments[6] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.6324555320336759*(2.5*v - 0.5*pow(u,2)*v - 0.5*pow(v,3) - 0.5*v*pow(w,2));
	value += moments[7] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.6324555320336759*(2.5*w - 0.5*pow(u,2)*w - 0.5*pow(v,2)*w - 0.5*pow(w,3));
	value += moments[8] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.6324555320336759*(-2.5*u + 0.5*pow(u,3) + 0.5*u*pow(v,2) + 0.5*u*pow(w,2));
	value += moments[9] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * u*v;
	value += moments[10] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * v*w;
	value += moments[11] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.2886751345948129*(-1.*pow(u,2) - 1.*pow(v,2) + 2.*pow(w,2));
	value += moments[12] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * -1.*u*w;
	value += moments[13] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.5*(pow(u,2) - 1.*pow(v,2));
	value += moments[14] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.5345224838248488*(3.5*u*v - 0.5*pow(u,3)*v - 0.5*u*pow(v,3) - 0.5*u*v*pow(w,2));
	value += moments[15] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.5345224838248488*(3.5*v*w - 0.5*pow(u,2)*v*w - 0.5*pow(v,3)*w - 0.5*v*pow(w,3));
	value += moments[16] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.1543033499620919*(-3.5*pow(u,2) + 0.5*pow(u,4) - 3.5*pow(v,2) + pow(u,2)*pow(v,2) + 0.5*pow(v,4) + 7.*pow(w,2) - 0.5*pow(u,2)*pow(w,2) - 0.5*pow(v,2)*pow(w,2) - 1.*pow(w,4));
	value += moments[17] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.5345224838248488*(-3.5*u*w + 0.5*pow(u,3)*w + 0.5*u*pow(v,2)*w + 0.5*u*pow(w,3));
	value += moments[18] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.2672612419124244*(3.5*pow(u,2) - 0.5*pow(u,4) - 3.5*pow(v,2) + 0.5*pow(v,4) - 0.5*pow(u,2)*pow(w,2) + 0.5*pow(v,2)*pow(w,2));
	value += moments[19] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.20412414523193154*(3.*pow(u,2)*v - 1.*pow(v,3));
	value += moments[20] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * u*v*w;
	value += moments[21] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.15811388300841897*(-1.*pow(u,2)*v - 1.*pow(v,3) + 4.*v*pow(w,2));
	value += moments[22] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.12909944487358055*(-3.*pow(u,2)*w - 3.*pow(v,2)*w + 2.*pow(w,3));
	value += moments[23] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.15811388300841897*(pow(u,3) + u*pow(v,2) - 4.*u*pow(w,2));
	value += moments[24] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.5*(pow(u,2)*w - 1.*pow(v,2)*w);
	value += moments[25] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.20412414523193154*(-1.*pow(u,3) + 3.*u*pow(v,2));
	value += moments[26] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.2886751345948129*(pow(u,3)*v - 1.*u*pow(v,3));
	value += moments[27] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.20412414523193154*(3.*pow(u,2)*v*w - 1.*pow(v,3)*w);
	value += moments[28] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.1091089451179962*(-1.*pow(u,3)*v - 1.*u*pow(v,3) + 6.*u*v*pow(w,2));
	value += moments[29] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.07715167498104596*(-3.*pow(u,2)*v*w - 3.*pow(v,3)*w + 4.*v*pow(w,3));
	value += moments[30] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.012198750911856666*(3.*pow(u,4) + 6.*pow(u,2)*pow(v,2) + 3.*pow(v,4) - 24.*pow(u,2)*pow(w,2) - 24.*pow(v,2)*pow(w,2) + 8.*pow(w,4));
	value += moments[31] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.07715167498104596*(3.*pow(u,3)*w + 3.*u*pow(v,2)*w - 4.*u*pow(w,3));
	value += moments[32] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.0545544725589981*(-1.*pow(u,4) + pow(v,4) + 6.*pow(u,2)*pow(w,2) - 6.*pow(v,2)*pow(w,2));
	value += moments[33] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.20412414523193154*(-1.*pow(u,3)*w + 3.*u*pow(v,2)*w);
	value += moments[34] * exp(-(pow(u,2) + pow(v,2) + pow(w,2))/2) * 0.07216878364870323*(pow(u,4) - 6.*pow(u,2)*pow(v,2) + pow(v,4));
	value *= moments[35]/pow(moments[39],3) * 0.06349363593424097;
	return value;
}

double f0(double x, double y, double z, double u, double v, double w) noexcept
{
    using std::sin;
    using std::cos;
    using std::exp;

    constexpr double alpha = 0.001;
    constexpr double k     = 0.2;

    // Weak Landau Damping:
    constexpr double c  = 0.06349363593424096978576330493464; // Weak Landau damping
    return c * ( 1. + alpha*cos(k*x) + alpha*cos(k*y) + alpha*cos(k*z)) * exp( -(u*u+v*v+w*w)/2 );
}

config_t<double> conf(8, 8, 8, 16, 16, 16, 100, 0.2, 0, 4*M_PI, 0, 4*M_PI, 0, 4*M_PI,
		-5, 5, -5, 5, -5, 5, &f0);
constexpr size_t order = 4;

void test_moments(size_t n)
{
    size_t stride_t = (conf.Nx + order - 1) *
                      (conf.Ny + order - 1) *
					  (conf.Nz + order - 1);
    size_t size = conf.Nx*conf.Ny*conf.Nz;

    std::unique_ptr<double[]> coeffs { new double[ conf.Nt*stride_t ] {} };
    // Vector to store moment-vectors at each grid-location.
    std::vector<std::vector<double>> moments_vec(size, std::vector<double>(40,0));

    // Read in coefficients.
    std::ifstream coeffs_ifstr( "lin_landau_coeffs/coeffs_Nt_100_dt_0.200000_Nx_8_Ny_8_Nz_8_stride_t_1331.txt" );
    for(size_t i = 0; i < (n+1)*stride_t; i++){
    	coeffs_ifstr >> coeffs.get()[i];
    }

    // Compute moments at time t=n*dt.
	#pragma omp parallel for
    for(size_t l = 0; l < size; l++){
    	calculate_moments<double,order>( n, l, coeffs.get(), conf, moments_vec[l].data());
    }

    // Sort the moments such that I can use them for my interpolation routine.
    std::vector<std::vector<double>> resorted_moments_vec(40,std::vector<double>(size,0));
	#pragma omp parallel for
    for(size_t l = 0; l < size; l++){
    	for(size_t k = 0; k < 40; k++){
    		resorted_moments_vec[k][l] = moments_vec[l][k];
    	}
    }

    // Interpolate the moments such that we can evaluate them outside of grid-points.
    std::vector<std::vector<double>> moment_coeffs_vec(40,std::vector<double>(stride_t, 0));
    for(size_t k = 0; k < 40; k++){
    	interpolate<double,order>( moment_coeffs_vec[k].data(), resorted_moments_vec[k].data(),
    								conf );
    }

    // Plotting sizes.
    size_t Nplot = 16;
    double dx_plot = conf.Lx/Nplot;
    double dy_plot = conf.Ly/Nplot;
    double dz_plot = conf.Lz/Nplot;
    double du_plot = (conf.u_max - conf.u_min)/Nplot;
    double dv_plot = (conf.v_max - conf.v_min)/Nplot;
    double dw_plot = (conf.w_max - conf.w_min)/Nplot;
    // Plot first m moments in cross-section along z=0;
    std::cout << "Start plotting moments:" << std::endl;
    dergeraet::stopwatch<double> timer;
    size_t m = 10;
	#pragma omp parallel for
    for(size_t k = 0; k < m; k++){
    	std::ofstream plot_m("plot_moment_" + std::to_string(k) + ".txt");
    	std::vector<double> moment_array(40, 0);
    	for(size_t i = 0; i <= Nplot; i++){
    		for(size_t j = 0; j <= Nplot; j++){
    			double x = i*dx_plot;
    			double y = j*dy_plot;
    			double z = 0;
    			double curr_mom = eval<double,order>(x,y,z,moment_coeffs_vec[k].data(),conf);

    			plot_m << x << " " << y << " " << curr_mom << std::endl;
    		}
    		plot_m << std::endl;
    	}
    }
    std::cout << "Plotting moments took: " << timer.elapsed() << " s." << std::endl;
    // Compute error between f and f_moments.
    std::cout << "Start computing error:" << std::endl;
    timer.reset();
    std::vector<double> moment_array(40,0);
    double f_l1_error = 0;
    double f_l2_error = 0;
    double f_max_error = 0;
	#pragma omp parallel for
    for(size_t ix = 0; ix < Nplot; ix++)
    for(size_t iy = 0; iy < Nplot; iy++)
    for(size_t iz = 0; iz < Nplot; iz++)
    for(size_t iu = 0; iu < Nplot; iu++)
    for(size_t iv = 0; iv < Nplot; iv++)
    for(size_t iw = 0; iw < Nplot; iw++){
    	double x = ix*dx_plot;
    	double y = iy*dy_plot;
    	double z = iz*dz_plot;
    	double u = conf.u_min + iu*du_plot;
    	double v = conf.v_min + iv*dv_plot;
    	double w = conf.w_min + iw*dw_plot;

    	for(size_t k = 0; k < 40; k++){
    		moment_array[k] = eval<double,order>(x,y,z,moment_coeffs_vec[k].data(),conf);
    	}

    	double f_exact = eval_f<double,order>( n, x, y, z, u, v, w, coeffs.get(), conf);
    	double f_mom = pi_inverse( u, v, w, moment_array.data());

    	double dist = std::abs(f_exact-f_mom);
		#pragma omp critical
    	{
			f_l1_error += dist;
			f_l2_error += dist*dist;
			f_max_error = std::max(dist,f_max_error);
    	}
    }

    f_l1_error *= dx_plot*dy_plot*dz_plot*du_plot*dv_plot*dw_plot;
    f_l2_error *= dx_plot*dy_plot*dz_plot*du_plot*dv_plot*dw_plot;
    f_l2_error = std::sqrt(f_l2_error);

    std::cout << "Computing errors took: " << timer.elapsed() << " s." << std::endl;
    std::cout << "f_l1_error = " << f_l1_error << std::endl;
    std::cout << "f_l2_error = " << f_l2_error << std::endl;
    std::cout << "f_max_error = " << f_max_error << std::endl;
}

void test()
{
    using std::abs;
    using std::max;

    size_t stride_t = (conf.Nx + order - 1) *
                      (conf.Ny + order - 1) *
					  (conf.Nz + order - 1);

    std::unique_ptr<double[]> coeffs { new double[ conf.Nt*stride_t ] {} };
    std::unique_ptr<double,decltype(std::free)*> rho { reinterpret_cast<double*>(std::aligned_alloc(64,
    													sizeof(double)*conf.Nx*conf.Ny*conf.Nz)), std::free };
    if ( rho == nullptr ) throw std::bad_alloc {};

    poisson<double> poiss( conf );

    std::ofstream coeffs_str( "coeffs_Nt_" + std::to_string(conf.Nt) + "_dt_" + std::to_string(conf.dt) + "_Nx_"
    						+ std::to_string(conf.Nx) + "_Ny_" + std::to_string(conf.Ny)
							+ "_Nz_" + std::to_string(conf.Nz) + "_stride_t_" + std::to_string(stride_t) + ".txt" );
    std::ofstream E_str("electric_energy.txt");
    double total_time = 0;
    for ( size_t n = 0; n < conf.Nt; ++n )
    {
    	dergeraet::stopwatch<double> timer;

    	// Compute rho:
		#pragma omp parallel for
    	for(size_t l = 0; l<conf.Nx*conf.Ny*conf.Nz; l++)
    	{
    		rho.get()[l] = eval_rho<double,order>(n, l, coeffs.get(), conf);
    	}

        double E_energy = poiss.solve( rho.get() );
        interpolate<double,order>( coeffs.get() + n*stride_t, rho.get(), conf );

        double timer_elapsed = timer.elapsed();
        total_time += timer_elapsed;
	    double t = n*conf.dt;
        std::cout << "n = " << n << " t = " << t << " Comp-time: " << timer_elapsed << ". Total time s.f.: " << total_time << std::endl;
        E_str << t  << " " << E_energy << std::endl;

        for(size_t i = 0; i < stride_t; i++){
        	coeffs_str << coeffs.get()[n*stride_t + i] << std::endl;
        }
    }
    std::cout << "Total time: " << total_time << std::endl;
}
}
}


int main()
{
	//dergeraet::dim3::test();
	dergeraet::dim3::test_moments(0);
}

