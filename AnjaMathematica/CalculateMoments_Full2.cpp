// This code has been generated automatically using the Mathematica file C++CodeGenerator.nb
// This file calculates moments from a distribution
// Author: Anja Matena (anja.matena@rwth-aachen.de)

#include <cstddef>
#include <cmath>
#include <math.h>
#include <algorithm>
#include <vector>

#include <dergeraet/config.hpp>

template <typename real, size_t order>
void eval_moments( size_t n, size_t l, const real *coeffs, const config_t<real> &conf, real* moments )
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

	std::vector<std::vector<std::vector<real>>> f(conf.Nu, std::vector<std::vector<real>>(conf.Nv, std::vector<real>(conf.Nw)));

	for ( size_t kk = 0; kk < conf.Nw; ++kk )
	for ( size_t jj = 0; jj < conf.Nv; ++jj )
	for ( size_t ii = 0; ii < conf.Nu; ++ii )
	{
		real u = u_min + ii*du;
		real v = v_min + jj*dv;
		real w = w_min + kk*dw;
		real func = eval_f<real,order>( n, x, y, z, u, v, w, coeffs, conf );

		rho += func;
		j_x += func * u;
		j_y += func * v;
		j_z += func * w;

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

		std_dev += f[ii][jj][kk] * 1/3 * (pow(unew, 2) + pow(vnew, 2) + pow(wnew, 2));
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

			moments[0] += f[ii][jj][kk]/rho * 1.; 
			moments[1] += f[ii][jj][kk]/rho * 0.816496580927726*(1.5 - 0.5*pow(unew,2) - 0.5*pow(vnew,2) - 0.5*pow(wnew,2)); 
			moments[2] += f[ii][jj][kk]/rho * 0.3651483716701107*(3.75 - 2.5*pow(unew,2) + 0.25*pow(unew,4) - 2.5*pow(vnew,2) + 0.5*pow(unew,2)*pow(vnew,2) + 0.25*pow(vnew,4) - 2.5*pow(wnew,2) + 0.5*pow(unew,2)*pow(wnew,2) + 0.5*pow(vnew,2)*pow(wnew,2) + 0.25*pow(wnew,4)); 
			moments[3] += f[ii][jj][kk]/rho * vnew; 
			moments[4] += f[ii][jj][kk]/rho * wnew; 
			moments[5] += f[ii][jj][kk]/rho * -1.*unew; 
			moments[6] += f[ii][jj][kk]/rho * 0.6324555320336759*(2.5*vnew - 0.5*pow(unew,2)*vnew - 0.5*pow(vnew,3) - 0.5*vnew*pow(wnew,2)); 
			moments[7] += f[ii][jj][kk]/rho * 0.6324555320336759*(2.5*wnew - 0.5*pow(unew,2)*wnew - 0.5*pow(vnew,2)*wnew - 0.5*pow(wnew,3)); 
			moments[8] += f[ii][jj][kk]/rho * 0.6324555320336759*(-2.5*unew + 0.5*pow(unew,3) + 0.5*unew*pow(vnew,2) + 0.5*unew*pow(wnew,2)); 
			moments[9] += f[ii][jj][kk]/rho * unew*vnew; 
			moments[10] += f[ii][jj][kk]/rho * vnew*wnew; 
			moments[11] += f[ii][jj][kk]/rho * 0.2886751345948129*(-1.*pow(unew,2) - 1.*pow(vnew,2) + 2.*pow(wnew,2)); 
			moments[12] += f[ii][jj][kk]/rho * -1.*unew*wnew; 
			moments[13] += f[ii][jj][kk]/rho * 0.5*(pow(unew,2) - 1.*pow(vnew,2)); 
			moments[14] += f[ii][jj][kk]/rho * 0.5345224838248488*(3.5*unew*vnew - 0.5*pow(unew,3)*vnew - 0.5*unew*pow(vnew,3) - 0.5*unew*vnew*pow(wnew,2)); 
			moments[15] += f[ii][jj][kk]/rho * 0.5345224838248488*(3.5*vnew*wnew - 0.5*pow(unew,2)*vnew*wnew - 0.5*pow(vnew,3)*wnew - 0.5*vnew*pow(wnew,3)); 
			moments[16] += f[ii][jj][kk]/rho * 0.1543033499620919*(-3.5*pow(unew,2) + 0.5*pow(unew,4) - 3.5*pow(vnew,2) + pow(unew,2)*pow(vnew,2) + 0.5*pow(vnew,4) + 7.*pow(wnew,2) - 0.5*pow(unew,2)*pow(wnew,2) - 0.5*pow(vnew,2)*pow(wnew,2) - 1.*pow(wnew,4)); 
			moments[17] += f[ii][jj][kk]/rho * 0.5345224838248488*(-3.5*unew*wnew + 0.5*pow(unew,3)*wnew + 0.5*unew*pow(vnew,2)*wnew + 0.5*unew*pow(wnew,3)); 
			moments[18] += f[ii][jj][kk]/rho * 0.2672612419124244*(3.5*pow(unew,2) - 0.5*pow(unew,4) - 3.5*pow(vnew,2) + 0.5*pow(vnew,4) - 0.5*pow(unew,2)*pow(wnew,2) + 0.5*pow(vnew,2)*pow(wnew,2)); 
			moments[19] += f[ii][jj][kk]/rho * 0.20412414523193154*(3.*pow(unew,2)*vnew - 1.*pow(vnew,3)); 
			moments[20] += f[ii][jj][kk]/rho * unew*vnew*wnew; 
			moments[21] += f[ii][jj][kk]/rho * 0.15811388300841897*(-1.*pow(unew,2)*vnew - 1.*pow(vnew,3) + 4.*vnew*pow(wnew,2)); 
			moments[22] += f[ii][jj][kk]/rho * 0.12909944487358055*(-3.*pow(unew,2)*wnew - 3.*pow(vnew,2)*wnew + 2.*pow(wnew,3)); 
			moments[23] += f[ii][jj][kk]/rho * 0.15811388300841897*(pow(unew,3) + unew*pow(vnew,2) - 4.*unew*pow(wnew,2)); 
			moments[24] += f[ii][jj][kk]/rho * 0.5*(pow(unew,2)*wnew - 1.*pow(vnew,2)*wnew); 
			moments[25] += f[ii][jj][kk]/rho * 0.20412414523193154*(-1.*pow(unew,3) + 3.*unew*pow(vnew,2)); 
			moments[26] += f[ii][jj][kk]/rho * 0.2886751345948129*(pow(unew,3)*vnew - 1.*unew*pow(vnew,3)); 
			moments[27] += f[ii][jj][kk]/rho * 0.20412414523193154*(3.*pow(unew,2)*vnew*wnew - 1.*pow(vnew,3)*wnew); 
			moments[28] += f[ii][jj][kk]/rho * 0.1091089451179962*(-1.*pow(unew,3)*vnew - 1.*unew*pow(vnew,3) + 6.*unew*vnew*pow(wnew,2)); 
			moments[29] += f[ii][jj][kk]/rho * 0.07715167498104596*(-3.*pow(unew,2)*vnew*wnew - 3.*pow(vnew,3)*wnew + 4.*vnew*pow(wnew,3)); 
			moments[30] += f[ii][jj][kk]/rho * 0.012198750911856666*(3.*pow(unew,4) + 6.*pow(unew,2)*pow(vnew,2) + 3.*pow(vnew,4) - 24.*pow(unew,2)*pow(wnew,2) - 24.*pow(vnew,2)*pow(wnew,2) + 8.*pow(wnew,4)); 
			moments[31] += f[ii][jj][kk]/rho * 0.07715167498104596*(3.*pow(unew,3)*wnew + 3.*unew*pow(vnew,2)*wnew - 4.*unew*pow(wnew,3)); 
			moments[32] += f[ii][jj][kk]/rho * 0.0545544725589981*(-1.*pow(unew,4) + pow(vnew,4) + 6.*pow(unew,2)*pow(wnew,2) - 6.*pow(vnew,2)*pow(wnew,2)); 
			moments[33] += f[ii][jj][kk]/rho * 0.20412414523193154*(-1.*pow(unew,3)*wnew + 3.*unew*pow(vnew,2)*wnew); 
			moments[34] += f[ii][jj][kk]/rho * 0.07216878364870323*(pow(unew,4) - 6.*pow(unew,2)*pow(vnew,2) + pow(vnew,4)); 
		}
	}
	moments[35] = rho; 
	moments[36] = j_x; 
	moments[37] = j_y; 
	moments[38] = j_z; 
	moments[39] = std_dev; 
} 
template <typename real, size_t order>
real pi_inverse( real u, real v, real w, real* moments )
{
	real value = 0;
	u = (u - moments[36]) / moments[39]; 
	v = (v - moments[37]) / moments[39]; 
	w = (w - moments[38]) / moments[39]; 
	value += moments[0] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 1.; 
	value += moments[1] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.816496580927726*(1.5 - 0.5*pow(u,2) - 0.5*pow(v,2) - 0.5*pow(w,2)); 
	value += moments[2] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.3651483716701107*(3.75 - 2.5*pow(u,2) + 0.25*pow(u,4) - 2.5*pow(v,2) + 0.5*pow(u,2)*pow(v,2) + 0.25*pow(v,4) - 2.5*pow(w,2) + 0.5*pow(u,2)*pow(w,2) + 0.5*pow(v,2)*pow(w,2) + 0.25*pow(w,4)); 
	value += moments[3] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * v; 
	value += moments[4] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * w; 
	value += moments[5] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * -1.*u; 
	value += moments[6] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.6324555320336759*(2.5*v - 0.5*pow(u,2)*v - 0.5*pow(v,3) - 0.5*v*pow(w,2)); 
	value += moments[7] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.6324555320336759*(2.5*w - 0.5*pow(u,2)*w - 0.5*pow(v,2)*w - 0.5*pow(w,3)); 
	value += moments[8] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.6324555320336759*(-2.5*u + 0.5*pow(u,3) + 0.5*u*pow(v,2) + 0.5*u*pow(w,2)); 
	value += moments[9] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * u*v; 
	value += moments[10] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * v*w; 
	value += moments[11] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.2886751345948129*(-1.*pow(u,2) - 1.*pow(v,2) + 2.*pow(w,2)); 
	value += moments[12] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * -1.*u*w; 
	value += moments[13] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.5*(pow(u,2) - 1.*pow(v,2)); 
	value += moments[14] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.5345224838248488*(3.5*u*v - 0.5*pow(u,3)*v - 0.5*u*pow(v,3) - 0.5*u*v*pow(w,2)); 
	value += moments[15] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.5345224838248488*(3.5*v*w - 0.5*pow(u,2)*v*w - 0.5*pow(v,3)*w - 0.5*v*pow(w,3)); 
	value += moments[16] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.1543033499620919*(-3.5*pow(u,2) + 0.5*pow(u,4) - 3.5*pow(v,2) + pow(u,2)*pow(v,2) + 0.5*pow(v,4) + 7.*pow(w,2) - 0.5*pow(u,2)*pow(w,2) - 0.5*pow(v,2)*pow(w,2) - 1.*pow(w,4)); 
	value += moments[17] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.5345224838248488*(-3.5*u*w + 0.5*pow(u,3)*w + 0.5*u*pow(v,2)*w + 0.5*u*pow(w,3)); 
	value += moments[18] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.2672612419124244*(3.5*pow(u,2) - 0.5*pow(u,4) - 3.5*pow(v,2) + 0.5*pow(v,4) - 0.5*pow(u,2)*pow(w,2) + 0.5*pow(v,2)*pow(w,2)); 
	value += moments[19] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.20412414523193154*(3.*pow(u,2)*v - 1.*pow(v,3)); 
	value += moments[20] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * u*v*w; 
	value += moments[21] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.15811388300841897*(-1.*pow(u,2)*v - 1.*pow(v,3) + 4.*v*pow(w,2)); 
	value += moments[22] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.12909944487358055*(-3.*pow(u,2)*w - 3.*pow(v,2)*w + 2.*pow(w,3)); 
	value += moments[23] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.15811388300841897*(pow(u,3) + u*pow(v,2) - 4.*u*pow(w,2)); 
	value += moments[24] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.5*(pow(u,2)*w - 1.*pow(v,2)*w); 
	value += moments[25] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.20412414523193154*(-1.*pow(u,3) + 3.*u*pow(v,2)); 
	value += moments[26] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.2886751345948129*(pow(u,3)*v - 1.*u*pow(v,3)); 
	value += moments[27] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.20412414523193154*(3.*pow(u,2)*v*w - 1.*pow(v,3)*w); 
	value += moments[28] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.1091089451179962*(-1.*pow(u,3)*v - 1.*u*pow(v,3) + 6.*u*v*pow(w,2)); 
	value += moments[29] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.07715167498104596*(-3.*pow(u,2)*v*w - 3.*pow(v,3)*w + 4.*v*pow(w,3)); 
	value += moments[30] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.012198750911856666*(3.*pow(u,4) + 6.*pow(u,2)*pow(v,2) + 3.*pow(v,4) - 24.*pow(u,2)*pow(w,2) - 24.*pow(v,2)*pow(w,2) + 8.*pow(w,4)); 
	value += moments[31] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.07715167498104596*(3.*pow(u,3)*w + 3.*u*pow(v,2)*w - 4.*u*pow(w,3)); 
	value += moments[32] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.0545544725589981*(-1.*pow(u,4) + pow(v,4) + 6.*pow(u,2)*pow(w,2) - 6.*pow(v,2)*pow(w,2)); 
	value += moments[33] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.20412414523193154*(-1.*pow(u,3)*w + 3.*u*pow(v,2)*w); 
	value += moments[34] * exp(-(pow(u,2) + pow(v,2) + pow(w,2)) /2) * 0.07216878364870323*(pow(u,4) - 6.*pow(u,2)*pow(v,2) + pow(v,4)); 
	value *= moments[35]/pow(moments[39],3) ;
	return value;
}

