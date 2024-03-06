// This code has been generated automatically using the Mathematica file C++CodeGenerator.nb
// This file calculates moments from a distribution
// Author: Anja Matena (anja.matena@rwth-aachen.de)

#include <cstddef>

template <typename real, size_t order>
void eval_rho( size_t n, size_t l, const real *coeffs, const config_t<real> &conf, real* moments )
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

	for ( size_t kk = 0; kk < conf.Nw; ++kk )
	for ( size_t jj = 0; jj < conf.Nv; ++jj )
	for ( size_t ii = 0; ii < conf.Nu; ++ii )
	{
		real u = u_min + ii*du;
		real v = v_min + jj*dv;
		real w = w_min + kk*dw;
		real f = eval_f<real,order>( n, x, y, z, u, v, w, coeffs, conf );

		rho += f;
		j_x += f*u;
		j_y += f*v;
		j_z += f*w;
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
		real u = u_min + ii * du;
		real v = v_min + jj * dv;
		real w = w_min + kk * dw;
		real f = eval_f<real,order>( n, x, y, z, u, v, w, coeffs, conf );

		// vmean = 0
		real unew = u - j_x;
		real vnew = v - j_y;
		real wnew = w - j_z;

		std_dev += f * (pow(unew, 2) + pow(vnew, 2) + pow(wnew, 2));
	}

	std_dev = sqrt(std_dev / rho); // Standard deviation

	// Calculate moments
	for ( size_t kk = 0; kk < conf.Nw; ++kk )
	for ( size_t jj = 0; jj < conf.Nv; ++jj )
	for ( size_t ii = 0; ii < conf.Nu; ++ii )
	{
		real u = u_min + ii*du;
		real v = v_min + jj*dv;
		real w = w_min + kk*dw;
		real f = eval_f<real,order>( n, x, y, z, u, v, w, coeffs, conf );

		// std_dev = 1
		real unew = (u - j_x)/std_dev;
		real vnew = (v - j_y)/std_dev;
		real wnew = (w - j_z)/std_dev;

		moments[0] += f* 1; 
		moments[1] += f* 1.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.; 
		moments[2] += f* 3.75 - (5*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.; 
		moments[3] += f* vnew; 
		moments[4] += f* wnew; 
		moments[5] += f* -unew; 
		moments[6] += f* vnew*(2.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[7] += f* wnew*(2.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[8] += f* -(unew*(2.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.)); 
		moments[9] += f* unew*vnew; 
		moments[10] += f* vnew*wnew; 
		moments[11] += f* -Power(unew,2) - Power(vnew,2) + 2*Power(wnew,2); 
		moments[12] += f* -(unew*wnew); 
		moments[13] += f* Power(unew,2) - Power(vnew,2); 
		moments[14] += f* unew*vnew*(3.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[15] += f* vnew*wnew*(3.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[16] += f* (-Power(unew,2) - Power(vnew,2) + 2*Power(wnew,2))*(3.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[17] += f* -(unew*wnew*(3.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.)); 
		moments[18] += f* (Power(unew,2) - Power(vnew,2))*(3.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[19] += f* 3*Power(unew,2)*vnew - Power(vnew,3); 
		moments[20] += f* unew*vnew*wnew; 
		moments[21] += f* -(Power(unew,2)*vnew) - Power(vnew,3) + 4*vnew*Power(wnew,2); 
		moments[22] += f* -3*Power(unew,2)*wnew - 3*Power(vnew,2)*wnew + 2*Power(wnew,3); 
		moments[23] += f* Power(unew,3) + unew*Power(vnew,2) - 4*unew*Power(wnew,2); 
		moments[24] += f* Power(unew,2)*wnew - Power(vnew,2)*wnew; 
		moments[25] += f* -Power(unew,3) + 3*unew*Power(vnew,2); 
		moments[26] += f* Power(unew,3)*vnew - unew*Power(vnew,3); 
		moments[27] += f* 3*Power(unew,2)*vnew*wnew - Power(vnew,3)*wnew; 
		moments[28] += f* -(Power(unew,3)*vnew) - unew*Power(vnew,3) + 6*unew*vnew*Power(wnew,2); 
		moments[29] += f* -3*Power(unew,2)*vnew*wnew - 3*Power(vnew,3)*wnew + 4*vnew*Power(wnew,3); 
		moments[30] += f* 3*Power(unew,4) + 6*Power(unew,2)*Power(vnew,2) + 3*Power(vnew,4) - 24*Power(unew,2)*Power(wnew,2) - 24*Power(vnew,2)*Power(wnew,2) + 8*Power(wnew,4); 
		moments[31] += f* 3*Power(unew,3)*wnew + 3*unew*Power(vnew,2)*wnew - 4*unew*Power(wnew,3); 
		moments[32] += f* -Power(unew,4) + Power(vnew,4) + 6*Power(unew,2)*Power(wnew,2) - 6*Power(vnew,2)*Power(wnew,2); 
		moments[33] += f* -(Power(unew,3)*wnew) + 3*unew*Power(vnew,2)*wnew; 
		moments[34] += f* Power(unew,4) - 6*Power(unew,2)*Power(vnew,2) + Power(vnew,4); 
	}
	rho = 1 - du*dv*dw*rho;
	moments /= rho; 
}