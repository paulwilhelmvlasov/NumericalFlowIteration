// This code has been generated automatically using the Mathematica file C++CodeGenerator.nb
// This file calculates moments from a distribution
// Author: Anja Matena (anja.matena@rwth-aachen.de)

#include <cstddef>
#include <cmath>
#include <algorithm>

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

		std_dev += f * 1/3 * (pow(unew, 2) + pow(vnew, 2) + pow(wnew, 2));
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

		if(rho != 0) {

			moments[0] += f/rho * 1; 
			moments[1] += f/rho * sqrt(0.6666666666666666)*(1.5 - pow(unew,2)/2. - pow(vnew,2)/2. - pow(wnew,2)/2.); 
			moments[2] += f/rho * sqrt(0.13333333333333333)*(3.75 - (5*pow(unew,2))/2. + pow(unew,4)/4. - (5*pow(vnew,2))/2. + (pow(unew,2)*pow(vnew,2))/2. + pow(vnew,4)/4. - (5*pow(wnew,2))/2. + (pow(unew,2)*pow(wnew,2))/2. + (pow(vnew,2)*pow(wnew,2))/2. + pow(wnew,4)/4.); 
			moments[3] += f/rho * vnew; 
			moments[4] += f/rho * wnew; 
			moments[5] += f/rho * -unew; 
			moments[6] += f/rho * sqrt(0.4)*((5*vnew)/2. - (pow(unew,2)*vnew)/2. - pow(vnew,3)/2. - (vnew*pow(wnew,2))/2.); 
			moments[7] += f/rho * sqrt(0.4)*((5*wnew)/2. - (pow(unew,2)*wnew)/2. - (pow(vnew,2)*wnew)/2. - pow(wnew,3)/2.); 
			moments[8] += f/rho * sqrt(0.4)*((-5*unew)/2. + pow(unew,3)/2. + (unew*pow(vnew,2))/2. + (unew*pow(wnew,2))/2.); 
			moments[9] += f/rho * unew*vnew; 
			moments[10] += f/rho * vnew*wnew; 
			moments[11] += f/rho * (-pow(unew,2) - pow(vnew,2) + 2*pow(wnew,2))/(2.*sqrt(3)); 
			moments[12] += f/rho * -(unew*wnew); 
			moments[13] += f/rho * (pow(unew,2) - pow(vnew,2))/2.; 
			moments[14] += f/rho * sqrt(0.2857142857142857)*((7*unew*vnew)/2. - (pow(unew,3)*vnew)/2. - (unew*pow(vnew,3))/2. - (unew*vnew*pow(wnew,2))/2.); 
			moments[15] += f/rho * sqrt(0.2857142857142857)*((7*vnew*wnew)/2. - (pow(unew,2)*vnew*wnew)/2. - (pow(vnew,3)*wnew)/2. - (vnew*pow(wnew,3))/2.); 
			moments[16] += f/rho * ((-7*pow(unew,2))/2. + pow(unew,4)/2. - (7*pow(vnew,2))/2. + pow(unew,2)*pow(vnew,2) + pow(vnew,4)/2. + 7*pow(wnew,2) - (pow(unew,2)*pow(wnew,2))/2. - (pow(vnew,2)*pow(wnew,2))/2. - pow(wnew,4))/sqrt(42); 
			moments[17] += f/rho * sqrt(0.2857142857142857)*((-7*unew*wnew)/2. + (pow(unew,3)*wnew)/2. + (unew*pow(vnew,2)*wnew)/2. + (unew*pow(wnew,3))/2.); 
			moments[18] += f/rho * ((7*pow(unew,2))/2. - pow(unew,4)/2. - (7*pow(vnew,2))/2. + pow(vnew,4)/2. - (pow(unew,2)*pow(wnew,2))/2. + (pow(vnew,2)*pow(wnew,2))/2.)/sqrt(14); 
			moments[19] += f/rho * (3*pow(unew,2)*vnew - pow(vnew,3))/(2.*sqrt(6)); 
			moments[20] += f/rho * unew*vnew*wnew; 
			moments[21] += f/rho * (-(pow(unew,2)*vnew) - pow(vnew,3) + 4*vnew*pow(wnew,2))/(2.*sqrt(10)); 
			moments[22] += f/rho * (-3*pow(unew,2)*wnew - 3*pow(vnew,2)*wnew + 2*pow(wnew,3))/(2.*sqrt(15)); 
			moments[23] += f/rho * (pow(unew,3) + unew*pow(vnew,2) - 4*unew*pow(wnew,2))/(2.*sqrt(10)); 
			moments[24] += f/rho * (pow(unew,2)*wnew - pow(vnew,2)*wnew)/2.; 
			moments[25] += f/rho * (-pow(unew,3) + 3*unew*pow(vnew,2))/(2.*sqrt(6)); 
			moments[26] += f/rho * (pow(unew,3)*vnew - unew*pow(vnew,3))/(2.*sqrt(3)); 
			moments[27] += f/rho * (3*pow(unew,2)*vnew*wnew - pow(vnew,3)*wnew)/(2.*sqrt(6)); 
			moments[28] += f/rho * (-(pow(unew,3)*vnew) - unew*pow(vnew,3) + 6*unew*vnew*pow(wnew,2))/(2.*sqrt(21)); 
			moments[29] += f/rho * (-3*pow(unew,2)*vnew*wnew - 3*pow(vnew,3)*wnew + 4*vnew*pow(wnew,3))/(2.*sqrt(42)); 
			moments[30] += f/rho * (3*pow(unew,4) + 6*pow(unew,2)*pow(vnew,2) + 3*pow(vnew,4) - 24*pow(unew,2)*pow(wnew,2) - 24*pow(vnew,2)*pow(wnew,2) + 8*pow(wnew,4))/(8.*sqrt(105)); 
			moments[31] += f/rho * (3*pow(unew,3)*wnew + 3*unew*pow(vnew,2)*wnew - 4*unew*pow(wnew,3))/(2.*sqrt(42)); 
			moments[32] += f/rho * (-pow(unew,4) + pow(vnew,4) + 6*pow(unew,2)*pow(wnew,2) - 6*pow(vnew,2)*pow(wnew,2))/(4.*sqrt(21)); 
			moments[33] += f/rho * (-(pow(unew,3)*wnew) + 3*unew*pow(vnew,2)*wnew)/(2.*sqrt(6)); 
			moments[34] += f/rho * (pow(unew,4) - 6*pow(unew,2)*pow(vnew,2) + pow(vnew,4))/(8.*sqrt(3)); 
		}
	}
	moments[35] = rho; 
	moments[36] = j_x; 
	moments[37] = j_y; 
	moments[38] = j_z; 
	moments[39] = std_dev; 
} 

