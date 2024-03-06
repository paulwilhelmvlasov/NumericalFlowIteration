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
		moments[3] += f* 13.125 - (105*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (21*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.; 
		moments[4] += f* vnew; 
		moments[5] += f* wnew; 
		moments[6] += f* -unew; 
		moments[7] += f* vnew*(2.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[8] += f* wnew*(2.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[9] += f* -(unew*(2.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.)); 
		moments[10] += f* vnew*(8.75 - (7*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[11] += f* wnew*(8.75 - (7*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[12] += f* -(unew*(8.75 - (7*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.)); 
		moments[13] += f* vnew*(39.375 - (189*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (27*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[14] += f* wnew*(39.375 - (189*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (27*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[15] += f* -(unew*(39.375 - (189*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (27*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.)); 
		moments[16] += f* unew*vnew; 
		moments[17] += f* vnew*wnew; 
		moments[18] += f* -Power(unew,2) - Power(vnew,2) + 2*Power(wnew,2); 
		moments[19] += f* -(unew*wnew); 
		moments[20] += f* Power(unew,2) - Power(vnew,2); 
		moments[21] += f* unew*vnew*(3.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[22] += f* vnew*wnew*(3.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[23] += f* (-Power(unew,2) - Power(vnew,2) + 2*Power(wnew,2))*(3.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[24] += f* -(unew*wnew*(3.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.)); 
		moments[25] += f* (Power(unew,2) - Power(vnew,2))*(3.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[26] += f* unew*vnew*(15.75 - (9*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[27] += f* vnew*wnew*(15.75 - (9*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[28] += f* (-Power(unew,2) - Power(vnew,2) + 2*Power(wnew,2))*(15.75 - (9*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[29] += f* -(unew*wnew*(15.75 - (9*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.)); 
		moments[30] += f* (Power(unew,2) - Power(vnew,2))*(15.75 - (9*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[31] += f* unew*vnew*(86.625 - (297*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (33*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[32] += f* vnew*wnew*(86.625 - (297*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (33*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[33] += f* (-Power(unew,2) - Power(vnew,2) + 2*Power(wnew,2))*(86.625 - (297*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (33*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[34] += f* -(unew*wnew*(86.625 - (297*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (33*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.)); 
		moments[35] += f* (Power(unew,2) - Power(vnew,2))*(86.625 - (297*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (33*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[36] += f* 3*Power(unew,2)*vnew - Power(vnew,3); 
		moments[37] += f* unew*vnew*wnew; 
		moments[38] += f* -(Power(unew,2)*vnew) - Power(vnew,3) + 4*vnew*Power(wnew,2); 
		moments[39] += f* -3*Power(unew,2)*wnew - 3*Power(vnew,2)*wnew + 2*Power(wnew,3); 
		moments[40] += f* Power(unew,3) + unew*Power(vnew,2) - 4*unew*Power(wnew,2); 
		moments[41] += f* Power(unew,2)*wnew - Power(vnew,2)*wnew; 
		moments[42] += f* -Power(unew,3) + 3*unew*Power(vnew,2); 
		moments[43] += f* (3*Power(unew,2)*vnew - Power(vnew,3))*(4.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[44] += f* unew*vnew*wnew*(4.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[45] += f* (-(Power(unew,2)*vnew) - Power(vnew,3) + 4*vnew*Power(wnew,2))*(4.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[46] += f* (-3*Power(unew,2)*wnew - 3*Power(vnew,2)*wnew + 2*Power(wnew,3))*(4.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[47] += f* (Power(unew,3) + unew*Power(vnew,2) - 4*unew*Power(wnew,2))*(4.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[48] += f* (Power(unew,2)*wnew - Power(vnew,2)*wnew)*(4.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[49] += f* (-Power(unew,3) + 3*unew*Power(vnew,2))*(4.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[50] += f* (3*Power(unew,2)*vnew - Power(vnew,3))*(24.75 - (11*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[51] += f* unew*vnew*wnew*(24.75 - (11*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[52] += f* (-(Power(unew,2)*vnew) - Power(vnew,3) + 4*vnew*Power(wnew,2))*(24.75 - (11*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[53] += f* (-3*Power(unew,2)*wnew - 3*Power(vnew,2)*wnew + 2*Power(wnew,3))*(24.75 - (11*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[54] += f* (Power(unew,3) + unew*Power(vnew,2) - 4*unew*Power(wnew,2))*(24.75 - (11*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[55] += f* (Power(unew,2)*wnew - Power(vnew,2)*wnew)*(24.75 - (11*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[56] += f* (-Power(unew,3) + 3*unew*Power(vnew,2))*(24.75 - (11*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[57] += f* (3*Power(unew,2)*vnew - Power(vnew,3))*(160.875 - (429*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (39*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[58] += f* unew*vnew*wnew*(160.875 - (429*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (39*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[59] += f* (-(Power(unew,2)*vnew) - Power(vnew,3) + 4*vnew*Power(wnew,2))*(160.875 - (429*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (39*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[60] += f* (-3*Power(unew,2)*wnew - 3*Power(vnew,2)*wnew + 2*Power(wnew,3))*(160.875 - (429*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (39*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[61] += f* (Power(unew,3) + unew*Power(vnew,2) - 4*unew*Power(wnew,2))*(160.875 - (429*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (39*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[62] += f* (Power(unew,2)*wnew - Power(vnew,2)*wnew)*(160.875 - (429*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (39*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[63] += f* (-Power(unew,3) + 3*unew*Power(vnew,2))*(160.875 - (429*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (39*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[64] += f* Power(unew,3)*vnew - unew*Power(vnew,3); 
		moments[65] += f* 3*Power(unew,2)*vnew*wnew - Power(vnew,3)*wnew; 
		moments[66] += f* -(Power(unew,3)*vnew) - unew*Power(vnew,3) + 6*unew*vnew*Power(wnew,2); 
		moments[67] += f* -3*Power(unew,2)*vnew*wnew - 3*Power(vnew,3)*wnew + 4*vnew*Power(wnew,3); 
		moments[68] += f* 3*Power(unew,4) + 6*Power(unew,2)*Power(vnew,2) + 3*Power(vnew,4) - 24*Power(unew,2)*Power(wnew,2) - 24*Power(vnew,2)*Power(wnew,2) + 8*Power(wnew,4); 
		moments[69] += f* 3*Power(unew,3)*wnew + 3*unew*Power(vnew,2)*wnew - 4*unew*Power(wnew,3); 
		moments[70] += f* -Power(unew,4) + Power(vnew,4) + 6*Power(unew,2)*Power(wnew,2) - 6*Power(vnew,2)*Power(wnew,2); 
		moments[71] += f* -(Power(unew,3)*wnew) + 3*unew*Power(vnew,2)*wnew; 
		moments[72] += f* Power(unew,4) - 6*Power(unew,2)*Power(vnew,2) + Power(vnew,4); 
		moments[73] += f* (Power(unew,3)*vnew - unew*Power(vnew,3))*(5.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[74] += f* (3*Power(unew,2)*vnew*wnew - Power(vnew,3)*wnew)*(5.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[75] += f* (-(Power(unew,3)*vnew) - unew*Power(vnew,3) + 6*unew*vnew*Power(wnew,2))*(5.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[76] += f* (-3*Power(unew,2)*vnew*wnew - 3*Power(vnew,3)*wnew + 4*vnew*Power(wnew,3))*(5.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[77] += f* (3*Power(unew,4) + 6*Power(unew,2)*Power(vnew,2) + 3*Power(vnew,4) - 24*Power(unew,2)*Power(wnew,2) - 24*Power(vnew,2)*Power(wnew,2) + 8*Power(wnew,4))*(5.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[78] += f* (3*Power(unew,3)*wnew + 3*unew*Power(vnew,2)*wnew - 4*unew*Power(wnew,3))*(5.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[79] += f* (-Power(unew,4) + Power(vnew,4) + 6*Power(unew,2)*Power(wnew,2) - 6*Power(vnew,2)*Power(wnew,2))*(5.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[80] += f* (-(Power(unew,3)*wnew) + 3*unew*Power(vnew,2)*wnew)*(5.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[81] += f* (Power(unew,4) - 6*Power(unew,2)*Power(vnew,2) + Power(vnew,4))*(5.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[82] += f* (Power(unew,3)*vnew - unew*Power(vnew,3))*(35.75 - (13*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[83] += f* (3*Power(unew,2)*vnew*wnew - Power(vnew,3)*wnew)*(35.75 - (13*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[84] += f* (-(Power(unew,3)*vnew) - unew*Power(vnew,3) + 6*unew*vnew*Power(wnew,2))*(35.75 - (13*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[85] += f* (-3*Power(unew,2)*vnew*wnew - 3*Power(vnew,3)*wnew + 4*vnew*Power(wnew,3))*(35.75 - (13*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[86] += f* (3*Power(unew,4) + 6*Power(unew,2)*Power(vnew,2) + 3*Power(vnew,4) - 24*Power(unew,2)*Power(wnew,2) - 24*Power(vnew,2)*Power(wnew,2) + 8*Power(wnew,4))*(35.75 - (13*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[87] += f* (3*Power(unew,3)*wnew + 3*unew*Power(vnew,2)*wnew - 4*unew*Power(wnew,3))*(35.75 - (13*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[88] += f* (-Power(unew,4) + Power(vnew,4) + 6*Power(unew,2)*Power(wnew,2) - 6*Power(vnew,2)*Power(wnew,2))*(35.75 - (13*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[89] += f* (-(Power(unew,3)*wnew) + 3*unew*Power(vnew,2)*wnew)*(35.75 - (13*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[90] += f* (Power(unew,4) - 6*Power(unew,2)*Power(vnew,2) + Power(vnew,4))*(35.75 - (13*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[91] += f* (Power(unew,3)*vnew - unew*Power(vnew,3))*(268.125 - (585*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (45*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[92] += f* (3*Power(unew,2)*vnew*wnew - Power(vnew,3)*wnew)*(268.125 - (585*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (45*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[93] += f* (-(Power(unew,3)*vnew) - unew*Power(vnew,3) + 6*unew*vnew*Power(wnew,2))*(268.125 - (585*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (45*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[94] += f* (-3*Power(unew,2)*vnew*wnew - 3*Power(vnew,3)*wnew + 4*vnew*Power(wnew,3))*(268.125 - (585*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (45*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[95] += f* (3*Power(unew,4) + 6*Power(unew,2)*Power(vnew,2) + 3*Power(vnew,4) - 24*Power(unew,2)*Power(wnew,2) - 24*Power(vnew,2)*Power(wnew,2) + 8*Power(wnew,4))*(268.125 - (585*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (45*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[96] += f* (3*Power(unew,3)*wnew + 3*unew*Power(vnew,2)*wnew - 4*unew*Power(wnew,3))*(268.125 - (585*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (45*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[97] += f* (-Power(unew,4) + Power(vnew,4) + 6*Power(unew,2)*Power(wnew,2) - 6*Power(vnew,2)*Power(wnew,2))*(268.125 - (585*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (45*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[98] += f* (-(Power(unew,3)*wnew) + 3*unew*Power(vnew,2)*wnew)*(268.125 - (585*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (45*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[99] += f* (Power(unew,4) - 6*Power(unew,2)*Power(vnew,2) + Power(vnew,4))*(268.125 - (585*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (45*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[100] += f* 5*Power(unew,4)*vnew - 10*Power(unew,2)*Power(vnew,3) + Power(vnew,5); 
		moments[101] += f* Power(unew,3)*vnew*wnew - unew*Power(vnew,3)*wnew; 
		moments[102] += f* -3*Power(unew,4)*vnew - 2*Power(unew,2)*Power(vnew,3) + Power(vnew,5) + 24*Power(unew,2)*vnew*Power(wnew,2) - 8*Power(vnew,3)*Power(wnew,2); 
		moments[103] += f* -(Power(unew,3)*vnew*wnew) - unew*Power(vnew,3)*wnew + 2*unew*vnew*Power(wnew,3); 
		moments[104] += f* Power(unew,4)*vnew + 2*Power(unew,2)*Power(vnew,3) + Power(vnew,5) - 12*Power(unew,2)*vnew*Power(wnew,2) - 12*Power(vnew,3)*Power(wnew,2) + 8*vnew*Power(wnew,4); 
		moments[105] += f* 15*Power(unew,4)*wnew + 30*Power(unew,2)*Power(vnew,2)*wnew + 15*Power(vnew,4)*wnew - 40*Power(unew,2)*Power(wnew,3) - 40*Power(vnew,2)*Power(wnew,3) + 8*Power(wnew,5); 
		moments[106] += f* -Power(unew,5) - 2*Power(unew,3)*Power(vnew,2) - unew*Power(vnew,4) + 12*Power(unew,3)*Power(wnew,2) + 12*unew*Power(vnew,2)*Power(wnew,2) - 8*unew*Power(wnew,4); 
		moments[107] += f* -(Power(unew,4)*wnew) + Power(vnew,4)*wnew + 2*Power(unew,2)*Power(wnew,3) - 2*Power(vnew,2)*Power(wnew,3); 
		moments[108] += f* Power(unew,5) - 2*Power(unew,3)*Power(vnew,2) - 3*unew*Power(vnew,4) - 8*Power(unew,3)*Power(wnew,2) + 24*unew*Power(vnew,2)*Power(wnew,2); 
		moments[109] += f* Power(unew,4)*wnew - 6*Power(unew,2)*Power(vnew,2)*wnew + Power(vnew,4)*wnew; 
		moments[110] += f* -Power(unew,5) + 10*Power(unew,3)*Power(vnew,2) - 5*unew*Power(vnew,4); 
		moments[111] += f* (5*Power(unew,4)*vnew - 10*Power(unew,2)*Power(vnew,3) + Power(vnew,5))*(6.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[112] += f* (Power(unew,3)*vnew*wnew - unew*Power(vnew,3)*wnew)*(6.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[113] += f* (-3*Power(unew,4)*vnew - 2*Power(unew,2)*Power(vnew,3) + Power(vnew,5) + 24*Power(unew,2)*vnew*Power(wnew,2) - 8*Power(vnew,3)*Power(wnew,2))*(6.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[114] += f* (-(Power(unew,3)*vnew*wnew) - unew*Power(vnew,3)*wnew + 2*unew*vnew*Power(wnew,3))*(6.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[115] += f* (Power(unew,4)*vnew + 2*Power(unew,2)*Power(vnew,3) + Power(vnew,5) - 12*Power(unew,2)*vnew*Power(wnew,2) - 12*Power(vnew,3)*Power(wnew,2) + 8*vnew*Power(wnew,4))*(6.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[116] += f* (15*Power(unew,4)*wnew + 30*Power(unew,2)*Power(vnew,2)*wnew + 15*Power(vnew,4)*wnew - 40*Power(unew,2)*Power(wnew,3) - 40*Power(vnew,2)*Power(wnew,3) + 8*Power(wnew,5))*(6.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[117] += f* (-Power(unew,5) - 2*Power(unew,3)*Power(vnew,2) - unew*Power(vnew,4) + 12*Power(unew,3)*Power(wnew,2) + 12*unew*Power(vnew,2)*Power(wnew,2) - 8*unew*Power(wnew,4))*(6.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[118] += f* (-(Power(unew,4)*wnew) + Power(vnew,4)*wnew + 2*Power(unew,2)*Power(wnew,3) - 2*Power(vnew,2)*Power(wnew,3))*(6.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[119] += f* (Power(unew,5) - 2*Power(unew,3)*Power(vnew,2) - 3*unew*Power(vnew,4) - 8*Power(unew,3)*Power(wnew,2) + 24*unew*Power(vnew,2)*Power(wnew,2))*(6.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[120] += f* (Power(unew,4)*wnew - 6*Power(unew,2)*Power(vnew,2)*wnew + Power(vnew,4)*wnew)*(6.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[121] += f* (-Power(unew,5) + 10*Power(unew,3)*Power(vnew,2) - 5*unew*Power(vnew,4))*(6.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[122] += f* (5*Power(unew,4)*vnew - 10*Power(unew,2)*Power(vnew,3) + Power(vnew,5))*(48.75 - (15*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[123] += f* (Power(unew,3)*vnew*wnew - unew*Power(vnew,3)*wnew)*(48.75 - (15*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[124] += f* (-3*Power(unew,4)*vnew - 2*Power(unew,2)*Power(vnew,3) + Power(vnew,5) + 24*Power(unew,2)*vnew*Power(wnew,2) - 8*Power(vnew,3)*Power(wnew,2))*(48.75 - (15*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[125] += f* (-(Power(unew,3)*vnew*wnew) - unew*Power(vnew,3)*wnew + 2*unew*vnew*Power(wnew,3))*(48.75 - (15*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[126] += f* (Power(unew,4)*vnew + 2*Power(unew,2)*Power(vnew,3) + Power(vnew,5) - 12*Power(unew,2)*vnew*Power(wnew,2) - 12*Power(vnew,3)*Power(wnew,2) + 8*vnew*Power(wnew,4))*(48.75 - (15*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[127] += f* (15*Power(unew,4)*wnew + 30*Power(unew,2)*Power(vnew,2)*wnew + 15*Power(vnew,4)*wnew - 40*Power(unew,2)*Power(wnew,3) - 40*Power(vnew,2)*Power(wnew,3) + 8*Power(wnew,5))*(48.75 - (15*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[128] += f* (-Power(unew,5) - 2*Power(unew,3)*Power(vnew,2) - unew*Power(vnew,4) + 12*Power(unew,3)*Power(wnew,2) + 12*unew*Power(vnew,2)*Power(wnew,2) - 8*unew*Power(wnew,4))*(48.75 - (15*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[129] += f* (-(Power(unew,4)*wnew) + Power(vnew,4)*wnew + 2*Power(unew,2)*Power(wnew,3) - 2*Power(vnew,2)*Power(wnew,3))*(48.75 - (15*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[130] += f* (Power(unew,5) - 2*Power(unew,3)*Power(vnew,2) - 3*unew*Power(vnew,4) - 8*Power(unew,3)*Power(wnew,2) + 24*unew*Power(vnew,2)*Power(wnew,2))*(48.75 - (15*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[131] += f* (Power(unew,4)*wnew - 6*Power(unew,2)*Power(vnew,2)*wnew + Power(vnew,4)*wnew)*(48.75 - (15*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[132] += f* (-Power(unew,5) + 10*Power(unew,3)*Power(vnew,2) - 5*unew*Power(vnew,4))*(48.75 - (15*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[133] += f* (5*Power(unew,4)*vnew - 10*Power(unew,2)*Power(vnew,3) + Power(vnew,5))*(414.375 - (765*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (51*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[134] += f* (Power(unew,3)*vnew*wnew - unew*Power(vnew,3)*wnew)*(414.375 - (765*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (51*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[135] += f* (-3*Power(unew,4)*vnew - 2*Power(unew,2)*Power(vnew,3) + Power(vnew,5) + 24*Power(unew,2)*vnew*Power(wnew,2) - 8*Power(vnew,3)*Power(wnew,2))*(414.375 - (765*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (51*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[136] += f* (-(Power(unew,3)*vnew*wnew) - unew*Power(vnew,3)*wnew + 2*unew*vnew*Power(wnew,3))*(414.375 - (765*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (51*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[137] += f* (Power(unew,4)*vnew + 2*Power(unew,2)*Power(vnew,3) + Power(vnew,5) - 12*Power(unew,2)*vnew*Power(wnew,2) - 12*Power(vnew,3)*Power(wnew,2) + 8*vnew*Power(wnew,4))*(414.375 - (765*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (51*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[138] += f* (15*Power(unew,4)*wnew + 30*Power(unew,2)*Power(vnew,2)*wnew + 15*Power(vnew,4)*wnew - 40*Power(unew,2)*Power(wnew,3) - 40*Power(vnew,2)*Power(wnew,3) + 8*Power(wnew,5))*(414.375 - (765*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (51*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[139] += f* (-Power(unew,5) - 2*Power(unew,3)*Power(vnew,2) - unew*Power(vnew,4) + 12*Power(unew,3)*Power(wnew,2) + 12*unew*Power(vnew,2)*Power(wnew,2) - 8*unew*Power(wnew,4))*(414.375 - (765*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (51*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[140] += f* (-(Power(unew,4)*wnew) + Power(vnew,4)*wnew + 2*Power(unew,2)*Power(wnew,3) - 2*Power(vnew,2)*Power(wnew,3))*(414.375 - (765*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (51*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[141] += f* (Power(unew,5) - 2*Power(unew,3)*Power(vnew,2) - 3*unew*Power(vnew,4) - 8*Power(unew,3)*Power(wnew,2) + 24*unew*Power(vnew,2)*Power(wnew,2))*(414.375 - (765*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (51*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[142] += f* (Power(unew,4)*wnew - 6*Power(unew,2)*Power(vnew,2)*wnew + Power(vnew,4)*wnew)*(414.375 - (765*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (51*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[143] += f* (-Power(unew,5) + 10*Power(unew,3)*Power(vnew,2) - 5*unew*Power(vnew,4))*(414.375 - (765*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (51*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[144] += f* 3*Power(unew,5)*vnew - 10*Power(unew,3)*Power(vnew,3) + 3*unew*Power(vnew,5); 
		moments[145] += f* 5*Power(unew,4)*vnew*wnew - 10*Power(unew,2)*Power(vnew,3)*wnew + Power(vnew,5)*wnew; 
		moments[146] += f* -(Power(unew,5)*vnew) + unew*Power(vnew,5) + 10*Power(unew,3)*vnew*Power(wnew,2) - 10*unew*Power(vnew,3)*Power(wnew,2); 
		moments[147] += f* -9*Power(unew,4)*vnew*wnew - 6*Power(unew,2)*Power(vnew,3)*wnew + 3*Power(vnew,5)*wnew + 24*Power(unew,2)*vnew*Power(wnew,3) - 8*Power(vnew,3)*Power(wnew,3); 
		moments[148] += f* Power(unew,5)*vnew + 2*Power(unew,3)*Power(vnew,3) + unew*Power(vnew,5) - 16*Power(unew,3)*vnew*Power(wnew,2) - 16*unew*Power(vnew,3)*Power(wnew,2) + 16*unew*vnew*Power(wnew,4); 
		moments[149] += f* 5*Power(unew,4)*vnew*wnew + 10*Power(unew,2)*Power(vnew,3)*wnew + 5*Power(vnew,5)*wnew - 20*Power(unew,2)*vnew*Power(wnew,3) - 20*Power(vnew,3)*Power(wnew,3) + 8*vnew*Power(wnew,5); 
		moments[150] += f* -5*Power(unew,6) - 15*Power(unew,4)*Power(vnew,2) - 15*Power(unew,2)*Power(vnew,4) - 5*Power(vnew,6) + 90*Power(unew,4)*Power(wnew,2) + 180*Power(unew,2)*Power(vnew,2)*Power(wnew,2) + 90*Power(vnew,4)*Power(wnew,2) - 120*Power(unew,2)*Power(wnew,4) - 120*Power(vnew,2)*Power(wnew,4) + 16*Power(wnew,6); 
		moments[151] += f* -5*Power(unew,5)*wnew - 10*Power(unew,3)*Power(vnew,2)*wnew - 5*unew*Power(vnew,4)*wnew + 20*Power(unew,3)*Power(wnew,3) + 20*unew*Power(vnew,2)*Power(wnew,3) - 8*unew*Power(wnew,5); 
		moments[152] += f* Power(unew,6) + Power(unew,4)*Power(vnew,2) - Power(unew,2)*Power(vnew,4) - Power(vnew,6) - 16*Power(unew,4)*Power(wnew,2) + 16*Power(vnew,4)*Power(wnew,2) + 16*Power(unew,2)*Power(wnew,4) - 16*Power(vnew,2)*Power(wnew,4); 
		moments[153] += f* 3*Power(unew,5)*wnew - 6*Power(unew,3)*Power(vnew,2)*wnew - 9*unew*Power(vnew,4)*wnew - 8*Power(unew,3)*Power(wnew,3) + 24*unew*Power(vnew,2)*Power(wnew,3); 
		moments[154] += f* -Power(unew,6) + 5*Power(unew,4)*Power(vnew,2) + 5*Power(unew,2)*Power(vnew,4) - Power(vnew,6) + 10*Power(unew,4)*Power(wnew,2) - 60*Power(unew,2)*Power(vnew,2)*Power(wnew,2) + 10*Power(vnew,4)*Power(wnew,2); 
		moments[155] += f* -(Power(unew,5)*wnew) + 10*Power(unew,3)*Power(vnew,2)*wnew - 5*unew*Power(vnew,4)*wnew; 
		moments[156] += f* Power(unew,6) - 15*Power(unew,4)*Power(vnew,2) + 15*Power(unew,2)*Power(vnew,4) - Power(vnew,6); 
		moments[157] += f* (3*Power(unew,5)*vnew - 10*Power(unew,3)*Power(vnew,3) + 3*unew*Power(vnew,5))*(7.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[158] += f* (5*Power(unew,4)*vnew*wnew - 10*Power(unew,2)*Power(vnew,3)*wnew + Power(vnew,5)*wnew)*(7.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[159] += f* (-(Power(unew,5)*vnew) + unew*Power(vnew,5) + 10*Power(unew,3)*vnew*Power(wnew,2) - 10*unew*Power(vnew,3)*Power(wnew,2))*(7.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[160] += f* (-9*Power(unew,4)*vnew*wnew - 6*Power(unew,2)*Power(vnew,3)*wnew + 3*Power(vnew,5)*wnew + 24*Power(unew,2)*vnew*Power(wnew,3) - 8*Power(vnew,3)*Power(wnew,3))*(7.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[161] += f* (Power(unew,5)*vnew + 2*Power(unew,3)*Power(vnew,3) + unew*Power(vnew,5) - 16*Power(unew,3)*vnew*Power(wnew,2) - 16*unew*Power(vnew,3)*Power(wnew,2) + 16*unew*vnew*Power(wnew,4))*(7.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[162] += f* (5*Power(unew,4)*vnew*wnew + 10*Power(unew,2)*Power(vnew,3)*wnew + 5*Power(vnew,5)*wnew - 20*Power(unew,2)*vnew*Power(wnew,3) - 20*Power(vnew,3)*Power(wnew,3) + 8*vnew*Power(wnew,5))*(7.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[163] += f* (-5*Power(unew,6) - 15*Power(unew,4)*Power(vnew,2) - 15*Power(unew,2)*Power(vnew,4) - 5*Power(vnew,6) + 90*Power(unew,4)*Power(wnew,2) + 180*Power(unew,2)*Power(vnew,2)*Power(wnew,2) + 90*Power(vnew,4)*Power(wnew,2) - 120*Power(unew,2)*Power(wnew,4) - 120*Power(vnew,2)*Power(wnew,4) + 16*Power(wnew,6))*(7.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[164] += f* (-5*Power(unew,5)*wnew - 10*Power(unew,3)*Power(vnew,2)*wnew - 5*unew*Power(vnew,4)*wnew + 20*Power(unew,3)*Power(wnew,3) + 20*unew*Power(vnew,2)*Power(wnew,3) - 8*unew*Power(wnew,5))*(7.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[165] += f* (Power(unew,6) + Power(unew,4)*Power(vnew,2) - Power(unew,2)*Power(vnew,4) - Power(vnew,6) - 16*Power(unew,4)*Power(wnew,2) + 16*Power(vnew,4)*Power(wnew,2) + 16*Power(unew,2)*Power(wnew,4) - 16*Power(vnew,2)*Power(wnew,4))*(7.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[166] += f* (3*Power(unew,5)*wnew - 6*Power(unew,3)*Power(vnew,2)*wnew - 9*unew*Power(vnew,4)*wnew - 8*Power(unew,3)*Power(wnew,3) + 24*unew*Power(vnew,2)*Power(wnew,3))*(7.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[167] += f* (-Power(unew,6) + 5*Power(unew,4)*Power(vnew,2) + 5*Power(unew,2)*Power(vnew,4) - Power(vnew,6) + 10*Power(unew,4)*Power(wnew,2) - 60*Power(unew,2)*Power(vnew,2)*Power(wnew,2) + 10*Power(vnew,4)*Power(wnew,2))*(7.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[168] += f* (-(Power(unew,5)*wnew) + 10*Power(unew,3)*Power(vnew,2)*wnew - 5*unew*Power(vnew,4)*wnew)*(7.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[169] += f* (Power(unew,6) - 15*Power(unew,4)*Power(vnew,2) + 15*Power(unew,2)*Power(vnew,4) - Power(vnew,6))*(7.5 + (-Power(unew,2) - Power(vnew,2) - Power(wnew,2))/2.); 
		moments[170] += f* (3*Power(unew,5)*vnew - 10*Power(unew,3)*Power(vnew,3) + 3*unew*Power(vnew,5))*(63.75 - (17*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[171] += f* (5*Power(unew,4)*vnew*wnew - 10*Power(unew,2)*Power(vnew,3)*wnew + Power(vnew,5)*wnew)*(63.75 - (17*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[172] += f* (-(Power(unew,5)*vnew) + unew*Power(vnew,5) + 10*Power(unew,3)*vnew*Power(wnew,2) - 10*unew*Power(vnew,3)*Power(wnew,2))*(63.75 - (17*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[173] += f* (-9*Power(unew,4)*vnew*wnew - 6*Power(unew,2)*Power(vnew,3)*wnew + 3*Power(vnew,5)*wnew + 24*Power(unew,2)*vnew*Power(wnew,3) - 8*Power(vnew,3)*Power(wnew,3))*(63.75 - (17*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[174] += f* (Power(unew,5)*vnew + 2*Power(unew,3)*Power(vnew,3) + unew*Power(vnew,5) - 16*Power(unew,3)*vnew*Power(wnew,2) - 16*unew*Power(vnew,3)*Power(wnew,2) + 16*unew*vnew*Power(wnew,4))*(63.75 - (17*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[175] += f* (5*Power(unew,4)*vnew*wnew + 10*Power(unew,2)*Power(vnew,3)*wnew + 5*Power(vnew,5)*wnew - 20*Power(unew,2)*vnew*Power(wnew,3) - 20*Power(vnew,3)*Power(wnew,3) + 8*vnew*Power(wnew,5))*(63.75 - (17*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[176] += f* (-5*Power(unew,6) - 15*Power(unew,4)*Power(vnew,2) - 15*Power(unew,2)*Power(vnew,4) - 5*Power(vnew,6) + 90*Power(unew,4)*Power(wnew,2) + 180*Power(unew,2)*Power(vnew,2)*Power(wnew,2) + 90*Power(vnew,4)*Power(wnew,2) - 120*Power(unew,2)*Power(wnew,4) - 120*Power(vnew,2)*Power(wnew,4) + 16*Power(wnew,6))*(63.75 - (17*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[177] += f* (-5*Power(unew,5)*wnew - 10*Power(unew,3)*Power(vnew,2)*wnew - 5*unew*Power(vnew,4)*wnew + 20*Power(unew,3)*Power(wnew,3) + 20*unew*Power(vnew,2)*Power(wnew,3) - 8*unew*Power(wnew,5))*(63.75 - (17*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[178] += f* (Power(unew,6) + Power(unew,4)*Power(vnew,2) - Power(unew,2)*Power(vnew,4) - Power(vnew,6) - 16*Power(unew,4)*Power(wnew,2) + 16*Power(vnew,4)*Power(wnew,2) + 16*Power(unew,2)*Power(wnew,4) - 16*Power(vnew,2)*Power(wnew,4))*(63.75 - (17*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[179] += f* (3*Power(unew,5)*wnew - 6*Power(unew,3)*Power(vnew,2)*wnew - 9*unew*Power(vnew,4)*wnew - 8*Power(unew,3)*Power(wnew,3) + 24*unew*Power(vnew,2)*Power(wnew,3))*(63.75 - (17*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[180] += f* (-Power(unew,6) + 5*Power(unew,4)*Power(vnew,2) + 5*Power(unew,2)*Power(vnew,4) - Power(vnew,6) + 10*Power(unew,4)*Power(wnew,2) - 60*Power(unew,2)*Power(vnew,2)*Power(wnew,2) + 10*Power(vnew,4)*Power(wnew,2))*(63.75 - (17*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[181] += f* (-(Power(unew,5)*wnew) + 10*Power(unew,3)*Power(vnew,2)*wnew - 5*unew*Power(vnew,4)*wnew)*(63.75 - (17*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[182] += f* (Power(unew,6) - 15*Power(unew,4)*Power(vnew,2) + 15*Power(unew,2)*Power(vnew,4) - Power(vnew,6))*(63.75 - (17*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/2. + Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2)/4.); 
		moments[183] += f* (3*Power(unew,5)*vnew - 10*Power(unew,3)*Power(vnew,3) + 3*unew*Power(vnew,5))*(605.625 - (969*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (57*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[184] += f* (5*Power(unew,4)*vnew*wnew - 10*Power(unew,2)*Power(vnew,3)*wnew + Power(vnew,5)*wnew)*(605.625 - (969*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (57*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[185] += f* (-(Power(unew,5)*vnew) + unew*Power(vnew,5) + 10*Power(unew,3)*vnew*Power(wnew,2) - 10*unew*Power(vnew,3)*Power(wnew,2))*(605.625 - (969*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (57*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[186] += f* (-9*Power(unew,4)*vnew*wnew - 6*Power(unew,2)*Power(vnew,3)*wnew + 3*Power(vnew,5)*wnew + 24*Power(unew,2)*vnew*Power(wnew,3) - 8*Power(vnew,3)*Power(wnew,3))*(605.625 - (969*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (57*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[187] += f* (Power(unew,5)*vnew + 2*Power(unew,3)*Power(vnew,3) + unew*Power(vnew,5) - 16*Power(unew,3)*vnew*Power(wnew,2) - 16*unew*Power(vnew,3)*Power(wnew,2) + 16*unew*vnew*Power(wnew,4))*(605.625 - (969*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (57*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[188] += f* (5*Power(unew,4)*vnew*wnew + 10*Power(unew,2)*Power(vnew,3)*wnew + 5*Power(vnew,5)*wnew - 20*Power(unew,2)*vnew*Power(wnew,3) - 20*Power(vnew,3)*Power(wnew,3) + 8*vnew*Power(wnew,5))*(605.625 - (969*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (57*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[189] += f* (-5*Power(unew,6) - 15*Power(unew,4)*Power(vnew,2) - 15*Power(unew,2)*Power(vnew,4) - 5*Power(vnew,6) + 90*Power(unew,4)*Power(wnew,2) + 180*Power(unew,2)*Power(vnew,2)*Power(wnew,2) + 90*Power(vnew,4)*Power(wnew,2) - 120*Power(unew,2)*Power(wnew,4) - 120*Power(vnew,2)*Power(wnew,4) + 16*Power(wnew,6))*(605.625 - (969*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (57*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[190] += f* (-5*Power(unew,5)*wnew - 10*Power(unew,3)*Power(vnew,2)*wnew - 5*unew*Power(vnew,4)*wnew + 20*Power(unew,3)*Power(wnew,3) + 20*unew*Power(vnew,2)*Power(wnew,3) - 8*unew*Power(wnew,5))*(605.625 - (969*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (57*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[191] += f* (Power(unew,6) + Power(unew,4)*Power(vnew,2) - Power(unew,2)*Power(vnew,4) - Power(vnew,6) - 16*Power(unew,4)*Power(wnew,2) + 16*Power(vnew,4)*Power(wnew,2) + 16*Power(unew,2)*Power(wnew,4) - 16*Power(vnew,2)*Power(wnew,4))*(605.625 - (969*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (57*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[192] += f* (3*Power(unew,5)*wnew - 6*Power(unew,3)*Power(vnew,2)*wnew - 9*unew*Power(vnew,4)*wnew - 8*Power(unew,3)*Power(wnew,3) + 24*unew*Power(vnew,2)*Power(wnew,3))*(605.625 - (969*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (57*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[193] += f* (-Power(unew,6) + 5*Power(unew,4)*Power(vnew,2) + 5*Power(unew,2)*Power(vnew,4) - Power(vnew,6) + 10*Power(unew,4)*Power(wnew,2) - 60*Power(unew,2)*Power(vnew,2)*Power(wnew,2) + 10*Power(vnew,4)*Power(wnew,2))*(605.625 - (969*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (57*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[194] += f* (-(Power(unew,5)*wnew) + 10*Power(unew,3)*Power(vnew,2)*wnew - 5*unew*Power(vnew,4)*wnew)*(605.625 - (969*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (57*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
		moments[195] += f* (Power(unew,6) - 15*Power(unew,4)*Power(vnew,2) + 15*Power(unew,2)*Power(vnew,4) - Power(vnew,6))*(605.625 - (969*(Power(unew,2) + Power(vnew,2) + Power(wnew,2)))/8. + (57*Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),2))/8. - Power(Power(unew,2) + Power(vnew,2) + Power(wnew,2),3)/8.); 
	}
	rho = 1 - du*dv*dw*rho;
	moments /= rho; 
}
