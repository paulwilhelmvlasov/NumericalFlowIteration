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


#include <iostream>
#include <fstream>
#include <vector>

#include <dergeraet/fields.hpp>

constexpr size_t order = 4;

double sign(double x)
{
	if(x >= 0){
		return 1;
	}else{
		return -1;
	}
}

void exp_jb_times_v(double B, double& v1, double& v2, double tol=1e-16)
{
	// Evaluates exp(-dt*J_B)*v.
	// If B=0, then exp(J_B)=I.

	if(std::abs(B) > tol)
	{
		double fac1 = std::sin(B)*sign(B);
		double fac2 = 1 - std::cos(B);

		double old_v1 = v1;
		double old_v2 = v2;

		v1 = old_v1 + fac1*old_v2 - fac2*old_v1;
		v2 = old_v2 - fac1*old_v1 - fac2*old_v2;
	}
}

template <size_t dx = 0>
double eval_E_1(size_t nt, double x, double* coeffs_E_1, const dergeraet::dim_1_half::config_t<double>& conf, size_t stride_t)
{
	return dergeraet::dim_1_half::eval<double,order,dx>(x, coeffs_E_1 + nt*stride_t, conf);
}

template <size_t dx = 0>
double eval_E_2(size_t nt, double x, double* coeffs_E_2, const dergeraet::dim_1_half::config_t<double>& conf, size_t stride_t)
{
	return dergeraet::dim_1_half::eval<double,order,dx>(x, coeffs_E_2 + nt*stride_t, conf);
}

template <size_t dx = 0>
double eval_B_3(size_t nt, double x, double* coeffs_B_3, const dergeraet::dim_1_half::config_t<double>& conf, size_t stride_t)
{
	return dergeraet::dim_1_half::eval<double,order,dx>(x, coeffs_B_3 + nt*stride_t, conf);
}

double eval_f(size_t nt, double x, double u, double v, double* coeffs_E_1, double* coeffs_E_2,
				double* coeffs_B_3, const dergeraet::dim_1_half::config_t<double>& conf, size_t stride_t)
{
	double B = 0;
	for(;nt > 0; nt--)
	{
		x -= conf.dt*u;
		B = -conf.dt*(eval_B_3(nt-1, x, coeffs_B_3, conf, stride_t) - conf.dt*eval_E_2<1>(nt-1, x, coeffs_E_2, conf, stride_t));

		exp_jb_times_v(B, u, v);

		u -= conf.dt*eval_E_1(nt-1,x,coeffs_E_1,conf,stride_t);
		v -= conf.dt*eval_E_2(nt-1,x,coeffs_E_2,conf,stride_t);
	}

	return conf.f0(x, u, v);
}

void backwards_iteration_J_Hf(size_t nt, double* coeffs_E_1, double* coeffs_E_2, double* coeffs_B_3,
								double* coeffs_J_Hf_1, double* coeffs_J_Hf_2, const dergeraet::dim_1_half::config_t<double>& conf, size_t stride_t)
{
	// nt is the next time-step for which one wants to compute J_Hf.
	// Computes J_Hf on a grid and interpolates it then using B-Splines.
	// Todo: Think about a clever way to parallelize this!
	if(nt == 0)
	{
		throw std::runtime_error("nt > 0 expected. J_Hf only exists after init.");
	}

	std::vector<double> j_hf_1(conf.Nx,0);
	std::vector<double> j_hf_2(conf.Nx,0);

	#pragma omp parallel for
	for(size_t i = 0; i < conf.Nx; i++)
	{
		double x = conf.x_min + i*conf.dx;
		for(size_t j = 0; j < conf.Nu; j++)
		{
			double u = conf.u_min + j*conf.du;
			for(size_t k = 0; k < conf.Nv; k++)
			{
				double v = conf.v_min + k*conf.dv;

				//x -= 0.5*conf.dt*u;
				x -= conf.dt*u;

				double B = -conf.dt*( eval_B_3(nt-1, x, coeffs_B_3, conf, stride_t)
							- conf.dt*eval_E_2<1>(nt-1, x, coeffs_E_2, conf, stride_t) );

				exp_jb_times_v(B, u, v);

				u -= conf.dt*eval_E_1(nt-1,x,coeffs_E_1,conf,stride_t);
				v -= conf.dt*eval_E_2(nt-1,x,coeffs_E_2,conf,stride_t);

				double f = eval_f(nt-1, x, u, v, coeffs_E_1, coeffs_E_2, coeffs_B_3, conf, stride_t);

				j_hf_1[i] += (conf.u_min+j*conf.du)*f;
				j_hf_2[i] += (conf.v_min+k*conf.dv)*f;
			}
		}

		j_hf_1[i] *= conf.du*conf.dv;
		j_hf_2[i] *= conf.du*conf.dv;
	}

	dergeraet::dim_1_half::interpolate<double,order>(coeffs_J_Hf_1 + (nt-1)*stride_t, j_hf_1.data(), conf);
	dergeraet::dim_1_half::interpolate<double,order>(coeffs_J_Hf_2 + (nt-1)*stride_t, j_hf_2.data(), conf);
}

void backwards_iteration_avrg_J_Hf(size_t nt, double* coeffs_E_1, double* coeffs_E_2, double* coeffs_B_3,
								double& avrg_J_Hf_1, double& avrg_J_Hf_2, const dergeraet::dim_1_half::config_t<double>& conf,
								size_t stride_t)
{
	// nt is the next time-step for which one wants to compute avrg_J_Hf. For nt it is already given.
	// Todo: Think about a clever way to parallelize this!
	// Do we really need this in the periodic case? See CROUSEILLES et.al. paper.

	if(nt == 0)
	{
		throw std::runtime_error("nt > 0 expected. J_Hf only exists after init.");
	}

	avrg_J_Hf_1 = 0;
	avrg_J_Hf_2 = 0;

	for(size_t i = 0; i < conf.Nx; i++)
	{
		double x = conf.x_min + i*conf.dx;
		for(size_t j = 0; j < conf.Nu; j++)
		{
			double u = conf.u_min + j*conf.du;
			for(size_t k = 0; k < conf.Nv; k++)
			{
				double v = conf.v_min + k*conf.dv;

				double B = -conf.dt*(eval_B_3(nt-1,x,coeffs_B_3,conf,stride_t) - conf.dt*eval_E_2<1>(nt-1,x,coeffs_E_2,conf,stride_t));

				exp_jb_times_v(B, u, v, conf.dt);
				u -= conf.dt*eval_E_1(nt-1,x,coeffs_E_1,conf,stride_t);
				v -= conf.dt*eval_E_2(nt-1,x,coeffs_E_2,conf,stride_t);

				double f = eval_f(nt-1, x, u, v, coeffs_E_1, coeffs_E_2, coeffs_B_3, conf, stride_t);

				avrg_J_Hf_1 += (conf.u_min+j*conf.du)*f;
				avrg_J_Hf_2 += (conf.v_min+k*conf.dv)*f;
			}
		}
	}

	double fac = conf.dt*conf.Lx_inv*conf.dx*conf.du*conf.dv;
	avrg_J_Hf_1 *= fac;
	avrg_J_Hf_2 *= fac;
}

void compute_E(size_t nt, double* coeffs_E_1, double* coeffs_E_2, double* coeffs_B_3, double* coeffs_J_Hf_1,
				double* coeffs_J_Hf_2, double* avrg_J_Hf_1, double* avrg_J_Hf_2, const dergeraet::dim_1_half::config_t<double>& conf, size_t stride_t)
{
	// Given nt > 0 computes the coefficients of E_1(n_t) and E_2(n_t).

	if(nt == 0)
	{
		throw std::runtime_error("nt > 0 expected. E_0 already computed.");
	}

	std::vector<double> E_values_1(conf.Nx, 0);
	std::vector<double> E_values_2(conf.Nx, 0);

	#pragma omp parallel for
	for(size_t i = 0; i< conf.Nx; i++)
	{
		double x = conf.x_min + i*conf.dx;

		E_values_1[i] = eval_E_1(nt-1,x,coeffs_E_1,conf,stride_t)
						- conf.dt*dergeraet::dim_1_half::eval<double,order>(x, coeffs_J_Hf_1 + (nt-1)*stride_t, conf);

		E_values_2[i] = eval_E_2(nt-1,x,coeffs_E_2,conf,stride_t) - conf.dt*eval_B_3<1>(nt-1,x,coeffs_B_3,conf,stride_t)
						+ conf.dt*conf.dt*eval_E_2<2>(nt-1,x,coeffs_E_2,conf,stride_t)
						- conf.dt*dergeraet::dim_1_half::eval<double,order>(x, coeffs_J_Hf_2 + (nt-1)*stride_t, conf);
	}

	dergeraet::dim_1_half::interpolate<double,order>(coeffs_E_1 + nt*stride_t, E_values_1.data(), conf);
	dergeraet::dim_1_half::interpolate<double,order>(coeffs_E_2 + nt*stride_t, E_values_2.data(), conf);
}

void compute_B(size_t nt, double* coeffs_E_2, double* coeffs_B_3, const dergeraet::dim_1_half::config_t<double>& conf,
				size_t stride_t)
{
	// Given nt > 0 computes the coefficients of B_3(n_t). E_1(nt-1), E_2(nt-1) and B_3(nt-1) is known.

	if(nt == 0)
	{
		throw std::runtime_error("nt > 0 expected. B_0 already computed.");
	}

	std::vector<double> B_values(conf.Nx, 0);

	#pragma omp parallel for
	for(size_t i = 0; i< conf.Nx; i++)
	{
		double x = conf.x_min + i*conf.dx;
		B_values[i] = eval_B_3(nt-1,x,coeffs_B_3,conf,stride_t) - conf.dt*eval_E_2<1>(nt-1,x,coeffs_E_2,conf,stride_t);
	}

	dergeraet::dim_1_half::interpolate<double,order>(coeffs_B_3 + nt*stride_t, B_values.data(), conf);
}


int main()
{
	dergeraet::dim_1_half::config_t<double> conf;

    const size_t stride_t = conf.Nx + order - 1;
    std::unique_ptr<double[]> coeffs_E_1 { new double[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<double[]> coeffs_E_2 { new double[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<double[]> coeffs_B_3 { new double[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<double[]> coeffs_J_Hf_1 { new double[ (conf.Nt)*stride_t ] {} };
    std::unique_ptr<double[]> coeffs_J_Hf_2 { new double[ (conf.Nt)*stride_t ] {} };
    std::vector<double> avrg_J_Hf_1(conf.Nt, 0) ;
    std::vector<double> avrg_J_Hf_2(conf.Nt, 0) ;

    std::vector<double> E_1(conf.Nx, 0);
    std::vector<double> E_2(conf.Nx, 0);
    std::vector<double> B_3(conf.Nx, 0);

    // Init values of E and B.
    std::ofstream E_1_str("E_1_0.txt");
    std::ofstream E_2_str("E_2_0.txt");
    std::ofstream B_3_str("B_3_0.txt");
    double elec_energy = 0;
    double magn_energy = 0;
    for(size_t i = 0; i < conf.Nx; i++)
    {
    	//  Weibel instability:
    	double alpha = 1e-2;
    	double beta = 1e-2;
    	double k = 0.5;//1.25;
    	double x = conf.x_min + i*conf.dx;
    	E_1[i] = alpha/k*std::sin(k*x);
    	E_2[i] = 0;
    	B_3[i] = 0;//beta*std::cos(k*x);

    	elec_energy += E_1[i]*E_1[i] + E_2[i]*E_2[i];
    	magn_energy += B_3[i]*B_3[i];

    	E_1_str << x << " " << E_1[i] << std::endl;
    	E_2_str << x << " " << E_2[i] << std::endl;
    	B_3_str << x << " " << B_3[i] << std::endl;
    }
    elec_energy *= conf.dx;
    magn_energy *= conf.dx;
    // Interpolate E_0 and B_0.
    dergeraet::dim_1_half::interpolate<double,order>(coeffs_E_1.get(), E_1.data(), conf);
    dergeraet::dim_1_half::interpolate<double,order>(coeffs_E_2.get(), E_2.data(), conf);
    dergeraet::dim_1_half::interpolate<double,order>(coeffs_B_3.get(), B_3.data(), conf);

    std::ofstream Electric_Energy_str("E_energy.txt");
    std::ofstream Magnetic_Energy_str("B_energy.txt");
    Electric_Energy_str << 0 << " " << elec_energy << std::endl;
    Magnetic_Energy_str << 0 << " " << magn_energy << std::endl;
    std::cout << "E_energy: " << 0 << " " << elec_energy << std::endl;
    std::cout << "B_energy: " << 0 << " " << magn_energy << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    for(size_t nt = 1; nt <= conf.Nt; nt++)
    {
    	// Compute next J_Hf.
    	backwards_iteration_J_Hf(nt, coeffs_E_1.get(), coeffs_E_2.get(), coeffs_B_3.get(), coeffs_J_Hf_1.get(),
    								coeffs_J_Hf_2.get(), conf, stride_t);

    	// Compute next E.
    	compute_E(nt, coeffs_E_1.get(), coeffs_E_2.get(), coeffs_B_3.get(), coeffs_J_Hf_1.get(), coeffs_J_Hf_2.get(), avrg_J_Hf_1.data(),
    				avrg_J_Hf_2.data(), conf, stride_t);

    	// Compute next B.
    	compute_B(nt, coeffs_E_2.get(), coeffs_B_3.get(), conf, stride_t);

    	// Do output etc.
    	size_t n_plot = 128;
    	double dx_plot = (conf.x_max-conf.x_min)/n_plot;
        std::ofstream E_1_nt_str("E_1_" + std::to_string(nt*conf.dt) + ".txt");
        std::ofstream E_2_nt_str("E_2_" + std::to_string(nt*conf.dt) + ".txt");
        std::ofstream B_3_nt_str("B_3_" + std::to_string(nt*conf.dt) + ".txt");
        elec_energy = 0;
        magn_energy = 0;
    	for(size_t i = 0; i < n_plot; i++)
    	{
    		double x = conf.x_min + i*dx_plot;
    		double E1 = dergeraet::dim_1_half::eval<double,order>(x, coeffs_E_1.get() + nt*stride_t, conf);
    		double E2 = dergeraet::dim_1_half::eval<double,order>(x, coeffs_E_2.get() + nt*stride_t, conf);
    		double B3 = dergeraet::dim_1_half::eval<double,order>(x, coeffs_B_3.get() + nt*stride_t, conf);

    		elec_energy += E1*E1 + E2*E2;
    		magn_energy += B3*B3;

        	E_1_nt_str << x << " " << E1 << std::endl;
        	E_2_nt_str << x << " " << E2 << std::endl;
        	B_3_nt_str << x << " " << B3 << std::endl;
    	}
        elec_energy *= dx_plot;
        magn_energy *= dx_plot;
        Electric_Energy_str << nt*conf.dt << " " << elec_energy << std::endl;
        Magnetic_Energy_str << nt*conf.dt << " " << magn_energy << std::endl;
        std::cout << "E_energy: " << nt*conf.dt << " " << elec_energy << std::endl;
        std::cout << "B_energy: " << nt*conf.dt << " " << magn_energy << std::endl;
        std::cout << "--------------------------------------" << std::endl;
    }

	return 0;
}
