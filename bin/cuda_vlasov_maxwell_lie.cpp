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
#include <numeric>
#include <vector>

#include <dergeraet/fields.hpp>
#include <dergeraet/cuda_kernel.hpp>
#include <dergeraet/stopwatch.hpp>

namespace dergeraet
{
namespace dim_1_half
{
template <size_t order>
void init_E_B(double* coeffs_E_1, double* coeffs_E_2, double* coeffs_B_3, const config_t<double>& conf)
{
	std::vector<double> E_1(conf.Nx, 0);
	std::vector<double> E_2(conf.Nx, 0);
	std::vector<double> B_3(conf.Nx, 0);

    for(size_t i = 0; i < conf.Nx; i++)
    {
    	//  Weibel instability:

    	double alpha = 1e-4;
    	double beta = 1e-4;
    	double k = 1.25;
    	double x = conf.x_min + i*conf.dx;
    	E_1[i] = alpha/k*std::sin(k*x);
    	E_2[i] = 0;
    	B_3[i] = beta*std::cos(k*x);

    	// Weak Landau Damping:
    	/*
    	double alpha = 1e-2;
    	double k = 0.5;
    	double x = conf.x_min + i*conf.dx;
    	E_1[i] = alpha/k*std::sin(k*x);
    	E_2[i] = 0;
    	B_3[i] = 0;
		*/
    }

    // Interpolate E_0 and B_0.
    interpolate<double,order>(coeffs_E_1, E_1.data(), conf);
    interpolate<double,order>(coeffs_E_2, E_2.data(), conf);
    interpolate<double,order>(coeffs_B_3, B_3.data(), conf);
}

template <size_t order>
double compute_electric_energy(size_t nt, const double* coeffs_E_1,
		const double* coeffs_E_2, const config_t<double>& conf,
		size_t n_plot = 128, bool plot = false)
{
	if(nt > conf.Nt)
	{
		throw std::runtime_error("Error: nt > conf.Nt.");
	}

    const size_t stride_t = conf.Nx + order - 1;

	double dx_plot = (conf.x_max-conf.x_min)/n_plot;
	double E_energy = 0;
	if(plot){
		std::ofstream E_1_str("E_1_" + std::to_string(nt*conf.dt) + ".txt");
		std::ofstream E_2_str("E_2_" + std::to_string(nt*conf.dt) + ".txt");

		E_1_str << conf.x_min << " " << eval<double,order>(conf.x_min, coeffs_E_1 + nt*stride_t, conf) << std::endl;
		E_2_str << conf.x_min << " " << eval<double,order>(conf.x_min, coeffs_E_2 + nt*stride_t, conf) << std::endl;
		for(size_t i = 0; i < n_plot; i++)
		{
			double x = conf.x_min + (i+0.5)*dx_plot;
			double E1 = eval<double,order>(x, coeffs_E_1 + nt*stride_t, conf);
			double E2 = eval<double,order>(x, coeffs_E_2 + nt*stride_t, conf);

			E_energy += E1*E1 + E2*E2;

			E_1_str << x << " " << E1 << std::endl;
			E_2_str << x << " " << E2 << std::endl;
		}
		E_1_str << conf.x_max << " " << eval<double,order>(conf.x_max, coeffs_E_1 + nt*stride_t, conf) << std::endl;
		E_2_str << conf.x_max << " " << eval<double,order>(conf.x_max, coeffs_E_2 + nt*stride_t, conf) << std::endl;
	}else{
		for(size_t i = 0; i < n_plot; i++)
		{
			double x = conf.x_min + (i+0.5)*dx_plot;
			double E1 = eval<double,order>(x, coeffs_E_1 + nt*stride_t, conf);
			double E2 = eval<double,order>(x, coeffs_E_2 + nt*stride_t, conf);

			E_energy += E1*E1 + E2*E2;
		}
	}

	E_energy *= 0.5*dx_plot;

	return E_energy;
}

template <size_t order>
double compute_magnetic_energy(size_t nt, const double* coeffs_B_3,
		const config_t<double>& conf, size_t n_plot = 128, bool plot = false)
{
	if(nt > conf.Nt)
	{
		throw std::runtime_error("Error: nt > conf.Nt.");
	}

    const size_t stride_t = conf.Nx + order - 1;

    double dx_plot = (conf.x_max-conf.x_min)/n_plot;
    double B_energy = 0;
	if(plot){
		std::ofstream B_3_str("B_3_" + std::to_string(nt*conf.dt) + ".txt");

		B_3_str << conf.x_min << " " << eval<double,order>(conf.x_min, coeffs_B_3 + nt*stride_t, conf) << std::endl;
		for(size_t i = 0; i < n_plot; i++)
		{
			double x = conf.x_min + (i+0.5)*dx_plot;
			double B3 = eval<double,order>(x, coeffs_B_3 + nt*stride_t, conf);

			B_energy += B3*B3;

			B_3_str << x << " " << B3 << std::endl;
		}
		B_3_str << conf.x_max << " " << eval<double,order>(conf.x_max, coeffs_B_3 + nt*stride_t, conf) << std::endl;
	}else{
		for(size_t i = 0; i < n_plot; i++)
		{
			double x = conf.x_min + (i+0.5)*dx_plot;
			double B3 = eval<double,order>(x, coeffs_B_3 + nt*stride_t, conf);

			B_energy += B3*B3;
		}
	}

	B_energy *= 0.5*dx_plot;

	return B_energy;
}

template <size_t order>
void compute_E(size_t nt, double* coeffs_E_1, double* coeffs_E_2, double* coeffs_B_3,
			double* coeffs_J_Hf_1, double* coeffs_J_Hf_2, const config_t<double>& conf)
{
	// Given nt > 0 computes the coefficients of E_1(n_t) and E_2(n_t).
	if(nt == 0)
	{
		throw std::runtime_error("nt > 0 expected. E_0 already computed.");
	}

	const size_t stride_t = conf.Nx + order - 1;

	std::vector<double> E_values_1(conf.Nx, 0);
	std::vector<double> E_values_2(conf.Nx, 0);

	#pragma omp parallel for
	for(size_t i = 0; i< conf.Nx; i++)
	{
		double x = conf.x_min + i*conf.dx;


		E_values_1[i] = eval<double,order>(x, coeffs_E_1 + (nt-1)*stride_t, conf)
						- conf.dt*eval<double,order>(x, coeffs_J_Hf_1 + (nt-1)*stride_t, conf);

		E_values_2[i] = eval<double,order>(x, coeffs_E_2 + (nt-1)*stride_t, conf)
						- conf.dt*eval<double,order,1>(x, coeffs_B_3 + (nt-1)*stride_t, conf)
						+ conf.dt*conf.dt*eval<double,order,2>(x, coeffs_E_2 + (nt-1)*stride_t, conf)
						- conf.dt*eval<double,order>(x, coeffs_J_Hf_2 + (nt-1)*stride_t, conf);
	}

	interpolate<double,order>(coeffs_E_1 + nt*stride_t, E_values_1.data(), conf);
	interpolate<double,order>(coeffs_E_2 + nt*stride_t, E_values_2.data(), conf);
}

template <size_t order>
void compute_B(size_t nt, double* coeffs_E_2, double* coeffs_B_3,
		const config_t<double>& conf)
{
	// Given nt > 0 computes the coefficients of B_3(n_t). E_1(nt-1), E_2(nt-1) and B_3(nt-1) is known.

	if(nt == 0)
	{
		throw std::runtime_error("nt > 0 expected. B_0 already computed.");
	}

	const size_t stride_t = conf.Nx + order - 1;

	std::vector<double> B_values(conf.Nx, 0);

	#pragma omp parallel for
	for(size_t i = 0; i< conf.Nx; i++)
	{
		double x = conf.x_min + i*conf.dx;

		B_values[i] = eval<double,order>(x, coeffs_B_3 + (nt-1)*stride_t, conf)
				- conf.dt*eval<double,order,1>(x, coeffs_E_2 + (nt-1)*stride_t, conf);
	}

	interpolate<double,order>(coeffs_B_3 + nt*stride_t, B_values.data(), conf);
}

void output(double* metrics, double E_energy, double B_energy, std::ofstream& stats_file,
		double t, double comp_time_step)
{
	double total_energy = E_energy + B_energy + metrics[2];

    for ( size_t i = 0; i < 80; ++i )
        std::cout << '=';
    std::cout << std::endl;

    std::cout << "t = " << t << ". Compute time = " << comp_time_step << ".\n";

    std::cout << "L¹-Norm:      " << std::setw(20) << metrics[0]      << std::endl;
    std::cout << "L²-Norm:      " << std::setw(20) << metrics[1]      << std::endl;
    std::cout << "Total energy: " << std::setw(20) << total_energy << std::endl;
    std::cout << "Entropy:      " << std::setw(20) << metrics[3]      << std::endl;

    std::cout << std::endl;

    stats_file << t << "; "
               << metrics[0] << "; "
               << metrics[1] << "; "
               << E_energy << "; "
               << B_energy << "; "
			   << metrics[2] << "; "
               << total_energy << "; "
               << metrics[3] << std::endl;
}

template <size_t order>
void test()
{
	dergeraet::stopwatch<double> timer_init;
    config_t<double> conf;

    const size_t stride_t = conf.Nx + order - 1;
    const size_t N = conf.Nx*conf.Nu*conf.Nv;

    std::unique_ptr<double[]> coeffs_E_1 { new double[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<double[]> coeffs_E_2 { new double[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<double[]> coeffs_B_3 { new double[ (conf.Nt+1)*stride_t ] {} };
    std::unique_ptr<double[]> coeffs_J_Hf_1 { new double[ (conf.Nt)*stride_t ] {} };
    std::unique_ptr<double[]> coeffs_J_Hf_2 { new double[ (conf.Nt)*stride_t ] {} };

    double metrics[4] = { 0, 0, 0, 0 };

	cuda_kernel_vm<double,order> ck_vm(conf);

    std::ofstream statistics_file( "statistics.csv" );
    statistics_file << R"("Time"; "L1-Norm"; "L2-Norm"; "Electric Energy"; "Magnetic Energy"; "Kinetic Energy"; "Total Energy"; "Entropy")";
    statistics_file << std::endl;

    statistics_file << std::scientific;
    std::cout << std::scientific;

    init_E_B<order>(coeffs_E_1.get(), coeffs_E_2.get(), coeffs_B_3.get(), conf);
    ck_vm.upload_coeffs(0, coeffs_E_1.get(), coeffs_E_2.get(), coeffs_B_3.get());
    double init_time = timer_init.elapsed();

    double E_energy = compute_electric_energy<order>(0, coeffs_E_1.get(),
    							coeffs_E_2.get(), conf, 128, true);
    double B_energy = compute_magnetic_energy<order>(0, coeffs_B_3.get(),
													conf, 128, true);

    ck_vm.compute_metrics(0, 0, N);
    ck_vm.download_metrics(metrics);

    output(metrics, E_energy, B_energy, statistics_file, 0, init_time);

    double total_time = 0;
    for(size_t nt = 1; nt <= conf.Nt; nt++)
    {
    	dergeraet::stopwatch<double> timer;
    	// Compute next J_Hf.
    	std::vector<double> j_hf_1(conf.Nx,0);
    	std::vector<double> j_hf_2(conf.Nx,0);
    	ck_vm.compute_j_hf(nt, 0, N);
    	ck_vm.download_j_hf(j_hf_1.data(), j_hf_2.data());
    	interpolate<double,order>(coeffs_J_Hf_1.get() + (nt-1)*stride_t, j_hf_1.data(), conf);
    	interpolate<double,order>(coeffs_J_Hf_2.get() + (nt-1)*stride_t, j_hf_2.data(), conf);

    	// Compute next E.
    	compute_E<order>(nt, coeffs_E_1.get(), coeffs_E_2.get(), coeffs_B_3.get(),
    				coeffs_J_Hf_1.get(), coeffs_J_Hf_2.get(), conf);

    	// Compute next B.
    	compute_B<order>(nt, coeffs_E_2.get(), coeffs_B_3.get(), conf);

    	// Upload E and B coefficients to GPU.
    	ck_vm.upload_coeffs(nt, coeffs_E_1.get(), coeffs_E_2.get(), coeffs_B_3.get());

    	// Do output.
    	// This should be completly outsourced to the output function!
        double time_elapsed = timer.elapsed();
        total_time += time_elapsed;
        if(nt % 4 == 0)
        {
			dergeraet::stopwatch<double> timer_metrics;
			metrics[0] = 0;
			metrics[1] = 0;
			metrics[2] = 0;
			metrics[3] = 0;
			ck_vm.compute_metrics(nt, 0, N);
			ck_vm.download_metrics(metrics);
			if(nt % 16 == 0){
				E_energy = compute_electric_energy<order>(nt, coeffs_E_1.get(),
											coeffs_E_2.get(), conf, 128, true);
				B_energy = compute_magnetic_energy<order>(nt, coeffs_B_3.get(),
																conf, 128, true);
			}else{
				E_energy = compute_electric_energy<order>(nt, coeffs_E_1.get(),
											coeffs_E_2.get(), conf, 128, false);
				B_energy = compute_magnetic_energy<order>(nt, coeffs_B_3.get(),
																conf, 128, false);
			}
			output(metrics, E_energy, B_energy, statistics_file, nt*conf.dt, time_elapsed);
			double time_with_metrics = time_elapsed + timer_metrics.elapsed();
			std::cout << "Time for time step with metrics: " << time_with_metrics << std::endl;
        }else{
            for ( size_t i = 0; i < 80; ++i )
                std::cout << '=';
            std::cout << std::endl;
            std::cout << "t = " << nt*conf.dt << ". Compute time = " << time_elapsed << ".\n";
        }
    }
}


}
}

int main()
{
    dergeraet::dim_1_half::test<4>();
}
