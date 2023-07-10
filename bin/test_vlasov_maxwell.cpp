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

#include <iomanip>

#include <dergeraet/forces.hpp>
#include <dergeraet/poisson.hpp>
#include <dergeraet/rho.hpp>
#include <dergeraet/stopwatch.hpp>

namespace dergeraet
{

namespace dim3
{

void get_weak_landau_first_two_time_step_electric_field(const config_t<double>& param,
		double* phi_0, double* phi_1, double* E0_x, double* E0_y, double* E0_z,
		double* E1_x, double* E1_y, double* E1_z)
{
    poisson<double> poiss( param );

    size_t stride_t = (param.Nx + electro_magnetic_force::order - 1) *
                      (param.Ny + electro_magnetic_force::order - 1) *
					  (param.Nz + electro_magnetic_force::order - 1);

    size_t Nt = 0;//2.0 / param.dt;
    //size_t Nt = 1;
    std::unique_ptr<double[]> coeffs { new double[ (Nt+1)*stride_t ] {} };

    void *tmp = std::aligned_alloc( poiss.alignment, sizeof(double)*param.Nx*param.Ny*param.Nz );
    if ( tmp == nullptr ) throw std::bad_alloc {};
    std::unique_ptr<double,decltype(std::free)*> rho { reinterpret_cast<double*>(tmp),
    													std::free };

	std::ofstream E_max_str("E_max.txt");
	//std::ofstream j_max_str("j_max.txt");
	//std::ofstream j_l2_str("j_l2.txt");

    std::cout << "Start precompute: t = 0." << std::endl;
    // t = 0:
    // Compute rho.
	#pragma omp parallel for
    for(size_t l = 0; l < param.Nx*param.Ny*param.Nz; l++)
    {
    	rho.get()[l] = dergeraet::dim3::eval_rho<double, electro_magnetic_force::order>( 0, l, coeffs.get(), param);
    }
    // Solve for phi.
    poiss.solve( rho.get() );
    // Interpolate phi.
    interpolate<double,electro_magnetic_force::order>( coeffs.get(), rho.get(), param );
    // Store the required values of E0.
    double Emax = 0;
    //double j_max = 0;
    //double j_l2 = 0;
	#pragma omp parallel for
    for(size_t l = 0; l < param.Nx*param.Ny*param.Nz; l++)
    {
        size_t k   = l   / (param.Nx * param.Ny);
        size_t tmp = l   % (param.Nx * param.Ny);
        size_t j   = tmp / param.Nx;
        size_t i   = tmp % param.Nx;

        double x = param.x_min + i*param.dx;
        double y = param.y_min + j*param.dy;
        double z = param.z_min + k*param.dz;

        phi_0[l] = eval<double, electro_magnetic_force::order, 0, 0, 0>(x,y,z,coeffs.get(), param);

        E0_x[l] = -eval<double, electro_magnetic_force::order, 1, 0, 0>(x,y,z,coeffs.get(), param);
        E0_y[l] = -eval<double, electro_magnetic_force::order, 0, 1, 0>(x,y,z,coeffs.get(), param);
        E0_z[l] = -eval<double, electro_magnetic_force::order, 0, 0, 1>(x,y,z,coeffs.get(), param);

        Emax = std::max(Emax, std::sqrt(E0_x[l]*E0_x[l] + E0_y[l]*E0_y[l] + E0_z[l]*E0_z[l]));


		// eval j at x,y,z:
        /*
		double j_1 = 0;
		double j_2 = 0;
		double j_3 = 0;
		for(size_t iv = 0; iv < param.Nv; iv++)
		for(size_t iu = 0; iu < param.Nu; iu++)
		for(size_t iw = 0; iw < param.Nw; iw++)
		{
			double v = param.v_min + double(iv + 0.5) * param.dv;
			double u = param.u_min + double(iu + 0.5) * param.du;
			double w = param.w_min + double(iw + 0.5) * param.dw;

			double f = eval_f<double, electro_magnetic_force::order>( 0, x, y, z, v, u, w, coeffs.get(), param);
			j_1 += v*f;
			j_2 += u*f;
			j_3 += w*f;
		}
		j_max = std::max(j_max, std::sqrt(j_1*j_1 + j_2*j_2 + j_3*j_3));
		j_l2 += j_1*j_1 + j_2*j_2 + j_3*j_3;
		*/
    }

    E_max_str << 0 << " " << Emax << std::endl;
    //j_max_str << 0 << " " << j_max << std::endl;
    //j_l2_str << 0 << " " << j_l2 << std::endl;

    std::cout << "Start precompute: t = dt." << std::endl;
    // t = dt.
    // Compute rho.
	#pragma omp parallel for
    for(size_t l = 0; l < param.Nx*param.Ny*param.Nz; l++)
    {
    	rho.get()[l] = eval_rho<double, electro_magnetic_force::order>( 1, l, coeffs.get() + stride_t, param);
    }
    // Solve for phi.
    poiss.solve( rho.get() );
    // Interpolate phi.
    interpolate<double,electro_magnetic_force::order>( coeffs.get() + stride_t, rho.get(), param );
    // Store the required values of E1.
    Emax = 0;
    //j_max = 0;
    //j_l2 = 0;
	#pragma omp parallel for
    for(size_t l = 0; l < param.Nx*param.Ny*param.Nz; l++)
    {
        size_t k   = l   / (param.Nx * param.Ny);
        size_t tmp = l   % (param.Nx * param.Ny);
        size_t j   = tmp / param.Nx;
        size_t i   = tmp % param.Nx;

        double x = param.x_min + i*param.dx;
        double y = param.y_min + j*param.dy;
        double z = param.z_min + k*param.dz;

        phi_1[l] = eval<double, electro_magnetic_force::order, 0, 0, 0>(x,y,z,coeffs.get() + stride_t, param);

        E1_x[l] = -eval<double, electro_magnetic_force::order, 1, 0, 0>(x,y,z,coeffs.get() + stride_t, param);
        E1_y[l] = -eval<double, electro_magnetic_force::order, 0, 1, 0>(x,y,z,coeffs.get() + stride_t, param);
        E1_z[l] = -eval<double, electro_magnetic_force::order, 0, 0, 1>(x,y,z,coeffs.get() + stride_t, param);
        Emax = std::max(Emax, std::sqrt(E1_x[l]*E1_x[l] + E1_y[l]*E1_y[l] + E1_z[l]*E1_z[l]));

		// eval j at x,y,z:
        /*
		double j_1 = 0;
		double j_2 = 0;
		double j_3 = 0;
		for(size_t iv = 0; iv < param.Nv; iv++)
		for(size_t iu = 0; iu < param.Nu; iu++)
		for(size_t iw = 0; iw < param.Nw; iw++)
		{
			double v = param.v_min + double(iv + 0.5) * param.dv;
			double u = param.u_min + double(iu + 0.5) * param.du;
			double w = param.w_min + double(iw + 0.5) * param.dw;

			double f = eval_f<double, electro_magnetic_force::order>( 1, x, y, z, v, u, w, coeffs.get() + 1*stride_t, param);
			j_1 += v*f;
			j_2 += u*f;
			j_3 += w*f;
		}
		j_max = std::max(j_max, std::sqrt(j_1*j_1 + j_2*j_2 + j_3*j_3));
		j_l2 += j_1*j_1 + j_2*j_2 + j_3*j_3;
		*/
    }

    E_max_str << param.dt << " " << Emax << std::endl;
    //j_max_str << param.dt << " " << j_max << std::endl;
    //j_l2_str << param.dt << " " << j_l2 << std::endl;

    // t > dt.
    for(size_t n = 2; n <= Nt; n++)
    {
        std::cout << "Start precompute: t = " << n << "*dt." << std::endl;
        // t = n*dt.
        // Compute rho.
    	#pragma omp parallel for
        for(size_t l = 0; l < param.Nx*param.Ny*param.Nz; l++)
        {
        	rho.get()[l] = eval_rho<double, electro_magnetic_force::order>( n, l, coeffs.get() + n*stride_t, param);
        }
        // Solve for phi.
        poiss.solve( rho.get() );
        // Interpolate phi.
        interpolate<double,electro_magnetic_force::order>( coeffs.get() + n*stride_t, rho.get(), param );
        // Plot.
        Emax = 0;
        double j_max = 0;
        double j_l2 = 0;
		//#pragma omp parallel for
        std::ofstream E_str("Exact_Electric_field_" + std::to_string(n*param.dt) +  ".txt");
        //std::ofstream j_str("j_exact_" + std::to_string(n*param.dt) +  ".txt");
		#pragma omp parallel for
		for(size_t l = 0; l < param.Nx*param.Ny*param.Nz; l++)
		{
			size_t k   = l   / (param.Nx * param.Ny);
			size_t tmp = l   % (param.Nx * param.Ny);
			size_t j   = tmp / param.Nx;
			size_t i   = tmp % param.Nx;

			double x = param.x_min + i*param.dx;
			double y = param.y_min + j*param.dy;
			double z = param.z_min + k*param.dz;

			double Ex = -eval<double, electro_magnetic_force::order, 1, 0, 0>(x,y,z,coeffs.get() + n*stride_t, param);
			double Ey = -eval<double, electro_magnetic_force::order, 0, 1, 0>(x,y,z,coeffs.get() + n*stride_t, param);
			double Ez = -eval<double, electro_magnetic_force::order, 0, 0, 1>(x,y,z,coeffs.get() + n*stride_t, param);
			Emax = std::max(Emax, std::sqrt(Ex*Ex + Ey*Ey + Ez*Ez));

			E_str << x << " " << y << " " << z << " " << Ex << " " << Ey << " " << Ez << std::endl;

			// eval j at x,y,z:
			/*
			double j_1 = 0;
			double j_2 = 0;
			double j_3 = 0;
			for(size_t iv = 0; iv < param.Nv; iv++)
			for(size_t iu = 0; iu < param.Nu; iu++)
			for(size_t iw = 0; iw < param.Nw; iw++)
			{
				double v = param.v_min + double(iv + 0.5) * param.dv;
				double u = param.u_min + double(iu + 0.5) * param.du;
				double w = param.w_min + double(iw + 0.5) * param.dw;

				double f = eval_f<double, electro_magnetic_force::order>( n, x, y, z, v, u, w, coeffs.get() + n*stride_t, param);
				j_1 += v*f;
				j_2 += u*f;
				j_3 += w*f;
			}
			j_str << x << " " << y << " " << z << " " << j_1 << " " << j_2 << " " << j_3 << std::endl;
			j_max = std::max(j_max, std::sqrt(j_1*j_1 + j_2*j_2 + j_3*j_3));
			j_l2 += j_1*j_1 + j_2*j_2 + j_3*j_3;
			*/
		}

		E_max_str << n*param.dt << " " << Emax << std::endl;
		//j_max_str << n*param.dt << " " << j_max << std::endl;
		//j_l2_str << n*param.dt << " " << std::sqrt(param.dv*param.dw*param.du*j_l2) << std::endl;
    }
}

void test_landau_damping()
{
	stopwatch<double> timer_total;
	// Initialize config.
	config_t<double> param;
	//param.Nt = 32;
	double alpha = 0.01;
	double kx = 0.5;
	double ky = 0.5;
	double kz = 0.5;

	// Initialize phi, A vectors at t=0,t_1.
	std::vector<double> phi_0(param.Nx*param.Ny*param.Nz, 0);
	std::vector<double> A_x_0(param.Nx*param.Ny*param.Nz, 0);
	std::vector<double> A_y_0(param.Nx*param.Ny*param.Nz, 0);
	std::vector<double> A_z_0(param.Nx*param.Ny*param.Nz, 0);

	std::vector<double> phi_1(param.Nx*param.Ny*param.Nz, 0);
	std::vector<double> A_x_1(param.Nx*param.Ny*param.Nz, 0);
	std::vector<double> A_y_1(param.Nx*param.Ny*param.Nz, 0);
	std::vector<double> A_z_1(param.Nx*param.Ny*param.Nz, 0);

	// Init E_0 and E_1.
	std::vector<double> E_x_0(param.Nx*param.Ny*param.Nz, 0);
	std::vector<double> E_y_0(param.Nx*param.Ny*param.Nz, 0);
	std::vector<double> E_z_0(param.Nx*param.Ny*param.Nz, 0);

	std::vector<double> E_x_1(param.Nx*param.Ny*param.Nz, 0);
	std::vector<double> E_y_1(param.Nx*param.Ny*param.Nz, 0);
	std::vector<double> E_z_1(param.Nx*param.Ny*param.Nz, 0);

	// Compute values for phi, A at t=0,t_1.
	get_weak_landau_first_two_time_step_electric_field(param,
						phi_0.data(), phi_1.data(),
						E_x_0.data(), E_y_0.data(), E_z_0.data(),
						E_x_1.data(), E_y_1.data(), E_z_1.data());

	// Init electro_magnetic_force.
	electro_magnetic_force emf(phi_0.data(), phi_1.data(), A_x_0.data(), A_x_1.data(),
				 A_y_0.data(), A_y_1.data(), A_z_0.data(), A_z_1.data(), E_x_0.data(),
				 E_x_1.data(), E_y_0.data(), E_y_1.data(), E_z_0.data(), E_z_1.data(),
				 param);
	// Arrays to store values of rho and j at previous time-step.
	std::vector<std::vector<double>> rj(4, std::vector<double>(param.Nx*param.Ny*param.Nz,0) );

	double init_time = timer_total.elapsed();
	std::cout << "Init done." << std::endl;
	std::cout << "Init time = " << init_time << std::endl;
	timer_total.reset();

	std::ofstream E_norm("E_norm.txt");
	E_norm << 0 << " " << emf.E_norm(0) << std::endl;
	std::cout << "t = " << 0 << ". E_norm = " << emf.E_norm(0) << std::endl;
	E_norm << param.dt << " " << emf.E_norm(1) << std::endl;
	std::cout << "t = " << param.dt << ". E_norm = " << emf.E_norm(1) << std::endl;

	std::ofstream B_norm("B_norm.txt");
	B_norm << 0 << " " << emf.B_norm(0) << std::endl;
	std::cout << "t = " << 0 << ". B_norm = " << emf.B_norm(0) << std::endl;
	B_norm << param.dt << " " << emf.B_norm(1) << std::endl;
	std::cout << "t = " << param.dt << ". B_norm = " << emf.B_norm(1) << std::endl;


	// Do NuFI loop.
	for(size_t t = 2; t <= param.Nt; t++)
	{
		stopwatch<double> inner_timer;
		// Get values of rho and j at previous time-step.
		rj = emf.eval_rho_j(t-1);

		// Compute next time-step.
		emf.solve_next_time_step(rj[0].data(), rj[1].data(), rj[2].data(), rj[3].data());

		// Test/output.
		double E_n = emf.E_norm(t);
		E_norm << t*param.dt << " " << E_n << std::endl;
		std::cout << "This time-step took " << inner_timer.elapsed()
				<< ". t = " << t*param.dt
				<< ". E_norm = " << E_n << std::endl;

		double B_n = emf.B_norm(t);
		B_norm << t*param.dt << " " << B_n << std::endl;
		std::cout << "This time-step took " << inner_timer.elapsed()
				<< ". t = " << t*param.dt
				<< ". B_norm = " << B_n << std::endl;

		/*
		for(size_t i = 0; i < param.Nx; i++)
		{
			for(size_t j = 0; j < param.Ny; j++)
			{
				for(size_t k = 0; k < param.Nz; k++)
				{
					size_t index = i + j*param.Nx + k*param.Nx*param.Ny;

					std::cout << rj[1][index] << " " << rj[2][index] << " " << rj[3][index] << std::endl;

					// There are non-zero entries already at tn = 3. Check if this is
					// the same for the electro-static case.
				}
			}
		}
		*/

	}

	double loop_time = timer_total.elapsed();
	double total_time = init_time + loop_time;
	std::cout << "Loop time = " << loop_time << std::endl;
	std::cout << "Total computation time = " << total_time << std::endl;
}


}
}



int main()
{
	dergeraet::dim3::test_landau_damping();

	return 0;
}
