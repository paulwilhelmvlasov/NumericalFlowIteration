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
#ifndef DERGERAET_STOPWATCH_HPP
#define DERGERAET_STOPWATCH_HPP

#include <chrono>

namespace dergeraet
{

template <typename real>
class stopwatch
{
public:
	void reset();
	real elapsed();

private:
	using clock = std::chrono::high_resolution_clock;
	clock::time_point t0 { clock::now() };
};


template <typename real> inline
void stopwatch<real>::reset()
{
	t0 = clock::now();
}

template <typename real> inline
real stopwatch<real>::elapsed()
{
	using seconds = std::chrono::duration<real,std::ratio<1,1>>;

	auto tnow = clock::now();
	auto duration = std::chrono::duration_cast<seconds>( tnow - t0 );

	return duration.count();
}

}

#endif

