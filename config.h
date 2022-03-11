/*
 * config.h
 *
 *  Created on: Mar 9, 2022
 *      Author: paul
 */

#ifndef CONFIG_H_
#define CONFIG_H_

#include <math.h>

const double Pi = M_PI;

const double L0x = 0; 
const double L0y = 0; 
const double L0z = 0; 

const double L1x = 2 * Pi; 
const double L1y = 2 * Pi; 
const double L1z = 2 * Pi; 

const double Lx = L1x - L0x; 
const double Ly = L1y - L0y; 
const double Lz = L1z - L0z; 

const size_t Nx = 21;
const size_t Ny = 21;
const size_t Nz = 21;

const double dx = Lx / Nx;
const double dy = Ly / Ny;
const double dz = Lz / Nz;


#endif /* CONFIG_H_ */
