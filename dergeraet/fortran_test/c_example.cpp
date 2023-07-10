// c_example.cpp
// 
// Author: Walter Dal'Maz Silva
// Date  : Jan 6 2019

#include <iostream>
#include <iomanip>
#include "c_example.hpp"

int main()
{
    double val = 0.0;
    example_type *ptr = new example_type {3.0, 4.0};

    std::cout << std::fixed << std::setprecision(2)
              << "\n Before Fortran call"
              << "\n x = " << ptr->x
              << "\n y = " << ptr->y
              << "\n v = " << val
              << "\n" << std::endl;

    c_example((void*)&ptr, &val);

    std::cout << std::fixed << std::setprecision(2)
              << "\n After Fortran call"
              << "\n x = " << ptr->x
              << "\n y = " << ptr->y
              << "\n v = " << val
              << "\n" << std::endl;
    return 0;
}
