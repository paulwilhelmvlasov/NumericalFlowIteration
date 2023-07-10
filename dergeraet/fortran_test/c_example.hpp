// c_example.hpp
//
// Author: Walter Dal'Maz Silva
// Date  : Jan 6 2019

#ifndef __C_EXAMPLE_HPP__
#define __C_EXAMPLE_HPP__

// Consider compatibility with plain C.
#ifdef __cplusplus
extern "C" {
#endif

// This has the same face as the example_type in module_f_example.95 but
// it was nowhere declared in (see the file for more) module_c_example.F95.
typedef struct {
    double x;
    double y;
} example_type;

// Function interfaced at module_c_example.F95
void c_example(void* mstruc, double* val);

#ifdef __cplusplus
}
#endif

#endif // (__C_EXAMPLE_HPP__)
