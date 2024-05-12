cdef extern from "func.cpp":
    double func(double u, double v, double w)

def compute_function_py(u, v, w) -> double:
    return func(u, v, w)
