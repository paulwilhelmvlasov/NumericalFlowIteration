cdef extern from "func.cpp":
    double f(double u, double v, double w)

def compute_function_py(u, v, w) -> double:
    return f(u, v, w)
