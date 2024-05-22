AC_INIT([dergeraet],[0.1])
AC_CANONICAL_TARGET
AC_CONFIG_MACRO_DIR([m4])

AM_INIT_AUTOMAKE([nostdinc dist-xz])
AM_SILENT_RULES([no])
AM_PROG_AR
AC_PROG_RANLIB

AC_LANG([C++])
AC_PROG_CC
AC_PROG_CXX
AC_OPENMP
AC_SUBST([AM_CXXFLAGS],["$OPENMP_CXXFLAGS"]) 
AX_CXX_COMPILE_STDCXX_17([noext],[mandatory])
AX_CHECK_CUDA
AX_BLAS

AC_SUBST([AM_CPPFLAGS],['-I$(top_srcdir)'])

AC_CONFIG_HEADERS([dergeraet/autoconfig.hpp])
AC_CONFIG_FILES([Makefile
                 AnjaMathematica/makefile
                 dergeraet/makefile
                 bin/makefile])

AC_OUTPUT

