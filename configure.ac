AC_INIT([nufi],[0.1])
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

# Check for MPI C++ compiler
AC_CHECK_PROGS([MPICXX], [mpic++], [no])
if test "$MPICXX" != "no"; then
    AC_MSG_NOTICE([MPI C++ compiler found: $MPICXX])
    CXX=$MPICXX
else
    AC_MSG_ERROR([MPI C++ compiler (mpic++) not found])
fi

AX_BLAS

AC_SUBST([AM_CPPFLAGS],['-I$(top_srcdir)'])

AC_CONFIG_HEADERS([nufi/autoconfig.hpp])
AC_CONFIG_FILES([Makefile
                 nufi/makefile
                 bin/makefile])

AC_OUTPUT

