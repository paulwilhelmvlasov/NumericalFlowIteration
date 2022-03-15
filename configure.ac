AC_INIT([dergeraet],[0.1])
AC_CANONICAL_SYSTEM
AC_CONFIG_MACRO_DIR([m4])

AM_INIT_AUTOMAKE([-Wall -Werror nostdinc dist-xz])
AM_SILENT_RULES([yes])
AM_PROG_AR

AC_LANG([C++])
AC_PROG_CC
AC_PROG_CXX

AX_CXX_COMPILE_STDCXX_17([noext],[mandatory])

AC_SUBST([AM_CPPFLAGS],['-I$(top_srcdir)'])

AC_CONFIG_HEADERS([dergeraet/autoconfig.hpp])
AC_CONFIG_FILES([Makefile
                 dergeraet/makefile
                 bin/makefile])

LT_INIT
AC_OUTPUT

