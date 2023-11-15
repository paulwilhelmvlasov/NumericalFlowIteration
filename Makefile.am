ACLOCAL_AMFLAGS = -I m4

lib_LTLIBRARIES = dergeraet/libdergeraet.la
dergeraet_libdergeraet_la_SOURCES = dergeraet/poisson.cpp dergeraet/lapack.cpp \
                                    dergeraet/blas.cpp dergeraet/device_kernel.cpp
dergeraet_libdergeraet_la_LDFLAGS = -version-info 0:0:0
                                    

noinst_PROGRAMS = bin/test_dergeraet bin/test_dergeraet_2d bin/test_dergeraet_3d \
                  bin/test_gmres bin/test_poisson bin/test_fields
 

bin_test_dergeraet_SOURCES = bin/test_dergeraet.cpp
bin_test_dergeraet_LDADD   = $(top_builddir)/dergeraet/libdergeraet.la $(BLAS_LIBS) -lfftw3 -lfftw3f

bin_test_dergeraet_2d_SOURCES = bin/test_dergeraet_2d.cpp
bin_test_dergeraet_2d_LDADD   = $(top_builddir)/dergeraet/libdergeraet.la $(BLAS_LIBS) -lfftw3 -lfftw3f

bin_test_dergeraet_3d_SOURCES = bin/test_dergeraet_3d.cpp
bin_test_dergeraet_3d_LDADD   = $(top_builddir)/dergeraet/libdergeraet.la $(BLAS_LIBS) -lfftw3 -lfftw3f

bin_test_gmres_SOURCES = bin/test_gmres.cpp
bin_test_gmres_LDADD   = $(top_builddir)/dergeraet/libdergeraet.la $(BLAS_LIBS) -lfftw3 -lfftw3f

bin_test_poisson_SOURCES = bin/test_poisson.cpp
bin_test_poisson_LDADD   = $(top_builddir)/dergeraet/libdergeraet.la $(BLAS_LIBS) -lfftw3 -lfftw3f

bin_test_fields_SOURCES = bin/test_fields.cpp
bin_test_fields_LDADD   = $(top_builddir)/dergeraet/libdergeraet.la $(BLAS_LIBS) -lfftw3 -lfftw3f

