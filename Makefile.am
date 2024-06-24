ACLOCAL_AMFLAGS = -I m4

lib_LTLIBRARIES = nufi/libnufi.la
nufi_libnufi_la_SOURCES = nufi/poisson.cpp nufi/lapack.cpp \
                                    nufi/blas.cpp nufi/device_kernel.cpp
nufi_libnufi_la_LDFLAGS = -version-info 0:0:0
                                    

noinst_PROGRAMS = bin/test_nufi_1d bin/test_nufi_2d bin/test_nufi_3d \
                  bin/test_gmres bin/test_poisson bin/test_fields
 

bin_test_nufi_1d_SOURCES = bin/test_nufi_1d.cpp
bin_test_nufi_1d_LDADD   = $(top_builddir)/nufi/libnufi.la $(BLAS_LIBS) -lfftw3 -lfftw3f

bin_test_nufi_2d_SOURCES = bin/test_nufi_2d.cpp
bin_test_nufi_2d_LDADD   = $(top_builddir)/nufi/libnufi.la $(BLAS_LIBS) -lfftw3 -lfftw3f

bin_test_nufi_3d_SOURCES = bin/test_nufi_3d.cpp
bin_test_nufi_3d_LDADD   = $(top_builddir)/nufi/libnufi.la $(BLAS_LIBS) -lfftw3 -lfftw3f

bin_test_gmres_SOURCES = bin/test_gmres.cpp
bin_test_gmres_LDADD   = $(top_builddir)/nufi/libnufi.la $(BLAS_LIBS) -lfftw3 -lfftw3f

bin_test_poisson_SOURCES = bin/test_poisson.cpp
bin_test_poisson_LDADD   = $(top_builddir)/nufi/libnufi.la $(BLAS_LIBS) -lfftw3 -lfftw3f

bin_test_fields_SOURCES = bin/test_fields.cpp
bin_test_fields_LDADD   = $(top_builddir)/nufi/libnufi.la $(BLAS_LIBS) -lfftw3 -lfftw3f

