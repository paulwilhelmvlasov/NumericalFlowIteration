! module_c_example.F95
! ISO-C Binding to module_f_example
!
! Author: Walter Dal'Maz Silva
! Date  : Jan 6 2019

module module_c_example
    ! Use ISO-C Bindings default
    use, intrinsic :: iso_c_binding
    !, only : c_f_pointer, c_loc, c_double, c_ptr

    ! The module we are interfacing.
    use module_f_example

    implicit none

    ! XXX this seems actually not to be necessary once example_type is
    ! defined as a struct in c_example.hpp. The only important factor is
    ! that the order of parameters and their names must respect the original
    ! interface from module_f_example. This is weird because the code works
    ! fine, against what has been stated in the following link:
    ! https://stackoverflow.com/tags/fortran-iso-c-binding/info
    ! Maybe it is linked to the fact we use 2008ts, not 2003 here!
    ! type, bind(c) :: c_example_type
    !     real(c_double) :: x
    !     real(c_double) :: y
    ! end type c_example_type

contains

    ! Declaration of C-callable interface for `example` function. The main
    ! feature in this interface is the `bind` method providing the C-name.
    subroutine c_example(mstruc,val) bind(c,name='c_example')
        implicit none

        ! Provide argument types. Notice here that declarations are not as
        ! usual in Fortran, but must respect the style of iso_c_binding.
        ! Another import point is about the typo of `mstruc`. Although in
        ! the headings of this module c_example_type was declared, that
        ! interface was only intended for the C-header file and here one
        ! must use the `c_ptr` type. See C interface with a `void*` in its
        ! place for compatibility. The type of `val` is now given by the
        ! C-binding `real(c_double)`.
        type(c_ptr), intent(inout) :: mstruc
        real(c_double), intent(inout) :: val

        ! Declare internal object of Fortran type for conversion and actual
        ! communication with underlining library.
        type(example_type), pointer :: f_mstruc

        ! Convert C to Fortran pointer.
        ! https://gcc.gnu.org/onlinedocs/gfortran/C_005fF_005fPOINTER.html
        call c_f_pointer(mstruc, f_mstruc)

        ! Call of Fortran function.
        call example(f_mstruc,val)

        ! Gets the address of Fortran pointer.
        ! https://gcc.gnu.org/onlinedocs/gfortran/C_005fLOC.html
        mstruc = c_loc(f_mstruc)
    end subroutine c_example

endmodule module_c_example
