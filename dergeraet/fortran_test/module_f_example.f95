! module_f_example.F95
! Module intended to be called from C/C++.
!
! Author: Walter Dal'Maz Silva
! Date  : Jan 6 2019

module module_f_example
    implicit none

    ! Declare a type to illustrate how to use from C/C++.
    type example_type
        double precision :: x
        double precision :: y
    end type example_type

contains

    ! Function to be called from C/C++.
    subroutine example(mstruc,val)
        implicit none

        type(example_type), pointer, intent(inout) :: mstruc
        double precision, intent(inout) :: val

        ! Be verbose to check from C/C++.
        write(*,*)'Point 0 at `module_f_example:example`: val =',val
        ! Modify all quantities.
        val = sqrt(mstruc%x * mstruc%x + mstruc%y * mstruc%y)
        mstruc%x = 0
        mstruc%y = 0
        write(*,*)'Point 1 at `module_f_example:example`: val =',val
    end subroutine example

endmodule module_f_example
