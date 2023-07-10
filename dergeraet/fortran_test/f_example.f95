! f_example.F95
!
! Author: Walter Dal'Maz Silva
! Date  : Jan 6 2019

program f_example
    use module_f_example

    implicit none

    double precision :: val
    type(example_type), pointer :: ptr

    allocate(ptr)

    val = 0.0
    ptr%x = 3.0
    ptr%y = 4.0

    call example(ptr,val)

end program f_example
