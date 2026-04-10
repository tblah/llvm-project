! RUN: %flang_fc1 -emit-hlfir -fopenmp -fdo-concurrent-to-openmp=host %s -o - \
! RUN:   | FileCheck %s

program main
    implicit none

    call foo(10)

    contains
        subroutine foo(n)
            implicit none
            integer :: n
            integer :: i
            integer, dimension(n) :: a

            do concurrent(i=1:n)
                a(i) = i
            end do
        end subroutine

end program main

! CHECK: %[[N_DECL:.*]]:2 = hlfir.declare %{{.*}} dummy_scope %{{.*}} {uniq_name = "_QFFfooEn"}

! CHECK: fir.load

! CHECK: %[[N_VAL:.*]] = fir.load %[[N_DECL]]#0 : !fir.ref<i32>

! CHECK: omp.parallel shared(%{{.*}} -> %{{.*}}, %[[N_VAL]] -> %[[PAR_N:.*]] : !fir.box<!fir.array<?xi32>>, i32) {

! Verify that the non-const upper bound (N) is passed as a shared var and
! converted to index inside the parallel region.

! CHECK:   %[[UB:.*]] = fir.convert %[[PAR_N]] : (i32) -> index
! CHECK:   omp.wsloop {
! CHECK:     omp.loop_nest (%{{.*}}) : index = (%{{.*}}) to (%[[UB]]) inclusive step (%{{.*}}) {
! CHECK:       omp.yield
! CHECK:     }
! CHECK:   }
! CHECK:   omp.terminator
! CHECK: }

