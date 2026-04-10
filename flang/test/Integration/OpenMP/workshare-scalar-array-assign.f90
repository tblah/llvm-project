!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -emit-hlfir -fopenmp -O3 %s -o - | FileCheck %s --check-prefix HLFIR
!RUN: %flang_fc1 -emit-fir -fopenmp -O3 %s -o - | FileCheck %s --check-prefix FIR

subroutine sb1(a, x)
  integer :: a
  integer :: x(:)
  !$omp parallel workshare
  x = a
  !$omp end parallel workshare
end subroutine

! HLFIR:     omp.parallel shared(%1#0 -> %[[A_SHARED:.*]], %2#0 -> %[[X_SHARED:.*]] : !fir.ref<i32>, !fir.box<!fir.array<?xi32>>) {
! HLFIR:       omp.workshare {
! HLFIR:         %[[SCALAR:.*]] = fir.load %[[A_SHARED]] : !fir.ref<i32>
! HLFIR:         hlfir.assign %[[SCALAR]] to %[[X_SHARED]] : i32, !fir.box<!fir.array<?xi32>>
! HLFIR:         omp.terminator
! HLFIR:       }
! HLFIR:       omp.terminator
! HLFIR:     }

! FIR:     omp.parallel shared(%{{.*}} -> %[[A_SHARED:.*]], %{{.*}} -> %[[X_SHARED:.*]] : !fir.ref<i32>, !fir.box<!fir.array<?xi32>>) {
! FIR:       %[[SCALAR_ALLOCA:.*]] = fir.alloca i32
! FIR:       omp.single copyprivate(%[[SCALAR_ALLOCA]] -> @_workshare_copy_i32 : !fir.ref<i32>) {
! FIR:         %[[SCALAR_LOAD:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! FIR:         fir.store %[[SCALAR_LOAD]] to %[[SCALAR_ALLOCA]] : !fir.ref<i32>
! FIR:         omp.terminator
! FIR:       }
! FIR:       %[[SCALAR_RELOAD:.*]] = fir.load %[[SCALAR_ALLOCA]] : !fir.ref<i32>
! FIR:       %6:3 = fir.box_dims %[[X_SHARED]], %c0 : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! FIR:       omp.wsloop nowait {
! FIR:         omp.loop_nest (%[[IV:.*]]) : index = (%c1) to (%6#1) inclusive step (%c1) {
! FIR:           fir.store %[[SCALAR_RELOAD]]
! FIR:           omp.yield
! FIR:         }
! FIR:       }
! FIR:       omp.barrier
! FIR:       omp.terminator
