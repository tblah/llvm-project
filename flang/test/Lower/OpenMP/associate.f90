! Check that constructs with associate and variables that have implicitly
! determined DSAs are lowered properly.
! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

!CHECK-LABEL: func @_QPtest_parallel_assoc
!CHECK:         omp.parallel shared(%[[A_DECL:.*]]#0 -> %[[A_SHARED:.*]], %[[I_DECL:.*]]#0 -> %[[I_SHARED:.*]] : !fir.ref<!fir.array<3xi32>>, !fir.ref<i32>) {
!CHECK-NOT:       hlfir.declare {{.*}} {uniq_name = "_QFtest_parallel_assocEa"}
!CHECK-NOT:       hlfir.declare {{.*}} {uniq_name = "_QFtest_parallel_assocEb"}
!CHECK:           omp.wsloop private(@_QFtest_parallel_assocEi_private_i32 %[[I_SHARED]] -> %[[I_PRIV:.*]] : !fir.ref<i32>) {
!CHECK:           }
!CHECK:         }
!CHECK:         omp.parallel {{.*}} {
!CHECK-NOT:       hlfir.declare {{.*}} {uniq_name = "_QFtest_parallel_assocEb"}
!CHECK:           omp.wsloop private({{.*}}) {
!CHECK:           }
!CHECK:         }
subroutine test_parallel_assoc()
  integer, parameter :: l = 3
  integer :: a(l)
  integer :: i
  a = 1

  !$omp parallel do
  do i = 1,l
    associate (b=>a)
      b(i) = b(i) * 2
    end associate
  enddo
  !$omp end parallel do

  !$omp parallel do default(private)
  do i = 1,l
    associate (b=>a)
      b(i) = b(i) * 2
    end associate
  enddo
  !$omp end parallel do
end subroutine
