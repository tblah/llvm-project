! Check that for parallel do, reduction is only processed for the loop

! RUN: bbc -fopenmp --force-byref-reduction -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -mmlir --force-byref-reduction -emit-hlfir %s -o - | FileCheck %s

! CHECK: omp.parallel shared(%[[IN_I:.*]] -> %[[SHARED_I:.*]], %[[IN_X:.*]] -> %[[SHARED_X:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
! CHECK: omp.wsloop private(@{{.*}} %[[SHARED_I]] -> %[[I_PRIV:.*]] : !fir.ref<i32>) reduction(byref @add_reduction_byref_i32 %[[SHARED_X]] -> %[[X_PRIV:.*]] : !fir.ref<i32>)
subroutine sb
  integer :: x
  x = 0
  !$omp parallel do reduction(+:x)
  do i=1,100
    x = x + 1
  end do
  !$omp end parallel do
end subroutine
