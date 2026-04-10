! Testcase adopted from the Fujitsu test suite:
! https://github.com/fujitsu/compiler-test-suite/blob/main/Fortran/0407/0407_0006.f90

!RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=60 %s -o - | FileCheck %s

! Make sure that the variable in the atomic construct is privatized at
! the task level

!CHECK: omp.parallel shared(%{{[0-9]+}}#0 -> %[[SHARED_PRV:[^[:space:]]+]] : !fir.ref<i32>) {
!CHECK: omp.task private(@_QFfredEprv_firstprivate_i32 %[[SHARED_PRV]] -> %[[PRIV_PRV:[^[:space:]]+]] : !fir.ref<i32>) {
!CHECK: %[[DECL:[0-9]+]]:2 = hlfir.declare %[[PRIV_PRV]] {uniq_name = "_QFfredEprv"}
!CHECK: omp.atomic.update memory_order(relaxed) %[[DECL]]#0

integer function fred
  integer :: prv

  prv = 1
  !$omp parallel shared(prv)
  !$omp task default(firstprivate)
  !$omp atomic
  prv = prv + 1
  !$omp end task
  !$omp end parallel
  fred = prv
end
