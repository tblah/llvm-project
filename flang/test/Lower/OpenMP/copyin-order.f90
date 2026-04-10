!RUN: bbc -fopenmp -emit-hlfir -o - %s | FileCheck %s

!https://github.com/llvm/llvm-project/issues/91205

!CHECK: omp.parallel if(%{{[0-9]+}}) shared(%{{[0-9]+}}#0 -> %[[SHARED_X1_SRC:arg0]], %{{[0-9]+}}#0 -> %[[SHARED_X1_VAL:arg1]], %{{[0-9]+}}#0 -> %[[SHARED_X2_SRC:arg2]], %{{[0-9]+}}#0 -> %[[SHARED_X2_VAL:arg3]] : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<!fir.array<10xi64>>, !fir.ref<!fir.array<10xi64>>) {
!CHECK:   %[[THP1:[0-9]+]] = omp.threadprivate %[[SHARED_X1_SRC]]
!CHECK:   %[[DCL1:[0-9]+]]:2 = hlfir.declare %[[THP1]] {uniq_name = "_QFcopyin_scalar_arrayEx1"}
!CHECK:   %[[LD1:[0-9]+]] = fir.load %[[SHARED_X1_VAL]]
!CHECK:   hlfir.assign %[[LD1]] to %[[DCL1]]#0
!CHECK:   %[[THP2:[0-9]+]] = omp.threadprivate %[[SHARED_X2_SRC]]
!CHECK:   %[[SHP2:[0-9]+]] = fir.shape %c{{[0-9]+}}
!CHECK:   %[[DCL2:[0-9]+]]:2 = hlfir.declare %[[THP2]](%[[SHP2]]) {uniq_name = "_QFcopyin_scalar_arrayEx2"}
!CHECK:   hlfir.assign %[[SHARED_X2_VAL]] to %[[DCL2]]#0
!CHECK:   omp.barrier
!CHECK:   fir.call @_QPsub1(%[[DCL1]]#0, %[[DCL2]]#0)
!CHECK:   omp.terminator
!CHECK: }

subroutine copyin_scalar_array()
  integer(kind=4), save :: x1
  integer(kind=8), save :: x2(10)
  !$omp threadprivate(x1, x2)

  ! Have x1 appear before x2 in the AST node for the `parallel` construct,
  ! but at the same time have them in a different order in `copyin`.
  !$omp parallel if (x1 .eq. x2(1)) copyin(x2, x1)
    call sub1(x1, x2)
  !$omp end parallel

end
