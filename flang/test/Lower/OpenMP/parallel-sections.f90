! REQUIRES: openmp_runtime

!RUN: %flang_fc1 -emit-hlfir %openmp_flags %s -o - | FileCheck %s

!===============================================================================
! Parallel sections construct
!===============================================================================

!CHECK: func @_QPomp_parallel_sections
subroutine omp_parallel_sections(x, y)
  integer, intent(inout) :: x, y
  !CHECK: omp.parallel shared(%[[X_DECL:.*]]#0 -> %[[X_SHARED:.*]], %[[Y_DECL:.*]]#0 -> %[[Y_SHARED:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
  !CHECK: omp.sections {
  !$omp parallel sections
    !CHECK: omp.section {
    !$omp section
      !CHECK: fir.load %[[X_SHARED]] : !fir.ref<i32>
      !CHECK: arith.addi
      !CHECK: hlfir.assign {{.*}} to %[[X_SHARED]] : i32, !fir.ref<i32>
      x = x + 12
      !CHECK: omp.terminator
    !CHECK: omp.section {
    !$omp section
      !CHECK: fir.load %[[Y_SHARED]] : !fir.ref<i32>
      !CHECK: arith.subi
      !CHECK: hlfir.assign {{.*}} to %[[Y_SHARED]] : i32, !fir.ref<i32>
      y = y - 5
      !CHECK: omp.terminator
  !CHECK: omp.terminator
  !CHECK: omp.terminator
  !$omp end parallel sections
end subroutine omp_parallel_sections

!===============================================================================
! Parallel sections construct with allocate clause
!===============================================================================

!CHECK: func @_QPomp_parallel_sections
subroutine omp_parallel_sections_allocate(x, y)
  use omp_lib
  integer, intent(inout) :: x, y
  !CHECK: omp.parallel
  !CHECK: %[[allocator_1:.*]] = arith.constant 4 : i64
  !CHECK: omp.sections allocate(%[[allocator_1]] : i64 -> %{{.*}} : !fir.ref<i32>) {
  !$omp parallel sections allocate(omp_high_bw_mem_alloc: x) private(x, y)
    !CHECK: omp.section {
    !$omp section
      x = x + 12
      !CHECK: omp.terminator
    !CHECK: omp.section {
    !$omp section
      y = y + 5
      !CHECK: omp.terminator
  !CHECK: omp.terminator
  !CHECK: omp.terminator
  !$omp end parallel sections
end subroutine omp_parallel_sections_allocate
