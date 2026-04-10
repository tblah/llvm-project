! This test checks lowering of OpenMP Threadprivate Directive.
! Test for character, array, and character array.

!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module test
  character :: x
  integer :: y(5)
  character(5) :: z(5)

  !$omp threadprivate(x, y, z)

contains
  subroutine sub()
!CHECK-LABEL: func.func @_QMtestPsub() {
!CHECK-DAG:  %[[X:.*]] = fir.address_of(@_QMtestEx) : !fir.ref<!fir.char<1>>
!CHECK-DAG:  %[[X_DECL:.*]]:2 = hlfir.declare %[[X]] typeparams %c1 {uniq_name = "_QMtestEx"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
!CHECK-DAG:  %[[OMP_X:.*]] = omp.threadprivate %[[X_DECL]]#0 : !fir.ref<!fir.char<1>> -> !fir.ref<!fir.char<1>>
!CHECK-DAG:  %[[OMP_X_DECL:.*]]:2 = hlfir.declare %[[OMP_X]] typeparams %c1 {uniq_name = "_QMtestEx"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
!CHECK-DAG:  %[[Y:.*]] = fir.address_of(@_QMtestEy) : !fir.ref<!fir.array<5xi32>>
!CHECK-DAG:  %[[Y_DECL:.*]]:2 = hlfir.declare %[[Y]](%{{.*}}) {uniq_name = "_QMtestEy"} : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<5xi32>>, !fir.ref<!fir.array<5xi32>>)
!CHECK-DAG:  %[[OMP_Y:.*]] = omp.threadprivate %[[Y_DECL]]#0 : !fir.ref<!fir.array<5xi32>> -> !fir.ref<!fir.array<5xi32>>
!CHECK-DAG:  %[[OMP_Y_DECL:.*]]:2 = hlfir.declare %[[OMP_Y]](%{{.*}}) {uniq_name = "_QMtestEy"} : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<5xi32>>, !fir.ref<!fir.array<5xi32>>)
!CHECK-DAG:  %[[Z:.*]] = fir.address_of(@_QMtestEz) : !fir.ref<!fir.array<5x!fir.char<1,5>>>
!CHECK-DAG:  %[[Z_DECL:.*]]:2 = hlfir.declare %[[Z]](%{{.*}}) typeparams %c5_0 {uniq_name = "_QMtestEz"} : (!fir.ref<!fir.array<5x!fir.char<1,5>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<5x!fir.char<1,5>>>, !fir.ref<!fir.array<5x!fir.char<1,5>>>)
!CHECK-DAG:  %[[OMP_Z:.*]] = omp.threadprivate %[[Z_DECL]]#0 : !fir.ref<!fir.array<5x!fir.char<1,5>>> -> !fir.ref<!fir.array<5x!fir.char<1,5>>>
!CHECK-DAG:  %[[OMP_Z_DECL:.*]]:2 = hlfir.declare %[[OMP_Z]](%{{.*}}) typeparams %c5_0 {uniq_name = "_QMtestEz"} : (!fir.ref<!fir.array<5x!fir.char<1,5>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<5x!fir.char<1,5>>>, !fir.ref<!fir.array<5x!fir.char<1,5>>>)
!CHECK-DAG:  %{{.*}} = fir.convert %[[OMP_X_DECL]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.ref<i8>
!CHECK-DAG:  %{{.*}} = fir.embox %[[OMP_Y_DECL]]#0(%{{.*}}) : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<5xi32>>
!CHECK-DAG:  %{{.*}} = fir.embox %[[OMP_Z_DECL]]#0(%{{.*}}) : (!fir.ref<!fir.array<5x!fir.char<1,5>>>, !fir.shape<1>) -> !fir.box<!fir.array<5x!fir.char<1,5>>>
    print *, x, y, z

    !$omp parallel
!CHECK:      omp.parallel shared(%[[X_DECL]]#0 -> %[[X_SHARED:.*]], %[[Y_DECL]]#0 -> %[[Y_SHARED:.*]], %[[Z_DECL]]#0 -> %[[Z_SHARED:.*]] : !fir.ref<!fir.char<1>>, !fir.ref<!fir.array<5xi32>>, !fir.ref<!fir.array<5x!fir.char<1,5>>>) {
!CHECK:      %[[C1_PAR:.*]] = arith.constant 1 : index
!CHECK:      %[[X_PVT:.*]] = omp.threadprivate %[[X_SHARED]] : !fir.ref<!fir.char<1>> -> !fir.ref<!fir.char<1>>
!CHECK:      %[[X_PVT_DECL:.*]]:2 = hlfir.declare %[[X_PVT]] typeparams %[[C1_PAR]] {uniq_name = "_QMtestEx"} : (!fir.ref<!fir.char<1>>, index) -> (!fir.ref<!fir.char<1>>, !fir.ref<!fir.char<1>>)
!CHECK:      %[[Y_PVT:.*]] = omp.threadprivate %[[Y_SHARED]] : !fir.ref<!fir.array<5xi32>> -> !fir.ref<!fir.array<5xi32>>
!CHECK:      %[[Y_PVT_DECL:.*]]:2 = hlfir.declare %[[Y_PVT]](%{{.*}}) {uniq_name = "_QMtestEy"} : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<5xi32>>, !fir.ref<!fir.array<5xi32>>)
!CHECK:      %[[Z_PVT:.*]] = omp.threadprivate %[[Z_SHARED]] : !fir.ref<!fir.array<5x!fir.char<1,5>>> -> !fir.ref<!fir.array<5x!fir.char<1,5>>>
!CHECK:      %[[Z_PVT_DECL:.*]]:2 = hlfir.declare %[[Z_PVT]](%{{.*}}) typeparams %[[C5_Z_TP:c5(_[0-9]+)?]] {uniq_name = "_QMtestEz"} : (!fir.ref<!fir.array<5x!fir.char<1,5>>>, !fir.shape<1>, index) -> (!fir.ref<!fir.array<5x!fir.char<1,5>>>, !fir.ref<!fir.array<5x!fir.char<1,5>>>)
!CHECK:      %{{.*}} = fir.convert %[[X_PVT_DECL]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.ref<i8>
!CHECK:      %{{.*}} = fir.embox %[[Y_PVT_DECL]]#0(%{{.*}}) : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<5xi32>>
!CHECK:      %{{.*}} = fir.embox %[[Z_PVT_DECL]]#0(%{{.*}}) : (!fir.ref<!fir.array<5x!fir.char<1,5>>>, !fir.shape<1>) -> !fir.box<!fir.array<5x!fir.char<1,5>>>
    print *, x, y, z
    !$omp end parallel
!CHECK-DAG:  %{{.*}} = fir.convert %[[OMP_X_DECL]]#0 : (!fir.ref<!fir.char<1>>) -> !fir.ref<i8>
!CHECK-DAG:  %{{.*}} = fir.embox %[[OMP_Y_DECL]]#0(%{{.*}}) : (!fir.ref<!fir.array<5xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<5xi32>>
!CHECK-DAG:  %{{.*}} = fir.embox %[[OMP_Z_DECL]]#0(%{{.*}}) : (!fir.ref<!fir.array<5x!fir.char<1,5>>>, !fir.shape<1>) -> !fir.box<!fir.array<5x!fir.char<1,5>>>
    print *, x, y, z

!CHECK-DAG: fir.global @_QMtestEx : !fir.char<1> {
!CHECK-DAG: fir.global @_QMtestEy : !fir.array<5xi32> {
!CHECK-DAG: fir.global @_QMtestEz : !fir.array<5x!fir.char<1,5>> {

  end
end
