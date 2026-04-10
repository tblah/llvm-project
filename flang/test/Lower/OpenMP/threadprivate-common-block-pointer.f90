! Simple test for lowering of OpenMP Threadprivate Directive with a pointer var
! from a common block.

!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s
!RUN: bbc -hlfir -emit-hlfir -fopenmp %s -o - | FileCheck %s

! Regression test for a compiler crash

module mmm
integer,pointer::nam1
common /com1/nam1,nam2
!$omp threadprivate(/com1/)
end
use mmm
!$omp parallel copyin(nam1)
!$omp end parallel
end


! CHECK-LABEL:   fir.global common @com1_(dense<0> : vector<28xi8>) {alignment = 8 : i64} : !fir.array<28xi8>

! CHECK-LABEL:   func.func @_QQmain() {
! CHECK:           %[[VAL_0:.*]] = fir.address_of(@com1_) : !fir.ref<!fir.array<28xi8>>
! CHECK:           %[[VAL_5:.*]] = omp.threadprivate %[[VAL_0]] : !fir.ref<!fir.array<28xi8>> -> !fir.ref<!fir.array<28xi8>>
! CHECK:           %[[VAL_8:.*]]:2 = hlfir.declare %{{.*}} storage(%[[VAL_5]][0]) {fortran_attrs = #{{.*}}<pointer>, uniq_name = "_QMmmmEnam1"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.array<28xi8>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
! CHECK:           omp.parallel shared(%[[VAL_8]]#0 -> %[[PAR_NAM1:.*]] : !fir.ref<!fir.box<!fir.ptr<i32>>>) {
! CHECK:             %[[PAR_CMN:.*]] = fir.address_of(@com1_) : !fir.ref<!fir.array<28xi8>>
! CHECK:             %[[VAL_17:.*]] = omp.threadprivate %[[PAR_CMN]] : !fir.ref<!fir.array<28xi8>> -> !fir.ref<!fir.array<28xi8>>
! CHECK:             %[[VAL_19:.*]] = arith.constant 0 : index
! CHECK:             %[[VAL_20:.*]] = fir.coordinate_of %[[VAL_17]], %[[VAL_19]] : (!fir.ref<!fir.array<28xi8>>, index) -> !fir.ref<i8>
! CHECK:             %[[VAL_21:.*]] = fir.convert %[[VAL_20]] : (!fir.ref<i8>) -> !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:             %[[VAL_22:.*]]:2 = hlfir.declare %[[VAL_21]] storage(%[[VAL_17]][0]) {fortran_attrs = #{{.*}}<pointer>, uniq_name = "_QMmmmEnam1"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.array<28xi8>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
! CHECK:             %[[VAL_23:.*]] = fir.load %[[PAR_NAM1]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:             fir.store %[[VAL_23]] to %[[VAL_22]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
