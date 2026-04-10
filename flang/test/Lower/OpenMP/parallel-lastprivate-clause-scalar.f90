! This test checks lowering of `LASTPRIVATE` clause for scalar types.

! RUN: bbc -fopenmp -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -emit-hlfir %s -o - | FileCheck %s

!CHECK: func @_QPlastprivate_character(%[[ARG1:.*]]: !fir.boxchar<1>{{.*}}) {
!CHECK-DAG: %[[ARG1_UNBOX:.*]]:2 = fir.unboxchar
!CHECK-DAG: %[[FIVE:.*]] = arith.constant 5 : index
!CHECK-DAG: %[[ARG1_REF:.*]] = fir.convert %[[ARG1_UNBOX]]#0 : (!fir.ref<!fir.char<1,?>>) -> !fir.ref<!fir.char<1,5>>
!CHECK-DAG: %[[ARG1_DECL:.*]]:2 = hlfir.declare %[[ARG1_REF]] typeparams %[[FIVE]] dummy_scope %{{[0-9]+}} arg {{[0-9]+}} {uniq_name = "_QFlastprivate_characterEarg1"} : (!fir.ref<!fir.char<1,5>>, index, !fir.dscope) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)

!CHECK: omp.parallel shared(%[[ARG1_DECL]]#0 -> %[[SHARED_ARG1:.*]], %{{.*}}#0 -> %[[SHARED_N:.*]] : !fir.ref<!fir.char<1,5>>, !fir.ref<i32>) {

! Check that we are accessing the clone inside the loop
!CHECK: omp.wsloop private(@_QFlastprivate_characterEarg1_private_c8x5 %[[SHARED_ARG1]] -> %[[ARG1_PVT:.*]], @_QFlastprivate_characterEn_private_i32 %[[SHARED_N]] -> %[[N_PVT:.*]] : !fir.ref<!fir.char<1,5>>, !fir.ref<i32>) {
!CHECK-NEXT: omp.loop_nest (%[[INDX_WS:.*]]) : {{.*}} {
!CHECK: %[[FIVE:.*]] = arith.constant 5 : index
!CHECK: %[[ARG1_PVT_DECL:.*]]:2 = hlfir.declare %[[ARG1_PVT]] typeparams %[[FIVE]] {uniq_name = "_QFlastprivate_characterEarg1"} : (!fir.ref<!fir.char<1,5>>, index) -> (!fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>)
!CHECK: %[[N_PVT_DECL:.*]]:2 = hlfir.declare %[[N_PVT]] {uniq_name = "_QFlastprivate_characterEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[UNIT:.*]] = arith.constant 6 : i32
!CHECK-NEXT: %[[ADDR:.*]] = fir.address_of(@_QQclX
!CHECK-NEXT: %[[CVT0:.*]] = fir.convert %[[ADDR]]
!CHECK-NEXT: %[[CNST:.*]] = arith.constant
!CHECK-NEXT: %[[CALL_BEGIN_IO:.*]] = fir.call @_FortranAioBeginExternalListOutput(%[[UNIT]], %[[CVT0]], %[[CNST]]) {{.*}}: (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
!CHECK-NEXT: %[[CVT_0_1:.*]] = fir.convert %[[ARG1_PVT_DECL]]#0
!CHECK-NEXT: %[[CVT_0_2:.*]] = fir.convert %[[FIVE]]
!CHECK-NEXT: %[[CALL_OP_ASCII:.*]] = fir.call @_FortranAioOutputAscii(%[[CALL_BEGIN_IO]], %[[CVT_0_1]], %[[CVT_0_2]])
!CHECK-NEXT: %[[CALL_END_IO:.*]] = fir.call @_FortranAioEndIoStatement(%[[CALL_BEGIN_IO]])

! Testing last iteration check
!CHECK: %[[V:.*]] = arith.addi %[[INDX_WS]], %{{.*}} : i32
!CHECK: %[[C0:.*]] = arith.constant 0 : i32
!CHECK: %[[T1:.*]] = arith.cmpi slt, %{{.*}}, %[[C0]] : i32
!CHECK: %[[T2:.*]] = arith.cmpi slt, %[[V]], %{{.*}} : i32
!CHECK: %[[T3:.*]] = arith.cmpi sgt, %[[V]], %{{.*}} : i32
!CHECK: %[[IV_CMP:.*]] = arith.select %[[T1]], %[[T2]], %[[T3]] : i1
!CHECK: fir.if %[[IV_CMP]] {
!CHECK: hlfir.assign %[[V]] to %{{.*}} : i32, !fir.ref<i32>

! Testing lastprivate val update
!CHECK: hlfir.assign %[[ARG1_PVT_DECL]]#0 to %[[SHARED_ARG1]] : !fir.ref<!fir.char<1,5>>, !fir.ref<!fir.char<1,5>>
!CHECK: }
!CHECK: omp.yield
!CHECK: }
!CHECK: }

subroutine lastprivate_character(arg1)
        character(5) :: arg1
!$OMP PARALLEL
!$OMP DO LASTPRIVATE(arg1)
do n = 1, 5
        arg1(n:n) = 'c'
        print *, arg1
end do
!$OMP END DO
!$OMP END PARALLEL
end subroutine

!CHECK: func @_QPlastprivate_int(%[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "arg1"}) {
!CHECK: %[[ARG1_DECL:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %{{[0-9]+}} arg {{[0-9]+}} {uniq_name = "_QFlastprivate_intEarg1"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[N_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFlastprivate_intEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.parallel shared(%[[ARG1_DECL]]#0 -> %[[SHARED_ARG1:.*]], %[[N_DECL]]#0 -> %[[SHARED_N:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
!CHECK: omp.wsloop private(@_QFlastprivate_intEarg1_private_i32 %[[SHARED_ARG1]] -> %[[CLONE:.*]], @_QFlastprivate_intEn_private_i32 %[[SHARED_N]] -> %[[IV:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
!CHECK-NEXT: omp.loop_nest (%[[INDX_WS:.*]]) : {{.*}} {
!CHECK:      %[[CLONE_DECL:.*]]:2 = hlfir.declare %[[CLONE]] {uniq_name = "_QFlastprivate_intEarg1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:      %[[IV_DECL:.*]]:2 = hlfir.declare %[[IV]] {uniq_name = "_QFlastprivate_intEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

! Testing last iteration check
!CHECK: %[[V:.*]] = arith.addi %[[INDX_WS]], %{{.*}} : i32
!CHECK: %[[C0:.*]] = arith.constant 0 : i32
!CHECK: %[[T1:.*]] = arith.cmpi slt, %{{.*}}, %[[C0]] : i32
!CHECK: %[[T2:.*]] = arith.cmpi slt, %[[V]], %{{.*}} : i32
!CHECK: %[[T3:.*]] = arith.cmpi sgt, %[[V]], %{{.*}} : i32
!CHECK: %[[IV_CMP:.*]] = arith.select %[[T1]], %[[T2]], %[[T3]] : i1
!CHECK: fir.if %[[IV_CMP]] {
!CHECK: hlfir.assign %[[V]] to %{{.*}} : i32, !fir.ref<i32>

! Testing lastprivate val update
!CHECK-NEXT: %[[CLONE_LD:.*]] = fir.load %[[CLONE_DECL]]#0 : !fir.ref<i32>
!CHECK:      hlfir.assign %[[CLONE_LD]] to %[[SHARED_ARG1]] : i32, !fir.ref<i32>
!CHECK: }
!CHECK: omp.yield
!CHECK: }
!CHECK: }

subroutine lastprivate_int(arg1)
        integer :: arg1
!$OMP PARALLEL
!$OMP DO LASTPRIVATE(arg1)
do n = 1, 5
        arg1 = 2
        print *, arg1
end do
!$OMP END DO
!$OMP END PARALLEL
print *, arg1
end subroutine

!CHECK: func.func @_QPmult_lastprivate_int(%[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "arg1"}, %[[ARG2:.*]]: !fir.ref<i32> {fir.bindc_name = "arg2"}) {
!CHECK: %[[ARG1_DECL:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %{{[0-9]+}} arg {{[0-9]+}} {uniq_name = "_QFmult_lastprivate_intEarg1"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[ARG2_DECL:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %{{[0-9]+}} arg {{[0-9]+}} {uniq_name = "_QFmult_lastprivate_intEarg2"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[N_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFmult_lastprivate_intEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.parallel shared(%[[ARG1_DECL]]#0 -> %[[SHARED_ARG1:.*]], %[[ARG2_DECL]]#0 -> %[[SHARED_ARG2:.*]], %[[N_DECL]]#0 -> %[[SHARED_N:.*]] : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>) {
!CHECK: omp.wsloop private(@_QFmult_lastprivate_intEarg1_private_i32 %[[SHARED_ARG1]] -> %[[CLONE1:.*]], @_QFmult_lastprivate_intEarg2_private_i32 %[[SHARED_ARG2]] -> %[[CLONE2:.*]], @_QFmult_lastprivate_intEn_private_i32 %[[SHARED_N]] -> %[[IV:.*]] : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>) {
!CHECK-NEXT: omp.loop_nest (%[[INDX_WS:.*]]) : {{.*}} {
!CHECK-DAG: %[[CLONE1_DECL:.*]]:2 = hlfir.declare %[[CLONE1]] {uniq_name = "_QFmult_lastprivate_intEarg1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-DAG: %[[CLONE2_DECL:.*]]:2 = hlfir.declare %[[CLONE2]] {uniq_name = "_QFmult_lastprivate_intEarg2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

! Testing last iteration check
!CHECK: %[[V:.*]] = arith.addi %[[INDX_WS]], %{{.*}} : i32
!CHECK: %[[C0:.*]] = arith.constant 0 : i32
!CHECK: %[[T1:.*]] = arith.cmpi slt, %{{.*}}, %[[C0]] : i32
!CHECK: %[[T2:.*]] = arith.cmpi slt, %[[V]], %{{.*}} : i32
!CHECK: %[[T3:.*]] = arith.cmpi sgt, %[[V]], %{{.*}} : i32
!CHECK: %[[IV_CMP:.*]] = arith.select %[[T1]], %[[T2]], %[[T3]] : i1
!CHECK: fir.if %[[IV_CMP]] {
!CHECK: hlfir.assign %[[V]] to %{{.*}} : i32, !fir.ref<i32>
! Testing lastprivate val update
!CHECK-DAG: %[[CLONE_LD1:.*]] = fir.load %[[CLONE1_DECL]]#0 : !fir.ref<i32>
!CHECK-DAG: hlfir.assign %[[CLONE_LD1]] to %[[SHARED_ARG1]] : i32, !fir.ref<i32>
!CHECK-DAG: %[[CLONE_LD2:.*]] = fir.load %[[CLONE2_DECL]]#0 : !fir.ref<i32>
!CHECK-DAG: hlfir.assign %[[CLONE_LD2]] to %[[SHARED_ARG2]] : i32, !fir.ref<i32>
!CHECK: }
!CHECK: omp.yield
!CHECK: }
!CHECK: }

subroutine mult_lastprivate_int(arg1, arg2)
        integer :: arg1, arg2
!$OMP PARALLEL
!$OMP DO LASTPRIVATE(arg1) LASTPRIVATE(arg2)
do n = 1, 5
        arg1 = 2
        arg2 = 3
        print *, arg1, arg2
end do
!$OMP END DO
!$OMP END PARALLEL
print *, arg1, arg2
end subroutine

!CHECK: func.func @_QPmult_lastprivate_int2(%[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "arg1"}, %[[ARG2:.*]]: !fir.ref<i32> {fir.bindc_name = "arg2"}) {
!CHECK: %[[ARG1_DECL:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %{{[0-9]+}} arg {{[0-9]+}} {uniq_name = "_QFmult_lastprivate_int2Earg1"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[ARG2_DECL:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %{{[0-9]+}} arg {{[0-9]+}} {uniq_name = "_QFmult_lastprivate_int2Earg2"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[N_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFmult_lastprivate_int2En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.parallel shared(%[[ARG1_DECL]]#0 -> %[[SHARED_ARG1:.*]], %[[ARG2_DECL]]#0 -> %[[SHARED_ARG2:.*]], %[[N_DECL]]#0 -> %[[SHARED_N:.*]] : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>) {
!CHECK: omp.wsloop private(@_QFmult_lastprivate_int2Earg1_private_i32 %[[SHARED_ARG1]] -> %[[CLONE1:.*]], @_QFmult_lastprivate_int2Earg2_private_i32 %[[SHARED_ARG2]] -> %[[CLONE2:.*]], @_QFmult_lastprivate_int2En_private_i32 %[[SHARED_N]] -> %[[IV:.*]] : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>) {
!CHECK-NEXT: omp.loop_nest (%[[INDX_WS:.*]]) : {{.*}} {
!CHECK-DAG: %[[CLONE1_DECL:.*]]:2 = hlfir.declare %[[CLONE1]] {uniq_name = "_QFmult_lastprivate_int2Earg1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK-DAG: %[[CLONE2_DECL:.*]]:2 = hlfir.declare %[[CLONE2]] {uniq_name = "_QFmult_lastprivate_int2Earg2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

!Testing last iteration check
!CHECK: %[[V:.*]] = arith.addi %[[INDX_WS]], %{{.*}} : i32
!CHECK: %[[C0:.*]] = arith.constant 0 : i32
!CHECK: %[[T1:.*]] = arith.cmpi slt, %{{.*}}, %[[C0]] : i32
!CHECK: %[[T2:.*]] = arith.cmpi slt, %[[V]], %{{.*}} : i32
!CHECK: %[[T3:.*]] = arith.cmpi sgt, %[[V]], %{{.*}} : i32
!CHECK: %[[IV_CMP:.*]] = arith.select %[[T1]], %[[T2]], %[[T3]] : i1
!CHECK: fir.if %[[IV_CMP]] {
!CHECK: hlfir.assign %[[V]] to %{{.*}} : i32, !fir.ref<i32>
!Testing lastprivate val update
!CHECK-DAG: %[[CLONE_LD2:.*]] = fir.load %[[CLONE2_DECL]]#0 : !fir.ref<i32>
!CHECK-DAG: hlfir.assign %[[CLONE_LD2]] to %[[SHARED_ARG2]] : i32, !fir.ref<i32>
!CHECK-DAG: %[[CLONE_LD1:.*]] = fir.load %[[CLONE1_DECL]]#0 : !fir.ref<i32>
!CHECK-DAG: hlfir.assign %[[CLONE_LD1]] to %[[SHARED_ARG1]] : i32, !fir.ref<i32>
!CHECK: }
!CHECK: omp.yield
!CHECK: }
!CHECK: }

subroutine mult_lastprivate_int2(arg1, arg2)
        integer :: arg1, arg2
!$OMP PARALLEL
!$OMP DO LASTPRIVATE(arg1, arg2)
do n = 1, 5
        arg1 = 2
        arg2 = 3
        print *, arg1, arg2
end do
!$OMP END DO
!$OMP END PARALLEL
print *, arg1, arg2
end subroutine

!CHECK: func.func @_QPfirstpriv_lastpriv_int(%[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "arg1"}, %[[ARG2:.*]]: !fir.ref<i32> {fir.bindc_name = "arg2"}) {
!CHECK:    %[[ARG1_DECL:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %{{[0-9]+}} arg {{[0-9]+}} {uniq_name = "_QFfirstpriv_lastpriv_intEarg1"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[ARG2_DECL:.*]]:2 = hlfir.declare %[[ARG2]] dummy_scope %{{[0-9]+}} arg {{[0-9]+}} {uniq_name = "_QFfirstpriv_lastpriv_intEarg2"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:    %[[N_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFfirstpriv_lastpriv_intEn"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.parallel shared(%[[ARG2_DECL]]#0 -> %[[SHARED_ARG2:.*]], %[[ARG1_DECL]]#0 -> %[[SHARED_ARG1:.*]], %[[N_DECL]]#0 -> %[[SHARED_N:.*]] : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>) {
! Firstprivate update
!CHECK-NOT: omp.barrier
!CHECK: omp.wsloop private(@_QFfirstpriv_lastpriv_intEarg1_firstprivate_i32 %[[SHARED_ARG1]] -> %[[CLONE1:.*]], @_QFfirstpriv_lastpriv_intEarg2_private_i32 %[[SHARED_ARG2]] -> %[[CLONE2:.*]], @_QFfirstpriv_lastpriv_intEn_private_i32 %[[SHARED_N]] -> %[[IV:.*]] : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>) {
!CHECK-NEXT: omp.loop_nest (%[[INDX_WS:.*]]) : {{.*}} {
!CHECK: %[[CLONE1_DECL:.*]]:2 = hlfir.declare %[[CLONE1]] {uniq_name = "_QFfirstpriv_lastpriv_intEarg1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[CLONE2_DECL:.*]]:2 = hlfir.declare %[[CLONE2]] {uniq_name = "_QFfirstpriv_lastpriv_intEarg2"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

! Testing last iteration check
!CHECK: %[[V:.*]] = arith.addi %[[INDX_WS]], %{{.*}} : i32
!CHECK: %[[C0:.*]] = arith.constant 0 : i32
!CHECK: %[[T1:.*]] = arith.cmpi slt, %{{.*}}, %[[C0]] : i32
!CHECK: %[[T2:.*]] = arith.cmpi slt, %[[V]], %{{.*}} : i32
!CHECK: %[[T3:.*]] = arith.cmpi sgt, %[[V]], %{{.*}} : i32
!CHECK: %[[IV_CMP:.*]] = arith.select %[[T1]], %[[T2]], %[[T3]] : i1
!CHECK: fir.if %[[IV_CMP]] {
!CHECK: hlfir.assign %[[V]] to %{{.*}} : i32, !fir.ref<i32>
! Testing lastprivate val update
!CHECK-NEXT: %[[CLONE_LD:.*]] = fir.load %[[CLONE2_DECL]]#0 : !fir.ref<i32>
!CHECK-NEXT: hlfir.assign %[[CLONE_LD]] to %[[SHARED_ARG2]] : i32, !fir.ref<i32>
!CHECK-NEXT: }
!CHECK-NEXT: omp.yield
!CHECK-NEXT: }
!CHECK-NEXT: }

subroutine firstpriv_lastpriv_int(arg1, arg2)
        integer :: arg1, arg2
!$OMP PARALLEL
!$OMP DO FIRSTPRIVATE(arg1) LASTPRIVATE(arg2)
do n = 1, 5
        arg1 = 2
        arg2 = 3
        print *, arg1, arg2
end do
!$OMP END DO
!$OMP END PARALLEL
print *, arg1, arg2
end subroutine

!CHECK: func.func @_QPfirstpriv_lastpriv_int2(%[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "arg1"}) {
!CHECK: %[[ARG1_DECL:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %{{[0-9]+}} arg {{[0-9]+}} {uniq_name = "_QFfirstpriv_lastpriv_int2Earg1"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: %[[N_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFfirstpriv_lastpriv_int2En"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK: omp.parallel shared(%[[ARG1_DECL]]#0 -> %[[SHARED_ARG1:.*]], %[[N_DECL]]#0 -> %[[SHARED_N:.*]] : !fir.ref<i32>, !fir.ref<i32>) {

! Firstprivate update


!CHECK: omp.wsloop private(@_QFfirstpriv_lastpriv_int2Earg1_firstprivate_i32 %[[SHARED_ARG1]] -> %[[CLONE1:.*]], @_QFfirstpriv_lastpriv_int2En_private_i32 %[[SHARED_N]] -> %[[IV:.*]] : !fir.ref<i32>, !fir.ref<i32>) private_barrier {
!CHECK-NEXT: omp.loop_nest (%[[INDX_WS:.*]]) : {{.*}} {
!CHECK: %[[CLONE1_DECL:.*]]:2 = hlfir.declare %[[CLONE1]] {uniq_name = "_QFfirstpriv_lastpriv_int2Earg1"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)

!CHECK-NEXT: hlfir.declare %[[IV]]
! Testing last iteration check
!CHECK: %[[V:.*]] = arith.addi %[[INDX_WS]], %{{.*}} : i32
!CHECK: %[[C0:.*]] = arith.constant 0 : i32
!CHECK: %[[T1:.*]] = arith.cmpi slt, %{{.*}}, %[[C0]] : i32
!CHECK: %[[T2:.*]] = arith.cmpi slt, %[[V]], %{{.*}} : i32
!CHECK: %[[T3:.*]] = arith.cmpi sgt, %[[V]], %{{.*}} : i32
!CHECK: %[[IV_CMP:.*]] = arith.select %[[T1]], %[[T2]], %[[T3]] : i1
!CHECK: fir.if %[[IV_CMP]] {
!CHECK: hlfir.assign %[[V]] to %{{.*}} : i32, !fir.ref<i32>
! Testing lastprivate val update
!CHECK-NEXT: %[[CLONE_LD:.*]] = fir.load %[[CLONE1_DECL]]#0 : !fir.ref<i32>
!CHECK-NEXT: hlfir.assign %[[CLONE_LD]] to %[[SHARED_ARG1]] : i32, !fir.ref<i32>
!CHECK-NEXT: }
!CHECK-NEXT: omp.yield
!CHECK-NEXT: }
!CHECK-NEXT: }

subroutine firstpriv_lastpriv_int2(arg1)
        integer :: arg1
!$OMP PARALLEL
!$OMP DO FIRSTPRIVATE(arg1) LASTPRIVATE(arg1)
do n = 1, 5
        arg1 = 2
        print *, arg1
end do
!$OMP END DO
!$OMP END PARALLEL
print *, arg1
end subroutine

! Check that LASTPRIVATE updates the private copy of `i` when used inside
! nested PARALLEL constructs in which `i` is private.
!CHECK-LABEL: func @_QPlastprivate_nested_parallel()
!CHECK:         %[[I:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFlastprivate_nested_parallelEi"} :
!CHECK-SAME:                  (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:         omp.parallel private(@_QFlastprivate_nested_parallelEi_private_i32 %[[I]]#0 {{.*}})
!CHECK:           %[[PRIV_I:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFlastprivate_nested_parallelEi"} :
!CHECK-SAME:                         (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:           omp.parallel shared(%[[PRIV_I]]#0 -> %[[SHARED_PRIV_I:.*]] : !fir.ref<i32>) {
!CHECK:             omp.wsloop private(@_QFlastprivate_nested_parallelEi_private_i32 %[[SHARED_PRIV_I]] -> %[[INNER_PRIV_I:.*]] : !fir.ref<i32>) {
!CHECK:               %[[INNER_PRIV_I_DECL:.*]]:2 = hlfir.declare %[[INNER_PRIV_I]] {uniq_name = "_QFlastprivate_nested_parallelEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:               hlfir.assign %{{.*}} to %[[INNER_PRIV_I_DECL]]#0
!CHECK:               %[[INNER_PRIV_I_LOAD:.*]] = fir.load %[[INNER_PRIV_I_DECL]]#0 : !fir.ref<i32>
!CHECK:               hlfir.assign %[[INNER_PRIV_I_LOAD]] to %[[SHARED_PRIV_I]] : i32, !fir.ref<i32>

subroutine lastprivate_nested_parallel()
  integer :: i

  !$OMP PARALLEL DEFAULT(PRIVATE)
    !$OMP PARALLEL
      !$OMP DO LASTPRIVATE(i)
      do i = 1, 5
      end do
    !$OMP END PARALLEL
  !$OMP END PARALLEL
end subroutine

!CHECK-LABEL: func @_QPlastprivate_nested_parallel2()
!CHECK:         %[[I:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFlastprivate_nested_parallel2Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:         %[[J:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFlastprivate_nested_parallel2Ej"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:         %[[K:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QFlastprivate_nested_parallel2Ek"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:         omp.parallel shared(%[[K]]#0 -> %[[SHARED_K:.*]], %[[I]]#0 -> %[[SHARED_I:.*]], %[[J]]#0 -> %[[SHARED_J:.*]] : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>) {
!CHECK:           omp.wsloop private(@_QFlastprivate_nested_parallel2Ei_private_i32 %[[SHARED_I]] -> %[[OUTER_PRIV_I:.*]], @_QFlastprivate_nested_parallel2Ej_private_i32 %[[SHARED_J]] -> %[[OUTER_PRIV_J:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
!CHECK:             %[[OUTER_PRIV_I_DECL:.*]]:2 = hlfir.declare %[[OUTER_PRIV_I]] {uniq_name = "_QFlastprivate_nested_parallel2Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:             %[[OUTER_PRIV_J_DECL:.*]]:2 = hlfir.declare %[[OUTER_PRIV_J]] {uniq_name = "_QFlastprivate_nested_parallel2Ej"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:             omp.parallel shared(%[[OUTER_PRIV_I_DECL]]#0 -> %[[INNER_SHARED_I:.*]], %[[SHARED_K]] -> %[[INNER_SHARED_K:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
!CHECK:               omp.wsloop private(@_QFlastprivate_nested_parallel2Ei_private_i32 %[[INNER_SHARED_I]] -> %[[INNER_PRIV_I:.*]], @_QFlastprivate_nested_parallel2Ek_private_i32 %[[INNER_SHARED_K]] -> %[[INNER_PRIV_K:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
!CHECK:                 %[[INNER_PRIV_I_DECL:.*]]:2 = hlfir.declare %[[INNER_PRIV_I]] {uniq_name = "_QFlastprivate_nested_parallel2Ei"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:                 %[[INNER_PRIV_K_DECL:.*]]:2 = hlfir.declare %[[INNER_PRIV_K]] {uniq_name = "_QFlastprivate_nested_parallel2Ek"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
!CHECK:                 hlfir.assign %{{.*}} to %[[INNER_PRIV_K_DECL]]#0
!CHECK:                 %[[INNER_PRIV_I_LOAD:.*]] = fir.load %[[INNER_PRIV_I_DECL]]#0 : !fir.ref<i32>
!CHECK:                 hlfir.assign %[[INNER_PRIV_I_LOAD]] to %[[INNER_SHARED_I]] : i32, !fir.ref<i32>
!CHECK:             hlfir.assign %{{.*}} to %[[OUTER_PRIV_J_DECL]]#0
!CHECK:             %[[OUTER_PRIV_I_LOAD:.*]] = fir.load %[[OUTER_PRIV_I_DECL]]#0 : !fir.ref<i32>
!CHECK:             hlfir.assign %[[OUTER_PRIV_I_LOAD]] to %[[SHARED_I]] : i32, !fir.ref<i32>

subroutine lastprivate_nested_parallel2()
  integer :: i, j, k

  !$omp parallel do lastprivate(i)
  do j = 1, 10
    !$omp parallel do lastprivate(i)
    do k = 2, 20
    end do
  end do
end subroutine
