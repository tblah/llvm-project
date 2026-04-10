! RUN: %flang_fc1 -fopenmp -mmlir --enable-delayed-privatization-staging=true -emit-hlfir %s -o - | FileCheck %s

subroutine first_and_lastprivate
  integer i
  integer :: var = 1

  !$omp parallel do firstprivate(var) lastprivate(var)
  do i=1,1
  end do
  !$omp end parallel do
end subroutine

! CHECK:  omp.private {type = firstprivate} @{{.*}}Evar_firstprivate_i32 : {{.*}} copy {
! CHECK: ^{{.*}}(%[[ORIG_REF:.*]]: {{.*}}, %[[PRIV_REF:.*]]: {{.*}}):
! CHECK:    %[[ORIG_VAL:.*]] = fir.load %[[ORIG_REF]]
! CHECK:    hlfir.assign %[[ORIG_VAL]] to %[[PRIV_REF]]
! CHECK:    omp.yield(%[[PRIV_REF]] : !fir.ref<i32>)
! CHECK:  }

! CHECK:  func.func @{{.*}}first_and_lastprivate()
! CHECK:    %[[ORIG_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "{{.*}}Ei"}
! CHECK:    %[[ORIG_I_DECL:.*]]:2 = hlfir.declare %[[ORIG_I]] {uniq_name = "{{.*}}Ei"}
! CHECK:    %[[ORIG_VAR:.*]] = fir.address_of(@{{.*}}Evar) : !fir.ref<i32>
! CHECK:    %[[ORIG_VAR_DECL:.*]]:2 = hlfir.declare %[[ORIG_VAR]] {uniq_name = "{{.*}}Evar"}
! CHECK:    omp.parallel shared(%[[ORIG_VAR_DECL]]#0 -> %[[SHARED_VAR:.*]], %[[ORIG_I_DECL]]#0 -> %[[SHARED_I:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
! CHECK:      omp.wsloop private(@{{.*}}var_firstprivate_i32 %[[SHARED_VAR]] -> %[[PRIV_VAR:.*]], @{{.*}}Ei_private_i32 %[[SHARED_I]] -> %[[PRIV_I:.*]] : !fir.ref<i32>, !fir.ref<i32>) private_barrier {
! CHECK:        omp.loop_nest {{.*}} {
! CHECK:          %[[PRIV_VAR_DECL:.*]]:2 = hlfir.declare %[[PRIV_VAR]] {uniq_name = "{{.*}}Evar"}
! CHECK:          fir.if %{{.*}} {
! CHECK:            %[[PRIV_VAR_VAL:.*]] = fir.load %[[PRIV_VAR_DECL]]#0 : !fir.ref<i32>
! CHECK:            hlfir.assign %[[PRIV_VAR_VAL]] to %[[SHARED_VAR]]
! CHECK:          }
! CHECK:          omp.yield
! CHECK:        }
! CHECK:      }
! CHECK:      omp.terminator
! CHECK:    }
