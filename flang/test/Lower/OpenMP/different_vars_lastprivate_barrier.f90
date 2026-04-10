! RUN: %flang_fc1 -fopenmp -mmlir --enable-delayed-privatization-staging=true -emit-hlfir %s -o - | FileCheck %s

subroutine first_and_lastprivate(var)
  integer i
  integer, dimension(:) :: var

  !$omp parallel do lastprivate(i) private(var)
  do i=1,1
  end do
  !$omp end parallel do
end subroutine

! CHECK:      omp.private {type = private} @[[VAR_PRIVATIZER:.*Evar_private_box_Uxi32]] : [[BOX_TYPE:!fir\.box<!fir\.array<\?xi32>>]] init {
! CHECK-NEXT: ^bb0(%[[ORIG_REF:.*]]: {{.*}}, %[[PRIV_REF:.*]]: {{.*}}):
! CHECK:        %[[ORIG_VAL:.*]] = fir.load %[[ORIG_REF]]
! CHECK:        %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[ORIG_VAL]], %{{.*}} : ([[BOX_TYPE]], index) -> (index, index, index)
! CHECK:        %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[ORIG_VAL]], %{{.*}} : ([[BOX_TYPE]], index) -> (index, index, index)
! CHECK:        %[[SHAPE_SHIFT:.*]] = fir.shape_shift %[[BOX_DIMS]]#0, %[[BOX_DIMS]]#1
! CHECK:        %[[EMBOX:.*]] = fir.rebox %{{.*}}(%[[SHAPE_SHIFT]]) : {{.*}} -> [[BOX_TYPE]]
! CHECK:        fir.store %[[EMBOX]] to %[[PRIV_REF]]
! CHECK:        omp.yield(%[[PRIV_REF]] : !fir.ref<[[BOX_TYPE]]>)
! CHECK:      }

! CHECK: omp.private {type = private} @[[I_PRIVATIZER:.*Ei_private_i32]] : i32

! CHECK:     func.func @{{.*}}first_and_lastprivate({{.*}})
! CHECK:       %[[ORIG_I_DECL:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "{{.*}}Ei"}
! CHECK:       %[[ORIG_VAR_DECL:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "{{.*}}Evar"}
! CHECK:       omp.parallel shared(%[[ORIG_VAR_DECL]]#0 -> %[[SHARED_VAR:.*]], %[[ORIG_I_DECL]]#0 -> %[[SHARED_I:.*]] : [[BOX_TYPE]], !fir.ref<i32>) {
! CHECK-NOT:      omp.barrier
! CHECK:          %[[PRIV_VAR_BOX:.*]] = fir.alloca !fir.box<!fir.array<?xi32>>
! CHECK:          fir.store %[[SHARED_VAR]] to %[[PRIV_VAR_BOX]] : !fir.ref<!fir.box<!fir.array<?xi32>>>
! CHECK:          omp.wsloop private(@[[I_PRIVATIZER]] %[[SHARED_I]] -> %[[I_ARG:.*]], @[[VAR_PRIVATIZER]] %[[PRIV_VAR_BOX]] -> %[[VAR_ARG:.*]] : !fir.ref<i32>, !fir.ref<!fir.box<!fir.array<?xi32>>>) {
! CHECK:            omp.loop_nest {{.*}} {
! CHECK:              %[[PRIV_I_DECL:.*]]:2 = hlfir.declare %[[I_ARG]] {uniq_name = "{{.*}}Ei"}
! CHECK:              fir.if %{{.*}} {
! CHECK:                %[[PRIV_I_VAL:.*]] = fir.load %[[PRIV_I_DECL]]#0 : !fir.ref<i32>
! CHECK:                hlfir.assign %[[PRIV_I_VAL]] to %[[SHARED_I]]
! CHECK:              }
! CHECK:              omp.yield
! CHECK:            }
! CHECK:          }
! CHECK:          omp.terminator
! CHECK:        }
