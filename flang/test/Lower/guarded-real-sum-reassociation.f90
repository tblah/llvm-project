! RUN: %flang_fc1 -emit-hlfir -o - %s | FileCheck %s --check-prefixes=DEFAULT,NO-REWRITE
! RUN: %flang_fc1 -emit-hlfir -mllvm -enable-guarded-real-sum-reassociation -o - %s | FileCheck %s --check-prefixes=GUARDED,NO-REWRITE
! RUN: %not_todo_abort_cmd %flang_fc1 -emit-hlfir -mllvm -enable-split-sum-expression-tree-lowering -mllvm -enable-guarded-real-sum-reassociation -o - %s 2>&1 | FileCheck %s --check-prefix=CONFLICT

! CONFLICT: enable-split-sum-expression-tree-lowering and enable-guarded-real-sum-reassociation cannot both be set

subroutine eligible_self_update3(x,a,b,c,d,e,f)
  real(8) :: x,a,b,c,d,e,f
  x = x + a*b + c*d + e*f
end

! GUARDED-LABEL: func.func @_QPeligible_self_update3
! GUARDED-DAG: %[[A:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update3Ea"}
! GUARDED-DAG: %[[B:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update3Eb"}
! GUARDED-DAG: %[[C:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update3Ec"}
! GUARDED-DAG: %[[D:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update3Ed"}
! GUARDED-DAG: %[[E:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update3Ee"}
! GUARDED-DAG: %[[F:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update3Ef"}
! GUARDED-DAG: %[[X:.*]]:2 = hlfir.declare {{.*}} {uniq_name = "_QFeligible_self_update3Ex"}
! GUARDED: %[[XV:.*]] = fir.load %[[X]]#0
! GUARDED: %[[AV:.*]] = fir.load %[[A]]#0
! GUARDED: %[[BV:.*]] = fir.load %[[B]]#0
! GUARDED: %[[AB:.*]] = arith.mulf %[[AV]], %[[BV]] fastmath<reassoc
! GUARDED: %[[XAB:.*]] = arith.addf %[[XV]], %[[AB]] fastmath<reassoc
! GUARDED: %[[CV:.*]] = fir.load %[[C]]#0
! GUARDED: %[[DV:.*]] = fir.load %[[D]]#0
! GUARDED: %[[CD:.*]] = arith.mulf %[[CV]], %[[DV]] fastmath<reassoc
! GUARDED: %[[XABCD:.*]] = arith.addf %[[XAB]], %[[CD]] fastmath<reassoc
! GUARDED: %[[EV:.*]] = fir.load %[[E]]#0
! GUARDED: %[[FV:.*]] = fir.load %[[F]]#0
! GUARDED: %[[EF:.*]] = arith.mulf %[[EV]], %[[FV]] fastmath<reassoc
! GUARDED: %[[RAW:.*]] = arith.addf %[[XABCD]], %[[EF]] fastmath<reassoc
! GUARDED: %[[RES:.*]] = hlfir.no_reassoc %[[RAW]] : f64
! GUARDED: hlfir.assign %[[RES]] to %[[X]]#0

! DEFAULT-LABEL: func.func @_QPeligible_self_update3
! DEFAULT-NOT: fastmath<reassoc
! DEFAULT-NOT: hlfir.no_reassoc
! DEFAULT: hlfir.assign

subroutine guard_parentheses(x,a,b,c,d,e,f)
  real(8) :: x,a,b,c,d,e,f
  x = (x + a*b) + c*d + e*f
end

! NO-REWRITE-LABEL: func.func @_QPguard_parentheses
! NO-REWRITE-NOT: fastmath<reassoc
! NO-REWRITE: hlfir.assign

subroutine guard_subtract(x,a,b,c,d,e,f)
  real(8) :: x,a,b,c,d,e,f
  x = x - a*b + c*d + e*f
end

! NO-REWRITE-LABEL: func.func @_QPguard_subtract
! NO-REWRITE-NOT: fastmath<reassoc
! NO-REWRITE: hlfir.assign

real(8) function foo(a)
  real(8) :: a
  foo = a
end

subroutine guard_call(x,a,b,c,d,e)
  real(8) :: x,a,b,c,d,e
  real(8) :: foo
  x = x + foo(a) + b*c + d*e
end

! NO-REWRITE-LABEL: func.func @_QPguard_call
! NO-REWRITE-NOT: fastmath<reassoc
! NO-REWRITE: hlfir.assign

subroutine guard_short_sum(x,a,b)
  real(8) :: x,a,b
  x = x + a*b
end

! GUARDED-LABEL: func.func @_QPguard_short_sum
! GUARDED: arith.mulf {{.*}} fastmath<reassoc
! GUARDED: arith.addf {{.*}} fastmath<reassoc
! GUARDED: hlfir.no_reassoc
! GUARDED: hlfir.assign

! DEFAULT-LABEL: func.func @_QPguard_short_sum
! DEFAULT-NOT: fastmath<reassoc
! DEFAULT-NOT: hlfir.no_reassoc
! DEFAULT: hlfir.assign

subroutine guard_array(x,a,b,c,d,e,f)
  real(8) :: x(:),a(:),b(:),c(:),d(:),e(:),f(:)
  x = x + a*b + c*d + e*f
end

! NO-REWRITE-LABEL: func.func @_QPguard_array
! NO-REWRITE-NOT: fastmath<reassoc
! NO-REWRITE: hlfir.assign

subroutine guard_volatile(x,a,b,c,d,e,f)
  real(8), volatile :: x
  real(8) :: a,b,c,d,e,f
  x = x + a*b + c*d + e*f
end

! NO-REWRITE-LABEL: func.func @_QPguard_volatile
! NO-REWRITE-NOT: fastmath<reassoc
! NO-REWRITE: hlfir.assign

subroutine guard_asynchronous(x,a,b,c,d,e,f)
  real(8), asynchronous :: x
  real(8) :: a,b,c,d,e,f
  x = x + a*b + c*d + e*f
end

! NO-REWRITE-LABEL: func.func @_QPguard_asynchronous
! NO-REWRITE-NOT: fastmath<reassoc
! NO-REWRITE: hlfir.assign
