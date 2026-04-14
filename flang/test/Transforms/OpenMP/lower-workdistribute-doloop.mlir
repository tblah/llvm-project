// RUN: fir-opt --lower-workdistribute %s | FileCheck %s

// CHECK-LABEL:   func.func @x({{.*}})
// CHECK:           omp.teams {
// CHECK:             omp.parallel {
// CHECK:               omp.distribute {
// CHECK:                 omp.wsloop {
// CHECK:                   omp.loop_nest (%[[VAL_1:.*]]) : index = (%[[ARG0:.*]]) to (%[[ARG1:.*]]) inclusive step (%[[ARG2:.*]]) {
// CHECK:                     %[[VAL_0:.*]] = arith.constant 0 : index
// CHECK:                     fir.store %[[VAL_0]] to %[[ARG4:.*]] : !fir.ref<index>
// CHECK:                     omp.yield
// CHECK:                   }
// CHECK:                 } {omp.composite}
// CHECK:               } {omp.composite}
// CHECK:               omp.terminator
// CHECK:             } {omp.composite}
// CHECK:             omp.terminator
// CHECK:           }
// CHECK:           return
// CHECK:         }
func.func @x(%lb : index, %ub : index, %step : index, %b : i1, %addr : !fir.ref<index>) {
  omp.teams shared(%lb -> %lb0, %ub -> %ub0, %step -> %step0, %b -> %b0, %addr -> %addr0 : index, index, index, i1, !fir.ref<index>) {
    omp.workdistribute {
      fir.do_loop %iv = %lb0 to %ub0 step %step0 unordered {
        %zero = arith.constant 0 : index
        fir.store %zero to %addr0 : !fir.ref<index>
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}
