// RUN: mlir-opt -convert-scf-to-openmp='num-threads=4' %s | FileCheck %s

// CHECK-LABEL: @parallel
func.func @parallel(%arg0: index, %arg1: index, %arg2: index,
                    %arg3: index, %arg4: index, %arg5: index) {
  // CHECK: %[[FOUR:.+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: omp.parallel num_threads(%[[FOUR]] : i32) shared(%arg0 -> %[[P0:[^ ,)]+]], %arg1 -> %[[P1:[^ ,)]+]], %arg2 -> %[[P2:[^ ,)]+]], %arg3 -> %[[P3:[^ ,)]+]], %arg4 -> %[[P4:[^ ,)]+]], %arg5 -> %[[P5:[^ ,)]+]] : index, index, index, index, index, index) {
  // CHECK: omp.wsloop {
  // CHECK: omp.loop_nest (%[[LVAR1:.*]], %[[LVAR2:.*]]) : index = (%[[P0]], %[[P1]]) to (%[[P2]], %[[P3]]) step (%[[P4]], %[[P5]]) collapse(2) {
  // CHECK: memref.alloca_scope
  scf.parallel (%i, %j) = (%arg0, %arg1) to (%arg2, %arg3) step (%arg4, %arg5) {
    // CHECK: "test.payload"(%[[LVAR1]], %[[LVAR2]]) : (index, index) -> ()
    "test.payload"(%i, %j) : (index, index) -> ()
    // CHECK:   omp.yield
    // CHECK: }
  }
  // CHECK:   }
  // CHECK:   omp.terminator
  // CHECK: }
  return
}

// CHECK-LABEL: @nested_loops
func.func @nested_loops(%arg0: index, %arg1: index, %arg2: index,
                   %arg3: index, %arg4: index, %arg5: index) {
  // CHECK: %[[FOUR:.+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: omp.parallel num_threads(%[[FOUR]] : i32) shared(%arg1 -> %[[NIN_LB:[^ ,)]+]], %arg3 -> %[[NIN_UB:[^ ,)]+]], %arg5 -> %[[NIN_STEP:[^ ,)]+]], %arg0 -> %[[NOUT_LB:[^ ,)]+]], %arg2 -> %[[NOUT_UB:[^ ,)]+]], %arg4 -> %[[NOUT_STEP:[^ ,)]+]] : index, index, index, index, index, index) {
  // CHECK: omp.wsloop {
  // CHECK: omp.loop_nest (%[[LVAR_OUT1:.*]]) : index = (%[[NOUT_LB]]) to (%[[NOUT_UB]]) step (%[[NOUT_STEP]]) {
  // CHECK: memref.alloca_scope
  scf.parallel (%i) = (%arg0) to (%arg2) step (%arg4) {
    // CHECK: omp.parallel{{.*}}shared(%[[LVAR_OUT1]] -> %[[OIV_IN:[^ ,)]+]], %[[NIN_LB]] -> %[[NIN_LB2:[^ ,)]+]], %[[NIN_UB]] -> %[[NIN_UB2:[^ ,)]+]], %[[NIN_STEP]] -> %[[NIN_STEP2:[^ ,)]+]] : index, index, index, index) {
    // CHECK: omp.wsloop {
    // CHECK: omp.loop_nest (%[[LVAR_IN1:.*]]) : index = (%[[NIN_LB2]]) to (%[[NIN_UB2]]) step (%[[NIN_STEP2]]) {
    // CHECK: memref.alloca_scope
    scf.parallel (%j) = (%arg1) to (%arg3) step (%arg5) {
      // CHECK: "test.payload"(%[[OIV_IN]], %[[LVAR_IN1]]) : (index, index) -> ()
      "test.payload"(%i, %j) : (index, index) -> ()
      // CHECK: }
    }
    // CHECK:     omp.yield
    // CHECK:   }
    // CHECK: }
  }
  // CHECK:   }
  // CHECK:   omp.terminator
  // CHECK: }
  return
}

// CHECK-LABEL: @adjacent_loops
func.func @adjacent_loops(%arg0: index, %arg1: index, %arg2: index,
                     %arg3: index, %arg4: index, %arg5: index) {
  // CHECK: %[[FOUR:.+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: omp.parallel num_threads(%[[FOUR]] : i32) shared(%arg0 -> %[[AL1_LB:[^ ,)]+]], %arg2 -> %[[AL1_UB:[^ ,)]+]], %arg4 -> %[[AL1_STEP:[^ ,)]+]] : index, index, index) {
  // CHECK: omp.wsloop {
  // CHECK: omp.loop_nest (%[[LVAR_AL1:.*]]) : index = (%[[AL1_LB]]) to (%[[AL1_UB]]) step (%[[AL1_STEP]]) {
  // CHECK: memref.alloca_scope
  scf.parallel (%i) = (%arg0) to (%arg2) step (%arg4) {
    // CHECK: "test.payload1"(%[[LVAR_AL1]]) : (index) -> ()
    "test.payload1"(%i) : (index) -> ()
    // CHECK:   omp.yield
    // CHECK: }
  }
  // CHECK:   }
  // CHECK:   omp.terminator
  // CHECK: }

  // CHECK: %[[FOUR:.+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: omp.parallel num_threads(%[[FOUR]] : i32) shared(%arg1 -> %[[AL2_LB:[^ ,)]+]], %arg3 -> %[[AL2_UB:[^ ,)]+]], %arg5 -> %[[AL2_STEP:[^ ,)]+]] : index, index, index) {
  // CHECK: omp.wsloop {
  // CHECK: omp.loop_nest (%[[LVAR_AL2:.*]]) : index = (%[[AL2_LB]]) to (%[[AL2_UB]]) step (%[[AL2_STEP]]) {
  // CHECK: memref.alloca_scope
  scf.parallel (%j) = (%arg1) to (%arg3) step (%arg5) {
    // CHECK: "test.payload2"(%[[LVAR_AL2]]) : (index) -> ()
    "test.payload2"(%j) : (index) -> ()
    // CHECK:   omp.yield
    // CHECK: }
  }
  // CHECK:   }
  // CHECK:   omp.terminator
  // CHECK: }
  return
}
