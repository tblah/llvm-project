// Use --mlir-disable-threading so that the AA queries are serialized
// as well as its diagnostic output.
// RUN: fir-opt %s -pass-pipeline='builtin.module(func.func(test-fir-alias-analysis))' -split-input-file --mlir-disable-threading 2>&1 | FileCheck %s

// Check that alias analysis can recover the original source when it reaches
// an OpenMP shared block argument after peeling a defining op.

// CHECK-LABEL: Testing : "test_shared_decl"
// CHECK-DAG: orig#0 <-> shared_decl#0: MustAlias

func.func @test_shared_decl() {
  %orig = fir.address_of(@global0) {test.ptr = "orig"} : !fir.ref<i32>
  omp.parallel shared(%orig -> %shared : !fir.ref<i32>) {
    %shared_decl:2 = hlfir.declare %shared {test.ptr = "shared_decl", uniq_name = "_QFshared"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %c1 = arith.constant 1 : i32
    hlfir.assign %c1 to %shared_decl#0 : i32, !fir.ref<i32>
    omp.terminator
  }
  return
}

fir.global internal @global0 : i32 {
  %c0 = arith.constant 0 : i32
  fir.has_value %c0 : i32
}

// -----

// Check that alias analysis can look through nested OpenMP shared clauses when
// getSource() is invoked directly on the innermost shared block argument.

// CHECK-LABEL: Testing : "test_nested_shared_direct"
// "parallel.region0#0" means the 0th block argument to parallel's 0th region.
// CHECK-DAG: orig#0 <-> parallel.region0#0: MustAlias

func.func @test_nested_shared_direct() {
  %orig = fir.address_of(@global1) {test.ptr = "orig"} : !fir.ref<i32>
  omp.teams shared(%orig -> %teams_shared : !fir.ref<i32>) {
    omp.parallel shared(%teams_shared -> %parallel_shared : !fir.ref<i32>) {
      %c2 = arith.constant 2 : i32
      hlfir.assign %c2 to %parallel_shared : i32, !fir.ref<i32>
      omp.terminator
    } {test.ptr = "parallel"}
    omp.terminator
  } {omp.composite}
  return
}

fir.global internal @global1 : i32 {
  %c0 = arith.constant 0 : i32
  fir.has_value %c0 : i32
}
