// RUN: fir-opt --split-input-file --fir-to-llvm-ir %s | FileCheck %s

func.func @real32(%arg0: f32) -> f32 {
  %0 = fir.no_reassoc %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: llvm.func @real32
// CHECK-SAME: %[[ARG0:.*]]: f32
// CHECK: %[[FENCE:.*]] = llvm.intr.arithmetic.fence(%[[ARG0]]) : (f32) -> f32
// CHECK: llvm.return %[[FENCE]] : f32

// -----

func.func @real64(%arg0: f64) -> f64 {
  %0 = fir.no_reassoc %arg0 : f64
  return %0 : f64
}

// CHECK-LABEL: llvm.func @real64
// CHECK-SAME: %[[ARG0:.*]]: f64
// CHECK: %[[FENCE:.*]] = llvm.intr.arithmetic.fence(%[[ARG0]]) : (f64) -> f64
// CHECK: llvm.return %[[FENCE]] : f64

// -----

func.func @integer(%arg0: i32) -> i32 {
  %0 = fir.no_reassoc %arg0 : i32
  return %0 : i32
}

// CHECK-LABEL: llvm.func @integer
// CHECK-SAME: %[[ARG0:.*]]: i32
// CHECK-NOT: llvm.intr.arithmetic.fence
// CHECK: llvm.return %[[ARG0]] : i32
