// Test LLVM IR translation of the OpenMP `shared` clause. Shared variables are
// passed into the outlined region via the struct arg mechanism and their block
// arguments inside the region are identity-mapped to the loaded outer value.

// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @foo(!llvm.ptr) -> ()

// CHECK-LABEL: @omp_parallel_shared
// CHECK-SAME:  (ptr [[ARG0:%.+]])
// CHECK:         [[STRUCTARG:%.+]] = alloca { ptr }
// CHECK:         [[GEP:%.+]] = getelementptr { ptr }, ptr [[STRUCTARG]], i32 0, i32 0
// CHECK:         store ptr [[ARG0]], ptr [[GEP]]
// CHECK:         @__kmpc_fork_call(ptr @{{.+}}, i32 1, ptr [[OUTLINED:@[^,)]+]], ptr [[STRUCTARG]])
// CHECK:         ret void
llvm.func @omp_parallel_shared(%arg0: !llvm.ptr) {
  omp.parallel shared(%arg0 -> %arg0_shared : !llvm.ptr) {
    llvm.call @foo(%arg0_shared) : (!llvm.ptr) -> ()
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define internal void @omp_parallel_shared..omp_par
// CHECK-SAME:  (ptr noalias {{.+}}, ptr noalias {{.+}}, ptr [[DATA:%.+]])
// CHECK:         [[GEP:%.+]] = getelementptr { ptr }, ptr [[DATA]], i32 0, i32 0
// CHECK:         [[LOADGEP:%.+]] = load ptr, ptr [[GEP]]
// CHECK:         call void @foo(ptr [[LOADGEP]])

// -----

llvm.func @bar(!llvm.ptr, !llvm.ptr) -> ()

// CHECK-LABEL: @omp_parallel_multiple_shared
// CHECK-SAME:  (ptr [[A:%.+]], ptr [[B:%.+]])
// CHECK:         [[STRUCTARG:%.+]] = alloca { ptr, ptr }
// CHECK:         [[GEP0:%.+]] = getelementptr { ptr, ptr }, ptr [[STRUCTARG]], i32 0, i32 0
// CHECK:         store ptr [[A]], ptr [[GEP0]]
// CHECK:         [[GEP1:%.+]] = getelementptr { ptr, ptr }, ptr [[STRUCTARG]], i32 0, i32 1
// CHECK:         store ptr [[B]], ptr [[GEP1]]
// CHECK:         @__kmpc_fork_call(ptr @{{.+}}, i32 1, ptr [[OUTLINED:@[^,)]+]], ptr [[STRUCTARG]])
// CHECK:         ret void
llvm.func @omp_parallel_multiple_shared(%a: !llvm.ptr, %b: !llvm.ptr) {
  omp.parallel shared(%a -> %a_shared, %b -> %b_shared : !llvm.ptr, !llvm.ptr) {
    llvm.call @bar(%a_shared, %b_shared) : (!llvm.ptr, !llvm.ptr) -> ()
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define internal void @omp_parallel_multiple_shared..omp_par
// CHECK-SAME:  (ptr noalias {{.+}}, ptr noalias {{.+}}, ptr [[DATA:%.+]])
// CHECK:         [[GEP0:%.+]] = getelementptr { ptr, ptr }, ptr [[DATA]], i32 0, i32 0
// CHECK:         [[LOADA:%.+]] = load ptr, ptr [[GEP0]]
// CHECK:         [[GEP1:%.+]] = getelementptr { ptr, ptr }, ptr [[DATA]], i32 0, i32 1
// CHECK:         [[LOADB:%.+]] = load ptr, ptr [[GEP1]]
// CHECK:         call void @bar(ptr [[LOADA]], ptr [[LOADB]])

// -----

llvm.func @baz(!llvm.ptr) -> ()

// CHECK-LABEL: @omp_teams_shared
// CHECK-SAME:  (ptr [[ARG0:%.+]])
// CHECK:         [[STRUCTARG:%.+]] = alloca { ptr }
// CHECK:         [[GEP:%.+]] = getelementptr { ptr }, ptr [[STRUCTARG]], i32 0, i32 0
// CHECK:         store ptr [[ARG0]], ptr [[GEP]]
// CHECK:         @__kmpc_fork_teams(ptr @{{.+}}, i32 1, ptr @{{.+}}, ptr [[STRUCTARG]])
// CHECK:         ret void
llvm.func @omp_teams_shared(%arg0: !llvm.ptr) {
  omp.teams shared(%arg0 -> %arg0_shared : !llvm.ptr) {
    llvm.call @baz(%arg0_shared) : (!llvm.ptr) -> ()
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define internal void @omp_teams_shared..omp_par
// CHECK-SAME:  (ptr {{.+}}, ptr {{.+}}, ptr [[DATA:%.+]])
// CHECK:         [[GEP:%.+]] = getelementptr { ptr }, ptr [[DATA]], i32 0, i32 0
// CHECK:         [[LOADGEP:%.+]] = load ptr, ptr [[GEP]]
// CHECK:         call void @baz(ptr [[LOADGEP]])

// -----

llvm.func @qux(!llvm.ptr) -> ()

// CHECK-LABEL: @omp_task_shared
// CHECK-SAME:  (ptr [[ARG0:%.+]])
// CHECK:         [[STRUCTARG:%.+]] = alloca { ptr }
// CHECK:         [[GEP:%.+]] = getelementptr { ptr }, ptr [[STRUCTARG]], i32 0, i32 0
// CHECK:         store ptr [[ARG0]], ptr [[GEP]]
// CHECK:         @__kmpc_omp_task_alloc(
// CHECK:         call void @llvm.memcpy{{.+}}(ptr {{.+}}, ptr {{.+}} [[STRUCTARG]], i64 8
// CHECK:         @__kmpc_omp_task(
// CHECK:         ret void
llvm.func @omp_task_shared(%arg0: !llvm.ptr) {
  omp.task shared(%arg0 -> %arg0_shared : !llvm.ptr) {
    llvm.call @qux(%arg0_shared) : (!llvm.ptr) -> ()
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define internal void @omp_task_shared..omp_par
// CHECK:         [[SHAREDS:%.+]] = load ptr, ptr %{{.+}}
// CHECK:         [[GEP:%.+]] = getelementptr { ptr }, ptr [[SHAREDS]], i32 0, i32 0
// CHECK:         [[LOADGEP:%.+]] = load ptr, ptr [[GEP]]
// CHECK:         call void @qux(ptr [[LOADGEP]])

// -----

llvm.func @use(!llvm.ptr) -> ()

// CHECK-LABEL: @omp_taskloop_context_shared
// CHECK-SAME:  (ptr [[ARG0:%.+]])
// CHECK:         [[STRUCTARG:%.+]] = alloca { i64, i64, i64, ptr }
// CHECK:         [[GEP:%.+]] = getelementptr { i64, i64, i64, ptr }, ptr [[STRUCTARG]], i32 0, i32 3
// CHECK:         store ptr [[ARG0]], ptr [[GEP]]
// CHECK:         @__kmpc_taskloop(
// CHECK:         ret void
llvm.func @omp_taskloop_context_shared(%arg0: !llvm.ptr) {
  %c1 = llvm.mlir.constant(1 : i32) : i32
  %c5 = llvm.mlir.constant(5 : i32) : i32
  omp.taskloop.context shared(%arg0 -> %arg0_shared : !llvm.ptr) {
    omp.taskloop.wrapper {
      omp.loop_nest (%iv) : i32 = (%c1) to (%c5) inclusive step (%c1) {
        llvm.call @use(%arg0_shared) : (!llvm.ptr) -> ()
        omp.yield
      }
    }
    omp.terminator
  }
  llvm.return
}

// CHECK-LABEL: define internal void @omp_taskloop_context_shared..omp_par
// CHECK:         [[SHAREDS:%.+]] = load ptr, ptr %{{.+}}
// CHECK:         [[GEP:%.+]] = getelementptr { i64, i64, i64, ptr }, ptr [[SHAREDS]], i32 0, i32 3
// CHECK:         [[LOADGEP:%.+]] = load ptr, ptr [[GEP]]
// CHECK:         call void @use(ptr [[LOADGEP]])
