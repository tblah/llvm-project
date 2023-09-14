//===-- TBAAForest.h - A TBAA tree for each function -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_ANALYSIS_TBAA_FOREST_H
#define FORTRAN_OPTIMIZER_ANALYSIS_TBAA_FOREST_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/DenseMap.h"
#include <string>

namespace fir {

//===----------------------------------------------------------------------===//
// SubtreeState
//===----------------------------------------------------------------------===//
/// TODO: documenation
class SubtreeState {
public:
  SubtreeState(mlir::MLIRContext *ctx, std::string name,
               mlir::LLVM::TBAANodeAttr grandParent)
      : parentId{std::move(name)}, context(ctx) {
    parent = mlir::LLVM::TBAATypeDescriptorAttr::get(
        context, parentId, mlir::LLVM::TBAAMemberAttr::get(grandParent, 0));
  }

  SubtreeState(const SubtreeState &) = delete;
  SubtreeState(SubtreeState &&) = default;

  mlir::LLVM::TBAATagAttr getTag(llvm::StringRef uniqueId) const;

private:
  const std::string parentId;
  mlir::MLIRContext *const context;
  mlir::LLVM::TBAATypeDescriptorAttr parent;
  llvm::DenseMap<llvm::StringRef, mlir::LLVM::TBAATagAttr> tagDedup;
};

//===----------------------------------------------------------------------===//
// TBAATree
//===----------------------------------------------------------------------===//
/// TODO: documentation
/// per-function TBAA tree
struct TBAATree {
  SubtreeState globalDataTree;
  SubtreeState allocatedDataTree;
  SubtreeState dummyArgDataTree;
  mlir::LLVM::TBAATypeDescriptorAttr boxMemberTypeDesc;
  mlir::LLVM::TBAATypeDescriptorAttr anyDataTypeDesc;

  static TBAATree buildTree(mlir::StringAttr functionName);

private:
  explicit TBAATree(mlir::LLVM::TBAATypeDescriptorAttr dataRoot,
                    mlir::LLVM::TBAATypeDescriptorAttr boxMemberTypeDesc);
};

//===----------------------------------------------------------------------===//
// TBAAForrest
//===----------------------------------------------------------------------===//
/// Colletion of TBAATrees, indexed by function
/// TODO: documentation
class TBAAForrest {
public:
  inline const TBAATree &operator[](mlir::func::FuncOp func) {
    return getFuncTree(func.getSymNameAttr());
  }
  inline const TBAATree &operator[](mlir::LLVM::LLVMFuncOp func) {
    return getFuncTree(func.getSymNameAttr());
  }

private:
  const TBAATree &getFuncTree(mlir::StringAttr symName) {
    if (!trees.contains(symName))
      trees.insert({symName, TBAATree::buildTree(symName)});
    return trees.at(symName);
  }

  // TBAA tree per function
  llvm::DenseMap<mlir::StringAttr, TBAATree> trees;
};
} // namespace fir

#endif // FORTRAN_OPTIMIZER_ANALYSIS_TBAA_FOREST_H
