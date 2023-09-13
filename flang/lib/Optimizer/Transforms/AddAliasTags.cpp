//===- AddAliasTags.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
/// \file
/// TODO: write documentation
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Analysis/AliasAnalysis.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Dialect/FirAliasAnalysisOpInterface.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace fir {
#define GEN_PASS_DEF_ADDALIASTAGS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

#define DEBUG_TYPE "fir-add-alias-tags"

namespace {

// TODO: documentation
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

  mlir::LLVM::TBAATagAttr getTag(llvm::StringRef uniqueId);

private:
  const std::string parentId;
  mlir::MLIRContext *const context;
  mlir::LLVM::TBAATypeDescriptorAttr parent;
  llvm::DenseMap<llvm::StringRef, mlir::LLVM::TBAATagAttr> tagDedup;
};

// TODO: documentation
struct PassState {
  SubtreeState globalDataTree;
  SubtreeState allocatedDataTree;
  // TODO: this might produce incorrect results in the presnence of function
  // inlining and instruction re-ordering
  SubtreeState dummyArgDataTree;

  explicit PassState(SubtreeState globalDataTree,
                     SubtreeState allocatedDataTree,
                     SubtreeState dummyArgDataTree)
      : globalDataTree{std::move(globalDataTree)},
        allocatedDataTree{std::move(allocatedDataTree)},
        dummyArgDataTree(std::move(dummyArgDataTree)) {}

  inline const fir::AliasAnalysis::Source &getSource(mlir::Value value) {
    if (!analysisCache.contains(value))
      analysisCache[value] = analysis.getSource(value);
    return analysisCache[value];
  }

private:
  fir::AliasAnalysis analysis;
  llvm::DenseMap<mlir::Value, fir::AliasAnalysis::Source> analysisCache;
};

class AddAliasTagsPass : public fir::impl::AddAliasTagsBase<AddAliasTagsPass> {
public:
  void runOnOperation() override;

private:
  void runOnAliasInterface(fir::FirAliasAnalysisOpInterface op,
                           PassState &state);
};

} // namespace

mlir::LLVM::TBAATagAttr SubtreeState::getTag(llvm::StringRef uniqueName) {
  mlir::LLVM::TBAATagAttr &tag = tagDedup[uniqueName];
  if (tag)
    return tag;
  std::string id = (parentId + "/" + uniqueName).str();
  mlir::LLVM::TBAATypeDescriptorAttr type =
      mlir::LLVM::TBAATypeDescriptorAttr::get(
          context, id, mlir::LLVM::TBAAMemberAttr::get(parent, 0));
  tag = mlir::LLVM::TBAATagAttr::get(type, type, 0);
  return tag;
}

void AddAliasTagsPass::runOnAliasInterface(fir::FirAliasAnalysisOpInterface op,
                                           PassState &state) {
  llvm::SmallVector<mlir::Value> accessedOperands = op.getAccessedOperands();
  assert(accessedOperands.size() == 1 &&
         "load and store only access one address");
  mlir::Value memref = accessedOperands.front();

  // skip boxes
  if (mlir::isa<fir::BaseBoxType>(fir::unwrapRefType(memref.getType())))
    return;
  LLVM_DEBUG(llvm::dbgs() << "Analysing " << op << "\n");

  const fir::AliasAnalysis::Source &source = state.getSource(memref);
  if (source.isTargetOrPointer()) {
    LLVM_DEBUG(llvm::dbgs().indent(2) << "Skipping TARGET/POINTER\n");
    return;
  }

  mlir::LLVM::TBAATagAttr tag;
  if (source.kind == fir::AliasAnalysis::SourceKind::Global) {
    mlir::SymbolRefAttr glbl = source.u.get<mlir::SymbolRefAttr>();
    const char *name = glbl.getRootReference().data();
    LLVM_DEBUG(llvm::dbgs().indent(2) << "Found reference to global " << name
                                      << " at " << *op << "\n");
    tag = state.globalDataTree.getTag(name);
  } else if (source.kind == fir::AliasAnalysis::SourceKind::Allocate) {
    std::optional<llvm::StringRef> name;
    mlir::Operation *sourceOp = source.u.get<mlir::Value>().getDefiningOp();
    if (auto alloc = mlir::dyn_cast_or_null<fir::AllocaOp>(sourceOp))
      name = alloc.getUniqName();
    else if (auto alloc = mlir::dyn_cast_or_null<fir::AllocMemOp>(sourceOp))
      name = alloc.getUniqName();
    if (name) {
      LLVM_DEBUG(llvm::dbgs().indent(2) << "Found reference to allocation "
                                        << name << " at " << *op << "\n");
      tag = state.allocatedDataTree.getTag(*name);
    } else {
      LLVM_DEBUG(llvm::dbgs().indent(2)
                 << "WARN: couldn't find a name for allocation " << *op
                 << "\n");
    }
  } else if (source.kind == fir::AliasAnalysis::SourceKind::Argument) {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "Found reference to dummy argument at " << *op << "\n");
    // TODO: make a better name
    // value impls should be unique within a given mlir context. Use its address
    // as a unique id
    tag = state.dummyArgDataTree.getTag(
        "dummy" +
        std::to_string((uintptr_t)source.u.get<mlir::Value>().getImpl()));
  } else {
    LLVM_DEBUG(llvm::dbgs().indent(2)
               << "WARN: unsupported value: " << source << "\n");
  }

  if (tag)
    op.setTBAATags(mlir::ArrayAttr::get(&getContext(), tag));
}

void AddAliasTagsPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "=== Begin " DEBUG_TYPE " ===\n");
  mlir::MLIRContext *ctx = &getContext();

  // TODO: share with fir::TBAABuilder
  static constexpr llvm::StringRef flangTBAARootId = "Flang Type TBAA Root";
  static constexpr llvm::StringRef anyAccessTypeDescId = "any access";
  static constexpr llvm::StringRef anyDataAccessTypeDescId = "any data access";
  mlir::LLVM::TBAARootAttr flangTBAARoot = mlir::LLVM::TBAARootAttr::get(
      ctx, mlir::StringAttr::get(ctx, flangTBAARootId));
  mlir::LLVM::TBAATypeDescriptorAttr anyAccessTypeDesc =
      mlir::LLVM::TBAATypeDescriptorAttr::get(
          ctx, anyAccessTypeDescId,
          mlir::LLVM::TBAAMemberAttr::get(flangTBAARoot, 0));
  mlir::LLVM::TBAATypeDescriptorAttr anyDataAccessTypeDesc =
      mlir::LLVM::TBAATypeDescriptorAttr::get(
          ctx, anyDataAccessTypeDescId,
          mlir::LLVM::TBAAMemberAttr::get(anyAccessTypeDesc, 0));

  // MLIR forbids storing state in a pass because different instances might be
  // used in different threads
  // Instead this pass stores state per mlir::ModuleOp (which is what MLIR
  // thinks the pass operates on), then the real work of the pass is done in
  // runOnAliasInterface
  PassState state{SubtreeState{ctx, "global data", anyDataAccessTypeDesc},
                  SubtreeState{ctx, "allocated data", anyDataAccessTypeDesc},
                  SubtreeState{ctx, "dummy arg data", anyDataAccessTypeDesc}};

  mlir::ModuleOp mod = getOperation();
  mod.walk([&](fir::FirAliasAnalysisOpInterface op) {
    runOnAliasInterface(op, state);
  });

  LLVM_DEBUG(llvm::dbgs() << "=== End " DEBUG_TYPE " ===\n");
}

std::unique_ptr<mlir::Pass> fir::createAliasTagsPass() {
  return std::make_unique<AddAliasTagsPass>();
}
