//===-- Optimizer/OpenMP/Utils.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_OPTIMIZER_OPENMP_UTILS_H
#define FORTRAN_OPTIMIZER_OPENMP_UTILS_H

#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/SmallVector.h"

namespace flangomp {

enum class DoConcurrentMappingKind {
  DCMK_None,  ///< Do not lower `do concurrent` to OpenMP.
  DCMK_Host,  ///< Lower to run in parallel on the CPU.
  DCMK_Device ///< Lower to run in parallel on the GPU.
};

/// Isolate an outlineable OpenMP op's region from above by threading any
/// non-clonable external values through the op's \c shared_vars clause. Pure
/// ops with no sub-regions are cloned into the region instead.
///
/// \c makeRegionIsolatedFromAbove appends the new shared block args AFTER the
/// existing block args, but \c BlockArgOpenMPOpInterface expects shared args
/// BEFORE reduction (and other post-shared) args. When the op already has
/// block args from other clauses (e.g. reduction_vars) at positions that
/// would conflict with the new shared args, we reorder the entry block's
/// arguments to satisfy the interface's expected layout.
template <typename OpTy>
void isolateOutlineableOpFromAbove(OpTy op, mlir::RewriterBase &rewriter) {
  auto isSafeToClone = [](mlir::Operation *op) -> bool {
    return mlir::isPure(op) && op->getNumRegions() == 0;
  };
  mlir::SmallVector<mlir::Value> captured =
      mlir::makeRegionIsolatedFromAbove(rewriter, op.getRegion(), isSafeToClone);
  if (captured.empty())
    return;

  // Determine where the interface expects shared block args to begin (after
  // private, before reduction).
  auto iface = llvm::cast<mlir::omp::BlockArgOpenMPOpInterface>(*op);
  unsigned sharedStart = iface.getSharedBlockArgsStart();
  mlir::Block &entry = op.getRegion().front();
  unsigned totalArgs = entry.getNumArguments();
  unsigned capturedCount = captured.size();
  unsigned existingCount = totalArgs - capturedCount;

  // If the appended shared args are already in the correct position (i.e.
  // there were no block args in [sharedStart..existingCount)), just assign.
  if (sharedStart == existingCount) {
    rewriter.modifyOpInPlace(op, [&]() {
      op.getSharedVarsMutable().assign(captured);
    });
    return;
  }

  // The current layout after makeRegionIsolatedFromAbove:
  //   [old_args[0..existingCount), captured_args[existingCount..totalArgs)]
  //
  // We need:
  //   [old_args[0..sharedStart), captured_args, old_args[sharedStart..existingCount)]
  //
  // Create a new entry block with the reordered argument types/locs.
  llvm::SmallVector<mlir::Type> newTypes;
  llvm::SmallVector<mlir::Location> newLocs;
  auto addArgInfo = [&](mlir::BlockArgument arg) {
    newTypes.push_back(arg.getType());
    newLocs.push_back(arg.getLoc());
  };
  for (unsigned i = 0; i < sharedStart; ++i)
    addArgInfo(entry.getArgument(i));
  for (unsigned i = existingCount; i < totalArgs; ++i)
    addArgInfo(entry.getArgument(i));
  for (unsigned i = sharedStart; i < existingCount; ++i)
    addArgInfo(entry.getArgument(i));

  mlir::Block *newEntry = rewriter.createBlock(
      &op.getRegion(), op.getRegion().begin(), newTypes, newLocs);

  // Map each old block arg to the corresponding new block arg:
  //   old[0..sharedStart)             → new[0..sharedStart)
  //   old[sharedStart..existingCount) → new[sharedStart+capturedCount..totalArgs)
  //   old[existingCount..totalArgs)   → new[sharedStart..sharedStart+capturedCount)
  llvm::SmallVector<mlir::Value> mergeMapping(totalArgs);
  for (unsigned i = 0; i < sharedStart; ++i)
    mergeMapping[i] = newEntry->getArgument(i);
  for (unsigned i = sharedStart; i < existingCount; ++i)
    mergeMapping[i] = newEntry->getArgument(i + capturedCount);
  for (unsigned i = 0; i < capturedCount; ++i)
    mergeMapping[existingCount + i] = newEntry->getArgument(sharedStart + i);

  rewriter.mergeBlocks(&entry, newEntry, mergeMapping);

  rewriter.modifyOpInPlace(op, [&]() {
    op.getSharedVarsMutable().assign(captured);
  });
}

} // namespace flangomp

#endif // FORTRAN_OPTIMIZER_OPENMP_UTILS_H
