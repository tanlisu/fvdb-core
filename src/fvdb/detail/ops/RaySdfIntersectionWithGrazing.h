// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_RAYSDFINTERSECTIONWITHGRAZING_H
#define FVDB_DETAIL_OPS_RAYSDFINTERSECTIONWITHGRAZING_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/ops/RaySdfPoints.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief The first zero crossing and the grazing point, from a single traversal.
///
/// Cheaper than calling raySdfIntersection() and raySdfGrazing() separately, for two reasons:
/// the crossing search reuses the roots of the derivative that the grazing search already
/// computes per primal voxel, and because voxels are visited in ray order, every grazing
/// candidate collected before the traversal stops at the crossing is automatically in front of
/// the surface -- so no separate tMax pass is needed, and none is accepted here.
///
/// @param batchHdl The grid the SDF is stored on (the dual grid; see raySdfIntersection).
/// @return Slot 0 holds the crossing, slot 1 the grazing point; see RaySdfPointsResult.
/// @param refine 0 = Newton (default), 1 = fixed 8-step bisection.
/// @param graze 0 = analytic quadratic (default), 1 = midpoint-pair bisection.
RaySdfPointsResult raySdfIntersectionWithGrazing(const GridBatchData &batchHdl,
                                                 const JaggedTensor &rayOrigins,
                                                 const JaggedTensor &rayDirections,
                                                 const JaggedTensor &sdf,
                                                 double tMin,
                                                 double grazeTMin,
                                                 double relaxationEps,
                                                 double itxEps,
                                                 double eps,
                                                 int refine,
                                                 int graze);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_RAYSDFINTERSECTIONWITHGRAZING_H
