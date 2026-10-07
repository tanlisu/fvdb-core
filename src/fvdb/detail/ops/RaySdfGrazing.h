// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_RAYSDFGRAZING_H
#define FVDB_DETAIL_OPS_RAYSDFGRAZING_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/ops/RaySdfPoints.h>

#include <torch/types.h>

#include <optional>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Grazing point: the local minimum of the SDF along each ray that lies inside the
///        relaxation band, i.e. where d/dt SDF(o + t*d) passes through zero from approaching
///        to receding and itxEps < SDF < relaxationEps.
///
/// Inside a primal voxel that derivative is a quadratic and is solved exactly. Because the
/// interpolant is C0 but not C1 the minimum can also sit exactly on a voxel face, which is
/// handled by minimising over the closed interval rather than by testing for a sign change.
///
/// The lower band edge is itxEps rather than 0 deliberately: SDF == 0 is the surface, so such
/// a point is a tangential hit belonging to the interior term, and testing against a bare zero
/// lets rounding flip points between the two sets.
///
/// @param batchHdl The grid the SDF is stored on (the dual grid; see raySdfIntersection).
/// @param tMax Optional per-ray upper bound on the search, shape (N,). Typically the first
///             surface hit, since a grazing point behind the surface is not visible. When
///             absent the search runs to the end of the grid.
/// @param rayMask Optional per-ray enable, shape (N,).
/// @param grazeTMin Earliest accepted candidate; keeps the search off the ray origin.
/// @return Slot 0 holds the grazing point; see RaySdfPointsResult.
/// @param graze 0 = analytic quadratic (default), 1 = midpoint-pair bisection.
RaySdfPointsResult raySdfGrazing(const GridBatchData &batchHdl,
                                 const JaggedTensor &rayOrigins,
                                 const JaggedTensor &rayDirections,
                                 const JaggedTensor &sdf,
                                 const std::optional<JaggedTensor> &tMax,
                                 const std::optional<JaggedTensor> &rayMask,
                                 double grazeTMin,
                                 double relaxationEps,
                                 double itxEps,
                                 double eps,
                                 int graze);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_RAYSDFGRAZING_H
