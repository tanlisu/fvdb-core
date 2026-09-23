// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_RAYSDFINTERSECTION_H
#define FVDB_DETAIL_OPS_RAYSDFINTERSECTION_H

#include <fvdb/GridBatchData.h>
#include <fvdb/JaggedTensor.h>
#include <fvdb/detail/ops/RaySdfPoints.h>

#include <torch/types.h>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief First zero crossing of the trilinearly interpolated SDF along each ray.
///
/// Where rayImplicitIntersection() treats the field as constant per voxel, this solves the
/// cubic the interpolant traces through each primal voxel, so the hit is exact within the
/// interpolant and a pair of roots inside a single voxel cannot be missed. A sign change in
/// either direction counts, so a ray starting inside the surface reports its exit.
///
/// @param batchHdl The grid the SDF is stored on, i.e. the DUAL of the grid whose voxels
///                 define the geometry: one scalar per voxel of `batchHdl`.
/// @param rayOrigins World-space ray origins, shape (N, 3).
/// @param rayDirections World-space ray directions, shape (N, 3). Assumed normalized.
/// @param sdf One SDF value per voxel of `batchHdl`, shape (N_voxels,).
/// @param tMin Earliest accepted crossing.
/// @param eps Skip primal voxels whose ray segment is shorter than this.
/// @return Slot 0 holds the crossing; see RaySdfPointsResult.
/// @param refine 0 = Newton (default), 1 = fixed 8-step bisection.
RaySdfPointsResult raySdfIntersection(const GridBatchData &batchHdl,
                                      const JaggedTensor &rayOrigins,
                                      const JaggedTensor &rayDirections,
                                      const JaggedTensor &sdf,
                                      double tMin,
                                      double eps,
                                      int refine);

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_RAYSDFINTERSECTION_H
