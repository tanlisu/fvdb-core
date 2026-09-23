// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_RAYSDFPOINTS_H
#define FVDB_DETAIL_OPS_RAYSDFPOINTS_H

#include <fvdb/JaggedTensor.h>

namespace fvdb {
namespace detail {
namespace ops {

/// @brief Result of the ray_sdf_* ops: per-ray points of the trilinear SDF interpolant.
///
/// Laid out per ray with a trailing slot dimension P -- P == 1 for the single-point ops,
/// P == 2 for ray_sdf_intersection_with_grazing (slot 0 = crossing, slot 1 = grazing):
///
///   times [N, P]    ray parameter t of the point (world units, ray_directions assumed unit)
///   mask  [N, P]    bool, whether that point was found and accepted
///   sdf   [N, P]    SDF value at the point
///   grad  [N, P, 3] spatial SDF gradient at the point
///
/// Rows where mask is false are zeroed.
struct RaySdfPointsResult {
    JaggedTensor times;
    JaggedTensor mask;
    JaggedTensor sdf;
    JaggedTensor grad;
};

} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_RAYSDFPOINTS_H
