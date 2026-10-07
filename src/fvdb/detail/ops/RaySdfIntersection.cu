// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/RaySdfCommon.cuh>
#include <fvdb/detail/ops/RaySdfIntersection.h>

namespace fvdb {
namespace detail {
namespace ops {

RaySdfPointsResult
raySdfIntersection(const GridBatchData &batchHdl,
                   const JaggedTensor &rayOrigins,
                   const JaggedTensor &rayDirections,
                   const JaggedTensor &sdf,
                   double tMin,
                   double eps,
                   int refine) {
    raysdf::checkListDims(rayOrigins, rayDirections, sdf);
    return FVDB_DISPATCH_KERNEL_DEVICE(rayOrigins.device(), [&]() {
        return raysdf::launchWithModes<DeviceTag, /*WantCrossing=*/true, /*WantGrazing=*/false>(
            refine,
            /*graze=*/0,

            batchHdl,
            rayOrigins,
            rayDirections,
            sdf,
            std::nullopt,
            std::nullopt,
            tMin,
            0.0,
            0.0,
            0.0,
            eps);
    });
}

} // namespace ops
} // namespace detail
} // namespace fvdb
