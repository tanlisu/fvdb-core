// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/RaySdfCommon.cuh>
#include <fvdb/detail/ops/RaySdfIntersectionWithGrazing.h>

namespace fvdb {
namespace detail {
namespace ops {

RaySdfPointsResult
raySdfIntersectionWithGrazing(const GridBatchData &batchHdl,
                              const JaggedTensor &rayOrigins,
                              const JaggedTensor &rayDirections,
                              const JaggedTensor &sdf,
                              double tMin,
                              double grazeTMin,
                              double relaxationEps,
                              double itxEps,
                              double eps,
                              int refine,
                              int graze) {
    raysdf::checkListDims(rayOrigins, rayDirections, sdf);
    return FVDB_DISPATCH_KERNEL_DEVICE(rayOrigins.device(), [&]() {
        return raysdf::launchWithModes<DeviceTag, /*WantCrossing=*/true, /*WantGrazing=*/true>(
            refine,
            graze,
            batchHdl,
            rayOrigins,
            rayDirections,
            sdf,
            std::nullopt,
            std::nullopt,
            tMin,
            grazeTMin,
            relaxationEps,
            itxEps,
            eps);
    });
}

} // namespace ops
} // namespace detail
} // namespace fvdb
