// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#include <fvdb/detail/ops/RaySdfCommon.cuh>
#include <fvdb/detail/ops/RaySdfGrazing.h>

namespace fvdb {
namespace detail {
namespace ops {

RaySdfPointsResult
raySdfGrazing(const GridBatchData &batchHdl,
              const JaggedTensor &rayOrigins,
              const JaggedTensor &rayDirections,
              const JaggedTensor &sdf,
              const std::optional<JaggedTensor> &tMax,
              const std::optional<JaggedTensor> &rayMask,
              double grazeTMin,
              double relaxationEps,
              double itxEps,
              double eps,
              int graze) {
    raysdf::checkListDims(rayOrigins, rayDirections, sdf);
    return FVDB_DISPATCH_KERNEL_DEVICE(rayOrigins.device(), [&]() {
        return raysdf::launchWithModes<DeviceTag, /*WantCrossing=*/false, /*WantGrazing=*/true>(
            /*refine=*/0,
            graze,

            batchHdl,
            rayOrigins,
            rayDirections,
            sdf,
            tMax,
            rayMask,
            0.0,
            grazeTMin,
            relaxationEps,
            itxEps,
            eps);
    });
}

} // namespace ops
} // namespace detail
} // namespace fvdb
