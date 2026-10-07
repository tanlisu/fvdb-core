// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
//
#ifndef FVDB_DETAIL_OPS_RAYSDFCOMMON_CUH
#define FVDB_DETAIL_OPS_RAYSDFCOMMON_CUH

// Shared implementation behind ray_sdf_intersection, ray_sdf_grazing and
// ray_sdf_intersection_with_grazing.
//
// The three ops are one traversal specialised on compile-time flags, so the algorithm lives
// here once and each op file instantiates it. Keeping one copy matters more than the extra
// compile time: the crossing and grazing searches share per-primal-voxel work (the roots of
// the derivative), and any drift between separate copies would be a silent correctness bug.

#include <fvdb/detail/ops/RaySdfPoints.h>
#include <fvdb/detail/utils/AccessorHelpers.cuh>
#include <fvdb/detail/utils/ForEachCPU.h>
#include <fvdb/detail/utils/TrilinearStencil.h>
#include <fvdb/detail/utils/Utils.h>
#include <fvdb/detail/utils/cuda/Caching.cuh>
#include <fvdb/detail/utils/cuda/ForEachCUDA.cuh>
#include <fvdb/detail/utils/nanovdb/HDDAIterators.h>

#include <ATen/OpMathType.h>
#include <c10/cuda/CUDAException.h>

#include <limits>

namespace fvdb {
namespace detail {
namespace ops {
namespace raysdf {

// How the crossing and the grazing point are searched for.
//
//   Both refine modes bracket each crossing exactly, by splitting every cell at the turning
//   points of the cubic the SDF traces along the ray -- so a surface thinner than a voxel is
//   seen either way -- and differ only in how the bracketed root is refined:
//   RefineMode::Newton   safeguarded Newton, to the limits of the arithmetic.
//   RefineMode::Bisect   8 halvings on the cubic's sign, to 1/256 of the bracket. (A scheme
//                        that brackets between one SDF sample per voxel instead misses features
//                        thinner than a voxel; the relaxed-boundary reference implementation's
//                        sphere tracer does not, and the cubic bracket matches its hits.)
//   GrazeMode::Analytic  minima of the per-cell cubic, including minima on cell faces.
//   GrazeMode::Bisect    midpoint derivative pairs of adjacent voxels, bisected 8 times on the
//                        sign of the along-ray slope, read from the two cells' cubics.
//
//   Either way the grazing candidate is a local minimum of the SDF along the ray, and it is
//   accepted when its value lies in (itxEps, relaxationEps); see the acceptance note below.
//
// Bisect is the default for both and the mode used in practice; Newton and Analytic are exact
// alternatives kept for comparison and have not been validated as far in optimisation.
//
// Both are compile-time: every combination is instantiated and the entry points select one at
// launch, so callers switch method per call with no rebuild. The reason not to make them plain
// runtime branches is register pressure -- the bisection grazing needs machinery the analytic
// path does not, and compiling it into both would inflate the analytic path's register use and
// skew exactly the efficiency comparison these options exist to measure.
enum class RefineMode { Newton, Bisect };
enum class GrazeMode { Analytic, Bisect };

// Tunables shared by all three entry points. Packed into a struct so the device callback
// keeps a sane arity; every field is a plain scalar so it lands in constant/param space.
template <typename MathType> struct SurfacePointParams {
    MathType tMin;          // earliest accepted crossing
    MathType grazeTMin;     // earliest accepted grazing candidate
    MathType relaxationEps; // upper edge of the band a grazing point must sit in
    MathType itxEps;        // lower edge of that band; see the acceptance note below
    MathType eps;           // skip cells whose ray segment is shorter than this
};

// The trilinear interpolant restricted to a ray inside one cell is exactly a cubic in the
// ray parameter. We carry it relative to the cell's ENTRY time (tau = t - tEnter) rather
// than absolute t: for a ray that has already travelled a long way, the absolute form loses
// most of its significant bits to cancellation, while the local form keeps tau in [0, dt].
template <typename MathType> struct CellCubic {
    MathType a3, a2, a1, a0;

    __hostdev__ inline MathType
    value(MathType tau) const {
        return ((a3 * tau + a2) * tau + a1) * tau + a0;
    }
    __hostdev__ inline MathType
    deriv(MathType tau) const {
        return (MathType(3) * a3 * tau + MathType(2) * a2) * tau + a1;
    }
    __hostdev__ inline MathType
    deriv2(MathType tau) const {
        return MathType(6) * a3 * tau + MathType(2) * a2;
    }
};

// Fetch the 8 corner values of the interpolation cell whose base corner is `ijk`.
//
// Walks the corners one axis component at a time, matching resolveTrilinearStencil()'s
// traversal order so the ReadAccessor's node cache sees the same access pattern.
//
// Returns false unless ALL 8 corners are active. A partially populated cell is skipped
// rather than interpolated: the grid that owns the SDF is the dual of the geometry grid, so
// dual_grid() guarantees all 8 corners exist for every cell of the original geometry, and a
// cell missing some corners is by definition outside it. Interpolating there would silently
// extend the surface half a voxel past the band with values decaying toward zero (inactive
// corners contribute weight 0, so the stencil is not a partition of unity).
template <typename MathType, typename GridAccessorType, typename SdfAccessorType>
__hostdev__ inline bool
fetchCellCorners(const nanovdb::Coord &base,
                 GridAccessorType &gridAcc,
                 const SdfAccessorType &sdf,
                 int64_t baseOffset,
                 MathType (&V)[8]) {
    nanovdb::Coord ijk = base;

    // V is indexed [4*a + 2*b + c] for corner (i+a, j+b, k+c).
#define FVDB_FETCH_CORNER(SLOT)   \
    if (!gridAcc.isActive(ijk)) { \
        return false;             \
    }                             \
    V[SLOT] = static_cast<MathType>(_loadReadOnly(&sdf[gridAcc.getValue(ijk) - 1 + baseOffset]));

    FVDB_FETCH_CORNER(0) // (i,   j,   k  )
    ijk[2] += 1;
    FVDB_FETCH_CORNER(1) // (i,   j,   k+1)
    ijk[1] += 1;
    FVDB_FETCH_CORNER(3) // (i,   j+1, k+1)
    ijk[2] -= 1;
    FVDB_FETCH_CORNER(2) // (i,   j+1, k  )
    ijk[0] += 1;
    ijk[1] -= 1;
    FVDB_FETCH_CORNER(4) // (i+1, j,   k  )
    ijk[2] += 1;
    FVDB_FETCH_CORNER(5) // (i+1, j,   k+1)
    ijk[1] += 1;
    FVDB_FETCH_CORNER(7) // (i+1, j+1, k+1)
    ijk[2] -= 1;
    FVDB_FETCH_CORNER(6) // (i+1, j+1, k  )

#undef FVDB_FETCH_CORNER

    return true;
}

// Expand the trilinear interpolant along the ray into cubic coefficients in tau.
//
// Trilinear in the multilinear basis is
//   s = c000 + c100*u + c010*v + c001*w + c110*uv + c101*uw + c011*vw + c111*uvw
// and substituting the affine u(tau), v(tau), w(tau) collects into a cubic. The uvw term
// alone produces tau^3, which is why the field is cubic and not quadratic along a ray.
template <typename MathType>
__hostdev__ inline CellCubic<MathType>
cubicAlongRay(const MathType (&V)[8],
              MathType u0,
              MathType v0,
              MathType w0,
              MathType du,
              MathType dv,
              MathType dw) {
    const MathType c000 = V[0];
    const MathType c100 = V[4] - V[0];
    const MathType c010 = V[2] - V[0];
    const MathType c001 = V[1] - V[0];
    const MathType c110 = V[6] - V[4] - V[2] + V[0];
    const MathType c101 = V[5] - V[4] - V[1] + V[0];
    const MathType c011 = V[3] - V[2] - V[1] + V[0];
    const MathType c111 = V[7] - V[6] - V[5] - V[3] + V[4] + V[2] + V[1] - V[0];

    CellCubic<MathType> c;
    c.a3 = c111 * du * dv * dw;
    c.a2 = c110 * du * dv + c101 * du * dw + c011 * dv * dw +
           c111 * (u0 * dv * dw + v0 * du * dw + w0 * du * dv);
    c.a1 = c100 * du + c010 * dv + c001 * dw + c110 * (u0 * dv + v0 * du) +
           c101 * (u0 * dw + w0 * du) + c011 * (v0 * dw + w0 * dv) +
           c111 * (u0 * v0 * dw + u0 * w0 * dv + v0 * w0 * du);
    c.a0 = c000 + c100 * u0 + c010 * v0 + c001 * w0 + c110 * u0 * v0 + c101 * u0 * w0 +
           c011 * v0 * w0 + c111 * u0 * v0 * w0;
    return c;
}

// Roots of s'(tau) = 3*a3*tau^2 + 2*a2*tau + a1 that lie strictly inside (0, dt), returned
// ascending. These are the cell's critical points, and they do double duty: they are the
// grazing-point candidates, AND they split [0, dt] into at most three monotonic pieces so
// the crossing search cannot miss a root (in particular a pair of roots inside one cell,
// which any scheme that only tests sampled endpoints silently drops).
//
// Uses the cancellation-free quadratic form (compute the larger root via -q, the smaller as
// c/q) rather than the textbook formula, which loses precision when b^2 >> 4ac -- the common
// case here, since a3 is often near zero for rays close to axis-aligned.
template <typename MathType>
__hostdev__ inline int
criticalPoints(const CellCubic<MathType> &c, MathType dt, MathType (&out)[2]) {
    const MathType A = MathType(3) * c.a3;
    const MathType B = MathType(2) * c.a2;
    const MathType C = c.a1;

    const MathType tiny = MathType(1e-20);
    int n               = 0;
    MathType roots[2]   = {MathType(0), MathType(0)};

    if (A > -tiny && A < tiny) {
        // Degenerate to linear: a single critical point, or none if the derivative is
        // constant (the field is linear along this ray).
        if (B > tiny || B < -tiny) {
            roots[n++] = -C / B;
        }
    } else {
        const MathType disc = B * B - MathType(4) * A * C;
        if (disc >= MathType(0)) {
            const MathType sq = nanovdb::math::Sqrt(disc);
            const MathType q =
                (B >= MathType(0)) ? MathType(-0.5) * (B + sq) : MathType(-0.5) * (B - sq);
            roots[n++] = q / A;
            if (q > tiny || q < -tiny) {
                roots[n++] = C / q;
            }
        }
    }

    if (n == 2 && roots[0] > roots[1]) {
        const MathType tmp = roots[0];
        roots[0]           = roots[1];
        roots[1]           = tmp;
    }

    int m = 0;
    for (int i = 0; i < n; ++i) {
        if (roots[i] > MathType(0) && roots[i] < dt) {
            out[m++] = roots[i];
        }
    }
    return m;
}

// Refine a root known to be bracketed by [lo, hi] on a piece where s is monotonic (used by
// RefineMode::Newton; RefineMode::Bisect brackets and bisects differently, see the march).
//
// Newton from the midpoint, falling back to bisection whenever a step would leave the
// bracket. On a monotonic piece Newton converges quadratically, so this reaches the limits
// of the arithmetic in a handful of iterations.
template <typename MathType>
__hostdev__ inline MathType
refineRootNewton(const CellCubic<MathType> &c, MathType lo, MathType hi, MathType sLo) {
    MathType tau = MathType(0.5) * (lo + hi);
    for (int i = 0; i < 8; ++i) {
        const MathType s = c.value(tau);
        if (s == MathType(0)) {
            break;
        }
        // Keep the bracket around the root so a bad Newton step can always be recovered.
        if ((s < MathType(0)) == (sLo < MathType(0))) {
            lo = tau;
        } else {
            hi = tau;
        }
        MathType next;
        {
            const MathType d = c.deriv(tau);
            next             = (d != MathType(0)) ? tau - s / d : MathType(0.5) * (lo + hi);
            // Test convergence BEFORE the bracket guard, and accept a step that lands on a
            // bracket endpoint. Once Newton has converged, s/d rounds to zero, so `next`
            // equals `tau` -- and `tau` is itself an endpoint, because the bracket update
            // just set lo or hi to it. A strict `next > lo && next < hi` therefore rejects
            // the converged answer and bisects away from it, and the loop then spends its
            // remaining iterations crawling back and stops short of the root. That cost
            // ~1.5e-4 on rays where the cubic degenerates to a line (axis-aligned ones,
            // where a3 and a2 are exactly zero and Newton lands on the root immediately).
            if (next == tau) {
                break;
            }
            if (!(next >= lo && next <= hi)) {
                next = MathType(0.5) * (lo + hi);
            }
        }
        if (next == tau) {
            break;
        }
        tau = next;
    }
    return tau;
}

// RefineMode::Bisect's refinement of a bracketed root: 8 halvings on the sign of the cubic, moving
// the end whose sign matches. Lands within 1/256 of the bracket.
template <typename MathType>
__hostdev__ inline MathType
refineRootBisect(const CellCubic<MathType> &c, MathType lo, MathType hi, MathType sLo) {
    for (int i = 0; i < 8; ++i) {
        const MathType mid  = MathType(0.5) * (lo + hi);
        const MathType sMid = c.value(mid);
        if (sMid * sLo > MathType(0)) {
            lo  = mid;
            sLo = sMid;
        } else {
            hi = mid;
        }
    }
    return MathType(0.5) * (lo + hi);
}

// Per-ray surface point search. One thread per ray, no inter-ray communication.
//
// WantCrossing / WantGrazing are compile-time so each entry point compiles down to only the
// work it needs; the fused instantiation is strictly cheaper than the two singles because
// criticalPoints() is computed once and consumed by both searches.
//
// Output slots: crossing occupies slot 0 when present, grazing takes the next free slot.
template <typename ScalarT,
          bool WantCrossing,
          bool WantGrazing,
          RefineMode Refine,
          GrazeMode Graze,
          template <typename T, int32_t D>
          typename JaggedAccessor,
          template <typename T, int32_t D>
          typename TensorAccessor>
__hostdev__ inline void
raySdfPointsCallback(int32_t bidx,
                     int32_t eidx,
                     JaggedAccessor<ScalarT, 2> raysO,
                     JaggedAccessor<ScalarT, 2> raysD,
                     JaggedAccessor<ScalarT, 1> sdfJ,
                     TensorAccessor<ScalarT, 1> tMaxAcc,
                     TensorAccessor<bool, 1> rayMaskAcc,
                     BatchGridAccessor batchAcc,
                     TensorAccessor<ScalarT, 2> outTimes,
                     TensorAccessor<bool, 2> outMask,
                     TensorAccessor<ScalarT, 2> outSdf,
                     TensorAccessor<ScalarT, 3> outGrad,
                     SurfacePointParams<at::opmath_type<ScalarT>> p) {
    using MathType = at::opmath_type<ScalarT>;

    constexpr int kCrossingSlot = WantCrossing ? 0 : -1;
    constexpr int kGrazingSlot  = WantGrazing ? (WantCrossing ? 1 : 0) : -1;

    // Absent optional inputs arrive as empty tensors rather than materialised defaults, so
    // the common path allocates nothing; a zero-length accessor means "not supplied".
    if (rayMaskAcc.size(0) > 0 && !rayMaskAcc[eidx]) {
        return;
    }

    const nanovdb::OnIndexGrid *gpuGrid = batchAcc.grid(bidx);
    auto gridAcc                        = gpuGrid->getAccessor();
    // The grid holding the SDF is sampled with its PRIMAL transform, which puts its voxel
    // centers on integers -- so the unit cubes the HDDA marches are exactly the trilinear
    // interpolation cells. (Marching with the dual transform, as voxels_along_rays does,
    // would step cells offset half a voxel from the cells the interpolant is defined over,
    // and a segment would then straddle two different cubics.)
    const VoxelCoordTransform transform = batchAcc.primalTransform(bidx);
    const nanovdb::CoordBBox bbox       = batchAcc.bbox(bidx);
    const int64_t baseOffset            = batchAcc.voxelOffset(bidx);

    const auto rayO        = raysO.data()[eidx];
    const auto rayD        = raysD.data()[eidx];
    const auto sdf         = sdfJ.data();
    const MathType tMaxRay = (tMaxAcc.size(0) > 0) ? static_cast<MathType>(tMaxAcc[eidx])
                                                   : std::numeric_limits<MathType>::infinity();

    nanovdb::math::Ray<MathType> rayVox = transform.applyToRay(static_cast<MathType>(rayO[0]),
                                                               static_cast<MathType>(rayO[1]),
                                                               static_cast<MathType>(rayO[2]),
                                                               static_cast<MathType>(rayD[0]),
                                                               static_cast<MathType>(rayD[1]),
                                                               static_cast<MathType>(rayD[2]));
    if (!rayVox.clip(bbox)) {
        return;
    }

    // applyToRay scales the direction but leaves the ray parameter alone, so every t below
    // is directly the world-space t the caller passed rays in with.
    const nanovdb::math::Vec3<MathType> oVox = rayVox.eye();
    const nanovdb::math::Vec3<MathType> dVox = rayVox.dir();

    auto evalAt = [&](MathType t, MathType &val, MathType(&grad)[3]) -> bool {
        const nanovdb::math::Vec3<MathType> xyz(
            oVox[0] + t * dVox[0], oVox[1] + t * dVox[1], oVox[2] + t * dVox[2]);
        const auto gradTransform = transform.template applyGrad<MathType>(xyz);

        int64_t indices[8] = {};
        MathType weights[8];
        MathType gradWeights[8][3];
        const uint8_t activeMask = resolveTrilinearStencilWithGrad(
            xyz, gridAcc, baseOffset, indices, weights, gradWeights);
        if (activeMask == 0) {
            return false;
        }

        val     = MathType(0);
        grad[0] = grad[1] = grad[2] = MathType(0);
#pragma unroll
        for (int corner = 0; corner < 8; ++corner) {
            const MathType v = static_cast<MathType>(_loadReadOnly(&sdf[indices[corner]]));
            val += weights[corner] * v;
            grad[0] += gradWeights[corner][0] * v;
            grad[1] += gradWeights[corner][1] * v;
            grad[2] += gradWeights[corner][2] * v;
        }
        grad[0] *= gradTransform[0];
        grad[1] *= gradTransform[1];
        grad[2] *= gradTransform[2];
        return true;
    };

    bool foundCrossing = false;
    MathType tCrossing = MathType(0);

    // Best grazing candidate so far, keyed on |SDF|: the analytic form of "the candidate
    // whose sample sits closest to the surface".
    // GrazeMode::Bisect carries the previous voxel midpoint so consecutive midpoints can be
    // tested for the derivative changing sign, mirroring the torch implementation's pairs.
    bool havePrevMid     = false;
    MathType prevMidT    = MathType(0);
    MathType prevMidDrv  = MathType(0);
    MathType prevMidAbsS = MathType(0);
    MathType braMinT     = MathType(0);
    MathType braMaxT     = MathType(0);
    bool haveBracket     = false;
    // The bracket runs from the previous cell's midpoint to this cell's, so it spans exactly two
    // cells whose cubics are already known. Both are kept with the bracket, each with its entry
    // time (the cubic's tau origin); the second entry time is also the face between them. The
    // bisection reads its slopes from these cubics and never resolves the stencil.
    CellCubic<MathType> prevCubic  = {MathType(0), MathType(0), MathType(0), MathType(0)};
    MathType prevT0                = MathType(0);
    CellCubic<MathType> braCubicLo = prevCubic;
    CellCubic<MathType> braCubicHi = prevCubic;
    MathType braT0Lo               = MathType(0);
    MathType braT0Hi               = MathType(0);

    // The previous cell, for everything that pairs a cell with its neighbour. Its exit t says
    // whether the two are adjacent: across a gap in the band nothing is defined, so no grazing
    // bracket may span one. GrazeMode::Analytic also needs the exit derivative (a minimum on
    // the shared face is recognised from the two one-sided derivatives straddling zero).
    bool havePrevCell    = false;
    MathType prevTExit   = MathType(0);
    MathType prevExitDrv = MathType(0);

    bool foundGrazing   = false;
    MathType tGrazing   = MathType(0);
    MathType bestAbsSdf = std::numeric_limits<MathType>::infinity();

    for (auto it = HDDAActiveValueIterator<decltype(gridAcc), MathType>(rayVox, gridAcc);
         it.isValid();
         ++it) {
        const MathType t0 = it->second.t0;
        const MathType t1 = it->second.t1;
        const MathType dt = t1 - t0;

        if (t0 > tMaxRay) {
            break;
        }
        if (dt < p.eps) {
            continue;
        }

        MathType V[8];
        if (!fetchCellCorners<MathType>(it->first, gridAcc, sdf, baseOffset, V)) {
            continue; // partial cell: outside the band, see fetchCellCorners
        }

        const nanovdb::Coord &base = it->first;
        const MathType u0          = oVox[0] + t0 * dVox[0] - static_cast<MathType>(base[0]);
        const MathType v0          = oVox[1] + t0 * dVox[1] - static_cast<MathType>(base[1]);
        const MathType w0          = oVox[2] + t0 * dVox[2] - static_cast<MathType>(base[2]);
        const CellCubic<MathType> c =
            cubicAlongRay<MathType>(V, u0, v0, w0, dVox[0], dVox[1], dVox[2]);

        // Shared by both searches: grazing candidates, and the monotonic split below.
        MathType crit[2];
        const int nCrit = criticalPoints<MathType>(c, dt, crit);

        const bool contiguous = havePrevCell && nanovdb::math::Abs(t0 - prevTExit) <= p.eps;

        MathType tauCrossing    = dt;
        bool crossingInThisCell = false;

        if constexpr (WantCrossing) {
            // Walk [0, dt] split at the critical points. Each piece is monotonic, so a sign
            // change at its ends means exactly one root, and no root can hide inside. Only the
            // refinement of a bracketed root depends on the mode.
            MathType bounds[4];
            int nb       = 0;
            bounds[nb++] = MathType(0);
            for (int i = 0; i < nCrit; ++i) {
                bounds[nb++] = crit[i];
            }
            bounds[nb++] = dt;

            MathType sLo = c.value(bounds[0]);
            for (int i = 0; i + 1 < nb && !crossingInThisCell; ++i) {
                const MathType hi  = bounds[i + 1];
                const MathType sHi = c.value(hi);
                MathType tau       = MathType(0);
                bool hit           = false;

                if (sLo == MathType(0)) {
                    tau = bounds[i];
                    hit = true;
                } else if ((sLo < MathType(0)) != (sHi < MathType(0))) {
                    // Either direction counts: a ray that starts inside the surface crosses
                    // from negative to positive, and is just as much a first hit.
                    if constexpr (Refine == RefineMode::Newton) {
                        tau = refineRootNewton<MathType>(c, bounds[i], hi, sLo);
                    } else {
                        tau = refineRootBisect<MathType>(c, bounds[i], hi, sLo);
                    }
                    hit = true;
                }

                if (hit && (t0 + tau) >= p.tMin) {
                    tauCrossing        = tau;
                    tCrossing          = t0 + tau;
                    crossingInThisCell = true;
                    foundCrossing      = true;
                }
                sLo = sHi;
            }
        }

        if constexpr (WantGrazing && Graze == GrazeMode::Bisect) {
            // Sample the derivative at this voxel's midpoint, pair it with the previous
            // midpoint, and keep pairs where the ray stops approaching and starts receding.
            // Selection is by the smaller |SDF| of the pair, i.e. the pair whose samples come
            // closest to the surface. The winning bracket is bisected once after the march, on
            // the two cells' cubics, which are saved with it.
            //
            // A minimum sitting exactly on a voxel face is still found: the bracket runs
            // midpoint to midpoint and so straddles the face, never reading the derivative
            // on it. See test_bisect_grazing_finds_axis_aligned_minimum.
            const MathType tauMid = MathType(0.5) * dt;
            const MathType midT   = t0 + tauMid;
            const MathType midDrv = c.deriv(tauMid);
            const MathType midAbs = nanovdb::math::Abs(c.value(tauMid));

            // Adjacent voxels only: across a gap neither cubic describes the field between the
            // two midpoints, and the final evaluation there reads missing corners as zero.
            if (havePrevMid && contiguous && prevMidDrv <= MathType(0) && midDrv > MathType(0)) {
                const MathType key = (prevMidAbsS < midAbs) ? prevMidAbsS : midAbs;
                if (prevMidT >= p.grazeTMin && midT <= tMaxRay && key < bestAbsSdf &&
                    !(crossingInThisCell && tauMid >= tauCrossing)) {
                    bestAbsSdf   = key;
                    braMinT      = prevMidT;
                    braMaxT      = midT;
                    braCubicLo   = prevCubic;
                    braT0Lo      = prevT0;
                    braCubicHi   = c;
                    braT0Hi      = t0;
                    haveBracket  = true;
                    foundGrazing = true;
                }
            }
            havePrevMid = true;
            prevMidT    = midT;
            prevMidDrv  = midDrv;
            prevMidAbsS = midAbs;
            prevCubic   = c;
            prevT0      = t0;
        }

        if constexpr (WantGrazing && Graze == GrazeMode::Analytic) {
            // A grazing point is a LOCAL MINIMUM of the field along the ray -- the ray stops
            // approaching the surface and starts receding. Candidates must satisfy that, not
            // merely have a small |SDF|.
            //
            // An earlier version minimised s over the closed interval [0, dt] and relied on a
            // tangency test at the end. That over-accepts badly (measured: 32 rays against
            // bisect's 14 on a 256^2 sphere render, inflating the boundary gradient 2.2x),
            // because a cell endpoint lies exactly ON a voxel face, where the trilinear field
            // is C0 but not C1: one of the two one-sided gradients there can look tangent while
            // the ray passes straight through the face with no minimum. Small |s| on a face is
            // not evidence of anything; only a minimum is.
            //
            // The two candidate kinds are therefore tested differently:
            //
            //   interior critical point   deriv == 0 by construction; it is a minimum rather
            //                             than a maximum exactly when deriv2 > 0.
            //   cell entry (tau == 0)     the derivative is discontinuous here, so the test is
            //                             whether the two ONE-SIDED derivatives straddle zero:
            //                             the previous cell exits with deriv <= 0 and this one
            //                             enters with deriv >= 0.
            //
            // Keeping the face case is what handles an axis-aligned ray against a
            // lattice-aligned surface, where symmetry puts the minimum exactly on a face and
            // both one-sided derivatives are exactly zero -- admitted by <= / >=, while a
            // strict sign change would miss it. That is the case this mode exists for.
            //
            // tau == dt is not a candidate: it is the same point as the next cell's tau == 0
            // and is handled there, with the neighbour's derivative available. Carrying the
            // exit derivative forward costs the contiguity bookkeeping the closed-interval
            // form avoided, but a face shared with a cell we never visited is one where the
            // field is only defined on one side, and that is not a minimum we can verify.
            //
            // Before the crossing the field has not changed sign, so among admitted minima
            // "smallest s" and "smallest |s|" agree -- the rule the torch path selects by.
            const MathType entryDrv = c.deriv(MathType(0));

            for (int i = 0; i <= nCrit; ++i) {
                MathType tau;
                const bool onFace = (i == 0);
                if (onFace) {
                    if (!(contiguous && prevExitDrv <= MathType(0) && entryDrv >= MathType(0))) {
                        continue; // cell entry, but not a minimum across the face
                    }
                    tau = MathType(0);
                } else {
                    tau = crit[i - 1];
                    if (!(c.deriv2(tau) > MathType(0))) {
                        continue; // a maximum or an inflection, not a minimum
                    }
                }

                // A candidate behind this cell's own crossing is behind the surface.
                if (crossingInThisCell && tau >= tauCrossing) {
                    continue;
                }
                const MathType t = t0 + tau;
                if (t < p.grazeTMin || t > tMaxRay) {
                    continue;
                }
                const MathType absS = nanovdb::math::Abs(c.value(tau));
                if (absS < bestAbsSdf) {
                    bestAbsSdf   = absS;
                    tGrazing     = t;
                    foundGrazing = true;
                }
            }
        }

        havePrevCell = true;
        prevTExit    = t1;
        prevExitDrv  = c.deriv(dt);

        if (crossingInThisCell) {
            // Everything collected so far is in front of the surface, because cells are
            // visited in ray order. That is what lets the fused op skip the separate tMax
            // pass the torch implementation needs.
            break;
        }
    }

    // Evaluate the interpolant and its spatial gradient once at an accepted point. This is
    // the only place the full 3D gradient is needed; the march itself only ever works with
    // the along-ray derivative, which comes from the cubic for free.

    auto writeSlot = [&](int slot, MathType t, MathType val, const MathType(&grad)[3]) {
        outTimes[eidx][slot]   = static_cast<ScalarT>(t);
        outSdf[eidx][slot]     = static_cast<ScalarT>(val);
        outGrad[eidx][slot][0] = static_cast<ScalarT>(grad[0]);
        outGrad[eidx][slot][1] = static_cast<ScalarT>(grad[1]);
        outGrad[eidx][slot][2] = static_cast<ScalarT>(grad[2]);
        outMask[eidx][slot]    = true;
    };

    if constexpr (WantGrazing && Graze == GrazeMode::Bisect) {
        // Bisect the winning midpoint-to-midpoint bracket on the sign of the along-ray slope, 8
        // steps. The bracket straddles the face between two cells, and on either side the slope
        // is that cell's cubic differentiated -- the same number as the stencil gradient dotted
        // with the ray, without resolving the stencil. A probe landing exactly on the face is
        // assigned to the second cell, so which side it reads is fixed rather than left to
        // rounding (resolving the stencil there made CPU and GPU disagree).
        if (haveBracket) {
            auto slopeAt = [&](MathType t) -> MathType {
                return (t < braT0Hi) ? braCubicLo.deriv(t - braT0Lo)
                                     : braCubicHi.deriv(t - braT0Hi);
            };
            MathType lo = braMinT, hi = braMaxT;
            for (int i = 0; i < 8; ++i) {
                const MathType mid = MathType(0.5) * (lo + hi);
                if (slopeAt(mid) < MathType(0)) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            // The bracket keeps slope <= 0 at lo and >= 0 at hi, so it always holds a local
            // minimum (an interior one with zero slope, or a crease on the face).
            tGrazing = MathType(0.5) * (lo + hi);
        }
    }

    if constexpr (WantCrossing) {
        if (foundCrossing) {
            MathType val, grad[3];
            if (evalAt(tCrossing, val, grad)) {
                writeSlot(kCrossingSlot, tCrossing, val, grad);
            }
        }
    }

    if constexpr (WantGrazing) {
        if (foundGrazing) {
            MathType val, grad[3];
            if (!evalAt(tGrazing, val, grad)) {
                return;
            }

            // itxEps rather than 0: SDF == 0 IS the surface, so a point there is a tangential
            // hit that the interior term already accounts for. Testing against a bare 0 sits
            // exactly on that knife edge and lets rounding flip points between the interior
            // and boundary sets from iteration to iteration.
            if (!(val > p.itxEps && val < p.relaxationEps)) {
                return;
            }

            // No slope (tangency) test. Both modes only produce local minima of the SDF along
            // the ray, and the relaxed boundary term counts the rays whose minimum lies in
            // (itxEps, relaxationEps): as the field changes, the ray starts or stops hitting the
            // surface when that minimum crosses zero, whether the minimum is smooth or sits on
            // a crease of the trilinear field at a cell face (C0 but not C1 there, so creases
            // are generic). A threshold on the averaged one-sided slopes rejected crease minima
            // with lopsided slopes -- measured against the relaxed-boundary reference
            // implementation, whose grazing points lie ~75% on cell faces, we accepted about
            // half as many points and its boundary gradient was ~1.8x ours.
            writeSlot(kGrazingSlot, tGrazing, val, grad);
        }
    }
}

// Validate the shared arguments of all three entry points.
inline void
checkCommonArgs(const GridBatchData &batchHdl,
                const JaggedTensor &rayOrigins,
                const JaggedTensor &rayDirections,
                const JaggedTensor &sdf) {
    batchHdl.checkDevice(rayOrigins);
    batchHdl.checkDevice(rayDirections);
    batchHdl.checkDevice(sdf);

    TORCH_CHECK_TYPE(rayOrigins.is_floating_point(), "ray_origins must have a floating point type");
    TORCH_CHECK_TYPE(rayDirections.is_floating_point(),
                     "ray_directions must have a floating point type");
    TORCH_CHECK_TYPE(sdf.is_floating_point(), "sdf must have a floating point type");
    TORCH_CHECK_TYPE(rayOrigins.dtype() == rayDirections.dtype(),
                     "ray_origins and ray_directions must have the same type");
    TORCH_CHECK_TYPE(rayDirections.dtype() == sdf.dtype(),
                     "ray_directions and sdf must have the same type");

    TORCH_CHECK(rayOrigins.rdim() == 2,
                "Expected ray_origins to have 2 dimensions (shape (n, 3)) but got ",
                rayOrigins.rdim(),
                " dimensions");
    TORCH_CHECK(rayDirections.rdim() == 2,
                "Expected ray_directions to have 2 dimensions (shape (n, 3)) but got ",
                rayDirections.rdim(),
                " dimensions");
    TORCH_CHECK(rayOrigins.rsize(0) == rayDirections.rsize(0),
                "Expected ray_origins and ray_directions to have the same number of rays");
    TORCH_CHECK(rayOrigins.rsize(1) == 3, "Expected ray_origins to have shape (n, 3)");
    TORCH_CHECK(rayDirections.rsize(1) == 3, "Expected ray_directions to have shape (n, 3)");

    TORCH_CHECK(sdf.rdim() == 1,
                "Expected sdf to have 1 dimension (shape (num_voxels,)) but got ",
                sdf.rdim(),
                " dimensions");
    TORCH_CHECK(sdf.rsize(0) == batchHdl.totalVoxels(),
                "ray_sdf_* interpolates the SDF trilinearly and needs exactly one value per "
                "active voxel of the grid it is stored on, but got ",
                sdf.rsize(0),
                " values for ",
                batchHdl.totalVoxels(),
                " voxels. Note this grid is the DUAL of the geometry grid: pass "
                "grid.dual_grid() and its per-corner values.");
}

template <torch::DeviceType DeviceTag,
          bool WantCrossing,
          bool WantGrazing,
          RefineMode Refine,
          GrazeMode Graze>
RaySdfPointsResult
launch(const GridBatchData &batchHdl,
       const JaggedTensor &rayOrigins,
       const JaggedTensor &rayDirections,
       const JaggedTensor &sdf,
       const std::optional<JaggedTensor> &tMax,
       const std::optional<JaggedTensor> &rayMask,
       double tMin,
       double grazeTMin,
       double relaxationEps,
       double itxEps,
       double eps) {
    checkCommonArgs(batchHdl, rayOrigins, rayDirections, sdf);
    TORCH_CHECK_VALUE(eps >= 0.0, "eps must be positive or zero");

    constexpr int64_t kSlots = (WantCrossing ? 1 : 0) + (WantGrazing ? 1 : 0);
    const int64_t numRays    = rayOrigins.rsize(0);

    auto optsF = torch::TensorOptions().dtype(rayOrigins.dtype()).device(rayOrigins.device());
    auto optsB = torch::TensorOptions().dtype(torch::kBool).device(rayOrigins.device());

    // Zero rather than empty: rows whose mask is false are documented as zeroed, and the
    // kernel only writes the slots it accepts.
    torch::Tensor outTimes = torch::zeros({numRays, kSlots}, optsF);
    torch::Tensor outMask  = torch::zeros({numRays, kSlots}, optsB);
    torch::Tensor outSdf   = torch::zeros({numRays, kSlots}, optsF);
    torch::Tensor outGrad  = torch::zeros({numRays, kSlots, 3}, optsF);

    // Materialise the optional per-ray inputs rather than plumbing optional accessors into
    // the device code: two small allocations, and the kernel stays branch-free on them.
    torch::Tensor tMaxT = tMax.has_value() ? tMax.value().jdata() : torch::empty({0}, optsF);
    torch::Tensor rayMaskT =
        rayMask.has_value() ? rayMask.value().jdata() : torch::empty({0}, optsB);

    // Only validate what was actually supplied: an absent optional is a zero-length tensor.
    TORCH_CHECK(!tMax.has_value() || (tMaxT.dim() == 1 && tMaxT.size(0) == numRays),
                "t_max must have shape (n,) matching the number of rays");
    TORCH_CHECK(!rayMask.has_value() || (rayMaskT.dim() == 1 && rayMaskT.size(0) == numRays),
                "ray_mask must have shape (n,) matching the number of rays");

    AT_DISPATCH_V2(rayOrigins.scalar_type(),
                   "RaySdfPoints",
                   AT_WRAP([&]() {
                       using MathType = at::opmath_type<scalar_t>;
                       SurfacePointParams<MathType> params{
                           static_cast<MathType>(tMin),
                           static_cast<MathType>(grazeTMin),
                           static_cast<MathType>(relaxationEps),
                           static_cast<MathType>(itxEps),
                           static_cast<MathType>(eps),
                       };

                       auto batchAcc    = gridBatchAccessor<DeviceTag>(batchHdl);
                       auto rayDAcc     = jaggedAccessor<DeviceTag, scalar_t, 2>(rayDirections);
                       auto sdfAcc      = jaggedAccessor<DeviceTag, scalar_t, 1>(sdf);
                       auto tMaxAcc     = tensorAccessor<DeviceTag, scalar_t, 1>(tMaxT);
                       auto rayMaskAcc  = tensorAccessor<DeviceTag, bool, 1>(rayMaskT);
                       auto outTimesAcc = tensorAccessor<DeviceTag, scalar_t, 2>(outTimes);
                       auto outMaskAcc  = tensorAccessor<DeviceTag, bool, 2>(outMask);
                       auto outSdfAcc   = tensorAccessor<DeviceTag, scalar_t, 2>(outSdf);
                       auto outGradAcc  = tensorAccessor<DeviceTag, scalar_t, 3>(outGrad);

                       if constexpr (DeviceTag == torch::kCUDA) {
                           auto cb = [=] __device__(int32_t bidx,
                                                    int32_t eidx,
                                                    int32_t cidx,
                                                    JaggedRAcc64<scalar_t, 2> rOA) {
                               raySdfPointsCallback<scalar_t,
                                                    WantCrossing,
                                                    WantGrazing,
                                                    Refine,
                                                    Graze,
                                                    JaggedRAcc64,
                                                    TorchRAcc64>(bidx,
                                                                 eidx,
                                                                 rOA,
                                                                 rayDAcc,
                                                                 sdfAcc,
                                                                 tMaxAcc,
                                                                 rayMaskAcc,
                                                                 batchAcc,
                                                                 outTimesAcc,
                                                                 outMaskAcc,
                                                                 outSdfAcc,
                                                                 outGradAcc,
                                                                 params);
                           };
                           forEachJaggedElementChannelCUDA<scalar_t, 2>(1, rayOrigins, cb);
                       } else {
                           auto cb = [=](int32_t bidx,
                                         int32_t eidx,
                                         int32_t cidx,
                                         JaggedAcc<scalar_t, 2> rOA) {
                               raySdfPointsCallback<scalar_t,
                                                    WantCrossing,
                                                    WantGrazing,
                                                    Refine,
                                                    Graze,
                                                    JaggedAcc,
                                                    TorchAcc>(bidx,
                                                              eidx,
                                                              rOA,
                                                              rayDAcc,
                                                              sdfAcc,
                                                              tMaxAcc,
                                                              rayMaskAcc,
                                                              batchAcc,
                                                              outTimesAcc,
                                                              outMaskAcc,
                                                              outSdfAcc,
                                                              outGradAcc,
                                                              params);
                           };
                           forEachJaggedElementChannelCPU<scalar_t, 2>(1, rayOrigins, cb);
                       }
                   }),
                   AT_EXPAND(AT_FLOATING_TYPES));

    return RaySdfPointsResult{rayOrigins.jagged_like(outTimes),
                              rayOrigins.jagged_like(outMask),
                              rayOrigins.jagged_like(outSdf),
                              rayOrigins.jagged_like(outGrad)};
}

// Map the caller's runtime mode selection onto the compiled instantiations. Every combination
// is built into the library, so switching method is a call-site argument, not a rebuild.
template <torch::DeviceType DeviceTag, bool WantCrossing, bool WantGrazing, typename... Args>
RaySdfPointsResult
launchWithModes(int refine, int graze, Args &&...args) {
    TORCH_CHECK_VALUE(refine == 0 || refine == 1, "refine must be 0 (newton) or 1 (bisect)");
    TORCH_CHECK_VALUE(graze == 0 || graze == 1, "graze must be 0 (analytic) or 1 (bisect)");
    const bool rb = (refine == 1);
    const bool gb = (graze == 1);
    if (!rb && !gb) {
        return launch<DeviceTag,
                      WantCrossing,
                      WantGrazing,
                      RefineMode::Newton,
                      GrazeMode::Analytic>(std::forward<Args>(args)...);
    } else if (!rb && gb) {
        return launch<DeviceTag, WantCrossing, WantGrazing, RefineMode::Newton, GrazeMode::Bisect>(
            std::forward<Args>(args)...);
    } else if (rb && !gb) {
        return launch<DeviceTag,
                      WantCrossing,
                      WantGrazing,
                      RefineMode::Bisect,
                      GrazeMode::Analytic>(std::forward<Args>(args)...);
    }
    return launch<DeviceTag, WantCrossing, WantGrazing, RefineMode::Bisect, GrazeMode::Bisect>(
        std::forward<Args>(args)...);
}

inline void
checkListDims(const JaggedTensor &rayOrigins,
              const JaggedTensor &rayDirections,
              const JaggedTensor &sdf) {
    TORCH_CHECK_VALUE(rayOrigins.ldim() == 1,
                      "Expected ray_origins to have 1 list dimension, but got ",
                      rayOrigins.ldim());
    TORCH_CHECK_VALUE(rayDirections.ldim() == 1,
                      "Expected ray_directions to have 1 list dimension, but got ",
                      rayDirections.ldim());
    TORCH_CHECK_VALUE(
        sdf.ldim() == 1, "Expected sdf to have 1 list dimension, but got ", sdf.ldim());
}

} // namespace raysdf
} // namespace ops
} // namespace detail
} // namespace fvdb

#endif // FVDB_DETAIL_OPS_RAYSDFCOMMON_CUH
