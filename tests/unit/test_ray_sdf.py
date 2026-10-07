# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import unittest

import torch

import fvdb


def _dense_cube_grid(vx: float, half: int, device: torch.device) -> "fvdb.Grid":
    """A dense (2*half+1)^3 cube grid, one voxel per lattice site."""
    rng = torch.arange(-half, half + 1, device=device, dtype=torch.float32)
    ii, jj, kk = torch.meshgrid(rng, rng, rng, indexing="ij")
    ijk = torch.stack([ii, jj, kk], dim=-1).reshape(-1, 3)
    return fvdb.Grid.from_points(ijk * vx, voxel_size=vx)


class RaySdfTestCase(unittest.TestCase):
    """Shared fixture: a dense cube and its dual, which is where the SDF lives."""

    def setUp(self):
        torch.manual_seed(0)
        # CUDA when available: the kernel shares one __hostdev__ body across both, but the
        # launch and accessor plumbing differ, so the GPU path needs exercising too.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.vx = 0.05
        self.grid = _dense_cube_grid(self.vx, half=12, device=self.device)
        self.dual = self.grid.dual_grid()
        # World-space position of every SDF sample.
        self.corners = self.dual.voxel_to_world(self.dual.ijk.float()).float()

    def plane_sdf(self, normal: torch.Tensor, offset: float) -> torch.Tensor:
        """A linear field, which trilinear interpolation reproduces *exactly*."""
        n = normal / normal.norm()
        return self.corners @ n - offset

    def sphere_sdf(self, radius: float, center=(0.0, 0.0, 0.0)) -> torch.Tensor:
        c = torch.tensor(center, device=self.device, dtype=torch.float32)
        return (self.corners - c).norm(dim=1) - radius


class RaySdfIntersectionTests(RaySdfTestCase):
    def test_plane_crossing_is_exact(self):
        """On a linear field the interpolant is exact, so the hit must be too.

        This is the tightest correctness check available: trilinear interpolation reproduces
        a linear function with no error, so the cubic solved per cell has the analytic root
        and any discrepancy is the kernel's own, not a discretization artifact.
        """
        normal = torch.tensor([0.3, -0.7, 0.55], device=self.device)
        n = normal / normal.norm()
        offset = 0.11
        sdf = self.plane_sdf(normal, offset)

        # Rays from the negative side, aimed at the plane from assorted directions.
        origins = torch.tensor(
            [
                [-0.4, 0.0, 0.0],
                [0.0, 0.4, 0.0],
                [0.1, -0.2, -0.35],
                [-0.25, 0.3, 0.15],
            ],
            device=self.device,
        )
        directions = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.2, -1.0, 0.1],
                [0.1, 0.3, 1.0],
                [0.6, -0.4, 0.2],
            ],
            device=self.device,
        )
        directions = directions / directions.norm(dim=1, keepdim=True)

        # Newton, not the bisect default: this checks the exact solver, and 8 halvings stop at
        # ~1/256 of a cell (bisect's own accuracy is test_bisect_refine_agrees_with_newton).
        hit = self.dual.ray_sdf_intersection(sdf, origins, directions, refine="newton")

        denom = directions @ n
        t_analytic = (offset - origins @ n) / denom

        self.assertTrue(bool(hit.mask.all()), "every ray should cross the plane")
        torch.testing.assert_close(hit.t, t_analytic, atol=1e-4, rtol=0)
        # The SDF at the hit is zero by definition of the crossing.
        torch.testing.assert_close(hit.sdf, torch.zeros_like(hit.sdf), atol=1e-5, rtol=0)
        # ...and the gradient of a linear field is its normal everywhere.
        expected_grad = n.expand_as(hit.grad)
        torch.testing.assert_close(hit.grad, expected_grad, atol=1e-3, rtol=0)

    def test_position_matches_t(self):
        sdf = self.plane_sdf(torch.tensor([0.0, 0.0, 1.0], device=self.device), 0.1)
        o = torch.tensor([[0.0, 0.0, -0.3]], device=self.device)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)
        hit = self.dual.ray_sdf_intersection(sdf, o, d)
        torch.testing.assert_close(hit.position, o + hit.t.unsqueeze(-1) * d)

    def test_rays_that_miss(self):
        """A ray pointing away from the surface reports no hit and zeroed outputs."""
        sdf = self.sphere_sdf(0.3)
        o = torch.tensor([[0.0, 0.0, 0.45]], device=self.device)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)  # outward
        hit = self.dual.ray_sdf_intersection(sdf, o, d)
        self.assertFalse(bool(hit.mask.any()))
        self.assertEqual(float(hit.t.abs().max()), 0.0)
        self.assertEqual(float(hit.sdf.abs().max()), 0.0)

    def test_ray_starting_inside_reports_exit(self):
        """A sign change in either direction counts, so an inside ray finds its exit."""
        radius = 0.3
        sdf = self.sphere_sdf(radius)
        o = torch.tensor([[0.0, 0.0, 0.0]], device=self.device)  # sphere centre
        d = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)
        hit = self.dual.ray_sdf_intersection(sdf, o, d)
        self.assertTrue(bool(hit.mask.all()))
        # Tolerance is set by discretization, not by the solver: trilinear interpolation of a
        # sphere's SDF puts the interpolated zero slightly inside the true one, by O(vx^2).
        # test_matches_independent_bisection pins the solver itself, to 1e-6.
        self.assertAlmostEqual(float(hit.t[0]), radius, delta=5e-3)

    def test_matches_independent_bisection(self):
        """The reported root must be the root of the field fvdb itself interpolates.

        Bisects :meth:`sample_trilinear` along the ray, which shares no code with the kernel,
        so this pins the solver against the interpolant rather than against an analytic shape
        the interpolant only approximates.
        """
        sdf = self.sphere_sdf(0.3)
        o = torch.tensor([0.0, 0.0, 0.0], device=self.device)
        d = torch.tensor([0.0, 0.0, 1.0], device=self.device)

        def field_at(t: float) -> float:
            p = (o + t * d).unsqueeze(0)
            return float(self.dual.sample_trilinear(p, sdf.unsqueeze(-1))[0, 0])

        lo, hi = 0.25, 0.35
        self.assertLess(field_at(lo), 0.0)
        self.assertGreater(field_at(hi), 0.0)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if field_at(mid) < 0.0:
                lo = mid
            else:
                hi = mid
        reference = 0.5 * (lo + hi)

        # Newton, not the bisect default: this checks the exact solver, and 8 halvings stop at
        # ~1/256 of a cell (bisect's own accuracy is test_bisect_refine_agrees_with_newton).
        hit = self.dual.ray_sdf_intersection(sdf, o.unsqueeze(0), d.unsqueeze(0), refine="newton")
        self.assertAlmostEqual(float(hit.t[0]), reference, delta=1e-6)

    def test_two_roots_inside_one_cell(self):
        """A feature thinner than a voxel is found by both refine modes.

        A sphere of radius < vx/2 centred on a lattice corner makes the SDF along a ray dip
        negative and come back inside one voxel, so any scheme that only tests one sample per
        voxel sees no sign change. Both modes here bracket from the
        cell's cubic, split at its turning points, so neither can miss it; they differ only in
        how the bracket is refined (Newton vs 8 halvings).
        """
        radius = 0.4 * self.vx
        center = (0.5 * self.vx, 0.5 * self.vx, 0.5 * self.vx)  # a lattice corner
        sdf = self.sphere_sdf(radius, center=center)

        c = torch.tensor(center, device=self.device)
        o = (c - torch.tensor([0.0, 0.0, 0.5], device=self.device)).unsqueeze(0)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=self.device)

        for refine, delta in (("newton", 1e-5), ("bisect", self.vx / 256.0)):
            hit = self.dual.ray_sdf_intersection(sdf, o, d, refine=refine)
            self.assertTrue(bool(hit.mask.all()), f"sub-cell feature was missed by {refine}")
            # The near side of the little sphere, i.e. 0.5 - radius along the ray.
            self.assertAlmostEqual(float(hit.t[0]), 0.5 - radius, delta=delta, msg=refine)


class RaySdfSampledReferenceTests(unittest.TestCase):
    """The bisect modes against an independent re-implementation of a per-voxel sampling search.

    That search takes one SDF sample per voxel, brackets a crossing between consecutive samples
    of opposite sign and a grazing point between adjacent midpoints where the derivative changes
    sign, and bisects 8 times. The kernel's bisect modes follow it with three deliberate
    differences, each matching the relaxed-boundary reference implementation instead: crossings
    are bracketed from each cell's cubic (so features thinner than a voxel are seen; on this
    smooth shell the two agree to bisection precision), the root rather than the bracket is
    tested against t_min, and the grazing point has no tangency test (the bracket always holds a
    minimum along the ray; only its SDF value is checked). The reference below is written against fvdb's own voxels_along_rays and sample_trilinear, not
    against the kernel, and runs on a pruned spherical shell so rays leave and re-enter the band.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.vx, half, self.radius = 0.05, 12, 0.3
        ijk = torch.stack(torch.meshgrid(*[torch.arange(-half, half + 1)] * 3, indexing="ij"), -1).reshape(-1, 3)
        dense = fvdb.Grid.from_ijk(ijk.int().to(self.device), voxel_size=self.vx, origin=0.0)
        centres = dense.voxel_to_world(dense.ijk.float())
        shell = (centres.norm(dim=1) - self.radius).abs() < 2.0 * self.vx
        self.grid = fvdb.Grid.from_ijk(dense.ijk[shell], voxel_size=self.vx, origin=0.0)
        self.dual = self.grid.dual_grid()
        corners = self.dual.voxel_to_world(self.dual.ijk.float()).float()
        self.sdf = (corners.norm(dim=1) - self.radius).contiguous()
        # A fan through and past the shell: head-on hits, grazes, and rays that cross the
        # hollow interior (a gap in the band) before hitting the far side.
        ys = torch.linspace(-0.36, 0.36, 97, device=self.device)
        self.o = torch.stack([torch.full_like(ys, -0.7), ys, 0.37 * ys + 0.011], dim=1).contiguous()
        d = torch.zeros_like(self.o)
        d[:, 0], d[:, 2] = 1.0, 0.013
        self.d = (d / d.norm(dim=1, keepdim=True)).contiguous()

    def _runs(self, i):
        """Per ray: runs of adjacent voxels, each a list of (t_enter, t_exit)."""
        vox, times = self.grid.voxels_along_rays(
            self.o[i : i + 1], self.d[i : i + 1], max_voxels=100000, eps=1e-4, return_ijk=False
        )
        runs = []
        for t0, t1 in times.jdata.tolist():
            if runs and abs(t0 - runs[-1][-1][1]) <= 1e-4:
                runs[-1].append((t0, t1))
            else:
                runs.append([(t0, t1)])
        return runs

    def _val(self, i, t):
        p = self.o[i] + t * self.d[i]
        return float(self.dual.sample_trilinear(p[None], self.sdf[:, None])[0, 0])

    def _drv(self, i, t):
        p = self.o[i] + t * self.d[i]
        _, g = self.dual.sample_trilinear_with_grad(p[None], self.sdf[:, None])
        return float((g[0, 0] * self.d[i]).sum())

    def _ref_crossing(self, i, t_min):
        for run in self._runs(i):
            samples = [(run[0][0], self._val(i, run[0][0]))]
            samples += [(0.5 * (a + b), self._val(i, 0.5 * (a + b))) for a, b in run]
            samples.append((run[-1][1], self._val(i, run[-1][1])))
            for (lo, s_lo), (hi, s_hi) in zip(samples, samples[1:]):
                if not (s_lo * s_hi < 0 or s_lo == 0):
                    continue
                for _ in range(8):
                    mid = 0.5 * (lo + hi)
                    s_mid = self._val(i, mid)
                    if s_mid * s_lo > 0:
                        lo, s_lo = mid, s_mid
                    else:
                        hi = mid
                t = 0.5 * (lo + hi)
                if t >= t_min:
                    return t
        return None

    def _ref_grazing(self, i, eps, graze_t_min, itx_eps):
        best, bracket = float("inf"), None
        for run in self._runs(i):
            mids = [0.5 * (a + b) for a, b in run]
            for lo, hi in zip(mids, mids[1:]):  # adjacent voxels only
                if not (self._drv(i, lo) <= 0 < self._drv(i, hi)) or lo < graze_t_min:
                    continue
                key = min(abs(self._val(i, lo)), abs(self._val(i, hi)))
                if key < best:
                    best, bracket = key, (lo, hi)
        if bracket is None:
            return None
        lo, hi = bracket
        for _ in range(8):
            mid = 0.5 * (lo + hi)
            if self._drv(i, mid) < 0:
                lo = mid
            else:
                hi = mid
        t = 0.5 * (lo + hi)
        p = self.o[i] + t * self.d[i]
        s = float(self.dual.sample_trilinear(p[None], self.sdf[:, None])[0, 0])
        return t if itx_eps < s < eps else None

    def test_bisect_crossing_matches_reference(self):
        """Same hits as the reference on a smooth shell; t agrees to bisection precision.

        The brackets differ (cubic pieces here, midpoint pairs there), so the two 8-step
        bisections converge to different points within 1/256 of their own bracket -- at most
        a voxel each -- hence the tolerance.
        """
        hit = self.dual.ray_sdf_intersection(self.sdf, self.o, self.d, t_min=1e-4, refine="bisect")
        n_hit = 0
        for i in range(self.o.shape[0]):
            ref = self._ref_crossing(i, 1e-4)
            self.assertEqual(bool(hit.mask[i]), ref is not None, f"ray {i}")
            if ref is not None:
                n_hit += 1
                self.assertAlmostEqual(float(hit.t[i]), ref, delta=self.vx / 128.0, msg=f"ray {i}")
        self.assertGreater(n_hit, 20, "fixture should produce hits")

    def test_bisect_grazing_matches_reference(self):
        kw = dict(relaxation_eps=0.02, graze_t_min=1e-4, itx_eps=1e-7)
        gr = self.dual.ray_sdf_grazing(self.sdf, self.o, self.d, graze="bisect", **kw)
        n_graze = 0
        for i in range(self.o.shape[0]):
            ref = self._ref_grazing(i, kw["relaxation_eps"], kw["graze_t_min"], kw["itx_eps"])
            self.assertEqual(bool(gr.mask[i]), ref is not None, f"ray {i}")
            if ref is not None:
                n_graze += 1
                self.assertAlmostEqual(float(gr.t[i]), ref, delta=1e-5, msg=f"ray {i}")
        self.assertGreater(n_graze, 3, "fixture should produce grazing points")

    def test_combined_matches_separate_on_the_shell(self):
        """Combined and separate agree bit for bit in bisect/bisect, gaps in the band included."""
        kw = dict(relaxation_eps=0.02, itx_eps=1e-7)
        both = self.dual.ray_sdf_intersection_with_grazing(
            self.sdf, self.o, self.d, t_min=1e-4, graze_t_min=1e-4, refine="bisect", graze="bisect", **kw
        )
        cross = self.dual.ray_sdf_intersection(self.sdf, self.o, self.d, t_min=1e-4, refine="bisect")
        t_max = torch.where(cross.mask, cross.t, torch.full_like(cross.t, float("inf")))
        graze = self.dual.ray_sdf_grazing(self.sdf, self.o, self.d, t_max=t_max, graze_t_min=1e-4, graze="bisect", **kw)
        for a, b in ((both.crossing, cross), (both.grazing, graze)):
            torch.testing.assert_close(a.mask, b.mask, rtol=0, atol=0)
            torch.testing.assert_close(a.t[a.mask], b.t[b.mask], rtol=0, atol=0)


class RaySdfGrazingTests(RaySdfTestCase):
    def test_grazing_on_sphere_is_analytic(self):
        """For a ray passing a sphere at impact parameter b, the SDF minimum is b - R.

        The minimum sits at the closest approach, whose position and value are both known in
        closed form, so this pins the grazing search against an independent answer rather
        than against the implementation's own output.
        """
        radius = 0.3
        band = 0.04
        b = radius + 0.4 * band  # inside the relaxation band, so it must be accepted
        sdf = self.sphere_sdf(radius)

        o = torch.tensor([[-0.6, b, 0.0]], device=self.device)
        d = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)

        graze = self.dual.ray_sdf_grazing(sdf, o, d, relaxation_eps=band)

        self.assertTrue(bool(graze.mask.all()), "grazing point should be found")
        # Closest approach is directly abeam the centre.
        self.assertAlmostEqual(float(graze.t[0]), 0.6, delta=2.0 * self.vx)
        self.assertAlmostEqual(float(graze.sdf[0]), b - radius, delta=0.01)
        # And it must be tangent: the gradient is perpendicular to the ray there.
        g = graze.grad[0] / graze.grad[0].norm()
        self.assertLess(abs(float(g @ d[0])), 1e-1)

    def test_grazing_rejected_outside_band(self):
        """A ray passing well outside the band has a minimum, but not an accepted one."""
        radius = 0.3
        sdf = self.sphere_sdf(radius)
        o = torch.tensor([[-0.6, radius + 0.15, 0.0]], device=self.device)
        d = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)
        graze = self.dual.ray_sdf_grazing(sdf, o, d, relaxation_eps=0.04)
        self.assertFalse(bool(graze.mask.any()))

    def test_grazing_rejected_on_surface(self):
        """A ray straight through the centre has its minimum *inside*, not in the band."""
        radius = 0.3
        sdf = self.sphere_sdf(radius)
        o = torch.tensor([[-0.6, 0.0, 0.0]], device=self.device)
        d = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)
        graze = self.dual.ray_sdf_grazing(sdf, o, d, relaxation_eps=0.04)
        self.assertFalse(bool(graze.mask.any()))

    def test_ray_mask_disables_rays(self):
        radius = 0.3
        band = 0.04
        b = radius + 0.4 * band
        sdf = self.sphere_sdf(radius)
        o = torch.tensor([[-0.6, b, 0.0], [-0.6, b, 0.0]], device=self.device)
        d = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], device=self.device)
        mask = torch.tensor([True, False], device=self.device)
        graze = self.dual.ray_sdf_grazing(sdf, o, d, relaxation_eps=band, ray_mask=mask)
        self.assertTrue(bool(graze.mask[0]))
        self.assertFalse(bool(graze.mask[1]))


class RaySdfFusedTests(RaySdfTestCase):
    def test_fused_matches_separate_ops(self):
        """The fused op must agree with the two single-point ops it subsumes."""
        radius = 0.3
        band = 0.04
        sdf = self.sphere_sdf(radius)

        # A spread of impact parameters: some hit, some graze, some miss entirely.
        ys = torch.linspace(0.0, radius + 0.1, 24, device=self.device)
        o = torch.stack([torch.full_like(ys, -0.6), ys, torch.zeros_like(ys)], dim=1)
        d = torch.zeros_like(o)
        d[:, 0] = 1.0

        fused = self.dual.ray_sdf_intersection_with_grazing(sdf, o, d, relaxation_eps=band)
        hit = self.dual.ray_sdf_intersection(sdf, o, d)

        torch.testing.assert_close(fused.crossing.mask, hit.mask)
        torch.testing.assert_close(fused.crossing.t, hit.t, atol=1e-5, rtol=0)

        # The grazing search is bounded by the crossing, which is what the separate op needs
        # t_max spelled out for.
        t_max = torch.where(hit.mask, hit.t, torch.full_like(hit.t, float("inf")))
        graze = self.dual.ray_sdf_grazing(sdf, o, d, relaxation_eps=band, t_max=t_max)
        torch.testing.assert_close(fused.grazing.mask, graze.mask)
        torch.testing.assert_close(fused.grazing.t, graze.t, atol=1e-5, rtol=0)

    def test_grazing_never_behind_the_hit(self):
        radius = 0.3
        sdf = self.sphere_sdf(radius)
        ys = torch.linspace(0.0, radius + 0.1, 32, device=self.device)
        o = torch.stack([torch.full_like(ys, -0.6), ys, torch.zeros_like(ys)], dim=1)
        d = torch.zeros_like(o)
        d[:, 0] = 1.0

        pts = self.dual.ray_sdf_intersection_with_grazing(sdf, o, d, relaxation_eps=0.04)
        both = pts.crossing.mask & pts.grazing.mask
        if bool(both.any()):
            self.assertTrue(bool((pts.grazing.t[both] <= pts.crossing.t[both]).all()))


class RaySdfGradientTests(RaySdfTestCase):
    def test_sdf_and_grad_are_differentiable(self):
        """Values and gradients carry autograd history; the points themselves do not."""
        sdf = self.plane_sdf(torch.tensor([0.0, 0.0, 1.0], device=self.device), 0.1)
        sdf = sdf.detach().requires_grad_(True)

        o = torch.tensor([[0.0, 0.0, -0.3], [0.05, 0.02, -0.3]], device=self.device)
        d = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], device=self.device)

        hit = self.dual.ray_sdf_intersection(sdf, o, d)

        self.assertIsNotNone(hit.sdf.grad_fn)
        self.assertIsNotNone(hit.grad.grad_fn)
        self.assertIsNone(hit.position.grad_fn, "positions must stay detached")
        self.assertIsNone(hit.t.grad_fn, "ray parameters must stay detached")

        hit.sdf.sum().backward()
        self.assertIsNotNone(sdf.grad)
        self.assertGreater(float(sdf.grad.abs().sum()), 0.0)

    def test_values_match_direct_sampling(self):
        """The differentiable path must agree with sampling the same points directly."""
        sdf = self.sphere_sdf(0.3).detach().requires_grad_(True)
        ys = torch.linspace(0.0, 0.2, 8, device=self.device)
        o = torch.stack([torch.full_like(ys, -0.6), ys, torch.zeros_like(ys)], dim=1)
        d = torch.zeros_like(o)
        d[:, 0] = 1.0

        hit = self.dual.ray_sdf_intersection(sdf, o, d)
        idx = hit.mask.nonzero(as_tuple=True)[0]
        self.assertGreater(idx.numel(), 0)

        val, grad = self.dual.sample_trilinear_with_grad(hit.position[idx], sdf.unsqueeze(-1))
        torch.testing.assert_close(hit.sdf[idx], val[:, 0], atol=1e-5, rtol=0)
        torch.testing.assert_close(hit.grad[idx], grad[:, 0, :], atol=1e-4, rtol=0)

    def test_backward_matches_direct_sampling(self):
        """d(sum sdf + sum grad)/d(corners) equals sampling the hit points directly, and the
        rays that found nothing contribute nothing (their rows are zero with no history)."""
        sdf = self.sphere_sdf(0.3).detach().requires_grad_(True)
        ys = torch.linspace(-0.5, 0.5, 21, device=self.device)  # some rays miss the sphere
        o = torch.stack([torch.full_like(ys, -0.6), ys, torch.zeros_like(ys)], dim=1)
        d = torch.zeros_like(o)
        d[:, 0] = 1.0

        hit = self.dual.ray_sdf_intersection(sdf, o, d)
        miss = ~hit.mask
        self.assertTrue(bool(miss.any()) and bool(hit.mask.any()))
        self.assertEqual(hit.sdf[miss].detach().abs().sum().item(), 0.0)
        self.assertEqual(hit.grad[miss].detach().abs().sum().item(), 0.0)

        w = torch.randn(3, device=self.device)
        (hit.sdf.sum() + (hit.grad @ w).sum()).backward()  # over ALL rows, misses included
        g_kernel = sdf.grad.clone()

        sdf.grad = None
        idx = hit.mask.nonzero(as_tuple=True)[0]
        val, grad = self.dual.sample_trilinear_with_grad(hit.position[idx], sdf.unsqueeze(-1))
        (val[:, 0].sum() + (grad[:, 0, :] @ w).sum()).backward()
        torch.testing.assert_close(g_kernel, sdf.grad, atol=1e-4, rtol=1e-4)

    def test_no_grad_path_returns_kernel_values(self):
        """With grad disabled nothing is recomputed, and the kernel's values stand."""
        sdf = self.sphere_sdf(0.3)
        o = torch.tensor([[-0.6, 0.0, 0.0]], device=self.device)
        d = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)
        with torch.no_grad():
            hit = self.dual.ray_sdf_intersection(sdf, o, d)
        self.assertTrue(bool(hit.mask.all()))
        self.assertLess(float(hit.sdf.abs().max()), 1e-4)


if __name__ == "__main__":
    unittest.main()


class RaySdfBatchTests(RaySdfTestCase):
    """The batch variants must agree with the single ones, per grid in the batch."""

    def _batch_of_two(self):
        """A batch holding the same grid twice, so per-grid results are directly comparable."""
        ijk = self.grid.ijk
        gb = fvdb.GridBatch.from_ijk(fvdb.JaggedTensor([ijk, ijk]), voxel_sizes=self.vx, origins=0.0)
        return gb.dual_grid()

    def _rays(self):
        radius = 0.3
        ys = torch.linspace(0.0, radius + 0.1, 12, device=self.device)
        o = torch.stack([torch.full_like(ys, -0.6), ys, torch.zeros_like(ys)], dim=1)
        d = torch.zeros_like(o)
        d[:, 0] = 1.0
        return o, d

    def test_batch_matches_single(self):
        radius, band = 0.3, 0.04
        sdf = self.sphere_sdf(radius)
        o, d = self._rays()

        single = self.dual.ray_sdf_intersection_with_grazing(sdf, o, d, relaxation_eps=band)

        dual_batch = self._batch_of_two()
        sdf_b = fvdb.JaggedTensor([sdf, sdf])
        o_b = fvdb.JaggedTensor([o, o])
        d_b = fvdb.JaggedTensor([d, d])
        batch = dual_batch.ray_sdf_intersection_with_grazing(sdf_b, o_b, d_b, relaxation_eps=band)

        n = o.shape[0]
        for name in ("crossing", "grazing"):
            sp = getattr(single, name)
            bp = getattr(batch, name)
            for which in range(2):  # both copies of the grid in the batch
                lo, hi = which * n, (which + 1) * n
                torch.testing.assert_close(bp.t.jdata[lo:hi], sp.t)
                torch.testing.assert_close(bp.mask.jdata[lo:hi], sp.mask)
                torch.testing.assert_close(bp.sdf.jdata[lo:hi], sp.sdf)
                torch.testing.assert_close(bp.grad.jdata[lo:hi], sp.grad)
                torch.testing.assert_close(bp.position.jdata[lo:hi], sp.position)

    def test_batch_returns_jagged_tensors(self):
        sdf = self.sphere_sdf(0.3)
        o, d = self._rays()
        dual_batch = self._batch_of_two()
        pts = dual_batch.ray_sdf_intersection_with_grazing(
            fvdb.JaggedTensor([sdf, sdf]),
            fvdb.JaggedTensor([o, o]),
            fvdb.JaggedTensor([d, d]),
            relaxation_eps=0.04,
        )
        for field in pts.crossing:
            self.assertIsInstance(field, fvdb.JaggedTensor)

    def test_batch_gradients_flow(self):
        """The shared differentiability rule must hold for the batch path too."""
        sdf = self.sphere_sdf(0.3).detach().requires_grad_(True)
        o, d = self._rays()
        dual_batch = self._batch_of_two()
        pts = dual_batch.ray_sdf_intersection_with_grazing(
            fvdb.JaggedTensor([sdf, sdf]),
            fvdb.JaggedTensor([o, o]),
            fvdb.JaggedTensor([d, d]),
            relaxation_eps=0.04,
        )
        self.assertIsNotNone(pts.crossing.sdf.jdata.grad_fn)
        self.assertIsNone(pts.crossing.position.jdata.grad_fn)
        pts.crossing.sdf.jdata.sum().backward()
        self.assertGreater(float(sdf.grad.abs().sum()), 0.0)


class RaySdfSolverModeTests(RaySdfTestCase):
    """The selectable solvers: refine=newton|bisect, graze=analytic|bisect."""

    def _oblique_rays(self):
        """Rays deliberately off-axis, so the bisection grazing is well conditioned."""
        radius, band = 0.3, 0.04
        b = radius + 0.4 * band
        o = torch.tensor([[-0.6, b, 0.013], [-0.6, b * 0.97, -0.021]], device=self.device)
        d = torch.tensor([[1.0, 0.03, 0.017], [1.0, -0.02, 0.011]], device=self.device)
        d = d / d.norm(dim=1, keepdim=True)
        return o, d

    def test_bisect_refine_agrees_with_newton(self):
        """Both refine modes find the same root, to bisection's resolution."""
        sdf = self.sphere_sdf(0.3)
        o, d = self._oblique_rays()

        newton = self.dual.ray_sdf_intersection(sdf, o, d, refine="newton")
        bisect = self.dual.ray_sdf_intersection(sdf, o, d, refine="bisect")

        torch.testing.assert_close(newton.mask, bisect.mask)
        m = newton.mask
        # 8 halvings leave the answer within 1/256 of the bracket, which is at most one voxel.
        self.assertLess(float((newton.t[m] - bisect.t[m]).abs().max()), self.vx / 256.0)

        # Pin Newton to its own standard rather than to bisection's. Comparing the two only
        # within bisection's resolution is too loose to catch a broken Newton: a guard-ordering
        # bug that cost ~1.5e-4 (see refineRoot) sat inside this tolerance and passed.
        residual = self.dual.sample_trilinear(newton.position[m], sdf.unsqueeze(-1))[:, 0]
        self.assertLess(float(residual.abs().max()), 1e-6, "Newton must land ON the surface")

    def test_newton_exact_on_axis_aligned_rays(self):
        """Axis-aligned rays are where the cubic degenerates to a line and Newton must be exact.

        Regression test for a guard-ordering bug in refineRoot: once Newton converged, its step
        landed exactly on the bracket endpoint, the strict interior test rejected it, and the
        search bisected away and ran out of iterations ~1.5e-4 short. It only manifested when a
        near-zero residual came out positive, so it reproduced on GPU and not on CPU.
        """
        normal = torch.tensor([0.3, -0.7, 0.55], device=self.device)
        n = normal / normal.norm()
        offset = 0.11
        sdf = self.plane_sdf(normal, offset)

        o = torch.tensor([[-0.4, 0.0, 0.0], [0.0, -0.4, 0.0], [0.0, 0.0, -0.4]], device=self.device)
        d = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=self.device)
        hit = self.dual.ray_sdf_intersection(sdf, o, d, refine="newton")

        self.assertTrue(bool(hit.mask.all()))
        t_analytic = (offset - o @ n) / (d @ n)
        # The interpolant reproduces a plane exactly, so this is the solver's own error.
        self.assertLess(float((hit.t - t_analytic).abs().max()), 1e-6)

    def test_bisect_grazing_agrees_with_analytic(self):
        """Both grazing modes locate the same minimum, to bisection's resolution."""
        sdf = self.sphere_sdf(0.3)
        o, d = self._oblique_rays()

        analytic = self.dual.ray_sdf_grazing(sdf, o, d, relaxation_eps=0.10, graze="analytic")
        bisect = self.dual.ray_sdf_grazing(sdf, o, d, relaxation_eps=0.10, graze="bisect")

        torch.testing.assert_close(analytic.mask, bisect.mask)
        m = analytic.mask
        self.assertTrue(bool(m.any()), "fixture should produce grazing points")
        self.assertLess(float((analytic.t[m] - bisect.t[m]).abs().max()), 0.5 * self.vx)

    def test_bisect_grazing_finds_axis_aligned_minimum(self):
        """Both modes handle a minimum sitting exactly on a voxel face.

        The face case is degenerate for the *derivative*: it is exactly zero on both sides. The
        analytic mode copes because it minimises over closed intervals, and the bisection mode
        copes because its bracket runs midpoint-to-midpoint and straddles the face, so it never
        evaluates the derivative at the degenerate point. They disagree on t by about half a
        voxel, which is bisection's bracket resolution, while agreeing on the SDF value.
        """
        radius, band = 0.3, 0.04
        b = radius + 0.4 * band
        sdf = self.sphere_sdf(radius)
        o = torch.tensor([[-0.6, b, 0.0]], device=self.device)
        d = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)

        analytic = self.dual.ray_sdf_grazing(sdf, o, d, relaxation_eps=band, graze="analytic")
        bisect = self.dual.ray_sdf_grazing(sdf, o, d, relaxation_eps=band, graze="bisect")

        self.assertTrue(bool(analytic.mask[0]))
        self.assertTrue(bool(bisect.mask[0]))
        self.assertLess(abs(float(analytic.t[0]) - float(bisect.t[0])), self.vx)
        self.assertAlmostEqual(float(analytic.sdf[0]), float(bisect.sdf[0]), delta=1e-4)

    def test_analytic_grazing_accepts_only_minima(self):
        """Analytic must not accept far more rays than bisection does.

        A grazing point is a local minimum of the field along the ray. An earlier version of
        the analytic mode minimised |SDF| over each cell's closed interval and relied on a
        tangency test at the end, which let a ray passing straight THROUGH a voxel face be
        accepted: a cell endpoint sits exactly on a face, where the trilinear field is C0 but
        not C1, so one of the two one-sided gradients can look tangent to a transversal
        crossing. Small |SDF| on a face is not evidence of a minimum.

        Measured on a 256^2 sphere render, that accepted 32 rays where bisection accepted 14,
        inflating the relaxed-boundary term's gradient 2.2x and breaking a downstream
        finite-difference check of the renderer.

        Bisection is the reference for the COUNT here, not for the positions: it only updates
        at a genuine turning point, so it cannot over-accept. Analytic is allowed to find
        slightly more -- it resolves minima sitting exactly on a face, which bisection's
        midpoint brackets smear -- but "slightly" is the point of the test.
        """
        sdf = self.sphere_sdf(0.3)
        # A dense fan sweeping through the silhouette, so most rays pass near-tangentially.
        ys = torch.linspace(0.26, 0.34, 400, device=self.device)
        o = torch.stack([torch.full_like(ys, -0.6), ys, torch.full_like(ys, 0.011)], dim=1)
        d = torch.zeros_like(o)
        d[:, 0] = 1.0
        d[:, 2] = 0.013
        d = d / d.norm(dim=1, keepdim=True)

        analytic = self.dual.ray_sdf_grazing(sdf, o, d, relaxation_eps=0.02, graze="analytic")
        bisect = self.dual.ray_sdf_grazing(sdf, o, d, relaxation_eps=0.02, graze="bisect")

        n_a, n_b = int(analytic.mask.sum()), int(bisect.mask.sum())
        self.assertGreater(n_b, 0, "fixture should produce grazing points")
        self.assertLessEqual(
            n_a, 1.5 * n_b, f"analytic accepted {n_a} vs bisection's {n_b}: it is accepting non-minima"
        )
        # Bounded from below too. These rays sweep the silhouette of a lattice-aligned sphere,
        # so many minima sit exactly on a voxel face -- the case this mode exists to resolve.
        # Accepting none of them means the face path is broken.
        self.assertGreaterEqual(
            n_a, 0.5 * n_b, f"analytic accepted {n_a} vs bisection's {n_b}: it is rejecting real minima"
        )

    def test_grazing_accepts_crease_minimum(self):
        """A minimum on a crease is a grazing point, however lopsided the crease.

        |x - x0| + c with x0 on a lattice plane is reproduced exactly by trilinear interpolation,
        with the kink sitting on a cell face. A ray along +x has its SDF minimum, c, exactly on
        that face. Minima on faces are generic for a C0 interpolant, not an edge case: the
        relaxed-boundary reference implementation's grazing points lie ~75% on faces. Whatever
        the two slopes, the ray starts hitting the surface when c drops below zero, so both
        modes accept the symmetric (-1 / +1) and the asymmetric (-1 / +0.2) crease. A kink
        without a minimum (-1 / -0.2, the SDF keeps falling into a crossing) is not a graze.
        """
        x0 = 0.5 * self.vx  # a plane of corner positions, so the kink is on a cell face
        self.assertTrue(bool((self.corners[:, 0] - x0).abs().min() < 1e-6))
        c = 0.01
        dx = self.corners[:, 0] - x0
        o = torch.tensor([[-0.4, 0.013, -0.021]], device=self.device)
        d = torch.tensor([[1.0, 0.0, 0.0]], device=self.device)
        creases = {"symmetric": dx.abs() + c, "asymmetric": torch.where(dx < 0, -dx, 0.2 * dx) + c}
        for graze in ("analytic", "bisect"):
            for name, sdf in creases.items():
                got = self.dual.ray_sdf_grazing(sdf, o, d, relaxation_eps=0.04, graze=graze)
                self.assertTrue(bool(got.mask[0]), f"graze={graze} rejected a {name} crease minimum")
                self.assertAlmostEqual(float(got.sdf[0]), c, delta=1e-4)
            no_min = torch.where(dx < 0, -dx, -0.2 * dx) + c
            got = self.dual.ray_sdf_grazing(no_min, o, d, relaxation_eps=0.04, graze=graze)
            self.assertFalse(bool(got.mask[0]), f"graze={graze} accepted a kink with no minimum")

    def test_invalid_mode_names_are_rejected(self):
        sdf = self.sphere_sdf(0.3)
        o, d = self._oblique_rays()
        with self.assertRaises(ValueError):
            self.dual.ray_sdf_intersection(sdf, o, d, refine="secant")
        with self.assertRaises(ValueError):
            self.dual.ray_sdf_grazing(sdf, o, d, relaxation_eps=0.04, graze="newton")
