# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
"""Functional API for ray operations on sparse grids."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import torch

from .. import _fvdb_cpp
from ..jagged_tensor import JaggedTensor

if TYPE_CHECKING:
    from ..grid import Grid
    from ..grid_batch import GridBatch


# ---------------------------------------------------------------------------
#  Batch variants (GridBatch + JaggedTensor)
# ---------------------------------------------------------------------------


def voxels_along_rays_batch(
    grid: GridBatch,
    ray_origins: JaggedTensor,
    ray_directions: JaggedTensor,
    max_voxels: int,
    eps: float = 0.0,
    return_ijk: bool = False,
    cumulative: bool = False,
) -> tuple[JaggedTensor, JaggedTensor]:
    """Enumerate voxels intersected by rays using a DDA traversal on a grid batch.

    Args:
        grid (GridBatch): The grid batch to trace through.
        ray_origins (JaggedTensor): Ray origin positions, shape ``(B, -1, 3)``.
        ray_directions (JaggedTensor): Ray direction vectors, shape ``(B, -1, 3)``.
        max_voxels (int): Maximum number of voxels to return per ray.
        eps (float): Small offset to avoid self-intersection. Default ``0.0``.
        return_ijk (bool): If ``True``, return voxel coordinates instead of linear indices.
        cumulative (bool): If ``True``, return cumulative indices across the batch.

    Returns:
        voxels (JaggedTensor): Voxel coordinates or linear indices per ray hit.
        distances (JaggedTensor): ``(t_entry, t_exit)`` pairs per ray hit.

    .. seealso:: :func:`voxels_along_rays_single`
    """
    grid_data = grid.data
    result = _fvdb_cpp.voxels_along_rays(
        grid_data, ray_origins._impl, ray_directions._impl, max_voxels, eps, return_ijk, cumulative
    )
    return JaggedTensor(impl=result[0]), JaggedTensor(impl=result[1])


def segments_along_rays_batch(
    grid: GridBatch,
    ray_origins: JaggedTensor,
    ray_directions: JaggedTensor,
    max_segments: int,
    eps: float = 0.0,
) -> JaggedTensor:
    """Return continuous segments of ray traversal through a grid batch.

    Args:
        grid (GridBatch): The grid batch to trace through.
        ray_origins (JaggedTensor): Ray origin positions, shape ``(B, -1, 3)``.
        ray_directions (JaggedTensor): Ray direction vectors, shape ``(B, -1, 3)``.
        max_segments (int): Maximum number of segments to return per ray.
        eps (float): Small offset to avoid self-intersection. Default ``0.0``.

    Returns:
        segments (JaggedTensor): ``(t_start, t_end)`` pairs per ray segment.

    .. seealso:: :func:`segments_along_rays_single`
    """
    grid_data = grid.data
    return JaggedTensor(
        impl=_fvdb_cpp.segments_along_rays(grid_data, ray_origins._impl, ray_directions._impl, max_segments, eps)
    )


def uniform_ray_samples_batch(
    grid: GridBatch,
    ray_origins: JaggedTensor,
    ray_directions: JaggedTensor,
    t_min: JaggedTensor,
    t_max: JaggedTensor,
    step_size: float,
    cone_angle: float = 0.0,
    include_end_segments: bool = True,
    return_midpoints: bool = False,
    eps: float = 0.0,
) -> JaggedTensor:
    """Generate uniformly spaced samples along rays that intersect active voxels of a grid batch.

    Args:
        grid (GridBatch): The grid batch to sample through.
        ray_origins (JaggedTensor): Ray origin positions, shape ``(B, -1, 3)``.
        ray_directions (JaggedTensor): Ray direction vectors, shape ``(B, -1, 3)``.
        t_min (JaggedTensor): Minimum ray distances per ray.
        t_max (JaggedTensor): Maximum ray distances per ray.
        step_size (float): Distance between consecutive samples.
        cone_angle (float): Cone angle for mip-mapping. Default ``0.0``.
        include_end_segments (bool): Include segment endpoints. Default ``True``.
        return_midpoints (bool): Return midpoints instead of boundaries. Default ``False``.
        eps (float): Small offset to avoid self-intersection. Default ``0.0``.

    Returns:
        samples (JaggedTensor): Sample distances along each ray.

    .. seealso:: :func:`uniform_ray_samples_single`
    """
    grid_data = grid.data
    return JaggedTensor(
        impl=_fvdb_cpp.uniform_ray_samples(
            grid_data,
            ray_origins._impl,
            ray_directions._impl,
            t_min._impl,
            t_max._impl,
            step_size,
            cone_angle,
            include_end_segments,
            return_midpoints,
            eps,
        )
    )


def ray_implicit_intersection_batch(
    grid: GridBatch,
    ray_origins: JaggedTensor,
    ray_directions: JaggedTensor,
    grid_scalars: JaggedTensor,
    eps: float = 0.0,
) -> JaggedTensor:
    """Find ray intersections with an implicit surface defined by grid scalars on a grid batch.

    The first valid (non-NaN) voxel sampled along each ray seeds the sign reference, and the first
    subsequent voxel with the opposite sign is reported as the intersection. Both "positive outside"
    and "negative outside" SDF conventions are therefore handled identically, and a ray that enters
    the bbox already inside the surface is reported at the *exit* of the surface along the ray.

    Args:
        grid (GridBatch): The grid batch defining the implicit surface topology.
        ray_origins (JaggedTensor): Ray origin positions, shape ``(B, -1, 3)``.
        ray_directions (JaggedTensor): Ray direction vectors, shape ``(B, -1, 3)``.
        grid_scalars (JaggedTensor): Per-voxel scalar values defining the implicit surface.
        eps (float): Small offset to avoid self-intersection. Default ``0.0``.

    Returns:
        distances (JaggedTensor): Intersection distance per ray, or ``-1`` if no intersection.

    .. seealso:: :func:`ray_implicit_intersection_single`
    """
    grid_data = grid.data
    result_impl = _fvdb_cpp.ray_implicit_intersection(
        grid_data, ray_origins._impl, ray_directions._impl, grid_scalars._impl, eps
    )
    return JaggedTensor(impl=result_impl)


def rays_intersect_voxels_batch(
    grid: GridBatch,
    ray_origins: JaggedTensor,
    ray_directions: JaggedTensor,
    eps: float = 0.0,
) -> JaggedTensor:
    """Check whether rays hit any voxels in a grid batch.

    Args:
        grid (GridBatch): The grid batch to test against.
        ray_origins (JaggedTensor): Ray origin positions, shape ``(B, -1, 3)``.
        ray_directions (JaggedTensor): Ray direction vectors, shape ``(B, -1, 3)``.
        eps (float): Small offset to avoid self-intersection. Default ``0.0``.

    Returns:
        hit (JaggedTensor): Boolean mask indicating whether each ray hit a voxel.

    .. seealso:: :func:`rays_intersect_voxels_single`
    """
    _, ray_times = voxels_along_rays_batch(
        grid,
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        max_voxels=1,
        eps=eps,
        return_ijk=False,
        cumulative=False,
    )
    did_hit = (ray_times.joffsets[1:] - ray_times.joffsets[:-1]) > 0
    return ray_origins.jagged_like(did_hit)


# ---------------------------------------------------------------------------
#  Single variants (Grid + torch.Tensor)
# ---------------------------------------------------------------------------


def voxels_along_rays_single(
    grid: Grid,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    max_voxels: int,
    eps: float = 0.0,
    return_ijk: bool = False,
) -> tuple[JaggedTensor, JaggedTensor]:
    """Enumerate voxels intersected by rays using a DDA traversal on a single grid.

    Args:
        grid (Grid): The single grid to trace through.
        ray_origins (torch.Tensor): Ray origin positions, shape ``(N, 3)``.
        ray_directions (torch.Tensor): Ray direction vectors, shape ``(N, 3)``.
        max_voxels (int): Maximum number of voxels to return per ray.
        eps (float): Small offset to avoid self-intersection. Default ``0.0``.
        return_ijk (bool): If ``True``, return voxel coordinates instead of linear indices.

    Returns:
        voxels (JaggedTensor): Voxel coordinates or linear indices per ray hit.
        distances (JaggedTensor): ``(t_entry, t_exit)`` pairs per ray hit.

    .. seealso:: :func:`voxels_along_rays_batch`
    """
    grid_data = grid.data
    origins_jt = JaggedTensor(ray_origins)
    directions_jt = JaggedTensor(ray_directions)
    result = _fvdb_cpp.voxels_along_rays(
        grid_data, origins_jt._impl, directions_jt._impl, max_voxels, eps, return_ijk, False
    )
    return JaggedTensor(impl=result[0]), JaggedTensor(impl=result[1])


def segments_along_rays_single(
    grid: Grid,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    max_segments: int,
    eps: float = 0.0,
) -> JaggedTensor:
    """Return continuous segments of ray traversal through a single grid.

    Args:
        grid (Grid): The single grid to trace through.
        ray_origins (torch.Tensor): Ray origin positions, shape ``(N, 3)``.
        ray_directions (torch.Tensor): Ray direction vectors, shape ``(N, 3)``.
        max_segments (int): Maximum number of segments to return per ray.
        eps (float): Small offset to avoid self-intersection. Default ``0.0``.

    Returns:
        segments (JaggedTensor): ``(t_start, t_end)`` pairs per ray segment.

    .. seealso:: :func:`segments_along_rays_batch`
    """
    grid_data = grid.data
    origins_jt = JaggedTensor(ray_origins)
    directions_jt = JaggedTensor(ray_directions)
    return JaggedTensor(
        impl=_fvdb_cpp.segments_along_rays(grid_data, origins_jt._impl, directions_jt._impl, max_segments, eps)
    )


def uniform_ray_samples_single(
    grid: Grid,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    t_min: torch.Tensor,
    t_max: torch.Tensor,
    step_size: float,
    cone_angle: float = 0.0,
    include_end_segments: bool = True,
    return_midpoints: bool = False,
    eps: float = 0.0,
) -> JaggedTensor:
    """Generate uniformly spaced samples along rays that intersect active voxels of a single grid.

    Args:
        grid (Grid): The single grid to sample through.
        ray_origins (torch.Tensor): Ray origin positions, shape ``(N, 3)``.
        ray_directions (torch.Tensor): Ray direction vectors, shape ``(N, 3)``.
        t_min (torch.Tensor): Minimum ray distances per ray.
        t_max (torch.Tensor): Maximum ray distances per ray.
        step_size (float): Distance between consecutive samples.
        cone_angle (float): Cone angle for mip-mapping. Default ``0.0``.
        include_end_segments (bool): Include segment endpoints. Default ``True``.
        return_midpoints (bool): Return midpoints instead of boundaries. Default ``False``.
        eps (float): Small offset to avoid self-intersection. Default ``0.0``.

    Returns:
        samples (JaggedTensor): Sample distances along each ray.

    .. seealso:: :func:`uniform_ray_samples_batch`
    """
    grid_data = grid.data
    origins_jt = JaggedTensor(ray_origins)
    directions_jt = JaggedTensor(ray_directions)
    t_min_jt = JaggedTensor(t_min)
    t_max_jt = JaggedTensor(t_max)
    return JaggedTensor(
        impl=_fvdb_cpp.uniform_ray_samples(
            grid_data,
            origins_jt._impl,
            directions_jt._impl,
            t_min_jt._impl,
            t_max_jt._impl,
            step_size,
            cone_angle,
            include_end_segments,
            return_midpoints,
            eps,
        )
    )


def ray_implicit_intersection_single(
    grid: Grid,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    grid_scalars: torch.Tensor,
    eps: float = 0.0,
) -> torch.Tensor:
    """Find ray intersections with an implicit surface defined by grid scalars on a single grid.

    The first valid (non-NaN) voxel sampled along each ray seeds the sign reference, and the first
    subsequent voxel with the opposite sign is reported as the intersection. Both "positive outside"
    and "negative outside" SDF conventions are therefore handled identically, and a ray that enters
    the bbox already inside the surface is reported at the *exit* of the surface along the ray.

    Args:
        grid (Grid): The single grid defining the implicit surface topology.
        ray_origins (torch.Tensor): Ray origin positions, shape ``(N, 3)``.
        ray_directions (torch.Tensor): Ray direction vectors, shape ``(N, 3)``.
        grid_scalars (torch.Tensor): Per-voxel scalar values defining the implicit surface.
        eps (float): Small offset to avoid self-intersection. Default ``0.0``.

    Returns:
        distances (torch.Tensor): Intersection distance per ray, or ``-1`` if no intersection.

    .. seealso:: :func:`ray_implicit_intersection_batch`
    """
    grid_data = grid.data
    origins_jt = JaggedTensor(ray_origins)
    directions_jt = JaggedTensor(ray_directions)
    scalars_jt = JaggedTensor(grid_scalars)
    result_impl = _fvdb_cpp.ray_implicit_intersection(
        grid_data, origins_jt._impl, directions_jt._impl, scalars_jt._impl, eps
    )
    return result_impl.jdata


def rays_intersect_voxels_single(
    grid: Grid,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    eps: float = 0.0,
) -> torch.Tensor:
    """Check whether rays hit any voxels in a single grid.

    Args:
        grid (Grid): The single grid to test against.
        ray_origins (torch.Tensor): Ray origin positions, shape ``(N, 3)``.
        ray_directions (torch.Tensor): Ray direction vectors, shape ``(N, 3)``.
        eps (float): Small offset to avoid self-intersection. Default ``0.0``.

    Returns:
        hit (torch.Tensor): Boolean mask indicating whether each ray hit a voxel.

    .. seealso:: :func:`rays_intersect_voxels_batch`
    """
    _, ray_times = voxels_along_rays_single(
        grid,
        ray_origins=ray_origins,
        ray_directions=ray_directions,
        max_voxels=1,
        eps=eps,
        return_ijk=False,
    )
    did_hit = (ray_times.joffsets[1:] - ray_times.joffsets[:-1]) > 0
    return did_hit


# ---------------------------------------------------------------------------
#  SDF surface points (trilinear interpolant)
# ---------------------------------------------------------------------------


_REFINE_MODES = {"newton": 0, "bisect": 1}
_GRAZE_MODES = {"analytic": 0, "bisect": 1}


def _mode(value: str, table: dict[str, int], name: str) -> int:
    """Validate a solver-method argument, failing by name rather than deep in the kernel."""
    try:
        return table[value]
    except KeyError:
        raise ValueError(f"{name} must be one of {sorted(table)}, got {value!r}") from None


class RaySdfPoint(NamedTuple):
    """One surface point per ray, with the SDF sampled there.

    ``position``, ``t`` and ``mask`` are always detached: they describe *where* the point is,
    which is a geometric query, not a differentiable function of the SDF values. ``sdf`` and
    ``grad`` are differentiable with respect to the SDF values when those require grad; the
    backward is attached only for rays that found a point, the others carry zeros with no
    history.
    """

    t: torch.Tensor  # (N,)    ray parameter of the point
    mask: torch.Tensor  # (N,)    whether a point was found and accepted
    position: torch.Tensor  # (N, 3) ray_origins + t * ray_directions
    sdf: torch.Tensor  # (N,)    SDF at the point
    grad: torch.Tensor  # (N, 3) spatial SDF gradient at the point


class RaySdfPoints(NamedTuple):
    """Both surface points per ray, from a single traversal."""

    crossing: RaySdfPoint
    grazing: RaySdfPoint


class _RaySdfPointValuesFn(torch.autograd.Function):
    """Attach autograd to the values the kernel already computed.

    Forward is a pass-through. The kernel evaluated the SDF and its spatial gradient while it
    had the voxel's stencil in registers, and those values are exact, so there is nothing to
    recompute -- only the backward is new. That backward is the same scatter
    ``sample_trilinear_with_grad`` uses, evaluated at the *detached* points, which is the
    correct derivative precisely because the points are not differentiable.
    """

    @staticmethod
    def forward(ctx, voxel_data, grid_data, pts_impl, value, gradient):
        ctx.grid_data = grid_data
        ctx.pts_impl = pts_impl
        ctx.save_for_backward(voxel_data)
        return value, gradient

    @staticmethod
    def backward(ctx: Any, *grad_outputs: torch.Tensor | None) -> tuple[torch.Tensor | None, ...]:
        grad_value, grad_gradient = grad_outputs
        (voxel_data,) = ctx.saved_tensors
        if grad_value is None:
            grad_value = torch.zeros(voxel_data.new_empty(0).shape)  # pragma: no cover
        # Match the layouts sample_trilinear_with_grad produces: (M, C) and (M, C, 3).
        grad_data = _fvdb_cpp.sample_trilinear_with_grad_bwd(
            ctx.grid_data,
            ctx.pts_impl,
            voxel_data,
            grad_value.unsqueeze(-1).contiguous(),
            grad_gradient.unsqueeze(1).contiguous(),
        )
        return grad_data, None, None, None, None


def _sdf_point(
    sdf: torch.Tensor,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    times: torch.Tensor,
    mask: torch.Tensor,
    sdf_vals: torch.Tensor,
    grads: torch.Tensor,
    grid_data,
    make_pts_impl,
) -> RaySdfPoint:
    """Assemble one output slot, making the SDF value and gradient differentiable.

    Works on flat tensors only, so the single and batch variants share it: the batch wrapper
    passes ``.jdata`` and re-wraps the result. The only difference between them is how the
    points are wrapped for the C++ layer, which is what ``make_pts_impl`` supplies.

    The kernel's own values are returned -- nothing is re-evaluated. When the SDF requires grad
    they are routed through :class:`_RaySdfPointValuesFn` so a backward exists; when it does not
    (the common geometry-only path, e.g. under ``torch.no_grad``) they are passed straight out.

    ``make_pts_impl(points, ray_index)`` wraps a subset of the points for the C++ layer;
    ``ray_index`` says which rays they belong to, which the batch variant needs to keep each
    point with its grid.
    """
    position = ray_origins + times.unsqueeze(-1) * ray_directions

    if sdf.requires_grad and torch.is_grad_enabled():
        # Only the rays that found a point get an autograd node. The backward is a stencil
        # scatter per point, so running it over every ray would cost the full ray count while
        # the misses -- which carry the ray origin as their "position" -- must contribute
        # nothing anyway. Compact to the accepted rows, attach the node there, scatter back.
        idx = mask.nonzero(as_tuple=True)[0]
        vals_hit, grads_hit = _RaySdfPointValuesFn.apply(
            sdf.unsqueeze(-1),
            grid_data,
            make_pts_impl(position[idx].contiguous(), idx),
            sdf_vals[idx],
            grads[idx],
        )
        sdf_vals = torch.zeros_like(sdf_vals).index_put((idx,), vals_hit)
        grads = torch.zeros_like(grads).index_put((idx,), grads_hit)

    return RaySdfPoint(t=times, mask=mask, position=position, sdf=sdf_vals, grad=grads)


def ray_sdf_intersection_single(
    grid: Grid,
    sdf: torch.Tensor,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    t_min: float = 1e-4,
    eps: float = 1e-4,
    refine: str = "bisect",
) -> RaySdfPoint:
    """First zero crossing of the trilinearly interpolated SDF along each ray.

    Each cell's crossing is bracketed from the cubic the SDF traces along the ray inside it,
    split at the cubic's turning points, so a surface thinner than a cell is still seen;
    ``refine`` only selects how the bracketed crossing is refined. A sign change in either direction
    counts, so rays that start inside the surface report their exit.

    Args:
        grid (Grid): The grid the SDF is stored on: the **dual** of the geometry grid, e.g.
            ``geometry_grid.dual_grid()``.
        sdf (torch.Tensor): One SDF value per voxel of the grid.
        ray_origins (torch.Tensor): Ray origins, shape ``(N, 3)``.
        ray_directions (torch.Tensor): Ray directions, shape ``(N, 3)``. Assumed normalized, so
            ``t`` is a world-space distance.
        t_min (float): Earliest accepted crossing. Default ``1e-4``.
        eps (float): Skip cells whose ray segment is shorter than this. Default ``1e-4``.
        refine (str): How a bracketed crossing is refined: ``"bisect"`` (default; 8 halvings
            on the sign of the cell's cubic, to 1/256 of the bracket) or ``"newton"`` (safeguarded
            Newton, to float precision). Both bracket every crossing from the cubic, so a surface
            thinner than a voxel is seen.

    Returns:
        crossing (RaySdfPoint): The hit point, its SDF value and gradient.

    .. seealso:: :func:`ray_sdf_grazing_single`, :func:`ray_sdf_intersection_with_grazing_single`
    """
    times, mask, sdf_vals, grads = _fvdb_cpp.ray_sdf_intersection(
        grid.data,
        JaggedTensor(ray_origins)._impl,
        JaggedTensor(ray_directions)._impl,
        JaggedTensor(sdf)._impl,
        t_min,
        eps,
        _mode(refine, _REFINE_MODES, "refine"),
    )
    return _sdf_point(
        sdf,
        ray_origins,
        ray_directions,
        times.jdata[:, 0],
        mask.jdata[:, 0],
        sdf_vals.jdata[:, 0],
        grads.jdata[:, 0, :],
        grid.data,
        lambda pts, idx: JaggedTensor(pts)._impl,
    )


def ray_sdf_grazing_single(
    grid: Grid,
    sdf: torch.Tensor,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    relaxation_eps: float,
    t_max: torch.Tensor | None = None,
    ray_mask: torch.Tensor | None = None,
    graze_t_min: float = 1e-4,
    itx_eps: float = 1e-7,
    deriv_eps: float = 1e-1,
    eps: float = 1e-4,
    graze: str = "bisect",
) -> RaySdfPoint:
    """Grazing point: the local minimum of the SDF along each ray inside the relaxation band.

    The point where the SDF along the ray stops approaching and starts receding, i.e. where
    the directional derivative crosses zero from negative to positive. Inside a cell that
    derivative is a quadratic and is solved exactly; because the interpolant is C0 but not
    C1, a minimum can also sit exactly on a cell face, and those are detected separately.

    This is the relaxed silhouette of the relaxed-boundary method (Wang et al. 2025): rays that
    pass within ``relaxation_eps`` of the surface without hitting it, rather than exactly
    tangent rays. The boundary term weights each such point by ``-SDF / relaxation_eps``.

    A candidate is accepted only when ``itx_eps < SDF < relaxation_eps`` and the surface is
    near-tangent to the ray. The lower bound is not zero on purpose: ``SDF == 0`` is the
    surface itself, so such a point is a tangential hit belonging to the interior term, and
    testing against a bare zero lets rounding flip points between the two sets.

    Args:
        grid (Grid): The grid the SDF is stored on: the **dual** of the geometry grid, e.g.
            ``geometry_grid.dual_grid()``.
        sdf (torch.Tensor): One SDF value per voxel of the grid.
        ray_origins (torch.Tensor): Ray origins, shape ``(N, 3)``.
        ray_directions (torch.Tensor): Ray directions, shape ``(N, 3)``. Assumed normalized, so
            ``t`` is a world-space distance.
        relaxation_eps (float): Upper edge of the band a grazing point's SDF must lie in.
        t_max (torch.Tensor | None): Optional per-ray search bound, shape ``(N,)``, typically
            the first hit. Default ``None`` (unbounded).
        ray_mask (torch.Tensor | None): Optional per-ray boolean enable, shape ``(N,)``. Default
            ``None`` (all rays).
        graze_t_min (float): Earliest accepted grazing point, to keep the search off the ray
            origin. Default ``1e-4``.
        itx_eps (float): Lower edge of the band: a smaller SDF counts as a hit, not a graze.
            Default ``1e-7``.
        deriv_eps (float): Tangency tolerance: a grazing point needs
            ``|normalize(grad) . direction|`` below this. Default ``0.1``.
        eps (float): Skip cells whose ray segment is shorter than this. Default ``1e-4``.
        graze (str): How the grazing point is found: ``"bisect"`` (default; brackets the
            minimum between adjacent cell midpoints, bisects 8 times on the derivative's sign, and
            judges tangency from the average of the gradients at the two ends of the final
            bracket -- at a cell face that is the average of the two one-sided slopes, matching
            the relaxed-boundary reference implementation's central-difference gradient) or
            ``"analytic"`` (solves each cell's derivative quadratic exactly, including minima on
            cell faces).

    Returns:
        grazing (RaySdfPoint): The grazing point, its SDF value and gradient.

    .. seealso:: :func:`ray_sdf_intersection_single`, :func:`ray_sdf_intersection_with_grazing_single`
    """
    times, mask, sdf_vals, grads = _fvdb_cpp.ray_sdf_grazing(
        grid.data,
        JaggedTensor(ray_origins)._impl,
        JaggedTensor(ray_directions)._impl,
        JaggedTensor(sdf)._impl,
        None if t_max is None else JaggedTensor(t_max)._impl,
        None if ray_mask is None else JaggedTensor(ray_mask)._impl,
        graze_t_min,
        relaxation_eps,
        itx_eps,
        deriv_eps,
        eps,
        _mode(graze, _GRAZE_MODES, "graze"),
    )
    return _sdf_point(
        sdf,
        ray_origins,
        ray_directions,
        times.jdata[:, 0],
        mask.jdata[:, 0],
        sdf_vals.jdata[:, 0],
        grads.jdata[:, 0, :],
        grid.data,
        lambda pts, idx: JaggedTensor(pts)._impl,
    )


def ray_sdf_intersection_with_grazing_single(
    grid: Grid,
    sdf: torch.Tensor,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    relaxation_eps: float,
    t_min: float = 1e-4,
    graze_t_min: float = 1e-4,
    itx_eps: float = 1e-7,
    deriv_eps: float = 1e-1,
    eps: float = 1e-4,
    refine: str = "bisect",
    graze: str = "bisect",
) -> RaySdfPoints:
    """Both the first crossing and the grazing point, from a single traversal.

    The grazing point is the relaxed-silhouette point of the relaxed-boundary method, see
    :func:`ray_sdf_grazing_single`.

    Cheaper than calling :func:`ray_sdf_intersection_single` and
    :func:`ray_sdf_grazing_single` separately, for two reasons: the crossing search reuses
    the derivative roots the grazing search already computes, and marching in ray order means
    every grazing candidate found before the march stops at the crossing is automatically in
    front of the surface -- so no separate ``t_max`` is needed or accepted here.

    Args:
        grid (Grid): The grid the SDF is stored on: the **dual** of the geometry grid, e.g.
            ``geometry_grid.dual_grid()``.
        sdf (torch.Tensor): One SDF value per voxel of the grid.
        ray_origins (torch.Tensor): Ray origins, shape ``(N, 3)``.
        ray_directions (torch.Tensor): Ray directions, shape ``(N, 3)``. Assumed normalized, so
            ``t`` is a world-space distance.
        relaxation_eps (float): Upper edge of the band a grazing point's SDF must lie in.
        t_min (float): Earliest accepted crossing. Default ``1e-4``.
        graze_t_min (float): Earliest accepted grazing point, to keep the search off the ray
            origin. Default ``1e-4``.
        itx_eps (float): Lower edge of the band: a smaller SDF counts as a hit, not a graze.
            Default ``1e-7``.
        deriv_eps (float): Tangency tolerance: a grazing point needs
            ``|normalize(grad) . direction|`` below this. Default ``0.1``.
        eps (float): Skip cells whose ray segment is shorter than this. Default ``1e-4``.
        refine (str): How a bracketed crossing is refined: ``"bisect"`` (default; 8 halvings
            on the sign of the cell's cubic, to 1/256 of the bracket) or ``"newton"`` (safeguarded
            Newton, to float precision). Both bracket every crossing from the cubic, so a surface
            thinner than a voxel is seen.
        graze (str): How the grazing point is found: ``"bisect"`` (default; brackets the
            minimum between adjacent cell midpoints, bisects 8 times on the derivative's sign, and
            judges tangency from the average of the gradients at the two ends of the final
            bracket -- at a cell face that is the average of the two one-sided slopes, matching
            the relaxed-boundary reference implementation's central-difference gradient) or
            ``"analytic"`` (solves each cell's derivative quadratic exactly, including minima on
            cell faces).

    Returns:
        points (RaySdfPoints): ``.crossing`` and ``.grazing``, each a
        :class:`RaySdfPoint`.

    .. seealso:: :func:`ray_sdf_intersection_single`, :func:`ray_sdf_grazing_single`
    """
    times, mask, sdf_vals, grads = _fvdb_cpp.ray_sdf_intersection_with_grazing(
        grid.data,
        JaggedTensor(ray_origins)._impl,
        JaggedTensor(ray_directions)._impl,
        JaggedTensor(sdf)._impl,
        t_min,
        graze_t_min,
        relaxation_eps,
        itx_eps,
        deriv_eps,
        eps,
        _mode(refine, _REFINE_MODES, "refine"),
        _mode(graze, _GRAZE_MODES, "graze"),
    )
    t, m, s, g = times.jdata, mask.jdata, sdf_vals.jdata, grads.jdata
    pts_impl = lambda pts, idx: JaggedTensor(pts)._impl
    return RaySdfPoints(
        crossing=_sdf_point(
            sdf, ray_origins, ray_directions, t[:, 0], m[:, 0], s[:, 0], g[:, 0, :], grid.data, pts_impl
        ),
        grazing=_sdf_point(
            sdf, ray_origins, ray_directions, t[:, 1], m[:, 1], s[:, 1], g[:, 1, :], grid.data, pts_impl
        ),
    )


def _sdf_point_batch(
    grid: GridBatch,
    sdf: JaggedTensor,
    ray_origins: JaggedTensor,
    ray_directions: JaggedTensor,
    times: torch.Tensor,
    mask: torch.Tensor,
    sdf_vals: torch.Tensor,
    grads: torch.Tensor,
) -> RaySdfPoint:
    """Batch counterpart of :func:`_sdf_point`: run it flat, then re-wrap as JaggedTensors.

    The rays of a batch are stored concatenated in ``.jdata``, so every operation in
    :func:`_sdf_point` is elementwise-correct on the flat form and only the wrapping differs.
    """
    point = _sdf_point(
        sdf.jdata,
        ray_origins.jdata,
        ray_directions.jdata,
        times,
        mask,
        sdf_vals,
        grads,
        grid.data,
        # A compacted subset no longer matches the batch's offsets, so rebuild the jagged
        # structure from the rays the points came from.
        lambda pts, idx: JaggedTensor.from_data_and_indices(pts, ray_origins.jidx[idx], ray_origins.num_tensors)._impl,
    )
    return RaySdfPoint(*(ray_origins.jagged_like(field) for field in point))


def ray_sdf_intersection_batch(
    grid: GridBatch,
    sdf: JaggedTensor,
    ray_origins: JaggedTensor,
    ray_directions: JaggedTensor,
    t_min: float = 1e-4,
    eps: float = 1e-4,
    refine: str = "bisect",
) -> RaySdfPoint:
    """Batched :func:`ray_sdf_intersection_single`.

    Args:
        grid (GridBatch): The grid batch the SDF is stored on: the **dual** of the geometry grid
            batch, e.g. ``geometry_batch.dual_grid()``.
        sdf (JaggedTensor): One SDF value per voxel of the grid.
        ray_origins (JaggedTensor): Ray origins, shape ``(B, -1, 3)``.
        ray_directions (JaggedTensor): Ray directions, shape ``(B, -1, 3)``. Assumed normalized,
            so ``t`` is a world-space distance.
        t_min (float): Earliest accepted crossing. Default ``1e-4``.
        eps (float): Skip cells whose ray segment is shorter than this. Default ``1e-4``.
        refine (str): How a bracketed crossing is refined: ``"bisect"`` (default; 8 halvings
            on the sign of the cell's cubic, to 1/256 of the bracket) or ``"newton"`` (safeguarded
            Newton, to float precision). Both bracket every crossing from the cubic, so a surface
            thinner than a voxel is seen.

    Returns:
        crossing (RaySdfPoint): Fields are :class:`JaggedTensor` rather than ``torch.Tensor``.

    .. seealso:: :func:`ray_sdf_intersection_single`
    """
    times, mask, sdf_vals, grads = _fvdb_cpp.ray_sdf_intersection(
        grid.data,
        ray_origins._impl,
        ray_directions._impl,
        sdf._impl,
        t_min,
        eps,
        _mode(refine, _REFINE_MODES, "refine"),
    )
    return _sdf_point_batch(
        grid,
        sdf,
        ray_origins,
        ray_directions,
        times.jdata[:, 0],
        mask.jdata[:, 0],
        sdf_vals.jdata[:, 0],
        grads.jdata[:, 0, :],
    )


def ray_sdf_grazing_batch(
    grid: GridBatch,
    sdf: JaggedTensor,
    ray_origins: JaggedTensor,
    ray_directions: JaggedTensor,
    relaxation_eps: float,
    t_max: JaggedTensor | None = None,
    ray_mask: JaggedTensor | None = None,
    graze_t_min: float = 1e-4,
    itx_eps: float = 1e-7,
    deriv_eps: float = 1e-1,
    eps: float = 1e-4,
    graze: str = "bisect",
) -> RaySdfPoint:
    """Batched :func:`ray_sdf_grazing_single`.

    Args:
        grid (GridBatch): The grid batch the SDF is stored on: the **dual** of the geometry grid
            batch, e.g. ``geometry_batch.dual_grid()``.
        sdf (JaggedTensor): One SDF value per voxel of the grid.
        ray_origins (JaggedTensor): Ray origins, shape ``(B, -1, 3)``.
        ray_directions (JaggedTensor): Ray directions, shape ``(B, -1, 3)``. Assumed normalized,
            so ``t`` is a world-space distance.
        relaxation_eps (float): Upper edge of the band a grazing point's SDF must lie in.
        t_max (JaggedTensor | None): Optional per-ray search bound, shape ``(B, -1)``, typically
            the first hit. Default ``None`` (unbounded).
        ray_mask (JaggedTensor | None): Optional per-ray boolean enable, shape ``(B, -1)``.
            Default ``None`` (all rays).
        graze_t_min (float): Earliest accepted grazing point, to keep the search off the ray
            origin. Default ``1e-4``.
        itx_eps (float): Lower edge of the band: a smaller SDF counts as a hit, not a graze.
            Default ``1e-7``.
        deriv_eps (float): Tangency tolerance: a grazing point needs
            ``|normalize(grad) . direction|`` below this. Default ``0.1``.
        eps (float): Skip cells whose ray segment is shorter than this. Default ``1e-4``.
        graze (str): How the grazing point is found: ``"bisect"`` (default; brackets the
            minimum between adjacent cell midpoints, bisects 8 times on the derivative's sign, and
            judges tangency from the average of the gradients at the two ends of the final
            bracket -- at a cell face that is the average of the two one-sided slopes, matching
            the relaxed-boundary reference implementation's central-difference gradient) or
            ``"analytic"`` (solves each cell's derivative quadratic exactly, including minima on
            cell faces).

    Returns:
        grazing (RaySdfPoint): Fields are :class:`JaggedTensor` rather than ``torch.Tensor``.

    .. seealso:: :func:`ray_sdf_grazing_single`
    """
    times, mask, sdf_vals, grads = _fvdb_cpp.ray_sdf_grazing(
        grid.data,
        ray_origins._impl,
        ray_directions._impl,
        sdf._impl,
        None if t_max is None else t_max._impl,
        None if ray_mask is None else ray_mask._impl,
        graze_t_min,
        relaxation_eps,
        itx_eps,
        deriv_eps,
        eps,
        _mode(graze, _GRAZE_MODES, "graze"),
    )
    return _sdf_point_batch(
        grid,
        sdf,
        ray_origins,
        ray_directions,
        times.jdata[:, 0],
        mask.jdata[:, 0],
        sdf_vals.jdata[:, 0],
        grads.jdata[:, 0, :],
    )


def ray_sdf_intersection_with_grazing_batch(
    grid: GridBatch,
    sdf: JaggedTensor,
    ray_origins: JaggedTensor,
    ray_directions: JaggedTensor,
    relaxation_eps: float,
    t_min: float = 1e-4,
    graze_t_min: float = 1e-4,
    itx_eps: float = 1e-7,
    deriv_eps: float = 1e-1,
    eps: float = 1e-4,
    refine: str = "bisect",
    graze: str = "bisect",
) -> RaySdfPoints:
    """Batched :func:`ray_sdf_intersection_with_grazing_single`.

    Args:
        grid (GridBatch): The grid batch the SDF is stored on: the **dual** of the geometry grid
            batch, e.g. ``geometry_batch.dual_grid()``.
        sdf (JaggedTensor): One SDF value per voxel of the grid.
        ray_origins (JaggedTensor): Ray origins, shape ``(B, -1, 3)``.
        ray_directions (JaggedTensor): Ray directions, shape ``(B, -1, 3)``. Assumed normalized,
            so ``t`` is a world-space distance.
        relaxation_eps (float): Upper edge of the band a grazing point's SDF must lie in.
        t_min (float): Earliest accepted crossing. Default ``1e-4``.
        graze_t_min (float): Earliest accepted grazing point, to keep the search off the ray
            origin. Default ``1e-4``.
        itx_eps (float): Lower edge of the band: a smaller SDF counts as a hit, not a graze.
            Default ``1e-7``.
        deriv_eps (float): Tangency tolerance: a grazing point needs
            ``|normalize(grad) . direction|`` below this. Default ``0.1``.
        eps (float): Skip cells whose ray segment is shorter than this. Default ``1e-4``.
        refine (str): How a bracketed crossing is refined: ``"bisect"`` (default; 8 halvings
            on the sign of the cell's cubic, to 1/256 of the bracket) or ``"newton"`` (safeguarded
            Newton, to float precision). Both bracket every crossing from the cubic, so a surface
            thinner than a voxel is seen.
        graze (str): How the grazing point is found: ``"bisect"`` (default; brackets the
            minimum between adjacent cell midpoints, bisects 8 times on the derivative's sign, and
            judges tangency from the average of the gradients at the two ends of the final
            bracket -- at a cell face that is the average of the two one-sided slopes, matching
            the relaxed-boundary reference implementation's central-difference gradient) or
            ``"analytic"`` (solves each cell's derivative quadratic exactly, including minima on
            cell faces).

    Returns:
        points (RaySdfPoints): ``.crossing`` and ``.grazing``, whose fields are
        :class:`JaggedTensor` rather than ``torch.Tensor``.

    .. seealso:: :func:`ray_sdf_intersection_with_grazing_single`
    """
    times, mask, sdf_vals, grads = _fvdb_cpp.ray_sdf_intersection_with_grazing(
        grid.data,
        ray_origins._impl,
        ray_directions._impl,
        sdf._impl,
        t_min,
        graze_t_min,
        relaxation_eps,
        itx_eps,
        deriv_eps,
        eps,
        _mode(refine, _REFINE_MODES, "refine"),
        _mode(graze, _GRAZE_MODES, "graze"),
    )
    t, m, s, g = times.jdata, mask.jdata, sdf_vals.jdata, grads.jdata
    return RaySdfPoints(
        crossing=_sdf_point_batch(grid, sdf, ray_origins, ray_directions, t[:, 0], m[:, 0], s[:, 0], g[:, 0, :]),
        grazing=_sdf_point_batch(grid, sdf, ray_origins, ray_directions, t[:, 1], m[:, 1], s[:, 1], g[:, 1, :]),
    )
