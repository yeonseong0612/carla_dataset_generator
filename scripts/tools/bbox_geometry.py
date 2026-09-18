"""
scripts/tools/bbox_geometry.py

Shared, ordering-agnostic 3D bounding-box geometry helpers used by
validate_annotations.py and visualize_annotations.py.

We deliberately do NOT assume a fixed CARLA BoundingBox.get_world_vertices()
vertex index ordering (e.g. the classic PythonAPI example's [0,1,3,2,...]
edge list). Instead, the box's 3 orthogonal axes are recovered directly from
the 8 stored vertices via SVD of the centered point cloud, vertices are
classified onto the +/-/+/- corner of those axes, and edges are derived from
that sign pattern (two corners are connected iff they differ along exactly
one axis). This works regardless of what index order the annotation pipeline
happened to store vertices in, and it comes from the actual data rather than
a guess.
"""

import numpy as np

# Cube corners in +/-1 sign space, connected iff they differ in exactly one
# axis. This is topology, not an index-into-vertices assumption.
_SIGN_CORNERS = [
    (sx, sy, sz)
    for sx in (-1, 1)
    for sy in (-1, 1)
    for sz in (-1, 1)
]


class BoxGeometryError(ValueError):
    pass


def local_box_axes(vertices):
    """
    Given 8x3 vertices (any order), recover the box centroid, its 3
    orthonormal principal axes (rows of a 3x3 matrix), and the projected
    per-axis extents (half-lengths).

    Returns
    -------
    centroid : (3,) ndarray
    axes : (3,3) ndarray, each row is a unit axis direction
    extents : (3,) ndarray, half-length along each axis (>= 0)
    proj : (8,3) ndarray, vertex coordinates in the local axis frame
    """

    v = np.asarray(vertices, dtype=np.float64)

    if v.shape != (8, 3):
        raise BoxGeometryError(f"Expected 8x3 vertices, got {v.shape}")

    if not np.all(np.isfinite(v)):
        raise BoxGeometryError("Non-finite vertex coordinates")

    centroid = v.mean(axis=0)
    centered = v - centroid

    try:
        _, _, vt = np.linalg.svd(centered)
    except np.linalg.LinAlgError as exc:
        raise BoxGeometryError(f"SVD failed: {exc}") from exc

    axes = vt  # 3x3, rows are principal directions

    proj = centered @ axes.T  # 8x3

    extents = np.abs(proj).max(axis=0)

    return centroid, axes, extents, proj


def derive_edges_from_vertices(vertices, sign_tol_ratio=0.35):
    """
    Derive the 12 box edges (as pairs of indices into `vertices`) purely
    from geometry, without assuming CARLA's internal vertex ordering.

    Each of the 8 vertices is classified by its sign along the 3 principal
    axes (+/-/+/-...), matched greedily to the nearest unused vertex for
    each of the 8 expected sign-corners, then edges are the corner pairs
    that differ along exactly one axis (12 for a proper box).

    Returns
    -------
    edges : list of (i, j) index pairs into the input `vertices` array
    extents : (3,) ndarray of half-lengths along the recovered axes
    """

    centroid, axes, extents, proj = local_box_axes(vertices)

    if np.any(extents <= 1e-9):
        raise BoxGeometryError(f"Degenerate box extents: {extents}")

    # Normalize projected coordinates to roughly +/-1 per axis.
    norm_proj = proj / extents[None, :]

    used = set()
    corner_to_index = {}

    for corner in _SIGN_CORNERS:
        target = np.array(corner, dtype=np.float64)

        best_idx = None
        best_dist = None

        for i in range(8):
            if i in used:
                continue

            dist = np.linalg.norm(norm_proj[i] - target)

            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx is None or best_dist > sign_tol_ratio * np.sqrt(3):
            raise BoxGeometryError(
                "Vertices do not form a well-defined rectangular box "
                f"(closest match for corner {corner} had normalized "
                f"distance {best_dist})"
            )

        used.add(best_idx)
        corner_to_index[corner] = best_idx

    edges = []

    for a in range(len(_SIGN_CORNERS)):
        for b in range(a + 1, len(_SIGN_CORNERS)):
            ca = _SIGN_CORNERS[a]
            cb = _SIGN_CORNERS[b]

            diff = sum(1 for k in range(3) if ca[k] != cb[k])

            if diff == 1:
                edges.append(
                    (corner_to_index[ca], corner_to_index[cb])
                )

    if len(edges) != 12:
        raise BoxGeometryError(
            f"Expected 12 edges from box topology, derived {len(edges)}"
        )

    return edges, extents
