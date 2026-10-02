"""Relation-driven hypothesis proposers built from the Atlas DSL.

Owner: Atlas

These proposers intentionally cover only compact, reusable rule families.
Every emitted candidate is deterministic; the verifier remains responsible
for accepting only exact fits across every training pair.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from . import dsl
from .dsl import Transform
from .scene import Scene


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """A candidate program plus search metadata."""

    transform: Transform
    description: str
    confidence: float = 0.0
    complexity: int = 1
    proposer: str = "unknown"

    @property
    def program_signature(self) -> tuple:
        return dsl.signature(self.transform)

    def __repr__(self) -> str:
        return f"Hypothesis({self.description}, conf={self.confidence:.2f})"


class Proposer:
    """Base class for hypothesis proposers.

    ``tier`` follows ``search.ProposerTier``: 0 fast, 1 medium, 2 deep,
    3 exotic.
    """

    tier: int = 1

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        raise NotImplementedError


def _paired(inputs: list[Scene], outputs: list[Scene] | None) -> bool:
    return outputs is not None and bool(inputs) and len(inputs) == len(outputs)


def _exact(transform: Transform, inputs: list[Scene], outputs: list[Scene]) -> bool:
    try:
        return all(
            np.array_equal(transform(source.grid), target.grid)
            for source, target in zip(inputs, outputs)
        )
    except Exception:
        return False


def _infer_recolor(
    source_grids: list[np.ndarray],
    outputs: list[Scene],
) -> dict[int, int] | None:
    """Infer one position-wise color map shared by every training pair."""

    mapping: dict[int, int] = {}
    for source, output in zip(source_grids, outputs):
        if source.shape != output.grid.shape:
            return None
        for old, new in zip(source.flat, output.grid.flat):
            old_i, new_i = int(old), int(new)
            previous = mapping.setdefault(old_i, new_i)
            if previous != new_i:
                return None
    return mapping


def _deduplicate(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    result: list[Hypothesis] = []
    seen: set[tuple] = set()
    for hypothesis in hypotheses:
        key = hypothesis.program_signature
        if key not in seen:
            seen.add(key)
            result.append(hypothesis)
    return result


def _infer_color_block_mapping(
    source: np.ndarray,
    target: np.ndarray,
) -> dict[int, np.ndarray] | None:
    """Infer a fixed output block for every color in one source grid."""

    source = np.asarray(source, dtype=int)
    target = np.asarray(target, dtype=int)
    if target.shape[0] % source.shape[0] or target.shape[1] % source.shape[1]:
        return None
    block_height = target.shape[0] // source.shape[0]
    block_width = target.shape[1] // source.shape[1]
    if block_height < 1 or block_width < 1 or (block_height, block_width) == (1, 1):
        return None

    mapping: dict[int, np.ndarray] = {}
    for row in range(source.shape[0]):
        for column in range(source.shape[1]):
            color = int(source[row, column])
            block = target[
                row * block_height:(row + 1) * block_height,
                column * block_width:(column + 1) * block_width,
            ]
            previous = mapping.get(color)
            if previous is not None and not np.array_equal(previous, block):
                return None
            mapping[color] = block.copy()
    return mapping


def _neighbor_kernel(connectivity: int) -> np.ndarray:
    """Return the foreground-neighbor kernel for 4- or 8-connectivity."""

    if connectivity == 4:
        return np.asarray(
            ((0, 1, 0), (1, 0, 1), (0, 1, 0)),
            dtype=int,
        )
    if connectivity == 8:
        return np.asarray(
            ((1, 1, 1), (1, 0, 1), (1, 1, 1)),
            dtype=int,
        )
    raise ValueError("connectivity must be 4 or 8")


def _foreground_neighbor_counts(grid: np.ndarray, connectivity: int) -> np.ndarray:
    """Count non-background neighbors around every cell in ``grid``."""

    source = np.asarray(grid, dtype=int)
    background = Counter(
        int(value) for value in source.flat
    ).most_common(1)[0][0]
    foreground = (source != background).astype(int)
    return ndimage.convolve(
        foreground,
        _neighbor_kernel(connectivity),
        mode="constant",
        cval=0,
    )


def _infer_neighbor_color_mapping(
    inputs: list[Scene],
    outputs: list[Scene],
    connectivity: int,
) -> dict[tuple[int, int], int] | None:
    """Infer one joint color/count mapping across every training pair."""

    mapping: dict[tuple[int, int], int] = {}
    changed = False
    for source, target in zip(inputs, outputs):
        if source.grid.shape != target.grid.shape:
            return None
        counts = _foreground_neighbor_counts(source.grid, connectivity)
        for old, count, new in zip(source.grid.flat, counts.flat, target.grid.flat):
            key = (int(old), int(count))
            value = int(new)
            previous = mapping.setdefault(key, value)
            if previous != value:
                return None
            changed = changed or int(old) != value
    return mapping if mapping and changed else None


def _neighbor_color_transform(
    mapping: dict[tuple[int, int], int],
    connectivity: int,
) -> Transform:
    """Build a stable transform from a learned neighbor-color mapping."""

    frozen_mapping = tuple(
        (color, count, output)
        for (color, count), output in sorted(mapping.items())
    )

    def transform(grid: np.ndarray) -> np.ndarray:
        source = np.asarray(grid, dtype=int)
        counts = _foreground_neighbor_counts(source, connectivity)
        result = source.copy()
        for index in np.ndindex(source.shape):
            key = (int(source[index]), int(counts[index]))
            result[index] = mapping.get(key, int(source[index]))
        return result

    transform.__name__ = "neighbor_color_recolor"
    transform.__qualname__ = "neighbor_color_recolor"
    transform._arc2_signature = (  # type: ignore[attr-defined]
        "neighbor_color_recolor",
        connectivity,
        frozen_mapping,
    )
    return transform


class GeometricProposer(Proposer):
    """Enumerate single-step D4 symmetries."""

    tier = 0

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        candidates = [
            Hypothesis(transform, name.replace("_", " "), 0.96, 1, "geometric")
            for name, transform in dsl.d4_transforms().items()
        ]
        if _paired(inputs, outputs):
            assert outputs is not None
            candidates = [h for h in candidates if _exact(h.transform, inputs, outputs)]
        return _deduplicate(candidates)


class ColorMapProposer(Proposer):
    """Infer a global recolor, optionally after a D4 transform."""

    tier = 0

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        if not _paired(inputs, outputs):
            return []
        assert outputs is not None
        candidates: list[Hypothesis] = []
        for name, base in dsl.d4_transforms().items():
            try:
                transformed = [base(scene.grid) for scene in inputs]
            except Exception:
                continue
            mapping = _infer_recolor(transformed, outputs)
            if mapping is None:
                continue
            recolor = dsl.recolor(mapping)
            transform = recolor if name == "identity" else dsl.compose(base, recolor)
            mapping_text = ", ".join(f"{old}->{new}" for old, new in sorted(mapping.items()))
            description = f"recolor {mapping_text}"
            if name != "identity":
                description = f"{name.replace('_', ' ')} then {description}"
            complexity = 1 if name == "identity" else 2
            candidate = Hypothesis(transform, description, 0.94, complexity, "color-map")
            if _exact(candidate.transform, inputs, outputs):
                candidates.append(candidate)
        return _deduplicate(candidates)


class ResizeProposer(Proposer):
    """Infer integer tiling or nearest-neighbor scaling, with recoloring."""

    tier = 0

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        if not _paired(inputs, outputs):
            return []
        assert outputs is not None

        factors: tuple[int, int] | None = None
        for source, target in zip(inputs, outputs):
            if target.height % source.height or target.width % source.width:
                return []
            current = (target.height // source.height, target.width // source.width)
            if current[0] < 1 or current[1] < 1:
                return []
            if factors is None:
                factors = current
            elif factors != current:
                return []
        assert factors is not None

        candidates: list[Hypothesis] = []
        for name, base in (
            ("tile", dsl.tile(*factors)),
            ("scale", dsl.scale(*factors)),
        ):
            description = f"{name} {factors[0]}x{factors[1]}"
            direct = Hypothesis(base, description, 0.92, 1, "resize")
            if _exact(base, inputs, outputs):
                candidates.append(direct)

            resized = [base(scene.grid) for scene in inputs]
            mapping = _infer_recolor(resized, outputs)
            if mapping is None or all(old == new for old, new in mapping.items()):
                continue
            recolor = dsl.recolor(mapping)
            combined = dsl.compose(recolor, base)
            mapping_text = ", ".join(f"{old}->{new}" for old, new in sorted(mapping.items()))
            candidate = Hypothesis(
                combined,
                f"recolor {mapping_text} then {description}",
                0.90,
                2,
                "resize",
            )
            if _exact(combined, inputs, outputs):
                candidates.append(candidate)
        return _deduplicate(candidates)


class CropProposer(Proposer):
    """Crop to the inferred foreground bounding box."""

    tier = 0

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        candidate = Hypothesis(
            dsl.crop_non_background,
            "crop non-background bounding box",
            0.90,
            1,
            "crop",
        )
        if _paired(inputs, outputs):
            assert outputs is not None
            return [candidate] if _exact(candidate.transform, inputs, outputs) else []
        return [candidate]


class LargestComponentProposer(Proposer):
    """Select all maximum-area components and render their color histogram."""

    tier = 1

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        candidates: list[Hypothesis] = []
        for order in (
            "left_to_right",
            "right_to_left",
            "top_to_bottom",
            "bottom_to_top",
        ):
            for layout in ("rows", "columns"):
                transform = dsl.largest_component_histogram(order, layout)
                candidate = Hypothesis(
                    transform,
                    f"largest components {order.replace('_', ' ')} as {layout}",
                    0.86,
                    2,
                    "largest-component",
                )
                if not _paired(inputs, outputs):
                    candidates.append(candidate)
                elif outputs is not None and _exact(transform, inputs, outputs):
                    candidates.append(candidate)
        return _deduplicate(candidates)


class RoutedRibbonProposer(Proposer):
    """Infer a singleton-anchored diagonal ribbon that routes around obstacles."""

    tier = 1

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        if not inputs:
            return []
        anchor_colors = {
            color
            for color in inputs[0].colors
            if inputs[0].color_counts[color] == 1
            and color != inputs[0].background_color()
        }
        for scene in inputs[1:]:
            anchor_colors &= {
                color
                for color in scene.colors
                if scene.color_counts[color] == 1 and color != scene.background_color()
            }

        candidates: list[Hypothesis] = []
        for anchor_color in sorted(anchor_colors):
            for row_direction in (-1, 1):
                for column_direction in (-1, 1):
                    for width in (1, 2, 3):
                        transform = dsl.routed_ribbon(
                            anchor_color,
                            row_direction,
                            column_direction,
                            width,
                        )
                        candidate = Hypothesis(
                            transform,
                            (
                                f"routed ribbon color {anchor_color} "
                                f"direction {row_direction},{column_direction} width {width}"
                            ),
                            0.82,
                            3,
                            "routed-ribbon",
                        )
                        if not _paired(inputs, outputs):
                            candidates.append(candidate)
                        elif outputs is not None and _exact(transform, inputs, outputs):
                            candidates.append(candidate)
        return _deduplicate(candidates)


class NestedRectangleProposer(Proposer):
    """Compress nested rectangular regions into a miniature ring diagram."""

    tier = 3

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        transform = dsl.nested_rectangle_miniature()
        candidate = Hypothesis(
            transform,
            "compress nested rectangles to one-cell rings",
            0.78,
            3,
            "nested-rectangle",
        )
        if _paired(inputs, outputs):
            assert outputs is not None
            return [candidate] if _exact(transform, inputs, outputs) else []
        return [candidate]


class D4OrbitProposer(Proposer):
    """Fill a masked color by consensus over each dihedral orbit."""

    tier = 3

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        colors = sorted({color for scene in inputs for color in scene.colors})
        candidates: list[Hypothesis] = []
        for hole_color in colors:
            transform = dsl.d4_orbit_consensus(hole_color)
            candidate = Hypothesis(
                transform,
                f"fill color {hole_color} by D4 orbit consensus",
                0.76,
                3,
                "d4-orbit",
            )
            if not _paired(inputs, outputs):
                candidates.append(candidate)
            elif outputs is not None and _exact(transform, inputs, outputs):
                candidates.append(candidate)
        return _deduplicate(candidates)


class FractalProposer(Proposer):
    """Reconstruct a self-similar motif lattice and mark each copy."""

    tier = 3

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        colors = (
            sorted({color for scene in outputs for color in scene.colors})
            if outputs is not None
            else list(range(10))
        )
        candidates: list[Hypothesis] = []
        for marker_color in colors:
            transform = dsl.self_similar_stamp(marker_color)
            candidate = Hypothesis(
                transform,
                f"self-similar motif lattice marked with color {marker_color}",
                0.74,
                4,
                "fractal",
            )
            if not _paired(inputs, outputs):
                candidates.append(candidate)
            elif outputs is not None and _exact(transform, inputs, outputs):
                candidates.append(candidate)
        return _deduplicate(candidates)


class AnalogyProposer(Proposer):
    """Infer local A-to-B rules and apply them to neighboring C blocks."""

    tier = 3

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        transform = dsl.block_analogy()
        candidate = Hypothesis(
            transform,
            "apply demonstrated block analogies",
            0.72,
            4,
            "analogy",
        )
        if _paired(inputs, outputs):
            assert outputs is not None
            return [candidate] if _exact(transform, inputs, outputs) else []
        return [candidate]


class PartitionBlueprintProposer(Proposer):
    """Expand a uniquely incomplete subgrid as a lattice blueprint."""

    tier = 3

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        transform = dsl.partition_blueprint_expand()
        candidate = Hypothesis(
            transform,
            "expand incomplete partition block as lattice blueprint",
            0.72,
            4,
            "partition-blueprint",
        )
        if _paired(inputs, outputs):
            assert outputs is not None
            return [candidate] if _exact(transform, inputs, outputs) else []
        return [candidate]


class PartitionMaxCountProposer(Proposer):
    """Fill partition blocks tied for the largest foreground count."""

    tier = 3

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        transform = dsl.partition_max_count_fill(2)
        candidate = Hypothesis(
            transform,
            "fill partition blocks with maximum foreground count",
            0.72,
            4,
            "partition-max-count",
        )
        if _paired(inputs, outputs):
            assert outputs is not None
            return [candidate] if _exact(transform, inputs, outputs) else []
        return [candidate]


class PartitionMarkerRouteProposer(Proposer):
    """Route a template block to the lattice coordinate encoded by a marker."""

    tier = 3

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        if not inputs:
            return []
        colors = set(inputs[0].colors)
        for scene in inputs[1:]:
            colors &= set(scene.colors)
        candidates: list[Hypothesis] = []
        for marker_color in sorted(colors):
            transform = dsl.partition_marker_route(marker_color)
            candidate = Hypothesis(
                transform,
                f"route marked partition block using color {marker_color}",
                0.70,
                4,
                "partition-marker-route",
            )
            if not _paired(inputs, outputs):
                candidates.append(candidate)
            elif outputs is not None and _exact(transform, inputs, outputs):
                candidates.append(candidate)
        return _deduplicate(candidates)


class PartitionAnchorRelocateProposer(Proposer):
    """Relocate a panel object so a bbox corner meets a singleton anchor."""

    tier = 3

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        if not inputs:
            return []
        colors = set(inputs[0].colors)
        for scene in inputs[1:]:
            colors &= set(scene.colors)
        candidates: list[Hypothesis] = []
        for anchor_color in sorted(colors):
            for corner in ("top_left", "top_right", "bottom_left", "bottom_right"):
                transform = dsl.partition_anchor_relocate(anchor_color, corner)
                candidate = Hypothesis(
                    transform,
                    f"relocate panel object {corner} to anchor color {anchor_color}",
                    0.68,
                    5,
                    "partition-anchor-relocate",
                )
                if not _paired(inputs, outputs):
                    candidates.append(candidate)
                elif outputs is not None and _exact(transform, inputs, outputs):
                    candidates.append(candidate)
        return _deduplicate(candidates)


class FramedObjectProposer(Proposer):
    """Infer object, outer-frame, and enclosed-hole colors."""

    tier = 3

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        if not _paired(inputs, outputs):
            return []
        assert outputs is not None
        object_colors = set(inputs[0].colors) - {inputs[0].background_color()}
        for scene in inputs[1:]:
            object_colors &= set(scene.colors) - {scene.background_color()}
        introduced = set(outputs[0].colors) - set(inputs[0].colors)
        for source, target in zip(inputs[1:], outputs[1:]):
            introduced &= set(target.colors) - set(source.colors)

        candidates: list[Hypothesis] = []
        for object_color in sorted(object_colors):
            for frame_color in sorted(introduced):
                for hole_color in sorted(introduced - {frame_color}):
                    transform = dsl.frame_and_fill_objects(
                        object_color,
                        frame_color,
                        hole_color,
                    )
                    candidate = Hypothesis(
                        transform,
                        (
                            f"frame color {object_color} with {frame_color} "
                            f"and fill holes with {hole_color}"
                        ),
                        0.70,
                        4,
                        "framed-object",
                    )
                    if _exact(transform, inputs, outputs):
                        candidates.append(candidate)
        return _deduplicate(candidates)


class AxialCrossProposer(Proposer):
    """Infer full row-and-column marker crosses with a collision color."""

    tier = 3

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        if not _paired(inputs, outputs):
            return []
        assert outputs is not None
        introduced = set(outputs[0].colors) - set(inputs[0].colors)
        for source, target in zip(inputs[1:], outputs[1:]):
            introduced &= set(target.colors) - set(source.colors)
        candidates: list[Hypothesis] = []
        for collision_color in sorted(introduced):
            transform = dsl.axial_cross_lines(collision_color)
            candidate = Hypothesis(
                transform,
                f"draw axial marker crosses with collision color {collision_color}",
                0.72,
                4,
                "axial-cross",
            )
            if _exact(transform, inputs, outputs):
                candidates.append(candidate)
        return _deduplicate(candidates)


class ClosestHorizontalPairProposer(Proposer):
    """Connect rows tied for the smallest gap between equal-color endpoints."""

    tier = 3

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        transform = dsl.connect_closest_horizontal_pairs()
        candidate = Hypothesis(
            transform,
            "connect horizontal endpoint pairs with the smallest gap",
            0.70,
            4,
            "closest-horizontal-pair",
        )
        if _paired(inputs, outputs):
            assert outputs is not None
            return [candidate] if _exact(transform, inputs, outputs) else []
        return [candidate]


class MarkerFrameProposer(Proposer):
    """Infer a clipped square expansion around singleton markers."""

    tier = 3

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        if not _paired(inputs, outputs):
            return []
        assert outputs is not None
        marker_colors = set(inputs[0].colors) - {inputs[0].background_color()}
        introduced = set(outputs[0].colors) - set(inputs[0].colors)
        for source, target in zip(inputs[1:], outputs[1:]):
            marker_colors &= set(source.colors) - {source.background_color()}
            introduced &= set(target.colors) - set(source.colors)
        candidates: list[Hypothesis] = []
        for marker_color in sorted(marker_colors):
            for frame_color in sorted(introduced):
                transform = dsl.frame_around_markers(marker_color, frame_color, radius=1)
                candidate = Hypothesis(
                    transform,
                    f"expand marker color {marker_color} with a clipped 3x3 frame {frame_color}",
                    0.72,
                    3,
                    "marker-frame",
                )
                if _exact(transform, inputs, outputs):
                    candidates.append(candidate)
        return _deduplicate(candidates)


class PeriodicStripeProposer(Proposer):
    """Extend two boundary markers as alternating periodic stripes."""

    tier = 3

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        transform = dsl.periodic_boundary_stripes()
        candidate = Hypothesis(
            transform,
            "extend boundary markers as alternating periodic stripes",
            0.70,
            4,
            "periodic-stripe",
        )
        if _paired(inputs, outputs):
            assert outputs is not None
            return [candidate] if _exact(transform, inputs, outputs) else []
        return [candidate]


class OrientedMarkerLineProposer(Proposer):
    """Infer which marker colors draw rows or columns and their precedence."""

    tier = 3

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        if not _paired(inputs, outputs):
            return []
        assert outputs is not None
        colors = sorted({
            color
            for scene in inputs
            for color in scene.colors
            if color != scene.background_color()
        })
        if not colors or len(colors) > 6:
            return []
        candidates: list[Hypothesis] = []
        for mask in range(1 << len(colors)):
            vertical = {color for index, color in enumerate(colors) if mask & (1 << index)}
            horizontal = set(colors) - vertical
            for horizontal_overwrites in (True, False):
                transform = dsl.oriented_marker_lines(
                    vertical,
                    horizontal,
                    horizontal_overwrites,
                )
                candidate = Hypothesis(
                    transform,
                    (
                        f"draw vertical colors {sorted(vertical)} and horizontal colors "
                        f"{sorted(horizontal)} with "
                        f"{'row' if horizontal_overwrites else 'column'} precedence"
                    ),
                    0.68,
                    5,
                    "oriented-marker-lines",
                )
                if _exact(transform, inputs, outputs):
                    candidates.append(candidate)
        return _deduplicate(candidates)


class WallpaperRepairProposer(Proposer):
    """Repair marker-column bands or damaged periodic wallpaper."""

    tier = 3

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        if not _paired(inputs, outputs):
            return []
        assert outputs is not None
        if any(source.grid.shape != target.grid.shape for source, target in zip(inputs, outputs)):
            return []

        candidates: list[Hypothesis] = []
        terminal_colors = set(int(value) for value in np.unique(inputs[0].grid[-1]))
        for scene in inputs[1:]:
            terminal_colors &= set(int(value) for value in np.unique(scene.grid[-1]))
        terminal_colors -= {scene.background_color() for scene in inputs}
        for terminal_color in sorted(terminal_colors):
            transform = dsl.marker_column_bands(terminal_color)
            candidate = Hypothesis(
                transform,
                f"fill marker-column bands ending in color {terminal_color}",
                0.70,
                4,
                "wallpaper-repair",
            )
            if _exact(transform, inputs, outputs):
                candidates.append(candidate)

        changed_colors: set[int] | None = None
        for source, target in zip(inputs, outputs):
            mask = source.grid != target.grid
            current = set(int(value) for value in np.unique(source.grid[mask]))
            changed_colors = current if changed_colors is None else changed_colors & current
        for hole_color in sorted(changed_colors or set()):
            transform = dsl.periodic_pattern_repair(hole_color)
            candidate = Hypothesis(
                transform,
                f"repair color {hole_color} from the smallest periodic wallpaper",
                0.72,
                5,
                "wallpaper-repair",
            )
            if _exact(transform, inputs, outputs):
                candidates.append(candidate)
        return _deduplicate(candidates)


class SymmetryCompleterProposer(Proposer):
    """Complete reflected block quartets or periodic boundary symmetry."""

    tier = 3

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        transforms = (
            (
                dsl.reflection_block_repair(),
                "complete damaged 2x2 block quartets by reflection",
                5,
            ),
            (
                dsl.periodic_perimeter_segments(),
                "repeat equal on/off segments around the perimeter",
                4,
            ),
        )
        candidates: list[Hypothesis] = []
        for transform, description, complexity in transforms:
            candidate = Hypothesis(
                transform,
                description,
                0.70,
                complexity,
                "symmetry-completer",
            )
            if not _paired(inputs, outputs):
                candidates.append(candidate)
            elif outputs is not None and _exact(transform, inputs, outputs):
                candidates.append(candidate)
        return _deduplicate(candidates)


class ShapeUnifierProposer(Proposer):
    """Replace anomalous object colors from exact same-shape exemplars."""

    tier = 3

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        if not _paired(inputs, outputs):
            return []
        assert outputs is not None
        changed_colors: set[int] | None = None
        for source, target in zip(inputs, outputs):
            if source.grid.shape != target.grid.shape:
                return []
            mask = source.grid != target.grid
            current = set(int(value) for value in np.unique(source.grid[mask]))
            changed_colors = current if changed_colors is None else changed_colors & current

        candidates: list[Hypothesis] = []
        for anomaly_color in sorted(changed_colors or set()):
            transform = dsl.unify_shape_colors(anomaly_color)
            candidate = Hypothesis(
                transform,
                f"recolor anomalous color {anomaly_color} from same-shape exemplars",
                0.72,
                4,
                "shape-unifier",
            )
            if _exact(transform, inputs, outputs):
                candidates.append(candidate)
        return _deduplicate(candidates)


class SparseSymmetryRepairProposer(Proposer):
    """Repair sparse rectangle, boundary, legend, ray, and duplicate anomalies."""

    tier = 3

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        if not _paired(inputs, outputs):
            return []
        assert outputs is not None
        if any(source.grid.shape != target.grid.shape for source, target in zip(inputs, outputs)):
            return []

        candidates: list[Hypothesis] = []
        fixed = (
            (
                dsl.complete_occluded_rectangles(),
                "complete solid rectangles occluded by other colors",
                4,
                "occluded-rectangles",
            ),
            (
                dsl.directed_marker_ray_cleanup(),
                "clear the first border intrusion opposite a marker-attached line",
                5,
                "marker-ray-cleanup",
            ),
            (
                dsl.legend_sample_erase(),
                "erase external instances of the L-legend sample color",
                4,
                "legend-sample-erase",
            ),
        )
        for transform, description, complexity, family in fixed:
            if _exact(transform, inputs, outputs):
                candidates.append(Hypothesis(transform, description, 0.74, complexity, family))

        shared_colors = set(int(value) for value in np.unique(inputs[0].grid))
        for scene in inputs[1:]:
            shared_colors &= set(int(value) for value in np.unique(scene.grid))
        shared_backgrounds = {scene.background_color() for scene in inputs}
        for foreground in sorted(shared_colors - shared_backgrounds):
            transform = dsl.opposite_edge_anomaly_pairs(foreground)
            if _exact(transform, inputs, outputs):
                candidates.append(Hypothesis(
                    transform,
                    f"complete opposite-edge dent/bulge pairs in color {foreground}",
                    0.76,
                    6,
                    "opposite-edge-anomalies",
                ))

        introduced: set[int] | None = None
        changed_sources: set[int] | None = None
        for source, target in zip(inputs, outputs):
            new_colors = set(int(value) for value in np.unique(target.grid)) - set(
                int(value) for value in np.unique(source.grid)
            )
            introduced = new_colors if introduced is None else introduced & new_colors
            changed = source.grid != target.grid
            old_colors = set(int(value) for value in np.unique(source.grid[changed]))
            changed_sources = old_colors if changed_sources is None else changed_sources & old_colors
        for result_color in sorted(introduced or set()):
            for mask_color in sorted(changed_sources or set()):
                references = shared_colors - shared_backgrounds - {result_color, mask_color}
                for reference_color in sorted(references):
                    transform = dsl.duplicate_template_refine(
                        result_color,
                        mask_color,
                        reference_color,
                    )
                    if _exact(transform, inputs, outputs):
                        candidates.append(Hypothesis(
                            transform,
                            (
                                f"trim color {mask_color} facing reference {reference_color} "
                                f"and mark its edge color {result_color}"
                            ),
                            0.72,
                            6,
                            "duplicate-template-refine",
                        ))
        return _deduplicate(candidates)


class FillFromBackgroundProposer(Proposer):
    """Complete contextual patterns by changing background cells only."""

    tier = 3

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        if not _paired(inputs, outputs):
            return []
        assert outputs is not None

        changed_targets: set[int] | None = None
        for source, target in zip(inputs, outputs):
            if source.grid.shape != target.grid.shape:
                return []
            changed = source.grid != target.grid
            if not np.any(changed) or np.any(source.grid[changed] != 0):
                return []
            colors = set(int(value) for value in np.unique(target.grid[changed]))
            changed_targets = colors if changed_targets is None else changed_targets & colors

        candidates: list[Hypothesis] = []
        fixed = (
            (
                dsl.complete_d4_orbits(0),
                "complete missing cells in same-color D4 orbits",
                4,
                "d4-orbit-fill",
            ),
            (
                dsl.mode_below_separator(0),
                "place the modal upper color below a full separator",
                3,
                "separator-mode-fill",
            ),
            (
                dsl.complete_local_symmetric_templates(0),
                "complete isolated local symmetric template copies",
                6,
                "local-template-fill",
            ),
        )
        for transform, description, complexity, family in fixed:
            if _exact(transform, inputs, outputs):
                candidates.append(Hypothesis(transform, description, 0.75, complexity, family))

        shared_colors = set(int(value) for value in np.unique(inputs[0].grid))
        for scene in inputs[1:]:
            shared_colors &= set(int(value) for value in np.unique(scene.grid))
        marker_colors = shared_colors - {0}
        for fill_color in sorted(changed_targets or set()):
            for marker_color in sorted(marker_colors - {fill_color}):
                transform = dsl.extend_diagonal_lattice(marker_color, fill_color, 0)
                if _exact(transform, inputs, outputs):
                    candidates.append(Hypothesis(
                        transform,
                        f"extend diagonal marker {marker_color} lattice with {fill_color}",
                        0.78,
                        4,
                        "diagonal-lattice-fill",
                    ))

            transform = dsl.complete_solid_template(fill_color, 0)
            if _exact(transform, inputs, outputs):
                candidates.append(Hypothesis(
                    transform,
                    f"complete partial solid templates with color {fill_color}",
                    0.76,
                    5,
                    "solid-template-fill",
                ))
        return _deduplicate(candidates)


class FixedSmallOutputProposer(Proposer):
    """Extract compact classifications, position maps, and count summaries."""

    tier = 3

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        if not _paired(inputs, outputs):
            return []
        assert outputs is not None
        shapes = {tuple(output.grid.shape) for output in outputs}
        if len(shapes) != 1:
            return []
        shape = next(iter(shapes))
        if shape[0] not in range(1, 4) or shape[1] not in range(1, 4):
            return []

        candidates: list[Hypothesis] = []
        if shape == (1, 1):
            transform = dsl.extract_full_span_color()
            if _exact(transform, inputs, outputs):
                candidates.append(Hypothesis(
                    transform,
                    "classify by the unique uninterrupted full-span color",
                    0.79,
                    3,
                    "full-span-classifier",
                ))

        if shape == (2, 2):
            transform = dsl.quadrant_object_map()
            if _exact(transform, inputs, outputs):
                candidates.append(Hypothesis(
                    transform,
                    "map embedded 2x2 objects to canvas quadrants",
                    0.78,
                    5,
                    "quadrant-object-map",
                ))

        output_colors = sorted({
            int(value)
            for output in outputs
            for value in np.unique(output.grid)
        })
        if shape == (3, 3):
            for output_background in output_colors:
                for output_color in output_colors:
                    if output_color == output_background:
                        continue
                    transforms = (
                        (
                            dsl.encode_cell_count_3x3(output_color, output_background),
                            "encode foreground-cell count on the top-and-center path",
                            4,
                            "cell-count-code",
                        ),
                        (
                            dsl.encode_square_component_count_3x3(
                                output_color,
                                output_background,
                            ),
                            "encode solid 2x2 component count on a five-cell path",
                            5,
                            "square-component-code",
                        ),
                    )
                    for transform, description, complexity, family in transforms:
                        if _exact(transform, inputs, outputs):
                            candidates.append(Hypothesis(
                                transform,
                                description,
                                0.77,
                                complexity,
                                family,
                            ))

        for output_background in output_colors:
            transform = dsl.component_segment_histogram(
                shape[0],
                shape[1],
                output_background,
            )
            if _exact(transform, inputs, outputs):
                candidates.append(Hypothesis(
                    transform,
                    "pack colors by descending horizontal-segment counts",
                    0.76,
                    5,
                    "segment-count-histogram",
                ))
        return _deduplicate(candidates)


class PixelExpansionProposer(Proposer):
    """Expand input pixels into input-sized, mirrored, or learned blocks."""

    tier = 3

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        if not _paired(inputs, outputs):
            return []
        assert outputs is not None

        candidates: list[Hypothesis] = []

        # Strategy A: a semantic or literal key expands to the complete input.
        selectors: list[tuple[str, int | None, str]] = [
            ("nonzero", None, "sole non-zero color"),
            ("most_frequent", None, "unique most-frequent color"),
            ("least_frequent", None, "unique least-frequent color"),
            ("full_line", None, "color spanning a complete row or column"),
        ]
        selectors.extend(
            ("literal", color, f"color {color}")
            for color in range(10)
        )
        for selector, key_color, label in selectors:
            transform = dsl.pixel_self_substitute(selector, key_color)
            if _exact(transform, inputs, outputs):
                candidates.append(Hypothesis(
                    transform,
                    f"expand each {label} pixel to a copy of the input",
                    0.82,
                    4,
                    "pixel-fractal-self-sub",
                ))

        # Strategy B: binary key pixels expand to a complemented input.
        for selector, key_color, label in selectors:
            transform = dsl.pixel_complement_substitute(selector, key_color)
            if _exact(transform, inputs, outputs):
                candidates.append(Hypothesis(
                    transform,
                    f"expand each {label} pixel to the binary complement",
                    0.81,
                    5,
                    "pixel-complement-sub",
                ))

        # Strategy C: four possible corners may anchor the original grid.
        for anchor in ("top_left", "bottom_right", "top_right", "bottom_left"):
            transform = dsl.mirror4_tile(anchor)
            if _exact(transform, inputs, outputs):
                candidates.append(Hypothesis(
                    transform,
                    f"mirror the input into four quadrants anchored {anchor}",
                    0.83,
                    2,
                    "mirror4-tile",
                ))

        # Strategy D: learn color-to-block substitution from pair zero, then
        # require that exact mapping to generalize across every later pair.
        mapping = _infer_color_block_mapping(inputs[0].grid, outputs[0].grid)
        if mapping is not None:
            transform = dsl.color_block_substitute(mapping)
            if _exact(transform, inputs, outputs):
                candidates.append(Hypothesis(
                    transform,
                    "replace colors with consistently learned output blocks",
                    0.80,
                    5,
                    "learned-color-block",
                ))

        return _deduplicate(candidates)


class MiscTransformProposer(Proposer):
    """Try compact same-size gravity, marker-connection, and shift rules."""

    tier = 1

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        if not _paired(inputs, outputs):
            return []
        assert outputs is not None
        if any(source.grid.shape != target.grid.shape for source, target in zip(inputs, outputs)):
            return []

        candidates: list[Hypothesis] = []

        for direction in ("down", "up", "left", "right"):
            transform = dsl.gravity(direction)
            if _exact(transform, inputs, outputs):
                candidates.append(Hypothesis(
                    transform,
                    f"pack foreground pixels toward the {direction} edge",
                    0.84,
                    2,
                    "gravity",
                ))

        for mode, label in (
            ("horizontal", "horizontally"),
            ("vertical", "vertically"),
            ("horizontal_vertical", "horizontally then vertically"),
        ):
            transform = dsl.connect_same_color(mode)
            if _exact(transform, inputs, outputs):
                candidates.append(Hypothesis(
                    transform,
                    f"connect matching color markers {label}",
                    0.83,
                    2 if mode != "horizontal_vertical" else 3,
                    "same-color-connect",
                ))

        for axis in ("row", "column"):
            for shift in (1, -1, 2, -2):
                transform = dsl.diagonal_shift(axis, shift)
                if _exact(transform, inputs, outputs):
                    candidates.append(Hypothesis(
                        transform,
                        f"circularly shift each {axis} by {shift} times its index",
                        0.82,
                        2,
                        "diagonal-shift",
                    ))

        return _deduplicate(candidates)


class ComponentBBoxProposer(Proposer):
    """Crop to a connected component selected by a compact structural rule."""

    tier = 1

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        if not _paired(inputs, outputs):
            return []
        assert outputs is not None
        if any(
            target.height > source.height
            or target.width > source.width
            or (target.height == source.height and target.width == source.width)
            for source, target in zip(inputs, outputs)
        ):
            return []

        candidates: list[Hypothesis] = []
        rules = (
            "largest",
            "smallest",
            "most_colors",
            "unique_color",
            "unique_pattern",
            "bottommost",
            "topmost",
            "leftmost",
            "rightmost",
        )
        for mode in ("foreground", "per_color"):
            for connectivity in (4, 8):
                for rule in rules:
                    if mode == "per_color" and rule == "most_colors":
                        continue
                    transform = dsl.component_bbox_crop(mode, connectivity, rule)
                    if _exact(transform, inputs, outputs):
                        candidates.append(Hypothesis(
                            transform,
                            (
                                f"crop {rule.replace('_', ' ')} {mode.replace('_', ' ')} "
                                f"component using {connectivity}-connectivity"
                            ),
                            0.82,
                            3,
                            "component-bbox",
                        ))
        return _deduplicate(candidates)


class OverlayFillProposer(Proposer):
    """Overlay geometric copies or fill holes enclosed by foreground."""

    tier = 1

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        if not _paired(inputs, outputs):
            return []
        assert outputs is not None
        if any(source.grid.shape != target.grid.shape for source, target in zip(inputs, outputs)):
            return []

        candidates: list[Hypothesis] = []
        for symmetry in (
            "rotate_90",
            "rotate_180",
            "rotate_270",
            "flip_h",
            "flip_v",
            "transpose",
        ):
            for operation in ("or", "and", "xor", "max", "min"):
                transform = dsl.transform_overlay(symmetry, operation)
                if _exact(transform, inputs, outputs):
                    candidates.append(Hypothesis(
                        transform,
                        f"combine input with {symmetry.replace('_', ' ')} using {operation}",
                        0.82,
                        3,
                        "transform-overlay",
                    ))

        for fill_color in range(10):
            transform = dsl.fill_enclosed_holes(fill_color)
            if _exact(transform, inputs, outputs):
                candidates.append(Hypothesis(
                    transform,
                    f"fill enclosed foreground holes with color {fill_color}",
                    0.84,
                    2,
                    "enclosed-hole-fill",
                ))

        return _deduplicate(candidates)


class NeighborColorProposer(Proposer):
    """Recolor cells by input color and local foreground-neighbor count."""

    tier = 1

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        if not _paired(inputs, outputs):
            return []
        assert outputs is not None
        if any(
            source.grid.shape != target.grid.shape
            for source, target in zip(inputs, outputs)
        ):
            return []

        candidates: list[Hypothesis] = []
        for connectivity in (4, 8):
            mapping = _infer_neighbor_color_mapping(inputs, outputs, connectivity)
            if mapping is None:
                continue
            transform = _neighbor_color_transform(mapping, connectivity)
            if _exact(transform, inputs, outputs):
                candidates.append(Hypothesis(
                    transform,
                    f"recolor by {connectivity}-neighbor foreground count",
                    0.84,
                    3,
                    "neighbor-color",
                ))
        return _deduplicate(candidates)


class GridCellOperationProposer(Proposer):
    """Apply structural operations to cells separated by uniform H+V dividers."""

    tier = 3

    def propose(
        self,
        inputs: list[Scene],
        outputs: list[Scene] | None = None,
    ) -> list[Hypothesis]:
        if not _paired(inputs, outputs):
            return []
        assert outputs is not None
        transforms = (
            (
                dsl.grid_key_pattern_recolor(),
                "recolor neutral cell patterns from aligned singleton keys",
                5,
                "grid-key-pattern-recolor",
            ),
            (
                dsl.grid_stamp_by_row_key(),
                "broadcast one cell template in each grid row's key color",
                6,
                "grid-row-stamp",
            ),
            (
                dsl.grid_fill_from_max_pattern(),
                "complete every grid cell to the largest observed pattern mask",
                5,
                "grid-pattern-fill",
            ),
            (
                dsl.grid_complete_cell_multiset(),
                "replace divider placeholders to equalize cell color multisets",
                6,
                "grid-cell-multiset",
            ),
            (
                dsl.extract_unique_nonuniform_grid_cell(),
                "extract the unique non-uniform grid cell",
                3,
                "grid-cell-extraction",
            ),
        )
        candidates: list[Hypothesis] = []
        for transform, description, complexity, family in transforms:
            if _exact(transform, inputs, outputs):
                candidates.append(Hypothesis(
                    transform,
                    description,
                    0.78,
                    complexity,
                    family,
                ))
        return _deduplicate(candidates)


def default_proposers() -> list[Proposer]:
    """Return the deterministic baseline proposer set in search order."""

    return [
        GeometricProposer(),
        ColorMapProposer(),
        ResizeProposer(),
        CropProposer(),
        LargestComponentProposer(),
        RoutedRibbonProposer(),
        NestedRectangleProposer(),
        D4OrbitProposer(),
        FractalProposer(),
        AnalogyProposer(),
        PartitionBlueprintProposer(),
        PartitionMaxCountProposer(),
        PartitionMarkerRouteProposer(),
        PartitionAnchorRelocateProposer(),
        FramedObjectProposer(),
        AxialCrossProposer(),
        ClosestHorizontalPairProposer(),
        MarkerFrameProposer(),
        PeriodicStripeProposer(),
        OrientedMarkerLineProposer(),
        WallpaperRepairProposer(),
        SymmetryCompleterProposer(),
        ShapeUnifierProposer(),
        SparseSymmetryRepairProposer(),
        FillFromBackgroundProposer(),
        FixedSmallOutputProposer(),
        PixelExpansionProposer(),
        MiscTransformProposer(),
        ComponentBBoxProposer(),
        OverlayFillProposer(),
        NeighborColorProposer(),
        GridCellOperationProposer(),
    ]
