"""Relation-driven hypothesis proposers built from the Atlas DSL.

Owner: Atlas

These proposers intentionally cover only compact, reusable rule families.
Every emitted candidate is deterministic; the verifier remains responsible
for accepting only exact fits across every training pair.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

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
    ]
