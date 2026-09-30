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


def default_proposers() -> list[Proposer]:
    """Return the deterministic baseline proposer set in search order."""

    return [
        GeometricProposer(),
        ColorMapProposer(),
        ResizeProposer(),
        CropProposer(),
        LargestComponentProposer(),
        RoutedRibbonProposer(),
    ]
