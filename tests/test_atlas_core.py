"""Unit tests for Atlas-owned SceneGraph, DSL, and proposers."""

from __future__ import annotations

import unittest

import numpy as np

from arc2 import dsl
from arc2.proposers import (
    ColorMapProposer,
    GeometricProposer,
    LargestComponentProposer,
    ResizeProposer,
    RoutedRibbonProposer,
    default_proposers,
)
from arc2.scene import Scene
from arc2.search import search


class SceneTests(unittest.TestCase):
    def test_nonzero_background_and_object_extraction(self) -> None:
        scene = Scene([
            [3, 3, 3, 3],
            [3, 8, 8, 3],
            [3, 3, 3, 3],
        ])
        self.assertEqual(scene.background_color(), 3)
        self.assertEqual(len(scene.objects()), 1)
        obj = scene.objects()[0]
        self.assertEqual((obj.color, obj.size), (8, 2))
        self.assertEqual(obj.bbox.as_list(), [1, 1, 1, 2])

    def test_hole_and_symmetry_detection(self) -> None:
        scene = Scene([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 1, 1, 1, 1, 1, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ])
        self.assertEqual(scene.background_color(), 0)
        self.assertEqual(len(scene.holes()), 1)
        self.assertEqual(scene.holes()[0].size, 9)
        symmetries = scene.symmetries()
        self.assertTrue(symmetries["flip_lr"])
        self.assertTrue(symmetries["flip_ud"])
        self.assertTrue(symmetries["rot90"])

    def test_nested_palette_preserves_repeated_colors(self) -> None:
        scene = Scene([
            [2, 2, 2, 2, 2],
            [2, 1, 1, 1, 2],
            [2, 1, 2, 1, 2],
            [2, 1, 1, 1, 2],
            [2, 2, 2, 2, 2],
        ])
        self.assertEqual(scene.nested_rectangular_palette(), (2, 1, 2))


class DSLTests(unittest.TestCase):
    def test_recolor_is_simultaneous(self) -> None:
        transform = dsl.recolor({1: 2, 2: 1})
        actual = transform(np.array([[1, 2], [2, 1]]))
        np.testing.assert_array_equal(actual, [[2, 1], [1, 2]])

    def test_compose_recolor_then_tile(self) -> None:
        transform = dsl.compose(dsl.recolor({0: 1, 1: 0}), dsl.tile(1, 2))
        actual = transform(np.array([[0, 1]]))
        np.testing.assert_array_equal(actual, [[1, 0, 1, 0]])

    def test_crop_uses_inferred_nonzero_background(self) -> None:
        actual = dsl.crop_non_background(np.array([
            [4, 4, 4, 4],
            [4, 7, 7, 4],
            [4, 4, 4, 4],
        ]))
        np.testing.assert_array_equal(actual, [[7, 7]])

    def test_largest_component_histogram(self) -> None:
        transform = dsl.largest_component_histogram("left_to_right", "rows")
        actual = transform(np.array([
            [0, 5, 0, 0, 2, 2],
            [0, 5, 5, 0, 0, 2],
            [9, 0, 0, 0, 0, 0],
        ]))
        np.testing.assert_array_equal(actual, [[5, 2], [5, 2], [5, 2]])

    def test_routed_ribbon_preserves_obstacle_and_bridges_detour(self) -> None:
        transform = dsl.routed_ribbon(2, -1, 1, width=2)
        actual = transform(np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0],
        ]))
        np.testing.assert_array_equal(actual, [
            [0, 0, 0, 0, 0, 2, 2],
            [0, 0, 0, 1, 2, 2, 0],
            [0, 0, 2, 2, 2, 0, 0],
            [0, 2, 2, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0],
        ])


class ProposerTests(unittest.TestCase):
    @staticmethod
    def scenes(grids: list[list[list[int]]]) -> list[Scene]:
        return [Scene(grid) for grid in grids]

    def test_geometric_proposer_learns_rotation(self) -> None:
        inputs = self.scenes([
            [[1, 0, 0], [2, 0, 0]],
            [[0, 3, 0], [4, 0, 0]],
        ])
        outputs = [Scene(np.rot90(scene.grid, 1)) for scene in inputs]
        proposals = GeometricProposer().propose(inputs, outputs)
        self.assertTrue(any(h.description == "rotate 90" for h in proposals))

    def test_color_map_proposer_learns_swap(self) -> None:
        inputs = self.scenes([[[1, 2], [0, 1]], [[2, 0], [1, 2]]])
        outputs = self.scenes([[[2, 1], [0, 2]], [[1, 0], [2, 1]]])
        proposals = ColorMapProposer().propose(inputs, outputs)
        self.assertTrue(proposals)
        for source, target in zip(inputs, outputs):
            np.testing.assert_array_equal(proposals[0].transform(source.grid), target.grid)

    def test_resize_proposer_learns_recolor_then_tile(self) -> None:
        inputs = self.scenes([[[0, 1]], [[1, 0]]])
        outputs = self.scenes([[[2, 3, 2, 3]], [[3, 2, 3, 2]]])
        proposals = ResizeProposer().propose(inputs, outputs)
        self.assertTrue(any("recolor" in h.description and "tile" in h.description for h in proposals))
        for proposal in proposals:
            for source, target in zip(inputs, outputs):
                np.testing.assert_array_equal(proposal.transform(source.grid), target.grid)

    def test_largest_component_proposer(self) -> None:
        source = Scene([
            [0, 5, 0, 0, 2, 2],
            [0, 5, 5, 0, 0, 2],
            [9, 0, 0, 0, 0, 0],
        ])
        target = Scene([[5, 2], [5, 2], [5, 2]])
        proposals = LargestComponentProposer().propose([source], [target])
        self.assertTrue(any("left to right" in h.description for h in proposals))

    def test_routed_ribbon_proposer(self) -> None:
        source = Scene([
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0, 0, 0],
        ])
        target = Scene(dsl.routed_ribbon(2, -1, 1, 2)(source.grid))
        proposals = RoutedRibbonProposer().propose([source], [target])
        self.assertTrue(any("direction -1,1 width 2" in h.description for h in proposals))

    def test_default_proposers_integrate_with_search(self) -> None:
        train = [{
            "input": [[1, 0, 0], [2, 0, 0]],
            "output": [[0, 0], [0, 0], [1, 2]],
        }]
        result = search(
            default_proposers(),
            train,
            max_hypotheses=100,
            time_budget_seconds=2.0,
        )
        self.assertTrue(result.solved)
        self.assertEqual(result.best.description, "rotate 90")


if __name__ == "__main__":
    unittest.main()
