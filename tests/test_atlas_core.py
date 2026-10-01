"""Unit tests for Atlas-owned SceneGraph, DSL, and proposers."""

from __future__ import annotations

import unittest

import numpy as np

from arc2 import dsl
from arc2.proposers import (
    AnalogyProposer,
    AxialCrossProposer,
    ClosestHorizontalPairProposer,
    ColorMapProposer,
    ComponentBBoxProposer,
    D4OrbitProposer,
    FillFromBackgroundProposer,
    FixedSmallOutputProposer,
    GridCellOperationProposer,
    FractalProposer,
    GeometricProposer,
    LargestComponentProposer,
    MarkerFrameProposer,
    MiscTransformProposer,
    NestedRectangleProposer,
    FramedObjectProposer,
    PartitionAnchorRelocateProposer,
    PartitionBlueprintProposer,
    PartitionMarkerRouteProposer,
    PartitionMaxCountProposer,
    PixelExpansionProposer,
    PeriodicStripeProposer,
    OrientedMarkerLineProposer,
    OverlayFillProposer,
    ShapeUnifierProposer,
    SparseSymmetryRepairProposer,
    SymmetryCompleterProposer,
    WallpaperRepairProposer,
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

    def test_no_op_programs_have_identity_signature(self) -> None:
        identity_signature = dsl.signature(dsl.identity)
        self.assertEqual(dsl.signature(dsl.tile(1, 1)), identity_signature)
        self.assertEqual(dsl.signature(dsl.scale(1, 1)), identity_signature)
        self.assertEqual(dsl.signature(dsl.rotate(0)), identity_signature)
        self.assertIs(dsl.compose(dsl.identity, dsl.tile(1, 1)), dsl.identity)

        recolor = dsl.recolor({1: 2})
        decorated = dsl.compose(recolor, dsl.scale(1, 1), dsl.rotate(0))
        self.assertIs(decorated, recolor)
        self.assertEqual(dsl.signature(decorated), dsl.signature(recolor))

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

    def test_nested_rectangle_miniature_preserves_repeated_colors(self) -> None:
        actual = dsl.nested_rectangle_miniature()(np.array([
            [2, 2, 2, 2, 2],
            [2, 1, 1, 1, 2],
            [2, 1, 2, 1, 2],
            [2, 1, 1, 1, 2],
            [2, 2, 2, 2, 2],
        ]))
        np.testing.assert_array_equal(actual, [
            [2, 2, 2, 2, 2],
            [2, 1, 1, 1, 2],
            [2, 1, 2, 1, 2],
            [2, 1, 1, 1, 2],
            [2, 2, 2, 2, 2],
        ])

    def test_d4_orbit_consensus_fills_mask(self) -> None:
        source = np.array([
            [1, 2, 2, 1],
            [3, 4, 3, 3],
            [3, 3, 4, 3],
            [1, 2, 2, 1],
        ])
        actual = dsl.d4_orbit_consensus(4)(source)
        np.testing.assert_array_equal(actual, [
            [1, 2, 2, 1],
            [3, 3, 3, 3],
            [3, 3, 3, 3],
            [1, 2, 2, 1],
        ])

    def test_self_similar_stamp_restores_occluded_copy(self) -> None:
        source = np.array([
            [2, 2, 0, 2, 2],
            [2, 0, 0, 2, 0],
            [0, 0, 0, 0, 0],
            [2, 2, 0, 3, 3],
            [2, 0, 0, 3, 3],
        ])
        expected = np.array([
            [8, 2, 0, 2, 8],
            [2, 0, 0, 2, 0],
            [0, 0, 0, 0, 0],
            [2, 2, 0, 0, 0],
            [8, 0, 0, 0, 0],
        ])
        np.testing.assert_array_equal(dsl.self_similar_stamp(8)(source), expected)

    def test_block_analogy_applies_geometry_and_color_map(self) -> None:
        source = np.array([
            [2, 4, 0, 4, 2, 0, 0, 8, 6],
            [4, 4, 0, 4, 4, 0, 0, 8, 8],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 7, 0, 8, 3, 0, 0, 3, 3],
            [3, 3, 0, 8, 8, 0, 0, 3, 7],
        ])
        np.testing.assert_array_equal(dsl.block_analogy()(source), [
            [6, 8],
            [8, 8],
            [0, 0],
            [8, 8],
            [8, 3],
        ])

    def test_partition_max_count_fill_preserves_separators(self) -> None:
        source = np.array([
            [2, 0, 5, 0, 0],
            [2, 0, 5, 0, 2],
            [5, 5, 5, 5, 5],
            [0, 0, 5, 2, 2],
            [0, 0, 5, 2, 0],
        ])
        np.testing.assert_array_equal(dsl.partition_max_count_fill()(source), [
            [0, 0, 5, 0, 0],
            [0, 0, 5, 0, 0],
            [5, 5, 5, 5, 5],
            [0, 0, 5, 2, 2],
            [0, 0, 5, 2, 2],
        ])

    def test_background_gaps_are_not_partition_separators(self) -> None:
        source = np.array([
            [8, 8, 8, 8, 8],
            [8, 6, 6, 6, 8],
            [8, 6, 8, 6, 8],
            [8, 6, 6, 6, 8],
            [8, 8, 8, 8, 8],
        ])
        with self.assertRaises(ValueError):
            dsl.partition_max_count_fill()(source)

    def test_frame_and_fill_objects(self) -> None:
        source = np.array([
            [8, 8, 8, 8, 8, 8, 8],
            [8, 8, 6, 6, 6, 8, 8],
            [8, 8, 6, 8, 6, 8, 8],
            [8, 8, 6, 6, 6, 8, 8],
            [8, 8, 8, 8, 8, 8, 8],
        ])
        np.testing.assert_array_equal(dsl.frame_and_fill_objects(6, 3, 4)(source), [
            [8, 3, 3, 3, 3, 3, 8],
            [8, 3, 6, 6, 6, 3, 8],
            [8, 3, 6, 4, 6, 3, 8],
            [8, 3, 6, 6, 6, 3, 8],
            [8, 3, 3, 3, 3, 3, 8],
        ])

    def test_axial_cross_lines_marks_different_color_collisions(self) -> None:
        source = np.array([
            [0, 8, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 7],
            [0, 0, 0, 0],
        ])
        np.testing.assert_array_equal(dsl.axial_cross_lines(2)(source), [
            [8, 8, 8, 2],
            [0, 8, 0, 7],
            [7, 2, 7, 7],
            [0, 8, 0, 7],
        ])

    def test_closest_horizontal_pairs_only_fills_minimum_gap(self) -> None:
        source = np.array([
            [2, 0, 0, 0, 0, 2],
            [0, 2, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 2, 0],
        ])
        np.testing.assert_array_equal(dsl.connect_closest_horizontal_pairs()(source), [
            [2, 0, 0, 0, 0, 2],
            [0, 2, 2, 2, 2, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 2, 2, 2, 2, 0],
        ])

    def test_marker_frame_clips_at_corner(self) -> None:
        source = np.array([
            [5, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        np.testing.assert_array_equal(dsl.frame_around_markers(5, 1)(source), [
            [5, 1, 0],
            [1, 1, 0],
            [0, 0, 0],
        ])

    def test_periodic_boundary_stripes(self) -> None:
        source = np.zeros((7, 4), dtype=int)
        source[1, 0] = 2
        source[3, 3] = 3
        np.testing.assert_array_equal(dsl.periodic_boundary_stripes()(source), [
            [0, 0, 0, 0],
            [2, 2, 2, 2],
            [0, 0, 0, 0],
            [3, 3, 3, 3],
            [0, 0, 0, 0],
            [2, 2, 2, 2],
            [0, 0, 0, 0],
        ])

    def test_oriented_marker_lines_row_precedence(self) -> None:
        source = np.array([
            [0, 0, 2, 0],
            [0, 0, 0, 3],
            [0, 0, 0, 0],
        ])
        np.testing.assert_array_equal(
            dsl.oriented_marker_lines({2}, {3}, horizontal_overwrites=True)(source),
            [
                [0, 0, 2, 0],
                [3, 3, 3, 3],
                [0, 0, 2, 0],
            ],
        )

    def test_oriented_marker_lines_rejects_non_singleton_components(self) -> None:
        source = np.array([
            [2, 2, 0],
            [0, 0, 0],
            [0, 0, 3],
        ])
        with self.assertRaises(ValueError):
            dsl.oriented_marker_lines({2}, {3})(source)

    def test_periodic_pattern_repair_fills_only_holes(self) -> None:
        source = np.array([
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
            [1, 2, 8, 2, 1, 2],
            [3, 4, 3, 8, 3, 4],
        ])
        np.testing.assert_array_equal(dsl.periodic_pattern_repair(8)(source), [
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4],
        ])

    def test_periodic_pattern_repair_rejects_uniform_and_noise(self) -> None:
        with self.assertRaises(ValueError):
            dsl.periodic_pattern_repair(0)(np.array([
                [1, 1, 1, 1],
                [1, 0, 1, 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ]))
        with self.assertRaises(ValueError):
            dsl.periodic_pattern_repair(0)(np.array([
                [1, 2, 3, 4],
                [5, 0, 6, 7],
                [8, 9, 1, 2],
                [3, 4, 5, 6],
            ]))

    def test_reflection_block_repair_completes_one_quartet(self) -> None:
        source = np.array([
            [1, 0, 5, 0, 1],
            [1, 1, 5, 1, 1],
            [5, 5, 5, 5, 5],
            [1, 1, 5, 0, 0],
            [1, 0, 5, 0, 0],
        ])
        np.testing.assert_array_equal(dsl.reflection_block_repair(0)(source), [
            [1, 0, 5, 0, 1],
            [1, 1, 5, 1, 1],
            [5, 5, 5, 5, 5],
            [1, 1, 5, 1, 1],
            [1, 0, 5, 0, 1],
        ])

    def test_shape_unifier_handles_multiple_twins(self) -> None:
        source = np.array([
            [1, 0, 7, 0, 7],
            [1, 0, 7, 0, 7],
        ])
        np.testing.assert_array_equal(dsl.unify_shape_colors(1)(source), [
            [7, 0, 7, 0, 7],
            [7, 0, 7, 0, 7],
        ])

    def test_periodic_perimeter_segments_uses_nonzero_background(self) -> None:
        source = np.full((4, 6), 8, dtype=int)
        source[0, 0:2] = 3
        np.testing.assert_array_equal(dsl.periodic_perimeter_segments()(source), [
            [3, 3, 8, 8, 3, 3],
            [8, 8, 8, 8, 8, 8],
            [8, 8, 8, 8, 8, 8],
            [3, 3, 8, 8, 3, 3],
        ])

    def test_complete_occluded_rectangles_uses_nonzero_background(self) -> None:
        source = np.full((6, 7), 7, dtype=int)
        source[1:5, 3:6] = 4
        source[3:5, 1:5] = 1
        actual = dsl.complete_occluded_rectangles()(source)
        expected = source.copy()
        expected[1:5, 3:6] = 4
        np.testing.assert_array_equal(actual, expected)

    def test_opposite_edge_anomaly_pairs_completes_dent_and_bulge(self) -> None:
        source = np.full((7, 8), 8, dtype=int)
        source[2:5, 2:6] = 1
        source[2, 4] = 8
        source[5, 3] = 1
        actual = dsl.opposite_edge_anomaly_pairs(1)(source)
        expected = source.copy()
        expected[5, 4] = 1
        expected[2, 3] = 8
        np.testing.assert_array_equal(actual, expected)

    def test_directed_marker_ray_cleanup_follows_diagonal(self) -> None:
        source = np.full((8, 8), 7, dtype=int)
        source[3, 3] = 2
        source[4, 4] = source[5, 5] = source[6, 6] = 6
        source[1, 1] = 9
        actual = dsl.directed_marker_ray_cleanup()(source)
        expected = source.copy()
        expected[1, 1] = 7
        np.testing.assert_array_equal(actual, expected)

    def test_legend_sample_erase_preserves_sample(self) -> None:
        source = np.zeros((7, 7), dtype=int)
        source[0:4, 3] = 5
        source[3, 0:4] = 5
        source[1, 1] = source[5, 5] = source[6, 2] = 4
        actual = dsl.legend_sample_erase()(source)
        expected = source.copy()
        expected[5, 5] = expected[6, 2] = 0
        np.testing.assert_array_equal(actual, expected)

    def test_duplicate_template_refine_marks_stair_anchor(self) -> None:
        source = np.full((12, 14), 7, dtype=int)
        source[0, 0:3] = 1
        source[1, 1:3] = 1
        source[2, 2] = 1
        source[5, 9:14] = 5
        source[6, 10:14] = 5
        source[7, 11:14] = 5
        source[8, 12:14] = 5
        first = [(3, 2), (4, 2)] + [(row, col) for row in range(5, 9) for col in range(1, 4)]
        second = [(9, col) for col in range(4, 8)] + [(10, col) for col in range(4, 10)] + [(11, col) for col in range(4, 8)]
        for cell in first + second:
            source[cell] = 0
        source[4, 6] = 3
        actual = dsl.duplicate_template_refine(6)(source)
        self.assertTrue(np.all(actual[5:9, 3] == 6))
        self.assertTrue(np.all(actual[5:9, 1:3] == 0))
        self.assertEqual(int(actual[3, 2]), 7)

    def test_extend_diagonal_lattice_fills_open_tail(self) -> None:
        source = np.zeros((9, 9), dtype=int)
        source[0, 0] = source[2, 2] = source[4, 4] = 1
        actual = dsl.extend_diagonal_lattice(1, 3)(source)
        expected = source.copy()
        expected[6, 6] = expected[8, 8] = 3
        np.testing.assert_array_equal(actual, expected)

    def test_complete_solid_template_fills_only_template_slots(self) -> None:
        source = np.array([
            [1, 4, 1, 0, 1, 0, 1],
            [4, 8, 4, 0, 0, 8, 0],
            [1, 4, 1, 0, 1, 0, 1],
        ])
        actual = dsl.complete_solid_template(4)(source)
        expected = source.copy()
        expected[:, 4:7] = np.array([[1, 4, 1], [4, 8, 4], [1, 4, 1]])
        np.testing.assert_array_equal(actual, expected)

    def test_complete_d4_orbits_fills_partial_corner_orbit(self) -> None:
        source = np.zeros((5, 5), dtype=int)
        source[2, 2] = 3
        source[0, 2] = source[2, 0] = source[2, 4] = source[4, 2] = 1
        source[0, 0] = 2
        actual = dsl.complete_d4_orbits()(source)
        expected = source.copy()
        expected[0, 4] = expected[4, 0] = expected[4, 4] = 2
        np.testing.assert_array_equal(actual, expected)

    def test_mode_below_separator_places_unique_mode_at_bottom_center(self) -> None:
        source = np.array([
            [2, 2, 3, 2, 4],
            [4, 2, 3, 2, 1],
            [5, 5, 5, 5, 5],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        actual = dsl.mode_below_separator()(source)
        expected = source.copy()
        expected[4, 2] = 2
        np.testing.assert_array_equal(actual, expected)

    def test_complete_local_symmetric_templates_propagates_motif(self) -> None:
        source = np.zeros((7, 7), dtype=int)
        source[1, 1:4] = [4, 8, 4]
        source[4, 4] = 8
        actual = dsl.complete_local_symmetric_templates()(source)
        expected = source.copy()
        expected[4, 3] = expected[4, 5] = 4
        np.testing.assert_array_equal(actual, expected)

    def test_fill_from_background_primitives_fail_closed_on_noise(self) -> None:
        ambiguous = np.array([[1, 0, 2], [0, 0, 0], [0, 0, 0]])
        empty_context = np.zeros((5, 5), dtype=int)
        border_marker = np.array([[8, 0, 0], [0, 0, 0], [0, 0, 0]])
        dense = np.array([[1, 2, 3], [4, 0, 6], [7, 8, 9]])
        off_lattice = np.zeros((7, 7), dtype=int)
        off_lattice[0, 0] = off_lattice[2, 2] = off_lattice[4, 5] = 1
        degenerate = np.array([[0, 1, 0, 1, 0]])
        cases = (
            (dsl.extend_diagonal_lattice(1, 3), off_lattice),
            (dsl.complete_solid_template(4), ambiguous),
            (dsl.complete_d4_orbits(), ambiguous),
            (dsl.mode_below_separator(), border_marker),
            (dsl.complete_local_symmetric_templates(), dense),
        )
        for transform, source in cases:
            np.testing.assert_array_equal(transform(source), source)
        for transform in (
            dsl.extend_diagonal_lattice(1, 3),
            dsl.complete_solid_template(4),
            dsl.complete_d4_orbits(),
            dsl.mode_below_separator(),
            dsl.complete_local_symmetric_templates(),
        ):
            np.testing.assert_array_equal(transform(empty_context), empty_context)
            np.testing.assert_array_equal(transform(degenerate), degenerate)

    def test_extract_full_span_color_finds_uninterrupted_bar(self) -> None:
        source = np.array([[0, 2, 0], [3, 2, 3], [0, 2, 0]])
        np.testing.assert_array_equal(dsl.extract_full_span_color()(source), [[2]])

    def test_quadrant_object_map_extracts_canvas_positions(self) -> None:
        source = np.zeros((8, 8), dtype=int)
        source[1:7, 1:7] = 8
        source[2:4, 2:4] = 3
        source[4:6, 4:6] = 5
        expected = np.array([[3, 0], [0, 5]])
        np.testing.assert_array_equal(dsl.quadrant_object_map()(source), expected)

    def test_encode_cell_count_3x3_uses_top_then_center(self) -> None:
        source = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]])
        expected = np.array([[2, 2, 2], [0, 2, 0], [0, 0, 0]])
        np.testing.assert_array_equal(dsl.encode_cell_count_3x3(2)(source), expected)

    def test_encode_square_component_count_3x3_counts_blocks(self) -> None:
        source = np.zeros((6, 6), dtype=int)
        source[0:2, 0:2] = 4
        source[3:5, 3:5] = 4
        expected = np.array([[1, 0, 1], [0, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(
            dsl.encode_square_component_count_3x3(1)(source),
            expected,
        )

    def test_component_segment_histogram_handles_nonzero_background(self) -> None:
        source = np.full((6, 8), 8, dtype=int)
        source[0, 0:2] = source[2, 4:7] = source[5, 1:3] = 3
        source[1, 5:7] = source[4, 0:4] = 2
        expected = np.array([[3, 3, 3], [2, 2, 0]])
        np.testing.assert_array_equal(
            dsl.component_segment_histogram(2, 3)(source),
            expected,
        )

    def test_grid_cell_operation_primitives(self) -> None:
        recolor = np.zeros((7, 7), dtype=int)
        recolor[3, :] = recolor[:, 3] = 4
        recolor[1, 1] = 5
        recolor[5, 1] = 6
        recolor[0, 4] = recolor[1, 5] = 1
        recolor[4, 4] = recolor[5, 5] = 1
        expected_recolor = recolor.copy()
        expected_recolor[expected_recolor == 1] = np.array([5, 5, 6, 6])
        np.testing.assert_array_equal(
            dsl.grid_key_pattern_recolor()(recolor),
            expected_recolor,
        )

        fill = np.zeros((7, 7), dtype=int)
        fill[3, :] = fill[:, 3] = 2
        fill[0, 0] = fill[0, 1] = fill[1, 0] = 1
        fill[0, 4] = 1
        expected_fill = fill.copy()
        for r0 in (0, 4):
            for c0 in (0, 4):
                mask = np.array([[True, True, False], [True, False, False], [False, False, False]])
                view = expected_fill[r0:r0 + 3, c0:c0 + 3]
                view[mask & (view == 0)] = 2
        np.testing.assert_array_equal(dsl.grid_fill_from_max_pattern()(fill), expected_fill)

        multiset = np.array([
            [7, 1, 7, 1, 7],
            [2, 3, 7, 2, 4],
            [7, 7, 7, 7, 7],
            [1, 3, 7, 2, 3],
            [7, 4, 7, 4, 7],
        ])
        completed = dsl.grid_complete_cell_multiset()(multiset)
        for r0 in (0, 3):
            for c0 in (0, 3):
                self.assertEqual(sorted(completed[r0:r0 + 2, c0:c0 + 2].flat), [1, 2, 3, 4])

        extract = np.full((7, 7), 4, dtype=int)
        extract[3, :] = extract[:, 3] = 2
        extract[1, 5] = 9
        np.testing.assert_array_equal(
            dsl.extract_unique_nonuniform_grid_cell()(extract),
            extract[0:3, 4:7],
        )

    def test_pixel_expansion_primitives(self) -> None:
        source = np.array([[4, 0], [0, 4]])
        blank = np.zeros_like(source)
        expected_self = np.block([[source, blank], [blank, source]])
        np.testing.assert_array_equal(
            dsl.pixel_self_substitute("nonzero")(source),
            expected_self,
        )

        complement = np.array([[0, 4], [4, 0]])
        expected_complement = np.block([[complement, blank], [blank, complement]])
        np.testing.assert_array_equal(
            dsl.pixel_complement_substitute("nonzero")(source),
            expected_complement,
        )

        expected_mirror = np.block([
            [source, np.fliplr(source)],
            [np.flipud(source), np.rot90(source, 2)],
        ])
        np.testing.assert_array_equal(dsl.mirror4_tile()(source), expected_mirror)

        mapping = {
            0: np.array([[0, 0], [0, 0]]),
            4: np.array([[1, 2], [2, 1]]),
        }
        expected_blocks = np.block([
            [mapping[4], mapping[0]],
            [mapping[0], mapping[4]],
        ])
        np.testing.assert_array_equal(
            dsl.color_block_substitute(mapping)(source),
            expected_blocks,
        )

    def test_misc_transform_primitives(self) -> None:
        gravity_source = np.array([
            [0, 2, 0],
            [3, 0, 4],
            [0, 5, 0],
        ])
        gravity_expected = np.array([
            [0, 0, 0],
            [0, 2, 0],
            [3, 5, 4],
        ])
        np.testing.assert_array_equal(
            dsl.gravity("down")(gravity_source),
            gravity_expected,
        )

        markers = np.array([
            [2, 0, 0, 2],
            [0, 3, 0, 0],
            [0, 0, 0, 0],
            [0, 3, 0, 0],
        ])
        connected = np.array([
            [2, 2, 2, 2],
            [0, 3, 0, 0],
            [0, 3, 0, 0],
            [0, 3, 0, 0],
        ])
        np.testing.assert_array_equal(
            dsl.connect_same_color("horizontal_vertical")(markers),
            connected,
        )

        shifted = np.array([[4, 8, 4, 8], [8, 4, 8, 4]])
        np.testing.assert_array_equal(
            dsl.diagonal_shift("column", 1)(np.array([[4] * 4, [8] * 4])),
            shifted,
        )

    def test_component_bbox_crop_uses_original_grid(self) -> None:
        source = np.array([
            [0, 2, 0, 0, 0, 0],
            [2, 0, 2, 0, 5, 5],
            [0, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ])
        expected = np.array([
            [0, 2, 0],
            [2, 0, 2],
            [0, 2, 0],
        ])
        np.testing.assert_array_equal(
            dsl.component_bbox_crop("foreground", 8, "largest")(source),
            expected,
        )

    def test_overlay_and_enclosed_hole_fill(self) -> None:
        source = np.array([
            [0, 2, 0],
            [0, 0, 0],
            [0, 0, 0],
        ])
        expected_overlay = np.array([
            [0, 2, 0],
            [0, 0, 0],
            [0, 2, 0],
        ])
        np.testing.assert_array_equal(
            dsl.transform_overlay("flip_v", "or")(source),
            expected_overlay,
        )

        ring = np.zeros((5, 5), dtype=int)
        ring[1, 1:4] = 3
        ring[3, 1:4] = 3
        ring[1:4, 1] = 3
        ring[1:4, 3] = 3
        filled = ring.copy()
        filled[2, 2] = 4
        np.testing.assert_array_equal(
            dsl.fill_enclosed_holes(4)(ring),
            filled,
        )


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

    def test_structural_proposer_family(self) -> None:
        nested = Scene([
            [2, 2, 2],
            [2, 1, 2],
            [2, 2, 2],
        ])
        self.assertTrue(NestedRectangleProposer().propose([nested], [nested]))

        orbit_input = Scene([
            [1, 2, 2, 1],
            [3, 4, 3, 3],
            [3, 3, 4, 3],
            [1, 2, 2, 1],
        ])
        orbit_output = Scene(dsl.d4_orbit_consensus(4)(orbit_input.grid))
        self.assertTrue(D4OrbitProposer().propose([orbit_input], [orbit_output]))

        analogy_input = Scene([
            [2, 4, 0, 4, 2, 0, 0, 8, 6],
            [4, 4, 0, 4, 4, 0, 0, 8, 8],
        ])
        analogy_output = Scene([[6, 8], [8, 8]])
        self.assertTrue(AnalogyProposer().propose([analogy_input], [analogy_output]))

        fractal_input = Scene([
            [2, 2, 0, 2, 2],
            [2, 0, 0, 2, 0],
            [0, 0, 0, 0, 0],
            [2, 2, 0, 3, 3],
            [2, 0, 0, 3, 3],
        ])
        fractal_output = Scene(dsl.self_similar_stamp(8)(fractal_input.grid))
        self.assertTrue(FractalProposer().propose([fractal_input], [fractal_output]))

        partition_input = Scene([
            [2, 0, 5, 0, 0],
            [2, 0, 5, 0, 2],
            [5, 5, 5, 5, 5],
            [0, 0, 5, 2, 2],
            [0, 0, 5, 2, 0],
        ])
        partition_output = Scene(dsl.partition_max_count_fill()(partition_input.grid))
        self.assertTrue(
            PartitionMaxCountProposer().propose([partition_input], [partition_output])
        )

        framed_input = Scene([
            [8, 8, 8, 8, 8, 8, 8],
            [8, 8, 6, 6, 6, 8, 8],
            [8, 8, 6, 8, 6, 8, 8],
            [8, 8, 6, 6, 6, 8, 8],
            [8, 8, 8, 8, 8, 8, 8],
        ])
        framed_output = Scene(dsl.frame_and_fill_objects(6, 3, 4)(framed_input.grid))
        self.assertTrue(FramedObjectProposer().propose([framed_input], [framed_output]))

        defaults = default_proposers()
        self.assertTrue(any(isinstance(p, PartitionBlueprintProposer) for p in defaults))
        self.assertTrue(any(isinstance(p, PartitionMarkerRouteProposer) for p in defaults))
        self.assertTrue(any(isinstance(p, PartitionAnchorRelocateProposer) for p in defaults))

        cross_input = Scene([
            [0, 8, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 7],
            [0, 0, 0, 0],
        ])
        cross_output = Scene(dsl.axial_cross_lines(2)(cross_input.grid))
        self.assertTrue(AxialCrossProposer().propose([cross_input], [cross_output]))

        pair_input = Scene([
            [2, 0, 0, 0, 0, 2],
            [0, 2, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 2, 0, 0, 2, 0],
        ])
        pair_output = Scene(dsl.connect_closest_horizontal_pairs()(pair_input.grid))
        self.assertTrue(
            ClosestHorizontalPairProposer().propose([pair_input], [pair_output])
        )

        self.assertTrue(any(isinstance(p, MarkerFrameProposer) for p in defaults))
        self.assertTrue(any(isinstance(p, PeriodicStripeProposer) for p in defaults))
        self.assertTrue(any(isinstance(p, OrientedMarkerLineProposer) for p in defaults))
        self.assertTrue(any(isinstance(p, WallpaperRepairProposer) for p in defaults))
        self.assertTrue(any(isinstance(p, SymmetryCompleterProposer) for p in defaults))
        self.assertTrue(any(isinstance(p, ShapeUnifierProposer) for p in defaults))
        self.assertTrue(any(isinstance(p, SparseSymmetryRepairProposer) for p in defaults))
        self.assertTrue(any(isinstance(p, FillFromBackgroundProposer) for p in defaults))
        self.assertTrue(any(isinstance(p, FixedSmallOutputProposer) for p in defaults))
        self.assertTrue(any(isinstance(p, PixelExpansionProposer) for p in defaults))
        self.assertTrue(any(isinstance(p, MiscTransformProposer) for p in defaults))
        self.assertTrue(any(isinstance(p, ComponentBBoxProposer) for p in defaults))
        self.assertTrue(any(isinstance(p, OverlayFillProposer) for p in defaults))
        self.assertTrue(any(isinstance(p, GridCellOperationProposer) for p in defaults))

    def test_pixel_expansion_proposer_adapts_nonzero_key_color(self) -> None:
        first = Scene([[6, 0], [0, 6]])
        second = Scene([[0, 3], [3, 0]])
        outputs = [
            Scene([[6, 0, 0, 0], [0, 6, 0, 0], [0, 0, 6, 0], [0, 0, 0, 6]]),
            Scene([[0, 0, 0, 3], [0, 0, 3, 0], [0, 3, 0, 0], [3, 0, 0, 0]]),
        ]
        proposals = PixelExpansionProposer().propose([first, second], outputs)
        self.assertTrue(any("sole non-zero" in h.description for h in proposals))

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
