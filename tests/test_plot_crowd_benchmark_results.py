"""Regression checks for the benchmark chart source."""

from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from tools.plot_crowd_benchmark_results import METHODS, create_figure


class CrowdBenchmarkPlotTests(unittest.TestCase):
    def test_values_are_complete_percentages(self):
        for method in METHODS:
            for model in ("double_integrator", "unicycle"):
                values = method[model]
                if values is None:
                    continue
                self.assertEqual(len(values), 3)
                self.assertTrue(all(sum(cell) == 100 for cell in values))

    def test_render_has_expected_dimensions_and_portable_svg_structure(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            svg_path = root / "benchmark.svg"
            png_path = root / "benchmark.png"

            create_figure(svg_path, png_path)

            svg = svg_path.read_text(encoding="utf-8")
            self.assertIn('width="707" height="352"', svg)
            self.assertNotIn("<mask", svg)
            self.assertNotIn("<clipPath", svg)

            png = png_path.read_bytes()
            self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(struct.unpack(">II", png[16:24]), (707, 352))


if __name__ == "__main__":
    unittest.main()
