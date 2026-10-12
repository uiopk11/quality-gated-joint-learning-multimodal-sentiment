from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from q1lib.pipeline import derive_mfa_window


class MFAWindowTests(unittest.TestCase):
    def test_prefix_and_suffix_are_excluded(self) -> None:
        quality = {
            "official_match_mask": [1, 1, 1],
            "official_asr_starts": [1.44, 2.0, 4.5],
            "official_asr_ends": [1.60, 2.3, 5.38],
        }
        result = derive_mfa_window(quality, 5.50, 0.03, True)
        self.assertTrue(result["eligible"])
        self.assertAlmostEqual(result["window_start"], 1.41)
        self.assertAlmostEqual(result["window_end"], 5.41)
        self.assertAlmostEqual(result["prefix_extra_seconds"], 1.44)
        self.assertAlmostEqual(result["suffix_extra_seconds"], 0.12)
        self.assertTrue(result["is_windowed"])

    def test_missing_boundary_anchor_downgrades_mfa(self) -> None:
        quality = {
            "official_match_mask": [0, 1, 1],
            "official_asr_starts": [None, 1.0, 2.0],
            "official_asr_ends": [None, 1.4, 2.5],
        }
        result = derive_mfa_window(quality, 3.0, 0.03, True)
        self.assertFalse(result["eligible"])
        self.assertEqual(result["reason"], "official_boundary_word_not_reliably_matched")

    def test_complete_clip_remains_valid(self) -> None:
        quality = {
            "official_match_mask": [1, 1],
            "official_asr_starts": [0.0, 0.5],
            "official_asr_ends": [0.4, 1.0],
        }
        result = derive_mfa_window(quality, 1.0, 0.03, True)
        self.assertTrue(result["eligible"])
        self.assertFalse(result["is_windowed"])


if __name__ == "__main__":
    unittest.main()
