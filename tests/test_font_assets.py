"""Offline plot-font preparation contracts."""
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


class FontTests(unittest.TestCase):
    def helper(self):
        try:
            from font_assets import ensure_plot_font
        except ImportError:
            self.fail("Offline font preparation is not implemented")
        return ensure_plot_font

    def test_copies_first_available_local_font_to_legacy_cache_name(self):
        prepare = self.helper()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "DejaVuSans.ttf"
            source.write_bytes(b"local-font")
            result = prepare(root / "cache", [root / "missing.ttf", source])
            self.assertEqual(result.name, "Arial.ttf")
            self.assertEqual(result.read_bytes(), source.read_bytes())

    def test_existing_font_is_preserved(self):
        prepare = self.helper()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "Arial.ttf"
            target.write_bytes(b"existing-font")
            self.assertEqual(prepare(root, []).read_bytes(), b"existing-font")

    def test_missing_local_fonts_fail_without_creating_fake_font(self):
        prepare = self.helper()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaises(FileNotFoundError):
                prepare(root, [])
            self.assertFalse((root / "Arial.ttf").exists())
