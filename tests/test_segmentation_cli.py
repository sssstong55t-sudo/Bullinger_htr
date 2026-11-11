import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class SegmentationCLI(unittest.TestCase):
    def test_default_contours_and_rectangle_option(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name, flags in (('contours', []), ('rectangles', ['--no-contours'])):
                with self.subTest(mode=name):
                    output = Path(tmp) / name
                    subprocess.run([
                        sys.executable, str(ROOT / 'scripts/page_segmentation_loose.py'),
                        '--image', str(ROOT / 'examples/611_01_r1l31.png'),
                        '--output-dir', str(output), *flags,
                    ], check=True, capture_output=True, text=True)
                    lines = json.loads((output / 'manifest.json').read_text())['lines']
                    self.assertTrue(lines)
                    self.assertTrue((output / 'preview.png').is_file())
                    for line in lines:
                        self.assertEqual('polygon' in line, name == 'contours')
                        self.assertTrue((output / line['image']).is_file())
