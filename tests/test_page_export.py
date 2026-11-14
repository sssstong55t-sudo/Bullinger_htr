"""Check the portable page boundary without requiring neural model downloads."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_page_pipeline import export_results, sha256


class PageExport(unittest.TestCase):
    def fixture(self, root):
        crops = root / 'crops'
        crops.mkdir()
        (root / 'input').mkdir()
        page = root / 'input/page.png'
        Image.new('RGB', (40, 20), 'white').save(page)
        crop = crops / 'line_0001.png'
        crop.write_bytes(page.read_bytes())
        (crops / 'preview.png').write_bytes(page.read_bytes())
        manifest = {'page_size': [40, 20], 'coordinate_system': 'EXIF-oriented pixels',
                    'reading_order': 'top-to-bottom', 'lines': [
                        {'line_id': 'line_0001', 'image': crop.name,
                         'bbox': [0, 0, 40, 20], 'polygon': [[0, 0], [40, 0], [40, 20]]}]}
        (crops / 'manifest.json').write_text(json.dumps(manifest))
        row = {'line_id': str(crop), 'image': str(crop), 'image_sha256': sha256(crop),
               'text': '😀 <Basel> & X'}
        candidate = {'official_place_id': 'l28', 'canonical_name': 'Basel',
                     'latitude': 47.55839, 'longitude': 7.57327, 'source': 'test'}
        ner = {**row, 'places': [
            {'start': 3, 'end': 8, 'text': 'Basel', 'sources': ['test'],
             'official_place_candidates': [candidate], 'place_link_status': 'candidate_for_review'},
            {'start': 12, 'end': 13, 'text': 'X', 'sources': ['test'],
             'official_place_candidates': [], 'place_link_status': 'unmatched'}]}
        for name, value in [('htr.jsonl', row), ('places.jsonl', ner)]:
            (root / name).write_text(json.dumps(value) + '\n')
        return page, crops

    def test_portable_archive_and_literal_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            page, crops = self.fixture(root)
            payload = export_results(root, page, crops, '<test>.png', {}, 'page-1')
            with zipfile.ZipFile(root / 'bullinger_page_results.zip') as archive:
                self.assertIsNone(archive.testzip())
                for relative in [payload['page']['image'], payload['lines'][0]['image']]:
                    self.assertIn(relative, archive.namelist())
                row = json.loads(archive.read('htr.jsonl'))
                self.assertEqual(row['image'], 'crops/line_0001.png')
                self.assertEqual(row['line_id'], 'line_0001')
                report = archive.read('report.html').decode()
                self.assertIn('😀 &lt;<mark>Basel</mark>&gt; &amp; <mark>X</mark>', report)
                self.assertIn('&lt;test&gt;.png', report)
            self.assertEqual(payload['summary']['place_link_status_counts']['unmatched'], 1)

    def test_misaligned_text_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            page, crops = self.fixture(root)
            path = root / 'places.jsonl'
            row = json.loads(path.read_text())
            row['text'] = 'different text'
            path.write_text(json.dumps(row))
            with self.assertRaises(ValueError):
                export_results(root, page, crops, 'page.png', {}, 'page-1')
            self.assertFalse((root / 'bullinger_page_results.zip').exists())
