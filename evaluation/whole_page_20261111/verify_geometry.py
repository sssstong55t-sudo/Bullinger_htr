"""Recompute polygon matches or regenerate crops from the recorded source scans."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEI = ROOT / 'outputs/bullinger_htr_delivery_EN/data/official_letters'


def polygon_mask(polygon):
    xs, ys = zip(*polygon)
    left, top, right, bottom = min(xs), min(ys), max(xs) + 1, max(ys) + 1
    mask = Image.new("1", (right - left, bottom - top))
    ImageDraw.Draw(mask).polygon([(x - left, y - top) for x, y in polygon], fill=1)
    pixels = np.asarray(mask)
    return (left, top, right, bottom), pixels, np.count_nonzero(pixels)


def polygon_overlaps(refs, predictions):
    scores = np.zeros((len(refs), len(predictions)))
    pred_masks = [polygon_mask(polygon) for polygon in predictions]
    for i, reference in enumerate(refs):
        (rx0, ry0, rx1, ry1), ref, ref_area = polygon_mask(reference)
        for j, ((px0, py0, px1, py1), pred, pred_area) in enumerate(pred_masks):
            x0, y0 = max(rx0, px0), max(ry0, py0)
            x1, y1 = min(rx1, px1), min(ry1, py1)
            if x0 < x1 and y0 < y1:
                inter = np.count_nonzero(ref[y0-ry0:y1-ry0, x0-rx0:x1-rx0]
                                         & pred[y0-py0:y1-py0, x0-px0:x1-px0])
                scores[i, j] = inter / (ref_area + pred_area - inter)
    return scores


def reference_polygons(page_id):
    letter, page = page_id.split('-')
    xml = ET.parse(TEI / f'{letter}.xml')
    surface = xml.findall('.//{*}surface')[int(page) - 1]
    return [([tuple(map(int, point.split(','))) for point in zone.get('points').split()],
             zone.get('{http://www.w3.org/XML/1998/namespace}id'))
            for zone in surface.findall('{*}zone') if zone.get('points')]


def prepare_images(page, images, crop_dir, download):
    from page_segmentation_loose import export_page

    source = images / f"{page['page']}.jpg"
    if not source.exists() and download:
        with urlopen(page['source'], timeout=60) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != page['image_sha256']:
            raise ValueError(f'Changed source bytes: {page["source"]}')
        source.write_bytes(data)
    if hashlib.sha256(source.read_bytes()).hexdigest() != page['image_sha256']:
        raise ValueError(f'Unexpected source bytes: {source}')
    path = crop_dir / page['page']
    manifest = export_page(source, path, contours=True)
    original = page['predictions']
    if [row['polygon'] for row in manifest['lines']] != [row['polygon'] for row in original]:
        raise ValueError(f'Contour output differs from frozen output on {page["page"]}')
    changed_pngs = sum(hashlib.sha256((path / live['image']).read_bytes()).hexdigest() != cached['sha256']
                       for live, cached in zip(manifest['lines'], original))
    if changed_pngs:
        print(f'{page["page"]}: {changed_pngs} PNG hashes differ from the recorded crops.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images-dir', type=Path, help='Directory for source scans named LETTER-PAGE.jpg')
    parser.add_argument('--download', action='store_true', help='Fetch missing source scans from the manifest')
    parser.add_argument('--crop-dir', type=Path, help='Fresh directory for regenerated crops')
    args = parser.parse_args()
    if (args.images_dir is None) != (args.crop_dir is None) or (args.download and args.images_dir is None):
        parser.error('Crop regeneration requires both --images-dir and --crop-dir')
    if args.images_dir is not None:
        if args.crop_dir.exists():
            parser.error('--crop-dir must be a new directory')
        args.images_dir.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(ROOT / 'scripts'))
    pages = json.loads((HERE / 'plan.json').read_text())
    for page in pages:
        refs, ids = zip(*reference_polygons(page['page']))
        if list(ids) != [row['zone'] for row in page['reference']]:
            raise ValueError(f'TEI reference mismatch: {page["page"]}')
        overlaps = polygon_overlaps(refs, [row['polygon'] for row in page['predictions']])
        valid = overlaps >= .5
        ixs, jxs = linear_sum_assignment(valid * (len(refs) + 1 + overlaps), maximize=True)
        pairs = {str(i): int(j) for i, j in zip(ixs, jxs) if valid[i, j]}
        if pairs != page['pairs']:
            raise ValueError(f'Polygon matching mismatch: {page["page"]}')
        if args.images_dir is not None:
            prepare_images(page, args.images_dir, args.crop_dir, args.download)
        print(page['page'], len(refs), len(page['predictions']), len(pairs))
    print('PASS: reproduced the frozen contour matches on all 10 pages.')


if __name__ == '__main__':
    main()
