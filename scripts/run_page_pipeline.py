"""Run one page through contours, public TrOCR and mBERT; export portable results.

Install the README dependencies and models first. No notebook or web service is
required. Use a fresh output directory; partial runs are retained for diagnosis.
"""

import argparse
import base64
from collections import Counter
import hashlib
import html
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    try:
        result = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]


def index_by_image(rows):
    indexed = {Path(row['image']).resolve(): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError('Duplicate line images were found.')
    return indexed


def run_stage(script, *arguments):
    env = {**os.environ, 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'PYTHONUNBUFFERED': '1'}
    subprocess.run([sys.executable, str(ROOT / 'scripts' / script),
                    *map(str, arguments)], cwd=ROOT, env=env, check=True)


def export_results(RESULT, PAGE, CROPS, original_name, upload_info, page_id):
    """Validate stage joins and create JSON, text, and a self-contained report."""
    HTR, PLACES = RESULT / 'htr.jsonl', RESULT / 'places.jsonl'
    manifest = json.loads((CROPS / 'manifest.json').read_text(encoding='utf-8'))
    crop_paths = [CROPS / line['image'] for line in manifest['lines']]
    htr = index_by_image(read_jsonl(HTR))
    ner = index_by_image(read_jsonl(PLACES))
    if not (set(htr) == set(ner) == set(path.resolve() for path in crop_paths)):
        raise ValueError('Line images do not align across pipeline stages.')
    counts = Counter()
    lines = []
    for segment in manifest['lines']:
        crop = (CROPS / segment['image']).resolve()
        prediction, entities = htr[crop], ner[crop]
        if not (prediction['text'] == entities['text']):
            raise ValueError("Pipeline integrity check failed: prediction['text'] == entities['text']")
        if not (prediction['image_sha256'] == sha256(crop)):
            raise ValueError("Pipeline integrity check failed: prediction['image_sha256'] == sha256(crop)")
        for mention in entities['places']:
            start, end = mention['start'], mention['end']
            if not (0 <= start < end <= len(prediction['text'])):
                raise ValueError("Pipeline integrity check failed: 0 <= start < end <= len(prediction['text'])")
            if not (prediction['text'][start:end] == mention['text']):
                raise ValueError("Pipeline integrity check failed: prediction['text'][start:end] == mention['text']")
            candidates = mention['official_place_candidates']
            if not (len({c['official_place_id'] for c in candidates}) == len(candidates)):
                raise ValueError("Pipeline integrity check failed: len({c['official_place_id'] for c in candidates}) == len(candidates)")
            status = 'unmatched' if not candidates else ('candidate_for_review' if len(candidates) == 1 else 'ambiguous')
            if not (mention['place_link_status'] == status):
                raise ValueError("Pipeline integrity check failed: mention['place_link_status'] == status")
            counts[status] += 1
        lines.append({'line_id': segment['line_id'], 'image': 'crops/' + segment['image'],
                      'bbox': segment['bbox'], 'polygon': segment['polygon'],
                      'text': prediction['text'], 'places': entities['places']})
    payload = {
        'page': {'page_id': page_id, **upload_info, 'image': 'input/page.png',
                 'image_sha256': sha256(PAGE), 'width': manifest['page_size'][0],
                 'height': manifest['page_size'][1], 'coordinate_system': manifest['coordinate_system'],
                 'reading_order': manifest['reading_order']},
        'pipeline': {'github_commit': git_commit(), 'segmentation': 'contours',
                     'htr_model': 'pstroe/bullinger-general-model',
                     'htr_revision': 'be32c883a4deb812ab6f461ae79f583c3f9edc53',
                     'place_extraction_mode': 'mbert_with_aliases',
                     'place_resolution': 'official lexicon candidates for review',
                     'text_offset_unit': 'Unicode code points; end exclusive'},
        'summary': {'line_count': len(lines), 'place_mention_count': sum(counts.values()),
                    'place_link_status_counts': {key: counts[key] for key in
                        ['candidate_for_review', 'ambiguous', 'unmatched']},
                    'mentions_with_candidate_coordinates': sum(
                        any(c['latitude'] is not None and c['longitude'] is not None
                            for c in m['official_place_candidates']) for line in lines for m in line['places'])},
        'lines': lines}
    JSON_PATH = RESULT / 'page_response.json'
    JSON_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (RESULT / 'transcription.txt').write_text('\n'.join(line['text'] for line in lines) + '\n', encoding='utf-8')

    def esc(value):
        return html.escape('—' if value is None else str(value))

    def marked_text(text, mentions):
        # Union overlapping spans so the display preserves every original character.
        spans = []
        for mention in sorted(mentions, key=lambda m: (m['start'], m['end'])):
            start, end = mention['start'], mention['end']
            if spans and start <= spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], end)
            else:
                spans.append([start, end])
        chunks, cursor = [], 0
        for start, end in spans:
            chunks += [esc(text[cursor:start]), '<mark>' + esc(text[start:end]) + '</mark>']
            cursor = end
        return ''.join(chunks) + esc(text[cursor:])

    labels = {'candidate_for_review': 'One candidate: review required', 'ambiguous': 'Multiple candidates', 'unmatched': 'Unmatched'}
    body = [f"<h1>Bullinger page recognition results</h1><p>{esc(original_name)} · {len(lines)} lines · {sum(counts.values())} place mentions</p>",
            '<p>Place mentions are highlighted in yellow. Modern place names are lexicon candidates that require review.</p>']
    for line in lines:
        with Image.open(CROPS / Path(line['image']).name) as im:
            thumb = im.copy()
        thumb.thumbnail((1000, 240))
        buffer = io.BytesIO()
        thumb.save(buffer, format='PNG')
        data = base64.b64encode(buffer.getvalue()).decode('ascii')
        body.append(f"<section><h3>{esc(line['line_id'])}</h3><div class='pair'><img src='data:image/png;base64,{data}'><p>{marked_text(line['text'], line['places'])}</p></div></section>")
    body.append('<h2>Modern place candidates</h2>')
    if not sum(counts.values()):
        body.append('<p>No place mentions were detected. Each line has an empty places array in the JSON.</p>')
    else:
        body.append('<div class="scroll"><table><tr><th>Line</th><th>Mention</th><th>Status</th><th>Candidate name</th><th>Place ID</th><th>Latitude</th><th>Longitude</th><th>Source</th></tr>')
        for line in lines:
            for mention in line['places']:
                for candidate in mention['official_place_candidates'] or [{}]:
                    values = [line['line_id'], mention['text'], labels[mention['place_link_status']],
                              candidate.get('canonical_name'), candidate.get('official_place_id'),
                              candidate.get('latitude'), candidate.get('longitude'), candidate.get('source')]
                    body.append('<tr>' + ''.join('<td>' + esc(v) + '</td>' for v in values) + '</tr>')
        body.append('</table></div>')
    style = 'body{font-family:system-ui,sans-serif;margin:24px;color:#222}section{border-top:1px solid #ddd;padding:12px 0}.pair{display:flex;gap:20px;align-items:center;flex-wrap:wrap}.pair img{max-width:100%;width:500px}.pair p{flex:1;min-width:240px;white-space:pre-wrap;font-size:18px}mark{background:#ffe58a}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px;text-align:left}.scroll{overflow:auto}'
    report = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bullinger recognition results</title><style>' + style + '</style></head><body>' + ''.join(body) + '</body></html>'
    (RESULT / 'report.html').write_text(report, encoding='utf-8')

    print('JSON created:', JSON_PATH.name)

    # Export portable paths in every artifact, keeping model provenance unchanged.
    paths = {str(path.resolve()): 'crops/' + path.name for path in crop_paths}
    ids = {str((CROPS / line['image']).resolve()): line['line_id']
           for line in manifest['lines']}
    for path in (HTR, PLACES):
        records = read_jsonl(path)
        for record in records:
            key = str(Path(record['image']).resolve())
            record['image'] = paths[key]
            record['line_id'] = ids[key]
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n'
                                for row in records), encoding='utf-8')
    manifest['source'] = 'input/page.png'
    (CROPS / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    zip_path = RESULT / 'bullinger_page_results.zip'
    with zipfile.ZipFile(zip_path, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(RESULT.rglob('*')):
            if path.is_file() and path != zip_path:
                archive.write(path, path.relative_to(RESULT).as_posix())
    with zipfile.ZipFile(zip_path) as archive:
        if archive.testzip() is not None:
            raise ValueError('ZIP integrity check failed.')
    print('Complete results:', zip_path, flush=True)
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='New directory for images, JSON, report, and ZIP')
    parser.add_argument('--page-id', help='Optional stable page identifier supplied by the backend')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'mps'], default='auto')
    args = parser.parse_args()
    source = args.image.expanduser().resolve()
    result = args.output_dir.expanduser().resolve()
    if not source.is_file():
        parser.error('Input image does not exist.')
    if result.exists():
        parser.error('Output directory already exists; choose a new path.')
    required = [ROOT / 'models/bullinger_public_trocr/pytorch_model.bin',
                ROOT / 'models/bullinger_public_trocr/tokenizer/vocab.json',
                ROOT / 'models/bullinger_mbert_20261022/model.safetensors']
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error('Install the README models first. Missing: ' + ', '.join(missing))
    result.mkdir(parents=True)
    (result / 'input').mkdir()
    page = result / 'input/page.png'
    with Image.open(source) as image:
        oriented = ImageOps.exif_transpose(image).convert('RGB')
        oriented.save(page)
    crops = result / 'crops'
    run_stage('page_segmentation_loose.py', '--image', page, '--output-dir', crops)
    manifest = json.loads((crops / 'manifest.json').read_text(encoding='utf-8'))
    crop_paths = [crops / line['image'] for line in manifest['lines']]
    if not crop_paths:
        raise ValueError('No text lines found; inspect crops/preview.png and try another image.')
    run_stage('transcribe_images.py', '--images', *crop_paths,
              '--output', result / 'htr.jsonl', '--device', args.device)
    run_stage('extract_places.py', '--input', result / 'htr.jsonl',
              '--output', result / 'places.jsonl')
    export_results(result, page, crops, source.name,
                   {'original_filename': source.name, 'original_sha256': sha256(source)},
                   args.page_id or result.name)


if __name__ == '__main__':
    main()
