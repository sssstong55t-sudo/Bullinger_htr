"""Recompute the frozen whole-page scores from the archived predictions."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import unicodedata

from ner_span_type import distance, metric, score_row

OUT = Path(__file__).resolve().parent


def score(ocr, places=None):
    pages = json.loads((OUT / 'plan.json').read_text())
    records = list(map(json.loads, ocr.read_text().splitlines()))
    predictions = {r['line_id']: r for r in records}
    expected_ids = {r['line_id'] for page in pages for r in page['predictions']}
    if len(records) != len(predictions) or set(predictions) != expected_ids:
        raise ValueError('HTR must contain exactly one prediction for each generated crop')
    ner = None if places is None else {
        r['line_id']: r for r in map(json.loads, places.read_text().splitlines())
    }
    if ner is not None and set(ner) != expected_ids:
        raise ValueError('Place predictions must cover every generated crop')
    totals = defaultdict(Counter)
    page_scores = []
    for page in pages:
        refs, crops = page['reference'], page['predictions']
        pairs = {int(i): j for i, j in page['pairs'].items()}
        assert all(row['line_id'] in predictions for row in crops)
        stats = Counter(seg_tp=len(pairs), seg_fp=len(crops)-len(pairs), seg_fn=len(refs)-len(pairs))
        for i, ref in enumerate(refs):
            if ref['exclusion']:
                stats['excluded_reference_lines'] += 1
                stats['excluded_reference_places'] += len(ref['mentions'])
                continue
            crop = crops[pairs[i]] if i in pairs else None
            lid = crop['line_id'] if crop else f"missing/{ref['zone']}"
            text = predictions[lid]['text'] if crop else ''
            ref_words = ref['text'].split()
            counts = Counter(char_errors=distance(ref['text'], text), chars=len(ref['text']),
                             word_errors=distance(ref_words, text.split()), words=len(ref_words))
            totals['full_page'].update(counts)
            if crop:
                totals['matched_only'].update(counts)
            stats.update(counts)
            if ner is not None and ref['cross_line_loc_parts']:
                stats['ner_excluded_cross_line'] += 1
                stats['ner_excluded_same_line_places'] += len(ref['mentions'])
            elif ner is not None:
                mentions = []
                if crop:
                    assert ner[lid]['text'] == text
                    places_for_line = ner[lid]['places']
                    assert all(text[m['start']:m['end']] == m['text'] for m in places_for_line)
                    mentions = [dict(start=m['start'], end=m['end'], type='LOC') for m in places_for_line]
                result, projected = score_row(dict(ref, line_id=lid), dict(line_id=lid, text=text, mentions=mentions))
                stats.update(result)
                if crop:
                    stats['ambiguous_or_unprojectable_places'] += sum(
                        not span['projection_unique'] for span in projected
                    )
        for j, crop in enumerate(crops):
            if j not in pairs.values():
                lid = crop['line_id']
                text = predictions[lid]['text']
                stats.update(char_errors=len(text), word_errors=len(text.split()))
                totals['full_page'].update(char_errors=len(text), word_errors=len(text.split()))
                if ner is not None:
                    stats.update(fp=len(ner[lid]['places']))
        page_scores.append(dict(page=page['page'], **stats))
    combined = sum((Counter({k: v for k, v in r.items() if k != 'page'}) for r in page_scores), Counter())
    result = dict(segmentation=metric(combined['seg_tp'], combined['seg_fp'], combined['seg_fn']),
                  transcription={scope: dict(counts, cer=counts['char_errors']/counts['chars'],
                                             wer=counts['word_errors']/counts['words'])
                                 for scope, counts in totals.items()},
                  ner=None if ner is None else metric(combined['tp'], combined['fp'], combined['fn']),
                  pages=page_scores)
    return result


def score_seed44_pair():
    inputs = json.loads((OUT / 'paired_seed44_inputs.json').read_text())
    predictions = {r['line_id']: r['text'] for r in map(
        json.loads, (OUT / 'public_trocr.jsonl').read_text().splitlines()
    )}

    def normalize(lines):
        return ' '.join(unicodedata.normalize('NFC', ' '.join(lines)).split())

    def totals(rows, use_seed44):
        counts = Counter(char_errors=0, ref_chars=0, word_errors=0,
                         ref_words=0, exact=0, n=0)
        for row in rows:
            ref = row['reference']
            lines = row['seed44'] if use_seed44 else [predictions[lid] for lid in row['line_ids']]
            hyp = normalize(lines)
            counts.update(char_errors=distance(ref, hyp), ref_chars=len(ref),
                          word_errors=distance(ref.split(), hyp.split()),
                          ref_words=len(ref.split()), exact=int(ref == hyp), n=1)
        return dict(counts, cer=counts['char_errors']/counts['ref_chars'],
                    wer=counts['word_errors']/counts['ref_words'])

    result = dict(byte_identical_crops_verified=len(inputs['shared_crop_ids']),
                  excluded_paired_references=len(inputs['excluded_paired_references']))
    for model in ('seed44_cached', 'public_trocr'):
        result[model] = {scope: totals(inputs[scope], model == 'seed44_cached')
                         for scope in ('paired_lines', 'full_pages')}
    return result


def main():
    htr = OUT / 'public_trocr.jsonl'
    results = {'scores_public_trocr': score(htr)}
    for method in ('mbert', 'gliner'):
        stem = f'public_trocr_{method}'
        results[f'scores_{stem}'] = score(htr, OUT / f'{stem}.jsonl')
    results['paired_seed44_comparison'] = score_seed44_pair()
    for name, result in results.items():
        expected = json.loads((OUT / f'{name}.json').read_text())
        if result != expected:
            raise SystemExit(f'Metric mismatch: {name}')
        print(name, json.dumps({k: v for k, v in result.items() if k != 'pages'}))
    print('PASS: reproduced all four saved score reports.')


if __name__ == '__main__':
    main()
