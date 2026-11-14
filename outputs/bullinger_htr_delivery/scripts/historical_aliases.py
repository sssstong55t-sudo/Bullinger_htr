"""Evidence-backed historical variants learned only from declared development records."""
from collections import defaultdict
import re
from nlp_baseline import fold


def build_lexicon(rows):
    aliases = defaultdict(lambda: {'variants': set(), 'places': {}, 'evidence': []})
    excluded = []
    for row in rows:
        if row['transcription_source_category'] != 'edited':
            continue
        for mention in row['mentions']:
            text = mention['text']
            if row['text'][mention['start']:mention['end']] != text:
                raise ValueError('Development mention offsets differ from source text')
            key = fold(text)[0]
            if len(key.strip()) < 4 or not mention.get('official_place_records'):
                excluded.append({'text': text, 'reason': 'short_or_missing_official_identity'})
                continue
            entry = aliases[key]
            entry['variants'].add(text)
            for place in mention['official_place_records']:
                entry['places'][place['place_id']] = place
            entry['evidence'].append({'document_id': row['document_id'],
                'paragraph_number': row['paragraph_number'], 'start': mention['start'],
                'end': mention['end'], 'source_sha256': row['source_sha256'],
                'certainty': mention.get('certainty', 'unspecified')})
    return {'version': 'official-edited-development-variants-v1',
        'learning_documents': sorted({r['document_id'] for r in rows}),
        'automatic_geolocation': False, 'excluded': excluded,
        'aliases': [{'key': key, 'variants': sorted(value['variants']),
                     'places': list(value['places'].values()), 'evidence': value['evidence']}
                    for key, value in sorted(aliases.items())]}


class HistoricalAliases:
    def __init__(self, lexicon):
        self.entries = {}
        for entry in lexicon['aliases']:
            key = entry['key']
            if not key or key in self.entries or any(fold(v)[0] != key for v in entry['variants']):
                raise ValueError('Invalid or duplicate folded historical alias')
            if not entry['places'] or not entry['evidence']:
                raise ValueError('Historical alias requires identity and source evidence')
            self.entries[key] = entry
        terms = sorted(self.entries, key=lambda s: (-len(s), s))
        self.pattern = re.compile(r'(?<!\w)(?:' + '|'.join(map(re.escape, terms)) + r')(?!\w)') if terms else None

    def extract(self, text, line_id):
        folded, starts, ends = fold(text)
        result = []
        for match in self.pattern.finditer(folded) if self.pattern else []:
            start, end = starts[match.start()], ends[match.end() - 1]
            entry = self.entries[match.group()]
            candidates = []
            for place in entry['places']:
                coords = place.get('coordinates_lat_lon')
                candidates.append({'id': 'bullinger:' + place['place_id'],
                    'canonical_name': place.get('name') or None,
                    'latitude': coords[0] if coords else None,
                    'longitude': coords[1] if coords else None,
                    'source': 'https://github.com/stazh/bullinger-korpus-tei',
                    'official_place_id': place['place_id'],
                    'coordinate_interpretation': 'Official record; historical identity and geometry require review'})
            result.append({'text': text[start:end],
                'segments': [{'line_id': line_id, 'start': start, 'end': end}],
                'sources': ['historical_alias'], 'source_scores': {},
                'needs_review': True, 'resolution_status': 'unresolved', 'candidates': candidates,
                'alias_evidence': entry['evidence'],
                'alias_interpretation': 'Observed geographic expression; may be a demonym or area, not a city'})
        return result


def augment_candidates(mentions, aliases):
    """Retain existing spans; alias support alone never certifies a place or raises priority."""
    output = {tuple((s['line_id'], s['start'], s['end']) for s in m['segments']):
              {**m, 'sources': list(m['sources']), 'source_scores': dict(m.get('source_scores', {})),
               'candidates': list(m.get('candidates', []))} for m in mentions}
    for alias in aliases:
        key = tuple((s['line_id'], s['start'], s['end']) for s in alias['segments'])
        if key not in output:
            output[key] = alias
            continue
        current = output[key]
        current['sources'] = list(dict.fromkeys(current['sources'] + alias['sources']))
        ids = {p['id'] for p in current['candidates']}
        current['candidates'].extend(p for p in alias['candidates'] if p['id'] not in ids)
        current['alias_evidence'] = alias['alias_evidence']
        current['alias_interpretation'] = alias['alias_interpretation']
    return list(output.values())
