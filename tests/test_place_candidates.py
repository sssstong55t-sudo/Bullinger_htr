import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
DELIVERY = ROOT / 'outputs/bullinger_htr_delivery'
sys.path[:0] = [str(ROOT / 'scripts'), str(DELIVERY / 'scripts')]

from extract_places import add_alias_candidates, finalize_mentions
from historical_aliases import HistoricalAliases


class PlaceCandidates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.matchers = [
            HistoricalAliases(json.loads((DELIVERY / 'data' / name).read_text()))
            for name in ('historical_aliases.json', 'source_aliases.json')
        ]

    def mention(self, text):
        mention = dict(start=0, end=len(text), text=text, sources=['mbert'])
        for matcher in self.matchers:
            for match in matcher.extract(text, 'test'):
                add_alias_candidates(mention, match['candidates'])
        return finalize_mentions({(0, len(text)): mention})[0]

    def test_duplicate_alias_returns_one_candidate_with_coordinates(self):
        result = self.mention('Basileę')
        self.assertEqual(result['place_link_status'], 'candidate_for_review')
        self.assertEqual(len(result['official_place_candidates']), 1)
        candidate, = result['official_place_candidates']
        self.assertEqual(candidate['official_place_id'], 'l28')
        self.assertEqual(candidate['canonical_name'], 'Basel')
        self.assertEqual((candidate['latitude'], candidate['longitude']), (47.55839, 7.57327))

    def test_multiple_and_missing_candidates(self):
        result = self.mention('Zurich')
        self.assertEqual(result['place_link_status'], 'ambiguous')
        self.assertEqual([c['official_place_id'] for c in result['official_place_candidates']],
                         ['l587', 'l803'])
        missing = self.mention('UnknownTestPlace')
        self.assertEqual(missing['place_link_status'], 'unmatched')
        self.assertEqual(missing['official_place_candidates'], [])
