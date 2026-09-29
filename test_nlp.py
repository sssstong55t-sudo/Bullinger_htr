import unittest
from nlp_baseline import GazetteerNLP, evaluate


def place(pid, aliases):
    return {"id": pid, "canonical_name": pid, "latitude": 0, "longitude": 0,
            "aliases": aliases, "source": "synthetic test fixture"}


def row(text):
    return {"document_id": "d", "page_id": "p", "line_id": "l", "text": text, "language": "de"}


class NLPTests(unittest.TestCase):
    def model(self, *places):
        return GazetteerNLP({"version": "test", "places": list(places)})

    def test_boundaries(self):
        m = self.model(place("bern", ["Bern"]))
        out = m.extract(row("Bernhard besucht Bern."))["mentions"]
        self.assertEqual([x["text"] for x in out], ["Bern"])

    def test_casefold_offsets(self):
        m = self.model(place("x", ["Strasse"]))
        text = "In Straße!"
        out = m.extract(row(text))["mentions"][0]
        self.assertEqual(text[out["start"]:out["end"]], "Straße")

    def test_longest_alias(self):
        m = self.model(place("a", ["York"]), place("b", ["New York"]))
        self.assertEqual(m.extract(row("New York"))["mentions"][0]["place_id"], "b")

    def test_ambiguity_not_arbitrarily_resolved(self):
        m = self.model(place("a", ["Neustadt"]), place("b", ["Neustadt"]))
        mention = m.extract(row("Neustadt"))["mentions"][0]
        self.assertIsNone(mention["place_id"])
        self.assertEqual(len(mention["candidates"]), 2)
        self.assertIsNone(mention["evidence"]["line_bbox"])

    def test_missing_and_spurious(self):
        m = self.model(place("bern", ["Bern"]))
        gold = {**row("Bern und Basel"), "gold_mentions": [
            {"start": 0, "end": 4, "place_id": "bern"},
            {"start": 9, "end": 14, "place_id": "basel"}]}
        scores = evaluate([gold], [m.extract(gold)], "span")["scores"]["overall"]
        self.assertEqual((scores["tp"], scores["fn"]), (1, 1))

    def test_duplicate_mentions_count(self):
        m = self.model(place("bern", ["Bern"]))
        gold = {**row("Bern Bern"), "gold_mentions": [
            {"start": 0, "end": 4, "place_id": "bern"},
            {"start": 5, "end": 9, "place_id": "bern"}]}
        pred = m.extract(row("Bern xxxx"))
        score = evaluate([gold], [pred], "places")["scores"]["overall"]
        self.assertEqual((score["tp"], score["fn"]), (1, 1))
        with self.assertRaises(ValueError):
            evaluate([gold], [pred], "span")

    def test_negative_and_empty(self):
        m = self.model(place("bern", ["Bern"]))
        gold = {**row("Bern"), "gold_mentions": []}
        score = evaluate([gold], [m.extract(gold)], "span")["scores"]["overall"]
        self.assertEqual(score["fp"], 1)
        self.assertEqual(m.extract(row(""))["mentions"], [])

    def test_reject_mismatched_lines(self):
        gold = {**row(""), "gold_mentions": []}
        pred = {**row(""), "line_id": "different", "mentions": []}
        with self.assertRaises(ValueError):
            evaluate([gold], [pred], "span")

    def test_unresolved_gold_is_explicitly_excluded(self):
        gold = {**row("Town"), "gold_mentions": [{"start": 0, "end": 4, "place_id": None}]}
        result = evaluate([gold], [{**row("Town"), "mentions": []}], "places")
        self.assertEqual(result["excluded_unresolved_gold_lines"], 1)


if __name__ == "__main__":
    unittest.main()
