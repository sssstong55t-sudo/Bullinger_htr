import unittest,sys,math,json,tempfile,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_htr import metrics,indexed
from rerank_htr import lm_scores,choose

class LocalTools(unittest.TestCase):
    def test_insertion_and_empty_language(self):
        m=metrics([{'reference':'ab cd','prediction':'ab X cd','language':'de'}])
        self.assertEqual(m['overall']['character_errors'],2);self.assertEqual(m['overall']['cer'],.4);self.assertEqual(m['de']['wer'],.5);self.assertIsNone(m['la']['cer'])
    def test_corpus_weighting_and_case(self):
        m=metrics([{'reference':'a','prediction':'A','language':'de'},{'reference':'bbbbbbbbb','prediction':'bbbbbbbbb','language':'la'}]);self.assertEqual(m['overall']['cer'],.1)
    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'rows.jsonl';p.write_text('{"line_id":"a"}\n{"line_id":"a"}\n')
            with self.assertRaises(ValueError):indexed(p)
    def test_smoothing_and_fixed_candidate_choice(self):
        stats={'languages':{'la':{'alphabet':['a','$'],'words':{'a':10},'grams':{},'contexts':{}}}}
        c,w=lm_scores('unknown','la',stats);self.assertTrue(math.isfinite(c+w))
        winner,_=choose([{'text':'a','model_score':0},{'text':'unknown','model_score':0}],'la',stats);self.assertEqual(winner['text'],'a')
        with self.assertRaises(ValueError):choose([{'text':'a','model_score':0}],'unknown',stats)
if __name__=='__main__':unittest.main()
