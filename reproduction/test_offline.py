"""Offline checks of evaluation equivalence and financial fail-closed behavior."""
import ast
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from common import ROOT, select_questions, upstream_f1
from api_meter import BudgetStop, Meter

TEMP_ROOT = ROOT / '.cache' / 'test_tmp'
TEMP_ROOT.resolve().relative_to(ROOT.resolve())
TEMP_ROOT.mkdir(parents=True, exist_ok=True)

class EvaluationChecks(unittest.TestCase):
    def test_matches_pinned_upstream_f1(self):
        tree = ast.parse((ROOT / 'test_locomo10.py').read_text(encoding='utf-8'))
        tokenize = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'simple_tokenize')
        metrics = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'calculate_metrics')
        source_module = ast.Module(body=[tokenize, metrics], type_ignores=[])
        namespace = {'Dict': dict, 'simple_tokenize': None,
                     'calculate_rouge_scores': lambda *a: {}, 'calculate_bleu_scores': lambda *a: {},
                     'calculate_bert_scores': lambda *a: {}, 'calculate_meteor_score': lambda *a: 0,
                     'calculate_sentence_similarity': lambda *a: 0}
        exec(compile(source_module, '<upstream-evaluator>', 'exec'), namespace)
        rng = random.Random(11)
        strings = ['', 'The red car.', 'red red car', 'Starbucks!', 'May 12, 2023', 'a/b', '1990']
        for _ in range(50):
            prediction, reference = rng.choice(strings), rng.choice(strings)
            self.assertEqual(upstream_f1(prediction, reference), namespace['calculate_metrics'](prediction, reference)['f1'])

    def test_subset_is_deterministic_and_excludes_adversarial(self):
        data = [{'qa': [{'category': category, 'question': f'{category}-{i}', 'answer': 'x'}
                        for category in range(1, 6) for i in range(5)]}]
        first = select_questions(data, [0], 2, 5660)
        self.assertEqual(first, select_questions(data, [0], 2, 5660))
        self.assertEqual(len(first), 8)
        self.assertNotIn(5, [row['category'] for row in first])


class BudgetChecks(unittest.TestCase):
    def client(self):
        return SimpleNamespace(api_key='test-placeholder-not-a-key', base_url='https://api.deepseek.com')

    def test_rejects_before_any_request_when_reservation_exceeds_limit(self):
        calls = []
        fake = SimpleNamespace(OpenAI=lambda **kw: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: calls.append(kw)))))
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp, patch.dict(sys.modules, {'openai': fake}):
            meter = Meter(Path(temp) / 'usage.jsonl', limit_cny=0.0001)
            client = self.client()
            meter.wrap(client)
            with self.assertRaises(BudgetStop):
                client.client.chat.completions.create(model='deepseek-flash', messages=[{'role': 'user', 'content': 'Hello'}])
            self.assertEqual(calls, [])

    def test_uncertain_failed_request_stays_reserved_after_restart(self):
        def fail(**kwargs):
            raise TimeoutError('Simulated uncertain request')
        fake = SimpleNamespace(OpenAI=lambda **kw: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fail))))
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp, patch.dict(sys.modules, {'openai': fake}):
            path = Path(temp) / 'usage.jsonl'
            meter = Meter(path)
            client = self.client()
            meter.wrap(client)
            with self.assertRaises(TimeoutError):
                client.client.chat.completions.create(model='deepseek-flash', messages=[{'role': 'user', 'content': 'Hello'}])
            self.assertGreater(Meter(path).reserved_cny, 0)
            raw = path.read_text()
            self.assertNotIn('test-placeholder-not-a-key', raw)
            self.assertNotIn('Hello', raw)

    def test_stream_is_assembled_and_final_usage_is_recorded(self):
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=2,
                                model_dump=lambda: {'prompt_tokens': 100, 'completion_tokens': 2})
        chunks = [SimpleNamespace(model='deepseek-flash', usage=None, choices=[SimpleNamespace(
                      delta=SimpleNamespace(content='Star'), finish_reason=None)]),
                  SimpleNamespace(model='deepseek-flash', usage=usage, choices=[SimpleNamespace(
                      delta=SimpleNamespace(content='bucks'), finish_reason='stop')])]
        class Stream:
            def __enter__(self): return self
            def __exit__(self, *args): return False
            def __iter__(self): return iter(chunks)
        fake = SimpleNamespace(OpenAI=lambda **kw: SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kw: Stream()))))
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp, patch.dict(sys.modules, {'openai': fake}):
            meter = Meter(Path(temp) / 'usage.jsonl')
            client = self.client()
            meter.wrap(client)
            response = client.client.chat.completions.create(model='deepseek-flash', messages=[{'role': 'user', 'content': 'Hello'}])
            self.assertEqual(response.choices[0].message.content, 'Starbucks')
            self.assertAlmostEqual(meter.reserved_cny, (100 * 2 + 2 * 8) / 1_000_000)
            self.assertEqual(meter.records[-1]['usage']['completion_tokens'], 2)


if __name__ == '__main__':
    unittest.main()
