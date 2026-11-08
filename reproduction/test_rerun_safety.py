"""Offline fault-recovery checks; no real model or private configuration used."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from reproduction.api_meter import Meter, BudgetStop
from reproduction.memory_window_cache import install_memory_window_cache
from reproduction.common import ROOT
from simplemem.core.models.memory_entry import MemoryEntry, Dialogue

class ReplayChecks(unittest.TestCase):
    def setUp(self):
        (ROOT / '.cache/test_tmp').mkdir(parents=True, exist_ok=True)

    def system(self, counter):
        def generate(dialogues):
            counter.append(1)
            return [MemoryEntry(lossless_restatement='Alice attends a meeting.')]
        return SimpleNamespace(memory_builder=SimpleNamespace(window_size=40, overlap_size=2,
            previous_entries=[], _generate_memory_entries=generate))

    def test_successful_window_replays_identical_entries_without_new_call(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache/test_tmp') as temp:
            calls=[];meter=Meter(Path(temp)/'ledger.jsonl');dialogues=[Dialogue(dialogue_id=1,speaker='Alice',content='Meeting')]
            first=self.system(calls);install_memory_window_cache(first,meter,Path(temp)/'windows',{'protocol':40})
            a=first.memory_builder._generate_memory_entries(dialogues)
            second=self.system(calls);install_memory_window_cache(second,meter,Path(temp)/'windows',{'protocol':40})
            b=second.memory_builder._generate_memory_entries(dialogues)
            self.assertEqual(len(calls),1);self.assertEqual(a[0].model_dump(),b[0].model_dump())
            self.assertEqual(meter.reserved_cny,0)
            second.memory_builder.previous_entries=[MemoryEntry(lossless_restatement='Different context.')]
            second.memory_builder._generate_memory_entries(dialogues)
            self.assertEqual(len(calls),2)

    def test_corrupt_window_cache_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache/test_tmp') as temp:
            meter=Meter(Path(temp)/'ledger.jsonl');system=self.system([])
            install_memory_window_cache(system,meter,Path(temp)/'windows',{'protocol':40})
            dialogues=[Dialogue(dialogue_id=1,speaker='Alice',content='Meeting')]
            system.memory_builder._generate_memory_entries(dialogues)
            path=next((Path(temp)/'windows').glob('*.json'));value=json.loads(path.read_text())
            value['entries'][0]['lossless_restatement']='Modified';path.write_text(json.dumps(value))
            with self.assertRaises(BudgetStop):system.memory_builder._generate_memory_entries(dialogues)

    def test_complete_usage_chunk_does_not_wait_for_extra_transport_data(self):
        usage=SimpleNamespace(prompt_tokens=100,completion_tokens=2,model_dump=lambda:{'prompt_tokens':100,'completion_tokens':2})
        class Stream:
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def __iter__(self):
                yield SimpleNamespace(model='deepseek-flash',usage=usage,choices=[SimpleNamespace(
                    delta=SimpleNamespace(content='Complete'),finish_reason='stop')])
                raise AssertionError('Read beyond terminal response with final usage')
        fake=SimpleNamespace(OpenAI=lambda **kw:SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw:Stream()))))
        with tempfile.TemporaryDirectory(dir=ROOT/'.cache/test_tmp') as temp,patch.dict(sys.modules,{'openai':fake}):
            meter=Meter(Path(temp)/'ledger.jsonl');client=SimpleNamespace(api_key='test-placeholder',base_url='https://api.deepseek.com')
            meter.wrap(client)
            result=client.client.chat.completions.create(model='deepseek-flash',messages=[{'role':'user','content':'test'}])
            self.assertEqual(result.choices[0].message.content,'Complete')
            self.assertEqual(meter.records[-1]['event'],'response')

if __name__=='__main__':unittest.main()
